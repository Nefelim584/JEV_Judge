"""One streaming interface for live ASR + end of turn (todo Phase 9b, "One streaming interface").

Every system takes audio chunks of any length (16 kHz mono float32) and returns events:

    partial   the text of the current turn so far (only systems that decode while the speaker talks)
    endpoint  the turn ended (t1); ``detail`` = eou / eob (the model's token) or silence / turn / timeout
              (the endpoint detector), eos at the end of the stream
    final     the turn's final text (t2 = endpoint time + its ``compute_ms``)

``audio_s`` is the stream time of the event: the end of the audio consumed when it was decided (so the
lookahead of a model and the silence a detector waits are inside it); ``compute_ms`` is the compute spent
on it on this machine. In a live stream fed in real time, t = ``audio_s`` + ``compute_ms`` (+ queueing).

Systems (todo Phase 9b decisions of 2026-11-18), named for ``parse_system``:

- ``parakeet-eou`` — **A1, combined**: NVIDIA Parakeet Realtime EOU 120M (FastConformer cache-aware RNNT,
  NVIDIA Open Model License) transcribes and emits ``<EOU>`` (or ``<EOB>``, end of a backchannel) itself:
  endpoint and final come from the same step; no punctuation / capitalisation.
- ``nemotron[-r<R>]+<endpoint>`` — **B1**: NVIDIA Nemotron Speech Streaming EN 0.6B (cache-aware RNNT,
  punctuation and capitalisation) decodes while the speaker talks (partials), the endpoint comes from the
  detector (``vad-500``, ``turn-400-p0.5-m800``, … see ``jevlite.voice.endpoint``); at the endpoint the
  model's lookahead is flushed with silence and the turn's text is final. ``-r<R>`` = right context in
  80 ms frames (0 / 1 / 6 / 13 → chunk 80 / 160 / 560 / 1120 ms), default 1.
- ``whisper+<endpoint>`` — **B2**: Whisper large-v3-turbo via faster-whisper (MIT), no partials; the turn's
  audio (from 300 ms before the turn opened to the endpoint) is decoded once at the endpoint, greedy
  (beam 1), English, no previous-text conditioning.

The NeMo loop follows NVIDIA's live voice agent (``NVIDIA-NeMo/labs-Voice-Agent``,
``nemo_voice_agent/pipecat/services/nemo/streaming_asr.py``, read 2026-11-20): greedy RNNT decoding,
``encoder.setup_streaming_params(chunk_size=model chunk // subsampling, shift_size=1 frame per 80 ms)``,
a rolling log-mel window of (pre-encode cache + model chunk) recomputed from the last 80 ms + one hop of
audio, ``encoder.cache_aware_stream_step(..., keep_all_outputs=False, drop_extra_pre_encoded=…)`` and
``decoding.rnnt_decoder_predictions_tensor(..., partial_hypotheses=…)``; float32 (cache-aware streaming
needs it); state reset after every end of turn. **Not run here** (no NeMo locally, no model runs by
Claude): the backends are checked against fakes in ``tests/test_voice_asr.py``; the first real check is
the stage-1 smoke on Modal. One stream per object; batching streams is stage 2.
"""

from __future__ import annotations

import math
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from .endpoint import CHUNK, SR, Endpoint, EndpointConfig, Endpointer, parse_setting

PARAKEET_EOU = "nvidia/parakeet_realtime_eou_120m-v1"
NEMOTRON = "nvidia/nemotron-speech-streaming-en-0.6b"
WHISPER = "large-v3-turbo"
EOU, EOB = "<EOU>", "<EOB>"
LOG_MEL_ZERO = -16.635  # log(2^-24): NeMo's log-mel value of digital silence
STEP_S = 0.08  # audio per streaming step (one 80 ms encoder frame)


@dataclass(frozen=True)
class AsrEvent:
    kind: str  # partial | endpoint | final
    audio_s: float
    text: str = ""
    compute_ms: float = 0.0
    detail: str = ""


class StreamingASR(Protocol):
    name: str

    def push(self, audio: np.ndarray) -> list[AsrEvent]: ...

    def flush(self) -> list[AsrEvent]: ...

    def reset(self) -> None: ...


class Rechunker:
    """Arbitrary pieces in → fixed ``size`` chunks out (the remainder waits for the next push)."""

    def __init__(self, size: int):
        self.size = size
        self.buf = np.zeros(0, np.float32)

    def push(self, audio: np.ndarray) -> list[np.ndarray]:
        self.buf = np.concatenate([self.buf, np.asarray(audio, np.float32).reshape(-1)])
        n = len(self.buf) // self.size
        out = [self.buf[i * self.size:(i + 1) * self.size] for i in range(n)]
        self.buf = self.buf[n * self.size:]
        return out

    def take_rest(self) -> np.ndarray:
        rest, self.buf = self.buf, np.zeros(0, np.float32)
        return rest


# ----------------------------------------------------------------------------------------------- NeMo


@dataclass
class NemoSpec:
    """A loaded cache-aware RNNT model and its streaming geometry (shared by all streams of the model)."""
    name: str
    model: object
    att_context_size: tuple[int, int]
    buffer_s: float  # log-mel window: pre-encode cache + model chunk
    step_s: float
    window_stride_s: float
    n_feat: int
    drop_extra_pre_encoded: int
    lookahead_steps: int  # silent steps that flush the right context at the end of a turn
    device: str


def _second(x):
    return x[1] if isinstance(x, (list, tuple)) else x


def load_nemo(name: str, right_context: int = 1, device: str = "cuda", left_context: int = 70) -> NemoSpec:
    """Load a NeMo cache-aware RNNT model for live streaming (see the module doc)."""
    import nemo.collections.asr as nemo_asr
    import torch
    from omegaconf import open_dict

    model = nemo_asr.models.ASRModel.from_pretrained(name, map_location=torch.device(device))
    if not isinstance(model, nemo_asr.models.EncDecRNNTModel):
        raise ValueError(f"{name}: expected an RNNT model, got {type(model).__name__}")
    att = [left_context, right_context]
    model.encoder.set_default_att_context_size(att_context_size=att)
    cfg = model.cfg.decoding
    with open_dict(cfg):
        cfg.strategy = "greedy"
        cfg.compute_timestamps = False
        cfg.preserve_alignments = False
        cfg.greedy.max_symbols = 10
        cfg.fused_batch_size = -1
    model.change_decoding_strategy(cfg)
    model.encoder.set_default_att_context_size(att_context_size=att)
    stride = float(model.cfg.preprocessor.window_stride)
    sub = int(model.cfg.encoder.subsampling_factor)
    chunk = int(_second(model.encoder.streaming_cfg.chunk_size))
    pre = int(_second(model.encoder.streaming_cfg.pre_encode_cache_size))
    per_step = math.ceil(np.trunc(STEP_S / stride) / sub)
    model.encoder.setup_streaming_params(chunk_size=chunk // sub, shift_size=per_step)
    model.eval()
    return NemoSpec(name=name, model=model, att_context_size=(left_context, right_context),
                    buffer_s=(pre + chunk) * stride, step_s=STEP_S, window_stride_s=stride,
                    n_feat=int(model.cfg.preprocessor.features),
                    drop_extra_pre_encoded=int(model.encoder.streaming_cfg.drop_extra_pre_encoded),
                    lookahead_steps=right_context, device=device)


def nemo_preprocessor(spec: NemoSpec):
    """A copy of the model's log-mel front end for the rolling window (no dither, no padding)."""
    import copy

    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    cfg = copy.deepcopy(spec.model.cfg.preprocessor)
    with open_dict(cfg):
        cfg.dither = 0.0
        cfg.pad_to = 0
    pre = nemo_asr.models.ASRModel.from_config_dict(cfg)
    return pre.to(spec.device).eval()


class FeatureWindow:
    """Rolling log-mel window for live cache-aware streaming (NVIDIA's ``CacheFeatureBufferer``): keeps
    ``buffer_s`` of audio; each ``update`` with ``step_s`` of new audio computes the log-mel of the last
    step + one hop (context for the first new frame), keeps its last ``step_s / stride`` frames and rolls
    them into a ``[n_feat, buffer_s / stride]`` window that starts as silence."""

    def __init__(self, preprocess: Callable, buffer_s: float, step_s: float, window_stride_s: float, n_feat: int,
                 device: str = "cpu"):
        import torch

        self.preprocess, self.device = preprocess, device
        self.look_back = int(round(window_stride_s * SR))
        self.step = int(round(step_s * SR))
        self.n_buffer = int(round(buffer_s * SR))
        self.frames = int(round(step_s / window_stride_s))
        self.n_window = int(round(buffer_s / window_stride_s))
        self.n_feat = n_feat
        self.samples = torch.zeros(self.n_buffer, dtype=torch.float32)
        self.window = torch.full((n_feat, self.n_window), LOG_MEL_ZERO, dtype=torch.float32, device=device)

    def reset(self) -> None:
        self.samples.zero_()
        self.window.fill_(LOG_MEL_ZERO)

    def update(self, chunk: np.ndarray):
        import torch

        x = torch.as_tensor(np.asarray(chunk, np.float32))
        if len(x) != self.step:
            raise ValueError(f"FeatureWindow expects {self.step} samples, got {len(x)}")
        self.samples = torch.cat([self.samples[len(x):], x])
        audio = self.samples[-(self.look_back + self.step):].unsqueeze(0).to(self.device)
        with torch.inference_mode():
            feats, _ = self.preprocess(input_signal=audio, length=torch.tensor([audio.shape[1]], device=self.device))
        feats = feats.reshape(self.n_feat, -1)
        if (extra := feats.shape[1] - self.frames - 1) > 0:  # the last frames may be incomplete
            feats = feats[:, :-extra]
        self.window = torch.cat([self.window[:, self.frames:], feats[:, -self.frames:].to(self.window)], dim=1)
        return self.window


class NemoStream:
    """One stream of a cache-aware RNNT model: ``step(80 ms of audio)`` → the new token ids."""

    def __init__(self, spec: NemoSpec, window: FeatureWindow | None = None):
        self.spec = spec
        self.model = spec.model
        self.window = window or FeatureWindow(nemo_preprocessor(spec), spec.buffer_s, spec.step_s,
                                              spec.window_stride_s, spec.n_feat, spec.device)
        self.chunk_samples = self.window.step
        self.reset()

    def reset(self) -> None:
        self.window.reset()
        self.cache = self.model.encoder.get_initial_cache_state(batch_size=1)
        self.hyps = None
        self.seen: list[int] = []

    def step(self, chunk: np.ndarray) -> list[int]:
        import torch

        feats = self.window.update(chunk).unsqueeze(0)
        length = torch.tensor([feats.shape[2]], device=feats.device)
        c1, c2, c3 = self.cache
        with torch.inference_mode():
            enc, enc_len, c1, c2, c3 = self.model.encoder.cache_aware_stream_step(
                processed_signal=feats, processed_signal_length=length, cache_last_channel=c1, cache_last_time=c2,
                cache_last_channel_len=c3, keep_all_outputs=False,
                drop_extra_pre_encoded=self.spec.drop_extra_pre_encoded)
            out = self.model.decoding.rnnt_decoder_predictions_tensor(enc, enc_len, return_hypotheses=True,
                                                                      partial_hypotheses=self.hyps)
        self.cache = (c1, c2, c3)
        self.hyps = out[0] if isinstance(out, tuple) else out
        return self._new_tokens(self.hyps[0].y_sequence)

    def _new_tokens(self, y) -> list[int]:
        """Tokens added by this step. With ``partial_hypotheses`` the hypothesis carries the whole sequence
        since the last reset (as in NeMo's file-based streaming script, whose last step holds the full text);
        if it does not extend the previous one, it is taken as this step's tokens only."""
        ids = [int(t) for t in (y.tolist() if hasattr(y, "tolist") else y)]
        new = ids[len(self.seen):] if ids[:len(self.seen)] == self.seen else ids
        self.seen = ids
        return new


def split_markers(tokenizer, ids: list[int]) -> tuple[list[int], str]:
    """(the ids without ``<EOU>`` / ``<EOB>`` pieces, the first marker seen or '')."""
    pieces = tokenizer.ids_to_tokens(ids) if ids else []
    marker = next((p for p in pieces if p in (EOU, EOB)), "")
    return [i for i, p in zip(ids, pieces) if p not in (EOU, EOB)], marker


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace(EOU, " ").replace(EOB, " ")).strip()


class ParakeetEOU:
    """A1, combined: the model's own ``<EOU>`` / ``<EOB>`` is the endpoint."""

    def __init__(self, stream: NemoStream, name: str = "parakeet-eou"):
        self.name, self.stream = name, stream
        self.tok = stream.model.tokenizer
        self.chunks = Rechunker(stream.chunk_samples)
        self.pos = 0
        self.ids: list[int] = []
        self.compute_ms = 0.0

    def reset(self) -> None:
        self.stream.reset()
        self.chunks = Rechunker(self.stream.chunk_samples)
        self.pos, self.ids, self.compute_ms = 0, [], 0.0

    def _text(self) -> str:
        return clean_text(self.tok.ids_to_text(self.ids)) if self.ids else ""

    def push(self, audio: np.ndarray) -> list[AsrEvent]:
        events = []
        for chunk in self.chunks.push(audio):
            t = time.perf_counter()
            new = self.stream.step(chunk)
            ms = (time.perf_counter() - t) * 1000
            self.compute_ms += ms
            self.pos += len(chunk)
            if not new:
                continue
            kept, marker = split_markers(self.tok, new)
            self.ids += kept
            if marker:
                events += [AsrEvent("endpoint", self.pos / SR, "", ms, "eou" if marker == EOU else "eob"),
                           AsrEvent("final", self.pos / SR, self._text(), 0.0)]
                self.stream.reset()  # as NVIDIA's voice agent: a fresh state for the next turn
                self.ids = []
            elif kept:
                events.append(AsrEvent("partial", self.pos / SR, self._text(), ms))
        return events

    def flush(self) -> list[AsrEvent]:
        if not self.ids:
            return []
        events = [AsrEvent("endpoint", self.pos / SR, "", 0.0, "eos"), AsrEvent("final", self.pos / SR, self._text())]
        self.ids = []
        return events


# ----------------------------------------------------------------------------------- B: ASR + detector


class TurnTranscriber(Protocol):
    name: str

    def accept(self, audio: np.ndarray) -> str | None:
        """Audio of the stream as it comes; the turn's partial text if it changed, else None."""

    def finalize(self, turn_audio: np.ndarray) -> str:
        """The endpoint: the turn's final text (``turn_audio``: the turn, for decoders that need it)."""

    def reset(self) -> None: ...


class NemotronTurns:
    """B1 transcriber: a cache-aware model decoding the whole stream; the turn ends where the detector says."""

    def __init__(self, stream: NemoStream, name: str = "nemotron"):
        self.name, self.stream = name, stream
        self.tok = stream.model.tokenizer
        self.chunks = Rechunker(stream.chunk_samples)
        self.ids: list[int] = []

    def reset(self) -> None:
        self.stream.reset()
        self.chunks = Rechunker(self.stream.chunk_samples)
        self.ids = []

    def _text(self) -> str:
        return clean_text(self.tok.ids_to_text(self.ids)) if self.ids else ""

    def accept(self, audio: np.ndarray) -> str | None:
        changed = False
        for chunk in self.chunks.push(audio):
            new, _ = split_markers(self.tok, self.stream.step(chunk))
            self.ids += new
            changed |= bool(new)
        return self._text() if changed else None

    def finalize(self, turn_audio: np.ndarray) -> str:
        rest = self.chunks.take_rest()
        silence = np.zeros(self.stream.chunk_samples, np.float32)
        tail = [np.pad(rest, (0, self.stream.chunk_samples - len(rest)))] if len(rest) else []
        for chunk in tail + [silence] * self.stream.spec.lookahead_steps:  # flush the right context
            self.ids += split_markers(self.tok, self.stream.step(chunk))[0]
        text = self._text()
        self.stream.reset()
        self.ids = []
        return text


class WhisperTurns:
    """B2 transcriber: faster-whisper decodes the turn's audio once, at the endpoint."""

    def __init__(self, model, name: str = "whisper", beam_size: int = 1, language: str = "en"):
        self.name, self.model, self.beam_size, self.language = name, model, beam_size, language

    @classmethod
    def load(cls, model: str = WHISPER, device: str = "cuda", compute_type: str = "float16", **kw) -> "WhisperTurns":
        from faster_whisper import WhisperModel

        return cls(WhisperModel(model, device=device, compute_type=compute_type), **kw)

    def reset(self) -> None:
        pass

    def accept(self, audio: np.ndarray) -> str | None:
        return None

    def finalize(self, turn_audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(np.asarray(turn_audio, np.float32), language=self.language,
                                            beam_size=self.beam_size, vad_filter=False,
                                            condition_on_previous_text=False, without_timestamps=True)
        return " ".join(s.text.strip() for s in segments).strip()


class SeparateASR:
    """B, separate: a ``TurnTranscriber`` + Silero VAD + an ``Endpointer`` setting."""

    def __init__(self, transcriber: TurnTranscriber, config: EndpointConfig, vad: Callable[[np.ndarray], float],
                 turn_model: Callable[[np.ndarray], float] | None = None, preroll_s: float = 0.3,
                 max_turn_s: float = 120.0, name: str | None = None):
        self.transcriber, self.config, self.vad, self.turn_model = transcriber, config, vad, turn_model
        self.name = name or f"{transcriber.name}+{config.name}"
        self.preroll, self.keep = int(preroll_s * SR), int((max_turn_s + preroll_s) * SR)
        self.reset()

    def reset(self) -> None:
        self.transcriber.reset()
        if hasattr(self.vad, "reset"):
            self.vad.reset()
        self.endpointer = Endpointer(self.config, self.turn_model)
        self.chunks = Rechunker(CHUNK)
        self.history: deque[tuple[int, np.ndarray]] = deque()
        self.pos = 0
        self.compute_ms = 0.0

    def _turn_audio(self, a: int, b: int) -> np.ndarray:
        parts = [c[max(0, a - s):max(0, b - s)] for s, c in self.history if s < b and s + len(c) > a]
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def _end(self, e: Endpoint, detect_ms: float) -> list[AsrEvent]:
        t = time.perf_counter()
        text = self.transcriber.finalize(self._turn_audio(max(0, e.start_sample - self.preroll), e.sample))
        ms = (time.perf_counter() - t) * 1000
        self.compute_ms += ms
        return [AsrEvent("endpoint", e.sample / SR, "", detect_ms + e.decide_ms, e.reason),
                AsrEvent("final", e.sample / SR, text, ms)]

    def push(self, audio: np.ndarray) -> list[AsrEvent]:
        events = []
        for chunk in self.chunks.push(audio):
            self.history.append((self.pos, chunk))
            self.pos += len(chunk)
            while self.history and self.history[0][0] + len(self.history[0][1]) < self.pos - self.keep:
                self.history.popleft()
            t = time.perf_counter()
            partial = self.transcriber.accept(chunk)
            asr_ms = (time.perf_counter() - t) * 1000
            if partial is not None:
                events.append(AsrEvent("partial", self.pos / SR, partial, asr_ms))
            t = time.perf_counter()
            e = self.endpointer.push(chunk, float(self.vad(chunk)))
            detect_ms = (time.perf_counter() - t) * 1000 - (e.decide_ms if e else 0.0)
            self.compute_ms += asr_ms + detect_ms + (e.decide_ms if e else 0.0)
            if e is not None:
                events += self._end(e, detect_ms)
        return events

    def flush(self) -> list[AsrEvent]:
        e = self.endpointer.close()
        return [] if e is None else self._end(e, 0.0)


# ----------------------------------------------------------------------------------------- the systems


@dataclass(frozen=True)
class SystemSpec:
    name: str
    asr: str  # parakeet-eou | nemotron | whisper
    right_context: int | None = None  # NeMo models: lookahead in 80 ms frames
    endpoint: str | None = None  # B systems: the detector setting


_SYSTEM = re.compile(r"(parakeet-eou|nemotron|whisper)(?:-r(\d+))?(?:\+(.+))?")


def parse_system(name: str) -> SystemSpec:
    """``parakeet-eou[-r<R>]`` · ``nemotron[-r<R>]+<endpoint>`` · ``whisper+<endpoint>`` (module doc)."""
    m = _SYSTEM.fullmatch(name.strip())
    if not m:
        raise ValueError(f"ASR system {name!r}: expected parakeet-eou, nemotron[-r<R>]+<endpoint> or whisper+<endpoint>")
    asr, r, ep = m.group(1), m.group(2), m.group(3)
    if asr == "parakeet-eou" and ep:
        raise ValueError(f"{name}: parakeet-eou decides the end of turn itself, no +<endpoint>")
    if asr != "parakeet-eou" and not ep:
        raise ValueError(f"{name}: {asr} needs an endpoint detector, e.g. {asr}+vad-500")
    if asr == "whisper" and r:
        raise ValueError(f"{name}: -r<R> is for the NeMo models only")
    if r is not None and int(r) not in (0, 1, 6, 13):
        raise ValueError(f"{name}: right context must be 0, 1, 6 or 13 (80 ms frames)")
    if ep:
        parse_setting(ep)
    right = int(r) if r is not None else (1 if asr != "whisper" else None)
    return SystemSpec(name=name.strip(), asr=asr, right_context=right, endpoint=ep)


class ModelPool:
    """Loads each model once and builds streams of the named systems on top (one object per stream)."""

    def __init__(self, device: str = "cuda", whisper_compute_type: str = "float16", turn_variant: str = "gpu"):
        self.device, self.whisper_compute_type, self.turn_variant = device, whisper_compute_type, turn_variant
        self.nemo: dict[tuple[str, int], NemoSpec] = {}
        self.whisper = None
        self.smart_turn = None

    def _nemo(self, name: str, right: int) -> NemoSpec:
        if (name, right) not in self.nemo:
            self.nemo[(name, right)] = load_nemo(name, right, self.device)
        return self.nemo[(name, right)]

    def stream(self, system: str | SystemSpec) -> StreamingASR:
        from .endpoint import SileroVAD, SmartTurn

        s = parse_system(system) if isinstance(system, str) else system
        if s.asr == "parakeet-eou":
            return ParakeetEOU(NemoStream(self._nemo(PARAKEET_EOU, s.right_context)), name=s.name)
        if s.asr == "nemotron":
            tr: TurnTranscriber = NemotronTurns(NemoStream(self._nemo(NEMOTRON, s.right_context)))
        else:
            if self.whisper is None:
                self.whisper = WhisperTurns.load(device=self.device, compute_type=self.whisper_compute_type)
            tr = self.whisper
        config = parse_setting(s.endpoint)
        turn = None
        if config.smart_turn:
            if self.smart_turn is None:
                self.smart_turn = SmartTurn(self.turn_variant)
            turn = self.smart_turn
        return SeparateASR(tr, config, SileroVAD(), turn, name=s.name)
