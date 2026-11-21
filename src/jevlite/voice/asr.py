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
  detector (``vad-500``, … see ``jevlite.voice.endpoint``). The model decodes the stream continuously
  (no reset at the turns): a token belongs to the turn if it was emitted by the step that ended at or
  before endpoint + the lookahead steps, so the final waits one 160 ms step of real audio after the
  endpoint at R=1 (the model's lookahead), which is its honest cost; at the end of the stream the
  lookahead is flushed with silence.
  ``-r<R>`` = right context in 80 ms frames (0 / 1 / 6 / 13 → chunk 80 / 160 / 560 / 1120 ms), default 1.
  Because the decoding does not depend on the detector, a recorded token timeline (``RecordedSteps``)
  can be replayed under any detector setting without the GPU (stage-1 quality study).
- ``whisper+<endpoint>`` — **B2**: Whisper large-v3-turbo via faster-whisper (MIT), no partials; the turn's
  audio (from 300 ms before the turn opened to the endpoint) is decoded once at the endpoint, greedy
  (beam 1), English, no previous-text conditioning.

The NeMo loop follows NVIDIA's live voice agent (``NVIDIA-NeMo/labs-Voice-Agent``,
``nemo_voice_agent/pipecat/services/nemo/streaming_asr.py``, read 2026-11-20) and NeMo 3.0.0's own
streaming pipeline (``nemo/collections/asr/inference/pipelines/cache_aware_rnnt_pipeline.py``, read from
the wheel 2026-11-20: same geometry, ``cache_aware_stream_step`` returns (encoded, lengths, 3 caches), the
hypothesis carries the tokens since the start, new tokens = past the previous length): greedy RNNT decoding,
``encoder.setup_streaming_params(chunk_size=model chunk // subsampling, shift_size=frames per step)``,
a rolling log-mel window of (pre-encode cache + model chunk) recomputed from the last step + one hop of
audio (step = the model chunk, 160 ms at R=1, as NeMo's own pipeline; NVIDIA's agent steps 80 ms, which
broke Nemotron here), ``encoder.cache_aware_stream_step(..., keep_all_outputs=False, drop_extra_pre_encoded=…)`` and
``decoding.rnnt_decoder_predictions_tensor(..., partial_hypotheses=…)``; float32 (cache-aware streaming
needs it); A1 resets the state after every ``<EOU>`` / ``<EOB>`` (as NVIDIA's agent). Text = the
SentencePiece pieces joined (``▁`` → space), the same live and in a replay. **Not run here** (no NeMo locally, no model runs by
Claude): the backends are checked against fakes in ``tests/test_voice_asr.py``; the first real check is
the stage-1 smoke on Modal. ``BatchedNemo`` steps many streams in one call (each slot reset on its own);
``NemoStream`` is its one-slot case, and ``record_batched`` records the stage-1 timelines with it.
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
STEP_S = 0.08  # one encoder frame (80 ms); the default step is the model chunk (``load_nemo``): 160 ms at R=1


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


def load_nemo(name: str, right_context: int = 1, device: str = "cuda", left_context: int = 70,
              step_s: float | None = None, cuda_graphs: bool = True) -> NemoSpec:
    """Load a NeMo cache-aware RNNT model for live streaming (see the module doc). ``step_s``: audio per
    step — None (default) = the model's own chunk (current + right context frames, NeMo's default, no
    overlap: 0.16 s at right context 1), or 0.08 (one encoder frame, as NVIDIA's voice agent; the chunk then
    overlaps the previous step's lookahead — Nemotron loses most words that way: stage-1 diag of
    2026-11-20, WER 69% vs 1.8% at 0.16 s, Parakeet 1.2% vs 0.6%). ``cuda_graphs``: NeMo's CUDA-graph greedy
    decoder (default on; no effect on the text in the diag)."""
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
        cfg.greedy.use_cuda_graph_decoder = cuda_graphs
        cfg.fused_batch_size = -1
    model.change_decoding_strategy(cfg)
    model.encoder.set_default_att_context_size(att_context_size=att)
    stride = float(model.cfg.preprocessor.window_stride)
    sub = int(model.cfg.encoder.subsampling_factor)
    chunk = int(_second(model.encoder.streaming_cfg.chunk_size))
    pre = int(_second(model.encoder.streaming_cfg.pre_encode_cache_size))
    if step_s is None:
        step_s = round(chunk * stride, 6)
    per_step = math.ceil(np.trunc(step_s / stride) / sub)
    model.encoder.setup_streaming_params(chunk_size=chunk // sub, shift_size=per_step)
    model.eval()
    return NemoSpec(name=name, model=model, att_context_size=(left_context, right_context),
                    buffer_s=(pre + chunk) * stride, step_s=step_s, window_stride_s=stride,
                    n_feat=int(model.cfg.preprocessor.features),
                    drop_extra_pre_encoded=int(model.encoder.streaming_cfg.drop_extra_pre_encoded),
                    lookahead_steps=math.ceil(right_context * sub * stride / step_s - 1e-9), device=device)


def nemo_preprocessor(spec: NemoSpec):
    """A copy of the model's log-mel front end for the rolling window: no dither, no padding, no
    normalisation (as NeMo 3.0's ``make_preprocessor_deterministic`` for its own streaming pipeline)."""
    import copy

    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    cfg = copy.deepcopy(spec.model.cfg.preprocessor)
    with open_dict(cfg):
        cfg.dither = 0.0
        cfg.pad_to = 0
        cfg.normalize = "None"
    pre = nemo_asr.models.ASRModel.from_config_dict(cfg)
    return pre.to(spec.device).eval()


class BatchedFeatureWindow:
    """Rolling log-mel windows for ``batch`` live streams (NVIDIA's ``CacheFeatureBufferer``), one
    preprocessor call per step for all of them: each slot keeps ``buffer_s`` of audio; each ``update`` with
    ``step_s`` of new audio per slot computes the log-mel of the last step + one hop (context for the first
    new frame), keeps its last ``step_s / stride`` frames and rolls them into a ``[n_feat, buffer_s / stride]``
    window that starts as silence. All slots have the same length, so the batch adds no padding and each
    slot's features equal a single-stream window's."""

    def __init__(self, preprocess: Callable, buffer_s: float, step_s: float, window_stride_s: float, n_feat: int,
                 batch: int = 1, device: str = "cpu"):
        import torch

        self.preprocess, self.device, self.batch = preprocess, device, batch
        self.look_back = int(round(window_stride_s * SR))
        self.step = int(round(step_s * SR))
        self.n_buffer = int(round(buffer_s * SR))
        self.frames = int(round(step_s / window_stride_s))
        self.n_window = int(round(buffer_s / window_stride_s))
        self.n_feat = n_feat
        self.samples = torch.zeros(batch, self.n_buffer, dtype=torch.float32)
        self.windows = torch.full((batch, n_feat, self.n_window), LOG_MEL_ZERO, dtype=torch.float32, device=device)

    def reset(self, slots: list[int] | None = None) -> None:
        """All slots, or only ``slots`` (a new stream / a new turn there), back to silence."""
        if slots is None:
            self.samples.zero_()
            self.windows.fill_(LOG_MEL_ZERO)
            return
        for s in slots:
            self.samples[s].zero_()
            self.windows[s].fill_(LOG_MEL_ZERO)

    def update_batch(self, chunks: np.ndarray):
        """``chunks``: ``[batch, step]`` samples → the windows ``[batch, n_feat, n_window]``."""
        import torch

        x = torch.as_tensor(np.asarray(chunks, np.float32)).reshape(self.batch, -1)
        if x.shape[1] != self.step:
            raise ValueError(f"FeatureWindow expects {self.step} samples, got {x.shape[1]}")
        self.samples = torch.cat([self.samples[:, x.shape[1]:], x], dim=1)
        audio = self.samples[:, -(self.look_back + self.step):].to(self.device)
        lengths = torch.full((self.batch,), audio.shape[1], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            feats, _ = self.preprocess(input_signal=audio, length=lengths)
        feats = feats.reshape(self.batch, self.n_feat, -1)
        if (extra := feats.shape[2] - self.frames - 1) > 0:  # the last frames may be incomplete
            feats = feats[:, :, :-extra]
        self.windows = torch.cat([self.windows[:, :, self.frames:], feats[:, :, -self.frames:].to(self.windows)],
                                 dim=2)
        return self.windows

    update = update_batch


class FeatureWindow(BatchedFeatureWindow):
    """One stream's rolling log-mel window (``BatchedFeatureWindow`` with one slot): ``update(chunk)`` →
    ``[n_feat, n_window]``."""

    def __init__(self, preprocess: Callable, buffer_s: float, step_s: float, window_stride_s: float, n_feat: int,
                 device: str = "cpu"):
        super().__init__(preprocess, buffer_s, step_s, window_stride_s, n_feat, 1, device)

    @property
    def window(self):
        return self.windows[0]

    def update(self, chunk: np.ndarray):
        return self.update_batch(np.asarray(chunk, np.float32).reshape(1, -1))[0]


class BatchedNemo:
    """``batch`` independent streams of one cache-aware RNNT model stepped together:
    ``step(chunks [batch, step samples])`` → the new SentencePiece pieces per slot; ``last_ms`` = the compute
    of that batched step. ``reset(slots)`` starts slots afresh (a new stream there, or Parakeet's new turn):
    their feature window back to silence, their encoder cache zeroed with cache length 0 (= the initial state:
    the encoder masks each slot by its own cache length) and their hypothesis dropped (NeMo's batched greedy
    decoder starts a ``None`` entry from the start-of-sequence state)."""

    def __init__(self, spec: NemoSpec, batch: int = 1, window: BatchedFeatureWindow | None = None,
                 preprocess: Callable | None = None):
        self.spec, self.batch = spec, batch
        self.model = spec.model
        self.window = window or BatchedFeatureWindow(preprocess or nemo_preprocessor(spec), spec.buffer_s,
                                                     spec.step_s, spec.window_stride_s, spec.n_feat, batch,
                                                     spec.device)
        if self.window.batch != batch:
            raise ValueError(f"window for {self.window.batch} slots, model for {batch}")
        self.chunk_samples = self.window.step
        self.tokenizer = spec.model.tokenizer
        self.last_ms = 0.0
        self.non_prefix_steps = 0
        self.reset()

    def reset(self, slots: list[int] | None = None) -> None:
        if slots is None:
            self.window.reset()
            self.cache = self.model.encoder.get_initial_cache_state(batch_size=self.batch)
            self.hyps = None
            self.seen: list[list[int]] = [[] for _ in range(self.batch)]
            return
        import torch

        self.window.reset(slots)
        c1, c2, c3 = self.cache
        idx = torch.as_tensor(slots, dtype=torch.long, device=c1.device)
        with torch.inference_mode():  # the caches are inference tensors (made in ``step``)
            c1.index_fill_(1, idx, 0.0)  # [layers, batch, …]
            c2.index_fill_(1, idx, 0.0)
            c3.index_fill_(0, idx.to(c3.device), 0)  # [batch]
        for s in slots:
            if self.hyps is not None:
                self.hyps[s] = None
            self.seen[s] = []

    def step(self, chunks: np.ndarray) -> list[list[str]]:
        import torch

        t = time.perf_counter()
        feats = self.window.update_batch(chunks)
        length = torch.full((self.batch,), feats.shape[2], dtype=torch.long, device=feats.device)
        c1, c2, c3 = self.cache
        with torch.inference_mode():
            enc, enc_len, c1, c2, c3 = self.model.encoder.cache_aware_stream_step(
                processed_signal=feats, processed_signal_length=length, cache_last_channel=c1, cache_last_time=c2,
                cache_last_channel_len=c3, keep_all_outputs=False,
                drop_extra_pre_encoded=self.spec.drop_extra_pre_encoded)
            out = self.model.decoding.rnnt_decoder_predictions_tensor(enc, enc_len, return_hypotheses=True,
                                                                      partial_hypotheses=self.hyps)
        self.cache = (c1, c2, c3)
        self.hyps = list(out[0] if isinstance(out, tuple) else out)
        pieces = []
        for s, h in enumerate(self.hyps):  # .tolist() waits for the GPU
            new = self._new_tokens(s, h.y_sequence)
            pieces.append(self.tokenizer.ids_to_tokens(new) if new else [])
        self.last_ms = (time.perf_counter() - t) * 1000
        return pieces

    def _new_tokens(self, slot: int, y) -> list[int]:
        """Tokens added by this step. With ``partial_hypotheses`` the hypothesis carries the whole sequence
        since the slot's last reset (as in NeMo's file-based streaming script, whose last step holds the full
        text); if it does not extend the previous one, it is taken as this step's tokens only (counted)."""
        ids = [int(t) for t in (y.tolist() if hasattr(y, "tolist") else y)]
        seen = self.seen[slot]
        if ids[:len(seen)] == seen:
            new = ids[len(seen):]
        else:
            new = ids
            self.non_prefix_steps += 1
        self.seen[slot] = ids
        return new


class NemoStream:
    """One stream of a cache-aware RNNT model (``BatchedNemo`` with one slot): ``step(one step of audio)`` →
    the new tokens as SentencePiece pieces; ``last_ms`` = the compute of that step."""

    def __init__(self, spec: NemoSpec, window: FeatureWindow | None = None):
        self.spec, self.model, self.tokenizer = spec, spec.model, spec.model.tokenizer
        self.batched = BatchedNemo(spec, 1, window)
        self.window = self.batched.window
        self.chunk_samples = self.batched.chunk_samples
        self.last_ms = 0.0

    @property
    def non_prefix_steps(self) -> int:
        return self.batched.non_prefix_steps

    def reset(self) -> None:
        self.batched.reset()

    def step(self, chunk: np.ndarray) -> list[str]:
        pieces = self.batched.step(np.asarray(chunk, np.float32).reshape(1, -1))[0]
        self.last_ms = self.batched.last_ms
        return pieces


def split_markers(pieces: list[str]) -> tuple[list[str], str]:
    """(the pieces without ``<EOU>`` / ``<EOB>``, the first marker seen or '')."""
    marker = next((p for p in pieces if p in (EOU, EOB)), "")
    return [p for p in pieces if p not in (EOU, EOB)], marker


def pieces_text(pieces: list[str]) -> str:
    """SentencePiece pieces → text (``▁`` starts a word)."""
    return re.sub(r"\s+", " ", "".join(pieces).replace("▁", " ")).strip()


class ParakeetEOU:
    """A1, combined: the model's own ``<EOU>`` / ``<EOB>`` is the endpoint."""

    def __init__(self, stream: NemoStream, name: str = "parakeet-eou"):
        self.name, self.stream = name, stream
        self.reset()

    def reset(self) -> None:
        self.stream.reset()
        self.chunks = Rechunker(self.stream.chunk_samples)
        self.pos, self.pieces, self.compute_ms = 0, [], 0.0

    def push(self, audio: np.ndarray) -> list[AsrEvent]:
        events = []
        for chunk in self.chunks.push(audio):
            new = self.stream.step(chunk)
            ms = self.stream.last_ms
            self.compute_ms += ms
            self.pos += len(chunk)
            if not new:
                continue
            kept, marker = split_markers(new)
            self.pieces += kept
            if marker:
                events += [AsrEvent("endpoint", self.pos / SR, "", ms, "eou" if marker == EOU else "eob"),
                           AsrEvent("final", self.pos / SR, pieces_text(self.pieces), 0.0)]
                self.stream.reset()  # as NVIDIA's voice agent: a fresh state for the next turn
                self.pieces = []
            elif kept:
                events.append(AsrEvent("partial", self.pos / SR, pieces_text(self.pieces), ms))
        return events

    def flush(self) -> list[AsrEvent]:
        if not self.pieces:
            return []
        text, self.pieces = pieces_text(self.pieces), []
        return [AsrEvent("endpoint", self.pos / SR, "", 0.0, "eos"), AsrEvent("final", self.pos / SR, text)]


# ----------------------------------------------------------------------------------- B: ASR + detector


class TurnTranscriber(Protocol):
    name: str
    lag_samples: int  # audio the transcriber needs after the endpoint before the turn's text is final
    processed: int  # samples consumed so far
    compute_ms: float

    def accept(self, audio: np.ndarray) -> str | None:
        """Audio of the stream as it comes; the open turn's text if it changed, else None."""

    def finalize(self, turn_audio: np.ndarray, until: int) -> str:
        """The turn's final text: ``turn_audio`` for decoders that re-decode the turn; ``until`` = the stream
        sample up to which streaming output belongs to this turn."""

    def finish(self) -> None:
        """End of the stream: emit whatever the lookahead still holds."""

    def reset(self) -> None: ...


class RecordedSteps:
    """A recorded token timeline of a ``NemoStream`` (``steps``: pieces and compute ms per step), replayed
    step by step in place of the model; the audio passed to ``step`` is ignored."""

    def __init__(self, steps: list[tuple[list[str], float]], chunk_samples: int = int(STEP_S * SR),
                 rewind: bool = True):
        """``rewind=False``: ``reset`` keeps the position — for a Parakeet timeline recorded with its state
        reset after every end of turn already in it (``record_batched(..., reset_on_marker=True)``), which
        ``ParakeetEOU`` resets again on replay."""
        self.steps, self.chunk_samples, self.rewind = steps, chunk_samples, rewind
        self.i, self.last_ms = 0, 0.0

    def reset(self) -> None:
        if self.rewind:
            self.i = 0
        self.last_ms = 0.0

    def step(self, chunk: np.ndarray) -> list[str]:
        pieces, self.last_ms = self.steps[self.i] if self.i < len(self.steps) else ([], 0.0)
        self.i += 1
        return list(pieces)


def record_steps(stream: NemoStream, audio: np.ndarray, flush_steps: int = 0) -> list[tuple[list[str], float]]:
    """Run a ``NemoStream`` over the whole stream and keep its timeline: the same steps as a live
    ``NemotronTurns`` (the last partial chunk zero-padded, as ``finish`` does, then ``flush_steps`` of
    silence = the right context), so ``RecordedSteps`` replays it exactly."""
    stream.reset()
    n = stream.chunk_samples
    audio = np.asarray(audio, np.float32)
    chunks = [audio[i:i + n] for i in range(0, len(audio), n)]
    if chunks and len(chunks[-1]) < n:
        chunks[-1] = np.pad(chunks[-1], (0, n - len(chunks[-1])))
    chunks += [np.zeros(n, np.float32)] * flush_steps
    out = []
    for c in chunks:
        pieces = stream.step(np.asarray(c, np.float32))
        out.append((pieces, stream.last_ms))
    return out


def stream_steps(audio: np.ndarray, n: int, pad_last: bool = True, flush_steps: int = 0) -> np.ndarray:
    """A stream cut into ``n``-sample steps as the live systems step it: B1 (``pad_last``) zero-pads the
    last partial step and adds ``flush_steps`` of silence (``NemotronTurns.finish``); A1 drops the partial
    step (``ParakeetEOU`` steps full chunks only). → ``[steps, n]``."""
    audio = np.asarray(audio, np.float32).reshape(-1)
    k = -(-len(audio) // n) if pad_last else len(audio) // n
    out = np.zeros((k + (flush_steps if pad_last else 0), n), np.float32)
    flat = out.reshape(-1)
    m = min(len(audio), k * n)
    flat[:m] = audio[:m]
    return out


def record_batched(batched: BatchedNemo, streams: list[np.ndarray], pad_last: bool = True, flush_steps: int = 0,
                   reset_on_marker: bool = False, log: Callable[[str], None] | None = None,
                   log_every: int = 2000) -> tuple[list[list[tuple[list[str], float]]], dict]:
    """The timelines of many streams through one ``BatchedNemo``, the same per stream as ``record_steps``
    (B1) or a live ``ParakeetEOU`` (A1, ``reset_on_marker``: the slot's state is reset after a step that
    emits ``<EOU>`` / ``<EOB>``, before its next step, as ``ParakeetEOU`` does). Streams go to free slots
    longest first; a slot whose stream ends is reset and takes the next one; slots with nothing left step
    silence and their output is dropped. Each stream's step is charged the batched step's compute divided
    by the slots that carried a stream in it (the GPU cost per stream at this batch size).
    → (timelines in the order of ``streams``, info: batch, steps, step ms p50 / p95, wall s)."""
    n, b = batched.chunk_samples, batched.batch
    steps = [stream_steps(a, n, pad_last, flush_steps) for a in streams]
    queue = deque(sorted(range(len(streams)), key=lambda i: -len(steps[i])))
    slot: list[int | None] = [None] * b
    pos = [0] * b
    out: list[list[tuple[list[str], float]]] = [[] for _ in streams]
    batched.reset()
    for s in range(b):
        slot[s] = queue.popleft() if queue else None
    step_ms, t0, k = [], time.perf_counter(), 0
    chunks = np.zeros((b, n), np.float32)
    while any(i is not None for i in slot):
        chunks.fill(0.0)
        for s, i in enumerate(slot):
            if i is not None:
                chunks[s] = steps[i][pos[s]]
        pieces = batched.step(chunks)
        step_ms.append(batched.last_ms)
        busy = sum(i is not None for i in slot)
        share = batched.last_ms / busy
        fresh = []
        for s, i in enumerate(slot):
            if i is None:
                continue
            out[i].append((pieces[s], share))
            pos[s] += 1
            if pos[s] == len(steps[i]):
                slot[s], pos[s] = (queue.popleft() if queue else None), 0
                fresh.append(s)
            elif reset_on_marker and split_markers(pieces[s])[1]:
                fresh.append(s)
        if fresh:
            batched.reset(fresh)
        k += 1
        if log and k % log_every == 0:
            log(f"  step {k}: {busy} streams busy, {len(queue)} waiting, {np.median(step_ms[-log_every:]):.1f} ms/step")
    ms = np.array(step_ms) if step_ms else np.zeros(1)
    info = {"batch": b, "steps": len(step_ms), "step_ms_p50": float(np.percentile(ms, 50)),
            "step_ms_p95": float(np.percentile(ms, 95)), "wall_s": time.perf_counter() - t0,
            "non_prefix_steps": batched.non_prefix_steps}
    return out, info


class NemotronTurns:
    """B1 transcriber: a cache-aware model decoding the whole stream continuously (module doc)."""

    def __init__(self, stepper: NemoStream | RecordedSteps, lag_steps: int, name: str = "nemotron"):
        self.name, self.stepper = name, stepper
        self.lag_samples = lag_steps * stepper.chunk_samples
        self.lag_steps = lag_steps
        self.reset()

    def reset(self) -> None:
        self.stepper.reset()
        self.chunks = Rechunker(self.stepper.chunk_samples)
        self.tokens: deque[tuple[int, str]] = deque()  # (stream sample at the end of the emitting step, piece)
        self.processed = 0
        self.compute_ms = 0.0

    def _step(self, chunk: np.ndarray) -> bool:
        pieces, _ = split_markers(self.stepper.step(chunk))
        self.processed += len(chunk)
        self.compute_ms += self.stepper.last_ms
        self.tokens.extend((self.processed, p) for p in pieces)
        return bool(pieces)

    def accept(self, audio: np.ndarray) -> str | None:
        changed = False
        for chunk in self.chunks.push(audio):
            changed |= self._step(chunk)
        return pieces_text([p for _, p in self.tokens]) if changed else None

    def finalize(self, turn_audio: np.ndarray, until: int) -> str:
        out = []
        while self.tokens and self.tokens[0][0] <= until:
            out.append(self.tokens.popleft()[1])
        return pieces_text(out)

    def finish(self) -> None:
        rest = self.chunks.take_rest()
        n = self.stepper.chunk_samples
        for chunk in ([np.pad(rest, (0, n - len(rest)))] if len(rest) else []) + [np.zeros(n, np.float32)] * self.lag_steps:
            self._step(chunk)


class WhisperTurns:
    """B2 transcriber: faster-whisper decodes the turn's audio once, at the endpoint."""

    lag_samples = 0

    def __init__(self, model, name: str = "whisper", beam_size: int = 1, language: str = "en"):
        self.name, self.model, self.beam_size, self.language = name, model, beam_size, language
        self.processed, self.compute_ms = 0, 0.0

    @classmethod
    def load(cls, model: str = WHISPER, device: str = "cuda", compute_type: str = "float16", **kw) -> "WhisperTurns":
        from faster_whisper import WhisperModel

        return cls(WhisperModel(model, device=device, compute_type=compute_type), **kw)

    def reset(self) -> None:
        self.processed, self.compute_ms = 0, 0.0

    def accept(self, audio: np.ndarray) -> str | None:
        self.processed += len(audio)
        return None

    def finalize(self, turn_audio: np.ndarray, until: int) -> str:
        t = time.perf_counter()
        segments, _ = self.model.transcribe(np.asarray(turn_audio, np.float32), language=self.language,
                                            beam_size=self.beam_size, vad_filter=False,
                                            condition_on_previous_text=False, without_timestamps=True)
        text = " ".join(s.text.strip() for s in segments).strip()  # the generator decodes here
        self.compute_ms += (time.perf_counter() - t) * 1000
        return text

    def finish(self) -> None:
        pass


class SeparateASR:
    """B, separate: a ``TurnTranscriber`` + Silero VAD + an ``Endpointer`` setting. The final of a turn is
    released once the transcriber has consumed ``lag_samples`` past the endpoint (at once for Whisper)."""

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
        self.pending: deque[Endpoint] = deque()
        self.pos = 0
        self.detect_ms = 0.0

    @property
    def compute_ms(self) -> float:
        return self.detect_ms + self.transcriber.compute_ms

    def _turn_audio(self, a: int, b: int) -> np.ndarray:
        parts = [c[max(0, a - s):max(0, b - s)] for s, c in self.history if s < b and s + len(c) > a]
        return np.concatenate(parts) if parts else np.zeros(0, np.float32)

    def _release(self, last_step_ms: float, done: bool = False) -> list[AsrEvent]:
        out = []
        lag = self.transcriber.lag_samples
        while self.pending and (done or self.transcriber.processed >= self.pending[0].sample + lag):
            e = self.pending.popleft()
            before = self.transcriber.compute_ms
            text = self.transcriber.finalize(self._turn_audio(max(0, e.start_sample - self.preroll), e.sample),
                                             e.sample + lag)
            ms = self.transcriber.compute_ms - before + (last_step_ms if lag else 0.0)
            at = e.sample if not lag else max(e.sample, min(self.transcriber.processed, e.sample + lag))
            out.append(AsrEvent("final", at / SR, text, ms))
        return out

    def push(self, audio: np.ndarray) -> list[AsrEvent]:
        events = []
        for chunk in self.chunks.push(audio):
            self.history.append((self.pos, chunk))
            self.pos += len(chunk)
            while self.history and self.history[0][0] + len(self.history[0][1]) < self.pos - self.keep:
                self.history.popleft()
            before = self.transcriber.compute_ms
            partial = self.transcriber.accept(chunk)
            asr_ms = self.transcriber.compute_ms - before
            if partial is not None:
                events.append(AsrEvent("partial", self.transcriber.processed / SR, partial, asr_ms))
            t = time.perf_counter()
            e = self.endpointer.push(chunk, float(self.vad(chunk)))
            detect_ms = (time.perf_counter() - t) * 1000
            self.detect_ms += detect_ms
            if e is not None:
                events.append(AsrEvent("endpoint", e.sample / SR, "", detect_ms, e.reason))
                self.pending.append(e)
            events += self._release(asr_ms)
        return events

    def flush(self) -> list[AsrEvent]:
        events = []
        e = self.endpointer.close()
        if e is not None:
            events.append(AsrEvent("endpoint", e.sample / SR, "", 0.0, e.reason))
            self.pending.append(e)
        before = self.transcriber.compute_ms
        self.transcriber.finish()
        return events + self._release(self.transcriber.compute_ms - before, done=True)


class ReplayVAD:
    """Recorded P(speech) per 512-sample chunk, served in order (the stream's VAD run once, on the CPU)."""

    def __init__(self, probs: np.ndarray):
        self.probs = np.asarray(probs, np.float32)
        self.i = 0

    def reset(self) -> None:
        self.i = 0

    def __call__(self, chunk: np.ndarray) -> float:
        p = float(self.probs[self.i]) if self.i < len(self.probs) else 0.0
        self.i += 1
        return p


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
        self.specs: dict[tuple[str, int], NemoSpec] = {}
        self.whisper = None
        self.smart_turn = None

    def nemo(self, name: str, right: int) -> NemoSpec:
        if (name, right) not in self.specs:
            self.specs[(name, right)] = load_nemo(name, right, self.device)
        return self.specs[(name, right)]

    def whisper_turns(self) -> WhisperTurns:
        if self.whisper is None:
            self.whisper = WhisperTurns.load(device=self.device, compute_type=self.whisper_compute_type)
        return self.whisper

    def stream(self, system: str | SystemSpec) -> StreamingASR:
        from .endpoint import SileroVAD, SmartTurn

        s = parse_system(system) if isinstance(system, str) else system
        if s.asr == "parakeet-eou":
            return ParakeetEOU(NemoStream(self.nemo(PARAKEET_EOU, s.right_context)), name=s.name)
        if s.asr == "nemotron":
            spec = self.nemo(NEMOTRON, s.right_context)
            tr: TurnTranscriber = NemotronTurns(NemoStream(spec), spec.lookahead_steps)
        else:
            tr = self.whisper_turns()
        config = parse_setting(s.endpoint)
        turn = None
        if config.smart_turn:
            if self.smart_turn is None:
                self.smart_turn = SmartTurn(self.turn_variant)
            turn = self.smart_turn
        return SeparateASR(tr, config, SileroVAD(), turn, name=s.name)
