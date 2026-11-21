"""End-of-utterance detection (todo Phase 9b, task 1): when has the speaker finished?

The detector sees 32 ms chunks (512 samples at 16 kHz, the Silero VAD frame) with their speech
probability, and decides the endpoint (t1). Two kinds of setting:

    vad-<ms>                Silero VAD only: the turn ends after <ms> of silence.
    turn-<ms>[-p<P>][-m<M>] Silero VAD + Smart Turn v3.2: after <ms> of silence Smart Turn looks at the
                            turn's audio (last 8 s); P(complete) ≥ P (default 0.5) ends the turn, below it
                            keeps waiting. New speech resets the check; after M ms of silence (default
                            3000, the Pipecat default) the turn ends anyway (timeout). ``turn-200-m800`` =
                            "Smart Turn may end early, otherwise as vad-800".

Speech / silence follow Silero's own streaming rule: a chunk is speech at P ≥ ``threshold`` (0.5); a
pause starts at the first chunk with P < ``threshold − 0.15`` and lasts until the next speech chunk
(chunks in between neither start nor end it). A turn opens only after ``start_ms`` of speech, so short
noises do not open turns.

Models (CPU, one per stream; licenses checked 2026-11-18):

- Silero VAD ``silero-vad`` 6.2.3, MIT (GitHub repo and PyPI).
- Smart Turn v3.2, ``pipecat-ai/smart-turn-v3`` at ``SMART_TURN_REVISION`` (2026-01-07), BSD-2-Clause;
  8M parameters, Whisper Tiny encoder + linear head; ``cpu`` = int8 ONNX (8 MB), ``gpu`` = fp32 (32 MB).
  Input: 16 kHz mono, the last 8 s of the turn, zero-padded at the start.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

SR = 16_000
CHUNK = 512  # Silero VAD frame at 16 kHz: 32 ms
SMART_TURN_REPO = "pipecat-ai/smart-turn-v3"
SMART_TURN_REVISION = "f766f81d3cfdf7737ac64aad813d91bbfd56bf93"  # main on 2026-11-18, v3.2 (2026-01-07)
SMART_TURN_SECONDS = 8
DEFAULT_SETTINGS = ("vad-300", "vad-500", "vad-800", "turn-200")
# Smart Turn tuning (todo Phase 9b): threshold × timeout around the Pipecat defaults, with the VAD baselines.
TUNE_SETTINGS = ("vad-500", "vad-800",
                 *(f"turn-200-p{p}-m{m}" for m in (3000, 1200, 800) for p in ("0.3", "0.5", "0.7", "0.9")),
                 "turn-400-p0.5-m800")
# Carried to AMI after the LibriSpeech tuning (2026-11-18), plus vad-300 and the Pipecat default turn-200.
SHORT_SETTINGS = ("vad-300", "vad-500", "vad-800", "turn-200", "turn-200-p0.5-m800", "turn-200-p0.9-m800",
                  "turn-400-p0.5-m800")


@dataclass(frozen=True)
class EndpointConfig:
    name: str
    silence_ms: int  # vad: the silence that ends a turn; turn: the silence that triggers a Smart Turn check
    smart_turn: bool = False
    max_silence_ms: int = 3000  # Smart Turn only: end the turn after this much silence whatever it says
    threshold: float = 0.5
    neg_threshold: float = 0.35
    start_ms: int = 200
    turn_threshold: float = 0.5


def parse_setting(name: str, max_silence_ms: int = 3000) -> EndpointConfig:
    """``vad-500`` → silence only, 500 ms; ``turn-200`` → Smart Turn checked after 200 ms of silence;
    ``turn-200-p0.7-m800`` → threshold 0.7, timeout 800 ms (``max_silence_ms`` is the default timeout)."""
    m = re.fullmatch(r"(vad|turn)-(\d+)(?:-p(0?\.\d+|1(?:\.0+)?))?(?:-m(\d+))?", name.strip())
    if not m or (m.group(1) == "vad" and (m.group(3) or m.group(4))):
        raise ValueError(f"endpoint setting {name!r}: expected vad-<ms> or turn-<ms>[-p<P>][-m<ms>]")
    kind, ms = m.group(1), int(m.group(2))
    timeout = int(m.group(4)) if m.group(4) else max_silence_ms
    if kind == "turn" and ms >= timeout:
        raise ValueError(f"{name}: the check ({ms} ms) must come before the timeout ({timeout} ms)")
    return EndpointConfig(name=name.strip(), silence_ms=ms, smart_turn=kind == "turn", max_silence_ms=timeout,
                          turn_threshold=float(m.group(3)) if m.group(3) else 0.5)


@dataclass(frozen=True)
class Endpoint:
    sample: int  # stream position at the end of the chunk that decided (audio-time part of t1)
    start_sample: int  # where the turn opened
    reason: str  # silence | turn | timeout
    decide_ms: float  # compute time of the Smart Turn call that decided (0 for silence / timeout)
    turn_calls: int  # Smart Turn calls during this turn
    turn_ms: float  # all Smart Turn compute during this turn
    p_turn: float | None  # Smart Turn's P(complete) of its last call


class Endpointer:
    """One stream's endpoint state machine: ``push(chunk, p_speech)`` per chunk → an ``Endpoint`` or None."""

    def __init__(self, config: EndpointConfig, turn_model: Callable[[np.ndarray], float] | None = None,
                 turn_cache: dict | None = None):
        """``turn_cache``: shared between settings replaying the same stream; Smart Turn's answer depends only
        on the turn's audio, i.e. on (turn start, position), so it is computed once (with its measured time)."""
        if config.smart_turn and turn_model is None:
            raise ValueError(f"{config.name} needs a Smart Turn model")
        self.config = config
        self.turn_model = turn_model
        self.turn_cache = turn_cache
        self.pos = 0
        self._idle()

    def _idle(self) -> None:
        self.in_turn = False
        self.pending: list[np.ndarray] = []
        self.pending_speech = 0
        self.buffer: list[np.ndarray] = []
        self.buffered = 0
        self.start_sample = 0
        self.silence_start: int | None = None
        self.checked = False
        self.turn_calls = 0
        self.turn_ms = 0.0
        self.p_turn: float | None = None

    def turn_audio(self) -> np.ndarray:
        return np.concatenate(self.buffer) if self.buffer else np.zeros(0, np.float32)

    def push(self, chunk: np.ndarray, p_speech: float) -> Endpoint | None:
        c = self.config
        n = len(chunk)
        self.pos += n
        speech = p_speech >= c.threshold
        quiet = p_speech < c.neg_threshold
        if not self.in_turn:
            if quiet:
                self.pending, self.pending_speech = [], 0
            elif speech or self.pending:
                self.pending.append(chunk)
                self.pending_speech += n if speech else 0
                if self.pending_speech * 1000 >= c.start_ms * SR:
                    self.in_turn = True
                    self.buffer, self.pending = self.pending, []
                    self.buffered = sum(len(x) for x in self.buffer)
                    self.start_sample = self.pos - self.buffered
            return None

        self.buffer.append(chunk)
        self.buffered += n
        keep = (SMART_TURN_SECONDS + 1) * SR  # Smart Turn reads the last 8 s; drop older audio
        while self.buffered - len(self.buffer[0]) >= keep:
            self.buffered -= len(self.buffer.pop(0))
        if speech:
            self.silence_start, self.checked = None, False
            return None
        if quiet and self.silence_start is None:
            self.silence_start = self.pos - n
        if self.silence_start is None:
            return None
        silence_ms = (self.pos - self.silence_start) * 1000 / SR
        if not c.smart_turn:
            return self._end("silence", 0.0) if silence_ms >= c.silence_ms else None
        if silence_ms >= c.max_silence_ms:
            return self._end("timeout", 0.0)
        if silence_ms >= c.silence_ms and not self.checked:
            self.checked = True
            key = (self.start_sample, self.pos)
            if self.turn_cache is not None and key in self.turn_cache:
                self.p_turn, ms = self.turn_cache[key]
            else:
                t = time.perf_counter()
                self.p_turn = float(self.turn_model(self.turn_audio()))
                ms = (time.perf_counter() - t) * 1000
                if self.turn_cache is not None:
                    self.turn_cache[key] = (self.p_turn, ms)
            self.turn_calls += 1
            self.turn_ms += ms
            if self.p_turn >= c.turn_threshold:
                return self._end("turn", ms)
        return None

    def close(self) -> Endpoint | None:
        """End of the stream: end an open turn now (reason ``eos``), else None."""
        return self._end("eos", 0.0) if self.in_turn else None

    def _end(self, reason: str, decide_ms: float) -> Endpoint:
        e = Endpoint(sample=self.pos, start_sample=self.start_sample, reason=reason, decide_ms=decide_ms,
                     turn_calls=self.turn_calls, turn_ms=self.turn_ms, p_turn=self.p_turn)
        self._idle()
        return e


class SileroVAD:
    """Silero VAD for one stream (the model keeps the stream's state): ``vad(chunk)`` → P(speech)."""

    def __init__(self, onnx: bool = True):
        from silero_vad import load_silero_vad

        self.model = load_silero_vad(onnx=onnx)

    def __call__(self, chunk: np.ndarray) -> float:
        import torch

        return float(self.model(torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32)), SR))

    def reset(self) -> None:
        self.model.reset_states()


def fit_turn_audio(audio: np.ndarray, seconds: int = SMART_TURN_SECONDS) -> np.ndarray:
    """The last ``seconds`` of the turn, zero-padded at the start (Smart Turn's input layout)."""
    n = seconds * SR
    audio = np.asarray(audio, dtype=np.float32)[-n:]
    return np.pad(audio, (n - len(audio), 0)) if len(audio) < n else audio


class SmartTurn:
    """Smart Turn v3.2 on ONNX Runtime: ``model(turn_audio)`` → P(turn complete)."""

    def __init__(self, variant: str = "cpu", path: str | Path | None = None, threads: int = 1):
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        if path is None:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(SMART_TURN_REPO, f"smart-turn-v3.2-{variant}.onnx", revision=SMART_TURN_REVISION)
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        self.features = WhisperFeatureExtractor(chunk_length=SMART_TURN_SECONDS)
        self.variant, self.threads, self.path = variant, threads, str(path)

    def __call__(self, audio: np.ndarray) -> float:
        n = SMART_TURN_SECONDS * SR
        feats = self.features(fit_turn_audio(audio), sampling_rate=SR, return_tensors="np", padding="max_length",
                              max_length=n, truncation=True, do_normalize=True).input_features
        out = self.session.run(None, {"input_features": feats.astype(np.float32)})
        return float(np.asarray(out[0]).reshape(-1)[0])
