"""Test streams from public speech (todo Phase 9b).

A stream is ``lead`` silence + utterance + gap + utterance + …; each utterance is trimmed to its speech
first (energy rule below, independent of the VAD under test), so its speech start and end in the stream,
and so t0 (speech ends), are known exactly. Optional white noise over the whole stream, so the silence is
not digital zero (a real microphone never is).

Sources: AMI turns (``jevlite.voice.ami``, conversational) and LibriSpeech test-clean (read speech,
CC BY 4.0), the extracted OpenSLR folder
(``LibriSpeech/test-clean/<speaker>/<chapter>/<id>.flac`` + ``<speaker>-<chapter>.trans.txt``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .endpoint import SR


@dataclass(frozen=True)
class Utterance:
    id: str
    audio: np.ndarray  # float32, 16 kHz mono
    text: str


@dataclass(frozen=True)
class Segment:
    id: str
    text: str
    start: int  # speech start in the stream (samples)
    end: int  # speech end in the stream (samples): t0


def speech_bounds(audio: np.ndarray, frame_ms: int = 20, floor_db: float = -40.0) -> tuple[int, int]:
    """[start, end) of the speech: from the first to the last frame whose RMS is within ``floor_db`` of the
    loudest frame. Whole audio if nothing passes."""
    f = SR * frame_ms // 1000
    n = len(audio) // f
    if n == 0:
        return 0, len(audio)
    rms = np.sqrt(np.mean(np.square(audio[: n * f].reshape(n, f).astype(np.float64)), axis=1))
    loud = np.flatnonzero(rms >= rms.max() * 10 ** (floor_db / 20)) if rms.max() > 0 else np.zeros(0, int)
    if len(loud) == 0:
        return 0, len(audio)
    return int(loud[0] * f), int(min(len(audio), (loud[-1] + 1) * f))


def noise_floor_db(audio: np.ndarray, frame_ms: int = 20) -> float:
    """Background level: the 10th percentile of 20 ms frame RMS, in dB relative to the loudest frame."""
    f = SR * frame_ms // 1000
    n = len(audio) // f
    if n == 0:
        return 0.0
    rms = np.sqrt(np.mean(np.square(audio[: n * f].reshape(n, f).astype(np.float64)), axis=1))
    return float(20 * np.log10(max(np.percentile(rms, 10), 1e-12) / max(rms.max(), 1e-12)))


def build_stream(utterances: list[Utterance], gap_s: tuple[float, float], lead_s: float = 0.5,
                 noise_db: float | None = -60.0, seed: int = 0, trim: bool = True) -> tuple[np.ndarray, list[Segment]]:
    """Utterances separated by gaps drawn uniformly from ``gap_s`` (seconds); ``noise_db`` = RMS of white
    noise in dBFS (None: none). The stream ends with one more gap, so the last endpoint fits.
    ``trim``: cut each utterance to ``speech_bounds`` (LibriSpeech); False when the audio is already cut at
    the annotated speech bounds (AMI)."""
    rng = np.random.default_rng(seed)
    parts = [np.zeros(int(lead_s * SR), np.float32)]
    pos = len(parts[0])
    segments = []
    for u in utterances:
        a, b = speech_bounds(u.audio) if trim else (0, len(u.audio))
        speech = np.asarray(u.audio[a:b], np.float32)
        segments.append(Segment(u.id, u.text, pos, pos + len(speech)))
        gap = np.zeros(int(rng.uniform(*gap_s) * SR), np.float32)
        parts += [speech, gap]
        pos += len(speech) + len(gap)
    stream = np.concatenate(parts)
    if noise_db is not None:
        stream = stream + rng.normal(0, 10 ** (noise_db / 20), len(stream)).astype(np.float32)
    return stream.astype(np.float32), segments


def librispeech_index(root: str | Path) -> dict[str, tuple[Path, str]]:
    """id → (flac path, transcript) for every utterance under an extracted LibriSpeech split."""
    out = {}
    for trans in sorted(Path(root).rglob("*.trans.txt")):
        for line in trans.read_text().splitlines():
            if line.strip():
                uid, text = line.split(" ", 1)
                out[uid] = (trans.parent / f"{uid}.flac", text.strip())
    if not out:
        raise FileNotFoundError(f"no LibriSpeech *.trans.txt under {root}")
    return out


def load_librispeech(root: str | Path, n: int, seed: int = 0, max_noise_db: float | None = None) -> list[Utterance]:
    """``n`` utterances drawn at random (fixed by ``seed``) from an extracted LibriSpeech split.

    ``max_noise_db``: keep only recordings whose ``noise_floor_db`` is at most this (e.g. −50). On noisier
    recordings the −40 dB trim keeps the noise tail, so t0 lands up to ~0.5 s after the last word (found
    2026-11-18: 37% of test-clean has a floor above −40 dB) and endpoints before t0 count as cuts. Noise for
    the VAD is then added under control by ``build_stream``, after t0 is fixed."""
    import soundfile as sf

    index = librispeech_index(root)
    ids = sorted(index)
    order = np.random.default_rng(seed).permutation(len(ids))
    out = []
    for i in order:
        path, text = index[ids[i]]
        audio, sr = sf.read(path, dtype="float32")
        if sr != SR or audio.ndim != 1:
            raise ValueError(f"{path}: expected 16 kHz mono, got {sr} Hz, shape {audio.shape}")
        if max_noise_db is None or noise_floor_db(audio) <= max_noise_db:
            out.append(Utterance(ids[i], audio, text))
            if len(out) == n:
                break
    return sorted(out, key=lambda u: u.id)
