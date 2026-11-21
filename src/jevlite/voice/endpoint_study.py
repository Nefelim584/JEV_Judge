"""Endpoint wait t1 − t0 and false endpoints per endpoint setting (todo Phase 9b, task 1). CPU only.

The VAD runs once over the stream (its probabilities do not depend on the setting); every setting then
replays the same chunks and probabilities through its ``Endpointer``. Per utterance (speech [start, end)
in the stream, the next one starting at ``next``):

- **endpoint**: the first endpoint in [end, next); **wait_ms** = its audio time − end (t1 − t0 without
  compute); **latency_ms** = wait_ms + the compute of the Smart Turn call that decided (t1 − t0).
- **missed**: no endpoint in [end, next): the turn ran into the next utterance. Split into **no turn**
  (no endpoint inside or after the speech: the VAD never opened a turn on it) and **ended early**
  (an endpoint inside the speech, none after: the VAD heard ≥ the setting's silence before t0, so the
  end of the speech is not speech to it — a ground-truth or a VAD problem, worth a look either way).
- **premature**: endpoints inside the speech (start, end): a pause taken for the end, the utterance cut.
- **extra**: further endpoints in [end, next) after the first (noise opening turns).

Smart Turn compute is wall time on this machine's CPU with the given threads; the GPU server will differ.
"""

from __future__ import annotations

import platform
import time
from dataclasses import asdict
from typing import Callable, Sequence

import numpy as np

from .audio import Segment
from .endpoint import CHUNK, SR, Endpoint, EndpointConfig, Endpointer


def vad_probs(stream: np.ndarray, vad: Callable[[np.ndarray], float]) -> tuple[np.ndarray, float]:
    """P(speech) per full 512-sample chunk, and the mean VAD time per chunk (ms)."""
    n = len(stream) // CHUNK
    probs = np.empty(n, np.float32)
    t = time.perf_counter()
    for i in range(n):
        probs[i] = vad(stream[i * CHUNK:(i + 1) * CHUNK])
    return probs, (time.perf_counter() - t) * 1000 / max(n, 1)


def run_setting(stream: np.ndarray, probs: np.ndarray, config: EndpointConfig,
                turn_model: Callable[[np.ndarray], float] | None = None, turn_cache: dict | None = None) -> list[Endpoint]:
    """``turn_cache``: pass the same dict to every setting on this stream (Smart Turn computed once per turn audio)."""
    ep = Endpointer(config, turn_model if config.smart_turn else None, turn_cache)
    out = []
    for i, p in enumerate(probs):
        e = ep.push(stream[i * CHUNK:(i + 1) * CHUNK], float(p))
        if e is not None:
            out.append(e)
    return out


def match(segments: Sequence[Segment], endpoints: Sequence[Endpoint], stream_len: int) -> list[dict]:
    """One row per utterance (see the module doc)."""
    samples = np.array([e.sample for e in endpoints], dtype=np.int64)
    rows = []
    for i, s in enumerate(segments):
        nxt = segments[i + 1].start if i + 1 < len(segments) else stream_len + 1
        inside = np.flatnonzero((samples > s.start) & (samples < s.end))
        after = np.flatnonzero((samples >= s.end) & (samples < nxt))
        row = {"id": s.id, "speech_s": (s.end - s.start) / SR, "premature": int(len(inside)),
               "extra": max(0, int(len(after)) - 1), "missed": len(after) == 0,
               "wait_ms": None, "decide_ms": None, "latency_ms": None, "reason": None,
               "turn_calls": None, "p_turn": None}
        if len(after):
            e = endpoints[after[0]]
            wait = (e.sample - s.end) * 1000 / SR
            row.update(wait_ms=wait, decide_ms=e.decide_ms, latency_ms=wait + e.decide_ms, reason=e.reason,
                       turn_calls=e.turn_calls, p_turn=e.p_turn)
        rows.append(row)
    return rows


def _stats(x: Sequence[float]) -> dict:
    if not len(x):
        return {"n": 0}
    a = np.asarray(x, float)
    return {"n": int(len(a)), "p50": float(np.percentile(a, 50)), "p90": float(np.percentile(a, 90)),
            "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)), "mean": float(a.mean()),
            "max": float(a.max())}


def summarize(config: EndpointConfig, rows: Sequence[dict], endpoints: Sequence[Endpoint]) -> dict:
    n = len(rows)
    hit = [r for r in rows if not r["missed"]]
    reasons = {}
    for r in hit:
        reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
    calls = [e.turn_ms / e.turn_calls for e in endpoints if e.turn_calls]
    return {
        "setting": config.name, "config": asdict(config), "utterances": n,
        "missed_pct": 100 * (n - len(hit)) / max(n, 1),
        "missed_no_turn_pct": 100 * sum(r["missed"] and r["premature"] == 0 for r in rows) / max(n, 1),
        "missed_early_pct": 100 * sum(r["missed"] and r["premature"] > 0 for r in rows) / max(n, 1),
        "premature_utt_pct": 100 * sum(r["premature"] > 0 for r in rows) / max(n, 1),
        "premature_per_100": 100 * sum(r["premature"] for r in rows) / max(n, 1),
        "extra_per_100": 100 * sum(r["extra"] for r in rows) / max(n, 1),
        "latency_ms": _stats([r["latency_ms"] for r in hit]),
        "wait_ms": _stats([r["wait_ms"] for r in hit]),
        "reasons_pct": {k: 100 * v / max(len(hit), 1) for k, v in sorted(reasons.items())},
        "turn_calls_per_endpoint": float(np.mean([e.turn_calls for e in endpoints])) if endpoints and config.smart_turn else 0.0,
        "turn_call_ms": _stats(calls),
    }


def env_info() -> dict:
    return {"machine": platform.machine(), "processor": platform.processor(), "system": platform.platform(),
            "python": platform.python_version()}


def report_markdown(result: dict) -> str:
    s = result["stream"]
    lines = [
        "# Endpoint study: t1 − t0 per endpoint setting",
        "",
        f"Audio: {s['source']}, {s['utterances']} utterances, {s['speech_min']:.1f} min of speech, gaps "
        f"{s['gap_s'][0]}–{s['gap_s'][1]} s, noise {s['noise_db']} dBFS, seed {s['seed']}"
        + (f", recordings with background ≤ {s['max_noise_db']} dB re peak. " if s.get("max_noise_db") is not None else ". ")
        + f"VAD: Silero ({'ONNX' if result['vad']['onnx'] else 'JIT'}), {result['vad']['ms_per_chunk']:.3f} ms per 32 ms chunk. "
        f"Smart Turn: {result.get('smart_turn') or 'not used'}. CPU: {result['env']['machine']} {result['env']['processor']}.",
        "",
        "t1 − t0 = endpoint wait (audio) + compute of the deciding Smart Turn call. "
        "Missed = no endpoint before the next utterance; premature = endpoints inside the speech (utterance cut).",
        "",
        "| setting | t1−t0 p50 | p90 | p95 | p99 | max | missed % (no turn / early) | cut utt. % | cuts / 100 | extra / 100 | ends by |",
        "| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | :-- |",
    ]
    for r in result["settings"]:
        lat = r["latency_ms"]
        ends = ", ".join(f"{k} {v:.0f}%" for k, v in r["reasons_pct"].items())
        if lat["n"]:
            cells = " | ".join(f"{lat[k]:.0f}" for k in ("p50", "p90", "p95", "p99", "max"))
        else:
            cells = " | ".join("–" for _ in range(5))
        miss = f"{r['missed_pct']:.1f} ({r.get('missed_no_turn_pct', 0):.1f} / {r.get('missed_early_pct', 0):.1f})"
        lines.append(f"| {r['setting']} | {cells} | {miss} | {r['premature_utt_pct']:.1f} | "
                     f"{r['premature_per_100']:.1f} | {r['extra_per_100']:.1f} | {ends} |")
    turn = [r for r in result["settings"] if r["config"]["smart_turn"] and r["turn_call_ms"]["n"]]
    if turn:
        lines += ["", "Smart Turn compute per call (ms, this CPU):", "",
                  "| setting | calls / endpoint | p50 | p95 | max |", "| :-- | --: | --: | --: | --: |"]
        for r in turn:
            c = r["turn_call_ms"]
            lines.append(f"| {r['setting']} | {r['turn_calls_per_endpoint']:.2f} | {c['p50']:.1f} | {c['p95']:.1f} | {c['max']:.1f} |")
    return "\n".join(lines) + "\n"
