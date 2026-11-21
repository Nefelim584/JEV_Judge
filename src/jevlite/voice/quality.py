"""Stage 1 of the voice study: quality of every ASR + end-of-turn system, one stream, one GPU (todo
Phase 9b, "Stage 1, quality"). Everything except the decoding runs on the CPU.

**Streams** (``prepare_ami`` / ``prepare_librispeech``, written as FLAC + JSON + the Silero P(speech)):

- **AMI**: every speaker's whole headset stream as in the pause study (``jevlite.voice.pauses``: levelled,
  other speakers muted, −60 dBFS noise), with **long silences shortened** (``keep_intervals``): around each
  of the speaker's active runs, ``after_s`` (3 s) after and ``before_s`` (0.5 s) before are kept, the rest
  of a longer gap is cut out. No setting waits more than 3 s, so the endpoints do not change; an endpoint
  later than that counts as missed. 369 min of headset audio → ~150 min (56 min of it speech). Pauses and
  active runs are mapped to the shortened time.
- **LibriSpeech** test-clean: clean recordings (background ≤ −50 dB re peak), ``n`` utterances with
  1.5–2 s gaps (longer than any setting's wait), −60 dBFS noise.

**Systems** (``jevlite.voice.asr``): A1 ``parakeet-eou`` runs live; B1 ``nemotron-r<R>`` is decoded once per
stream (``asr.record_steps``) and **replayed** under every detector setting on the CPU (its decoding does
not depend on the detector); B2 ``whisper+<setting>`` decodes each turn at the endpoint, so it runs per
setting on the GPU.

**Metrics** per system and source: t1 − t0 (endpoint), t2 − t0 (final text; = t1 for A1, + R × 80 ms of
lookahead + step for B1, + the Whisper decode for B2), p50 / p95; AMI: extra endpoints (= extra JEV calls)
per turn, missed turns (all / ≥ 0.5 s), as ``pauses.summarize_setting``; LibriSpeech: cut utterances,
missed; **WER** of the finals against the reference (Whisper's English normaliser, corpus level: all
streams' errors / reference words; the AMI reference = all of the speaker's words, so text lost to a
missed turn counts); compute ms per audio second (this GPU, one stream, no batching).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from . import ami
from . import pauses as P
from .asr import (BatchedNemo, NemotronTurns, ParakeetEOU, RecordedSteps, ReplayVAD, SeparateASR, WhisperTurns,
                  parse_system, record_batched, record_steps)
from .audio import Segment, build_stream, load_librispeech
from .endpoint import SR, Endpoint, parse_setting
from .endpoint_study import match, vad_probs

FRAME = int(round(P.FRAME_S * SR))  # 320 samples
AFTER_S, BEFORE_S = 3.0, 0.5
NEMOTRON_SETTINGS = ("vad-300", "vad-500", "vad-800")


# ------------------------------------------------------------------------------ shortening the silences


def keep_intervals(active: Sequence[tuple[int, int]], n_frames: int, after_s: float = AFTER_S,
                   before_s: float = BEFORE_S, frame_s: float = P.FRAME_S) -> list[tuple[int, int]]:
    """Frame intervals kept around the active runs (merged where they touch)."""
    aft, bef = int(round(after_s / frame_s)), int(round(before_s / frame_s))
    out: list[list[int]] = []
    for a, b in active:
        lo, hi = max(0, a - bef), min(n_frames, b + aft)
        if out and lo <= out[-1][1]:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return [(a, b) for a, b in out]


class TimeMap:
    """Original time ↔ shortened time for the kept intervals (frames)."""

    def __init__(self, intervals: Sequence[tuple[int, int]], frame_s: float = P.FRAME_S):
        self.iv = np.asarray(intervals, np.int64).reshape(-1, 2)
        self.frame_s = frame_s
        lens = self.iv[:, 1] - self.iv[:, 0]
        self.cum = np.concatenate([[0], np.cumsum(lens)])

    @property
    def n_frames(self) -> int:
        return int(self.cum[-1])

    def frame(self, f: float) -> float:
        """An original frame position → the shortened one (a position in a cut gap → the cut point)."""
        i = int(np.searchsorted(self.iv[:, 0], f, "right")) - 1
        if i < 0:
            return 0.0
        return float(self.cum[i] + min(f - self.iv[i, 0], self.iv[i, 1] - self.iv[i, 0]))

    def seconds(self, t: float) -> float:
        return round(self.frame(t / self.frame_s) * self.frame_s, 6)

    def audio(self, x: np.ndarray) -> np.ndarray:
        return np.concatenate([x[a * FRAME:b * FRAME] for a, b in self.iv]) if len(self.iv) else x[:0]


# ----------------------------------------------------------------------------------------- preparation


def _write_stream(out: Path, sid: str, audio: np.ndarray, meta: dict, vad) -> dict:
    import soundfile as sf

    out.mkdir(parents=True, exist_ok=True)
    sf.write(out / f"{sid}.flac", np.clip(audio, -1, 1), SR, subtype="PCM_16")
    audio, _ = sf.read(out / f"{sid}.flac", dtype="float32")  # the VAD sees what the ASR will see
    if hasattr(vad, "reset"):
        vad.reset()
    probs, ms = vad_probs(audio, vad)
    np.save(out / f"{sid}.vad.npy", probs)
    meta = {"id": sid, "duration_s": len(audio) / SR, "vad_ms_per_chunk": ms, **meta}
    (out / f"{sid}.json").write_text(json.dumps(meta))
    return {"id": sid, "source": meta["source"], "duration_s": meta["duration_s"]}


def prepare_ami(root: str | Path, out: str | Path, meetings=ami.DEFAULT_MEETINGS, vad=None, noise_db: float = -60.0,
                target_dbfs: float = -20.0, seed: int = 0, after_s: float = AFTER_S, before_s: float = BEFORE_S,
                log=print) -> list[dict]:
    """Per speaker: the shortened stream, its labelled pauses / active runs in the shortened time, the
    reference (all the speaker's words) and P(speech) per chunk."""
    out, index = Path(out), []
    for k, m in enumerate(meetings):
        mt = P.load_meeting(root, m, target_dbfs=target_dbfs)
        n = len(next(iter(mt.masks.values())))
        for spk in sorted(mt.channels):
            tm = TimeMap(keep_intervals(mt.active[spk], n, after_s, before_s))
            audio = tm.audio(P.speaker_stream(root, mt, spk, noise_db=noise_db, seed=seed + k))
            pauses = [{**asdict(p), "start": tm.seconds(p.start), "end": tm.seconds(p.end), "orig_start": p.start}
                      for p in mt.pauses if p.speaker == spk]
            active = [[int(tm.frame(a)), int(tm.frame(b))] for a, b in mt.active[spk]]
            meta = {"source": "ami", "meeting": m, "speaker": spk, "orig_s": n * P.FRAME_S, "pauses": pauses,
                    "active": active, "reference": " ".join(w.text for w in mt.words.get(spk, []))}
            index.append(_write_stream(out, f"ami.{m}.{spk}", audio, meta, vad))
            log(f"{m}.{spk}: {n * P.FRAME_S / 60:.1f} → {len(audio) / SR / 60:.1f} min")
    return index


def prepare_librispeech(root: str | Path, out: str | Path, n: int = 100, vad=None, gap_s=(1.5, 2.0),
                        noise_db: float = -60.0, seed: int = 0, max_noise_db: float = -50.0,
                        sid: str = "librispeech") -> list[dict]:
    utts = load_librispeech(root, n, seed, max_noise_db)
    stream, segments = build_stream(utts, gap_s, noise_db=noise_db, seed=seed)
    meta = {"source": "librispeech", "segments": [asdict(s) for s in segments], "gap_s": list(gap_s),
            "reference": " ".join(s.text for s in segments)}
    return [_write_stream(Path(out), sid, stream, meta, vad)]


def load_stream(out: str | Path, sid: str) -> tuple[np.ndarray, dict, np.ndarray]:
    import soundfile as sf

    out = Path(out)
    audio, _ = sf.read(out / f"{sid}.flac", dtype="float32")
    return audio, json.loads((out / f"{sid}.json").read_text()), np.load(out / f"{sid}.vad.npy")


# --------------------------------------------------------------------------------------------- decoding


def run_live(asr, audio: np.ndarray, piece: int = 1280) -> dict:
    """Feed a stream through a ``StreamingASR`` in ``piece``-sample pieces (as from a socket) + flush."""
    asr.reset()
    events = []
    for i in range(0, len(audio), piece):
        events += asr.push(audio[i:i + piece])
    events += asr.flush()
    return {"events": [asdict(e) for e in events], "compute_ms": float(asr.compute_ms), "audio_s": len(audio) / SR}


def replay_nemotron(rec: dict, probs: np.ndarray, setting: str, right_context: int, n_samples: int) -> dict:
    """B1 under one detector setting, from the recorded timeline (no GPU); the step size and the lookahead
    steps come from the recording (older recordings: 80 ms steps, lag = right context)."""
    chunk = int(round(rec.get("step_s", 0.08) * SR))
    tr = NemotronTurns(RecordedSteps([(p, ms) for p, ms in rec["steps"]], chunk), rec.get("lag_steps", right_context))
    asr = SeparateASR(tr, parse_setting(setting), ReplayVAD(probs), name=f"nemotron-r{right_context}+{setting}")
    return run_live(asr, np.zeros(n_samples, np.float32))


def run_whisper(model, audio: np.ndarray, probs: np.ndarray, setting: str) -> dict:
    asr = SeparateASR(WhisperTurns(model), parse_setting(setting), ReplayVAD(probs), name=f"whisper+{setting}")
    return run_live(asr, audio)


def record_nemotron(stream, audio: np.ndarray, right_context: int | None = None) -> dict:
    lag = stream.spec.lookahead_steps
    steps = record_steps(stream, audio, flush_steps=lag)
    return _nemotron_result(steps, len(audio), stream.spec)


def _nemotron_result(steps, n_samples: int, spec) -> dict:
    return {"steps": [[p, ms] for p, ms in steps], "compute_ms": float(sum(ms for _, ms in steps)),
            "audio_s": n_samples / SR, "step_s": spec.step_s, "lag_steps": spec.lookahead_steps}


def parakeet_events(steps, n_samples: int, chunk_samples: int) -> dict:
    """A1's events from a timeline recorded with the reset after each end of turn (``record_batched(...,
    reset_on_marker=True)``): the live ``ParakeetEOU`` over a replay of it, so the events are exactly the
    live ones (``run_live`` format, + the steps)."""
    asr = ParakeetEOU(RecordedSteps([(p, ms) for p, ms in steps], chunk_samples, rewind=False))
    res = run_live(asr, np.zeros(n_samples, np.float32))
    res["steps"] = [[p, ms] for p, ms in steps]
    return res


def decode_batched(spec, audios: list[np.ndarray], kind: str, batch: int | None = None,
                   log: Callable[[str], None] | None = None, preprocess: Callable | None = None
                   ) -> tuple[list[dict], dict]:
    """Many streams through one NeMo model at once (``asr.record_batched``): ``kind`` ``nemotron`` → B1
    timelines (``replay_nemotron``), ``parakeet`` → A1 events. ``batch`` = slots (default: one per
    stream). → (one result per stream, in order; the batch info)."""
    b = max(1, min(batch or len(audios), len(audios)))
    batched = BatchedNemo(spec, b, preprocess=preprocess)
    if kind == "nemotron":
        timelines, info = record_batched(batched, audios, pad_last=True, flush_steps=spec.lookahead_steps, log=log)
        out = [_nemotron_result(t, len(a), spec) for t, a in zip(timelines, audios)]
    elif kind == "parakeet":
        timelines, info = record_batched(batched, audios, pad_last=False, reset_on_marker=True, log=log)
        out = [parakeet_events(t, len(a), batched.chunk_samples) for t, a in zip(timelines, audios)]
    else:
        raise ValueError(kind)
    for r in out:
        r["batch"] = b
    return out, info


# ------------------------------------------------------------------------------------------- evaluation


def normalizer():
    """Whisper's English normaliser (``whisper-normalizer``) if installed, else lowercase / no punctuation."""
    try:
        from whisper_normalizer.english import EnglishTextNormalizer

        return EnglishTextNormalizer()
    except ImportError:
        return lambda s: re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]+", " ", s.lower().replace("_", ""))).strip()


def word_errors(ref: str, hyp: str) -> tuple[int, int]:
    """(word errors, reference words) after normalisation; jiwer if installed, else dynamic programming."""
    norm = normalizer()
    r, h = norm(ref).split(), norm(hyp).split()
    if not r:
        return len(h), 0
    try:
        import jiwer

        o = jiwer.process_words(" ".join(r), " ".join(h) if h else "")
        return o.substitutions + o.deletions + o.insertions, len(r)
    except ImportError:
        prev = list(range(len(h) + 1))
        for i, rw in enumerate(r, 1):
            cur = [i] + [0] * len(h)
            for j, hw in enumerate(h, 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw))
            prev = cur
        return prev[-1], len(r)


def pairs(events: Sequence[dict]) -> list[tuple[dict, dict | None]]:
    """(endpoint, its final) in order: every endpoint gets one final, in the same order."""
    ends = [e for e in events if e["kind"] == "endpoint"]
    finals = [e for e in events if e["kind"] == "final"]
    return [(e, finals[i] if i < len(finals) else None) for i, e in enumerate(ends)]


def _endpoint(e: dict) -> Endpoint:
    return Endpoint(sample=int(round(e["audio_s"] * SR)), start_sample=0, reason=e["detail"],
                    decide_ms=float(e["compute_ms"]), turn_calls=0, turn_ms=0.0, p_turn=None)


def _t2_extra(ep: dict, final: dict | None) -> float | None:
    """t2 − t1: from the endpoint to the final text (its wait + its compute)."""
    if final is None:
        return None
    return (final["audio_s"] - ep["audio_s"]) * 1000 + final["compute_ms"]


def eval_ami(meta: dict, events: Sequence[dict]) -> tuple[list[dict], list[dict], tuple[int, int]]:
    ps = [P.Pause(**{k: v for k, v in p.items() if k != "orig_start"}) for p in meta["pauses"]]
    pr = pairs(events)
    ep_rows, pause_rows = P.match_pauses(ps, [tuple(a) for a in meta["active"]], [_endpoint(e) for e, _ in pr])
    for row, (e, f) in zip(ep_rows, pr):
        extra = _t2_extra(e, f)
        row["t2_ms"] = row["latency_ms"] + extra if row["latency_ms"] is not None and extra is not None else None
    hyp = " ".join(e["text"] for e in events if e["kind"] == "final")
    return ep_rows, pause_rows, word_errors(meta["reference"], hyp)


def eval_librispeech(meta: dict, events: Sequence[dict], n_samples: int) -> tuple[list[dict], tuple[int, int]]:
    segs = [Segment(**s) for s in meta["segments"]]
    pr = pairs(events)
    eps = [_endpoint(e) for e, _ in pr]
    rows = match(segs, eps, n_samples)
    samples = np.array([ep.sample for ep in eps], dtype=np.int64)
    for row, s in zip(rows, segs):
        row["t2_ms"] = None
        if row["latency_ms"] is not None:  # the first endpoint at or after the end of the speech
            e, f = pr[int(np.searchsorted(samples, s.end, "left"))]
            extra = _t2_extra(e, f)
            row["t2_ms"] = row["latency_ms"] + extra if extra is not None else None
    hyp = " ".join(e["text"] for e in events if e["kind"] == "final")
    return rows, word_errors(meta["reference"], hyp)


def _q(x: Sequence[float], q: float) -> float | None:
    x = [v for v in x if v is not None]
    return float(np.percentile(x, q)) if x else None


def summarize_ami(system: str, per_stream: Sequence[tuple[list[dict], list[dict], tuple[int, int]]],
                  compute_ms: float, audio_s: float, min_turn_s: float = 0.5) -> dict:
    ep_rows = [r for e, _, _ in per_stream for r in e]
    pause_rows = [r for _, p, _ in per_stream for r in p]
    s = P.summarize_setting(system, ep_rows, pause_rows, min_turn_s)
    t2 = [r["t2_ms"] for r in ep_rows if r["kind"] == "end"]
    err, words = (sum(x) for x in zip(*(w for _, _, w in per_stream)))
    s.update(source="ami", t2_ms={"p50": _q(t2, 50), "p95": _q(t2, 95)}, wer=100 * err / max(words, 1),
             ref_words=words, compute_ms_per_audio_s=compute_ms / max(audio_s, 1e-9), audio_min=audio_s / 60)
    return s


def summarize_librispeech(system: str, rows: list[dict], errors: tuple[int, int], compute_ms: float,
                          audio_s: float) -> dict:
    hit = [r for r in rows if not r["missed"]]
    n = max(len(rows), 1)
    return {"setting": system, "source": "librispeech", "utterances": len(rows),
            "latency_ms": {"p50": _q([r["latency_ms"] for r in hit], 50), "p95": _q([r["latency_ms"] for r in hit], 95)},
            "t2_ms": {"p50": _q([r["t2_ms"] for r in hit], 50), "p95": _q([r["t2_ms"] for r in hit], 95)},
            "missed_pct": 100 * (len(rows) - len(hit)) / n,
            "cut_utt_pct": 100 * sum(r["premature"] > 0 for r in rows) / n,
            "cuts_per_utt": sum(r["premature"] for r in rows) / n,
            "wer": 100 * errors[0] / max(errors[1], 1), "ref_words": errors[1],
            "compute_ms_per_audio_s": compute_ms / max(audio_s, 1e-9), "audio_min": audio_s / 60}


def _f(x, fmt="{:.0f}") -> str:
    return "–" if x is None else fmt.format(x)


def report_markdown(result: dict) -> str:
    jobs = (result.get("decode") or {}).get("jobs", {})
    batches = {j: v["batch"] for j, v in jobs.items() if isinstance(v, dict) and "batch" in v}
    how = (f"the NeMo systems decoded all streams at once ({', '.join(f'{j}: batch {b}' for j, b in batches.items())}), "
           "each stream charged the batched step / the busy slots; Whisper one stream at a time"
           if batches else "one stream, no batching")
    lines = ["# Voice stage 1: quality of ASR + end of turn", "",
             f"GPU: {result.get('gpu', '?')}. Versions: {result.get('versions', {})}.", "",
             f"t1 = endpoint, t2 = final text, both from the end of the speech (t0), ms. Compute = this GPU, {how} "
             "(the B systems' VAD runs on the CPU, ~0.14 ms per 32 ms chunk, not included).", ""]
    for j, c in ((result.get("decode") or {}).get("batch_check") or {}).items():
        lines.append(f"Batch check {j} (smoke stream): WER single {c['single_wer']:.1f}%, batched "
                     f"{c['batched_wer']:.1f}%; batched vs single words (whole, half, half after a slot reset): "
                     f"{', '.join(f'{d:.1f}%' for d in c['disagree_pct'])}.")
    lines.append("")
    ami_rows = [r for r in result["summaries"] if r["source"] == "ami"]
    if ami_rows:
        lines += [f"## AMI ({ami_rows[0]['turns']} turns, {ami_rows[0]['audio_min']:.0f} min of audio)", "",
                  "| system | t1 p50 | t1 p95 | t2 p50 | t2 p95 | extra JEV calls / turn | missed % (all / ≥ 0.5 s) | WER % | compute ms / audio s |",
                  "| :-- | --: | --: | --: | --: | --: | --: | --: | --: |"]
        for r in ami_rows:
            lines.append(f"| {r['setting']} | {_f(r['latency_ms']['p50'])} | {_f(r['latency_ms']['p95'])} | "
                         f"{_f(r['t2_ms']['p50'])} | {_f(r['t2_ms']['p95'])} | {r['extra_per_turn']:.2f} | "
                         f"{r['missed_pct']:.1f} / {r['missed_long_pct']:.1f} | {r['wer']:.1f} | "
                         f"{r['compute_ms_per_audio_s']:.1f} |")
        lines.append("")
    ls = [r for r in result["summaries"] if r["source"] == "librispeech"]
    if ls:
        lines += [f"## LibriSpeech test-clean ({ls[0]['utterances']} utterances)", "",
                  "| system | t1 p50 | t1 p95 | t2 p50 | t2 p95 | cut utt. % | missed % | WER % | compute ms / audio s |",
                  "| :-- | --: | --: | --: | --: | --: | --: | --: | --: |"]
        for r in ls:
            lines.append(f"| {r['setting']} | {_f(r['latency_ms']['p50'])} | {_f(r['latency_ms']['p95'])} | "
                         f"{_f(r['t2_ms']['p50'])} | {_f(r['t2_ms']['p95'])} | {r['cut_utt_pct']:.1f} | "
                         f"{r['missed_pct']:.1f} | {r['wer']:.1f} | {r['compute_ms_per_audio_s']:.1f} |")
    return "\n".join(lines) + "\n"


def evaluate(streams_dir: str | Path, decode_dir: str | Path, nemotron_settings=NEMOTRON_SETTINGS,
             stream_ids: Sequence[str] | None = None, log=print) -> list[dict]:
    """Every decoded system × source → summary rows (replays B1 under ``nemotron_settings``); only the
    decoded streams count (``stream_ids`` narrows further). LibriSpeech streams are pooled."""
    streams_dir, decode_dir = Path(streams_dir), Path(decode_dir)
    index = json.loads((streams_dir / "index.json").read_text())
    if stream_ids is not None:
        index = [s for s in index if s["id"] in set(stream_ids)]
    runs: dict[str, dict[str, dict]] = {}  # system → stream id → decoded
    for d in sorted(p for p in decode_dir.iterdir() if p.is_dir()):
        for f in sorted(d.glob("*.json")):
            rec = json.loads(f.read_text())
            if d.name.startswith("nemotron-r"):
                r = int(d.name.split("-r")[1])
                audio_n = int(round(rec["audio_s"] * SR))
                _, _, probs = load_stream(streams_dir, f.stem)
                for st in nemotron_settings:
                    out = replay_nemotron(rec, probs, st, r, audio_n)
                    out["compute_ms"] = rec["compute_ms"]  # the GPU decode, not the replay
                    runs.setdefault(f"{d.name}+{st}", {})[f.stem] = out
            else:
                runs.setdefault(d.name, {})[f.stem] = rec
    summaries = []
    for system, by_stream in runs.items():
        for source in ("ami", "librispeech"):
            ids = [s["id"] for s in index if s["source"] == source and s["id"] in by_stream]
            if not ids:
                continue
            comp = sum(by_stream[i]["compute_ms"] for i in ids)
            audio = sum(by_stream[i]["audio_s"] for i in ids)
            if source == "ami":
                per = [eval_ami(load_stream(streams_dir, i)[1], by_stream[i]["events"]) for i in ids]
                summaries.append(summarize_ami(system, per, comp, audio))
            else:
                rows, errs = [], (0, 0)
                for i in ids:
                    r, e = eval_librispeech(load_stream(streams_dir, i)[1], by_stream[i]["events"],
                                            int(round(by_stream[i]["audio_s"] * SR)))
                    rows, errs = rows + r, (errs[0] + e[0], errs[1] + e[1])
                summaries.append(summarize_librispeech(system, rows, errs, comp, audio))
            log(f"{system} / {source}: {len(ids)} stream(s)")
    return summaries


def system_dirs(systems: Sequence[str]) -> list[str]:
    """Decode jobs for the named systems: ``nemotron-r<R>`` once for all its settings; others as named."""
    out = []
    for name in systems:
        s = parse_system(name)
        job = f"nemotron-r{s.right_context}" if s.asr == "nemotron" else s.name
        if job not in out:
            out.append(job)
    return out

