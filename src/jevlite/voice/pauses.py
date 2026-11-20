"""Pause labels on AMI and the endpoint cost table (todo Phase 9b, "Pause labels and the cost table").

The question: per endpoint setting, how long do we wait after the speech ends (latency) and how many
**extra JEV calls** do we pay for pauses that were not the end? Turn-level streams (``ami.load_ami``) cut
turns out of the meeting and put silence between them, which mixes the two; here every speaker's
**whole headset stream** is replayed, and every endpoint is matched to a labelled pause.

**Speech and pauses, per speaker.** ``active`` = the speaker's 20 ms frames that are speaking by the audio
(``ami.speaking_mask``: own headset above the noise, near their speech level and louder than every other
headset) **and** inside one of their annotated words (± ``word_pad_s``), so coughs and laughs do not count.
Active runs closer than ``min_pause_s`` (150 ms) are merged; runs shorter than ``min_run_s`` are dropped.
A **pause** is the gap after an active run, up to the speaker's next run (or the end of the meeting).

**Label of a pause** (from the annotations, the same rule as ``ami.build_turns``):

- **end**: ≥ ``floor_words`` words of other speakers start inside the pause (floor change), or the pause
  lasts ≥ ``split_pause_s`` (2 s);
- **continue**: the speaker resumes first (1–2 words of others, e.g. "yeah", allowed);
- **unclear**: another speaker had already been speaking (an active run of theirs, by the audio) for
  ≥ ``overlap_s`` (200 ms) when this speaker stopped, and does not stop with them (overlap: the speaker may
  have been interrupted or kept talking under the other); excluded from the costs, counted. The word
  times cannot tell this (they absorb the pauses, so quick floor changes look like overlaps).

**The stream a detector sees** (one per speaker): their headset, levelled like ``load_ami``
(``target_dbfs``), with the **other speakers muted** (``mute_others``): frames where another speaker is
speaking and this one is not are faded to zero, as an ideal echo canceller / separate channel would
leave them, then white noise at ``noise_db``. Muting uses the audio masks only, the same way at every
pause, never the labels, so it gives the detector no hint about the end. Checked 2026-11-20: inside the
speaker's own active runs only ~0–4% of frames get muted, so the muting does not chop their speech. What
the masks miss stays audible: crosstalk of others whose mask is off (two others at once, a quiet one) and
the speaker's own non-word sounds (laughs, breaths); in end pauses where the endpoint came late or never,
10–16% of the frames of the first 3 s are above −45 dBFS vs 2% where it came within 1 s.

**Matching** an endpoint at time t to the speaker's speech:

- inside an active run → **in speech** (the detector heard a pause the audio mask does not have);
- in the pause after run j, the first endpoint there → by the pause's label: **end** → latency t − pause
  start (t1 − t0); **continue** → an extra JEV call (premature); **unclear** → counted apart;
- a further endpoint in the same pause → **repeat** (noise reopened a turn): an extra call too.

A **turn** here = one end pause; an end pause with no endpoint is **missed** (the VAD never opened a turn on
the speech before it, or already ended it inside). JEV calls per turn = all endpoints / turns.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from . import ami
from .endpoint import SR, Endpoint

FRAME_S = ami.FRAME_S
MIN_PAUSE_S = 0.15
MIN_RUN_S = 0.04
WORD_PAD_S = 0.05
# Pause-duration bins for the extra calls (ms; the last bin is open).
BINS_MS = (150, 300, 500, 800, 2000)
# Settings for the pause study: a VAD threshold sweep (the latency vs extra-calls curve) + the Smart Turn
# shortlist carried from the turn-level studies.
PAUSE_SETTINGS = ("vad-200", "vad-300", "vad-400", "vad-500", "vad-600", "vad-800", "vad-1000",
                  "turn-200", "turn-200-p0.5-m800", "turn-200-p0.9-m800", "turn-400-p0.5-m800")
LABELS = ("end", "continue", "unclear")


@dataclass(frozen=True)
class Pause:
    meeting: str
    speaker: str
    start: float  # end of the active run before it: t0 if it is an end
    end: float  # start of the speaker's next active run, or the end of the meeting
    label: str  # end | continue | unclear
    other_words: int  # words of other speakers starting inside the pause
    turn_s: float  # speech since the previous end pause (first run start → this pause), seconds
    before: str  # the speaker's last words before the pause
    after: str  # the first words after it (the speaker's own if continue, the others' if end)

    @property
    def ms(self) -> float:
        return round((self.end - self.start) * 1000, 1)  # start / end are on the 20 ms grid; no 199.999…


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) frame index pairs of the True runs of a boolean array."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def word_mask(words: Sequence[ami.Word], n: int, pad_s: float = WORD_PAD_S, frame_s: float = FRAME_S) -> np.ndarray:
    m = np.zeros(n, bool)
    for w in words:
        m[max(0, int(np.floor((w.start - pad_s) / frame_s))):min(n, int(np.ceil((w.end + pad_s) / frame_s)))] = True
    return m


def active_runs(mask: np.ndarray, min_pause_s: float = MIN_PAUSE_S, min_run_s: float = MIN_RUN_S,
                frame_s: float = FRAME_S) -> list[tuple[int, int]]:
    """Runs of ``mask`` with gaps < ``min_pause_s`` merged, then runs < ``min_run_s`` dropped (frames)."""
    gap, short = int(round(min_pause_s / frame_s)), int(round(min_run_s / frame_s))
    out: list[list[int]] = []
    for a, b in runs(mask):
        if out and a - out[-1][1] < gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(a, b) for a, b in out if b - a >= short]


def _text(words: Sequence[ami.Word], k: int = 6) -> str:
    return " ".join(w.text for w in words[-k:]) if words else ""


def label_pauses(meeting: str, words: dict[str, list[ami.Word]], active: dict[str, list[tuple[int, int]]],
                 duration_s: float, floor_words: int = 3, split_pause_s: float = 2.0, overlap_s: float = 0.2,
                 frame_s: float = FRAME_S) -> list[Pause]:
    """The labelled pauses of every speaker (see the module doc), sorted by start."""
    out = []
    for spk, rs in active.items():
        if not rs:
            continue
        own = words.get(spk, [])
        own_start = np.array([w.start for w in own], float)
        others = sorted((w for s, ws in words.items() if s != spk for w in ws), key=lambda w: w.start)
        o_start = np.array([w.start for w in others], float)
        o_runs = [r for s, rr in active.items() if s != spk for r in rr]
        margin = int(round(overlap_s / frame_s))
        turn_from = rs[0][0] * frame_s
        for j, (a, b) in enumerate(rs):
            start = b * frame_s
            end = rs[j + 1][0] * frame_s if j + 1 < len(rs) else duration_s
            if end <= start or (j + 1 == len(rs) and end - start < split_pause_s):
                continue  # the meeting ends too soon after the last run to tell
            lo, hi = np.searchsorted(o_start, start, "left"), np.searchsorted(o_start, end, "left")
            n_other = int(hi - lo)
            overlap = any(oa <= b - margin and ob > b for oa, ob in o_runs)
            if end - start >= split_pause_s:
                label = "end"
            elif overlap:  # another speaker was mid-word when this one stopped: interrupted / talking under
                label = "unclear"
            elif n_other >= floor_words:
                label = "end"
            else:
                label = "continue"
            before = own[:np.searchsorted(own_start, start + WORD_PAD_S, "right")]
            if label == "continue":
                after = own[np.searchsorted(own_start, end - WORD_PAD_S, "left"):][:6]
            else:
                after = others[lo:min(hi, lo + 6)]  # the others' words inside the pause
            out.append(Pause(meeting, spk, round(start, 3), round(end, 3), label, n_other,
                             round(start - turn_from, 3), _text(before), " ".join(w.text for w in after)))
            if label == "end" and j + 1 < len(rs):
                turn_from = rs[j + 1][0] * frame_s
    return sorted(out, key=lambda p: (p.start, p.speaker))


def mute_gain(own: np.ndarray, others: np.ndarray, fade: int = 3) -> np.ndarray:
    """Per-frame gain: 0 where another speaker is speaking and this one is not, 1 elsewhere, smoothed over
    ``fade`` frames so the edges are ramps, not clicks."""
    g = (~(others & ~own)).astype(np.float32)
    if fade > 1 and len(g):
        g = np.convolve(np.pad(g, (fade // 2, fade - 1 - fade // 2), mode="edge"), np.ones(fade, np.float32) / fade,
                        mode="valid")
    return g


def frame_gain_to_samples(g: np.ndarray, n: int, frame_s: float = FRAME_S) -> np.ndarray:
    f = int(round(frame_s * SR))
    out = np.repeat(g.astype(np.float32), f)[:n]
    return np.pad(out, (0, n - len(out)), constant_values=1.0) if len(out) < n else out


@dataclass
class Meeting:
    """One meeting, prepared: the words, levels, speaking masks and labelled pauses."""
    name: str
    channels: dict[str, int]
    words: dict[str, list[ami.Word]]
    masks: dict[str, np.ndarray]
    active: dict[str, list[tuple[int, int]]]
    gain_db: dict[str, float]
    pauses: list[Pause]
    duration_s: float


def load_meeting(root: str | Path, meeting: str, floor_words: int = 3, split_pause_s: float = 2.0,
                 min_pause_s: float = MIN_PAUSE_S, target_dbfs: float | None = -20.0) -> Meeting:
    root = Path(root)
    chan = ami.channels(root / "annotations" / "corpusResources" / "meetings.xml", meeting)
    words = {}
    for p in sorted((root / "annotations" / "words").glob(f"{meeting}.*.words.xml")):
        m = re.fullmatch(rf"{re.escape(meeting)}\.([A-Z])\.words\.xml", p.name)
        if m and m.group(1) in chan:
            words[m.group(1)] = ami.parse_words(p, m.group(1))
    levels = {s: ami.frame_db(root / "audio" / f"{meeting}.Headset-{c}.wav") for s, c in chan.items()}
    masks = ami.speaking_mask(levels)
    n = min(len(v) for v in masks.values())
    active = {s: active_runs(masks[s][:n] & word_mask(words.get(s, []), n), min_pause_s) for s in chan}
    gain = ({s: target_dbfs - float(np.percentile(v, 99)) for s, v in levels.items()}
            if target_dbfs is not None else {})
    pauses = label_pauses(meeting, words, active, n * FRAME_S, floor_words, split_pause_s)
    return Meeting(meeting, chan, words, {s: v[:n] for s, v in masks.items()}, active, gain, pauses, n * FRAME_S)


def speaker_stream(root: str | Path, mt: Meeting, speaker: str, mute_others: bool = True,
                   noise_db: float | None = -60.0, seed: int = 0) -> np.ndarray:
    """The speaker's whole headset recording as the detector sees it (see the module doc)."""
    import soundfile as sf

    wav = Path(root) / "audio" / f"{mt.name}.Headset-{mt.channels[speaker]}.wav"
    audio, sr = sf.read(wav, dtype="float32", always_2d=False)
    if sr != SR or audio.ndim != 1:
        raise ValueError(f"{wav}: expected 16 kHz mono, got {sr} Hz, shape {audio.shape}")
    if speaker in mt.gain_db:
        audio = audio * np.float32(10 ** (mt.gain_db[speaker] / 20))
    if mute_others:
        own = mt.masks[speaker]
        others = np.max(np.stack([m for s, m in mt.masks.items() if s != speaker]), axis=0) if len(mt.masks) > 1 \
            else np.zeros_like(own)
        audio = audio * frame_gain_to_samples(mute_gain(own, others), len(audio))
    if noise_db is not None:
        audio = audio + np.random.default_rng(seed).normal(0, 10 ** (noise_db / 20), len(audio)).astype(np.float32)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


def match_pauses(pauses: Sequence[Pause], active: Sequence[tuple[int, int]], endpoints: Sequence[Endpoint],
                 frame_s: float = FRAME_S) -> tuple[list[dict], list[dict]]:
    """One speaker: (one row per endpoint, one row per pause) — see the module doc.

    Endpoint row: time_s, kind (end | continue | unclear | repeat | in_speech | outside), pause_ms, latency_ms (end only),
    reason / decide_ms / turn_calls of the endpoint. Pause row: the pause fields + calls (endpoints in it),
    latency_ms of its first endpoint (None if none)."""
    starts = np.array([a * frame_s for a, _ in active], float)
    ends = np.array([b * frame_s for _, b in active], float)
    by_start = {round(p.start, 3): p for p in pauses}
    first: dict[float, dict] = {}
    calls: dict[float, int] = {}
    ep_rows = []
    for e in endpoints:
        t = e.sample / SR
        j = int(np.searchsorted(starts, t, "right")) - 1
        row = {"time_s": round(t, 3), "reason": e.reason, "decide_ms": e.decide_ms, "turn_calls": e.turn_calls,
               "pause_ms": None, "latency_ms": None}
        if j < 0:
            row["kind"] = "outside"
        elif t < ends[j]:
            row["kind"] = "in_speech"
        else:
            p = by_start.get(round(ends[j], 3))
            if p is None:  # after the last run, at the very end of the meeting
                row["kind"] = "outside"
            else:
                row["pause_ms"] = p.ms
                calls[p.start] = calls.get(p.start, 0) + 1
                if p.start in first:
                    row["kind"] = "repeat"
                else:
                    row["kind"] = p.label
                    if p.label == "end":
                        row["latency_ms"] = (t - p.start) * 1000 + e.decide_ms
                    first[p.start] = row
        ep_rows.append(row)
    pause_rows = []
    for p in pauses:
        f = first.get(p.start)
        pause_rows.append({"meeting": p.meeting, "speaker": p.speaker, "start": p.start, "ms": p.ms,
                           "label": p.label, "turn_s": p.turn_s, "calls": calls.get(p.start, 0),
                           "latency_ms": f["latency_ms"] if f else None})
    return ep_rows, pause_rows


def _bin(ms: float, bins: Sequence[int] = BINS_MS) -> str:
    for lo, hi in zip(bins, bins[1:]):
        if lo <= ms < hi:
            return f"{lo}–{hi}"
    return f"≥{bins[-1]}" if ms >= bins[-1] else f"<{bins[0]}"


def bin_names(bins: Sequence[int] = BINS_MS) -> list[str]:
    return [f"{lo}–{hi}" for lo, hi in zip(bins, bins[1:])] + [f"≥{bins[-1]}"]


def _pct(x: Sequence[float], q: float) -> float | None:
    return float(np.percentile(np.asarray(x, float), q)) if len(x) else None


def summarize_setting(name: str, ep_rows: Sequence[dict], pause_rows: Sequence[dict], min_turn_s: float = 0.5) -> dict:
    """Costs of one setting over all speakers."""
    ends = [r for r in pause_rows if r["label"] == "end"]
    turns = max(len(ends), 1)
    kinds = {k: sum(r["kind"] == k for r in ep_rows) for k in ("end", "continue", "unclear", "repeat", "in_speech",
                                                               "outside")}
    lat = [r["latency_ms"] for r in ends if r["latency_ms"] is not None]
    long_ends = [r for r in ends if r["turn_s"] >= min_turn_s]
    extra = kinds["continue"] + kinds["repeat"] + kinds["in_speech"]
    cont_bins = {b: 0 for b in bin_names()}
    for r in ep_rows:
        if r["kind"] == "continue":
            cont_bins[_bin(r["pause_ms"])] += 1
    return {
        "setting": name, "turns": len(ends), "endpoints": len(ep_rows), "kinds": kinds,
        "calls_per_turn": len(ep_rows) / turns,
        "extra_per_turn": extra / turns,
        "continue_per_turn": kinds["continue"] / turns,
        "unclear_per_turn": kinds["unclear"] / turns,
        "missed_pct": 100 * sum(r["latency_ms"] is None for r in ends) / turns,
        "missed_long_pct": 100 * sum(r["latency_ms"] is None for r in long_ends) / max(len(long_ends), 1),
        "latency_ms": {"n": len(lat), **{f"p{q}": _pct(lat, q) for q in (50, 90, 95, 99)}},
        "continue_by_pause_ms": {b: v / turns for b, v in cont_bins.items()},
    }


def pause_stats(pauses: Sequence[Pause], thresholds_ms: Sequence[int] = (200, 300, 400, 500, 600, 800, 1000)) -> dict:
    """Model-free view of the labels: counts per label and duration bin, and for a plain silence threshold T
    the continue pauses ≥ T per turn (the extra calls an ideal VAD with threshold T would pay)."""
    by = {lab: [p for p in pauses if p.label == lab] for lab in LABELS}
    turns = max(len(by["end"]), 1)
    bins = {lab: {b: 0 for b in bin_names()} for lab in LABELS}
    for p in pauses:
        bins[p.label][_bin(p.ms)] += 1
    return {
        "counts": {lab: len(v) for lab, v in by.items()},
        "bins": bins,
        "ideal_vad_extra_per_turn": {t: sum(p.ms >= t for p in by["continue"]) / turns for t in thresholds_ms},
        "turn_s": {q: _pct([p.turn_s for p in by["end"]], q) for q in (10, 50, 90)},
    }


def export_check(root: str | Path, meetings: Sequence[Meeting], out: str | Path, n: int = 30, seed: int = 0,
                 before_s: float = 2.0, after_s: float = 1.0) -> list[dict]:
    """``n`` random pauses (≈ evenly over the labels) as wav clips of the **mix of all headsets** (levelled),
    from ``before_s`` before the pause to ``after_s`` after it (pauses longer than 2 s are cut to 2 s), named
    ``NN_<label>_<meeting>_<speaker>_<start>.wav``, plus ``pauses.csv`` with the words before / after."""
    import csv

    import soundfile as sf

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pool = {lab: [(mt, p) for mt in meetings for p in mt.pauses if p.label == lab] for lab in LABELS}
    per = {lab: min(len(v), n // len(LABELS) + (1 if i < n % len(LABELS) else 0)) for i, (lab, v) in enumerate(pool.items())}
    pick, rest = [], []
    for lab in LABELS:
        shuffled = rng.sample(pool[lab], len(pool[lab]))
        pick += shuffled[:per[lab]]
        rest += shuffled[per[lab]:]
    pick += rng.sample(rest, min(len(rest), n - len(pick)))  # a label with too few pauses: fill from the others
    rng.shuffle(pick)
    rows = []
    for i, (mt, p) in enumerate(pick):
        a = max(0.0, p.start - before_s)
        b = min(mt.duration_s, p.start + min(p.end - p.start, 2.0) + after_s)
        mix = None
        for s, c in mt.channels.items():
            x, _ = sf.read(Path(root) / "audio" / f"{mt.name}.Headset-{c}.wav", start=int(a * SR), stop=int(b * SR),
                           dtype="float32", always_2d=False)
            x = x * np.float32(10 ** (mt.gain_db.get(s, 0.0) / 20))
            mix = x if mix is None else mix[:len(x)] + x[:len(mix)]
        mix = mix / max(1.0, float(np.max(np.abs(mix))) / 0.9)
        name = f"{i:02d}_{p.label}_{mt.name}_{p.speaker}_{p.start:.2f}.wav"
        sf.write(out / name, mix, SR)
        rows.append({"file": name, "label": p.label, "meeting": mt.name, "speaker": p.speaker, "pause_start_s": p.start,
                     "pause_ms": round(p.ms), "pause_at_s_in_clip": round(p.start - a, 2), "other_words": p.other_words,
                     "before": p.before, "after": p.after, "verdict (ok / wrong / unsure)": ""})
    with open(out / "pauses.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def report_markdown(result: dict) -> str:
    s, ps = result["source"], result["pause_stats"]
    c = ps["counts"]
    lines = [
        "# Endpoint cost table: latency vs extra JEV calls (AMI pauses)",
        "",
        f"Audio: {s['description']}. VAD: Silero ({'ONNX' if result['vad']['onnx'] else 'JIT'}), "
        f"{result['vad']['ms_per_chunk']:.3f} ms per chunk. Smart Turn: {result.get('smart_turn') or 'not used'}. "
        f"CPU: {result['env']['machine']} {result['env']['processor']}.",
        "",
        f"Pauses (≥ {s['min_pause_ms']} ms in a speaker's speech): **{c['end']} ends (turns)**, {c['continue']} continue, "
        f"{c['unclear']} unclear (overlap at the stop, left out of the costs). Turn length p10 / p50 / p90: "
        + " / ".join(f"{ps['turn_s'][q]:.1f}" if ps['turn_s'][q] is not None else "–" for q in (10, 50, 90)) + " s.",
        "",
        "- **t1 − t0**: from the end of the speech (audio) to the endpoint, + the deciding Smart Turn call.",
        "- **JEV calls / turn**: all endpoints / turns (1.00 = ideal). **Extra**: at continue pauses + repeats "
        "(noise reopening a turn) + in speech (the VAD heard a pause the audio mask does not).",
        f"- **Missed**: turns with no endpoint (all / turns ≥ {s['min_turn_s']} s of speech).",
        "",
        "| setting | t1−t0 p50 | p90 | p95 | p99 | JEV calls / turn | extra / turn | at continue | repeat | in speech | at unclear / turn | missed % (all / long) |",
        "| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |",
    ]
    for r in result["settings"]:
        lat, k, t = r["latency_ms"], r["kinds"], max(r["turns"], 1)
        cells = " | ".join(f"{lat[f'p{q}']:.0f}" if lat[f"p{q}"] is not None else "–" for q in (50, 90, 95, 99))
        lines.append(f"| {r['setting']} | {cells} | {r['calls_per_turn']:.2f} | {r['extra_per_turn']:.2f} | "
                     f"{k['continue'] / t:.2f} | {k['repeat'] / t:.2f} | {k['in_speech'] / t:.2f} | "
                     f"{r['unclear_per_turn']:.2f} | {r['missed_pct']:.1f} / {r['missed_long_pct']:.1f} |")
    names = bin_names()
    lines += ["", "Extra calls at continue pauses per turn, by pause length (ms) — the short ones only a smarter "
              "detector can save; at ≥ 800 ms a plain VAD pays them too:", "",
              "| setting | " + " | ".join(names) + " |", "| :-- | " + " | ".join("--:" for _ in names) + " |"]
    for r in result["settings"]:
        lines.append(f"| {r['setting']} | " + " | ".join(f"{r['continue_by_pause_ms'][b]:.2f}" for b in names) + " |")
    lines += ["", "Labels alone (no model): pauses per label and length (ms):", "",
              "| label | " + " | ".join(names) + " |", "| :-- | " + " | ".join("--:" for _ in names) + " |"]
    for lab in LABELS:
        lines.append(f"| {lab} | " + " | ".join(str(ps["bins"][lab][b]) for b in names) + " |")
    ideal = ps["ideal_vad_extra_per_turn"]
    lines += ["", "An ideal silence-only detector with threshold T (exactly the audio pauses) would pay this many "
              "calls at continue pauses per turn:", "",
              "| T, ms | " + " | ".join(str(t) for t in ideal) + " |", "| :-- | " + " | ".join("--:" for _ in ideal) + " |",
              "| extra / turn | " + " | ".join(f"{v:.2f}" for v in ideal.values()) + " |"]
    turn = [r for r in result["settings"] if r.get("turn_call_ms")]
    if turn:
        lines += ["", "Smart Turn compute per call (ms, this CPU): " + ", ".join(
            f"{r['setting']} p50 {r['turn_call_ms']['p50']:.1f} / p95 {r['turn_call_ms']['p95']:.1f}" for r in turn)]
    return "\n".join(lines) + "\n"
