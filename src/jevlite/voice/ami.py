"""AMI Meeting Corpus turns for the endpoint and ASR studies (todo Phase 9b). CC BY 4.0.

Source: the manual annotations ``ami_public_manual_1.6.2.zip`` (words with forced-aligned start / end
times per speaker, ``words/<meeting>.<agent>.words.xml``; agent → headset channel in
``corpusResources/meetings.xml``) and one headset recording per speaker
(``<meeting>.Headset-<channel>.wav``, 16 kHz mono), from the Edinburgh mirror.

**A turn** is a run of one speaker's words that ends when
- another speaker takes the floor: ≥ ``floor_words`` words of other speakers start inside the speaker's
  pause (1–2 word backchannels such as "yeah" do not end it), or
- the speaker is silent for ≥ ``split_pause_s`` (default 2 s) anyway.

**The word times absorb the pauses** (found 2026-11-18: 98–99% of gaps between a speaker's consecutive
words are 0, the last word of a turn lasts 0.5 s median, 1.2 s p90), so the annotated turn end can lie
well after the speech. ``refine_bounds`` moves the start and end to the speech, from the audio and
without a model: per 20 ms frame, the speaker's own headset is "speaking" when its level is ≥ the
speaker's noise floor + ``above_noise_db``, ≥ their speech level − ``below_peak_db``, and ≥ every other
headset + ``margin_db`` (so crosstalk from the others does not count). t0 = the end of the last speaking
frame inside the last word (the first word for the start); unchanged if no frame qualifies.
The in-turn pauses therefore stay in the audio (with any crosstalk), as they would live.

**Level**: the headsets differ a lot in gain (found 2026-11-19: turns the VAD never opened were ~18 dB
quieter, median −45 dBFS, mostly two EN2002a headsets), and a live microphone path has automatic gain
control. ``target_dbfs`` (default −20) scales each speaker so their speech level (99th percentile of
their 20 ms frame levels over the meeting) sits there, before the stream noise is added.

Punctuation tokens, vocal sounds and disfluency markers are not words; words without times are skipped.
Default meetings: the first session of four series of the standard AMI test split (``ES2004a``,
``IS1009a``, ``TS3003a``, ``EN2002a``), so the ASR models are less likely to have trained on them.
"""

from __future__ import annotations

import re
import shutil
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from .audio import Utterance
from .endpoint import SR

ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
AUDIO_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Headset-{c}.wav"
DEFAULT_MEETINGS = ("ES2004a", "IS1009a", "TS3003a", "EN2002a")
FRAME_S = 0.02
_UA = {"User-Agent": "jevlite-ami-fetch/1.0"}


def _fetch(url: str, dest: Path, tries: int = 5, wait_s: float = 30.0, log=print) -> None:
    """Download with retries: the mirror answers 503 at times (seen 2026-11-18)."""
    part = dest.with_name(dest.name + ".part")
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=_UA), timeout=120) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            part.rename(dest)
            return
        except (urllib.error.URLError, TimeoutError) as e:
            if i == tries - 1:
                raise
            log(f"  {e}; retry in {wait_s * (i + 1):.0f} s")
            time.sleep(wait_s * (i + 1))


@dataclass(frozen=True)
class Word:
    speaker: str  # NXT agent letter
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Turn:
    meeting: str
    speaker: str
    start: float  # speech start, seconds in the meeting (first word start, or refined)
    end: float  # speech end: t0 (last word end, or refined)
    words: tuple[Word, ...]

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    @property
    def max_pause(self) -> float:
        return max((b.start - a.end for a, b in zip(self.words, self.words[1:])), default=0.0)


def parse_words(path: str | Path, speaker: str) -> list[Word]:
    """The timed words of one ``*.words.xml`` (no punctuation, vocal sounds, markers or untimed words)."""
    out = []
    for el in ET.parse(path).getroot():
        if el.tag != "w" or el.get("punc") == "true" or el.get("starttime") is None or el.get("endtime") is None:
            continue
        start, end = float(el.get("starttime")), float(el.get("endtime"))
        if end > start and (el.text or "").strip():
            out.append(Word(speaker, start, end, el.text.strip()))
    return sorted(out, key=lambda w: w.start)


def channels(meetings_xml: str | Path, meeting: str) -> dict[str, int]:
    """NXT agent letter → headset channel for one meeting."""
    for m in ET.parse(meetings_xml).getroot():
        if m.get("observation") == meeting:
            return {s.get("nxt_agent"): int(s.get("channel")) for s in m if s.get("nxt_agent")}
    raise KeyError(f"{meeting} not in {meetings_xml}")


def build_turns(meeting: str, words: dict[str, list[Word]], floor_words: int = 3,
                split_pause_s: float = 2.0) -> list[Turn]:
    """Turns of every speaker (see the module doc), sorted by start."""
    turns = []
    for spk, ws in words.items():
        others = np.sort(np.array([w.start for s, o in words.items() if s != spk for w in o], dtype=float))
        cur = [ws[0]] if ws else []
        for a, b in zip(ws, ws[1:]):
            n_other = np.searchsorted(others, b.start, "left") - np.searchsorted(others, a.end, "right")
            if b.start - a.end >= split_pause_s or n_other >= floor_words:
                turns.append(Turn(meeting, spk, cur[0].start, cur[-1].end, tuple(cur)))
                cur = []
            cur.append(b)
        if cur:
            turns.append(Turn(meeting, spk, cur[0].start, cur[-1].end, tuple(cur)))
    return sorted(turns, key=lambda t: (t.start, t.speaker))


def frame_db(wav: str | Path, frame_s: float = FRAME_S) -> np.ndarray:
    """Level (dBFS RMS) of consecutive frames of a recording, read in blocks."""
    import soundfile as sf

    f = int(round(frame_s * SR))
    out = []
    for block in sf.blocks(str(wav), blocksize=f * 4096, dtype="float32", always_2d=False):
        n = len(block) // f
        if n:
            rms = np.sqrt(np.mean(np.square(block[: n * f].reshape(n, f).astype(np.float64)), axis=1))
            out.append(20 * np.log10(np.maximum(rms, 1e-10)))
    return np.concatenate(out) if out else np.zeros(0)


def speaking_mask(levels: dict[str, np.ndarray], above_noise_db: float = 10.0, below_peak_db: float = 30.0,
                  margin_db: float = 3.0) -> dict[str, np.ndarray]:
    """Per speaker, the frames where their own headset carries their speech (see the module doc)."""
    n = min(len(v) for v in levels.values())
    lv = {s: v[:n] for s, v in levels.items()}
    out = {}
    for s, v in lv.items():
        others = np.max(np.stack([o for k, o in lv.items() if k != s]), axis=0) if len(lv) > 1 else np.full(n, -np.inf)
        floor, peak = np.percentile(v, 10), np.percentile(v, 99)
        out[s] = (v >= floor + above_noise_db) & (v >= peak - below_peak_db) & (v >= others + margin_db)
    return out


def refine_bounds(turn: Turn, mask: np.ndarray, frame_s: float = FRAME_S) -> Turn:
    """The turn with its start / end moved to the first / last speaking frame of its first / last word."""
    def frames(w):
        return max(0, int(np.floor(w.start / frame_s))), min(len(mask), int(np.ceil(w.end / frame_s)))

    a, b = frames(turn.words[-1])
    hit = np.flatnonzero(mask[a:b])
    end = min(turn.end, (a + hit[-1] + 1) * frame_s) if len(hit) else turn.end
    a, b = frames(turn.words[0])
    hit = np.flatnonzero(mask[a:b])
    start = max(turn.start, (a + hit[0]) * frame_s) if len(hit) else turn.start
    return replace(turn, start=start, end=end) if end > start else turn


def download(root: str | Path, meetings=DEFAULT_MEETINGS, log=print) -> None:
    """Fetch the annotations (23 MB) and the headset recordings of ``meetings`` (~30–80 MB per speaker)
    into ``root`` if missing."""
    root = Path(root)
    ann = root / "annotations"
    if not (ann / "corpusResources" / "meetings.xml").exists():
        root.mkdir(parents=True, exist_ok=True)
        z = root / "ami_public_manual_1.6.2.zip"
        if not z.exists():
            log(f"downloading {ANNOTATIONS_URL}")
            _fetch(ANNOTATIONS_URL, z, log=log)
        with zipfile.ZipFile(z) as f:
            f.extractall(ann)
    for m in meetings:
        for c in sorted(channels(ann / "corpusResources" / "meetings.xml", m).values()):
            wav = root / "audio" / f"{m}.Headset-{c}.wav"
            if not wav.exists():
                wav.parent.mkdir(parents=True, exist_ok=True)
                log(f"downloading {m} headset {c}")
                _fetch(AUDIO_URL.format(m=m, c=c), wav, log=log)


def meeting_turns(root: str | Path, meeting: str, floor_words: int = 3, split_pause_s: float = 2.0) -> list[Turn]:
    ann = Path(root) / "annotations"
    words = {}
    for p in sorted((ann / "words").glob(f"{meeting}.*.words.xml")):
        spk = re.fullmatch(rf"{re.escape(meeting)}\.([A-Z])\.words\.xml", p.name)
        if spk:
            words[spk.group(1)] = parse_words(p, spk.group(1))
    if not words:
        raise FileNotFoundError(f"no word annotations for {meeting} under {ann}")
    return build_turns(meeting, words, floor_words, split_pause_s)


def load_ami(root: str | Path, n: int, meetings=DEFAULT_MEETINGS, seed: int = 0, min_turn_s: float = 0.5,
             floor_words: int = 3, split_pause_s: float = 2.0, refine: bool = True,
             target_dbfs: float | None = -20.0, stats: dict | None = None) -> list[Utterance]:
    """``n`` turns drawn at random (fixed by ``seed``) from ``meetings``, each cut from its speaker's headset
    recording at its speech bounds (``refine``: moved to the audio, see the module doc; else the
    annotation's). Turns shorter than ``min_turn_s`` after refining are left out (the VAD needs 200 ms of
    speech to open a turn). The id carries the meeting, speaker and times. ``stats`` (a dict) receives the
    shifts of the refined ends vs the annotation, in ms, under ``end_shift_ms``, and the gain per speaker
    (dB) under ``gain_db``. ``target_dbfs``: speaker level normalisation (module doc); None = raw."""
    import soundfile as sf

    root = Path(root)
    chan = {m: channels(root / "annotations" / "corpusResources" / "meetings.xml", m) for m in meetings}
    pool, shifts, gain = [], [], {}
    for m in meetings:
        turns = meeting_turns(root, m, floor_words, split_pause_s)
        if refine or target_dbfs is not None:
            levels = {s: frame_db(root / "audio" / f"{m}.Headset-{c}.wav") for s, c in chan[m].items()}
            gain.update({(m, s): target_dbfs - float(np.percentile(v, 99)) for s, v in levels.items()}
                        if target_dbfs is not None else {})
        if refine:
            masks = speaking_mask(levels)
            refined = [refine_bounds(t, masks[t.speaker]) for t in turns]
            shifts += [(t.end - r.end) * 1000 for t, r in zip(turns, refined)]
            turns = refined
        pool += [t for t in turns if t.end - t.start >= min_turn_s]
    if not pool:
        raise ValueError(f"no AMI turns ≥ {min_turn_s} s in {meetings}")
    if stats is not None:
        stats["end_shift_ms"] = shifts
        stats["gain_db"] = {f"{m}.{s}": g for (m, s), g in gain.items()}
    pick = sorted(np.random.default_rng(seed).choice(len(pool), size=min(n, len(pool)), replace=False))
    out = []
    for i in pick:
        t = pool[i]
        wav = root / "audio" / f"{t.meeting}.Headset-{chan[t.meeting][t.speaker]}.wav"
        a, b = int(round(t.start * SR)), int(round(t.end * SR))
        audio, sr = sf.read(wav, start=a, stop=b, dtype="float32", always_2d=False)
        if sr != SR or audio.ndim != 1:
            raise ValueError(f"{wav}: expected 16 kHz mono, got {sr} Hz, shape {audio.shape}")
        if (t.meeting, t.speaker) in gain:
            audio = np.clip(audio * np.float32(10 ** (gain[(t.meeting, t.speaker)] / 20)), -1.0, 1.0)
        out.append(Utterance(f"{t.meeting}.{t.speaker}.{t.start:.2f}-{t.end:.2f}", audio, t.text))
    return out
