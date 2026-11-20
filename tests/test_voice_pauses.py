import csv

import numpy as np
import pytest

from jevlite.voice import pauses as P
from jevlite.voice.ami import Word
from jevlite.voice.endpoint import SR, Endpoint

MEETINGS_XML = """<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
   <meeting nite:id="meet_1" observation="M1">
      <speaker nite:id="M1_1" channel="1" nxt_agent="A"/>
      <speaker nite:id="M1_2" channel="0" nxt_agent="B"/>
   </meeting>
</nite:root>
"""


def words_xml(ws):
    body = "\n".join(f'   <w nite:id="w{i}" starttime="{a}" endtime="{b}">{t}</w>' for i, (a, b, t) in enumerate(ws))
    return f'<?xml version="1.0" encoding="ISO-8859-1"?>\n<nite:root xmlns:nite="http://nite.sourceforge.net/">\n{body}\n</nite:root>\n'


def w(spk, start, end, text="x"):
    return Word(spk, start, end, text)


def ep(t, decide_ms=0.0):
    return Endpoint(sample=int(round(t * SR)), start_sample=0, reason="silence", decide_ms=decide_ms, turn_calls=0,
                    turn_ms=0.0, p_turn=None)


def test_runs_and_active_runs():
    m = np.zeros(40, bool)
    m[2:10] = m[13:20] = True  # gap of 3 frames (60 ms) → merged at 150 ms
    m[30] = True  # 1 frame (20 ms) → dropped at 40 ms
    assert P.runs(m) == [(2, 10), (13, 20), (30, 31)]
    assert P.active_runs(m) == [(2, 20)]
    assert P.active_runs(m, min_pause_s=0.04) == [(2, 10), (13, 20)]


def test_word_mask():
    m = P.word_mask([w("A", 0.1, 0.2)], 20, pad_s=0.0)
    assert np.flatnonzero(m).tolist() == [5, 6, 7, 8, 9]


# Speaker A: runs 0–1.0, 1.2–2.0, 3.0–4.0, 8.0–9.0, 10.0–11.0, 12.5–13.0 s; meeting 20 s.
A_RUNS = [(0, 50), (60, 100), (150, 200), (400, 450), (500, 550), (625, 650)]
WORDS = {
    "A": [w("A", 0.0, 1.0, "so"), w("A", 1.2, 2.0, "then"), w("A", 3.0, 4.0, "okay"), w("A", 8.0, 9.0, "and"),
          w("A", 10.0, 11.0, "but"), w("A", 12.5, 13.0, "right")],
    "B": [w("B", 1.04, 1.12, "yeah"), w("B", 2.1, 2.3, "I"), w("B", 2.3, 2.5, "think"), w("B", 2.5, 2.9, "so")],
    "C": [w("C", 10.4, 10.8, "wait"), w("C", 11.1, 11.3, "no"), w("C", 11.4, 11.7, "that's"), w("C", 11.8, 12.0, "it")],
}
ACTIVE = {"A": A_RUNS, "B": [(52, 56), (105, 145)], "C": [(520, 600)]}


def labelled():
    return [p for p in P.label_pauses("M", WORDS, ACTIVE, 20.0) if p.speaker == "A"]


def test_label_pauses():
    a = labelled()
    got = [(p.start, p.end, p.label, p.other_words) for p in a]
    assert got == [
        (1.0, 1.2, "continue", 1),  # B's "yeah" only: A keeps the floor
        (2.0, 3.0, "end", 3),  # B says 3 words: floor change
        (4.0, 8.0, "end", 0),  # ≥ 2 s of silence
        (9.0, 10.0, "continue", 0),
        (11.0, 12.5, "unclear", 3),  # C had been talking for 0.6 s when A stopped
        (13.0, 20.0, "end", 0),  # to the end of the meeting, ≥ 2 s
    ]
    assert a[0].before == "so" and a[0].after.startswith("then okay")  # up to 6 words
    assert a[1].after == "I think so"  # the others' words, not A's
    assert a[1].turn_s == pytest.approx(2.0) and a[2].turn_s == pytest.approx(1.0)  # turns restart after an end
    assert a[2].ms == pytest.approx(4000)
    short = P.label_pauses("M", WORDS, ACTIVE, 14.0)  # the meeting ends 1 s after A's last run: dropped
    assert all(p.start != 13.0 for p in short if p.speaker == "A")


def test_match_pauses_and_summary():
    a = labelled()
    endpoints = [ep(1.15), ep(2.45, decide_ms=45.0), ep(2.8), ep(3.5), ep(11.6), ep(13.4)]
    er, pr = P.match_pauses(a, A_RUNS, endpoints)
    assert [r["kind"] for r in er] == ["continue", "end", "repeat", "in_speech", "unclear", "end"]
    assert er[1]["latency_ms"] == pytest.approx(450 + 45) and er[5]["latency_ms"] == pytest.approx(400)
    assert er[0]["pause_ms"] == pytest.approx(200)
    assert [r["calls"] for r in pr] == [1, 2, 0, 0, 1, 1]
    assert pr[2]["latency_ms"] is None  # the 4 s end pause got no endpoint: missed
    s = P.summarize_setting("vad-x", er, pr, min_turn_s=0.5)
    assert s["turns"] == 3 and s["endpoints"] == 6
    assert s["calls_per_turn"] == pytest.approx(2.0)
    assert s["extra_per_turn"] == pytest.approx(1.0)  # continue + repeat + in speech
    assert s["unclear_per_turn"] == pytest.approx(1 / 3)
    assert s["missed_pct"] == pytest.approx(100 / 3)
    assert s["latency_ms"]["n"] == 2 and s["latency_ms"]["p50"] == pytest.approx((495 + 400) / 2)
    assert s["continue_by_pause_ms"]["150–300"] == pytest.approx(1 / 3)


def test_pause_stats_and_report():
    st = P.pause_stats(labelled(), thresholds_ms=(200, 500, 1000))
    assert st["counts"] == {"end": 3, "continue": 2, "unclear": 1}
    assert st["ideal_vad_extra_per_turn"] == {200: pytest.approx(2 / 3), 500: pytest.approx(1 / 3), 1000: pytest.approx(1 / 3)}
    assert st["bins"]["continue"]["150–300"] == 1 and st["bins"]["continue"]["800–2000"] == 1
    er, pr = P.match_pauses(labelled(), A_RUNS, [ep(2.45)])
    result = {"source": {"description": "test", "min_pause_ms": 150, "min_turn_s": 0.5}, "pause_stats": st,
              "vad": {"onnx": True, "ms_per_chunk": 0.1}, "smart_turn": None,
              "env": {"machine": "x", "processor": "y"}, "settings": [P.summarize_setting("vad-500", er, pr)]}
    md = P.report_markdown(result)
    assert "| vad-500 | 450 |" in md and "**3 ends (turns)**" in md


def test_mute_gain():
    own = np.array([1, 1, 0, 0, 0, 0, 1], bool)
    others = np.array([0, 0, 1, 1, 1, 0, 0], bool)
    assert P.mute_gain(own, others, fade=1).tolist() == [1, 1, 0, 0, 0, 1, 1]
    g = P.mute_gain(own, others, fade=3)
    assert len(g) == 7 and g[3] == 0 and 0 < g[2] < 1 and g[0] == 1
    assert P.frame_gain_to_samples(np.array([1.0, 0.0]), 2 * 320 + 5).tolist()[319:321] == [1.0, 0.0]


def make_meeting(root, sf):
    (root / "annotations" / "words").mkdir(parents=True)
    (root / "annotations" / "corpusResources").mkdir()
    (root / "annotations" / "corpusResources" / "meetings.xml").write_text(MEETINGS_XML)
    # A: speech 0.5–1.1 and 1.5–2.0 (word times absorb the 400 ms pause), B: 2.4–3.2; 6 s
    (root / "annotations" / "words" / "M1.A.words.xml").write_text(
        words_xml([(0.5, 1.1, "one"), (1.1, 2.0, "two")]), encoding="iso-8859-1")
    (root / "annotations" / "words" / "M1.B.words.xml").write_text(
        words_xml([(2.4, 2.6, "a"), (2.6, 2.9, "b"), (2.9, 3.2, "c")]), encoding="iso-8859-1")
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1e-4, 6 * SR).astype(np.float32)
    b = rng.normal(0, 1e-4, 6 * SR).astype(np.float32)
    for lo, hi in ((0.5, 1.1), (1.5, 2.0)):
        a[int(lo * SR):int(hi * SR)] += rng.normal(0, 0.1, int(hi * SR) - int(lo * SR)).astype(np.float32)
    b[int(2.4 * SR):int(3.2 * SR)] += rng.normal(0, 0.1, int(0.8 * SR)).astype(np.float32)
    (root / "audio").mkdir()
    sf.write(root / "audio" / "M1.Headset-1.wav", a + 0.1 * b, SR, subtype="FLOAT")  # A, with B's crosstalk
    sf.write(root / "audio" / "M1.Headset-0.wav", b + 0.1 * a, SR, subtype="FLOAT")


def test_load_meeting_stream_and_check(tmp_path):
    sf = pytest.importorskip("soundfile")
    root = tmp_path / "ami"
    make_meeting(root, sf)
    mt = P.load_meeting(root, "M1", target_dbfs=None)
    assert mt.duration_s == pytest.approx(6.0)
    assert mt.active["A"] == [(25, 55), (75, 100)] and mt.active["B"] == [(120, 160)]
    got = [(p.speaker, p.start, p.end, p.label) for p in mt.pauses]
    assert got == [("A", 1.1, 1.5, "continue"), ("A", 2.0, 6.0, "end"), ("B", 3.2, 6.0, "end")]

    s = P.speaker_stream(root, mt, "A", noise_db=None)
    rms = lambda lo, hi: float(np.sqrt(np.mean(s[int(lo * SR):int(hi * SR)] ** 2)))  # noqa: E731
    assert rms(0.6, 1.0) > 0.05  # A's own speech kept
    assert rms(2.5, 3.1) < 1e-6  # B's crosstalk muted
    kept = P.speaker_stream(root, mt, "A", mute_others=False, noise_db=None)
    assert float(np.sqrt(np.mean(kept[int(2.5 * SR):int(3.1 * SR)] ** 2))) > 5e-3

    rows = P.export_check(root, [mt], tmp_path / "check", n=3, seed=0)
    assert sorted(r["label"] for r in rows) == ["continue", "end", "end"]  # no unclear pauses to draw from
    files = sorted(p.name for p in (tmp_path / "check").glob("*.wav"))
    assert len(files) == 3 and all(f.split("_")[1] in ("end", "continue") for f in files)
    with open(tmp_path / "check" / "pauses.csv") as f:
        assert len(list(csv.DictReader(f))) == 3
