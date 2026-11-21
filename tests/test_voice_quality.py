import json

import numpy as np
import pytest

from jevlite.voice import quality as Q
from jevlite.voice.endpoint import SR

from test_voice_pauses import make_meeting


def test_keep_intervals_and_time_map():
    # runs at frames 100–150 and 600–650 (2 s, 12 s); 1000 frames; keep 3 s after (150 fr), 0.5 s before (25 fr)
    iv = Q.keep_intervals([(100, 150), (600, 650), (700, 710)], 1000)
    assert iv == [(75, 300), (575, 860)]
    tm = Q.TimeMap(iv)
    assert tm.n_frames == 225 + 285
    assert tm.frame(150) == 75 and tm.frame(600) == 225 + 25  # run end / next run start
    assert tm.frame(400) == 225  # inside a cut gap → the cut point
    assert tm.frame(10) == 0.0 and tm.frame(999) == 510
    assert tm.seconds(12.0) == pytest.approx(250 * 0.02)
    x = np.arange(1000 * Q.FRAME, dtype=np.float32)
    y = tm.audio(x)
    assert len(y) == 510 * Q.FRAME and y[0] == 75 * Q.FRAME and y[225 * Q.FRAME] == 575 * Q.FRAME


def test_word_errors_fallback():
    assert Q.word_errors("Hello, world!", "hello world") == (0, 2)
    assert Q.word_errors("a b c d", "a x c") == (2, 4)  # 1 substitution + 1 deletion
    assert Q.word_errors("", "noise") == (1, 0)


def test_pairs():
    ev = [{"kind": "partial"}, {"kind": "endpoint", "i": 1}, {"kind": "endpoint", "i": 2}, {"kind": "final", "i": 1}]
    p = Q.pairs(ev)
    assert [(e["i"], f["i"] if f else None) for e, f in p] == [(1, 1), (2, None)]


def ev(kind, t, text="", ms=0.0, detail=""):
    return {"kind": kind, "audio_s": t, "text": text, "compute_ms": ms, "detail": detail}


def test_eval_librispeech_t1_t2_wer():
    segs = [{"id": "u1", "text": "HELLO THERE", "start": SR, "end": 2 * SR},
            {"id": "u2", "text": "GOOD BYE", "start": 4 * SR, "end": 5 * SR}]
    meta = {"segments": segs, "reference": "HELLO THERE GOOD BYE"}
    events = [ev("endpoint", 1.5, ms=1.0, detail="silence"), ev("final", 1.5, "hello", 2.0),  # a cut inside u1
              ev("endpoint", 2.5, ms=1.0, detail="silence"), ev("final", 2.58, "there", 4.0),
              ev("endpoint", 5.6, ms=2.0, detail="eou"), ev("final", 5.6, "good bye")]
    rows, errs = Q.eval_librispeech(meta, events, 7 * SR)
    assert [r["premature"] for r in rows] == [1, 0]
    assert rows[0]["latency_ms"] == pytest.approx(501) and rows[0]["t2_ms"] == pytest.approx(501 + 80 + 4)
    assert rows[1]["latency_ms"] == pytest.approx(602) and rows[1]["t2_ms"] == pytest.approx(602)
    assert errs == (0, 4)
    s = Q.summarize_librispeech("x", rows, errs, compute_ms=70.0, audio_s=7.0)
    assert s["cut_utt_pct"] == 50 and s["missed_pct"] == 0 and s["wer"] == 0 and s["compute_ms_per_audio_s"] == 10


def test_prepare_ami_evaluate_and_report(tmp_path):
    sf = pytest.importorskip("soundfile")
    root = tmp_path / "ami"
    make_meeting(root, sf)  # A: 0.5–1.1, 1.5–2.0 s; B: 2.4–3.2 s; 6 s
    out = tmp_path / "streams"
    vad = lambda c: 1.0 if np.abs(c).mean() > 0.01 else 0.0  # noqa: E731
    index = Q.prepare_ami(root, out, meetings=("M1",), vad=vad, target_dbfs=None, log=lambda s: None)
    (out / "index.json").write_text(json.dumps(index))
    assert [i["id"] for i in index] == ["ami.M1.A", "ami.M1.B"]
    audio, meta, probs = Q.load_stream(out, "ami.M1.A")
    # A: kept 0–6 s whole (0.5 s before the first run, 3 s after the last: 2.0 + 3.0 = 5.0 s)
    assert meta["duration_s"] == pytest.approx(5.0) and len(probs) == len(audio) // 512
    assert [(p["start"], p["label"]) for p in meta["pauses"]] == [(1.1, "continue"), (2.0, "end")]
    assert meta["pauses"][1]["end"] == pytest.approx(5.0)  # the meeting end, cut to the kept audio
    assert meta["reference"] == "one two"

    dec = tmp_path / "decode"
    (dec / "parakeet-eou").mkdir(parents=True)
    a_events = [ev("endpoint", 2.4, ms=3.0, detail="eou"), ev("final", 2.4, "one two")]
    (dec / "parakeet-eou" / "ami.M1.A.json").write_text(json.dumps({"events": a_events, "compute_ms": 50.0,
                                                                    "audio_s": 5.0}))
    (dec / "parakeet-eou" / "ami.M1.B.json").write_text(json.dumps({"events": [], "compute_ms": 50.0,
                                                                    "audio_s": 5.0}))
    # B1 timeline for A: "one" at step 10 (0.8 s), "two" at step 22 (1.76 s), 1 lookahead step
    steps = [[[], 1.0] for _ in range(64)]
    steps[9][0], steps[21][0] = ["▁one"], ["▁two"]
    (dec / "nemotron-r1").mkdir()
    (dec / "nemotron-r1" / "ami.M1.A.json").write_text(json.dumps({"steps": steps, "compute_ms": 64.0, "audio_s": 5.0}))
    rows = Q.evaluate(out, dec, nemotron_settings=("vad-300",), log=lambda s: None)
    by = {r["setting"]: r for r in rows}
    p = by["parakeet-eou"]
    assert p["turns"] == 2 and p["latency_ms"]["p50"] == pytest.approx(403) and p["missed_pct"] == 50  # B: missed
    assert p["wer"] == pytest.approx(100 * 3 / 5)  # A right; B's 3 words lost
    assert p["compute_ms_per_audio_s"] == pytest.approx(10.0)
    n = by["nemotron-r1+vad-300"]
    assert n["turns"] == 1 and n["latency_ms"]["n"] == 1
    assert n["t2_ms"]["p50"] >= n["latency_ms"]["p50"] + 80 - 1e-6  # the final waits the lookahead step
    assert n["wer"] == 0 and n["compute_ms_per_audio_s"] == pytest.approx(64 / 5)
    md = Q.report_markdown({"gpu": "test", "summaries": rows})
    assert "| parakeet-eou | 403 |" in md and "nemotron-r1+vad-300" in md


def test_system_dirs():
    assert Q.system_dirs(["parakeet-eou", "nemotron+vad-300", "nemotron+vad-500", "nemotron-r6+vad-500",
                          "whisper+vad-500"]) == ["parakeet-eou", "nemotron-r1", "nemotron-r6", "whisper+vad-500"]
