import numpy as np
import pytest

from jevlite.voice.ami import (FRAME_S, Turn, Word, build_turns, channels, frame_db, load_ami, meeting_turns,
                               parse_words, refine_bounds, speaking_mask)
from jevlite.voice.endpoint import SR

WORDS_XML = """<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>
<nite:root nite:id="M1.A.words" xmlns:nite="http://nite.sourceforge.net/">
   <w nite:id="a0" starttime="0.5" endtime="0.9">Hello</w>
   <w nite:id="a1" starttime="0.9" endtime="0.9" punc="true">,</w>
   <vocalsound nite:id="a2" starttime="0.9" endtime="1.0" type="cough"/>
   <w nite:id="a3" starttime="1.0" endtime="1.4">there</w>
   <w nite:id="a4">untimed</w>
   <disfmarker nite:id="a5" starttime="1.4" endtime="1.4"/>
   <w nite:id="a6" starttime="1.5" endtime="1.5">zero</w>
</nite:root>
"""

MEETINGS_XML = """<?xml version="1.0" encoding="ISO-8859-1" standalone="yes"?>
<nite:root xmlns:nite="http://nite.sourceforge.net/">
   <meeting nite:id="meet_1" observation="M1">
      <speaker nite:id="M1_1" channel="1" nxt_agent="A"/>
      <speaker nite:id="M1_2" channel="0" nxt_agent="B"/>
   </meeting>
</nite:root>
"""


def w(spk, start, end, text="x"):
    return Word(spk, start, end, text)


def test_parse_words_keeps_timed_words_only(tmp_path):
    p = tmp_path / "M1.A.words.xml"
    p.write_text(WORDS_XML, encoding="iso-8859-1")
    words = parse_words(p, "A")
    assert [x.text for x in words] == ["Hello", "there"]
    assert words[1].start == 1.0 and words[1].end == 1.4 and words[0].speaker == "A"


def test_channels(tmp_path):
    p = tmp_path / "meetings.xml"
    p.write_text(MEETINGS_XML)
    assert channels(p, "M1") == {"A": 1, "B": 0}
    with pytest.raises(KeyError):
        channels(p, "M2")


def test_turns_backchannel_floor_change_and_long_pause():
    a = [w("A", 0.0, 0.5), w("A", 1.2, 1.6),  # B's "yeah" (1 word) in the pause: same turn
         w("A", 3.0, 3.4),  # B says 3 words in the pause: floor change → new turn
         w("A", 6.0, 6.5)]  # 2.6 s pause, nobody else: ≥ 2 s → new turn
    b = [w("B", 0.7, 0.9), w("B", 1.8, 2.0), w("B", 2.1, 2.3), w("B", 2.4, 2.7)]
    turns = build_turns("M", {"A": a, "B": b})
    ta = [t for t in turns if t.speaker == "A"]
    assert [(t.start, t.end) for t in ta] == [(0.0, 1.6), (3.0, 3.4), (6.0, 6.5)]
    assert ta[0].max_pause == pytest.approx(0.7)
    tb = [t for t in turns if t.speaker == "B"]
    assert [(t.start, t.end) for t in tb] == [(0.7, 2.7)]  # only 1 word of A between B's words: B keeps the turn
    assert [t.start for t in turns] == sorted(t.start for t in turns)


def test_load_ami_cuts_turn_audio(tmp_path):
    sf = pytest.importorskip("soundfile")
    root = tmp_path / "ami"
    (root / "annotations" / "words").mkdir(parents=True)
    (root / "annotations" / "corpusResources").mkdir()
    (root / "annotations" / "corpusResources" / "meetings.xml").write_text(MEETINGS_XML)
    (root / "annotations" / "words" / "M1.A.words.xml").write_text(WORDS_XML, encoding="iso-8859-1")
    (root / "audio").mkdir()
    sig = np.arange(3 * SR, dtype=np.float32) / (3 * SR)
    sf.write(root / "audio" / "M1.Headset-1.wav", sig, SR, subtype="FLOAT")
    turns = meeting_turns(root, "M1")
    assert len(turns) == 1 and turns[0].text == "Hello there"
    (u,) = load_ami(root, 5, meetings=("M1",), min_turn_s=0.5, refine=False, target_dbfs=None)
    assert u.id == "M1.A.0.50-1.40" and u.text == "Hello there"
    assert len(u.audio) == int(1.4 * SR) - int(0.5 * SR)
    assert u.audio[0] == pytest.approx(sig[int(0.5 * SR)])
    with pytest.raises(ValueError):
        load_ami(root, 5, meetings=("M1",), min_turn_s=2.0, refine=False, target_dbfs=None)

    # refine: A speaks 0.5–1.1 s (the annotation says to 1.4); B's headset hears A as crosstalk, 20 dB lower
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1e-4, 3 * SR).astype(np.float32)
    a[int(0.5 * SR):int(1.1 * SR)] += rng.normal(0, 0.1, int(0.6 * SR)).astype(np.float32)
    sf.write(root / "audio" / "M1.Headset-1.wav", a, SR, subtype="FLOAT")
    sf.write(root / "audio" / "M1.Headset-0.wav", a * 0.1, SR, subtype="FLOAT")
    stats = {}
    (u,) = load_ami(root, 5, meetings=("M1",), min_turn_s=0.3, target_dbfs=None, stats=stats)
    assert u.id == "M1.A.0.50-1.10"
    assert stats["end_shift_ms"] == [pytest.approx(300)]
    raw_db = 20 * np.log10(np.sqrt(np.mean(u.audio ** 2)))
    (lv,) = load_ami(root, 5, meetings=("M1",), min_turn_s=0.3, target_dbfs=-10.0, stats=stats)
    assert stats["gain_db"]["M1.A"] == pytest.approx(-10.0 - (-20.0), abs=0.5)  # speech at ~−20 dBFS → +10 dB
    assert 20 * np.log10(np.sqrt(np.mean(lv.audio ** 2))) == pytest.approx(raw_db + stats["gain_db"]["M1.A"], abs=0.01)


def test_frame_db(tmp_path):
    sf = pytest.importorskip("soundfile")
    x = np.full(SR, 0.1, np.float32)
    sf.write(tmp_path / "x.wav", x, SR, subtype="FLOAT")
    db = frame_db(tmp_path / "x.wav")
    assert len(db) == int(1 / FRAME_S) and np.allclose(db, -20, atol=0.01)


def test_speaking_mask_and_refine():
    n = 100
    noise = np.full(n, -80.0)
    own = noise.copy()
    own[20:40] = -20  # A speaks
    own[60:70] = -45  # crosstalk of B in A's headset
    other = noise.copy()
    other[60:70] = -20  # B speaks
    other[20:40] = -45  # crosstalk of A
    m = speaking_mask({"A": own, "B": other})
    assert m["A"][20:40].all() and not m["A"][60:70].any() and m["B"][60:70].all() and not m["B"][20:40].any()
    t = Turn("M", "A", 0.3, 1.2, (Word("A", 0.3, 0.6, "hi"), Word("A", 0.6, 1.2, "there")))
    r = refine_bounds(t, m["A"])
    assert r.start == pytest.approx(0.4) and r.end == pytest.approx(0.8)
    silent = refine_bounds(t, np.zeros(n, bool))
    assert (silent.start, silent.end) == (0.3, 1.2)
