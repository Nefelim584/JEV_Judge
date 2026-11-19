import numpy as np
import pytest

from jevlite.voice.audio import Segment, Utterance, build_stream, librispeech_index, speech_bounds
from jevlite.voice.endpoint import CHUNK, SR, Endpointer, fit_turn_audio, parse_setting
from jevlite.voice.endpoint_study import match, report_markdown, run_setting, summarize

MS = CHUNK * 1000 / SR  # 32 ms per chunk


def chunks_for(pattern):
    """pattern: list of (p_speech, n_chunks) → per-chunk probabilities."""
    return np.concatenate([np.full(n, p, np.float32) for p, n in pattern])


def run(config, probs, turn_model=None):
    stream = np.zeros(len(probs) * CHUNK, np.float32)
    return run_setting(stream, probs, config, turn_model)


def test_parse_setting():
    assert parse_setting("vad-500").silence_ms == 500 and not parse_setting("vad-500").smart_turn
    t = parse_setting("turn-200")
    assert t.smart_turn and t.silence_ms == 200 and t.max_silence_ms == 3000
    for bad in ("vad", "turn-x", "silero-300", "turn-3000"):
        with pytest.raises(ValueError):
            parse_setting(bad)


def test_vad_endpoint_after_silence():
    # 0.5 s idle, 2 s speech, 1 s silence
    probs = chunks_for([(0.0, 16), (0.9, 63), (0.0, 32)])
    (e,) = run(parse_setting("vad-500"), probs)
    speech_end = (16 + 63) * CHUNK
    assert e.reason == "silence" and e.decide_ms == 0
    assert 500 <= (e.sample - speech_end) * 1000 / SR < 500 + MS
    assert e.start_sample == 16 * CHUNK


def test_short_blip_does_not_open_a_turn():
    probs = chunks_for([(0.0, 10), (0.9, 3), (0.0, 40)])  # 96 ms < start_ms 200
    assert run(parse_setting("vad-300"), probs) == []


def test_in_between_probabilities_neither_start_nor_end_a_pause():
    # speech, then 0.4 (between neg 0.35 and 0.5) for a long time: no pause starts
    probs = chunks_for([(0.9, 20), (0.4, 40)])
    assert run(parse_setting("vad-300"), probs) == []
    # pause started by 0.1, continued through 0.4 chunks: still ends
    probs = chunks_for([(0.9, 20), (0.1, 2), (0.4, 20)])
    (e,) = run(parse_setting("vad-300"), probs)
    assert (e.sample - 20 * CHUNK) * 1000 / SR >= 300


def test_short_pause_inside_speech_is_not_an_endpoint():
    probs = chunks_for([(0.9, 30), (0.0, 6), (0.9, 30), (0.0, 30)])  # 192 ms pause < 300
    (e,) = run(parse_setting("vad-300"), probs)
    assert e.sample > 66 * CHUNK


def test_smart_turn_complete_ends_early():
    calls = []

    def turn(audio):
        calls.append(len(audio))
        return 0.9

    probs = chunks_for([(0.9, 40), (0.0, 40)])
    (e,) = run(parse_setting("turn-200"), probs, turn)
    assert e.reason == "turn" and e.turn_calls == 1 and len(calls) == 1
    assert 200 <= (e.sample - 40 * CHUNK) * 1000 / SR < 200 + MS
    assert calls[0] >= 40 * CHUNK  # the whole turn so far


def test_smart_turn_incomplete_waits_rechecks_after_new_speech_then_times_out():
    p = iter([0.1, 0.2])
    probs = chunks_for([(0.9, 20), (0.0, 10), (0.9, 10), (0.0, 120)])  # second pause 3.84 s
    (e,) = run(parse_setting("turn-200"), probs, lambda a: next(p))
    assert e.reason == "timeout" and e.turn_calls == 2 and e.p_turn == pytest.approx(0.2)
    assert 3000 <= (e.sample - 40 * CHUNK) * 1000 / SR < 3000 + MS


def test_smart_turn_required_for_turn_setting():
    with pytest.raises(ValueError):
        Endpointer(parse_setting("turn-200"))


def test_turn_buffer_keeps_last_seconds():
    lengths = []
    probs = chunks_for([(0.9, int(12 * SR / CHUNK)), (0.0, 10)])
    run(parse_setting("turn-200"), probs, lambda a: lengths.append(len(a)) or 0.9)
    assert 8 * SR <= lengths[0] <= 9 * SR + CHUNK


def test_fit_turn_audio_pads_front_and_keeps_end():
    a = np.arange(1, 101, dtype=np.float32)
    out = fit_turn_audio(a, seconds=1)
    assert len(out) == SR and out[-1] == 100 and out[0] == 0
    long = np.arange(2 * SR, dtype=np.float32)
    assert np.array_equal(fit_turn_audio(long, seconds=1), long[-SR:])


def test_speech_bounds_trims_silence():
    a = np.zeros(SR, np.float32)
    a[4000:8000] = 0.5
    s, e = speech_bounds(a)
    assert 3680 <= s <= 4000 and 8000 <= e <= 8320
    assert speech_bounds(np.zeros(SR, np.float32)) == (0, SR)


def test_build_stream_segments_point_at_the_speech():
    rng = np.random.default_rng(1)
    utts = []
    for i in range(3):
        a = np.zeros(SR, np.float32)
        a[3200:12800] = rng.uniform(-0.5, 0.5, 9600)
        utts.append(Utterance(f"u{i}", a, f"text {i}"))
    stream, segs = build_stream(utts, (1.0, 2.0), lead_s=0.5, noise_db=None, seed=0)
    assert segs[0].start == SR // 2
    for s in segs:
        assert s.end - s.start == 9600
        assert np.all(stream[s.start:s.end] != 0)
    for a, b in zip(segs, segs[1:]):
        assert SR <= b.start - a.end <= 2 * SR
        assert not np.any(stream[a.end:b.start])
    noisy, _ = build_stream(utts, (1.0, 2.0), noise_db=-60, seed=0)
    assert 0 < np.std(noisy[: SR // 4]) < 0.01


def test_match_and_summarize():
    from jevlite.voice.endpoint import Endpoint

    def ep(sample, reason="silence", ms=0.0):
        return Endpoint(sample, 0, reason, ms, 0, 0.0, None)

    segs = [Segment("a", "", 1000, 20000), Segment("b", "", 60000, 90000), Segment("c", "", 130000, 150000)]
    eps = [ep(10000), ep(20000 + 8000), ep(40000), ep(95000, "turn", 12.0)]  # a: cut + endpoint + extra; c: missed
    rows = match(segs, eps, 170000)
    a, b, c = rows
    assert a["premature"] == 1 and a["extra"] == 1 and a["wait_ms"] == pytest.approx(500)
    assert b["latency_ms"] == pytest.approx(5000 * 1000 / SR + 12.0) and b["reason"] == "turn"
    assert c["missed"] and c["latency_ms"] is None
    s = summarize(parse_setting("vad-500"), rows, eps)
    assert s["missed_pct"] == pytest.approx(100 / 3) and s["premature_utt_pct"] == pytest.approx(100 / 3)
    assert s["latency_ms"]["n"] == 2
    md = report_markdown({
        "stream": {"source": "test", "utterances": 3, "speech_min": 0.1, "gap_s": [1, 2], "noise_db": None, "seed": 0},
        "vad": {"onnx": True, "ms_per_chunk": 0.1}, "smart_turn": None,
        "env": {"machine": "x", "processor": "y"}, "settings": [s],
    })
    assert "| vad-500 |" in md


def test_librispeech_index(tmp_path):
    d = tmp_path / "test-clean" / "1" / "2"
    d.mkdir(parents=True)
    (d / "1-2.trans.txt").write_text("1-2-0000 HELLO WORLD\n1-2-0001 SECOND ONE\n")
    idx = librispeech_index(tmp_path / "test-clean")
    assert idx["1-2-0001"] == (d / "1-2-0001.flac", "SECOND ONE")
    with pytest.raises(FileNotFoundError):
        librispeech_index(tmp_path / "nothing")


def test_noise_floor_db():
    from jevlite.voice.audio import noise_floor_db

    rng = np.random.default_rng(0)
    a = rng.normal(0, 1e-3, SR).astype(np.float32)  # floor −60 dB re a 1.0-RMS burst
    a[4000:8000] = rng.choice([-1.0, 1.0], 4000)
    assert -62 < noise_floor_db(a) < -58
    assert noise_floor_db(np.zeros(10, np.float32)) == 0.0


def test_parse_setting_threshold_and_timeout():
    from jevlite.voice.endpoint import TUNE_SETTINGS

    c = parse_setting("turn-200-p0.7-m800")
    assert c.turn_threshold == 0.7 and c.max_silence_ms == 800 and c.silence_ms == 200
    assert parse_setting("turn-200-m1200").turn_threshold == 0.5
    assert parse_setting("turn-200-p0.3").max_silence_ms == 3000
    for bad in ("vad-500-p0.5", "vad-500-m800", "turn-800-m800", "turn-200-p2", "turn-200-m800-p0.5"):
        with pytest.raises(ValueError):
            parse_setting(bad)
    assert len({parse_setting(s).name for s in TUNE_SETTINGS}) == len(TUNE_SETTINGS)


def test_smart_turn_threshold_and_short_timeout():
    (e,) = run(parse_setting("turn-200-p0.7"), chunks_for([(0.9, 40), (0.0, 100)]), lambda a: 0.6)
    assert e.reason == "timeout"  # 0.6 < 0.7: waits for the 3 s timeout
    probs = chunks_for([(0.9, 40), (0.0, 40)])
    (e,) = run(parse_setting("turn-200-p0.7-m800"), probs, lambda a: 0.6)
    assert e.reason == "timeout" and 800 <= (e.sample - 40 * CHUNK) * 1000 / SR < 800 + MS
    (e,) = run(parse_setting("turn-200-p0.5-m800"), probs, lambda a: 0.6)
    assert e.reason == "turn"


def test_turn_cache_shared_between_settings():
    calls = []

    def turn(audio):
        calls.append(len(audio))
        return 0.4

    probs = chunks_for([(0.9, 40), (0.0, 30), (0.9, 40), (0.0, 120)])
    cache = {}
    stream = np.zeros(len(probs) * CHUNK, np.float32)
    a = run_setting(stream, probs, parse_setting("turn-200-p0.5-m3000"), turn, cache)
    n = len(calls)
    b = run_setting(stream, probs, parse_setting("turn-200-p0.3-m3000"), turn, cache)
    assert len(calls) == n + 1  # the first pause reused; b's second turn starts elsewhere: new audio, new call
    assert [e.reason for e in a] == ["timeout"] and [e.reason for e in b] == ["turn", "turn"]
    assert b[0].decide_ms == cache[(b[0].start_sample, b[0].sample)][1]
