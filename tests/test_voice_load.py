import dataclasses
import time
from concurrent.futures import Future

import numpy as np
import pytest

from jevlite.voice import asr as A
from jevlite.voice import load as L
from jevlite.voice.endpoint import CHUNK, SR

from test_voice_asr import fake_preprocess, state_spec

STEP = 2560  # 160 ms, the real step (5 VAD chunks)


def spec16():
    return dataclasses.replace(state_spec(), step_s=0.16)


def echo(items):
    return [it[1] for it in items]


def test_batcher_groups_requests_within_the_wait_and_respects_max_seqs():
    seen = []

    def forward(items):
        seen.append(len(items))
        return echo(items)

    b = L.JevBatcher(forward, max_wait_ms=50, max_seqs=8, cuda_stream=False)
    futures = [b.submit([("s", f"r{i}q{k}") for k in range(4)]) for i in range(3)]  # queued before the worker
    b.start()
    res = [f.result(timeout=5) for f in futures]
    b.stop()
    assert seen == [8, 4]  # 2 requests fit 8 sequences, the third waits for the next batch
    assert [r["rows"] for r in res] == [[f"r{i}q{k}" for k in range(4)] for i in range(3)]
    assert res[0]["batch_requests"] == 2 and res[2]["batch_requests"] == 1
    assert all(r["t_done"] >= r["t_start"] >= r["t_arrival"] for r in res)


def test_batcher_splits_on_out_of_memory():
    calls = []

    def forward(items):
        calls.append(len(items))
        if len(items) > 4:
            raise RuntimeError("CUDA out of memory")
        return echo(items)

    b = L.JevBatcher(forward, max_wait_ms=50, max_seqs=16, cuda_stream=False)
    futures = [b.submit([("s", i)] * 4) for i in range(3)]
    b.start()
    assert [f.result(timeout=5)["rows"] for f in futures] == [[i] * 4 for i in range(3)]
    b.stop()
    assert b.ooms >= 1 and b.max_seqs < 16 and sorted(c for c in calls if c <= 4) == [4, 4, 4]


def test_service_curve_and_oom_stop():
    def forward(items):
        if len(items) > 8:
            raise RuntimeError("out of memory")
        time.sleep(0.001 * len(items))
        return echo(items)

    make = L.request_factory({"document": "x"}, ["a", "b", "c", "d"])
    rows = L.service_curve(forward, make, sizes=(1, 2, 4), reps=2)
    assert [r["requests"] for r in rows] == [1, 2, 4] and rows[2]["oom"] and rows[0]["seqs"] == 4
    assert rows[1]["requests_per_s"] > 0 and rows[1]["ms_per_request"] == pytest.approx(rows[1]["ms"]["p50"] / 2)


def test_request_factory_copies_the_state_per_request():
    make = L.request_factory({"document": "x"}, ["a", "b"])
    r1, r2 = make(), make()
    assert r1[0][0] is r1[1][0] and r1[0][0] is not r2[0][0] and r1[0][0] == r2[0][0]


def test_open_loop_measures_response_time():
    def forward(items):
        time.sleep(0.002)
        return echo(items)

    b = L.JevBatcher(forward, max_wait_ms=1, max_seqs=64, cuda_stream=False).start()
    make = L.request_factory({"document": "x"}, list("abcd"))
    r = L.open_loop(b, make, rate=200, duration_s=0.5, warmup_s=0.1)
    b.stop()
    assert r["incomplete"] == 0 and r["counted"] > 20 and r["latency_ms"]["p50"] >= 2.0
    assert r["achieved_per_s"] > 50 and r["forward_ms"]["n"] >= 1


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, d):
        self.t += d


class InstantJev:
    """Answers at once, 50 ms after arrival (no thread)."""

    def __init__(self):
        self.requests = []

    def submit(self, items, t_arrival=None, meta=None):
        self.requests.append(items)
        f: Future = Future()
        f.set_result({"rows": [], "t_arrival": t_arrival, "t_start": t_arrival, "t_done": t_arrival + 0.05,
                      "batch_requests": 1, "batch_seqs": len(items), "forward_ms": 50.0, "meta": {}})
        return f

    def cancel_pending(self):
        return 0


def talk_streams(seed, n):
    """Turns of 0.5–1.5 s (step values 1–6: the fake decoder emits) and pauses of 0.2–1.0 s (zeros), so
    vad-300 finds ends of turns and also continues through short pauses."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        parts = []
        for _ in range(8):
            k = int(rng.integers(4, 12))  # 160 ms blocks
            parts.append(np.repeat(rng.integers(1, 7, size=k).astype(np.float32), STEP))
            parts.append(np.zeros(int(rng.uniform(0.2, 1.0) * SR), np.float32))
        out.append(np.concatenate(parts))
    return out


def vad_probs(audio):
    """Speech where the step value is ≥ 1 (the fake decoder emits there), per 512-sample chunk."""
    n = len(audio) // CHUNK
    return (audio[:n * CHUNK].reshape(n, CHUNK).max(axis=1) >= 1).astype(np.float32)


def single_stream_finals(spec, audio, probs, offset, n_ticks):
    """Stage 1's B1 path on one stream: NemotronTurns over a NemoStream + SeparateASR, same looped audio."""
    looped = np.take(audio, np.arange(offset, offset + n_ticks * STEP), mode="wrap")
    p = np.take(probs, offset // CHUNK + np.arange(n_ticks * STEP // CHUNK), mode="wrap")
    stream = A.NemoStream(spec, A.FeatureWindow(fake_preprocess, 0.2, 0.16, 0.01, 4))
    asr = A.SeparateASR(A.NemotronTurns(stream, 1), L.parse_setting("vad-300"), A.ReplayVAD(p))
    out = []
    for k in range(n_ticks):
        out += [e.text for e in asr.push(looped[k * STEP:(k + 1) * STEP]) if e.kind == "final"]
    return out


def test_live_run_matches_single_streams_and_times_turns():
    """3 streams batched, real-time paced on a fake clock: each stream's finals equal stage 1's single-stream
    B1 path on the same looped audio; every non-empty final is one JEV request; t3 − t1 = t2 − t1 + 50 ms."""
    spec = spec16()
    sources = [(a, vad_probs(a)) for a in talk_streams(5, 2)]
    clock = FakeClock()
    jev = InstantJev()
    batched = A.BatchedNemo(spec, 3, preprocess=fake_preprocess)
    make = L.request_factory({"document": "x"}, list("abcd"))
    run = L.LiveRun(batched, sources, batcher=jev, make_request=make, clock=clock, sleep=clock.sleep)
    run.offsets = [0, STEP * 2, STEP]
    n_ticks = 150
    r = run.run(3, duration_s=n_ticks * STEP / SR, warmup_s=0.0, drain_s=0.0)
    for j, off in enumerate(run.offsets):
        audio, probs = sources[j % 2]
        expect = single_stream_finals(spec, audio, probs, off, n_ticks)
        assert [t for s, _, t in run.finals if s == j] == expect
    texts = [t for _, _, t in run.finals]
    assert any(texts) and r["turns"] == len(texts)
    assert r["jev_calls"] == sum(1 for t in texts if t.strip()) == len(jev.requests)
    assert r["empty_finals"] == len(texts) - r["jev_calls"]
    assert all(len(items) == 4 for items in jev.requests)
    assert r["lag_ms"]["p95"] == pytest.approx(0.0, abs=1e-6) and r["asr_keeps_up"]
    # the final waits one step of lookahead after the endpoint (fake clock: no compute time)
    assert r["t2_minus_t1_ms"]["p50"] >= 0 and r["t3_minus_t1_ms"]["p50"] == pytest.approx(r["t2_minus_t1_ms"]["p50"] + 50)
    assert clock.t == pytest.approx(n_ticks * STEP / SR)


def test_live_run_reports_lag_when_steps_are_slow():
    spec = spec16()
    sources = [(a, vad_probs(a)) for a in talk_streams(6, 1)]
    clock = FakeClock()
    batched = A.BatchedNemo(spec, 2, preprocess=fake_preprocess)
    step = batched.step

    def slow(chunks):
        clock.t += 0.25  # 250 ms of compute per 160 ms step: falls behind
        return step(chunks)

    batched.step = slow
    run = L.LiveRun(batched, sources, clock=clock, sleep=clock.sleep)
    r = run.run(2, duration_s=40 * STEP / SR, warmup_s=0.0, drain_s=0.0)
    assert not r["asr_keeps_up"] and r["final_lag_ms"] > 1000 and r["step_ms"]["p50"] == pytest.approx(250)


def test_report_markdown_and_capacity_at():
    lat = lambda p95: {"p50": p95 / 2, "p95": p95, "p99": p95}  # noqa: E731
    rows = [{"rate": 10, "latency_ms": lat(40), "incomplete": 0},
            {"rate": 20, "latency_ms": lat(150), "incomplete": 0},
            {"rate": 40, "latency_ms": lat(90), "incomplete": 3}]
    for r in rows:
        r |= {"achieved_per_s": r["rate"], "batch_requests": {"mean": 1.0}}
    assert L.capacity_at(rows, 100) == 10 and L.capacity_at(rows, 300) == 20 and L.capacity_at(rows, 10) is None
    curve = [{"requests": 1, "seqs": 4, "ms": {"p50": 25.0}, "requests_per_s": 40, "ms_per_request": 25.0}]
    live = {"length": 256, "streams": 16, "setting": "vad-300", "step_ms": {"p50": 40, "p95": 50},
            "bookkeeping_ms": {"p50": 5}, "lag_ms": {"p95": 0}, "jev_calls_per_s": 2.0, "jev_calls_per_stream_min": 7.5,
            "t2_minus_t1_ms": {"p50": 170, "p95": 300}, "t3_minus_t1_ms": {"p50": 200, "p95": 350}, "asr_keeps_up": True}
    md = L.report_markdown([{"gpu": "X", "jev_variant": "bf16-fa2", "jev": [{"length": 256, "curve": curve, "load": rows}],
                             "live": [live]}])
    assert "| 256 | 25.0 | 1 | 25.0 | 40 | 10 | 20 |" in md
    assert "| 256 | 16 | 40 / 50 | 5 | 0 | 2.0 (7.5) | 170 / 300 | 200 / 350 | 624 / 1099 | yes |" in md
