"""Stage 2, capacity of one GPU (todo Phase 9b): JEV under load, and the live pipeline with N streams.

Two measurements on the same GPU (``scripts/modal_voice_capacity.py`` runs them on Modal):

1. **JEV alone.** A request = one JEV call with the 4 criteria (4 questions on one input of ``L`` tokens:
   4 sequences in one forward, the input tokenised once). ``JevBatcher`` is the server side: requests that
   arrive within ``max_wait_ms`` go into one forward (micro-batching, up to ``max_seqs`` sequences).
   ``service_curve``: time of one forward for 1, 2, 4, … requests at once (the GPU cost per request and the
   throughput ceiling); ``open_loop``: requests arriving at random (Poisson) at a fixed rate for some
   seconds → the response time under load (arrival → result: queue + batch + forward). The highest rate
   whose p95 stays under the target is the JEV capacity of the GPU at that length.
2. **The live pipeline** (``LiveRun``): N real-time audio streams (the stage-1 streams, looped, each from
   a random offset) → Nemotron (``BatchedNemo``, one step of 160 ms for all streams at once, paced by the
   wall clock) → per stream the B1 logic of stage 1 (``SeparateASR`` + ``NemotronTurns`` over the batched
   tokens, VAD from the recorded Silero probabilities, ``vad-300``) → at each turn's final text a JEV
   request through the same ``JevBatcher`` (the ASR loop and the JEV worker share the GPU on separate CUDA
   streams). Per turn: t2 − t1 (final text after the endpoint) and t3 − t1 (JEV result after the
   endpoint), in wall time; the endpoint's own wait after the end of speech (t1 − t0) is stage 1's
   (vad-300: 424 ms p50, 749 ms p95), so t3 − t0 = stage 1's t1 − t0 + this t3 − t1. Per step: the GPU step,
   the per-stream bookkeeping on the CPU, and the lag behind real time (a step that starts later than its
   audio is complete). The GPU keeps up with N streams while the lag stays under one step.

The JEV inputs are synthetic (``jevlite.latency.fit_request``: Bool claims on a filler document of exactly
``L`` tokens), not our questions or criteria: the cost depends on the number of tokens, not on the words, and
no confidential text leaves the laptop. Input lengths from our data (2026-11-21, ModernBERT tokens): question
p95 ~150–270, question + answer p95 ~700–900, + retrieved documents ~3.3k p50 / 4.3k p95 / 6.9k max → 256,
1024, 4096, 8192.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .asr import BatchedNemo, NemotronTurns, ReplayVAD, SeparateASR
from .endpoint import CHUNK, SR, parse_setting

N_CRITERIA = 4
LENGTHS = (256, 1024, 4096, 8192)


def percentiles(values: Sequence[float]) -> dict:
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return {"n": 0}
    return {**{f"p{q}": float(np.percentile(v, q)) for q in (50, 90, 95, 99)},
            "mean": float(v.mean()), "max": float(v.max()), "n": int(v.size)}


def is_oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


# ------------------------------------------------------------------------------------------ JEV server


@dataclass
class _Job:
    items: list
    future: Future
    t_arrival: float
    meta: dict = field(default_factory=dict)


class JevBatcher:
    """Micro-batching JEV server in a worker thread. ``forward(items) → one row per item`` (e.g.
    ``Predictor._probs(items)[0]``; an item = (state, question)). A batch starts with the first waiting
    request and takes every request that arrives within ``max_wait_ms`` after it, up to ``max_seqs``
    sequences; on out-of-memory the batch is split in two and ``max_seqs`` halved. The result of a request:
    its rows, the times (arrival, start of its forward, done; ``clock`` seconds), the batch size and the
    forward's ms. On a GPU the forwards run on their own CUDA stream (``cuda_stream``), so they can overlap
    the ASR's work on the default stream."""

    def __init__(self, forward: Callable[[list], list], max_wait_ms: float = 5.0, max_seqs: int = 64,
                 cuda_stream: bool = True, clock: Callable[[], float] = time.perf_counter):
        self.forward, self.max_wait, self.max_seqs = forward, max_wait_ms / 1000, max_seqs
        self.clock = clock
        self.q: queue.Queue[_Job | None] = queue.Queue()
        self.held: _Job | None = None
        self.thread: threading.Thread | None = None
        self.stream = None
        self.batches: list[tuple[int, int, float]] = []  # (requests, sequences, forward ms)
        self.ooms = 0
        if cuda_stream:
            try:
                import torch

                if torch.cuda.is_available():
                    self.stream = torch.cuda.Stream()
            except ImportError:
                pass

    def start(self) -> "JevBatcher":
        self.thread = threading.Thread(target=self._run, name="jev-batcher", daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.q.put(None)
        if self.thread:
            self.thread.join(timeout=60)

    def submit(self, items: list, t_arrival: float | None = None, meta: dict | None = None) -> Future:
        f: Future = Future()
        self.q.put(_Job(items, f, self.clock() if t_arrival is None else t_arrival, meta or {}))
        return f

    def cancel_pending(self) -> int:
        """Drop the requests still waiting (after an overloaded rate, so the next one starts clean)."""
        n = 0
        while True:
            try:
                job = self.q.get_nowait()
            except queue.Empty:
                break
            if job is None:
                self.q.put(None)
                break
            job.future.cancel()
            n += 1
        return n

    def _take(self) -> list[_Job] | None:
        job = self.held or self.q.get()
        self.held = None
        if job is None:
            return None
        jobs, seqs = [job], len(job.items)
        deadline = self.clock() + self.max_wait
        while seqs < self.max_seqs:
            left = deadline - self.clock()
            try:
                nxt = self.q.get(timeout=max(left, 0)) if left > 0 else self.q.get_nowait()
            except queue.Empty:
                break
            if nxt is None:
                self.q.put(None)
                break
            if seqs + len(nxt.items) > self.max_seqs:
                self.held = nxt
                break
            jobs.append(nxt)
            seqs += len(nxt.items)
        return jobs

    def _forward(self, jobs: list[_Job]) -> None:
        items = [it for j in jobs for it in j.items]
        t = self.clock()
        try:
            if self.stream is not None:
                import torch

                with torch.cuda.stream(self.stream):
                    rows = self.forward(items)
                self.stream.synchronize()
            else:
                rows = self.forward(items)
        except Exception as e:  # noqa: BLE001
            if is_oom(e) and len(jobs) > 1:
                self.ooms += 1
                self.max_seqs = max(len(jobs[0].items), self.max_seqs // 2)
                try:
                    import torch

                    torch.cuda.empty_cache()
                except ImportError:
                    pass
                half = len(jobs) // 2
                self._forward(jobs[:half])
                self._forward(jobs[half:])
                return
            for j in jobs:
                j.future.set_exception(e)
            return
        done = self.clock()
        ms = (done - t) * 1000
        self.batches.append((len(jobs), len(items), ms))
        i = 0
        for j in jobs:
            j.future.set_result({"rows": rows[i:i + len(j.items)], "t_arrival": j.t_arrival, "t_start": t,
                                 "t_done": done, "batch_requests": len(jobs), "batch_seqs": len(items),
                                 "forward_ms": ms, "meta": j.meta})
            i += len(j.items)

    def _run(self) -> None:
        while True:
            jobs = self._take()
            if jobs is None:
                return
            jobs = [j for j in jobs if not j.future.cancelled()]
            if jobs:
                self._forward(jobs)


def service_curve(forward: Callable[[list], list], make_request: Callable[[], list], sizes: Sequence[int],
                  reps: int = 5, warmup: int = 1, clock: Callable[[], float] = time.perf_counter,
                  sync: Callable[[], None] = lambda: None, log: Callable[[str], None] | None = None) -> list[dict]:
    """One forward over ``b`` requests at once, for each ``b`` in ``sizes``: ms (p50 …), requests/s and the
    GPU ms per request. Stops at the first out-of-memory (recorded)."""
    out = []
    for b in sizes:
        items = [it for _ in range(b) for it in make_request()]
        try:
            for _ in range(warmup):
                forward(items)
            times = []
            for _ in range(reps):
                sync()
                t = clock()
                forward(items)
                sync()
                times.append((clock() - t) * 1000)
        except Exception as e:  # noqa: BLE001
            if not is_oom(e):
                raise
            out.append({"requests": b, "seqs": len(items), "oom": True})
            if log:
                log(f"  service b={b}: out of memory")
            break
        s = percentiles(times)
        row = {"requests": b, "seqs": len(items), "ms": s, "requests_per_s": b / (s["p50"] / 1000),
               "ms_per_request": s["p50"] / b}
        out.append(row)
        if log:
            log(f"  service b={b}: {s['p50']:.1f} ms p50, {row['requests_per_s']:.0f} req/s")
    return out


def open_loop(batcher: JevBatcher, make_request: Callable[[], list], rate: float, duration_s: float,
              warmup_s: float = 2.0, drain_s: float = 10.0, seed: int = 0,
              clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep) -> dict:
    """Requests arriving at random (Poisson, ``rate`` per s) for ``duration_s``; the response time from the
    scheduled arrival to the result (so a late generator counts against the server, not for it). Requests
    after ``warmup_s`` are counted; requests not done ``drain_s`` after the end are cut (``incomplete``) and
    the queue is cleared."""
    rng = np.random.default_rng(seed)
    arrivals, t = [], 0.0
    while True:
        t += float(rng.exponential(1 / rate))
        if t >= duration_s:
            break
        arrivals.append(t)
    batches_before = len(batcher.batches)
    t0 = clock()
    sent = []
    for a in arrivals:
        wait = t0 + a - clock()
        if wait > 0:
            sleep(wait)
        sent.append((a, batcher.submit(make_request(), t_arrival=t0 + a)))
    end = clock() + drain_s
    lat, wait_ms, sizes, incomplete = [], [], [], 0
    for a, f in sent:
        try:
            r = f.result(timeout=max(end - clock(), 0.0))
        except Exception:  # noqa: BLE001 — timeout or cancelled
            incomplete += 1
            continue
        if a >= warmup_s:
            lat.append((r["t_done"] - r["t_arrival"]) * 1000)
            wait_ms.append((r["t_start"] - r["t_arrival"]) * 1000)
            sizes.append(r["batch_requests"])
    if incomplete:
        batcher.cancel_pending()
    fwd = [ms for _, _, ms in batcher.batches[batches_before:]]
    return {"rate": rate, "offered": len(sent), "counted": len(lat), "incomplete": incomplete,
            "achieved_per_s": len(lat) / max(duration_s - warmup_s, 1e-9), "latency_ms": percentiles(lat),
            "queue_ms": percentiles(wait_ms), "batch_requests": percentiles(sizes), "forward_ms": percentiles(fwd)}


def request_factory(template_state: dict, questions: Sequence) -> Callable[[], list]:
    """Items of one request: the questions on a fresh copy of the state (shared by the request's questions,
    so it is tokenised once per request, as ``Predictor.predict`` does; distinct between requests, so a
    batch does not get a tokenisation it would not get in production)."""
    def make() -> list:
        state = dict(template_state)
        return [(state, q) for q in questions]
    return make


# -------------------------------------------------------------------------------------- live pipeline


class PushedSteps:
    """A ``NemotronTurns`` stepper fed from outside: the pieces of one stream from a batched step are
    ``put`` before the stream's audio is pushed, and ``step`` hands them over."""

    def __init__(self, chunk_samples: int):
        self.chunk_samples = chunk_samples
        self.pending: deque[tuple[list[str], float]] = deque()
        self.last_ms = 0.0

    def reset(self) -> None:
        self.pending.clear()
        self.last_ms = 0.0

    def put(self, pieces: list[str], ms: float) -> None:
        self.pending.append((pieces, ms))

    def step(self, chunk: np.ndarray) -> list[str]:
        pieces, self.last_ms = self.pending.popleft() if self.pending else ([], 0.0)
        return list(pieces)


@dataclass
class _Stream:
    source: int
    offset: int  # sample of the source where this stream starts (a multiple of the step)
    asr: SeparateASR
    stepper: PushedSteps
    endpoints: deque = field(default_factory=deque)  # wall times of endpoints waiting for their final


class LiveRun:
    """N real-time streams through one ``BatchedNemo`` and, at each turn's final, one JEV request (module
    doc). ``sources``: (audio, Silero P(speech) per 512 samples) of the stage-1 streams; stream j plays source
    j mod len(sources), looped, from a random step-aligned offset. ``batcher`` / ``make_request``: the JEV
    side (None: ASR only). The first ``warmup_s`` of endpoints are not counted."""

    def __init__(self, batched: BatchedNemo, sources: Sequence[tuple[np.ndarray, np.ndarray]],
                 setting: str = "vad-300", batcher: JevBatcher | None = None,
                 make_request: Callable[[], list] | None = None, lag_steps: int = 1, seed: int = 0,
                 clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep):
        self.batched, self.sources, self.setting = batched, list(sources), setting
        self.batcher, self.make_request = batcher, make_request
        self.n = batched.chunk_samples
        self.step_s = self.n / SR
        if self.n % CHUNK:
            raise ValueError(f"step of {self.n} samples is not a multiple of the VAD chunk {CHUNK}")
        self.lag_steps, self.rng, self.clock, self.sleep = lag_steps, np.random.default_rng(seed), clock, sleep
        self.offsets: list[int] | None = None  # fixed start offsets (tests); else random
        self.finals: list[tuple[int, float, str]] = []  # (stream, endpoint audio s, text) of the last run

    def _streams(self, n_streams: int, n_ticks: int) -> list[_Stream]:
        out = []
        per = self.n // CHUNK
        for j in range(n_streams):
            s = j % len(self.sources)
            audio, probs = self.sources[s]
            offset = (self.offsets[j] if self.offsets is not None
                      else int(self.rng.integers(0, max(len(audio) // self.n, 1))) * self.n)
            idx = (offset // CHUNK + np.arange(n_ticks * per)) % max(len(probs), 1)
            looped = np.asarray(probs, np.float32)[idx] if len(probs) else np.zeros(n_ticks * per, np.float32)
            stepper = PushedSteps(self.n)
            # max_turn_s=1: the turn audio SeparateASR keeps is for Whisper only (NemotronTurns ignores it);
            # 120 s of it per stream would be ~8 MB × N of RAM
            asr = SeparateASR(NemotronTurns(stepper, self.lag_steps), parse_setting(self.setting), ReplayVAD(looped),
                              max_turn_s=1.0)
            out.append(_Stream(s, offset, asr, stepper))
        return out

    def _chunks(self, streams: list[_Stream], k: int) -> np.ndarray:
        out = np.empty((len(streams), self.n), np.float32)
        for j, st in enumerate(streams):
            audio = self.sources[st.source][0]
            a = (st.offset + k * self.n) % len(audio)
            if a + self.n <= len(audio):
                out[j] = audio[a:a + self.n]
            else:
                out[j] = np.take(audio, np.arange(a, a + self.n), mode="wrap")
        return out

    def run(self, n_streams: int, duration_s: float, warmup_s: float = 5.0, drain_s: float = 15.0,
            log: Callable[[str], None] | None = None) -> dict:
        if self.batched.batch != n_streams:
            raise ValueError(f"BatchedNemo has {self.batched.batch} slots, run asks for {n_streams} streams")
        n_ticks = int(round(duration_s / self.step_s))
        streams = self._streams(n_streams, n_ticks)
        zeros = np.zeros(self.n, np.float32)
        self.batched.reset()
        step_ms, book_ms, lag_ms = [], [], []
        t2, jobs, empty, endpoints = [], [], 0, 0
        self.finals = []
        t0 = self.clock()
        for k in range(n_ticks):
            due = t0 + (k + 1) * self.step_s  # this step's audio is complete in real time
            wait = due - self.clock()
            if wait > 0:
                self.sleep(wait)
            start = self.clock()
            lag_ms.append((start - due) * 1000)
            pieces = self.batched.step(self._chunks(streams, k))
            gpu_done = self.clock()
            step_ms.append((gpu_done - start) * 1000)
            for j, st in enumerate(streams):
                st.stepper.put(pieces[j], 0.0)
                for e in st.asr.push(zeros):
                    if e.kind == "endpoint":
                        st.endpoints.append(t0 + e.audio_s)
                    elif e.kind == "final":
                        t_end = st.endpoints.popleft() if st.endpoints else t0 + e.audio_s
                        self.finals.append((j, t_end - t0, e.text))
                        if t_end - t0 < warmup_s:
                            continue
                        now = self.clock()
                        endpoints += 1
                        t2.append((now - t_end) * 1000)
                        if not e.text.strip():
                            empty += 1  # no words: the pipeline does not call JEV
                        elif self.batcher is not None and self.make_request is not None:
                            jobs.append((t_end, self.batcher.submit(self.make_request(), t_arrival=now)))
            book_ms.append((self.clock() - gpu_done) * 1000)
            if log and (k + 1) % max(int(10 / self.step_s), 1) == 0:
                log(f"  N={n_streams} t={(k + 1) * self.step_s:.0f}s: step {np.median(step_ms[-60:]):.0f} ms, "
                    f"lag {lag_ms[-1]:.0f} ms, finals {len(t2)}, JEV {len(jobs)}")
        end = self.clock() + drain_s
        t3, queue_ms, fwd_ms, sizes, incomplete = [], [], [], [], 0
        for t_end, f in jobs:
            try:
                r = f.result(timeout=max(end - self.clock(), 0.0))
            except Exception:  # noqa: BLE001
                incomplete += 1
                continue
            t3.append((r["t_done"] - t_end) * 1000)
            queue_ms.append((r["t_start"] - r["t_arrival"]) * 1000)
            fwd_ms.append(r["forward_ms"])
            sizes.append(r["batch_requests"])
        if incomplete and self.batcher is not None:
            self.batcher.cancel_pending()
        counted_s = max(duration_s - warmup_s, 1e-9)
        lag = percentiles(lag_ms[int(warmup_s / self.step_s):] or lag_ms)
        return {"streams": n_streams, "setting": self.setting, "duration_s": duration_s, "ticks": n_ticks,
                "step_s": self.step_s, "step_ms": percentiles(step_ms), "bookkeeping_ms": percentiles(book_ms),
                "lag_ms": lag, "final_lag_ms": lag_ms[-1] if lag_ms else 0.0,
                "turns": endpoints, "empty_finals": empty, "jev_calls": len(jobs), "jev_incomplete": incomplete,
                "jev_calls_per_s": len(jobs) / counted_s,
                "jev_calls_per_stream_min": len(jobs) / counted_s * 60 / n_streams,
                "t2_minus_t1_ms": percentiles(t2), "t3_minus_t1_ms": percentiles(t3),
                "jev_queue_ms": percentiles(queue_ms), "jev_forward_ms": percentiles(fwd_ms),
                "jev_batch_requests": percentiles(sizes),
                "asr_keeps_up": bool(lag.get("p95", 0.0) <= self.step_s * 1000 and incomplete == 0)}


# ---------------------------------------------------------------------------------------------- report

STAGE1_T1 = {"vad-300": (424, 749), "vad-500": (612, 993)}  # AMI t1 − t0 p50 / p95, stage 1 (2026-11-20)


def capacity_at(rows: Sequence[dict], target_ms: float, key: str = "rate") -> float | None:
    """The highest ``key`` among rows whose p95 response time is within ``target_ms`` (and none cut)."""
    ok = [r[key] for r in rows if r.get("latency_ms", {}).get("p95", float("inf")) <= target_ms
          and not r.get("incomplete")]
    return max(ok) if ok else None


def _f(x, fmt: str = ".0f") -> str:
    return "–" if x is None else format(x, fmt)


def report_markdown(results: Sequence[dict]) -> str:
    """Per GPU: JEV alone (one call with the 4 criteria, by input length) and the live pipeline (N streams)."""
    lines = ["# Voice stage 2: capacity of one GPU", "",
             "JEV request = one call with the 4 criteria (4 sequences of L tokens in one forward; synthetic "
             "inputs of exactly L tokens). Response time = arrival → result (queue + micro-batch + forward). "
             "Live: Nemotron R=1 batched over N real-time streams (160 ms steps) + vad-300 + a JEV request at each "
             "final; t1 = endpoint, t2 = final text, t3 = JEV result (wall time after the endpoint). t3 − t0 ≈ "
             "stage 1's t1 − t0 (vad-300: 424 ms p50, 749 ms p95) + t3 − t1.", ""]
    for res in results:
        lines += [f"## {res.get('gpu', '?')} (JEV {res.get('jev_variant', '?')})", ""]
        if res.get("asr_check"):
            c = res["asr_check"]
            lines += [f"ASR check (smoke stream, one stream): WER {c.get('wer', float('nan')):.1f}% "
                      f"({'ok' if c.get('ok') else 'FAILED'}).", ""]
        jev = res.get("jev", [])
        if jev:
            lines += ["### JEV alone", "",
                      "| L tokens | 1 call ms (p50) | best batch | GPU ms / call | max calls/s | calls/s at p95 ≤ 100 ms | ≤ 300 ms |",
                      "| --: | --: | --: | --: | --: | --: | --: |"]
            for j in jev:
                curve = [c for c in j["curve"] if not c.get("oom")]
                one = curve[0]["ms"]["p50"] if curve else None
                best = max(curve, key=lambda c: c["requests_per_s"]) if curve else None
                lines.append(f"| {j['length']} | {_f(one, '.1f')} | {best['requests'] if best else '–'} | "
                             f"{_f(best and best['ms_per_request'], '.1f')} | {_f(best and best['requests_per_s'])} | "
                             f"{_f(capacity_at(j['load'], 100))} | {_f(capacity_at(j['load'], 300))} |")
            lines += ["", "Under load (offered rate → response time):", "",
                      "| L | calls/s offered | done/s | p50 ms | p95 ms | p99 ms | batch (mean calls) | cut |",
                      "| --: | --: | --: | --: | --: | --: | --: | --: |"]
            for j in jev:
                for r in j["load"]:
                    lat = r["latency_ms"]
                    lines.append(f"| {j['length']} | {r['rate']:.0f} | {r['achieved_per_s']:.0f} | {_f(lat.get('p50'))} | "
                                 f"{_f(lat.get('p95'))} | {_f(lat.get('p99'))} | "
                                 f"{_f(r['batch_requests'].get('mean'), '.1f')} | {r['incomplete']} |")
            lines.append("")
        live = res.get("live", [])
        if live:
            lines += ["### Live pipeline", "",
                      "| L | N streams | ASR step ms p50 / p95 | CPU per step ms | lag p95 ms | JEV calls/s (per stream/min) | "
                      "t2 − t1 p50 / p95 | t3 − t1 p50 / p95 | t3 − t0 ≈ p50 / p95 | keeps up |",
                      "| --: | --: | --: | --: | --: | --: | --: | --: | --: | :-: |"]
            for r in live:
                t1 = STAGE1_T1.get(r["setting"], (0, 0))
                t3 = r["t3_minus_t1_ms"]
                lines.append(
                    f"| {r['length']} | {r['streams']} | {_f(r['step_ms'].get('p50'))} / {_f(r['step_ms'].get('p95'))} | "
                    f"{_f(r['bookkeeping_ms'].get('p50'))} | {_f(r['lag_ms'].get('p95'))} | "
                    f"{r['jev_calls_per_s']:.1f} ({r['jev_calls_per_stream_min']:.1f}) | "
                    f"{_f(r['t2_minus_t1_ms'].get('p50'))} / {_f(r['t2_minus_t1_ms'].get('p95'))} | "
                    f"{_f(t3.get('p50'))} / {_f(t3.get('p95'))} | "
                    f"{_f(t3.get('p50') and t3['p50'] + t1[0])} / {_f(t3.get('p95') and t3['p95'] + t1[1])} | "
                    f"{'yes' if r['asr_keeps_up'] else 'no'} |")
            lines.append("")
        if res.get("errors"):
            lines += ["Errors:", "", *[f"- {e}" for e in res["errors"]], ""]
    return "\n".join(lines) + "\n"
