"""Voice stage 2 on Modal: capacity of one GPU for JEV and for the live pipeline (todo Phase 9b; the library is
``jevlite.voice.load``). Same launch pattern as stage 1: deploy once, start with ``spawn`` (returns at once,
runs without the laptop), results in the volume ``jevlite-voice`` under ``capacity/``.

    # once, and again after every code change (a deployed app with no calls runs no containers: free):
    uv run --extra bench modal deploy scripts/modal_voice_capacity.py

    uv run --extra bench python scripts/modal_voice_capacity.py check          # CPU, ~3 min, cents
    uv run --extra bench python scripts/modal_voice_capacity.py run --gpu L40S  # ~16 min ≈ $0.55
    uv run --extra bench python scripts/modal_voice_capacity.py run --gpu H100! # ~16 min ≈ $1.05

    uv run --extra bench modal app logs jevlite-voice-capacity
    uv run --extra bench modal app stop jevlite-voice-capacity
    uv run --extra bench modal volume get jevlite-voice capacity data/speed/voice

``check`` (CPU): the image imports NeMo next to jev-lite, both models load (and are cached in the volume:
the GPU run pays no download), one batched Nemotron step and one JEV call with 4 criteria run on the CPU.
``run`` (one GPU): (0) ASR check — the smoke stream through one Nemotron stream, WER must stay ≈ stage 1's
1.8% (else the live part is skipped); (1) JEV alone at 256 / 1024 / 4096 / 8192 tokens: the service curve
(1, 2, 4, … calls in one forward) and 10 s of random arrivals at 30 / 60 / 80 / 95% of the best throughput;
(2) the live pipeline at 256 and 4096 tokens: N = 16, 32, 64, … real-time streams (40 s each, 8 s
warm-up), doubling until the ASR falls behind or t3 − t1 p95 passes 1 s, then one N half-way.
``H100!`` pins an H100 (plain ``H100`` may become an H200). Prices for the estimates: H100 ~$3.95/h,
L40S ~$1.95/h (Modal list prices, approximate).

Image: this repo's ``uv.lock`` (torch 2.14.1 + CUDA 13.0, transformers 5.18, kernels 0.17 — the setup of
the 2026-11-09 speed run, whose FlashAttention kernels are known to work) + NeMo 3.0.0 [asr] installed with
those versions pinned. JEV = the public ``Nefelim584/jev-lite@v0.3.0``, bf16 with FlashAttention 3 on
Hopper, 2 elsewhere (bf16 SDPA if the kernel fails to load).
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

APP = "jevlite-voice-capacity"
ROOT = Path("/vol")
STREAMS, OUT = ROOT / "streams", ROOT / "capacity"
SMOKE_ID = "librispeech-smoke"
JEV_REPO, JEV_REVISION = "Nefelim584/jev-lite", "v0.3.0"
JEV_LENGTHS = (256, 1024, 4096, 8192)
LIVE_LENGTHS = (256, 4096)
LIVE_STREAMS = (16, 32, 64, 128, 256, 512)
TOKEN_BUDGET = 1 << 19  # tokens in one JEV forward (calls × 4 × L), the service curve's and the batcher's cap
LOAD_FRACTIONS = (0.3, 0.6, 0.8, 0.95)
SMOKE_WER_MAX = 4.0  # stage 1: 1.8% one stream at 160 ms

app = modal.App(APP)
vol = modal.Volume.from_name("jevlite-voice", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libsndfile1")
    .uv_sync(".", extras=["bench", "voice"], frozen=True)
    .run_commands(
        # NeMo on top of uv_sync's venv (/.uv/.venv, no pip in it; uv at /.uv/uv), with the locked torch /
        # transformers / hub / kernels / numpy kept as they are
        "PY=/.uv/.venv/bin/python && /.uv/uv pip freeze --python $PY | grep -iE "
        "'^(torch|transformers|huggingface[-_]hub|kernels|numpy|tokenizers|safetensors)==' > /tmp/pins.txt "
        "&& cat /tmp/pins.txt && /.uv/uv pip install --python $PY -c /tmp/pins.txt "
        "--extra-index-url https://download.pytorch.org/whl/cu130 --index-strategy unsafe-best-match "
        "'nemo_toolkit[asr]==3.0.0'",
    )
    .env({"HF_HOME": str(ROOT / "hf"), "TOKENIZERS_PARALLELISM": "false", "NEMO_CACHE_DIR": str(ROOT / "nemo")})
    .add_local_python_source("jevlite")
)


def _log(s: str) -> None:
    print(s, flush=True)


def _versions() -> dict:
    import importlib.metadata as md

    out = {}
    for p in ("torch", "transformers", "kernels", "nemo_toolkit", "numpy", "huggingface-hub"):
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    return out


def _jev_final() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(JEV_REPO, revision=JEV_REVISION))


def _load_jev(device: str, log=_log):
    """The release predictor in the best variant for this device; falls back to bf16 SDPA (or fp32 on CPU)."""
    import torch

    from jevlite.latency import fit_request, load_variant, unsupported_reason, variants_by_name

    final = _jev_final()
    if device == "cpu":
        names = ["fp32-sdpa"]
    else:
        cap = torch.cuda.get_device_capability(0)
        names = [n for n in ("bf16-fa3", "bf16-fa2") if unsupported_reason(variants_by_name([n])[0], cap) is None]
        names.append("bf16-sdpa")
    errors = []
    for name in names:
        try:
            pred = load_variant(final, variants_by_name([name])[0], device)
            req, _ = fit_request(pred.encoder, 256, n_claims=4)
            pred.predict(req)  # the kernel really runs
            return pred, name, errors
        except Exception as e:  # noqa: BLE001
            errors.append(f"JEV {name}: {type(e).__name__}: {e}"[:500])
            log(errors[-1])
    raise RuntimeError(f"no JEV variant loads: {errors}")


def _sources():
    """(audio, Silero P(speech)) of the stage-1 streams (AMI headsets + LibriSpeech), the smoke stream apart."""
    from jevlite.voice import quality

    index = json.loads((STREAMS / "index.json").read_text())
    out = []
    for s in index:
        if s["id"] != SMOKE_ID:
            audio, _, probs = quality.load_stream(STREAMS, s["id"])
            out.append((audio, probs))
    return out


@app.function(image=image, cpu=4.0, memory=16384, timeout=1800, volumes={str(ROOT): vol})
def check() -> dict:
    """CPU: imports, both models load (cached into the volume), one batched ASR step, one JEV call."""
    import time

    import numpy as np

    from jevlite.latency import fit_request
    from jevlite.voice import load as L
    from jevlite.voice import quality
    from jevlite.voice.asr import NEMOTRON, BatchedNemo, load_nemo

    vol.reload()
    out = {"versions": _versions()}
    _log(json.dumps(out))
    t = time.perf_counter()
    spec = load_nemo(NEMOTRON, 1, "cpu", cuda_graphs=False)
    b = BatchedNemo(spec, 2)
    audio, meta, _ = quality.load_stream(STREAMS, SMOKE_ID)
    n = b.chunk_samples
    pieces = []
    for k in range(25):  # 4 s of two streams
        pieces.append(b.step(np.stack([audio[k * n:(k + 1) * n], audio[(k + 3) * n:(k + 4) * n]])))
    out["nemotron"] = {"step_s": spec.step_s, "lookahead_steps": spec.lookahead_steps, "load_and_25_steps_s":
                       time.perf_counter() - t, "text_slot0": "".join(p for ps in pieces for p in ps[0])[:200]}
    _log(json.dumps(out["nemotron"]))
    t = time.perf_counter()
    pred, variant, errors = _load_jev("cpu")
    req, length = fit_request(pred.encoder, 256, n_claims=4)
    make = L.request_factory(req.state, req.questions)
    rows = pred._probs(make() + make())[0]
    out["jev"] = {"variant": variant, "length": length, "rows": len(rows), "s": time.perf_counter() - t,
                  "errors": errors}
    _log(json.dumps(out["jev"]))
    vol.commit()  # JEV model + NeMo caches into the volume
    out["sources"] = len(json.loads((STREAMS / "index.json").read_text())) - 1
    out["ok"] = bool(out["nemotron"]["text_slot0"]) and len(rows) == 8
    _log(f"check ok: {out['ok']}")
    return out


def _asr_check(spec) -> dict:
    from jevlite.voice import quality
    from jevlite.voice.asr import EOB, EOU, NemoStream, pieces_text, record_steps

    audio, meta, _ = quality.load_stream(STREAMS, SMOKE_ID)
    steps = record_steps(NemoStream(spec), audio, flush_steps=spec.lookahead_steps)
    text = pieces_text([p for ps, _ in steps for p in ps if p not in (EOU, EOB)])
    e, n = quality.word_errors(meta["reference"], text)
    wer = 100 * e / max(n, 1)
    return {"wer": wer, "ok": wer <= SMOKE_WER_MAX}


def _jev_alone(pred, length: int, log=_log) -> dict:
    import torch

    from jevlite.latency import fit_request
    from jevlite.voice import load as L

    req, got = fit_request(pred.encoder, length, n_claims=L.N_CRITERIA)
    make = L.request_factory(req.state, req.questions)
    forward = lambda items: pred._probs(items)[0]  # noqa: E731
    max_calls = max(1, TOKEN_BUDGET // (L.N_CRITERIA * length))
    sizes = [b for b in (1, 2, 4, 8, 16, 32, 64, 128) if b <= max_calls]
    log(f"JEV L={length} ({got} tokens): service curve {sizes}")
    curve = L.service_curve(forward, make, sizes, reps=5, warmup=2, sync=torch.cuda.synchronize, log=log)
    ok = [c for c in curve if not c.get("oom")]
    best = max(c["requests_per_s"] for c in ok)
    batcher = L.JevBatcher(forward, max_wait_ms=5.0, max_seqs=max(L.N_CRITERIA, TOKEN_BUDGET // length)).start()
    load = []
    try:
        for frac in LOAD_FRACTIONS:
            r = L.open_loop(batcher, make, rate=frac * best, duration_s=10.0, warmup_s=2.0, drain_s=10.0)
            r["fraction"] = frac
            load.append(r)
            log(f"  load {r['rate']:.0f}/s: p50 {r['latency_ms'].get('p50', 0):.0f} ms, p95 "
                f"{r['latency_ms'].get('p95', 0):.0f} ms, batch {r['batch_requests'].get('mean', 0):.1f}, "
                f"cut {r['incomplete']}")
            if r["incomplete"] or r["latency_ms"].get("p95", 0) > 1000:
                break
    finally:
        batcher.stop()
    return {"length": length, "tokens": got, "curve": curve, "load": load, "best_requests_per_s": best,
            "ooms": batcher.ooms}


def _live(spec, pred, length: int, sources, log=_log) -> list[dict]:
    import gc

    import torch

    from jevlite.latency import fit_request
    from jevlite.voice import load as L
    from jevlite.voice.asr import BatchedNemo

    req, _ = fit_request(pred.encoder, length, n_claims=L.N_CRITERIA)
    make = L.request_factory(req.state, req.questions)
    forward = lambda items: pred._probs(items)[0]  # noqa: E731
    rows = []

    def one(n: int) -> dict:
        batcher = L.JevBatcher(forward, max_wait_ms=5.0, max_seqs=max(L.N_CRITERIA, TOKEN_BUDGET // length)).start()
        try:
            run = L.LiveRun(BatchedNemo(spec, n), sources, "vad-300", batcher, make, lag_steps=spec.lookahead_steps,
                            seed=n)
            r = run.run(n, duration_s=40.0, warmup_s=8.0, drain_s=15.0, log=log)
        finally:
            batcher.stop()
            gc.collect()
            torch.cuda.empty_cache()
        r["length"] = length
        t3 = r["t3_minus_t1_ms"]
        log(f"LIVE L={length} N={n}: step {r['step_ms'].get('p50', 0):.0f}/{r['step_ms'].get('p95', 0):.0f} ms, "
            f"cpu {r['bookkeeping_ms'].get('p50', 0):.0f} ms, lag p95 {r['lag_ms'].get('p95', 0):.0f} ms, "
            f"JEV {r['jev_calls_per_s']:.1f}/s, t3-t1 {t3.get('p50', 0):.0f}/{t3.get('p95', 0):.0f} ms, "
            f"keeps up {r['asr_keeps_up']}")
        return r

    def good(r: dict) -> bool:  # no counted turns is not an overload (keeps_up covers cut JEV calls)
        return r["asr_keeps_up"] and r["t3_minus_t1_ms"].get("p95", 0.0) <= 1000

    last_ok = None
    for n in LIVE_STREAMS:
        try:
            r = one(n)
        except Exception as e:  # noqa: BLE001 — e.g. out of memory at a large N
            rows.append({"length": length, "streams": n, "error": f"{type(e).__name__}: {e}"[:500]})
            log(rows[-1]["error"])
            break
        rows.append(r)
        if not good(r):
            break
        last_ok = n
    else:
        return rows
    if last_ok and "error" not in rows[-1]:
        mid = (last_ok + rows[-1]["streams"]) // 2
        if mid not in (last_ok, rows[-1]["streams"]):
            try:
                rows.append(one(mid))
            except Exception as e:  # noqa: BLE001
                rows.append({"length": length, "streams": mid, "error": f"{type(e).__name__}: {e}"[:500]})
    return sorted(rows, key=lambda r: r["streams"])


@app.function(image=image, gpu="L40S", cpu=8.0, memory=32768, timeout=3600, volumes={str(ROOT): vol})
def run(label: str) -> dict:
    """One GPU: ASR check, JEV alone, the live pipeline; the JSON and report.md into ``capacity/``."""
    import datetime
    import re
    import time

    import torch

    from jevlite.voice import load as L
    from jevlite.voice.asr import NEMOTRON, load_nemo

    vol.reload()
    t_start = time.perf_counter()
    res = {"gpu": torch.cuda.get_device_name(0), "label": label, "versions": _versions(), "errors": []}
    _log(json.dumps({k: res[k] for k in ("gpu", "versions")}))
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    out = OUT / f"{re.sub(r'[^A-Za-z0-9_-]+', '_', res['gpu'])}-{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    def save() -> None:
        res["wall_s"] = time.perf_counter() - t_start
        (out / "capacity.json").write_text(json.dumps(res, indent=1))
        (out / "report.md").write_text(L.report_markdown([res]))
        vol.commit()

    pred, res["jev_variant"], errs = _load_jev("cuda")
    res["errors"] += errs
    spec = load_nemo(NEMOTRON, 1, "cuda")
    res["asr_check"] = _asr_check(spec)
    _log(f"ASR check: {res['asr_check']}")
    save()
    res["jev"] = []
    for length in JEV_LENGTHS:
        try:
            res["jev"].append(_jev_alone(pred, length))
        except Exception as e:  # noqa: BLE001
            res["errors"].append(f"JEV alone L={length}: {type(e).__name__}: {e}"[:500])
            _log(res["errors"][-1])
        save()
    res["live"] = []
    if res["asr_check"]["ok"]:
        sources = _sources()
        for length in LIVE_LENGTHS:
            res["live"] += _live(spec, pred, length, sources)
            save()
    else:
        res["errors"].append("live part skipped: the ASR check failed")
    save()
    _log(L.report_markdown([res]))
    return {"out": str(out), "wall_s": res["wall_s"]}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Start a stage on the deployed app; returns at once.")
    ap.add_argument("stage", choices=["check", "run"])
    ap.add_argument("--gpu", default="L40S", help="L40S, H100! (pinned H100), …")
    a = ap.parse_args()
    if a.stage == "check":
        call = modal.Function.from_name(APP, "check").spawn()
    else:
        call = modal.Function.from_name(APP, "run").with_options(gpu=a.gpu).spawn(a.gpu)
    print(f"started {a.stage}{'' if a.stage == 'check' else ' on ' + a.gpu}: call {call.object_id} ({APP})")
    print(f"logs:    uv run --extra bench modal app logs {APP}")
    print(f"stop:    uv run --extra bench modal app stop {APP}")
    print("results: uv run --extra bench modal volume get jevlite-voice capacity data/speed/voice")
