"""Speed benchmark of jev-lite v0.2.0 (Stage B + calibration) on Modal GPUs (todo Phase 9).

    uv run --extra bench modal run scripts/modal_latency.py      # full grid: 5 GPUs × 4 lengths × 4 variants
    uv run --extra bench modal run scripts/modal_latency.py --gpus L4 --variants fp32-sdpa,bf16-fa2 --runs 10   # smoke

The grid (``jevlite.latency``): context lengths 512 / 2048 / 4096 / 8192 tokens × fp32-sdpa, bf16-sdpa,
bf16-fa2, bf16-fa3 (FlashAttention has no fp32 kernels; FA3 is Hopper only, n/a on the others), on
H100, A100-80GB, L40S, L4 and A10 in parallel, one container per GPU, same settings on every GPU
(L4 / A10 have 24 GB: the throughput batch may be halved at 8k, the batch used is recorded). Per cell: response time of one request
(100 calls: p50 / p90 / p95 / p99, mean, std, min, max, raw samples), forward-only time, throughput
(requests/s and tokens/s at batch 8) and the drift of the probabilities vs fp32.

Output in ``data/speed/modal/``: ``<gpu>.json`` (everything, with raw samples), ``speed.csv`` (one row per
GPU × variant × length) and ``report.md`` (tables per GPU and all GPUs at 8k).

Setup once: ``uv run --extra bench modal setup`` (log in). The model is the public release
``Nefelim584/jev-lite@v0.2.0`` (weights + ``calibration.json``), so no Hugging Face secret is needed.
The model (~1.6 GB) and the Hub kernels are cached in the Modal volume ``jevlite-hf-cache``; each GPU
also saves its JSON there (``speed/<gpu>-<time>.json``), fetch with
``modal volume get jevlite-hf-cache speed/ data/speed/modal/volume/`` if the local run was cut off.

Image: ``uv sync --frozen --extra bench`` from this repo's ``uv.lock`` (the same torch / transformers as
locally) plus the ``jevlite`` source. FlashAttention comes from the Hub through ``kernels``, no
``flash-attn`` build. ``H100!`` pins an H100 (plain ``H100`` may be upgraded by Modal to an H200).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import modal

MODEL_REPO = "Nefelim584/jev-lite"
REVISION = "v0.2.0"
HF_CACHE = "/root/.cache/huggingface"
OUT = Path("data/speed/modal")

app = modal.App("jevlite-latency")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(".", extras=["bench"], frozen=True)
    .env({"HF_HOME": HF_CACHE, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("jevlite")
)
hf_cache = modal.Volume.from_name("jevlite-hf-cache", create_if_missing=True)


# cpu=4: tokenisation runs on the CPU and is part of the response time; Modal's default share is a fraction of a core.
@app.function(image=image, gpu="H100!", cpu=4.0, memory=16384, timeout=3 * 3600, volumes={HF_CACHE: hf_cache})
def bench(variants: list[str] | None, lengths: list[int], n_warmup: int, n_runs: int, throughput_batch: int,
          n_throughput: int) -> dict:
    from huggingface_hub import snapshot_download

    from jevlite.latency import run_benchmark, variants_by_name

    final = Path(snapshot_download(MODEL_REPO, revision=REVISION))
    hf_cache.commit()
    if not (final / "calibration.json").exists():
        print(f"WARNING: {MODEL_REPO}@{REVISION} has no calibration.json: timing without the calibration step")
    result = run_benchmark(final, variants_by_name(variants), lengths, "cuda", n_warmup, n_runs, throughput_batch,
                           n_throughput, log=lambda s: print(s, flush=True))
    result["model"] = f"{MODEL_REPO}@{REVISION}"
    result["calibration"] = (final / "calibration.json").exists()
    # A copy on the volume, so the results survive a closed laptop:
    #   modal volume get jevlite-hf-cache speed/ data/speed/modal/volume/
    import datetime
    import json
    import re

    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    out = Path(HF_CACHE) / "speed" / f"{re.sub(r'[^A-Za-z0-9_-]+', '_', result['env'].get('gpu', 'gpu'))}-{stamp}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"saved to volume jevlite-hf-cache: {out.relative_to(HF_CACHE)}", flush=True)
    hf_cache.commit()  # Hub kernels downloaded during the run, and the copy
    return result


@app.local_entrypoint()
def main(gpus: str = "H100!,A100-80GB,L40S,L4,A10", variants: str = "", lengths: str = "", warmup: int = 10, runs: int = 100,
         batch: int = 8, throughput_runs: int = 20):
    from jevlite.latency import DEFAULT_LENGTHS, MAX_LEN, report_markdown, rows_long, variants_by_name

    names = [v for v in variants.split(",") if v] or None
    variants_by_name(names)  # fail here, not after the containers start
    length_list = [int(n) for n in lengths.split(",") if n] or list(DEFAULT_LENGTHS)
    if max(length_list) > MAX_LEN:
        raise ValueError(f"lengths above {MAX_LEN}")

    calls = {gpu: bench.with_options(gpu=gpu).spawn(names, length_list, warmup, runs, batch, throughput_runs)
             for gpu in gpus.split(",") if gpu}
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for gpu, call in calls.items():
        try:
            res = call.get()
        except Exception as e:  # noqa: BLE001 — keep the other GPUs
            print(f"{gpu}: FAILED {type(e).__name__}: {e}")
            continue
        res["modal_gpu"] = gpu
        results.append(res)
        path = OUT / f"{re.sub(r'[^A-Za-z0-9_-]+', '_', gpu)}.json"
        path.write_text(json.dumps(res, indent=2))
        print(f"{gpu}: {res['env'].get('gpu')} → {path}")

    if results:
        rows = rows_long(results)
        fields = list(dict.fromkeys(k for r in rows for k in r))
        with (OUT / "speed.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        report = report_markdown(results)
        (OUT / "report.md").write_text(f"# jev-lite {REVISION} speed on Modal\n\n" + report)
        print("\n" + report)
