"""Linear probe on a frozen Qwen3-Embedding on the criteria bench (``jevlite.embed_probe``), next to jev-lite.

    # smoke: 32 items per task + 200 training records, a few cents
    uv run --extra bench modal run scripts/modal_embed_probe.py --limit 32 --run probe-smoke
    # the run: Qwen3-Embedding-8B, the 8,233 items of crit-all + 12k Bool records of the Stage D mix (~$1–2)
    uv run --extra bench modal run scripts/modal_embed_probe.py
    # the probe and the report again from the saved embeddings, locally (no GPU, no Modal call)
    uv run --extra bench modal run scripts/modal_embed_probe.py --fit-only

1. On Modal (``--gpu``, H100 by default; ``timeout`` 40 min caps the cost): the model embeds the bench items
   of ``--source`` (a criteria bench run folder, read locally; public data) and ``--n-train`` Bool records
   with a confident target drawn from ``/vol/mix/<mix>/train.e0.jsonl`` (the data jev-lite trained on).
   The embeddings come back to ``data/embed_probe/<run>/emb.npz``.
2. Locally: a logistic regression on the training embeddings (``jevlite.embed_probe.fit_probe``) scores the
   bench items. ``data/criteria_bench/<run>/`` gets the items and the scores of ``--source`` plus the
   probe's, and the usual report (``modal_criteria_bench.write_report``): AUROC and BA@fit side by side.

The probe has no timing in the report: it runs on another GPU than the L40S of the other scorers.
"""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
from pathlib import Path

import modal

HF_CACHE = "/root/.cache/huggingface"
VOL = "/vol"
OUT = Path("data/embed_probe")
BENCH = Path("data/criteria_bench")

app = modal.App("jevlite-embed-probe")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(".", extras=["bench"], frozen=True)
    .env({"HF_HOME": HF_CACHE, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("jevlite")
)
hf_cache = modal.Volume.from_name("jevlite-hf-cache", create_if_missing=True)
stage_c = modal.Volume.from_name("jevlite-stage-c", create_if_missing=True)


@app.function(image=image, gpu="H100", cpu=4.0, memory=32768, timeout=40 * 60, volumes={HF_CACHE: hf_cache, VOL: stage_c})
def embed_remote(model_name: str, bench: list[dict], mix: str, n_train: int, seed: int, max_len: int,
                 tokens_per_batch: int) -> dict:
    import time

    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    from jevlite.data.unified import read_jsonl
    from jevlite.embed_probe import READOUT_TOKEN, embed, hard_label, query_ids, sample_bool_records
    from jevlite.model import disable_kv_cache

    stage_c.reload()
    train = sample_bool_records(read_jsonl(f"{VOL}/mix/{mix}/train.e0.jsonl"), n_train, seed)
    print(f"{len(train)} training records, {len(bench)} bench items", flush=True)

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(model_name)
    model = disable_kv_cache(AutoModel.from_pretrained(model_name, dtype=torch.bfloat16, attn_implementation="sdpa")).cuda().eval()
    hf_cache.commit()  # the weights, for the next run
    load_s = time.perf_counter() - t0
    print(f"{model_name} loaded in {load_s:.0f} s on {torch.cuda.get_device_name(0)}", flush=True)

    readout = tok.convert_tokens_to_ids(READOUT_TOKEN)
    train_rows = [query_ids(tok, r["prompt"], r["state"], max_len, readout) for r in train]
    bench_rows = [query_ids(tok, d["prompt"], d["state"], max_len, readout) for d in bench]
    n_tokens = sum(map(len, train_rows)) + sum(map(len, bench_rows))
    print(f"{n_tokens:,} tokens to embed", flush=True)

    t0 = time.perf_counter()
    train_x = embed(model, train_rows, tok.pad_token_id, "cuda", tokens_per_batch)
    bench_x = embed(model, bench_rows, tok.pad_token_id, "cuda", tokens_per_batch)
    embed_s = time.perf_counter() - t0
    print(f"embedded in {embed_s:.0f} s ({n_tokens / embed_s:,.0f} tokens/s)", flush=True)
    return {
        "train_x": train_x, "train_y": np.array([hard_label(float(r["target"])) for r in train], dtype=np.int8),
        "train_ids": [r["id"] for r in train], "train_source": [r["source"] for r in train],
        "bench_x": bench_x, "bench_ids": [d["id"] for d in bench],
        "meta": {"model": model_name, "gpu": torch.cuda.get_device_name(0), "load_s": load_s, "embed_s": embed_s,
                 "n_tokens": n_tokens, "max_len": max_len, "mix": mix, "seed": seed},
    }


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _bench_items(source: Path, limit: int) -> list[dict]:
    items = _read_jsonl(source / "items.jsonl")
    if not limit:
        return items
    per_task: dict[str, list[dict]] = {}
    for d in items:
        if len(per_task.setdefault(d["task"], [])) < limit:
            per_task[d["task"]].append(d)
    return [d for ds in per_task.values() for d in ds]


def _write_report(run_dir: Path, calib_frac: float, seed: int, n_boot: int) -> None:
    """``write_report`` of the criteria bench script (one report format for both)."""
    spec = importlib.util.spec_from_file_location("modal_criteria_bench", Path(__file__).with_name("modal_criteria_bench.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.write_report(run_dir, calib_frac, seed, n_boot)


def fit_and_report(emb_dir: Path, source: Path, run: str, seed: int, calib_frac: float, n_boot: int) -> None:
    import numpy as np

    from jevlite.embed_probe import fit_probe

    data = np.load(emb_dir / "emb.npz")
    meta = json.loads((emb_dir / "meta.json").read_text())
    clf, info = fit_probe(data["train_x"], data["train_y"], seed=seed)
    print(f"probe: C={info['C']}, held-out AUROC on the training mix {info['val_auroc']}")
    probs = clf.predict_proba(np.asarray(data["bench_x"], dtype=np.float32))[:, 1]
    scores = dict(zip(meta["bench_ids"], map(float, probs), strict=True))

    run_dir = BENCH / run
    (run_dir / "scores").mkdir(parents=True, exist_ok=True)
    shutil.copy(source / "items.jsonl", run_dir / "items.jsonl")
    for path in (source / "scores").glob("*.json"):  # the reference scorers, as measured
        shutil.copy(path, run_dir / "scores" / path.name)
    by_task: dict[str, dict[str, float]] = {}
    for d in _read_jsonl(source / "items.jsonl"):
        if d["id"] in scores:
            by_task.setdefault(d["task"], {})[d["id"]] = scores[d["id"]]
    short = meta["model"].split("/")[-1]
    result = {"key": f"probe_{re.sub(r'[^A-Za-z0-9]+', '_', short).lower()}", "name": f"{short} frozen + linear probe",
              "gpu": None, "tasks": {t: {"scores": s, "timing": {}} for t, s in by_task.items()}, "errors": {},
              "probe": info}
    (run_dir / "scores" / f"{result['key']}.json").write_text(json.dumps(result))
    (emb_dir / "probe.json").write_text(json.dumps(info, indent=2))
    _write_report(run_dir, calib_frac, seed, n_boot)


@app.local_entrypoint()
def main(model: str = "Qwen/Qwen3-Embedding-8B", source: str = "data/criteria_bench/crit-all", mix: str = "stage_d",
         n_train: int = 12000, seed: int = 0, max_len: int = 1024, tokens_per_batch: int = 32768, limit: int = 0,
         gpu: str = "H100", run: str = "", fit_only: bool = False, calib_frac: float = 0.3, n_boot: int = 1000):
    import numpy as np

    source_dir = Path(source)
    run = run or f"probe-{re.sub(r'[^A-Za-z0-9_.-]+', '_', model.split('/')[-1]).lower()}"
    emb_dir = OUT / run
    if not fit_only:
        bench = _bench_items(source_dir, limit)
        n = min(n_train, 200) if limit else n_train
        print(f"{run}: {model} on {gpu}, {len(bench)} bench items from {source_dir}, {n} training records from {mix}")
        res = embed_remote.with_options(gpu=gpu).remote(
            model, [{"id": d["id"], "prompt": d["prompt"], "state": d["state"]} for d in bench], mix, n, seed, max_len,
            tokens_per_batch)
        emb_dir.mkdir(parents=True, exist_ok=True)
        np.savez(emb_dir / "emb.npz", train_x=res["train_x"], train_y=res["train_y"], bench_x=res["bench_x"])
        (emb_dir / "meta.json").write_text(json.dumps({**res["meta"], "source": str(source_dir), "bench_ids": res["bench_ids"],
                                                       "train_ids": res["train_ids"], "train_source": res["train_source"]}))
        print(f"embeddings → {emb_dir}")
    fit_and_report(emb_dir, source_dir, run, seed, calib_frac, n_boot)
