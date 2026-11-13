"""Train a stage on a Modal GPU (Stage C on the eval-stand data, todo Phase 6; ``dataset_gen.md`` section 1).

    # 1. upload the mix built by scripts/build_mix.py (company data → the private volume only)
    uv run --extra bench modal run scripts/modal_train.py --upload data/mix/stage_c
    # 2. smoke: 200 steps, a dev eval and a checkpoint, real speed in run.log (no final predictions: they need a finished run)
    uv run --extra bench modal run scripts/modal_train.py --run stage_c-smoke --max-steps 200
    # 3. the run, on its own in Modal: --background hands the whole run (with its resume loop) to a CPU
    #    driver container and returns; --detach keeps the app alive after the laptop disconnects
    uv run --extra bench modal run --detach scripts/modal_train.py --run stage_c-full-s42 --gpu H100! --background
    #    progress: Modal dashboard → app jevlite-train → logs (or `modal app logs jevlite-train`)
    # 4. results (metrics, run.log, preds.jsonl; the final model only with --fetch-final) → data/runs/<run>/
    uv run --extra bench modal run scripts/modal_train.py --fetch stage_c-full-s42

Company data rules: the mix, the checkpoints and the final model live in the private Modal volume
``jevlite-stage-c`` (``/mix/<mix>/``, ``/runs/<run>/``) and in the git-ignored ``data/`` locally. The Hub
mirror of ``train.checkpoint.hub_repo`` is always off here. Base weights come from the public release
(``--init``, default ``hub:Nefelim584/jev-lite@v0.2.0`` = Stage B), so no Hugging Face token is needed.

The run: ``jevlite.train.train`` with the ``cuda`` stack, ``model.init=checkpoint``, bf16 (``precision:
auto`` on Ampere+), FlashAttention 2 from the Hub kernel (``--attn``; ``sdpa`` as the fallback), and a
micro-batch by GPU memory (``--batch``; the effective batch stays ``train.effective_batch_size``). One
call stops with a checkpoint before the function timeout (``max_minutes``); the entrypoint then calls
again, which resumes. A preempted container is retried by Modal and resumes from the newest checkpoint
on the volume. ``--set key=value;key=value`` passes config overrides as in ``scripts/train.py``.

Image: ``uv sync --frozen --extra cuda --extra bench`` from ``uv.lock`` (bitsandbytes for adamw_8bit,
``kernels`` for FlashAttention) plus the ``jevlite`` source and ``configs/``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

HF_CACHE = "/root/.cache/huggingface"
VOL = "/vol"
CONFIGS = "/root/configs"
TIMEOUT_H = 8
FA2 = "kernels-community/flash-attn2"
DEFAULT_INIT = "hub:Nefelim584/jev-lite@v0.2.0"
# Micro-batch by GPU memory (8k-token records, gradient checkpointing); prefix of the Modal GPU name.
MICRO_BATCH = {"H100": 16, "H200": 16, "B200": 16, "A100-80GB": 16, "RTX-PRO-6000": 16, "A100": 8, "L40S": 8, "A10": 4, "L4": 4}

app = modal.App("jevlite-train")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(".", extras=["cuda", "bench"], frozen=True)
    .env({"HF_HOME": HF_CACHE, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("jevlite")
    .add_local_dir("configs", CONFIGS)
)
hf_cache = modal.Volume.from_name("jevlite-hf-cache", create_if_missing=True)
data = modal.Volume.from_name("jevlite-stage-c", create_if_missing=True)


def set_path(cfg: dict, dotted: str, value) -> None:
    *head, last = dotted.split(".")
    node = cfg
    for k in head:
        node = node.setdefault(k, {})
    node[last] = value


def micro_batch(gpu: str) -> int:
    name = gpu.rstrip("!").upper()
    for prefix in sorted(MICRO_BATCH, key=len, reverse=True):  # "A100-80GB" before "A100"
        if name.startswith(prefix.upper()):
            return MICRO_BATCH[prefix]
    return 8


@app.function(image=image, gpu="H100!", cpu=8.0, memory=65536, timeout=TIMEOUT_H * 3600,
              volumes={HF_CACHE: hf_cache, VOL: data}, retries=modal.Retries(max_retries=3, initial_delay=30.0))
def train_run(run: str, stage: str, overrides: list[str], max_steps: int | None, meta: dict) -> dict:
    import yaml

    from jevlite import checkpoint as ckpt
    from jevlite.config import apply_env, load_config

    cfg = load_config("cuda", config_dir=CONFIGS)
    for item in overrides:
        key, _, value = item.partition("=")
        set_path(cfg, key.strip(), yaml.safe_load(value))
    apply_env(cfg)

    from jevlite.log import logger, setup_logging
    from jevlite.train import train

    # Commit the volume after every checkpoint: a preempted or timed-out container must leave its newest
    # checkpoint for the retry to resume from.
    save = ckpt.CheckpointStore.save

    def save_and_commit(self, *args, **kwargs):
        out = save(self, *args, **kwargs)
        data.commit()
        return out

    ckpt.CheckpointStore.save = save_and_commit

    # A warm container may still hold the previous call's run.log sink, and an open file on the volume makes
    # Volume.reload() fail ("there are open files preventing the operation"): close every sink first.
    logger.remove()
    data.reload()
    run_dir = Path(VOL) / "runs" / run
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log")
    logger.info("run {} on Modal ({}), overrides {}", run, meta.get("gpu"), overrides)
    try:
        result = train(cfg, stage=stage, data_root=f"{VOL}/mix", runs_dir=f"{VOL}/runs", run_name=run, hub_repo=None,
                       max_steps=max_steps, max_minutes=TIMEOUT_H * 60 - 25, meta={"stack": "cuda", "overrides": overrides, **meta})
        logger.info("result: {}", result)
    finally:
        logger.remove()  # close run.log before the commit, and for the next call in this container
        data.commit()
        hf_cache.commit()
    return result


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def upload_mix(local: str) -> None:
    src = Path(local)
    if not (src / "train.e0.jsonl").exists():
        raise FileNotFoundError(f"{src} is not a built mix (no train.e0.jsonl)")
    with data.batch_upload(force=True) as batch:
        batch.put_directory(str(src), f"/mix/{src.name}")
    print(f"uploaded {src} → volume jevlite-stage-c:/mix/{src.name}")


def fetch_run(run: str, final: bool) -> None:
    """The run's files → ``data/runs/<run>/``; checkpoints never, the final model only with ``final``."""
    out = Path("data/runs")
    n = 0
    for entry in data.listdir(f"/runs/{run}", recursive=True):
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        rel = entry.path.lstrip("/")
        parts = Path(rel).parts
        if "checkpoints" in parts or ("final" in parts and not final):
            continue
        dest = out / Path(rel).relative_to("runs")
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            for chunk in data.read_file(rel):
                f.write(chunk)
        n += 1
    print(f"fetched {n} files of {run} → {out / run}")


@app.function(image=image, cpu=1.0, memory=2048, timeout=24 * 3600, volumes={VOL: data})
def train_driver(run: str, stage: str, overrides: list[str], max_steps: int | None, meta: dict, gpu: str) -> dict:
    """The run in Modal, laptop-free: GPU calls until the run ends (a call that hits the function
    timeout pauses with a checkpoint, and the next one resumes from it). A preempted GPU container is
    retried by ``train_run``'s own retries. The result also goes to ``/vol/runs/<run>/driver_result.json``."""
    import json

    # A preempted driver is restarted by Modal with the same input: a run that already ended is not run again.
    data.reload()
    out = Path(VOL) / "runs" / run / "driver_result.json"
    if out.exists():
        print(f"{run}: already finished ({out}), not started again", flush=True)
        return json.loads(out.read_text())
    # the decorator's GPU needs no option binding; another one is bound per call
    call = train_run if gpu == "H100!" else train_run.with_options(gpu=gpu)
    while True:
        result = call.remote(run, stage, overrides, max_steps, meta)
        print(f"{run}: {result}", flush=True)
        if result.get("status") != "paused":
            break
        print("paused before the function timeout: resuming in a new call", flush=True)
    data.reload()
    out = Path(VOL) / "runs" / run / "driver_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    data.commit()
    return result


@app.local_entrypoint()
def main(run: str = "", upload: str = "", fetch: str = "", gpu: str = "H100!", stage: str = "stage_c", set: str = "",
         max_steps: int = 0, init: str = DEFAULT_INIT, attn: str = FA2, batch: int = 0, seed: int = 42, fetch_final: bool = False,
         background: bool = False):
    if upload:
        upload_mix(upload)
    if fetch:
        fetch_run(fetch, fetch_final)
    if not run:
        if not (upload or fetch):
            print("nothing to do: pass --upload, --run or --fetch")
        return
    overrides = [
        "model.init=checkpoint", f"model.init_checkpoint={init}", f"runtime.attn_implementation={attn}",
        f"train.per_device_batch_size={batch or micro_batch(gpu)}", f"seed={seed}",
        *[s.strip() for s in set.split(";") if s.strip()],  # last, so they win
    ]
    meta = {"commit": git_commit(), "gpu": gpu, "world_size": 1}
    if background:
        # needs `modal run --detach`: without it the app (and the run) stops when this entrypoint returns
        fc = train_driver.spawn(run, stage, overrides, max_steps or None, meta, gpu)
        print(f"{run}: started in the background (driver call {fc.object_id}). The laptop may disconnect now.\n"
              f"  logs:    Modal dashboard → app jevlite-train, or `modal app logs jevlite-train`\n"
              f"  results: uv run --extra bench modal run scripts/modal_train.py --fetch {run}")
        return
    call = train_run.with_options(gpu=gpu)
    while True:
        result = call.remote(run, stage, overrides, max_steps or None, meta)
        print(f"{run}: {result}")
        if result.get("status") != "paused":
            break
        print("paused before the function timeout: resuming in a new call")
    fetch_run(run, fetch_final)
