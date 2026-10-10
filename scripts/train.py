"""Train one run of a stage (TODO Phase 6); resumes automatically from its newest checkpoint.

    # Kaggle 2×T4 (one process per GPU)
    torchrun --nproc_per_node 2 scripts/train.py --stack cuda --run stage_a-full-s42 \\
        --data-root /kaggle/input/datasets/mrgrin548/jevlite-mix --runs-dir /tmp/jevlite-runs --max-minutes 690
    # M1
    uv run python scripts/train.py --stack apple_silicon --run stage_a-lora-s42
    # smoke run: 30 steps, eval and a checkpoint at the end, local only
    uv run python scripts/train.py --stack cuda --run smoke --max-steps 30 --no-hub

``--set key=value`` overrides any config value (YAML syntax), e.g. ``--set train.stage_a.mix=mixes/stage_a_nosa.yaml``
for the no-SA version, ``--set seed=7``. The Hub repo is ``train.checkpoint.hub_repo``; the token comes
from ``HF_TOKEN``.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from datetime import timedelta
from pathlib import Path

import yaml


def set_path(cfg: dict, dotted: str, value) -> None:
    *head, last = dotted.split(".")
    node = cfg
    for k in head:
        node = node.setdefault(k, {})
    node[last] = value


def commit() -> str:
    try:
        root = Path(__file__).resolve().parents[1]
        return subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stack", required=True, choices=["apple_silicon", "cuda"])
    ap.add_argument("--stage", default="stage_a")
    ap.add_argument("--run", required=True, help="run name: one folder locally and on the Hub; the same name resumes")
    ap.add_argument("--data-root", default="data/mix", help="folder holding <mix>/train.e*.jsonl")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps (smoke runs)")
    ap.add_argument("--max-minutes", type=float, default=None, help="stop with a checkpoint after this long (Kaggle: 690)")
    ap.add_argument("--no-hub", action="store_true", help="keep checkpoints local only")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)

    from jevlite.config import apply_env, load_config

    cfg = load_config(args.stack)
    for item in args.set:
        key, _, value = item.partition("=")
        set_path(cfg, key, yaml.safe_load(value))
    apply_env(cfg)

    import torch
    import torch.distributed as dist

    from jevlite.log import logger, setup_logging
    from jevlite.train import Dist, train

    d = Dist(rank=int(os.environ.get("RANK", 0)), world=int(os.environ.get("WORLD_SIZE", 1)), local_rank=int(os.environ.get("LOCAL_RANK", 0)))
    if d.world > 1:
        if torch.cuda.is_available():
            torch.cuda.set_device(d.local_rank)
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo", timeout=timedelta(minutes=60))
    run_dir = Path(args.runs_dir) / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(run_dir / "run.log" if d.main else None, level="INFO" if d.main else "WARNING")
    hub = None if args.no_hub else cfg["train"].get("checkpoint", {}).get("hub_repo")
    logger.info("run {} (stack {}, stage {}, world {}), hub {}", args.run, args.stack, args.stage, d.world, hub or "off")
    try:
        result = train(
            cfg, stage=args.stage, data_root=args.data_root, runs_dir=args.runs_dir, run_name=args.run, hub_repo=hub,
            max_steps=args.max_steps, max_minutes=args.max_minutes, dist_info=d,
            meta={"stack": args.stack, "commit": commit(), "overrides": args.set, "world_size": d.world},
        )
        logger.info("result: {}", result)
    finally:
        if d.world > 1:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
