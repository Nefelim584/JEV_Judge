"""Build a training mix from converted datasets (``configs/mixes/*.yaml``).

    uv run python scripts/build_mix.py --config configs/mixes/stage_a.yaml --out data/mix/stage_a --epochs 2

Writes ``train.e{N}.jsonl`` per epoch (shuffled; subsampled pools draw a fresh subset each epoch),
``calib.jsonl``, ``test_in.jsonl``, ``test_ood.jsonl``, and ``mix_report.json`` / ``mix_report.md``
with pool sizes, draws and repeat factors per source and primitive.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevlite.data.mix import EVAL_SPLITS, allocate, build_pools, epoch_lines, eval_lines, load_mix_config, mix_report, report_markdown


def _write(path: Path, lines: list[str]) -> None:
    with path.open("w") as f:
        for line in lines:
            f.write(line + "\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--dir", default=None, help="override the config's data dir")
    args = ap.parse_args(argv)

    cfg = load_mix_config(args.config)
    if args.dir:
        cfg = cfg.__class__(**{**cfg.__dict__, "dir": Path(args.dir)})
    print(f"building pools from {cfg.dir} ({len(cfg.sources)} sources)", flush=True)
    pools = build_pools(cfg)
    allocations = allocate(cfg, pools)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        lines = epoch_lines(cfg, pools, allocations, epoch)
        _write(out / f"train.e{epoch}.jsonl", lines)
        print(f"train.e{epoch}: {len(lines)} records", flush=True)
    for split in EVAL_SPLITS:
        lines = eval_lines(pools, split)
        _write(out / f"{split}.jsonl", lines)
        print(f"{split}: {len(lines)} records", flush=True)

    report = mix_report(cfg, pools, allocations)
    (out / "mix_report.json").write_text(json.dumps(report, indent=2))
    (out / "mix_report.md").write_text(report_markdown(report))
    print(f"report → {out / 'mix_report.md'}")


if __name__ == "__main__":
    main()
