"""Markdown reports with charts (matplotlib PNGs + interactive plotly HTML twins).

Compare eval reports (baselines, variants, seeds, learning curves):

    uv run python scripts/make_report.py compare --out reports/baselines \\
        --run nli=reports/nli/report.json --run laya=reports/laya/report.json
    # seeds: repeat a name; learning-curve points: name@x
    uv run python scripts/make_report.py compare --out reports/h1 \\
        --run synthetic@1000=runs/s1k-s1/report.json --run synthetic@1000=runs/s1k-s2/report.json \\
        --run human@1000=runs/h1k-s1/report.json

Training runs (``metrics.jsonl`` written by ``jevlite.reports.training.TrainingLog``):

    uv run python scripts/make_report.py training --out reports/stage_a \\
        --run lora=runs/stage_a_lora --run full=runs/stage_a_full

Needs the ``reports`` extra: ``uv sync --extra reports``.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _named(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    return name.strip(), Path(path.strip())


def main(argv: list[str] | None = None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)

    cmp = sub.add_parser("compare", help="compare scripts/eval.py reports")
    cmp.add_argument("--run", action="append", required=True, help="name=report.json or name@x=report.json; repeat a name for seeds")
    cmp.add_argument("--out", required=True)
    cmp.add_argument("--title", default="Comparison report")
    cmp.add_argument("--slice-metric", default="accuracy", help="metric for the per-slice charts (accuracy, kappa, ece, …)")
    cmp.add_argument("--slice-limit", type=int, default=15, help="largest N slice values per chart")

    tr = sub.add_parser("training", help="report on training runs")
    tr.add_argument("--run", action="append", required=True, type=_named, help="name=run_dir (or name=metrics.jsonl)")
    tr.add_argument("--out", required=True)
    tr.add_argument("--title", default="Training report")
    tr.add_argument("--smooth", type=float, default=0.9, help="EMA smoothing of train curves, 0 = off")

    args = ap.parse_args(argv)
    if args.kind == "compare":
        from jevlite.reports.compare import build_report, parse_run

        path = build_report([parse_run(s) for s in args.run], Path(args.out), args.title, args.slice_metric, args.slice_limit)
    else:
        from jevlite.reports.training import build_report

        runs = dict(args.run)
        if len(runs) != len(args.run):
            ap.error("training run names must be unique")
        path = build_report(runs, Path(args.out), args.title, args.smooth)
    print(f"report → {path}")
    return path


if __name__ == "__main__":
    main()
