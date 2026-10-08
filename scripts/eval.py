"""Evaluation report for any model or baseline (TODO Phase 2).

    uv run python scripts/eval.py --data data/test.jsonl --pred preds/nli.jsonl --out reports/nli

Writes ``<out>/report.json`` (every metric, reliability-diagram data) and ``<out>/report.md``
(tables per primitive and slice). Formats of both inputs: ``jevlite.data.unified``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevlite.data.unified import load_predictions, load_records
from jevlite.evaluation import evaluate, json_safe, parse_slices, to_markdown


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, nargs="+", help="records, unified-format JSONL (one or more files)")
    ap.add_argument("--pred", required=True, help="predictions JSONL")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--splits", default=None, help="comma-separated splits to keep, e.g. test_in,test_ood")
    ap.add_argument(
        "--slices",
        default=None,
        help="comma-separated slicing fields, '*' for a cross, e.g. 'source,domain,source*domain,family'. "
        "Default: split, source, domain, criterion, source*domain",
    )
    ap.add_argument("--bins", type=int, default=15, help="equal-mass ECE bins")
    ap.add_argument("--bootstrap", type=int, default=0, help="bootstrap resamples for 95%% CIs (0 = off)")
    ap.add_argument("--min-n", type=int, default=1, help="drop slice cells with fewer examples")
    ap.add_argument("--allow-missing", action="store_true", help="evaluate records that have predictions only")
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",")] if args.splits else None
    records = [r for path in args.data for r in load_records(path, splits)]
    report = evaluate(
        records,
        load_predictions(args.pred),
        slices=parse_slices(args.slices),
        n_bins=args.bins,
        n_boot=args.bootstrap,
        min_n=args.min_n,
        allow_missing=args.allow_missing,
    )
    report["inputs"] = {"data": [str(d) for d in args.data], "pred": str(args.pred), "splits": splits}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(json_safe(report), indent=1, ensure_ascii=False))
    (out / "report.md").write_text(to_markdown(report, args.title or f"Evaluation: {Path(args.pred).stem}"))
    print(f"{report['n_evaluated']} records evaluated → {out / 'report.md'}")
    return report


if __name__ == "__main__":
    main()
