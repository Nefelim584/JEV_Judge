"""Run a zero-shot baseline over unified records and write predictions for ``scripts/eval.py``.

    uv run python scripts/predict_baseline.py --baseline nli  --data data/test.jsonl --out preds/nli.jsonl
    uv run python scripts/predict_baseline.py --baseline laya --data data/test.jsonl --out preds/laya.jsonl

``--model`` picks the NLI checkpoint, or the Laya checkpoint (base | multilingual | typed-decisions).
"""

from __future__ import annotations

import argparse

from jevlite.baselines import BASELINES, load_baseline
from jevlite.baselines.base import predict_records
from jevlite.data.unified import load_records, write_jsonl


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, choices=BASELINES)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--splits", default=None, help="comma-separated splits to keep")
    ap.add_argument("--limit", type=int, default=None, help="first N records only (smoke runs)")
    ap.add_argument("--max-len", type=int, default=512, help="NLI only: max tokens per premise + hypothesis")
    args = ap.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",")] if args.splits else None
    records = load_records(args.data, splits)[: args.limit]
    predictor = load_baseline(args.baseline, args.model, args.device, args.max_len)
    preds = predict_records(predictor, records, progress=True)
    n = write_jsonl(args.out, preds)
    print(f"{n} predictions from {predictor.name} → {args.out}")


if __name__ == "__main__":
    main()
