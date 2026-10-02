"""Black-box checks of a baseline: option-order sensitivity, IIA, confidence on nonsense inputs.

    uv run python scripts/probe_blackbox.py --baseline laya --data data/test.jsonl --out reports/laya_probes.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevlite.baselines import BASELINES, load_baseline
from jevlite.data.unified import load_records
from jevlite.probes import run_probes


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, choices=BASELINES)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--n-items", type=int, default=50)
    ap.add_argument("--n-perms", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",")] if args.splits else None
    predictor = load_baseline(args.baseline, args.model, args.device)
    result = run_probes(predictor, load_records(args.data, splits), args.n_items, args.n_perms, args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    main()
