"""Run a zero-shot baseline over unified records and write predictions for ``scripts/eval.py``.

    uv run python scripts/predict_baseline.py --baseline nli  --data data/test.jsonl --out preds/nli.jsonl
    uv run python scripts/predict_baseline.py --baseline laya --data a.jsonl b.jsonl --out preds/laya.jsonl --resume

``--model`` picks the NLI checkpoint, or the Laya checkpoint (base | multilingual | typed-decisions).

Predictions are appended to ``--out`` as they come, so an interrupted run loses at most one call;
``--resume`` skips the records already in ``--out``. A record the baseline cannot handle (e.g. Laya:
options longer than its 192-token header) is skipped and listed in ``<out>.skipped.jsonl``; evaluate
with ``eval.py --allow-missing`` and look at ``n_missing`` in the report.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from jevlite.baselines import BASELINES, load_baseline
from jevlite.baselines.base import iter_predictions
from jevlite.data.unified import load_records


def _ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open() as f:
        return {json.loads(line)["id"] for line in f if line.strip()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True, choices=BASELINES)
    ap.add_argument("--data", required=True, nargs="+", help="one or more record files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--splits", default=None, help="comma-separated splits to keep")
    ap.add_argument("--limit", type=int, default=None, help="first N records only (smoke runs)")
    ap.add_argument("--max-len", type=int, default=512, help="NLI only: max tokens per premise + hypothesis")
    ap.add_argument("--resume", action="store_true", help="keep --out and skip the records already in it")
    args = ap.parse_args(argv)

    splits = [s.strip() for s in args.splits.split(",")] if args.splits else None
    records = [r for path in args.data for r in load_records(path, splits)][: args.limit]
    if len({r.id for r in records}) != len(records):
        raise SystemExit("record ids must be unique across --data files")

    out, skipped_path = Path(args.out), Path(f"{args.out}.skipped.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume:
        done = _ids(out) | _ids(skipped_path)
        records = [r for r in records if r.id not in done]
        print(f"resume: {len(done)} records done, {len(records)} to go", flush=True)
    else:
        for path in (out, skipped_path):
            path.unlink(missing_ok=True)

    predictor = load_baseline(args.baseline, args.model, args.device, args.max_len)
    n_pred = n_skip = 0
    start = time.perf_counter()
    with out.open("a") as f_pred, skipped_path.open("a") as f_skip:
        for preds, skipped in iter_predictions(predictor, records, skip_errors=True):
            for p in preds:
                f_pred.write(p.model_dump_json(exclude_none=True) + "\n")
            for s in skipped:
                f_skip.write(json.dumps({"id": s.id, "error": s.error}) + "\n")
            f_pred.flush()
            f_skip.flush()
            n_pred += len(preds)
            n_skip += len(skipped)
            done = n_pred + n_skip
            rate = done / max(time.perf_counter() - start, 1e-9)
            eta = (len(records) - done) / rate / 60 if rate else 0.0
            print(f"\r  {predictor.name}: {done}/{len(records)}  skipped {n_skip}  {rate:.1f} rec/s  ETA {eta:.0f} min",
                  end="", flush=True)
    print()
    print(f"{n_pred} predictions from {predictor.name} → {out}" + (f"; {n_skip} skipped → {skipped_path}" if n_skip else ""))
    if skipped_path.stat().st_size == 0:
        skipped_path.unlink()


if __name__ == "__main__":
    main()
