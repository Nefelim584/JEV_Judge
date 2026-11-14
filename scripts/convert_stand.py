"""Convert eval-stand rows (CSV exports) into jev-lite records for Stage C (``jevlite.data.stand``).

    uv run python scripts/convert_stand.py --rows data/stand/rejudged/*.csv --criteria data/stand/criteria.json \\
        --out data/stand/records --eval-ids data/stand/fresh_ids.csv --test 0.12 --calib 0.08 --dev 0.04

The rows and the criteria spec are company data: keep them under the git-ignored ``data/`` and pass
them by path. Writes ``<out>/stand.jsonl`` (all splits; the Stage C mix reads it with ``path``),
``<out>/splits.csv`` (row id → split: freeze it before training and pass it back with ``--splits``, so
later conversions keep the same test) and ``<out>/report.json`` (counts only, no text).

Splits: ``--splits`` (a CSV with ``id,split``) fixes them; otherwise groups (a question or a whole
dialogue) are hashed into test_in / calib / dev by the shares, and only rows listed in ``--eval-ids``
(fresh runs of the current assistant; a CSV with an ``id`` column) may leave train.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from jevlite.data.stand import CriteriaSpec, convert_rows
from jevlite.data.unified import write_jsonl


def read_rows(paths: list[str]) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    rows: list[dict] = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    return rows


def read_ids(path: str) -> set[str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"].strip() for row in csv.DictReader(f) if row.get("id")}


def read_splits(path: str) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        return {row["id"].strip(): row["split"].strip() for row in csv.DictReader(f)}


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", nargs="+", required=True, help="stand CSV exports (same columns as data/mapped_q2.csv)")
    ap.add_argument("--criteria", required=True, help="criteria spec JSON (see jevlite.data.stand)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", default=None, help="CSV id,split: fixed splits (a frozen test)")
    ap.add_argument("--eval-ids", default=None, help="CSV with an id column: rows that may go to test / calib / dev")
    ap.add_argument("--test", type=float, default=0.0, help="share of eligible groups in test_in")
    ap.add_argument("--calib", type=float, default=0.0)
    ap.add_argument("--dev", type=float, default=0.0)
    ap.add_argument("--tokenizer", default="answerdotai/ModernBERT-large", help="'' = no history trimming")
    ap.add_argument("--max-len", type=int, default=8192, help="= train.stage_c.max_len")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    spec = CriteriaSpec.load(args.criteria)
    rows = read_rows(args.rows)
    count = None
    if args.tokenizer:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        count = lambda text: len(tok(text, add_special_tokens=False).input_ids)  # noqa: E731
    eligible_ids = read_ids(args.eval_ids) if args.eval_ids else None
    records, report = convert_rows(
        rows, spec, explicit_splits=read_splits(args.splits) if args.splits else None,
        shares={"test_in": args.test, "calib": args.calib, "dev": args.dev},
        eval_eligible=(lambda r: r.id in eligible_ids) if eligible_ids is not None else None,
        count_tokens=count, max_len=args.max_len, seed=args.seed,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "stand.jsonl", records)
    split_of = {r.field("stand_row"): r.split for r in records}
    with (out / "splits.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split"])
        w.writerows(sorted(split_of.items()))
    report = {"inputs": args.rows, "criteria": args.criteria, "max_len": args.max_len, **report}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
