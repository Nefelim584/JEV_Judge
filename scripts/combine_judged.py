"""Stack judged stand exports into one CSV: scored rows to ``--out``, the rest to ``<out stem>-failed.csv``.

    uv run python scripts/combine_judged.py --out <REdo>/results/combined.csv \\
        <REdo>/results/rejudge-judged.csv <REdo>/results/rerun-judged-merged.csv

Columns are the union of the inputs in first-seen order (missing ones are empty); ``results_file`` names
the input a row came from. A key (``question_id, repeat, turn``) found in two inputs is an error: pass
each question once (e.g. the merged rerun file, not the raw one). A row is scored when it has no error
and an ``overall_score``. Input and output are company data: keep them under a git-ignored directory.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

KEY = ("question_id", "repeat", "turn")


def read(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def scored(row: dict[str, str]) -> bool:
    return not (row.get("error") or "").strip() and bool((row.get("overall_score") or "").strip())


def combine(inputs: list[tuple[str, list[dict[str, str]]]]):
    fields: list[str] = []
    seen: dict[tuple[str, ...], str] = {}
    ok, failed = [], []
    for name, rows in inputs:
        for r in rows:
            fields += [c for c in r if c not in fields]
            k = tuple((r.get(c) or "").strip() for c in KEY)
            if k in seen:
                raise ValueError(f"key {k} is in both {seen[k]} and {name}")
            seen[k] = name
            (ok if scored(r) else failed).append({**r, "results_file": name})
    fields = [c for c in fields if c != "results_file"] + ["results_file"]
    return ok, failed, fields


def write(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL, restval="")
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    args = ap.parse_args(argv)
    if args.out.resolve() in {p.resolve() for p in args.inputs}:
        ap.error("--out must not be one of the inputs")
    if args.out.exists() and not args.force:
        ap.error(f"{args.out} exists; pass --force to overwrite it")

    csv.field_size_limit(sys.maxsize)
    ok, failed, fields = combine([(p.name, read(p)) for p in args.inputs])
    failed_path = args.out.with_name(f"{args.out.stem}-failed{args.out.suffix}")
    write(args.out, ok, fields)
    write(failed_path, failed, fields)

    by_file = Counter((r["results_file"], scored(r)) for r in ok + failed)
    for p in args.inputs:
        print(f"  {p.name}: {by_file[(p.name, True)]} scored, {by_file[(p.name, False)]} failed")
    print(f"{len(ok)} scored -> {args.out}\n{len(failed)} failed -> {failed_path}")
    print("scored by category:", dict(Counter(r.get("category", "") for r in ok).most_common()))


if __name__ == "__main__":
    main()
