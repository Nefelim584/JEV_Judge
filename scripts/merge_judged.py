"""Put the judged rows of re-runs back into one stand export: one row per question, scored where any run scored it.

    uv run python scripts/merge_judged.py --base <REdo>/results/rerun-judged.csv \\
        --judged <redo>/rerun2/judged-*.csv --out <REdo>/results/rerun-judged-merged.csv

``--base`` is the export to fix (rows with tracebacks). ``--judged`` are the stand's judge outputs, in
order: for each base row (key ``question_id, repeat, turn``) the last scored row wins; a row with an
error never replaces a scored one. Rows that no file scored keep their latest error and go to
``<out stem>-failed.csv`` (same columns), so ``--out`` holds scored rows only.

Columns follow the base file; base-only columns (``source_file``) keep the base value, and
``judged_from`` names the file a row came from. Judged rows whose key is not in the base are counted and
ignored. Input and output are company data: keep them under a git-ignored directory.
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


def key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple((row.get(k) or "").strip() for k in KEY)


def scored(row: dict[str, str]) -> bool:
    return not (row.get("error") or "").strip() and bool((row.get("overall_score") or "").strip())


def last_error_line(row: dict[str, str]) -> str:
    lines = [ln.strip() for ln in (row.get("error") or "").strip().splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.startswith(("For further information", "During task with name"))]
    return lines[-1][:120] if lines else "(no error text, no score)"


def merge(base: list[dict[str, str]], judged: list[tuple[str, list[dict[str, str]]]]):
    keys = [key(r) for r in base]
    dup = [k for k, n in Counter(keys).items() if n > 1]
    if dup:
        raise ValueError(f"{len(dup)} duplicate keys in the base, e.g. {dup[0]}")
    known = set(keys)
    best: dict[tuple[str, ...], tuple[str, dict[str, str]]] = {}
    unknown = 0
    for name, rows in judged:
        for r in rows:
            k = key(r)
            if k not in known:
                unknown += 1
                continue
            prev = best.get(k)
            if scored(r) or prev is None or not scored(prev[1]):
                best[k] = (name, r)

    ok, failed, origin = [], [], Counter()
    for k, b in zip(keys, base):
        name, r = best.get(k, ("base", b))
        row = {c: (r[c] if c in r else b.get(c, "")) for c in b}
        row["judged_from"] = name
        (ok if scored(row) else failed).append(row)
        origin[name if scored(row) else "unresolved"] += 1
    return ok, failed, origin, unknown


def write(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--judged", type=Path, nargs="+", required=True, help="judge outputs, later wins")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--force", action="store_true", help="overwrite --out")
    args = ap.parse_args(argv)
    if args.out.resolve() == args.base.resolve():
        ap.error("--out must differ from --base")
    if args.out.exists() and not args.force:
        ap.error(f"{args.out} exists; pass --force to overwrite it")

    csv.field_size_limit(sys.maxsize)
    base = read(args.base)
    ok, failed, origin, unknown = merge(base, [(p.name, read(p)) for p in args.judged])
    fields = list(base[0].keys()) + ["judged_from"]
    failed_path = args.out.with_name(f"{args.out.stem}-failed{args.out.suffix}")
    write(args.out, ok, fields)
    write(failed_path, failed, fields)

    print(f"{len(base)} base rows -> {len(ok)} scored, {len(failed)} unresolved")
    for name, n in sorted(origin.items()):
        print(f"  {name}: {n}")
    if unknown:
        print(f"  ignored {unknown} judged rows whose key is not in the base")
    if failed:
        print("unresolved by last error:")
        for msg, n in Counter(last_error_line(r) for r in failed).most_common():
            print(f"  {n:4d}  {msg}")
    print(f"wrote {args.out}\nwrote {failed_path}")


if __name__ == "__main__":
    main()
