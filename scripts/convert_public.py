"""Convert public datasets to unified records: one JSONL and one stats file per dataset.

    uv run python scripts/convert_public.py --datasets all --out data/public
    uv run python scripts/convert_public.py --datasets boolq,squad_v2 --limit 200 --out data/smoke

``--limit N`` reads the first N rows of every source split and streams HF datasets instead of
downloading them (smoke runs). Raw files from original sources are cached in ``--raw-dir``.
TabFact downloads its whole repo archive (~770 MB) on the first run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from jevlite.data.public import CONVERTERS, get_converter
from jevlite.data.unified import write_jsonl
from jevlite.encoding import serialize_state


def stats(records) -> dict:
    by_split_type: dict[str, Counter] = defaultdict(Counter)
    targets: dict[str, Counter] = defaultdict(Counter)
    templates: Counter = Counter()
    k: Counter = Counter()
    chars = []
    for r in records:
        by_split_type[r.split][r.type] += 1
        templates[r.field("template")] += 1
        if r.type == "bool":
            targets["bool"][str(int(r.target))] += 1
        else:
            targets[r.type][r.candidates[int(r.target)] if r.type == "score" else f"pos{int(r.target)}"] += 1
            k[r.n_candidates] += 1
        chars.append(len(serialize_state(r.state)))
    chars.sort()
    return {
        "n": len(records),
        "splits": {s: dict(c) for s, c in sorted(by_split_type.items())},
        "targets": {t: dict(c.most_common()) for t, c in targets.items()},
        "n_candidates": dict(sorted(k.items())),
        "templates": dict(templates.most_common()),
        "state_chars": {"p50": chars[len(chars) // 2], "p95": chars[int(len(chars) * 0.95)], "max": chars[-1]} if chars else {},
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", required=True, help=f"comma-separated, or 'all': {', '.join(CONVERTERS)}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="first N rows per source split (smoke runs)")
    ap.add_argument("--raw-dir", default=None, help="cache for raw downloads (default: data/raw)")
    args = ap.parse_args(argv)

    if args.raw_dir:
        os.environ["JEVLITE_RAW_DIR"] = args.raw_dir
    names = list(CONVERTERS) if args.datasets == "all" else [n.strip() for n in args.datasets.split(",")]
    out = Path(args.out)
    failed = []
    for name in names:
        conv = get_converter(name)
        conv.streaming = args.limit is not None
        print(f"[{name}] converting ({conv.origin})", flush=True)
        try:
            records = list(conv.records(limit=args.limit))
        except Exception as e:  # keep going: one broken source should not stop the others
            print(f"[{name}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
            failed.append(name)
            continue
        ids = [r.id for r in records]
        if len(set(ids)) != len(ids):
            dup = next(i for i, c in Counter(ids).items() if c > 1)
            print(f"[{name}] FAILED: duplicate record id {dup!r}", file=sys.stderr)
            failed.append(name)
            continue
        write_jsonl(out / f"{name}.jsonl", records)
        s = {"source": name, "license": conv.license, "origin": conv.origin, **stats(records)}
        (out / f"{name}.stats.json").write_text(json.dumps(s, indent=2, ensure_ascii=False))
        splits = ", ".join(f"{sp} {sum(c.values())}" for sp, c in s["splits"].items())
        print(f"[{name}] {s['n']} records: {splits}", flush=True)
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
