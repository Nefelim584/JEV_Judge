"""Ids of eval records whose passage occurs in training files (``jevlite.data.overlap``), one per line:
the exclusion list for ``make_eval_v2.py --exclude`` (todo Phase 4b).

    uv run python scripts/overlap_ids.py --train data/mix/stage_b/train.e*.jsonl \\
        --data data/public/{contextual_judge_bench,ragbench,halubench,faithbench}.jsonl --out stage_b_overlap_ids.txt

Prints the count per eval source. ``--mode`` as in ``clean_eval.py`` (default ``prefix``).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from jevlite.data.overlap import MODES, overlaps, train_index


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=MODES, default="prefix")
    args = ap.parse_args(argv)

    index = train_index(args.train, args.mode)
    ids, per_source = [], Counter()
    for path in args.data:
        with Path(path).open() as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    if overlaps(row, index, args.mode):
                        ids.append(row["id"])
                        per_source[row["source"]] += 1
    Path(args.out).write_text("".join(f"{i}\n" for i in sorted(set(ids))))
    print(json.dumps({"n": len(set(ids)), "by_source": dict(per_source)}, indent=2))


if __name__ == "__main__":
    main()
