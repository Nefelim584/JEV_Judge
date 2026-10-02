"""Token-length statistics of exported judge triples; decides whether Stage C is needed (TODO 2.11).

    uv run python scripts/judge_length_stats.py --data data/judge_export.jsonl --out reports/judge_lengths
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from jevlite.data.unified import read_jsonl
from jevlite.judge.export import JudgeSample
from jevlite.judge.lengths import length_stats, to_markdown


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="exported triples, see jevlite.judge.export")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--tokenizer", default="answerdotai/ModernBERT-large")
    args = ap.parse_args(argv)

    samples = [JudgeSample.model_validate(row) for row in read_jsonl(args.data)]
    stats = length_stats(samples, AutoTokenizer.from_pretrained(args.tokenizer))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "lengths.json").write_text(json.dumps(stats, indent=1))
    md = to_markdown(stats)
    (out / "lengths.md").write_text("# Judge input lengths\n\n" + md + "\n")
    print(md)
    return stats


if __name__ == "__main__":
    main()
