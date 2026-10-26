"""eval v2 of the judge benchmarks (todo Phase 4b, ``DATASETS.md`` 8.4): the record lists that keep the
Stage A → Stage B comparison on the same records once Stage B trains on 7 RAGBench subsets.

    uv run python scripts/make_eval_v2.py \\
        --data data/public/{contextual_judge_bench,ragbench,halubench,faithbench}.jsonl \\
        --exclude docs/judge_bench/overlap_ids.txt --out docs/judge_bench/eval_v2

Writes two exclusion files for ``judge_bench_report.py --exclude`` and a manifest:

- ``heldout_exclude.txt``: the comparable slice. Leaves out the passage-overlap ids (``--exclude``,
  repeatable: the Stage B overlap ids are added here once the Stage B mix is built) and every record of
  a trained domain.
- ``in_distribution_exclude.txt``: everything except the trained domains (and the overlap ids), for the
  separate in-distribution report.
- ``manifest.json``: counts per source and domain in each slice, the trained domains, the "seen source"
  flags.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from jevlite.data.public.judge_train import RAGBenchTrain
from jevlite.data.unified import load_records

# RAGBench subsets whose train/validation splits go into Stage B (DATASETS.md 8.2, approved 2026-10-26)
TRAINED_DOMAINS = tuple(f"ragbench/{s}" for s in RAGBenchTrain.SUBSETS)
# held out, but their source is trained on in another form: report them flagged
SEEN_SOURCE = {
    "halubench/pubmedqa": "PubMedQA via RAGBench pubmedqa",
    "contextual_judge_bench (QA-Feedback / ASQA pairs)": "ASQA via Trust-Data, if its ASQA part is trained",
}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--exclude", action="append", default=[], help="id file to leave out of both slices (repeatable)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    records = [r for path in args.data for r in load_records(path)]
    overlap = {i for path in args.exclude for i in Path(path).read_text().split()}
    trained = {r.id for r in records if r.domain in TRAINED_DOMAINS}
    all_ids = {r.id for r in records}

    heldout_ex = overlap | trained
    in_dist_ex = overlap | (all_ids - trained)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "heldout_exclude.txt").write_text("".join(f"{i}\n" for i in sorted(heldout_ex)))
    (out / "in_distribution_exclude.txt").write_text("".join(f"{i}\n" for i in sorted(in_dist_ex)))

    def counts(keep: set[str]) -> dict:
        by_source: Counter = Counter()
        by_domain: Counter = Counter()
        for r in records:
            if r.id in keep:
                by_source[r.source] += 1
                by_domain[r.domain] += 1
        return {"n": sum(by_source.values()), "by_source": dict(sorted(by_source.items())),
                "by_domain": dict(sorted(by_domain.items()))}

    manifest = {
        "data": args.data,
        "exclude_files": args.exclude,
        "trained_domains": list(TRAINED_DOMAINS),
        "seen_source_flags": SEEN_SOURCE,
        "n_records": len(records),
        "n_overlap_excluded": len(overlap & all_ids),
        "heldout": counts(all_ids - heldout_ex),
        "in_distribution": counts(all_ids - in_dist_ex),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: manifest[k] for k in ("n_records", "n_overlap_excluded")}
                     | {s: manifest[s]["by_source"] for s in ("heldout", "in_distribution")}, indent=2))


if __name__ == "__main__":
    main()
