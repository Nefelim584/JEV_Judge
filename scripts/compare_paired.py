"""Paired comparison of a model against one or more references on the same records (``jevlite.paired``).

    uv run python scripts/compare_paired.py --data data/mix/stage_a/clean/test_in.jsonl \\
        data/mix/stage_a/clean/test_ood_source.jsonl data/mix/stage_a/clean/test_ood_template.jsonl \\
        --model jevlite=data/runs/stage_a-full-s42/preds.jsonl \\
        --ref nli=data/baselines/preds/nli.jsonl --ref laya=data/baselines/preds/laya.jsonl \\
        --out data/runs/stage_a-full-s42/paired

Writes ``<out>/paired_<model>_vs_<ref>.json`` and ``.md`` per reference. Records without a prediction
from either side are left out of that comparison (the count is in the JSON).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevlite.data.unified import load_predictions, load_records
from jevlite.evaluation import json_safe, parse_slices
from jevlite.paired import paired_compare, to_markdown


def _named(spec: str) -> tuple[str, str]:
    name, _, path = spec.partition("=")
    if not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {spec!r}")
    return name, path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--model", type=_named, required=True, help="NAME=preds.jsonl")
    ap.add_argument("--ref", type=_named, action="append", required=True, help="NAME=preds.jsonl (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--slices", default="split,source,split*source", help="as in eval.py; the overall row is always added")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--min-n", type=int, default=20)
    args = ap.parse_args(argv)

    records = [r for path in args.data for r in load_records(path)]
    slices = ((), *[s for s in parse_slices(args.slices) if s])
    name, path = args.model
    model = load_predictions(path)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for ref_name, ref_path in args.ref:
        ref = load_predictions(ref_path)
        rows = paired_compare(records, model, ref, slices=slices, n_boot=args.bootstrap, min_n=args.min_n)
        both = sum(r.id in model and r.id in ref for r in records)
        stem = f"paired_{name}_vs_{ref_name}"
        (out / f"{stem}.json").write_text(json.dumps(json_safe({"model": name, "reference": ref_name, "n_records": len(records),
                                                                  "n_both": both, "rows": rows}), indent=1))
        (out / f"{stem}.md").write_text(to_markdown(rows, name, ref_name))
        overall = next(r for r in rows if r["slicing"] == "all") if rows else None
        print(f"{name} vs {ref_name}: {both} of {len(records)} records paired → {out / stem}.md", flush=True)
        if overall:
            print(f"  e.g. {overall['primitive']}: Δacc {100 * overall['d_acc']:+.1f} ({overall['verdict']})")


if __name__ == "__main__":
    main()
