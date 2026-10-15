"""Clean eval sets: drop records whose passage was trained on, and the mid-run eval subsets that took
part in decisions (todo, Stage A review, Q1 and Q2; logic in ``jevlite.data.overlap``).

    uv run python scripts/clean_eval.py \\
        --train data/mix/stage_a/train.e*.jsonl data/mix/stage_a_nosa/train.e*.jsonl \\
        --data data/mix/stage_a/test_in.jsonl data/mix/stage_a/test_ood.jsonl data/mix/stage_a/dev.jsonl \\
        --drop-eval-subset test_in test_ood --out data/mix/stage_a/clean

``--train`` lists the training files of **every** run that will be compared on the cleaned sets.
``--drop-eval-subset`` names the splits whose ``train.eval_max_records`` subset was evaluated during
earlier runs (the ``test_*`` files of Stage A); ``dev`` is not in it. The subset is the first records
of the **whole file** by hash, so it moves when records are added: if the split has grown since the
run (e.g. new ``test_ood`` sources), pass the file the run saw with ``--eval-subset-ref`` (for Stage A:
``data/mix/stage_a/run_eval/``, the run's ``test_ood`` = today's without MMLU and FaithEval). Writes the cleaned files under
the same names to ``--out``; the cleaned ``test_ood`` is also split into ``test_ood_source.jsonl`` and
``test_ood_template.jsonl`` (``split`` field rewritten, so ``eval.py`` reports them separately), plus ``overlap_report.json`` / ``.md`` (counts per split and source).
Predictions made on the full files stay valid: ``eval.py --data <clean files> --pred <full preds>``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from jevlite.data.overlap import MODES, eval_subset_ids, ood_kind, overlaps, train_domains, train_index


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=MODES, default="prefix")
    ap.add_argument("--drop-eval-subset", nargs="*", default=[], metavar="SPLIT",
                    help="file stems whose mid-run eval subset is dropped (e.g. test_in test_ood)")
    ap.add_argument("--eval-max-records", type=int, default=2000, help="train.eval_max_records of the earlier runs")
    ap.add_argument("--eval-subset-ref", nargs="*", default=[], metavar="SPLIT=PATH",
                    help="the file a run evaluated on, if the split has changed since (the subset depends on the whole "
                         "file: adding records moves it). Default: the --data file itself")
    ap.add_argument("--seed", type=int, default=42, help="seed of the earlier runs (eval subset order)")
    args = ap.parse_args(argv)

    print(f"indexing passages of {len(args.train)} training files ({args.mode})", flush=True)
    index = train_index(args.train, args.mode)
    domains = train_domains(args.train)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"mode": args.mode, "train": args.train, "eval_max_records": args.eval_max_records, "splits": {}}
    refs = dict(spec.split("=", 1) for spec in args.eval_subset_ref)
    for path in map(Path, args.data):
        rows = [json.loads(line) for line in path.open() if line.strip()]
        burned: set[str] = set()
        if path.stem in args.drop_eval_subset:
            ref = [json.loads(line) for line in open(refs[path.stem]) if line.strip()] if path.stem in refs else rows
            burned = eval_subset_ids(ref, args.eval_max_records, args.seed)
        stats: dict[str, Counter] = defaultdict(Counter)
        kept = []
        for r in rows:
            s = stats[r["source"]]
            s["total"] += 1
            if r["id"] in burned:
                s["eval_subset"] += 1
            elif overlaps(r, index, args.mode):
                s["overlap"] += 1
            else:
                s["kept"] += 1
                kept.append(r)
        with (out / path.name).open("w") as f:
            for r in kept:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        report["splits"][path.stem] = {src: dict(c) for src, c in sorted(stats.items())}
        print(f"{path.stem}: kept {len(kept)} of {len(rows)} (eval subset {len(burned)})", flush=True)
        if path.stem == "test_ood":
            parts: dict[str, list[dict]] = {"test_ood_source": [], "test_ood_template": []}
            for r in kept:
                kind = ood_kind(r, domains)
                parts[kind].append({**r, "split": kind})
            for kind, part in parts.items():
                with (out / f"{kind}.jsonl").open("w") as f:
                    for r in part:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                report.setdefault("ood", {})[kind] = dict(sorted(Counter(r["source"] for r in part).items()))
                print(f"  {kind}: {len(part)}", flush=True)
    (out / "overlap_report.json").write_text(json.dumps(report, indent=2))
    (out / "overlap_report.md").write_text(_markdown(report))


def _markdown(report: dict) -> str:
    lines = [
        "# Clean eval sets", "",
        f"Passage overlap mode `{report['mode']}` against {len(report['train'])} training files; "
        f"mid-run eval subsets of {report['eval_max_records']} records dropped where listed. "
        "A record dropped for both reasons counts as `eval_subset`.", "",
    ]
    for split, sources in report["splits"].items():
        tot = Counter()
        for c in sources.values():
            tot.update(c)
        lines += [f"## {split}: kept {tot['kept']} of {tot['total']}", "",
                  "| Source | Total | Eval subset | Passage overlap | Overlap % | Kept |", "|---|---|---|---|---|---|"]
        for src, c in sources.items():
            rest = c.get("total", 0) - c.get("eval_subset", 0)
            pct = c.get("overlap", 0) / rest if rest else 0.0
            lines.append(f"| {src} | {c.get('total', 0)} | {c.get('eval_subset', 0)} | {c.get('overlap', 0)} | {pct:.0%} | {c.get('kept', 0)} |")
        lines.append("")
    for kind, sources in report.get("ood", {}).items():
        lines += [f"**{kind}:** {sum(sources.values())} records — " + ", ".join(f"{k} {v}" for k, v in sources.items()), ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
