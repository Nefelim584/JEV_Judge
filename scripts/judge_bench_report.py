"""Judge benchmarks report: Bool balanced accuracy + AUROC, pairwise consistent accuracy (metrics in
``jevlite.judge_bench``), per benchmark and subset, for several models side by side.

    uv run python scripts/judge_bench_report.py --data data/public/{contextual_judge_bench,ragbench,halubench,faithbench}.jsonl \\
        --pred jevlite=data/judge_bench/preds/jevlite_512.jsonl --pred nli=... --pred laya=... \\
        --exclude data/judge_bench/overlap_ids.txt --tokenizer data/runs/stage_a-full-s42/final/encoder \\
        --out data/judge_bench/report --bootstrap 1000

Metrics are computed on the records **every** model predicted (a record a model skipped, e.g. a prompt
longer than its max_len, drops out for all), minus ``--exclude`` (passages seen in training). With
``--tokenizer``, each table also gets a ``fits512`` slice: records whose packed sequence fits 512 tokens
untruncated, the training length of Stage A.

Every table also gets the response time of each model, from ``latency_ms`` of its predictions (written by
``predict_baseline.py``: the wall time of one model call, tokenisation to probabilities, divided by the
questions in that call), as p50 / p95 / mean in milliseconds per record, per slice and over all records.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from jevlite.data.unified import load_predictions, load_records
from jevlite.evaluation import json_safe
from jevlite.judge_bench import bool_summary, pairwise_summary

FIT_LEN = 512
LATENCY = ("p50", "p95", "mean")


def _latency(preds) -> dict | None:
    """p50 / p95 / mean of ``latency_ms`` in milliseconds; None when the predictions carry no timing."""
    lat = [p.latency_ms for p in preds if p.latency_ms is not None]
    if not lat:
        return None
    return {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95)), "mean": float(np.mean(lat))}


def _fits(records, tokenizer_path: str) -> set[str]:
    from transformers import AutoTokenizer

    from jevlite.encoding import serialize_state

    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    out = set()
    for r in records:
        header = r.prompt + " " + " ".join(r.candidates or [])
        n = sum(len(x) for x in tok([header, serialize_state(r.state)], add_special_tokens=False)["input_ids"])
        if n + 3 + len(r.candidates or [None]) <= FIT_LEN:  # [CLS] [SEP] [SEP] + one marker per candidate
            out.add(r.id)
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--pred", nargs="+", required=True, metavar="NAME=PATH")
    ap.add_argument("--exclude", default=None, help="file with one record id per line to leave out")
    ap.add_argument("--tokenizer", default=None, help="adds the fits512 slice")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    records = [r for path in args.data for r in load_records(path)]
    excluded = set(Path(args.exclude).read_text().split()) if args.exclude else set()
    preds = {name: load_predictions(path) for name, path in (spec.split("=", 1) for spec in args.pred)}
    common = [r for r in records if r.id not in excluded and all(r.id in p for p in preds.values())]
    fits = _fits(common, args.tokenizer) if args.tokenizer else None

    # benchmark → slice → records; a slice is "all", a domain, or "fits512"
    slices: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in common:
        s = slices[r.source]
        s["all"].append(r)
        s[r.domain].append(r)
        if fits is not None and r.id in fits:
            s["fits512"].append(r)

    report = {
        "n_records": len(records), "n_excluded": len(excluded & {r.id for r in records}), "n_common": len(common),
        "missing": {name: sum(r.id not in p for r in records) for name, p in preds.items()},
        "latency_ms": {name: _latency([p[r.id] for r in common]) for name, p in preds.items()}, "benchmarks": {},
    }
    for source, by_slice in sorted(slices.items()):
        rows = {}
        for name, slice_records in by_slice.items():
            pairwise = slice_records[0].type == "choice"
            fn = pairwise_summary if pairwise else bool_summary
            rows[name] = {}
            for m, p in preds.items():
                slice_preds = [p[r.id] for r in slice_records]
                rows[name][m] = fn(slice_records, slice_preds, args.bootstrap) | {"latency_ms": _latency(slice_preds)}
        report["benchmarks"][source] = rows

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(json_safe(report), indent=2))
    (out / "report.md").write_text(_markdown(report, list(preds)))
    print((out / "report.md").read_text())


def _cell(m: dict, key: str) -> str:
    v = m.get(key)
    if v is None or v != v:
        return "—"
    ci = m.get("ci95", {}).get(key)
    return f"{v:.3f}" + (f" [{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "")


def _lat_cell(lat: dict | None) -> str:
    return " / ".join(f"{lat[k]:.1f}" for k in LATENCY) if lat else "—"


def _markdown(report: dict, models: list[str]) -> str:
    lines = ["# Judge benchmarks", "",
             f"{report['n_common']} records predicted by every model, of {report['n_records']}; "
             f"{report['n_excluded']} left out for passage overlap with training. "
             "Missing per model (skipped or not run): " + ", ".join(f"{k} {v}" for k, v in report["missing"].items()) + ".", "",
             "## Response time, ms per record (p50 / p95 / mean)", "",
             "| Records | n | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models),
             f"| all | {report['n_common']} | " + " | ".join(_lat_cell(report["latency_ms"][m]) for m in models) + " |", ""]
    for source, rows in report["benchmarks"].items():
        pairwise = "n_pairs" in next(iter(rows["all"].values()))
        keys = ("consistent_accuracy", "accuracy", "picks_first") if pairwise else ("balanced_accuracy", "auroc", "accuracy")
        lines += [f"## {source}", "", "| Slice | n | " + " | ".join(f"{m} {k}" for k in keys for m in models) + " |",
                  "|---|---|" + "---|" * (len(keys) * len(models))]
        order = ["all", "fits512"] + sorted(k for k in rows if k not in ("all", "fits512"))
        for name in (k for k in order if k in rows):
            first = next(iter(rows[name].values()))
            n = first["n_pairs"] if pairwise else first["n"]
            lines.append(f"| {name} | {n} | " + " | ".join(_cell(rows[name][m], k) for k in keys for m in models) + " |")
        lines += ["", "Response time, ms per record (p50 / p95 / mean):", "",
                  "| Slice | n | " + " | ".join(models) + " |", "|---|---|" + "---|" * len(models)]
        for name in (k for k in order if k in rows):
            first = next(iter(rows[name].values()))
            n = first["n_pairs"] if pairwise else first["n"]
            lines.append(f"| {name} | {n} | " + " | ".join(_lat_cell(rows[name][m].get("latency_ms")) for m in models) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
