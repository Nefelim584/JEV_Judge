"""Fit post-hoc calibration on saved predictions, or apply it to a predictions file (``jevlite.calibration``).

    # fit: model layer on the calib splits of the mixes, judge layer on held-out RAG answers
    uv run python scripts/calibrate.py fit \\
        --data data/mix/stage_a/calib.jsonl data/calib/stage_b_calib.jsonl \\
        --pred data/calib/preds/stage_a_calib.jsonl data/calib/preds/stage_b_calib.jsonl \\
        --judge-data data/calib/ragbench_calib.jsonl --judge-pred data/calib/preds/ragbench_calib.jsonl \\
        --out data/calib/calibration.json

    # apply: model layer only (Stage A tests), or both layers (judge benchmarks: every Bool is a faithfulness claim)
    uv run python scripts/calibrate.py apply --calibration data/calib/calibration.json --layers model \\
        --data data/mix/stage_a/test_in.jsonl data/mix/stage_a/test_ood.jsonl \\
        --pred data/runs/stage_b-full-s42-r1/preds.jsonl --out data/runs/stage_b-full-s42-r1/preds.cal.jsonl

``fit`` prints accuracy / NLL / ECE before and after on the data it was fitted on (a sanity check,
not an evaluation: evaluate the applied predictions with ``eval.py`` / ``judge_bench_report.py``).
``apply`` reads the records only for their type (Score and Choice get different temperatures).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevlite.calibration import LAYERS, apply, fit_judge_layer, fit_model_layer
from jevlite.data.unified import Prediction, Record, load_predictions, load_records, read_jsonl, write_jsonl
from jevlite.evaluation import primitive_metrics
from jevlite.judge_bench import balanced_accuracy


def _load(data: list[str], pred: list[str]) -> tuple[list[Record], dict[str, Prediction]]:
    if len(data) != len(pred):
        raise SystemExit("--data and --pred take one predictions file per records file")
    records, preds = [], {}
    for d, p in zip(data, pred):
        file_preds = load_predictions(p)
        records += [r for r in load_records(d) if r.id in file_preds]
        preds.update(file_preds)
    return records, preds


def _check(title: str, records: list[Record], before: dict[str, Prediction], cal: dict, layers: tuple[str, ...]) -> None:
    print(f"\n{title}\n| primitive | n | accuracy | balanced acc. | NLL | ECE |\n|---|---|---|---|---|---|")
    for prim in ("bool", "choice", "score"):
        rs = [r for r in records if r.type == prim]
        if not rs:
            continue
        b = primitive_metrics(prim, rs, [before[r.id] for r in rs])
        a = primitive_metrics(prim, rs, [apply(prim, before[r.id], cal, layers) for r in rs])
        ba = "—"
        if prim == "bool":
            y = [r.target >= 0.5 for r in rs]
            ba = " → ".join(f"{balanced_accuracy([p.prob >= 0.5 for p in ps], y):.3f}"
                            for ps in ([before[r.id] for r in rs], [apply(prim, before[r.id], cal, layers) for r in rs]))
        print(f"| {prim} | {len(rs)} | {b['accuracy']:.3f} → {a['accuracy']:.3f} | {ba} | {b['nll']:.3f} → {a['nll']:.3f} "
              f"| {b['ece']:.3f} → {a['ece']:.3f} |")


def fit(args) -> None:
    cal: dict = {}
    if args.data:
        records, preds = _load(args.data, args.pred)
        cal["model"] = fit_model_layer(records, preds)
        cal["model"]["fitted_on"] = args.data
    if args.judge_data:
        jrecords, jpreds = _load(args.judge_data, args.judge_pred)
        cal["judge"] = fit_judge_layer(jrecords, jpreds, cal.get("model"), balanced=not args.judge_keep_prior)
        cal["judge"]["fitted_on"] = args.judge_data
    if not cal:
        raise SystemExit("nothing to fit: give --data/--pred and/or --judge-data/--judge-pred")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(cal, indent=2))
    print(json.dumps(cal, indent=2))
    if args.data:
        _check("model layer, on its fit data", records, preds, cal, ("model",))
    if args.judge_data:
        _check("model + judge layers, on the judge fit data", jrecords, jpreds, cal, LAYERS)


def run_apply(args) -> None:
    cal = json.loads(Path(args.calibration).read_text())
    layers = tuple(s.strip() for s in args.layers.split(","))
    if unknown := set(layers) - set(LAYERS):
        raise SystemExit(f"unknown layers {sorted(unknown)}; choose from {LAYERS}")
    types = {r.id: r.type for d in args.data for r in load_records(d)}
    out = []
    for row in read_jsonl(args.pred):
        p = Prediction.model_validate(row)
        if p.id not in types:
            raise SystemExit(f"{p.id}: no record in --data (needed to tell Score from Choice)")
        out.append(apply(types[p.id], p, cal, layers))
    n = write_jsonl(args.out, out)
    print(f"{n} predictions, layers {','.join(layers)} → {args.out}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--data", nargs="*", default=[], help="records for the model layer (calib splits)")
    f.add_argument("--pred", nargs="*", default=[], help="their predictions, one file per --data file")
    f.add_argument("--judge-data", nargs="*", default=[], help="records for the judge layer (held-out RAG answers)")
    f.add_argument("--judge-pred", nargs="*", default=[])
    f.add_argument("--judge-keep-prior", action="store_true",
                   help="fit the judge layer on the plain NLL (learns the fit set's share of 'supported'); default: class-balanced")
    f.add_argument("--out", required=True)
    a = sub.add_parser("apply")
    a.add_argument("--calibration", required=True)
    a.add_argument("--pred", required=True)
    a.add_argument("--data", nargs="+", required=True, help="the records of --pred (their types: Score and Choice differ)")
    a.add_argument("--layers", default="model", help=f"comma-separated, of {LAYERS}")
    a.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "fit":
        fit(args)
    else:
        run_apply(args)


if __name__ == "__main__":
    main()
