"""Charts for ``docs/overview`` (Stage A → Stage B on eval v2) from two reports of
``scripts/judge_bench_report.py`` (needs the ``reports`` extra). PNG + interactive plotly twin per chart.

    uv run --extra reports python scripts/make_overview_figures.py \\
        --heldout data/judge_bench/overview/heldout/report.json \\
        --in-distribution data/judge_bench/overview/in_distribution/report.json --out docs/overview

The reports must hold the runs of ``MODELS`` (``stage_b_8192_cal`` = Stage B 8192 with both calibration
layers, ``scripts/calibrate.py apply --layers model,judge``). The chart style is the one of
``make_judge_bench_figures.py`` (its ``dot_chart`` is reused).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from make_judge_bench_figures import CJB, FAITH, HALU, RAG, REFERENCES, dot_chart

from jevlite.reports.style import TEXT, series_colors

MODELS = ["stage_b_8192_cal", "stage_b_8192", "stage_a_8192", "nli", "laya"]
LABELS = {"stage_b_8192_cal": "Stage B, calibrated", "stage_b_8192": "Stage B", "stage_a_8192": "Stage A",
          "nli": "NLI (DeBERTa-v3)", "laya": "Laya"}


def headline_rows(held: dict, ind: dict) -> list[dict]:
    def row(label, report, bench, slice_, metric, ref=None):
        cells = report["benchmarks"][bench][slice_]
        first = cells[MODELS[0]]
        return {"label": label, "n": first.get("n_pairs", first.get("n")), "ref": ref, **{m: cells[m][metric] for m in MODELS}}

    h = held
    return [
        row("CJB · consistent accuracy, all", h, CJB, "all", "consistent_accuracy", REFERENCES[(CJB, "all", "consistent_accuracy")]),
        row("CJB · refusal, unanswerable", h, CJB, "cjb/refusal_unanswerable", "consistent_accuracy"),
        row("CJB · faithfulness QA", h, CJB, "cjb/faithfulness_qa", "consistent_accuracy",
            REFERENCES[(CJB, "cjb/faithfulness_qa", "consistent_accuracy")]),
        row("FaithBench · balanced accuracy", h, FAITH, "all", "balanced_accuracy", REFERENCES[(FAITH, "all", "balanced_accuracy")]),
        row("HaluBench · balanced accuracy", h, HALU, "all", "balanced_accuracy"),
        row("RAGBench held-out ×5 · balanced accuracy", h, RAG, "all", "balanced_accuracy"),
        row("RAGBench held-out ×5 · AUROC", h, RAG, "all", "auroc"),
        row("RAGBench trained ×7 · balanced accuracy", ind, RAG, "all", "balanced_accuracy"),
        row("RAGBench trained ×7 · AUROC", ind, RAG, "all", "auroc"),
    ]


def rate_rows(held: dict, ind: dict) -> list[dict]:
    series = ["stage_a_8192", "stage_b_8192", "stage_b_8192_cal"]
    rows = []
    for report, tag in ((held, "held-out"), (ind, "trained")):
        b = report["benchmarks"]
        benches = ((RAG, "RAGBench"), (HALU, "HaluBench"), (FAITH, "FaithBench")) if tag == "held-out" else ((RAG, "RAGBench"),)
        for bench, name in benches:
            for s in sorted(k for k in b[bench] if "/" in k and k.split("/")[0] in (RAG, HALU)):
                cells = b[bench][s]
                rows.append({"label": f"{name} · {s.split('/', 1)[1]} ({tag})", "n": cells[MODELS[0]]["n"],
                             "labels": cells[MODELS[0]]["positive_rate"], **{m: cells[m]["predicted_positive_rate"] for m in series}})
            if bench == FAITH:
                cells = b[bench]["all"]
                rows.append({"label": "FaithBench (held-out)", "n": cells[MODELS[0]]["n"], "labels": cells[MODELS[0]]["positive_rate"],
                             **{m: cells[m]["predicted_positive_rate"] for m in series}})
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--in-distribution", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    held = json.loads(Path(args.heldout).read_text())
    ind = json.loads(Path(args.in_distribution).read_text())
    out = Path(args.out)
    colors = series_colors(MODELS)
    rate_series = ["labels", "stage_a_8192", "stage_b_8192", "stage_b_8192_cal"]
    blocks = [
        dot_chart(headline_rows(held, ind), MODELS, colors, LABELS, out, "stage_b_headline",
                  "Judge benchmarks, eval v2: Stage A vs Stage B",
                  "8192 tokens; grey ticks: published numbers from the papers (orientation only)",
                  "metric value", refs=True),
        dot_chart(rate_rows(held, ind), rate_series, {"labels": TEXT, **colors},
                  {"labels": "share of 'supported' labels", **{m: f"{LABELS[m].replace(',', '')} says 'supported'" for m in rate_series[1:]}},
                  out, "stage_b_positive_rate", "How often each model says 'supported', against the labels",
                  "Stage A said 'no' too often, Stage B says 'yes' too often; calibration moves the threshold",
                  "share of records"),
    ]
    print("\n".join(blocks))


if __name__ == "__main__":
    main()
