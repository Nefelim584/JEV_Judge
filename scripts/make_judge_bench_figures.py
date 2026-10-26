"""Charts for ``JUDGE_BENCH.md`` from the report of ``scripts/judge_bench_report.py`` (needs the
``reports`` extra). PNG + interactive plotly twin per chart, written to ``--out``/figures.

    uv run --extra reports python scripts/make_judge_bench_figures.py \\
        --report data/judge_bench/report/report.json --out docs/judge_bench

Run names are the report's (``jevlite_8192``, ``jevlite_512``, ``nli``, ``laya``); colours follow the
run in that fixed order. Published reference points (``REFERENCES``) are drawn as grey ticks: they come
from the benchmarks' papers, other prompts and label versions, so they are orientation, not a paired
comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevlite.reports.style import (
    AXIS, GRID, MUTED, SURFACE, TEXT, TEXT_SECONDARY, md_figure, plotly_axis, plotly_layout, save_figure,
    series_colors, setup_matplotlib,
)

MODELS = ["jevlite_8192", "jevlite_512", "nli", "laya"]
LABELS = {"jevlite_8192": "jev-lite, 8192 tokens", "jevlite_512": "jev-lite, 512 tokens", "nli": "NLI (DeBERTa-v3)", "laya": "Laya"}
CJB, RAG, HALU, FAITH = "contextual_judge_bench", "ragbench", "halubench", "faithbench"
# (benchmark, slice, metric) → (reference model, value) from the papers (JUDGE_BENCH.md section 5).
REFERENCES = {
    (CJB, "all", "consistent_accuracy"): ("GPT-4o", 0.512),
    (CJB, "cjb/faithfulness_qa", "consistent_accuracy"): ("GPT-4o", 0.704),
    (FAITH, "all", "balanced_accuracy"): ("GPT-4o", 0.563),
    (HALU, "all", "accuracy"): ("GPT-4o", 0.865),
    (RAG, "subset_mean", "auroc"): ("GPT-3.5", 0.56),
}


def headline_rows(report: dict) -> list[dict]:
    b = report["benchmarks"]

    def row(label, bench, slice_, metric, n_key="n"):
        cells = b[bench][slice_]
        first = next(iter(cells.values()))
        return {"label": label, "n": first.get(n_key, first.get("n_pairs")), "ref": REFERENCES.get((bench, slice_, metric)),
                **{m: cells[m][metric] for m in MODELS}}

    rag_subsets = [k for k in b[RAG] if k.startswith("ragbench/")]
    rag_mean = {"label": "RAGBench · AUROC, mean of 12 subsets", "n": b[RAG]["all"][MODELS[0]]["n"],
                "ref": REFERENCES[(RAG, "subset_mean", "auroc")],
                **{m: sum(b[RAG][s][m]["auroc"] for s in rag_subsets) / len(rag_subsets) for m in MODELS}}
    return [
        row("CJB · consistent accuracy, all splits", CJB, "all", "consistent_accuracy"),
        row("CJB · consistent accuracy, faithfulness QA", CJB, "cjb/faithfulness_qa", "consistent_accuracy"),
        row("FaithBench · balanced accuracy", FAITH, "all", "balanced_accuracy"),
        row("HaluBench · accuracy", HALU, "all", "accuracy"),
        row("HaluBench · AUROC", HALU, "all", "auroc"),
        row("RAGBench · balanced accuracy", RAG, "all", "balanced_accuracy"),
        rag_mean,
    ]


def length_rows(report: dict) -> list[dict]:
    """jev-lite at 512 vs 8192 per slice: CJB splits (consistent accuracy) and Bool sources (balanced accuracy)."""
    b = report["benchmarks"]
    rows = []
    for bench, metric, prefix in ((CJB, "consistent_accuracy", "cjb/"), (HALU, "balanced_accuracy", "halubench/"),
                                  (RAG, "balanced_accuracy", "ragbench/")):
        for s in sorted(k for k in b[bench] if k.startswith(prefix)):
            cells = b[bench][s]
            rows.append({"label": f"{bench.split('_')[0] if bench != CJB else 'CJB'} · {s.split('/', 1)[1]}",
                         "n": cells[MODELS[0]].get("n_pairs", cells[MODELS[0]].get("n")),
                         "jevlite_512": cells["jevlite_512"][metric], "jevlite_8192": cells["jevlite_8192"][metric]})
    cells = b[FAITH]["all"]
    rows.append({"label": "faithbench", "n": cells[MODELS[0]]["n"],
                 "jevlite_512": cells["jevlite_512"]["balanced_accuracy"], "jevlite_8192": cells["jevlite_8192"]["balanced_accuracy"]})
    for r in rows:
        r["label"] = r["label"].replace("halubench ·", "HaluBench ·").replace("ragbench ·", "RAGBench ·").replace("faithbench", "FaithBench")
    return sorted(rows, key=lambda r: r["jevlite_8192"] - r["jevlite_512"], reverse=True)


def rate_rows(report: dict) -> list[dict]:
    b = report["benchmarks"]
    rows = []
    for bench, slices in ((RAG, ["all"] + sorted(k for k in b[RAG] if k.startswith("ragbench/"))), (HALU, ["all"]), (FAITH, ["all"])):
        for s in slices:
            cells = b[bench][s]
            name = {RAG: "RAGBench", HALU: "HaluBench", FAITH: "FaithBench"}[bench]
            rows.append({"label": f"{name} · {s.split('/', 1)[1] if '/' in s else 'all'}", "n": cells[MODELS[0]]["n"],
                         "labels": cells[MODELS[0]]["positive_rate"],
                         **{m: cells[m]["predicted_positive_rate"] for m in ("jevlite_8192", "nli")}})
    return rows


def _axes(ax, labels):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels) - 0.4, -0.6)  # first row on top
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def dot_chart(rows, series: list[str], colors: dict[str, str], names: dict[str, str], out: Path, name: str,
              title: str, subtitle: str, xlabel: str, xlim=(0, 1), refs: bool = False) -> str:
    """One dot per series per row, joined over their range; optional grey reference tick per row."""
    plt = setup_matplotlib()
    import plotly.graph_objects as go
    from matplotlib.lines import Line2D

    labels = [r["label"] for r in rows]
    height = 0.36 * len(rows) + 1.7
    fig, ax = plt.subplots(figsize=(8.2, height))
    for i, r in enumerate(rows):
        vals = [r[s] for s in series]
        ax.plot([min(vals), max(vals)], [i, i], color=AXIS, linewidth=2, zorder=1)
    for s in series:
        ax.scatter([r[s] for r in rows], range(len(rows)), s=64, color=colors[s], edgecolor=SURFACE, linewidth=2, zorder=3)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, color=colors[s], label=names[s]) for s in series]
    if refs:
        for i, r in enumerate(rows):
            if r.get("ref"):
                who, v = r["ref"]
                ax.plot([v, v], [i - 0.28, i + 0.28], color=MUTED, linewidth=2, zorder=2)
                ax.annotate(who, (v, i - 0.3), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                            fontsize=8, color=TEXT_SECONDARY)
        handles.append(Line2D([], [], marker="|", linestyle="", markersize=10, markeredgewidth=2, color=MUTED,
                              label="published reference (paper)"))
    _axes(ax, labels)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    fig.subplots_adjust(top=1 - 0.75 / height, bottom=0.75 / height)
    fig.suptitle(title, x=0.01, y=1 - 0.05 / height, va="top", ha="left", fontweight="bold", fontsize=11, color=TEXT)
    fig.text(0.01, 1 - 0.32 / height, subtitle, va="top", ha="left", fontsize=9, color=TEXT_SECONDARY)
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(handles), fontsize=9)

    hfig = go.Figure()
    hfig.add_trace(go.Scatter(
        x=[v for r in rows for v in (min(r[s] for s in series), max(r[s] for s in series), None)],
        y=[v for lab in labels for v in (lab, lab, None)], mode="lines", line={"color": AXIS, "width": 2},
        hoverinfo="skip", showlegend=False,
    ))
    for s in series:
        hfig.add_trace(go.Scatter(
            x=[r[s] for r in rows], y=labels, mode="markers", name=names[s],
            marker={"size": 11, "color": colors[s], "line": {"color": SURFACE, "width": 2}},
            customdata=[[r["n"]] for r in rows],
            hovertemplate=f"<b>%{{y}}</b><br>{names[s]}: %{{x:.3f}}<br>n = %{{customdata[0]:,}}<extra></extra>",
        ))
    if refs:
        ref_rows = [r for r in rows if r.get("ref")]
        hfig.add_trace(go.Scatter(
            x=[r["ref"][1] for r in ref_rows], y=[r["label"] for r in ref_rows], mode="markers",
            name="published reference (paper)", marker={"symbol": "line-ns-open", "size": 16, "color": MUTED, "line": {"width": 2}},
            customdata=[[r["ref"][0]] for r in ref_rows],
            hovertemplate="<b>%{y}</b><br>%{customdata[0]} (paper): %{x:.3f}<extra></extra>",
        ))
    hfig.update_layout(**plotly_layout(title, height=30 * len(rows) + 170))
    hfig.update_xaxes(**plotly_axis(range=list(xlim), title=xlabel))
    hfig.update_yaxes(**plotly_axis(showgrid=False, autorange="reversed"))
    png, html = save_figure(fig, out, name, hfig)
    return md_figure(title, png, html)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    report = json.loads(Path(args.report).read_text())
    out = Path(args.out)
    colors = series_colors(MODELS)
    rates_colors = {"labels": TEXT, "jevlite_8192": colors["jevlite_8192"], "nli": colors["nli"]}
    blocks = [
        dot_chart(headline_rows(report), MODELS, colors, LABELS, out, "headline",
                  "Headline metric per benchmark, four runs",
                  "Grey ticks: published numbers from the papers (other prompts and label versions: orientation only)",
                  "metric value", refs=True),
        dot_chart(length_rows(report), ["jevlite_512", "jevlite_8192"], colors, LABELS, out, "context_length",
                  "jev-lite at 512 vs 8192 tokens, per split and source",
                  "Consistent accuracy for CJB, balanced accuracy elsewhere; sorted by the gain from the longer input",
                  "metric value"),
        dot_chart(rate_rows(report), ["labels", "jevlite_8192", "nli"], rates_colors,
                  {"labels": "share of 'supported' labels", "jevlite_8192": "jev-lite 8192 says 'supported'", "nli": "NLI says 'supported'"},
                  out, "positive_rate", "How often each model says 'supported', against the labels",
                  "At the 0.5 threshold: jev-lite says 'no' too often on RAGBench, NLI says 'yes' to almost everything",
                  "share of records"),
    ]
    print("\n".join(blocks))


if __name__ == "__main__":
    main()
