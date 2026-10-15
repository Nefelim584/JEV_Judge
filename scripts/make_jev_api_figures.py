"""Figures for STAGE_A_API.md: paired Δ accuracy of JEV against jev-lite with 95% paired-bootstrap CIs,
per split × primitive and per source × primitive (from ``scripts/compare_paired.py``). PNG + interactive
plotly twin per chart, in <out>/figures/. The other charts of STAGE_A_API.md come from
``make_baselines_figures.py`` (see the Reproduce section there).

    uv run --extra reports python scripts/make_jev_api_figures.py \\
        --paired-split data/jev_api/paired/paired_jev_vs_jevlite.json \\
        --paired-source data/jev_api/paired_all/paired_jev_vs_jevlite.json --out docs/stage_a_api

The split chart reads the balanced sample (700 per cell); the source chart can read a larger one (sample + top-up).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevlite.reports.style import (
    AXIS,
    MUTED,
    SURFACE,
    TEXT,
    TEXT_SECONDARY,
    md_figure,
    plotly_axis,
    plotly_layout,
    save_figure,
    series_colors,
    setup_matplotlib,
)

# the slot order of make_baselines_figures.py with --model jevlite … --model jev, so colours match its charts
COLORS = series_colors(["nli", "laya", "jevlite", "jev"])
SPLITS = ["test_in", "test_ood_template", "test_ood_source"]


def _rows(paired: dict, slicing: str, min_n: int) -> list[dict]:
    rows = [r for r in paired["rows"] if r["slicing"] == slicing and r["n"] >= min_n]
    for r in rows:
        r["label"] = f"{r['cell']} · {r['primitive']}"
    if slicing == "split":
        return sorted(rows, key=lambda r: (SPLITS.index(r["cell"]), r["primitive"]), reverse=True)
    return sorted(rows, key=lambda r: r["d_acc"])


def _color(r: dict) -> str:
    return {"win": COLORS["jev"], "loss": COLORS["jevlite"]}.get(r["verdict"], MUTED)


def forest_chart(rows: list[dict], model: str, ref: str, out: Path, name: str, title: str, subtitle: str) -> str:
    plt = setup_matplotlib()
    import plotly.graph_objects as go
    from matplotlib.lines import Line2D

    labels = [f"{r['label']}  (n = {r['n']:,})" for r in rows]
    height = 0.3 * len(rows) + 1.6
    fig, ax = plt.subplots(figsize=(7.6, height))
    ax.axvline(0, color=AXIS, linewidth=1, zorder=0)
    for i, r in enumerate(rows):
        lo, hi = (100 * v for v in r["d_acc_ci"])
        ax.plot([lo, hi], [i, i], color=_color(r), linewidth=2, zorder=1)
    ax.scatter([100 * r["d_acc"] for r in rows], range(len(rows)), s=64, color=[_color(r) for r in rows],
               edgecolor=SURFACE, linewidth=2, zorder=2)
    ax.set_yticks(range(len(rows)), labels)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(f"Δ accuracy, {model} − {ref} (points)")
    fig.subplots_adjust(top=1 - 0.75 / height, bottom=0.75 / height)
    fig.suptitle(title, x=0.01, y=1 - 0.05 / height, va="top", ha="left", fontweight="bold", fontsize=11, color=TEXT)
    fig.text(0.01, 1 - 0.3 / height, subtitle, va="top", ha="left", fontsize=9, color=TEXT_SECONDARY)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, color=c, label=lab)
               for c, lab in ((COLORS["jev"], f"{model} wins"), (MUTED, "tie"), (COLORS["jevlite"], f"{ref} wins"))]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3)

    hlabels = [r["label"] for r in rows]
    hfig = go.Figure()
    hfig.add_vline(x=0, line={"color": AXIS, "width": 1})
    for verdict, lab in (("win", f"{model} wins"), ("tie", "tie"), ("loss", f"{ref} wins")):
        sub = [r for r in rows if r["verdict"] == verdict]
        if not sub:
            continue
        color = _color(sub[0])
        hfig.add_trace(go.Scatter(
            x=[v for r in sub for v in (100 * r["d_acc_ci"][0], 100 * r["d_acc_ci"][1], None)],
            y=[v for r in sub for v in (r["label"], r["label"], None)],
            mode="lines", line={"color": color, "width": 2}, hoverinfo="skip", showlegend=False,
        ))
        hfig.add_trace(go.Scatter(
            x=[100 * r["d_acc"] for r in sub], y=[r["label"] for r in sub], mode="markers", name=lab,
            marker={"size": 11, "color": color, "line": {"color": SURFACE, "width": 2}},
            customdata=[[r["n"], r["acc_model"], r["acc_ref"], 100 * r["d_acc_ci"][0], 100 * r["d_acc_ci"][1], r["p_holm"]]
                        for r in sub],
            hovertemplate=(f"<b>%{{y}}</b><br>{model} %{{customdata[1]:.3f}} vs {ref} %{{customdata[2]:.3f}}"
                           "<br>Δ %{x:+.1f} [%{customdata[3]:+.1f}, %{customdata[4]:+.1f}]"
                           "<br>p (Holm) %{customdata[5]:.2g}<br>n = %{customdata[0]:,}<extra></extra>"),
        ))
    hfig.update_layout(**plotly_layout(title, height=28 * len(rows) + 170))
    hfig.update_xaxes(**plotly_axis(title=f"Δ accuracy, {model} − {ref} (points)"))
    hfig.update_yaxes(**plotly_axis(showgrid=False, categoryorder="array", categoryarray=hlabels))
    png, html = save_figure(fig, out, name, hfig)
    return md_figure(title, png, html)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paired-split", default="data/jev_api/paired/paired_jev_vs_jevlite.json")
    ap.add_argument("--paired-source", default="data/jev_api/paired_all/paired_jev_vs_jevlite.json")
    ap.add_argument("--out", default="docs/stage_a_api")
    ap.add_argument("--min-n", type=int, default=40, help="per-source chart: cells with at least this many records")
    args = ap.parse_args(argv)

    paired = json.loads(Path(args.paired_split).read_text())
    by_source = json.loads(Path(args.paired_source).read_text())
    model, ref, out = paired["model"], paired["reference"], Path(args.out)
    if (by_source["model"], by_source["reference"]) != (model, ref):
        ap.error("--paired-split and --paired-source compare different models")
    note = "dot = Δ accuracy, line = 95% paired-bootstrap CI; win / loss = CI excludes 0 and Holm-adjusted McNemar p < 0.05"
    snippets = {
        "paired_by_split": forest_chart(
            _rows(paired, "split", 0), model, ref, out, "paired_by_split",
            f"{model} vs {ref}: far ahead on new sources, behind on Score, tied elsewhere", note),
        "paired_by_source": forest_chart(
            _rows(by_source, "source", args.min_n), model, ref, out, "paired_by_source",
            f"{model} vs {ref} by source (n ≥ {args.min_n})", note),
    }
    for name, snippet in snippets.items():
        print(f"--- {name}\n{snippet}")


if __name__ == "__main__":
    main()
