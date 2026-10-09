"""Figures for BASELINES.md (todo Phase 2): PNG + interactive plotly twin per chart, in docs/baselines/figures/.

    uv run python scripts/make_baselines_figures.py

Inputs (git-ignored, produced by the Phase 2 runs; see BASELINES.md section 10):
- data/baselines/reports/{nli,laya}/report.json  (scripts/eval.py)
- data/baselines/preds/{nli,laya}.jsonl          (scripts/predict_baseline.py)
- data/mix/stage_a/test_{in,ood}.jsonl           (targets and number of options)

Charts:
1. reliability diagrams and 2. risk–coverage curves: the same charts as ``make_report.py compare``;
3. accuracy per source, NLI vs Laya, sorted by the gap (BASELINES.md section 3);
4. accuracy vs mean top probability per number of options K (section 4: Laya's K-bucket temperatures).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jevlite.reports.compare import load_runs, parse_run, reliability_chart, risk_coverage_chart
from jevlite.reports.style import (
    AXIS,
    GRID,
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

MODELS = ["nli", "laya"]  # slot order = colours: NLI blue, Laya orange (as in the compare report)
K_BUCKETS = [("2", 2, 2), ("3–5", 3, 5), ("6–10", 6, 10), ("11+", 11, 10**6)]


def source_rows(reports: dict[str, dict]) -> list[dict]:
    """Accuracy of both models per (source, primitive), sorted by NLI − Laya."""
    acc: dict[tuple[str, str], dict] = {}
    for model, report in reports.items():
        for row in report["rows"]:
            if set(row["slice"]) == {"source"}:
                key = (row["slice"]["source"], row["type"])
                acc.setdefault(key, {"source": key[0], "type": key[1], "n": row["n"]})[model] = row["accuracy"]
    rows = [r for r in acc.values() if all(m in r for m in MODELS)]
    return sorted(rows, key=lambda r: r["nli"] - r["laya"])


def k_bucket_rows(preds: dict[str, Path], tests: list[Path]) -> list[dict]:
    """Choice records only: accuracy and mean top probability per K bucket and model."""
    targets: dict[str, tuple[int, int]] = {}
    for path in tests:
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                if r["type"] == "choice":
                    targets[r["id"]] = (int(r["target"]), len(r["candidates"]))
    out = []
    for model, path in preds.items():
        stats = {name: [0, 0.0, 0] for name, _, _ in K_BUCKETS}  # n, sum top prob, n correct
        with path.open() as f:
            for line in f:
                p = json.loads(line)
                if p["id"] not in targets or not p.get("probs"):
                    continue
                target, k = targets[p["id"]]
                probs = p["probs"]
                top = max(range(len(probs)), key=probs.__getitem__)
                name = next(b for b, lo, hi in K_BUCKETS if lo <= k <= hi)
                s = stats[name]
                s[0] += 1
                s[1] += probs[top]
                s[2] += top == target
        for name, (n, conf, correct) in stats.items():
            if n:
                out.append({"model": model, "k": name, "n": n, "accuracy": correct / n, "confidence": conf / n})
    return out


def _dumbbell_axes(ax, labels: list[str]):
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)


def source_chart(rows: list[dict], colors: dict[str, str], out: Path) -> str:
    plt = setup_matplotlib()
    import plotly.graph_objects as go

    labels = [f"{r['source']} · {r['type']}" for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 0.32 * len(rows) + 1.6))
    for i, r in enumerate(rows):
        ax.plot([r["nli"], r["laya"]], [i, i], color=AXIS, linewidth=2, zorder=1)
    for model in MODELS:
        ax.scatter([r[model] for r in rows], range(len(rows)), s=64, color=colors[model], edgecolor=SURFACE, linewidth=2, zorder=2, label=model)
    _dumbbell_axes(ax, labels)
    ax.axhline(sum(r["nli"] < r["laya"] for r in rows) - 0.5, color=GRID, linewidth=1, linestyle="--", zorder=0)
    ax.set_xlabel("accuracy")
    ax.set_title("Accuracy by source: NLI wins above the dashed line, Laya below", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.8 / (0.32 * len(rows) + 1.6)), ncol=2)

    hfig = go.Figure()
    hfig.add_trace(go.Scatter(
        x=[v for r in rows for v in (r["nli"], r["laya"], None)], y=[v for lab in labels for v in (lab, lab, None)],
        mode="lines", line={"color": AXIS, "width": 2}, hoverinfo="skip", showlegend=False,
    ))
    for model in MODELS:
        hfig.add_trace(go.Scatter(
            x=[r[model] for r in rows], y=labels, mode="markers", name=model,
            marker={"size": 11, "color": colors[model], "line": {"color": SURFACE, "width": 2}},
            customdata=[[r["n"], r["nli"] - r["laya"]] for r in rows],
            hovertemplate=f"<b>%{{y}}</b><br>{model}: %{{x:.3f}}<br>NLI − Laya: %{{customdata[1]:+.3f}}<br>n = %{{customdata[0]:,}}<extra></extra>",
        ))
    hfig.update_layout(**plotly_layout("Accuracy by source, NLI vs Laya", height=26 * len(rows) + 160))
    hfig.update_xaxes(**plotly_axis(range=[0, 1], title="accuracy"))
    hfig.update_yaxes(**plotly_axis(showgrid=False))
    png, html = save_figure(fig, out, "accuracy_by_source", hfig)
    return md_figure("Accuracy by source, NLI vs Laya", png, html)


def k_chart(rows: list[dict], colors: dict[str, str], out: Path) -> str:
    plt = setup_matplotlib()
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    names = [b for b, _, _ in K_BUCKETS]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.0), sharey=True)
    hfig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=MODELS, horizontal_spacing=0.08)
    for col, (ax, model) in enumerate(zip(axes, MODELS), start=1):
        by_k = {r["k"]: r for r in rows if r["model"] == model}
        ks = [k for k in names if k in by_k]
        labels = [f"K = {k}  (n = {by_k[k]['n']:,})" for k in ks]
        c = colors[model]
        for i, k in enumerate(ks):
            r = by_k[k]
            ax.plot([r["accuracy"], r["confidence"]], [i, i], color=c, linewidth=2, alpha=0.45, zorder=1)
            gap = r["confidence"] - r["accuracy"]
            ax.annotate(f"{gap:+.2f}", (max(r["accuracy"], r["confidence"]), i), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=9, color=TEXT if abs(gap) >= 0.1 else MUTED)
        ax.scatter([by_k[k]["accuracy"] for k in ks], range(len(ks)), s=64, color=c, edgecolor=SURFACE, linewidth=2, zorder=2, label="accuracy")
        ax.scatter([by_k[k]["confidence"] for k in ks], range(len(ks)), s=64, facecolor=SURFACE, edgecolor=c, linewidth=2, zorder=2, label="mean top probability")
        _dumbbell_axes(ax, labels)
        ax.set_title(model, loc="left")
        ax.set_xlabel("share")

        for measure, symbol in (("accuracy", "circle"), ("confidence", "circle-open")):
            hfig.add_trace(go.Scatter(
                x=[by_k[k][measure] for k in ks], y=[f"K = {k}" for k in ks], mode="markers",
                name=f"{model}: {'accuracy' if measure == 'accuracy' else 'mean top probability'}",
                marker={"size": 11, "color": c, "symbol": symbol, "line": {"width": 2, "color": c}},
                customdata=[[by_k[k]["n"], by_k[k]["confidence"] - by_k[k]["accuracy"]] for k in ks],
                hovertemplate=f"<b>{model}, %{{y}}</b><br>{measure}: %{{x:.3f}}<br>overconfidence: %{{customdata[1]:+.3f}}<br>n = %{{customdata[0]:,}}<extra></extra>",
            ), row=1, col=col)
    from matplotlib.lines import Line2D

    fig.suptitle("Choice: accuracy vs claimed confidence by number of options K", x=0.01, y=1.1, ha="left", fontweight="bold", fontsize=11, color=TEXT)
    fig.text(0.01, 0.98, "labels: mean top probability − accuracy (+ = overconfident)", ha="left", fontsize=9, color=TEXT_SECONDARY)
    handles = [
        Line2D([], [], marker="o", linestyle="", markersize=8, color=TEXT_SECONDARY, label="accuracy"),
        Line2D([], [], marker="o", linestyle="", markersize=8, markerfacecolor=SURFACE, markeredgecolor=TEXT_SECONDARY, markeredgewidth=2, label="mean top probability"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2)
    hfig.update_layout(**plotly_layout("Choice: accuracy vs mean top probability by number of options K", height=380))
    hfig.update_xaxes(**plotly_axis(range=[0, 1.05], title="share"))
    hfig.update_yaxes(**plotly_axis(showgrid=False, categoryorder="array", categoryarray=[f"K = {k}" for k in names]))
    png, html = save_figure(fig, out, "calibration_by_k", hfig)
    return md_figure("Accuracy vs confidence by K", png, html)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baselines", default="data/baselines")
    ap.add_argument("--tests", nargs="+", default=["data/mix/stage_a/test_in.jsonl", "data/mix/stage_a/test_ood.jsonl"])
    ap.add_argument("--out", default="docs/baselines")
    args = ap.parse_args(argv)

    base, out = Path(args.baselines), Path(args.out)
    colors = series_colors(MODELS)
    runs = load_runs([parse_run(f"{m}={base / 'reports' / m / 'report.json'}") for m in MODELS])
    reports = {run.name: run.reports[0] for run in runs}
    prims = ["choice", "score", "bool"]

    snippets = {
        "reliability": reliability_chart(runs, prims, colors, out),
        "risk_coverage": risk_coverage_chart(runs, prims, colors, out),
        "accuracy_by_source": source_chart(source_rows(reports), colors, out),
        "calibration_by_k": k_chart(k_bucket_rows({m: base / "preds" / f"{m}.jsonl" for m in MODELS}, [Path(t) for t in args.tests]), colors, out),
    }
    for name, snippet in snippets.items():
        print(f"--- {name}\n{snippet}")


if __name__ == "__main__":
    main()
