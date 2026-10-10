"""Figures for BASELINES.md (todo Phase 2): PNG + interactive plotly twin per chart, in docs/baselines/figures/.

    uv run python scripts/make_baselines_figures.py

Inputs (git-ignored, produced by the Phase 2 runs; see BASELINES.md section 10):
- data/baselines/reports/{nli,laya}/report.json  (scripts/eval.py)
- data/baselines/preds/{nli,laya}.jsonl          (scripts/predict_baseline.py)
- data/mix/stage_a/test_{in,ood}.jsonl           (targets and number of options)

Charts:
1. reliability diagrams and 2. risk–coverage curves: the same charts as ``make_report.py compare``;
3. accuracy per source, NLI vs Laya, sorted by the gap (BASELINES.md section 3);
4. accuracy vs mean top probability per number of options K (section 4: Laya's K-bucket temperatures);
5. Bool accuracy on the claim-vs-text groups, HoVer by number of hops (section 3.1);
6. Bool recall per label on fact-checking and answerability sources (sections 3.1, 3.2);
7. Score accuracy per criterion (section 3.4);
8. confidence on real vs nonsense states, from data/baselines/probes/{nli,laya}.json (section 6.3).
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
CLAIM_VS_TEXT = {"multi_nli", "snli", "wanli", "vitaminc", "contract_nli", "fever", "hover"}
# (label, sources): the Bool sources of the per-label recall chart, positive class = target 1
RECALL_SOURCES = [
    ("FEVER", {"fever"}), ("HoVer", {"hover"}), ("ContractNLI", {"contract_nli"}),
    ("Natural Questions", {"nq"}), ("SQuAD 2.0", {"squad_v2"}), ("ClapNQ", {"clapnq"}),
]
NONSENSE_STATES = [("real", "real text"), ("shuffled_words", "shuffled words"), ("random_chars", "random characters")]


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


def bool_records(preds: dict[str, Path], tests: list[Path]) -> list[dict]:
    """Bool records joined with both models' P(true): source, target, hops, and ``{model: prob}``."""
    recs: dict[str, dict] = {}
    for path in tests:
        with path.open() as f:
            for line in f:
                r = json.loads(line)
                if r["type"] == "bool":
                    recs[r["id"]] = {"source": r["source"], "target": float(r["target"]), "hops": r.get("num_hops")}
    for model, path in preds.items():
        with path.open() as f:
            for line in f:
                p = json.loads(line)
                if p["id"] in recs and p.get("prob") is not None:
                    recs[p["id"]][model] = p["prob"]
    return [r for r in recs.values() if all(m in r for m in MODELS)]


def _accuracy_row(label: str, recs: list[dict]) -> dict:
    row = {"label": label, "n": len(recs)}
    for model in MODELS:
        row[model] = sum((r[model] >= 0.5) == (r["target"] >= 0.5) for r in recs) / len(recs)
    return row


def faithfulness_rows(recs: list[dict]) -> list[dict]:
    """Bool accuracy on the claim-vs-text groups of BASELINES.md section 3.1, top to bottom."""
    claim = [r for r in recs if r["source"] in CLAIM_VS_TEXT]
    hover = [r for r in recs if r["source"] == "hover"]
    rows = [
        _accuracy_row("claim vs text, all", claim),
        _accuracy_row("ContractNLI (held out)", [r for r in recs if r["source"] == "contract_nli"]),
        _accuracy_row("FEVER", [r for r in recs if r["source"] == "fever"]),
        _accuracy_row("HoVer, all", hover),
    ]
    for hops in sorted({r["hops"] for r in hover if r["hops"]}):
        rows.append(_accuracy_row(f"HoVer, {hops} hops", [r for r in hover if r["hops"] == hops]))
    rows.append(_accuracy_row("other Bool sources", [r for r in recs if r["source"] not in CLAIM_VS_TEXT]))
    return rows


def recall_rows(recs: list[dict]) -> dict[str, list[dict]]:
    """Recall of each label per source: {"positive": rows, "negative": rows}."""
    out: dict[str, list[dict]] = {"positive": [], "negative": []}
    for label, sources in RECALL_SOURCES:
        sel = [r for r in recs if r["source"] in sources]
        for side, want in (("positive", True), ("negative", False)):
            part = [r for r in sel if (r["target"] >= 0.5) == want]
            if part:
                out[side].append(_accuracy_row(label, part))
    return out


def score_rows(reports: dict[str, dict]) -> list[dict]:
    """Score accuracy per criterion (HelpSteer2 attributes, HelpSteer3 preference), sorted by Laya."""
    acc: dict[str, dict] = {}
    for model, report in reports.items():
        for row in report["rows"]:
            if set(row["slice"]) == {"criterion"} and row["type"] == "score":
                name = row["slice"]["criterion"]
                acc.setdefault(name, {"label": name, "n": row["n"]})[model] = row["accuracy"]
    rows = [r for r in acc.values() if all(m in r for m in MODELS)]
    return sorted(rows, key=lambda r: r["laya"])


def nonsense_rows(probes: dict[str, dict]) -> list[dict]:
    rows = []
    for key, label in NONSENSE_STATES:
        row = {"label": label, "n": probes[MODELS[0]]["nonsense_confidence"]["n"]}
        for model in MODELS:
            row[model] = probes[model]["nonsense_confidence"][key]
        rows.append(row)
    return rows[::-1]  # matplotlib draws bottom-up: "real" ends on top


def dumbbell_chart(
    panels: list[tuple[str | None, list[dict]]], colors: dict[str, str], out: Path, name: str,
    title: str, subtitle: str, xlabel: str = "accuracy", xlim: tuple[float, float] = (0, 1),
) -> str:
    """NLI vs Laya dot pairs per row; one or more panels side by side with shared row labels."""
    plt = setup_matplotlib()
    import plotly.graph_objects as go
    from matplotlib.lines import Line2D
    from plotly.subplots import make_subplots

    n_rows = max(len(rows) for _, rows in panels)
    height = 0.34 * n_rows + 1.5
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels) + 2.4, height), sharey=True, squeeze=False)
    hfig = make_subplots(rows=1, cols=len(panels), shared_yaxes=True, horizontal_spacing=0.06,
                         subplot_titles=[t or "" for t, _ in panels])
    for col, (ax, (panel_title, rows)) in enumerate(zip(axes[0], panels), start=1):
        # n differs per panel, and shared row labels show only one: multi-panel charts keep n in the hover
        labels = [r["label"] if len(panels) > 1 else f"{r['label']}  (n = {r['n']:,})" for r in rows]
        for i, r in enumerate(rows):
            ax.plot([r["nli"], r["laya"]], [i, i], color=AXIS, linewidth=2, zorder=1)
        for model in MODELS:
            ax.scatter([r[model] for r in rows], range(len(rows)), s=64, color=colors[model], edgecolor=SURFACE, linewidth=2, zorder=2)
        _dumbbell_axes(ax, labels)
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        if panel_title:
            ax.set_title(panel_title, loc="left")

        hlabels = [r["label"] for r in rows]
        hfig.add_trace(go.Scatter(
            x=[v for r in rows for v in (r["nli"], r["laya"], None)], y=[v for lab in hlabels for v in (lab, lab, None)],
            mode="lines", line={"color": AXIS, "width": 2}, hoverinfo="skip", showlegend=False,
        ), row=1, col=col)
        for model in MODELS:
            hfig.add_trace(go.Scatter(
                x=[r[model] for r in rows], y=hlabels, mode="markers", name=model, showlegend=col == 1,
                marker={"size": 11, "color": colors[model], "line": {"color": SURFACE, "width": 2}},
                customdata=[[r["n"], r["nli"] - r["laya"]] for r in rows],
                hovertemplate=f"<b>%{{y}}</b><br>{model}: %{{x:.3f}}<br>NLI − Laya: %{{customdata[1]:+.3f}}<br>n = %{{customdata[0]:,}}<extra></extra>",
            ), row=1, col=col)

    # fixed margins in inches, so title, subtitle and legend sit the same way at any number of rows
    top = 0.95 if any(t for t, _ in panels) else 0.7
    fig.subplots_adjust(top=1 - top / height, bottom=0.6 / height)
    fig.suptitle(title, x=0.01, y=1 - 0.05 / height, va="top", ha="left", fontweight="bold", fontsize=11, color=TEXT)
    fig.text(0.01, 1 - 0.3 / height, subtitle, va="top", ha="left", fontsize=9, color=TEXT_SECONDARY)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, color=colors[m], label=m) for m in MODELS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(MODELS))
    hfig.update_layout(**plotly_layout(title, height=30 * n_rows + 170))
    hfig.update_xaxes(**plotly_axis(range=list(xlim), title=xlabel))
    hfig.update_yaxes(**plotly_axis(showgrid=False, autorange="reversed"))
    png, html = save_figure(fig, out, name, hfig)
    return md_figure(title, png, html)


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
    tests = [Path(t) for t in args.tests]
    preds = {m: base / "preds" / f"{m}.jsonl" for m in MODELS}
    recs = bool_records(preds, tests)
    recall = recall_rows(recs)
    probes = {m: json.loads((base / "probes" / f"{m}.json").read_text()) for m in MODELS}

    snippets = {
        "reliability": reliability_chart(runs, prims, colors, out),
        "risk_coverage": risk_coverage_chart(runs, prims, colors, out),
        "accuracy_by_source": source_chart(source_rows(reports), colors, out),
        "calibration_by_k": k_chart(k_bucket_rows(preds, tests), colors, out),
        "faithfulness": dumbbell_chart(
            [(None, faithfulness_rows(recs)[::-1])], colors, out, "faithfulness",
            "Claim vs text (Bool): NLI leads on every group, both drop on multi-hop HoVer",
            "accuracy per group; HoVer claims need 2–4 chunks at once; bottom row (non-claim Bool tasks) for reference",
        ),
        "recall_by_label": dumbbell_chart(
            [("supported / answerable", recall["positive"][::-1]), ("not supported / unanswerable", recall["negative"][::-1])],
            colors, out, "recall_by_label",
            "Recall per label: Laya misses the negative class, and on HoVer both do",
            "share of records of each label answered correctly (threshold 0.5)", xlabel="recall",
        ),
        "score_by_criterion": dumbbell_chart(
            [(None, score_rows(reports))], colors, out, "score_by_criterion",
            "Score accuracy per criterion: both weak, Laya ahead except on preference",
            "exact-level accuracy on 5- and 7-level rubrics (HelpSteer2 attributes, HelpSteer3 preference)",
        ),
        "nonsense_confidence": dumbbell_chart(
            [(None, nonsense_rows(probes))], colors, out, "nonsense_confidence",
            "Confidence barely reacts to nonsense states; Laya's rises on random characters",
            "mean heuristic confidence (1 − H(p)/log K), 200 test_in items per state", xlabel="mean confidence",
        ),
    }
    for name, snippet in snippets.items():
        print(f"--- {name}\n{snippet}")


if __name__ == "__main__":
    main()
