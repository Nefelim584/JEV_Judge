"""Comparison report: several ``scripts/eval.py`` reports side by side (baselines, variants, H0–H3).

Runs are given as ``name=path/to/report.json``:

- the same ``name`` repeated → seeds of one variant: the report shows the mean and the min–max range;
- ``name@x=path`` → a point of a learning curve at ``x`` (e.g. the number of training examples);
  the report adds metric-vs-x charts (TODO 2.13, H1 and H2).

The first run name is the reference for the Δ table.
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from .charts import BarPanel, Bars, Line, LinePanel, bar_panels, line_panels
from .style import series_colors

TYPES = ("choice", "score", "bool")
HIGHER_IS_BETTER = {"accuracy", "kappa", "spearman", "selective_acc@80", "selective_acc@90"}
LOWER_IS_BETTER = {"nll", "brier", "ece", "mce", "ece_top", "aurc", "eaurc", "mae", "mae_expected"}
TABLE_METRICS = {
    "choice": ["accuracy", "kappa", "nll", "brier", "ece", "aurc", "selective_acc@80"],
    "score": ["accuracy", "kappa", "spearman", "mae", "nll", "ece", "aurc"],
    "bool": ["accuracy", "kappa", "nll", "brier", "ece", "ece_top", "aurc", "selective_acc@80"],
}
NAMES = {"kappa": "κ", "selective_acc@80": "sel.acc@80", "selective_acc@90": "sel.acc@90", "ece_top": "ece (top)", "mae_expected": "mae(E)"}
CHART_METRICS = [("accuracy", "higher is better"), ("kappa", "higher is better"), ("ece", "lower is better"), ("aurc", "lower is better")]

_RUN = re.compile(r"^(?P<name>[^=@]+?)(?:@(?P<x>[0-9.eE+-]+))?=(?P<path>.+)$")


@dataclass
class RunSpec:
    name: str
    path: Path
    x: float | None = None


def parse_run(spec: str) -> RunSpec:
    m = _RUN.match(spec.strip())
    if not m:
        raise ValueError(f"run must look like 'name=path' or 'name@x=path', got {spec!r}")
    return RunSpec(m["name"].strip(), Path(m["path"].strip()), float(m["x"]) if m["x"] else None)


@dataclass
class Run:
    """One variant at one x: all its seeds."""

    name: str
    x: float | None
    reports: list[dict]

    @property
    def n_seeds(self) -> int:
        return len(self.reports)

    def rows(self, prim: str, slice_: dict | None = None) -> list[dict]:
        slice_ = slice_ or {}
        out = []
        for rep in self.reports:
            match = [r for r in rep["rows"] if r["type"] == prim and r["slice"] == slice_]
            if match:
                out.append(match[0])
        return out

    def stat(self, prim: str, metric: str, slice_: dict | None = None) -> tuple[float, float, float] | None:
        """(mean, min, max) over seeds, ``None`` if the metric is absent."""
        values = [r.get(metric) for r in self.rows(prim, slice_)]
        values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
        if not values:
            return None
        return float(np.mean(values)), float(np.min(values)), float(np.max(values))


def load_runs(specs: list[RunSpec]) -> list[Run]:
    grouped: dict[tuple[str, float | None], list[dict]] = {}
    for spec in specs:
        report = json.loads(Path(spec.path).read_text())
        if "rows" not in report:
            raise ValueError(f"{spec.path} is not a scripts/eval.py report")
        report["_path"] = str(spec.path)
        grouped.setdefault((spec.name, spec.x), []).append(report)
    return [Run(name, x, reports) for (name, x), reports in grouped.items()]


# ---------------------------------------------------------------------------- tables


def _fmt(stat: tuple[float, float, float] | None, n_seeds: int, metric: str = "") -> str:
    if stat is None:
        return "–"
    mean, lo, hi = stat
    text = f"{mean:.3f}"
    if n_seeds > 1 and hi > lo:
        text += f" ({lo:.3f}–{hi:.3f})"
    return text


def _best(values: list[float | None], metric: str) -> float | None:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return None
    if metric in HIGHER_IS_BETTER:
        return max(vals)
    if metric in LOWER_IS_BETTER:
        return min(vals)
    return None


def _label(run: Run) -> str:
    return run.name if run.x is None else f"{run.name} @ {run.x:g}"


def metrics_table(runs: list[Run], prim: str, slice_: dict | None = None) -> str:
    metrics = [m for m in TABLE_METRICS[prim] if any(r.stat(prim, m, slice_) for r in runs)]
    lines = ["| run | seeds | n | " + " | ".join(NAMES.get(m, m) for m in metrics) + " |", "|---|---|---|" + "---|" * len(metrics)]
    best = {m: _best([s[0] if (s := r.stat(prim, m, slice_)) else None for r in runs], m) for m in metrics}
    for run in runs:
        n = run.stat(prim, "n", slice_)
        cells = []
        for m in metrics:
            s = run.stat(prim, m, slice_)
            text = _fmt(s, run.n_seeds, m)
            if s is not None and best[m] is not None and math.isclose(s[0], best[m], abs_tol=1e-12):
                text = f"**{text}**"
            cells.append(text)
        lines.append(f"| {_label(run)} | {run.n_seeds} | {int(n[0]) if n else '–'} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def delta_table(runs: list[Run], prims: list[str]) -> str:
    ref = runs[0]
    key = ["accuracy", "kappa", "ece", "aurc"]
    lines = [f"Δ = run − {_label(ref)} (mean over seeds). For ECE and AURC lower is better, so a negative Δ is an improvement.", ""]
    lines.append("| run | primitive | " + " | ".join(f"Δ {NAMES.get(m, m)}" for m in key) + " |")
    lines.append("|---|---|" + "---|" * len(key))
    for run in runs[1:]:
        for prim in prims:
            cells = []
            for m in key:
                a, b = run.stat(prim, m), ref.stat(prim, m)
                cells.append("–" if a is None or b is None else f"{a[0] - b[0]:+.3f}")
            lines.append(f"| {_label(run)} | {prim} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def ops_table(runs: list[Run]) -> str | None:
    rows = []
    for run in runs:
        lat = [r["ops"]["latency_ms"] for r in run.reports if "latency_ms" in r.get("ops", {})]
        cost = [r["ops"]["cost_usd"] for r in run.reports if "cost_usd" in r.get("ops", {})]
        if not lat and not cost:
            continue
        p50 = f"{np.mean([x['p50'] for x in lat]):.1f}" if lat else "–"
        p95 = f"{np.mean([x['p95'] for x in lat]):.1f}" if lat else "–"
        per = f"{np.mean([c['mean'] for c in cost]):.6f}" if cost else "–"
        rows.append(f"| {_label(run)} | {p50} | {p95} | {per} |")
    if not rows:
        return None
    return "\n".join(["| run | latency p50, ms | latency p95, ms | cost per record, $ |", "|---|---|---|---|", *rows])


# ---------------------------------------------------------------------------- charts


def _present(runs: list[Run]) -> list[str]:
    return [p for p in TYPES if any(r.rows(p) for r in runs)]


def overall_chart(runs: list[Run], prims: list[str], colors: dict, out: Path) -> str:
    panels = []
    for metric, note in CHART_METRICS:
        series = {}
        for run in runs:
            stats = [run.stat(p, metric) for p in prims]
            series[_label(run)] = Bars(
                y=[s[0] if s else None for s in stats], lo=[s[1] if s else None for s in stats], hi=[s[2] if s else None for s in stats]
            )
        if any(v is not None for b in series.values() for v in b.y):
            panels.append(BarPanel(NAMES.get(metric, metric), prims, series, note=note))
    return bar_panels(panels, colors, out, "overall", "Overall metrics by primitive")


def reliability_chart(runs: list[Run], prims: list[str], colors: dict, out: Path) -> str:
    panels = []
    for prim in prims:
        series = {}
        for run in runs:
            rows = run.rows(prim)
            if rows and rows[0].get("reliability"):
                bins = rows[0]["reliability"]
                series[_label(run)] = Line([b["mean_confidence"] for b in bins], [b["mean_correct"] for b in bins], markers=True)
        what = "P(yes) vs observed rate" if prim == "bool" else "top probability vs accuracy"
        panels.append(LinePanel(prim, series, xlabel="predicted", ylabel="observed", xlim=(0, 1), ylim=(0, 1), reference=([0, 1], [0, 1]), reference_label="perfect calibration", note=what))
    return line_panels(panels, colors, out, "reliability", "Reliability diagrams (seed 1 of each run)")


def risk_coverage_chart(runs: list[Run], prims: list[str], colors: dict, out: Path) -> str | None:
    panels = []
    for prim in prims:
        series = {}
        for run in runs:
            rows = run.rows(prim)
            if rows and rows[0].get("risk_coverage"):
                rc = rows[0]["risk_coverage"]
                series[_label(run)] = Line(rc["coverage"], rc["risk"])
        if series:
            panels.append(LinePanel(prim, series, xlabel="coverage (share answered)", ylabel="risk (error rate)", xlim=(0, 1), note="lower is better"))
    if not panels:
        return None
    return line_panels(panels, colors, out, "risk_coverage", "Risk vs coverage: answer the most confident first, escalate the rest")


def slice_keys(runs: list[Run]) -> list[tuple[str, ...]]:
    keys = []
    for run in runs:
        for rep in run.reports:
            for row in rep["rows"]:
                k = tuple(row["slice"])
                if k and k not in keys:
                    keys.append(k)
    return keys


def slice_values(runs: list[Run], prim: str, keys: tuple[str, ...], limit: int) -> list[dict]:
    """Slice cells of one key set, largest first, at most ``limit``."""
    sizes: dict[tuple, int] = defaultdict(int)
    for run in runs:
        for rep in run.reports:
            for row in rep["rows"]:
                if row["type"] == prim and tuple(row["slice"]) == keys:
                    value = tuple(row["slice"][k] for k in keys)
                    sizes[value] = max(sizes[value], row["n"])
    top = sorted(sizes, key=lambda v: (-sizes[v], [str(x) for x in v]))[:limit]
    return [dict(zip(keys, v)) for v in top]


def slice_chart(runs: list[Run], keys: tuple[str, ...], metric: str, colors: dict, out: Path, limit: int) -> tuple[str, str] | None:
    panels, tables = [], []
    for prim in TYPES:
        cells = slice_values(runs, prim, keys, limit)
        if len(cells) < 2:
            continue
        labels = [" / ".join(str(c[k]) for k in keys) for c in cells]
        series = {}
        for run in runs:
            stats = [run.stat(prim, metric, c) for c in cells]
            series[_label(run)] = Bars([s[0] if s else None for s in stats], [s[1] if s else None for s in stats], [s[2] if s else None for s in stats])
        panels.append(BarPanel(f"{prim}", labels, series, ylabel=NAMES.get(metric, metric)))
        header = "| " + " × ".join(keys) + " | n | " + " | ".join(_label(r) for r in runs) + " |"
        rows = [header, "|---|---|" + "---|" * len(runs)]
        for c, label in zip(cells, labels):
            n = max((s[0] for r in runs if (s := r.stat(prim, "n", c))), default=0)
            rows.append(f"| {label} | {int(n)} | " + " | ".join(_fmt(r.stat(prim, metric, c), r.n_seeds) for r in runs) + " |")
        tables.append(f"**{prim}** — {NAMES.get(metric, metric)}\n\n" + "\n".join(rows))
    if not panels:
        return None
    name = "slice_" + "_x_".join(keys)
    md = bar_panels(panels, colors, out, name, f"{NAMES.get(metric, metric)} by {' × '.join(keys)}", horizontal=True, max_cols=3)
    return md, "\n\n".join(tables)


def learning_curve_chart(runs: list[Run], prims: list[str], out: Path) -> tuple[str, str] | None:
    curves = [r for r in runs if r.x is not None]
    if not curves:
        return None
    names = list(dict.fromkeys(r.name for r in curves))
    colors = series_colors(names)
    panels = []
    for metric, note in [("accuracy", "higher is better"), ("ece", "lower is better")]:
        for prim in prims:
            series = {}
            for name in names:
                pts = sorted((r.x, r.stat(prim, metric)) for r in curves if r.name == name and r.stat(prim, metric))
                if pts:
                    series[name] = Line([p[0] for p in pts], [p[1][0] for p in pts], lo=[p[1][1] for p in pts], hi=[p[1][2] for p in pts], markers=True)
            if series:
                panels.append(LinePanel(f"{prim}: {NAMES.get(metric, metric)}", series, xlabel="x (e.g. training examples)", ylabel=metric, xlog=True, note=note))
    if not panels:
        return None
    md = line_panels(panels, colors, out, "learning_curves", "Learning curves (band = min–max over seeds)")
    lines = ["| variant | x | seeds | primitive | accuracy | ece |", "|---|---|---|---|---|---|"]
    for r in sorted(curves, key=lambda r: (names.index(r.name), r.x)):
        for prim in prims:
            if r.rows(prim):
                lines.append(f"| {r.name} | {r.x:g} | {r.n_seeds} | {prim} | {_fmt(r.stat(prim, 'accuracy'), r.n_seeds)} | {_fmt(r.stat(prim, 'ece'), r.n_seeds)} |")
    return md, "\n".join(lines)


# ---------------------------------------------------------------------------- report

GLOSSARY = """\
- **accuracy** — share of correct answers (for soft targets: the target mass on the predicted answer).
- **κ** — Cohen's kappa: agreement with the labels beyond chance; 0 = chance, 1 = perfect. Quadratic-weighted for Score.
- **NLL**, **Brier** — proper scores of the whole probability distribution; lower is better.
- **ECE** — expected calibration error: the average gap between stated confidence and observed accuracy (equal-mass bins); 0 = perfectly calibrated. For Bool it is measured on P(yes); `ece (top)` on max(p, 1 − p).
- **AURC**, **sel.acc@80** — selective prediction: answer the most confident examples first. AURC is the area under the risk–coverage curve (lower is better); sel.acc@80 is the accuracy on the 80% most confident examples. This is the cascade trade-off: coverage = share answered locally, the rest escalates.
- Seeds: values are means over seeds, with the min–max range in brackets and as error bars.
"""


def build_report(specs: list[RunSpec], out: Path, title: str = "Comparison report", slice_metric: str = "accuracy", slice_limit: int = 15) -> Path:
    runs = load_runs(specs)
    if not runs:
        raise ValueError("no runs")
    out.mkdir(parents=True, exist_ok=True)
    prims = _present(runs)
    point_runs = [r for r in runs if r.x is None] or runs
    colors = series_colors([_label(r) for r in point_runs])

    md = [f"# {title}", "", f"Generated {date.today().isoformat()} by `scripts/make_report.py compare`.", ""]
    md += ["## Runs", "", "| run | seeds | x | reports |", "|---|---|---|---|"]
    for r in runs:
        md.append(f"| {r.name} | {r.n_seeds} | {'' if r.x is None else f'{r.x:g}'} | " + "<br>".join(f"`{rep['_path']}`" for rep in r.reports) + " |")
    md.append("")

    md += ["## Overall", "", overall_chart(point_runs, prims, colors, out), ""]
    for prim in prims:
        md += [f"### {prim}", "", metrics_table(point_runs, prim), ""]
    if len(point_runs) > 1:
        md += ["### Δ vs reference", "", delta_table(point_runs, prims), ""]

    md += ["## Calibration", "", "Points on the diagonal mean that stated probabilities match observed frequencies. Points below it: overconfident.", ""]
    md += [reliability_chart(point_runs, prims, colors, out), ""]

    rc = risk_coverage_chart(point_runs, prims, colors, out)
    if rc:
        md += ["## Selective prediction (cascade)", "", "Read the curve at the coverage you are willing to answer locally: its height is the error rate there. The rest is escalated to the LLM judge.", "", rc, ""]
        lines = ["| run | primitive | AURC | sel.acc@80 | sel.acc@90 |", "|---|---|---|---|---|"]
        for r in point_runs:
            for prim in prims:
                if r.rows(prim):
                    lines.append(f"| {_label(r)} | {prim} | " + " | ".join(_fmt(r.stat(prim, m), r.n_seeds) for m in ("aurc", "selective_acc@80", "selective_acc@90")) + " |")
        md += ["\n".join(lines), ""]

    ops = ops_table(point_runs)
    if ops:
        md += ["## Cost and latency", "", ops, ""]

    sections = [s for keys in slice_keys(point_runs) if (s := slice_chart(point_runs, keys, slice_metric, colors, out, slice_limit))]
    if sections:
        md += [f"## Slices ({NAMES.get(slice_metric, slice_metric)})", ""]
        for chart, table in sections:
            md += [chart, "", table, ""]

    lc = learning_curve_chart(runs, prims, out)
    if lc:
        md += ["## Learning curves", "", lc[0], "", lc[1], ""]

    md += ["## How to read the metrics", "", GLOSSARY]
    path = out / "report.md"
    path.write_text("\n".join(md))
    return path
