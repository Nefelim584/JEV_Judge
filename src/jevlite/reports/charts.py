"""Two chart forms used by every report, each rendered twice from the same data: a matplotlib PNG
for the markdown and a plotly HTML twin with hover.

- :func:`bar_panels` — small multiples of grouped bars (one panel per metric or primitive, one bar
  per run). Different metrics never share an axis.
- :func:`line_panels` — small multiples of lines (one line per run), with an optional min–max band
  and an optional reference line (e.g. the diagonal of a reliability diagram).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .style import (
    BAR_MAX_PX,
    LINE_WIDTH,
    MARKER_SIZE,
    MUTED,
    SAVE_DPI,
    SURFACE,
    TEXT_SECONDARY,
    md_figure,
    plotly_axis,
    plotly_layout,
    save_figure,
    setup_matplotlib,
)


@dataclass
class Bars:
    """Values of one run over a panel's categories; ``lo``/``hi`` give a min–max range (seeds)."""

    y: list[float | None]
    lo: list[float | None] | None = None
    hi: list[float | None] | None = None


@dataclass
class BarPanel:
    title: str
    categories: list[str]
    series: dict[str, Bars]
    ylabel: str = ""
    ylim: tuple[float, float] | None = None
    note: str = ""  # e.g. "lower is better", shown under the title


@dataclass
class Line:
    x: list[float]
    y: list[float]
    lo: list[float] | None = None
    hi: list[float] | None = None
    raw_y: list[float] | None = None  # unsmoothed values, drawn faint behind the line
    markers: bool = False


@dataclass
class LinePanel:
    title: str
    series: dict[str, Line]
    xlabel: str = ""
    ylabel: str = ""
    xlog: bool = False
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    reference: tuple[list[float], list[float]] | None = None
    reference_label: str = ""
    note: str = ""
    extra: dict = field(default_factory=dict)


def _grid(n: int, max_cols: int) -> tuple[int, int]:
    cols = min(n, max_cols)
    return math.ceil(n / cols), cols


def _clean(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def _panel_title(title: str, note: str) -> str:
    return f"{title}\n({note})" if note else title


# ---------------------------------------------------------------------------- bars


def bar_panels(
    panels: list[BarPanel],
    colors: dict[str, str],
    out_dir: Path,
    name: str,
    title: str,
    horizontal: bool = False,
    max_cols: int = 4,
) -> str:
    """Grouped bars per panel. Returns the markdown snippet (image + link to the interactive twin)."""
    plt = setup_matplotlib()
    import numpy as np

    rows, cols = _grid(len(panels), max_cols)
    n_runs = len(colors)
    n_cat = max(len(p.categories) for p in panels)
    if horizontal:
        size = (4.6 * cols, max(2.4, 0.32 * n_cat * max(n_runs, 1) + 1.2) * rows)
    else:
        size = (max(3.0, 0.55 * n_cat * max(n_runs, 1) + 1.2) * cols, 3.4 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False)

    for ax, panel in zip(axes.flat, panels):
        idx = np.arange(len(panel.categories))
        # Bars are capped at BAR_MAX_PX thick: convert pixels to data units of the band axis.
        extent = ax.get_window_extent()
        axis_px = ((extent.height if horizontal else extent.width) or 1) * SAVE_DPI / fig.dpi
        px_per_unit = axis_px / max(len(panel.categories), 1)
        width = min(0.8 / max(n_runs, 1), BAR_MAX_PX / px_per_unit)
        offsets = (np.arange(n_runs) - (n_runs - 1) / 2) * width * 1.12  # 12% gap between neighbours
        for k, (run, color) in enumerate(colors.items()):
            bars = panel.series.get(run)
            if bars is None:
                continue
            y = np.array([np.nan if v is None else v for v in bars.y], dtype=float)
            err = None
            if bars.lo is not None and bars.hi is not None:
                lo = np.array([np.nan if v is None else v for v in bars.lo], dtype=float)
                hi = np.array([np.nan if v is None else v for v in bars.hi], dtype=float)
                if np.nanmax(hi - lo, initial=0) > 0:
                    err = np.vstack([y - lo, hi - y])
            pos = idx + offsets[k]
            kw = dict(color=color, label=run, zorder=2)
            ekw = dict(ecolor=TEXT_SECONDARY, elinewidth=1, capsize=0)
            if horizontal:
                ax.barh(pos, y, height=width, xerr=err, error_kw=ekw, **kw)
            else:
                ax.bar(pos, y, width=width, yerr=err, error_kw=ekw, **kw)
        if horizontal:
            ax.set_yticks(idx, panel.categories)
            ax.invert_yaxis()
            ax.grid(axis="y", visible=False)
            if panel.ylim:
                ax.set_xlim(*panel.ylim)
            ax.set_xlabel(panel.ylabel)
        else:
            ax.set_xticks(idx, panel.categories)
            ax.grid(axis="x", visible=False)
            if panel.ylim:
                ax.set_ylim(*panel.ylim)
            ax.set_ylabel(panel.ylabel)
        ax.tick_params(length=0)
        ax.set_title(_panel_title(panel.title, panel.note), loc="left")
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    if n_runs >= 2:
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=min(n_runs, 4), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.06 if n_runs >= 2 else 0, 1, 0.97))

    html = _bars_plotly(panels, colors, title, horizontal, rows, cols)
    return md_figure(title, *save_figure(fig, out_dir, name, html))


def _bars_plotly(panels, colors, title, horizontal, rows, cols):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[_panel_title(p.title, p.note).replace("\n", "<br>") for p in panels])
    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        for run, color in colors.items():
            bars = panel.series.get(run)
            if bars is None:
                continue
            y = [_clean(v) for v in bars.y]
            err = None
            if bars.lo is not None and bars.hi is not None:
                err = dict(
                    type="data",
                    symmetric=False,
                    array=[None if v is None or h is None else h - v for v, h in zip(y, bars.hi)],
                    arrayminus=[None if v is None or lo is None else v - lo for v, lo in zip(y, bars.lo)],
                    color=TEXT_SECONDARY,
                    thickness=1,
                    width=0,
                )
            common = dict(name=run, legendgroup=run, showlegend=i == 0, marker_color=color, hovertemplate=f"{run}<br>%{{{'y' if horizontal else 'x'}}}: %{{{'x' if horizontal else 'y'}:.3f}}<extra></extra>")
            if horizontal:
                fig.add_trace(go.Bar(x=y, y=panel.categories, orientation="h", error_x=err, **common), row=r + 1, col=c + 1)
            else:
                fig.add_trace(go.Bar(x=panel.categories, y=y, error_y=err, **common), row=r + 1, col=c + 1)
        fig.update_xaxes(plotly_axis(), row=r + 1, col=c + 1)
        fig.update_yaxes(plotly_axis(autorange="reversed" if horizontal else True), row=r + 1, col=c + 1)
        if panel.ylim:
            (fig.update_xaxes if horizontal else fig.update_yaxes)(range=list(panel.ylim), row=r + 1, col=c + 1)
    fig.update_layout(plotly_layout(title, barmode="group", bargap=0.3, bargroupgap=0.12, height=360 * rows))
    return fig


# ---------------------------------------------------------------------------- lines


def line_panels(panels: list[LinePanel], colors: dict[str, str], out_dir: Path, name: str, title: str, max_cols: int = 3) -> str:
    plt = setup_matplotlib()

    rows, cols = _grid(len(panels), len(panels) if len(panels) <= 4 else max_cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.5 * rows), squeeze=False)
    for ax, panel in zip(axes.flat, panels):
        if panel.reference:
            ax.plot(*panel.reference, color=MUTED, linewidth=1, zorder=1, label=panel.reference_label or None)
        for run, color in colors.items():
            line = panel.series.get(run)
            if line is None or not line.x:
                continue
            if line.raw_y is not None:
                ax.plot(line.x, line.raw_y, color=color, linewidth=1, alpha=0.25, zorder=2)
            if line.lo is not None and line.hi is not None:
                ax.fill_between(line.x, line.lo, line.hi, color=color, alpha=0.1, linewidth=0, zorder=2)
            style = dict(marker="o", markersize=MARKER_SIZE, markeredgecolor=SURFACE, markeredgewidth=2) if line.markers else {}
            ax.plot(line.x, line.y, color=color, linewidth=LINE_WIDTH, label=run, zorder=3, **style)
        if panel.xlog:
            ax.set_xscale("log")
        if panel.xlim:
            ax.set_xlim(*panel.xlim)
        if panel.ylim:
            ax.set_ylim(*panel.ylim)
        ax.set_xlabel(panel.xlabel)
        ax.set_ylabel(panel.ylabel)
        ax.tick_params(length=0)
        ax.set_title(_panel_title(panel.title, panel.note), loc="left")
    for ax in list(axes.flat)[len(panels) :]:
        ax.set_visible(False)
    handles, labels = [], []
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    n_items = len(labels)
    if len(colors) >= 2 or any(p.reference_label for p in panels):
        legend = fig.legend(handles, labels, loc="lower center", ncol=min(n_items, 4), bbox_to_anchor=(0.5, -0.02))
        for h in legend.legend_handles:  # the surface ring separates marks on the plot; in the key it reads as a dash
            if hasattr(h, "set_markeredgewidth"):
                h.set_markeredgewidth(0)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0.07 if n_items >= 2 else 0, 1, 0.97))

    html = _lines_plotly(panels, colors, title, rows, cols)
    return md_figure(title, *save_figure(fig, out_dir, name, html))


def _hex_rgba(color: str, alpha: float) -> str:
    color = color.lstrip("#")
    r, g, b = (int(color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _lines_plotly(panels, colors, title, rows, cols):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[_panel_title(p.title, p.note).replace("\n", "<br>") for p in panels])
    shown: set[str] = set()
    for i, panel in enumerate(panels):
        r, c = divmod(i, cols)
        pos = dict(row=r + 1, col=c + 1)
        if panel.reference:
            label = panel.reference_label or "reference"
            fig.add_trace(go.Scatter(x=panel.reference[0], y=panel.reference[1], mode="lines", line=dict(color=MUTED, width=1), name=label, legendgroup=label, showlegend=label not in shown and bool(panel.reference_label), hoverinfo="skip"), **pos)
            shown.add(label)
        for run, color in colors.items():
            line = panel.series.get(run)
            if line is None or not line.x:
                continue
            if line.lo is not None and line.hi is not None:
                fig.add_trace(go.Scatter(x=list(line.x) + list(line.x)[::-1], y=list(line.hi) + list(line.lo)[::-1], fill="toself", fillcolor=_hex_rgba(color, 0.1), line=dict(width=0), hoverinfo="skip", showlegend=False, legendgroup=run), **pos)
            if line.raw_y is not None:
                fig.add_trace(go.Scatter(x=line.x, y=line.raw_y, mode="lines", line=dict(color=_hex_rgba(color, 0.25), width=1), hoverinfo="skip", showlegend=False, legendgroup=run), **pos)
            fig.add_trace(
                go.Scatter(
                    x=line.x,
                    y=line.y,
                    mode="lines+markers" if line.markers else "lines",
                    line=dict(color=color, width=2),
                    marker=dict(size=8, line=dict(color=SURFACE, width=2)),
                    name=run,
                    legendgroup=run,
                    showlegend=run not in shown,
                    hovertemplate=f"{run}<br>{panel.xlabel or 'x'}: %{{x}}<br>{panel.ylabel or 'y'}: %{{y:.4f}}<extra></extra>",
                ),
                **pos,
            )
            shown.add(run)
        fig.update_xaxes(plotly_axis(type="log" if panel.xlog else "linear", title_text=panel.xlabel, **({"range": [math.log10(v) for v in panel.xlim] if panel.xlog else list(panel.xlim)} if panel.xlim else {})), **pos)
        fig.update_yaxes(plotly_axis(title_text=panel.ylabel, **({"range": list(panel.ylim)} if panel.ylim else {})), **pos)
    fig.update_layout(plotly_layout(title, hovermode="x unified" if all(not p.reference for p in panels) else "closest", height=380 * rows))
    return fig
