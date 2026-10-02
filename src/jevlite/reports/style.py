"""Chart tokens and the matplotlib / plotly setup shared by all reports.

Categorical palette: the validated 8-slot reference palette (light mode, surface ``#fcfcfb``); slots
are assigned in fixed order and follow the run, never its rank. Three slots are below 3:1 contrast
on the surface, so every chart ships with a table of the same numbers (the reports always do).
"""

from __future__ import annotations

from pathlib import Path

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MAX_SERIES = len(SERIES)

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FONT = ["system-ui", "-apple-system", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"]

LINE_WIDTH = 2.0
MARKER_SIZE = 8  # points, matplotlib `markersize`
BAR_MAX_PX = 24
SAVE_DPI = 150


def series_colors(names: list[str]) -> dict[str, str]:
    """Fixed-order slot per run name, in the order the runs were given."""
    if len(names) > MAX_SERIES:
        raise ValueError(f"at most {MAX_SERIES} runs per chart (got {len(names)}); split the comparison")
    return {name: SERIES[i] for i, name in enumerate(names)}


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": FONT[2:],
            "font.size": 10,
            "text.color": TEXT,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.edgecolor": AXIS,
            "axes.linewidth": 1.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 1.0,
            "grid.linestyle": "-",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": TEXT_SECONDARY,
            "ytick.labelcolor": TEXT_SECONDARY,
            "lines.linewidth": LINE_WIDTH,
            "lines.solid_capstyle": "round",
            "lines.solid_joinstyle": "round",
            "legend.frameon": False,
            "legend.numpoints": 1,
            "legend.handlelength": 1.6,
            "legend.labelcolor": TEXT_SECONDARY,
        }
    )
    return plt


def plotly_layout(title: str | None = None, **extra) -> dict:
    layout = {
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "font": {"family": ", ".join(FONT), "color": TEXT_SECONDARY, "size": 12},
        "title": {"text": title, "font": {"color": TEXT, "size": 15}} if title else None,
        "hovermode": "closest",
        "legend": {"orientation": "h", "y": -0.18},
        "margin": {"l": 60, "r": 20, "t": 60 if title else 30, "b": 60},
    }
    layout.update(extra)
    return {k: v for k, v in layout.items() if v is not None}


def plotly_axis(**extra) -> dict:
    return {"gridcolor": GRID, "linecolor": AXIS, "zeroline": False, "tickfont": {"color": TEXT_SECONDARY}, **extra}


def save_figure(fig, out_dir: Path, name: str, html_fig=None) -> tuple[str, str | None]:
    """Save the matplotlib figure as PNG and the plotly twin as HTML; returns relative paths."""
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"{name}.png"
    fig.savefig(png, dpi=SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    html_rel = None
    if html_fig is not None:
        html = fig_dir / f"{name}.html"
        # "directory": plotly.min.js is written once next to the charts, so they work offline.
        html_fig.write_html(html, include_plotlyjs="directory", full_html=True)
        html_rel = f"figures/{html.name}"
    return f"figures/{png.name}", html_rel


def md_figure(title: str, png: str, html: str | None) -> str:
    link = f"\n\n[Interactive version]({html})" if html else ""
    return f"![{title}]({png}){link}\n"
