"""Training logs and the training report (TODO Phase 6: per-primitive loss, accuracy and ECE every N steps).

Log format: ``metrics.jsonl`` in the run directory, one JSON object per line.

    {"phase": "meta", "run": "stage_a-lora-s42", "stack": "cuda", "config": {...}}      # first line, optional
    {"phase": "train", "step": 100, "epoch": 0.12, "lr": 1.9e-4, "loss": 0.71, "loss/bool": 0.52, "loss/choice": 0.93}
    {"phase": "eval",  "step": 500, "split": "test_in", "loss": 0.64, "accuracy/bool": 0.81, "ece/bool": 0.04}

Keys are ``metric`` (total) or ``metric/primitive``. Any numeric key is plotted; ``step`` is the x axis.
Write it with :class:`TrainingLog` so notebooks on both stacks produce the same files.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

from .charts import Line, LinePanel, line_panels
from .style import series_colors

TYPES = ("choice", "score", "bool")
LOWER_IS_BETTER = ("loss", "nll", "brier", "ece", "mce", "aurc", "mae")


class TrainingLog:
    """Append-only ``metrics.jsonl`` writer. Flushes every line, so a dead Colab session loses nothing."""

    def __init__(self, run_dir: str | Path, run: str | None = None, meta: dict | None = None):
        self.path = Path(run_dir) / "metrics.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if meta is not None or run is not None:
            if not self.path.exists() or self.path.stat().st_size == 0:
                self._write({"phase": "meta", "run": run or self.path.parent.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"), **(meta or {})})

    def _write(self, row: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(row, default=float) + "\n")

    def log(self, phase: str, step: int, split: str | None = None, **metrics: float) -> None:
        """One row; ``split`` names the eval split (``test_in``, ``calib`` …)."""
        if phase not in ("train", "eval"):
            raise ValueError("phase must be 'train' or 'eval'")
        row = {"phase": phase, "step": int(step), **({"split": split} if split else {})}
        self._write(row | {k: float(v) for k, v in metrics.items() if v is not None})


def read_log(path: str | Path) -> tuple[dict, list[dict]]:
    """``(meta, rows)``; ``path`` is a ``metrics.jsonl`` or the run directory holding it."""
    path = Path(path)
    if path.is_dir():
        path = path / "metrics.jsonl"
    meta, rows = {}, []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("phase") == "meta":
                meta = row
            else:
                rows.append(row)
    if not rows:
        raise ValueError(f"{path}: no train/eval rows")
    return meta, rows


def series(rows: list[dict], phase: str, key: str, split: str | None = None) -> tuple[list[float], list[float]]:
    pts = [
        (r["step"], r[key])
        for r in rows
        if r.get("phase") == phase and key in r and isinstance(r[key], (int, float)) and math.isfinite(r[key]) and (split is None or r.get("split", split) == split)
    ]
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def ema(values: list[float], alpha: float) -> list[float]:
    """Bias-corrected exponential moving average (as TensorBoard's smoothing)."""
    out, acc = [], 0.0
    for i, v in enumerate(values, start=1):
        acc = alpha * acc + (1 - alpha) * v
        out.append(acc / (1 - alpha**i) if alpha < 1 else v)
    return out


def _keys(logs: dict[str, tuple[dict, list[dict]]], phase: str) -> list[str]:
    keys = []
    for _, rows in logs.values():
        for r in rows:
            if r.get("phase") == phase:
                for k, v in r.items():
                    if k not in ("phase", "step", "split", "epoch") and isinstance(v, (int, float)) and k not in keys:
                        keys.append(k)
    return keys


def _families(keys: list[str]) -> dict[str, list[str]]:
    """``loss``, ``loss/bool`` … → ``{"loss": ["loss", "loss/bool", …]}``."""
    fam: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        fam[k.split("/", 1)[0]].append(k)
    return fam


def _panel_order(keys: list[str]) -> list[str]:
    def rank(k: str) -> tuple:
        sub = k.split("/", 1)[1] if "/" in k else ""
        return (0 if not sub else 1 + (TYPES.index(sub) if sub in TYPES else len(TYPES)), sub)

    return sorted(keys, key=rank)


def _splits(logs) -> list[str | None]:
    splits = []
    for _, rows in logs.values():
        for r in rows:
            if r.get("phase") == "eval" and r.get("split") not in splits:
                splits.append(r.get("split"))
    return splits or [None]


def _note(family: str) -> str:
    return "lower is better" if family in LOWER_IS_BETTER else "higher is better" if family not in ("lr", "grad_norm") else ""


def build_report(runs: dict[str, Path], out: Path, title: str = "Training report", smooth: float = 0.9) -> Path:
    logs = {name: read_log(path) for name, path in runs.items()}
    colors = series_colors(list(logs))
    out.mkdir(parents=True, exist_ok=True)

    md = [f"# {title}", "", f"Generated {date.today().isoformat()} by `scripts/make_report.py training`.", ""]
    md += ["## Runs", "", "| run | train steps | last step | eval points | stack | log |", "|---|---|---|---|---|---|"]
    for name, (meta, rows) in logs.items():
        train = [r for r in rows if r.get("phase") == "train"]
        evals = [r for r in rows if r.get("phase") == "eval"]
        last = max(r["step"] for r in rows)
        md.append(f"| {name} | {len(train)} | {last} | {len(evals)} | {meta.get('stack', '')} | `{runs[name]}` |")
    md.append("")

    configs = {n: m.get("config") for n, (m, _) in logs.items() if m.get("config")}
    if configs:
        md += ["### Config", "", "```json", json.dumps(configs, indent=1, default=str), "```", ""]

    # Train curves: one figure per metric family, one panel per key (total first, then per primitive).
    train_fams = _families(_keys(logs, "train"))
    if train_fams:
        md += ["## Training curves", "", f"Lines are smoothed with an EMA (α = {smooth}); the faint line is the raw value.", ""]
    for family, keys in train_fams.items():
        panels = []
        for key in _panel_order(keys):
            lines = {}
            for name, (_, rows) in logs.items():
                x, y = series(rows, "train", key)
                if x:
                    smoothed = ema(y, smooth) if family != "lr" else y
                    lines[name] = Line(x, smoothed, raw_y=y if family != "lr" else None)
            if lines:
                panels.append(LinePanel(key, lines, xlabel="step", ylabel=family, note=_note(family)))
        if panels:
            md += [line_panels(panels, colors, out, f"train_{family}", f"Train: {family}"), ""]

    # Eval curves: per split, one figure per metric family.
    eval_fams = _families(_keys(logs, "eval"))
    for split in _splits(logs):
        if not eval_fams:
            break
        md += [f"## Evaluation{f' — {split}' if split else ''}", ""]
        for family, keys in eval_fams.items():
            panels = []
            for key in _panel_order(keys):
                lines = {}
                for name, (_, rows) in logs.items():
                    x, y = series(rows, "eval", key, split)
                    if x:
                        lines[name] = Line(x, y, markers=len(x) <= 30)
                if lines:
                    panels.append(LinePanel(key, lines, xlabel="step", ylabel=family, note=_note(family)))
            if panels:
                md += [line_panels(panels, colors, out, f"eval_{split or 'all'}_{family}", f"Eval{f' ({split})' if split else ''}: {family}"), ""]
        md += [summary_table(logs, split, list(eval_fams)), ""]

    path = out / "report.md"
    path.write_text("\n".join(md))
    return path


def summary_table(logs, split: str | None, families: list[str]) -> str:
    """Last eval value of every key and, for the total loss, the best value and its step."""
    keys = [k for fam in families for k in _panel_order(_families(_keys(logs, "eval"))[fam])]
    lines = ["| key | " + " | ".join(logs) + " |", "|---|" + "---|" * len(logs)]
    for key in keys:
        cells = []
        for _, rows in logs.values():
            x, y = series(rows, "eval", key, split)
            if not y:
                cells.append("–")
                continue
            text = f"{y[-1]:.4f}"
            pick = min if key.split("/", 1)[0] in LOWER_IS_BETTER else max
            i = pick(range(len(y)), key=y.__getitem__)
            if i != len(y) - 1:
                text += f" (best {y[i]:.4f} @ {x[i]})"
            cells.append(text)
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    return "Last evaluation, with the best value and its step when it is not the last:\n\n" + "\n".join(lines)
