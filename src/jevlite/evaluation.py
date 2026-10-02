"""Evaluation report: records + predictions → metrics per primitive and slice (TODO Phase 2).

A report is a list of rows. Each row is one primitive (``choice`` / ``score`` / ``bool``) on one
slice: the whole set, or one value of a slicing field (``source``, ``domain``, ``criterion`` …), or one
cell of a cross of fields. The same function is used for every model and baseline, so their reports
are comparable.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np

from .data.unified import Prediction, Record
from .metrics import (
    DEFAULT_BINS,
    bool_metrics,
    bootstrap_ci,
    categorical_metrics,
    cohen_kappa,
    ece_mce,
    score_metrics,
    spearman,
)

DEFAULT_SLICES: tuple[tuple[str, ...], ...] = (
    (),
    ("split",),
    ("source",),
    ("domain",),
    ("criterion",),
    ("source", "domain"),
)
TYPES = ("choice", "score", "bool")


def join(records: Sequence[Record], predictions: dict[str, Prediction], allow_missing: bool = False):
    """Pair each record with its prediction and check that the prediction fits the record's schema."""
    pairs, missing = [], []
    for r in records:
        p = predictions.get(r.id)
        if p is None:
            missing.append(r.id)
            continue
        if r.type == "bool":
            if p.prob is None:
                raise ValueError(f"{r.id}: bool record needs 'prob', got 'probs'")
        elif p.probs is None or len(p.probs) != r.n_candidates:
            got = "prob" if p.probs is None else f"{len(p.probs)} probs"
            raise ValueError(f"{r.id}: {r.type} record has {r.n_candidates} candidates, prediction has {got}")
        pairs.append((r, p))
    if missing and not allow_missing:
        raise ValueError(f"{len(missing)} records have no prediction, e.g. {missing[:3]}")
    return pairs, missing


def _soft_matrix(records: Sequence[Record], k_max: int) -> np.ndarray:
    t = np.zeros((len(records), k_max))
    for i, r in enumerate(records):
        if isinstance(r.target, list):
            t[i, : len(r.target)] = r.target
        else:
            t[i, int(r.target)] = 1.0
    return t


def _probs_matrix(preds: Sequence[Prediction], k_max: int) -> np.ndarray:
    p = np.zeros((len(preds), k_max))
    for i, pr in enumerate(preds):
        p[i, : len(pr.probs)] = pr.probs
    return p / p.sum(1, keepdims=True)


def _confidence(preds: Sequence[Prediction]) -> np.ndarray | None:
    conf = [p.confidence for p in preds]
    return None if any(c is None for c in conf) else np.asarray(conf, dtype=float)


def primitive_metrics(
    prim: str, records: Sequence[Record], preds: Sequence[Prediction], n_bins: int = DEFAULT_BINS, n_boot: int = 0
) -> dict:
    """All metrics for one primitive on one slice, plus agreement (κ, Spearman) and optional bootstrap CIs."""
    conf = _confidence(preds)
    ci: dict[str, tuple[float, float]] = {}

    if prim == "bool":
        prob = np.asarray([p.prob for p in preds], dtype=float)
        target = np.asarray([float(r.target) for r in records])
        out = bool_metrics(prob, target, conf, n_bins)
        hard_p, hard_t = (prob >= 0.5).astype(int), (target >= 0.5).astype(int)
        out["kappa"] = cohen_kappa(hard_p, hard_t, n_labels=2)
        out["positive_rate"] = float(hard_t.mean())
        if n_boot:
            correct = np.where(hard_p == 1, target, 1.0 - target)
            ci["accuracy"] = bootstrap_ci(np.mean, correct, n_boot=n_boot)
            ci["kappa"] = bootstrap_ci(lambda a, b: cohen_kappa(a, b, n_labels=2), hard_p, hard_t, n_boot=n_boot)
            ci["ece"] = bootstrap_ci(lambda a, b: ece_mce(a, b, n_bins)[0], prob, target, n_boot=n_boot)
    else:
        k_max = max(r.n_candidates for r in records)
        p = _probs_matrix(preds, k_max)
        t = _soft_matrix(records, k_max)
        fn = score_metrics if prim == "score" else categorical_metrics
        out = fn(p, t, conf, n_bins)
        hard_p, hard_t = p.argmax(1), t.argmax(1)
        weights = "quadratic" if prim == "score" else None
        out["kappa"] = cohen_kappa(hard_p, hard_t, weights=weights, n_labels=k_max)
        if prim == "score":
            # Levels are normalised to [0, 1] so that scales with different K can share a slice.
            span = np.asarray([max(r.n_candidates - 1, 1) for r in records], dtype=float)
            levels = np.arange(k_max)
            out["spearman"] = spearman(p @ levels / span, t @ levels / span) if len(records) > 1 else float("nan")
        if n_boot:
            top = p[np.arange(len(p)), hard_p]
            correct = t[np.arange(len(t)), hard_p]
            ci["accuracy"] = bootstrap_ci(np.mean, correct, n_boot=n_boot)
            ci["kappa"] = bootstrap_ci(
                lambda a, b: cohen_kappa(a, b, weights=weights, n_labels=k_max), hard_p, hard_t, n_boot=n_boot
            )
            ci["ece"] = bootstrap_ci(lambda a, b: ece_mce(a, b, n_bins)[0], top, correct, n_boot=n_boot)
    if ci:
        out["ci95"] = {k: list(v) for k, v in ci.items()}
    return out


def _ops(preds: Sequence[Prediction]) -> dict:
    out = {}
    lat = [p.latency_ms for p in preds if p.latency_ms is not None]
    if lat:
        out["latency_ms"] = {
            "mean": float(np.mean(lat)),
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
        }
    cost = [p.cost_usd for p in preds if p.cost_usd is not None]
    if cost:
        out["cost_usd"] = {"total": float(np.sum(cost)), "mean": float(np.mean(cost))}
    return out


def evaluate(
    records: Sequence[Record],
    predictions: dict[str, Prediction],
    slices: Iterable[tuple[str, ...]] = DEFAULT_SLICES,
    n_bins: int = DEFAULT_BINS,
    n_boot: int = 0,
    min_n: int = 1,
    allow_missing: bool = False,
) -> dict:
    """Build the report. Slices whose field is absent from every record are skipped; cells with fewer
    than ``min_n`` examples are dropped."""
    pairs, missing = join(records, predictions, allow_missing)
    if not pairs:
        raise ValueError("nothing to evaluate: no record has a prediction")

    rows = []
    for keys in slices:
        if keys and all(r.field(k) is None for r, _ in pairs for k in keys):
            continue
        cells: dict[tuple, list] = defaultdict(list)
        for r, p in pairs:
            values = tuple(r.field(k) for k in keys)
            if any(v is None for v in values):
                continue
            cells[(r.type, values)].append((r, p))
        for (prim, values), cell in sorted(cells.items(), key=lambda kv: (TYPES.index(kv[0][0]), [str(v) for v in kv[0][1]])):
            if len(cell) < min_n:
                continue
            recs, preds = zip(*cell)
            rows.append(
                {
                    "type": prim,
                    "slice": dict(zip(keys, values)),
                    **primitive_metrics(prim, recs, preds, n_bins, n_boot),
                    **_ops(preds),
                }
            )
    return {
        "n_records": len(records),
        "n_evaluated": len(pairs),
        "n_missing": len(missing),
        "ops": _ops([p for _, p in pairs]),
        "rows": rows,
    }


# ---------------------------------------------------------------------------- rendering

_COLUMNS = {
    "choice": ["n", "accuracy", "kappa", "nll", "brier", "ece", "mce", "aurc", "selective_acc@80"],
    "score": ["n", "accuracy", "kappa", "spearman", "mae", "mae_expected", "nll", "ece", "aurc"],
    "bool": ["n", "positive_rate", "accuracy", "kappa", "nll", "brier", "ece", "ece_top", "aurc", "selective_acc@80"],
}
_NAMES = {"kappa": "κ", "selective_acc@80": "sel.acc@80", "positive_rate": "pos.rate", "mae_expected": "mae(E)"}


def _fmt(row: dict, col: str) -> str:
    v = row.get(col)
    if v is None:
        return ""
    if col == "n":
        return str(v)
    if isinstance(v, float) and math.isnan(v):
        return "–"
    text = f"{v:.3f}"
    lo_hi = row.get("ci95", {}).get(col)
    if lo_hi and not any(math.isnan(x) for x in lo_hi):
        text += f" [{lo_hi[0]:.3f}, {lo_hi[1]:.3f}]"
    return text


def to_markdown(report: dict, title: str = "Evaluation report") -> str:
    lines = [f"# {title}", ""]
    lines.append(f"Records: {report['n_records']}, evaluated: {report['n_evaluated']}, missing predictions: {report['n_missing']}.")
    ops = report.get("ops") or {}
    if "latency_ms" in ops:
        lat = ops["latency_ms"]
        lines.append(f"Latency per record: mean {lat['mean']:.1f} ms, p50 {lat['p50']:.1f} ms, p95 {lat['p95']:.1f} ms.")
    if "cost_usd" in ops:
        lines.append(f"Cost: total ${ops['cost_usd']['total']:.4f}, per record ${ops['cost_usd']['mean']:.6f}.")
    lines.append("")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in report["rows"]:
        groups[(row["type"], tuple(row["slice"]))].append(row)
    for (prim, keys), rows in groups.items():
        header = "overall" if not keys else " × ".join(keys)
        lines += [f"## {prim} — {header}", ""]
        cols = [c for c in _COLUMNS[prim] if any(c in r for r in rows)]
        lines.append("| " + " | ".join([*keys, *(_NAMES.get(c, c) for c in cols)]) + " |")
        lines.append("|" + "---|" * (len(keys) + len(cols)))
        for r in rows:
            cells = [str(r["slice"][k]) for k in keys] + [_fmt(r, c) for c in cols]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def json_safe(obj):
    """NaN/inf → None and numpy scalars → Python, so the report is strict JSON."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def parse_slices(spec: str | None) -> tuple[tuple[str, ...], ...]:
    """``"source,domain,source*domain"`` → ``((), ("source",), ("domain",), ("source", "domain"))``."""
    if not spec:
        return DEFAULT_SLICES
    out: list[tuple[str, ...]] = [()]
    for part in spec.split(","):
        part = part.strip()
        if part:
            out.append(tuple(k.strip() for k in part.split("*")))
    return tuple(dict.fromkeys(out))  # dedupe, keep order


__all__ = ["DEFAULT_SLICES", "evaluate", "join", "json_safe", "parse_slices", "primitive_metrics", "to_markdown"]
