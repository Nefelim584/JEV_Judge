"""Paired comparison of two models on the same records (todo, Stage A review: statistics).

Unpaired bootstrap intervals of two models overlap even when one model is right on the same records
the other gets wrong, and differences of 1–4 points on one seed are noise unless a paired test says
otherwise. Per record we compute correctness (Bool: P(yes) ≥ 0.5 vs target ≥ 0.5; Choice / Score:
argmax vs argmax of the target) and NLL, then per slice:

- Δ accuracy and Δ NLL (model − reference) with a **paired bootstrap** CI (records resampled jointly);
- **McNemar's exact test** on the discordant records (two-sided binomial), with a **Holm** correction
  over the cells of one slicing (e.g. all sources), since one table holds many tests.

A difference is called a win or a loss only if the CI of Δ accuracy excludes 0 and the Holm-adjusted
McNemar p is below ``alpha``; otherwise it is a tie.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

import numpy as np

from .data.unified import Prediction, Record

EPS = 1e-12


def record_scores(r: Record, p: Prediction) -> tuple[bool, float]:
    """(correct, NLL) of one prediction."""
    if r.type == "bool":
        y, q = float(r.target), min(max(float(p.prob), EPS), 1 - EPS)
        return (q >= 0.5) == (y >= 0.5), -(y * math.log(q) + (1 - y) * math.log(1 - q))
    probs = np.clip(np.asarray(p.probs, dtype=np.float64), EPS, 1.0)
    if isinstance(r.target, list):
        t = np.asarray(r.target, dtype=np.float64)
    else:
        t = np.zeros(len(probs))
        t[int(r.target)] = 1.0
    return int(np.argmax(probs)) == int(np.argmax(t)), float(-(t * np.log(probs)).sum())


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: ``b``, ``c`` = records only the model / only the reference gets right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(pvalues: Sequence[float]) -> list[float]:
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adj, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[i]))
        adj[i] = running
    return adj


def _bootstrap(d_acc: np.ndarray, d_nll: np.ndarray, n_boot: int, seed: int) -> tuple[tuple[float, float], tuple[float, float]]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d_acc), size=(n_boot, len(d_acc)))
    acc, nll = d_acc[idx].mean(axis=1), d_nll[idx].mean(axis=1)
    return tuple(np.percentile(acc, [2.5, 97.5])), tuple(np.percentile(nll, [2.5, 97.5]))


def paired_compare(
    records: Sequence[Record], model: dict[str, Prediction], reference: dict[str, Prediction],
    slices: Sequence[tuple[str, ...]] = ((), ("split",), ("source",), ("split", "source")),
    n_boot: int = 2000, alpha: float = 0.05, min_n: int = 20, seed: int = 0,
) -> list[dict]:
    """One row per (slicing, cell, primitive) on the records both models predicted."""
    rows_by_key: dict[tuple, list[tuple[bool, float, bool, float]]] = defaultdict(list)
    for r in records:
        if r.id not in model or r.id not in reference:
            continue
        a, b = record_scores(r, model[r.id]), record_scores(r, reference[r.id])
        for fields in slices:
            cell = tuple(str(r.field(f)) for f in fields)
            rows_by_key[(fields, cell, r.type)].append((*a, *b))
    out: list[dict] = []
    for (fields, cell, prim), vals in sorted(rows_by_key.items()):
        if len(vals) < min_n:
            continue
        v = np.asarray(vals, dtype=np.float64)
        ca, na, cb, nb = v[:, 0], v[:, 1], v[:, 2], v[:, 3]
        (lo, hi), (nlo, nhi) = _bootstrap(ca - cb, na - nb, n_boot, seed)
        only_a, only_b = int(((ca == 1) & (cb == 0)).sum()), int(((ca == 0) & (cb == 1)).sum())
        out.append({
            "slicing": "*".join(fields) or "all", "cell": "/".join(cell) or "all", "primitive": prim, "n": len(v),
            "acc_model": ca.mean(), "acc_ref": cb.mean(), "d_acc": (ca - cb).mean(), "d_acc_ci": [lo, hi],
            "nll_model": na.mean(), "nll_ref": nb.mean(), "d_nll": (na - nb).mean(), "d_nll_ci": [nlo, nhi],
            "only_model": only_a, "only_ref": only_b, "p_mcnemar": mcnemar_exact(only_a, only_b),
        })
    by_slicing: dict[str, list[dict]] = defaultdict(list)
    for row in out:
        by_slicing[row["slicing"]].append(row)
    for rows in by_slicing.values():
        for row, p in zip(rows, holm([r["p_mcnemar"] for r in rows])):
            row["p_holm"] = p
            significant = p < alpha and (row["d_acc_ci"][0] > 0 or row["d_acc_ci"][1] < 0)
            row["verdict"] = ("win" if row["d_acc"] > 0 else "loss") if significant else "tie"
    return out


def to_markdown(rows: list[dict], model: str, reference: str) -> str:
    lines = [
        f"# Paired comparison: {model} vs {reference}", "",
        "Same records for both models. Δ = model − reference; 95% paired-bootstrap CI; McNemar exact p, Holm-adjusted "
        "within each slicing. **win / loss** only if the CI of Δ accuracy excludes 0 and p_holm < 0.05.", "",
    ]
    for slicing in dict.fromkeys(r["slicing"] for r in rows):
        lines += [f"## By {slicing}", "",
                  f"| Cell | Primitive | n | {model} | {reference} | Δ acc [95% CI] | Δ NLL [95% CI] | p (Holm) | Verdict |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in (r for r in rows if r["slicing"] == slicing):
            lines.append(
                f"| {r['cell']} | {r['primitive']} | {r['n']} | {r['acc_model']:.3f} | {r['acc_ref']:.3f} "
                f"| {100 * r['d_acc']:+.1f} [{100 * r['d_acc_ci'][0]:+.1f}, {100 * r['d_acc_ci'][1]:+.1f}] "
                f"| {r['d_nll']:+.3f} [{r['d_nll_ci'][0]:+.3f}, {r['d_nll_ci'][1]:+.3f}] | {r['p_holm']:.3g} | {r['verdict']} |"
            )
        lines.append("")
    return "\n".join(lines)
