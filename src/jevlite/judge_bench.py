"""Metrics for the judge benchmarks (``jevlite.data.public.judge_bench``) that ``evaluation`` lacks.

- Bool: **balanced accuracy** at 0.5 (the labels are skewed: HaluBench ~half, RAGBench and FaithBench
  not) and **AUROC**, which ignores the threshold. On new sources the 0.5 threshold of jev-lite drifts
  (Stage A), so AUROC tells ranking from calibration.
- Pairwise Choice (ContextualJudgeBench): accuracy per response order, **consistent accuracy** (the
  pair counts only if both orders are right, the paper's main metric) and **position bias** (share of
  answers that pick response 1; 0.5 is unbiased).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from .data.unified import Prediction, Record
from .metrics import bootstrap_ci


def auroc(score, target) -> float:
    """Area under the ROC curve via the rank-sum (ties get the mean rank); NaN with one class only."""
    s, t = np.asarray(score, dtype=float), np.asarray(target, dtype=float) >= 0.5
    n_pos, n_neg = int(t.sum()), int((~t).sum())
    if not n_pos or not n_neg:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    _, inv, counts = np.unique(s[order], return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks[order] = (ends - (counts - 1) / 2.0)[inv]
    return float((ranks[t].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def balanced_accuracy(pred, target) -> float:
    """Mean of the recalls of the two classes (a class absent from ``target`` is skipped)."""
    p, t = np.asarray(pred, dtype=bool), np.asarray(target, dtype=float) >= 0.5
    parts = [float(np.mean(p[m] == t[m])) for m in (t, ~t) if m.any()]
    return float(np.mean(parts)) if parts else float("nan")


def bool_summary(records: Sequence[Record], preds: Sequence[Prediction], n_boot: int = 0) -> dict:
    prob = np.asarray([p.prob for p in preds], dtype=float)
    target = np.asarray([float(r.target) for r in records])
    yes = prob >= 0.5
    out = {
        "n": len(records),
        "accuracy": float(np.mean(yes == (target >= 0.5))),
        "balanced_accuracy": balanced_accuracy(yes, target),
        "auroc": auroc(prob, target),
        "positive_rate": float(np.mean(target >= 0.5)),
        "predicted_positive_rate": float(np.mean(yes)),
    }
    if n_boot:
        out["ci95"] = {
            "balanced_accuracy": list(bootstrap_ci(lambda a, b: balanced_accuracy(a >= 0.5, b), prob, target, n_boot=n_boot)),
            "auroc": list(bootstrap_ci(auroc, prob, target, n_boot=n_boot)),
        }
    return out


def pairwise_summary(records: Sequence[Record], preds: Sequence[Prediction], n_boot: int = 0) -> dict:
    """Records come in pairs sharing ``group``; a group with one order missing is left out."""
    by_group: dict[str, dict[int, bool]] = defaultdict(dict)
    first = []
    for r, p in zip(records, preds, strict=True):
        pick = int(np.argmax(p.probs))
        by_group[r.group][int(r.field("order"))] = pick == int(r.target)
        first.append(pick == 0)
    pairs = [g for g in by_group.values() if len(g) == 2]
    both = np.asarray([g[0] and g[1] for g in pairs], dtype=float)
    out = {
        "n_pairs": len(pairs),
        "accuracy": float(np.mean([v for g in pairs for v in g.values()])) if pairs else float("nan"),
        "accuracy_order0": float(np.mean([g[0] for g in pairs])) if pairs else float("nan"),
        "accuracy_order1": float(np.mean([g[1] for g in pairs])) if pairs else float("nan"),
        "consistent_accuracy": float(both.mean()) if pairs else float("nan"),
        "picks_first": float(np.mean(first)) if first else float("nan"),
    }
    if n_boot and pairs:
        out["ci95"] = {"consistent_accuracy": list(bootstrap_ci(np.mean, both, n_boot=n_boot))}
    return out


__all__ = ["auroc", "balanced_accuracy", "bool_summary", "pairwise_summary"]
