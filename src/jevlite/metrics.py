"""Evaluation metrics: quality, proper scores, calibration and selective prediction (TODO 2.7, 2.8).

Conventions
-----------
- Choice / Score predictions are ``probs [N, K]``. Padded candidates carry probability 0
  (softmax over ``-inf``), so batches with different K can be stacked after padding.
- Targets are either class indices ``[N]`` or (soft) distributions ``[N, K]``.
  Bool targets are floats in ``[0, 1]``.
- With soft targets, "correct" generalises to **expected correctness**: the target mass on the
  predicted answer. For one-hot targets this is 0/1; for a uniform "insufficient information"
  target over K options it is 1/K. Accuracy, ECE and risk-coverage all use it.
- ECE uses equal-mass bins: examples are sorted by confidence and split into ``n_bins``
  chunks of (nearly) equal size. Tied confidences always share a bin, so heavy ties give
  fewer, larger bins.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

EPS = 1e-12
DEFAULT_BINS = 15
SELECTIVE_COVERAGES = (0.8, 0.9)


def _np(x, dtype=np.float64) -> np.ndarray:
    if hasattr(x, "detach"):  # torch tensor, without importing torch
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, dtype=dtype)


def _categorical_target(target, n: int, k: int) -> np.ndarray:
    t = np.asarray(target.detach().cpu().numpy() if hasattr(target, "detach") else target)
    if t.ndim == 1:
        if not np.issubdtype(t.dtype, np.integer):
            raise TypeError("1-D categorical targets must be integer class indices")
        if t.shape[0] != n or t.min() < 0 or t.max() >= k:
            raise ValueError("class index targets must have shape [N] with values in [0, K)")
        onehot = np.zeros((n, k))
        onehot[np.arange(n), t] = 1.0
        return onehot
    t = t.astype(np.float64)
    if t.shape != (n, k):
        raise ValueError(f"soft targets must have shape {(n, k)}, got {t.shape}")
    if (t < 0).any() or not np.allclose(t.sum(1), 1.0, atol=1e-4):
        raise ValueError("soft targets must be non-negative and sum to 1")
    return t


def _check_probs(probs: np.ndarray) -> None:
    if probs.ndim != 2 or probs.shape[0] == 0:
        raise ValueError("probs must have shape [N, K] with N > 0")
    if (probs < -1e-6).any() or not np.allclose(probs.sum(1), 1.0, atol=1e-4):
        raise ValueError("each row of probs must be a distribution")


# ------------------------------------------------------------------------------- calibration


def reliability_bins(confidence, correct, n_bins: int = DEFAULT_BINS) -> list[dict]:
    """Equal-mass bins of (confidence, correctness). One dict per non-empty bin."""
    conf, corr = _np(confidence), _np(correct)
    if conf.shape != corr.shape or conf.ndim != 1 or conf.size == 0:
        raise ValueError("confidence and correct must be equal-length non-empty 1-D arrays")
    order = np.argsort(conf, kind="stable")
    sorted_conf = conf[order]
    # Nominal equal-mass bin per sorted position, then move every tie group into the bin of its
    # first member, so identical confidences are never split (otherwise ECE depends on input order).
    nominal = np.concatenate([np.full(len(c), i) for i, c in enumerate(np.array_split(order, min(n_bins, conf.size)))])
    _, first, inverse = np.unique(sorted_conf, return_index=True, return_inverse=True)
    bin_of = nominal[first][inverse]

    bins = []
    for b in np.unique(bin_of):
        idx = order[bin_of == b]
        bins.append(
            {
                "count": int(idx.size),
                "conf_min": float(conf[idx].min()),
                "conf_max": float(conf[idx].max()),
                "mean_confidence": float(conf[idx].mean()),
                "mean_correct": float(corr[idx].mean()),
            }
        )
    return bins


def ece_mce(confidence, correct, n_bins: int = DEFAULT_BINS) -> tuple[float, float]:
    """Expected and maximum calibration error over equal-mass bins."""
    bins = reliability_bins(confidence, correct, n_bins)
    total = sum(b["count"] for b in bins)
    gaps = [abs(b["mean_confidence"] - b["mean_correct"]) for b in bins]
    ece = sum(b["count"] / total * g for b, g in zip(bins, gaps))
    return float(ece), float(max(gaps))


# ------------------------------------------------------------------------- selective prediction


def risk_coverage(confidence, correct) -> dict:
    """Risk–coverage curve: answer the most confident examples first.

    Returns ``coverage`` / ``risk`` arrays (one point per accepted example), ``aurc``,
    ``eaurc`` (AURC minus the best AURC achievable for the same correctness values)
    and selective accuracy at the coverages in ``SELECTIVE_COVERAGES``.
    Tied confidences are handled as the expectation over their possible orderings.
    """
    conf, corr = _np(confidence), _np(correct)
    if conf.shape != corr.shape or conf.ndim != 1 or conf.size == 0:
        raise ValueError("confidence and correct must be equal-length non-empty 1-D arrays")
    n = conf.size
    ranks = np.arange(1, n + 1)

    def _risks(ordered_correct: np.ndarray) -> np.ndarray:
        return 1.0 - np.cumsum(ordered_correct) / ranks

    order = np.argsort(-conf, kind="stable")
    ordered = corr[order]
    # Within a group of tied confidences the order is arbitrary: replace it by the group mean,
    # which gives the expected curve over all orderings of the ties.
    _, group, counts = np.unique(conf[order], return_inverse=True, return_counts=True)
    group_mean = np.bincount(group, weights=ordered) / counts
    risk = _risks(group_mean[group])
    optimal = _risks(np.sort(corr)[::-1])
    out = {
        "coverage": ranks / n,
        "risk": risk,
        "aurc": float(risk.mean()),
        "eaurc": float(risk.mean() - optimal.mean()),
    }
    for cov in SELECTIVE_COVERAGES:
        k = max(1, math.ceil(cov * n))
        out[f"selective_acc@{int(cov * 100)}"] = float(1.0 - risk[k - 1])
    return out


# ---------------------------------------------------------------------------- per primitive


def _summary(n: int, accuracy: float, nll: float, brier: float, conf, correct, confidence, n_bins: int) -> dict:
    """Shared tail of every per-primitive report. ``conf`` drives calibration, ``confidence`` drives selection."""
    ece, mce = ece_mce(conf, correct, n_bins)
    rc = risk_coverage(conf if confidence is None else confidence, correct)
    return {
        "n": n,
        "accuracy": accuracy,
        "nll": nll,
        "brier": brier,
        "ece": ece,
        "mce": mce,
        "aurc": rc["aurc"],
        "eaurc": rc["eaurc"],
        **{k: v for k, v in rc.items() if k.startswith("selective_acc@")},
        "reliability": reliability_bins(conf, correct, n_bins),
    }


def categorical_metrics(probs, target, confidence=None, n_bins: int = DEFAULT_BINS) -> dict:
    """Metrics for Choice (and the categorical part of Score).

    ``confidence`` is an optional separate confidence signal (TODO 2.8) used only for the
    risk–coverage metrics; by default the top probability is used. Calibration (ECE/MCE) is
    always measured on the top probability.
    Brier is the multi-class Brier score ``Σ_k (p_k − t_k)²``, in ``[0, 2]``.
    """
    p = _np(probs)
    _check_probs(p)
    n, k = p.shape
    t = _categorical_target(target, n, k)

    pred = p.argmax(1)
    top_prob = p[np.arange(n), pred]
    correct = t[np.arange(n), pred]
    nll = -(t * np.log(np.clip(p, EPS, 1.0))).sum(1)
    brier = ((p - t) ** 2).sum(1)
    return _summary(n, float(correct.mean()), float(nll.mean()), float(brier.mean()), top_prob, correct, confidence, n_bins)


def score_metrics(probs, target, confidence=None, n_bins: int = DEFAULT_BINS) -> dict:
    """Categorical metrics plus ordinal errors, in level units.

    ``mae``: |argmax level − target level|; ``mae_expected``: |expected level − target level|.
    For soft targets the target level is the target's expected level.
    """
    out = categorical_metrics(probs, target, confidence, n_bins)
    p = _np(probs)
    t = _categorical_target(target, *p.shape)
    levels = np.arange(p.shape[1])
    target_level = t @ levels
    out["mae"] = float(np.abs(p.argmax(1) - target_level).mean())
    out["mae_expected"] = float(np.abs(p @ levels - target_level).mean())
    return out


def bool_metrics(prob, target, confidence=None, n_bins: int = DEFAULT_BINS) -> dict:
    """Metrics for Bool. ``prob`` is P(yes) ``[N]``; ``target`` is ``[N]`` in [0, 1] (soft allowed).

    ``ece`` is calibration of P(yes) itself: bins by P(yes), compares with the mean target —
    this is the number callers consume. ``ece_top`` is top-label ECE (confidence
    ``max(p, 1 − p)`` vs correctness of the yes/no decision), comparable with Choice.
    """
    p, t = _np(prob), _np(target)
    if p.ndim != 1 or p.shape != t.shape or p.size == 0:
        raise ValueError("prob and target must be equal-length non-empty 1-D arrays")
    if (p < 0).any() or (p > 1).any() or (t < 0).any() or (t > 1).any():
        raise ValueError("prob and target must lie in [0, 1]")

    yes = p >= 0.5
    correct = np.where(yes, t, 1.0 - t)
    top = np.where(yes, p, 1.0 - p)
    pc = np.clip(p, EPS, 1.0 - EPS)
    nll = -(t * np.log(pc) + (1.0 - t) * np.log(1.0 - pc))

    out = _summary(p.size, float(correct.mean()), float(nll.mean()), float(((p - t) ** 2).mean()), top, correct, confidence, n_bins)
    out["ece_top"], out["mce_top"] = out.pop("ece"), out.pop("mce")
    out["reliability_top"] = out.pop("reliability")
    out["ece"], out["mce"] = ece_mce(p, t, n_bins)
    out["reliability"] = reliability_bins(p, t, n_bins)
    return out


# ------------------------------------------------------------------------ agreement (TODO 2.13)


def cohen_kappa(pred, target, weights: str | None = None, n_labels: int | None = None) -> float:
    """Cohen's κ between hard predictions and hard targets (class indices).

    ``weights="quadratic"`` gives weighted κ for ordinal labels (Score). NaN when κ is undefined,
    e.g. when both sides use a single label.
    """
    from sklearn.metrics import cohen_kappa_score

    p, t = np.asarray(pred).astype(int), np.asarray(target).astype(int)
    if p.shape != t.shape or p.ndim != 1 or p.size == 0:
        raise ValueError("pred and target must be equal-length non-empty 1-D arrays")
    labels = list(range(n_labels)) if n_labels else None
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.simplefilter("ignore")  # undefined κ (a single label) is reported as NaN
        k = cohen_kappa_score(t, p, weights=weights, labels=labels)
    return float(k)


def spearman(a, b) -> float:
    """Spearman rank correlation; NaN when either side is constant."""
    from scipy.stats import spearmanr

    a, b = _np(a), _np(b)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("a and b must be equal-length 1-D arrays with at least 2 values")
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return float(spearmanr(a, b).statistic)


def bootstrap_ci(fn, *arrays, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI of ``fn(*arrays)``, resampling rows of all arrays together.

    Resamples where ``fn`` is undefined (NaN) are dropped.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = arrays[0].shape[0]
    if n == 0 or any(a.shape[0] != n for a in arrays):
        raise ValueError("arrays must be non-empty and share the first dimension")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        value = fn(*(a[idx] for a in arrays))
        if not math.isnan(value):
            stats.append(value)
    if not stats:
        return float("nan"), float("nan")
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
