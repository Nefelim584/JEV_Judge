"""Post-hoc calibration on saved predictions (todo 2.7): no GPU, the model is not touched.

Two layers, applied in this order:

- **model**: a temperature per primitive. Choice gets a smooth ``T(K)``, ``log T = a + b·log K``
  (K = number of options; a single T rarely fits both K = 2 and K = 16, and buckets make probabilities
  jump at their edges); Bool and Score get ``log T = a``. Fitted by NLL on the ``calib`` splits of the
  mixes. Dividing logits by T keeps every argmax and the Bool 0.5 threshold: only confidence changes.
- **judge**: Platt scaling of the Bool logit, ``sigmoid(scale·z + bias)``, for the judge's
  faithfulness claims. Fitted on held-out RAG answers (``ragbench_calib``, todo Phase 4b, 2026-10-30)
  on top of the model layer. Unlike a temperature, the bias moves the decision threshold (Stage B says
  "supported" too often on real RAG answers). By default it is fitted **class-balanced** (both labels
  weigh the same in the NLL): the result is P(supported) at an even prior, so 0.5 is the threshold for
  balanced accuracy and the fit does not learn the 89% "supported" share of RAGBench, which differs per
  benchmark (HaluBench 50%, FaithBench 33%) and in production. The prior of a target set goes back in
  as a shift of the logit by ``log(prior / (1 − prior))`` once it is known (Phase 11).

Predictions carry probabilities, not logits. ``log p`` is a logit up to a per-question constant, which
softmax ignores; the Bool logit is ``log(p / (1 − p))``. Probabilities are clipped to
``[EPS, 1 − EPS]`` first, so a saturated probability stays finite.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import minimize

from .baselines.base import entropy_confidence
from .data.unified import Prediction, Record

EPS = 1e-7
LAYERS = ("model", "judge")


def _soft_target(r: Record, k: int) -> np.ndarray:
    if isinstance(r.target, list):
        t = np.asarray(r.target, dtype=float)
        return t / t.sum()
    t = np.zeros(k)
    t[int(r.target)] = 1.0
    return t


def bool_logit(p: float) -> float:
    p = float(np.clip(p, EPS, 1.0 - EPS))
    return float(np.log(p) - np.log1p(-p))


def option_logits(probs: Sequence[float]) -> np.ndarray:
    return np.log(np.clip(np.asarray(probs, dtype=float), EPS, 1.0))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _bool_nll(z: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    # −[y log σ(z) + (1 − y) log σ(−z)] = softplus(z) − y·z, stable for large |z|
    return float(np.average(np.logaddexp(0.0, z) - y * z, weights=w))


def _pairs(records: Iterable[Record], preds: dict[str, Prediction], prim: str) -> list[tuple[Record, Prediction]]:
    return [(r, preds[r.id]) for r in records if r.type == prim and r.id in preds]


def fit_model_layer(records: Sequence[Record], preds: dict[str, Prediction]) -> dict:
    """``{"bool": {"a"}, "choice": {"a", "b"}, "score": {"a"}, "n": {...}}``; a primitive without
    records keeps T = 1 (``a = 0``)."""
    out: dict = {"n": {}}

    pairs = _pairs(records, preds, "bool")
    z = np.array([bool_logit(p.prob) for _, p in pairs])
    y = np.array([float(r.target) for r, _ in pairs])
    out["n"]["bool"] = len(pairs)
    out["bool"] = {"a": float(minimize(lambda a: _bool_nll(z / np.exp(a[0]), y), [0.0], method="L-BFGS-B").x[0])
                   if len(pairs) else 0.0}

    for prim, smooth in (("choice", True), ("score", False)):
        pairs = _pairs(records, preds, prim)
        out["n"][prim] = len(pairs)
        groups: dict[int, tuple[list, list]] = defaultdict(lambda: ([], []))
        for r, p in pairs:
            k = len(p.probs)
            groups[k][0].append(option_logits(p.probs))
            groups[k][1].append(_soft_target(r, k))
        stacked = [(k, np.stack(z), np.stack(t)) for k, (z, t) in groups.items()]
        n = sum(len(z) for _, z, _ in stacked)

        def nll(x: np.ndarray) -> float:
            a, b = x[0], (x[1] if smooth else 0.0)
            total = 0.0
            for k, z, t in stacked:
                logp = z / np.exp(a + b * np.log(k))
                logp = logp - np.logaddexp.reduce(logp, axis=1, keepdims=True)
                total -= float((t * logp).sum())
            return total / n

        if not n:
            out[prim] = {"a": 0.0, "b": 0.0} if smooth else {"a": 0.0}
            continue
        x = minimize(nll, [0.0, 0.0] if smooth else [0.0], method="L-BFGS-B").x
        out[prim] = {"a": float(x[0]), "b": float(x[1])} if smooth else {"a": float(x[0])}
    return out


def temperature(model_layer: dict, prim: str, k: int | None = None) -> float:
    c = model_layer[prim]
    return float(np.exp(c["a"] + c.get("b", 0.0) * np.log(k or 1)))


def fit_judge_layer(
    records: Sequence[Record], preds: dict[str, Prediction], model_layer: dict | None, balanced: bool = True
) -> dict:
    """Platt ``{"scale", "bias", "balanced", "n", "positive_rate"}`` on the Bool logits after the model
    layer; ``balanced`` weighs each label by the inverse of its share."""
    pairs = _pairs(records, preds, "bool")
    t = temperature(model_layer, "bool") if model_layer else 1.0
    z = np.array([bool_logit(p.prob) for _, p in pairs]) / t
    y = np.array([float(r.target) for r, _ in pairs])
    pos = float(y.mean())
    w = np.where(y >= 0.5, 0.5 / pos, 0.5 / (1.0 - pos)) if balanced else None
    x = minimize(lambda v: _bool_nll(v[0] * z + v[1], y, w), [1.0, 0.0], method="L-BFGS-B").x
    return {"scale": float(x[0]), "bias": float(x[1]), "balanced": balanced, "n": len(pairs), "positive_rate": pos}


def apply(prim: str, p: Prediction, calibration: dict, layers: Sequence[str] = LAYERS) -> Prediction:
    """``p`` (a prediction for a question of type ``prim``) with calibrated probabilities and confidence;
    other fields (latency) are kept. The judge layer only touches Bool."""
    model, judge = calibration.get("model"), calibration.get("judge")
    if prim == "bool":
        z = bool_logit(p.prob)
        if "model" in layers and model:
            z /= temperature(model, "bool")
        if "judge" in layers and judge:
            z = judge["scale"] * z + judge["bias"]
        prob = float(_sigmoid(np.array(z)))
        return p.model_copy(update={"prob": prob, "confidence": entropy_confidence(np.array([prob]))})
    if "model" not in layers or not model:
        return p
    probs = _softmax(option_logits(p.probs) / temperature(model, prim, len(p.probs)))
    return p.model_copy(update={"probs": probs.tolist(), "confidence": entropy_confidence(probs)})
