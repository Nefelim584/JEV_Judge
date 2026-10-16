import math

import numpy as np

from jevlite.data.unified import Prediction, Record
from jevlite.judge_bench import auroc, balanced_accuracy, bool_summary, pairwise_summary


def test_auroc_matches_pairwise_definition():
    rng = np.random.default_rng(0)
    s, t = rng.integers(0, 5, 60).astype(float), rng.integers(0, 2, 60)
    pos, neg = s[t == 1], s[t == 0]
    brute = np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])
    assert math.isclose(auroc(s, t), brute)
    assert auroc([0.1, 0.9], [0, 1]) == 1.0 and math.isnan(auroc([0.3, 0.4], [1, 1]))


def test_balanced_accuracy_ignores_class_skew():
    # always "yes" on 9 positives + 1 negative: accuracy 0.9, balanced 0.5
    assert balanced_accuracy([True] * 10, [1] * 9 + [0]) == 0.5
    assert balanced_accuracy([True, False], [1, 0]) == 1.0


def _bool(i, target):
    return Record(id=f"b{i}", source="s", split="test_ood", state="x", type="bool", prompt="p", target=target)


def test_bool_summary():
    recs = [_bool(i, t) for i, t in enumerate([1.0, 1.0, 0.0, 0.0])]
    preds = [Prediction(id=r.id, prob=p) for r, p in zip(recs, [0.9, 0.4, 0.3, 0.6])]
    out = bool_summary(recs, preds)
    assert out["accuracy"] == 0.5 and out["balanced_accuracy"] == 0.5 and out["auroc"] == 0.75


def _pair(g, order, target):
    return Record(id=f"{g}:{order}", source="cjb", split="test_ood", state="x", type="choice", prompt="p",
                  candidates=["response 1", "response 2"], target=target, group=g, order=order)


def test_pairwise_consistent_accuracy_and_position_bias():
    # pair a: right in both orders; pair b: always picks response 1, so right only in order 0
    recs = [_pair("a", 0, 0), _pair("a", 1, 1), _pair("b", 0, 0), _pair("b", 1, 1), _pair("c", 0, 0)]
    probs = [[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.6, 0.4], [0.9, 0.1]]
    out = pairwise_summary(recs, [Prediction(id=r.id, probs=p) for r, p in zip(recs, probs)])
    assert out["n_pairs"] == 2  # c has one order only
    assert out["consistent_accuracy"] == 0.5 and out["accuracy"] == 0.75
    assert out["accuracy_order0"] == 1.0 and out["accuracy_order1"] == 0.5
