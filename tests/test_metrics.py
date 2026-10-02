import numpy as np
import pytest
import torch
from sklearn.metrics import brier_score_loss, log_loss

from jevlite.metrics import (
    bool_metrics,
    categorical_metrics,
    ece_mce,
    reliability_bins,
    risk_coverage,
    score_metrics,
)

rng = np.random.default_rng(0)


def _random_probs(n, k):
    logits = rng.normal(size=(n, k))
    e = np.exp(logits)
    return e / e.sum(1, keepdims=True)


# ------------------------------------------------------------------ calibration


def test_equal_mass_bins():
    conf = np.linspace(0, 1, 30)
    bins = reliability_bins(conf, np.ones(30), n_bins=15)
    assert len(bins) == 15 and all(b["count"] == 2 for b in bins)
    assert bins[0]["conf_min"] == 0.0 and bins[-1]["conf_max"] == 1.0


def test_fewer_examples_than_bins():
    assert len(reliability_bins([0.2, 0.9], [0, 1], n_bins=15)) == 2


def test_ece_known_value():
    # Always 80% confident, right half of the time.
    ece, mce = ece_mce(np.full(100, 0.8), np.tile([0, 1], 50))
    assert ece == pytest.approx(0.3)
    assert mce == pytest.approx(0.3)


def test_ece_zero_when_calibrated_per_bin():
    conf = np.repeat([0.25, 0.75], 4)
    correct = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    ece, mce = ece_mce(conf, correct, n_bins=2)
    assert ece == pytest.approx(0.0) and mce == pytest.approx(0.0)


def test_mce_is_worst_bin():
    conf = np.array([0.9, 0.9, 0.6, 0.6])
    correct = np.array([1, 1, 0, 0])  # bin(0.6) gap 0.6, bin(0.9) gap 0.1
    ece, mce = ece_mce(conf, correct, n_bins=2)
    assert mce == pytest.approx(0.6)
    assert ece == pytest.approx(0.35)


# ------------------------------------------------------------- risk–coverage


def test_risk_coverage_known_values():
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    correct = np.array([1, 0, 1, 1])
    rc = risk_coverage(conf, correct)
    np.testing.assert_allclose(rc["risk"], [0, 0.5, 1 / 3, 0.25])
    assert rc["aurc"] == pytest.approx((0 + 0.5 + 1 / 3 + 0.25) / 4)
    # Best ordering of the same outcomes: 1,1,1,0 -> risks 0,0,0,0.25
    assert rc["eaurc"] == pytest.approx(rc["aurc"] - 0.25 / 4)
    assert rc["selective_acc@80"] == pytest.approx(1 - 0.25)  # ceil(3.2) = 4 accepted
    assert rc["selective_acc@90"] == pytest.approx(1 - 0.25)


def test_perfect_ranking_has_zero_eaurc():
    correct = rng.integers(0, 2, 200)
    conf = correct + rng.uniform(0, 0.5, 200)
    assert risk_coverage(conf, correct)["eaurc"] == pytest.approx(0.0)


def test_selective_accuracy_improves_with_good_confidence():
    correct = rng.integers(0, 2, 1000)
    conf = 0.5 * correct + rng.uniform(0, 0.6, 1000)
    rc = risk_coverage(conf, correct)
    assert rc["selective_acc@80"] > correct.mean()


# --------------------------------------------------------------- categorical


def test_perfect_categorical():
    m = categorical_metrics(np.eye(4), np.arange(4))
    assert m["accuracy"] == 1.0 and m["brier"] == 0.0 and m["ece"] == 0.0
    assert m["nll"] == pytest.approx(0.0, abs=1e-9)


def test_categorical_matches_sklearn():
    p = _random_probs(300, 5)
    y = rng.integers(0, 5, 300)
    m = categorical_metrics(p, y)
    assert m["nll"] == pytest.approx(log_loss(y, p, labels=range(5)))
    assert m["accuracy"] == pytest.approx((p.argmax(1) == y).mean())
    onehot = np.eye(5)[y]
    assert m["brier"] == pytest.approx(((p - onehot) ** 2).sum(1).mean())


def test_index_and_onehot_targets_agree():
    p = _random_probs(50, 3)
    y = rng.integers(0, 3, 50)
    a, b = categorical_metrics(p, y), categorical_metrics(p, np.eye(3)[y])
    assert {k: v for k, v in a.items() if k != "reliability"} == {k: v for k, v in b.items() if k != "reliability"}


def test_padded_candidates_do_not_change_metrics():
    p = _random_probs(40, 3)
    y = rng.integers(0, 3, 40)
    padded = np.concatenate([p, np.zeros((40, 2))], axis=1)
    a, b = categorical_metrics(p, y), categorical_metrics(padded, y)
    for key in ("accuracy", "nll", "brier", "ece", "aurc"):
        assert a[key] == pytest.approx(b[key])


def test_uniform_soft_target_gives_expected_correctness():
    k = 4
    m = categorical_metrics(np.full((10, k), 1 / k), np.full((10, k), 1 / k))
    assert m["accuracy"] == pytest.approx(1 / k)
    assert m["nll"] == pytest.approx(np.log(k))
    assert m["ece"] == pytest.approx(0.0)  # predicting uniform on uniform targets is calibrated


def test_separate_confidence_only_affects_selection():
    p = _random_probs(100, 3)
    y = rng.integers(0, 3, 100)
    a = categorical_metrics(p, y)
    b = categorical_metrics(p, y, confidence=rng.uniform(size=100))
    assert a["ece"] == b["ece"] and a["nll"] == b["nll"]
    assert a["aurc"] != b["aurc"]


def test_torch_inputs():
    p = torch.softmax(torch.tensor([[2.0, 0.0, float("-inf")], [0.0, 1.0, float("-inf")]]), -1)
    m = categorical_metrics(p, torch.tensor([0, 1]))
    assert m["accuracy"] == 1.0


@pytest.mark.parametrize(
    "probs, target",
    [
        (np.array([[0.5, 0.6]]), [0]),  # not a distribution
        (np.array([[0.5, 0.5]]), [2]),  # index out of range
        (np.array([[0.5, 0.5]]), np.array([0.0])),  # float 1-D target
        (np.array([[0.5, 0.5]]), np.array([[0.7, 0.7]])),  # soft target not a distribution
        (np.zeros((0, 2)), np.array([], dtype=int)),  # empty
    ],
)
def test_categorical_validation(probs, target):
    with pytest.raises((ValueError, TypeError)):
        categorical_metrics(probs, target)


# --------------------------------------------------------------------- score


def test_score_mae():
    p = np.array([[0.1, 0.2, 0.7], [0.6, 0.4, 0.0]])
    m = score_metrics(p, np.array([0, 0]))
    assert m["mae"] == pytest.approx((2 + 0) / 2)
    assert m["mae_expected"] == pytest.approx((1.6 + 0.4) / 2)


def test_score_soft_target_level():
    p = np.array([[0.0, 1.0, 0.0]])
    m = score_metrics(p, np.array([[0.5, 0.0, 0.5]]))  # target level 1.0
    assert m["mae"] == 0.0 and m["mae_expected"] == 0.0


# ---------------------------------------------------------------------- bool


def test_bool_matches_sklearn():
    p = rng.uniform(0.01, 0.99, 500)
    y = (rng.uniform(size=500) < p).astype(float)
    m = bool_metrics(p, y)
    assert m["nll"] == pytest.approx(log_loss(y, p))
    assert m["brier"] == pytest.approx(brier_score_loss(y, p))
    assert m["accuracy"] == pytest.approx(((p >= 0.5) == y).mean())


def test_bool_calibrated_probabilities_have_low_ece():
    p = rng.uniform(0, 1, 20000)
    y = (rng.uniform(size=20000) < p).astype(float)
    m = bool_metrics(p, y)
    assert m["ece"] < 0.02 and m["ece_top"] < 0.02


def test_bool_ece_vs_top_label_ece():
    # Always says P(yes)=0.7, half the targets are yes.
    p, y = np.full(100, 0.7), np.tile([0.0, 1.0], 50)
    m = bool_metrics(p, y)
    assert m["ece"] == pytest.approx(0.2)  # mean P(yes) 0.7 vs base rate 0.5
    assert m["ece_top"] == pytest.approx(0.2)  # top confidence 0.7 vs accuracy 0.5
    # Symmetric miscalibration: P(yes)=0.9 on yes-cases and 0.1 on no-cases, but only 60% right each way.
    p = np.concatenate([np.full(50, 0.9), np.full(50, 0.1)])
    y = np.concatenate([np.r_[np.ones(30), np.zeros(20)], np.r_[np.zeros(30), np.ones(20)]])
    m = bool_metrics(p, y)
    assert m["ece_top"] == pytest.approx(0.3)
    assert m["ece"] == pytest.approx(0.3)


def test_bool_soft_targets():
    m = bool_metrics(np.full(10, 0.5), np.full(10, 0.5))
    assert m["accuracy"] == 0.5 and m["brier"] == 0.0 and m["ece"] == pytest.approx(0.0)
    assert m["nll"] == pytest.approx(np.log(2))


def test_bool_report_keys():
    m = bool_metrics(rng.uniform(size=50), rng.integers(0, 2, 50).astype(float))
    for key in ("ece", "mce", "ece_top", "mce_top", "reliability", "reliability_top", "aurc", "selective_acc@90"):
        assert key in m


@pytest.mark.parametrize("prob, target", [([1.2], [1.0]), ([0.5], [2.0]), ([0.5, 0.5], [1.0]), ([], [])])
def test_bool_validation(prob, target):
    with pytest.raises(ValueError):
        bool_metrics(np.array(prob), np.array(target))


def test_ties_never_split_and_order_does_not_matter():
    conf = np.repeat([0.3, 0.6, 0.9], [7, 11, 5])
    correct = rng.integers(0, 2, conf.size)
    bins = reliability_bins(conf, correct, n_bins=15)
    assert [b["count"] for b in bins] == [7, 11, 5]
    perm = rng.permutation(conf.size)
    assert ece_mce(conf, correct) == pytest.approx(ece_mce(conf[perm], correct[perm]))


def test_risk_coverage_ties_use_expected_order():
    # All tied: the curve is flat at the overall error rate, whatever the input order.
    correct = np.array([1, 0, 1, 1])
    rc = risk_coverage(np.full(4, 0.5), correct)
    np.testing.assert_allclose(rc["risk"], 0.25)
    perm = rng.permutation(4)
    assert risk_coverage(np.full(4, 0.5), correct[perm])["aurc"] == pytest.approx(rc["aurc"])


# ---------------------------------------------------------------------------- agreement


def test_cohen_kappa():
    from jevlite.metrics import cohen_kappa

    assert cohen_kappa([0, 1, 2, 1], [0, 1, 2, 1]) == pytest.approx(1.0)
    # Quadratic weights punish far errors more than near ones.
    near = cohen_kappa([0, 1, 2, 3, 3], [0, 1, 2, 2, 3], weights="quadratic", n_labels=4)
    far = cohen_kappa([0, 1, 2, 0, 3], [0, 1, 2, 3, 3], weights="quadratic", n_labels=4)
    assert near > far
    assert np.isnan(cohen_kappa([1, 1], [1, 1]))


def test_spearman():
    from jevlite.metrics import spearman

    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert np.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_bootstrap_ci():
    from jevlite.metrics import bootstrap_ci

    x = np.random.default_rng(0).normal(size=500)
    lo, hi = bootstrap_ci(np.mean, x, n_boot=300)
    assert lo < x.mean() < hi and hi - lo < 0.3
    assert all(np.isnan(bootstrap_ci(lambda a: float("nan"), x, n_boot=5)))
