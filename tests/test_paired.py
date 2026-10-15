import math

import numpy as np

from jevlite.data.unified import Prediction, Record
from jevlite.paired import holm, mcnemar_exact, paired_compare, record_scores, to_markdown


def _bool(i, target, src="a"):
    return Record(id=f"r{i}", source=src, split="test_in", state="s", type="bool", prompt="p", target=target)


def test_record_scores():
    ok, nll = record_scores(_bool(0, 1.0), Prediction(id="r0", prob=0.8))
    assert ok and math.isclose(nll, -math.log(0.8))
    c = Record(id="c", source="a", split="test_in", state="s", type="choice", prompt="p", candidates=["x", "y", "z"], target=2)
    ok, nll = record_scores(c, Prediction(id="c", probs=[0.2, 0.2, 0.6]))
    assert ok and math.isclose(nll, -math.log(0.6))


def test_mcnemar_and_holm():
    assert mcnemar_exact(0, 0) == 1.0
    assert math.isclose(mcnemar_exact(5, 5), 1.0)
    assert mcnemar_exact(20, 2) < 0.001
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_paired_compare_detects_a_real_difference_and_calls_noise_a_tie():
    rng = np.random.default_rng(0)
    recs = [_bool(i, float(i % 2), src="big" if i < 600 else "small") for i in range(640)]
    truth = {r.id: r.target for r in recs}
    good = {k: Prediction(id=k, prob=0.9 if t else 0.1) for k, t in truth.items()}
    # the reference is wrong on 20% of "big" and right on all of "small"
    ref = {k: Prediction(id=k, prob=(0.1 if t else 0.9) if (int(k[1:]) < 600 and rng.random() < 0.2) else (0.9 if t else 0.1)) for k, t in truth.items()}
    rows = paired_compare(recs, good, ref, slices=((), ("source",)), n_boot=500)
    by = {(r["slicing"], r["cell"]): r for r in rows}
    assert by[("all", "all")]["verdict"] == "win" and by[("source", "big")]["d_acc"] > 0.1
    assert by[("source", "small")]["verdict"] == "tie" and by[("source", "small")]["d_acc"] == 0
    assert "| big | bool |" in to_markdown(rows, "m", "r")
    assert not paired_compare(recs[:5], good, ref, min_n=20)  # cells below min_n are dropped
