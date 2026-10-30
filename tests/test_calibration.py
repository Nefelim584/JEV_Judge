import math

import numpy as np

from jevlite.calibration import _softmax, apply, bool_logit, fit_judge_layer, fit_model_layer, temperature
from jevlite.data.public import CONVERTERS
from jevlite.data.unified import Prediction, Record


def _bool(i, target):
    return Record(id=f"b{i}", source="s", split="calib", state="x", type="bool", prompt="p", target=target)


def _choice(i, k, target):
    return Record(id=f"c{i}", source="s", split="calib", state="x", type="choice", prompt="p",
                  candidates=[f"o{j}" for j in range(k)], target=target)


def _overconfident_choice(n=600, seed=0):
    """True probabilities from logits z; the model reports softmax(3·z): T = 3 recovers them, for any K."""
    rng = np.random.default_rng(seed)
    records, preds = [], {}
    for i in range(n):
        k = int(rng.choice([2, 4, 8]))
        z = rng.normal(0, 1, k)
        y = int(rng.choice(k, p=_softmax(z)))
        records.append(_choice(i, k, y))
        preds[f"c{i}"] = Prediction(id=f"c{i}", probs=_softmax(3 * z).tolist())
    return records, preds


def test_model_layer_recovers_temperature_and_keeps_argmax():
    records, preds = _overconfident_choice()
    layer = fit_model_layer(records, preds)
    for k in (2, 4, 8):
        assert 2.2 < temperature(layer, "choice", k) < 4.0
    assert layer["n"] == {"bool": 0, "choice": 600, "score": 0} and layer["bool"]["a"] == 0.0
    cal = {"model": layer}
    for r in records[:50]:
        before, after = preds[r.id], apply("choice", preds[r.id], cal)
        assert np.argmax(after.probs) == np.argmax(before.probs)
        assert max(after.probs) < max(before.probs) and math.isclose(sum(after.probs), 1.0)


def test_judge_layer_moves_the_threshold_and_balances_classes():
    # a "lenient" model: the true logit is z − 2, the model reports z; 80% positives
    rng = np.random.default_rng(1)
    records, preds = [], {}
    for i in range(3000):
        y = float(rng.random() < 0.8)
        z = rng.normal(1.5 if y else -1.5, 1.0) + 2.0
        records.append(_bool(i, y))
        preds[f"b{i}"] = Prediction(id=f"b{i}", prob=float(1 / (1 + np.exp(-z))))
    plain = fit_judge_layer(records, preds, None, balanced=False)
    balanced = fit_judge_layer(records, preds, None)
    assert balanced["balanced"] and math.isclose(balanced["positive_rate"], 0.8, abs_tol=0.03)
    # the class-balanced fit leaves the 80% prior out: its bias is lower by about log(0.8 / 0.2)
    assert math.isclose(plain["bias"] - balanced["bias"], math.log(4), abs_tol=0.3)
    cal = {"judge": balanced}
    y = np.array([r.target for r in records])
    p = np.array([apply("bool", preds[r.id], cal).prob for r in records])
    recall_neg = np.mean(p[y == 0] < 0.5)
    raw_recall_neg = np.mean(np.array([preds[r.id].prob for r in records])[y == 0] < 0.5)
    assert raw_recall_neg < 0.6 and recall_neg > 0.85


def test_layers_are_selectable_and_latency_is_kept():
    cal = {"model": {"bool": {"a": math.log(2)}, "choice": {"a": 0.0, "b": 0.0}, "score": {"a": 0.0}},
           "judge": {"scale": 1.0, "bias": -1.0}}
    p = Prediction(id="x", prob=0.9, latency_ms=12.0)
    z = bool_logit(0.9)
    model_only = apply("bool", p, cal, ("model",))
    both = apply("bool", p, cal)
    assert math.isclose(bool_logit(model_only.prob), z / 2, rel_tol=1e-6)
    assert math.isclose(bool_logit(both.prob), z / 2 - 1.0, rel_tol=1e-6)
    assert both.latency_ms == 12.0
    q = Prediction(id="y", probs=[0.7, 0.3])
    assert apply("choice", q, cal, ("judge",)) is q  # the judge layer leaves Choice alone


def test_ragbench_calib_converter():
    conv = CONVERTERS["ragbench_calib"]()
    row = {"documents": ["Doc one.", " "], "response": "An answer.", "question": "Q?", "adherence_score": True,
           "subset": "covidqa", "idx": 3, "generation_model_name": "m"}
    (rec,) = conv.convert(row, "validation")
    assert rec.split == "calib" and rec.source == "ragbench_calib" and rec.domain == "ragbench/covidqa"
    assert rec.id == "ragbench_calib:validation:covidqa:3" and rec.state == ["Doc one."] and rec.target == 1.0
    assert set(conv.SUBSETS) == {"covidqa", "cuad", "delucionqa", "emanual"}
    assert "msmarco" not in conv.SUBSETS
    # the eval converter still sends the same row to test_ood
    (test,) = CONVERTERS["ragbench"]().convert(row, "test")
    assert test.split == "test_ood" and test.id != rec.id
