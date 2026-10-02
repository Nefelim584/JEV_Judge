import numpy as np
import pytest

from jevlite.data.unified import Record
from jevlite.probes import iia_violation, nonsense_confidence, order_sensitivity, run_probes
from jevlite.schema import BoolQuestion, ChoiceQuestion


class IndependentScorer:
    """Scores every option on its own (like pair mode): order-invariant and IIA by construction."""

    name = "independent"

    def predict(self, state, questions):
        out = []
        for q in questions:
            if q.type == "bool":
                out.append(np.array([0.9 if "rain" in str(state) else 0.5]))
            else:
                s = np.array([len(c) for c in q.candidates], dtype=float)
                e = np.exp(s - s.max())
                out.append(e / e.sum())
        return out


class FirstPositionBias(IndependentScorer):
    """Adds a bonus to whatever option comes first."""

    name = "biased"

    def predict(self, state, questions):
        out = []
        for q, p in zip(questions, super().predict(state, questions)):
            if q.type != "bool":
                p = p.copy()
                p[0] += 1.0
                p /= p.sum()
            out.append(p)
        return out


Q = ChoiceQuestion(id="q", prompt="p", options=["a", "bbb", "cc", "dddd"])


def test_order_sensitivity():
    assert order_sensitivity(IndependentScorer(), "s", Q)["mean_spread"] == pytest.approx(0.0)
    biased = order_sensitivity(FirstPositionBias(), "s", Q, n_perms=10)
    assert biased["mean_spread"] > 0.1 and biased["argmax_flip_rate"] > 0


def test_iia():
    assert iia_violation(IndependentScorer(), "s", Q, "eeeee")["mean_abs_log_ratio_change"] == pytest.approx(0.0, abs=1e-9)
    with pytest.raises(ValueError):
        iia_violation(IndependentScorer(), "s", Q, "a")


def test_nonsense_confidence():
    out = nonsense_confidence(IndependentScorer(), "heavy rain today", [BoolQuestion(id="b", prompt="It rains.")])
    assert out["real"] > out["random_chars"]


def test_run_probes_summary():
    records = [
        Record(id=f"c{i}", source="s", split="test_in", state="rain", type="choice", prompt="p", candidates=["a", "bb", f"c{i}"], target=0)
        for i in range(4)
    ] + [Record(id="b", source="s", split="test_in", state="rain", type="bool", prompt="p", target=1.0)]
    out = run_probes(IndependentScorer(), records, n_items=3, n_perms=2)
    assert out["order_sensitivity"]["n"] == 3 and out["iia"]["n"] == 3 and out["nonsense_confidence"]["n"] == 3
    assert out["order_sensitivity"]["mean_spread"] == pytest.approx(0.0)
