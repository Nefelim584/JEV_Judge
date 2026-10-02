import numpy as np
import pytest
import torch

from jevlite.baselines.base import entropy_confidence, predict_records
from jevlite.baselines.laya import to_laya_question
from jevlite.baselines.nli import NLIZeroShot, entailment_index, hypotheses
from jevlite.data.unified import Record
from jevlite.schema import BoolQuestion, ChoiceQuestion, ScoreQuestion


class FixedPredictor:
    """Choice/Score: probability ∝ 1 + index; Bool: 0.8. Records the calls it gets."""

    name = "fixed"

    def __init__(self):
        self.calls = []

    def predict(self, state, questions):
        self.calls.append((state, [q.id for q in questions]))
        out = []
        for q in questions:
            if q.type == "bool":
                out.append(np.array([0.8]))
            else:
                w = np.arange(1, len(q.candidates) + 1, dtype=float)
                out.append(w / w.sum())
        return out


def _rec(i, state, type_="bool", **kw):
    base = {"id": f"r{i}", "source": "s", "split": "test_in", "state": state, "type": type_, "prompt": f"p{i}", "target": 1.0}
    return Record.model_validate(base | kw)


def test_predict_records_groups_by_state_and_keeps_order():
    records = [
        _rec(0, "A"),
        _rec(1, {"k": 1}, "choice", candidates=["x", "y"], target=0),
        _rec(2, "A", "score", candidates=["lo", "mid", "hi"], target=2),
        _rec(3, {"k": 1}),
    ]
    predictor = FixedPredictor()
    preds = predict_records(predictor, records, max_questions=16)
    assert [p.id for p in preds] == ["r0", "r1", "r2", "r3"]
    assert len(predictor.calls) == 2  # one call per distinct state
    assert preds[0].prob == pytest.approx(0.8)
    assert preds[2].probs == pytest.approx([1 / 6, 2 / 6, 3 / 6])
    assert all(p.latency_ms is not None and 0 <= p.confidence <= 1 for p in preds)


def test_predict_records_chunks_large_groups():
    predictor = FixedPredictor()
    predict_records(predictor, [_rec(i, "A") for i in range(5)], max_questions=2)
    assert [len(ids) for _, ids in predictor.calls] == [2, 2, 1]


def test_entropy_confidence():
    assert entropy_confidence([0.5, 0.5]) == pytest.approx(0.0)
    assert entropy_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert entropy_confidence([0.5]) == pytest.approx(0.0)  # Bool P(yes) = 0.5
    assert entropy_confidence([0.99]) > 0.9


def test_entailment_index():
    assert entailment_index({0: "entailment", 1: "neutral", 2: "contradiction"}) == 0
    assert entailment_index({0: "not_entailment", 1: "entailment"}) == 1
    assert entailment_index({"0": "CONTRADICTION", "1": "NEUTRAL", "2": "ENTAILMENT"}) == 2
    with pytest.raises(ValueError):
        entailment_index({0: "positive", 1: "negative"})


def test_hypotheses():
    assert hypotheses(BoolQuestion(id="b", prompt="It rains.")) == ["It rains."]
    choice = hypotheses(ChoiceQuestion(id="c", prompt="Who?", options=["Ann", "Bob"]))
    assert choice == ['The answer to "Who?" is: Ann.', 'The answer to "Who?" is: Bob.']
    score = hypotheses(ScoreQuestion(id="s", prompt="How bad?", levels=["low", "high"]))
    assert all('from "low" to "high"' in h for h in score)


class _FakeNLI(torch.nn.Module):
    """Entailment logit = +3 when the hypothesis ends with 'yes.', else −3. Labels: [contradiction, entailment]."""

    def __init__(self):
        super().__init__()
        self.config = type("C", (), {"id2label": {0: "contradiction", 1: "entailment"}})()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, **_):
        ent = torch.where(input_ids[:, -1] == 1, 3.0, -3.0)
        return type("O", (), {"logits": torch.stack([-ent, ent], -1)})()


class _FakeTok:
    def __call__(self, premises, hyps, **_):
        ids = torch.tensor([[0, 1 if h.endswith("yes.") else 2] for h in hyps])

        class Enc(dict):
            def to(self, device):
                return self

        return Enc(input_ids=ids)


def test_nli_predict_with_fake_model():
    nli = NLIZeroShot(model=_FakeNLI(), tokenizer=_FakeTok(), device="cpu", batch_size=2, choice_template="{candidate}.")
    p_bool, p_choice = nli.predict("state", [BoolQuestion(id="b", prompt="yes."), ChoiceQuestion(id="c", prompt="?", options=["no", "yes", "maybe"])])
    assert p_bool[0] == pytest.approx(1 / (1 + np.exp(-6)))
    assert p_choice.argmax() == 1 and p_choice.sum() == pytest.approx(1.0)
    assert p_choice[0] == pytest.approx(p_choice[2])


def test_to_laya_question():
    assert to_laya_question(BoolQuestion(id="b", prompt="It rains.")) == {"t": "noul", "ins": "It rains.", "crit": None}
    assert to_laya_question(ChoiceQuestion(id="c", prompt="Who?", options=["a", "b"])) == {"t": "choice", "ins": "Who?", "crit": {"a": None, "b": None}}
    assert to_laya_question(ScoreQuestion(id="s", prompt="How?", levels=["lo", "hi"]))["crit"] == ["lo", "hi"]


def test_load_baseline_rejects_unknown():
    from jevlite.baselines import load_baseline

    with pytest.raises(ValueError, match="unknown baseline"):
        load_baseline("gpt")
