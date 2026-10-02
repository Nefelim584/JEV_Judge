import json

import pytest
from pydantic import ValidationError

from jevlite.data.unified import Prediction, Record, load_predictions, load_records, write_jsonl
from jevlite.schema import BoolQuestion, ChoiceQuestion, ScoreQuestion


def rec(**kw):
    base = {"id": "r1", "source": "s", "split": "test_in", "state": "x", "type": "bool", "prompt": "p", "target": 1.0}
    return Record.model_validate(base | kw)


def test_valid_records_and_questions():
    assert isinstance(rec().to_question(), BoolQuestion)
    c = rec(type="choice", candidates=["a", "b"], target=1)
    assert isinstance(c.to_question(), ChoiceQuestion) and c.n_candidates == 2
    s = rec(type="score", candidates=["lo", "mid", "hi"], target=[0.0, 0.5, 0.5])
    assert isinstance(s.to_question("q"), ScoreQuestion) and s.to_question("q").id == "q"
    assert rec(type="choice", candidates=["a", "b"], target=1.0).target == 1.0  # integral float index is fine


@pytest.mark.parametrize(
    "kw",
    [
        {"split": "dev"},
        {"candidates": ["a"]},
        {"target": 1.5},
        {"target": [0.5, 0.5]},
        {"type": "choice", "candidates": None, "target": 0},
        {"type": "choice", "candidates": ["a", "a"], "target": 0},
        {"type": "choice", "candidates": ["a", "b"], "target": 2},
        {"type": "choice", "candidates": ["a", "b"], "target": 0.5},
        {"type": "score", "candidates": ["a", "b"], "target": [0.3, 0.3]},
        {"type": "score", "candidates": ["a", "b"], "target": [1.0]},
    ],
)
def test_invalid_records_rejected(kw):
    with pytest.raises(ValidationError):
        rec(**kw)


def test_extra_fields_are_kept_for_slicing():
    r = rec(family="bool.negation", criterion="faithfulness")
    assert r.field("family") == "bool.negation"
    assert r.field("criterion") == "faithfulness"
    assert r.field("missing") is None


def test_prediction_validation():
    assert Prediction(id="a", prob=0.3).prob == 0.3
    with pytest.raises(ValidationError):
        Prediction(id="a")
    with pytest.raises(ValidationError):
        Prediction(id="a", prob=0.3, probs=[1.0])
    with pytest.raises(ValidationError):
        Prediction(id="a", probs=[0.3, 0.3])


def test_jsonl_round_trip_and_split_filter(tmp_path):
    rows = [rec(id="a").model_dump(), rec(id="b", split="calib").model_dump()]
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    assert [r.id for r in load_records(path)] == ["a", "b"]
    assert [r.id for r in load_records(path, ["calib"])] == ["b"]

    write_jsonl(tmp_path / "p.jsonl", [Prediction(id="a", prob=0.2)])
    assert load_predictions(tmp_path / "p.jsonl")["a"].prob == 0.2


def test_duplicate_ids_rejected(tmp_path):
    path = tmp_path / "data.jsonl"
    write_jsonl(path, [rec(id="a"), rec(id="a")])
    with pytest.raises(ValueError, match="unique"):
        load_records(path)
