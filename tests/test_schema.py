import pytest
from pydantic import ValidationError

from jevlite.envcheck import SAMPLE_REQUEST
from jevlite.schema import (
    MAX_CANDIDATES,
    BoolAnswer,
    BoolQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    Request,
    Response,
    ScoreAnswer,
    ScoreQuestion,
)


def _request(*questions, state="some state"):
    return {"state": state, "questions": list(questions)}


def test_sample_request_parses_into_typed_questions():
    req = Request.model_validate(SAMPLE_REQUEST)
    assert [type(q) for q in req.questions] == [ChoiceQuestion, ScoreQuestion, BoolQuestion]
    assert list(req.state) == ["ticket_id", "channel", "message"]  # key order preserved


@pytest.mark.parametrize("state", ["text", {"a": 1, "b": {"c": [1, 2]}}, ["one", "two"]])
def test_state_kinds(state):
    req = Request.model_validate(_request({"id": "q", "type": "bool", "prompt": "p"}, state=state))
    assert req.state == state


@pytest.mark.parametrize(
    "question",
    [
        {"id": "q", "type": "choice", "prompt": "p", "options": []},
        {"id": "q", "type": "choice", "prompt": "p", "options": ["a", "a"]},
        {"id": "q", "type": "choice", "prompt": "p", "options": ["a", "  "]},
        {"id": "q", "type": "choice", "prompt": "p", "options": [str(i) for i in range(MAX_CANDIDATES + 1)]},
        {"id": "q", "type": "score", "prompt": "p", "levels": []},
        {"id": "q", "type": "score", "prompt": "p", "levels": ["low", "low"]},
        {"id": "q", "type": "bool", "prompt": ""},
        {"id": "", "type": "bool", "prompt": "p"},
        {"id": "q", "type": "ranking", "prompt": "p"},
        {"id": "q", "type": "bool", "prompt": "p", "options": ["x"]},  # extra field
        {"id": "q", "prompt": "p"},  # missing type
    ],
)
def test_invalid_questions_are_rejected(question):
    with pytest.raises(ValidationError):
        Request.model_validate(_request(question))


def test_max_candidates_is_allowed():
    q = {"id": "q", "type": "choice", "prompt": "p", "options": [str(i) for i in range(MAX_CANDIDATES)]}
    Request.model_validate(_request(q))


def test_duplicate_question_ids_rejected():
    q = {"id": "dup", "type": "bool", "prompt": "p"}
    with pytest.raises(ValidationError, match="unique"):
        Request.model_validate(_request(q, q))


def test_empty_questions_rejected():
    with pytest.raises(ValidationError):
        Request.model_validate({"state": "s", "questions": []})


def test_invalid_state_rejected():
    with pytest.raises(ValidationError):
        Request.model_validate({"state": 42, "questions": [{"id": "q", "type": "bool", "prompt": "p"}]})


def test_candidates_property():
    req = Request.model_validate(SAMPLE_REQUEST)
    assert req.questions[0].candidates == SAMPLE_REQUEST["questions"][0]["options"]
    assert req.questions[1].candidates == SAMPLE_REQUEST["questions"][1]["levels"]
    assert req.questions[2].candidates == [SAMPLE_REQUEST["questions"][2]["prompt"]]


RESPONSE = {
    "answers": {
        "intent": {
            "value": "refund",
            "probabilities": {"refund": 0.91, "exchange": 0.01, "order status": 0.02, "cancel subscription": 0.04, "other": 0.02},
            "confidence": 0.88,
        },
        "urgency": {
            "value": "high",
            "expected_level": 2.35,
            "probabilities": {"low": 0.01, "medium": 0.07, "high": 0.48, "critical": 0.44},
            "confidence": 0.61,
        },
        "churn_risk": {"probability": 0.93, "confidence": 0.90},
    },
    "meta": {"model": "jev-lite-v1", "latency_ms": 41},
}


def test_response_round_trip_picks_correct_answer_types():
    resp = Response.model_validate(RESPONSE)
    assert type(resp.answers["intent"]) is ChoiceAnswer
    assert type(resp.answers["urgency"]) is ScoreAnswer
    assert type(resp.answers["churn_risk"]) is BoolAnswer
    assert Response.model_validate_json(resp.model_dump_json()) == resp


def test_answer_value_must_be_in_schema():
    with pytest.raises(ValidationError):
        ChoiceAnswer(value="nope", probabilities={"a": 0.5, "b": 0.5}, confidence=0.5)


def test_answer_probabilities_must_sum_to_one():
    with pytest.raises(ValidationError):
        ChoiceAnswer(value="a", probabilities={"a": 0.5, "b": 0.6}, confidence=0.5)


def test_bool_probability_range():
    with pytest.raises(ValidationError):
        BoolAnswer(probability=1.2, confidence=0.5)
