import json

import pytest

from jevlite.encoding import PairEncoder, segment_a_texts, serialize_state, truncate_head_tail
from jevlite.schema import BoolQuestion, ChoiceQuestion, ScoreQuestion


def test_serialize_string_as_is():
    assert serialize_state("  hello\nworld ") == "  hello\nworld "


def test_serialize_json_compact_and_ordered():
    state = {"z": 1, "a": {"y": [1, 2], "b": "ü"}}
    text = serialize_state(state)
    assert text == '{"z":1,"a":{"y":[1,2],"b":"ü"}}'
    assert json.loads(text) == state


def test_serialize_array():
    assert serialize_state(["first", "second"]) == "[1] first\n[2] second"


def test_templates():
    choice = ChoiceQuestion(id="q", prompt="Pick", options=["x", "y"])
    score = ScoreQuestion(id="s", prompt="Rate", levels=["low", "high"])
    claim = BoolQuestion(id="b", prompt="It is raining.")
    assert segment_a_texts(choice) == ["type: choice | question: Pick | answer: x", "type: choice | question: Pick | answer: y"]
    assert segment_a_texts(score) == [
        "type: score | question: Rate | level 1 of 2: low",
        "type: score | question: Rate | level 2 of 2: high",
    ]
    assert segment_a_texts(claim) == ["type: bool | claim: It is raining."]


@pytest.mark.parametrize("head_ratio", [0.0, 0.3, 0.5, 1.0])
def test_truncate_head_tail(head_ratio):
    ids = list(range(100))
    out, truncated = truncate_head_tail(ids, 10, head_ratio)
    head = int(10 * head_ratio)
    assert truncated and len(out) == 10
    assert out == ids[:head] + ids[100 - (10 - head) :]


def test_truncate_noop_when_fits():
    assert truncate_head_tail([1, 2, 3], 3, 0.5) == ([1, 2, 3], False)


def _segments(row, tok):
    """Split [CLS] a [SEP] b [SEP] back into (a, b)."""
    assert row[0] == tok.cls_token_id and row[-1] == tok.sep_token_id
    first_sep = row.index(tok.sep_token_id)
    return row[1:first_sep], row[first_sep + 1 : -1]


def test_pair_layout_matches_tokenizer_pair_encoding(tokenizer):
    enc = PairEncoder(tokenizer, max_len=512)
    q = BoolQuestion(id="b", prompt="The customer is angry.")
    state = "I want my money back now."
    row = enc.encode_question(q, enc.tokenize_state(state)).input_ids[0]
    reference = tokenizer(segment_a_texts(q)[0], state)["input_ids"]
    assert row == reference


@pytest.mark.parametrize("max_len", [32, 64, 128])
def test_question_and_candidate_never_truncated(tokenizer, max_len):
    enc = PairEncoder(tokenizer, max_len=max_len, head_ratio=0.5)
    q = ChoiceQuestion(id="q", prompt="Which department should handle it?", options=["billing", "technical support team", "x"])
    long_state = " ".join(f"word{i}" for i in range(2000))
    encoded = enc.encode_question(q, enc.tokenize_state(long_state))

    assert encoded.truncated
    expected_a = tokenizer(segment_a_texts(q), add_special_tokens=False)["input_ids"]
    states = []
    for row, a in zip(encoded.input_ids, expected_a, strict=True):
        assert len(row) <= max_len
        got_a, got_b = _segments(row, tokenizer)
        assert got_a == a
        states.append(got_b)
    # Every candidate of a question sees the same state tokens, and the longest row uses the full budget.
    assert all(s == states[0] for s in states)
    assert max(len(r) for r in encoded.input_ids) == max_len


def test_head_and_tail_of_state_are_kept(tokenizer):
    enc = PairEncoder(tokenizer, max_len=64, head_ratio=0.5)
    q = BoolQuestion(id="b", prompt="p")
    state_ids = enc.tokenize_state("START " + "filler " * 500 + "END")
    _, b = _segments(enc.encode_question(q, state_ids).input_ids[0], tokenizer)
    assert b[:3] == state_ids[:3] and b[-3:] == state_ids[-3:]


def test_short_state_untouched(tokenizer):
    enc = PairEncoder(tokenizer, max_len=512)
    state_ids = enc.tokenize_state({"a": 1})
    encoded = enc.encode_question(BoolQuestion(id="b", prompt="p"), state_ids)
    assert not encoded.truncated
    assert _segments(encoded.input_ids[0], tokenizer)[1] == state_ids


def test_question_too_long_for_max_len_raises(tokenizer):
    enc = PairEncoder(tokenizer, max_len=16)
    q = BoolQuestion(id="b", prompt="a very long claim " * 20)
    with pytest.raises(ValueError, match="no room for the state"):
        enc.encode_question(q, enc.tokenize_state("state"))
