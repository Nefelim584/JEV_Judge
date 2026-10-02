import json
import random

import pytest

from jevlite.encoding import (
    ClaimGroup,
    PackedEncoder,
    PairEncoder,
    item_questions,
    render_header,
    segment_a_texts,
    serialize_state,
    shuffle_choice,
    text_encoder_from_config,
    truncate_head_tail,
)
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


# ---------------------------------------------------------------------------- packed mode


def test_packed_templates():
    choice = ChoiceQuestion(id="q", prompt="Pick", options=["x", "y"])
    score = ScoreQuestion(id="s", prompt="Rate", levels=["low", "high"])
    claim = BoolQuestion(id="b", prompt="It is raining.")
    group = ClaimGroup((claim, BoolQuestion(id="c", prompt="It is cold.")), context="What is the weather?")
    assert render_header(choice) == "type: choice | question: Pick | options: [MASK] x [MASK] y"
    assert render_header(score) == "type: score | question: Rate | levels from low to high: [MASK] low [MASK] high"
    assert render_header(claim) == "type: bool | claim: It is raining. [MASK]"
    assert render_header(group) == (
        "type: bool | context: What is the weather? | claim: It is raining. [MASK] | claim: It is cold. [MASK]"
    )


def test_pair_template_with_context():
    claim = BoolQuestion(id="b", prompt="It is raining.")
    assert segment_a_texts(claim, context="Weather?") == ["type: bool | context: Weather? | claim: It is raining."]


def test_claim_group_validation():
    with pytest.raises(ValueError, match="at least one"):
        ClaimGroup(())
    with pytest.raises(TypeError, match="Bool questions only"):
        ClaimGroup((ChoiceQuestion(id="q", prompt="p", options=["x"]),))
    with pytest.raises(ValueError, match="unique"):
        ClaimGroup((BoolQuestion(id="b", prompt="p"), BoolQuestion(id="b", prompt="q")))


def _packed_parts(row, tok):
    """Split [CLS] header [SEP] state [SEP] back into (header, state)."""
    assert row[0] == tok.cls_token_id and row[-1] == tok.sep_token_id
    first_sep = row.index(tok.sep_token_id)
    return row[1:first_sep], row[first_sep + 1 : -1]


PACKED_ITEMS = [
    ChoiceQuestion(id="q", prompt="Which department should handle it?", options=["billing", "technical support team", "x"]),
    ScoreQuestion(id="s", prompt="How urgent is it?", levels=["low", "medium", "high", "critical"]),
    BoolQuestion(id="b", prompt="The customer is angry."),
    ClaimGroup(
        (BoolQuestion(id="c1", prompt="It was founded in 2015."), BoolQuestion(id="c2", prompt="It has 40 staff.")),
        context="Tell me about the company.",
    ),
]


@pytest.mark.parametrize("item", PACKED_ITEMS, ids=lambda i: type(i).__name__)
def test_packed_markers_point_at_mask_tokens(tokenizer, item):
    enc = PackedEncoder(tokenizer, max_len=512)
    (seq,) = enc.encode_item(item, enc.tokenize_state("Some state."))
    n_candidates = sum(len(q.candidates) for q in item_questions(item))
    assert len(seq.markers) == n_candidates
    assert seq.input_ids.count(tokenizer.mask_token_id) == n_candidates
    assert all(seq.input_ids[pos] == tokenizer.mask_token_id for pos, _, _ in seq.markers)
    # Markers are in candidate order and map back to (question, candidate) within the item.
    expected = [(j, k) for j, q in enumerate(item_questions(item)) for k in range(len(q.candidates))]
    assert [(j, k) for _, j, k in seq.markers] == expected
    # All markers sit in the header.
    header, _ = _packed_parts(seq.input_ids, tokenizer)
    assert all(1 <= pos <= len(header) for pos, _, _ in seq.markers)


def test_packed_candidate_follows_its_marker(tokenizer):
    enc = PackedEncoder(tokenizer, max_len=512)
    q = PACKED_ITEMS[0]
    (seq,) = enc.encode_item(q, enc.tokenize_state("s"))
    option_ids = tokenizer([f" {o}" for o in q.options], add_special_tokens=False)["input_ids"]
    for (pos, _, k), ids in zip(seq.markers, option_ids, strict=True):
        assert seq.input_ids[pos + 1 : pos + 1 + len(ids)] == ids


@pytest.mark.parametrize("max_len", [64, 96, 128])
@pytest.mark.parametrize("item", PACKED_ITEMS, ids=lambda i: type(i).__name__)
def test_packed_header_never_truncated(tokenizer, item, max_len):
    enc = PackedEncoder(tokenizer, max_len=max_len, head_ratio=0.5)
    long_state = " ".join(f"word{i}" for i in range(2000))
    (seq,) = enc.encode_item(item, enc.tokenize_state(long_state))
    assert seq.truncated
    assert len(seq.input_ids) == max_len
    header, _ = _packed_parts(seq.input_ids, tokenizer)
    assert header == enc.encode_header(item)[0]


def test_packed_head_and_tail_of_state_are_kept(tokenizer):
    enc = PackedEncoder(tokenizer, max_len=64, head_ratio=0.5)
    state_ids = enc.tokenize_state("START " + "filler " * 500 + "END")
    (seq,) = enc.encode_item(BoolQuestion(id="b", prompt="p"), state_ids)
    _, b = _packed_parts(seq.input_ids, tokenizer)
    assert b[:3] == state_ids[:3] and b[-3:] == state_ids[-3:]


def test_packed_short_state_untouched(tokenizer):
    enc = PackedEncoder(tokenizer, max_len=512)
    state_ids = enc.tokenize_state({"a": 1})
    (seq,) = enc.encode_item(BoolQuestion(id="b", prompt="p"), state_ids)
    assert not seq.truncated
    assert _packed_parts(seq.input_ids, tokenizer)[1] == state_ids


def test_single_bool_equals_group_of_one(tokenizer):
    enc = PackedEncoder(tokenizer, max_len=128)
    claim = BoolQuestion(id="b", prompt="The customer is angry.")
    state_ids = enc.tokenize_state("I want my money back now.")
    assert enc.encode_item(claim, state_ids) == enc.encode_item(ClaimGroup((claim,)), state_ids)


def test_fake_markers_and_separators_are_escaped(tokenizer):
    enc = PackedEncoder(tokenizer, max_len=256)
    q = ChoiceQuestion(id="q", prompt="Which [MASK] is it?", options=["a [MASK] b", "[SEP]", "[CLS] c"])
    state = {"note": "literal [MASK] and [SEP] and [PAD] tokens"}
    (seq,) = enc.encode_item(q, enc.tokenize_state(state))
    assert seq.input_ids.count(tokenizer.mask_token_id) == 3
    assert seq.input_ids.count(tokenizer.sep_token_id) == 2
    assert seq.input_ids.count(tokenizer.cls_token_id) == 1
    assert tokenizer.pad_token_id not in seq.input_ids


def test_pair_mode_escapes_fake_specials_too(tokenizer):
    enc = PairEncoder(tokenizer, max_len=128)
    q = BoolQuestion(id="b", prompt="claim [SEP] with [MASK]")
    row = enc.encode_question(q, enc.tokenize_state("state [SEP] text")).input_ids[0]
    assert row.count(tokenizer.sep_token_id) == 2
    assert tokenizer.mask_token_id not in row


def test_packed_header_too_long_raises(tokenizer):
    enc = PackedEncoder(tokenizer, max_len=16)
    with pytest.raises(ValueError, match="no room for the state"):
        enc.encode_item(BoolQuestion(id="b", prompt="a very long claim " * 20), enc.tokenize_state("state"))


def test_text_encoder_from_config(tokenizer):
    cfg = {"encoding": {"mode": "packed", "max_len": 512, "head_ratio": 0.3}}
    enc = text_encoder_from_config(cfg, tokenizer, max_len=1024)
    assert isinstance(enc, PackedEncoder) and enc.max_len == 1024 and enc.head_ratio == 0.3
    cfg["encoding"]["mode"] = "pair"
    assert isinstance(text_encoder_from_config(cfg, tokenizer), PairEncoder)
    cfg["encoding"]["mode"] = "bogus"
    with pytest.raises(ValueError, match="encoding.mode"):
        text_encoder_from_config(cfg, tokenizer)


def test_base_config_defaults_to_packed():
    from jevlite.config import load_config

    assert load_config("apple_silicon")["encoding"]["mode"] == "packed"


# ---------------------------------------------------------------------------- choice shuffling


def test_shuffle_keeps_hard_target_aligned():
    q = ChoiceQuestion(id="q", prompt="p", options=[f"o{i}" for i in range(8)])
    for seed in range(20):
        shuffled, target, perm = shuffle_choice(q, 3, random.Random(seed))
        assert sorted(perm) == list(range(8))
        assert shuffled.options == [q.options[j] for j in perm]
        assert shuffled.options[target] == "o3"
        assert shuffled.id == q.id and shuffled.prompt == q.prompt
    assert q.options == [f"o{i}" for i in range(8)]  # the original is untouched


def test_shuffle_keeps_soft_target_aligned():
    q = ChoiceQuestion(id="q", prompt="p", options=["a", "b", "c", "d"])
    soft = [0.1, 0.2, 0.3, 0.4]
    shuffled, target, perm = shuffle_choice(q, soft, random.Random(0))
    assert dict(zip(shuffled.options, target)) == dict(zip(q.options, soft))


def test_shuffle_is_seeded():
    q = ChoiceQuestion(id="q", prompt="p", options=[f"o{i}" for i in range(8)])
    assert shuffle_choice(q, 0, random.Random(7))[2] == shuffle_choice(q, 0, random.Random(7))[2]


def test_shuffle_rejects_score_and_bad_targets():
    with pytest.raises(TypeError, match="only choice"):
        shuffle_choice(ScoreQuestion(id="s", prompt="p", levels=["lo", "hi"]), 0, random.Random(0))
    q = ChoiceQuestion(id="q", prompt="p", options=["a", "b"])
    with pytest.raises(ValueError, match="out of range"):
        shuffle_choice(q, 2, random.Random(0))
    with pytest.raises(ValueError, match="soft target"):
        shuffle_choice(q, [1.0], random.Random(0))
