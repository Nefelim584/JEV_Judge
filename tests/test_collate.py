import pytest
import torch

from jevlite.collate import TYPE_IDS, build_batch, build_request_batch, gather_markers, group_logits, ungroup
from jevlite.encoding import ClaimGroup, PackedEncoder, PairEncoder
from jevlite.envcheck import SAMPLE_REQUEST
from jevlite.schema import BoolQuestion, Request


@pytest.fixture
def encoder(tokenizer):
    return PairEncoder(tokenizer, max_len=128)


@pytest.fixture
def request_batch(encoder):
    return build_request_batch(Request.model_validate(SAMPLE_REQUEST), encoder)


def test_shapes_and_mask(request_batch):
    b = request_batch
    assert b.question_ids == ["intent", "urgency", "churn_risk"]
    assert b.input_ids.shape[0] == 5 + 4 + 1
    assert b.k_max == 5
    assert b.candidate_mask.tolist() == [
        [True] * 5,
        [True] * 4 + [False],
        [True] + [False] * 4,
    ]
    assert b.question_type.tolist() == [TYPE_IDS["choice"], TYPE_IDS["score"], TYPE_IDS["bool"]]
    assert b.marker_type.tolist() == [0] * 5 + [1] * 4 + [2]
    assert b.marker_row.tolist() == list(range(10))
    assert b.marker_pos.tolist() == [0] * 10  # pair mode reads out at [CLS]


def test_padding_and_attention_mask(request_batch, encoder):
    b = request_batch
    lengths = b.attention_mask.sum(-1)
    for i, n in enumerate(lengths.tolist()):
        assert (b.input_ids[i, n:] == encoder.tokenizer.pad_token_id).all()
        assert b.input_ids[i, n - 1] == encoder.tokenizer.sep_token_id
    assert lengths.max() == b.input_ids.shape[1]


def test_row_mapping_round_trip(request_batch):
    b = request_batch
    marker_logits = torch.randn(b.n_markers)
    grouped = group_logits(marker_logits, b)
    assert grouped.dtype == torch.float32
    assert torch.equal(ungroup(grouped, b), marker_logits)
    assert torch.isneginf(grouped[~b.candidate_mask]).all()
    assert torch.isfinite(grouped[b.candidate_mask]).all()
    # (question, candidate) pairs are unique and cover exactly the real candidates
    pairs = set(zip(b.marker_question.tolist(), b.marker_candidate.tolist()))
    assert len(pairs) == b.n_markers
    assert pairs == {tuple(ix) for ix in b.candidate_mask.nonzero().tolist()}


def test_masked_candidates_get_zero_probability(request_batch):
    probs = torch.softmax(group_logits(torch.randn(request_batch.n_markers), request_batch), -1)
    assert (probs[~request_batch.candidate_mask] == 0).all()
    assert torch.allclose(probs.sum(-1), torch.ones(3))


def test_rows_follow_candidate_order(request_batch, encoder):
    req = Request.model_validate(SAMPLE_REQUEST)
    q = req.questions[0]
    rows = [request_batch.input_ids[r] for r, qi in zip(request_batch.marker_row, request_batch.marker_question) if qi == 0]
    expected = encoder.encode_question(q, encoder.tokenize_state(req.state)).input_ids
    for row, exp, mask in zip(rows, expected, request_batch.attention_mask[:5]):
        assert row[: mask.sum()].tolist() == exp


def test_multi_state_batch_tracks_truncation(tokenizer):
    enc = PairEncoder(tokenizer, max_len=48)
    req = Request.model_validate(
        {
            "state": "short",
            "questions": [
                {"id": "a", "type": "bool", "prompt": "p"},
                {"id": "b", "type": "choice", "prompt": "p", "options": ["x", "y"]},
            ],
        }
    )
    long_state = "long " * 300
    b = build_batch([(req.state, req.questions[0]), (long_state, req.questions[1])], enc)
    assert b.truncated.tolist() == [False, True]
    assert b.input_ids.shape[1] <= 48


def test_to_device_keeps_metadata(request_batch):
    moved = request_batch.to("cpu")
    assert moved.question_ids == request_batch.question_ids
    assert torch.equal(moved.input_ids, request_batch.input_ids)


def test_empty_batch_rejected(encoder):
    with pytest.raises(ValueError):
        build_batch([], encoder)


# ---------------------------------------------------------------------------- packed mode


def _mixed_items():
    req = Request.model_validate(SAMPLE_REQUEST)
    group = ClaimGroup(
        (
            BoolQuestion(id="s1", prompt="It was charged twice."),
            BoolQuestion(id="s2", prompt="The order number is A-993."),
            BoolQuestion(id="s3", prompt="The customer wants a discount."),
        ),
        context="What happened to the order?",
    )
    chunk = "Order A-993 was charged twice on May 3."
    return [(req.state, q) for q in req.questions] + [(chunk, group)]


@pytest.fixture
def packed_batch(tokenizer):
    return build_batch(_mixed_items(), PackedEncoder(tokenizer, max_len=128))


def test_packed_one_sequence_per_item(packed_batch, tokenizer):
    b = packed_batch
    assert b.input_ids.shape[0] == 4  # choice, score, bool, one claim group
    assert b.question_ids == ["intent", "urgency", "churn_risk", "s1", "s2", "s3"]
    assert b.marker_row.tolist() == [0] * 5 + [1] * 4 + [2] + [3] * 3
    assert (b.input_ids[b.marker_row, b.marker_pos] == tokenizer.mask_token_id).all()
    assert b.marker_type.tolist() == [0] * 5 + [1] * 4 + [2] * 4
    assert b.question_type.tolist() == [0, 1, 2, 2, 2, 2]
    assert b.candidate_mask.sum(-1).tolist() == [5, 4, 1, 1, 1, 1]


def test_packed_marker_mapping_round_trip(packed_batch):
    b = packed_batch
    marker_logits = torch.randn(b.n_markers)
    grouped = group_logits(marker_logits, b)
    assert torch.equal(ungroup(grouped, b), marker_logits)
    assert torch.isneginf(grouped[~b.candidate_mask]).all()
    pairs = list(zip(b.marker_question.tolist(), b.marker_candidate.tolist()))
    assert len(set(pairs)) == b.n_markers
    assert set(pairs) == {tuple(ix) for ix in b.candidate_mask.nonzero().tolist()}


def test_gather_markers(packed_batch):
    b = packed_batch
    hidden = torch.randn(*b.input_ids.shape, 8)
    out = gather_markers(hidden, b)
    assert out.shape == (b.n_markers, 8)
    for m in range(b.n_markers):
        assert torch.equal(out[m], hidden[b.marker_row[m], b.marker_pos[m]])


def test_claim_group_truncation_is_shared(tokenizer):
    group = ClaimGroup((BoolQuestion(id="a", prompt="p"), BoolQuestion(id="b", prompt="q")))
    b = build_batch([("long " * 300, group), ("short", BoolQuestion(id="c", prompt="r"))], PackedEncoder(tokenizer, 48))
    assert b.truncated.tolist() == [True, True, False]
    assert b.input_ids.shape[1] <= 48


def test_both_modes_group_questions_identically(tokenizer):
    items = _mixed_items()
    packed = build_batch(items, PackedEncoder(tokenizer, max_len=128))
    pair = build_batch(items, PairEncoder(tokenizer, max_len=128))
    assert pair.input_ids.shape[0] == pair.n_markers == packed.n_markers  # pair: one sequence per candidate
    assert packed.question_ids == pair.question_ids
    for name in ("question_type", "candidate_mask", "marker_question", "marker_candidate", "marker_type"):
        assert torch.equal(getattr(packed, name), getattr(pair, name)), name
