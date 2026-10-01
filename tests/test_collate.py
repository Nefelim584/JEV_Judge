import pytest
import torch

from jevlite.collate import TYPE_IDS, build_batch, build_request_batch, group_logits, ungroup
from jevlite.encoding import PairEncoder
from jevlite.envcheck import SAMPLE_REQUEST
from jevlite.schema import Request


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
    assert b.row_type.tolist() == [0] * 5 + [1] * 4 + [2]


def test_padding_and_attention_mask(request_batch, encoder):
    b = request_batch
    lengths = b.attention_mask.sum(-1)
    for i, n in enumerate(lengths.tolist()):
        assert (b.input_ids[i, n:] == encoder.tokenizer.pad_token_id).all()
        assert b.input_ids[i, n - 1] == encoder.tokenizer.sep_token_id
    assert lengths.max() == b.input_ids.shape[1]


def test_row_mapping_round_trip(request_batch):
    b = request_batch
    row_logits = torch.randn(b.input_ids.shape[0])
    grouped = group_logits(row_logits, b)
    assert grouped.dtype == torch.float32
    assert torch.equal(ungroup(grouped, b), row_logits)
    assert torch.isneginf(grouped[~b.candidate_mask]).all()
    assert torch.isfinite(grouped[b.candidate_mask]).all()
    # (question, candidate) pairs are unique and cover exactly the real candidates
    pairs = set(zip(b.row_question.tolist(), b.row_candidate.tolist()))
    assert len(pairs) == b.input_ids.shape[0]
    assert pairs == {tuple(ix) for ix in b.candidate_mask.nonzero().tolist()}


def test_masked_candidates_get_zero_probability(request_batch):
    probs = torch.softmax(group_logits(torch.randn(request_batch.input_ids.shape[0]), request_batch), -1)
    assert (probs[~request_batch.candidate_mask] == 0).all()
    assert torch.allclose(probs.sum(-1), torch.ones(3))


def test_rows_follow_candidate_order(request_batch, encoder):
    req = Request.model_validate(SAMPLE_REQUEST)
    q = req.questions[0]
    rows = [r for r, qi in zip(request_batch.input_ids, request_batch.row_question) if qi == 0]
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
