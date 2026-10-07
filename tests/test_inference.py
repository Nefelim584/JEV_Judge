import random

import pytest
import torch

from jevlite.data.loader import TrainCollator
from jevlite.data.unified import Record
from jevlite.encoding import PackedEncoder
from jevlite.inference import Predictor, heuristic_confidence
from jevlite.model import JevLite
from jevlite.schema import BoolAnswer, BoolQuestion, ChoiceAnswer, ChoiceQuestion, Request, Response, ScoreAnswer, ScoreQuestion

HIDDEN = 32


@pytest.fixture(scope="module")
def predictor(tokenizer):
    from transformers import ModernBertConfig, ModernBertModel

    torch.manual_seed(0)
    encoder = ModernBertModel(ModernBertConfig(
        vocab_size=len(tokenizer), hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, pad_token_id=tokenizer.pad_token_id, reference_compile=False,
    ))
    model = JevLite(encoder, hidden=HIDDEN, trunk_dim=8, dropout=0.0)
    return Predictor(model, PackedEncoder(tokenizer, 96))


def _random_request(rng: random.Random) -> Request:
    words = ["refund", "exchange", "status", "other", "low", "medium", "high", "critical", "billing", "spam", "a b", "x"]
    questions = []
    for i in range(rng.randint(1, 4)):
        kind = rng.choice(["choice", "score", "bool"])
        if kind == "bool":
            questions.append(BoolQuestion(id=f"q{i}", prompt=f"Claim {i}: the user is upset [MASK]."))
        else:
            cands = rng.sample(words, rng.randint(1, 6))
            cls = ChoiceQuestion if kind == "choice" else ScoreQuestion
            field = "options" if kind == "choice" else "levels"
            questions.append(cls(id=f"q{i}", prompt=f"Question {i}?", **{field: cands}))
    state = rng.choice(["short text", {"message": "I was charged twice " * rng.randint(1, 60)}, ["chunk one", "chunk two"]])
    return Request(state=state, questions=questions)


def test_output_always_inside_schema(predictor):
    rng = random.Random(0)
    for _ in range(25):
        request = _random_request(rng)
        response = predictor.predict(request)
        Response.model_validate_json(response.model_dump_json())  # survives a JSON round trip
        assert list(response.answers) == [q.id for q in request.questions]
        for q in request.questions:
            a = response.answers[q.id]
            if q.type == "bool":
                assert isinstance(a, BoolAnswer) and 0 <= a.probability <= 1
            else:
                assert isinstance(a, ScoreAnswer if q.type == "score" else ChoiceAnswer)
                assert list(a.probabilities) == q.candidates  # caller's order, nothing else
                assert a.value in q.candidates
                assert a.value == max(a.probabilities, key=a.probabilities.get)
                assert sum(a.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
            assert 0 <= a.confidence <= 1


def test_score_expected_level(predictor):
    request = Request(state="text", questions=[ScoreQuestion(id="u", prompt="How urgent?", levels=["low", "mid", "high"])])
    a = predictor.predict(request).answers["u"]
    p = list(a.probabilities.values())
    assert a.expected_level == pytest.approx(p[1] + 2 * p[2])


def test_truncation_is_reported(predictor):
    long = Request(state="word " * 500, questions=[BoolQuestion(id="b", prompt="A claim.")])
    assert predictor.predict(long).meta.truncated
    assert not predictor.predict(Request(state="short", questions=[BoolQuestion(id="b", prompt="A claim.")])).meta.truncated


def test_heuristic_confidence():
    assert heuristic_confidence([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert heuristic_confidence([0.25] * 4) == pytest.approx(0.0, abs=1e-9)
    assert heuristic_confidence([1.0]) == 1.0
    assert 0 < heuristic_confidence([0.9, 0.1]) < 1


def test_predict_records_matches_predict(predictor):
    records = [
        Record(id="c", source="t", split="test_in", state="text", type="choice", prompt="Pick", candidates=["a", "b", "c"], target=1),
        Record(id="b", source="t", split="test_in", state="text", type="bool", prompt="A claim.", target=1.0),
    ]
    preds = predictor.predict_records(records, batch_size=1)
    assert [p.id for p in preds] == ["c", "b"]
    assert len(preds[0].probs) == 3 and preds[1].prob is not None
    single = predictor.predict(Request(state="text", questions=[records[0].to_question("x")])).answers["x"]
    assert preds[0].probs == pytest.approx(list(single.probabilities.values()), abs=1e-5)


def test_train_collator_shuffles_choice_only(tokenizer):
    records = [
        Record(id="c", source="t", split="train", state="s", type="choice", prompt="Pick", candidates=["a", "b", "c", "d", "e"], target=3),
        Record(id="s", source="t", split="train", state="s", type="score", prompt="Rate", candidates=["1", "2", "3", "4", "5"], target=4),
        Record(id="b", source="t", split="train", state="s", type="bool", prompt="Claim.", target=0.0),
    ]
    collate = TrainCollator(PackedEncoder(tokenizer, 96), seed=1)
    orders = set()
    for epoch in range(8):
        collate.set_epoch(epoch)
        batch, target = collate(records)
        # the gold option moves with the shuffle: decode the option text at the target position
        header = tokenizer.decode(batch.input_ids[0])
        options = [o.strip() for o in header.split("[SEP]")[0].split("[MASK]")[1:]]
        assert options[int(target[0].argmax())] == "d"
        orders.add(tuple(options))
        assert target[1].tolist() == [0, 0, 0, 0, 1] and target[2, 0] == 0.0
    assert len(orders) > 1  # a fresh order per epoch
    collate.set_epoch(0)
    first, _ = collate(records)
    again, _ = collate(records)
    assert torch.equal(first.input_ids, again.input_ids)  # deterministic per (seed, epoch, id)
