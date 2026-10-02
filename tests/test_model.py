import math

import pytest
import torch

from jevlite.collate import TYPE_IDS, build_batch, build_request_batch
from jevlite.encoding import ClaimGroup, PackedEncoder, PairEncoder
from jevlite.model import JevLite, count_parameters, question_probs
from jevlite.schema import BoolQuestion, ChoiceQuestion, Request, ScoreQuestion

HIDDEN = 64


@pytest.fixture(scope="module")
def tiny_encoder(tokenizer):
    """A random 2-layer ModernBERT with the real vocabulary: no weights to download."""
    from transformers import ModernBertConfig, ModernBertModel

    torch.manual_seed(0)
    cfg = ModernBertConfig(
        vocab_size=len(tokenizer), hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, pad_token_id=tokenizer.pad_token_id, reference_compile=False,
    )
    return ModernBertModel(cfg).eval()


def _model(encoder, **kw):
    torch.manual_seed(0)
    return JevLite(encoder, hidden=HIDDEN, trunk_dim=16, dropout=0.0, **kw).eval()


@pytest.fixture
def request_obj():
    return Request(
        state={"message": "I was charged twice for order #A-993. Please fix this or I am cancelling."},
        questions=[
            ChoiceQuestion(id="intent", prompt="What does the customer want?", options=["refund", "exchange", "status", "other"]),
            ScoreQuestion(id="urgency", prompt="How urgent is this?", levels=["low", "medium", "high"]),
            BoolQuestion(id="churn", prompt="The customer threatens to leave."),
        ],
    )


@pytest.fixture
def packed(tokenizer, request_obj):
    return build_request_batch(request_obj, PackedEncoder(tokenizer, 128))


def test_forward_shapes_and_grouping(tiny_encoder, packed):
    model = _model(tiny_encoder)
    with torch.no_grad():
        logits = model(packed)
        grouped = model.grouped_logits(packed)
    assert logits.shape == (packed.n_markers,) and logits.dtype == torch.float32
    assert grouped.shape == (3, 4)
    assert torch.isinf(grouped[~packed.candidate_mask]).all()
    assert torch.isfinite(grouped[packed.candidate_mask]).all()


def test_probabilities(tiny_encoder, packed):
    model = _model(tiny_encoder)
    with torch.no_grad():
        probs = question_probs(model.grouped_logits(packed), packed.question_type)
    assert (probs[~packed.candidate_mask] == 0).all()  # masked options get 0
    multi = packed.question_type != TYPE_IDS["bool"]
    assert torch.allclose(probs[multi].sum(-1), torch.ones(int(multi.sum())))
    p_bool = probs[~multi]
    assert ((p_bool[:, 0] > 0) & (p_bool[:, 0] < 1)).all() and (p_bool[:, 1:] == 0).all()


def test_bool_claims_are_independent():
    grouped = torch.tensor([[2.0, float("-inf")], [-1.0, float("-inf")], [0.5, float("-inf")]])
    qtype = torch.full((3,), TYPE_IDS["bool"])
    p = question_probs(grouped, qtype)
    assert p[:, 0] == pytest.approx(torch.sigmoid(grouped[:, 0]))
    changed = grouped.clone()
    changed[0, 0] = -5.0
    assert question_probs(changed, qtype)[1:, 0] == pytest.approx(p[1:, 0])  # no normalisation across claims


def test_claim_group_gives_one_sigmoid_per_claim(tiny_encoder, tokenizer):
    group = ClaimGroup(claims=tuple(BoolQuestion(id=f"c{i}", prompt=f"Claim number {i}.") for i in range(3)), context="Q?")
    batch = build_batch([("A chunk of text.", group)], PackedEncoder(tokenizer, 128))
    assert batch.input_ids.shape[0] == 1 and batch.n_questions == 3
    with torch.no_grad():
        probs = question_probs(_model(tiny_encoder).grouped_logits(batch), batch.question_type)
    assert probs.shape == (3, 1)


def test_temperature_divides_logits_per_primitive(tiny_encoder, packed):
    model = _model(tiny_encoder)
    with torch.no_grad():
        raw = model(packed, apply_temperature=False)
        model.log_temperature.copy_(torch.log(torch.tensor([2.0, 4.0, 0.5])))
        scaled = model(packed)
    temps = torch.tensor([2.0, 4.0, 0.5])[packed.marker_type]
    assert torch.allclose(scaled, raw / temps, atol=1e-6)


def test_mean_readout_needs_pair_mode(tiny_encoder, tokenizer, request_obj, packed):
    model = _model(tiny_encoder, readout="mean")
    pair = build_request_batch(request_obj, PairEncoder(tokenizer, 128))
    with torch.no_grad():
        assert model(pair).shape == (pair.n_markers,)
        with pytest.raises(ValueError, match="pair mode"):
            model(packed)
    with pytest.raises(ValueError, match="readout"):
        _model(tiny_encoder, readout="cls")


def test_gradients_reach_heads_and_encoder(tiny_encoder, packed):
    model = _model(tiny_encoder, decision_layers=1).train()
    model.zero_grad()
    model(packed).sum().backward()
    for name in ("trunk.0.weight", "heads.choice.weight", "heads.score.weight", "heads.bool.weight", "decision.layers.0.linear1.weight"):
        assert model.get_parameter(name).grad is not None, name
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert model.log_temperature.requires_grad is False  # a buffer: fitted after training, not learned


def test_lora_keeps_heads_trainable(tokenizer, packed):
    from peft import LoraConfig, get_peft_model
    from transformers import ModernBertConfig, ModernBertModel

    torch.manual_seed(0)
    base = ModernBertModel(ModernBertConfig(
        vocab_size=len(tokenizer), hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, pad_token_id=tokenizer.pad_token_id, reference_compile=False,
    ))
    encoder = get_peft_model(base, LoraConfig(r=4, lora_alpha=8, target_modules=["Wqkv", "Wo", "Wi"]))
    model = _model(encoder)
    heads = model.head_parameters()
    assert heads and all(p.requires_grad for p in heads)
    encoder_trainable = [n for n, p in model.encoder.named_parameters() if p.requires_grad]
    assert encoder_trainable and all("lora_" in n for n in encoder_trainable)
    trainable, total = count_parameters(model)
    assert trainable == sum(p.numel() for p in heads) + sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    assert 0 < trainable / total < 1 and not math.isclose(trainable, total)
