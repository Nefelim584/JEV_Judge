"""pair_last: pair mode for causal backbones (Qwen3-Embedding), read out at the last token."""

import pytest
import torch

from jevlite.collate import build_batch, build_request_batch
from jevlite.encoding import LastTokenPairEncoder, PairEncoder, segment_a_texts, text_encoder_from_config
from jevlite.model import JevLite, disable_kv_cache, load_base_encoder
from jevlite.schema import BoolQuestion, ChoiceQuestion, Request, ScoreQuestion

QWEN = "Qwen/Qwen3-Embedding-0.6B"
HIDDEN = 64


@pytest.fixture(scope="module")
def qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(QWEN)
    except OSError as e:  # not cached and no network
        pytest.skip(f"tokenizer {QWEN} unavailable: {e}")


@pytest.fixture(scope="module")
def tiny_qwen(qwen_tokenizer):
    """A random 2-layer Qwen3 with the real vocabulary: no weights to download."""
    from transformers import Qwen3Config, Qwen3Model

    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=len(qwen_tokenizer), hidden_size=HIDDEN, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, pad_token_id=qwen_tokenizer.pad_token_id,
    )
    return disable_kv_cache(Qwen3Model(cfg)).eval()


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


def _model(encoder):
    torch.manual_seed(0)
    return JevLite(encoder, hidden=HIDDEN, trunk_dim=16, dropout=0.0).eval()


def test_layout_state_first_readout_last(qwen_tokenizer):
    enc = LastTokenPairEncoder(qwen_tokenizer, max_len=512)
    q = ChoiceQuestion(id="q", prompt="Which team?", options=["billing", "tech"])
    state = "My card was charged twice."
    seqs = enc.encode_item(q, enc.tokenize_state(state))
    expected_a = qwen_tokenizer(segment_a_texts(q), add_special_tokens=False)["input_ids"]
    state_ids = qwen_tokenizer(state, add_special_tokens=False)["input_ids"]
    assert len(seqs) == 2
    for k, (seq, a) in enumerate(zip(seqs, expected_a, strict=True)):
        assert seq.input_ids == [*state_ids, *enc.sep_ids, *a, enc.readout_id]
        assert seq.markers == [(len(seq.input_ids) - 1, 0, k)]
        assert seq.input_ids[-1] == qwen_tokenizer.convert_tokens_to_ids("<|endoftext|>")


def test_literal_readout_token_in_text_stays_plain(qwen_tokenizer):
    enc = LastTokenPairEncoder(qwen_tokenizer, max_len=512)
    seq = enc.encode_item(BoolQuestion(id="b", prompt="p <|endoftext|>"), enc.tokenize_state("x <|endoftext|> y"))[0]
    assert seq.input_ids.count(enc.readout_id) == 1  # only the real read-out token


@pytest.mark.parametrize("max_len", [32, 64, 128])
def test_question_never_truncated_and_state_shared(qwen_tokenizer, max_len):
    enc = LastTokenPairEncoder(qwen_tokenizer, max_len=max_len, head_ratio=0.5)
    q = ChoiceQuestion(id="q", prompt="Which department should handle it?", options=["billing", "technical support team", "x"])
    encoded = enc.encode_question(q, enc.tokenize_state(" ".join(f"word{i}" for i in range(2000))))
    assert encoded.truncated
    expected_a = qwen_tokenizer(segment_a_texts(q), add_special_tokens=False)["input_ids"]
    states = []
    for row, a in zip(encoded.input_ids, expected_a, strict=True):
        assert len(row) <= max_len
        assert row[-1 - len(a) : -1] == a
        states.append(row[: -1 - len(a) - len(enc.sep_ids)])
    assert all(s == states[0] for s in states)
    assert max(len(r) for r in encoded.input_ids) == max_len


def test_config_selects_pair_last(qwen_tokenizer):
    cfg = {"encoding": {"mode": "pair_last", "max_len": 256, "head_ratio": 0.5}}
    assert isinstance(text_encoder_from_config(cfg, qwen_tokenizer), LastTokenPairEncoder)


def test_markers_line_up_with_pair_mode(qwen_tokenizer, tokenizer, request_obj):
    """Same questions and candidates in the same order as ModernBERT pair mode; only positions differ."""
    last = build_request_batch(request_obj, LastTokenPairEncoder(qwen_tokenizer, 256))
    pair = build_request_batch(request_obj, PairEncoder(tokenizer, 256))
    for name in ("marker_row", "marker_type", "marker_question", "marker_candidate", "question_type", "candidate_mask"):
        assert torch.equal(getattr(last, name), getattr(pair, name)), name
    lengths = last.attention_mask.sum(1)
    assert torch.equal(last.marker_pos, lengths[last.marker_row] - 1)


def test_logits_do_not_depend_on_padding(tiny_qwen, qwen_tokenizer, request_obj):
    """Right padding: the last real token never attends to the pads, so a batch equals its rows one by one."""
    model = _model(tiny_qwen)
    enc = LastTokenPairEncoder(qwen_tokenizer, 256)
    batch = build_request_batch(request_obj, enc)
    assert batch.attention_mask.min() == 0  # rows of different lengths
    with torch.no_grad():
        together = model(batch)
        alone = torch.cat([model(build_batch([(request_obj.state, q)], enc)) for q in request_obj.questions])
    torch.testing.assert_close(together, alone, rtol=1e-4, atol=1e-5)


def test_readout_sees_the_state(tiny_qwen, qwen_tokenizer):
    model = _model(tiny_qwen)
    enc = LastTokenPairEncoder(qwen_tokenizer, 256)
    q = BoolQuestion(id="b", prompt="The customer is angry.")
    with torch.no_grad():
        a = model(build_batch([("I am furious.", q)], enc))
        b = model(build_batch([("Thanks, all good.", q)], enc))
    assert not torch.allclose(a, b)


def test_lora_trains_adapters_and_heads_only(tiny_qwen, qwen_tokenizer, request_obj):
    peft = pytest.importorskip("peft")
    import copy

    encoder = peft.get_peft_model(
        copy.deepcopy(tiny_qwen),
        peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, bias="none",
                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
    )
    model = _model(encoder).train()
    model(build_request_batch(request_obj, LastTokenPairEncoder(qwen_tokenizer, 256))).sum().backward()
    with_grad = {n for n, p in model.named_parameters() if p.grad is not None and p.grad.abs().sum() > 0}
    assert any("lora_" in n for n in with_grad)
    assert any(n.startswith("heads.") for n in with_grad)
    assert all("lora_" in n or not n.startswith("encoder.") for n in with_grad)


def test_bf16_base_needs_lora():
    cfg = {"model": {"init": "base", "base": QWEN, "base_dtype": "bf16"}, "tuning": {"mode": "full"}}
    with pytest.raises(ValueError, match="needs tuning.mode: lora"):
        load_base_encoder(cfg, rt=None)
