"""jevlite.embed_probe: query format, batching, embedding order and the probe."""

import numpy as np
import pytest
import torch

from jevlite.embed_probe import embed, fit_probe, hard_label, query_ids, sample_bool_records, token_batches

QWEN = "Qwen/Qwen3-Embedding-0.6B"


@pytest.fixture(scope="module")
def qwen_tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(QWEN)
    except OSError as e:
        pytest.skip(f"tokenizer {QWEN} unavailable: {e}")


@pytest.fixture(scope="module")
def tiny_qwen(qwen_tokenizer):
    from transformers import Qwen3Config, Qwen3Model

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(qwen_tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, pad_token_id=qwen_tokenizer.pad_token_id,
                      use_cache=False)
    return Qwen3Model(cfg).eval()


def test_query_format(qwen_tokenizer):
    eot = qwen_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ids = query_ids(qwen_tokenizer, "The text is toxic.", "you are great", 128, eot)
    assert ids[-1] == eot and ids.count(eot) == 1
    assert qwen_tokenizer.decode(ids[:-1]) == (
        "Instruct: Decide whether the statement is true of the text. Statement: The text is toxic.\nQuery:you are great")


def test_query_keeps_head_and_tail_within_max_len(qwen_tokenizer):
    eot = qwen_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    ids = query_ids(qwen_tokenizer, "p", "START " + "filler " * 3000 + "END", 256, eot)
    text = qwen_tokenizer.decode(ids)
    assert len(ids) == 256 and "START" in text and "END" in text and "Statement: p" in text
    long_claim = query_ids(qwen_tokenizer, "claim " * 3000, "the text", 256, eot)
    assert len(long_claim) <= 256 and "the text" in qwen_tokenizer.decode(long_claim)


def test_hard_labels_and_sampling():
    assert [hard_label(t) for t in (0.0, 0.2, 0.5, 0.8, 1.0)] == [0, 0, None, 1, 1]
    rows = [{"type": "bool", "target": t, "id": str(i)} for i, t in enumerate([0.0, 1.0, 0.5, 1.0])]
    rows.append({"type": "choice", "target": 1, "id": "c"})
    assert {r["id"] for r in sample_bool_records(rows, 10)} == {"0", "1", "3"}
    assert len(sample_bool_records(rows, 2, seed=1)) == 2


def test_token_batches_cover_all_within_budget():
    lengths = [5, 100, 30, 30, 7, 300, 1]
    batches = token_batches(lengths, 120)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    for b in batches:
        assert len(b) == 1 or len(b) * max(lengths[i] for i in b) <= 120


def test_embed_is_in_input_order_and_padding_free(tiny_qwen, qwen_tokenizer):
    eot = qwen_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    rows = [query_ids(qwen_tokenizer, "p", s, 128, eot) for s in ("short", "a much longer text " * 5, "mid size text")]
    together = embed(tiny_qwen, rows, qwen_tokenizer.pad_token_id, "cpu", tokens_per_batch=10_000)
    alone = np.concatenate([embed(tiny_qwen, [r], qwen_tokenizer.pad_token_id, "cpu") for r in rows])
    assert together.shape == (3, 64) and together.dtype == np.float16
    np.testing.assert_allclose(together.astype(np.float32), alone.astype(np.float32), atol=2e-3)
    np.testing.assert_allclose(np.linalg.norm(together.astype(np.float32), axis=1), 1.0, atol=2e-3)


def test_fit_probe_learns_a_separable_signal():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 400)
    x = rng.normal(size=(400, 16)) + y[:, None] * 1.5
    clf, info = fit_probe(x, y)
    assert info["val_auroc"][str(info["C"])] > 0.95 and info["n_train"] == 400
    assert clf.predict_proba(x.astype(np.float32))[:, 1][y == 1].mean() > 0.8
