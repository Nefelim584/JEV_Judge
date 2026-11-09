import pytest
import torch

from jevlite.encoding import PackedEncoder
from jevlite.inference import Predictor
from jevlite.latency import (
    DEFAULT_VARIANTS,
    fit_request,
    report_markdown,
    rows_long,
    run_benchmark,
    summarize,
    time_latency,
    time_throughput,
    unsupported_reason,
    variants_by_name,
)
from jevlite.model import JevLite

HIDDEN = 32


@pytest.fixture(scope="module")
def predictor(tokenizer):
    from transformers import ModernBertConfig, ModernBertModel

    torch.manual_seed(0)
    encoder = ModernBertModel(ModernBertConfig(
        vocab_size=len(tokenizer), hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=1,
        num_attention_heads=2, pad_token_id=tokenizer.pad_token_id, reference_compile=False,
    ))
    return Predictor(JevLite(encoder, hidden=HIDDEN, trunk_dim=8, dropout=0.0), PackedEncoder(tokenizer, 1024))


@pytest.mark.parametrize("n, claims", [(128, 1), (1000, 1), (256, 3)])
def test_fit_request_fills_the_budget_untruncated(predictor, n, claims):
    request, length = fit_request(predictor.encoder, n, claims)
    assert len(request.questions) == claims
    assert n - 40 <= length <= n  # one filler sentence is < 40 tokens


def test_fit_request_needs_room(predictor):
    with pytest.raises(ValueError):
        fit_request(predictor.encoder, 4096)


def test_time_latency_and_throughput(predictor):
    request, tokens = fit_request(predictor.encoder, 128)
    r = time_latency(predictor, request, n_warmup=1, n_runs=3)
    assert r["e2e_ms"]["n"] == r["forward_ms"]["n"] == len(r["samples"]["e2e_ms"]) == 3
    assert len(r["probs"]) == 1 and 0 <= r["probs"][0] <= 1
    th = time_throughput(predictor, request, tokens, batch=4, n_warmup=1, n_runs=2)
    assert th["batch"] == 4 and th["tokens_per_s"] == pytest.approx(th["requests_per_s"] * tokens)


def test_summarize():
    s = summarize([1, 2, 3, 4, 100])
    assert s["p50"] == 3 and s["min"] == 1 and s["max"] == 100 and s["n"] == 5 and s["p99"] > s["p95"] > 4 and s["std"] > 0


def test_variants_and_capabilities():
    assert variants_by_name(None) == list(DEFAULT_VARIANTS)
    assert [v.name for v in variants_by_name(["bf16-flex"])] == ["bf16-flex"]
    with pytest.raises(ValueError):
        variants_by_name(["fp8"])
    fa2, fa3 = variants_by_name(["bf16-fa2", "bf16-fa3"])
    assert unsupported_reason(fa3, (9, 0)) is None and unsupported_reason(fa2, (9, 0)) is None
    assert unsupported_reason(fa3, (8, 0)) and unsupported_reason(fa3, (8, 9)) and unsupported_reason(fa3, (10, 0))
    assert unsupported_reason(fa2, (8, 9)) is None and unsupported_reason(fa2, (7, 5))
    assert unsupported_reason(fa2, None) and unsupported_reason(variants_by_name(["bf16-sdpa"])[0], None) is None


def test_run_benchmark_grid_and_reports(predictor, monkeypatch):
    import jevlite.latency as lat

    monkeypatch.setattr(lat, "load_variant", lambda final, variant, device: predictor)
    res = run_benchmark("unused", variants_by_name(["fp32-sdpa", "bf16-sdpa", "bf16-fa3"]), [128, 256], "cpu",
                        n_warmup=1, n_runs=2, throughput_batch=2, n_throughput=1, log=lambda s: None)
    assert res["variants"]["bf16-fa3"]["skipped"]
    assert set(res["variants"]["bf16-sdpa"]["lengths"]) == {"128", "256"}
    assert res["variants"]["bf16-sdpa"]["lengths"]["256"]["max_abs_dp"] == 0.0  # same predictor as the baseline
    rows = rows_long([res])
    assert len([r for r in rows if r["status"] == "ok"]) == 4 and any(r["variant"] == "bf16-fa3" for r in rows)
    md = report_markdown([res])
    assert "| 256 |" in md and "n/a" in md and "All GPUs at 256 tokens" in md
