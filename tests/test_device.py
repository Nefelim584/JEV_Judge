import pytest
import torch

from jevlite.config import load_config
from jevlite.device import autocast, make_grad_scaler, make_optimizer, resolve_runtime


def _cfg(**runtime):
    cfg = load_config("apple_silicon")
    cfg["runtime"] = {"device": "cpu", "precision": "fp32", "attn_implementation": "sdpa", "optimizer": "adamw"} | runtime
    return cfg


def test_cpu_fp32_runtime():
    rt = resolve_runtime(_cfg())
    assert rt.device.type == "cpu" and rt.amp_dtype is None and not rt.use_grad_scaler
    assert not make_grad_scaler(rt).is_enabled()
    with autocast(rt):
        assert torch.ones(2, 2).matmul(torch.ones(2, 2)).dtype == torch.float32
    assert isinstance(make_optimizer([torch.nn.Parameter(torch.zeros(1))], rt, 1e-3, 0.0), torch.optim.AdamW)


def test_auto_precision_on_non_cuda_is_fp32():
    assert resolve_runtime(_cfg(precision="auto", attn_implementation="auto")).precision == "fp32"


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-CUDA error path")
def test_cuda_requested_without_cuda_raises():
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_runtime(_cfg(device="cuda"))


def test_8bit_adam_requires_cuda():
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_runtime(_cfg(optimizer="adamw_8bit"))


def test_flash_attention_requires_cuda():
    with pytest.raises(RuntimeError, match="flash_attention_2"):
        resolve_runtime(_cfg(attn_implementation="flash_attention_2"))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="needs MPS")
def test_mps_half_precision_warns():
    with pytest.warns(UserWarning, match="MPS"):
        resolve_runtime(_cfg(device="mps", precision="fp16"))
