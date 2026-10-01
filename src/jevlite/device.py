"""Device, precision, attention implementation and optimizer selection from config.

Everything stack-specific that is not a config value lives here, so model / training code
stays identical on MPS, CUDA and CPU.
"""

from __future__ import annotations

import contextlib
import importlib.util
import platform
import warnings
from dataclasses import dataclass

import torch

_AMP_DTYPES = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    precision: str  # fp32 | fp16 | bf16
    attn_implementation: str
    optimizer: str  # adamw | adamw_8bit

    @property
    def amp_dtype(self) -> torch.dtype | None:
        return _AMP_DTYPES[self.precision]

    @property
    def use_grad_scaler(self) -> bool:
        return self.device.type == "cuda" and self.precision == "fp16"


def _cuda_capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(0)


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("runtime.device=cuda but CUDA is not available")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("runtime.device=mps but MPS is not available")
    return torch.device(name)


def _resolve_precision(name: str, device: torch.device) -> str:
    if name == "auto":
        if device.type == "cuda":
            return "bf16" if _cuda_capability() >= (8, 0) else "fp16"
        return "fp32"
    if name not in _AMP_DTYPES:
        raise ValueError(f"unknown precision {name!r}")
    if device.type == "mps" and name != "fp32":
        warnings.warn(f"precision={name} on MPS is not recommended; use fp32", stacklevel=3)
    return name


def _resolve_attn(name: str, device: torch.device) -> str:
    has_flash = importlib.util.find_spec("flash_attn") is not None
    if name == "auto":
        if device.type == "cuda" and _cuda_capability() >= (8, 0) and has_flash:
            return "flash_attention_2"
        return "sdpa"
    if name == "flash_attention_2" and not (device.type == "cuda" and has_flash):
        raise RuntimeError("flash_attention_2 needs CUDA (Ampere+) and the flash-attn package")
    return name


def _resolve_optimizer(name: str, device: torch.device) -> str:
    if name == "adamw_8bit":
        if device.type != "cuda":
            raise RuntimeError("adamw_8bit (bitsandbytes) is only supported on CUDA")
        if importlib.util.find_spec("bitsandbytes") is None:
            raise RuntimeError("adamw_8bit needs bitsandbytes: install the [cuda] extra")
    elif name != "adamw":
        raise ValueError(f"unknown optimizer {name!r}")
    return name


def resolve_runtime(cfg: dict) -> Runtime:
    rt = cfg["runtime"]
    device = _resolve_device(rt["device"])
    return Runtime(
        device=device,
        precision=_resolve_precision(rt["precision"], device),
        attn_implementation=_resolve_attn(rt["attn_implementation"], device),
        optimizer=_resolve_optimizer(rt["optimizer"], device),
    )


def autocast(rt: Runtime):
    """Mixed-precision context. A no-op under fp32."""
    if rt.amp_dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=rt.device.type, dtype=rt.amp_dtype)


def make_grad_scaler(rt: Runtime) -> torch.amp.GradScaler:
    """GradScaler, enabled only for fp16 on CUDA. Disabled scalers pass calls straight through."""
    return torch.amp.GradScaler(rt.device.type, enabled=rt.use_grad_scaler)


def make_optimizer(params, rt: Runtime, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if rt.optimizer == "adamw_8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def synchronize(rt: Runtime) -> None:
    if rt.device.type == "cuda":
        torch.cuda.synchronize()
    elif rt.device.type == "mps":
        torch.mps.synchronize()


class PeakMemory:
    """Peak device memory in bytes.

    CUDA tracks the peak natively. MPS has no peak counter, so call :meth:`sample` at the
    points of interest (after forward, after backward, ...) and the maximum is kept.
    """

    def __init__(self, rt: Runtime):
        self.rt = rt
        self._peak = 0
        self.reset()

    def reset(self) -> None:
        self._peak = 0
        if self.rt.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def sample(self) -> None:
        if self.rt.device.type == "mps":
            synchronize(self.rt)
            self._peak = max(self._peak, torch.mps.current_allocated_memory())

    @property
    def peak_bytes(self) -> int:
        if self.rt.device.type == "cuda":
            return torch.cuda.max_memory_allocated()
        return self._peak


def device_report(rt: Runtime) -> dict:
    """Versions and hardware facts for the env-check notebooks and run metadata."""
    report = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(rt.device),
        "precision": rt.precision,
        "attn_implementation": rt.attn_implementation,
        "optimizer": rt.optimizer,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "bitsandbytes_installed": importlib.util.find_spec("bitsandbytes") is not None,
        "flash_attn_installed": importlib.util.find_spec("flash_attn") is not None,
    }
    if rt.device.type == "cuda":
        report["cuda_device_count"] = torch.cuda.device_count()
        report["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        report["cuda_capability"] = _cuda_capability()
        report["bf16_supported"] = torch.cuda.is_bf16_supported()
        report["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    elif rt.device.type == "mps":
        report["mps_recommended_max_gb"] = round(torch.mps.recommended_max_memory() / 2**30, 1)
    return report
