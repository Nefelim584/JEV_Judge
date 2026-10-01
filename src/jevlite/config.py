"""Config loading: ``base.yaml`` plus one stack override file.

This module must not import torch: notebooks call :func:`apply_env` before torch is imported,
so that variables like ``PYTORCH_ENABLE_MPS_FALLBACK`` take effect.
"""

from __future__ import annotations

import copy
import os
import random
from pathlib import Path
from typing import Any

import yaml

STACKS = ("apple_silicon", "cuda")
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

# Stack files may only touch these keys (dotted prefixes). Everything else must stay identical
# across stacks so that experiments are comparable (see the parity rules in TODO section 3).
STACK_OVERRIDABLE = ("runtime", "env", "tuning.mode", "train.per_device_batch_size")


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _leaf_paths(d: dict, prefix: str = "") -> list[str]:
    paths = []
    for key, value in d.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            paths.extend(_leaf_paths(value, f"{path}."))
        else:
            paths.append(path)
    return paths


def _check_stack_overrides(stack: str, override: dict) -> None:
    for path in _leaf_paths(override):
        if not any(path == p or path.startswith(f"{p}.") for p in STACK_OVERRIDABLE):
            raise ValueError(
                f"{stack}.yaml overrides '{path}', but stack files may only override {STACK_OVERRIDABLE}"
            )


def _read_yaml(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def load_config(stack: str, overrides: dict | None = None, config_dir: str | Path | None = None) -> dict[str, Any]:
    """Return ``base.yaml`` merged with ``<stack>.yaml`` and then with ``overrides`` (experiment-level)."""
    if stack not in STACKS:
        raise ValueError(f"unknown stack {stack!r}, expected one of {STACKS}")
    config_dir = Path(config_dir) if config_dir is not None else DEFAULT_CONFIG_DIR
    stack_override = _read_yaml(config_dir / f"{stack}.yaml")
    _check_stack_overrides(stack, stack_override)

    cfg = _deep_merge(_read_yaml(config_dir / "base.yaml"), stack_override)
    if overrides:
        cfg = _deep_merge(cfg, overrides)
    cfg["stack"] = stack
    return cfg


def apply_env(cfg: dict) -> None:
    """Export ``cfg['env']`` into ``os.environ``. Call before importing torch."""
    for key, value in (cfg.get("env") or {}).items():
        os.environ[key] = str(value)


def grad_accum_steps(cfg: dict, world_size: int = 1) -> int:
    """Accumulation steps that keep ``train.effective_batch_size`` fixed across stacks."""
    effective = cfg["train"]["effective_batch_size"]
    per_step = cfg["train"]["per_device_batch_size"] * world_size
    if effective % per_step:
        raise ValueError(
            f"effective_batch_size={effective} is not divisible by per_device_batch_size*world_size={per_step}"
        )
    return effective // per_step


def seed_everything(seed: int):
    """Seed Python, NumPy and torch (all devices). Returns a torch.Generator for data order."""
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds CUDA and MPS
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator
