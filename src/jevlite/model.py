"""Encoder construction. The JevLite heads (TODO 2.5) are added in Phase 5."""

from __future__ import annotations

import torch
from transformers import AutoModel

from .device import Runtime


def build_encoder(cfg: dict, rt: Runtime, mode: str | None = None) -> torch.nn.Module:
    """Load the base encoder in fp32 (AMP casts on the fly), optionally wrapped with LoRA.

    ``mode`` defaults to ``cfg['tuning']['mode']``: ``lora`` or ``full``.
    """
    mode = mode or cfg["tuning"]["mode"]
    encoder = AutoModel.from_pretrained(
        cfg["model"]["base"],
        attn_implementation=rt.attn_implementation,
        dtype=torch.float32,
    )
    if cfg["train"]["gradient_checkpointing"]:
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    if mode == "lora":
        from peft import LoraConfig, get_peft_model

        lora = cfg["tuning"]["lora"]
        encoder = get_peft_model(
            encoder,
            LoraConfig(
                r=lora["r"],
                lora_alpha=lora["alpha"],
                lora_dropout=lora["dropout"],
                target_modules=list(lora["target_modules"]),
                bias="none",
            ),
        )
    elif mode != "full":
        raise ValueError(f"unknown tuning mode {mode!r}, expected 'lora' or 'full'")
    return encoder.to(rt.device)


def count_parameters(module: torch.nn.Module) -> tuple[int, int]:
    """(trainable, total)."""
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total
