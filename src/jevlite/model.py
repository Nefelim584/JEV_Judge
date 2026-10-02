"""The jev-lite model (TODO 2.5): encoder → read-out at the markers → shared trunk → one head per primitive.

    model = build_model(cfg, rt)              # encoder (LoRA or full FT) + heads, on rt.device
    marker_logits = model(batch)              # [M], one logit per candidate marker, divided by T
    grouped = model.grouped_logits(batch)     # [Q, K_max], padded candidates = -inf
    probs = question_probs(grouped, batch.question_type)

LoRA wraps the encoder only (``build_encoder``); the trunk, heads and the optional decision layers sit
outside the PEFT wrapper, so they are always fully trainable.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModel

from .collate import TYPE_IDS, Batch, gather_markers, group_logits
from .device import Runtime
from .log import logger
from .schema import PRIM_TYPES

READOUTS = ("marker", "mean")


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


class JevLite(nn.Module):
    """Scores every candidate marker of a batch.

    ``readout``:
      - ``marker``: the hidden state at each marker: ``[MASK]`` in packed mode, ``[CLS]`` of the
        candidate's row in pair mode;
      - ``mean``: the mean over the candidate's row (pair mode only: one candidate per row).

    ``decision_layers > 0`` adds a Laya-style block of transformer layers over the encoder output
    before the read-out (an ablation, TODO 2.5; 2 layers ≈ 25M parameters at hidden size 1024).

    Temperatures are a buffer ``log_temperature`` per primitive, 0 (T = 1) during training and fitted
    afterwards on ``calib`` (TODO 2.7).
    """

    def __init__(
        self,
        encoder: nn.Module,
        hidden: int,
        trunk_dim: int = 256,
        dropout: float = 0.1,
        readout: str = "marker",
        decision_layers: int = 0,
        decision_heads: int = 16,
    ):
        super().__init__()
        if readout not in READOUTS:
            raise ValueError(f"unknown readout {readout!r}, expected one of {READOUTS}")
        self.encoder = encoder
        self.readout = readout
        self.decision = (
            nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    hidden, decision_heads, dim_feedforward=4 * hidden, dropout=dropout, activation="gelu",
                    batch_first=True, norm_first=True,
                ),
                num_layers=decision_layers,
                enable_nested_tensor=False,
            )
            if decision_layers
            else None
        )
        self.trunk = nn.Sequential(nn.Linear(hidden, trunk_dim), nn.GELU(), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({t: nn.Linear(trunk_dim, 1) for t in PRIM_TYPES})
        self.register_buffer("log_temperature", torch.zeros(len(PRIM_TYPES)))

    def _read_out(self, hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
        if self.readout == "marker":
            return gather_markers(hidden, batch)
        if torch.unique(batch.marker_row).numel() != batch.n_markers:
            raise ValueError("readout 'mean' needs one candidate per row (pair mode)")
        mask = batch.attention_mask.unsqueeze(-1).to(hidden.dtype)
        mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        return mean[batch.marker_row]

    def forward(self, batch: Batch, apply_temperature: bool = True) -> torch.Tensor:
        """``[M]`` fp32 logits, one per candidate marker."""
        hidden = self.encoder(input_ids=batch.input_ids, attention_mask=batch.attention_mask).last_hidden_state
        if self.decision is not None:
            hidden = self.decision(hidden, src_key_padding_mask=batch.attention_mask == 0)
        z = self.trunk(self._read_out(hidden, batch))
        per_head = torch.cat([self.heads[t](z) for t in PRIM_TYPES], dim=-1).float()  # [M, 3]
        logits = per_head.gather(1, batch.marker_type[:, None]).squeeze(1)
        if apply_temperature:
            logits = logits / self.log_temperature.float()[batch.marker_type].exp()
        return logits

    def grouped_logits(self, batch: Batch, apply_temperature: bool = True) -> torch.Tensor:
        """``[Q, K_max]`` fp32 logits per question; padded candidates are ``-inf``."""
        return group_logits(self(batch, apply_temperature), batch)

    def head_parameters(self) -> list[nn.Parameter]:
        """Everything outside the encoder: decision layers, trunk and heads."""
        encoder_ids = {id(p) for p in self.encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in encoder_ids]


def question_probs(grouped: torch.Tensor, question_type: torch.Tensor) -> torch.Tensor:
    """``[Q, K_max]`` probabilities from grouped logits.

    Choice and Score: softmax over the real candidates (padded ones get 0). Bool: an independent
    sigmoid of its single logit in column 0, the rest 0. Claims of a Bool claim group are separate
    questions, so they never normalise against each other.
    """
    probs = torch.zeros_like(grouped, dtype=torch.float32)
    is_bool = question_type == TYPE_IDS["bool"]
    if (~is_bool).any():
        probs[~is_bool] = F.softmax(grouped[~is_bool].float(), dim=-1)
    if is_bool.any():
        probs[is_bool, 0] = torch.sigmoid(grouped[is_bool, 0].float())
    return probs


def build_model(cfg: dict, rt: Runtime, mode: str | None = None) -> JevLite:
    """The encoder from ``build_encoder`` plus the heads from ``cfg['model']``, on ``rt.device``."""
    encoder = build_encoder(cfg, rt, mode)
    m = cfg["model"]
    model = JevLite(
        encoder,
        hidden=encoder.config.hidden_size,
        trunk_dim=m["trunk_dim"],
        dropout=m["dropout"],
        readout=m.get("readout", "marker"),
        decision_layers=m.get("decision_layers", 0),
    ).to(rt.device)
    trainable, total = count_parameters(model)
    head = sum(p.numel() for p in model.head_parameters())
    logger.info(
        "model {} ({}): {:,} / {:,} parameters trainable ({:.2%}), heads {:,}",
        m["base"], mode or cfg["tuning"]["mode"], trainable, total, trainable / total, head,
    )
    return model
