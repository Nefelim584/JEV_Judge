"""Training losses (TODO 2.6): one loss per primitive over grouped logits, all in fp32.

    target = loss_targets([r.target for r in records], batch)        # [Q, K_max]
    total, parts = jev_loss(model.grouped_logits(batch, apply_temperature=False), target, batch, LossConfig.from_config(cfg))

| Primitive | Loss |
|---|---|
| Choice | cross-entropy over the options |
| Score  | cross-entropy over the levels + ``emd_lambda · EMD²`` between the predicted and target CDFs |
| Bool   | binary cross-entropy + ``brier_mu · (p − t)²`` |

Targets are soft everywhere: a class index becomes a one-hot row, a distribution is used as is, and a
Bool target is any number in [0, 1]. ``EMD²`` is the squared distance between CDFs averaged over the
``K − 1`` level boundaries, i.e. the ranked probability score, so ``emd_lambda`` means the same for 3
and for 10 levels.

The total is ``Σ_type w_type · mean(L_type)`` over the primitives present in the batch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn.functional as F

from .collate import TYPE_IDS, Batch
from .schema import PRIM_TYPES


@dataclass(frozen=True)
class LossConfig:
    weights: dict[str, float] = field(default_factory=lambda: {t: 1.0 for t in PRIM_TYPES})
    emd_lambda: float = 0.5
    brier_mu: float = 0.1

    @classmethod
    def from_config(cls, cfg: dict) -> LossConfig:
        loss = cfg.get("loss", {})
        weights = {t: float(loss.get("weights", {}).get(t, 1.0)) for t in PRIM_TYPES}
        return cls(weights=weights, emd_lambda=float(loss.get("emd_lambda", 0.5)), brier_mu=float(loss.get("brier_mu", 0.1)))


def loss_targets(targets: Sequence[int | float | Sequence[float]], batch: Batch) -> torch.Tensor:
    """``[Q, K_max]`` fp32 target rows in question order. Choice / Score: a distribution over the real
    candidates (zeros on padding). Bool: the target probability in column 0."""
    if len(targets) != batch.n_questions:
        raise ValueError(f"{len(targets)} targets for {batch.n_questions} questions")
    out = torch.zeros(batch.n_questions, batch.k_max, dtype=torch.float32)
    n_real = batch.candidate_mask.sum(-1).tolist()
    for q, (t, qtype, k) in enumerate(zip(targets, batch.question_type.tolist(), n_real, strict=True)):
        if qtype == TYPE_IDS["bool"]:
            if not isinstance(t, (int, float)) or not 0.0 <= float(t) <= 1.0:
                raise ValueError(f"question {batch.question_ids[q]!r}: bool target must be in [0, 1], got {t!r}")
            out[q, 0] = float(t)
        elif isinstance(t, (int, float)):
            if float(t) != int(t) or not 0 <= int(t) < k:
                raise ValueError(f"question {batch.question_ids[q]!r}: target index {t!r} out of range for {k} candidates")
            out[q, int(t)] = 1.0
        else:
            if len(t) != k:
                raise ValueError(f"question {batch.question_ids[q]!r}: soft target has {len(t)} entries for {k} candidates")
            out[q, :k] = torch.tensor([float(x) for x in t])
    return out.to(batch.candidate_mask.device)


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``[N]`` cross-entropy of soft targets; padded candidates (``mask`` False) are ignored."""
    logp = F.log_softmax(logits.float().masked_fill(~mask, float("-inf")), dim=-1)
    # -inf · 0 would be NaN: padded entries contribute nothing.
    return -torch.where(mask, target * logp, torch.zeros_like(logp)).sum(-1)


def emd2(probs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """``[N]`` squared distance between the CDFs of ``probs`` and ``target`` over ordered levels,
    averaged over the ``K − 1`` boundaries (the ranked probability score). 0 for K = 1."""
    diff = (probs.float().cumsum(-1) - target.float().cumsum(-1)) ** 2
    # Boundaries are 0..K-2 of each row: the last real level's CDF is 1 for both, padding adds 0.
    k = mask.sum(-1)
    boundary = torch.arange(mask.shape[-1], device=mask.device)[None, :] < (k - 1)[:, None]
    return (diff * boundary).sum(-1) / (k - 1).clamp_min(1)


def jev_loss(
    grouped: torch.Tensor, target: torch.Tensor, batch: Batch, cfg: LossConfig = LossConfig()
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``(total, parts)``: the weighted total and the mean loss of every primitive present (detached)."""
    grouped, target = grouped.float(), target.float()
    mask = batch.candidate_mask
    parts: dict[str, torch.Tensor] = {}
    total = grouped.new_zeros(())
    for t in PRIM_TYPES:
        rows = batch.question_type == TYPE_IDS[t]
        if not rows.any():
            continue
        g, y, m = grouped[rows], target[rows], mask[rows]
        if t == "bool":
            logit, p_true = g[:, 0], y[:, 0]
            loss = F.binary_cross_entropy_with_logits(logit, p_true, reduction="none")
            if cfg.brier_mu:
                loss = loss + cfg.brier_mu * (torch.sigmoid(logit) - p_true) ** 2
        else:
            loss = soft_cross_entropy(g, y, m)
            if t == "score" and cfg.emd_lambda:
                probs = F.softmax(g.masked_fill(~m, float("-inf")), dim=-1)
                loss = loss + cfg.emd_lambda * emd2(probs, y, m)
        mean = loss.mean()
        parts[t] = mean.detach()
        total = total + cfg.weights.get(t, 1.0) * mean
    return total, parts
