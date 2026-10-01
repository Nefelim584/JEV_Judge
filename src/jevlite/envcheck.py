"""Checks run by ``notebooks/*/00_env_check.ipynb``. Shared so both stacks run identical code."""

from __future__ import annotations

import gc
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from .collate import build_request_batch, group_logits
from .device import PeakMemory, Runtime, autocast, make_grad_scaler, make_optimizer, synchronize
from .encoding import PairEncoder
from .model import build_encoder, count_parameters
from .schema import Request

SAMPLE_REQUEST = {
    "state": {
        "ticket_id": 4812,
        "channel": "email",
        "message": "I was charged twice for order #A-993. Please fix this today or I'm cancelling.",
    },
    "questions": [
        {
            "id": "intent",
            "type": "choice",
            "prompt": "What does the customer want?",
            "options": ["refund", "exchange", "order status", "cancel subscription", "other"],
        },
        {
            "id": "urgency",
            "type": "score",
            "prompt": "How urgent is this ticket?",
            "levels": ["low", "medium", "high", "critical"],
        },
        {"id": "churn_risk", "type": "bool", "prompt": "The customer is threatening to leave."},
    ],
}


def _free(rt: Runtime) -> None:
    gc.collect()
    if rt.device.type == "cuda":
        torch.cuda.empty_cache()
    elif rt.device.type == "mps":
        torch.mps.empty_cache()


def request_forward_check(cfg: dict, rt: Runtime) -> dict:
    """Run the sample request through schema → encoding → collate → encoder → grouping on the device.

    The probe head is random, so the probabilities are meaningless; this only checks shapes and masking.
    """
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["base"])
    encoder = build_encoder(cfg, rt, mode="full").eval()
    probe = torch.nn.Linear(encoder.config.hidden_size, 1).to(rt.device)

    request = Request.model_validate(SAMPLE_REQUEST)
    pair_encoder = PairEncoder(tokenizer, cfg["encoding"]["max_len"], cfg["encoding"]["head_ratio"])
    batch = build_request_batch(request, pair_encoder).to(rt.device)

    with torch.no_grad(), autocast(rt):
        h = encoder(input_ids=batch.input_ids, attention_mask=batch.attention_mask).last_hidden_state[:, 0]
        row_logits = probe(h).squeeze(-1)
    grouped = group_logits(row_logits, batch)
    probs = torch.softmax(grouped, dim=-1)  # Bool rows have K=1 here; the real model uses a sigmoid

    padded_mass = probs.masked_fill(batch.candidate_mask, 0.0).sum().item()
    result = {
        "rows": batch.input_ids.shape[0],
        "seq_len": batch.input_ids.shape[1],
        "grouped_shape": tuple(grouped.shape),
        "padded_candidate_mass": padded_mass,
        "row_sums": probs.sum(-1).tolist(),
        "ok": padded_mass == 0.0 and torch.allclose(probs.sum(-1), torch.ones(len(batch.question_ids), device=rt.device)),
    }
    del encoder, probe, batch
    _free(rt)
    return result


def train_step_check(cfg: dict, rt: Runtime, mode: str, batch_size: int, seq_len: int, steps: int = 3) -> dict:
    """Forward + backward + optimizer steps at training shape; reports peak memory and step time."""
    _free(rt)
    memory = PeakMemory(rt)
    encoder = build_encoder(cfg, rt, mode=mode).train()
    probe = torch.nn.Linear(encoder.config.hidden_size, 1).to(rt.device)
    trainable, total = count_parameters(encoder)

    encoder_params = [p for p in encoder.parameters() if p.requires_grad]
    params = encoder_params + list(probe.parameters())
    lr = cfg["train"]["lr"][mode]
    optimizer = make_optimizer(params, rt, lr=lr, weight_decay=cfg["train"]["weight_decay"])
    scaler = make_grad_scaler(rt)

    vocab = encoder.config.vocab_size
    generator = torch.Generator().manual_seed(cfg["seed"])
    input_ids = torch.randint(1000, vocab - 1000, (batch_size, seq_len), generator=generator).to(rt.device)
    attention_mask = torch.ones_like(input_ids)
    targets = torch.randint(0, 2, (batch_size,), generator=generator).float().to(rt.device)

    # The last trainable encoder tensor: for LoRA that is a lora_B, which gets a gradient from step one.
    watched = encoder_params[-1]
    before = watched.detach().clone()
    losses, step_times = [], []
    memory.reset()
    memory.sample()
    for _ in range(steps):
        synchronize(rt)
        t0 = time.perf_counter()
        with autocast(rt):
            h = encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
            logits = probe(h).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits.float(), targets)
        memory.sample()
        scaler.scale(loss).backward()
        memory.sample()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        memory.sample()
        synchronize(rt)
        step_times.append(time.perf_counter() - t0)
        losses.append(loss.item())

    result = {
        "mode": mode,
        "precision": rt.precision,
        "optimizer": rt.optimizer,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 3),
        "losses": [round(x, 4) for x in losses],
        "params_updated": not torch.equal(before, watched.detach()),
        "first_step_s": round(step_times[0], 2),
        "step_s": round(sum(step_times[1:]) / max(1, len(step_times) - 1), 2),
        "peak_memory_gb": round(memory.peak_bytes / 2**30, 2),
    }
    del encoder, probe, optimizer, params, encoder_params, watched, before
    _free(rt)
    return result
