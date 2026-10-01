"""Batching: flatten (state, question) items into candidate rows, pad, and map rows back.

Rows are the unit the encoder sees (one per candidate). Questions are the unit losses and
outputs are computed over: logits are regrouped into ``[n_questions, K_max]`` with a mask.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

import torch

from .encoding import PairEncoder
from .schema import PRIM_TYPES, Question, Request, State

TYPE_IDS = {t: i for i, t in enumerate(PRIM_TYPES)}


@dataclass
class Batch:
    input_ids: torch.Tensor  # [R, L] long
    attention_mask: torch.Tensor  # [R, L] long
    row_type: torch.Tensor  # [R] long, TYPE_IDS of the row's question
    row_question: torch.Tensor  # [R] long, question index within the batch
    row_candidate: torch.Tensor  # [R] long, candidate index within its question
    question_type: torch.Tensor  # [Q] long
    candidate_mask: torch.Tensor  # [Q, K_max] bool, True for real candidates
    truncated: torch.Tensor  # [Q] bool
    question_ids: list[str]  # [Q], ids as given by the caller

    @property
    def n_questions(self) -> int:
        return len(self.question_ids)

    @property
    def k_max(self) -> int:
        return self.candidate_mask.shape[1]

    def to(self, device: torch.device | str) -> Batch:
        moved = {
            f.name: (v.to(device) if isinstance(v := getattr(self, f.name), torch.Tensor) else v)
            for f in fields(self)
        }
        return Batch(**moved)


def build_batch(items: Sequence[tuple[State, Question]], encoder: PairEncoder) -> Batch:
    """Encode every candidate of every (state, question) item into one padded batch."""
    if not items:
        raise ValueError("cannot build an empty batch")

    rows: list[list[int]] = []
    row_type, row_question, row_candidate = [], [], []
    question_type, n_candidates, truncated, question_ids = [], [], [], []

    state_cache: dict[int, list[int]] = {}  # states shared between items (one request) are tokenised once
    for q_index, (state, question) in enumerate(items):
        key = id(state)
        if key not in state_cache:
            state_cache[key] = encoder.tokenize_state(state)
        encoded = encoder.encode_question(question, state_cache[key])

        type_id = TYPE_IDS[question.type]
        for k, ids in enumerate(encoded.input_ids):
            rows.append(ids)
            row_type.append(type_id)
            row_question.append(q_index)
            row_candidate.append(k)
        question_type.append(type_id)
        n_candidates.append(len(encoded.input_ids))
        truncated.append(encoded.truncated)
        question_ids.append(question.id)

    input_ids, attention_mask = pad_rows(rows, encoder.tokenizer.pad_token_id)
    k_max = max(n_candidates)
    candidate_mask = torch.arange(k_max)[None, :] < torch.tensor(n_candidates)[:, None]

    return Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        row_type=torch.tensor(row_type),
        row_question=torch.tensor(row_question),
        row_candidate=torch.tensor(row_candidate),
        question_type=torch.tensor(question_type),
        candidate_mask=candidate_mask,
        truncated=torch.tensor(truncated),
        question_ids=question_ids,
    )


def build_request_batch(request: Request, encoder: PairEncoder) -> Batch:
    """All candidates of all questions of a request in one batch (the v1 "single parallel pass")."""
    return build_batch([(request.state, q) for q in request.questions], encoder)


def pad_rows(rows: list[list[int]], pad_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-pad to the longest row."""
    length = max(len(r) for r in rows)
    input_ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(rows), length), dtype=torch.long)
    for i, r in enumerate(rows):
        input_ids[i, : len(r)] = torch.tensor(r, dtype=torch.long)
        attention_mask[i, : len(r)] = 1
    return input_ids, attention_mask


def group_logits(row_logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    """``[R]`` row logits → ``[Q, K_max]`` fp32, with padded candidates set to ``-inf``."""
    grouped = torch.full(
        (batch.n_questions, batch.k_max), float("-inf"), dtype=torch.float32, device=row_logits.device
    )
    grouped[batch.row_question, batch.row_candidate] = row_logits.float()
    return grouped


def ungroup(grouped: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Inverse of :func:`group_logits`: ``[Q, K_max]`` → ``[R]`` in row order."""
    return grouped[batch.row_question, batch.row_candidate]
