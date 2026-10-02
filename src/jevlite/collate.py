"""Batching: encode (state, item) pairs into padded sequences and map markers back to questions.

Sequences are the unit the encoder sees. Markers are the read-out slots: a ``[MASK]`` position per
candidate in packed mode, ``[CLS]`` of each candidate row in pair mode. Questions are the unit losses and
outputs are computed over: marker logits are regrouped into ``[n_questions, K_max]`` with a mask, the
same way for both modes.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Sequence

import torch

from .encoding import Item, PackedEncoder, PairEncoder, item_questions
from .schema import PRIM_TYPES, Request, State

TYPE_IDS = {t: i for i, t in enumerate(PRIM_TYPES)}


@dataclass
class Batch:
    input_ids: torch.Tensor  # [R, L] long
    attention_mask: torch.Tensor  # [R, L] long
    marker_row: torch.Tensor  # [M] long, sequence that holds the marker
    marker_pos: torch.Tensor  # [M] long, token position of the marker in its sequence
    marker_type: torch.Tensor  # [M] long, TYPE_IDS of the marker's question
    marker_question: torch.Tensor  # [M] long, question index within the batch
    marker_candidate: torch.Tensor  # [M] long, candidate index within its question
    question_type: torch.Tensor  # [Q] long
    candidate_mask: torch.Tensor  # [Q, K_max] bool, True for real candidates
    truncated: torch.Tensor  # [Q] bool
    question_ids: list[str]  # [Q], ids as given by the caller

    @property
    def n_questions(self) -> int:
        return len(self.question_ids)

    @property
    def n_markers(self) -> int:
        return self.marker_row.shape[0]

    @property
    def k_max(self) -> int:
        return self.candidate_mask.shape[1]

    def to(self, device: torch.device | str) -> Batch:
        moved = {
            f.name: (v.to(device) if isinstance(v := getattr(self, f.name), torch.Tensor) else v)
            for f in fields(self)
        }
        return Batch(**moved)


def build_batch(items: Sequence[tuple[State, Item]], encoder: PairEncoder | PackedEncoder) -> Batch:
    """Encode every (state, item) pair into one padded batch. An item is a question or a claim group."""
    if not items:
        raise ValueError("cannot build an empty batch")

    rows: list[list[int]] = []
    marker_row, marker_pos, marker_type, marker_question, marker_candidate = [], [], [], [], []
    question_type, n_candidates, truncated, question_ids = [], [], [], []

    state_cache: dict[int, list[int]] = {}  # states shared between items (one request) are tokenised once
    for state, item in items:
        key = id(state)
        if key not in state_cache:
            state_cache[key] = encoder.tokenize_state(state)

        questions = item_questions(item)
        q_offset = len(question_ids)
        q_truncated = [False] * len(questions)
        for seq in encoder.encode_item(item, state_cache[key]):
            for pos, j, k in seq.markers:
                marker_row.append(len(rows))
                marker_pos.append(pos)
                marker_type.append(TYPE_IDS[questions[j].type])
                marker_question.append(q_offset + j)
                marker_candidate.append(k)
                q_truncated[j] |= seq.truncated
            rows.append(seq.input_ids)

        for question, was_truncated in zip(questions, q_truncated, strict=True):
            question_type.append(TYPE_IDS[question.type])
            n_candidates.append(len(question.candidates))
            truncated.append(was_truncated)
            question_ids.append(question.id)

    input_ids, attention_mask = pad_rows(rows, encoder.tokenizer.pad_token_id)
    k_max = max(n_candidates)
    candidate_mask = torch.arange(k_max)[None, :] < torch.tensor(n_candidates)[:, None]

    return Batch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        marker_row=torch.tensor(marker_row),
        marker_pos=torch.tensor(marker_pos),
        marker_type=torch.tensor(marker_type),
        marker_question=torch.tensor(marker_question),
        marker_candidate=torch.tensor(marker_candidate),
        question_type=torch.tensor(question_type),
        candidate_mask=candidate_mask,
        truncated=torch.tensor(truncated),
        question_ids=question_ids,
    )


def build_request_batch(request: Request, encoder: PairEncoder | PackedEncoder) -> Batch:
    """All questions of a request in one batch, one item per question (the v1 "single parallel pass")."""
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


def gather_markers(hidden: torch.Tensor, batch: Batch) -> torch.Tensor:
    """``[R, L, H]`` encoder output → ``[M, H]`` hidden states at the markers."""
    return hidden[batch.marker_row, batch.marker_pos]


def group_logits(marker_logits: torch.Tensor, batch: Batch) -> torch.Tensor:
    """``[M]`` marker logits → ``[Q, K_max]`` fp32, with padded candidates set to ``-inf``."""
    grouped = torch.full(
        (batch.n_questions, batch.k_max), float("-inf"), dtype=torch.float32, device=marker_logits.device
    )
    grouped[batch.marker_question, batch.marker_candidate] = marker_logits.float()
    return grouped


def ungroup(grouped: torch.Tensor, batch: Batch) -> torch.Tensor:
    """Inverse of :func:`group_logits`: ``[Q, K_max]`` → ``[M]`` in marker order."""
    return grouped[batch.marker_question, batch.marker_candidate]
