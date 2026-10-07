"""Training data loading: unified records → (batch, targets) (TODO Phase 5).

    records = JsonlRecords("data/mix/stage_a/train.e0.jsonl")
    collate = TrainCollator(encoder, seed=cfg["seed"])
    loader = DataLoader(records, batch_size=8, shuffle=True, collate_fn=collate)
    for epoch in range(n):
        collate.set_epoch(epoch)
        for batch, target in loader:
            total, parts = jev_loss(model.grouped_logits(batch, apply_temperature=False), target.to(dev), batch.to(dev))

Choice options are shuffled in training (TODO 2.4): the order is a stable function of
``(seed, epoch, record id)``, so a fresh order every epoch, the same order on every rerun and every
DataLoader worker. Score levels keep their order. ``shuffle=False`` keeps the stored order (eval).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import torch

from ..collate import Batch, build_batch
from ..encoding import PackedEncoder, PairEncoder, shuffle_choice
from ..losses import loss_targets
from .public.base import stable_hash
from .unified import Record


class JsonlRecords(torch.utils.data.Dataset):
    """A map-style dataset over a JSONL file of records. Lines are kept as strings and parsed on access,
    so a 300k-record epoch costs its file size in memory, not pydantic objects."""

    def __init__(self, path: str | Path, splits: Sequence[str] | None = None):
        with Path(path).open() as f:
            lines = [line for line in f if line.strip()]
        if splits is not None:
            keep = set(splits)
            lines = [line for line in lines if Record.model_validate_json(line).split in keep]
        self.lines = lines

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, i: int) -> Record:
        return Record.model_validate_json(self.lines[i])


class TrainCollator:
    """``list[Record] → (Batch, target [Q, K_max])``, one question (one item) per record."""

    def __init__(self, encoder: PackedEncoder | PairEncoder, seed: int = 42, shuffle: bool = True):
        self.encoder = encoder
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __call__(self, records: Sequence[Record]) -> tuple[Batch, torch.Tensor]:
        items, targets = [], []
        for i, r in enumerate(records):
            # Question ids only need to be unique within the batch; record ids can repeat when a
            # small pool is upsampled.
            question, target = r.to_question(f"{i}:{r.id}"), r.target
            if self.shuffle and r.type == "choice":
                if isinstance(target, float):  # an index stored as 2.0
                    target = int(target)
                rng = random.Random(stable_hash(self.seed, "options", self.epoch, r.id))
                question, target, _ = shuffle_choice(question, target, rng)
            items.append((r.state, question))
            targets.append(target)
        batch = build_batch(items, self.encoder)
        return batch, loss_targets(targets, batch)
