"""Unified data format (TODO section 4) and the prediction format consumed by ``scripts/eval.py``.

Records: one JSONL line per question.

    {"id": "...", "source": "boolq", "domain": "wiki", "split": "test_in",
     "state": "...", "type": "bool", "prompt": "...", "candidates": null, "target": 1.0}

- Choice and Score: ``candidates`` is a list; ``target`` is a class index or a soft distribution.
- Bool: ``candidates`` is null; ``target`` is a float in [0, 1].
- Judge data adds ``criterion``, ``group`` and ``user_question``. Any other field is kept as is
  (``family``, ``difficulty``, ``perturbation`` …) and can be used to slice the report.

Predictions: one JSONL line per record id.

    {"id": "...", "probs": [0.1, 0.9]}          # Choice / Score, in candidate order
    {"id": "...", "prob": 0.93}                  # Bool, P(yes)

Optional: ``confidence`` (a separate confidence signal), ``latency_ms`` and ``cost_usd`` per record.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..schema import BoolQuestion, ChoiceQuestion, NonEmptyStr, PrimType, Question, ScoreQuestion, State, _check_candidates

# Evaluation splits (TODO Phase 2):
#   train     — training data;
#   calib     — temperatures, the confidence head and the answer-level calibrator only;
#   test_in   — the same domains and templates as train;
#   test_ood  — held-out domains and question templates (and, for synthetic judge data, held-out
#               perturbations and criteria, see dataset_dicr.md section 12);
#   judge_test — the human-labelled judge test set from our system. Frozen, never trained on.
SPLITS = ("train", "calib", "test_in", "test_ood", "judge_test")


class Record(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: NonEmptyStr
    source: NonEmptyStr
    domain: str = "unknown"
    split: str
    state: State
    type: PrimType
    prompt: NonEmptyStr
    candidates: list[str] | None = None
    target: int | float | list[float]
    criterion: str | None = None
    group: str | None = None
    user_question: str | None = None

    @model_validator(mode="after")
    def _check(self) -> Record:
        if self.split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {self.split!r}")
        if self.type == "bool":
            if self.candidates is not None:
                raise ValueError("bool records have no candidates")
            if isinstance(self.target, list) or not 0.0 <= float(self.target) <= 1.0:
                raise ValueError("bool target must be a number in [0, 1]")
            return self
        if self.candidates is None:
            raise ValueError(f"{self.type} records need candidates")
        _check_candidates(self.candidates, "candidates")
        k = len(self.candidates)
        if isinstance(self.target, list):
            if len(self.target) != k or any(t < 0 for t in self.target) or not math.isclose(sum(self.target), 1.0, abs_tol=1e-4):
                raise ValueError(f"soft target must be a distribution over the {k} candidates")
        elif isinstance(self.target, float) and not self.target.is_integer():
            raise ValueError(f"{self.type} target must be a class index or a distribution")
        elif not 0 <= int(self.target) < k:
            raise ValueError(f"target index {self.target} is out of range for {k} candidates")
        return self

    @property
    def n_candidates(self) -> int:
        return 1 if self.type == "bool" else len(self.candidates)

    def to_question(self, qid: str | None = None) -> Question:
        qid = qid or self.id
        if self.type == "choice":
            return ChoiceQuestion(id=qid, prompt=self.prompt, options=self.candidates)
        if self.type == "score":
            return ScoreQuestion(id=qid, prompt=self.prompt, levels=self.candidates)
        return BoolQuestion(id=qid, prompt=self.prompt)

    def field(self, name: str) -> Any:
        """A declared or extra field, ``None`` if absent (used for slicing)."""
        if name in type(self).model_fields:
            return getattr(self, name)
        return (self.model_extra or {}).get(name)


class Prediction(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: NonEmptyStr
    probs: list[float] | None = None
    prob: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    latency_ms: float | None = None
    cost_usd: float | None = None

    @model_validator(mode="after")
    def _one_of(self) -> Prediction:
        if (self.probs is None) == (self.prob is None):
            raise ValueError("a prediction has exactly one of 'probs' (choice/score) or 'prob' (bool)")
        if self.probs is not None and (
            any(p < -1e-6 for p in self.probs) or not math.isclose(sum(self.probs), 1.0, abs_tol=1e-3)
        ):
            raise ValueError("probs must be a distribution")
        return self


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e


def write_jsonl(path: str | Path, rows: Iterable[BaseModel | dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for row in rows:
            data = row.model_dump(exclude_none=True) if isinstance(row, BaseModel) else row
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            n += 1
    return n


def load_records(path: str | Path, splits: Iterable[str] | None = None) -> list[Record]:
    keep = set(splits) if splits else None
    records = [Record.model_validate(row) for row in read_jsonl(path)]
    if keep is not None:
        records = [r for r in records if r.split in keep]
    ids = [r.id for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: record ids must be unique")
    return records


def load_predictions(path: str | Path) -> dict[str, Prediction]:
    preds: dict[str, Prediction] = {}
    for row in read_jsonl(path):
        p = Prediction.model_validate(row)
        if p.id in preds:
            raise ValueError(f"{path}: duplicate prediction for {p.id!r}")
        preds[p.id] = p
    return preds
