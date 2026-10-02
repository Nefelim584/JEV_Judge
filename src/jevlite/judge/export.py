"""Format of (question, chunks, answer) triples exported from our system (TODO Phase 11).

One JSONL line per judged answer:

    {"id": "...", "question": "...", "chunks": ["...", "..."], "answer": "...",
     "llm_judge": {...}, "meta": {...}}

- ``chunks`` are the retrieved chunks in retrieval order: strings, or objects with ``text`` (and any
  other fields, e.g. ``id``, ``score``).
- ``llm_judge`` holds the production judge's verdicts as they are (needed for H0), including
  ``cost_usd`` and ``latency_ms`` when available. Its exact fields are fixed once we see the real export.
- ``meta`` is anything else (timestamps, product area); it is never shown to a model.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..schema import NonEmptyStr


class JudgeSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: NonEmptyStr
    question: str
    chunks: list[str]
    answer: str
    llm_judge: dict[str, Any] | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("chunks", mode="before")
    @classmethod
    def _chunk_texts(cls, v):
        return [c["text"] if isinstance(c, dict) else c for c in v]
