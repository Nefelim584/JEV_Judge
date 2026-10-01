"""Request / response models (TODO section 2.2)."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_CANDIDATES = 32

PrimType = Literal["choice", "score", "bool"]
PRIM_TYPES: tuple[PrimType, ...] = ("choice", "score", "bool")

NonEmptyStr = Annotated[str, Field(min_length=1)]
Probability = Annotated[float, Field(ge=0.0, le=1.0)]

# A state is free text, a JSON object, or an array of texts.
State = str | dict[str, Any] | list[str]


def _check_candidates(values: list[str], what: str) -> list[str]:
    if not values:
        raise ValueError(f"{what} must be non-empty")
    if len(values) > MAX_CANDIDATES:
        raise ValueError(f"at most {MAX_CANDIDATES} {what} are allowed, got {len(values)}")
    if any(not v.strip() for v in values):
        raise ValueError(f"{what} must not be blank")
    if len(set(values)) != len(values):
        raise ValueError(f"{what} must be unique")
    return values


class _Question(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: NonEmptyStr
    prompt: NonEmptyStr


class ChoiceQuestion(_Question):
    type: Literal["choice"] = "choice"
    options: list[str]

    @field_validator("options")
    @classmethod
    def _options(cls, v: list[str]) -> list[str]:
        return _check_candidates(v, "options")

    @property
    def candidates(self) -> list[str]:
        return self.options


class ScoreQuestion(_Question):
    type: Literal["score"] = "score"
    levels: list[str] = Field(description="Ordered from low to high.")

    @field_validator("levels")
    @classmethod
    def _levels(cls, v: list[str]) -> list[str]:
        return _check_candidates(v, "levels")

    @property
    def candidates(self) -> list[str]:
        return self.levels


class BoolQuestion(_Question):
    type: Literal["bool"] = "bool"

    @property
    def candidates(self) -> list[str]:
        # Bool scores the statement itself: one candidate.
        return [self.prompt]


Question = Annotated[ChoiceQuestion | ScoreQuestion | BoolQuestion, Field(discriminator="type")]


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: State
    questions: list[Question] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> Request:
        ids = [q.id for q in self.questions]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"question ids must be unique, duplicated: {dupes}")
        return self


# ---------------------------------------------------------------------------- response


def _check_distribution(probs: dict[str, float]) -> dict[str, float]:
    if not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-3):
        raise ValueError(f"probabilities must sum to 1, got {sum(probs.values()):.6f}")
    return probs


class ChoiceAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str
    probabilities: dict[str, Probability]
    confidence: Probability

    @field_validator("probabilities")
    @classmethod
    def _sum(cls, v: dict[str, float]) -> dict[str, float]:
        return _check_distribution(v)

    @model_validator(mode="after")
    def _value_in_schema(self) -> ChoiceAnswer:
        if self.value not in self.probabilities:
            raise ValueError(f"value {self.value!r} is not one of the candidates")
        return self


class ScoreAnswer(ChoiceAnswer):
    expected_level: float = Field(description="Expected 0-based level index under the distribution.")

    @model_validator(mode="after")
    def _expected_in_range(self) -> ScoreAnswer:
        if not 0.0 <= self.expected_level <= len(self.probabilities) - 1 + 1e-6:
            raise ValueError("expected_level is outside the level range")
        return self


class BoolAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    probability: Probability
    confidence: Probability


# extra="forbid" on every answer keeps this union unambiguous when validating from JSON.
Answer = ScoreAnswer | ChoiceAnswer | BoolAnswer


class Meta(BaseModel):
    model: str
    latency_ms: float
    truncated: bool = False


class Response(BaseModel):
    answers: dict[str, Answer]
    meta: Meta
