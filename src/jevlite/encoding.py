"""Templates, state serialisation and pair building (TODO section 2.4).

Each candidate becomes one sequence pair ``[CLS] segment_a [SEP] state [SEP]``:
segment A is the question + candidate, segment B is the serialised state.
Only the state is ever truncated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from .schema import BoolQuestion, ChoiceQuestion, Question, ScoreQuestion, State


def serialize_state(state: State) -> str:
    """Strings as is; JSON objects as compact JSON in original key order; arrays as ``[i] text`` lines."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    if isinstance(state, list):
        return "\n".join(f"[{i}] {text}" for i, text in enumerate(state, start=1))
    raise TypeError(f"unsupported state type {type(state).__name__}")


def segment_a_texts(question: Question) -> list[str]:
    """One segment-A string per candidate, in candidate order."""
    if isinstance(question, ChoiceQuestion):
        return [f"type: choice | question: {question.prompt} | answer: {option}" for option in question.options]
    if isinstance(question, ScoreQuestion):
        n = len(question.levels)
        return [
            f"type: score | question: {question.prompt} | level {i} of {n}: {level}"
            for i, level in enumerate(question.levels, start=1)
        ]
    if isinstance(question, BoolQuestion):
        return [f"type: bool | claim: {question.prompt}"]
    raise TypeError(f"unsupported question type {type(question).__name__}")


def truncate_head_tail(ids: list[int], budget: int, head_ratio: float) -> tuple[list[int], bool]:
    """Keep the first ``head_ratio * budget`` and the last remaining tokens. Returns (ids, truncated)."""
    if budget < 0:
        raise ValueError("budget must be >= 0")
    if len(ids) <= budget:
        return ids, False
    head = int(budget * head_ratio)
    tail = budget - head
    return ids[:head] + (ids[len(ids) - tail :] if tail else []), True


@dataclass
class EncodedQuestion:
    """Token ids for every candidate of one question against one state."""

    input_ids: list[list[int]]  # one row per candidate
    truncated: bool


class PairEncoder:
    """Builds token-id pairs with a hard guarantee that segment A is never truncated.

    The state is truncated once per question with a budget set by the longest segment A of that
    question, so all candidates of a question see exactly the same state tokens.
    """

    # [CLS] a [SEP] b [SEP]
    NUM_SPECIAL = 3

    def __init__(self, tokenizer, max_len: int, head_ratio: float = 0.5):
        if not 0.0 <= head_ratio <= 1.0:
            raise ValueError("head_ratio must be in [0, 1]")
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.head_ratio = head_ratio
        self.cls_id = tokenizer.cls_token_id
        self.sep_id = tokenizer.sep_token_id

    def _tokenize(self, texts: list[str]) -> list[list[int]]:
        return self.tokenizer(texts, add_special_tokens=False)["input_ids"]

    def tokenize_state(self, state: State) -> list[int]:
        return self._tokenize([serialize_state(state)])[0]

    def encode_question(self, question: Question, state_ids: list[int]) -> EncodedQuestion:
        a_ids = self._tokenize(segment_a_texts(question))
        budget = self.max_len - self.NUM_SPECIAL - max(len(a) for a in a_ids)
        if budget < 1:
            raise ValueError(
                f"question {question.id!r}: question/candidate text needs {self.max_len - budget} tokens, "
                f"which leaves no room for the state within max_len={self.max_len}"
            )
        b_ids, truncated = truncate_head_tail(state_ids, budget, self.head_ratio)
        rows = [[self.cls_id, *a, self.sep_id, *b_ids, self.sep_id] for a in a_ids]
        return EncodedQuestion(input_ids=rows, truncated=truncated)
