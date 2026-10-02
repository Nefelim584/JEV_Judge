"""Templates, state serialisation and the two encoders (TODO section 2.4).

Packed mode (v1): one sequence per question, ``[CLS] header [SEP] state [SEP]``. The header holds the
question and all its candidates; every candidate has a ``[MASK]`` marker whose output is its logit.
Several Bool claims about one state can share a sequence as a :class:`ClaimGroup`.

Pair mode (ablation): one sequence per candidate, ``[CLS] question + candidate [SEP] state [SEP]``,
read out at ``[CLS]``.

Both encoders return :class:`EncodedSequence` objects with explicit marker positions, so batching and
read-out are the same for both modes. Only the state is ever truncated.
"""

from __future__ import annotations

import json
import numbers
import random
from dataclasses import dataclass
from typing import Sequence

from .schema import BoolQuestion, ChoiceQuestion, Question, ScoreQuestion, State

ENCODING_MODES = ("packed", "pair")


def serialize_state(state: State) -> str:
    """Strings as is; JSON objects as compact JSON in original key order; arrays as ``[i] text`` lines."""
    if isinstance(state, str):
        return state
    if isinstance(state, dict):
        return json.dumps(state, ensure_ascii=False, separators=(",", ":"), sort_keys=False)
    if isinstance(state, list):
        return "\n".join(f"[{i}] {text}" for i, text in enumerate(state, start=1))
    raise TypeError(f"unsupported state type {type(state).__name__}")


@dataclass(frozen=True)
class ClaimGroup:
    """Bool claims about one state, encoded in one packed sequence with one marker per claim.

    Each claim keeps an independent sigmoid. ``context`` goes into the header, e.g. the user's
    question in judge mode, so that claims with pronouns can be resolved.
    """

    claims: tuple[BoolQuestion, ...]
    context: str | None = None

    def __post_init__(self):
        if not self.claims:
            raise ValueError("a claim group needs at least one claim")
        if not all(isinstance(c, BoolQuestion) for c in self.claims):
            raise TypeError("a claim group holds Bool questions only")
        ids = [c.id for c in self.claims]
        if len(set(ids)) != len(ids):
            raise ValueError(f"claim ids must be unique within a group, got {ids}")


# A unit of encoding: one question, or a group of Bool claims sharing one state.
Item = Question | ClaimGroup


def item_questions(item: Item) -> list[Question]:
    return list(item.claims) if isinstance(item, ClaimGroup) else [item]


def item_context(item: Item) -> str | None:
    return item.context if isinstance(item, ClaimGroup) else None


def _context_part(context: str | None) -> str:
    return f" | context: {context}" if context else ""


def segment_a_texts(question: Question, context: str | None = None) -> list[str]:
    """Pair mode: one segment-A string per candidate, in candidate order."""
    ctx = _context_part(context)
    if isinstance(question, ChoiceQuestion):
        return [f"type: choice{ctx} | question: {question.prompt} | answer: {option}" for option in question.options]
    if isinstance(question, ScoreQuestion):
        n = len(question.levels)
        return [
            f"type: score{ctx} | question: {question.prompt} | level {i} of {n}: {level}"
            for i, level in enumerate(question.levels, start=1)
        ]
    if isinstance(question, BoolQuestion):
        return [f"type: bool{ctx} | claim: {question.prompt}"]
    raise TypeError(f"unsupported question type {type(question).__name__}")


MARKER = None  # placeholder for a marker in a header template


def packed_header(item: Item) -> tuple[list[str | None], list[tuple[int, int]]]:
    """Packed mode: header text pieces with ``MARKER`` placeholders, and ``(question, candidate)``
    indices within the item for each marker, in marker order.

    Choice and Score put the marker before each candidate; Bool puts it after each claim.
    """
    if isinstance(item, BoolQuestion):
        item = ClaimGroup((item,))
    if isinstance(item, ClaimGroup):
        pieces: list[str | None] = [f"type: bool{_context_part(item.context)}"]
        for claim in item.claims:
            pieces += [f" | claim: {claim.prompt}", MARKER]
        return pieces, [(j, 0) for j in range(len(item.claims))]
    if isinstance(item, ChoiceQuestion):
        prefix = f"type: choice | question: {item.prompt} | options:"
    elif isinstance(item, ScoreQuestion):
        prefix = f"type: score | question: {item.prompt} | levels from low to high:"
    else:
        raise TypeError(f"unsupported item type {type(item).__name__}")
    pieces = [prefix]
    for candidate in item.candidates:
        pieces += [MARKER, f" {candidate}"]
    return pieces, [(0, k) for k in range(len(item.candidates))]


def render_header(item: Item, marker_text: str = "[MASK]") -> str:
    """Packed header as a readable string (for logs and tests; encoding never re-parses it)."""
    pieces, _ = packed_header(item)
    return "".join(f" {marker_text}" if p is MARKER else p for p in pieces)


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
    """Pair mode: token ids for every candidate of one question against one state."""

    input_ids: list[list[int]]  # one row per candidate
    truncated: bool


@dataclass
class EncodedSequence:
    """One encoder input and the read-out positions it carries."""

    input_ids: list[int]
    markers: list[tuple[int, int, int]]  # (position, question index within the item, candidate index)
    truncated: bool


class _TextEncoder:
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
        # split_special_tokens: a literal "[MASK]" or "[SEP]" in user text is tokenised as plain text,
        # so it can never become a fake marker or separator.
        return self.tokenizer(texts, add_special_tokens=False, split_special_tokens=True)["input_ids"]

    def tokenize_state(self, state: State) -> list[int]:
        return self._tokenize([serialize_state(state)])[0]

    def _state_budget(self, header_len: int, what: str) -> int:
        budget = self.max_len - self.NUM_SPECIAL - header_len
        if budget < 1:
            raise ValueError(
                f"{what}: question/candidate text needs {header_len + self.NUM_SPECIAL} tokens, "
                f"which leaves no room for the state within max_len={self.max_len}"
            )
        return budget

    def encode_item(self, item: Item, state_ids: list[int]) -> list[EncodedSequence]:
        raise NotImplementedError


class PairEncoder(_TextEncoder):
    """Builds token-id pairs with a hard guarantee that segment A is never truncated.

    The state is truncated once per question with a budget set by the longest segment A of that
    question, so all candidates of a question see exactly the same state tokens.
    """

    def encode_question(self, question: Question, state_ids: list[int], context: str | None = None) -> EncodedQuestion:
        a_ids = self._tokenize(segment_a_texts(question, context))
        budget = self._state_budget(max(len(a) for a in a_ids), f"question {question.id!r}")
        b_ids, truncated = truncate_head_tail(state_ids, budget, self.head_ratio)
        rows = [[self.cls_id, *a, self.sep_id, *b_ids, self.sep_id] for a in a_ids]
        return EncodedQuestion(input_ids=rows, truncated=truncated)

    def encode_item(self, item: Item, state_ids: list[int]) -> list[EncodedSequence]:
        """One sequence per candidate, read out at ``[CLS]`` (position 0). A claim group is split into
        one sequence per claim."""
        out = []
        for j, question in enumerate(item_questions(item)):
            encoded = self.encode_question(question, state_ids, item_context(item))
            out += [EncodedSequence(ids, [(0, j, k)], encoded.truncated) for k, ids in enumerate(encoded.input_ids)]
        return out


class PackedEncoder(_TextEncoder):
    """One sequence per item, ``[CLS] header [SEP] state [SEP]``, with a ``[MASK]`` marker per candidate."""

    def __init__(self, tokenizer, max_len: int, head_ratio: float = 0.5):
        super().__init__(tokenizer, max_len, head_ratio)
        self.marker_id = tokenizer.mask_token_id
        if self.marker_id is None:
            raise ValueError("packed encoding needs a tokenizer with a mask token")

    def encode_header(self, item: Item) -> tuple[list[int], list[tuple[int, int, int]]]:
        """Header token ids and markers, with positions relative to the start of the header."""
        pieces, slots = packed_header(item)
        texts = iter(self._tokenize([p for p in pieces if p is not MARKER]))
        ids: list[int] = []
        positions: list[int] = []
        for piece in pieces:
            if piece is MARKER:
                positions.append(len(ids))
                ids.append(self.marker_id)
            else:
                ids.extend(next(texts))
        return ids, [(pos, q, k) for pos, (q, k) in zip(positions, slots, strict=True)]

    def encode_item(self, item: Item, state_ids: list[int]) -> list[EncodedSequence]:
        header, markers = self.encode_header(item)
        budget = self._state_budget(len(header), f"item {[q.id for q in item_questions(item)]}")
        b_ids, truncated = truncate_head_tail(state_ids, budget, self.head_ratio)
        row = [self.cls_id, *header, self.sep_id, *b_ids, self.sep_id]
        markers = [(pos + 1, q, k) for pos, q, k in markers]  # shift past [CLS]

        n_markers = sum(len(q.candidates) for q in item_questions(item))
        if row.count(self.marker_id) != n_markers or len(markers) != n_markers:
            raise AssertionError(f"expected {n_markers} markers, found {row.count(self.marker_id)} in the sequence")
        return [EncodedSequence(row, markers, truncated)]


def text_encoder_from_config(cfg: dict, tokenizer, max_len: int | None = None) -> PairEncoder | PackedEncoder:
    """The encoder selected by ``encoding.mode``; ``max_len`` overrides ``encoding.max_len`` (e.g. per stage)."""
    enc = cfg["encoding"]
    mode = enc.get("mode", "packed")
    cls = {"packed": PackedEncoder, "pair": PairEncoder}.get(mode)
    if cls is None:
        raise ValueError(f"unknown encoding.mode {mode!r}, expected one of {ENCODING_MODES}")
    return cls(tokenizer, max_len or enc["max_len"], enc["head_ratio"])


def shuffle_choice(
    question: ChoiceQuestion, target: int | Sequence[float], rng: random.Random
) -> tuple[ChoiceQuestion, int | list[float], list[int]]:
    """Training-time option shuffle. Returns the shuffled question, the target in the new order and
    ``perm``, where new position ``i`` holds original option ``perm[i]``.

    Only Choice is shuffled: Score levels keep their order, which carries their meaning.
    """
    if not isinstance(question, ChoiceQuestion):
        raise TypeError(f"only choice questions are shuffled, got {question.type!r}")
    n = len(question.options)
    perm = list(range(n))
    rng.shuffle(perm)
    shuffled = question.model_copy(update={"options": [question.options[j] for j in perm]})
    if isinstance(target, numbers.Integral):
        target = int(target)
        if not 0 <= target < n:
            raise ValueError(f"target index {target} is out of range for {n} options")
        new_target: int | list[float] = perm.index(target)
    else:
        if len(target) != n:
            raise ValueError(f"soft target has {len(target)} entries for {n} options")
        new_target = [float(target[j]) for j in perm]
    return shuffled, new_target, perm
