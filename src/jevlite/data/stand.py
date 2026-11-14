"""Rows of the company's eval stand → jev-lite records (Stage C, ``dataset_gen.md`` section 7).

A stand row is one judged answer: the user's question (and, in a dialogue, the earlier turns), the
guard's verdict, the retrieved chunks, the assistant's answer and the production judge's 1–5 score
per criterion. The rows themselves are company data: they are read from ``data/`` by path and never
live in the code, and neither do the judge's criterion names and wordings. Those come from a
**criteria spec** (JSON, git-ignored, next to the rows)::

    {
      "criteria": {
        "<column name of the score>": {
          "prompts": ["short question 1", "short question 2"],   # one is drawn per row (stable)
          "context": true                                       # the chunks go into the state
        }
      },
      "level_schemes": {"digits": ["1", "2", "3", "4", "5"], "words": ["very low", "low", "medium", "high", "very high"]},
      "train_schemes": {"digits": 0.5, "words": 0.5},           # drawn per record in train
      "eval_scheme": "digits",                                  # fixed in dev / calib / test_in
      "faithfulness": {"from": "<criterion>", "pos_min": 4, "neg_max": 2}   # optional, weak Bool labels
    }

Per row:

- **one Score record per criterion**: none for a criterion that reads the chunks when an allowed question
  has none (old rows whose chunks were not logged: their scores cannot be checked). The prompt is a *short* wording of the criterion, not the rubric
  (zero-shot, the rubric text did not help: ``data/judge_mapped_rubric/summary.md``). The state is the
  guard line + the dialogue, with the chunks in front for criteria that need them.
- **optionally one Bool faithfulness record** (the answer against the chunks, Stage B wording), labelled
  from a criterion's score: ≥ ``pos_min`` supported, ≤ ``neg_max`` not, in between no record. A weak
  label (``label_source: judge_weak``); the main faithfulness data stays Stage B's.

Repeats of the judge (S4: the same row judged several times, rows sharing an ``id``) are merged: in
``train`` the target is the distribution of the repeats (a soft target), in the eval splits the mode
(a tie → the median).

Long dialogues: with a tokenizer, the oldest turns are dropped until the state fits ``max_len``
minus a reserve for the question and its levels; the number dropped is kept in ``history_trimmed``.
What still does not fit is dropped later by the mix (``fit_only``).

Splits are drawn per **group**: the stand's ``question_id`` if present, else the first user message
(old rows have no ``question_id``; the first message also merges repeated questions and all turns of a
dialogue). An explicit ``{id: split}`` map wins; otherwise groups are hashed into ``test_in`` /
``calib`` / ``dev`` by share, and only groups whose rows are all ``eval_eligible`` (fresh runs of the
current assistant, ``dataset_gen.md`` section 8) can leave ``train``.

Not yet: Bool records on single rubric rules. Their
labels come from the S2 perturbations, which do not exist yet (``dataset_gen.md`` sections 4, 7).
"""

from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from .public.base import stable_hash, unit
from .public.judge_train import ANSWER_CLAIM
from .unified import Record

SOURCE = "stand"
CHUNK_SEP = "\n---\n"
LEVELS = 5  # the judge scores 1..5
DEFAULT_SCHEMES = {
    "digits": ["1", "2", "3", "4", "5"],
    "words": ["very low", "low", "medium", "high", "very high"],
}
EVAL_SPLITS = ("dev", "calib", "test_in")
# The Stage B wording of a whole-answer claim; the held-out template of Stage B is not used.
FAITHFULNESS_TEMPLATE = next(t for t in ANSWER_CLAIM if t.id == "per_documents").prompt
# Room for the question, its levels and the special tokens when the history is trimmed to max_len.
HEADER_RESERVE = 256


# ---------------------------------------------------------------------------- spec


@dataclass(frozen=True)
class Criterion:
    name: str
    prompts: tuple[str, ...]
    context: bool


@dataclass(frozen=True)
class CriteriaSpec:
    criteria: tuple[Criterion, ...]
    level_schemes: dict[str, list[str]] = field(default_factory=lambda: dict(DEFAULT_SCHEMES))
    train_schemes: dict[str, float] = field(default_factory=lambda: {"digits": 0.5, "words": 0.5})
    eval_scheme: str = "digits"
    faithfulness: dict | None = None

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CriteriaSpec:
        crit = tuple(
            Criterion(name, tuple(c["prompts"]), bool(c.get("context", False))) for name, c in d["criteria"].items()
        )
        if not crit or any(not c.prompts for c in crit):
            raise ValueError("the spec needs criteria, each with at least one prompt")
        schemes = {k: list(v) for k, v in d.get("level_schemes", DEFAULT_SCHEMES).items()}
        if any(len(v) != LEVELS for v in schemes.values()):
            raise ValueError(f"every level scheme needs {LEVELS} levels (the judge scores 1..{LEVELS})")
        train = dict(d.get("train_schemes", {k: 1.0 for k in schemes}))
        eval_scheme = d.get("eval_scheme", next(iter(schemes)))
        unknown = (set(train) | {eval_scheme}) - set(schemes)
        if unknown:
            raise ValueError(f"unknown level schemes {sorted(unknown)}")
        faith = d.get("faithfulness")
        if faith and faith["from"] not in {c.name for c in crit}:
            raise ValueError(f"faithfulness.from {faith['from']!r} is not a criterion of the spec")
        return cls(crit, schemes, train, eval_scheme, faith)

    @classmethod
    def load(cls, path: str | Path) -> CriteriaSpec:
        return cls.from_dict(json.loads(Path(path).read_text()))


# ---------------------------------------------------------------------------- one stand row


def _blank(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def _text(v: Any) -> str:
    return "" if _blank(v) else str(v).strip()


def parse_bool(v: Any) -> bool | None:
    if _blank(v):
        return None
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "1", "1.0", "yes", "allowed"):
        return True
    if s in ("false", "0", "0.0", "no", "blocked"):
        return False
    return None


def parse_score(v: Any) -> int | None:
    """A 1..5 judge score, None when missing or out of range."""
    if _blank(v):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return int(x) if x.is_integer() and 1 <= x <= LEVELS else None


_TURN = re.compile(r"^\[turn \d+( \(judged\))?\]\s*$", re.M)


def history_turns(transcript: Any) -> list[str]:
    """The turns before the judged one, each ``"User: …\\n\\nAssistant: …"``. The stand's transcript marks
    turns ``[turn N]`` and the judged one ``[turn N (judged)]``; the judged turn itself comes from the
    row's question and answer."""
    if not isinstance(transcript, str) or "(judged)]" not in transcript:
        return []
    before = transcript.split("(judged)]")[0].rsplit("[turn", 1)[0]
    text = _TURN.sub("", before).strip()
    text = re.sub(r"^USER:\s*", "User: ", text, flags=re.M)
    text = re.sub(r"^ASSISTANT:\s*", "Assistant: ", text, flags=re.M)
    return [t.strip() for t in re.split(r"\n(?=User: )", text) if t.strip()]


def first_user_message(row: Mapping[str, Any]) -> str:
    turns = history_turns(row.get("transcript"))
    if turns:
        return turns[0].split("\nAssistant:", 1)[0].removeprefix("User:").strip()
    return _text(row.get("question"))


def chunks_of(row: Mapping[str, Any]) -> list[str]:
    return [c.strip() for c in _text(row.get("retrieved_context")).split(CHUNK_SEP) if c.strip()]


def judge_model(row: Mapping[str, Any]) -> str | None:
    m = re.search(r"JUDGE_MODEL=([^;]+)", _text(row.get("models")))
    return m.group(1).strip() if m else None


@dataclass
class StandRow:
    """A stand row with its judge scores; ``scores[criterion]`` holds one score per judge repeat."""

    id: str
    group: str
    question: str
    answer: str
    guard_allowed: bool | None
    history: list[str]
    chunks: list[str]
    scores: dict[str, list[int]]
    fields: dict[str, Any]

    def guard_line(self) -> str:
        verdict = {True: "ALLOWED", False: "BLOCKED", None: "UNKNOWN"}[self.guard_allowed]
        return f"Guard verdict on the question: {verdict}"

    def dialogue(self, n_history: int | None = None) -> str:
        hist = self.history if n_history is None else self.history[len(self.history) - n_history:]
        return "\n\n".join([self.guard_line(), *hist, f"User: {self.question}", f"Assistant: {self.answer}"])

    def with_documents(self, n_history: int | None = None) -> str:
        docs = "\n\n".join(f"Document [{i}]: {c}" for i, c in enumerate(self.chunks, 1)) or "(none)"
        return f"RETRIEVED CONTEXT:\n{docs}\n\n{self.dialogue(n_history)}"

    def lacks_context(self) -> bool:
        """An allowed question without chunks: the logger did not keep them (old rows), so a criterion that
        reads the chunks cannot be judged from this row. A blocked question needs no chunks."""
        return self.guard_allowed is not False and not self.chunks


def row_id(row: Mapping[str, Any]) -> str:
    """The stand's ``id``, else ``question_id`` (+ ``_t<turn>`` in a dialogue).

    The stand's judge-only and re-run outputs drop ``id`` but carry it as ``question_id``
    (``scripts/freeze_stand_splits.py`` sets ``question_id`` = ``id``), so a judge repeat (S4) of a row
    gets the row's own id and merges with it."""
    if not _blank(row.get("id")):
        return str(row["id"]).strip()
    parts = [_text(row.get(k)).removesuffix(".0") for k in ("question_id", "turn")]
    if not parts[0]:
        raise ValueError("a stand row needs an 'id' or a 'question_id'")
    return "_t".join(p for p in parts if p)


def group_of(row: Mapping[str, Any]) -> str:
    qid = _text(row.get("question_id")).removesuffix(".0")
    if qid:
        return f"{SOURCE}:q{qid}"
    first = re.sub(r"\s+", " ", first_user_message(row).lower())
    return f"{SOURCE}:m{stable_hash('stand-group', first):016x}"


# Carried into every record when present: slicing in the report (categories, S2 perturbations).
CARRY = ("category", "perturbation", "base_row_id", "stand_part", "commit_sha")


def merge_rows(rows: Iterable[Mapping[str, Any]], spec: CriteriaSpec) -> tuple[list[StandRow], Counter]:
    """Stand rows → ``StandRow``s, judge repeats (same ``id``) merged. Returns the rows and a count of the
    rows skipped by reason."""
    names = [c.name for c in spec.criteria]
    merged: dict[str, StandRow] = {}
    skipped: Counter = Counter()
    for row in rows:
        if not _blank(row.get("error")):
            skipped["error"] += 1
            continue
        answer, question = _text(row.get("answer")), _text(row.get("question"))
        if not answer or not question:
            skipped["no question or answer"] += 1
            continue
        scores = {n: parse_score(row.get(n)) for n in names}
        if all(s is None for s in scores.values()):
            skipped["no judge scores"] += 1
            continue
        rid = row_id(row)
        if rid in merged:
            for n, s in scores.items():
                if s is not None:
                    merged[rid].scores[n].append(s)
            continue
        fields = {k: _text(row.get(k)) for k in CARRY if _text(row.get(k))}
        fields["judge_model"] = judge_model(row)
        merged[rid] = StandRow(
            id=rid, group=group_of(row), question=question, answer=answer, guard_allowed=parse_bool(row.get("guard_is_allowed")),
            history=history_turns(row.get("transcript")), chunks=chunks_of(row),
            scores={n: [s] if s is not None else [] for n, s in scores.items()}, fields=fields,
        )
    return list(merged.values()), skipped


# ---------------------------------------------------------------------------- targets and splits


def soft_target(scores: Sequence[int]) -> list[float]:
    counts = Counter(scores)
    return [counts.get(level, 0) / len(scores) for level in range(1, LEVELS + 1)]


def mode_score(scores: Sequence[int]) -> int:
    """The most frequent score; a tie → the median of the tied scores, rounded down."""
    counts = Counter(scores)
    top = max(counts.values())
    tied = sorted(s for s, n in counts.items() if n == top)
    return tied[0] if len(tied) == 1 else int(statistics.median_low(tied))


def assign_splits(
    rows: Sequence[StandRow], explicit: Mapping[str, str] | None = None, shares: Mapping[str, float] | None = None,
    eval_eligible: Callable[[StandRow], bool] | None = None, seed: int = 42,
) -> dict[str, str]:
    """``{row id: split}``. ``explicit`` (row id → split) wins and must keep groups whole. Otherwise a group
    goes to ``test_in`` / ``calib`` / ``dev`` by a stable hash and ``shares`` (fractions of the eligible
    groups), else ``train``. A group with any row that is not ``eval_eligible`` stays in ``train``."""
    by_group: dict[str, list[StandRow]] = defaultdict(list)
    for r in rows:
        by_group[r.group].append(r)
    out: dict[str, str] = {}
    if explicit:
        for g, members in by_group.items():
            splits = {explicit.get(r.id, "train") for r in members}
            if len(splits) > 1:
                raise ValueError(f"group {g} is split across {sorted(splits)}: a question (dialogue) must stay in one split")
            out |= {r.id: splits.pop() for r in members}
        return out
    shares = {s: float(v) for s, v in (shares or {}).items() if v}
    bad = set(shares) - set(EVAL_SPLITS)
    if bad or sum(shares.values()) >= 1:
        raise ValueError(f"shares must be fractions of {EVAL_SPLITS} summing below 1, got {shares}")
    for g, members in by_group.items():
        split = "train"
        if eval_eligible is None or all(eval_eligible(r) for r in members):
            x = unit(seed, "stand-split", g)
            for s in EVAL_SPLITS:
                if x < shares.get(s, 0.0):
                    split = s
                    break
                x -= shares.get(s, 0.0)
        out |= {r.id: split for r in members}
    return out


# ---------------------------------------------------------------------------- records


def _pick_weighted(weights: Mapping[str, float], *key: Any) -> str:
    total = sum(weights.values())
    x = unit(*key) * total
    for name, w in weights.items():
        x -= w
        if x < 0:
            return name
    return next(reversed(weights))


def fit_history(row: StandRow, state_of: Callable[[int], str], count: Callable[[str], int] | None, budget: int) -> int:
    """How many earlier turns to keep (newest first) so that the state fits ``budget`` tokens."""
    n = len(row.history)
    if count is None:
        return n
    while n > 0 and count(state_of(n)) > budget:
        n -= 1
    return n


def row_records(
    row: StandRow, split: str, spec: CriteriaSpec, count_tokens: Callable[[str], int] | None = None, max_len: int = 8192,
) -> Iterator[Record]:
    budget = max_len - HEADER_RESERVE
    base = {"source": SOURCE, "domain": "stand", "split": split, "group": row.group, "user_question": row.question,
            "family": "judge", "label_source": "judge", "stand_row": row.id, "multi_turn": bool(row.history),
            "judge_repeats": max((len(v) for v in row.scores.values()), default=0), **row.fields}
    kept: dict[bool, int] = {}
    for crit in spec.criteria:
        scores = row.scores.get(crit.name) or []
        if not scores or (crit.context and row.lacks_context()):
            continue
        if crit.context not in kept:
            state_of = row.with_documents if crit.context else row.dialogue
            kept[crit.context] = fit_history(row, state_of, count_tokens, budget)
        n_hist = kept[crit.context]
        state = row.with_documents(n_hist) if crit.context else row.dialogue(n_hist)
        prompt = crit.prompts[stable_hash("stand-prompt", row.id, crit.name) % len(crit.prompts)]
        if split == "train":
            scheme = _pick_weighted(spec.train_schemes, "stand-scheme", row.id, crit.name)
            target: int | list[float] = soft_target(scores) if len(set(scores)) > 1 else scores[0] - 1
        else:
            scheme = spec.eval_scheme
            target = mode_score(scores) - 1
        yield Record(
            id=f"{SOURCE}:{row.id}:{crit.name}", state=state, type="score", prompt=prompt,
            candidates=list(spec.level_schemes[scheme]), target=target, criterion=crit.name, level_scheme=scheme,
            uses_context=crit.context, history_trimmed=len(row.history) - n_hist, judge_scores=list(scores), **base,
        )
    faith = spec.faithfulness
    if faith and row.chunks and row.scores.get(faith["from"]):
        s = mode_score(row.scores[faith["from"]])
        label = 1.0 if s >= faith.get("pos_min", 4) else 0.0 if s <= faith.get("neg_max", 2) else None
        if label is not None:
            yield Record(
                id=f"{SOURCE}:{row.id}:faithfulness", state=list(row.chunks), type="bool",
                prompt=FAITHFULNESS_TEMPLATE.format(question=row.question, answer=row.answer), target=label,
                criterion="faithfulness_full", template="judge/per_documents",
                **{**base, "label_source": "judge_weak"},
            )


def live_records(
    question: str, answer: str, guard_allowed: bool | None, chunks: Sequence[str], spec: CriteriaSpec,
    history: Sequence[str] = (), count_tokens: Callable[[str], int] | None = None, max_len: int = 8192,
) -> list[Record]:
    """The Score records of a new, unjudged answer, built exactly as the stand's test records were (the
    eval level scheme, the same states, history trimming and prompt choice), for a Stage C model at serving
    time. ``chunks``: the retrieved chunks as the assistant returned them (the stand logs them joined and
    the converter splits them back); ``history``: earlier turns, each ``"User: …\\nAssistant: …"``;
    ``guard_allowed``: as the stand logs it (``any`` of the guarded queries allowed). Targets are dummies."""
    row = StandRow(id=f"live-{stable_hash('live', question, answer):016x}", group="live", question=question.strip(),
                   answer=answer.strip(), guard_allowed=guard_allowed, history=list(history),
                   chunks=[c.strip() for c in chunks if c and c.strip()],
                   scores={c.name: [3] for c in spec.criteria}, fields={})
    return [r for r in row_records(row, "test_in", spec, count_tokens, max_len) if r.type == "score"]


def convert_rows(
    rows: Iterable[Mapping[str, Any]], spec: CriteriaSpec, *, explicit_splits: Mapping[str, str] | None = None,
    shares: Mapping[str, float] | None = None, eval_eligible: Callable[[StandRow], bool] | None = None,
    count_tokens: Callable[[str], int] | None = None, max_len: int = 8192, seed: int = 42,
) -> tuple[list[Record], dict]:
    """All records of the stand rows and a report (rows, skips, records per split / criterion / level)."""
    stand_rows, skipped = merge_rows(rows, spec)
    splits = assign_splits(stand_rows, explicit_splits, shares, eval_eligible, seed)
    records = [rec for r in stand_rows for rec in row_records(r, splits[r.id], spec, count_tokens, max_len)]
    report = {
        "rows": len(stand_rows), "groups": len({r.group for r in stand_rows}), "skipped": dict(skipped),
        "rows_per_split": dict(Counter(splits.values())), "records": len(records),
        "records_per_split": dict(Counter(r.split for r in records)),
        "records_per_criterion": dict(Counter(r.criterion for r in records)),
        "judge_repeats": dict(Counter(max((len(v) for v in r.scores.values()), default=0) for r in stand_rows)),
        "context_criteria_skipped_no_chunks": sum(1 for r in stand_rows if r.lacks_context()),
        "history_trimmed_records": sum(1 for r in records if (r.field("history_trimmed") or 0) > 0),
        "score_levels": {
            c.name: dict(sorted(Counter(mode_score(r.scores[c.name]) for r in stand_rows if r.scores.get(c.name)).items()))
            for c in spec.criteria
        },
    }
    return records, report
