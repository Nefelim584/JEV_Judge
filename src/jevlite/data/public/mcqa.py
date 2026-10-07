"""Multiple-choice QA: Choice over the given answer options (``DATASETS.md`` section 0).

Every MCQA set has a fixed number of options (ARC 3–5, OpenBookQA 4, CommonsenseQA 5, Cosmos QA 4,
Social IQa 3). To vary K, about half of the records get 1–3 extra options **borrowed from other
questions** of the same dataset and source split (``borrow_distractors``), inserted at random positions.
"""

from __future__ import annotations

import csv
import itertools
import json
import zipfile
from typing import Any, Iterable, Iterator, Sequence

from ...schema import MAX_CANDIDATES
from ..unified import Record
from .base import Converter, Template, fetch, hf_rows, pick_template, rng_for

BORROW_SHARE = 0.5
BORROW_MAX = 3

# No passage (ARC, OpenBookQA, CommonsenseQA): the question is the state.
NO_CONTEXT = (
    Template("correct", "Which answer to the question is correct?"),
    Template("best", "Choose the best answer."),
    Template("option", "Which option correctly answers the question in the text?"),
)
# With a passage (Cosmos QA, Social IQa): the passage is the state, the question is the prompt.
WITH_CONTEXT = (
    Template("question", "{question}"),
    Template("based_on", "Based on the text: {question}"),
    Template("best_answer", "What is the best answer to this question about the text? {question}"),
)


def _valid_options(options: Sequence[str]) -> bool:
    stripped = [o.strip() for o in options]
    return all(stripped) and len(set(stripped)) == len(stripped)


def mcqa_record(
    conv: Converter, *, key: Sequence, source_split: str, state: str, options: Sequence[str], answer: int,
    question: str | None = None,
) -> list[Record]:
    if not _valid_options(options) or not 0 <= answer < len(options) or not state.strip():
        return []
    templates = WITH_CONTEXT if question is not None else NO_CONTEXT
    tpl, split = pick_template(templates, conv.target_split(source_split, *key), conv.name, *key)
    prompt = tpl.prompt.format(question=(question or "").strip())
    return [
        conv.record(
            key=key, split=split, state=state, type="choice", prompt=prompt,
            candidates=[o.strip() for o in options], target=answer, template=f"mcqa/{tpl.id}",
        )
    ]


def borrow_distractors(
    records: Sequence[Record], *key: Any, share: float = BORROW_SHARE, n_max: int = BORROW_MAX
) -> list[Record]:
    """Add 1…``n_max`` options taken from the other records to about ``share`` of the records.

    A borrowed option never repeats one of the record's own options (case-insensitive), so the gold
    option stays the only correct one. Records keep their order; the field ``borrowed_distractors``
    holds the number of options added (0 for untouched records).
    """
    pool = sorted({c.strip() for r in records for c in r.candidates})
    out = []
    for r in records:
        rng = rng_for("borrow", *key, r.id)
        room = MAX_CANDIDATES - len(r.candidates)
        if rng.random() >= share or room <= 0 or isinstance(r.target, list):
            out.append(Record.model_validate({**r.model_dump(), "borrowed_distractors": 0}))
            continue
        n = min(rng.randint(1, n_max), room)
        taken = {c.casefold() for c in r.candidates}
        extra = [c for c in rng.sample(pool, min(len(pool), 4 * n + len(r.candidates))) if c.casefold() not in taken][:n]
        options, gold = list(r.candidates), r.candidates[int(r.target)]
        for c in extra:
            options.insert(rng.randint(0, len(options)), c)
        out.append(Record.model_validate({
            **r.model_dump(), "candidates": options, "target": options.index(gold), "borrowed_distractors": len(extra),
        }))
    return out


class _MCQA(Converter):
    """Converts a whole source split at once, so that options can be borrowed across questions."""

    def records(self, limit: int | None = None) -> Iterator[Record]:
        for source_split in self.splits:
            rows = itertools.islice(self.rows(source_split), limit)
            records = [r for row in rows for r in self.convert(row, source_split)]
            yield from borrow_distractors(records, self.name, source_split)


def _letter_index(labels: Sequence[str], key: str) -> int:
    return list(labels).index(key) if key in labels else -1


class ARC(_MCQA):
    name = "arc"
    domain = "science_exams"
    family = "mcqa"
    license = "CC BY-SA 4.0"
    share_alike = True
    origin = "hf:allenai/ai2_arc"
    splits = {"train": "train", "test": "test_in"}

    def rows(self, source_split: str) -> Iterator[dict]:
        for config in ("ARC-Challenge", "ARC-Easy"):
            for row in hf_rows("allenai/ai2_arc", config, source_split, self.streaming):
                yield {**row, "config": config}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        labels = row["choices"]["label"]
        return mcqa_record(
            self, key=(source_split, row["id"]), source_split=source_split, state=row["question"],
            options=row["choices"]["text"], answer=_letter_index(labels, row["answerKey"]),
        )


class OpenBookQA(_MCQA):
    name = "openbookqa"
    domain = "science_exams"
    family = "mcqa"
    license = "Apache-2.0 (GitHub allenai/OpenBookQA)"
    origin = "hf:allenai/openbookqa/main"
    splits = {"train": "train", "test": "test_in"}

    def rows(self, source_split: str) -> Iterable[dict]:
        return hf_rows("allenai/openbookqa", "main", source_split, self.streaming)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        labels = row["choices"]["label"]
        return mcqa_record(
            self, key=(source_split, row["id"]), source_split=source_split, state=row["question_stem"],
            options=row["choices"]["text"], answer=_letter_index(labels, row["answerKey"]),
        )


class CommonsenseQA(_MCQA):
    name = "commonsense_qa"
    domain = "commonsense"
    family = "mcqa"
    license = "MIT (HF card)"
    origin = "hf:tau/commonsense_qa"
    # Test answers are hidden.
    splits = {"train": "train", "validation": "test_in"}

    def rows(self, source_split: str) -> Iterable[dict]:
        return hf_rows("tau/commonsense_qa", None, source_split, self.streaming)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        labels = row["choices"]["label"]
        return mcqa_record(
            self, key=(source_split, row["id"]), source_split=source_split, state=row["question"],
            options=row["choices"]["text"], answer=_letter_index(labels, row["answerKey"]),
        )


class CosmosQA(_MCQA):
    name = "cosmos_qa"
    domain = "blogs"
    family = "mcqa"
    license = "CC BY 4.0 (HF card)"
    origin = "https://github.com/wilburOne/cosmosqa"
    # test.jsonl has no labels.
    splits = {"train": "train", "valid": "test_in"}

    def rows(self, source_split: str) -> Iterator[dict]:
        path = fetch(f"https://github.com/wilburOne/cosmosqa/raw/master/data/{source_split}.csv", f"cosmosqa_{source_split}.csv")
        with path.open(newline="") as f:
            yield from csv.DictReader(f)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        return mcqa_record(
            self, key=(source_split, row["id"]), source_split=source_split, state=row["context"],
            question=row["question"], options=[row[f"answer{i}"] for i in range(4)], answer=int(row["label"]),
        )


class SocialIQa(_MCQA):
    name = "social_iqa"
    domain = "social"
    family = "mcqa"
    license = "CC BY 4.0 (card text)"
    origin = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"
    splits = {"train": "train", "dev": "test_in"}

    def rows(self, source_split: str) -> Iterator[dict]:
        with zipfile.ZipFile(fetch(self.origin)) as z:
            base = "socialiqa-train-dev"
            items = z.read(f"{base}/{source_split}.jsonl").decode().splitlines()
            labels = z.read(f"{base}/{source_split}-labels.lst").decode().split()
        for i, (line, label) in enumerate(zip(items, labels, strict=True)):
            yield {**json.loads(line), "idx": i, "label": int(label)}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        return mcqa_record(
            self, key=(source_split, row["idx"]), source_split=source_split, state=row["context"],
            question=row["question"], options=[row["answerA"], row["answerB"], row["answerC"]],
            answer=row["label"] - 1,  # labels are 1-based
        )
