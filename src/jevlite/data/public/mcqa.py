"""Multiple-choice QA: Choice over the given answer options (``DATASETS.md`` section 0)."""

from __future__ import annotations

import csv
import json
import zipfile
from typing import Iterable, Iterator, Sequence

from ..unified import Record
from .base import Converter, Template, fetch, hf_rows, pick_template

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


def _letter_index(labels: Sequence[str], key: str) -> int:
    return list(labels).index(key) if key in labels else -1


class ARC(Converter):
    name = "arc"
    domain = "science_exams"
    family = "mcqa"
    license = "CC BY-SA 4.0"
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


class OpenBookQA(Converter):
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


class CommonsenseQA(Converter):
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


class CosmosQA(Converter):
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


class SocialIQa(Converter):
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
