"""Question answering over a passage: yes/no questions and answerability (``DATASETS.md`` section 0)."""

from __future__ import annotations

import json
from typing import Iterable, Iterator

from ..unified import Record
from .base import Converter, Template, answerable_record, fetch, hf_rows, pick_template

BOOLQ = (
    Template("question", "{question}?"),
    Template("answer_yes", "The answer to \"{question}?\" is yes, according to the text."),
)


class BoolQ(Converter):
    name = "boolq"
    domain = "wiki"
    family = "yes_no_qa"
    license = "CC BY-SA 3.0"
    origin = "hf:google/boolq"

    def rows(self, source_split: str) -> Iterable[dict]:
        return hf_rows("google/boolq", None, source_split, self.streaming)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        # No ids in the source; the question is unique within a split.
        key = (source_split, row["question"])
        tpl, split = pick_template(BOOLQ, self.target_split(source_split, *key), self.name, *key)
        return [
            self.record(
                key=key, split=split, state=row["passage"], type="bool",
                prompt=tpl.prompt.format(question=row["question"].strip().rstrip("?")),
                target=float(bool(row["answer"])), template=f"boolq/{tpl.id}",
            )
        ]


class SQuAD2(Converter):
    """Answerable vs adversarially unanswerable questions over a Wikipedia paragraph."""

    name = "squad_v2"
    domain = "wiki"
    family = "answerable"
    license = "CC BY-SA 4.0"
    origin = "hf:rajpurkar/squad_v2"

    def rows(self, source_split: str) -> Iterable[dict]:
        return hf_rows("rajpurkar/squad_v2", "squad_v2", source_split, self.streaming)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        return [
            answerable_record(
                self, key=(source_split, row["id"]), source_split=source_split, state=row["context"],
                question=row["question"], answerable=bool(row["answers"]["text"]),
            )
        ]


class ClapNQ(Converter):
    """Natural Questions passages with answerable and unanswerable questions (the GitHub release; the HF
    copy has the answerable ones only)."""

    name = "clapnq"
    domain = "wiki"
    family = "answerable"
    license = "Apache-2.0 (NQ text: CC BY-SA 3.0)"
    origin = "https://github.com/primeqa/clapnq"
    splits = {"train": "train", "dev": "test_in"}

    _URL = "https://raw.githubusercontent.com/primeqa/clapnq/main/annotated_data/{split}/clapnq_{split}_{kind}.jsonl"

    def rows(self, source_split: str) -> Iterator[dict]:
        for kind in ("answerable", "unanswerable"):
            path = fetch(self._URL.format(split=source_split, kind=kind))
            with path.open() as f:
                for line in f:
                    if line.strip():
                        yield json.loads(line)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        passages = [f"{p['title']}\n{p['text']}" for p in row["passages"]]
        state = passages[0] if len(passages) == 1 else passages
        answerable = any(o.get("answer") for o in row["output"])
        return [
            answerable_record(
                self, key=(source_split, row["id"]), source_split=source_split, state=state,
                question=row["input"], answerable=answerable,
            )
        ]
