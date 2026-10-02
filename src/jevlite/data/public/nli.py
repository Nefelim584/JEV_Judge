"""Sentence-level NLI and paraphrase sets: low-weight warm-up data (``DATASETS.md`` section 0)."""

from __future__ import annotations

from typing import Iterable

from ..unified import Record
from .base import (
    CONTRADICTED,
    ENTAILED,
    NOT_MENTIONED,
    Converter,
    Template,
    answerable_record,
    hf_rows,
    nli_records,
    pick_template,
    unique_pairs,
)

# MultiNLI, SNLI and the GLUE copies: 0 entailment, 1 neutral, 2 contradiction (-1 = no gold label).
_GLUE_NLI = {0: ENTAILED, 1: NOT_MENTIONED, 2: CONTRADICTED}
_WANLI = {"entailment": ENTAILED, "neutral": NOT_MENTIONED, "contradiction": CONTRADICTED}


class MultiNLI(Converter):
    name = "multi_nli"
    family = "nli"
    license = "OANC + CC BY-SA 3.0 / CC BY 3.0 (fiction)"
    origin = "hf:nyu-mll/multi_nli"
    # The mismatched dev set has genres absent from train: it is OOD by construction.
    splits = {"train": "train", "validation_matched": "test_in", "validation_mismatched": "test_ood"}

    def rows(self, source_split: str) -> Iterable[dict]:
        # pairID is not unique (it repeats across different hypotheses), and a few pairs repeat with
        # conflicting labels.
        rows = hf_rows("nyu-mll/multi_nli", None, source_split, self.streaming)
        return unique_pairs(rows, ("premise", "hypothesis"), "label", self.streaming)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        if row["label"] not in _GLUE_NLI:
            return []
        return nli_records(
            self, key=(source_split, row["idx"]), source_split=source_split, state=row["premise"],
            claim=row["hypothesis"], label=_GLUE_NLI[row["label"]], domain=row["genre"],
        )


class SNLI(Converter):
    name = "snli"
    domain = "captions"
    family = "nli"
    license = "CC BY-SA 4.0"
    origin = "hf:stanfordnlp/snli"
    splits = {"train": "train", "test": "test_in"}

    def rows(self, source_split: str) -> Iterable[dict]:
        # No ids, and some pairs repeat, a few of them with conflicting labels.
        rows = hf_rows("stanfordnlp/snli", None, source_split, self.streaming)
        return unique_pairs(rows, ("premise", "hypothesis"), "label", self.streaming)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        if row["label"] not in _GLUE_NLI:
            return []
        return nli_records(
            self, key=(source_split, row["idx"]), source_split=source_split,
            state=row["premise"], claim=row["hypothesis"], label=_GLUE_NLI[row["label"]],
        )


class WANLI(Converter):
    name = "wanli"
    domain = "generated"
    family = "nli"
    license = "CC BY 4.0 (pairs written by GPT-3, labelled by humans)"
    origin = "hf:alisawuffles/WANLI"
    splits = {"train": "train", "test": "test_in"}

    def rows(self, source_split: str) -> Iterable[dict]:
        return hf_rows("alisawuffles/WANLI", None, source_split, self.streaming)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        return nli_records(
            self, key=(source_split, row["id"]), source_split=source_split, state=row["premise"],
            claim=row["hypothesis"], label=_WANLI[row["gold"]], llm_generated=True,
        )


class QNLI(Converter):
    """Does the sentence answer the question? Bool only, as answerability over one sentence."""

    name = "qnli"
    domain = "wiki"
    family = "answerable"
    license = "CC BY-SA 4.0 (from SQuAD 1.1)"
    origin = "hf:nyu-mll/glue/qnli"
    # GLUE test labels are hidden.
    splits = {"train": "train", "validation": "test_in"}

    def rows(self, source_split: str) -> Iterable[dict]:
        return hf_rows("nyu-mll/glue", "qnli", source_split, self.streaming)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        if row["label"] not in (0, 1):
            return []
        return [
            answerable_record(
                self, key=(source_split, row["idx"]), source_split=source_split, state=row["sentence"],
                question=row["question"], answerable=row["label"] == 0,  # 0 = entailment
            )
        ]


PARAPHRASE = (
    Template("same", "Sentence A and sentence B mean the same thing."),
    Template("paraphrase", "Sentence B is a paraphrase of sentence A."),
    Template("equivalent", "The two sentences state the same fact."),
)


class PAWS(Converter):
    """Paraphrase vs word-swap: the state is a JSON object with both sentences."""

    name = "paws"
    domain = "wiki"
    family = "paraphrase"
    license = "free for any purpose (repo LICENSE)"
    origin = "hf:google-research-datasets/paws/labeled_final"
    splits = {"train": "train", "test": "test_in"}

    def rows(self, source_split: str) -> Iterable[dict]:
        return hf_rows("google-research-datasets/paws", "labeled_final", source_split, self.streaming)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        key = (source_split, row["id"])
        tpl, split = pick_template(PARAPHRASE, self.target_split(source_split, *key), self.name, *key)
        return [
            self.record(
                key=key, split=split, state={"sentence A": row["sentence1"], "sentence B": row["sentence2"]},
                type="bool", prompt=tpl.prompt, target=float(row["label"] == 1), template=f"paraphrase/{tpl.id}",
            )
        ]
