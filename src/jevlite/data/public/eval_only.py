"""Eval-only sources for ``test_ood_source`` (todo, Stage A review, Q3; ``DATASETS.md`` sections 3 and 6).

All three are held out: their test split goes to ``test_ood`` only, never to train, calib or dev.

- **LLM-AggreFact**: is the claim supported by the document? 11 subsets (RAG, summarisation, dialogue,
  expert QA), the subset is the domain. Bool only (two classes). Gated on the Hub: request access at
  https://huggingface.co/datasets/lytang/LLM-AggreFact with the account of ``HF_TOKEN``. CC BY-ND 4.0;
  the card says it "should not be used in pretraining or fine-tuning", so eval only. Its documents are
  often longer than 512 tokens: report it by length bucket.
- **FaithEval**: faithfulness to the context against the model's own knowledge. ``unanswerable``: the
  answer was removed from the context, so every record is answerability = 0 (one class: accuracy is
  the recall of "not answerable"). ``counterfactual``: ARC questions with a context that supports a
  counterfactual option; Choice, the target is the option the context supports. ``inconsistent``
  (two conflicting answers in one context) is left out: it has no label in our schema. HF: no
  license, "research purposes only" — eval only, never trained on.
- **MMLU**: 57 subjects, K = 4, no passage; a new Choice source (Choice OOD had only BANKING77 and
  ContractNLI). MIT. Only the ``test`` split; ``auxiliary_train`` is never read (it bundles RACE, NC).
"""

from __future__ import annotations

import ast
from typing import Iterable, Iterator

from ..unified import Record
from .base import NOT_MENTIONED, ENTAILED, Converter, answerable_record, hf_rows, nli_records
from .mcqa import _letter_index, mcqa_record


def _parsed(value):
    """FaithEval stores some fields as Python reprs (``"{'label': [...], ...}"``)."""
    return ast.literal_eval(value) if isinstance(value, str) else value


class LLMAggreFact(Converter):
    name = "llm_aggrefact"
    domain = "aggrefact"
    family = "factcheck"
    license = "CC BY-ND 4.0; eval only (card: not for training)"
    origin = "hf:lytang/LLM-AggreFact (gated)"
    held_out = True
    splits = {"test": "test_ood"}

    def rows(self, source_split: str) -> Iterable[dict]:
        for i, row in enumerate(hf_rows("lytang/LLM-AggreFact", None, source_split, self.streaming)):
            yield {**row, "idx": i}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        subset = str(row.get("dataset") or "unknown")
        return nli_records(
            self, key=(source_split, subset, row["idx"]), source_split=source_split, state=row["doc"], claim=row["claim"],
            label=ENTAILED if int(row["label"]) == 1 else NOT_MENTIONED, three_way=False, domain=f"aggrefact/{subset}",
        )


class FaithEval(Converter):
    name = "faitheval"
    domain = "faitheval"
    family = "answerable"
    license = "HF: none, \"research purposes only\"; code Apache-2.0; eval only"
    origin = "hf:Salesforce/FaithEval-{unanswerable,counterfactual}-v1.0"
    held_out = True
    splits = {"test": "test_ood"}
    TASKS = ("unanswerable", "counterfactual")

    def rows(self, source_split: str) -> Iterator[dict]:
        for task in self.TASKS:
            # qids repeat in ``unanswerable``: the row position keeps ids unique
            for i, row in enumerate(hf_rows(f"Salesforce/FaithEval-{task}-v1.0", None, source_split, self.streaming)):
                yield {**row, "task": task, "idx": i}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        if row["task"] == "unanswerable":
            subset = str(row.get("subset") or "unknown").lower()
            return [answerable_record(
                self, key=(source_split, "unanswerable", row["idx"]), source_split=source_split, state=row["context"],
                question=row["question"], answerable=False, domain=f"faitheval/unanswerable/{subset}",
            )]
        choices = _parsed(row["choices"])
        return [
            Record.model_validate({**r.model_dump(), "domain": "faitheval/counterfactual", "family": "mcqa"})
            for r in mcqa_record(
                self, key=(source_split, "counterfactual", row["idx"]), source_split=source_split, state=row["context"],
                options=choices["text"], answer=_letter_index(choices["label"], row["answerKey"]), question=row["question"],
            )
        ]


class MMLU(Converter):
    name = "mmlu"
    domain = "mmlu"
    family = "mcqa"
    license = "MIT (HF, GitHub)"
    origin = "hf:cais/mmlu/all"
    held_out = True
    splits = {"test": "test_ood"}

    def rows(self, source_split: str) -> Iterable[dict]:
        for i, row in enumerate(hf_rows("cais/mmlu", "all", source_split, self.streaming)):
            yield {**row, "idx": i}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        subject = row.get("subject") or "unknown"
        return [
            Record.model_validate({**r.model_dump(), "domain": f"mmlu/{subject}"})
            for r in mcqa_record(
                self, key=(source_split, subject, row["idx"]), source_split=source_split, state=row["question"],
                options=row["choices"], answer=int(row["answer"]),
            )
        ]
