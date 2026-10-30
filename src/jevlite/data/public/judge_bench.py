"""Judge benchmarks: eval only, never trained on (``DATASETS.md`` section 7, approved 2026-10-16).

Each tests a judge on a context it must stay faithful to, the setting of our judge (criteria a and b):

- **ContextualJudgeBench** (Salesforce, ACL 2025): 2,000 pairs of a positive and a negative response to
  one (context, question), 8 splits = 4 criteria (refusal, faithfulness, completeness, conciseness) ×
  QA / summarisation. Choice K = 2 with both responses in the state, like ``summarize_from_feedback``.
  Every pair gives two records, one per response order, with the same ``group``: the paper's main
  metric, *consistent accuracy*, counts a pair as right only when both orders are. CC BY 4.0. Its QA
  sources include MRQA (SQuAD, Natural Questions), so passage overlap with our training data is
  expected; ``clean_eval.py`` filters it.
- **RAGBench** (Galileo): 12 RAG subsets, test splits only. Bool "the response is supported by the
  documents" from ``adherence_score``. Responses are LLM-written and the labels come from GPT-4o, so
  this measures agreement with an LLM judge, not with people. CC BY 4.0.
- **HaluBench** (Patronus AI): 14.9k (passage, question, answer) triples from HaluEval, DROP, CovidQA,
  PubMedQA, FinanceBench and RAGTruth, ``PASS`` / ``FAIL``. Bool. CC BY-NC 2.0: eval only.
- **FaithBench** (Vectara, NAACL 2025): 800 LLM summaries with human hallucination spans. Bool on the
  sample level from the ``worst-label`` pooling of their ``binarize.py``: ``Unwanted`` → hallucinated,
  ``Consistent`` / ``Benign`` → faithful, ``Questionable`` is left out (no label in our schema).
  CC BY-NC-SA 4.0: eval only.

Faithfulness records (RAGBench, HaluBench, FaithBench) use the NLI form of LLM-AggreFact: the context
is the state and the response is the claim, so the NLI baseline reads them the same way.
"""

from __future__ import annotations

import ast
import csv
import io
from typing import Iterable, Iterator

from ..unified import Record
from .base import Converter, Template, fetch, hf_rows, pick_template


def _answer_text(value) -> str:
    """HaluBench stores DROP answers as a Python list repr (``"['Rams', 'second']"``)."""
    if isinstance(value, str) and value.startswith("["):
        try:
            parsed = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value.strip()
        if isinstance(parsed, list):
            return ", ".join(str(p).strip() for p in parsed)
    return str(value).strip()


# One wording per source: eval only, so there is no held-out template to carve.
RAG_CLAIM = Template("rag_answer", "The answer to the question \"{question}\" is: {answer}")


class ContextualJudgeBench(Converter):
    name = "contextual_judge_bench"
    domain = "cjb"
    family = "judge_pairwise"
    license = "CC BY 4.0; eval only"
    origin = "hf:Salesforce/ContextualJudgeBench"
    held_out = True
    CRITERIA = ("refusal_answerable", "refusal_unanswerable", "faithfulness_qa", "faithfulness_summ",
                "completeness_qa", "completeness_summ", "conciseness_qa", "conciseness_summ")
    splits = {s: "test_ood" for s in CRITERIA}
    # Criterion → (QA wording, summarisation wording). Refusal splits are QA only.
    PROMPTS = {
        "refusal": ("Which response handles the question correctly given the context: it answers if the "
                    "context contains the answer and declines if it does not?", None),
        "faithfulness": ("Which response is fully supported by the context?",
                         "Which summary is fully supported by the context?"),
        "completeness": ("Which response covers all the information from the context that the question needs?",
                         "Which summary covers all the key information from the context?"),
        "conciseness": ("Which response answers the question more concisely, without unnecessary content?",
                        "Which summary is more concise, without unnecessary content?"),
    }

    def rows(self, source_split: str) -> Iterator[dict]:
        for i, row in enumerate(hf_rows("Salesforce/ContextualJudgeBench", None, source_split, self.streaming)):
            yield {**row, "idx": i}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        criterion, _, task = source_split.partition("_")
        qa_prompt, summ_prompt = self.PROMPTS[criterion]
        summ = task == "summ"
        prompt = summ_prompt if summ else qa_prompt
        noun = "Summary" if summ else "Response"
        pos, neg = row["positive_response"].strip(), row["negative_response"].strip()
        if not pos or not neg or pos == neg:
            return []
        out = []
        for order, (first, second) in enumerate(((pos, neg), (neg, pos))):
            state = (
                f"Context:\n{row['context'].strip()}\n\nRequest:\n{row['question'].strip()}"
                f"\n\n{noun} 1:\n{first}\n\n{noun} 2:\n{second}"
            )
            out.append(self.record(
                key=(source_split, row["idx"], order), split="test_ood", state=state, type="choice", prompt=prompt,
                candidates=[f"{noun.lower()} 1", f"{noun.lower()} 2"], target=order, domain=f"cjb/{source_split}",
                criterion=criterion, group=f"{source_split}:{row['idx']}", template=f"cjb/{criterion}/{task}",
                order=order, origin_source=row.get("source"),
            ))
        return out


class RAGBench(Converter):
    name = "ragbench"
    domain = "ragbench"
    family = "factcheck"
    license = "CC BY 4.0; labels by GPT-4o; eval only"
    origin = "hf:galileo-ai/ragbench (formerly rungalileo/ragbench)"
    held_out = True
    splits = {"test": "test_ood"}
    SUBSETS = ("covidqa", "cuad", "delucionqa", "emanual", "expertqa", "finqa", "hagrid", "hotpotqa",
               "msmarco", "pubmedqa", "tatqa", "techqa")

    def rows(self, source_split: str) -> Iterator[dict]:
        for subset in self.SUBSETS:
            for i, row in enumerate(hf_rows("galileo-ai/ragbench", subset, source_split, self.streaming)):
                yield {**row, "subset": subset, "idx": i}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        docs = [d.strip() for d in row["documents"] if d and d.strip()]
        response = (row.get("response") or "").strip()
        if not docs or not response or row.get("adherence_score") is None:
            return []
        return [self.record(
            key=(source_split, row["subset"], row["idx"]), split="test_ood", state=docs, type="bool",
            prompt=RAG_CLAIM.prompt.format(question=row["question"].strip(), answer=response),
            target=float(bool(row["adherence_score"])), domain=f"ragbench/{row['subset']}", criterion="faithfulness",
            template=f"judge_bench/{RAG_CLAIM.id}", generation_model=row.get("generation_model_name"),
        )]


class RAGBenchCalib(RAGBench):
    """Train + validation of the 4 held-out RAGBench subsets, in the eval form of :class:`RAGBench`: the
    judge layer of ``jevlite.calibration`` is fitted on them (user, 2026-10-30). Never trained on and not
    in eval v2 (which holds their test split); msmarco stays out (NC). Train and test share documents:
    records whose passages occur in the RAGBench test file are dropped with ``overlap_ids.py``."""

    name = "ragbench_calib"
    license = "CC BY 4.0; labels by GPT-4o; calibration only"
    origin = "hf:galileo-ai/ragbench, train + validation of covidqa, cuad, delucionqa, emanual"
    held_out = False
    splits = {"train": "calib", "validation": "calib"}
    SUBSETS = ("covidqa", "cuad", "delucionqa", "emanual")

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        return [r.model_copy(update={"split": "calib"}) for r in super().convert(row, source_split)]


class HaluBench(Converter):
    name = "halubench"
    domain = "halubench"
    family = "factcheck"
    license = "CC BY-NC 2.0; eval only"
    origin = "hf:PatronusAI/HaluBench"
    held_out = True
    splits = {"test": "test_ood"}

    def rows(self, source_split: str) -> Iterator[dict]:
        for i, row in enumerate(hf_rows("PatronusAI/HaluBench", None, source_split, self.streaming)):
            yield {**row, "idx": i}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        passage, answer = (row.get("passage") or "").strip(), _answer_text(row.get("answer") or "")
        if not passage or not answer or row["label"] not in ("PASS", "FAIL"):
            return []
        subset = str(row.get("source_ds") or "unknown").lower()
        return [self.record(
            key=(source_split, row["idx"]), split="test_ood", state=passage, type="bool",
            prompt=RAG_CLAIM.prompt.format(question=row["question"].strip(), answer=answer),
            target=float(row["label"] == "PASS"), domain=f"halubench/{subset}", criterion="faithfulness",
            template=f"judge_bench/{RAG_CLAIM.id}",
        )]


class FaithBench(Converter):
    name = "faithbench"
    domain = "faithbench"
    family = "factcheck"
    license = "CC BY-NC-SA 4.0; eval only"
    # Pinned commit: the CSV is the sample-level release (source, summary, LLM, worst-label, best-label).
    COMMIT = "cf89797d82812c23b5d5e5c121f1d9b8983bbbce"
    origin = f"github:vectara/FaithBench@{COMMIT[:7]}/FaithBench.csv"
    held_out = True
    splits = {"test": "test_ood"}
    FAITHFUL = {"Consistent": 1.0, "Benign": 1.0, "Unwanted": 0.0}  # Questionable: left out

    def rows(self, source_split: str) -> Iterator[dict]:
        url = f"https://raw.githubusercontent.com/vectara/FaithBench/{self.COMMIT}/FaithBench.csv"
        path = fetch(url, f"faithbench_{self.COMMIT[:7]}.csv")
        csv.field_size_limit(1 << 30)
        with io.open(path, newline="", encoding="utf-8") as f:
            for i, row in enumerate(csv.DictReader(f)):
                yield {**row, "idx": i}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        source, summary = (row.get("source") or "").strip(), (row.get("summary") or "").strip()
        label = self.FAITHFUL.get(row.get("worst-label", ""))
        if not source or not summary or label is None:
            return []
        return [self.record(
            key=(source_split, row["idx"]), split="test_ood", state=source, type="bool", prompt=summary,
            target=label, criterion="faithfulness", template="nli/claim", summarizer=row.get("LLM"),
            worst_label=row["worst-label"], best_label=row.get("best-label"),
        )]
