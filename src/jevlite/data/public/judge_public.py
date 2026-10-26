"""Judge training data from new public sources, Stage B (``DATASETS.md`` 8.2, approved 2026-10-26).

- **Faithfulness-QA** (``faithfulness_qa``): SQuAD / TriviaQA passages with the answer entity replaced.
  The world-true original answer against the edited passage is not supported (``contradicts``); the
  replacement is. Teaches "the text wins over what you know".
- **MiniCheck C2D / D2C** (``minicheck``): claims that need several sentences of a document, Bool.
  C2D claims come from WiCE, which sits inside LLM-AggreFact (an eval set): C2D claims found in
  LLM-AggreFact are dropped, and without access to it (gated) C2D is left out entirely.
- **Trust-Data** (``trust_data``), Trust-Align DPO pairs: documents + question + a preferred and a
  dispreferred answer, the preferred one a refusal when the documents do not answer. Kept only for
  QAMPARI (its own instruction) and ASQA (question found in ASQA); the rest, which holds the
  unlicensed ELI5 part, is dropped. Choice K = 2 (both orders) and answerability.
- **PsiloQA**, English (``psiloqa``): Wikipedia passage, question, an open LLM's answer written without
  the passage, GPT-4o hallucination spans. The answer and one sentence against the passage, Bool only:
  the spans mark deviation from the golden answer and do not separate contradicts from silent. Almost
  every such answer has a span, so the golden answer (GPT-4o, written from the passage) is added as
  the positive on the same passage.
- **Qasper** (``qasper``): research papers. Answerability on chunks of the same paper (evidence +
  the paragraphs closest to the question, and the same with the evidence swapped out), and the
  annotators' answers against the evidence, the negative being another question's answer from the
  same paper.
- **RAGTruth**, summaries of CNN/DM articles (``ragtruth``): human spans with types. The summary and one
  sentence against the article; "Recent News" articles are left out (unclear source terms).

Same conventions as ``judge_train.py``: the question is written into the claim, the eval wording is
not used, the split is drawn per source document.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Iterable, Iterator

from ...log import logger
from ..unified import Record
from .base import ANSWERABLE, ENTAILED, NOT_MENTIONED, Converter, Template, fetch, hf_rows, nli_records, pick_template, rng_for
from .judge_train import ANSWER_CLAIM, SENTENCE_CLAIM, judge_record

SUMMARY_CLAIM = (
    Template("summary_says", "A summary of the text says: {summary}"),
    Template("summary_of", "Summary of the article: {summary}"),
    Template("summary_faithful", "This summary sticks to the text: {summary}"),  # held out
)
SUMMARY_SENTENCE = (
    Template("summary_sentence", "A summary of the text contains the sentence: {sentence}"),
    Template("summary_line", "One line of a summary of this article: {sentence}"),
    Template("summary_sentence_held", "The text backs this sentence of its summary: {sentence}"),  # held out
)
PREFERENCE = (
    Template("grounded_or_refuse", "Which answer uses only what the documents say, and declines when they do not answer?"),
    Template("better_grounded", "Which answer is the better reply given these documents?"),
    Template("trustworthy", "Which answer would a careful assistant with these documents give?"),  # held out
)


def first_by_id(rows: Iterable[dict], name: str, split: str, key: str = "id") -> Iterator[dict]:
    """Rows with a repeated ``key`` collapsed to the first one (record ids must be unique per source;
    TriviaQA-based rows repeat a question id across evidence documents)."""
    seen, dropped = set(), 0
    for row in rows:
        k = row.get(key)
        if k in seen:
            dropped += 1
            continue
        seen.add(k)
        yield row
    if dropped:
        logger.info("[{}] {}: {} rows with a repeated {} dropped", name, split, dropped, key)


def _sentences(text: str) -> list[str]:
    from ...judge.claims import split_sentences

    return split_sentences(text)


def sentence_labels(text: str, spans: list[tuple[int, int]]) -> list[tuple[str, bool]]:
    """Sentences of ``text`` (the judge's splitter) and whether each is free of every span."""
    out, pos = [], 0
    for s in _sentences(text):
        start = text.find(s, pos)
        if start < 0:
            continue
        end = start + len(s)
        pos = end
        out.append((s, not any(a < end and b > start for a, b in spans)))
    return out


# ---------------------------------------------------------------------------------------------------


class FaithfulnessQA(Converter):
    name = "faithfulness_qa"
    domain = "wiki/counterfactual"
    family = "judge"
    license = "MIT (HF card, README); SQuAD CC BY-SA 4.0, TriviaQA Wikipedia evidence"
    share_alike = True
    origin = "hf:Laurie/faithfulness-qa-dataset (all)"
    splits = {"train": "train", "validation": "test_in"}

    def rows(self, source_split: str) -> Iterator[dict]:
        rows = hf_rows("Laurie/faithfulness-qa-dataset", "all", source_split, self.streaming)
        yield from first_by_id(({**r, "uid": f"{r.get('source')}:{r['id']}"} for r in rows), self.name, source_split, "uid")

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        context = (row.get("modified_context") or "").strip()
        question = (row.get("question") or "").strip()
        original, faithful = (row.get("original_answer") or "").strip(), (row.get("faithful_answer") or "").strip()
        low = context.lower()
        # the edit must have happened: the new answer is in the text, the original one is gone
        if not context or not question or not original or not faithful or faithful.lower() not in low \
                or original.lower() in low or original.lower() == faithful.lower():
            return []
        # a year must be replaced by a year ("In 42 years, Apple faced …" reads as noise)
        if re.fullmatch(r"\d{4}", str(row.get("original_entity") or "").strip()) and not re.fullmatch(r"\d{4}", faithful):
            return []
        row_key = (source_split, row.get("source", ""), row["id"])
        claim = dict(row_key=row_key, source_split=source_split, state=context, templates=ANSWER_CLAIM,
                     criterion="faithfulness", group=f"{row['id']}", user_question=question,
                     domain=f"wiki/counterfactual/{row.get('source', 'unknown')}", entity_type=row.get("entity_type"))
        return [
            judge_record(self, rec_key=("faithful",), values={"question": question, "answer": faithful}, target=1.0,
                         label_source="program", **claim),
            judge_record(self, rec_key=("original",), values={"question": question, "answer": original}, target=0.0,
                         label_source="program", perturbation="world_true_answer", **claim),
        ]


class MiniCheck(Converter):
    name = "minicheck"
    domain = "minicheck"
    family = "judge"
    license = "MIT (HF card); D2C documents: news articles"
    origin = "hf:lytang/C2D-and-D2C-MiniCheck"
    splits = {"c2d": "train", "d2c": "train"}

    def _aggrefact_claims(self) -> set[str] | None:
        claims = set()
        try:
            for split in ("dev", "test"):
                for row in hf_rows("lytang/LLM-AggreFact", None, split, self.streaming):
                    claims.add(" ".join(str(row["claim"]).lower().split()))
        except Exception as e:  # gated or offline: fail closed
            logger.warning("[minicheck] LLM-AggreFact unavailable ({}): C2D left out", type(e).__name__)
            return None
        return claims

    def rows(self, source_split: str) -> Iterator[dict]:
        banned: set[str] | None = set()
        if source_split == "c2d":
            banned = self._aggrefact_claims()
            if banned is None:
                return
        dropped = 0
        for i, row in enumerate(hf_rows("lytang/C2D-and-D2C-MiniCheck", None, source_split, self.streaming)):
            if " ".join(str(row["claim"]).lower().split()) in banned:
                dropped += 1
                continue
            yield {**row, "idx": i}
        logger.info("[minicheck] {}: {} claims dropped (found in LLM-AggreFact)", source_split, dropped)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        doc = (row.get("doc") or "").strip()
        if not doc:
            return []
        return nli_records(
            self, key=(source_split, row["idx"]), source_split=source_split, state=doc, claim=row["claim"],
            label=ENTAILED if int(row["label"]) == 1 else NOT_MENTIONED, three_way=False,
            domain=f"minicheck/{source_split}", criterion="faithfulness", label_source="llm",
        )


_DOC = re.compile(r"^Document \[(\d+)\]\(Title: (.*?)\):\s*(.*)$")
_REFUSAL = "I apologize, but I couldn't find an answer to your question in the search results."


def trust_prompt(prompt: str) -> tuple[str, str, list[str]] | None:
    """(kind, question, documents) of a Trust-Align prompt; kind is ``qampari`` (its own instruction) or
    ``long`` (the instruction ASQA and ELI5 share)."""
    kind = "qampari" if prompt.startswith("Instruction: Provide a list of accurate answers") else "long"
    questions = re.findall(r"^Question: (.*)$", prompt, flags=re.M)
    docs = []
    for line in prompt.splitlines():
        m = _DOC.match(line.strip())
        if m:
            docs.append(f"{m.group(2).strip()}\n{m.group(3).strip()}")
    if not questions or not docs:
        return None
    return kind, questions[-1].strip(), docs


def _norm_q(q: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", q.lower()).split())


class TrustData(Converter):
    name = "trust_data"
    domain = "trust"
    family = "judge"
    license = "Apache-2.0 (HF); kept: ASQA Apache-2.0, QAMPARI CC0-1.0; ELI5 dropped (no license)"
    origin = "hf:declare-lab/Trust-Data (trust_align_dpo), ASQA whitelist from hf:din0s/asqa"
    splits = {"train": "train", "test": "test_in"}

    def _asqa_questions(self) -> set[str]:
        return {_norm_q(r["ambiguous_question"]) for split in ("train", "dev")
                for r in hf_rows("din0s/asqa", None, split, self.streaming)}

    def rows(self, source_split: str) -> Iterator[dict]:
        asqa = self._asqa_questions()
        kept = dropped = 0
        for i, row in enumerate(hf_rows("declare-lab/Trust-Data", "trust_align_dpo", source_split, self.streaming)):
            parsed = trust_prompt(row["prompt"])
            if parsed is None:
                continue
            kind, question, docs = parsed
            source = "qampari" if kind == "qampari" else "asqa" if _norm_q(question) in asqa else None
            if source is None:
                dropped += 1
                continue
            kept += 1
            yield {"idx": i, "source": source, "question": question, "docs": docs,
                   "chosen": row["chosen"].strip(), "rejected": row["rejected"].strip()}
        logger.info("[trust_data] {}: kept {} (ASQA, QAMPARI), dropped {} (ELI5 or unknown)", source_split, kept, dropped)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        chosen, rejected = row["chosen"], row["rejected"]
        if not chosen or not rejected or chosen == rejected or (chosen == _REFUSAL and rejected == _REFUSAL):
            return []
        question, docs = row["question"], row["docs"]
        row_key = (source_split, row["source"], _norm_q(question))
        common = dict(domain=f"trust/{row['source']}", user_question=question, label_source="llm")
        split = self.target_split(source_split, *row_key)
        tpl, split = pick_template(PREFERENCE, split, self.name, *row_key, row["idx"])
        out = []
        for order, (first, second) in enumerate(((chosen, rejected), (rejected, chosen))):
            state = "\n\n".join(f"Document [{i}]: {d}" for i, d in enumerate(docs, 1)) + \
                f"\n\nQuestion:\n{question}\n\nAnswer A:\n{first}\n\nAnswer B:\n{second}"
            out.append(self.record(
                key=(*row_key, row["idx"], order), split=split, state=state, type="choice", prompt=tpl.prompt,
                candidates=["answer A", "answer B"], target=order, criterion="grounded_preference",
                template=f"judge/{tpl.id}", group=f"{row['idx']}:pair", order=order, **common,
            ))
        out.append(judge_record(
            self, row_key=row_key, rec_key=(row["idx"], "answerable"), source_split=source_split, state=docs,
            templates=ANSWERABLE, values={"question": question}, target=float(chosen != _REFUSAL),
            criterion="answerable", group=f"{row['idx']}:docs", **common,
        ))
        return out


class PsiloQA(Converter):
    name = "psiloqa"
    domain = "wiki"
    family = "judge"
    license = "CC BY-4.0 (HF); Wikipedia passages CC BY-SA"
    share_alike = True
    origin = "hf:s-nlp/PsiloQA (lang = en)"
    splits = {"train": "train", "validation": "test_in"}

    def rows(self, source_split: str) -> Iterator[dict]:
        rows = ({**row, "idx": i} for i, row in enumerate(hf_rows("s-nlp/PsiloQA", None, source_split, self.streaming))
                if row.get("lang") == "en")
        yield from first_by_id(rows, self.name, source_split)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        passage, question = (row.get("wiki_passage") or "").strip(), (row.get("question") or "").strip()
        answer = row.get("llm_answer") or ""
        if not passage or not question or not answer.strip():
            return []
        labels = row.get("labels") or "[]"
        spans = [tuple(s) for s in (ast.literal_eval(labels) if isinstance(labels, str) else labels)]
        row_key = (source_split, row.get("wiki_title") or row["idx"])
        rid = str(row.get("id") or row["idx"])
        common = dict(row_key=row_key, source_split=source_split, state=passage, user_question=question,
                      label_source="llm", group=rid, llm=row.get("llm_checkpoint"))
        out = [judge_record(self, rec_key=(rid, "answer", "llm"), templates=ANSWER_CLAIM,
                            values={"question": question, "answer": answer.strip()}, target=float(not spans),
                            criterion="faithfulness", **common)]
        golden = (row.get("golden_answer") or "").strip()
        if golden and golden != answer.strip():
            out.append(judge_record(self, rec_key=(rid, "answer", "golden"), templates=ANSWER_CLAIM,
                                    values={"question": question, "answer": golden}, target=1.0,
                                    criterion="faithfulness", **common))
        sents = [(s, ok) for s, ok in sentence_labels(answer, spans) if len(s.split()) >= 3]
        if sents:
            rng = rng_for(self.name, rid)
            bad = [x for x in sents if not x[1]]
            s, ok = rng.choice(bad if bad and rng.random() < 0.5 else sents)
            out.append(judge_record(self, rec_key=(rid, "sentence"), templates=SENTENCE_CLAIM,
                                    values={"question": question, "sentence": s}, target=float(ok),
                                    criterion="faithfulness", **common))
        return out


def _overlap(question: str, paragraph: str) -> int:
    q = {w for w in re.findall(r"\w+", question.lower()) if len(w) > 3}
    return len(q & set(re.findall(r"\w+", paragraph.lower())))


class Qasper(Converter):
    name = "qasper"
    domain = "science/nlp_papers"
    family = "judge"
    license = "CC BY 4.0"
    origin = "hf:allenai/qasper"
    splits = {"train": "train", "validation": "test_in"}

    def rows(self, source_split: str) -> Iterator[dict]:
        # the repo still has a loading script, which `datasets` no longer runs: read HF's parquet export
        yield from hf_rows("allenai/qasper", "qasper", source_split, self.streaming, parquet_export=True)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        paragraphs = [p.strip() for sec in row["full_text"]["paragraphs"] for p in sec if p and len(p.split()) >= 20]
        if len(paragraphs) < 6:
            return []
        row_key = (source_split, row["id"])
        qas = row["qas"]
        parsed = []
        for qid, question, answers in zip(qas["question_id"], qas["question"], qas["answers"]):
            anns = answers["answer"]
            if not anns:
                continue
            unanswerable = {bool(a["unanswerable"]) for a in anns}
            if len(unanswerable) != 1:  # annotators disagree on answerability
                continue
            a = anns[0]
            text = (a.get("free_form_answer") or "").strip() or ", ".join(a.get("extractive_spans") or []).strip()
            if not text and a.get("yes_no") is not None:
                text = "Yes" if a["yes_no"] else "No"
            evidence = [e.strip() for e in a.get("evidence") or [] if e.strip() in paragraphs]
            parsed.append({"qid": qid, "question": question.strip(), "unanswerable": unanswerable.pop(),
                           "answer": text, "evidence": evidence, "yes_no": a.get("yes_no")})
        out = []
        for q in parsed:
            rng = rng_for(self.name, q["qid"])
            close = sorted((p for p in paragraphs if p not in q["evidence"]), key=lambda p: -_overlap(q["question"], p))
            k = rng.randint(2, 3)
            common = dict(row_key=row_key, source_split=source_split, user_question=q["question"], group=q["qid"])
            ans = dict(templates=ANSWERABLE, values={"question": q["question"]}, criterion="answerable", **common)
            if q["unanswerable"]:
                state = close[: k + rng.randint(1, 2)]   # as many chunks as an answerable state
                rng.shuffle(state)
                out.append(judge_record(self, rec_key=(q["qid"], "unanswerable"), state=state, target=0.0,
                                        label_source="human", **ans))
                continue
            if not q["evidence"] or len(q["evidence"]) > 3:
                continue
            state = q["evidence"] + close[:k]
            rng.shuffle(state)
            out.append(judge_record(self, rec_key=(q["qid"], "with_evidence"), state=state, target=1.0,
                                    label_source="human", **ans))
            # the same number of chunks with the evidence swapped for the next-closest paragraphs
            swapped = close[: k + len(q["evidence"])]
            rng.shuffle(swapped)
            out.append(judge_record(self, rec_key=(q["qid"], "without_evidence"), state=swapped, target=0.0,
                                    label_source="program", **ans))
            if not q["answer"]:
                continue
            claim = dict(templates=ANSWER_CLAIM, state=state, criterion="faithfulness_full", **common)
            out.append(judge_record(self, rec_key=(q["qid"], "answer", "gold"), values={"question": q["question"], "answer": q["answer"]},
                                    target=1.0, label_source="human", **claim))
            if q["yes_no"] is not None:
                neg, kind = ("No" if q["yes_no"] else "Yes"), "yes_no_flip"
            else:
                others = [o["answer"] for o in parsed if o["qid"] != q["qid"] and o["answer"] and o["yes_no"] is None
                          and not o["unanswerable"] and o["answer"].lower() not in q["answer"].lower()]
                neg, kind = (rng.choice(others), "other_question_answer") if others else (None, None)
            if neg:
                out.append(judge_record(self, rec_key=(q["qid"], "answer", "neg"), values={"question": q["question"], "answer": neg},
                                        target=0.0, label_source="program", perturbation=kind, **claim))
        return out


class RAGTruth(Converter):
    name = "ragtruth"
    domain = "news/cnn_dm"
    family = "judge"
    license = "MIT (GitHub); CNN/DM articles only"
    origin = "github:ParticleMedia/RAGTruth (dataset), task Summary, source CNN/DM"
    splits = {"train": "train", "test": "test_in"}
    _URL = "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/main/dataset/{name}.jsonl"

    def rows(self, source_split: str) -> Iterator[dict]:
        sources = {}
        with fetch(self._URL.format(name="source_info"), "ragtruth/source_info.jsonl").open() as f:
            for line in f:
                s = json.loads(line)
                if s["task_type"] == "Summary" and s["source"] == "CNN/DM":
                    sources[s["source_id"]] = s
        with fetch(self._URL.format(name="response"), "ragtruth/response.jsonl").open() as f:
            for line in f:
                r = json.loads(line)
                if r["split"] == source_split and r["source_id"] in sources:
                    yield {**r, "article": sources[r["source_id"]]["source_info"]}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        article, summary = str(row.get("article") or "").strip(), (row.get("response") or "").strip()
        if not article or not summary:
            return []
        spans = [(lab["start"], lab["end"]) for lab in row.get("labels") or []]
        types = sorted({lab["label_type"] for lab in row.get("labels") or []})
        row_key = (source_split, row["source_id"])
        rid = str(row["id"])
        common = dict(row_key=row_key, source_split=source_split, state=article, criterion="faithfulness",
                      group=rid, label_source="human", summarizer=row.get("model"))
        out = [judge_record(self, rec_key=(rid, "summary"), templates=SUMMARY_CLAIM, values={"summary": summary},
                            target=float(not spans), hallucination_types=types, **common)]
        sents = [(s, ok) for s, ok in sentence_labels(summary, spans) if len(s.split()) >= 4]
        if sents:
            rng = rng_for(self.name, rid)
            bad = [x for x in sents if not x[1]]
            s, ok = rng.choice(bad if bad and rng.random() < 0.5 else sents)
            out.append(judge_record(self, rec_key=(rid, "sentence"), templates=SUMMARY_SENTENCE,
                                    values={"sentence": s}, target=float(ok), **common))
        return out
