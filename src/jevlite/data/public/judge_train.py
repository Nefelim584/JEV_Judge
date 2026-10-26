"""Judge training data from public sources, built by code (todo Phase 4b, ``DATASETS.md`` 8.1–8.2).

Stage B trains the judge criteria on public data instead of generated cases (decided 2026-10-26):

- **RAGBench** train/validation of 7 subsets (``ragbench_train``): real LLM answers over several
  documents with GPT-4 sentence-support keys. Per row: the whole answer against all documents, one
  answer sentence against all documents, and that sentence against one document. The other 5 subsets
  stay out (held-out slice of eval v2, ``docs/judge_bench/eval_v2``).
- **HotpotQA**, distractor setting (``hotpotqa_judge``): the 2 gold paragraphs plus 2–4 distractors
  are the chunks. The gold answer is supported only by the chunks together; a negative answer swaps
  in the other compared entity, a distractor's title or a distractor's number. Answerability: the
  same chunks (answerable) and the same chunks with the answer's paragraph replaced by another
  distractor (not answerable), so both labels come from one paragraph pool (``DATASETS.md`` 5.1).
- **TAT-QA** and **FinQA** (``tatqa_judge``, ``finqa_judge``): table + text, the gold answer (often
  computed) as the positive; the negative takes another number of the **same** table or text, a
  shifted value or a wrong scale, so "derived from the table" counts as supported and a number that
  merely occurs in the table does not.

Wording. Training does not build claim groups yet, so the user's question goes into every claim. The
templates avoid the eval wording (``judge_bench.RAG_CLAIM``), so Stage B gains on the judge benchmarks
are not a template effect. Records carry ``criterion`` (``faithfulness``: one chunk,
``faithfulness_full``: all chunks, ``answerable``), ``user_question``, ``group`` (records on the same
state) and ``label_source`` (``human``, ``llm`` or ``program``; ``perturbation`` names the edit).

The split is drawn per source document (RAGBench question, HotpotQA question, TAT-QA table, FinQA
page), so the records of one state never straddle train and calib.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any, Iterable, Iterator, Sequence

from ...schema import State
from ..unified import Record
from .base import ANSWERABLE, Converter, Template, fetch, hf_rows, pick_template, rng_for
from .factcheck import clean_abstract

ANSWER_CLAIM = (
    Template("asked_answered", 'Asked "{question}", the assistant answered: {answer}'),
    Template("qa_pair", "Question: {question}\nAnswer: {answer}"),
    Template("per_documents", 'According to the documents, the answer to "{question}" is: {answer}'),
    Template("backed_answer", 'The text backs this answer to "{question}": {answer}'),  # held out
)
SENTENCE_CLAIM = (
    Template("wrote", 'Answering "{question}", the assistant wrote: {sentence}'),
    Template("in_reply", "{sentence} (said in reply to: {question})"),
    Template("backed_sentence", 'In an answer to "{question}", the text supports the sentence: {sentence}'),  # held out
)


def judge_record(
    conv: Converter, *, row_key: Sequence[Any], rec_key: Sequence[Any], source_split: str, state: State,
    templates: Sequence[Template], values: dict[str, str], target: float, criterion: str, **fields: Any,
) -> Record:
    """A Bool judge record. The split comes from ``row_key`` (the source document), the template and
    the held-out move from ``rec_key`` without its last element, which names the variant: the gold and
    the negative answer of a pair share their wording, so the template is no cue to the label."""
    split = conv.target_split(source_split, *row_key)
    tpl, split = pick_template(templates, split, conv.name, *row_key, *rec_key[:-1])
    return conv.record(
        key=(*row_key, *rec_key), split=split, state=state, type="bool", prompt=tpl.prompt.format(**values),
        target=float(target), criterion=criterion, template=f"judge/{tpl.id}", **fields,
    )


# ---------------------------------------------------------------------------------------------------
# Numbers: negatives that look like the answer


_NUMBER = re.compile(r"[-−]?\$?\(?\d[\d,]*(?:\.\d+)?\)?%?")


def parse_number(text: str) -> float | None:
    """``$1,234.5`` → 1234.5, ``(5,580)`` → -5580, ``14.1%`` → 14.1; None when it is not a number."""
    s = str(text).strip().replace("−", "-")
    m = _NUMBER.fullmatch(s)
    if not m:
        return None
    neg = s.startswith("-") or (s.startswith("(") and s.endswith(")")) or s.startswith("$(")
    digits = re.sub(r"[^\d.]", "", s)
    try:
        value = float(digits)
    except ValueError:
        return None
    return -value if neg else value


def numbers_in(texts: Iterable[str]) -> list[str]:
    return [m.group(0).strip("()") for t in texts for m in _NUMBER.finditer(str(t)) if parse_number(m.group(0)) is not None]


def format_like(value: float, like: str) -> str:
    """``value`` written the way ``like`` is: decimals, thousands separators, ``$`` and ``%``."""
    like = str(like).strip()
    decimals = len(like.split(".", 1)[1].rstrip("%)")) if "." in like else 0
    body = f"{abs(value):,.{decimals}f}" if "," in like else f"{abs(value):.{decimals}f}"
    return ("-" if value < 0 else "") + ("$" if "$" in like else "") + body + ("%" if like.endswith("%") else "")


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= max(1e-9, 0.005 * max(abs(a), abs(b)))


def numeric_negative(rng: random.Random, answer: str, pool: Sequence[str]) -> tuple[str, str] | None:
    """A wrong number written like ``answer``: another value of the document (``other_value``, the hard
    case) or the answer shifted (``shift``). None when ``answer`` is not a number."""
    a = parse_number(answer)
    if a is None:
        return None
    if re.fullmatch(r"(?:1[0-9]|20)\d\d", answer.strip()):   # a year: another year, not a multiple
        years = [p for p in pool if re.fullmatch(r"(?:1[0-9]|20)\d\d", p) and p != answer.strip()]
        if years and rng.random() < 0.6:
            return rng.choice(years), "other_value"
        return str(int(a) + rng.choice([-10, -5, -3, -2, -1, 1, 2, 3, 5])), "shift"
    others = []
    for p in pool:
        v = parse_number(p)
        if v is not None and not _close(v, a) and format_like(v, answer) != answer:
            others.append(v)
    if others and rng.random() < 0.6:
        return format_like(rng.choice(others), answer), "other_value"
    for _ in range(20):
        v = a * rng.choice([0.5, 0.8, 1.25, 1.5, 2.0]) if abs(a) >= 1 else a + rng.choice([-0.3, -0.1, 0.1, 0.2, 0.5])
        if not _close(v, a) and format_like(v, answer) != answer:
            return format_like(v, answer), "shift"
    return None


def grid_to_markdown(rows: Sequence[Sequence[Any]]) -> str:
    rows = [[str(c).strip().replace("\n", " ") for c in r] for r in rows if any(str(c).strip() for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(out)


def _row_labels(table: Sequence[Sequence[Any]]) -> list[str]:
    return [str(r[0]).strip() for r in table[1:] if r and str(r[0]).strip() and parse_number(r[0]) is None]


# ---------------------------------------------------------------------------------------------------
# RAGBench, train / validation


_DOC_KEY = re.compile(r"(\d+)[a-z]+\.?")


def ragbench_sentences(row: dict) -> list[dict]:
    """Answer sentences with a usable label: ``supported`` (not in the unsupported keys, backed by
    document sentences) or not; ``docs`` = indices of the documents its support keys point to.
    Sentences backed only by "general" / "well_known_fact" keys, and contradictory annotations, are
    left out."""
    unsupported = set(row.get("unsupported_response_sentence_keys") or [])
    texts = {k: t for k, t in row.get("response_sentences") or []}
    out = []
    for info in row.get("sentence_support_information") or []:
        key = info.get("response_sentence_key")
        text = (texts.get(key) or "").strip()
        if not text or len(text.split()) < 3:
            continue
        docs = sorted({int(m.group(1)) for k in info.get("supporting_sentence_keys") or [] if (m := _DOC_KEY.fullmatch(k))})
        fully = info.get("fully_supported")
        if key in unsupported:
            if fully is True:
                continue
            out.append({"key": key, "text": text, "supported": False, "docs": docs})
        elif fully is not False and docs:
            out.append({"key": key, "text": text, "supported": True, "docs": docs})
    return out


class RAGBenchTrain(Converter):
    name = "ragbench_train"
    domain = "ragbench"
    family = "judge"
    license = "CC BY 4.0 (sources: HotpotQA CC BY-SA 4.0, FinQA, TAT-QA, HAGRID, ExpertQA, TechQA, PubMedQA); LLM answers, GPT-4 labels"
    origin = "hf:galileo-ai/ragbench, train + validation of 7 subsets"
    SUBSETS = ("hotpotqa", "finqa", "tatqa", "hagrid", "expertqa", "techqa", "pubmedqa")
    share_alike_domains = frozenset({"ragbench/hotpotqa"})
    splits = {"train": "train", "validation": "test_in"}

    def rows(self, source_split: str) -> Iterator[dict]:
        for subset in self.SUBSETS:
            for i, row in enumerate(hf_rows("galileo-ai/ragbench", subset, source_split, self.streaming)):
                yield {**row, "subset": subset, "idx": i}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        docs = {i: d.strip() for i, d in enumerate(row.get("documents") or []) if d and d.strip()}
        question, response = (row.get("question") or "").strip(), (row.get("response") or "").strip()
        if not docs or not question or not response:
            return []
        state = list(docs.values())
        # one response per (question, model): split by question, so all its responses land together
        qid = str(row.get("id") or row["idx"]).rsplit("_", 1)[0]
        row_key = (source_split, row["subset"], qid)
        rid = str(row.get("id") or row["idx"])
        rng = rng_for(self.name, rid)
        common = dict(domain=f"ragbench/{row['subset']}", user_question=question, label_source="llm",
                      generation_model=row.get("generation_model_name"))
        out = []
        if row.get("adherence_score") is not None:
            out.append(judge_record(
                self, row_key=row_key, rec_key=(rid, "answer"), source_split=source_split, state=state,
                templates=ANSWER_CLAIM, values={"question": question, "answer": response},
                target=float(bool(row["adherence_score"])), criterion="faithfulness_full", group=f"{rid}:all", **common,
            ))
        sentences = [s for s in ragbench_sentences(row) if all(d in docs for d in s["docs"])]
        if not sentences:
            return out
        negatives = [s for s in sentences if not s["supported"]]
        s = rng.choice(negatives if negatives and rng.random() < 0.5 else sentences)
        values = {"question": question, "sentence": s["text"]}
        out.append(judge_record(
            self, row_key=row_key, rec_key=(rid, s["key"], "all"), source_split=source_split, state=state,
            templates=SENTENCE_CLAIM, values=values, target=float(s["supported"]), criterion="faithfulness_full",
            group=f"{rid}:all", sentence_key=s["key"], **common,
        ))
        # the sentence against one document: its own (supported) or another (not)
        others = [d for d in docs if d not in s["docs"]]
        if s["supported"] and len(s["docs"]) == 1 and (not others or rng.random() < 0.5):
            doc, target = s["docs"][0], 1.0
        elif others:
            doc, target = rng.choice(others), 0.0
        else:
            return out
        out.append(judge_record(
            self, row_key=row_key, rec_key=(rid, s["key"], f"doc{doc}"), source_split=source_split, state=docs[doc],
            templates=SENTENCE_CLAIM, values=values, target=target, criterion="faithfulness",
            group=f"{rid}:doc{doc}", sentence_key=s["key"], **common,
        ))
        return out


# ---------------------------------------------------------------------------------------------------
# HotpotQA, distractor setting


_TOKEN = r"(?:[A-Z](?:\.[A-Z])+\.?|[A-Z][\w'&-]*)"     # "F.E.A.R." or "Leeds"; a full stop ends a name
_NAME = re.compile(rf"\b{_TOKEN}(?:\s+(?:of|the|de|von|van|and)?\s*{_TOKEN})*")
_STOP = frozenset("A An The In On At By For From Of To It Its He She His Her They Their There This That These Those "
                  "After Before During While When Where Which Who".split())


def names_in(text: str) -> list[str]:
    """Capitalised names of ``text``, without sentence-initial words (capitalised for another reason)
    and single letters."""
    out = []
    for m in _NAME.finditer(text):
        before = text[: m.start()].rstrip()
        if not before or before.endswith((".", "!", "?", ":", '"')) or len(m.group(0)) < 3 or m.group(0).split()[0] in _STOP:
            continue
        out.append(m.group(0))
    return out


def hotpot_negative(rng: random.Random, row: dict, gold: Sequence[str], distractors: Sequence[str],
                    paras: dict[str, str]) -> tuple[str, str] | None:
    answer = row["answer"].strip()
    low = answer.lower()
    if low in ("yes", "no"):
        return ("no" if low == "yes" else "yes"), "yes_no_flip"
    if row.get("type") == "comparison" and any(t.lower() == low for t in gold):
        return next(t for t in gold if t.lower() != low), "other_entity"   # the other compared entity
    if any(ch.isdigit() for ch in answer):
        neg = numeric_negative(rng, answer, numbers_in(paras[t] for t in distractors))
        if neg:
            return neg[0], "number_from_distractor" if neg[1] == "other_value" else neg[1]
        return None
    # a name with as many words as the answer, from the gold paragraphs first (present in the evidence,
    # wrong role), then the distractors; distractor titles only as a last resort (often another type)
    n = len(answer.split())
    for source, kind in ((gold, "gold_entity"), (distractors, "distractor_entity")):
        names = [m for t in source for m in names_in(paras[t])
                 if len(m.split()) == n and m.lower() not in low and low not in m.lower()]
        if names:
            return rng.choice(sorted(set(names))), kind
    titles = [t for t in distractors if low not in t.lower() and t.lower() not in low]
    return (rng.choice(titles), "distractor_title") if titles else None


class HotpotQAJudge(Converter):
    name = "hotpotqa_judge"
    domain = "wiki"
    family = "judge"
    license = "CC BY-SA 4.0"
    share_alike = True
    origin = "hf:hotpotqa/hotpot_qa (distractor)"
    splits = {"train": "train", "validation": "test_in"}

    def rows(self, source_split: str) -> Iterator[dict]:
        yield from hf_rows("hotpotqa/hotpot_qa", "distractor", source_split, self.streaming)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        ctx = row["context"]
        paras = {t: clean_abstract(" ".join(s)) for t, s in zip(ctx["title"], ctx["sentences"])}
        paras = {t: p for t, p in paras.items() if p}
        gold = [t for t in dict.fromkeys(row["supporting_facts"]["title"]) if t in paras]
        question, answer = row["question"].strip(), row["answer"].strip()
        if len(gold) < 2 or not question or not answer:
            return []
        rng = rng_for(self.name, row["id"])
        distractors = [t for t in paras if t not in gold]
        picked = rng.sample(distractors, min(len(distractors), rng.randint(2, 4) + 1))
        spare, chosen = (picked[-1], picked[:-1]) if len(picked) >= 3 else (None, picked)
        titles = gold + chosen
        rng.shuffle(titles)
        state = [f"{t}\n{paras[t]}" for t in titles]
        row_key = (source_split, row["id"])
        common = dict(user_question=question, hotpot_type=row.get("type"), level=row.get("level"))
        claim = dict(row_key=row_key, source_split=source_split, state=state, templates=ANSWER_CLAIM,
                     criterion="faithfulness_full", group=f"{row['id']}:main", **common)
        out = [judge_record(self, rec_key=("answer", "gold"), values={"question": question, "answer": answer},
                            target=1.0, label_source="human", **claim)]
        neg = hotpot_negative(rng, row, gold, chosen, paras)
        if neg:
            out.append(judge_record(self, rec_key=("answer", "neg"), values={"question": question, "answer": neg[0]},
                                    target=0.0, label_source="program", perturbation=neg[1], **claim))
        ans = dict(row_key=row_key, source_split=source_split, templates=ANSWERABLE, values={"question": question},
                   criterion="answerable", label_source="program", **common)
        out.append(judge_record(self, rec_key=("answerable", "full"), state=state, target=1.0, group=f"{row['id']}:main", **ans))
        if spare is not None:
            # replace the paragraph that holds the answer (else a random gold one) with another distractor
            holding = [t for t in gold if answer.lower() in paras[t].lower()]
            removed = holding[0] if holding else rng.choice(gold)
            reduced = [f"{spare}\n{paras[spare]}" if t == removed else f"{t}\n{paras[t]}" for t in titles]
            out.append(judge_record(self, rec_key=("answerable", "minus_gold"), state=reduced, target=0.0,
                                    group=f"{row['id']}:minus_gold", removed=removed, **ans))
        return out


# ---------------------------------------------------------------------------------------------------
# TAT-QA and FinQA: answers computed from tables


_SCALES = ("thousand", "million", "billion")


def tatqa_answer(answer: Any) -> str:
    if isinstance(answer, list):
        parts = [str(a).strip() for a in answer if str(a).strip()]
        if not parts:
            return ""
        return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]
    if isinstance(answer, float):
        return f"{answer:.4f}".rstrip("0").rstrip(".")
    return str(answer).strip()


def with_scale(text: str, scale: str) -> str:
    if scale == "percent":
        return text if text.endswith("%") else f"{text}%"
    if scale in _SCALES:
        return f"{text} {scale}"
    return text


class TATQAJudge(Converter):
    name = "tatqa_judge"
    domain = "finance/tatqa"
    family = "judge"
    license = "CC BY 4.0"
    origin = "github:NExTplusplus/TAT-QA (dataset_raw)"
    splits = {"train": "train", "dev": "test_in"}
    _URL = "https://raw.githubusercontent.com/NExTplusplus/TAT-QA/master/dataset_raw/tatqa_dataset_{split}.json"

    def rows(self, source_split: str) -> Iterator[dict]:
        path = fetch(self._URL.format(split=source_split), f"tatqa/tatqa_dataset_{source_split}.json")
        for doc in json.loads(path.read_text()):
            paragraphs = [p["text"].strip() for p in sorted(doc["paragraphs"], key=lambda p: p["order"]) if p["text"].strip()]
            for q in doc["questions"]:
                yield {**q, "table": doc["table"]["table"], "table_uid": doc["table"]["uid"], "paragraphs": paragraphs}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        question = row["question"].strip()
        base = tatqa_answer(row["answer"])
        if not question or not base:
            return []
        scale = row.get("scale") or ""
        table = grid_to_markdown(row["table"])
        state = [x for x in (table, "\n".join(row["paragraphs"])) if x]
        rng = rng_for(self.name, row["uid"])
        neg = None
        if row.get("answer_type") in ("arithmetic", "count") or (row.get("answer_type") == "span" and parse_number(base) is not None):
            options = []
            num = numeric_negative(rng, base, numbers_in([c for r in row["table"] for c in r] + row["paragraphs"]))
            if num:
                options.append((with_scale(num[0], scale), num[1]))
            if scale in _SCALES:
                options.append((with_scale(base, rng.choice([s for s in _SCALES if s != scale])), "scale"))
            neg = rng.choice(options) if options else None
        else:
            labels = [x for x in _row_labels(row["table"]) if x.lower() not in base.lower()]
            if labels and row.get("answer_from") in ("table", "table-text"):
                if row.get("answer_type") == "multi-span" and isinstance(row["answer"], list) and len(row["answer"]) > 1:
                    parts = list(row["answer"])
                    parts[rng.randrange(len(parts))] = rng.choice(labels)
                    neg = (tatqa_answer(parts), "other_span")
                else:
                    neg = (rng.choice(labels), "other_span")
        claim = dict(row_key=(source_split, row["table_uid"]), source_split=source_split, state=state,
                     templates=ANSWER_CLAIM, criterion="faithfulness_full", group=f"{row['uid']}", user_question=question,
                     answer_type=row.get("answer_type"), answer_from=row.get("answer_from"))
        out = [judge_record(self, rec_key=(row["uid"], "gold"), values={"question": question, "answer": with_scale(base, scale)},
                            target=1.0, label_source="human", **claim)]
        if neg:
            out.append(judge_record(self, rec_key=(row["uid"], "neg"), values={"question": question, "answer": neg[0]},
                                    target=0.0, label_source="program", perturbation=neg[1], **claim))
        return out


def clean_finqa(sentence: str) -> str:
    """FinQA text is lower-cased and tokenised: ``2.05% ( 2.05 % )``, ``$ 3.8 million``, ``2009 ,``."""
    s = re.sub(r"\s*\(\s*-?[\d.,]+\s*%\s*\)", "", sentence)
    s = re.sub(r"\$\s+(?=\d)", "$", s)
    s = re.sub(r"\s+([,.;:%)])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def finqa_text(sentences: Sequence[str], gold: set[int], budget: int = 1500) -> str:
    """The gold sentences plus their nearest neighbours, in document order, up to ``budget`` characters."""
    keep = [i for i in sorted(gold) if i < len(sentences)]
    size = sum(len(sentences[i]) for i in keep)
    centre = keep or [0]
    for i in sorted(range(len(sentences)), key=lambda i: min(abs(i - c) for c in centre)):
        if i in keep:
            continue
        if size + len(sentences[i]) > budget:
            break
        keep.append(i)
        size += len(sentences[i])
    return " ".join(sentences[i] for i in sorted(keep))


class FinQAJudge(Converter):
    name = "finqa_judge"
    domain = "finance/finqa"
    family = "judge"
    license = "MIT (GitHub) / CC BY 4.0 (HF)"
    origin = "github:czyssrs/FinQA (dataset)"
    splits = {"train": "train", "dev": "test_in"}
    _URL = "https://raw.githubusercontent.com/czyssrs/FinQA/main/dataset/{split}.json"

    def rows(self, source_split: str) -> Iterator[dict]:
        path = fetch(self._URL.format(split=source_split), f"finqa/{source_split}.json")
        for i, ex in enumerate(json.loads(path.read_text())):
            yield {**ex, "idx": i}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        qa = row.get("qa") or {}
        question = (qa.get("question") or "").strip()
        answer = str(qa.get("answer") or "").strip()
        if not answer and isinstance(qa.get("exe_ans"), (int, float)):
            answer = f"{qa['exe_ans']:.4f}".rstrip("0").rstrip(".")
        if not question or not answer:
            return []
        sentences = [c for c in (clean_finqa(s) for s in row.get("pre_text", []) + row.get("post_text", [])) if len(c) > 2]
        raw = row.get("pre_text", []) + row.get("post_text", [])
        # gold_inds keys: text_<i> over pre_text + post_text; map raw indices onto the cleaned list
        kept = [i for i, s in enumerate(raw) if len(clean_finqa(s)) > 2]
        pos = {r: k for k, r in enumerate(kept)}
        gold = {pos[int(k.split("_")[1])] for k in (qa.get("gold_inds") or {}) if k.startswith("text_") and int(k.split("_")[1]) in pos}
        table_rows = row.get("table_ori") or row.get("table") or []
        state = [x for x in (grid_to_markdown(table_rows), finqa_text(sentences, gold)) if x]
        rng = rng_for(self.name, row.get("id", row["idx"]))
        if answer.lower() in ("yes", "no"):
            neg = ("no" if answer.lower() == "yes" else "yes", "yes_no_flip")
        else:
            neg = numeric_negative(rng, answer, numbers_in([c for r in table_rows for c in r] + sentences))
        claim = dict(row_key=(source_split, row.get("filename", row["idx"])), source_split=source_split, state=state,
                     templates=ANSWER_CLAIM, criterion="faithfulness_full", group=str(row.get("id", row["idx"])),
                     user_question=question)
        rid = str(row.get("id", row["idx"]))
        out = [judge_record(self, rec_key=(rid, "gold"), values={"question": question, "answer": answer},
                            target=1.0, label_source="human", **claim)]
        if neg:
            out.append(judge_record(self, rec_key=(rid, "neg"), values={"question": question, "answer": neg[0]},
                                    target=0.0, label_source="program", perturbation=neg[1], **claim))
        return out


# ---------------------------------------------------------------------------------------------------
# ClapNQ: answerability without the passage shortcut, relevance, conciseness and completeness


RELEVANCE_LEVELS = ["does not address", "partially addresses", "mostly addresses", "fully addresses"]
RELEVANCE = (
    Template("how_well", "How well does the answer address the user's question?"),
    Template("covers", "To what degree does the answer respond to what was asked?"),
    Template("on_point", "How directly does the answer deal with the question?"),  # held out
)
PAIR_CHOICE = {
    "conciseness": (
        Template("to_the_point", "Which answer is more to the point, with no material the question did not ask for?"),
        Template("less_padding", "Which answer gives the same information with less padding?"),
        Template("tighter", "Which answer stays closer to what was asked?"),  # held out
    ),
    "completeness": (
        Template("everything", "Which answer gives everything from the document that the question asks for?"),
        Template("fuller", "Which answer leaves out less of what the question needs from the document?"),
        Template("whole", "Which answer covers the question in full, using the document?"),  # held out
    ),
}


class ClapNQJudge(Converter):
    """ClapNQ rebuilt for Stage B (``DATASETS.md`` 5.1, 8.1). Only the answerable file is used: every
    passage gets its own question (answerable) and the question of a passage from another page (not
    answerable), so the label is a property of the pair, not of the passage. The long-form answers give
    relevance (own answer → fully, the other page's answer → does not address, own answer to the two
    questions joined → partially) and pairwise conciseness (answer vs answer + extra passage sentences)
    and completeness (answer vs answer without its last sentence), so that length alone never decides.
    The old ``clapnq`` converter stays as it is for the Stage A mixes."""

    name = "clapnq_judge"
    domain = "wiki"
    family = "judge"
    license = "Apache-2.0 (NQ text: CC BY-SA 3.0)"
    share_alike = True
    origin = "https://github.com/primeqa/clapnq (answerable files only)"
    splits = {"train": "train", "dev": "test_in"}
    _URL = "https://raw.githubusercontent.com/primeqa/clapnq/main/annotated_data/{split}/clapnq_{split}_answerable.jsonl"

    def rows(self, source_split: str) -> Iterator[dict]:
        path = fetch(self._URL.format(split=source_split))
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        rows = [r for r in rows if r.get("passages") and any(o.get("answer") for o in r["output"])]
        for i, row in enumerate(rows):
            title = row["passages"][0]["title"]
            rng = rng_for(self.name, "partner", row["id"])
            for _ in range(50):  # a partner from another page
                other = rows[rng.randrange(len(rows))]
                if other["passages"][0]["title"] != title:
                    break
            else:
                continue
            yield {**row, "partner": {"id": other["id"], "question": other["input"],
                                      "answer": next(o["answer"] for o in other["output"] if o.get("answer"))}}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        from ...judge.claims import split_sentences

        passage = f"{row['passages'][0]['title']}\n{row['passages'][0]['text']}".strip()
        question = row["input"].strip()
        answer = next((o["answer"].strip() for o in row["output"] if o.get("answer")), "")
        partner = row["partner"]
        if not question or not answer:
            return []
        rid = row["id"]
        row_key = (source_split, rid)
        rng = rng_for(self.name, rid)
        out = []
        # answerability: the same passage with its own question and with another page's question
        for variant, (q, target) in (("own", (question, 1.0)), ("other", (partner["question"], 0.0))):
            out.append(judge_record(
                self, row_key=row_key, rec_key=("answerable", variant), source_split=source_split, state=passage,
                templates=ANSWERABLE, values={"question": q.strip()}, target=target, criterion="answerable",
                group=f"{rid}:passage", user_question=q.strip(), label_source="human" if target else "program",
            ))
        # relevance: Score on (question, answer), whatever the documents say
        joined = f"{question.rstrip('?')}, and {partner['question'].strip().rstrip('?')}?"
        for variant, (q, a, level) in (("own", (question, answer, 3)), ("other", (question, partner["answer"].strip(), 0)),
                                       ("joined", (joined, answer, 1))):
            split = self.target_split(source_split, *row_key)
            tpl, split = pick_template(RELEVANCE, split, self.name, *row_key, "relevance")
            out.append(self.record(
                key=(*row_key, "relevance", variant), split=split, state={"question": q, "answer": a}, type="score",
                prompt=tpl.prompt, candidates=RELEVANCE_LEVELS, target=level, criterion="relevance",
                template=f"judge/{tpl.id}", user_question=q, label_source="human" if variant == "own" else "program",
                group=f"{rid}:relevance",
            ))
        # pairs: conciseness (the bare answer wins) and completeness (the whole answer wins)
        answer_sents = split_sentences(answer)
        extra = [s for s in split_sentences(row["passages"][0]["text"]) if s not in answer and len(s.split()) >= 5]
        pairs = {}
        if extra:
            padded = answer + " " + " ".join(rng.sample(extra, min(len(extra), rng.randint(1, 2))))
            pairs["conciseness"] = (answer, padded)
        if len(answer_sents) >= 2:
            pairs["completeness"] = (answer, " ".join(answer_sents[:-1]))
        for criterion, (better, worse) in pairs.items():
            split = self.target_split(source_split, *row_key)
            tpl, split = pick_template(PAIR_CHOICE[criterion], split, self.name, *row_key, criterion)
            for order, (first, second) in enumerate(((better, worse), (worse, better))):
                state = f"Document:\n{passage}\n\nQuestion:\n{question}\n\nAnswer A:\n{first}\n\nAnswer B:\n{second}"
                out.append(self.record(
                    key=(*row_key, criterion, order), split=split, state=state, type="choice", prompt=tpl.prompt,
                    candidates=["answer A", "answer B"], target=order, criterion=criterion, template=f"judge/{tpl.id}",
                    group=f"{rid}:{criterion}", order=order, user_question=question, label_source="program",
                ))
        return out
