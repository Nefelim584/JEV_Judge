"""Question answering over a passage: yes/no questions and answerability (``DATASETS.md`` section 0)."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Iterable, Iterator

from ...log import logger
from ..unified import Record
from .base import Converter, Template, answerable_record, fetch, hf_rows, pick_template, raw_dir, rng_for

BOOLQ = (
    Template("question", "{question}?"),
    Template("answer_yes", "The answer to \"{question}?\" is yes, according to the text."),
)


class BoolQ(Converter):
    name = "boolq"
    domain = "wiki"
    family = "yes_no_qa"
    license = "CC BY-SA 3.0"
    share_alike = True
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
    share_alike = True
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
    share_alike = True
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


def _columns(x) -> dict:
    """A nested field as a dict of lists, whether Arrow stored it as a struct of lists or as a list of
    structs."""
    if isinstance(x, list):
        return {k: [d[k] for d in x] for k in (x[0] if x else {})}
    return x


def join_tokens(tokens: list[str]) -> str:
    """NQ's whitespace tokens back to text.

    No space before closing punctuation or after opening; PTB quotes (````…''``) and ``"`` attach to
    the quoted words (opening and closing alternate); a lone ``'`` (``1950s '``) and ``-`` inside a
    word (``code - named``) attach to their neighbours; ``--`` is an en dash.
    """
    out: list[str] = []
    glue_next, in_quote = False, False
    for t in tokens:
        if t in ("``", "''", '"'):
            in_quote = t == "``" or (t == '"' and not in_quote)
            if in_quote:
                if out and glue_next:
                    out[-1] += '"'
                else:
                    out.append('"')
                glue_next = True
            else:
                out[-1:] = [(out[-1] if out else "") + '"']
            continue
        if t == "--":
            t = "–"
        if out and (glue_next or t in ("'", "-") or re.match(r"^([,.;:!?%)\]}]|n't|'s|'re|'ve|'m|'ll|'d)$", t)):
            out[-1] += t
        else:
            out.append(t)
        glue_next = t in ("(", "[", "{", "$", "-")
    return " ".join(out).strip()


def nq_example(row: dict, *key, first_candidates: int = 10, min_words: int = 8) -> dict | None:
    """One NQ row (HF parquet layout) as ``{id, question, title, passage, answerable}``, or None.

    Answerable: a long answer from at least 1 of 1 (train) or 2 of 5 (dev) annotators, the official
    NQ rule; the passage is the most chosen candidate. Unanswerable: no annotator found a long answer
    anywhere on the page, so any paragraph is a true negative; one of the first ``first_candidates``
    top-level paragraphs is drawn, which keeps it on the page's topic. Dev rows with a single
    long answer out of 5 are ambiguous and dropped. Only paragraph (``<P>``) passages are kept:
    tables and lists lose their structure without the HTML.
    """
    tokens = _columns(row["document"]["tokens"])
    words, is_html = tokens["token"], tokens["is_html"]
    cands = _columns(row["long_answer_candidates"])
    longs = _columns(row["annotations"])["long_answer"]
    chosen = [la["candidate_index"] for la in longs if la["candidate_index"] >= 0]

    def is_paragraph(c: int) -> bool:
        return words[cands["start_token"][c]] == "<P>"

    if chosen:
        if len(chosen) < (2 if len(longs) > 1 else 1):
            return None
        cand, answerable = Counter(chosen).most_common(1)[0][0], True
    else:
        pool = [c for c in range(min(first_candidates, len(cands["start_token"]))) if cands["top_level"][c] and is_paragraph(c)]
        if not pool:
            return None
        cand, answerable = rng_for("nq-negative", *key).choice(pool), False
    if not is_paragraph(cand):
        return None
    span = range(cands["start_token"][cand], cands["end_token"][cand])
    passage = join_tokens([words[i] for i in span if not is_html[i]])
    if len(passage.split()) < min_words:
        return None
    return {"id": row["id"], "question": row["question"]["text"], "title": row["document"]["title"], "passage": passage, "answerable": answerable}


class NaturalQuestions(Converter):
    """Real Google queries against a Wikipedia page: does a paragraph of the page answer the question?

    Read from HF's parquet copy (``storage.googleapis.com`` refuses anonymous downloads now). The
    train split is 287 files / 55 GB; the first ``TRAIN_FILES`` (~1.1k questions each) cover the 30k
    cap of ``DATASETS.md`` after filtering. Dev (7 files, 1.3 GB) is read whole.

    ``JEVLITE_DROP_RAW=1`` deletes every parquet file once it is read, so the peak disk use is one
    file (~0.2 GB) instead of 7 GB (Kaggle).
    """

    name = "nq"
    domain = "wiki"
    family = "answerable"
    license = "CC BY-SA 3.0"
    share_alike = True
    origin = "hf:google-research-datasets/natural_questions (parquet)"
    splits = {"train": "train", "validation": "test_in"}

    TRAIN_FILES = 30
    _REPO = "google-research-datasets/natural_questions"

    def _files(self, source_split: str) -> list[str]:
        if source_split == "train":
            names = [f"default/train-{i:05d}-of-00287.parquet" for i in range(self.TRAIN_FILES)]
        else:
            names = [f"dev/validation-{i:05d}-of-00007.parquet" for i in range(7)]
        return names[:1] if self.streaming else names

    def rows(self, source_split: str) -> Iterator[dict]:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download

        kept = dropped = 0
        for name in self._files(source_split):
            path = hf_hub_download(self._REPO, name, repo_type="dataset", local_dir=raw_dir() / "nq")
            columns = ["id", "document", "question", "long_answer_candidates", "annotations"]
            for batch in pq.ParquetFile(path).iter_batches(batch_size=64, columns=columns):
                for row in batch.to_pylist():
                    example = nq_example(row, self.name, source_split, row["id"])
                    if example is None:
                        dropped += 1
                        continue
                    kept += 1
                    yield example
            if os.environ.get("JEVLITE_DROP_RAW") == "1":
                os.remove(path)
        logger.info("[nq] {}: {} kept, {} dropped (non-paragraph, ambiguous or short)", source_split, kept, dropped)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        return [
            answerable_record(
                self, key=(source_split, row["id"]), source_split=source_split,
                state=f"{row['title']}\n{row['passage']}", question=row["question"], answerable=row["answerable"],
            )
        ]
