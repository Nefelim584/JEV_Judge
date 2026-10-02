"""Claim vs evidence sets with paragraph- or document-length states (``DATASETS.md`` section 0).

FEVER, HoVer and FEVEROUS are not here yet: their evidence text lives in separate Wikipedia dumps.
"""

from __future__ import annotations

import json
import zipfile
from typing import Iterable, Iterator, Sequence

from ..unified import Record
from .base import CONTRADICTED, ENTAILED, NOT_MENTIONED, Converter, fetch, hf_rows, nli_records, rng_for

_FEVER_LABELS = {"SUPPORTS": ENTAILED, "REFUTES": CONTRADICTED, "NOT ENOUGH INFO": NOT_MENTIONED}


class VitaminC(Converter):
    """Contrastive Wikipedia revisions: the same claim against slightly different evidence."""

    name = "vitaminc"
    domain = "wiki"
    family = "factcheck"
    license = "CC BY-SA 3.0"
    origin = "hf:tals/vitaminc"
    splits = {"train": "train", "test": "test_in"}

    def rows(self, source_split: str) -> Iterable[dict]:
        return hf_rows("tals/vitaminc", None, source_split, self.streaming)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        return nli_records(
            self, key=(source_split, row["unique_id"]), source_split=source_split, state=row["evidence"],
            claim=row["claim"], label=_FEVER_LABELS[row["label"]], revision_type=row["revision_type"],
        )


def evidence_window(text: str, spans: Sequence[Sequence[int]], evidence: Sequence[int], *key, n_random: tuple[int, int] = (4, 10)) -> str:
    """A chunk-sized view of a long document.

    With evidence: the evidence spans and one neighbour on each side, in document order, with ``…``
    over the gaps. Without evidence ("not mentioned"): a random run of ``n_random`` consecutive spans.
    """
    if evidence:
        keep = sorted({j for i in evidence for j in (i - 1, i, i + 1) if 0 <= j < len(spans)})
    else:
        rng = rng_for("window", *key)
        n = min(rng.randint(*n_random), len(spans))
        start = rng.randint(0, len(spans) - n)
        keep = list(range(start, start + n))
    parts, prev = [], None
    for i in keep:
        if prev is not None and i != prev + 1:
            parts.append("…")
        parts.append(text[spans[i][0]:spans[i][1]].strip())
        prev = i
    return "\n".join(p for p in parts if p)


class ContractNLI(Converter):
    """NDAs × 17 fixed hypotheses. Held out: the test split goes to ``test_ood`` only.

    Whole NDAs are far longer than 512 tokens, so the state is a chunk-sized evidence window
    (``evidence_window``), which also matches the judge's claims × chunks setting.
    """

    name = "contract_nli"
    domain = "legal"
    family = "factcheck"
    license = "CC BY 4.0 (official TERMS; the HF card's NC-SA tag is wrong)"
    origin = "https://stanfordnlp.github.io/contract-nli/resources/contract-nli.zip"
    held_out = True
    splits = {"test": "test_ood"}

    _LABELS = {"Entailment": ENTAILED, "Contradiction": CONTRADICTED, "NotMentioned": NOT_MENTIONED}

    def rows(self, source_split: str) -> Iterator[dict]:
        with zipfile.ZipFile(fetch(self.origin)) as z:
            data = json.loads(z.read(f"contract-nli/{source_split}.json"))
        hypotheses = {k: v["hypothesis"] for k, v in data["labels"].items()}
        for doc in data["documents"]:
            for hyp_id, ann in doc["annotation_sets"][0]["annotations"].items():
                yield {"doc_id": doc["id"], "text": doc["text"], "spans": doc["spans"], "hyp_id": hyp_id,
                       "hypothesis": hypotheses[hyp_id], "choice": ann["choice"], "evidence": ann["spans"]}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        key = (source_split, row["doc_id"], row["hyp_id"])
        evidence = row["evidence"] if row["choice"] != "NotMentioned" else []
        return nli_records(
            self, key=key, source_split=source_split, state=evidence_window(row["text"], row["spans"], evidence, *key),
            claim=row["hypothesis"], label=self._LABELS[row["choice"]],
        )


def table_to_text(csv_text: str, caption: str = "", sep: str = "#") -> str:
    """A ``#``-separated TabFact table as a markdown table, with the caption on top."""
    lines = [ln for ln in csv_text.strip().splitlines() if ln.strip()]
    if not lines:
        return caption
    rows = [[c.strip() for c in ln.split(sep)] for ln in lines]
    out = [f"Table: {caption}"] if caption else []
    out.append("| " + " | ".join(rows[0]) + " |")
    out.append("|" + "---|" * len(rows[0]))
    out.extend("| " + " | ".join(r) + " |" for r in rows[1:])
    return "\n".join(out)


class TabFact(Converter):
    """Statements about Wikipedia tables: entailed or refuted (two classes, so Bool only)."""

    name = "tabfact"
    domain = "wiki_tables"
    family = "factcheck"
    license = "CC BY 4.0 (HF), MIT (repo)"
    # The commit the HF loading script pins. The archive is large (~770 MB): it is the whole repo.
    origin = "https://github.com/wenhuchen/Table-Fact-Checking/archive/948b5560e2f7f8c9139bd91c7f093346a2bb56a8.zip"
    splits = {"train": "train", "test": "test_in"}

    _ROOT = "Table-Fact-Checking-948b5560e2f7f8c9139bd91c7f093346a2bb56a8"

    def rows(self, source_split: str) -> Iterator[dict]:
        with zipfile.ZipFile(fetch(self.origin, "tabfact.zip")) as z:
            examples = json.loads(z.read(f"{self._ROOT}/tokenized_data/{source_split}_examples.json"))
            for table_id, (statements, labels, caption) in examples.items():
                table = z.read(f"{self._ROOT}/data/all_csv/{table_id}").decode("utf-8")
                for i, (statement, label) in enumerate(zip(statements, labels)):
                    yield {"table_id": table_id, "idx": i, "table_text": table, "caption": caption,
                           "statement": statement, "label": label}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        return nli_records(
            self, key=(source_split, row["table_id"], row["idx"]), source_split=source_split,
            state=table_to_text(row["table_text"], row["caption"]), claim=row["statement"],
            label=ENTAILED if row["label"] == 1 else CONTRADICTED, three_way=False,
        )
