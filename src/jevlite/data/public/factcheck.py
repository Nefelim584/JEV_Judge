"""Claim vs evidence sets with paragraph- or document-length states (``DATASETS.md`` section 0).

FEVER, HoVer and FEVEROUS keep their evidence text in separate Wikipedia dumps (1.7 GB, 2.2 GB and
10 GB zipped / 53 GB unzipped). ``rows`` downloads them into the raw dir and resolves the evidence, so
convert them where the disk allows it (``notebooks/kaggle/05_convert_wiki_sources.ipynb``).
"""

from __future__ import annotations

import io
import json
import re
import shutil
import sqlite3
import unicodedata
import zipfile
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from ...log import logger
from ..unified import Record
from .base import (
    CONTRADICTED,
    ENTAILED,
    NOT_MENTIONED,
    Converter,
    fetch,
    hf_rows,
    nli_records,
    raw_dir,
    rng_for,
)

# Claims read per source split in smoke runs (``streaming``): the wiki lookups happen before
# ``records(limit=…)`` can stop the iteration, so the claim files are cut here instead.
SMOKE_CLAIMS = 1000

_FEVER_LABELS = {"SUPPORTS": ENTAILED, "REFUTES": CONTRADICTED, "NOT ENOUGH INFO": NOT_MENTIONED}


class VitaminC(Converter):
    """Contrastive Wikipedia revisions: the same claim against slightly different evidence."""

    name = "vitaminc"
    domain = "wiki"
    family = "factcheck"
    license = "CC BY-SA 3.0"
    share_alike = True
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


# ---------------------------------------------------------------------------------------------------
# Wikipedia evidence: FEVER, HoVer, FEVEROUS
#
# ``rows`` resolves the evidence and yields self-contained rows; ``convert`` stays pure. A resolved
# page is ``{"title", "sentences", "evidence"}`` (sentence indices) or ``{"title", "text"}`` (a whole
# paragraph); every page becomes one chunk of the state, so multi-page evidence gives a list state,
# like the judge's chunks.

_PTB = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}", "-COLON-": ":", "``": '"', "''": '"'}
_PTB_RE = re.compile("|".join(map(re.escape, _PTB)))


def detokenize(text: str) -> str:
    """FEVER's PTB-tokenised Wikipedia text (``-LRB-``, ``word ,``) back to plain text."""
    text = _PTB_RE.sub(lambda m: _PTB[m.group()], text)
    text = re.sub(r" ([,.;:!?%)\]}])", r"\1", text)
    text = re.sub(r"([(\[{$]) ", r"\1", text)
    text = re.sub(r" (n't|'s|'re|'ve|'m|'ll|'d)\b", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def sentence_window(sentences: Sequence[str], evidence: Sequence[int], *key, n_random: tuple[int, int] = (4, 10)) -> str:
    """``evidence_window`` over a list of sentences: the evidence sentences and one neighbour on each
    side, with ``…`` over gaps; without evidence a random run of ``n_random`` sentences. Empty
    sentences (gaps in the source numbering) are skipped."""
    present = [i for i, s in enumerate(sentences) if s.strip()]
    if not present:
        return ""
    if evidence:
        keep = sorted({j for i in evidence for j in (i - 1, i, i + 1) if 0 <= j < len(sentences) and sentences[j].strip()})
    else:
        rng = rng_for("window", *key)
        n = min(rng.randint(*n_random), len(present))
        start = rng.randint(0, len(present) - n)
        keep = present[start:start + n]
    parts, prev = [], None
    for i in keep:
        if prev is not None and any(sentences[j].strip() for j in range(prev + 1, i)):
            parts.append("…")
        parts.append(sentences[i].strip())
        prev = i
    return " ".join(parts)


def wiki_chunks(pages: Sequence[dict], *key) -> str | list[str]:
    """One ``title\\ntext`` chunk per resolved page; a single chunk is a plain string state."""
    chunks = []
    for page in pages:
        if "text" in page:
            body = page["text"].strip()
        else:
            body = sentence_window(page["sentences"], page["evidence"], page["title"], *key)
        if body:
            chunks.append(f"{page['title']}\n{body}")
    return chunks[0] if len(chunks) == 1 else chunks


def clean_abstract(text: str) -> str:
    """HotpotQA abstracts lose pronunciations and leave ``Name ( ) is`` or ``Name (] ; born …)``."""
    text = re.sub(r"\(\s*[\];,]*\s*\)", "", text)
    text = re.sub(r"\(\s*(?:[\];,]\s*)+", "(", text)
    return re.sub(r"\s+([,.;])", r"\1", re.sub(r"\s{2,}", " ", text)).strip()


def _open_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def fever_title(page_id: str) -> str:
    """``Savages_-LRB-2012_film-RRB-`` → ``Savages (2012 film)``."""
    return detokenize(page_id.replace("_", " "))


def title_key(title: str) -> str:
    """A title without its disambiguation suffix: ``Savages (2012 film)`` → ``Savages``."""
    return re.sub(r" \([^)]*\)$", "", title)


def fever_lines(lines: str) -> list[str]:
    """The ``lines`` field of a FEVER wiki page (``idx\\tsentence\\tlinks…`` per line) as a list
    indexed by sentence number."""
    out: list[str] = []
    for line in lines.split("\n"):
        parts = line.split("\t")
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        i = int(parts[0])
        out.extend([""] * (i + 1 - len(out)))
        out[i] = detokenize(parts[1])
    return out


# Capitalised function words that open sentences; as one-word titles they are never the subject.
_STOP_TITLES = frozenset("A An The In On At By For From Of To It Its He She His Her They Their There This That These Those".split())


def claim_grams(claim: str, max_words: int = 6) -> list[str]:
    """Capitalised word n-grams of a claim, longest first: candidate titles of its subject."""
    words = [re.sub(r"'s$", "", w.strip(",.;:!?\"")) for w in claim.split()]
    words = [w for w in words if w]
    grams = {
        " ".join(words[i:i + n])
        for n in range(max_words, 0, -1)
        for i in range(len(words) - n + 1)
        if words[i][:1].isupper() and not (n == 1 and words[i] in _STOP_TITLES)
    }
    return sorted(grams, key=lambda g: (-len(g.split()), -len(g), g))


def subject_page(claim: str, by_key: dict[str, list[str]]) -> str | None:
    """The page of the longest title match in the claim; a title without a disambiguation suffix
    wins over ``Name (film)`` and the like."""
    for gram in claim_grams(claim):
        if gram in by_key:
            return min(by_key[gram], key=lambda pid: (fever_title(pid) != gram, pid))
    return None


_ID_RE = re.compile(r'^\{"id": "((?:[^"\\]|\\.)*)"')


class FEVER(Converter):
    """Claims written from Wikipedia lead sentences, verified against the 2017 Wikipedia dump.

    Supported and refuted claims: one chunk per page of one evidence set, the evidence sentences with
    a neighbour on each side. NEI claims have no evidence: they get a random run of sentences from the
    page of their likely subject (the longest title match in the claim), so "not mentioned" stays
    on-topic instead of trivially unrelated.
    """

    name = "fever"
    domain = "wiki"
    family = "factcheck"
    license = "CC BY-SA 3.0 (fever.ai)"
    share_alike = True
    origin = "https://fever.ai/download/fever/"
    splits = {"train": "train", "shared_task_dev": "test_in"}

    _resolved: dict[str, list[dict]] | None = None

    def rows(self, source_split: str) -> Iterator[dict]:
        if self._resolved is None:
            self._resolved = self._resolve()
        yield from self._resolved[source_split]

    def _claims(self, split: str) -> list[dict]:
        path = fetch(f"{self.origin}{split}.jsonl", f"fever/{split}.jsonl")
        with path.open() as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return rows[:SMOKE_CLAIMS] if self.streaming else rows

    def _resolve(self) -> dict[str, list[dict]]:
        claims = {split: self._claims(split) for split in self.splits}
        needed, grams = set(), set()
        for rows in claims.values():
            for c in rows:
                if c["label"] == "NOT ENOUGH INFO":
                    grams.update(claim_grams(c["claim"]))
                else:
                    needed.update(ev[2] for evset in c["evidence"] for ev in evset if ev[2])
        sentences, by_key = self._scan_wiki(needed, grams)
        out: dict[str, list[dict]] = {}
        for split, rows in claims.items():
            out[split], dropped = [], 0
            for c in rows:
                pages = self._pages(c, split, sentences, by_key)
                if pages:
                    out[split].append({"id": c["id"], "claim": c["claim"], "label": c["label"], "pages": pages})
                else:
                    dropped += 1
            logger.info("[fever] {}: {} claims resolved, {} dropped (pages missing / no subject page)", split, len(out[split]), dropped)
        return out

    def _scan_wiki(self, needed: set[str], grams: set[str]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """One pass over the dump: sentences of the evidence pages and of every page whose title
        matches a claim n-gram. Only those lines are fully parsed."""
        sentences: dict[str, list[str]] = {}
        by_key: dict[str, list[str]] = {}
        with zipfile.ZipFile(fetch(f"{self.origin}wiki-pages.zip", "fever/wiki-pages.zip")) as z:
            for info in z.infolist():
                # The archive also carries macOS resource forks (``__MACOSX/…/._wiki-001.jsonl``).
                if not info.filename.startswith("wiki-pages/") or not info.filename.endswith(".jsonl"):
                    continue
                with z.open(info) as f:
                    for line in io.TextIOWrapper(f, encoding="utf-8"):
                        m = _ID_RE.match(line)
                        if not m or not m.group(1):
                            continue
                        pid = json.loads(f'"{m.group(1)}"')
                        key = title_key(fever_title(pid))
                        if key in grams:
                            by_key.setdefault(key, []).append(pid)
                        elif pid not in needed:
                            continue
                        sentences[pid] = fever_lines(json.loads(line)["lines"])
        return sentences, by_key

    def _pages(self, claim: dict, split: str, sentences: dict[str, list[str]], by_key: dict[str, list[str]]) -> list[dict]:
        if claim["label"] == "NOT ENOUGH INFO":
            pid = subject_page(claim["claim"], by_key)
            return [{"title": fever_title(pid), "sentences": sentences[pid], "evidence": []}] if pid else []
        sets = [s for s in claim["evidence"] if all(ev[2] in sentences for ev in s)]
        if not sets:
            return []
        evset = rng_for("evidence", self.name, split, claim["id"]).choice(sets)
        wanted: dict[str, list[int]] = {}
        for ev in evset:
            wanted.setdefault(ev[2], []).append(ev[3])
        return [{"title": fever_title(pid), "sentences": sentences[pid], "evidence": sorted(set(ev))} for pid, ev in wanted.items()]

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        key = (source_split, row["id"])
        state = wiki_chunks(row["pages"], self.name, *key)
        if not state:
            return []
        return nli_records(self, key=key, source_split=source_split, state=state, claim=row["claim"], label=_FEVER_LABELS[row["label"]])


class HoVer(Converter):
    """Multi-hop claims over 2–4 Wikipedia abstracts (HotpotQA's 2017 dump): every abstract is one
    chunk, and the claim needs all of them. Two classes (supported / not supported), so Bool only."""

    name = "hover"
    domain = "wiki"
    family = "factcheck"
    license = "CC BY-SA 4.0"
    share_alike = True
    origin = "https://github.com/hover-nlp/hover"
    splits = {"train": "train", "dev": "test_in"}

    _CLAIMS = "https://raw.githubusercontent.com/hover-nlp/hover/main/data/hover/hover_{split}_release_v1.1.json"
    _WIKI = "https://nlp.cs.unc.edu/data/hover/wiki_wo_links.db"

    def rows(self, source_split: str) -> Iterator[dict]:
        path = fetch(self._CLAIMS.format(split=source_split), f"hover/hover_{source_split}_release_v1.1.json")
        claims = json.loads(path.read_text())
        if self.streaming:
            claims = claims[:SMOKE_CLAIMS]
        db = _open_ro(fetch(self._WIKI, "hover/wiki_wo_links.db"))
        dropped = 0
        try:
            for c in claims:
                pages = []
                for title in dict.fromkeys(t for t, _ in c["supporting_facts"]):
                    hit = db.execute("SELECT text FROM documents WHERE id = ?", (unicodedata.normalize("NFD", title),)).fetchone()
                    if hit is None:
                        break
                    pages.append({"title": title, "text": clean_abstract(unicodedata.normalize("NFC", hit[0]))})
                else:
                    yield {"uid": c["uid"], "claim": c["claim"], "label": c["label"], "num_hops": c["num_hops"], "pages": pages}
                    continue
                dropped += 1
        finally:
            db.close()
        logger.info("[hover] {}: {} claims dropped (abstract missing from the dump)", source_split, dropped)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        key = (source_split, row["uid"])
        state = wiki_chunks(row["pages"], self.name, *key)
        if not state:
            return []
        # NOT_SUPPORTED mixes refuted and unverifiable claims; only the Bool target (0) is used.
        label = ENTAILED if row["label"] == "SUPPORTED" else NOT_MENTIONED
        return nli_records(
            self, key=key, source_split=source_split, state=state, claim=row["claim"], label=label,
            three_way=False, num_hops=row["num_hops"],
        )


_LINK_RE = re.compile(r"\[\[([^\]|]*)\|([^\]]*)\]\]")
_BARE_LINK_RE = re.compile(r"\[\[([^\]]*)\]\]")


def strip_wiki_links(text: str) -> str:
    """``[[Algebraic_logic|algebraic logic]]`` → ``algebraic logic``; ``[[Paris]]`` → ``Paris``."""
    text = _LINK_RE.sub(r"\2", text)
    return _BARE_LINK_RE.sub(lambda m: m.group(1).replace("_", " "), text).strip()


def feverous_sentences(page: dict) -> list[str]:
    """The ``sentence_N`` elements of a FEVEROUS page JSON as a list indexed by N."""
    out: list[str] = []
    for k, v in page.items():
        if k.startswith("sentence_") and k[9:].isdigit():
            i = int(k[9:])
            out.extend([""] * (i + 1 - len(out)))
            out[i] = strip_wiki_links(v)
    return out


class FEVEROUS(Converter):
    """Claims over Wikipedia sentences, tables and lists; only the text-only part is used (an evidence
    set made of sentences only). NEI claims keep the partial evidence annotators found, which makes
    them near-miss "not mentioned" examples.

    The page database is 53 GB unzipped: ``rows`` extracts it once and deletes the 10 GB archive.
    """

    name = "feverous"
    domain = "wiki"
    family = "factcheck"
    license = "CC BY-SA 3.0 (as FEVER); code Apache-2.0"
    share_alike = True
    origin = "https://fever.ai/download/feverous/"
    splits = {"train": "train", "dev": "test_in"}

    def _db(self) -> Path:
        db = raw_dir() / "feverous" / "feverous_wikiv1.db"
        if db.exists():
            return db
        archive = fetch(f"{self.origin}feverous-wiki-pages-db.zip", "feverous/feverous-wiki-pages-db.zip")
        tmp = db.with_suffix(".db.part")
        with zipfile.ZipFile(archive) as z:
            member = next(n for n in z.namelist() if n.endswith(".db"))
            with z.open(member) as src, tmp.open("wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 24)
        tmp.rename(db)
        archive.unlink()
        return db

    def rows(self, source_split: str) -> Iterator[dict]:
        path = fetch(f"{self.origin}feverous_{source_split}_challenges.jsonl", f"feverous/feverous_{source_split}_challenges.jsonl")
        with path.open() as f:
            claims = [c for c in map(json.loads, filter(str.strip, f)) if c.get("claim")]  # line 1 is an empty record
        if self.streaming:
            claims = claims[:SMOKE_CLAIMS]
        db = _open_ro(self._db())
        not_text, missing = 0, 0
        try:
            for c in claims:
                sets = [s["content"] for s in c["evidence"] if s["content"] and all("_sentence_" in e for e in s["content"])]
                if not sets:
                    not_text += 1
                    continue
                wanted: dict[str, list[int]] = {}
                for element in rng_for("evidence", self.name, source_split, c["id"]).choice(sets):
                    title, idx = element.rsplit("_sentence_", 1)
                    wanted.setdefault(title, []).append(int(idx))
                pages = []
                for title, evidence in wanted.items():
                    hit = db.execute("SELECT data FROM wiki WHERE id = ?", (title,)).fetchone()
                    if hit is None:
                        break
                    pages.append({"title": title, "sentences": feverous_sentences(json.loads(hit[0])), "evidence": sorted(set(evidence))})
                else:
                    yield {"id": c["id"], "claim": c["claim"], "label": c["label"], "pages": pages}
                    continue
                missing += 1
        finally:
            db.close()
        logger.info("[feverous] {}: {} claims with table/list evidence skipped, {} with pages missing", source_split, not_text, missing)

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        key = (source_split, row["id"])
        state = wiki_chunks(row["pages"], self.name, *key)
        if not state:
            return []
        return nli_records(self, key=key, source_split=source_split, state=state, claim=row["claim"], label=_FEVER_LABELS[row["label"]])
