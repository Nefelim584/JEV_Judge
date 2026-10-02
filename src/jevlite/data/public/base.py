"""Shared machinery of the public-data converters (TODO Phase 3, ``DATASETS.md`` section 0).

A converter turns the rows of one public dataset into unified records:

- ``rows(source_split)`` loads raw rows (Hugging Face or the original source). It needs the network.
- ``convert(row, source_split)`` maps one row to records. It is pure, so tests feed it handmade rows.

Everything random (split carving, Bool vs Choice-3, templates, distractors) is derived from a stable
hash of the record key, so a rerun produces the same files.

Splits:

- the source ``train`` split becomes ``train``, with ``CALIB_SHARE`` carved out as ``calib``;
- labelled source dev / test splits become ``test_in``;
- held-out datasets (``held_out = True``) go entirely to ``test_ood``;
- in ``test_in`` rows, ``HELD_OUT_TEMPLATE_SHARE`` of the records use the held-out template of their
  family (the last one of each list) and go to ``test_ood``. Train and calib never use it.
"""

from __future__ import annotations

import hashlib
import os
import random
import ssl
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Iterable, Iterator, Sequence

from ...schema import MAX_CANDIDATES, State
from ..unified import Record

CALIB_SHARE = 0.05
HELD_OUT_TEMPLATE_SHARE = 0.5

# Choice-3 label set of NLI and fact-checking data, in canonical order (collate shuffles options).
ENTAILED, CONTRADICTED, NOT_MENTIONED = 0, 1, 2


def stable_hash(*parts: Any) -> int:
    """A 64-bit hash of ``parts`` that is the same across runs and machines (unlike ``hash``)."""
    key = "\x1f".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


def unit(*parts: Any) -> float:
    """A stable pseudo-random number in [0, 1)."""
    return stable_hash(*parts) / 2**64


def rng_for(*parts: Any) -> random.Random:
    return random.Random(stable_hash(*parts))


@dataclass(frozen=True)
class Template:
    """One wording of a question. ``prompt`` is formatted with the row's fields."""

    id: str
    prompt: str
    candidates: tuple[str, ...] | None = None


def pick_template(templates: Sequence[Template], split: str, *key: Any) -> tuple[Template, str]:
    """Pick a template and the final split.

    The last template of the list is held out: only ``test_in`` rows use it, and those records move
    to ``test_ood``. A list with one template has no held-out wording.
    """
    if len(templates) == 1:
        return templates[0], split
    if split == "test_in" and unit("held-out-template", *key) < HELD_OUT_TEMPLATE_SHARE:
        return templates[-1], "test_ood"
    pool = templates[:-1]
    return pool[stable_hash("template", *key) % len(pool)], split


def sample_candidates(
    labels: Sequence[str], gold: int, *key: Any, k_min: int = 3, k_max: int = 16
) -> tuple[list[str], int]:
    """Options for a classification row: the gold label plus random distractors.

    The number of options varies per row (``k_min``…``k_max``, capped by the label set and
    ``MAX_CANDIDATES``). Returns the options and the index of the gold one.
    """
    n = len(labels)
    k_hi = min(k_max, n, MAX_CANDIDATES)
    k_lo = min(k_min, k_hi)
    rng = rng_for("candidates", *key)
    k = rng.randint(k_lo, k_hi)
    others = [i for i in range(n) if i != gold]
    chosen = rng.sample(others, k - 1) + [gold]
    rng.shuffle(chosen)
    return [labels[i] for i in chosen], chosen.index(gold)


def humanize(label: str) -> str:
    """``card_declined`` → ``card declined``."""
    return label.replace("_", " ").strip()


# ---------------------------------------------------------------------------------------------------
# Raw downloads


def raw_dir() -> Path:
    return Path(os.environ.get("JEVLITE_RAW_DIR", "data/raw"))


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch(url: str, name: str | None = None) -> Path:
    """Download ``url`` into ``raw_dir()`` once and return the local path."""
    path = raw_dir() / (name or url.rsplit("/", 1)[-1])
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "jevlite"})
    with urllib.request.urlopen(req, context=_ssl_context(), timeout=120) as resp, tmp.open("wb") as f:
        while chunk := resp.read(1 << 20):
            f.write(chunk)
    tmp.rename(path)
    return path


def hf_rows(hf_id: str, config: str | None, split: str, streaming: bool = False, parquet_export: bool = False) -> Iterator[dict]:
    """``parquet_export`` reads HF's parquet export (``refs/convert/parquet``) of a dataset whose repo
    still has a loading script, which ``datasets`` no longer runs. The config is a folder there."""
    from datasets import load_dataset

    if parquet_export:
        yield from load_dataset(hf_id, data_dir=config, revision="refs/convert/parquet", split=split, streaming=streaming)
    else:
        yield from load_dataset(hf_id, config, split=split, streaming=streaming)


# ---------------------------------------------------------------------------------------------------
# Converters


class Converter:
    """Base class. Subclasses set the class attributes and implement ``rows`` and ``convert``."""

    name: ClassVar[str]
    domain: ClassVar[str] = "unknown"
    family: ClassVar[str]
    license: ClassVar[str]
    origin: ClassVar[str]
    held_out: ClassVar[bool] = False
    # source split → target split; "train" is carved into train + calib.
    splits: ClassVar[dict[str, str]] = {"train": "train", "validation": "test_in"}

    streaming: bool = False

    def rows(self, source_split: str) -> Iterable[dict]:
        raise NotImplementedError

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        raise NotImplementedError

    def target_split(self, source_split: str, *key: Any) -> str:
        if self.held_out:
            return "test_ood"
        split = self.splits[source_split]
        if split == "train" and unit("calib", self.name, *key) < CALIB_SHARE:
            return "calib"
        return split

    def records(self, limit: int | None = None) -> Iterator[Record]:
        """All records of all source splits; ``limit`` caps the rows read per split (smoke runs)."""
        for source_split in self.splits:
            for i, row in enumerate(self.rows(source_split)):
                if limit is not None and i >= limit:
                    break
                yield from self.convert(row, source_split)

    def record(self, *, key: Sequence[Any], split: str, state: State, **fields: Any) -> Record:
        """A record with the converter's metadata; ``key`` makes the id unique within the source."""
        return Record(
            id=":".join([self.name, *map(str, key)]),
            source=self.name,
            domain=fields.pop("domain", self.domain),
            split=split,
            state=state,
            family=fields.pop("family", self.family),
            **fields,
        )


# ---------------------------------------------------------------------------------------------------
# NLI and fact checking: one record per row, Bool or Choice-3

NLI_BOOL = (
    Template("claim", "{claim}"),
)
NLI_CHOICE = (
    Template(
        "support",
        "Is the claim supported by the text, contradicted by it, or not mentioned? Claim: {claim}",
        ("supported", "contradicted", "not mentioned"),
    ),
    Template(
        "verdict",
        "According to the text, the claim \"{claim}\" is",
        ("true", "false", "not stated"),
    ),
    Template(
        "evidence",
        "What does the text say about this claim: {claim}",
        ("it confirms it", "it contradicts it", "it says nothing about it"),
    ),
)


def nli_records(
    conv: Converter,
    *,
    key: Sequence[Any],
    source_split: str,
    state: State,
    claim: str,
    label: int,
    three_way: bool = True,
    **fields: Any,
) -> list[Record]:
    """One record for a (state, claim, label) triple with ``label`` in ENTAILED / CONTRADICTED /
    NOT_MENTIONED.

    About half of the rows become Bool ("the claim follows from the state"), the rest Choice-3; never
    both for the same row. ``three_way=False`` (two-class data such as TabFact) keeps Bool only.
    """
    claim = claim.strip()
    if not claim:
        return []
    split = conv.target_split(source_split, *key)
    if three_way and stable_hash("nli-type", conv.name, *key) % 2:
        tpl, split = pick_template(NLI_CHOICE, split, conv.name, *key)
        return [
            conv.record(
                key=key, split=split, state=state, type="choice", prompt=tpl.prompt.format(claim=claim),
                candidates=list(tpl.candidates), target=label, template=f"nli/{tpl.id}", **fields,
            )
        ]
    tpl, split = pick_template(NLI_BOOL, split, conv.name, *key)
    return [
        conv.record(
            key=key, split=split, state=state, type="bool", prompt=tpl.prompt.format(claim=claim),
            target=float(label == ENTAILED), template=f"nli/{tpl.id}", **fields,
        )
    ]


# ---------------------------------------------------------------------------------------------------
# Answerability: "the text answers the question"

ANSWERABLE = (
    Template("answers", "The text answers the question: {question}"),
    Template("contains", "The text contains the information needed to answer \"{question}\""),
    Template("enough", "Based only on the text, one can answer the question: {question}"),
)


def answerable_record(
    conv: Converter, *, key: Sequence[Any], source_split: str, state: State, question: str, answerable: bool, **fields: Any
) -> Record:
    split = conv.target_split(source_split, *key)
    tpl, split = pick_template(ANSWERABLE, split, conv.name, *key)
    return conv.record(
        key=key, split=split, state=state, type="bool", prompt=tpl.prompt.format(question=question.strip()),
        target=float(answerable), criterion="answerable", template=f"answerable/{tpl.id}", **fields,
    )
