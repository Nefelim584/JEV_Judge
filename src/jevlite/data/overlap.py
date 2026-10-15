"""Passage overlap between training data and eval sets (todo, Stage A review, Q2).

Many sources reuse one passage for several questions (BoolQ, TabFact tables, Wikipedia intros in
FEVER / HoVer / VitaminC, HelpSteer conversations). A test record whose passage the model has trained
on measures recall of that passage as much as the skill, so headline numbers are computed on the
records whose passages appear in the training data of **none** of the compared runs. Training data is
not rebuilt (Q2): the eval sets are filtered instead.

A record's **passages** are the elements of a list state, the string values of a dict state, or the
whole string state. Two keys per passage, on whitespace-collapsed lowercase text:

- ``exact``: the whole passage;
- ``prefix``: its first ``PREFIX_CHARS`` characters (only for passages at least that long), which
  catches a shared conversation with a different response (HelpSteer) or the same table / article
  opening with different rows or a different cut.

A record overlaps if any of its passages shares a key with any training record of any source (the
same Wikipedia intro occurs in several sources).

``test_ood`` mixes two kinds of OOD (todo, Stage A review): records of a (source, domain) never
trained on (``test_ood_source``: ContractNLI, BANKING77, the mismatched MultiNLI genres) and held-out
templates on trained sources (``test_ood_template``). :func:`ood_kind` tells them apart from the same
training files.

The mid-run eval subsets of earlier runs (``train.eval_subset``) took part in decisions; they are
dropped from the tests by id with :func:`eval_subset_ids` (todo, Stage A review, Q1).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from .public.base import stable_hash

PREFIX_CHARS = 200
MODES = ("exact", "prefix")
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def passages(state: Any) -> list[str]:
    if isinstance(state, str):
        units = [state]
    elif isinstance(state, list):
        units = [u for u in state if isinstance(u, str)]
    elif isinstance(state, dict):
        units = [v for v in state.values() if isinstance(v, str)]
    else:
        units = []
    return [n for n in map(_norm, units) if n]


def _digest(text: str) -> bytes:
    return hashlib.blake2b(text.encode(), digest_size=10).digest()


def passage_keys(state: Any, mode: str = "prefix") -> set[bytes]:
    """``exact``: whole-passage keys only; ``prefix``: whole-passage and prefix keys."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    keys = set()
    for p in passages(state):
        keys.add(b"x" + _digest(p))
        if mode == "prefix" and len(p) >= PREFIX_CHARS:
            keys.add(b"p" + _digest(p[:PREFIX_CHARS]))
    return keys


def _rows(paths: Iterable[str | Path]) -> Iterable[dict]:
    for path in paths:
        with Path(path).open() as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def train_index(paths: Iterable[str | Path], mode: str = "prefix") -> set[bytes]:
    """Keys of every passage in the given training files (e.g. ``train.e*.jsonl`` of all compared runs)."""
    index: set[bytes] = set()
    for row in _rows(paths):
        index |= passage_keys(row["state"], mode)
    return index


def overlaps(row: dict, index: set[bytes], mode: str = "prefix") -> bool:
    return not passage_keys(row["state"], mode).isdisjoint(index)


def train_domains(paths: Iterable[str | Path]) -> set[tuple[str, str | None]]:
    """(source, domain) pairs present in the training files."""
    return {(row["source"], row.get("domain")) for row in _rows(paths)}


def ood_kind(row: dict, domains: set[tuple[str, str | None]]) -> str:
    """``test_ood_source`` if the record's (source, domain) was never trained on, else ``test_ood_template``."""
    return "test_ood_template" if (row["source"], row.get("domain")) in domains else "test_ood_source"


def eval_subset_ids(rows: list[dict], n: int, seed: int) -> set[str]:
    """Ids of ``train.eval_subset(records, n, seed)``: the fixed mid-run eval subset of a split."""
    if len(rows) <= n:
        return {r["id"] for r in rows}
    return {r["id"] for r in sorted(rows, key=lambda r: stable_hash(seed, "eval", r["id"]))[:n]}
