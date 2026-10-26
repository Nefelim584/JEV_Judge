"""Code-side checks and edits: quotes, answer layout, programmatic perturbations, near-duplicates.

Everything here is deterministic given an rng, so a label that comes from these functions is a
``program`` label (dataset_dicr.md 8.7).
"""

from __future__ import annotations

import random
import re
import zlib

import numpy as np

from jevlite.judge.claims import split_sentences

# ---------------------------------------------------------------- quotes


def _normalize(text: str) -> tuple[str, list[int]]:
    """Lower-cased text with runs of whitespace collapsed and curly quotes straightened; plus the
    index in the original text of every normalized character."""
    out, index, prev_space = [], [], False
    table = {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-"}
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space or not out:
                continue
            out.append(" "); index.append(i); prev_space = True
            continue
        out.append(table.get(ch, ch).lower()); index.append(i); prev_space = False
    return "".join(out), index


def find_quote(chunk: str, quote: str) -> list[int] | None:
    """Character span of ``quote`` in ``chunk``, tolerant to case, whitespace and an ellipsis inside
    the quote (then the longest piece must match). None when it is not there."""
    quote = quote.strip().strip("\"'“”‘’").strip()
    if not quote:
        return None
    norm, index = _normalize(chunk)
    pieces = [quote] + sorted(re.split(r"\s*(?:\.\.\.|…)\s*", quote), key=len, reverse=True)
    for piece in pieces:
        q, _ = _normalize(piece.strip(" .,;:"))
        if len(q) < 3:
            continue
        pos = norm.find(q)
        if pos >= 0:
            return [index[pos], index[pos + len(q) - 1] + 1]
    return None


# ---------------------------------------------------------------- answer layout

_FINAL_PUNCT = re.compile(r"^(.*?)([.!?:]*)$", re.S)


def render_sentence(core: str, cites: list[int]) -> str:
    """``core`` with ``[n]`` citations put before its final punctuation (dataset_dicr.md 6.2)."""
    if not cites:
        return core
    body, punct = _FINAL_PUNCT.match(core.strip()).groups()
    return f"{body} " + "".join(f"[{c}]" for c in cites) + punct


def join_answer(sentences: list[str], form: str) -> str:
    if form == "list" and len(sentences) >= 3:
        return sentences[0] + "\n" + "\n".join(f"- {s}" for s in sentences[1:])
    return " ".join(sentences)


def sentence_spans(text: str, sentences: list[str]) -> list[list[int]] | None:
    """Spans of ``sentences`` in ``text``, or None unless the judge's splitter yields exactly them."""
    if split_sentences(text) != sentences:
        return None
    spans, pos = [], 0
    for s in sentences:
        start = text.find(s, pos)
        if start < 0:
            return None
        spans.append([start, start + len(s)])
        pos = start + len(s)
    return spans


def length_ok(new: str, old: str, bounds: tuple[float, float]) -> bool:
    r = len(new.split()) / max(1, len(old.split()))
    return bounds[0] <= r <= bounds[1]


# ---------------------------------------------------------------- programmatic perturbations

_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
           "November", "December"]
_UNITS = r"(?:second|minute|hour|day|week|month|year|business day|working day)s?"
_YEAR = re.compile(r"(?<![\d.,$])(?:1[89]\d\d|20\d\d)(?![\d.,%])")
_MONTH = re.compile(r"\b(" + "|".join(_MONTHS) + r")\b")
_DURATION = re.compile(r"(?<![\d.,])(\d+)(\s*-?\s*" + _UNITS + r")\b", re.I)
_NUMBER = re.compile(r"(?<![\w.,\[])(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\d\]])")


def _numbers(text: str) -> list[re.Match]:
    years = {m.span() for m in _YEAR.finditer(text)}
    return [m for m in _NUMBER.finditer(text) if m.span() not in years]


def _as_float(m: re.Match) -> float:
    return float(m.group(0).replace(",", ""))


def _format_like(value: float, like: re.Match) -> str:
    decimals = len(like.group(2) or "")
    s = f"{value:,.{decimals}f}" if "," in like.group(1) else f"{value:.{decimals}f}"
    return s


def _shifted(rng: random.Random, m: re.Match) -> str:
    x = _as_float(m)
    if x == 0:
        return _format_like(rng.choice([1, 2, 5]), m)
    for _ in range(20):
        factor = rng.choice([0.5, 0.75, 1.25, 1.5, 2, 3]) if x >= 10 else 1
        y = x * factor if x >= 10 else x + rng.choice([-2, -1, 1, 2, 3])
        if y > 0 and _format_like(y, m) != m.group(0):
            return _format_like(round(y) if not m.group(2) else y, m)
    return _format_like(x + 1, m)


def number_swap(rng: random.Random, sentence: str, others: list[str], avoid: str, prefer_others: bool):
    """Replace one number (not a year) of ``sentence``. The new value comes from ``others`` (the chunks
    the sentence is not based on) when ``prefer_others`` and possible, else a shifted value; never a
    number that occurs in ``avoid`` (the evidence chunk). Returns (sentence, from, to) or None."""
    targets = _numbers(sentence)
    if not targets:
        return None
    m = rng.choice(targets)
    banned = {n.group(0) for n in _numbers(avoid)} | {m.group(0)}
    pool = [n.group(0) for o in others for n in _numbers(o)
            if n.group(0) not in banned and bool(n.group(2)) == bool(m.group(2))]
    new = rng.choice(pool) if pool and prefer_others else _shifted(rng, m)
    if new in banned:
        return None
    return sentence[: m.start()] + new + sentence[m.end():], m.group(0), new


def date_swap(rng: random.Random, sentence: str, others: list[str], avoid: str, prefer_others: bool):
    """Replace a year, a month name or the number of a duration ("30 days")."""
    options = [("year", m) for m in _YEAR.finditer(sentence)] + [("month", m) for m in _MONTH.finditer(sentence)] \
        + [("duration", m) for m in _DURATION.finditer(sentence)]
    if not options:
        return None
    kind, m = rng.choice(options)
    if kind == "duration":
        old = m.group(1)
        banned = {d.group(1) for d in _DURATION.finditer(avoid)} | {old}
        pool = [d.group(1) for o in others for d in _DURATION.finditer(o) if d.group(1) not in banned]
        if pool and prefer_others:
            new = rng.choice(pool)
        else:
            x = int(old)
            new = str(max(1, x * rng.choice([2, 3]) if x < 10 else int(x * rng.choice([0.5, 1.5, 2]))))
            if new in banned:
                new = str(x + 1)
        return sentence[: m.start(1)] + new + sentence[m.end(1):], old, new
    old = m.group(0)
    if kind == "year":
        banned = set(_YEAR.findall(avoid)) | {old}
        pool = [y for o in others for y in _YEAR.findall(o) if y not in banned]
        new = rng.choice(pool) if pool and prefer_others else str(int(old) + rng.choice([-3, -2, -1, 1, 2]))
    else:
        banned = set(_MONTH.findall(avoid)) | {old}
        new = rng.choice([mo for mo in _MONTHS if mo not in banned])
    if new in banned:
        return None
    return sentence[: m.start()] + new + sentence[m.end():], old, new


_CAPS = re.compile(r"\b([A-Z][\w&'-]*(?:\s+(?:of|the|and|for|de|von)?\s*[A-Z][\w&'-]*)*)")
_STOP = {"The", "A", "An", "This", "That", "These", "Those", "It", "If", "You", "Your", "We", "Our", "In", "On",
         "For", "To", "When", "After", "Before", "Note", "Yes", "No", "All", "Each", "Some", "Please"}


def entity_candidates(texts: list[str], exclude: str, limit: int = 8) -> list[str]:
    """Capitalised names in ``texts`` that do not occur in ``exclude``: values for entity_swap."""
    seen: list[str] = []
    for t in texts:
        for m in _CAPS.finditer(t):
            name = m.group(1).strip()
            if name in _STOP or len(name) < 3 or name in exclude or name in seen:
                continue
            seen.append(name)
    return seen[:limit]


# ---------------------------------------------------------------- near-duplicates


def _shingles(text: str, n: int = 5) -> set[int]:
    words = re.findall(r"\w+", text.lower())
    return {zlib.crc32(" ".join(words[i : i + n]).encode()) for i in range(max(1, len(words) - n + 1))}


def near_duplicates(texts: list[str], threshold: float, n_perm: int = 64, bands: int = 16, seed: int = 0) -> set[int]:
    """Indices of texts whose word 5-gram Jaccard with an earlier text is ≥ ``threshold`` (MinHash +
    LSH for candidates, exact Jaccard to confirm). The first of each group is kept."""
    rng = np.random.default_rng(seed)
    prime = (1 << 61) - 1
    a = rng.integers(1, prime, n_perm, dtype=np.uint64)
    b = rng.integers(0, prime, n_perm, dtype=np.uint64)
    shingles = [_shingles(t) for t in texts]
    rows = n_perm // bands
    buckets: dict[tuple, list[int]] = {}
    drop: set[int] = set()
    for i, sh in enumerate(shingles):
        x = np.fromiter(sh, dtype=np.uint64)
        sig = ((np.outer(x, a) + b) % np.uint64(prime)).min(axis=0)  # wraps mod 2^64: fine for hashing
        candidates: set[int] = set()
        keys = [(band, sig[band * rows : (band + 1) * rows].tobytes()) for band in range(bands)]
        for key in keys:
            candidates.update(buckets.get(key, ()))
        if any(j not in drop and len(sh & shingles[j]) / len(sh | shingles[j]) >= threshold for j in candidates):
            drop.add(i)
            continue
        for key in keys:
            buckets.setdefault(key, []).append(i)
    return drop
