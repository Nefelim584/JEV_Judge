"""Token-length statistics of exported judge triples (TODO 2.11, Phase 2).

Lengths are measured with the packed layout the judge will use (2.12), not estimated:

- ``faithfulness_per_chunk``: one sequence per chunk, the header holding the user's question and every
  sentence of the answer as a claim group;
- ``faithfulness_full``: the same header with all chunks as the state;
- ``relevance``: a Score question with the question and the answer as the state.

The shares above 512 / 1024 / … decide whether the long-context Stage C is needed.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ..encoding import ClaimGroup, PackedEncoder
from ..schema import BoolQuestion, ScoreQuestion
from .claims import split_sentences
from .export import JudgeSample

THRESHOLDS = (512, 1024, 2048, 4096, 8192)
PERCENTILES = (50, 90, 95, 99)
RELEVANCE_LEVELS = ["does not address", "partially addresses", "mostly addresses", "fully addresses"]


def _summary(values: Sequence[int]) -> dict:
    v = np.asarray(values)
    out = {"n": int(v.size), "mean": float(v.mean()), "max": int(v.max())}
    out |= {f"p{q}": float(np.percentile(v, q)) for q in PERCENTILES}
    out |= {f"share_gt_{t}": float((v > t).mean()) for t in THRESHOLDS}
    return out


def sample_lengths(sample: JudgeSample, enc: PackedEncoder) -> dict[str, list[int]]:
    sentences = split_sentences(sample.answer) or [sample.answer or " "]
    group = ClaimGroup(tuple(BoolQuestion(id=f"s{i}", prompt=s) for i, s in enumerate(sentences)), context=sample.question or None)
    header = len(enc.encode_header(group)[0]) + enc.NUM_SPECIAL
    chunks = [len(enc.tokenize_state(c)) for c in sample.chunks]
    relevance_q = ScoreQuestion(id="relevance", prompt="How well does the answer address the user's question?", levels=RELEVANCE_LEVELS)
    relevance = len(enc.encode_header(relevance_q)[0]) + enc.NUM_SPECIAL + len(
        enc.tokenize_state({"question": sample.question, "answer": sample.answer})
    )
    return {
        "question": [len(enc.tokenize_state(sample.question))],
        "answer": [len(enc.tokenize_state(sample.answer))],
        "answer_sentences": [len(sentences)],
        "n_chunks": [len(sample.chunks)],
        "chunk": chunks,
        "faithfulness_per_chunk": [header + c for c in chunks],
        "faithfulness_full": [header + len(enc.tokenize_state(sample.chunks))] if sample.chunks else [],
        "relevance": [relevance],
    }


def length_stats(samples: Iterable[JudgeSample], tokenizer) -> dict:
    enc = PackedEncoder(tokenizer, max_len=10**9)
    acc: dict[str, list[int]] = {}
    n = 0
    for s in samples:
        n += 1
        for key, values in sample_lengths(s, enc).items():
            acc.setdefault(key, []).extend(values)
    if not n:
        raise ValueError("no samples")
    return {"n_samples": n, **{k: _summary(v) for k, v in acc.items() if v}}


def to_markdown(stats: dict) -> str:
    cols = ["mean", *(f"p{q}" for q in PERCENTILES), "max", *(f"share_gt_{t}" for t in THRESHOLDS)]
    names = ["mean", *(f"p{q}" for q in PERCENTILES), "max", *(f">{t}" for t in THRESHOLDS)]
    lines = [f"Samples: {stats['n_samples']}. Token counts with the ModernBERT tokenizer; sequences include special tokens.", ""]
    lines.append("| | n | " + " | ".join(names) + " |")
    lines.append("|---|---|" + "---|" * len(cols))
    for key, row in stats.items():
        if key == "n_samples":
            continue
        cells = [f"{row[c]:.1%}" if c.startswith("share") else f"{row[c]:.0f}" for c in cols]
        lines.append(f"| {key} | {row['n']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
