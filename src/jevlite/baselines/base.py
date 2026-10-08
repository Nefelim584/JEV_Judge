"""The predictor interface shared by baselines and our model, and a runner over unified records."""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterator, Protocol, Sequence

import numpy as np

from ..data.unified import Prediction, Record
from ..encoding import serialize_state
from ..schema import Question, State


class Predictor(Protocol):
    name: str

    def predict(self, state: State, questions: Sequence[Question]) -> list[np.ndarray]:
        """One array per question, in order: probabilities over the candidates for Choice and Score,
        ``[P(yes)]`` for Bool."""
        ...


def pick_device(device: str | None = None):
    import torch

    if device and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def entropy_confidence(p: np.ndarray) -> float:
    """v1 heuristic confidence (TODO 2.8): ``1 − H(p) / log K``; Bool uses K = 2."""
    p = np.asarray(p, dtype=float)
    if p.size == 1:
        p = np.array([1.0 - p[0], p[0]])
    if p.size < 2:
        return 1.0
    h = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(1.0 - h / np.log(p.size))


@dataclass(frozen=True)
class Skipped:
    """A record the predictor could not handle (e.g. Laya: options longer than its 192-token header)."""

    id: str
    error: str


def _prediction(r: Record, p: np.ndarray, latency_ms: float) -> Prediction:
    p = np.asarray(p, dtype=float)
    conf = entropy_confidence(p)
    if r.type == "bool":
        return Prediction(id=r.id, prob=float(np.clip(p[0], 0.0, 1.0)), confidence=conf, latency_ms=latency_ms)
    return Prediction(id=r.id, probs=(p / p.sum()).tolist(), confidence=conf, latency_ms=latency_ms)


def _call(predictor: Predictor, chunk: Sequence[Record]) -> list[Prediction]:
    questions = [r.to_question(qid=f"q{i}") for i, r in enumerate(chunk)]
    t0 = time.perf_counter()
    results = predictor.predict(chunk[0].state, questions)
    per_q = (time.perf_counter() - t0) * 1000 / len(chunk)
    return [_prediction(r, p, per_q) for r, p in zip(chunk, results, strict=True)]


def _call_many(predictor, chunks: Sequence[Sequence[Record]]) -> list[Prediction]:
    calls = [(c[0].state, [r.to_question(qid=f"q{i}") for i, r in enumerate(c)]) for c in chunks]
    t0 = time.perf_counter()
    results = predictor.predict_many(calls)
    per_q = (time.perf_counter() - t0) * 1000 / sum(len(c) for c in chunks)
    return [_prediction(r, p, per_q) for c, res in zip(chunks, results, strict=True) for r, p in zip(c, res, strict=True)]


def _chunks(records: Sequence[Record], max_questions: int) -> list[list[Record]]:
    """Records grouped by state, at most ``max_questions`` per group, in first-seen order."""
    by_state: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_state[serialize_state(r.state)].append(r)
    return [group[s : s + max_questions] for group in by_state.values() for s in range(0, len(group), max_questions)]


def iter_predictions(
    predictor: Predictor, records: Sequence[Record], max_questions: int = 16, skip_errors: bool = False, block: int = 1
) -> Iterator[tuple[list[Prediction], list[Skipped]]]:
    """Run ``predictor`` over records, one call per state with up to ``max_questions`` questions, and
    yield ``(predictions, skipped)`` as results come, so callers can write them incrementally.

    ``block > 1`` with a predictor that has ``predict_many`` (NLI) sends ``block`` states per call, so
    the model sees full batches. ``latency_ms`` of a record is the wall time of its call divided by the
    number of questions in it (amortised over the block).

    With ``skip_errors``, a failed call is retried state by state and then question by question, and a
    question that still fails is skipped instead of stopping the run.
    """
    chunks = _chunks(records, max_questions)
    step = block if hasattr(predictor, "predict_many") else 1
    for b in range(0, len(chunks), step):
        group = chunks[b : b + step]
        if len(group) > 1:
            try:
                yield _call_many(predictor, group), []
                continue
            except Exception:
                if not skip_errors:
                    raise
        for chunk in group:
            try:
                yield _call(predictor, chunk), []
                continue
            except Exception as e:
                if not skip_errors:
                    raise
                if len(chunk) == 1:
                    yield [], [Skipped(chunk[0].id, f"{type(e).__name__}: {e}")]
                    continue
            for r in chunk:
                try:
                    yield _call(predictor, [r]), []
                except Exception as e:
                    yield [], [Skipped(r.id, f"{type(e).__name__}: {e}")]


def predict_records(
    predictor: Predictor, records: Sequence[Record], max_questions: int = 16, progress: bool = False
) -> list[Prediction]:
    """All predictions in record order (errors propagate)."""
    out: dict[str, Prediction] = {}
    for preds, _ in iter_predictions(predictor, records, max_questions):
        out.update((p.id, p) for p in preds)
        if progress:
            print(f"\r  {predictor.name}: {len(out)}/{len(records)}", end="", flush=True)
    if progress:
        print()
    return [out[r.id] for r in records]
