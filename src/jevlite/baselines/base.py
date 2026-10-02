"""The predictor interface shared by baselines and our model, and a runner over unified records."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Protocol, Sequence

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


def predict_records(
    predictor: Predictor, records: Sequence[Record], max_questions: int = 16, progress: bool = False
) -> list[Prediction]:
    """Run ``predictor`` over records, one call per state with up to ``max_questions`` questions.

    ``latency_ms`` of a record is the wall time of its call divided by the number of questions in it.
    """
    by_state: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        by_state[serialize_state(r.state)].append(r)

    out: dict[str, Prediction] = {}
    done = 0
    for group in by_state.values():
        for start in range(0, len(group), max_questions):
            chunk = group[start : start + max_questions]
            questions = [r.to_question(qid=f"q{i}") for i, r in enumerate(chunk)]
            t0 = time.perf_counter()
            results = predictor.predict(chunk[0].state, questions)
            per_q = (time.perf_counter() - t0) * 1000 / len(chunk)
            for r, p in zip(chunk, results, strict=True):
                p = np.asarray(p, dtype=float)
                conf = entropy_confidence(p)
                if r.type == "bool":
                    out[r.id] = Prediction(id=r.id, prob=float(np.clip(p[0], 0.0, 1.0)), confidence=conf, latency_ms=per_q)
                else:
                    p = p / p.sum()
                    out[r.id] = Prediction(id=r.id, probs=p.tolist(), confidence=conf, latency_ms=per_q)
            done += len(chunk)
            if progress:
                print(f"\r  {predictor.name}: {done}/{len(records)}", end="", flush=True)
    if progress:
        print()
    return [out[r.id] for r in records]
