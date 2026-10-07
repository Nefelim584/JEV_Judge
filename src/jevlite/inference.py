"""Inference: ``predict(request) -> response`` (TODO 2.2, 2.8).

    predictor = Predictor(model, text_encoder_from_config(cfg, tokenizer))
    response = predictor.predict(request)                    # one forward pass for all questions
    preds = predictor.predict_records(records)               # unified records → eval predictions

All questions of a request go into one batch; temperatures are applied; options keep the caller's
order (no shuffling at inference). Confidence is the v1 heuristic ``1 − H(p) / log K`` (Bool: K = 2);
the learned confidence head (v2) comes in Phase 7.

The answer only ever names the request's own candidates, so it always lies inside the schema.
"""

from __future__ import annotations

import math
import time
from typing import Iterable, Sequence

import torch

from .collate import TYPE_IDS, build_batch
from .data.unified import Prediction, Record
from .encoding import PackedEncoder, PairEncoder
from .model import JevLite, question_probs
from .schema import BoolAnswer, ChoiceAnswer, Meta, Question, Request, Response, ScoreAnswer


def heuristic_confidence(probs: Sequence[float]) -> float:
    """``1 − H(p) / log K`` over a distribution of K ≥ 1 entries; 1 for a single candidate."""
    p = torch.tensor(probs, dtype=torch.float64)
    k = p.numel()
    if k < 2:
        return 1.0
    h = -(p * p.clamp_min(1e-12).log()).sum()
    return float((1.0 - h / math.log(k)).clamp(0.0, 1.0))


def _normalise(p: list[float]) -> list[float]:
    s = sum(p)
    return [x / s for x in p]


def answer_for(question: Question, probs: Sequence[float]):
    """The typed answer of one question from its ``K`` probabilities (Bool: ``[P(true)]``)."""
    if question.type == "bool":
        p = float(probs[0])
        return BoolAnswer(probability=p, confidence=heuristic_confidence([p, 1.0 - p]))
    p = _normalise([float(x) for x in probs])
    best = max(range(len(p)), key=p.__getitem__)
    dist = dict(zip(question.candidates, p, strict=True))
    conf = heuristic_confidence(p)
    if question.type == "score":
        expected = sum(i * x for i, x in enumerate(p))
        return ScoreAnswer(value=question.candidates[best], probabilities=dist, confidence=conf,
                           expected_level=min(max(expected, 0.0), len(p) - 1))
    return ChoiceAnswer(value=question.candidates[best], probabilities=dist, confidence=conf)


class Predictor:
    def __init__(self, model: JevLite, encoder: PackedEncoder | PairEncoder, name: str = "jev-lite-v1", amp_dtype: torch.dtype | None = None):
        self.model = model.eval()
        self.encoder = encoder
        self.name = name
        self.amp_dtype = amp_dtype

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @torch.no_grad()
    def _probs(self, items) -> tuple[list[list[float]], list[bool]]:
        """Per question: its real-candidate probabilities, and the truncation flag."""
        batch = build_batch(items, self.encoder).to(self.device)
        with torch.autocast(self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            grouped = self.model.grouped_logits(batch)
        probs = question_probs(grouped, batch.question_type).cpu()
        n_real = batch.candidate_mask.sum(-1).tolist()
        is_bool = (batch.question_type == TYPE_IDS["bool"]).tolist()
        rows = [probs[q, : 1 if b else k].tolist() for q, (k, b) in enumerate(zip(n_real, is_bool))]
        return rows, batch.truncated.cpu().tolist()

    def predict(self, request: Request) -> Response:
        start = time.perf_counter()
        rows, truncated = self._probs([(request.state, q) for q in request.questions])
        answers = {q.id: answer_for(q, p) for q, p in zip(request.questions, rows, strict=True)}
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        latency = (time.perf_counter() - start) * 1000
        return Response(answers=answers, meta=Meta(model=self.name, latency_ms=latency, truncated=any(truncated)))

    def predict_records(self, records: Iterable[Record], batch_size: int = 16) -> list[Prediction]:
        """Predictions in the format of ``scripts/eval.py``, in record order."""
        out: list[Prediction] = []
        records = list(records)
        for i in range(0, len(records), batch_size):
            chunk = records[i : i + batch_size]
            questions = [r.to_question(f"{j}") for j, r in enumerate(chunk)]
            rows, _ = self._probs([(r.state, q) for r, q in zip(chunk, questions)])
            for r, p in zip(chunk, rows, strict=True):
                if r.type == "bool":
                    out.append(Prediction(id=r.id, prob=p[0], confidence=heuristic_confidence([p[0], 1 - p[0]])))
                else:
                    p = _normalise(p)
                    out.append(Prediction(id=r.id, probs=p, confidence=heuristic_confidence(p)))
        return out
