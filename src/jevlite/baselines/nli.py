"""Zero-shot NLI baseline: entailment scoring with an off-the-shelf NLI cross-encoder, no training.

- Bool: premise = state, hypothesis = claim; P(yes) = P(entailment). "Neutral" and "contradiction"
  both count as "no", which matches the faithfulness semantics ("not in the chunk" → not supported).
- Choice / Score: one hypothesis per candidate from a template; the candidates' entailment logits
  go through a softmax over the question, as in the Hugging Face zero-shot pipeline.

Works with 3-way NLI models (entailment / neutral / contradiction) and with 2-way zero-shot models
(entailment / not_entailment).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..encoding import serialize_state
from ..schema import BoolQuestion, ChoiceQuestion, Question, ScoreQuestion, State
from .base import pick_device

DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
CHOICE_TEMPLATE = 'The answer to "{prompt}" is: {candidate}.'
SCORE_TEMPLATE = 'On a scale from "{low}" to "{high}", the answer to "{prompt}" is: {candidate}.'


def entailment_index(id2label: dict) -> int:
    """Index of the entailment class from a model config (``not_entailment`` is not entailment)."""
    for i, label in id2label.items():
        name = str(label).lower().replace("-", "_")
        if "entail" in name and not name.startswith(("not", "non")):
            return int(i)
    raise ValueError(f"no entailment label in {id2label}")


def hypotheses(question: Question, choice_template: str = CHOICE_TEMPLATE, score_template: str = SCORE_TEMPLATE) -> list[str]:
    if isinstance(question, BoolQuestion):
        return [question.prompt]
    if isinstance(question, ChoiceQuestion):
        return [choice_template.format(prompt=question.prompt, candidate=c) for c in question.options]
    if isinstance(question, ScoreQuestion):
        low, high = question.levels[0], question.levels[-1]
        return [score_template.format(prompt=question.prompt, candidate=c, low=low, high=high) for c in question.levels]
    raise TypeError(f"unsupported question type {type(question).__name__}")


class NLIZeroShot:
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        max_len: int = 512,
        batch_size: int = 16,
        choice_template: str = CHOICE_TEMPLATE,
        score_template: str = SCORE_TEMPLATE,
        model=None,
        tokenizer=None,
        fp16: bool = False,
    ):
        """``fp16``: run the forward pass under fp16 autocast (CUDA only; ~3–4× faster on T4 tensor
        cores, probabilities shift by ~1e-3). Logits are read out in fp32."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = pick_device(device)
        if fp16 and self.device.type != "cuda":
            raise ValueError(f"fp16 is for CUDA only, the device is {self.device.type}")
        self.fp16 = fp16
        self.name = f"nli:{model_name}" + ("+fp16" if fp16 else "")
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.model = (model or AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch.float32)).to(self.device).eval()
        self.entail = entailment_index(self.model.config.id2label)
        self.max_len = max_len
        self.batch_size = batch_size
        self.choice_template = choice_template
        self.score_template = score_template

    def _logits(self, pairs: Sequence[tuple[str, str]]) -> np.ndarray:
        """Logits for (premise, hypothesis) pairs, in input order.

        Pairs from different premises share batches, sorted by length: batches are full even when
        every record has its own state and one hypothesis (Bool), and padding stays small.
        """
        import torch

        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        out: np.ndarray | None = None
        for start in range(0, len(order), self.batch_size):
            idx = order[start : start + self.batch_size]
            enc = self.tokenizer(
                [pairs[i][0] for i in idx], [pairs[i][1] for i in idx],
                truncation="only_first", max_length=self.max_len, padding=True, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.fp16):
                logits = self.model(**enc).logits.float().cpu().numpy()
            if out is None:
                out = np.empty((len(pairs), logits.shape[1]), dtype=np.float32)
            out[idx] = logits
        return out

    def _probs(self, question: Question, z: np.ndarray) -> np.ndarray:
        if isinstance(question, BoolQuestion):
            p = np.exp(z[0] - z[0].max())
            return np.array([p[self.entail] / p.sum()])
        e = z[:, self.entail]
        p = np.exp(e - e.max())
        return p / p.sum()

    def predict_many(self, calls: Sequence[tuple[State, Sequence[Question]]]) -> list[list[np.ndarray]]:
        """``predict`` for several states at once, with all their pairs batched together."""
        per_call = [
            (serialize_state(state), [hypotheses(q, self.choice_template, self.score_template) for q in questions])
            for state, questions in calls
        ]
        logits = self._logits([(premise, h) for premise, per_q in per_call for hs in per_q for h in hs])
        results, i = [], 0
        for (_, per_q), (_, questions) in zip(per_call, calls, strict=True):
            row = []
            for q, hs in zip(questions, per_q, strict=True):
                row.append(self._probs(q, logits[i : i + len(hs)]))
                i += len(hs)
            results.append(row)
        return results

    def predict(self, state: State, questions: Sequence[Question]) -> list[np.ndarray]:
        return self.predict_many([(state, questions)])[0]
