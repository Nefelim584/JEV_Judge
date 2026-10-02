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
    ):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.name = f"nli:{model_name}"
        self.device = pick_device(device)
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.model = (model or AutoModelForSequenceClassification.from_pretrained(model_name, dtype=torch.float32)).to(self.device).eval()
        self.entail = entailment_index(self.model.config.id2label)
        self.max_len = max_len
        self.batch_size = batch_size
        self.choice_template = choice_template
        self.score_template = score_template

    def _logits(self, premise: str, hyps: list[str]) -> np.ndarray:
        import torch

        out = []
        for start in range(0, len(hyps), self.batch_size):
            chunk = hyps[start : start + self.batch_size]
            enc = self.tokenizer(
                [premise] * len(chunk), chunk, truncation="only_first", max_length=self.max_len, padding=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out.append(self.model(**enc).logits.float().cpu().numpy())
        return np.concatenate(out)

    def predict(self, state: State, questions: Sequence[Question]) -> list[np.ndarray]:
        premise = serialize_state(state)
        per_q = [hypotheses(q, self.choice_template, self.score_template) for q in questions]
        logits = self._logits(premise, [h for hs in per_q for h in hs])

        results, i = [], 0
        for q, hs in zip(questions, per_q, strict=True):
            z = logits[i : i + len(hs)]
            i += len(hs)
            if isinstance(q, BoolQuestion):
                p = np.exp(z[0] - z[0].max())
                results.append(np.array([p[self.entail] / p.sum()]))
            else:
                e = z[:, self.entail]
                p = np.exp(e - e.max())
                results.append(p / p.sum())
        return results
