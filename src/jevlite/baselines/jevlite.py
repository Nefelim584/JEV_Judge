"""Our own trained model behind the baseline interface (TODO Phase 6, Stage A evaluation; Phase 8).

Loads a ``save_final`` folder (``final/`` of a run), so ``probe_blackbox.py`` and ``predict_baseline.py``
run on jev-lite exactly as on the zero-shot baselines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..checkpoint import load_final
from ..encoding import text_encoder_from_config
from ..inference import Predictor, _normalise
from ..schema import Question, State
from .base import pick_device


class JevLiteModel:
    def __init__(self, path: str | Path, device: str | None = None, max_len: int | None = None):
        path = Path(path)
        self.device = pick_device(device)
        model, tokenizer, cfg = load_final(path, device=self.device)
        # max_len None: the encoding.max_len the model was saved with (the training length of its last stage)
        self.predictor = Predictor(model, text_encoder_from_config(cfg, tokenizer, max_len), name=f"jevlite:{path.parent.name or path.name}")
        self.name = self.predictor.name

    def predict(self, state: State, questions: Sequence[Question]) -> list[np.ndarray]:
        rows, _ = self.predictor._probs([(state, q) for q in questions])
        return [np.asarray(p if q.type == "bool" else _normalise(p), dtype=float) for q, p in zip(questions, rows, strict=True)]
