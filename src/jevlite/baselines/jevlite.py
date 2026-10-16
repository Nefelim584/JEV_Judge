"""Our own trained model behind the baseline interface (TODO Phase 6, Stage A evaluation; Phase 8).

Loads a ``save_final`` folder (``final/`` of a run), so ``probe_blackbox.py`` and ``predict_baseline.py``
run on jev-lite exactly as on the zero-shot baselines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..checkpoint import load_final
from ..encoding import text_encoder_from_config
from ..inference import Predictor, _normalise
from ..schema import Question, State
from .base import pick_device


class JevLiteModel:
    def __init__(self, path: str | Path, device: str | None = None, max_len: int | None = None, fp16: bool = False):
        path = Path(path)
        self.device = pick_device(device)
        # fp16 autocast: CUDA only (Stage A trained with it on T4); on MPS fp16 can overflow (configs/apple_silicon.yaml)
        if fp16 and self.device.type != "cuda":
            raise ValueError(f"fp16 is for CUDA only, the device is {self.device.type}")
        model, tokenizer, cfg = load_final(path, device=self.device)
        # max_len None: the encoding.max_len the model was saved with (the training length of its last stage)
        name = f"jevlite:{path.parent.name or path.name}" + ("+fp16" if fp16 else "")
        self.predictor = Predictor(model, text_encoder_from_config(cfg, tokenizer, max_len), name=name,
                                   amp_dtype=torch.float16 if fp16 else None)
        self.name = self.predictor.name

    def predict(self, state: State, questions: Sequence[Question]) -> list[np.ndarray]:
        rows, _ = self.predictor._probs([(state, q) for q in questions])
        return [np.asarray(p if q.type == "bool" else _normalise(p), dtype=float) for q, p in zip(questions, rows, strict=True)]
