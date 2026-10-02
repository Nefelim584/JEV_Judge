"""Laya zero-shot baseline (``convaiinnovations/laya``, Apache 2.0; TODO 1.8).

Uses Laya's own ``rl_common.py`` from the Hub snapshot for the sequence layout and the model, and its
fitted temperatures. It mirrors ``RLAgent.system_one`` but keeps full-precision probabilities: the
shipped API rounds them to 4 decimals, which distorts NLL for confident answers.

Mapping of our primitives: Choice → ``choice`` with the options as criteria, Score → ``score`` with
the levels, Bool → ``noul`` (P(yes) = probability of its "true" option).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ..schema import BoolQuestion, ChoiceQuestion, Question, ScoreQuestion, State
from .base import pick_device

REPO = "convaiinnovations/laya"
CHECKPOINTS = {"base": "", "multilingual": "multilingual", "typed-decisions": "typed-decisions"}


def to_laya_question(question: Question) -> dict:
    """Our question → Laya's internal question dict (``t`` / ``ins`` / ``crit``)."""
    if isinstance(question, ChoiceQuestion):
        return {"t": "choice", "ins": question.prompt, "crit": {o: None for o in question.options}}
    if isinstance(question, ScoreQuestion):
        return {"t": "score", "ins": question.prompt, "crit": list(question.levels)}
    if isinstance(question, BoolQuestion):
        return {"t": "noul", "ins": question.prompt, "crit": None}
    raise TypeError(f"unsupported question type {type(question).__name__}")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def download(checkpoint: str = "base", cache_dir: str | None = None) -> Path:
    """Fetch one checkpoint (≈1.7 GB for base) and Laya's code; returns the checkpoint directory."""
    from huggingface_hub import snapshot_download

    sub = CHECKPOINTS[checkpoint]
    prefix = f"{sub}/" if sub else ""
    files = [f"{prefix}rl_agent_config.json", f"{prefix}model.safetensors", f"{prefix}tokenizer/*", f"{prefix}encoder/*"]
    root = Path(snapshot_download(REPO, allow_patterns=[*files, "rl_common.py", "rl_agent_api.py"], cache_dir=cache_dir))
    return root / sub if sub else root


class LayaZeroShot:
    def __init__(self, checkpoint: str = "base", device: str | None = None, model_dir: str | Path | None = None):
        import torch
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        model_dir = Path(model_dir) if model_dir else download(checkpoint)
        code_dir = model_dir if (model_dir / "rl_common.py").exists() else model_dir.parent
        self.rl = _load_module(code_dir / "rl_common.py", "laya_rl_common")

        self.name = f"laya:{checkpoint}"
        self.cfg = json.loads((model_dir / "rl_agent_config.json").read_text())
        self.device = pick_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
        self.model = self.rl.build_model(self.cfg, encoder_dir=str(model_dir / "encoder"))
        self.model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
        self.model.to(self.device).eval()
        self.model.encoder.config.reference_compile = False
        self.temperature = self.cfg.get("temperature", [1.0, 1.0, 1.0])
        self.temperature_by_options = self.cfg.get("temperature_by_options", {})
        self.amp_dtype = self.rl.amp_dtype(self.cfg.get("amp_dtype", "fp16"))
        if self.device.type == "cuda" and torch.cuda.get_device_capability(self.device)[0] < 8:
            self.amp_dtype = torch.float16

    def _temperature(self, qtype: int, k: int) -> float:
        return self.temperature_by_options.get(self.rl.temp_bucket(qtype, k), self.temperature[qtype])

    def predict(self, state: State, questions: Sequence[Question]) -> list[np.ndarray]:
        import torch

        rl, items, qs = self.rl, [], [to_laya_question(q) for q in questions]
        for question, q in zip(questions, qs, strict=True):
            ids, markers = rl.build_sequence(self.tokenizer, state, q, self.cfg["max_len"], self.cfg["head_max_len"])
            if len(markers) != len(rl.render_options(q)):
                raise ValueError(f"question {question.id!r}: options do not fit in Laya's head_max_len={self.cfg['head_max_len']}")
            items.append({"ids": ids, "markers": markers, "qtype": rl.QTYPES[q["t"]], "target": [0.0] * len(markers),
                          "label": -1, "episode": 0, "ep_step": 0, "ep_len": 1, "src": "jevlite"})
        b = rl.collate_items([items], self.tokenizer.pad_token_id)
        with torch.no_grad(), torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.device.type == "cuda"):
            logits, _ = self.model(
                b["input_ids"].to(self.device),
                b["attention_mask"].to(self.device),
                b["marker_pos"].to(self.device),
                b["marker_mask"].to(self.device),
                b["qtype"].to(self.device),
            )
        logits = logits.float().cpu().numpy()

        results = []
        for r, (item, q) in enumerate(zip(items, qs, strict=True)):
            k = len(item["markers"])
            z = logits[r, :k] / self._temperature(item["qtype"], k)
            p = np.exp(z - z.max())
            p /= p.sum()
            results.append(np.array([p[1]]) if q["t"] == "noul" else p)
        return results
