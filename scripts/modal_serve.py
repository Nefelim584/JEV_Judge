"""jev-lite as a live judge endpoint on Modal (RTX PRO 6000, bf16 weights + FlashAttention 2).

    uv run --extra serve modal deploy scripts/modal_serve.py                                   # v0.2.0, scales to zero
    JEVLITE_MIN_CONTAINERS=1 uv run --extra serve modal deploy scripts/modal_serve.py          # demo: one container kept warm
    JEVLITE_MODEL=vol:runs/stage_c-full-s42/final uv run --extra serve modal deploy scripts/modal_serve.py   # a Stage C model
    uv run --extra serve modal app stop jevlite-serve                                          # after the demo

Endpoints (Modal proxy auth: every call needs the headers ``Modal-Key`` / ``Modal-Secret`` of a proxy auth
token, created in the Modal dashboard under Settings → Proxy Auth Tokens; the browser never calls them, the
demo's local server does):

- ``POST …-judge.modal.run``: body = ``jevlite.schema.Request`` (state + typed questions), reply =
  ``jevlite.schema.Response`` plus ``server_ms`` (time inside the handler: parsing, model, serialisation).
  ``meta.latency_ms`` is the model alone (tokenisation + forward + calibration).
- ``POST …-judge-stand.modal.run``: the eval stand's criteria for one answer, for a Stage C model. Body:
  ``{"question", "answer", "guard_allowed": true|false|null, "chunks": [...], "history": [{"user", "assistant"}, …]}``
  — the answer and the chunks exactly as the assistant returned them (the stand logs them raw, HTML
  included, and Stage C was trained on that), ``guard_allowed`` as the stand logs it (any guarded query
  allowed). The records are built by ``jevlite.data.stand.live_records``, the same code as the stand's
  test records. Reply: per criterion ``level`` (1–5), ``label``, ``probs``, ``expected``; ``skipped``
  (criteria that read the chunks, for an allowed question without any); ``grounding`` (with chunks only):
  raw P(supported) of the whole answer against the chunks, ``claim`` in the Stage B wording of the criteria
  bench and ``documents`` in the stand's wording — the Score criteria miss answers that contradict the
  chunks, these claims are the check for that; ``model_ms``, ``server_ms``. Needs
  the criteria spec (company data) in the volume, read at container start:
  ``modal volume put jevlite-stage-c data/stand/criteria.json /stand/criteria.json``.
- ``GET …-health.modal.run``: the model in use and one timed dummy call; call it before a demo to wake the
  container.

The URLs are printed by ``modal deploy``.

Settings are read at deploy time from the environment and baked into the image:

- ``JEVLITE_MODEL``: a ``save_final`` folder, ``hub:<repo>[@rev][/dir]`` or ``vol:<path>`` in the private
  volume ``jevlite-stage-c`` (default ``hub:Nefelim584/jev-lite@v0.2.0``).
- ``JEVLITE_LAYERS``: calibration layers, default ``model,judge``: the judge layer is the one for
  faithfulness claims (README); it applies to every Bool question of a request, so send only faithfulness
  Bools here, or deploy a second app with ``model``.
- ``JEVLITE_MIN_CONTAINERS`` (default 0): 1 keeps a PRO 6000 running (and billed) all the time; set it for
  the demo window only. ``scaledown_window`` keeps a container for 15 min after the last call either way.
- ``JEVLITE_GPU`` (default ``RTX-PRO-6000``).
- ``JEVLITE_GROUNDING_MODEL`` (default: the main model): a second model, in the same container and on the same
  GPU, for the ``grounding`` claims of ``judge_stand``. Stage C trusts answers in the stand's style more
  than Stage D does (a probe of wrong details on one question, 2026-11-13: D caught a wrong title at 0.05,
  C gave 0.80), so the demo uses ``vol:runs/stage_d-full-s42/final`` here.

One container at most (``max_containers=1``): calls queue on the warm GPU instead of starting a second
cold container. A cold start (image + model load + kernels) takes on the order of a minute.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import modal

HF_CACHE = "/root/.cache/huggingface"
VOL = "/vol"
FA2 = "kernels-community/flash-attn2"

MODEL = os.environ.get("JEVLITE_MODEL", "hub:Nefelim584/jev-lite@v0.2.0")
GROUNDING_MODEL = os.environ.get("JEVLITE_GROUNDING_MODEL", "")
LAYERS = os.environ.get("JEVLITE_LAYERS", "model,judge")
MIN_CONTAINERS = int(os.environ.get("JEVLITE_MIN_CONTAINERS", "0"))
GPU = os.environ.get("JEVLITE_GPU", "RTX-PRO-6000")
STAND_BATCH = 8  # the 4 criteria of one answer: two states (with / without the chunks), one call

app = modal.App("jevlite-serve")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(".", extras=["serve"], frozen=True)
    .env({"HF_HOME": HF_CACHE, "TOKENIZERS_PARALLELISM": "false", "JEVLITE_MODEL": MODEL, "JEVLITE_LAYERS": LAYERS,
          "JEVLITE_GROUNDING_MODEL": GROUNDING_MODEL})
    .add_local_python_source("jevlite")
)
hf_cache = modal.Volume.from_name("jevlite-hf-cache", create_if_missing=True)
stage_c = modal.Volume.from_name("jevlite-stage-c", create_if_missing=True)

WARMUP = {
    "state": ["The gateway rejects a limit order without a price.", "Rate limits are applied per API key."],
    "questions": [{"id": "q", "type": "bool", "prompt": 'The answer to the question "Are rate limits per key?" is: yes'}],
}


@app.cls(image=image, gpu=GPU, cpu=4.0, memory=16384, volumes={HF_CACHE: hf_cache, VOL: stage_c},
         min_containers=MIN_CONTAINERS, max_containers=1, scaledown_window=15 * 60, timeout=120)
class JevLiteJudge:
    @modal.enter()
    def load(self):
        import torch

        from jevlite.calibration import load_calibration
        from jevlite.checkpoint import load_final
        from jevlite.encoding import text_encoder_from_config
        from jevlite.inference import Predictor
        from jevlite.model import resolve_checkpoint
        from jevlite.schema import Request

        stage_c.reload()
        layers = tuple(x for x in os.environ["JEVLITE_LAYERS"].split(",") if x)
        warm = Request.model_validate(WARMUP)

        def load(spec: str):
            path = f"{VOL}/{spec.removeprefix('vol:').lstrip('/')}" if spec.startswith("vol:") else resolve_checkpoint(spec)
            # bf16 weights + FA2 from the Hub kernel: the bf16-fa2 setting of the Phase 9 speed bench
            model, tokenizer, cfg = load_final(path, device="cuda", attn_implementation=FA2)
            model = model.to(torch.bfloat16)
            predictor = Predictor(model, text_encoder_from_config(cfg, tokenizer, None), name=spec,
                                  calibration=load_calibration(path), layers=layers)
            for _ in range(5):
                predictor.predict(warm)
            return predictor, model, tokenizer

        spec = os.environ["JEVLITE_MODEL"]
        t0 = time.perf_counter()
        self.predictor, model, tokenizer = load(spec)
        self.grounding_spec = os.environ.get("JEVLITE_GROUNDING_MODEL") or spec
        self.grounding = self.predictor if self.grounding_spec == spec else load(self.grounding_spec)[0]
        hf_cache.commit()
        # the stand's criteria spec (company data): only in the private volume, read for judge_stand
        self.spec, self.count_tokens = None, None
        spec_path = Path(VOL) / "stand" / "criteria.json"
        if spec_path.exists():
            from jevlite.data.stand import CriteriaSpec, live_records

            self.spec = CriteriaSpec.load(spec_path)
            self.count_tokens = lambda text: len(tokenizer(text, add_special_tokens=False).input_ids)  # noqa: E731
            for _ in range(3):
                self.predictor.predict_records(live_records("Warm-up?", "Warm-up.", True, ["Warm-up chunk."], self.spec,
                                                            count_tokens=self.count_tokens), STAND_BATCH)
        self.info = {"model": spec, "layers": list(self.predictor.layers), "gpu": torch.cuda.get_device_name(0),
                     "attn": getattr(model.encoder.config, "_attn_implementation", None),
                     "stand_criteria": [c.name for c in self.spec.criteria] if self.spec else None,
                     "grounding_model": self.grounding_spec,
                     "load_s": round(time.perf_counter() - t0, 1)}
        print(f"ready: {self.info}", flush=True)

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def judge(self, body: dict) -> dict:
        from jevlite.schema import Request

        t0 = time.perf_counter()
        response = self.predictor.predict(Request.model_validate(body))
        out = response.model_dump(mode="json")
        out["server_ms"] = (time.perf_counter() - t0) * 1000
        return out

    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def judge_stand(self, body: dict) -> dict:
        """The stand's criteria for one answer, as a Stage C model was trained on them (``live_records``)."""
        from fastapi import HTTPException

        from jevlite.data.stand import live_records

        if self.spec is None:
            raise HTTPException(503, "no criteria spec in the volume (/stand/criteria.json): upload it, redeploy")
        t0 = time.perf_counter()
        history = [h if isinstance(h, str) else f"User: {h['user']}\nAssistant: {h['assistant']}"
                   for h in body.get("history") or []]
        records = live_records(str(body["question"]), str(body["answer"]), body.get("guard_allowed"),
                               list(body.get("chunks") or []), self.spec, history=history,
                               count_tokens=self.count_tokens)
        t1 = time.perf_counter()
        preds = self.predictor.predict_records(records, STAND_BATCH)
        model_ms = (time.perf_counter() - t1) * 1000
        criteria = {}
        for r, p in zip(records, preds, strict=True):
            probs = list(p.probs)
            k = max(range(len(probs)), key=probs.__getitem__)
            criteria[r.criterion] = {"level": k + 1, "label": r.candidates[k], "probs": probs,
                                     "expected": sum((i + 1) * q for i, q in enumerate(probs)),
                                     "uses_context": r.uses_context, "history_trimmed": r.field("history_trimmed")}
        # Grounding of the whole answer against the chunks, as Bool claims (the Stage B wording of the
        # criteria bench, and the stand's own weak faithfulness wording). The Score criteria follow the
        # judge's mostly-4-5 distribution and miss answers that contradict the chunks; these claims do not
        # depend on it. Raw probabilities (no calibration for Stage C yet).
        grounding = None
        chunks = [c.strip() for c in body.get("chunks") or [] if c and c.strip()]
        if chunks:
            from jevlite.data.stand import FAITHFULNESS_TEMPLATE
            from jevlite.schema import Request

            q, a = str(body["question"]).strip(), str(body["answer"]).strip()
            t2 = time.perf_counter()
            resp = self.grounding.predict(Request.model_validate({"state": chunks, "questions": [
                {"id": "claim", "type": "bool", "prompt": f'The answer to the question "{q}" is: {a}'},
                {"id": "documents", "type": "bool", "prompt": FAITHFULNESS_TEMPLATE.format(question=q, answer=a)},
            ]}))
            grounding = {k: v.probability for k, v in resp.answers.items()}
            grounding.update(model=self.grounding_spec, truncated=resp.meta.truncated,
                             model_ms=(time.perf_counter() - t2) * 1000)
        return {"model": self.info["model"], "criteria": criteria,
                "skipped": [c.name for c in self.spec.criteria if c.name not in criteria],
                "grounding": grounding, "model_ms": model_ms, "server_ms": (time.perf_counter() - t0) * 1000}

    @modal.fastapi_endpoint(method="GET", requires_proxy_auth=True)
    def health(self) -> dict:
        from jevlite.schema import Request

        response = self.predictor.predict(Request.model_validate(WARMUP))
        return {**self.info, "latency_ms": response.meta.latency_ms}
