"""Predictions of other models on a mix's eval records, on Modal (baselines for Stage C on the eval stand).

    # jev-lite before Stage C (Stage D, the v0.2.0 release) and an open LLM judge, on the stand's test
    uv run --extra bench modal run scripts/modal_predict.py \\
        --models "stage_d=vol:runs/stage_d-full-s42/final,v020=hub:Nefelim584/jev-lite@v0.2.0,qwen3_8b=llm:Qwen/Qwen3-8B" \\
        --mix stage_c --split test_in --source stand --out data/runs/stage_c-full-s42/baselines

Models (``NAME=SPEC``, comma-separated, one container each, all on ``--gpu``):

- ``vol:<path>``: a ``save_final`` folder on the private volume ``jevlite-stage-c``; ``hub:...``: a released
  one (``jevlite.model.resolve_checkpoint``). Raw outputs (no ``calibration.json``), as in a run's
  ``preds.jsonl``, so they compare with it.
- ``llm:<hub id>``: an open instruction model as a plain LLM-as-a-judge with the record's own short
  prompt (not the production rubric): a Score record → P(level) from the logits of the first answer
  token over the digits ``1``–``k``; a Bool record → P(Yes). No sampling, no thinking.

The records are read in the containers from ``/vol/mix/<mix>/<split>.jsonl`` (company data stays in the
private volume); only the predictions come back (ids and probabilities, the format of ``scripts/eval.py``)
to ``<out>/<name>.preds.jsonl``, with ``<out>/<name>.timing.json``.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import modal

HF_CACHE = "/root/.cache/huggingface"
VOL = "/vol"

app = modal.App("jevlite-predict")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(".", extras=["bench"], frozen=True)
    .env({"HF_HOME": HF_CACHE, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("jevlite")
)
hf_cache = modal.Volume.from_name("jevlite-hf-cache", create_if_missing=True)
data = modal.Volume.from_name("jevlite-stage-c", create_if_missing=True)

LLM_SYSTEM = "You are a careful evaluator of an AI assistant's answers. You judge only by the text you are given."
LLM_SCORE = "{state}\n\n---\n{prompt}\nAnswer with one digit from 1 (lowest) to {k} (highest), nothing else."
LLM_BOOL = "Text:\n<<<\n{state}\n>>>\n\nStatement: {prompt}\n\nIs the statement supported by the text? Answer with one word: Yes or No."


def _state_text(state) -> str:
    return "\n\n".join(state) if isinstance(state, list) else str(state)


def _softmax_groups(z, groups: list[list[int]]) -> list[float]:
    import numpy as np

    z = np.asarray(z, dtype=np.float64)
    lse = [float(np.logaddexp.reduce(z[g])) for g in groups]
    m = max(lse)
    e = [math.exp(v - m) for v in lse]
    s = sum(e)
    return [v / s for v in e]


class LLMPredictor:
    def __init__(self, model: str, max_len: int = 8192, batch_size: int = 4):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from jevlite.baselines.criteria import single_token_ids

        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
        self.max_len, self.batch_size = max_len, batch_size
        self.yes, self.no = single_token_ids(self.tok, ["yes"]), single_token_ids(self.tok, ["no"])
        self.digits = {d: single_token_ids(self.tok, [str(d)]) for d in range(1, 10)}

    def _fit(self, text: str, budget: int) -> str:
        ids = self.tok.encode(text, add_special_tokens=False)
        if len(ids) <= budget:
            return text
        head = budget // 2
        return self.tok.decode(ids[:head]) + "\n[…]\n" + self.tok.decode(ids[-(budget - head):])

    def _chat(self, r) -> str:
        state = self._fit(_state_text(r.state), self.max_len - 400)
        user = (LLM_BOOL.format(state=state, prompt=r.prompt) if r.type == "bool"
                else LLM_SCORE.format(state=state, prompt=r.prompt, k=len(r.candidates)))
        msgs = [{"role": "system", "content": LLM_SYSTEM}, {"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    def _batch(self, recs) -> list[dict]:
        import torch

        enc = self.tok([self._chat(r) for r in recs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        with torch.no_grad():
            logits = self.model(**enc, use_cache=False, logits_to_keep=1).logits[:, -1, :].float().cpu().numpy()
        out = []
        for r, z in zip(recs, logits):
            if r.type == "bool":
                p = _softmax_groups(z, [self.yes, self.no])[0]
                out.append({"id": r.id, "prob": p, "confidence": max(p, 1 - p)})
            else:
                probs = _softmax_groups(z, [self.digits[d] for d in range(1, len(r.candidates) + 1)])
                out.append({"id": r.id, "probs": probs, "confidence": max(probs)})
        return out

    def predict(self, recs) -> list[dict]:
        import torch

        order = sorted(range(len(recs)), key=lambda i: len(_state_text(recs[i].state)))
        res: dict[int, dict] = {}

        def run(ix: list[int]) -> None:
            try:
                for i, p in zip(ix, self._batch([recs[i] for i in ix])):
                    res[i] = p
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(ix) == 1:
                    print(f"{recs[ix[0]].id}: does not fit, left out", flush=True)
                    return
                run(ix[: len(ix) // 2])
                run(ix[len(ix) // 2 :])

        for s in range(0, len(order), self.batch_size):
            run(order[s : s + self.batch_size])
        return [res[i] for i in range(len(recs)) if i in res]


@app.function(image=image, gpu="L40S", cpu=4.0, memory=32768, timeout=3 * 3600, volumes={HF_CACHE: hf_cache, VOL: data})
def predict(name: str, spec: str, mix: str, split: str, source: str, batch_size: int) -> dict:
    import torch

    from jevlite.data.unified import Record, read_jsonl

    data.reload()
    recs = [Record.model_validate(r) for r in read_jsonl(f"{VOL}/mix/{mix}/{split}.jsonl")
            if not source or r.get("source") == source]
    print(f"{name}: {len(recs)} records of {mix}/{split}" + (f" (source {source})" if source else ""), flush=True)
    t0 = time.perf_counter()
    if spec.startswith("llm:"):
        predictor = LLMPredictor(spec.removeprefix("llm:"))
        load_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        preds = predictor.predict(recs)
    else:
        from jevlite.inference import Predictor
        from jevlite.model import resolve_checkpoint

        path = f"{VOL}/{spec.removeprefix('vol:').lstrip('/')}" if spec.startswith("vol:") else str(resolve_checkpoint(spec))
        predictor = Predictor.from_final(path, device="cuda", layers=(), amp_dtype=torch.bfloat16, name=name)
        load_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        preds = [p.model_dump(exclude_none=True) for p in predictor.predict_records(recs, batch_size)]
    pred_s = time.perf_counter() - t1
    hf_cache.commit()
    timing = {"gpu": torch.cuda.get_device_name(0), "records": len(recs), "predicted": len(preds),
              "load_s": load_s, "predict_s": pred_s, "ms_per_record": 1000 * pred_s / max(len(preds), 1)}
    print(f"{name}: {timing}", flush=True)
    return {"preds": preds, "timing": timing}


@app.local_entrypoint()
def main(models: str, mix: str = "stage_c", split: str = "test_in", source: str = "stand", out: str = "",
         gpu: str = "L40S", batch_size: int = 8):
    specs = dict(m.strip().split("=", 1) for m in models.split(",") if m.strip())
    out_dir = Path(out or f"data/runs/predict-{mix}-{split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = {n: predict.with_options(gpu=gpu).spawn(n, s, mix, split, source, batch_size) for n, s in specs.items()}
    failed = []
    for n, call in calls.items():
        try:
            res = call.get()
        except Exception as e:  # noqa: BLE001 — the other models go on
            failed.append(n)
            print(f"{n}: FAILED {type(e).__name__}: {e}")
            continue
        with (out_dir / f"{n}.preds.jsonl").open("w") as f:
            for p in res["preds"]:
                f.write(json.dumps(p) + "\n")
        (out_dir / f"{n}.timing.json").write_text(json.dumps({"spec": specs[n], **res["timing"]}, indent=2))
        print(f"{n}: {len(res['preds'])} predictions → {out_dir / (n + '.preds.jsonl')}")
    if failed:
        raise SystemExit(f"failed: {', '.join(failed)}")
