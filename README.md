# jev-lite

A typed decision model: given a **state** (text, JSON or an array of texts) and typed **questions**
(Choice, Score, Bool), it returns answers from the caller's schema with calibrated probabilities and a
confidence score. No text generation. Not affiliated with TypeSafe AI. See `todo.md` for the full design and plan.

## Setup

Local (Apple Silicon or any machine with [uv](https://docs.astral.sh/uv/)):

```bash
uv sync --all-groups          # core deps + pytest / jupyter
uv run pytest
```

Colab / Kaggle (CUDA): keep the preinstalled torch and install the package on top:

```bash
git clone <repo-url> jev-lite && cd jev-lite
pip install -e ".[cuda]"
```

## Layout

| Path | What |
|---|---|
| `configs/base.yaml` | Shared experiment config: model, encoding, tuning, schedule, losses |
| `configs/{apple_silicon,cuda}.yaml` | Stack overrides. Only `runtime`, `env`, `tuning.mode`, `train.per_device_batch_size` are allowed, which `load_config` enforces |
| `src/jevlite/` | Shared library code. Notebooks import it and never copy it |
| `notebooks/{apple_silicon,cuda}/` | Training notebooks. Twins have the same cells and differ only in the first code cell |
| `tests/` | Unit tests |

## Notebooks

`00_env_check.ipynb` checks the device, runs the sample request end to end on the device, and runs
LoRA and full fine-tuning train steps at training shape, reporting peak memory.

```bash
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/apple_silicon/00_env_check.ipynb
```

## Evaluation and baselines

Data and predictions are JSONL files in the unified format (`jevlite.data.unified`).

```bash
# zero-shot baselines → predictions
uv run python scripts/predict_baseline.py --baseline nli  --data data/test.jsonl --out preds/nli.jsonl
uv run python scripts/predict_baseline.py --baseline laya --data data/test.jsonl --out preds/laya.jsonl
# report per primitive and slice (JSON + markdown), with bootstrap CIs
uv run python scripts/eval.py --data data/test.jsonl --pred preds/nli.jsonl --out reports/nli --bootstrap 1000
# black-box checks: option-order sensitivity, IIA, confidence on nonsense inputs
uv run python scripts/probe_blackbox.py --baseline laya --data data/test.jsonl --out reports/laya_probes.json
# token lengths of exported judge triples (decides whether long-context Stage C is needed)
uv run python scripts/judge_length_stats.py --data data/judge_export.jsonl --out reports/judge_lengths
```
