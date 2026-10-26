# jev-lite

[![Tests](https://github.com/Nefelim584/JEV_Judge/actions/workflows/tests.yml/badge.svg)](https://github.com/Nefelim584/JEV_Judge/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20model-jev--lite--stage--a-yellow)](https://huggingface.co/Nefelim584/jev-lite-stage-a)

A typed decision model, built to replace an LLM-as-a-judge on narrow checks. Given a **state** (text, a JSON
object or a list of texts) and typed **questions**, it returns answers from the caller's own schema with
calibrated probabilities and a confidence score. No text generation, so the answer always fits the schema.

| Type | Question | Output |
|---|---|---|
| `bool` | a statement about the state | P(true) + confidence |
| `choice` | a prompt + 1–32 options | distribution over the options + argmax + confidence |
| `score` | a prompt + 1–32 ordered levels | distribution over the levels + argmax + expected level + confidence |

The model is [`answerdotai/ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) (395M) with a
packed encoding (one `[MASK]` marker per candidate) and three small heads. English only. Not affiliated with TypeSafe AI.

## Status: Stage A

The first checkpoint, trained on 25 public datasets, is on the Hub:
[`Nefelim584/jev-lite-stage-a`](https://huggingface.co/Nefelim584/jev-lite-stage-a). Against zero-shot baselines
(an NLI cross-encoder and Laya), on cleaned test sets:

| Accuracy: NLI / Laya / **jev-lite** | trained sources | held-out templates | new sources |
|---|---|---|---|
| Bool | 0.763 / 0.723 / **0.882** | 0.569 / 0.770 / **0.902** | 0.542 / 0.519 / **0.603** |
| Choice | 0.595 / 0.596 / **0.803** | 0.617 / 0.622 / **0.794** | 0.558 / 0.585 / **0.635** |
| Score | 0.242 / 0.309 / **0.620** | 0.234 / 0.315 / **0.591** | — |

The lead is large on trained sources and shrinks on new ones, where calibration also breaks. Full reports:

| Report | What |
|---|---|
| [`STAGE_A.md`](STAGE_A.md) | Stage A vs the baselines: per source, calibration, selective prediction, probes, limitations |
| [`STAGE_A_API.md`](STAGE_A_API.md) | Stage A vs the JEV API, latency |
| [`JUDGE_BENCH.md`](JUDGE_BENCH.md) | Stage A on four judge benchmarks (ContextualJudgeBench, RAGBench, HaluBench, FaithBench), 512 vs 8192 tokens |
| [`BASELINES.md`](BASELINES.md) | Zero-shot baselines and the test sets |
| [`DATASETS.md`](DATASETS.md) | Dataset candidates and license checks |
| [`judge_labeling_guide.md`](judge_labeling_guide.md) | Labelling guide for the judge data |

## Quick start

```python
from jevlite.checkpoint import load_final
from jevlite.encoding import text_encoder_from_config
from jevlite.inference import Predictor
from jevlite.schema import Request

model, tokenizer, cfg = load_final("path/to/final", device="cpu")   # encoder/ + heads.pt + jevlite.yaml
predictor = Predictor(model, text_encoder_from_config(cfg, tokenizer))

response = predictor.predict(Request(
    state="The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
    questions=[
        {"id": "faithful", "type": "bool", "prompt": "The text states that the tower was finished in 1889."},
        {"id": "topic", "type": "choice", "prompt": "What is the text about?",
         "options": ["architecture", "sports", "finance"]},
    ],
))
print(response.answers["faithful"].probability, response.answers["topic"].value)
```

## Setup

Local (Apple Silicon or any machine with [uv](https://docs.astral.sh/uv/)):

```bash
uv sync --all-groups          # core deps + pytest / jupyter
uv run pytest
```

Colab / Kaggle (CUDA): keep the preinstalled torch and install the package on top:

```bash
git clone https://github.com/Nefelim584/JEV_Judge.git && cd JEV_Judge
pip install -e ".[cuda]"
```

## Layout

| Path | What |
|---|---|
| `src/jevlite/` | Library code: schema, encoding, model, losses, training, inference, metrics, baselines, reports. Notebooks import it and never copy it |
| `configs/base.yaml` | Shared experiment config: model, encoding, tuning, schedule, losses |
| `configs/{apple_silicon,cuda}.yaml` | Stack overrides. Only `runtime`, `env`, `tuning.mode`, `train.per_device_batch_size` are allowed, which `load_config` enforces |
| `configs/mixes/` | Training mixes (`stage_a`, `stage_a_nosa`) |
| `scripts/` | Data conversion, mix building, training, prediction, evaluation, reports |
| `notebooks/{apple_silicon,cuda}/` | Environment checks, training and benchmarks. Twins have the same cells and differ only in the first code cell |
| `docs/` | Charts and generated reports behind the Markdown reports |
| `tests/` | Unit tests |

## Workflow

`notebooks/*/00_env_check.ipynb` checks the device, runs a sample request end to end, and runs LoRA and
full fine-tuning train steps at training shape, reporting peak memory.

### Data

Converters for the approved datasets (`DATASETS.md`, section 0) write one JSONL file and one stats file per
dataset. Raw downloads are cached in `data/raw`.

```bash
# smoke run: the first 200 rows of every source split, HF datasets streamed
uv run python scripts/convert_public.py --datasets all --limit 200 --out data/smoke
# full conversion (TabFact downloads a ~770 MB archive on the first run)
uv run python scripts/convert_public.py --datasets all --out data/public
# training mix: caps per source, primitive weights, eval subsets
uv run python scripts/build_mix.py --config configs/mixes/stage_a.yaml --out data/mix/stage_a --epochs 2
```

### Training

Stage A was trained with `notebooks/cuda/10_train_stage_a_public.ipynb` on Kaggle (2×T4, about 9.5 h).
The notebook wraps `scripts/train.py`; the recipe is in `configs/base.yaml` and `STAGE_A.md` section 1.

### Evaluation and baselines

Data and predictions are JSONL files in the unified format (`jevlite.data.unified`).

```bash
# zero-shot baselines → predictions
uv run python scripts/predict_baseline.py --baseline nli  --data data/test.jsonl --out preds/nli.jsonl
uv run python scripts/predict_baseline.py --baseline laya --data data/test.jsonl --out preds/laya.jsonl
# report per primitive and slice (JSON + markdown), with bootstrap CIs
uv run python scripts/eval.py --data data/test.jsonl --pred preds/nli.jsonl --out reports/nli --bootstrap 1000
# paired comparison: bootstrap CI of Δ accuracy + McNemar, Holm-corrected
uv run python scripts/compare_paired.py --data data/test.jsonl --model jevlite=preds/jevlite.jsonl \
    --ref nli=preds/nli.jsonl --ref laya=preds/laya.jsonl --out reports/paired
# black-box checks: option-order sensitivity, IIA, confidence on nonsense inputs
uv run python scripts/probe_blackbox.py --baseline laya --data data/test.jsonl --out reports/laya_probes.json
```

### Reports with charts

Markdown reports with matplotlib charts and an interactive plotly twin of each (`uv sync --extra reports`):

```bash
# compare eval reports; repeat a name for seeds, use name@x for learning-curve points
uv run python scripts/make_report.py compare --out reports/baselines \
    --run nli=reports/nli/report.json --run laya=reports/laya/report.json
# training runs (metrics.jsonl written by jevlite.reports.training.TrainingLog)
uv run python scripts/make_report.py training --out reports/stage_a --run full=runs/stage_a_full
```

## License

Code and model weights: [Apache-2.0](LICENSE). The training datasets keep their own licenses; see
[`DATASETS.md`](DATASETS.md) and the [model card](https://huggingface.co/Nefelim584/jev-lite-stage-a).
