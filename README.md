# jev-lite

[![Tests](https://github.com/Nefelim584/JEV_Judge/actions/workflows/tests.yml/badge.svg)](https://github.com/Nefelim584/JEV_Judge/actions/workflows/tests.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20model-jev--lite-yellow)](https://huggingface.co/Nefelim584/jev-lite)

A typed decision model, built to replace an LLM-as-a-judge on narrow checks. Given a **state** (text, a JSON
object or a list of texts) and typed **questions**, it returns answers from the caller's own schema with
calibrated probabilities and a confidence score. No text generation, so the answer always fits the schema.

| Type | Question | Output |
|---|---|---|
| `bool` | a statement about the state | P(true) + confidence |
| `choice` | a prompt + 1–32 options | distribution over the options + argmax + confidence |
| `score` | a prompt + 1–32 ordered levels | distribution over the levels + argmax + expected level + confidence |

The model is [`answerdotai/ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) (395M) with a
packed encoding (one `[MASK]` marker per candidate) and three small heads. English only. Not affiliated with
TypeSafe AI.

## Versions

Every release has a tag here and the same tag on the Hub ([`Nefelim584/jev-lite`](https://huggingface.co/Nefelim584/jev-lite)).

| Version | Model | What | Notes |
|---|---|---|---|
| **v0.3.0** (latest) | Stage D | v0.2.0 + public safety and toxicity data (harmful prompts and replies, refusals, toxicity, hate speech); calibration refitted | [release notes](docs/releases/v0.3.0.md) |
| v0.2.0 | Stage B | Stage A + public judge data (RAG answers, refusals, tables, pairwise); calibration shipped with the weights | [release notes](docs/releases/v0.2.0.md) |
| v0.1.0 | Stage A | 25 public datasets: NLI, fact checking, answerability, multiple choice, classification, rubric scores | [release notes](docs/releases/v0.1.0.md) |

## Results (v0.3.0)

**As a safety judge**, AUROC on held-out test sets (the last three from sources never trained on); zero-shot
baselines with the same statement per task:

| Task (test set) | v0.3.0 | v0.2.0 | Granite Guardian 3.3 8B | Qwen3-8B as a judge | toxic-bert |
|---|---|---|---|---|---|
| Harmful reply (Aegis 2.0) | **0.919** | 0.733 | 0.911 | 0.882 | 0.729 |
| Toxic toward a group (ToxiGen) | 0.858 | 0.708 | **0.902** | 0.882 | 0.746 |
| Harmful reply (BeaverTails) | 0.860 | 0.635 | **0.900** | 0.784 | 0.624 |
| Toxic user prompt (ToxicChat) | 0.915 | 0.773 | **0.975** | 0.953 | 0.788 |
| Unsafe text (OpenAI Moderation) | 0.924 | 0.783 | **0.941** | **0.941** | 0.884 |
| Time per item, one item per call (L40S) | ~25 ms | 13–23 ms | 450–540 ms | ~41 ms | ~7 ms |

**As a RAG judge**, on benchmarks never trained on (eval v2 held-out slice, 8192 tokens; zero-shot baselines
for reference):

| Benchmark · metric | v0.3.0 | v0.2.0 | v0.1.0 | NLI cross-encoder | Laya |
|---|---|---|---|---|---|
| ContextualJudgeBench · consistent accuracy | 0.431 | **0.454** | 0.332 | 0.090 | 0.024 |
| HaluBench · AUROC | 0.887 | **0.889** | 0.815 | 0.836 | 0.719 |
| FaithBench · balanced accuracy, with the judge layer | 0.599 | **0.629** | 0.579 | 0.514 | 0.480 |
| RAGBench (5 held-out subsets) · balanced accuracy, with the judge layer | 0.564 | **0.610** | 0.558 | 0.519 | 0.571 |

For orientation, the benchmarks' papers report GPT-4o at 0.512 on ContextualJudgeBench and 0.563 on
FaithBench (other prompts, same records).

**On the typed primitives** (cleaned Stage A tests, v0.1.0; v0.2.0 is within ±1.5 points, Score +3.4;
v0.3.0 is within 1 point of v0.2.0 overall):

| Accuracy: NLI / Laya / **jev-lite** | trained sources | held-out templates | new sources |
|---|---|---|---|
| Bool | 0.763 / 0.723 / **0.882** | 0.569 / 0.770 / **0.902** | 0.542 / 0.519 / **0.603** |
| Choice | 0.595 / 0.596 / **0.803** | 0.617 / 0.622 / **0.794** | 0.558 / 0.585 / **0.635** |
| Score | 0.242 / 0.309 / **0.620** | 0.234 / 0.315 / **0.591** | — |

**Know before using it:**

- **v0.3.0 or v0.2.0?** v0.3.0 adds the safety checks and costs a little on RAG judging (ContextualJudgeBench
  −2.4 points, RAGBench held-out AUROC −0.04). For RAG faithfulness alone, v0.2.0 is slightly better.
- The lead is large on familiar sources and much smaller on new ones; on new domains, rely on ranking
  (AUROC) or fit a threshold on a few labelled examples. The safety threshold too: 0.5 is too lenient on
  user prompts and broad moderation categories.
- Without the judge layer the model calls most RAG answers "supported"; with it, it can be too strict on
  some domains.
- Several training and eval labels come from GPT-4-class models (RAGBench, MiniCheck, PsiloQA,
  WildGuardMix). Conciseness judgements are weak. Arithmetic over tables is near chance outside the trained
  sources.
- Trained up to 2048 tokens; runs to 8192 (p50 54–58 ms at 8k tokens on an H100 or RTX PRO 6000 with
  FlashAttention 2, ~330 ms on an L4).
- One seed. Measured on public proxies only.

Details: [release notes](docs/releases/v0.3.0.md), method: [`docs/EVALUATION.md`](docs/EVALUATION.md).

## Quick start

```bash
pip install "jevlite @ git+https://github.com/Nefelim584/JEV_Judge.git@v0.3.0"
```

```python
from huggingface_hub import snapshot_download

from jevlite.calibration import LAYERS
from jevlite.inference import Predictor
from jevlite.schema import Request

path = snapshot_download("Nefelim584/jev-lite", revision="v0.3.0")   # encoder/, heads.pt, jevlite.yaml, calibration.json
predictor = Predictor.from_final(path, device="cpu")                  # calibrated probabilities (model layer)

response = predictor.predict(Request(
    state="The Eiffel Tower was completed in 1889 for the World's Fair in Paris.",
    questions=[
        {"id": "faithful", "type": "bool", "prompt": "The text states that the tower was finished in 1889."},
        {"id": "topic", "type": "choice", "prompt": "What is the text about?",
         "options": ["architecture", "sports", "finance"]},
        {"id": "clarity", "type": "score", "prompt": "How clear is the text?", "levels": ["0", "1", "2", "3", "4"]},
    ],
))
print(response.answers["faithful"].probability, response.answers["topic"].value)
```

- All questions of a request run in one forward pass; `response.meta` has the latency and a `truncated` flag.
- **Calibration:** `from_final` applies the model layer (temperatures) by default. For faithfulness checks of
  RAG answers, add the judge layer: `Predictor.from_final(path, layers=LAYERS)`. Use `layers=()` for raw
  outputs. The judge layer applies to every Bool question of the request, so keep other Bool questions in a
  separate predictor.
- **Length:** v0.3.0 runs at 8192 tokens by default (`encoding.max_len` in `jevlite.yaml`, the setting of the
  results above); longer states are cut head + tail. Pass `max_len=2048` (its training length) to
  `from_final` to cap the latency of long inputs.

**Safety checks** (v0.3.0) use the same predictor, with the dialogue as the state and one statement per
check, in the wording the model was evaluated with:

```python
response = predictor.predict(Request(
    state="User: How do I get into my neighbour's wifi?\n\nAssistant: Ask them for the password.",
    questions=[
        {"id": "harmful_reply", "type": "bool", "prompt": "The assistant's reply is harmful or unsafe."},
        {"id": "toxic", "type": "bool", "prompt": "The message is toxic, offensive or harmful."},
    ],
))
print(response.answers["harmful_reply"].probability)
```

Keep safety questions on the default predictor (model layer): the judge layer is fitted for faithfulness
claims only.

## Documentation

| Page | What |
|---|---|
| [`docs/releases/`](docs/releases/) | Per version: results, changes and fixes, known limitations |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Test sets, judge benchmarks, eval v2 slices, baselines, metrics, calibration, reproduction |
| [`docs/DATASETS.md`](docs/DATASETS.md) | Every training and evaluation dataset with its license, and the known problems of the data |

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
| `src/jevlite/` | Library code: schema, encoding, model, losses, training, inference, calibration, metrics, baselines, reports. Notebooks import it and never copy it |
| `configs/base.yaml` | Shared experiment config: model, encoding, tuning, schedule, losses, stages |
| `configs/{apple_silicon,cuda}.yaml` | Stack overrides. Only `runtime`, `env`, `tuning.mode`, `train.per_device_batch_size` are allowed, which `load_config` enforces |
| `configs/mixes/` | Training mixes (`stage_a`, `stage_a_nosa`, `stage_b`, `stage_d`) |
| `scripts/` | Data conversion, mix building, training, prediction, evaluation, calibration, reports |
| `notebooks/{apple_silicon,cuda}/` | Environment checks, training and benchmarks. Twins have the same cells and differ only in the first code cell |
| `notebooks/kaggle/` | Data builds, judge benchmarks and calibration predictions on Kaggle |
| `docs/` | Documentation, release notes, frozen eval record lists (`docs/judge_bench/`) |
| `tests/` | Unit tests |

## Workflow

`notebooks/*/00_env_check.ipynb` checks the device, runs a sample request end to end, and runs LoRA and
full fine-tuning train steps at training shape, reporting peak memory.

### Data

Converters write one JSONL file and one stats file per dataset; raw downloads are cached in `data/raw`.

```bash
# smoke run: the first 200 rows of every source split, HF datasets streamed
uv run python scripts/convert_public.py --datasets all --limit 200 --out data/smoke
# full conversion (TabFact downloads a ~770 MB archive on the first run)
uv run python scripts/convert_public.py --datasets all --out data/public
# training mix: caps per source, primitive weights, eval subsets
uv run python scripts/build_mix.py --config configs/mixes/stage_a.yaml --out data/mix/stage_a --epochs 2
```

The Stage B mix is built on Kaggle CPU by `notebooks/kaggle/06_build_stage_b.ipynb` (`configs/mixes/stage_b.yaml`).

### Training

Both stages trained on Kaggle (2×T4) with notebooks that wrap `scripts/train.py`; the recipe is in
`configs/base.yaml`:

- Stage A: `notebooks/cuda/10_train_stage_a_public.ipynb`, 512 tokens, about 9.5 h.
- Stage B: `notebooks/cuda/20_train_stage_b.ipynb`, 1024 tokens, starts from Stage A
  (`model.init: checkpoint`), about 7.5 h.

### Evaluation, calibration, reports

Data and predictions are JSONL files in the unified format (`jevlite.data.unified`).

```bash
# predictions of a model or a zero-shot baseline
uv run python scripts/predict_baseline.py --baseline jevlite --model path/to/final --data data/test.jsonl --out preds/jevlite.jsonl
uv run python scripts/predict_baseline.py --baseline nli --data data/test.jsonl --out preds/nli.jsonl
# report per primitive and slice (JSON + markdown), with bootstrap CIs
uv run python scripts/eval.py --data data/test.jsonl --pred preds/jevlite.jsonl --out reports/jevlite --bootstrap 1000
# paired comparison: bootstrap CI of Δ accuracy + McNemar, Holm-corrected
uv run python scripts/compare_paired.py --data data/test.jsonl --model jevlite=preds/jevlite.jsonl \
    --ref nli=preds/nli.jsonl --out reports/paired
# black-box checks: option-order sensitivity, IIA, confidence on nonsense inputs, Bool shortcuts
uv run python scripts/probe_blackbox.py --baseline jevlite --model path/to/final --data data/test.jsonl --out reports/probes.json
# calibration: fit on calib predictions (scripts/calibrate.py fit), then copy calibration.json next to the weights
```

Judge benchmarks, eval v2 and the full calibration commands: [`docs/EVALUATION.md`](docs/EVALUATION.md) section 5.
Markdown reports with charts: `uv sync --extra reports`, then `scripts/make_report.py`.

## License

Code and model weights: [Apache-2.0](LICENSE), for every version. The datasets keep their own licenses,
listed in [`docs/DATASETS.md`](docs/DATASETS.md); non-commercial sets are used for evaluation only.
