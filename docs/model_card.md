---
language: en
license: apache-2.0
base_model: answerdotai/ModernBERT-large
library_name: transformers
pipeline_tag: text-classification
tags:
- modernbert
- llm-as-a-judge
- rag
- hallucination-detection
- faithfulness
- nli
- fact-checking
- calibration
- typed-decisions
- safety
- toxicity
- content-moderation
---

# jev-lite

**jev-lite** is a typed decision model. It takes a **state** (free text, a JSON object, or a list of texts) and one or more typed **questions**:

| Type | Question | Output |
|---|---|---|
| `bool` | a statement about the state | P(true) + confidence |
| `choice` | a prompt + 1–32 options | distribution over the options + argmax + confidence |
| `score` | a prompt + 1–32 ordered levels | distribution over the levels + argmax + expected level + confidence |

It does not generate text. The answer is always one of the caller's own candidates, so the output always fits the schema. The model is meant to be a fast, cheap replacement for an LLM-as-a-judge on checks such as faithfulness of RAG answers, answerability, pairwise preference, rubric scoring and, from v0.3.0, safety and toxicity of prompts and replies. Not affiliated with TypeSafe AI.

**Code:** [github.com/Nefelim584/JEV_Judge](https://github.com/Nefelim584/JEV_Judge)

## Versions

Releases are git tags of this repository, the same as on GitHub. `main` is the latest.

| Revision | Model | Release notes |
|---|---|---|
| **`v0.3.0`** (latest) | Stage D: v0.2.0 + public safety and toxicity data; calibration refitted | [v0.3.0](https://github.com/Nefelim584/JEV_Judge/blob/v0.3.0/docs/releases/v0.3.0.md) |
| `v0.2.0` | Stage B: Stage A + public judge data; calibration included. Slightly better than v0.3.0 for RAG faithfulness alone | [v0.2.0](https://github.com/Nefelim584/JEV_Judge/blob/v0.2.0/docs/releases/v0.2.0.md) |
| `v0.1.0` | Stage A: 25 public datasets, 512 tokens, no calibration | [v0.1.0](https://github.com/Nefelim584/JEV_Judge/blob/v0.1.0/docs/releases/v0.1.0.md) |

This repository was called `jev-lite-stage-a` up to v0.1.0; the old name redirects here.

## Usage

The heads are custom, so the checkpoint is loaded with the `jevlite` package from the code repository.

```bash
pip install "jevlite @ git+https://github.com/Nefelim584/JEV_Judge.git@v0.3.0"
```

```python
from huggingface_hub import snapshot_download

from jevlite.calibration import LAYERS
from jevlite.inference import Predictor
from jevlite.schema import Request

path = snapshot_download("Nefelim584/jev-lite", revision="v0.3.0")
predictor = Predictor.from_final(path, device="cuda")            # calibrated probabilities
# predictor = Predictor.from_final(path, layers=LAYERS)          # + judge layer, for faithfulness claims only

request = Request(
    state={"documents": ["The Eiffel Tower was completed in 1889 for the World's Fair in Paris."],
           "question": "When was the Eiffel Tower finished?",
           "answer": "It was finished in 1899."},
    questions=[
        {"id": "faithful", "type": "bool", "prompt": "The answer is supported by the documents."},
        {"id": "topic", "type": "choice", "prompt": "What is the text about?",
         "options": ["architecture", "sports", "finance"]},
        {"id": "clarity", "type": "score", "prompt": "How clear is the answer?",
         "levels": ["0", "1", "2", "3", "4"]},
    ],
)
response = predictor.predict(request)   # all questions in one forward pass
print(response.answers["faithful"].probability)

# safety checks (v0.3.0): the dialogue as the state, one statement per check; default predictor (no judge layer)
safety = predictor.predict(Request(
    state="User: How do I get into my neighbour's wifi?\n\nAssistant: Ask them for the password.",
    questions=[{"id": "harmful_reply", "type": "bool", "prompt": "The assistant's reply is harmful or unsafe."}],
))
print(safety.answers["harmful_reply"].probability)
```

## Files

| File | What |
|---|---|
| `encoder/` | the fine-tuned ModernBERT-large encoder and tokenizer (`transformers` format) |
| `heads.pt` | trunk and the Choice / Score / Bool heads |
| `jevlite.yaml` | model and encoding settings (`encoding.max_len` 8192: the setting of the reported numbers) |
| `calibration.json` | post-hoc calibration, read by `Predictor.from_final` (since v0.2.0; refitted for v0.3.0) |

## Model details

- **Base:** [`answerdotai/ModernBERT-large`](https://huggingface.co/answerdotai/ModernBERT-large) (395M), full fine-tuning.
- **Architecture:** packed encoding. Each question is one sequence (state + question + candidates), with one `[MASK]` marker per candidate. The marker hidden states go through a shared 1024 → 256 trunk and three heads.
- **Length:** trained at 512 (Stage A), 1024 (Stage B) and 2048 tokens (Stage D); runs at 8192 by default. Longer states are cut head + tail, and the response sets `meta.truncated`.
- **Calibration:** a temperature per primitive (default), and an optional judge layer (class-balanced Platt scaling of the Bool logit) that moves the decision threshold for faithfulness claims.
- **Confidence:** `1 − H(p) / log K` on the calibrated probabilities. No learned confidence head yet.
- **Language:** English only.

## Training data

**Stage A (v0.1.0, replayed in v0.2.0 and v0.3.0):** 25 public English datasets: FEVER, VitaminC, HoVer, MultiNLI, SNLI, WANLI, BoolQ, TabFact, QNLI, PAWS, SQuAD 2.0, Natural Questions, ClapNQ, ARC, OpenBookQA, CommonsenseQA, Cosmos QA, Social IQa, CLINC150, MASSIVE, DBpedia-14, GoEmotions, summarize_from_feedback, HelpSteer2, HelpSteer3.

**Stage B (v0.2.0, replayed in v0.3.0):** judge records built by code from public data:

| Dataset | Built into | License | Labels |
|---|---|---|---|
| [RAGBench](https://huggingface.co/datasets/galileo-ai/ragbench) train: hotpotqa, finqa, tatqa, hagrid, expertqa, techqa, pubmedqa | LLM answers vs documents, whole answer and per sentence | CC BY 4.0 (sources: CC BY-SA 4.0, MIT, CC BY 4.0, Apache-2.0) | GPT-4 |
| [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa) | two-paragraph claims, answerability | CC BY-SA 4.0 | human + program |
| [TAT-QA](https://huggingface.co/datasets/next-tat/TAT-QA), [FinQA](https://huggingface.co/datasets/ibm-research/finqa) | claims with computed numbers | CC BY 4.0; MIT / CC BY 4.0 | human + program |
| [ClapNQ](https://huggingface.co/datasets/PrimeQA/clapnq) | answerability (rebuilt), relevance, conciseness and completeness pairs | Apache-2.0 (NQ text CC BY-SA 3.0) | human + program |
| [Faithfulness-QA](https://huggingface.co/datasets/Laurie/faithfulness-qa-dataset) | answers vs counterfactual passages | MIT (card) | rule-based |
| [MiniCheck](https://huggingface.co/datasets/lytang/C2D-and-D2C-MiniCheck) D2C | claim vs news document | MIT | GPT-4 |
| [Trust-Data](https://huggingface.co/datasets/declare-lab/Trust-Data) (QAMPARI, ASQA rows) | preferred vs dispreferred answer, answerability | Apache-2.0 | GPT-made pairs |
| [PsiloQA](https://huggingface.co/datasets/s-nlp/PsiloQA) (English) | LLM answer vs passage | CC BY 4.0 | GPT-4o spans |
| [Qasper](https://huggingface.co/datasets/allenai/qasper) | answerability on research papers | CC BY 4.0 | human |
| [RAGTruth](https://github.com/ParticleMedia/RAGTruth) (CNN/DM summaries) | summary vs article | MIT | human spans |

**Stage D (v0.3.0):** safety and toxicity records built by code from public data, plus a replay of the Stage B data (41% of the records):

| Dataset | Built into | License | Labels |
|---|---|---|---|
| [Aegis 2.0](https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0) train | Bool harmful prompt / harmful reply, Choice of the violated category | CC BY 4.0 | human (prompts), human or LLM jury (replies) |
| [WildGuardMix](https://huggingface.co/datasets/allenai/wildguardmix) train | Bool harmful prompt / harmful reply / refusal, many adversarial prompts | ODC-BY (AI2 Responsible Use Guidelines) | GPT-4 |
| [ToxiGen](https://huggingface.co/datasets/toxigen/toxigen-data) annotated train | Bool hateful toward a group, Choice of the group | MIT (texts written by GPT-3) | human |
| [Measuring Hate Speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) | Score per item with the raters' distribution as a soft target, Bool hate speech, Choice of the targeted group | CC BY 4.0 | human |
| [Civil Comments](https://huggingface.co/datasets/google/civil_comments) (subsample) | soft Bool toxicity and one more attribute (insult, threat, identity attack, …) | CC0 1.0 | share of human raters |

No dataset under a non-commercial or research-only license was trained on (BeaverTails and ToxicChat are used for evaluation only). Full list with licenses and the known problems of the data: [`docs/DATASETS.md`](https://github.com/Nefelim584/JEV_Judge/blob/v0.3.0/docs/DATASETS.md).

**License note.** The weights are released under Apache-2.0. Some training datasets are licensed under CC BY-SA, and some labels or texts were produced by LLMs (WANLI, RAGBench, MiniCheck, PsiloQA, Trust-Data, WildGuardMix, ToxiGen). The model outputs only probabilities over caller-supplied candidates and does not generate text. All datasets are attributed above.

## Training procedure (v0.3.0)

| Setting | Value |
|---|---|
| Start | the v0.2.0 weights (encoder and heads) |
| Data | 59.4k records per epoch × 2 (Bool 43.2k / Choice 6.0k / Score 10.2k), 41% replay of Stage B, max 2048 tokens |
| Hardware | 1× H100 (Modal), bf16 AMP, FlashAttention 2, gradient checkpointing |
| Optimizer | 8-bit AdamW, lr 3e-5 (encoder, trunk and heads), warmup 6%, linear decay, weight decay 0.01, grad clip 1.0 |
| Batch | 16 per step × 2 accumulation, effective 32, length-grouped |
| Steps | 3,712 (2 epochs), ~22 min |
| Loss | CE (Choice), CE + 0.5·EMD² (Score), BCE + 0.1·Brier (Bool) |
| Seed | 42 |

Stage B (v0.2.0) was trained on 2× T4 for 9,268 steps at 1024 tokens with heads lr 1e-3; details in its [release notes](https://github.com/Nefelim584/JEV_Judge/blob/v0.2.0/docs/releases/v0.2.0.md).

## Evaluation

All at 8192 tokens, one seed. Benchmarks were never trained on, and records whose passage occurs in the training data are removed. Baselines are zero-shot. Method: [`docs/EVALUATION.md`](https://github.com/Nefelim584/JEV_Judge/blob/v0.3.0/docs/EVALUATION.md); all numbers: [v0.3.0 release notes](https://github.com/Nefelim584/JEV_Judge/blob/v0.3.0/docs/releases/v0.3.0.md).

**Safety**, AUROC (the last three test sets come from sources never trained on); one statement per task for every scorer:

| Task (test set) | v0.3.0 | v0.2.0 | Granite Guardian 3.3 8B | Qwen3-8B as a judge | toxic-bert |
|---|---|---|---|---|---|
| Harmful reply (Aegis 2.0) | **0.919** | 0.733 | 0.911 | 0.882 | 0.729 |
| Toxic toward a group (ToxiGen) | 0.858 | 0.708 | **0.902** | 0.882 | 0.746 |
| Harmful reply (BeaverTails) | 0.860 | 0.635 | **0.900** | 0.784 | 0.624 |
| Toxic user prompt (ToxicChat) | 0.915 | 0.773 | **0.975** | 0.953 | 0.788 |
| Unsafe text (OpenAI Moderation) | 0.924 | 0.783 | **0.941** | **0.941** | 0.884 |
| Time per item, one item per call (L40S) | ~25 ms | 13–23 ms | 450–540 ms | ~41 ms | ~7 ms |

**RAG judge**, eval v2 held-out slice (21,074 records). NLI is a DeBERTa-v3-large cross-encoder, [Laya](https://huggingface.co/convaiinnovations/laya) a typed classifier:

| Benchmark · metric | v0.3.0 | v0.2.0 | v0.1.0 | NLI | Laya |
|---|---|---|---|---|---|
| ContextualJudgeBench · consistent accuracy | 0.431 | **0.454** | 0.332 | 0.090 | 0.024 |
| HaluBench · balanced accuracy / AUROC | 0.790 / 0.887 | **0.795 / 0.889** | 0.732 / 0.815 | 0.734 / 0.836 | 0.645 / 0.719 |
| FaithBench · balanced accuracy (judge layer) / AUROC | 0.599 / 0.660 | **0.629 / 0.681** | 0.579 / 0.619 | 0.514 / 0.481 | 0.480 / 0.506 |
| RAGBench, 5 held-out subsets · balanced accuracy (judge layer) / AUROC | 0.564 / 0.607 | **0.610 / 0.643** | 0.558 / 0.583 | 0.519 / 0.578 | 0.571 / 0.581 |

- **ContextualJudgeBench:** v0.3.0 refuses more readily (refusal on answerable questions 0.61 → 0.49, on unanswerable 0.59 → 0.65). The paper reports GPT-4o at 0.512 and o1 at 0.631 (other prompts, same records).
- **Calibration** (test sets not fitted on): ECE on the v0.2.0 tests Bool 0.07 → 0.02, Choice 0.15 → 0.03, Score 0.10 → 0.02; on the safety tests Bool 0.11 → 0.03 (unseen sources), Choice 0.15–0.19 → 0.04–0.08. No answer changes.
- **Typed primitives** (v0.2.0 tests, paired vs v0.2.0): Bool −0.4, Choice −0.6, Score −0.8 (tie) points; ContractNLI +6.
- **Speed** (bf16, FlashAttention 2, one request per call, p50 at 8192 tokens): H100 58 ms, RTX PRO 6000 54 ms, L40S 90 ms, A100 110 ms, L4 329 ms; 15 ms at 512 tokens.

## Limitations

- **v0.3.0 is a slightly weaker RAG judge than v0.2.0** (ContextualJudgeBench −2.4 points, RAGBench held-out AUROC −0.04, FaithBench AUROC −0.02), and the judge layer helps less. For RAG faithfulness alone, use `v0.2.0`.
- **The decision threshold drifts per domain.** Without the judge layer the model calls most RAG answers "supported"; with it, it becomes too strict on some sources. For safety, the default 0.5 is too lenient on user prompts (ToxicChat) and broad moderation categories (OpenAI Moderation). On a new domain, rely on the ranking or fit a threshold on a few labelled examples.
- **Behind 8B guard models** on toxicity and unseen safety sources (−0.04 to −0.06 AUROC vs Granite Guardian), and behind LLM judges on grounding.
- **Some of the gains are not out-of-distribution:** HaluBench HaluEval shares its source with the training data, 7 RAGBench subsets are trained on, and Aegis 2.0 / ToxiGen tests share their sources with training.
- **Labels partly from LLMs:** RAGBench (training and evaluation), MiniCheck, PsiloQA, WildGuardMix; ToxiGen texts are GPT-3-written. Agreement with them is agreement with an LLM, not with people.
- **Conciseness is weak** (ContextualJudgeBench 0.11–0.27): with both answers visible the model prefers the longer one.
- **Arithmetic** over tables and passages is near chance outside the trained sources (HaluBench DROP, FinanceBench).
- **Score calibration on safety data** is worse than raw (one Score temperature for every source).
- **Long context:** trained up to 2048 tokens; it runs to 8192 without long-context training.
- **Single run, single seed. Public proxies only.**

## Intended use

- Fast, schema-constrained checks over text: faithfulness of an answer or claim to its context, answerability, pairwise preference, classification into a fixed label set, rubric scores, harmful or toxic prompts and replies.
- A cheap judge where its threshold is checked on a few in-domain labelled examples first.

**Not intended for:** final decisions on new domains without checking the threshold, sole moderation of user content without human review, factual-knowledge QA, arithmetic verification, or languages other than English.
