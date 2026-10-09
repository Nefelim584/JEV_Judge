# Phase 2 baselines: Laya and zero-shot NLI

> Run on 2026-10-08 (todo Phase 2). Both baselines are used **zero-shot**: no training, no fitted
> temperatures of our own. These are the numbers jev-lite has to beat offline (Phases 6–8).
> They do **not** answer H0 (jev-lite vs the production LLM judge): that needs the human-labelled
> judge test set from Phase 11.

## 1. Setup

**Test data.** `data/mix/stage_a/test_in.jsonl` (19,189 records) and `test_ood.jsonl` (23,368 records),
42,557 records in total, from 24 public datasets (`DATASETS.md`, section 0). Both Stage A versions
(`stage_a` and `stage_a_nosa`) share these files, so the numbers hold for both.

| Primitive | Records | Sources |
|---|---|---|
| Choice | 26,635 | MCQA (ARC, OpenBookQA, CommonsenseQA, Cosmos QA, Social IQa), intents and topics (CLINC150, BANKING77, MASSIVE, DBpedia-14, GoEmotions), 3-way NLI (MultiNLI, SNLI, WANLI, VitaminC, ContractNLI), summary preference |
| Score | 2,220 | HelpSteer2 (5 attributes), HelpSteer3 (preference) |
| Bool | 13,702 | NLI and fact checking, BoolQ, answerability (SQuAD 2.0, ClapNQ, QNLI), PAWS, TabFact |

`test_ood` holds the held-out sources (ContractNLI, BANKING77, MultiNLI mismatched genres) and the
held-out question templates. Up to 1,000 records per source and split.

**Baselines**

| | NLI | Laya |
|---|---|---|
| Model | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (435M, MIT) | `convaiinnovations/laya`, base (421M, Apache 2.0) |
| What it is | a 3-way NLI cross-encoder | a Jev-like typed-decision model: ModernBERT-large + 2-layer decision head |
| Bool | P(entailment) of (state, claim) | a 2-option question (false / true) |
| Choice / Score | one templated hypothesis per candidate, softmax over the entailment logits | all options in one sequence, one `[MASK]` marker each |
| Temperatures | none | Laya's own, per question type and K bucket |
| Wrapper | `src/jevlite/baselines/nli.py` | `src/jevlite/baselines/laya.py` |

Coverage: both baselines predicted all 42,557 records, with no skipped questions.

**Hardware.** Laya ran on an M1 Pro (MPS, fp32). NLI ran partly on the M1 (MPS, fp32), partly on a
Colab T4 in fp32, and mostly on the T4 with fp16 autocast. Probabilities differ by about 1e-3
between these settings, which does not move any metric below. **Latency numbers are not comparable**
between the two baselines (section 7).

## 2. Headline numbers

Point estimates over all of `test_in` + `test_ood`; 95% bootstrap intervals in the per-model reports.
Bold marks the better baseline.

| Primitive | Metric | NLI | Laya |
|---|---|---|---|
| Choice | accuracy | 0.607 | **0.631** |
| | κ | 0.529 | **0.558** |
| | NLL | **1.067** | 1.561 |
| | ECE | **0.057** | 0.094 |
| | AURC | 0.246 | **0.221** |
| Score | accuracy | 0.241 | **0.314** |
| | weighted κ | 0.294 | **0.366** |
| | Spearman | 0.005 | **0.222** |
| | MAE (levels) | 1.205 | **1.027** |
| | ECE | **0.115** | 0.187 |
| Bool | accuracy | 0.712 | **0.729** |
| | κ | 0.421 | **0.462** |
| | NLL | 0.891 | **0.546** |
| | ECE | 0.198 | **0.072** |
| | AURC | 0.161 | **0.142** |
| | selective accuracy at 80% coverage | 0.762 | **0.781** |

The overall averages hide the main result: **the two baselines are good at different things**
(section 3).

## 3. Where each baseline wins

![Accuracy by source, NLI vs Laya](docs/baselines/figures/accuracy_by_source.png)

*Accuracy per source and primitive, sorted by the gap. Above the dashed line NLI wins (claim vs text, MCQA), below it Laya wins (classification, answerability, PAWS).* [Interactive version](docs/baselines/figures/accuracy_by_source.html)

### 3.1 Claim vs text (faithfulness-like): NLI wins clearly

This is the core of the judge: does a claim follow from the retrieved text.

| Bool sources | n | NLI | Laya |
|---|---|---|---|
| NLI family: MultiNLI, SNLI, WANLI, VitaminC, ContractNLI | 4,102 | **0.875** | 0.690 |
| ContractNLI only (held out, `test_ood`) | 1,054 | **0.819** | 0.591 |
| All other Bool sources | 9,600 | 0.642 | **0.746** |

- On ContractNLI, a source neither model has seen, NLI leads by 23 points. This is the cleanest
  faithfulness comparison we have.
- NLI's 0.95 on MultiNLI and 0.83 on WANLI are inflated: both are in its training data. SNLI (0.947)
  and ContractNLI are not.
- **The bar for jev-lite on faithfulness is the NLI model, not Laya.**

### 3.2 Other Bool tasks: Laya wins

| Source | Task | NLI | Laya |
|---|---|---|---|
| PAWS | paraphrase | 0.513 (κ 0.05, chance) | **0.890** |
| QNLI | sentence answers the question | 0.672 | **0.801** |
| BoolQ | yes/no QA | 0.725 | **0.777** |
| ClapNQ | answerable | 0.618 | **0.692** |
| SQuAD 2.0 | answerable | **0.693** | 0.649 |
| TabFact | claim vs table | 0.583 | 0.513 (κ 0.02, chance) |

- Answerability (`criterion = answerable`, 4,600 records): Laya 0.721, NLI 0.674. This is the
  closest public proxy for judging refusals.
- PAWS shows the limit of NLI: paraphrase is not entailment, and the model is at chance there.
- **Tables are unsolved by both.** TabFact is at or near chance. The judge must handle tables
  (financial reports, docs), so table data in Phase 4 is not optional.

### 3.3 Choice: Laya wins classification, NLI wins multiple-choice QA

| Source | K | NLI | Laya |
|---|---|---|---|
| CLINC150 | 3–16 | 0.736 | **0.935** |
| BANKING77 (held out) | 3–16 | 0.670 | **0.843** |
| GoEmotions | 3–16 | 0.433 | **0.643** |
| DBpedia-14 | 3–14 | 0.905 | **0.916** |
| ARC | 3–8 | **0.604** | 0.433 |
| Cosmos QA | 4–7 | **0.463** | 0.273 |
| CommonsenseQA | 5–8 | **0.554** | 0.444 |
| OpenBookQA | 4–7 | **0.548** | 0.338 |
| Social IQa | 3–6 | **0.541** | 0.467 |
| 3-way NLI as Choice (VitaminC, SNLI, MultiNLI) | 3 | 0.54–0.58 | **0.65–0.71** |

- Laya is strong on intents and topics, its training domain, but weak on reasoning MCQA. On Cosmos
  QA it is close to chance.
- 3-way NLI as Choice: NLI is weaker here than on the same data as Bool, because the
  `supported / contradicted / not mentioned` options are scored through a hypothesis template,
  not through the model's own three classes.

### 3.4 Score: both are weak

- Accuracy 0.24 (NLI) and 0.31 (Laya) on 5- and 7-level rubrics. NLI has no rank correlation at
  all (Spearman 0.005); Laya has 0.22.
- By criterion, Laya: coherence 0.22, correctness 0.26, helpfulness 0.28, verbosity 0.44.
- Rubric scoring is the easiest place for jev-lite to beat both baselines, and also the most
  needed: the relevance rubric of the judge is a Score question.

## 4. Calibration

![Reliability diagrams](docs/baselines/figures/reliability.png)

*Reliability per primitive: on the diagonal = calibrated. NLI's Bool curve is flat in the middle and steep at the ends, i.e. extreme probabilities; both models are overconfident on Score.* [Interactive version](docs/baselines/figures/reliability.html)

**Laya's temperature buckets are miscalibrated at the edges.** Choice by number of options:

![Accuracy vs confidence by number of options](docs/baselines/figures/calibration_by_k.png)

*Choice records: accuracy (filled) vs mean top probability (hollow) per K bucket. Laya is overconfident at K = 2 (+0.33) and K ≥ 11 (+0.22); NLI is mildly underconfident at large K.* [Interactive version](docs/baselines/figures/calibration_by_k.html)


| K | n | Laya accuracy | Laya mean confidence | Laya ECE | NLI ECE |
|---|---|---|---|---|---|
| 2 | 2,000 | 0.520 | 0.851 | 0.331 | 0.102 |
| 3–5 | 13,634 | 0.588 | 0.605 | **0.029** | 0.051 |
| 6–10 | 6,435 | 0.662 | 0.745 | 0.085 | 0.065 |
| 11+ | 4,566 | 0.765 | **0.985** | 0.220 | 0.118 |

- For K ≥ 11 Laya's fitted temperature is 0.10, which multiplies the logits by 10. It claims 98.5%
  confidence while it is right 76.5% of the time. NLL there is 4.2, which explains GoEmotions'
  NLL of 4.9.
- For K = 2 (summary preference) it is also overconfident: 0.85 claimed vs 0.52 observed.
- In the 3–5 bucket, the one with most data, it is well calibrated (ECE 0.029).
- This confirms the Phase 2 smoke result and the decision in todo 2.7: **fit a smooth T(K) instead
  of per-bucket temperatures**, and report ECE per K bucket in Phase 7.

**NLI is overconfident on Bool** (ECE 0.198, NLL 0.89). It gives extreme probabilities on tasks
it was not built for: PAWS ECE 0.31, TabFact 0.33. On its own family it is well calibrated: MultiNLI
and SNLI ECE 0.03.

## 5. Selective prediction (the cascade view)

Answer the most confident share locally and escalate the rest. Accuracy on the most confident 80%:

![Risk vs coverage](docs/baselines/figures/risk_coverage.png)

*Error rate of the answered share as coverage grows. For Bool both curves are close to straight lines: confidence barely separates right from wrong answers.* [Interactive version](docs/baselines/figures/risk_coverage.html)


| Primitive | NLI | Laya |
|---|---|---|
| Choice | 0.654 | 0.685 |
| Score | 0.233 | 0.336 |
| Bool | 0.762 | 0.781 |

Skipping the 20% least confident adds only about 5 points of accuracy for either model. Their
confidence (`1 − H(p)/log K`) is a weak escalation signal. See also section 6.3.

## 6. Black-box probes (200 items each, `test_in`)

### 6.1 Order sensitivity of Choice

| | NLI | Laya |
|---|---|---|
| mean spread of an option's probability over 8 permutations | 0.000 | **0.099** |
| max spread | 0.000 | 0.227 |
| argmax changes under a permutation | 0% | **14%** |

NLI scores each option independently, so it cannot depend on order. Laya, like our packed v1, reads
all options in one sequence: **the answer changes in 14% of permutations.** This is larger than the
smoke run of 2026-10-02 suggested (0.04 spread, no flips). Option shuffling in training (already in
`TrainCollator`) is required, and order sensitivity is a Phase 8 metric for jev-lite.

### 6.2 IIA (adding one option borrowed from another question)

| | NLI | Laya |
|---|---|---|
| mean change of log-ratios between the original options | 0.000 | **0.92** |
| probability mass taken by the added (wrong) option | 0.077 | 0.102 |

NLI satisfies IIA by construction. For Laya, adding an irrelevant option changes the ratios between
the existing ones substantially. Part of this is the K-bucket temperature jump (section 4).

### 6.3 Confidence on nonsense inputs

Mean heuristic confidence when the state is replaced:

| State | NLI | Laya |
|---|---|---|
| real | 0.449 | 0.412 |
| shuffled words | 0.393 | 0.319 |
| random characters | 0.361 | **0.519** |

**Neither model's confidence reliably drops on garbage.** Laya is *more* confident on random
characters than on real text. A confidence signal that does not notice a meaningless input cannot
decide escalation. This is the evidence for the learned confidence head (v2, todo 2.8) and for
keeping it in the v1 release (open decision in todo section 7).

## 7. Latency

| | p50 | p95 | Notes |
|---|---|---|---|
| Laya | 50 ms | 142 ms | M1 MPS fp32, one question per call |
| NLI | 26 ms | 329 ms | mixed hardware (M1, T4 fp32, T4 fp16); per record, amortised over batched calls |

These are not comparable and not a latency benchmark. Latency is measured properly in Phase 9
(`bench_latency.py`) and per judged answer in Phase 10. Structurally, NLI costs K forward passes per
Choice/Score question and one per claim; Laya costs one per question.

## 8. What this means for jev-lite

Targets for Stage A/B on `test_in` + `test_ood` (to be fixed together with the H0–H3 success
criteria before the first training run):

| Area | Baseline to beat | Current best |
|---|---|---|
| Faithfulness (claim vs text) | NLI | 0.875 on the NLI family, **0.819 on held-out ContractNLI** |
| Answerability / refusals | Laya | 0.721 |
| Choice, classification | Laya | 0.94 CLINC150, 0.84 BANKING77 |
| Choice, MCQA | NLI | 0.46–0.60 |
| Score (rubrics) | Laya | accuracy 0.31, Spearman 0.22 |
| Calibration | best of both per primitive | ECE 0.057 (Choice, NLI), 0.115 (Score, NLI), 0.072 (Bool, Laya) |

Design consequences:

1. **Smooth T(K)**, not K buckets (section 4).
2. **Option shuffling in training** stays mandatory, and order sensitivity and IIA are reported for
   jev-lite in Phase 8 with the same probes (section 6).
3. **Learned confidence (v2)** is needed for the cascade: the entropy heuristic ignores nonsense
   inputs (section 6.3).
4. **Tables** need dedicated data (Phase 4); both baselines fail on TabFact.
5. A per-criterion H3 comparison should include NLI-style checkers. NLI is already a strong
   faithfulness specialist, which is exactly what H3 tests against.

## 9. Caveats

- **Contamination.** The NLI model was trained on MultiNLI, FEVER, ANLI, LingNLI and WANLI, so its
  MultiNLI and WANLI numbers are in-distribution. Laya's training data is not published; it may
  overlap with our sources (it lists NLI, intents and quality rubrics among its domains).
- **`test_in` vs `test_ood` is confounded by source mix.** For example, Laya's Choice accuracy is
  *higher* on `test_ood` (0.652 vs 0.598) because BANKING77, where it is strong, is held out
  entirely. Compare per source, not per split.
- **These are public-data proxies.** The offline judge benchmark (LLM-AggreFact, RAGTruth, …) is
  not converted yet, and human-labelled data from our system only arrives in Phase 11.
- **Templates matter for NLI.** Choice and Score through hypothesis templates are a weak use of an
  NLI model; better templates would raise its Choice numbers somewhat.
- One run each, no seeds: zero-shot inference is deterministic up to hardware noise.

## 10. Files and reproduction

All outputs are under `data/baselines/` (git-ignored):

- `preds/{laya,nli}.jsonl`: predictions;
- `reports/{laya,nli}/report.{md,json}`: full reports with bootstrap CIs, per split, source,
  domain and criterion;
- `reports/compare/report.md` and `figures/`: side-by-side tables, reliability diagrams,
  risk–coverage curves (PNG and interactive HTML);
- `probes/{laya,nli}.json`: black-box probes.

```
# predictions (NLI: --device cuda --fp16 --batch-size 64 on a CUDA GPU)
HF_HUB_OFFLINE=1 uv run python scripts/predict_baseline.py --baseline laya \
  --data data/mix/stage_a/test_in.jsonl data/mix/stage_a/test_ood.jsonl --out data/baselines/preds/laya.jsonl
HF_HUB_OFFLINE=1 uv run python scripts/predict_baseline.py --baseline nli \
  --data data/mix/stage_a/test_in.jsonl data/mix/stage_a/test_ood.jsonl --out data/baselines/preds/nli.jsonl

# reports
uv run python scripts/eval.py --data data/mix/stage_a/test_in.jsonl data/mix/stage_a/test_ood.jsonl \
  --pred data/baselines/preds/laya.jsonl --out data/baselines/reports/laya --bootstrap 1000
uv run python scripts/eval.py --data data/mix/stage_a/test_in.jsonl data/mix/stage_a/test_ood.jsonl \
  --pred data/baselines/preds/nli.jsonl --out data/baselines/reports/nli --bootstrap 1000
uv run python scripts/make_report.py compare --out data/baselines/reports/compare \
  --run nli=data/baselines/reports/nli/report.json --run laya=data/baselines/reports/laya/report.json

# probes
HF_HUB_OFFLINE=1 uv run python scripts/probe_blackbox.py --baseline laya \
  --data data/mix/stage_a/test_in.jsonl --out data/baselines/probes/laya.json --n-items 200
HF_HUB_OFFLINE=1 uv run python scripts/probe_blackbox.py --baseline nli \
  --data data/mix/stage_a/test_in.jsonl --out data/baselines/probes/nli.json --n-items 200
```

The K-bucket table in section 4 and the source groups in section 3.1 were computed from
`preds/*.jsonl` and the test files; they are not part of `eval.py`'s output.

The figures (PNG + interactive HTML, tracked in `docs/baselines/figures/`) are rebuilt from the same inputs.
The rebuilt `stage_a` test files (2026-10-09, with FEVER, HoVer and NQ) keep every earlier record
unchanged, so they work as input; records without predictions (the new sources) are ignored:

```
uv run --extra reports python scripts/make_baselines_figures.py
```
