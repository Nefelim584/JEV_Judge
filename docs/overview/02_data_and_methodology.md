# Part 2. Data, training and evaluation methodology

> jev-lite overview, October 2026 (revised 2026-10-30 for Stage B) · Part 2 of 5 ·
> [← Previous: Introduction](01_introduction.md) · [Index](README.md) ·
> [Next: Model test results →](03_stage_a_results.md)

## 4. Data used so far

**Selection rules.** Public English datasets with **human labels** and a license that allows
commercial use. Exceptions for LLM-made data are listed per source: WANLI in Stage A (GPT-3 wrote the
pairs, humans labelled and revised them); in Stage B, RAGBench, MiniCheck and PsiloQA are LLM-labelled
and marked so per record. Share-alike (CC BY-SA) datasets are included after legal confirmed it on
2026-10-07. Full license checks are in `DATASETS.md`.

### 4.1 Stage A training: 25 datasets

| Primitive | Share of the mix | Datasets | What they teach |
|---|---|---|---|
| **Bool** | 50% | FEVER, VitaminC, HoVer (multi-hop), MultiNLI, SNLI, WANLI | is a claim supported by the text (faithfulness) |
| | | SQuAD 2.0, Natural Questions, ClapNQ, QNLI | is the question answerable from the text (refusals) |
| | | BoolQ, PAWS, TabFact (tables) | yes/no QA, paraphrase, claims about tables |
| **Choice** | 20% | VitaminC, FEVER, MultiNLI, SNLI, WANLI as `supported / contradicted / not mentioned` | 3-way faithfulness |
| | | ARC, OpenBookQA, CommonsenseQA, Cosmos QA, Social IQa | multiple-choice QA |
| | | CLINC150 (151 classes), MASSIVE, DBpedia-14, GoEmotions | classification with labels as options |
| | | summarize_from_feedback | pairwise preference (K = 2) |
| **Score** | 30% | HelpSteer2 (5 attributes), HelpSteer3 | rubric scoring of answers |

- **300k records per epoch, 2 epochs**, drawn from a pool of ~500k with per-dataset caps.
- **The 3-way NLI and fact-checking sets are used both ways**: each example goes either to Bool
  (entailed vs not) or to Choice (supported / contradicted / not mentioned), never both.
- **Question templates** vary the wording of every question, and some templates are held out from
  training to test generalisation to new wording.
- **Two versions** were prepared as a legal precaution: with share-alike data (`stage_a`, trained) and
  without (`stage_a_nosa`, prepared, not trained yet).

### 4.2 Stage B training: public judge data

Stage B targets the gaps Stage A showed on the judge benchmarks (Part 4 of the first version of this
overview; `JUDGE_BENCH.md`): refusals, tables and computed numbers, claims across several chunks,
real LLM answers, length bias in pairwise judging. The plan was synthetic data from a teacher LLM;
generating it locally would have taken about a week, so Stage B takes **public data, turned into judge
records by code** (decision of 2026-10-26; synthetic data comes back later as a small top-up).

| Source | Share of the mix | Records per epoch | What we build | Gap it targets |
|---|---|---|---|---|
| **Stage A replay** | 26% | 39.1k | Stage A's own training records | forgetting |
| **RAGBench**, train of 7 subsets (HotpotQA, FinQA, TAT-QA, HAGRID, ExpertQA, TechQA, PubMedQA) | 12% | 18.1k | real LLM answers over several documents; the whole answer, one sentence against all documents, one sentence against one document (labels by GPT-4) | judging real RAG answers |
| **HotpotQA** (distractor setting) | 11% | 16.1k | "the answer to Q is A" supported only by the chunks together; negatives swap in an entity from a distractor; **unanswerable** by dropping a gold paragraph while on-topic distractors stay | claims across chunks; refusals on on-topic passages |
| **Faithfulness-QA** | 7% | 10.1k | an edited passage where the true-world answer is now wrong | reading the text, not world knowledge |
| **TAT-QA, FinQA** | 9% | 12.8k | the computed answer vs another number of the same table, a scale error or a small shift | tables and derived numbers |
| **PsiloQA** (English) | 5% | 8.1k | LLM answers with hallucinated spans vs the golden answer | hallucinations of real LLMs |
| **Trust-Data** (ASQA, QAMPARI) | 7% | 10.0k | preferred vs dispreferred answer (Choice K = 2); a refusal as the right answer | refusals; pairwise judging |
| **ClapNQ** (fixed) | 7% | 10.6k | the passage with its own vs another page's question; relevance levels; conciseness and completeness pairs in both orders | the Stage A shortcut; criterion (b); length bias |
| MiniCheck C2D/D2C, Qasper, RAGTruth (CNN/DM summaries) | 5% | 7.0k | claim vs document; answerability within one paper; summaries with human hallucination spans | variety of documents |
| **HelpSteer2 / HelpSteer3** | 11% | 16.4k | rubric scores, as in Stage A | Score |

- **148k records per epoch × 2 epochs = 297k**, max 1024 tokens: Bool 71%, Choice 15%, Score 14%.
  Records longer than 1024 tokens are dropped (RAGTruth 38%, HelpSteer3 64%, RAGBench 10%): long
  inputs are left for long-context training.
- **The user's question is written into every claim**, and the wording avoids the templates of the
  judge benchmarks, so gains there are not a template effect.
- Every record carries `label_source` (`human` / `llm` / `program`), so results can be split by it.

### 4.3 Test sets of the model

All test sets are **cleaned**: any record whose passage also occurs in the training data is removed
(~4.5% overall for Stage A, up to 27–60% for BoolQ, HoVer and HelpSteer3, which reuse passages).

| Split | Records | What it measures |
|---|---|---|
| `test_in` | 19,272 | Stage A sources, trained templates: in-distribution |
| `test_ood_template` | 17,302 | Stage A sources, **held-out question wording** |
| `test_ood_source` | 11,883 | **sources never trained on**: ContractNLI (legal NDAs), BANKING77 (intents), MultiNLI mismatched genres, FaithEval (unanswerable and counterfactual contexts), MMLU |
| Stage B new sources | ~21,000 | 1,000 per new source and split, same distribution as their training part |

`test_ood_source` is the number that matters most among these for the judge: in production the model
meets new domains first. Stage B keeps the Stage A tests unchanged, so forgetting is measured on the
same records (47,557 of them compared, Part 3, section 6.7).

### 4.4 Judge benchmarks (evaluation only) and eval v2

Four public benchmarks that test a model *as a judge* of RAG answers, never trained on:

| Benchmark | Records | Task | Labels |
|---|---|---|---|
| ContextualJudgeBench (CJB) | 2,000 pairs | which of two answers is better, on refusal / faithfulness / completeness / conciseness | human + perturbations |
| RAGBench | 11,802 | is the answer supported by the documents, 12 domains incl. finance tables | GPT-4o |
| HaluBench | 14,900 | is the answer hallucinated, 6 sources incl. medical and finance | source datasets + perturbations |
| FaithBench | 723 | subtle hallucinations in summaries, chosen where detectors disagree | human spans |

Stage B trains on the train splits of 7 RAGBench subsets, so their test splits are no longer
out-of-distribution. Before Stage B was trained, the benchmarks were frozen as **eval v2**, in two
slices:

- **Held-out slice** (comparable across all stages, ~21,000 records): CJB, FaithBench, HaluBench, and
  the RAGBench subsets never trained on (covidqa, cuad, delucionqa, emanual, msmarco; 1,491 records,
  so ±0.04 on them).
- **In-distribution slice** (reported separately, 7,641 records): the other 7 RAGBench subsets.
- Both leave out every record whose passage occurs in the Stage B training files (2,639 records).
- **"Seen source" flags** in the held-out slice: HaluBench HaluEval (built on HotpotQA, which Stage B
  trains on) and PubMedQA (trained via RAGBench); CJB's ASQA part (via Trust-Data). HaluBench is
  therefore also reported without HaluEval.

Stage A, NLI and Laya were re-scored on eval v2 from their saved predictions, so all numbers of Part 4
are on the same records.

### 4.5 Calibration data

Calibration is fitted on data that neither training nor evaluation uses:

- the `calib` splits of the Stage A and Stage B mixes (19,921 + 10,182 records), for the temperatures;
- **train + validation of the 4 held-out RAGBench subsets** (covidqa, cuad, delucionqa, emanual), minus
  every record whose passages occur in their test split: **2,972 records, 325 unsupported**, for the
  judge threshold. Their test splits stay in the held-out slice; the ids are frozen.

### 4.6 Data issues found on the way

- **ClapNQ** (Stage A): answerable and unanswerable examples come from different passage pools, so the
  model could tell the label from the passage alone (Stage A scored 1.000). **Fixed in Stage B**: both
  labels now share the passage; Stage B scores 0.83 on the old test, and every error is an unanswerable
  question called answerable, i.e. the shortcut is gone and the real skill is partial.
- **TabFact**: with an unrelated table, Stage A still says "supported" to 54% of claims; it judges the
  claim's plausibility more than the table. Stage B adds hard negatives on the same table (TAT-QA,
  FinQA).
- **Some Stage B sources are easy**: Faithfulness-QA 0.99 and PsiloQA 0.96 on their own tests. Their
  negatives may differ from the positives in surface features; to be checked with the probes.
- **Duplicated test ids**: the Stage A test replay and HelpSteer's own test split shared ids; this broke
  the first three Stage B launches and was fixed in the mix builder (each eval id in one split only).

These were found with a **passage-swap probe** (give the model another record's passage; a model that
reads the passage should then say "no") and with per-source reports. Both are part of the standard
evaluation.

## 5. Training methodology, calibration and baselines

### 5.1 Training recipe

| | Stage A | Stage B |
|---|---|---|
| Start | raw ModernBERT-large | Stage A final checkpoint (encoder + heads) |
| Hardware | Kaggle, 2× T4 (free tier), DDP, fp16 | same |
| Optimiser | 8-bit AdamW, lr 3e-5, warmup 6%, linear decay | same; heads at lr 1e-3 (they barely moved at 3e-5 in Stage A) |
| Batch | effective 32, length-grouped | same (8 per GPU × 2 GPUs × 2 accumulation) |
| Max length | 512 tokens | 1024 tokens |
| Length | 2 epochs = 18,750 steps, 9.5 h | 2 epochs = 9,268 steps, 7.5 h (2.9 s/step) |
| Augmentation | Choice options shuffled every epoch | same |

Stage B ran in one Kaggle session with no errors; gradient norms as in Stage A; dev accuracy on the
new sources rose to the end (Bool 0.85 → 0.92), with no sign of overfitting there.

### 5.2 Calibration

Fitted after training on saved predictions (section 4.5): no GPU, seconds on a laptop.

| Layer | What it is | Fitted on | What it changes |
|---|---|---|---|
| **Model** (default) | one temperature per primitive; for Choice it depends on the number of options: T = 2.46·K^−0.17 (2.2 at K = 2), T_Bool = 1.68, T_Score = 1.56 | `calib` of the Stage A and B mixes | probabilities only; **no answer changes** |
| **Judge** (optional, faithfulness of RAG answers) | a shift and scale of the Bool logit (Platt scaling), fitted with both classes weighted equally | 2,972 held-out RAGBench records | the decision threshold for "supported" |

The judge layer is fitted with equal class weights on purpose: otherwise it would learn RAGBench's 89%
share of "supported" answers, which is a property of that dataset, not of our system. The real share can
be added back as one constant once it is known from our own labelled answers.

### 5.3 Baselines

| | What it is | Why it is here |
|---|---|---|
| **NLI** | `DeBERTa-v3-large` NLI cross-encoder (435M), zero-shot | the standard open faithfulness checker; strong on claim-vs-text |
| **Laya** | Convai's open JEV-like model (ModernBERT-large + decision head, 421M), zero-shot | the closest open analogue to jev-lite |
| **JEV API** | `jev-1.13.0`, 16,460 records (against Stage A) | the closed model we imitate; the reference |
| **Stage A** | our previous model | the baseline for Stage B on every test |
| **Published LLM judges** | GPT-4o, o1, 8B judges, etc., from the benchmark papers | orientation on the judge benchmarks (not run by us) |

The open baselines are zero-shot, while jev-lite was trained on the training splits of the same
sources. That is why the fair comparisons are on `test_ood_source` and on the held-out slice of the
judge benchmarks.

### 5.4 How results are compared

- **Paired tests on the same records**: Δ accuracy with a paired bootstrap 95% CI and McNemar's exact
  test, Holm-corrected for multiple comparisons. A cell is called a **win** or a **loss** only when both
  tests agree; otherwise it is a **tie**.
- **Metrics**:
  - *Accuracy*: share of correct answers; *balanced accuracy* (mean of per-class recall) where labels
    are skewed.
  - *AUROC*: how well the probabilities rank right above wrong, independent of the 0.5 threshold.
  - *ECE* (calibration error): average gap between "the model says 80%" and "it is right 80% of the
    time". Lower is better; 0 is perfect.
  - *Risk–coverage*: error rate when only the most confident X% of answers are kept: how well the
    confidence separates right from wrong answers.
  - *Response time*: ms per question (wall time of a call / questions in it), p50 / p95 / mean.
- **Black-box probes**: option order shuffles, adding an irrelevant option, nonsense inputs, empty and
  swapped passages.

[Next: Model test results →](03_stage_a_results.md)
