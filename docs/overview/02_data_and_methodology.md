# Part 2. Data, training and evaluation methodology

> jev-lite overview, October 2026 · Part 2 of 5 ·
> [← Previous: Introduction](01_introduction.md) · [Index](README.md) ·
> [Next: Stage A results →](03_stage_a_results.md)

## 4. Data used so far

**Selection rules.** Public English datasets with **human labels** and a license that allows
commercial use. One exception for LLM-written data: WANLI (GPT-3 wrote the pairs, humans labelled
and revised them). Share-alike (CC BY-SA) datasets are included after legal confirmed it on
2026-10-07. Full license checks are in `DATASETS.md`.

### 4.1 Training: 25 datasets

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

### 4.2 Test sets

All test sets are **cleaned**: any record whose passage also occurs in the training data is removed
(~4.5% overall, up to 27–60% for BoolQ, HoVer and HelpSteer3, which reuse passages).

| Split | Records | What it measures |
|---|---|---|
| `test_in` | 19,272 | trained sources, trained templates: in-distribution |
| `test_ood_template` | 17,302 | trained sources, **held-out question wording** |
| `test_ood_source` | 11,883 | **sources never trained on**: ContractNLI (legal NDAs), BANKING77 (intents), MultiNLI mismatched genres, FaithEval (unanswerable and counterfactual contexts), MMLU |

`test_ood_source` is the number that matters most for the judge: in production the model meets new
domains first.

### 4.3 Judge benchmarks (evaluation only)

Four public benchmarks that test a model *as a judge* of RAG answers, never trained on, 31,300 records
in total:

| Benchmark | Records | Task | Labels |
|---|---|---|---|
| ContextualJudgeBench (CJB) | 2,000 pairs | which of two answers is better, on refusal / faithfulness / completeness / conciseness | human + perturbations |
| RAGBench | 11,802 | is the answer supported by the documents, 12 domains incl. finance tables | GPT-4o |
| HaluBench | 14,900 | is the answer hallucinated, 6 sources incl. medical and finance | source datasets + perturbations |
| FaithBench | 723 | subtle hallucinations in summaries, chosen where detectors disagree | human spans |

### 4.4 Data issues found on the way

- **ClapNQ**: answerable and unanswerable examples come from different passage pools, so the model can
  tell the label from the passage alone. jev-lite scored 1.000 on it, which was a shortcut. It is kept
  out of the headline answerability numbers and fixed for Stage B.
- **TabFact**: with an unrelated table, jev-lite still says "supported" to 54% of claims; it judges the
  claim's plausibility more than the table. Table data needs hard negatives on the same table.

Both were found with a **passage-swap probe** (give the model another record's passage; a model that
reads the passage should then say "no"). It is now part of the standard evaluation.

## 5. Training methodology and baselines

### 5.1 Training recipe

| | |
|---|---|
| Hardware | Kaggle, 2× T4 (free tier), DDP, fp16 mixed precision |
| Optimiser | 8-bit AdamW, lr 3e-5, warmup 6%, linear decay |
| Batch | effective 32, length-grouped; max length 512 tokens |
| Length | 2 epochs = 18,750 steps, 566 min of training, one Kaggle session |
| Augmentation | Choice options shuffled every epoch (prevents position bias) |
| Calibration | none yet: raw probabilities, no temperature scaling |

### 5.2 Baselines

| | What it is | Why it is here |
|---|---|---|
| **NLI** | `DeBERTa-v3-large` NLI cross-encoder (435M), zero-shot | the standard open faithfulness checker; strong on claim-vs-text |
| **Laya** | Convai's open JEV-like model (ModernBERT-large + decision head, 421M), zero-shot | the closest open analogue to jev-lite |
| **JEV API** | `jev-1.13.0`, 16,460 records | the closed model we imitate; the reference |
| **Published LLM judges** | GPT-4o, o1, 8B judges, etc., from the benchmark papers | orientation on the judge benchmarks (not run by us) |

The open baselines are zero-shot, while jev-lite was trained on the training splits of the same
sources. That is why the fair comparisons are on `test_ood_source` and on the judge benchmarks.

### 5.3 How results are compared

- **Paired tests on the same records**: Δ accuracy with a paired bootstrap 95% CI and McNemar's exact
  test, Holm-corrected for multiple comparisons. A cell is called a **win** or a **loss** only when both
  tests agree; otherwise it is a **tie**.
- **Metrics**:
  - *Accuracy*: share of correct answers; *balanced accuracy* (mean of per-class recall) where labels
    are skewed.
  - *AUROC*: how well the probabilities rank right above wrong, independent of the 0.5 threshold.
  - *ECE* (calibration error): average gap between "the model says 80%" and "it is right 80% of the
    time". Lower is better; 0 is perfect.
  - *Risk–coverage*: error rate when only the most confident X% of answers are kept. This is the
    cascade view.
- **Black-box probes**: option order shuffles, adding an irrelevant option, nonsense inputs, empty and
  swapped passages.

[Next: Stage A results →](03_stage_a_results.md)
