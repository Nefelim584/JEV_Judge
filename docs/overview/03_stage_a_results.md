# Part 3. Stage A results and the comparison with the JEV API

> jev-lite overview, October 2026 · Part 3 of 5 ·
> [← Previous: Data and methodology](02_data_and_methodology.md) · [Index](README.md) ·
> [Next: Judge benchmarks →](04_judge_benchmarks.md)
>
> Model: run `stage_a-full-s42` (= `Nefelim584/jev-lite-stage-a`). Clean test sets of Part 2,
> section 4.2. Full reports: `STAGE_A.md`, `STAGE_A_API.md`, `BASELINES.md`.

## 6. Results

### 6.1 Against the open baselines

Accuracy, **NLI / Laya / jev-lite**, on all 48,457 clean test records:

| | `test_in` (trained) | `test_ood_template` (new wording) | `test_ood_source` (new sources) |
|---|---|---|---|
| Bool | 0.763 / 0.723 / **0.882** | 0.569 / 0.770 / **0.902** | 0.542 / 0.519 / **0.603** |
| Choice | 0.595 / 0.596 / **0.803** | 0.617 / 0.622 / **0.794** | 0.558 / 0.585 / **0.635** |
| Score | 0.242 / 0.309 / **0.620** | 0.234 / 0.315 / **0.591** | — |
| Bool calibration error (ECE, lower is better) | 0.161 / 0.076 / **0.031** | 0.314 / 0.051 / **0.026** | 0.411 / 0.367 / **0.295** |

- **On trained sources and on new wording jev-lite is far ahead**: +12 (Bool), +17–21 (Choice) and
  +28–31 (Score) points over the better baseline, all significant. New question wording costs nothing,
  so the model does not depend on exact templates.
- **On new sources the lead shrinks to +5–6 points** and is not uniform. Over all 61
  source × split cells: against NLI 43 wins, 16 ties, **2 losses** (ContractNLI Bool −4.8, FaithEval
  counterfactual −6.1); against Laya 52 wins, 9 ties, 0 losses.
- **Calibration is excellent on familiar data** (ECE 0.03) and **breaks on new sources** (0.30).

### 6.2 Against the JEV API

JEV (`jev-1.13.0`) on a balanced sample of 5,600 records (700 per split × primitive), plus a top-up to
16,460 records for per-source numbers. Δ = JEV − jev-lite, paired.

| Split | Bool: JEV / jev-lite | Choice: JEV / jev-lite | Score: JEV / jev-lite |
|---|---|---|---|
| `test_in` | 0.853 / 0.864 (tie) | 0.856 / 0.810 (tie) | 0.461 / **0.627** (jev-lite +16.6) |
| `test_ood_template` | 0.891 / 0.880 (tie) | 0.827 / 0.791 (tie) | 0.443 / **0.584** (jev-lite +14.1) |
| `test_ood_source` | **0.763** / 0.601 (JEV +16.1) | **0.884** / 0.627 (JEV +25.7) | — |

![JEV vs jev-lite by split](../stage_a_api/figures/paired_by_split.png)

*Paired Δ accuracy (JEV − jev-lite) with 95% CI. Amber = JEV wins, green = jev-lite wins, gray = tie.*

- **JEV beats NLI and Laya everywhere.** Against jev-lite it depends on the data: **tied on what
  jev-lite was trained on, far ahead on new sources**, behind on Score (HelpSteer2 rubrics).
- **Per source: 10 wins, 10 losses, 16 ties.** JEV wins on knowledge-heavy multiple choice (MMLU +52,
  ARC +38, OpenBookQA +34), on new faithfulness data (FaithEval +21, ContractNLI +6 / +28) and on
  tables (TabFact +23). jev-lite wins on the entailment and label sets it was trained on (VitaminC +17,
  HoVer +12, MultiNLI +10, go_emotions +25, HelpSteer2 +17).

![Accuracy by source, four models](../stage_a_api/figures/accuracy_by_source.png)

*Accuracy per source and primitive, all 16,460 records. JEV (amber) leads on most rows; where
jev-lite (green) is ahead of it, the source is one jev-lite was trained on. Below the dashed line
jev-lite does not beat both open baselines.*

### 6.3 Faithfulness: the judge's main criterion

![Claim vs text accuracy by group](../stage_a_api/figures/faithfulness.png)

*Bool accuracy on "is this claim supported by the text". jev-lite leads overall (0.86 vs JEV 0.81),
on FEVER and on multi-hop HoVer at every hop count; JEV leads on held-out legal contracts
(ContractNLI, 0.83 vs 0.77).*

- **Multi-hop claims (HoVer, 2–4 documents per claim)** are the closest public proxy for checking an
  answer against several chunks. jev-lite keeps its accuracy as the number of hops grows and catches
  "not supported" in 82–84% of cases (NLI 46%, Laya 25%; JEV 60% on its sample).
- **ContractNLI shows the failure mode on a new domain**: jev-lite *ranks* claims better than NLI
  (AUROC 0.93 vs 0.88), but its probabilities are shifted down, so at the 0.5 threshold it calls half
  of the supported claims unsupported. On a new domain the model is **too strict, not too lenient**.

### 6.4 Calibration and the cascade

![Risk vs coverage](../stage_a_api/figures/risk_coverage.png)

*Error rate when only the most confident share is answered. Lower and flatter is better for a cascade.*

- jev-lite's confidence separates right from wrong answers: on Bool and Choice its error stays near
  zero for the most confident 10–20% and rises slowly; the open baselines' curves rise almost linearly
  from the start.
- On Stage A's full test set, escalating the 20% least confident Bool answers raises jev-lite's
  accuracy from 0.88 to 0.95 (first-version numbers in `STAGE_A.md` section 5). This is what makes
  the cascade viable.

### 6.5 Behaviour probes

| | NLI | Laya | jev-lite | JEV |
|---|---|---|---|---|
| Answer flips when options are reordered | 0% (by design) | 12.4% | 2.8% | **2.1%** |
| Mass taken by an irrelevant extra option | 0.103 | 0.108 | 0.020 | **0.010** |
| Claim only, no passage: accuracy on FEVER (majority 0.68; should stay at it) | 0.68 | 0.68 | 0.68 | 0.81 |
| Unrelated passage: share "supported" on VitaminC | ≤ 0.10 | — | 0.01 | 0.45 |

- **Option order**: jev-lite and JEV barely depend on it; option shuffling in training worked.
- **JEV answers partly from world knowledge, not from the text.** Without the passage it still gets
  FEVER and BoolQ claims right well above the majority rate, and with an unrelated passage it calls 45%
  of VitaminC claims supported. For a judge that must check an answer *against the retrieved context*
  this is the main risk found in JEV. jev-lite falls to the majority rate without the passage: it reads
  the text.
- **JEV is not deterministic**: the same request twice gave different probabilities for 61 of 100
  records (by up to 0.11), and 2 of 100 answers flipped.

### 6.6 Speed

Apple M1 Pro laptop, one record per call, the same 300 records for every model; JEV's times are API
round trips.

| ms per record | jev-lite | Laya | NLI | JEV (API) |
|---|---|---|---|---|
| median | **73** | 77 | 241 | 195 |
| p90 | **152** | 160 | 1,895 | 224 |

![Latency per model and primitive](../stage_a_api/figures/speed_latency_by_primitive.png)

*Dot = median, line = 10th–90th percentile, log scale.*

jev-lite on a laptop GPU is faster than a JEV API call at every input length in the test sets: 4.2×
on the shortest fifth, 1.3× on the longest; no network and no per-call cost. A server GPU would be
faster still. JEV's side scales by running requests in parallel.

[Next: Judge benchmarks →](04_judge_benchmarks.md)
