# Evaluation

How jev-lite is evaluated: test sets, benchmarks, baselines, metrics and how to reproduce them. This
page holds the method only; the numbers of each version are in its release notes
([`docs/releases/`](releases/)). Sources and their licenses: [`DATASETS.md`](DATASETS.md).

All numbers are offline, on public data. None of them measures agreement with a production LLM judge
on our own answers; that needs a human-labelled judge test set, which is planned and not built yet.

## 1. Test sets

### 1.1 Primitive tests (Stage A mix)

The `stage_a` mix (`configs/mixes/stage_a.yaml`, built by `scripts/build_mix.py`) cuts every source into
`train` / `dev` / `calib` / `test_in` / `test_ood`. The tests are frozen and shared by every model and
stage, so all versions are compared on the same records.

| Split | What it is | Used for |
|---|---|---|
| `test_in` | trained sources, trained templates | in-distribution accuracy |
| `test_ood_template` | trained sources, held-out question templates | wording robustness |
| `test_ood_source` | sources never trained on: ContractNLI, BANKING77, MultiNLI mismatched genres, FaithEval (unanswerable, counterfactual), MMLU | **the fair test**: the only split where zero-shot baselines and jev-lite start equal |
| `dev` | the unused remainder of the `test_in` pools | mid-run eval and decisions during training |
| `calib` | held out from training and tests | fitting the calibration (section 4) |

**Cleaning** (`scripts/clean_eval.py`, `jevlite/data/overlap.py`): a test record is dropped when its
passage occurs in any training file (exact or first-200-characters match per passage), and so are the
records that took part in decisions. This removed ~4.5% overall, but up to 28–60% on BoolQ, HoVer and
HelpSteer3, where many records share Wikipedia intros or responses. Headline numbers are on the clean sets.

### 1.2 Judge benchmarks (eval only, never trained on)

| Benchmark | Records | Primitive | Labels | Main metric |
|---|---|---|---|---|
| **ContextualJudgeBench** (CJB) | 2,000 pairs → 4,000 records, 8 splits (refusal, faithfulness, completeness, conciseness × QA / summaries) | Choice K = 2, both responses in the state, both orders | human + perturbations | consistent accuracy: right in both orders (random 0.25) |
| **RAGBench** | 11,802, 12 test subsets | Bool "the answer is supported by the documents" | GPT-4o | balanced accuracy, AUROC |
| **HaluBench** | 14,900, 6 sources | Bool | PASS / FAIL (source datasets + perturbations) | balanced accuracy, AUROC |
| **FaithBench** | 723 of 800 (Questionable left out) | Bool, the summary is the claim | human spans, `worst-label` pooling | balanced accuracy |

Converters: `src/jevlite/data/public/judge_bench.py`. Faithfulness records use the NLI form of
LLM-AggreFact (context = state, answer = claim), so the NLI baseline reads them the same way. Prompts are
ours, written once for this evaluation and never tuned.

### 1.3 eval v2: held-out and in-distribution slices

From Stage B on, the train/validation splits of 7 RAGBench subsets are training data, so the judge
benchmarks are reported in two slices. The record lists are frozen in `docs/judge_bench/eval_v2/`
(`scripts/make_eval_v2.py`) **before** a stage trains; earlier models are re-scored on them from their
saved predictions.

| Slice | Contents | Records |
|---|---|---|
| **held-out** (comparable across versions) | CJB, FaithBench, HaluBench, RAGBench covidqa / cuad / emanual / delucionqa / msmarco | 21,054 |
| **in-distribution** (reported separately) | RAGBench hotpotqa, finqa, tatqa, hagrid, expertqa, techqa, pubmedqa | 7,630 |

- Left out: records whose passage occurs in any training file (`docs/judge_bench/overlap_ids.txt`,
  `docs/judge_bench/eval_v2/stage_b_overlap_ids.txt`; 2,688 ids).
- **"Seen source" flags** inside the held-out slice, reported with and without: HaluBench HaluEval (built
  on HotpotQA, which Stage B trains on; two thirds of HaluBench) and HaluBench PubMedQA (via RAGBench
  pubmedqa); CJB's QA-Feedback pairs (ASQA, via Trust-Data).
- The held-out RAGBench part is small (1,490 records, ±0.04 overall, ±0.07–0.13 per subset): differences of
  a few points there are noise. CJB and HaluBench carry most of the held-out signal.
- **Rollback rule:** a new stage that is worse than the previous one on the held-out slice is not shipped.

### 1.4 Context length

Most judge-benchmark records are longer than 512 tokens (CJB 52–88% per split, RAGBench 39–100% per
subset). jev-lite runs at several budgets (512, 1024, 8192); at a budget shorter than the input the state is
cut head + tail, the question header never. Reports carry a `fits512` slice of records that fit untruncated.

## 2. Baselines

| | NLI | Laya |
|---|---|---|
| Model | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (435M, MIT) | `convaiinnovations/laya` base (421M, Apache-2.0) |
| What it is | 3-way NLI cross-encoder | a Jev-like typed-decision model: ModernBERT-large + decision head |
| Bool | P(entailment) of (state, claim) | 2-option question |
| Choice / Score | one templated hypothesis per candidate, softmax over entailment logits | all options in one sequence, one `[MASK]` marker each |
| Wrapper | `src/jevlite/baselines/nli.py` | `src/jevlite/baselines/laya.py` |

Both are zero-shot, and jev-lite is trained on the train splits of most `test_in` sources, so `test_in`
favours jev-lite by construction. The NLI baseline was trained on ANLI (non-commercial): fine for an
offline comparison, not for shipping. Published LLM-judge numbers from the benchmarks' papers are quoted
for orientation only (other prompts, label versions and binarisation).

## 3. Metrics and statistics

- **Bool:** accuracy, **balanced accuracy** (mean recall of both classes), AUROC, NLL, ECE, and the share
  answered "supported" next to the label share. On skewed labels plain accuracy misleads (RAGBench is 86%
  "supported": always saying yes scores 0.86), so balanced accuracy and AUROC are the headline.
- **Choice:** accuracy, NLL, ECE; per K bucket. CJB: consistent accuracy and the share of first-position picks.
- **Score:** accuracy of the argmax level, Spearman, NLL, ECE.
- **Response time:** ms per record, p50 / p95 / mean, one request per call, by input-length bucket.
- **Intervals:** 95% bootstrap (1,000 resamples). **Paired** tests on the same records: Δ with a paired
  bootstrap CI and McNemar's exact test, Holm-corrected per table; "win" / "loss" only when both agree
  (`jevlite/paired.py`, `scripts/compare_paired.py`).

**Probes** (`scripts/probe_blackbox.py`, 200 items per test): order sensitivity of Choice over 8
permutations, IIA (an added wrong option should not change the ratios of the others), confidence on
nonsense states (shuffled words, random characters), and two Bool shortcut tests: **claim-only** (empty
state) and **passage swap** (the state of another record of the same source). Every answerability source
must pass the 2 × 2 passage-swap test before it enters a training mix (`DATASETS.md` 3.1).

## 4. Calibration

Post-hoc, on saved predictions, no GPU (`jevlite.calibration`, `scripts/calibrate.py`):

- **model layer:** a temperature per primitive (Choice `log T = a + b·log K`), fitted by NLL on the `calib`
  splits. Answers never change, only probabilities and confidence. On by default in
  `Predictor.from_final`.
- **judge layer:** class-balanced Platt scaling of the Bool logit on held-out RAG answers
  (`ragbench_calib`: RAGBench covidqa / cuad / delucionqa / emanual train + validation, minus every record
  sharing a passage with the RAGBench test; ids frozen in `docs/judge_bench/calib/ragbench_calib_ids.txt`).
  It moves the decision threshold, so it is opt-in and only for faithfulness claims.

Calibration is evaluated on data it was not fitted on: the Stage A tests for the model layer, eval v2 for
the judge layer.

## 5. Reproduction

Predictions run on Kaggle (2×T4, fp16) with the notebooks in `notebooks/kaggle/`; reports and charts run
locally from the saved predictions. Everything under `data/` is git-ignored.

```bash
# primitive tests: report, paired comparison, probes
R=data/runs/<run>; C=data/mix/stage_a/clean
uv run python scripts/eval.py --data $C/test_in.jsonl $C/test_ood_source.jsonl $C/test_ood_template.jsonl \
  --pred $R/preds.jsonl --out $R/report_clean --bootstrap 1000
uv run python scripts/compare_paired.py --data $C/test_in.jsonl $C/test_ood_source.jsonl $C/test_ood_template.jsonl \
  --model jevlite=$R/preds.jsonl --ref nli=data/baselines/preds/nli.jsonl --ref laya=data/baselines/preds/laya.jsonl --out $R/paired
HF_HUB_OFFLINE=1 uv run python scripts/probe_blackbox.py --baseline jevlite --model $R/final --data $C/test_in.jsonl \
  --out $R/probes.json --n-items 200 --shortcuts 200

# judge benchmarks: data (eval only), predictions on Kaggle (notebooks/kaggle/30_judge_bench.ipynb), report per slice
uv run python scripts/convert_public.py --datasets contextual_judge_bench,ragbench,halubench,faithbench --out data/public
P=data/judge_bench/kaggle/preds
for s in heldout in_distribution; do
  uv run python scripts/judge_bench_report.py --data data/public/{contextual_judge_bench,ragbench,halubench,faithbench}.jsonl \
    --pred jevlite=$P/jevlite_8192.jsonl nli=$P/nli.jsonl laya=$P/laya.jsonl \
    --exclude docs/judge_bench/eval_v2/${s}_exclude.txt --tokenizer $R/final/encoder \
    --out data/judge_bench/eval_v2/$s --bootstrap 1000
done

# calibration: fit on calib predictions, apply to a predictions file
uv run python scripts/calibrate.py fit \
  --data data/mix/stage_a/calib.jsonl data/calib/stage_b_calib.jsonl \
  --pred data/calib/preds/stage_a_calib.jsonl data/calib/preds/stage_b_calib.jsonl \
  --judge-data data/calib/ragbench_calib.jsonl --judge-pred data/calib/preds/ragbench_calib.jsonl \
  --out data/calib/calibration.json
uv run python scripts/calibrate.py apply --calibration data/calib/calibration.json --layers model \
  --data data/mix/stage_a/test_in.jsonl data/mix/stage_a/test_ood.jsonl --pred $R/preds.jsonl --out $R/preds.cal.jsonl
```
