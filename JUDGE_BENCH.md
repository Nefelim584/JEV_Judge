# Judge benchmarks: jev-lite Stage A as a context judge

> Run `stage_a-full-s42` (the model on the Hub, [`Nefelim584/jev-lite-stage-a`](https://huggingface.co/Nefelim584/jev-lite-stage-a)),
> NLI and Laya on four public judge benchmarks, evaluated on 2026-10-18. jev-lite runs twice: at
> 512 tokens (its training length, longer inputs truncated) and at 8192 (untruncated). None of the
> benchmarks was trained on (`DATASETS.md` section 7). Offline public-data numbers: they do **not**
> answer H0 (jev-lite vs the production LLM judge, Phase 11). No LLM judge was run here; the frontier
> numbers in section 5 come from the benchmarks' papers.

## Summary

- **jev-lite holds up beyond its training length.** At 8192 tokens it is better than at 512 almost
  everywhere the inputs are long (HaluBench CovidQA balanced accuracy 0.544 → 0.755, AUROC 0.55 → 0.85;
  ContextualJudgeBench faithfulness QA 0.286 → 0.440), and never worse on Bool overall (+3.7 accuracy,
  paired). The exception is **conciseness** (CJB QA 0.280 → 0.102): with the full answers visible it
  prefers the longer one.
- **Pairwise judging (ContextualJudgeBench): far ahead of NLI and Laya, at the level of 8B judges,
  far from frontier.** Consistent accuracy 0.332 (random 0.25; NLI 0.090, Laya 0.024), against
  GPT-4o 0.512 and o1 0.631 in the paper; 0.440 on faithfulness QA, where 8B judges get 0.31–0.41 and
  GPT-4o 0.704.
- **FaithBench (subtle summary hallucinations): at the level of GPT-4-class detectors.** Balanced
  accuracy 0.579 [0.538, 0.614] vs GPT-4-Turbo 0.577 and GPT-4o 0.563 in the paper; NLI 0.514, Laya 0.480.
- **HaluBench: a tie with NLI on accuracy, NLI ranks slightly better** (AUROC 0.836 vs 0.815). About
  13 points below GPT-4o's vendor-reported 0.865.
- **RAGBench: the weakest result.** Mean AUROC over the 12 subsets 0.629 (NLI 0.590; the paper:
  RAGAS 0.60, GPT-3.5 0.56, a DeBERTa fine-tuned on RAGBench 0.79), balanced accuracy 0.53. jev-lite
  says "not supported" far too often: 51% "supported" against 86% in the labels, and **4% on FinQA**
  (91% in the labels), where the answers are computed from tables.
- **The threshold drifts on new data, again** (Stage A, `STAGE_A.md` 3.2): sceptical on RAGBench,
  slightly credulous on FaithBench. Ranking (AUROC) is the more reliable signal on new sources.

## 1. Setup

| Benchmark | Records | Primitive | Labels | Main metric |
|---|---|---|---|---|
| **ContextualJudgeBench** (CJB) | 2,000 pairs → 4,000 records, 8 splits | Choice K = 2: both responses in the state, both orders | human annotations + perturbations | consistent accuracy (right in both orders; random 0.25) |
| **RAGBench** | 11,802, 12 test subsets | Bool "the answer is supported by the documents" | GPT-4o (`annotating_model_name`) | AUROC (mean over subsets, as in the paper), balanced accuracy |
| **HaluBench** | 14,900, 6 sources | Bool | PASS / FAIL (source datasets + perturbations) | accuracy (as in the vendor table), AUROC |
| **FaithBench** | 723 of 800 (Questionable left out) | Bool, the summary is the claim | human spans, `worst-label` pooling | balanced accuracy |

- **Converters:** `src/jevlite/data/public/judge_bench.py`; prompts and mapping in its docstring.
  Faithfulness records use the NLI form of LLM-AggreFact (context = state, answer = claim), so the NLI
  baseline reads them the same way.
- **Passage overlap with Stage A training:** 49 records (31 RAGBench, 4 HaluBench, 7 CJB pairs) share
  a passage with the `stage_a` / `stage_a_nosa` training data (`clean_eval.py` prefix mode; CJB checked
  per passage of its concatenated contexts). They are left out (`docs/judge_bench/overlap_ids.txt`).
- **Common records:** every number is on the **31,300 records all four runs predicted**. Skipped:
  jev-lite 512: 58 (RAGBench / HaluBench answers longer than the 512-token budget: the header is never
  truncated); NLI: 40 (same reason); Laya: 18 CJB records (a bug in our wrapper: a confidence of
  1.0000001 fails validation; section 7); jev-lite 8192: none.
- **Length:** most records exceed 512 tokens (CJB 52–88% per split, RAGBench 39–100% per subset). At
  512 the state is cut head + tail. Each table in `data/judge_bench/report/report.md` has a `fits512`
  slice (records that fit untruncated).
- **Hardware:** Kaggle 2×T4, fp16 autocast for jev-lite and NLI. Throughput over the whole mix:
  jev-lite 27 rec/s at 512, 9.3 rec/s at 8192; NLI 21 rec/s; Laya 26 rec/s. 71 minutes for all four runs.
- **Statistics:** 95% bootstrap intervals (1,000 resamples) in brackets; Δ between two runs are
  paired (same records). Accuracy-based paired tests (`compare_paired.py`, McNemar + Holm) are in
  `data/judge_bench/paired/`; paired Δ of balanced accuracy, AUROC and consistent accuracy were
  computed ad hoc on the same records (500–1,000 resamples).

## 2. Headline

![Headline metric per benchmark, four runs](docs/judge_bench/figures/headline.png)

[Interactive version](docs/judge_bench/figures/headline.html)

| Benchmark · metric | jev-lite 8192 | jev-lite 512 | NLI | Laya | Published (paper) |
|---|---|---|---|---|---|
| CJB · consistent accuracy | **0.332** | 0.312 | 0.090 | 0.024 | GPT-4o 0.512, o1 0.631 |
| CJB · faithfulness QA | **0.440** | 0.286 | 0.121 | 0.069 | GPT-4o 0.704, o1 0.852 |
| FaithBench · balanced accuracy | **0.579** | 0.567 | 0.514 | 0.480 | GPT-4o 0.563, GPT-4-Turbo 0.577 |
| HaluBench · accuracy | 0.735 | 0.717 | **0.740** | 0.654 | GPT-4o 0.865 (vendor) |
| HaluBench · AUROC | 0.815 | 0.790 | **0.836** | 0.719 | — |
| RAGBench · AUROC, mean of 12 subsets | **0.629** | 0.602 | 0.590 | 0.569 | GPT-3.5 0.56, RAGAS 0.60, fine-tuned DeBERTa 0.79 |
| RAGBench · balanced accuracy (pooled) | 0.532 | **0.538** | 0.508 | 0.490 | — |

Paired differences, jev-lite 8192 minus the reference:

| | vs NLI | vs Laya | vs jev-lite 512 |
|---|---|---|---|
| CJB consistent accuracy | **+24.2** [+22.1, +26.4] | **+30.8** [+28.7, +33.0] | **+2.0** [+0.1, +3.9] |
| FaithBench balanced accuracy | **+6.4** [+2.3, +10.4] | **+9.8** [+5.4, +14.2] | +1.2 [−0.9, +3.1] |
| FaithBench AUROC | **+0.138** [+0.066, +0.201] | **+0.113** [+0.046, +0.177] | **+0.026** [+0.003, +0.049] |
| HaluBench balanced accuracy | −0.2 [−0.9, +0.6] | **+8.7** [+7.9, +9.7] | **+1.6** [+1.3, +1.9] |
| HaluBench AUROC | **−0.021** [−0.026, −0.015] | **+0.096** [+0.089, +0.104] | **+0.025** [+0.022, +0.028] |
| RAGBench balanced accuracy | **+2.4** [+1.0, +3.8] | **+4.2** [+2.7, +6.1] | −0.6 [−1.5, +0.3] |
| RAGBench AUROC (pooled) | −0.009 [−0.024, +0.008] | **+0.053** [+0.037, +0.071] | +0.005 [−0.002, +0.013] |

Bold: the interval excludes 0. **Plain accuracy misleads on skewed labels.** On RAGBench (86%
"supported") NLI reaches accuracy 0.832 by saying "supported" to 96% of the records; the paired
accuracy test then calls it a −31-point loss for jev-lite (`data/judge_bench/paired/`), while balanced
accuracy puts jev-lite ahead. The same effect, reversed, gives jev-lite +22 accuracy on FaithBench (33%
"supported"). Balanced accuracy and AUROC are the fair comparison here.

## 3. Per benchmark

### 3.1 ContextualJudgeBench: pairwise judging

| Split | jev-lite 8192 | jev-lite 512 | NLI | Laya | GPT-4o | o1 | SFRJudge-8B | Llama-3.1-8B |
|---|---|---|---|---|---|---|---|---|
| refusal, answerable | **0.460** | 0.360 | 0.192 | 0.040 | 0.496 | 0.948 | 0.320 | 0.000 |
| refusal, unanswerable | 0.280 | **0.296** | 0.020 | 0.012 | 0.604 | 0.472 | 0.468 | 0.936 |
| faithfulness, QA | **0.440** | 0.286 | 0.121 | 0.069 | 0.704 | 0.852 | 0.388 | 0.308 |
| faithfulness, summaries | **0.465** | 0.419 | 0.083 | 0.033 | 0.520 | 0.612 | 0.376 | 0.356 |
| completeness, QA | **0.317** | 0.272 | 0.041 | 0.008 | 0.388 | 0.500 | 0.404 | 0.272 |
| completeness, summaries | **0.430** | 0.355 | 0.143 | 0.016 | 0.566 | 0.645 | 0.430 | 0.482 |
| conciseness, QA | 0.102 | **0.280** | 0.008 | 0.008 | 0.467 | 0.643 | 0.671 | 0.561 |
| conciseness, summaries | 0.168 | **0.234** | 0.111 | 0.004 | 0.348 | 0.373 | 0.443 | 0.496 |
| **all** | **0.332** | 0.312 | 0.090 | 0.024 | 0.512 | 0.631 | 0.438 | 0.427 |

LLM columns: Table 5 of the paper (criterion-specific prompts, like ours). 248–254 pairs per split
(intervals ±0.06): differences below ~0.08 within a split are noise.

- **Faithfulness is the strongest split** and the one closest to our criterion (a): 0.44–0.47,
  above the 8B judges, below GPT-4o. Stage A saw pairwise data from one source only
  (`summarize_from_feedback`), so this is mostly transfer from the Bool / Choice training.
- **Conciseness is the weakest** and gets worse with the full answers (section 4).
- **Position bias:** jev-lite 8192 picks the first response in 43.5% of the answers (order accuracy
  0.44 / 0.57): a mild preference for the second. NLI picks the first in 80%, Laya in 21%; their
  consistent accuracy near 0 is that bias, not judgement (on records that fit 512, NLI rises to 0.26).

### 3.2 HaluBench: a tie with NLI

| Source | n | jev-lite 8192 (BA / AUROC) | jev-lite 512 | NLI | Laya |
|---|---|---|---|---|---|
| HaluEval (HotpotQA-based) | 9,998 | 0.788 / 0.881 | 0.788 / 0.881 | **0.792 / 0.887** | 0.699 / 0.786 |
| CovidQA (long) | 1,000 | **0.755 / 0.849** | 0.544 / 0.553 | 0.595 / 0.666 | 0.519 / 0.534 |
| PubMedQA | 1,000 | 0.607 / 0.680 | 0.606 / 0.680 | **0.713 / 0.803** | 0.567 / 0.653 |
| RAGTruth | 899 | 0.635 / **0.688** | **0.643** / 0.684 | 0.524 / 0.606 | 0.539 / 0.567 |
| DROP | 998 | 0.565 / 0.589 | 0.563 / 0.587 | **0.587 / 0.623** | 0.519 / 0.528 |
| FinanceBench | 1,000 | **0.556 / 0.579** | 0.544 / 0.561 | 0.515 / 0.549 | 0.512 / 0.529 |

Two thirds of HaluBench is HaluEval, where jev-lite and NLI are equal. jev-lite wins where the
context is long (CovidQA, at 8192 only) and on RAGTruth; NLI wins PubMedQA clearly (+0.11 BA) and DROP.
DROP and FinanceBench need arithmetic over the passage: everyone is near chance.

### 3.3 RAGBench: too sceptical, and numbers

![How often each model says 'supported', against the labels](docs/judge_bench/figures/positive_rate.png)

[Interactive version](docs/judge_bench/figures/positive_rate.html)

| Subset | n | labels "supported" | jev-lite 8192 says so | BA | AUROC | NLI AUROC |
|---|---|---|---|---|---|---|
| hagrid | 1,290 | 0.84 | 0.85 | 0.614 | **0.726** | 0.687 |
| hotpotqa | 389 | 0.92 | 0.95 | 0.574 | **0.733** | 0.656 |
| covidqa | 246 | 0.84 | 0.81 | 0.615 | **0.735** | 0.667 |
| emanual | 132 | 0.89 | 0.66 | 0.669 | **0.721** | 0.569 |
| expertqa | 197 | 0.48 | 0.35 | 0.613 | **0.685** | 0.535 |
| msmarco | 419 | 0.87 | 0.82 | 0.599 | **0.630** | 0.576 |
| tatqa | 3,338 | 0.97 | 0.54 | 0.562 | 0.615 | **0.673** |
| pubmedqa | 2,449 | 0.71 | 0.63 | 0.559 | **0.586** | 0.565 |
| cuad | 509 | 0.92 | 0.29 | 0.544 | **0.565** | 0.451 |
| techqa | 267 | 0.65 | 0.29 | 0.536 | 0.543 | 0.541 |
| delucionqa | 184 | 0.93 | 0.79 | 0.520 | 0.506 | 0.509 |
| finqa | 2,294 | 0.91 | **0.04** | 0.501 | 0.499 | **0.650** |

- **Prose QA works** (hagrid, hotpotqa, covidqa, emanual, expertqa: AUROC 0.69–0.74, ahead of NLI).
- **Tables and computed numbers fail** (finqa, tatqa: AUROC 0.50–0.62, NLI ahead). The responses state
  values derived from the tables; the text has no literal match, so jev-lite says "not supported". The
  same weakness as TabFact in Stage A (`STAGE_A.md` 6.4), now on real RAG data.
- **Long documents** (cuad, techqa: 5–24k tokens) are weak at either length.
- **The labels are GPT-4o's**, with its notion of "supported" (a correct computation counts). RAGBench
  measures agreement with an LLM judge, which is our H0 setting, but not ground truth.

### 3.4 FaithBench: subtle hallucinations

723 summaries chosen because hallucination detectors disagree on them; 67% are labelled hallucinated.
jev-lite 8192: balanced accuracy 0.579, AUROC 0.619; it says "supported" to 47% (labels 33%). NLI and
Laya say "supported" to 96% and 89%, so their balanced accuracy is near 0.5 and their AUROC at chance
(0.48, 0.51). The paper's GPT-4-class detectors score 0.53–0.58 balanced accuracy with zero-shot
prompts: jev-lite is in that range, with the caveats of section 5.

## 4. Long context: 512 vs 8192 tokens

![jev-lite at 512 vs 8192 tokens, per split and source](docs/judge_bench/figures/context_length.png)

[Interactive version](docs/judge_bench/figures/context_length.html)

Stage A was trained at 512 tokens only. Running it at 8192 (ModernBERT's native maximum) without any
long-context training:

- **Gains where the evidence was being cut:** HaluBench CovidQA +0.21 BA (AUROC +0.30), CJB
  faithfulness QA +0.15, refusal (answerable) +0.10, RAGBench expertqa +0.10, completeness (summaries)
  +0.08. On records that fit 512 both runs are identical, as they should be.
- **Overall:** Bool accuracy +3.7 [+3.3, +4.0] (paired, 27,332 records), CJB consistent accuracy +2.0
  [+0.1, +3.9].
- **Losses:** conciseness (QA −0.18, summaries −0.07) and RAGBench cuad (−0.06 BA, 5–24k-token
  contracts, many still cut at 8192). Conciseness at 512 was partly an artefact: truncation hid the
  difference in length, and with both answers whole the model prefers the more complete one.

**For Stage C:** the encoder keeps working far beyond its training length, so long-context training
starts from a usable model, not from scratch. The cost is speed: 9.3 vs 27 rec/s on a T4 over this mix.

## 5. Against published numbers

| Benchmark | jev-lite 8192 | Published | Source and how comparable |
|---|---|---|---|
| CJB, consistent accuracy | 0.332 | o1 0.631, R1 0.619, o3-mini 0.608, GPT-4o 0.512, SFRJudge-8B 0.438, GPT-4o-mini 0.418, Skywork-8B 0.320, Glider-3.8B 0.206 | Paper, Table 5 (criterion-specific prompts). Same records, other prompts. Close. |
| FaithBench, balanced accuracy | 0.579 | GPT-4-Turbo 0.577, GPT-4o 0.563, GPT-4 0.535, GPT-3.5-Turbo 0.449 | Paper, Table 2, zero-shot. Their binarization may treat Questionable differently (we drop it). Approximate. |
| HaluBench, accuracy | 0.735 | Lynx-70B 0.874, GPT-4o 0.865, GPT-4-Turbo 0.850, Claude-3.5-Sonnet 0.837, Lynx-8B 0.829, Claude-3-Haiku 0.690 | Patronus docs (vendor, authors of Lynx); not independently replicated. |
| RAGBench, mean AUROC | 0.629 | fine-tuned DeBERTa-v3-large 0.79, RAGAS 0.60, TruLens 0.59, GPT-3.5 judge 0.56 | Paper, Table 3, labels by GPT-4-turbo; the HF version we used is labelled by GPT-4o. Approximate. |

The published models are from 2024–early 2025; none of these papers reports Claude 4.x, GPT-5 or
Gemini 3. Same-record comparisons with current LLMs are planned (todo Phase 2, LLM baseline).

## 6. What it means

- **For the judge (criterion a):** on faithfulness of prose answers jev-lite is a usable signal
  (HaluBench HaluEval and RAG-prose subsets AUROC 0.69–0.88, CJB faithfulness above 8B judges), and the
  cheapest one here. It is not yet a replacement for a frontier LLM judge: 15–30 points behind on most
  metrics. Whether that gap is worth closing depends on cost and latency, which the LLM baseline will
  put side by side.
- **Phase 4 (synthetic data) should target:**
  - claims with **computed numbers from tables** (FinQA / TAT-QA style), positives included, so that
    "derived from the text" counts as supported;
  - **conciseness and completeness** pairs with answers of different length, so length is not the cue;
  - pairwise judge data in general (only `summarize_from_feedback` today);
  - refusals (CJB unanswerable 0.28; FaithEval-unanswerable 0.49 in Stage A).
- **Phase 7 (calibration):** the threshold drifts per source in both directions (51% vs 86%
  "supported" on RAGBench, 47% vs 33% on FaithBench). A calibrator that moves the bias per domain, or a
  threshold chosen per deployment on a few labelled examples, matters more than a global temperature.
- **Stage C (long context):** start from Stage B as planned; the 8192 run shows the base generalises.
  Evaluate long-context gains on CovidQA-like data and keep conciseness in the test set.

## 7. Caveats

- One seed, one checkpoint (`stage_a-full-s42`); no `stage_a_nosa` run yet.
- **Laya lost 18 CJB records** (9 pairs) to a wrapper bug (confidence 1.0000001 > 1 fails `Prediction`
  validation). The common-record rule drops those pairs for every model; re-running Laya on them after
  the fix will not change any conclusion.
- jev-lite's wording of the questions is ours: the RAG claim template (`The answer to the question
  "…" is: …`) and the CJB criterion prompts were written for this evaluation, never tuned.
- RAGBench labels are GPT-4o's; HaluBench mixes human-labelled sources with perturbation-generated
  negatives (HaluEval, DROP); FaithBench is adversarially selected (detectors disagree), so all
  detectors score low by design.
- Published numbers use other prompts, label versions or binarization (section 5): orientation, not
  paired comparisons.

## 8. Files and reproduction

| What | Where |
|---|---|
| Converters | `src/jevlite/data/public/judge_bench.py` (`DATASETS.md` section 7) |
| Metrics (balanced accuracy, AUROC, consistent accuracy) | `src/jevlite/judge_bench.py` |
| Kaggle notebook (all four runs) | `notebooks/kaggle/30_judge_bench.ipynb` |
| Predictions, logs (git-ignored) | `data/judge_bench/kaggle/{preds,logs}/` |
| Full report, paired tests (git-ignored) | `data/judge_bench/report/`, `data/judge_bench/paired/` |
| Overlap ids | `docs/judge_bench/overlap_ids.txt` |
| Charts | `docs/judge_bench/figures/` |

```bash
# data (eval only; ~70 MB of downloads)
uv run python scripts/convert_public.py --datasets contextual_judge_bench,ragbench,halubench,faithbench --out data/public
# predictions: notebooks/kaggle/30_judge_bench.ipynb on Kaggle 2×T4 (LIMIT = None), Output → data/judge_bench/kaggle
D=(data/public/contextual_judge_bench.jsonl data/public/ragbench.jsonl data/public/halubench.jsonl data/public/faithbench.jsonl)
P=data/judge_bench/kaggle/preds
uv run python scripts/judge_bench_report.py --data $D \
  --pred jevlite_8192=$P/jevlite_8192.jsonl jevlite_512=$P/jevlite_512.jsonl nli=$P/nli.jsonl laya=$P/laya.jsonl \
  --exclude docs/judge_bench/overlap_ids.txt --tokenizer data/runs/stage_a-full-s42/final/encoder \
  --out data/judge_bench/report --bootstrap 1000
uv run --extra reports python scripts/make_judge_bench_figures.py --report data/judge_bench/report/report.json --out docs/judge_bench
# paired accuracy tests on the Bool benchmarks (clean files = data/public minus overlap_ids.txt)
uv run python scripts/compare_paired.py --data data/judge_bench/clean/{ragbench,halubench,faithbench}.jsonl \
  --model jevlite_8192=$P/jevlite_8192.jsonl --ref nli=$P/nli.jsonl --ref laya=$P/laya.jsonl --ref jevlite_512=$P/jevlite_512.jsonl \
  --slices source --bootstrap 1000 --out data/judge_bench/paired
```

## 9. eval v2: the Stage B baseline (2026-10-26)

Stage B trains on the train/validation splits of 7 RAGBench subsets (`DATASETS.md` 8.2–8.4). Their test
records stay in the eval but are no longer out-of-distribution, so from Stage B on the judge benchmarks
are reported in two slices, and the Stage A numbers below (re-scored from the saved predictions, no new
runs) are the baseline every later stage is compared with on the same records.

- **Held-out slice** (comparable across stages): CJB, FaithBench, HaluBench, RAGBench covidqa / cuad /
  emanual / delucionqa / msmarco. 21,054 records.
- **In-distribution slice** (reported separately): RAGBench hotpotqa, finqa, tatqa, hagrid, expertqa,
  techqa, pubmedqa. 7,630 records.
- Both leave out the 49 overlap ids of section 8 and the 2,639 ids whose passages occur in the Stage B
  training files (`stage_b` mix of 2026-10-27, `prefix` mode): 2,615 of them in the in-distribution
  slice (RAGBench train and test share documents), 24 in the held-out one (22 HaluEval, 2 msmarco).
- "Seen source" flags in the held-out slice: HaluBench pubmedqa (PubMedQA is trained via RAGBench);
  CJB's QA-Feedback pairs (ASQA, via Trust-Data).

| Slice | Metric | n | jev-lite 8192 | jev-lite 512 | NLI | Laya |
|---|---|---|---|---|---|---|
| held-out: CJB | consistent accuracy | 1,984 | 0.332 [0.314, 0.354] | 0.312 | 0.090 | 0.024 |
| held-out: FaithBench | balanced accuracy | 723 | 0.579 [0.538, 0.614] | 0.567 | 0.514 | 0.480 |
| held-out: HaluBench | balanced accuracy | 14,873 | 0.732 [0.725, 0.739] | 0.716 | 0.734 | 0.645 |
| held-out: RAGBench, 5 subsets | balanced accuracy | 1,490 | 0.558 [0.515, 0.600] | 0.559 | 0.519 | 0.571 |
| in-distribution: RAGBench, 7 subsets | balanced accuracy | 7,630 | 0.540 [0.523, 0.556] | 0.552 | 0.513 | 0.499 |

- **The held-out RAGBench part is small** (1,490 records, intervals ±0.04, per subset ±0.07–0.13):
  changes of a few points there are noise. The sensitive held-out numbers are CJB and HaluBench.
- Full tables per subset: `data/judge_bench/eval_v2/{heldout,in_distribution}/report.md` (git-ignored).

```bash
uv run python scripts/make_eval_v2.py --data data/public/{contextual_judge_bench,ragbench,halubench,faithbench}.jsonl \
  --exclude docs/judge_bench/overlap_ids.txt --exclude docs/judge_bench/eval_v2/stage_b_overlap_ids.txt \
  --out docs/judge_bench/eval_v2
P=data/judge_bench/kaggle/preds
for s in heldout in_distribution; do
  uv run python scripts/judge_bench_report.py --data data/public/{contextual_judge_bench,ragbench,halubench,faithbench}.jsonl \
    --pred jevlite_8192=$P/jevlite_8192.jsonl jevlite_512=$P/jevlite_512.jsonl nli=$P/nli.jsonl laya=$P/laya.jsonl \
    --exclude docs/judge_bench/eval_v2/${s}_exclude.txt --tokenizer data/runs/stage_a-full-s42/final/encoder \
    --out data/judge_bench/eval_v2/$s --bootstrap 1000
done
```
