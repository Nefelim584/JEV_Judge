# Part 5. Known issues and next steps

> jev-lite overview, October 2026 · Part 5 of 5 ·
> [← Previous: Judge benchmarks](04_judge_benchmarks.md) · [Index](README.md)

## 8. Known issues

### 8.1 jev-lite

| Issue | Evidence | Why it matters for the judge |
|---|---|---|
| **Threshold drift on new domains** | ContractNLI: ranks well (AUROC 0.93) but says "supported" too rarely; RAGBench too sceptical, FaithBench slightly too lenient | the decision threshold will not transfer to a new customer domain as is |
| **Calibration breaks on new sources** | Bool ECE 0.03 on trained sources, 0.30 on new ones | probabilities cannot be used for escalation on a new domain without recalibration |
| **Refusals do not transfer** | FaithEval-unanswerable 0.49 (the answer removed from an on-topic passage); CJB refusal on unanswerable 0.28 | a judge must recognise a correct "the documents do not say" |
| **Tables and computed numbers** | TabFact passage-swap shortcut (54% "supported" on an unrelated table); RAGBench FinQA 4% "supported" vs 91% | RAG answers often state values derived from tables |
| **Length bias in pairwise judging** | CJB conciseness 0.10 at 8192 tokens | prefers the longer answer when comparing two |
| **Score on the judge-like rubrics is the weakest** | HelpSteer2 correctness 0.52, helpfulness 0.48 | criterion (b), "does the answer address the question", is a rubric of this kind |
| **Confidence is a heuristic** | entropy-based; drops on random text, but little on shuffled words | the cascade needs a confidence signal that notices odd inputs |
| **Knowledge-heavy multiple choice** | MMLU 0.38, ARC 0.60 vs JEV 0.89 / 0.98 | expected from a 395M encoder; not needed for the judge |

### 8.2 JEV API (as a reference judge)

- **Uses world knowledge instead of the given text** on Wikipedia-style claims (Part 3, section 6.5).
- **Not deterministic**: 2% of answers flip between identical calls.
- **`jev-latest` is an alias**: results are comparable only with the version pinned.
- **Unknown training data**: we cannot tell which of our test sources are new to it.

### 8.3 Limits of the evaluation so far

- **Public proxies only.** None of these numbers answers the product question (H0 below): that needs
  our own human-labelled answers.
- **No LLM judge run on the same records yet.** The LLM numbers on the judge benchmarks come from the
  papers (2024–early 2025 models, other prompts).
- **One training run, one seed.** Differences of a few points between future runs will need a second
  seed.
- **No trained-vs-trained control.** The open baselines are zero-shot; what the packed architecture adds
  over fine-tuning a plain classifier on the same data is not measured yet.
- **The no-share-alike version** (`stage_a_nosa`) is prepared but not trained.

## 9. What we do next

1. **Stage B: synthetic training data.** A teacher LLM generates judge-like data over many domains,
   with some domains held out for testing: unanswerable cases built by removing the answer from an
   on-topic passage, claims with numbers computed from tables, hard negatives on the same passage or
   table, multi-chunk claims, and pairwise answers of different length.
2. **Calibration and confidence.** A calibrator that can move the decision threshold per domain, not
   only sharpen or soften probabilities, and a learned confidence head for escalation.
3. **Long context (Stage C).** Train on inputs up to 8k tokens, starting from Stage B; the 8192-token
   run shows the base already generalises.
4. **The judge pipeline.** Faithfulness as claims × chunks, the relevance rubric, and the cascade to
   the LLM judge.
5. **LLM baseline on the same records.** Current frontier LLMs as judges on the judge benchmarks, for a
   like-for-like comparison of quality, cost and latency.
6. **The production test.** On human-labelled answers from our own system:
   - **H0, the product decision:** jev-lite with a cascade that sends **≤ 40%** of answers to the LLM
     judge agrees with human labels within **0.1 κ** of the LLM judge, for faithfulness and for
     relevance separately, at **≤ 50% of the cost**. These criteria were fixed before any training.
   - **H1:** synthetic vs human-labelled training data.
   - **H2:** general data vs our data vs both.
   - **H3:** one jev-lite vs a separate classifier per criterion.

[← Back to the index](README.md)
