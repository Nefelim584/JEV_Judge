# Part 5. Known issues, targets and roadmap

> jev-lite overview, October 2026 (revised 2026-10-30 for Stage B) · Part 5 of 5 ·
> [← Previous: Judge benchmarks](04_judge_benchmarks.md) · [Index](README.md)

## 8. Known issues

### 8.1 jev-lite (Stage B, the current model)

| Issue | Evidence | Why it matters for the judge | Status |
|---|---|---|---|
| **The decision threshold depends on the domain** | raw Stage B calls 93% of RAG answers and 81% of FaithBench summaries supported; one calibrated threshold fixes RAG (+5 to +11 points) but hurts short factual QA (DROP −3.5, PubMedQA −6.4) | the threshold will not transfer to a customer domain as is | partly solved: calibration; final threshold on our own labelled answers |
| **Choice probabilities were overconfident** | confidence 0.91 at accuracy 0.80, ECE 0.11 | a judge's probabilities must mean what they say | **solved** by the temperatures: ECE 0.02–0.03 |
| **Long inputs** | at 8192 tokens CJB is worse than at 1024 (0.454 vs 0.479); never trained beyond 1024 tokens | real RAG contexts with many chunks are long | open: long-context training (Stage C) |
| **Response time at 8k tokens** | 1.1 s median at 6–8k tokens on a T4; goal ≤ 100 ms | latency budget of the judge | open: server GPU + inference optimisation, to be measured |
| **Length bias in pairwise judging** | CJB conciseness 0.20–0.29, despite the conciseness pairs in training | prefers the longer answer when comparing two | open |
| **Computed numbers** | HaluBench DROP and FinanceBench near chance; RAGBench FinQA ranking 0.72 (trained) | RAG answers often state derived values | improved in-domain; open on new domains |
| **Some of the gain is on related sources** | HaluBench +6 overall, +2 without HaluEval (built on HotpotQA, trained) | the held-out improvement is real but smaller than the headline | reported both ways |
| **Score on judge-like rubrics** | HelpSteer2 0.64 (Stage A 0.61) | criterion (b) is a rubric of this kind | improving |
| **Confidence is a heuristic** | entropy-based; drops on random text, but little on shuffled words | the judge needs a signal that notices odd inputs | open: learned confidence head |
| **Easy sources in the Stage B mix** | Faithfulness-QA 0.99, PsiloQA 0.96 on their tests | possible surface shortcuts, like ClapNQ in Stage A | to check with the passage-swap probe |
| **Knowledge-heavy multiple choice** | MMLU, ARC far below JEV; Stage B lost 2–5 points on reasoning MCQ | expected from a 395M encoder; not needed for the judge | accepted |

Solved by Stage B: the ClapNQ passage shortcut; refusals on CJB (0.28 → 0.59); Stage A's scepticism on
RAG answers with numbers from tables (in the trained domains).

### 8.2 JEV API (as a reference judge)

- **Uses world knowledge instead of the given text** on Wikipedia-style claims (Part 3, section 6.5).
- **Not deterministic**: 2% of answers flip between identical calls.
- **`jev-latest` is an alias**: results are comparable only with the version pinned.
- **Unknown training data**: we cannot tell which of our test sources are new to it.
- **Compared with Stage A only**; Stage B has not been run against it.

### 8.3 Limits of the evaluation so far

- **Public proxies only.** None of these numbers answers the product question (H0 below): that needs
  our own human-labelled answers.
- **No LLM judge run on the same records yet.** The LLM numbers on the judge benchmarks come from the
  papers (2024–early 2025 models, other prompts).
- **The held-out RAGBench part is small** (1,491 records, ±0.04): CJB, FaithBench and HaluBench carry
  most of the held-out signal.
- **One training run, one seed per stage.** Differences of a few points between future runs will need
  a second seed.
- **No trained-vs-trained control.** The open baselines are zero-shot; what the packed architecture adds
  over fine-tuning a plain classifier on the same data is not measured yet.
- **Speed measured on a T4 only**; the target GPU is not measured yet.
- **The no-share-alike version** (`stage_a_nosa`) is prepared but not trained.

## 9. Targets and roadmap

### 9.1 Targets

| | Target | How it is measured | Status |
|---|---|---|---|
| **Quality** | **≥ 95% agreement with the production LLM judge** (Claude Opus 5.5), for faithfulness and relevance **separately** | on our production answers, against the production judge's verdicts; no human labelling needed, so it can run on thousands of answers | not measured yet (Phase 11) |
| **Quality, guardrails** | agreement on the answers the production judge **rejects** ≥ 90% *(proposed)*; Cohen's κ reported next to the raw agreement | same data | proposed 2026-10-30, to be confirmed |
| **Quality, check against people** | κ with human labels not worse than the production judge's by more than 0.1 (H0, fixed 2026-10-09) | frozen human-labelled test set, 500–800 answers | not started |
| **Latency** | **≤ 100 ms per request at the maximum context, 8,192 tokens**, on the serving GPU; the model is not shrunk | p50 and p95 by input length, one request per call | T4: 35–112 ms up to 2k tokens, **1.2 s at 6–8k** (~12× off) |
| **Cost** | ≤ 50% of the production judge's cost per answer | cost per answer on production traffic | not measured |

**Why the agreement target needs guardrails.** Most production answers are fine, and the production
judge accepts most of them. If it accepts 90%, a model that always says "fine" already agrees 90% of the
time and catches nothing. So 95% raw agreement is only meaningful together with agreement on the
answers the judge rejects, and κ, which corrects for that. The 90% value is a proposal; it is fixed from
the production judge's real rejection rate before the test is run.

**Why the human check stays.** Agreement with the production judge measures whether jev-lite can stand
in for it, including its mistakes. The human-labelled test (H0 below) shows where both are wrong and
keeps a 95% agreement from hiding a shared blind spot. The production judge's model is never used to
make training data, so the comparison stays fair.

**Latency.** Within the target the model must handle the full 8k context. Typical judge inputs (a
question, an answer, a few chunks) are under 2k tokens and already within 100 ms on a T4; the gap is at
long inputs (Part 4, section 7.6). If our real inputs turn out to need more than 8k tokens (step 3),
the target is revisited together with the choice of a longer-context model.

### 9.2 Roadmap

```text
1 Ship calibration ──► 2 Speed on target GPU ────────────────────────────────────────┐
                                                                                     ▼
3 Synthetic data ──► 4 Stage C, long context ──► 5 Offline eval + LLM baseline ──► 6 Judge pipeline ──► 7 Production test (H0–H3)
```

| # | Step | Phase in `todo.md` | Output / done when | Depends on | Status |
|---|---|---|---|---|---|
| 1 | Ship the calibration | 7 | `calibration.json` next to the weights, read by the inference code; calibrated Stage B on the Hub | — | fitted, not packaged |
| 2 | Speed on the target GPU | 9 | p50 / p95 by input length up to 8k on an H100 (or A100); serving setup chosen against ≤ 100 ms | 1 | not started |
| 3 | Synthetic data | 4 | smoke of 200 cases read by hand; wave 1 (~7k cases); wave 2 (~7k) only if wave 1 helps; long states 2–8k; token-length study of our data | plan approval | plan drafted (`dataset_gen.md`), code scaffold |
| 4 | Stage C: long context | 6 | Stage B + synthetic + long states, trained up to 8k tokens; at 8192 tokens at least as good as at 1024 | 3 | compute under discussion |
| 5 | Offline evaluation and LLM baseline | 2, 8 | eval v2 + synthetic tests + probes for every stage; frontier LLM judges on the same records | 4 (partly 1) | harness ready |
| 6 | Judge pipeline | 10 | claims × chunks, relevance rubric, answer-level verdict, learned confidence | 4, 5 | not started |
| 7 | Production test | 11 | targets of 9.1 and H0–H3 on our answers; the replace / do-not-replace decision | 6, labelled test set | not started; labelling is the long pole |

#### Step 1. Ship the calibration

- The two fitted layers (Part 2, section 5.2) go into a `calibration.json` next to `final/`; the
  inference code applies the temperatures by default and the judge threshold when asked.
- The calibration is published together with the model, with its numbers (ECE, the judge-layer table of Part 4).
- **Later in the roadmap**: the judge threshold is refitted on our own answers (step 7), possibly per
  domain, since one threshold does not fit all domains (Part 4, section 7.4).

#### Step 2. Speed on the target GPU

- Measure Stage B on an **H100** (AWS `p5.4xlarge`, the same machine considered for Stage C) or an
  A100: p50 / p95 by input length, 512 to 8k tokens, one request per call.
- Apply, one at a time, so the gain of each is known: bf16; FlashAttention 2 with unpadded sequences
  (not possible on a T4); `torch.compile`; ONNX / TensorRT export (Phase 9); checking several claims of
  one answer in one sequence (the packed encoding already allows it).
- **Done when** the 8k-token p95 is measured on the target GPU. **Decision point**: if ≤ 100 ms is not
  reached on an H100 with all of the above, options are a faster serving stack or more GPUs per
  request; shrinking the model is ruled out.

#### Step 3. Synthetic data

Plan in `dataset_gen.md` (draft); what is generated in `dataset_dicr.md`.

- **Teacher**: Claude **Sonnet 5.5** on Amazon Bedrock (company account), Haiku 5.5 for simple steps;
  legal cleared training on its outputs. **Opus 5.5 is never used**: it is the production judge, and a
  teacher from the same model would make the comparison of step 7 unfair.
- **Budget ≤ $600** including a buffer: smoke 200 cases (~$35), wave 1 **~7k cases** (~$230), wave 2
  **~7k** (~$230) only if wave 1 improves a short training run; wave sizes refitted from the smoke's
  real cost. Cost is logged per call; a wave does not start if it would exceed the cap.
- **Content**: a question, chunks, 3–4 answers of different quality (grounded, partly unsupported,
  refusal, off-topic), labels per sentence × chunk. About **20% in our domain**: FIX and REST API
  specifications, mostly fictional, few real documents; the real ones are split between generation and
  the step 7 test before generation.
- **Long states**: ~5k groups of 2–8k tokens assembled by code from the labelled cases plus unrelated
  distractors, with the evidence at the start, middle or end: the data for Stage C.
- **Quality checks**: a second model verifies the labels, every smoke case is read by hand, shortcut
  probes (passage swap, claim only) on each wave.
- **Token-length study** of our real questions, answers and chunks: does 8k cover our cases (the
  concern noted in 9.1)?

#### Step 4. Stage C: long context

- Start from Stage B (calibrated weights are the same weights); mix: synthetic short cases, long states,
  replay of Stage B's data against forgetting; max length up to 8k tokens.
- Compute, under discussion: AWS `p5.4xlarge` (H100) on demand with stop / start (preferred) or Kaggle;
  Kaggle has a compliance caveat for data from our domain.
- **Done when** at 8192 tokens the model is at least as good as at 1024 on CJB (now 0.454 vs 0.479), and
  accuracy does not depend on where the evidence sits in a long context.
- Calibration (step 1) is refitted for the new weights.

#### Step 5. Offline evaluation and LLM baseline (in parallel with 3–4)

- Every new model: eval v2 (held-out slice), the synthetic test splits including our domain, the Stage A
  tests (forgetting), probes, speed.
- **LLM baseline**: current frontier LLMs as judges on the judge benchmarks and the synthetic test, on
  the same records, with cost and latency; gives an early public-data view of the quality target before
  our own data exists.
- Open checks from Stage B: passage-swap probe on the easy new sources (Faithfulness-QA, PsiloQA);
  a second seed when two runs need to be told apart.

#### Step 6. Judge pipeline

- **Faithfulness**: the answer split into claims, each claim checked against each chunk in packed
  passes, aggregated into an answer-level verdict with the unsupported sentences named.
- **Relevance**: a Score rubric "does the answer address the question", with refusals defined in the
  rubric.
- **Confidence**: a learned confidence head instead of the entropy heuristic; the decision threshold
  per criterion chosen on validation data.
- Output: one call per answer from the RAG system, verdict + probability per criterion.

#### Step 7. Production test (Phase 11; labelling starts early)

- **Data**: answers from our own system with the production judge's verdicts (for the agreement target),
  and a frozen **human-labelled** test set of 500–800 answers labelled without seeing any model's output;
  a separate 1k human-labelled training split for H1. The labelling decision (who labels) is the long
  pole and should start now.
- **Targets of 9.1** on production answers; then the hypotheses, criteria fixed before training:
  - **H0, the product decision**: jev-lite agrees with the production judge on **≥ 95%** of verdicts,
    for faithfulness and relevance separately (9.1); against human labels its κ is not worse than the
    production judge's by more than 0.1, with guardrails on catching bad answers, refusals and cost ≤ 50%.
  - **H1**: synthetic vs human-labelled training data (learning curves; the two synthetic waves are two
    points).
  - **H2**: general data vs our data vs both, and new criteria absent from training (citation
    correctness, completeness).
  - **H3**: one jev-lite vs a separate classifier per criterion, and vs specialised faithfulness
    checkers.
- **Decision**: replace the production judge with jev-lite, or keep it.

#### Backlog

- The no-share-alike version (`stage_a_nosa` and its descendants), if legal needs it.
- A longer-context model, if the token-length study shows 8k is not enough for our cases.
- Multi-language support (v1 is English only).

[← Back to the index](README.md)
