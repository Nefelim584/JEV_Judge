# Part 1. JEV as a Judge: what, why, how

> jev-lite overview, October 2026 (revised 2026-10-30 for Stage B) · Part 1 of 5 ·
> [Index](README.md) · [Next: Data and methodology →](02_data_and_methodology.md)

## 1. What JEV is and why it fits the judge role

**JEV** (TypeSafe AI, `jev-1.13.0`) is a *typed decision model*: a "System One" model that decides
and does not write. The caller sends a **state** (a text, a JSON object or a list of texts) and one or
more typed **questions**. The model answers in one pass, with values from the caller's own schema:

| Type | Question | Answer |
|---|---|---|
| `bool` | a statement about the state | P(true) + confidence |
| `choice` | a prompt + 1–32 options | distribution over the options + argmax + confidence |
| `score` | a prompt + 1–32 ordered levels | distribution over the levels + argmax + expected level + confidence |

TypeSafe claims 70–500 ms per call and 40–400× lower cost than frontier LLMs (self-reported). The model
is a black box: no paper, no weights, no description of its architecture or training data.

**Why this is a good replacement for an LLM-as-a-judge.** Most of what our LLM judge does is a
decision with a fixed set of answers: "is this claim supported by the chunk?", "how well does the
answer address the question, on a 1–5 scale?". For that job a decision model has structural
advantages:

| | LLM as a judge | JEV-like model as a judge |
|---|---|---|
| Output | free text, parsed into a verdict | a typed answer + a probability; cannot leave the schema |
| Latency | seconds | ~40–100 ms for typical inputs, up to ~1 s at 8k tokens on a T4 (Part 4, 7.6) |
| Cost | tokens per call | one small model, local or cheap API |
| New criterion | edit the prompt | add a question |
| Uncertainty | often overconfident, hard to threshold | a calibrated probability per answer: how sure each verdict is |

Cheap calls change the shape of the judge. Instead of one LLM call per answer, every claim of the
answer can be checked against every retrieved chunk, which tells which sentence is unsupported.

**Why we build our own (jev-lite).** JEV is API-only and closed. We cannot fine-tune it on our
criteria, run it on our infrastructure, or know what it was trained on, and our tests found that it
partly answers from world knowledge rather than from the given text (Part 3, section 6.5), which is a
risk for a faithfulness judge. jev-lite is an open model with the same interface, built from scratch,
code and weights under Apache-2.0.

## 2. Abstract: what is done, what is planned

**Goal.** Replace the LLM-as-a-judge in our RAG system for its two main criteria, (a) faithfulness of
the answer to the retrieved chunks and (b) how well the answer addresses the question, at a fraction of
the cost and latency of the current production LLM judge.

**Targets** (details and how they are measured: Part 5, section 9.1):

| | Target | Where we are (Stage B, public benchmarks) |
|---|---|---|
| **Quality** | **≥ 95% agreement with the production LLM judge**, per criterion; checked against human labels | not measured yet: needs our own answers (Phase 11) |
| **Latency** | **≤ 100 ms per request at the maximum context (8,192 tokens)** on the serving GPU | 35–112 ms up to 2k tokens, 1.2 s at 6–8k on a T4 |
| **Cost** | ≤ 50% of the LLM-judge-only cost per answer | one 395M model; no per-call fee |

**Done.**

- **Stage A (early October 2026): jev-lite v0.1.0.** ModernBERT-large (395M) with a packed
  typed-question encoding and three heads, trained on 25 English datasets with human labels (600k
  records, 9.5 h on a free Kaggle 2×T4); released on GitHub and on the Hugging Face Hub
  ([`Nefelim584/jev-lite-stage-a`](https://huggingface.co/Nefelim584/jev-lite-stage-a)).
- **Stage B (late October 2026): judge data.** Stage A further trained on public data built for the
  judge task: real LLM answers over several documents (RAGBench), claims across chunks and
  unanswerable questions on on-topic passages (HotpotQA), numbers from tables (TAT-QA, FinQA), refusals
  and answer pairs (Trust-Data, ClapNQ), hallucinated LLM answers (PsiloQA, RAGTruth) and more: 11 new
  sources plus a replay of Stage A, 297k records over 2 epochs at 1024 tokens, 7.5 h on the same 2×T4.
  Synthetic data from a teacher LLM was not ready in time, so public data came first; synthetic data is
  the next data step (Part 5).
- **Calibration** fitted after training on held-out data, no GPU: temperatures for honest
  probabilities, and an optional threshold shift for judging RAG answers.
- **An evaluation harness**: clean test sets (no passage overlap with training), paired significance
  tests, calibration, selective prediction, black-box probes, response time in ms for every run; the
  judge benchmarks frozen as **eval v2** before Stage B, so Stage A → B is compared on the same records.
- **Comparisons**: two open zero-shot baselines (an NLI cross-encoder and Laya, another open JEV-like
  model), the JEV API itself on 16,460 records (Stage A), and four public judge benchmarks with
  published LLM results.

**Main results.**

- **Stage A** on sources and question templates it was trained on beats both open baselines by
  **12–31 accuracy points** and is **on par with the JEV API** on Bool and Choice; it beats JEV on
  Score. On **new sources** the lead over the baselines shrinks to 5–6 points and **JEV is far ahead**
  (+16 Bool, +26 Choice).
- **Stage B is a clear step forward on the judge benchmarks** (held-out part, never trained on):
  pairwise judging (ContextualJudgeBench) **0.33 → 0.45** consistent accuracy, now **between 8B LLM
  judges (0.32–0.44) and GPT-4o (0.51)**; recognising a correct refusal **0.28 → 0.59**; HaluBench
  balanced accuracy 0.73 → 0.80 (most of the gain on a source that Stage B's data is related to);
  better ranking everywhere (AUROC +0.06 on FaithBench and held-out RAGBench).
- **The price is small**: on Stage A's own tests Stage B loses 1.1–1.3 points on Bool and Choice and
  gains 3.4 on Score. The ClapNQ shortcut of Stage A is gone.
- **Stage A was too strict, Stage B is too lenient**: it calls 93% of RAG answers supported. Calibration
  fixes the probabilities (ECE on Choice 0.11 → 0.03) and, for RAG answers, the threshold (balanced
  accuracy **+5 to +11 points** on FaithBench and RAGBench); on short factual QA the same shift hurts,
  so the threshold has to be set per domain.
- **Speed** (Stage B, Kaggle T4, fp16, ms per question; full tables in Part 4, section 7.6):

  | Input tokens | ≤ 512 | 513–1k | 1–2k | 2–4k | 4–6k | 6–8k | Target at 8k |
  |---|---|---|---|---|---|---|---|
  | p50, ms | 35 | 61 | 112 | 334 | 723 | 1,167 | **≤ 100** |
  | p95, ms | 42 | 84 | 166 | 488 | 948 | 1,526 | |
  | share of judge-benchmark inputs | 68% | 14% | 10% | 4% | 3% | 1% | |

  Typical inputs are already fast; at the maximum context we are ~12× off the target. The plan is a
  server GPU and inference optimisation, not a smaller model.

**Planned** (roadmap in Part 5, section 9): ship the calibration with the model; measure and optimise
speed on the target GPU; a synthetic dataset from a teacher LLM (Claude Sonnet 5.5) with our own
domain and long contexts; long-context training (Stage C); the judge pipeline; and finally
the test that decides: jev-lite against the production LLM judge on our own answers.

## 3. Architecture of jev-lite

```text
 [CLS] question · [MASK] option₁ · [MASK] option₂ · … · [MASK] optionₖ [SEP] state [SEP]
                                   │
                     ModernBERT-large encoder (395M)
                                   │
              hidden vector at each [MASK] marker (1024-d)
                                   │
                       shared trunk 1024 → 256
                 ┌─────────────────┼─────────────────┐
             bool head        choice head        score head
          P(true) from      one logit per      one logit per level,
          its marker        option → softmax   ordinal loss
                 └─────────────────┼─────────────────┘
                     post-hoc calibration (no training):
            temperature per primitive · optional judge threshold
```

- **Everything is candidate scoring.** Each candidate (option, level, or "true" for Bool) gets one
  `[MASK]` marker; the head reads one logit at each marker. The question, all candidates and the state
  are packed into **one sequence**, so candidates see each other and the state in one forward pass.
  Cost is one pass per question, whatever the number of options (an NLI model needs one pass per option).
  Several Bool claims about the same state can share one sequence, one marker and one independent
  sigmoid each: this is how one answer's claims are checked against a chunk in a single pass.
- **Confidence** is `1 − H(p) / log K` (normalised entropy of the answer distribution). A learned
  confidence head is planned.
- **Calibration** is applied to the logits after the model (Part 2, section 5.2): a temperature per
  primitive (for Choice it depends on the number of options), which keeps every answer and only makes
  the probabilities honest; and, for faithfulness of RAG answers, an optional shift of the Bool decision
  threshold.
- **Base model**: raw `answerdotai/ModernBERT-large`, full fine-tuning; no NLI pre-tuning. Stage A
  trained at 512 tokens, Stage B at 1024; both run up to ModernBERT's native 8192 tokens without
  retraining (Part 4, section 7.5).
- **Losses**: cross-entropy (Choice), cross-entropy + 0.5·EMD² (Score, so that being one level off
  costs less than being four off), BCE + 0.1·Brier (Bool).
- **English only** in v1.

[Next: Data and methodology →](02_data_and_methodology.md)
