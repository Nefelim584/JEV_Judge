# Part 1. JEV as a Judge: what, why, how

> jev-lite overview, October 2026 · Part 1 of 5 ·
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
| Latency | seconds | ~50–200 ms |
| Cost | tokens per call | one small model, local or cheap API |
| New criterion | edit the prompt | add a question |
| Uncertainty | often overconfident, hard to threshold | a probability per answer: escalate only the unsure cases |

Cheap calls change the shape of the judge. Instead of one LLM call per answer, every claim of the
answer can be checked against every retrieved chunk, which tells which sentence is unsupported. And a
**cascade** becomes possible: the decision model answers the confident cases, the LLM judge only the
rest.

**Why we build our own (jev-lite).** JEV is API-only and closed. We cannot fine-tune it on our
criteria, run it on our infrastructure, or know what it was trained on, and our tests found that it
partly answers from world knowledge rather than from the given text (Part 3, section 6.5), which is a
risk for a faithfulness judge. jev-lite is an open model with the same interface, built from scratch,
code and weights under Apache-2.0.

## 2. Abstract: what is done, what is planned

**Goal.** Replace the LLM-as-a-judge in our RAG system for its two main criteria, (a) faithfulness of
the answer to the retrieved chunks and (b) how well the answer addresses the question, at a fraction of
the cost and latency, with agreement with human labels close to the current LLM judge.

**Done (Stage A, October 2026).**

- **jev-lite v0.1.0**: ModernBERT-large (395M) with a packed typed-question encoding and three heads;
  released on GitHub and on the Hugging Face Hub
  ([`Nefelim584/jev-lite-stage-a`](https://huggingface.co/Nefelim584/jev-lite-stage-a)).
- **Training on public data**: 25 English datasets with human labels, 600k training records over 2
  epochs, 9.5 h on a free Kaggle 2×T4.
- **An evaluation harness**: clean test sets (no passage overlap with training), paired significance
  tests, calibration, selective prediction, black-box probes for shortcuts and order effects.
- **Comparisons**: two open zero-shot baselines (an NLI cross-encoder and Laya, another open JEV-like
  model), the JEV API itself on 16,460 records, and four public judge benchmarks with published LLM
  results.

**Main results.**

- On sources and question templates it was trained on, jev-lite beats both open baselines by
  **12–31 accuracy points** and is **on par with the JEV API** on Bool and Choice; it beats JEV on Score.
- On **new sources** the lead over the baselines shrinks to **5–6 points**, and **JEV is far ahead**
  (+16 Bool, +26 Choice). The model generalises in ranking but its decision threshold drifts on new
  domains.
- On judge benchmarks it is at the level of **8B LLM judges** on pairwise judging, at the level of
  **GPT-4-class detectors** on subtle summary hallucinations (FaithBench), and 15–30 points behind
  frontier judges on most other metrics.
- It runs **73 ms median on a laptop GPU**, 2.7× faster than a JEV API call, with no per-call cost.

**Planned.** Synthetic training data from a teacher LLM to cover new domains, refusals and hard
negatives (Stage B); calibration and a learned confidence signal; long context; the judge pipeline and
cascade; and finally the test that decides: jev-lite against the current LLM judge on our own
human-labelled answers (Part 5).

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
```

- **Everything is candidate scoring.** Each candidate (option, level, or "true" for Bool) gets one
  `[MASK]` marker; the head reads one logit at each marker. The question, all candidates and the state
  are packed into **one sequence**, so candidates see each other and the state in one forward pass.
  Cost is one pass per question, whatever the number of options (an NLI model needs one pass per option).
  Several Bool claims about the same state can share one sequence, one marker and one independent
  sigmoid each: this is how one answer's claims are checked against a chunk in a single pass.
- **Confidence** is `1 − H(p) / log K` (normalised entropy of the answer distribution). A learned
  confidence head is planned.
- **Base model**: raw `answerdotai/ModernBERT-large`, full fine-tuning; no NLI pre-tuning. Trained at
  512 tokens; runs up to ModernBERT's native 8192 tokens without retraining (Part 4, section 7.4).
- **Losses**: cross-entropy (Choice), cross-entropy + 0.5·EMD² (Score, so that being one level off
  costs less than being four off), BCE + 0.1·Brier (Bool).
- **English only** in v1.

[Next: Data and methodology →](02_data_and_methodology.md)
