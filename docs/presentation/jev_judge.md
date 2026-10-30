---
marp: true
title: JEV as a Judge
paginate: true
---

<!-- Markdown twin of jev_judge.html: same slides, same order, same numbers. Slides are separated by
"---"; with the Marp extension (VS Code) or marp-cli this file renders as a deck. Edit both together. -->

*jev-lite · October 2026*

# JEV as a Judge

Typed decision models as a fast, cheap replacement for LLM-as-a-judge

`1 · What JEV is` · `2 · Where it is useful` · `3 · What we built`

---

# 01 · What is a JEV model?

A model that decides, not one that writes

---

*JEV · TypeSafe AI*

## A "System One" model

| ⚡ No text generation | 🎯 Probabilities | 🔒 Schema-safe |
|---|---|---|
| Returns typed values, not prose. Output goes to software, not to people. | Every answer comes with a probability and a confidence score. | Answers come only from options the caller defines, so it cannot return a value outside the schema. |

*Early access since 15 Sep 2026 · $40M seed · named after Jevons: cheaper intelligence → far wider use*

---

*Interface*

## State + typed questions → answers

| State | + | Questions | → | One pass |
|---|---|---|---|---|
| text · JSON object · list of texts | | `bool`: a statement → P(true) · `choice`: 1 of N options · `score`: ordered levels | | value + distribution + confidence |

---

*Claimed by TypeSafe (self-reported)*

## Fast and cheap by design

| **70–500 ms** | **40–200×** | **40–400×** |
|---|---|---|
| end-to-end latency | faster than frontier LLMs | cheaper than frontier LLMs |

Trained only on **synthetic data** with **RLCD** (Reinforcement Learning for Calibrated Decisions):
probabilities are optimised against outcomes, not human preferences.

*TypeSafe's own benchmarks; they note the results may be biased.*

---

*A black box*

## TypeSafe has published no technical documents

- ? Architecture & model size
- ? Training algorithm (RLCD objective, reward)
- ? Dataset contents & synthetic pipeline
- ? How probability vs confidence is computed
- ? Weights or a paper
- ? Third-party evaluations

Only the API is available. Everything we know comes from the interface and black-box tests.

---

# 02 · Where JEV is useful

Any place where an LLM is used just to pick an answer

---

*Technical use*

## Decision points in software

| 🧭 Routing & intent | 🛡️ Guardrails | 📑 Fact checking |
|---|---|---|
| Which team, model or tool handles this request? | Policy, PII, toxicity and jailbreak checks on every message. | Is this claim supported by the document? |
| **🤖 Agent steps** | **✅ Data validation** | **📊 Scoring & ranking** |
| Tool choice, stop or continue, ask a human. | Check extracted fields and labels against the source. | Quality, relevance and priority on a fixed scale. |

---

*Business value*

## Why it matters

| 💸 Cost per decision | ⏱ Real-time |
|---|---|
| Checks that are too expensive with an LLM become affordable at scale: every message, every document. | Milliseconds instead of seconds, so it fits inside user-facing flows. |
| **🧩 No parsing failures** | **⚖️ Calibrated probabilities** |
| Typed output plugs straight into code: no retries, no broken JSON. | Every verdict comes with how sure it is; the decision threshold is set per criterion. |

*Examples: support ticket triage · contract clause review · content moderation · RAG answer QA*

---

*Our focus*

## JEV as a Judge vs LLM as a Judge

| | LLM as a judge | JEV as a judge |
|---|---|---|
| **Output** | Free text → parse | ✅ Typed answer + probability |
| **Latency** | Seconds | ✅ ~100 ms |
| **Cost** | Tokens per call | ✅ One small model |
| **New criterion** | Edit the prompt | ✅ Add a question |
| **Uncertainty** | Often overconfident | ✅ Calibrated probability |

---

*JEV as a Judge · RAG faithfulness*

## Every claim checked against every chunk

| | chunk 1 | chunk 2 | chunk 3 | chunk 4 | max |
|---|---|---|---|---|---|
| **claim 1** | **0.95** | 0.10 | 0.05 | 0.20 | ✓ |
| **claim 2** | 0.08 | 0.12 | **0.91** | 0.07 | ✓ |
| **claim 3** | 0.10 | 0.15 | 0.09 | 0.11 | ✗ |

- **Per-claim verdict.** Shows which sentence is unsupported and which chunk supports the rest.
- **Calibrated verdict.** Each claim and the whole answer get a probability, not just yes / no.

*Second criterion: does the answer address the question? (Score rubric)*

---

# 03 · What we built

jev-lite: an open JEV-like model, built from scratch

---

*Approach*

## Same interface, our own design

```text
question · [MASK] opt₁ [MASK] opt₂ … | state
                    ↓
        ModernBERT-large · 395M
                    ↓  vector at each [MASK]
        bool     choice     score
```

- **Everything is candidate scoring:** one logit per option, read at its marker.
- **All options in one sequence**, so they see each other.
- **Open:** code and weights under Apache-2.0.

---

*Stage A*

## First model trained on public data

| **25** | **300k** | **9.5 h** | **73 ms** |
|---|---|---|---|
| public English datasets | records per epoch, 2 epochs | Kaggle, 2×T4 (free tier) | median latency, laptop GPU |

| Eval harness | Baselines | Released |
|---|---|---|
| Paired tests, calibration, black-box probes | Zero-shot NLI, Laya, and the JEV API | v0.1.0 on GitHub + Hugging Face |

---

*Results vs open zero-shot baselines*

## Clear lead, smaller on new sources

Accuracy:

| | NLI | Laya | jev-lite |
|---|---|---|---|
| Bool · seen sources | 0.76 | 0.72 | **0.88** |
| Choice · seen sources | 0.60 | 0.60 | **0.80** |
| Score · seen sources | 0.24 | 0.31 | **0.62** |
| Bool · new sources | 0.54 | 0.52 | **0.60** |
| Choice · new sources | 0.56 | 0.59 | **0.64** |

---

*Results vs the JEV API (jev-1.13.0, 16k records)*

## Close on seen data, behind on new data

Accuracy:

| | JEV | jev-lite |
|---|---|---|
| Bool · seen sources | 0.85 | **0.86** |
| Choice · seen sources | **0.86** | 0.81 |
| Score · seen sources | 0.46 | **0.63** |
| Bool · new sources | **0.76** | 0.60 |
| Choice · new sources | **0.88** | 0.63 |

*jev-lite: 73 ms vs 195 ms per JEV API call · JEV wins on knowledge-heavy tasks (MMLU, ARC)*

---

*Findings*

## What the tests revealed

| JEV answers from its own knowledge | JEV is not deterministic |
|---|---|
| With the passage removed it still gets 81% on FEVER (majority 68%). A risk for a faithfulness judge. | The same request twice: probabilities differ by up to 0.11, and 2% of answers flip. |
| **jev-lite: domain shift** | **jev-lite: refusals** |
| Too strict on unseen domains; calibration breaks (ECE 0.03 → 0.30). | Detecting unanswerable questions does not transfer to new data (0.49). |

---

*Next*

## Roadmap

| ✓ Stage A | ✓ Stage B + calibration | ▶ Synthetic data & long context | Judge pipeline | Production test |
|---|---|---|---|---|
| Public data, evaluation, v0.1.0 | Public judge data: refusals, tables, real RAG answers; calibrated probabilities | Teacher LLM, our domain, inputs up to 8k tokens; speed on a server GPU | Claims × chunks, relevance | Our answers vs the production judge and human labels |

> **Target.** ≥ 95% agreement with the production LLM judge, for both faithfulness and relevance;
> ≤ 100 ms per request at 8k tokens.

---

*Thank you*

# Questions?

`github.com/Nefelim584/JEV_Judge`
`huggingface.co/Nefelim584/jev-lite-stage-a`
