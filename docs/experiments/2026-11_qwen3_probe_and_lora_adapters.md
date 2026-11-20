# Large backbone and domain adapters: two experiments (November 2026)

Two questions, answered cheaply (about $5 of Modal GPU time in total), before spending money on either path:

1. **Is ModernBERT-large the limit?** Would a much larger encoder (Qwen3-Embedding-8B) judge better?
2. **Can a domain adapter replace sequential fine-tuning?** Train Stage C as a LoRA adapter on a frozen
   Stage D model instead of fine-tuning all weights, and see whether it forgets less.

Short answers: (1) scale helps on single-text toxicity, and a frozen 8B with one linear layer already beats
jev-lite there; on tasks that relate two texts the probe says nothing. (2) A merged LoRA adapter is as good as
full fine-tuning on the new domain and forgets just as much; an adapter only avoids forgetting if it is kept
separate and switched on per request.

All numbers: one seed, 95% bootstrap intervals.

## 1. Qwen3-Embedding vs jev-lite on the criteria bench

### Approach

A **linear probe on a frozen embedding model** (`jevlite.embed_probe`, `scripts/modal_embed_probe.py`), a
cheap stand-in for fine-tuning the large backbone:

- Each item becomes Qwen3-Embedding's query format,
  `Instruct: Decide whether the statement is true of the text. Statement: <claim>\nQuery:<text><|endoftext|>`,
  cut to 1,024 tokens (text head + tail). The embedding is the last token's hidden state, L2-normalised.
- A logistic regression is trained on **12,000 Bool records of the Stage D training mix** (confident targets
  only: ≤ 0.2 or ≥ 0.8; 42% positive), the same data jev-lite D learned from. `C` is chosen on a held-out 20%
  (held-out AUROC on the mix: 8B 0.823, 0.6B 0.756).
- The probe then scores the **criteria bench** items (`crit-all`, 8,233 items, as in the v0.3.0 release
  notes) and is reported next to the scorers measured there, with the same AUROC and BA@fit (threshold fitted
  on 30% of each task).
- Cost: the 8B model embeds the 4.4M tokens of the bench and the training records in 2.6 min on one H100
  (plus 2 min to load), ≈ $0.5; the 0.6B model in 40 s.

How to read it: a probe understates what a fine-tune of the same backbone could do. If a frozen model plus one
linear layer matches the fine-tuned jev-lite, the backbone is the limit; if it falls short, the result is
inconclusive.

jev-lite rows are fine-tuned end to end (ModernBERT-large, 395M): D = v0.3.0 (Stages A → B → D), C = D further
trained on the private eval stand (section 2). Granite Guardian 3.3 8B and Qwen3-8B as a judge are run
zero-shot with the same statement per task.

### Results (AUROC)

| Task (n) | Qwen3-Emb-8B probe | Qwen3-Emb-0.6B probe | jev-lite D (v0.3.0) | jev-lite C | Granite Guardian 3.3 8B | Qwen3-8B judge | toxic-bert |
|---|---|---|---|---|---|---|---|
| Toxic toward a group, ToxiGen (940) | **0.879** [0.857, 0.900] | 0.723 | 0.858 [0.835, 0.882] | 0.842 | **0.902** | 0.882 | 0.746 |
| Toxic user prompt, ToxicChat (1,500) | **0.942** [0.921, 0.959] | 0.861 | 0.915 [0.891, 0.937] | 0.897 | **0.975** | 0.953 | 0.788 |
| Unsafe text, OpenAI Moderation (1,680) | 0.915 [0.900, 0.929] | 0.815 | 0.924 [0.911, 0.937] | 0.922 | **0.941** | **0.941** | 0.884 |
| Harmful reply, BeaverTails (1,500) | 0.761 [0.738, 0.784] | 0.621 | **0.860** [0.842, 0.878] | 0.841 | **0.900** | 0.784 | 0.624 |
| Harmful reply, Aegis 2.0 (813) | 0.863 [0.839, 0.886] | 0.797 | **0.919** [0.901, 0.936] | 0.902 | 0.911 | 0.882 | 0.729 |
| Grounding, HaluBench + RAGTruth (1,800) | 0.557 [0.530, 0.583] | 0.497 | 0.724 [0.700, 0.747] | 0.739 | 0.816 | **0.834** | — |

BA@fit of the 8B probe: ToxiGen 0.808 (D 0.774), ToxicChat 0.885 (D 0.862), OpenAI Moderation 0.833 (D 0.853),
BeaverTails 0.688 (D 0.777), Aegis 0.748 (D 0.801), grounding 0.534 (D 0.670).

### Findings

- **On single-text toxicity the frozen 8B already matches or beats jev-lite**: ToxiGen +0.02, ToxicChat +0.03
  AUROC over D (intervals overlap), OpenAI Moderation a tie. One linear layer on 12k records is enough there,
  so the backbone is part of jev-lite's gap to the 8B guards on these tasks.
- **On tasks that relate two texts the probe falls far behind** (harmful *reply* to a prompt −0.06 to −0.10,
  grounding −0.17, near chance). A single pooled embedding cannot compare a claim with its context; this is a
  limit of the probe, not evidence about a fine-tuned 8B.
- **The gain comes from scale, not from Qwen's embedding training**: the 0.6B model of the same family (a
  bit larger than ModernBERT-large) is 0.08–0.16 below the 8B everywhere and below jev-lite on every task.
- **Speed is unchanged as a constraint**: the 8B models take 85–110 ms per item batched on an L40S
  (Granite Guardian), against ~4 ms for jev-lite; an 8B judge does not fit the ≤100 ms at 8k target.

What it means for jev-lite: an 8B encoder is a candidate **teacher** (distil its scores on toxicity criteria
into jev-lite), not a replacement. Whether a fine-tuned 8B would also win on relational tasks is open; the
code for it is in place (`encoding.mode: pair_last`, LoRA on a bf16 base, see "Code" below) but was not run
(est. $15–25 per run).

## 2. Domain adapter vs sequential fine-tuning (Stage C)

### The two paths

jev-lite is trained in stages, each fine-tuning **all weights** of the previous model: A (base primitives)
→ B (judge data) → D (safety; v0.3.0) → C (the private eval stand: company texts scored by the production LLM
judge on 4 Score criteria, plus one Bool criterion). Each stage costs a little of the previous ones
(forgetting).

The alternative tested here: keep D **frozen** and train only a **LoRA adapter** for Stage C (r = 16, α = 32
on the attention and MLP projections `Wqkv`, `Wo`, `Wi`) plus the output heads: 7.5M of 402M parameters
(1.85%); the adapter alone is ~15 MB against ~790 MB for the full model. Everything else is the same as the
full Stage C run: data, 3 epochs, 2,019 steps, effective batch 32, seed 42, one H100.

| | Full fine-tune (`stage_c-full-s42`) | LoRA on frozen D (`stage_c-lora-s42-r2`) |
|---|---|---|
| Trained weights | all 402M | adapter + heads, 7.5M |
| Learning rate | 3e-5 | 2e-4 (adapter and heads) |
| Training time (H100) | 21 min | 36 min |
| Saved model | full weights | adapter merged into the weights (`merge_and_unload`) |

### Results: the new domain (eval stand, 250 test rows × 4 criteria)

Agreement with the production judge over all four Score criteria:

| | exact | within ±1 | QWK | NLL |
|---|---|---|---|---|
| LoRA on frozen D | 65.5% [62.6, 68.5] | 98.1% | 0.58 [0.53, 0.63] | 0.90 |
| Full fine-tune | 66.9% [63.9, 69.9] | 97.5% | 0.59 [0.54, 0.63] | 0.87 |
| Constant (most frequent level) | 54.7% | — | — | — |

- **Equivalent.** Per criterion the two runs differ by at most 3 points of exact agreement, in both
  directions. On the rows the judge scored three times, both agree with a single judge call 67% of the time,
  against 85% for the judge with itself.
- The Bool criterion: accuracy 91.4% vs 92.0%; AUROC 0.78 vs 0.91, but on only 15 negative rows, so mostly
  noise.

### Results: forgetting (Stage D test sets, 80,100 records, paired vs D)

Δ accuracy against jev-lite D on the same records, in points:

| | LoRA on frozen D | Full fine-tune |
|---|---|---|
| Bool | −0.3 [−0.4, −0.1] | −0.6 |
| Choice | −0.3 [−0.5, −0.1] | +0.1 (tie) |
| Score | **−3.0** [−3.9, −2.1] | **−3.0** |
| Score, out of distribution | −3.4 | −4.7 |
| Δ NLL Bool / Choice / Score | +0.18 / +0.47 / +0.41 | +0.17 / +0.44 / +0.40 |
| Significant losses by source | HelpSteer2 Score −5.8, FinQA judge Bool −3.8, ContractNLI Choice −3.2 and Bool −2.3 | HelpSteer2 Score −4.9, Aegis 2.0 Bool −3.7 |

Directly LoRA vs full (81,287 records): Bool +0.3 (win), Choice −0.4 (loss), Score −0.2 (tie). The losses
moved between sources (Aegis 2.0 +4.4 for LoRA, ContractNLI −4.5 to −5.9) but did not shrink.

### Findings

- **A merged LoRA adapter forgets as much as full fine-tuning.** Once merged, the adapter changes the encoder
  for every task, not only for the stand; and the shared heads were trained at the adapter's learning rate
  (2e-4, ~7× the full run's 3e-5) and drifted ~8× further (relative drift 0.20 vs 0.026).
- **It is not cheaper to train** at this size: 36 vs 21 min on one H100. LoRA pays off for multi-billion
  backbones, not for a 400M encoder.
- **What the adapter does buy is isolation, if it is not merged.** Serving D as is for public tasks and
  D + the stand adapter (with its own copy of the heads, 263k parameters) for stand requests gives D's
  numbers on public tasks by construction and full-C quality on the stand. The cost is routing in the
  service and ~2% more compute on adapter requests. This setup is not built or measured yet.

### Recommendation

For one model that keeps everything: **replay** (mix records of the earlier stages into Stage C, as Stage D
already does with 41% Stage B data) is the next experiment, at the price of one Stage C run; joint training
from raw ModernBERT on all stages is the thorough option (est. $15–20). Per-domain adapters are worth building
only if the service is going to route requests by domain anyway.

## Code added on this branch

- `encoding.mode: pair_last` (`LastTokenPairEncoder`): pair encoding for causal backbones such as
  Qwen3-Embedding; the text first, the question last, read out at a final `<|endoftext|>` token.
- `model.base_dtype: bf16` for a frozen base under LoRA (fp32 stays the default; a bf16 base without LoRA is
  refused); KV cache off for decoder backbones (`disable_kv_cache`).
- `jevlite.embed_probe` and `scripts/modal_embed_probe.py`: the frozen-embedding probe of section 1,
  reported in the criteria bench format.
- Tests: `tests/test_pair_last.py`, `tests/test_embed_probe.py`.

Reproduce:

```bash
# section 1 (≈ $0.5 on H100): embeddings on Modal, probe and report locally
uv run --extra bench modal run scripts/modal_embed_probe.py --model Qwen/Qwen3-Embedding-8B
# section 2: Stage C as LoRA on the frozen D model (private data; needs the jevlite-stage-c volume)
uv run --extra bench modal run --detach scripts/modal_train.py --background --set=tuning.mode=lora \
    --run stage_c-lora-s42-r2 --init /vol/runs/stage_d-full-s42/final
```
