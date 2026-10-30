# Datasets and licenses

Every public dataset jev-lite is trained or evaluated on, with its license and the known problems of the
data. The code and the model weights are Apache-2.0; the datasets keep their own terms, listed here.
How the data is used in evaluation: [`EVALUATION.md`](EVALUATION.md).

Licenses were checked against the HF Hub API (`cardData.license`), the dataset cards, the original
repositories and project pages (Stage A sources on 2026-10-02, Stage B sources on 2026-10-26). This page
is our reading of those terms, not legal advice; check the original terms before reusing the data.

## 1. Rules we follow

- **No non-commercial or research-only data in training.** NC sources (HaluBench, FaithBench, RAGBench
  msmarco, FaithEval, LLM-AggreFact's subsets) are used for evaluation only.
- **Share-alike (CC BY-SA) data is trained on.** Most Wikipedia-based sets are CC BY-SA; we use them for
  training (legal sign-off, 2026-10-07). Whether share-alike terms carry over to trained weights is
  legally unsettled. No SA-free variant of the model has been trained yet.
- **LLM-made data is marked.** Records carry `label_source` (`human` / `llm` / `program`). LLM-written
  texts or labels (WANLI, RAGBench, MiniCheck, PsiloQA, Trust-Data, HaluBench) may fall under the terms of
  the LLM provider that produced them.
- **Source text keeps its own copyright** (Wikipedia, news articles, Reddit posts), separate from the
  label license.
- **Never train on an evaluation set**, and drop training records whose passage occurs in an eval set
  (`jevlite/data/overlap.py`).

## 2. Training data

### 2.1 Stage A: primitives (v0.1.0 on; replayed in Stage B)

| Primitive | Dataset | License | Labels |
|---|---|---|---|
| Bool / Choice-3 | FEVER | CC BY-SA 3.0 | human |
| Bool / Choice-3 | VitaminC | CC BY-SA 3.0 | human |
| Bool / Choice-3 | HoVer | CC BY-SA 4.0 | human |
| Bool / Choice-3 | MultiNLI | OANC + CC BY-SA 3.0 / CC BY 3.0 (fiction) | human |
| Bool / Choice-3 | SNLI | CC BY-SA 4.0 | human |
| Bool / Choice-3 | WANLI | CC BY 4.0 | pairs written by GPT-3, labelled by humans |
| Bool | BoolQ | CC BY-SA 3.0 | human |
| Bool | TabFact | CC BY 4.0 | human |
| Bool | QNLI (from SQuAD 1.1) | CC BY-SA 4.0 | human |
| Bool | PAWS (Wiki) | free use (repo LICENSE) | human |
| Bool, answerability | SQuAD 2.0 | CC BY-SA 4.0 | human |
| Bool, answerability | Natural Questions | CC BY-SA 3.0 | human |
| Bool, answerability | ClapNQ | Apache-2.0 (NQ text CC BY-SA 3.0) | human; see 3.1 |
| Choice | ARC | CC BY-SA 4.0 | human |
| Choice | OpenBookQA | Apache-2.0 | human |
| Choice | CommonsenseQA | MIT (HF card only) | human |
| Choice | Cosmos QA | CC BY 4.0 (HF card only) | human |
| Choice | Social IQa | CC BY 4.0 (card text) | human |
| Choice | CLINC150 (K = 151) | CC BY 3.0 | human |
| Choice | MASSIVE (en-US) | CC BY 4.0 | human |
| Choice | DBpedia-14 | CC BY-SA 3.0 | ontology |
| Choice | GoEmotions (single-label) | Apache-2.0 | human |
| Choice K = 2 | summarize_from_feedback (comparisons) | modified MIT; Reddit posts CC BY 4.0 | human |
| Score | HelpSteer2 (5 attributes) | CC BY 4.0 | human |
| Score | HelpSteer3 | CC BY 4.0 | human; responses from permissively licensed LLMs |

3-way NLI and fact-checking sets are used either as Bool (entailed vs not) or as Choice
(supported / contradicted / not mentioned), never both for the same example.

### 2.2 Stage B: judge data (v0.2.0 on)

Built into judge records by code (`src/jevlite/data/public/judge_train.py`, `judge_public.py`).

| Dataset | What we build | License | Labels |
|---|---|---|---|
| **RAGBench** train, 7 subsets (hotpotqa, finqa, tatqa, hagrid, expertqa, techqa, pubmedqa) | real LLM answers vs several documents: the whole answer, one sentence vs all documents, one sentence vs one document | CC BY 4.0; sources: HotpotQA CC BY-SA 4.0, FinQA MIT / CC BY 4.0, TAT-QA CC BY 4.0, HAGRID Apache-2.0, ExpertQA MIT, TechQA Apache-2.0, PubMedQA MIT | GPT-4 |
| HotpotQA (distractor) | claims supported only by two paragraphs together; negatives with a distractor's entity; answerability by dropping a gold paragraph | CC BY-SA 4.0 | human + program |
| TAT-QA, FinQA | claims with computed numbers; negatives with another number of the same table, a scale error or a small shift | CC BY 4.0; MIT / CC BY 4.0 | human + program |
| ClapNQ | answerability rebuilt (3.1), relevance (Score), conciseness and completeness pairs | Apache-2.0 (NQ text CC BY-SA 3.0) | human + program |
| Faithfulness-QA | answers against passages with a substituted entity (counterfactual contexts) | MIT (HF card; no LICENSE file in the repo); SQuAD CC BY-SA 4.0, TriviaQA | rule-based |
| MiniCheck D2C | claim vs news document | MIT; news articles are third-party text | GPT-4 |
| Trust-Data (QAMPARI and ASQA rows only, ELI5 removed) | preferred vs dispreferred answer (Choice K = 2), answerability | Apache-2.0; ASQA Apache-2.0, QAMPARI CC0-1.0 | GPT-made pairs |
| PsiloQA (English) | LLM answer vs Wikipedia passage, sentence-level | CC BY 4.0; Wikipedia CC BY-SA | GPT-4o spans |
| Qasper | answerability and answers on chunks of research papers | CC BY 4.0 | human |
| RAGTruth (summaries of CNN/DM articles only) | summary and sentence vs article | MIT; CNN/DM articles are third-party text | human spans |

Plus a replay of the Stage A training data (26% of the records) against forgetting.

- **MiniCheck C2D is not in v0.2.0**: its claims overlap WiCE inside LLM-AggreFact, and the overlap check
  needs gated access we did not have in the build.
- **Left out:** ConditionalQA (no license, research only); Trust-Data's ELI5 part (no license); RAGTruth's
  "Recent News" articles (unclear terms).

## 3. Known problems in the data

### 3.1 ClapNQ: the passage gives the answerability label away

In the original ClapNQ files, answerable and unanswerable questions come with passages from different
pools, so "which file is this passage from" predicts the label. Stage A learned it: 1.000 on the ClapNQ
test, and with a swapped passage the answer follows the passage, not the question. Natural Questions and
SQuAD 2.0 pass the same swap test.

Stage B rebuilds ClapNQ from the answerable file only (same passage, its own question → 1, a question
from another page → 0). After the fix the old ClapNQ test drops to 0.830, as expected. **Stage A's ClapNQ
numbers are not evidence of answerability skill.** Every answerability source now has to pass the
2 × 2 passage-swap test before it enters a mix.

### 3.2 TabFact: partly a table-type shortcut

With the table of another record, Stage A still says "supported" to 54% of TabFact claims (64% with the
right table). Part of its TabFact accuracy is claim plausibility, not table reading.

### 3.3 Overlap between training and evaluation sources

- **HaluBench HaluEval** (two thirds of HaluBench) is built on HotpotQA, which Stage B trains on. HaluBench
  is reported with and without it.
- **HaluBench PubMedQA** and RAGBench pubmedqa share a source; **CJB's QA-Feedback pairs** are ASQA-based,
  as is part of Trust-Data. Flagged "seen source".
- The RAGBench test splits of the 7 trained subsets are reported as an in-distribution slice, not as a
  held-out result.

### 3.4 Label quality

- **RAGBench labels are GPT-4o's** (GPT-4 in the paper), with its notion of "supported" (a correct
  computation counts). Scores on RAGBench measure agreement with an LLM judge, not ground truth.
- **PsiloQA spans** mark deviation from a GPT-4o golden answer, not from the passage; they do not separate
  "contradicts" from "not mentioned". Used as Bool only.
- **HaluBench** mixes human-labelled sources with perturbation-generated negatives; **FaithBench** is
  selected for detector disagreement, so every detector scores low on it by design.
- **License metadata disagrees** for some sources (ContractNLI: CC BY 4.0 on the official site, CC BY-NC-SA
  on HF; FEVER: an extra gpl-3.0 tag on HF). We follow the original source and keep a copy of its license.

## 4. Evaluation only

Never trained on.

| Dataset | License | Use |
|---|---|---|
| ContextualJudgeBench | CC BY 4.0 | pairwise judging |
| RAGBench test (all 12 subsets) and msmarco train | CC BY 4.0; msmarco NC (MS MARCO terms) | faithfulness of RAG answers |
| RAGBench covidqa / cuad / delucionqa / emanual train + validation | CC BY 4.0 | judge calibration layer only (`ragbench_calib`) |
| HaluBench | CC BY-NC 2.0 | faithfulness |
| FaithBench | CC BY-NC-SA 4.0 | subtle summary hallucinations |
| FaithEval (unanswerable, counterfactual) | research only | refusals, conflicting context |
| ContractNLI | CC BY 4.0 (official site) | held-out source |
| BANKING77 | CC BY 4.0 | held-out source |
| MMLU (test) | MIT | held-out source |

## 5. Considered and rejected

Non-commercial or unclear terms: ANLI, RACE (also inside MMLU `aux_train`), DREAM, QuAIL, SciQ, SICK,
MultiRC, ConTRoL, Yelp and Amazon reviews, AG News, MS MARCO, WikiQA, ASNQ, DocNLI, HellaSwag (wikiHow DMCA),
ConditionalQA; Yahoo Answers, TREC, SST-5, STS-B and Climate-FEVER (license unverified). LLM-AggreFact and
TofuEval ask not to be used for training.
