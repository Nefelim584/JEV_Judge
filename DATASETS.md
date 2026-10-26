# Public datasets: candidates and licenses

> **Status:** the Stage A list is approved (section 0, 2026-10-02). Sections 1–6 hold all the candidates and the license checks behind it. Section 7: judge benchmarks (eval). **Section 8: Stage B public judge data, option B (decided 2026-10-26; new sources pending license checks).**
>
> Licenses were checked on 2026-10-02 against the HF Hub API (`cardData.license`), HF cards, and the original repos and project pages.
> - "Commercial" assumes the model may ship inside a commercial product.
> - **SA** = share-alike (CC BY-SA). Whether SA terms carry over to trained weights is legally unsettled, so confirm with legal.
> - An HF tag of "unknown" does **not** mean permission.
> - **E / C / NM** = entailed / contradicted / not mentioned.
>
> The RAG / hallucination / relevance group is in section 6.

## 0. Stage A: approved list (2026-10-02)

Decisions:
- **Share-alike (CC BY-SA) datasets are in.** Legal confirmed on 2026-10-07 that SA data may be used. As a precaution there are two Stage A versions: with SA (`stage_a`) and without (`stage_a_nosa`, where Bool leans on MultiNLI, PAWS, TabFact and WANLI plus synthetic data); todo Phase 3.
- **LLM-generated data: WANLI only** (GPT-3 wrote the pairs; humans labelled and revised them). RAGBench, MiniCheck C2D/D2C, UltraFeedback and Prometheus stay out of Stage A (ablations, section 2). *Stage B (2026-10-26) admits LLM-labelled judge data: section 8.*
- **3-way NLI and fact-checking sets go in both as Bool and as Choice.** FEVER, VitaminC, HoVer, FEVEROUS, MultiNLI, SNLI and WANLI: each example is used either as Bool (entailed vs not) or as Choice `supported / contradicted / not mentioned`, about 50/50 per dataset, never both. This gives data for the open decision on the faithfulness label (todo section 7).
- **Mix by primitive: Bool 50%, Choice 20%, Score 30%.** Score has little variety (HelpSteer2/3), so vary the rubric wording and the number of levels in the templates, and cap upsampling (per-dataset caps are set in the mix config, Phase 3).
- **`test_ood` (public data): ContractNLI and BANKING77 are held out entirely.** No training on them.

| Primitive | Dataset | Cap (questions) | Note |
|---|---|---|---|
| Bool / Choice-3 | FEVER | 40k | evidence rebuilt as chunks |
| Bool / Choice-3 | VitaminC | 60k | near-miss evidence |
| Bool / Choice-3 | HoVer | 18k | multi-hop |
| Bool / Choice-3 | FEVEROUS (text-only evidence) | 30k | |
| Bool | BoolQ | 9.4k | |
| Bool | TabFact | 20k | tables |
| Bool `answerable` | SQuAD 2.0 | 30k | refusals |
| Bool `answerable` | Natural Questions | 30k | |
| Bool `answerable` / Score | ClapNQ | 3.7k | criterion B and refusals |
| Bool / Choice-3, low weight | MultiNLI | 40k | |
| Bool / Choice-3, low weight | WANLI | 20k | |
| Bool / Choice-3, low weight | SNLI | 20k | |
| Bool, low weight | QNLI | 15k | |
| Bool, low weight | PAWS (Wiki) | 10k | |
| Choice | ARC | all (7.8k) | |
| Choice | OpenBookQA | all (6k) | |
| Choice | CommonsenseQA | all (12k) | |
| Choice | Cosmos QA | all (35k) | |
| Choice | Social IQa | 20k | |
| Choice | CLINC150 | 15k | K = 151 |
| Choice | MASSIVE (en-US) | all (16.5k) | |
| Choice | DBpedia-14 | 10k | |
| Choice | GoEmotions (single-label) | 15k | |
| Score | HelpSteer2 | all (~105k = 21k × 5 attributes) | core |
| Score | HelpSteer3 (feedback) | all (~41k) | |
| Choice K = 2 | summarize_from_feedback (comparisons) | 30k | the axis split is eval-only |
| `test_ood` | ContractNLI, BANKING77 | — | held out |
| `test_ood_source` (2026-10-10, Stage A review Q3) | MMLU (test, 3k of 14k), FaithEval (unanswerable + counterfactual, all 3.5k), LLM-AggreFact (test, gated: after HF access) | — | eval only, converters in `data/public/eval_only.py`; FaithEval-inconsistent left out (no label in our schema) |
| eval only | MMLU (test/dev), LLM-AggreFact, RAGTruth, WiCE, SciFact, SummEval, FaithEval | — | section 3 and 6 |

Not in Stage A, but kept as **seeds for Phase 4 perturbations**: HotpotQA (two-chunk claims), TAT-QA and FinQA (tables, numbers). They have no negatives of their own.

## 1. Proposed for training (Stage A): commercial use OK, human labels

### Bool: claim follows from the state, answerability

| Dataset | HF id | Size (train / test) | Mapping | License (source) | Note |
|---|---|---|---|---|---|
| BoolQ | `google/boolq` | 9.4k / 3.3k dev | yes/no over a passage → Bool | CC BY-SA 3.0 (google-research-datasets/boolean-questions) | Paragraph-length premise. |
| MultiNLI | `nyu-mll/multi_nli` | 393k / 19.6k dev | 3-way → Bool E vs not, or Choice E/C/NM | OANC (free use) and CC BY-SA 3.0 / CC BY 3.0 / PD for fiction (HF card, paper) | Short premises; low weight. |
| SNLI | `stanfordnlp/snli` | 550k / 10k | 3-way | CC BY-SA 4.0 (nlp.stanford.edu/projects/snli) | One-sentence captions; low weight. |
| QNLI | `nyu-mll/glue` (qnli) | 105k / 5.5k dev | sentence answers question → Bool | from SQuAD 1.1, CC BY-SA 4.0 | Answer-addresses-question signal; low weight. |
| PAWS (Wiki) | `google-research-datasets/paws` | 49k / 8k | paraphrase vs not → Bool | "may be freely used for any purpose" (repo LICENSE) | Paraphrase, not entailment; low weight. |
| FEVER | `fever/fever` | 145k / 20k dev | S/R/NEI → Choice E/C/NM | CC BY-SA 3.0 (fever.ai license page) | Rebuild the evidence as chunks. HF card also tags gpl-3.0 (probably the old loading script; unconfirmed). |
| VitaminC | `tals/vitaminc` | ~370k / ~55k | S/R/NEI, contrastive | CC BY-SA 3.0 (HF); code MIT | Near-miss evidence; strong for perturbations. |
| HoVer | `hover-nlp/hover` | 18k / 4k dev | supported / not, multi-hop | CC BY-SA 4.0 (hover-nlp.github.io) | Multi-document evidence: claims that need two chunks. |
| FEVEROUS | `fever/feverous` | 71k / 7.9k dev | S/R/NEI, text + tables | CC BY-SA 3.0 (as FEVER); code Apache-2.0 | Start with the text-only evidence subset. |
| ContractNLI | `kiddothe2b/contract-nli` | 607 NDAs × 17 hyp. (~10k pairs) | E/C/NM + evidence spans | **CC BY 4.0 on the official site and repo; the HF card says CC BY-NC-SA 4.0** | Long documents, real NM class. Download from stanfordnlp.github.io/contract-nli. |
| Natural Questions | `google-research-datasets/natural_questions` | 307k / 7.8k | has an answer → Bool `answerable` | CC BY-SA 3.0 (HF); repo Apache-2.0 | Answerability over a real page. |
| SQuAD 2.0 | `rajpurkar/squad_v2` | 130k / 11.9k | answerable → Bool | CC BY-SA 4.0 (SQuAD explorer) | Adversarial unanswerable questions; refusal training. |
| WANLI | `alisawuffles/WANLI` | 103k / 5k | 3-way | CC BY 4.0 (HF card; repo has no license file) | Pairs written by GPT-3, labelled and revised by humans. Check the OpenAI-terms caveat (section 5). |

### Choice: multiple-choice QA and classification with labels as options

| Dataset | HF id | Size | Mapping | License (source) | Note |
|---|---|---|---|---|---|
| ARC | `allenai/ai2_arc` | 7.8k | K = 3–5 | CC BY-SA 4.0 (HF) | SA. |
| OpenBookQA | `allenai/openbookqa` | 6k | K = 4 | Apache-2.0 (GitHub allenai/OpenBookQA); HF: unknown | |
| CommonsenseQA | `tau/commonsense_qa` | 12k | K = 5 | MIT (HF only) | Medium confidence: the original page states no license. |
| Cosmos QA | `allenai/cosmos_qa` | 35k | K = 4 | CC BY 4.0 (HF only) | Medium confidence. |
| Social IQa | `allenai/social_i_qa` | 38k | K = 3 | CC BY 4.0 (card text only, no YAML tag) | Medium confidence. |
| CLINC150 | `clinc/clinc_oos` | 23.7k | K = 151 incl. out-of-scope | CC BY 3.0 (HF, GitHub) | Large K; out-of-scope option. |
| BANKING77 | `PolyAI/banking77` | 13k | K = 77 | CC BY 4.0 (HF, GitHub) | |
| MASSIVE (en-US) | `AmazonScience/massive` | ~16.5k | K = 60 intents | CC BY 4.0 (HF, NOTICE.md) | |
| DBpedia-14 | `fancyzhx/dbpedia_14` | 630k | K = 14 | CC BY-SA 3.0 (HF) | Labels from the ontology; take a small sample. |
| GoEmotions | `google-research-datasets/go_emotions` | 58k | K = 28 (multi-label) | Apache-2.0 (HF and repo LICENSE; the data folder states none) | Single-label subset only. |
| MMLU | `cais/mmlu` | 14k test | K = 4 | MIT (HF, GitHub) | Eval or train, but **drop `aux_train`**: it bundles RACE (NC). |

### Score: ordinal labels, judge-like rubrics

| Dataset | HF id | Size | Mapping | License (source) | Note |
|---|---|---|---|---|---|
| **HelpSteer2** | `nvidia/HelpSteer2` | 21k | N = 5 (0–4) × 5 attributes (helpfulness, correctness, coherence, …) | CC BY 4.0 (HF) | **Core.** Human rubric scores; the closest to our relevance criterion. |
| **HelpSteer3** | `nvidia/HelpSteer3` | 40k pref + 41k feedback | Choice K = 2 / Score N = 7 (−3..+3) | CC BY 4.0 (HF) | Responses from permissively licensed LLMs, human labels. |
| summarize_from_feedback | `openai/summarize_from_feedback` | ~15k axis + 93k comparisons | Score N = 7 (accuracy, coverage, coherence, overall) / Choice K = 2 | "Modified MIT" (GitHub) for the labels; TL;DR posts CC BY 4.0 | The axis split exists only as val/test. Reddit source text. |

## 2. LLM-labelled: ablations and the synthetic side of H1 only

Keep these out of the main human-label Stage A. They blur H1 (synthetic vs human), and OpenAI's terms may restrict using GPT outputs to train a model.

| Dataset | HF id | Size | Mapping | License | Labels by |
|---|---|---|---|---|---|
| MiniCheck C2D / D2C | `lytang/C2D-and-D2C-MiniCheck` | 14k | supported / not, multi-sentence document → Bool | MIT (HF); repo Apache-2.0 | GPT-4 (synthetic), built for exactly this task |
| UltraFeedback | `openbmb/UltraFeedback` | 64k prompts × 4 responses | Score N = 5 × 4 aspects | MIT (HF) | GPT-4 |
| Prometheus Feedback Collection | `prometheus-eval/Feedback-Collection` | 100k | Score N = 5 with a rubric | CC BY 4.0 (HF) | GPT-4 |
| Prometheus Preference Collection | `prometheus-eval/Preference-Collection` | 200k | Choice K = 2 | CC BY 4.0 (HF) | GPT-4 |

## 3. Eval only: offline judge benchmark (Phases 6–10)

| Dataset | HF id | Size | What it measures | License | Why eval-only |
|---|---|---|---|---|---|
| **LLM-AggreFact** | `lytang/LLM-AggreFact` (gated) | 30k dev / 29k test | claim supported by document → Bool, 11 subsets | CC BY-ND 4.0; the card says it "should not be used in pretraining or fine-tuning" | Main offline faithfulness benchmark; the standard one for MiniCheck-style checkers. |
| RAGTruth | no official HF id (`wandb/RAGTruth-processed` has no license) | ~15k / 2.7k responses | human span labels of hallucinations in RAG answers | MIT (GitHub ParticleMedia/RAGTruth); source passages from MS MARCO, CNN/DM and Yelp carry their own terms | Inside LLM-AggreFact; source-text terms unclear. |
| WiCE | `tasksource/wice` | ~1.3k claims | S / partial / NS | annotations ODC-BY, Wikipedia CC BY-SA | Inside LLM-AggreFact. |
| SciFact | `allenai/scifact` | 809 / 300 claims | S/R/NEI + rationales | **repo: CC BY 4.0 claims, ODC-By abstracts; HF: CC BY-NC 2.0** | Small; in BEIR. |
| SummEval | `mteb/summeval` | 1.6k summaries | human 1–5 on consistency, relevance, … | MIT (HF, GitHub) | Small, standard benchmark. CNN/DM source text. |

**Contamination rule:** do not train on WiCE, RAGTruth or the AggreFact subsets if LLM-AggreFact is the benchmark.

## 4. Not usable (non-commercial or unverified)

| Dataset | Reason |
|---|---|
| **ANLI** | CC BY-NC 4.0 (facebookresearch/anli LICENSE). |
| **RACE** | "non-commercial research purpose only" (also inside MMLU `aux_train`). |
| DREAM, QuAIL, SciQ, SICK | NC / research-only. |
| MultiRC | CogComp research and academic use license; no commercial use. |
| ConTRoL | CC BY-NC-SA 4.0. |
| **Yelp Review Full, amazon_reviews_multi** | Yelp Dataset Agreement and Amazon license: academic / non-commercial only. |
| AG News | Source corpus is for non-commercial activity. |
| Amazon Reviews '23 | No license stated anywhere. |
| Yahoo Answers Topics, TREC, SST-5, STS-B, Climate-FEVER | License unverified ("unknown" on HF, nothing in the originals). |
| DocNLI | BSD-3 for the code; data built from ANLI (NC), CNN/DM and DUC (NIST agreement). Unusable until legal clears it. |
| HellaSwag | HF says MIT, but the GitHub repo is blocked under a wikiHow DMCA notice; the wikiHow text is not covered. |

Consequences for the todo plan (Phase 3 named ANLI, RACE and review ratings):
- **ANLI is out**, so adversarial NLI has to come from VitaminC, WANLI and our own perturbations.
- **RACE is out.** Passage-based MCQA comes from Cosmos QA, ARC and OpenBookQA.
- **No review-star dataset is usable.** Score comes from HelpSteer2/3 and summarize_from_feedback, which is closer to a judge rubric anyway.
- The default NLI baseline (`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`) was trained on ANLI. That is fine for an offline baseline, but it should not ship.

## 5. Cross-cutting caveats

1. **HF cards and original sources disagree** for ContractNLI and SciFact. Download from the original source and keep a copy of its license.
2. **Share-alike:** most Wikipedia-based sets are CC BY-SA. Redistributing the processed data triggers SA; for trained weights it is unsettled.
3. **LLM-generated data** (WANLI: GPT-3; MiniCheck, UltraFeedback, Prometheus: GPT-4): check OpenAI's terms on using outputs to build models.
4. **Hidden test labels:** GLUE QNLI, SciFact, the FEVER and FEVEROUS shared tasks, and the TabFact challenge set. Use the dev splits as test.
5. **Source text** (CNN/DM, Reddit, MS MARCO, Yelp) has its own copyright, separate from the label license.
6. **Size:** section 1 alone is well above the ~200k target of Phase 3. The sampling mix caps the large NLI sets.
7. **ClapNQ answerability is solvable from the passage alone** (found 2026-10-10 on `stage_a-full-s42`; details below). Kept as is in Stage A, fixed for Stage B.

### 5.1 ClapNQ: the passage gives the label away

**What we saw.** `stage_a-full-s42` scored **1.000 on all 600 ClapNQ test records** (`test_in` 278, `test_ood` 322, both classes), against 0.907 on Natural Questions and 0.883 on SQuAD 2.0. The zero-shot baselines got 0.618 (NLI) and 0.692 (Laya). A perfect score on a hard task is a data signal, not a skill.

**How we checked it** (the final model, 100 test records per cell, mean P(answerable); scripts were ad hoc, the method is reproducible with `baselines.load_baseline("jevlite", …)`):

| ClapNQ | passage of an answerable record | passage of an unanswerable record |
|---|---|---|
| answerable question | 0.978 (other record's passage) | 0.002 |
| unanswerable question | 0.948 | 0.002 (other record's passage) |

- The label follows the **passage**, not the question: an unanswerable question with a passage taken from an answerable record is called answerable 95% of the time, and an answerable question with a mismatched answerable passage 98%.
- The model does read the state: with the state "No text." it gives 0.01 for both kinds of question.
- **Natural Questions and SQuAD 2.0 are clean**: with any mismatched passage, from either class, P(answerable) is 0.01–0.04. The problem is specific to ClapNQ.
- Not a train/test leak: only 5% of the ClapNQ test passages occur in train, and accuracy is 1.000 on the other 95% as well. Not an obvious formatting difference either: title line, tokenisation spaces, length (median 190 vs 210 tokens) and question wording are alike; passages of unanswerable records end mid-sentence more often (34–39% vs 8–15%, depending on the split), which alone does not explain 1.000.

**Why it happened.** ClapNQ builds its two classes with different passage selection: an answerable record pairs the question with the gold passage that holds the answer, an unanswerable record pairs it with a passage from a page where no answer was found. Our converter (`data/public/qa.py`, `ClapNQ`) used both files as they are, so **no passage ever appears with both labels**, and "which file did this passage come from" is a learnable feature. What exactly in the passages separates the files is not established (likely the kind of paragraph that gets selected in each case); the swap test shows that it is enough. SQuAD 2.0 does not have this problem because its unanswerable questions were written against the same paragraphs as the answerable ones. Our NQ conversion asks whether a paragraph of the question's own page answers it; it passes the swap test, but whether its two classes share a paragraph pool was not checked separately.

**What it affects.**
- ClapNQ numbers of jev-lite (1.000) are **not evidence of answerability skill**; exclude ClapNQ from headline numbers and from the answerability average. The zero-shot baselines did not learn the shortcut, so their ClapNQ numbers stay meaningful.
- Little else: ClapNQ is 1.7% of the Bool training mix (2.5k of 150k per epoch); NQ and SQuAD 2.0 are not affected.
- The risk for the judge is the pattern, not this set: any training set where positives and negatives come from different passage pools teaches "what kind of text is this" instead of "does this text answer the question". The synthetic data of Phase 4 must avoid it (todo Phase 4).

**Decision (2026-10-10).** Keep Stage A as trained (`stage_a`, and `stage_a_nosa` with the same recipe, so the two stay comparable); a retrain costs ~10 GPU hours of the Kaggle quota for a 0.8% share of the data. **Fix for Stage B:**
- build ClapNQ answerability from the answerable file only: positive = (question, its gold passage); negative = (another question, the same passage), with the other question taken from a different Wikipedia page so it is not answered by accident;
- drop the unanswerable file from training, or keep it only in a diagnostic test set marked as such;
- rebuild the ClapNQ test records the same way, and add the 2×2 swap test above to the Phase 8 evaluation for every answerability source.

## 6. RAG / hallucination / relevance group

Checked on 2026-10-02 against the HF Hub API, the GitHub API (`license.spdx_id`) and the raw LICENSE and README files.

### Train

| Dataset | HF id / repo | Size | Labels | Gives us | License (source) | Note |
|---|---|---|---|---|---|---|
| HotpotQA | `hotpotqa/hotpot_qa` | ~90k / 7.4k val | sentence-level supporting facts, human | **claims that need two chunks** | CC BY-SA 4.0 (HF, hotpotqa.github.io); code Apache-2.0 | No hallucination labels: build negatives with our perturbations (Phase 4). |
| TAT-QA | `next-tat/TAT-QA` | ~16.5k Qs | answer-level, human | **tables + text** | CC BY 4.0 (HF, GH README) | Use with perturbed answers. |
| FinQA | `ibm-research/finqa` | 6.3k / 0.9k / 1.1k | answer + program, human | tables, numbers | HF CC BY 4.0, GH MIT (both permissive) | Use with perturbed answers. |
| TabFact | `wenhu/tab_fact` | 92k / 12.8k / 12.8k | entailed / refuted, human | tables, contradiction (no NM class) | CC BY 4.0 (HF), MIT (GH): confirmed | Tables only. |
| ClapNQ | `PrimeQA/clapnq` | 3.7k / 0.6k | long-form answers, human | **answer relevance (criterion B), unanswerable questions** | Apache-2.0 (HF); NQ text CC BY-SA 3.0 | The only commercially clean source of refusals; check which HF splits contain the unanswerable questions. |
| FaithDial | `McGill-NLP/FaithDial` | 18.4k / 3.4k / 3.5k | turn-level, crowd-edited | faithfulness in dialogue | MIT (HF, GH) | Optional: domain mismatch; Wizard of Wikipedia source terms unverified. |

### Weak supervision (LLM labels): validate on human-labelled sets

| Dataset | HF id / repo | Size | Labels | License | Note |
|---|---|---|---|---|---|
| **RAGBench** | `galileo-ai/ragbench` | ~100k, 12 subsets | response sentence → context sentence keys; relevance, utilization, adherence, completeness; **GPT-4** | CC BY 4.0 (HF); the GH repo has no license file | Closest in shape to criterion A (sentence level, several chunks, tables). **Drop the msmarco subset** (MS MARCO is NC) and check each subset's source terms. **Stage B: train/validation of 7 subsets (section 8).** |
| rag-bioasq (TinyLettuce) | `KRLabsOrg/rag-bioasq-lettucedetect` | 3k / 0.6k | span level, LLM-synthetic | MIT (HF); BioASQ terms unverified | Optional. |
| HaluEval | `pminervini/HaluEval` | 10k × QA / dialogue / summ | answer level; ChatGPT-generated hallucinations | HF Apache-2.0 vs GH MIT | Seeds from HotpotQA, CNN/DM and OpenDialKG; easy and full of artefacts. Skip, or use a little. |

### Eval only

| Dataset | Repo | Why eval-only | License |
|---|---|---|---|
| FaithEval (unanswerable / inconsistent / counterfactual) | `Salesforce/FaithEval-*-v1.0` | Test split only. **Refusals** and conflicting chunks. | HF: no license, "research purposes only"; code Apache-2.0 |
| TofuEval | GH `amazon-science/tofueval` | Inside LLM-AggreFact; README: "should not be used in training". | annotations MIT-0; MediaSum research-only, MeetingBank CC BY-NC-SA |
| ExpertQA | GH `chaitanyamalaviya/ExpertQA` | Inside LLM-AggreFact. | MIT |
| SummaC benchmark | GH `tingofurro/summac` | Standard SummaC / AlignScore benchmark. | code Apache-2.0; six source sets with their own terms |
| FRANK | GH `artidoro/frank` | In SummaC and AggreFact; sentence-level error types. | MIT; CNN/DM and XSum terms |
| FaithBench | GH `vectara/FaithBench` | Span-level, human, ~750 summaries. | **CC BY-NC-SA 4.0** |
| HaluBench | `PatronusAI/HaluBench` | Built from RAGTruth and HaluEval (overlap). | **CC BY-NC 2.0** |
| AttributionBench | `osunlp/AttributionBench` | Merges ExpertQA, LFQA, HAGRID, BEGIN and others; train only after deduplicating against LLM-AggreFact. | Apache-2.0; sources with their own terms |

### Not usable

| Dataset | Reason |
|---|---|
| MS MARCO | "non-commercial research purposes only" (msmarco site). Also taints the RAGBench msmarco subset. |
| WikiQA | MSR Data License: "may not use the data directly in a commercial product". |
| ASNQ | The LICENSE file says CC BY-NC-SA 3.0 and LICENSE-SUMMARY says CC BY-SA 3.0, so treat it as NC. Regenerate a similar set from NQ ourselves if needed. |
| AggreFact (Tang 2022) | No license file; its SOTA subset is already inside LLM-AggreFact. |

### What the specialised checkers trained on (H3 baselines)

- **LettuceDetect:** RAGTruth (model cards), plus rag-bioasq for TinyLettuce. RAGTruth is its benchmark, so it stays eval-only for us.
- **HHEM-1.0:** multi_nli, snli, fever, vitaminc and paws (model card). **HHEM-2.1-Open does not disclose its training data.** Its benchmarks are AggreFact-SOTA and RAGTruth.

### Gaps this group leaves

- **There is no clean three-way supported / contradicted / not-mentioned label at RAG sentence level.** TabFact and FaithEval-inconsistent cover contradiction only. The three-way label has to come from our synthetic data (Phase 4) and our own data (Phase 11).
- **Few commercially clean refusals:** ClapNQ and SQuAD 2.0 / NQ (section 1). FaithEval stays eval-only.
- **Unverified source terms:** AggreFact (Tang), the ASNQ data, BioASQ, OpenDialKG and Wizard of Wikipedia. Get legal sign-off before training on them.

## 7. Judge benchmarks: eval only (approved 2026-10-16)

Converters in `src/jevlite/data/public/judge_bench.py`; every record goes to `test_ood`, none to train or calib. Run on Kaggle with `notebooks/kaggle/30_judge_bench.ipynb`; report by `scripts/judge_bench_report.py`. Results: [`JUDGE_BENCH.md`](JUDGE_BENCH.md).

| Benchmark | Source | Records | Mapping | License | Overlap with Stage A training |
|---|---|---|---|---|---|
| **ContextualJudgeBench** | `Salesforce/ContextualJudgeBench` | 2,000 pairs → 4,000 records | Choice K = 2, both responses in the state, both orders (`group` = pair): consistent accuracy | CC BY 4.0 | 7 pairs (QA-Feedback passages). MRQA (SQuAD, NQ) is inside, but no passage matched our mix |
| **RAGBench** | `galileo-ai/ragbench` (formerly `rungalileo/ragbench`), 12 test splits | 11,802 | Bool "the response is supported by the documents" (`adherence_score`) | CC BY 4.0; msmarco subset NC (MS MARCO terms) | 31 records |
| **HaluBench** | `PatronusAI/HaluBench` | 14,900 | Bool PASS / FAIL | **CC BY-NC 2.0** | 4 records |
| **FaithBench** | GH `vectara/FaithBench@cf89797`, `FaithBench.csv` | 723 of 800 | Bool by `worst-label`: Unwanted → 0, Consistent / Benign → 1, Questionable left out | **CC BY-NC-SA 4.0** | none |

- **Eval only, never trained on** (user, 2026-10-16): the NC licenses allow internal evaluation, not training. *Revised 2026-10-26 for RAGBench only (CC BY 4.0): its train/validation splits of 7 subsets go into Stage B; its test splits stay eval, and those 7 become an in-distribution slice (section 8). CJB, HaluBench and FaithBench stay eval only.*
- **RAGBench labels come from GPT-4o** (responses from LLMs too): it measures agreement with an LLM judge, not with people.
- **Length:** most records exceed 512 tokens (CJB 52–88% per split, RAGBench 39–100% per subset, HaluBench CovidQA 100%). jev-lite runs at 512 (truncated) and at 8192 (untruncated, beyond its training length); NLI and Laya at 512.
- **Excluded from the report:** the 49 record ids in `docs/judge_bench/overlap_ids.txt` (passage overlap with `stage_a` and `stage_a_nosa` training files, `prefix` mode; CJB checked per passage of its concatenated contexts).

## 8. Stage B: public judge data, option B (decided 2026-10-26)

**Why.** Generating 20k synthetic cases locally (Qwen3.5-4B, llama.cpp, M1 Pro 16 GB) would take about a week, and the free GPU quotas are kept for training. Stage B takes what public data gives, built into judge records by code; synthetic generation (todo Phase 4) comes back later as a small top-up. Plan and checklist: todo Phase 4b.

**Option B** (chosen over option A, which kept the eval untouched): besides the code-built and new sources, Stage B trains on **RAGBench train/validation of 7 subsets**, the only open source of real LLM answers over several documents with sentence-level support labels. The price is a smaller OOD part of the eval (section 8.4). **Rollback:** if Stage B is worse than Stage A on the held-out slice of eval v2, go back to the Stage A checkpoint and investigate there.

### 8.1 Built by code from sources already approved

No new license questions; each needs a converter (`src/jevlite/data/public/`).

| Source | What we build | Gap it targets (`JUDGE_BENCH.md`, `STAGE_A.md`) |
|---|---|---|
| HotpotQA (distractor setting) | 10 paragraphs as chunks; "the answer to Q is A" supported only by the chunks together; negatives with the answer swapped for an entity of a distractor paragraph; **answerability** by dropping one gold paragraph while the on-topic distractors stay | claims across chunks; refusals on on-topic passages (FaithEval-unanswerable 0.49, CJB 0.28) |
| TAT-QA, FinQA | claims with the computed answer as positives; negatives with another number of the same table, a scale error or a small shift | tables and derived numbers (RAGBench FinQA 4% "supported"); the TabFact passage-swap shortcut |
| SQuAD 2.0, NQ, ClapNQ | relevance: the answer to a neighbouring question on the same passage → `does not address`; two questions joined, one answered → `partially addresses` | criterion (b), weakest primitive (HelpSteer2 ~0.5) |
| NQ, ClapNQ | conciseness pairs: gold answer vs gold plus irrelevant sentences from the same page | length bias (CJB conciseness 0.10 at 8192) |
| ClapNQ | the fix of section 5.1 (same passage, another page's question as the negative) | the passage shortcut |

HotpotQA, FinQA and TAT-QA are also RAGBench subsets; that is one reason those subsets move to the in-distribution slice (8.4).

### 8.2 New sources: license and format check (2026-10-26)

Checked against the HF Hub API (`cardData.license`), the GitHub API (`license.spdx_id`), the dataset cards / READMEs and the papers. Each source still needs the user's OK before its converter is written. Verdict: ✅ use, ⚠️ use with the condition in the note, ❌ out.

**Approved by the user (2026-10-26):** all ✅ sources; MiniCheck **C2D and D2C** (D2C news documents accepted; WiCE claims matching LLM-AggreFact dropped); RAGTruth summarisation, **CNN/DM sources only** ("Recent News" out); Trust-Data **without its ELI5 part**, identified by matching questions to the source sets (if that cannot be done cleanly, Trust-Data is left out). ConditionalQA is out.

| Source | License (where checked) | Size | Format and labels | Verdict |
|---|---|---|---|---|
| **RAGBench**, train + validation of hotpotqa, finqa, tatqa, hagrid, expertqa, techqa, pubmedqa | dataset CC BY 4.0 (HF). Sources: HotpotQA CC BY-SA 4.0, FinQA MIT / CC BY 4.0, TAT-QA CC BY 4.0, HAGRID Apache-2.0 (GH, HF), ExpertQA MIT, TechQA Apache-2.0 (GH `IBM/techqa`), PubMedQA MIT (GH) | 74.9k: tatqa 29.8k, pubmedqa 22.1k, finqa 14.3k, hagrid 3.2k, hotpotqa 2.3k, expertqa 1.8k, techqa 1.5k | question, documents, LLM response (`generation_model_name`), GPT-4 labels: `sentence_support_information` (response sentence → supporting document sentences), adherence, relevance, utilization | ✅ **Cap tatqa / pubmedqa / finqa** (85% of the rows). Responses come from commercial LLMs: the provider-terms caveat of section 5 applies, as for every LLM-made set |
| **Faithfulness-QA** (GH `qzhangFDU/faithfulness-qa-dataset`, HF `Laurie/faithfulness-qa-dataset`) | MIT (HF card, README badge); **no LICENSE file in the GH repo**. Sources: SQuAD (CC BY-SA 4.0), TriviaQA (Wikipedia evidence) | 99.1k: 79.3k / 9.9k / 9.9k (SQuAD 49.1k, TriviaQA 50k) | JSONL: question, `original_context`, modified context, original and substituted answer, entity type; deterministic rule-based substitution, no LLM | ✅ SA is approved (section 0). Passage overlap with CJB's MRQA part goes through the overlap filter |
| **MiniCheck C2D / D2C** (`lytang/C2D-and-D2C-MiniCheck`) | MIT (HF card); code Apache-2.0 (GH) | 14.4k: C2D 7,076, D2C 7,319 | claim, doc, label 0/1. C2D: ~400 Wikipedia claims with cited web articles (WiCE, Kamoi et al. 2023; Petroni et al. 2023), documents written by GPT; D2C: ~300 scraped Google News articles (from Nov 2023), claims by GPT | ⚠️ **WiCE is inside LLM-AggreFact** (our eval): check claim overlap and drop matches; D2C documents are third-party news (same copyright class as CNN/DM): take D2C only with the user's OK |
| **Trust-Data** (`declare-lab/Trust-Data`) | Apache-2.0 (HF). Sources: ASQA Apache-2.0, QAMPARI CC0-1.0 (GH), ExpertQA MIT, **ELI5: no license** (GH NOASSERTION, HF unknown; Reddit questions) | DPO 15.6k / 1.7k pairs; SFT 8.9k; Trust-Score eval 5.1k (question, answers, docs) | prompt with the retrieved documents, `chosen` / `rejected`; a refusal is `chosen` when the documents do not answer | ⚠️ Rows do not carry their source: drop **ELI5** by matching questions to the source sets, else leave Trust-Data out. ASQA overlaps CJB's QA-Feedback part (ASQA-based) and ExpertQA LLM-AggreFact: overlap filter + "seen source" flag. Use as Choice K = 2 (faithful / refusal-correct answer) and as answerability |
| **PsiloQA**, English part (`s-nlp/PsiloQA`) | CC BY 4.0 (HF); Wikipedia passages CC BY-SA | English 16.1k / 0.9k / 1.1k (of 63.8k / 3.4k / 2.9k, 14 languages) | Wikipedia passage, question and golden answer (GPT-4o), answer of one of 24 open LLMs **written without the passage**, hallucinated spans by GPT-4o (alignment with the golden answer) | ✅ with a label-noise flag: spans mark deviation from the golden answer, not from the passage, and do not separate contradicts from silent → Bool only, sentence with a span → 0 |
| **Qasper** (`allenai/qasper`) | CC BY 4.0 (HF) | 888 / 281 / 416 papers (~5k questions) | full paper text, questions, several annotator answers with evidence paragraphs, `unanswerable` flag | ✅ small; research-paper domain and human answerability |
| **ConditionalQA** (GH `haitian-sun/ConditionalQA`) | **no license**; README: "should ONLY be used for NLP research purpose" | — | — | ❌ Same status as FaithEval. The government domain with conditions stays a gap (Phase 4 top-up, e.g. from OGL-licensed gov.uk pages) |
| **RAGTruth**, summarisation part (GH `ParticleMedia/RAGTruth`) | MIT (GH). Sources: CNN/DM (628 articles) and scraped "Recent News" (315) | train 4,758 responses (1,482 with hallucinations), test 900 | response with human spans: Evident / Subtle Conflict (→ contradicts), Evident / Subtle Baseless Info (→ silent), `implicit_true` (true but not in the source → silent) | ⚠️ News copyright as for D2C: CNN/DM is the standard summarisation corpus, "Recent News" is unclear → CNN/DM only, with the user's OK. **HaluBench's RAGTruth subset is the QA part (MS MARCO passages)**, so no HaluBench records are cut |

- **Label noise:** RAGBench, MiniCheck and PsiloQA are LLM-labelled. Keep `label_source` per record and report Stage B per source.
- **Answerability sources** (HotpotQA construction, Trust-Data, Qasper) must pass the 2×2 passage-swap test (section 5.1) before they enter the mix.

### 8.3 Never trained on

CJB, HaluBench (NC), FaithBench (NC), FaithEval (research only), LLM-AggreFact (TofuEval inside: "should not be used in training"), RAGBench msmarco (NC), and the RAGBench test splits of all subsets.

### 8.4 eval v2: what we cut and how the dynamics stay comparable

| Eval set | Status from Stage B on |
|---|---|
| CJB, FaithBench | unchanged, held-out slice; CJB's QA-Feedback (ASQA) part flagged "seen source" if Trust-Data's ASQA part is trained |
| HaluBench | held-out slice; the pubmedqa part flagged "seen source" (RAGBench pubmedqa is trained). Its RAGTruth subset is the QA (MS MARCO) part, so training on the RAGTruth summaries cuts nothing |
| RAGBench test: covidqa, cuad, emanual, delucionqa | held-out slice (never trained) |
| RAGBench test: hotpotqa, finqa, tatqa, hagrid, expertqa, techqa, pubmedqa | **in-distribution slice**, reported separately, no longer OOD |
| RAGBench test: msmarco | unchanged (never trained) |
| LLM-AggreFact | held-out; the ExpertQA part flagged "seen source"; WiCE claims checked against MiniCheck C2D |
| Stage A `test_in` / `test_ood` | unchanged (Stage A sources stay in the mix as replay) |

- **The record list of eval v2 is frozen before Stage B trains** (`docs/judge_bench/eval_v2/`, built by `scripts/make_eval_v2.py`; Stage A baseline in `JUDGE_BENCH.md` section 9), together with the passage overlap of the Stage B training files (`data/overlap.py`, `prefix` mode).
- **Stage A is re-scored on eval v2 from its saved predictions** (`judge_bench_report.py --exclude …`, no GPU), and so are NLI and Laya. Every later stage is compared on the same records.

