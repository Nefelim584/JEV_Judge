# Public datasets: candidates and licenses

> **Status:** the Stage A list is approved (section 0, 2026-10-02). Sections 1–6 hold all the candidates and the license checks behind it.
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
- **LLM-generated data: WANLI only** (GPT-3 wrote the pairs; humans labelled and revised them). RAGBench, MiniCheck C2D/D2C, UltraFeedback and Prometheus stay out of Stage A (ablations, section 2).
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
| **RAGBench** | `galileo-ai/ragbench` | ~100k, 12 subsets | response sentence → context sentence keys; relevance, utilization, adherence, completeness; **GPT-4** | CC BY 4.0 (HF); the GH repo has no license file | Closest in shape to criterion A (sentence level, several chunks, tables). **Drop the msmarco subset** (MS MARCO is NC) and check each subset's source terms. |
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
