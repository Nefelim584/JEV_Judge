# JEV as a Judge: project overview

> jev-lite, October 2026. A typed decision model built to replace an LLM-as-a-judge on narrow checks.
> Written for the dev team; the base for the business presentation.

| Part | Sections |
|---|---|
| [1. Introduction](01_introduction.md) | 1. What JEV is and why it fits the judge role · 2. Abstract: done and planned · 3. Architecture of jev-lite |
| [2. Data and methodology](02_data_and_methodology.md) | 4. Data used so far · 5. Training methodology and baselines |
| [3. Stage A results](03_stage_a_results.md) | 6. Results against the open baselines and the JEV API: faithfulness, calibration, probes, speed |
| [4. Judge benchmarks](04_judge_benchmarks.md) | 7. ContextualJudgeBench, FaithBench, HaluBench, RAGBench; long context |
| [5. Known issues and next steps](05_issues_and_next.md) | 8. Known issues · 9. What we do next |

**For sending.** `dist/` holds self-contained copies with every chart embedded (base64), so nothing is
lost when the file is forwarded: `dist/OVERVIEW.md` (all five parts in one file) and
`dist/01_introduction.md` … `dist/05_issues_and_next.md` (one file per part). They open with images in
VS Code, PyCharm, Typora and Obsidian; GitHub does not display embedded images, so on GitHub use the
files in this folder. Rebuild after editing:

```bash
python scripts/inline_md_images.py
```

**Detailed reports**: `BASELINES.md`, `STAGE_A.md`, `STAGE_A_API.md`, `JUDGE_BENCH.md`, `DATASETS.md`.
