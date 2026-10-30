# JEV as a Judge: project overview

> jev-lite, October 2026 (revised 2026-10-30 for Stage B). A typed decision model built to replace an
> LLM-as-a-judge on narrow checks. Written for the dev team; the base for the business presentation.

| Part | Sections |
|---|---|
| [1. Introduction](01_introduction.md) | 1. What JEV is and why it fits the judge role · 2. Abstract: done and planned · 3. Architecture of jev-lite |
| [2. Data and methodology](02_data_and_methodology.md) | 4. Data: Stage A, Stage B, tests, judge benchmarks (eval v2), calibration · 5. Training methodology, calibration and baselines |
| [3. Model test results](03_stage_a_results.md) | 6. Stage A against the open baselines and the JEV API; Stage B against Stage A on the same tests; calibration, probes, speed |
| [4. Judge benchmarks](04_judge_benchmarks.md) | 7. ContextualJudgeBench, FaithBench, HaluBench, RAGBench on eval v2: Stage A → Stage B, calibration, long context, response time |
| [5. Known issues, targets and roadmap](05_issues_and_next.md) | 8. Known issues · 9. Targets (quality, latency, cost) and the roadmap |

**Models.** *Stage A* = run `stage_a-full-s42`, public data with human labels, 512 tokens
([`Nefelim584/jev-lite-stage-a`](https://huggingface.co/Nefelim584/jev-lite-stage-a)). *Stage B* = run
`stage_b-full-s42-r1`, Stage A further trained on public judge data, 1024 tokens; the current main
model. *Stage B, calibrated* = Stage B with post-hoc calibration (Part 2, section 5.2).

**For sending.** `dist/` holds self-contained copies with every chart embedded (base64), so nothing is
lost when the file is forwarded:

- **`dist/OVERVIEW.html`: the one to send.** All five parts in one page with a table of contents,
  styles and charts inside the file; opens in any browser, prints to PDF (File → Print → Save as PDF,
  one part per page).
- `dist/OVERVIEW.md` (all five parts) and `dist/01_introduction.md` … `dist/05_issues_and_next.md`
  (one per part): open with images in VS Code, PyCharm, Typora and Obsidian. GitHub does not display
  embedded images, so on GitHub use the files in this folder.

Rebuild after editing:

```bash
python scripts/inline_md_images.py
uv run --extra reports python scripts/overview_html.py
```

**Charts of Part 4** come from `scripts/make_overview_figures.py` (eval v2 reports of
`scripts/judge_bench_report.py`); the Stage A charts of Part 3 from `scripts/make_jev_api_figures.py`.

**Detailed reports**: `BASELINES.md`, `STAGE_A.md`, `STAGE_A_API.md`, `JUDGE_BENCH.md`, `DATASETS.md`;
the Stage B status and numbers in `todo.md` (Phase 6, status 2026-10-26 onwards).
