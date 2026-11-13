"""Criteria bench on Modal: jev-lite vs specialist guards vs an open LLM judge (``jevlite.criteria_bench``).

    uv run --extra bench modal run scripts/modal_criteria_bench.py                                   # everything
    uv run --extra bench modal run scripts/modal_criteria_bench.py --scorers jevlite,nli --per-source 50 --single 5   # smoke
    uv run --extra bench modal run scripts/modal_criteria_bench.py --custom data/criteria/advice.jsonl --tasks ""     # own labels only
    uv run --extra bench modal run scripts/modal_criteria_bench.py --jevlite vol:runs/stage_c-full-s42/final   # a Stage C model
    uv run --extra bench modal run scripts/modal_criteria_bench.py --report-only data/criteria_bench/<run>     # metrics again, no GPU

Tasks (public, labelled): ``grounding`` (HaluBench, ``--per-source`` per source), ``answer_harm`` (Aegis 2.0
test, rows with a response), ``toxicity`` (ToxiGen annotated test); out-of-domain for Stage D (never trained
on): ``beavertails`` (30k test, ``5 × --per-source`` drawn), ``toxic_chat`` (test, same cap), ``oai_moderation`` (all). ``--custom`` adds a file of Bool records
in the unified format as task ``custom:<file stem>`` (unauthorised advice, disclosures, … — anything without
a public set); a record may carry ``context`` / ``question`` / ``answer`` / ``user`` fields for the
specialists. Custom records are read locally and sent to the Modal containers only (company data rule:
never to the Hub or Kaggle; results land in the git-ignored ``data/``).

Scorers (``jevlite.baselines.criteria``): ``jevlite``, ``nli``, ``llm_judge`` (Qwen3-8B), ``granite_guardian``,
``hhem``, ``lettucedetect``, ``toxic_bert``. One container per scorer, all on the same GPU type (``--gpu``,
default L40S), so the timings compare. The items are drawn once (a CPU container) and sent to every scorer.
A scorer that fails is reported and the others go on; ``--resume`` keeps the scores already saved in the
run folder and runs only what is missing: scorers without a file or that failed to load, and the tasks a
scorer failed on (merged into its file). Batches that run out of GPU memory are halved and retried; an item
that does not fit alone is counted as missing.

Output: ``data/criteria_bench/<run>/``: ``items.jsonl``, ``scores/<scorer>.json`` (P(positive) per item +
timing), ``summary.json`` and ``report.md``. Every scorer's JSON is also saved to the volume
``jevlite-hf-cache`` (``criteria/<run>/``), fetch with ``modal volume get jevlite-hf-cache criteria/<run>/ <dir>``.

Read the numbers with these in mind: Granite Guardian was trained on Aegis 1.0, BeaverTails and ToxicChat
(close to ``answer_harm``); LettuceDetect on RAGTruth (a HaluBench source); jev-lite's Stage B on HotpotQA
(close to HaluEval). jev-lite has seen no toxicity / safety data at all. Slices show where each one's lead
comes from.

Setup once: ``uv run --extra bench modal setup``. All models and datasets are public and ungated, so no
Hugging Face secret is needed (a ``vol:`` jev-lite reads the private volume ``jevlite-stage-c``).
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path

import modal

HF_CACHE = "/root/.cache/huggingface"
VOL = "/vol"
OUT = Path("data/criteria_bench")
DEFAULT_SCORERS = "jevlite,nli,llm_judge,granite_guardian,hhem,lettucedetect,toxic_bert"
DEFAULT_TASKS = "grounding,answer_harm,toxicity,beavertails,toxic_chat,oai_moderation"

app = modal.App("jevlite-criteria-bench")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(".", extras=["bench"], frozen=True)
    .env({"HF_HOME": HF_CACHE, "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("jevlite")
)
hf_cache = modal.Volume.from_name("jevlite-hf-cache", create_if_missing=True)
stage_c = modal.Volume.from_name("jevlite-stage-c", create_if_missing=True)


@app.function(image=image, cpu=4.0, memory=16384, timeout=3600, volumes={HF_CACHE: hf_cache})
def build_items(tasks: list[str], per_source: int, seed: int) -> dict[str, list[dict]]:
    from jevlite.criteria_bench import load_task

    out = {}
    for task in tasks:
        out[task] = [it.to_dict() for it in load_task(task, per_source or None, seed)]
        print(f"{task}: {len(out[task])} items", flush=True)
    hf_cache.commit()
    return out


@app.function(image=image, gpu="L40S", cpu=4.0, memory=32768, timeout=4 * 3600, volumes={HF_CACHE: hf_cache, VOL: stage_c})
def run_scorer(key: str, items_by_task: dict[str, list[dict]], jevlite: str, llm: str, single_k: int, run: str) -> dict:
    import time
    import traceback

    import numpy as np
    import torch

    from jevlite.baselines.criteria import load_scorer
    from jevlite.criteria_bench import Item

    if jevlite.startswith("vol:"):
        stage_c.reload()
        jevlite = f"{VOL}/{jevlite.removeprefix('vol:').lstrip('/')}"
    result: dict = {"key": key, "gpu": torch.cuda.get_device_name(0), "tasks": {}, "errors": {}}
    t0 = time.perf_counter()
    try:
        scorer = load_scorer(key, "cuda", jevlite=jevlite, llm=llm)
    except Exception as e:  # noqa: BLE001 — report it, the other scorers go on
        traceback.print_exc()
        return {**result, "name": key, "error": f"{type(e).__name__}: {e}"}
    result.update(name=scorer.name, load_s=time.perf_counter() - t0)
    hf_cache.commit()  # downloaded weights, for the next run
    print(f"{scorer.name} loaded in {result['load_s']:.0f} s on {result['gpu']}", flush=True)

    for task, dicts in items_by_task.items():
        items = [it for it in (Item.from_dict(d) for d in dicts) if scorer.supports(it)]
        if not items:
            continue
        try:
            scorer.score(items[:2])  # warm-up
            torch.cuda.synchronize()
            start = time.perf_counter()
            scores = scorer.score(items)
            batched = (time.perf_counter() - start) * 1000 / len(items)
            single = []
            for it in items[:single_k]:
                s = time.perf_counter()
                scorer.score([it])
                torch.cuda.synchronize()
                single.append((time.perf_counter() - s) * 1000)
            timing = {"batched_ms_per_item": batched, "n_single": len(single)}
            if single:
                timing.update(single_p50_ms=float(np.percentile(single, 50)), single_p90_ms=float(np.percentile(single, 90)))
            result["tasks"][task] = {"scores": {it.id: s for it, s in zip(items, scores, strict=True)}, "timing": timing}
            n_nan = sum(1 for s in scores if s != s)
            print(f"{scorer.name} / {task}: {len(items)} items, {batched:.1f} ms/item batched"
                  + (f", p50 1-item {timing['single_p50_ms']:.1f} ms" if single else "") + (f", {n_nan} unreadable" if n_nan else ""),
                  flush=True)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            result["errors"][task] = f"{type(e).__name__}: {e}"

    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    path = Path(HF_CACHE) / "criteria" / run / f"{key}-{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result))
    hf_cache.commit()
    return result


def _write_jsonl(path: Path, rows) -> None:
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def tasks_to_run(path: Path, tasks: list[str]) -> list[str]:
    """What ``--resume`` still has to run for one scorer: every task if it has no file or failed to load,
    else the tasks it failed on (tasks it does not support are absent from its file without an error)."""
    if not path.exists():
        return tasks
    saved = json.loads(path.read_text())
    if saved.get("error"):
        return tasks
    return [t for t in tasks if t in (saved.get("errors") or {})]


def merge_scores(old: dict, new: dict) -> dict:
    """A resumed scorer's result on top of its saved one: new tasks win, their old errors are dropped."""
    if old.get("error") or new.get("error"):
        # a failed load keeps the saved result (its task errors stay, so the next resume retries them)
        return old if new.get("error") and not old.get("error") else new
    errors = {t: e for t, e in (old.get("errors") or {}).items() if t not in new.get("tasks", {})}
    errors.update(new.get("errors") or {})
    return {**old, **{k: v for k, v in new.items() if k not in ("tasks", "errors")},
            "tasks": {**old.get("tasks", {}), **new.get("tasks", {})}, "errors": errors}


def write_report(run_dir: Path, calib_frac: float, seed: int, n_boot: int) -> None:
    from jevlite.criteria_bench import Item, report_markdown, summarize_by_slice

    items = [Item.from_dict(d) for d in _read_jsonl(run_dir / "items.jsonl")]
    tasks: dict[str, list[Item]] = {}
    for it in items:
        tasks.setdefault(it.task, []).append(it)
    results = {}
    for path in sorted((run_dir / "scores").glob("*.json")):
        res = json.loads(path.read_text())
        entry = {"key": res["key"], "gpu": res.get("gpu"), "error": res.get("error"), "tasks": {}}
        for task, t in (res.get("tasks") or {}).items():
            supported = [it for it in tasks.get(task, []) if it.id in t["scores"]]
            entry["tasks"][task] = {
                "metrics": summarize_by_slice(supported, t["scores"], calib_frac=calib_frac, seed=seed, n_boot=n_boot),
                "timing": t.get("timing", {}),
            }
        if res.get("errors"):
            entry["error"] = "; ".join(f"{k}: {v}" for k, v in res["errors"].items())
        results[res.get("name", res["key"])] = entry
    (run_dir / "summary.json").write_text(json.dumps(results, indent=2))
    gpus = sorted({r["gpu"] for r in results.values() if r.get("gpu")})
    head = f"# Criteria bench: {run_dir.name}\n\nGPU: {', '.join(gpus) or '?'}. Seed {seed}.\n\n"
    report = head + report_markdown(results, tasks, calib_frac)
    (run_dir / "report.md").write_text(report)
    print("\n" + report)
    print(f"→ {run_dir / 'report.md'}")


@app.local_entrypoint()
def main(scorers: str = DEFAULT_SCORERS, tasks: str = DEFAULT_TASKS, per_source: int = 300, custom: str = "",
         gpu: str = "L40S", jevlite: str = "hub:Nefelim584/jev-lite@v0.2.0", llm: str = "Qwen/Qwen3-8B",
         run: str = "", seed: int = 0, calib_frac: float = 0.3, n_boot: int = 1000, single: int = 50,
         resume: bool = False, report_only: str = ""):
    from jevlite.baselines.criteria import SCORERS
    from jevlite.criteria_bench import BUILTIN_TASKS, custom_items

    if report_only:
        write_report(Path(report_only), calib_frac, seed, n_boot)
        return

    keys = [s.strip() for s in scorers.split(",") if s.strip()]
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    if bad := [k for k in keys if k not in SCORERS]:
        raise SystemExit(f"unknown scorers {bad}, expected {SCORERS}")
    if bad := [t for t in task_list if t not in BUILTIN_TASKS]:
        raise SystemExit(f"unknown tasks {bad}, expected {BUILTIN_TASKS} (own data: --custom)")
    custom_paths = [Path(p.strip()) for p in custom.split(",") if p.strip()]
    if not task_list and not custom_paths:
        raise SystemExit("nothing to score: pass --tasks and/or --custom")

    run = run or datetime.datetime.now().strftime("%Y%m%d-%H%M")
    run_dir = OUT / re.sub(r"[^A-Za-z0-9_.-]+", "_", run)
    (run_dir / "scores").mkdir(parents=True, exist_ok=True)

    if resume and (run_dir / "items.jsonl").exists():
        all_items = _read_jsonl(run_dir / "items.jsonl")
        by_task: dict[str, list[dict]] = {}
        for d in all_items:
            by_task.setdefault(d["task"], []).append(d)
        print(f"resume {run_dir}: {len(all_items)} items")
    else:
        by_task = build_items.remote(task_list, per_source, seed) if task_list else {}
        for path in custom_paths:
            name = path.stem
            by_task[f"custom:{name}"] = [it.to_dict() for it in custom_items(_read_jsonl(path), name)]
        _write_jsonl(run_dir / "items.jsonl", [d for ds in by_task.values() for d in ds])
        for task, ds in by_task.items():
            print(f"{task}: {len(ds)} items, positive rate {sum(d['target'] for d in ds) / max(len(ds), 1):.2f}")

    (run_dir / "config.json").write_text(json.dumps({
        "scorers": keys, "tasks": list(by_task), "per_source": per_source, "gpu": gpu, "jevlite": jevlite, "llm": llm,
        "seed": seed, "calib_frac": calib_frac, "single": single, "custom": [str(p) for p in custom_paths],
    }, indent=2))

    todo = {k: tasks_to_run(run_dir / "scores" / f"{k}.json", list(by_task)) if resume else list(by_task) for k in keys}
    if resume:
        print("to run: " + (", ".join(f"{k} ({', '.join(t)})" for k, t in todo.items() if t) or "nothing"))
    calls = {k: run_scorer.with_options(gpu=gpu).spawn(k, {t: by_task[t] for t in ts}, jevlite, llm, single, run_dir.name)
             for k, ts in todo.items() if ts}
    for key, call in calls.items():
        try:
            res = call.get()
        except Exception as e:  # noqa: BLE001 — keep the other scorers
            res = {"key": key, "name": key, "error": f"{type(e).__name__}: {e}", "tasks": {}}
        path = run_dir / "scores" / f"{key}.json"
        if resume and path.exists():
            res = merge_scores(json.loads(path.read_text()), res)
        path.write_text(json.dumps(res))
        status = res.get("error") or ", ".join(res.get("tasks", {})) or "no supported task"
        print(f"{key}: {status}")

    write_report(run_dir, calib_frac, seed, n_boot)
