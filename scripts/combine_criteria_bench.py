"""Put criteria-bench runs on the same items into one run folder, for one report (``modal_criteria_bench.py``).

    uv run python scripts/combine_criteria_bench.py --out data/criteria_bench/crit-all \\
        data/criteria_bench/crit-v020 data/criteria_bench/crit-ood \\
        "data/criteria_bench/crit-stage-d-8k:jevlite=jev-lite D (stage_d-full-s42)" \\
        "data/criteria_bench/crit-stage-c:jevlite=jev-lite C (stage_c-full-s42)"
    uv run --extra bench modal run scripts/modal_criteria_bench.py --report-only data/criteria_bench/crit-all

A run is ``DIR`` or ``DIR:scorer=new name,…`` (renames a scorer, e.g. to tell jev-lite versions apart; the
renamed scorer gets its own file). Scores of the same scorer from several runs are merged by task (later
runs win). Items are merged by id; an id with a different target in two runs is an error (not the same
items: the runs must share ``--per-source`` and ``--seed``).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def parse(spec: str) -> tuple[Path, dict[str, str]]:
    path, _, ren = spec.partition(":")
    renames = dict(r.split("=", 1) for r in ren.split(",") if r.strip()) if ren else {}
    return Path(path), renames


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    items: dict[str, dict] = {}
    scores: dict[str, dict] = {}
    for spec in args.runs:
        run, renames = parse(spec)
        for it in read_jsonl(run / "items.jsonl"):
            old = items.get(it["id"])
            if old is not None and old["target"] != it["target"]:
                raise SystemExit(f"{it['id']}: target {old['target']} vs {it['target']} in {run}: not the same items")
            items[it["id"]] = it
        for f in sorted((run / "scores").glob("*.json")):
            res = json.loads(f.read_text())
            key = res["key"]
            if key in renames:
                res["name"] = renames[key]
                key = re.sub(r"[^A-Za-z0-9_]+", "_", renames[key]).strip("_").lower()
                res["key"] = key
            if key in scores:
                merged = scores[key]
                merged["tasks"] = {**merged.get("tasks", {}), **res.get("tasks", {})}
                merged["errors"] = {**merged.get("errors", {}), **res.get("errors", {})}
            else:
                scores[key] = res
            print(f"{run.name}/{f.name} → {key}: {', '.join(res.get('tasks', {}))}")

    out = Path(args.out)
    (out / "scores").mkdir(parents=True, exist_ok=True)
    (out / "items.jsonl").write_text("".join(json.dumps(it) + "\n" for it in items.values()))
    for key, res in scores.items():
        (out / "scores" / f"{key}.json").write_text(json.dumps(res))
    print(f"{len(items)} items, {len(scores)} scorers → {out}")


if __name__ == "__main__":
    main()
