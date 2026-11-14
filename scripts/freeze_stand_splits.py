"""Freeze test / calib splits of fresh eval-stand runs and pick the S4 rows (judge repeats).

    uv run python scripts/freeze_stand_splits.py --fresh data/mapped_q2_split/combined_newQ.csv \\
        --criteria data/stand/criteria.json \\
        --batch n1 --other data/mapped_q2_split/{1_drop,2_rejudge,3_rerun}.csv \\
        --out data/stand/freeze_n1 --test 250 --s4 150

``dataset_gen.md`` sections 6, 8. Steps:

1. Every fresh row gets a stable ``id`` = ``<batch>-<question_id>``, and ``question_id`` gets the same
   prefix: the stand numbers questions per run (``<category>-001``), so two exports would clash.
2. Rows with an error or without judge scores are left out (the converter skips them too).
3. Groups: questions that are the same or near the same (Jaccard of word 3-grams >= ``--jaccard``) form
   one group, across the fresh rows and the ``--other`` files (single linkage: paraphrase chains of the
   asker merge into large groups). A group that touches an ``--other`` question, or has more than
   ``--max-group`` rows, stays in train.
4. Groups are drawn (seeded) into test, then calib, per category in proportion to the fresh rows, up to
   ``--test`` / ``--calib`` rows. The rest is train. ``--calib`` defaults to 0: the first fresh batch
   (n1) has too few free edge-case groups for both, so calib is drawn later from re-runs and dialogues.
5. S4: ``--s4`` rows of test, stratified by category, to be judged again (judge-only mode, twice).

Writes to ``--out``: ``fresh_with_ids.csv`` (the fresh rows, stand columns + ``id``: use this file from
now on), ``splits.csv`` (``id,split`` for ``convert_stand.py --splits``), ``s4_rows.csv`` (the S4 rows,
same columns, ready for the stand's judge-only mode) and ``report.json`` (counts only).
Input and output are company data: keep them under ``data/`` (git-ignored).
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

def score_columns(criteria_path: str) -> list[str]:
    """The judge's score columns: the criteria of the spec (company data, git-ignored, see jevlite.data.stand)."""
    return list(json.loads(Path(criteria_path).read_text())["criteria"])


def shingles(text: str, n: int = 3) -> frozenset[str]:
    words = re.findall(r"[a-z0-9]+", str(text).lower())
    if len(words) < n:
        return frozenset([" ".join(words)])
    return frozenset(" ".join(words[i : i + n]) for i in range(len(words) - n + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def cluster(questions: list[str], threshold: float) -> list[int]:
    """Union-find over questions: same normalised text or Jaccard >= threshold → one cluster. O(n²)."""
    parent = list(range(len(questions)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sh = [shingles(q) for q in questions]
    for i in range(len(sh)):
        for j in range(i + 1, len(sh)):
            if find(i) != find(j) and (sh[i] == sh[j] or jaccard(sh[i], sh[j]) >= threshold):
                parent[find(i)] = find(j)
    return [find(i) for i in range(len(questions))]


def usable(df: pd.DataFrame, scores: list[str]) -> pd.Series:
    no_error = df["error"].isna() | (df["error"].astype(str).str.strip() == "")
    return no_error & df[scores].notna().any(axis=1)


def draw(groups: dict[int, list[int]], category: pd.Series, eligible: set[int], target: int, seed: int, tag: str) -> set[int]:
    """Groups into one split: per category, quota = target × the category's share of all fresh rows."""
    share = category.value_counts(normalize=True)
    by_cat: dict[str, list[int]] = defaultdict(list)
    for g in sorted(eligible):
        by_cat[category.iloc[groups[g][0]]].append(g)
    taken: set[int] = set()
    for cat, gs in sorted(by_cat.items()):
        quota = round(target * share.get(cat, 0.0))
        random.Random(f"{seed}-{tag}-{cat}").shuffle(gs)
        n = 0
        for g in gs:
            if n >= quota:
                break
            if n + len(groups[g]) <= quota:
                taken.add(g)
                n += len(groups[g])
    return taken


def freeze(fresh: pd.DataFrame, other_questions: list[str], *, scores: list[str], batch: str, test: int, calib: int, s4: int,
           jaccard_min: float = 0.6, max_group: int = 20, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    fresh = fresh.copy()
    if fresh["question_id"].isna().any():
        raise ValueError("every fresh row needs a question_id")
    fresh["question_id"] = f"{batch}-" + fresh["question_id"].astype(str)
    if fresh["question_id"].duplicated().any():
        raise ValueError("question_id is not unique inside the fresh export")
    fresh.insert(0, "id", fresh["question_id"])

    ok = usable(fresh, scores)
    rows = fresh[ok].reset_index(drop=True)
    qs = rows["question"].astype(str).tolist()
    labels = cluster(qs + list(other_questions), jaccard_min)
    fresh_lab, other_lab = labels[: len(qs)], set(labels[len(qs) :])

    groups: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(fresh_lab):
        groups[g].append(i)
    eligible = {g for g, m in groups.items() if g not in other_lab and len(m) <= max_group}

    t = draw(groups, rows["category"], eligible, test, seed, "test")
    c = draw(groups, rows["category"], eligible - t, calib, seed, "calib")
    split = pd.Series("train", index=rows.index)
    for g in t:
        split.iloc[groups[g]] = "test_in"
    for g in c:
        split.iloc[groups[g]] = "calib"
    splits = pd.DataFrame({"id": rows["id"], "split": split})

    test_rows = rows[split == "test_in"]
    share = test_rows["category"].value_counts(normalize=True)
    picked: list[int] = []
    for cat, sub in sorted(test_rows.groupby("category")):
        n = min(len(sub), round(s4 * share[cat]))
        picked += sub.sample(n=n, random_state=seed).index.tolist()
    s4_rows = rows.loc[sorted(picked)]

    report = {
        "fresh_rows": len(fresh), "left_out_error_or_no_scores": int((~ok).sum()), "usable_rows": len(rows),
        "groups": len(groups), "groups_multi_row": sum(len(m) > 1 for m in groups.values()),
        "groups_locked_train_other": sum(g in other_lab for g in groups),
        "groups_locked_train_size": sum(len(m) > max_group for m in groups.values()),
        "rows_per_split": dict(Counter(split)),
        "split_by_category": {k: dict(v) for k, v in pd.crosstab(rows["category"], split).to_dict().items()},
        "s4_rows": len(s4_rows), "s4_by_category": dict(Counter(s4_rows["category"])),
        "test_score_dist": {s: {str(k): int(v) for k, v in test_rows[s].value_counts().sort_index().items()} for s in scores},
        "settings": {"batch": batch, "jaccard": jaccard_min, "max_group": max_group, "seed": seed, "test": test, "calib": calib, "s4": s4},
    }
    return fresh, splits, s4_rows, report


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fresh", required=True, help="stand export of fresh runs (current assistant)")
    ap.add_argument("--criteria", required=True, help="criteria spec JSON (its keys are the score columns)")
    ap.add_argument("--batch", required=True, help="prefix for ids of this export, e.g. n1")
    ap.add_argument("--other", nargs="*", default=[], help="other stand files: their questions lock groups to train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test", type=int, default=250)
    ap.add_argument("--calib", type=int, default=0)
    ap.add_argument("--s4", type=int, default=150)
    ap.add_argument("--jaccard", type=float, default=0.6)
    ap.add_argument("--max-group", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    fresh = pd.read_csv(args.fresh)
    other = [q for p in args.other for q in pd.read_csv(p)["question"].dropna().astype(str)]
    fresh_ids, splits, s4_rows, report = freeze(
        fresh, other, scores=score_columns(args.criteria), batch=args.batch, test=args.test, calib=args.calib, s4=args.s4,
        jaccard_min=args.jaccard, max_group=args.max_group, seed=args.seed,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fresh_ids.to_csv(out / "fresh_with_ids.csv", index=False)
    splits.to_csv(out / "splits.csv", index=False)
    s4_rows.to_csv(out / "s4_rows.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("usable_rows", "groups", "rows_per_split", "s4_rows")}, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
