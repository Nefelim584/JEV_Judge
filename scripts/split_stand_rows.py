"""Split an export of old eval-stand rows into: drop / re-judge / re-run.

    uv run python scripts/split_stand_rows.py data/mapped_q2.csv data/mapped_q2_split

Groups (one per row, every row lands in exactly one file):
  2_rejudge.csv  the stored row is complete enough for the current judge:
                 an answer, and either the guard blocked the question (the judge
                 needs no retrieved context for a refusal) or retrieved_context
                 was logged.
  3_rerun.csv    the question is usable but the row is not: an allowed question
                 whose retrieved_context was not logged, an asker question with
                 no answer, or a pipeline error on a real question. Deduplicated
                 on (question, transcript); the first row is kept.
  1_drop.csv     placeholder questions from failed runs, multi-turn attacks
                 that would need a re-run, duplicate re-run rows.

Columns are kept as exported, so the stand can read the files as before; the
group and reason of every row are in manifest.csv. Input files are company data:
keep input and output under data/ (git-ignored).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PLACEHOLDERS = {"(conversation failed)", "(asker failed)"}


def classify(df: pd.DataFrame) -> pd.DataFrame:
    has_answer = df["answer"].notna() & (df["answer"].astype(str).str.strip() != "")
    has_ctx = df["retrieved_context"].notna() & (df["retrieved_context"].astype(str).str.strip() != "")
    blocked = df["guard_is_allowed"].astype(str).str.lower().eq("false")
    placeholder = df["question"].astype(str).str.strip().isin(PLACEHOLDERS) | df["question"].isna()

    group = pd.Series("rerun", index=df.index)
    reason = pd.Series("", index=df.index)

    rejudge_blocked = has_answer & blocked
    rejudge_ctx = has_answer & ~blocked & has_ctx
    group[rejudge_blocked | rejudge_ctx] = "rejudge"
    reason[rejudge_blocked] = "guard blocked: refusal, no context needed"
    reason[rejudge_ctx] = "allowed, retrieved_context logged"

    rerun = group.eq("rerun")
    reason[rerun & has_answer] = "allowed, retrieved_context not logged"
    reason[rerun & ~has_answer & df["error"].notna()] = "pipeline error, question usable"
    reason[rerun & ~has_answer & df["error"].isna()] = "question only (asker), no answer"

    group[placeholder] = "drop"
    reason[placeholder] = "placeholder question from a failed run"

    # multi-turn attacks are not re-run: the assistant is fixed against them, so
    # they only add more refusals (decision 2026-11-11)
    mt_attack = group.eq("rerun") & df["transcript"].notna() & df["category"].astype(str).str.startswith("attack_")
    group[mt_attack] = "drop"
    reason[mt_attack] = "multi-turn attack, not re-run"

    # one re-run per (question, transcript)
    key = df["question"].astype(str) + "\x00" + df["transcript"].fillna("").astype(str)
    dup = group.eq("rerun") & key.where(group.eq("rerun")).duplicated(keep="first")
    group[dup] = "drop"
    reason[dup] = "duplicate of another re-run question"

    return pd.DataFrame({"id": df["id"], "group": group, "reason": reason,
                         "multi_turn": df["transcript"].notna(), "category": df["category"]})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("out_dir", type=Path)
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    m = classify(df)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, g in (("1_drop", "drop"), ("2_rejudge", "rejudge"), ("3_rerun", "rerun")):
        df[m["group"].eq(g)].to_csv(args.out_dir / f"{name}.csv", index=False)
    m.to_csv(args.out_dir / "manifest.csv", index=False)

    assert m["group"].value_counts().sum() == len(df)
    print(f"{len(df)} rows from {args.src}")
    print(m.groupby(["group", "reason"]).size().to_string())
    print("\nmulti-turn by group:", m[m["multi_turn"]].groupby("group").size().to_dict())
    print("\ncategory x group:")
    print(pd.crosstab(m["category"], m["group"]).to_string())


if __name__ == "__main__":
    main()
