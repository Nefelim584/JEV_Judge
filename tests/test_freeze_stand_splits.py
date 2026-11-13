import importlib.util
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("freeze_stand_splits", Path(__file__).parents[1] / "scripts" / "freeze_stand_splits.py")
fz = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fz)


SCORES = ["grounded", "tone"]  # made-up criteria, as in tests/test_stand.py


def _rows(n_cat: int = 40) -> pd.DataFrame:
    rows = []
    for cat in ("cat_a", "cat_b"):
        for i in range(n_cat):
            rows.append({"question_id": f"{cat}-{i:03d}", "category": cat, "error": None,
                         "question": f"{cat} topic {i} asks about field number {i} and its allowed values today",
                         "grounded": 5, "tone": 4})
    return pd.DataFrame(rows)


def test_cluster_merges_paraphrases_only():
    qs = ["what does setting 38 mean in the export dialog",
          "What does setting 38 mean in the Export dialog?",
          "how do i undo a change before the file is saved"]
    lab = fz.cluster(qs, 0.6)
    assert lab[0] == lab[1] != lab[2]


def test_freeze_keeps_groups_whole_and_locks_other():
    df = _rows()
    dup = df.iloc[[0]].assign(question_id="cat_a-900")  # same question asked twice
    df = pd.concat([df, dup, df.iloc[[5]].assign(question_id="x-1", error="boom")], ignore_index=True)
    other = [df.loc[1, "question"]]  # an old question: its group must stay in train
    fresh, splits, s4, report = fz.freeze(df, other, scores=SCORES, batch="n1", test=20, calib=10, s4=8)

    assert fresh["id"].str.startswith("n1-").all() and fresh["id"].is_unique
    assert report["left_out_error_or_no_scores"] == 1
    split_of = dict(zip(splits["id"], splits["split"]))
    assert split_of["n1-cat_a-000"] == split_of["n1-cat_a-900"]
    assert split_of["n1-cat_a-001"] == "train"
    counts = splits["split"].value_counts()
    assert counts["test_in"] <= 20 and counts["calib"] <= 10
    assert set(s4["id"]) <= set(splits.loc[splits["split"] == "test_in", "id"])
    assert len(s4) == 8
