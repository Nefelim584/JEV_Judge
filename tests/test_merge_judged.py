import csv
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("merge_judged", Path(__file__).parents[1] / "scripts" / "merge_judged.py")
mj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mj)


def _row(qid, score="", error="", **kw):
    return {"question_id": qid, "repeat": "1", "turn": "", "overall_score": score, "error": error, **kw}


def test_last_scored_wins_and_error_never_replaces_score():
    base = [_row(q, error="Traceback\nConnectionError: boom", source_file="old.csv") for q in ("a", "b", "c", "d")]
    run1 = [_row("a", "4.0"), _row("b", error="ValidationError: score missing"), _row("x", "5.0")]
    run2 = [_row("a", error="ValidationError: later failure"), _row("b", "3.5"), _row("a", "4.5")]
    ok, failed, origin, unknown = mj.merge(base, [("run1", run1), ("run2", run2)])

    got = {r["question_id"]: r for r in ok}
    assert got["a"]["overall_score"] == "4.5" and got["a"]["judged_from"] == "run2"
    assert got["b"]["overall_score"] == "3.5"
    assert got["a"]["source_file"] == "old.csv"  # base-only column keeps the base value
    assert [r["question_id"] for r in failed] == ["c", "d"] and failed[0]["judged_from"] == "base"
    assert origin == {"run2": 2, "unresolved": 2} and unknown == 1


def test_cli_writes_scored_and_failed(tmp_path):
    def w(p, rows):
        with open(p, "w", newline="") as f:
            d = csv.DictWriter(f, fieldnames=rows[0].keys())
            d.writeheader()
            d.writerows(rows)

    w(tmp_path / "base.csv", [_row("a", error="x"), _row("b", error="y")])
    w(tmp_path / "j.csv", [_row("a", "4.0")])
    out = tmp_path / "merged.csv"
    mj.main(["--base", str(tmp_path / "base.csv"), "--judged", str(tmp_path / "j.csv"), "--out", str(out)])
    assert [r["question_id"] for r in mj.read(out)] == ["a"]
    assert [r["question_id"] for r in mj.read(tmp_path / "merged-failed.csv")] == ["b"]


cspec = importlib.util.spec_from_file_location("combine_judged", Path(__file__).parents[1] / "scripts" / "combine_judged.py")
cj = importlib.util.module_from_spec(cspec)
cspec.loader.exec_module(cj)


def test_combine_unions_columns_and_rejects_duplicate_keys():
    ok, failed, fields = cj.combine([("r1", [_row("a", "4.0", source_file="s"), _row("b", error="x")]),
                                     ("r2", [_row("c", "3.0", judged_from="j")])])
    assert [r["question_id"] for r in ok] == ["a", "c"] and [r["question_id"] for r in failed] == ["b"]
    assert {"source_file", "judged_from"} <= set(fields) and fields[-1] == "results_file"
    try:
        cj.combine([("r1", [_row("a", "4.0")]), ("r2", [_row("a", "5.0")])])
        raise AssertionError("duplicate key accepted")
    except ValueError:
        pass
