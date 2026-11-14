"""scripts/stand_report.py and scripts/combine_criteria_bench.py on made-up records (no company data)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sr = _load("stand_report")
cb = _load("combine_criteria_bench")


def test_qwk_bucket_and_auroc():
    a = np.array([0, 1, 2, 3, 4])
    assert sr.qwk(a, a, 5) == pytest.approx(1.0)
    assert sr.qwk(a, a[::-1], 5) < 0
    assert sr.bucket(np.array([0, 1, 2, 3, 4])).tolist() == [0, 0, 1, 2, 2]
    assert sr.auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)
    assert sr.auroc(np.array([0, 1]), np.array([0.5, 0.5])) == pytest.approx(0.5)


def _rec(row, crit, split, target, scores=None, typ="score", category="cat_a"):
    r = {"id": f"stand:{row}:{crit}", "source": "stand", "split": split, "type": typ, "criterion": crit,
         "stand_row": row, "category": category, "target": target, "multi_turn": False}
    if typ == "score":
        r["candidates"] = ["1", "2", "3", "4", "5"]
        r["judge_scores"] = scores or [target + 1]
    return r


def test_report_exact_s4_and_constant():
    recs = [_rec(f"t{i}", "tone", "train", 3) for i in range(5)]  # train mode: level 4
    recs += [_rec("a", "tone", "test_in", 3, [4, 4, 5]), _rec("b", "tone", "test_in", 1, [2]),
             _rec("a", "faith", "test_in", 1.0, typ="bool"), _rec("b", "faith", "test_in", 0.0, typ="bool")]
    onehot = lambda k: [1.0 if i == k else 0.0 for i in range(5)]  # noqa: E731
    preds = {"m": {"stand:a:tone": {"probs": onehot(3)}, "stand:b:tone": {"probs": onehot(2)},
                   "stand:a:faith": {"prob": 0.9}, "stand:b:faith": {"prob": 0.2}}}
    rep = sr.report(recs, preds, reps=0)
    m = rep["models"]["m"]["by_criterion"]["tone"]
    assert m["exact"] == 0.5 and m["within1"] == 1.0
    s4 = rep["models"]["m"]["s4"]["tone"]
    assert s4["rows"] == 1 and s4["model_vs_single_call_exact"] == pytest.approx(2 / 3)
    assert s4["judge_vs_judge_exact"] == pytest.approx(1 / 3)
    assert rep["constant"]["tone"]["level"] == 4 and rep["constant"]["tone"]["exact"] == 0.5
    assert rep["models"]["m"]["faithfulness"]["auroc"] == pytest.approx(1.0)
    assert "tone" in sr.to_markdown(rep)


def test_combine_renames_and_merges(tmp_path):
    def run(name, items, scores):
        d = tmp_path / name
        (d / "scores").mkdir(parents=True)
        (d / "items.jsonl").write_text("".join(json.dumps(i) + "\n" for i in items))
        for s in scores:
            (d / "scores" / f"{s['key']}.json").write_text(json.dumps(s))
        return d

    it = [{"id": "x", "task": "t1", "target": 1}]
    a = run("a", it, [{"key": "jevlite", "name": "jl", "tasks": {"t1": {"scores": {"x": 0.9}}}}])
    b = run("b", it, [{"key": "jevlite", "name": "jl", "tasks": {"t1": {"scores": {"x": 0.2}}}}])
    out = tmp_path / "all"
    cb.main([f"{a}:jevlite=jev A", f"{b}:jevlite=jev B", "--out", str(out)])
    assert sorted(p.name for p in (out / "scores").glob("*.json")) == ["jev_a.json", "jev_b.json"]
    assert json.loads((out / "scores" / "jev_b.json").read_text())["name"] == "jev B"
    bad = run("c", [{"id": "x", "task": "t1", "target": 0}], [])
    with pytest.raises(SystemExit):
        cb.main([str(a), str(bad), "--out", str(tmp_path / "bad")])
