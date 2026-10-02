import json

import numpy as np
import pytest

from jevlite.data.unified import Prediction, Record, write_jsonl
from jevlite.evaluation import evaluate, json_safe, parse_slices, to_markdown


def _records(n=60, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        src = "a" if i % 2 else "b"
        out.append(Record(id=f"b{i}", source=src, domain="d1", split="test_in", state="s", type="bool", prompt="p", target=float(rng.integers(0, 2)), criterion="faithfulness"))
        out.append(Record(id=f"c{i}", source=src, domain="d2", split="test_ood", state="s", type="choice", prompt="p", candidates=["x", "y", "z"], target=int(rng.integers(0, 3))))
        out.append(Record(id=f"s{i}", source=src, domain="d2", split="test_ood", state="s", type="score", prompt="p", candidates=["lo", "mid", "hi", "top"], target=int(rng.integers(0, 4))))
    return out


def _perfect(records):
    preds = {}
    for r in records:
        if r.type == "bool":
            preds[r.id] = Prediction(id=r.id, prob=float(r.target), latency_ms=10.0, cost_usd=0.001)
        else:
            p = np.zeros(r.n_candidates)
            p[int(r.target)] = 1.0
            preds[r.id] = Prediction(id=r.id, probs=p.tolist(), latency_ms=20.0)
    return preds


def _uniform(records):
    return {
        r.id: Prediction(id=r.id, prob=0.5) if r.type == "bool" else Prediction(id=r.id, probs=[1 / r.n_candidates] * r.n_candidates)
        for r in records
    }


def _row(report, prim, **slice_):
    (row,) = [r for r in report["rows"] if r["type"] == prim and r["slice"] == slice_]
    return row


def test_perfect_predictions():
    records = _records()
    report = evaluate(records, _perfect(records))
    assert report["n_evaluated"] == len(records) and report["n_missing"] == 0
    for prim in ("bool", "choice", "score"):
        row = _row(report, prim)
        assert row["accuracy"] == 1.0 and row["kappa"] == pytest.approx(1.0) and row["ece"] == pytest.approx(0.0)
    assert _row(report, "score")["spearman"] == pytest.approx(1.0)
    assert _row(report, "score")["mae"] == 0.0
    assert report["ops"]["latency_ms"]["p50"] in (10.0, 15.0, 20.0)
    assert report["ops"]["cost_usd"]["total"] == pytest.approx(0.06)


def test_uniform_predictions_have_no_agreement():
    records = _records()
    report = evaluate(records, _uniform(records))
    assert _row(report, "choice")["nll"] == pytest.approx(np.log(3))
    assert _row(report, "bool")["kappa"] == pytest.approx(0.0, abs=1e-9)


def test_slices():
    records = _records()
    report = evaluate(records, _perfect(records))
    assert _row(report, "bool", source="a")["n"] == 30
    assert _row(report, "bool", criterion="faithfulness")["n"] == 60
    assert _row(report, "choice", split="test_ood")["n"] == 60
    assert _row(report, "score", source="b", domain="d2")["n"] == 30
    # Records without the field are left out of that slice: choice has no criterion.
    assert not [r for r in report["rows"] if r["type"] == "choice" and "criterion" in r["slice"]]


def test_custom_slices_and_min_n():
    records = _records()
    report = evaluate(records, _perfect(records), slices=parse_slices("source*domain"), min_n=31)
    assert all(r["slice"] == {} for r in report["rows"])  # every source × domain cell has 30 < 31 examples
    assert parse_slices("a,b*c,a") == ((), ("a",), ("b", "c"))


def test_bootstrap_ci_brackets_the_estimate():
    records = _records(n=80)
    rng = np.random.default_rng(1)
    preds = {r.id: Prediction(id=r.id, prob=float(np.clip(r.target * 0.6 + rng.uniform(0, 0.4), 0, 1))) for r in records if r.type == "bool"}
    report = evaluate([r for r in records if r.type == "bool"], preds, slices=[()], n_boot=200)
    row = _row(report, "bool")
    for key in ("accuracy", "kappa", "ece"):
        lo, hi = row["ci95"][key]
        assert lo <= row[key] + 1e-9 and row[key] - 1e-9 <= hi


def test_missing_and_mismatched_predictions():
    records = _records(n=5)
    preds = _perfect(records)
    del preds["b0"]
    with pytest.raises(ValueError, match="no prediction"):
        evaluate(records, preds)
    assert evaluate(records, preds, allow_missing=True)["n_missing"] == 1
    preds["b0"] = Prediction(id="b0", probs=[0.5, 0.5])
    with pytest.raises(ValueError, match="needs 'prob'"):
        evaluate(records, preds)
    preds["b0"] = Prediction(id="b0", prob=0.5)
    preds["c0"] = Prediction(id="c0", probs=[0.5, 0.5])
    with pytest.raises(ValueError, match="3 candidates"):
        evaluate(records, preds)


def test_mixed_k_in_one_slice():
    recs = [
        Record(id="a", source="s", split="test_in", state="x", type="choice", prompt="p", candidates=["x", "y"], target=0),
        Record(id="b", source="s", split="test_in", state="x", type="choice", prompt="p", candidates=["x", "y", "z", "w"], target=[0, 0, 0.5, 0.5]),
    ]
    preds = {"a": Prediction(id="a", probs=[0.9, 0.1]), "b": Prediction(id="b", probs=[0.1, 0.1, 0.7, 0.1])}
    row = _row(evaluate(recs, preds, slices=[()]), "choice")
    assert row["accuracy"] == pytest.approx((1.0 + 0.5) / 2)


def test_markdown_and_json(tmp_path):
    records = _records(n=10)
    report = evaluate(records, _uniform(records), n_boot=20)
    md = to_markdown(report)
    assert "## bool — overall" in md and "## choice — source × domain" in md and "κ" in md
    text = json.dumps(json_safe(report), allow_nan=False)  # strict JSON: NaN → null
    assert json.loads(text)["n_records"] == 30


def test_eval_script(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("eval_script", Path(__file__).parents[1] / "scripts" / "eval.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    records = _records(n=10)
    write_jsonl(tmp_path / "data.jsonl", records)
    write_jsonl(tmp_path / "pred.jsonl", _perfect(records).values())
    script.main(["--data", str(tmp_path / "data.jsonl"), "--pred", str(tmp_path / "pred.jsonl"), "--out", str(tmp_path / "out"), "--splits", "test_ood"])
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert report["n_evaluated"] == 20
    assert (tmp_path / "out" / "report.md").read_text().startswith("# Evaluation: pred")
