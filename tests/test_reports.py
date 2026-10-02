import json

import numpy as np
import pytest

pytest.importorskip("matplotlib")
pytest.importorskip("plotly")

from jevlite.data.unified import Prediction, Record  # noqa: E402
from jevlite.evaluation import evaluate, json_safe  # noqa: E402
from jevlite.reports.compare import RunSpec, build_report, load_runs, parse_run  # noqa: E402
from jevlite.reports.style import MAX_SERIES, series_colors  # noqa: E402
from jevlite.reports.training import TrainingLog, ema, read_log  # noqa: E402
from jevlite.reports.training import build_report as build_training_report  # noqa: E402


def _records(n=40):
    rng = np.random.default_rng(0)
    recs = []
    for i in range(n):
        dom = ["faq", "docs"][i % 2]
        recs.append(Record(id=f"b{i}", source="syn", domain=dom, split="test_in", state="s", type="bool", prompt="p", target=float(rng.integers(0, 2)), criterion="faithfulness"))
        recs.append(Record(id=f"c{i}", source="syn", domain=dom, split="test_in", state="s", type="choice", prompt="p", candidates=["x", "y", "z"], target=int(rng.integers(0, 3))))
    return recs


def _report(tmp_path, name, skill, seed):
    """An eval report where predictions put ``skill`` extra mass on the right answer."""
    rng = np.random.default_rng(seed)
    records = _records()
    preds = {}
    for r in records:
        if r.type == "bool":
            p = np.clip(0.5 + (skill if r.target else -skill) + rng.normal(0, 0.15), 0.01, 0.99)
            preds[r.id] = Prediction(id=r.id, prob=float(p), latency_ms=5.0 + seed)
        else:
            p = rng.dirichlet([1, 1, 1]) + np.eye(3)[int(r.target)] * skill * 3
            preds[r.id] = Prediction(id=r.id, probs=(p / p.sum()).tolist(), latency_ms=5.0 + seed)
    path = tmp_path / f"{name}-{seed}.json"
    path.write_text(json.dumps(json_safe(evaluate(records, preds))))
    return path


def test_parse_run():
    assert parse_run("nli=a/b.json") == RunSpec("nli", __import__("pathlib").Path("a/b.json"))
    spec = parse_run("synthetic@1e3=r.json")
    assert spec.name == "synthetic" and spec.x == 1000.0
    with pytest.raises(ValueError):
        parse_run("no-path")


def test_series_colors_fixed_order_and_cap():
    colors = series_colors(["a", "b"])
    assert colors == {"a": "#2a78d6", "b": "#eb6834"}
    with pytest.raises(ValueError, match="at most"):
        series_colors([str(i) for i in range(MAX_SERIES + 1)])


def test_seeds_are_aggregated(tmp_path):
    specs = [RunSpec("good", _report(tmp_path, "good", 0.3, s)) for s in (1, 2, 3)] + [RunSpec("weak", _report(tmp_path, "weak", 0.05, 1))]
    runs = load_runs(specs)
    assert [(r.name, r.n_seeds) for r in runs] == [("good", 3), ("weak", 1)]
    mean, lo, hi = runs[0].stat("bool", "accuracy")
    assert lo <= mean <= hi and runs[0].stat("bool", "accuracy")[0] > runs[1].stat("bool", "accuracy")[0]


def test_compare_report(tmp_path):
    specs = [RunSpec("good", _report(tmp_path, "good", 0.3, s)) for s in (1, 2)] + [RunSpec("weak", _report(tmp_path, "weak", 0.05, 1))]
    path = build_report(specs, tmp_path / "out", "Test comparison")
    md = path.read_text()
    for section in ("## Overall", "### bool", "### Δ vs reference", "## Calibration", "## Selective prediction", "## Cost and latency", "## Slices", "## How to read"):
        assert section in md, section
    assert "**" in md  # best values are bold
    figs = tmp_path / "out" / "figures"
    for name in ("overall", "reliability", "risk_coverage", "slice_domain"):
        assert (figs / f"{name}.png").stat().st_size > 1000
        assert (figs / f"{name}.html").exists()
        assert f"figures/{name}.png" in md
    assert (figs / "plotly.min.js").exists()


def test_learning_curves(tmp_path):
    specs = []
    for x, skill in [(100, 0.05), (1000, 0.15), (10000, 0.3)]:
        specs += [RunSpec("synthetic", _report(tmp_path, f"s{x}", skill, s), x) for s in (1, 2)]
        specs.append(RunSpec("human", _report(tmp_path, f"h{x}", skill + 0.05, 1), x))
    md = build_report(specs, tmp_path / "lc").read_text()
    assert "## Learning curves" in md and "| synthetic | 1000 | 2 |" in md
    assert (tmp_path / "lc" / "figures" / "learning_curves.png").exists()


def test_training_log_and_report(tmp_path):
    for name, speed in [("a", 1.0), ("b", 2.0)]:
        log = TrainingLog(tmp_path / name, run=name, meta={"stack": "cuda", "config": {"lr": 1e-4}})
        for step in range(10, 501, 10):
            log.log("train", step, lr=1e-4, loss=float(np.exp(-speed * step / 200)), **{"loss/bool": 0.5})
            if step % 100 == 0:
                log.log("eval", step, split="test_in", loss=float(np.exp(-step / 300)), **{"accuracy/bool": step / 600})
    meta, rows = read_log(tmp_path / "a")
    assert meta["run"] == "a" and meta["stack"] == "cuda"
    assert rows[0] == {"phase": "train", "step": 10, "lr": 1e-4, "loss": pytest.approx(np.exp(-0.05)), "loss/bool": 0.5}
    assert any(r.get("split") == "test_in" for r in rows)

    md = build_training_report({"a": tmp_path / "a", "b": tmp_path / "b" / "metrics.jsonl"}, tmp_path / "rep").read_text()
    assert "## Training curves" in md and "## Evaluation — test_in" in md and "| accuracy/bool |" in md
    for name in ("train_loss", "train_lr", "eval_test_in_loss", "eval_test_in_accuracy"):
        assert (tmp_path / "rep" / "figures" / f"{name}.png").exists(), name


def test_training_log_rejects_bad_phase(tmp_path):
    with pytest.raises(ValueError):
        TrainingLog(tmp_path).log("test", 1, loss=1.0)


def test_ema():
    assert ema([1.0, 1.0, 1.0], 0.9) == pytest.approx([1.0, 1.0, 1.0])  # bias-corrected: no warm-up dip
    assert ema([0.0, 10.0], 0.0) == [0.0, 10.0]
