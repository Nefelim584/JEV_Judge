import json
from collections import Counter

import pytest

from jevlite.data.mix import MixConfig, allocate, build_pools, epoch_lines, eval_lines, mix_report, report_markdown
from jevlite.data.unified import Record, write_jsonl


def _source(tmp_path, name, n_train, prim="bool", n_test=0, long_every=0):
    rows = []
    for i in range(n_train + n_test):
        split = "train" if i < n_train else "test_in"
        state = "long " * 50 if long_every and i % long_every == 0 else "s"
        if prim == "bool":
            rows.append(Record(id=f"{name}:{i}", source=name, split=split, state=state, type="bool", prompt="p", target=1.0))
        else:
            rows.append(Record(id=f"{name}:{i}", source=name, split=split, state=state, type=prim, prompt="p", candidates=["a", "b"], target=0))
    write_jsonl(tmp_path / f"{name}.jsonl", rows)


def _cfg(tmp_path, sources, total=100, weights=None, **kw):
    return MixConfig.from_dict({
        "dir": str(tmp_path), "total": total, "primitive_weights": weights or {"bool": 0.5, "choice": 0.2, "score": 0.3},
        "sources": sources, **kw,
    })


@pytest.fixture
def data(tmp_path):
    _source(tmp_path, "b1", 300, "bool", n_test=50)
    _source(tmp_path, "b2", 100, "bool")
    _source(tmp_path, "c1", 500, "choice", n_test=20)
    _source(tmp_path, "s1", 40, "score", long_every=4)
    return tmp_path


def test_config_validation(tmp_path):
    with pytest.raises(ValueError, match="sum to 1"):
        _cfg(tmp_path, {"b1": {}}, weights={"bool": 0.5, "choice": 0.2, "score": 0.2})
    with pytest.raises(ValueError, match="primitive_weights"):
        _cfg(tmp_path, {"b1": {}}, weights={"bool": 0.5, "rank": 0.5})


def test_caps_are_stable_and_nested(data):
    small = build_pools(_cfg(data, {"b1": {"cap": 50}}, weights={"bool": 1.0}))
    large = build_pools(_cfg(data, {"b1": {"cap": 120}}, weights={"bool": 1.0}))
    again = build_pools(_cfg(data, {"b1": {"cap": 50}}, weights={"bool": 1.0}))
    s, l = set(small.train[("b1", "bool")]), set(large.train[("b1", "bool")])
    assert len(s) == 50 and len(l) == 120 and s < l
    assert small.train == again.train


def test_fit_only_filters_train_and_eval(data):
    fits = lambda row: not row["state"].startswith("long")
    pools = build_pools(_cfg(data, {"s1": {"fit_only": True}}, weights={"score": 1.0}), fits=fits)
    assert len(pools.train[("s1", "score")]) == 30 and pools.dropped_unfit["s1"] == 10
    assert all(not json.loads(line)["state"].startswith("long") for line in pools.train[("s1", "score")])


def test_allocation_follows_weights_and_pool_sizes(data):
    cfg = _cfg(data, {"b1": {}, "b2": {}, "c1": {}, "s1": {}}, total=100)
    allocations = allocate(cfg, build_pools(cfg))
    drawn = {(a.source, a.primitive): a.drawn for a in allocations}
    assert sum(drawn.values()) == 100
    assert drawn[("b1", "bool")] + drawn[("b2", "bool")] == 50 and drawn[("c1", "choice")] == 20 and drawn[("s1", "score")] == 30
    assert drawn[("b1", "bool")] == pytest.approx(37.5, abs=1)  # pools 300 vs 100: 3/4 of 50


def test_source_weight_shifts_share(data):
    cfg = _cfg(data, {"b1": {}, "b2": {"weight": 3.0}}, total=100, weights={"bool": 1.0})
    drawn = {a.source: a.drawn for a in allocate(cfg, build_pools(cfg))}
    assert drawn == {"b1": 50, "b2": 50}


def test_max_repeat_and_missing_primitive(data):
    cfg = _cfg(data, {"b1": {}, "c1": {}, "s1": {}}, total=1000)  # score quota 300 from a pool of 40
    with pytest.raises(ValueError, match="max_repeat"):
        allocate(cfg, build_pools(cfg))
    cfg = _cfg(data, {"b1": {}, "c1": {}}, total=100)
    with pytest.raises(ValueError, match="no training records"):
        allocate(cfg, build_pools(cfg))


def test_epochs_have_fixed_counts_and_fresh_subsets(data):
    cfg = _cfg(data, {"b1": {}, "c1": {}, "s1": {}}, total=200, max_repeat=2.0)  # score: 60 from 40 = 1.5×
    pools = build_pools(cfg)
    allocations = allocate(cfg, pools)
    e0, e1 = epoch_lines(cfg, pools, allocations, 0), epoch_lines(cfg, pools, allocations, 1)
    assert len(e0) == len(e1) == 200
    assert e0 == epoch_lines(cfg, pools, allocations, 0)
    types = lambda lines: Counter(json.loads(x)["type"] for x in lines)
    assert types(e0) == types(e1) == {"bool": 100, "choice": 40, "score": 60}
    bool0 = {x for x in e0 if '"type":"bool"' in x.replace(" ", "")}
    bool1 = {x for x in e1 if '"type":"bool"' in x.replace(" ", "")}
    assert bool0 != bool1  # subsampled pool: a fresh draw per epoch
    score = Counter(x for x in e0 if '"type":"score"' in x.replace(" ", ""))
    assert len(score) == 40 and set(score.values()) <= {1, 2}  # every record once, 20 of them twice


def test_eval_caps_and_report(data):
    cfg = _cfg(data, {"b1": {}, "c1": {"eval_cap": 5}, "s1": {}}, total=100, eval_cap=30)
    pools = build_pools(cfg)
    assert len(eval_lines(pools, "test_in")) == 30 + 5
    report = mix_report(cfg, pools, allocate(cfg, pools))
    assert report["eval"]["test_in"] == {"b1": 30, "c1": 5}
    assert sum(r["drawn"] for r in report["train"]) == 100
    assert "| bool |" in report_markdown(report)
