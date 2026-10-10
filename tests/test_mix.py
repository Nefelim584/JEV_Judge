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


def test_no_share_alike_drops_sa_from_train_and_calib_only(tmp_path):
    rows = []
    for i in range(40):
        split = ["train", "calib", "test_in", "test_ood"][i % 4]
        rows.append(Record(id=f"sa:{i}", source="sa", split=split, state="s", type="bool", prompt="p", target=1.0))
    write_jsonl(tmp_path / "sa.jsonl", rows)
    _source(tmp_path, "free", 50, "bool")
    nli = [
        Record(id=f"multi_nli:{i}", source="multi_nli", domain="fiction" if i % 2 else "travel", split="train" if i < 20 else "calib",
               state="s", type="bool", prompt="p", target=0.0)
        for i in range(30)
    ]
    write_jsonl(tmp_path / "multi_nli.jsonl", nli)
    sources = {"sa": {"share_alike": True}, "free": {"share_alike": False}, "multi_nli": {}}  # multi_nli: from its converter

    full = build_pools(_cfg(tmp_path, sources, weights={"bool": 1.0}))
    nosa = build_pools(_cfg(tmp_path, sources, weights={"bool": 1.0}, share_alike=False))
    assert ("sa", "bool") in full.train and ("sa", "bool") not in nosa.train
    assert ("calib", "sa") not in nosa.eval and ("calib", "sa") in full.eval
    for split in ("test_in", "test_ood"):  # the same test sets in both versions
        assert nosa.eval[(split, "sa")] == full.eval[(split, "sa")]
    kept = [json.loads(x)["domain"] for x in nosa.train[("multi_nli", "bool")] + nosa.eval[("calib", "multi_nli")]]
    assert kept and set(kept) == {"travel"}  # only the fiction genre of MultiNLI is SA
    assert len(nosa.train[("free", "bool")]) == len(full.train[("free", "bool")])
    assert nosa.dropped_sa == {"sa": 20, "free": 0, "multi_nli": 15}
    report = mix_report(_cfg(tmp_path, sources, weights={"bool": 1.0}, share_alike=False), nosa, [])
    assert "No CC BY-SA data" in report_markdown(report)


def test_unknown_source_needs_explicit_share_alike(tmp_path):
    _source(tmp_path, "mystery", 10, "bool")
    with pytest.raises(ValueError, match="share_alike"):
        build_pools(_cfg(tmp_path, {"mystery": {}}, weights={"bool": 1.0}, share_alike=False))
    build_pools(_cfg(tmp_path, {"mystery": {}}, weights={"bool": 1.0}))  # with SA allowed the flag is not needed


def test_extends_merges_sources(tmp_path):
    from jevlite.data.mix import load_mix_config

    (tmp_path / "base.yaml").write_text(
        "dir: d\ntotal: 10\nprimitive_weights: {bool: 1.0}\nsources:\n  a: {cap: 5}\n  b: {cap: 7, fit_only: true}\n"
    )
    (tmp_path / "child.yaml").write_text("extends: base.yaml\nshare_alike: false\nsources:\n  b: {cap: null}\n")
    cfg = load_mix_config(tmp_path / "child.yaml")
    assert cfg.name == "child" and cfg.share_alike is False and cfg.total == 10
    specs = {s.name: s for s in cfg.sources}
    assert specs["a"].cap == 5 and specs["b"].cap is None and specs["b"].fit_only is True


def test_real_mix_configs_load():
    from pathlib import Path

    from jevlite.data.mix import load_mix_config

    mixes = Path(__file__).resolve().parents[1] / "configs" / "mixes"
    full, nosa = load_mix_config(mixes / "stage_a.yaml"), load_mix_config(mixes / "stage_a_nosa.yaml")
    assert full.share_alike and not nosa.share_alike
    assert {s.name for s in full.sources} == {s.name for s in nosa.sources}
    assert {s.name: s.cap for s in nosa.sources}["wanli"] is None


def test_run_repeat_caps_repeats_over_the_whole_run(data):
    from jevlite.data.mix import suggest_loss_weights

    sources = {"b1": {}, "c1": {}, "s1": {}}
    plain = _cfg(data, sources, total=200)
    capped = _cfg(data, sources, total=200, run_repeat={"score": 1.0})
    pools = build_pools(capped)

    # without the cap Score (pool 40, quota 60) repeats 1.5× per epoch
    assert {a.source: a.drawn for a in allocate(plain, pools, epochs=2)}["s1"] == 60
    allocations = allocate(capped, pools, epochs=2)
    drawn = {a.source: a.drawn for a in allocations}
    assert drawn == {"b1": 100, "c1": 40, "s1": 20}  # Score shrinks to 40 / 2 epochs, the others keep their quota

    seen = Counter()
    for epoch in range(2):
        lines = epoch_lines(capped, pools, allocations, epoch)
        assert len(lines) == 160
        seen.update(line for line in lines if json.loads(line)["type"] == "score")
    assert len(seen) == 40 and set(seen.values()) == {1}  # every Score record exactly once over the run
    assert epoch_lines(capped, pools, allocations, 1) == epoch_lines(capped, pools, allocations, 1)

    report = mix_report(capped, pools, allocations, epochs=2)
    assert report["total"] == 160 and report["nominal_total"] == 200
    assert {r["source"]: r["run_repeat"] for r in report["train"]}["s1"] == 1.0
    assert "Run-level repeat cap" in report_markdown(report)

    lw = suggest_loss_weights(capped, pools, epochs=2, batch_size=8)
    assert lw["presence"]["score"] < lw["presence_nominal"]["score"]
    assert lw["weights"]["score"] > 1.0


def test_run_repeat_validation(tmp_path):
    with pytest.raises(ValueError, match="run_repeat"):
        _cfg(tmp_path, {"b1": {}}, weights={"bool": 1.0}, run_repeat={"rank": 1.0})
    with pytest.raises(ValueError, match="run_repeat"):
        _cfg(tmp_path, {"b1": {}}, weights={"bool": 1.0}, run_repeat={"bool": 0})
