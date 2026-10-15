import json

import pytest

from jevlite.data.overlap import PREFIX_CHARS, eval_subset_ids, ood_kind, overlaps, passage_keys, passages, train_domains, train_index
from jevlite.train import eval_subset


def test_passages_of_every_state_shape():
    assert passages("  A  b\n c ") == ["a b c"]
    assert passages(["One", "", "Two"]) == ["one", "two"]
    assert passages({"table": "T", "n": 3}) == ["t"]


def test_exact_and_prefix_keys():
    long = "x" * PREFIX_CHARS
    a, b = long + " response one", long + " response two"
    assert passage_keys(a, "exact").isdisjoint(passage_keys(b, "exact"))
    assert not passage_keys(a, "prefix").isdisjoint(passage_keys(b, "prefix"))
    assert passage_keys("short", "prefix") == passage_keys("short", "exact")  # no prefix key below PREFIX_CHARS
    with pytest.raises(ValueError):
        passage_keys("s", "fuzzy")


def test_overlap_against_train_files(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text("\n".join(json.dumps({"state": s}) for s in ["Shared  passage", ["chunk one", "chunk two"]]) + "\n")
    idx = train_index([train], "exact")
    assert overlaps({"state": "shared passage"}, idx, "exact")  # whitespace and case do not matter
    assert overlaps({"state": ["new chunk", "Chunk two"]}, idx, "exact")  # one shared chunk is enough
    assert not overlaps({"state": "unseen"}, idx, "exact")


def test_eval_subset_ids_match_training():
    rows = [{"id": f"r{i}"} for i in range(50)]

    class R:
        def __init__(self, id):
            self.id = id

    assert eval_subset_ids(rows, 10, 42) == {r.id for r in eval_subset([R(r["id"]) for r in rows], 10, 42)}
    assert eval_subset_ids(rows, 100, 42) == {r["id"] for r in rows}


def test_ood_kind_by_source_and_domain(tmp_path):
    train = tmp_path / "train.jsonl"
    train.write_text("\n".join(json.dumps({"source": "multi_nli", "domain": d, "state": "s"}) for d in ["fiction", "travel"]) + "\n")
    domains = train_domains([train])
    assert ood_kind({"source": "multi_nli", "domain": "fiction"}, domains) == "test_ood_template"
    assert ood_kind({"source": "multi_nli", "domain": "letters"}, domains) == "test_ood_source"  # mismatched genre
    assert ood_kind({"source": "contract_nli", "domain": "legal"}, domains) == "test_ood_source"


def test_clean_eval_burns_the_subset_of_the_file_the_run_saw(tmp_path):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("clean_eval", Path(__file__).parents[1] / "scripts" / "clean_eval.py")
    clean_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(clean_eval)

    def rows(src, n):
        return [{"id": f"{src}:{i}", "source": src, "domain": "d", "split": "test_ood", "state": f"{src} text {i}"} for i in range(n)]

    old, new = rows("a", 40), rows("a", 40) + rows("added", 40)
    for name, rs in (("run.jsonl", old), ("test_ood.jsonl", new), ("train.jsonl", [{"source": "a", "domain": "d", "state": "x"}])):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rs))
    out = tmp_path / "out"
    clean_eval.main(["--train", str(tmp_path / "train.jsonl"), "--data", str(tmp_path / "test_ood.jsonl"), "--out", str(out),
                     "--drop-eval-subset", "test_ood", "--eval-max-records", "10", "--eval-subset-ref", f"test_ood={tmp_path / 'run.jsonl'}"])
    kept = [json.loads(line) for line in (out / "test_ood.jsonl").open()]
    assert len(kept) == 70 and all(r["id"] not in eval_subset_ids(old, 10, 42) for r in kept)
    assert sum(r["source"] == "added" for r in kept) == 40  # records added after the run are never burned
