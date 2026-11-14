import math

import numpy as np
import pytest

from jevlite.baselines.criteria import (
    GraniteGuardianScorer, lettuce_prompt, p_yes, p_yes_from_generation, single_token_ids,
)
from jevlite.criteria_bench import (
    Item, aegis_items, best_threshold, custom_items, grounding_items, in_calib, report_markdown, summarize,
    summarize_by_slice, toxigen_items, toxigen_label,
)


def _halu(i, source, label="PASS", answer="Paris"):
    return {"passage": f"passage {i}", "question": f"q{i}?", "answer": answer, "label": label, "source_ds": source}


def test_grounding_items_cap_per_source_and_claim():
    rows = [_halu(i, "halueval") for i in range(10)] + [_halu(10, "RAGTruth", "FAIL", "['a', 'b']"),
                                                        {"passage": "", "question": "q", "answer": "x", "label": "PASS"}]
    items = grounding_items(rows, per_source=3, seed=1)
    by = {}
    for it in items:
        by.setdefault(it.slice, []).append(it)
    assert len(by["halueval"]) == 3 and len(by["ragtruth"]) == 1
    rt = by["ragtruth"][0]
    assert rt.target == 0.0 and rt.answer == "a, b"
    assert rt.prompt == 'The answer to the question "q10?" is: a, b'
    assert rt.context == rt.state == "passage 10"
    assert grounding_items(rows, 3, seed=1) == items  # the draw is seeded
    assert len(grounding_items(rows, None)) == 11


def test_aegis_items_keep_rows_with_a_labelled_response():
    rows = [
        {"id": "a", "prompt": "hi", "response": "bad words", "response_label": "unsafe", "response_label_source": "human"},
        {"id": "b", "prompt": "hi", "response": "hello", "response_label": "safe", "response_label_source": "llm_jury"},
        {"id": "c", "prompt": "hi", "response": None, "response_label": None},
        {"id": "d", "prompt": "hi", "response": "x", "response_label": None},
    ]
    items = aegis_items(rows)
    assert [it.id for it in items] == ["aegis2/a", "aegis2/b"]
    assert [it.target for it in items] == [1.0, 0.0]
    assert items[0].state == "User: hi\n\nAssistant: bad words" and items[0].answer == "bad words"
    assert items[1].slice == "all"


def test_toxigen_label_rule():
    assert toxigen_label({"toxicity_ai": 3.0, "toxicity_human": 3.0}) == 1.0
    assert toxigen_label({"toxicity_ai": 2.5, "toxicity_human": 3.0}) == 0.0
    items = toxigen_items([{"text": "t", "toxicity_ai": 5, "toxicity_human": 5, "target_group": "women"}, {"text": " "}])
    assert len(items) == 1 and items[0].target == 1.0 and items[0].slice == "women"


def test_custom_items_from_bool_records():
    recs = [{"id": "r1", "source": "own", "split": "test_in", "state": {"answer": "Buy it"}, "type": "bool",
             "prompt": "The reply gives personal investment advice.", "target": 1.0, "answer": "Buy it", "slice": "advice"}]
    (it,) = custom_items(recs, "advice")
    assert it.task == "custom:advice" and it.answer == "Buy it" and it.slice == "advice"
    assert it.state == '{"answer":"Buy it"}'
    with pytest.raises(ValueError, match="Bool"):
        custom_items([{**recs[0], "type": "choice", "candidates": ["a", "b"], "target": 0}], "x")


def test_calib_fold_is_stable_and_near_the_fraction():
    ids = [f"x{i}" for i in range(4000)]
    share = np.mean([in_calib(i, 0.3) for i in ids])
    assert 0.27 < share < 0.33
    assert [in_calib(i, 0.3) for i in ids[:50]] == [in_calib(i, 0.3) for i in ids[:50]]
    assert [in_calib(i, 0.3, seed=1) for i in ids[:200]] != [in_calib(i, 0.3) for i in ids[:200]]


def test_best_threshold():
    t = best_threshold([0.1, 0.2, 0.3, 0.8, 0.9], [0, 0, 0, 1, 1])
    assert 0.3 < t < 0.8
    assert best_threshold([0.4, 0.6], [1, 1]) == 0.5


def _items(n, task="t"):
    return [Item(id=f"i{k}", task=task, target=float(k % 2), state="s", prompt="p", slice="a" if k < n // 2 else "b")
            for k in range(n)]


def test_summarize_perfect_scorer_and_missing():
    items = _items(200)
    scores = {it.id: 0.9 if it.target else 0.1 for it in items}
    del scores["i0"]
    m = summarize(items, scores, calib_frac=0.3)
    assert m["n_missing"] == 1 and m["auroc"] == 1.0 and m["balanced_accuracy@fit"] == 1.0
    assert m["n_calib"] + m["n_test"] == 199
    s = summarize_by_slice(items, scores, min_n=30, calib_frac=0.3)
    assert set(s) == {"all", "a", "b"}


def test_summarize_inverted_scorer():
    items = _items(100)
    m = summarize(items, {it.id: 0.1 if it.target else 0.9 for it in items})
    assert m["auroc"] == 0.0 and m["balanced_accuracy@0.5"] == 0.0


def test_report_markdown_lists_scorers_slices_and_errors():
    items = _items(100)
    good = {it.id: 0.9 if it.target else 0.1 for it in items}
    results = {
        "good": {"tasks": {"t": {"metrics": summarize_by_slice(items, good, min_n=10), "timing": {"batched_ms_per_item": 2.0}}}},
        "broken": {"error": "OOM", "tasks": {}},
    }
    md = report_markdown(results, {"t": items}, 0.3)
    assert "| good | 1.000" in md and "AUROC by slice" in md and "**broken**: `OOM`" in md


def test_p_yes_and_generation_readout():
    z = np.zeros(10)
    z[3], z[4] = 2.0, 0.0
    assert math.isclose(p_yes(z, [3], [4]), 1 / (1 + math.exp(-2)))
    steps = [np.zeros(10), z]
    assert math.isclose(p_yes_from_generation([7, 3], steps, [3], [4]), p_yes(z, [3], [4]))
    assert math.isnan(p_yes_from_generation([7, 8], steps, [3], [4]))


class _Tok:
    vocab = {"yes": 1, " yes": 2, "Yes": 3}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[text]] if text in self.vocab else [9, 9]


def test_single_token_ids():
    assert single_token_ids(_Tok(), ["yes"]) == [1, 2, 3]
    with pytest.raises(ValueError):
        single_token_ids(_Tok(), ["no"])


def test_granite_inputs_polarity_and_roles():
    g = Item(id="g", task="grounding", target=1, state="ctx", prompt="c", context="ctx", question="q", answer="a")
    msgs, cfg, docs, invert = GraniteGuardianScorer.inputs(g)
    assert cfg == {"criteria_id": "groundedness"} and invert and docs[0]["text"] == "ctx"
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    h = Item(id="h", task="answer_harm", target=1, state="s", prompt="p", user="u", answer="a")
    assert GraniteGuardianScorer.inputs(h)[1:] == ({"criteria_id": "harm"}, None, False)
    t = Item(id="t", task="toxicity", target=1, state="text", prompt="p")
    msgs, cfg, _, invert = GraniteGuardianScorer.inputs(t)
    assert cfg == {"criteria_id": "social_bias"} and msgs == [{"role": "user", "content": "text"}] and not invert
    c = Item(id="c", task="custom:advice", target=1, state="s", prompt="The reply gives advice.", answer="Buy")
    msgs, cfg, _, invert = GraniteGuardianScorer.inputs(c)
    assert cfg == {"custom_criteria": "The reply gives advice."} and msgs[-1] == {"role": "assistant", "content": "Buy"}


def test_lettuce_prompt_has_question_and_passage():
    p = lettuce_prompt("Paris is in France.", "Where is Paris?")
    assert "Where is Paris?" in p and "passage 1: Paris is in France." in p and p.endswith("output:")


def test_split_on_oom_halves_until_items_fit():
    import torch

    from jevlite.baselines.criteria import split_on_oom

    calls = []

    def fn(batch):
        calls.append(len(batch))
        if len(batch) > 2 or "huge" in batch:
            raise torch.cuda.OutOfMemoryError("oom")
        return [1.0] * len(batch)

    out = split_on_oom(fn, ["a", "b", "c", "d", "e", "huge"])
    assert out[:5] == [1.0] * 5 and math.isnan(out[5])
    assert calls[0] == 6


def test_resume_reruns_failed_tasks_and_merges(tmp_path):
    pytest.importorskip("modal")
    import importlib.util
    import json

    spec = importlib.util.spec_from_file_location("mcb", "scripts/modal_criteria_bench.py")
    mcb = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mcb)
    tasks = ["grounding", "answer_harm", "toxicity"]
    saved = {"key": "g", "name": "G", "tasks": {"answer_harm": {"scores": {"x": 0.5}}}, "errors": {"grounding": "OOM"}}
    path = tmp_path / "g.json"
    path.write_text(json.dumps(saved))
    assert mcb.tasks_to_run(path, tasks) == ["grounding"]
    assert mcb.tasks_to_run(tmp_path / "none.json", tasks) == tasks
    path.write_text(json.dumps({"key": "g", "error": "load failed", "tasks": {}}))
    assert mcb.tasks_to_run(path, tasks) == tasks
    merged = mcb.merge_scores(saved, {"key": "g", "name": "G", "tasks": {"grounding": {"scores": {"y": 0.1}}}, "errors": {}})
    assert set(merged["tasks"]) == {"answer_harm", "grounding"} and merged["errors"] == {}
    assert mcb.merge_scores(saved, {"key": "g", "error": "boom", "tasks": {}}) == saved


def test_ood_loaders():
    from jevlite.criteria_bench import beavertails_items, oai_moderation_items, toxic_chat_items

    bt = beavertails_items([{"prompt": "p", "response": "r", "is_safe": False}, {"prompt": "p", "response": "", "is_safe": True}])
    assert len(bt) == 1 and bt[0].target == 1.0 and bt[0].answer == "r"
    assert GraniteGuardianScorer.inputs(bt[0])[1] == {"criteria_id": "harm"}
    assert [m["role"] for m in GraniteGuardianScorer.inputs(bt[0])[0]] == ["user", "assistant"]
    tc = toxic_chat_items([{"conv_id": "a", "user_input": "x", "toxicity": 1, "human_annotation": True},
                           {"conv_id": "b", "user_input": "y", "toxicity": 0, "human_annotation": False}])
    assert [it.id for it in tc] == ["toxic_chat/a"] and tc[0].target == 1.0
    om = oai_moderation_items([{"prompt": "t", "S": 0, "H": 0, "V": 0, "HR": 0, "SH": 1, "S3": 0, "H2": 0, "V2": 0}])
    assert om[0].target == 1.0 and GraniteGuardianScorer.inputs(om[0])[0] == [{"role": "user", "content": "t"}]
    many = beavertails_items([{"prompt": f"p{i}", "response": "r", "is_safe": True} for i in range(50)], n=10, seed=1)
    assert len(many) == 10
