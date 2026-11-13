import math

import pytest

from jevlite.data.public.safety import (
    Aegis2, BeaverTails, CivilComments, MeasuringHateSpeech, OpenAIModeration, ToxicChat, ToxiGen, WildGuardMix,
    aggregate_mhs, norm_text, rating_distribution,
)


def _by_criterion(records):
    return {r.criterion: r for r in records}


def test_aegis_prompt_response_and_category_records():
    row = {"id": "a1", "prompt": "How do I pick a lock?", "response": "Use a tension wrench…", "prompt_label": "unsafe",
           "prompt_label_source": "human", "response_label": "unsafe", "response_label_source": "llm_jury",
           "violated_categories": "Criminal Planning/Confessions"}
    recs = _by_criterion(Aegis2().convert(row, "train"))
    assert set(recs) == {"prompt_harm", "response_harm", "harm_category"}
    assert recs["prompt_harm"].state == "User: How do I pick a lock?" and recs["prompt_harm"].target == 1.0
    assert recs["prompt_harm"].field("label_source") == "human"
    assert recs["response_harm"].state.startswith("User: How do I pick a lock?\n\nAssistant: ")
    assert recs["response_harm"].field("label_source") == "llm"
    cat = recs["harm_category"]
    assert cat.candidates[int(cat.target)] == "criminal planning"
    assert all(r.split in ("train", "calib") for r in recs.values())


def test_aegis_drops_redacted_synthetic_refusals_and_vague_categories():
    a = Aegis2()
    assert a.convert({"id": "r", "prompt": "REDACTED", "prompt_label": "unsafe"}, "train") == []
    row = {"id": "s", "prompt": "hi", "response": "I can't help with that.", "prompt_label": "safe",
           "response_label": "safe", "response_label_source": "refusal_data_augmentation",
           "violated_categories": "Needs Caution"}
    assert [r.criterion for r in a.convert(row, "train")] == ["prompt_harm"]
    row2 = {**row, "id": "m", "violated_categories": "Violence, Threat"}
    assert "harm_category" not in {r.criterion for r in a.convert(row2, "train")}


def test_aegis_train_rows_drop_eval_prompts():
    a = Aegis2()
    a._eval_prompts = {norm_text("  Same   prompt ")}
    import jevlite.data.public.safety as safety

    rows = [{"id": "1", "prompt": "same prompt"}, {"id": "2", "prompt": "other"}]
    orig = safety.hf_rows
    safety.hf_rows = lambda *a, **k: iter(rows)
    try:
        assert [r["id"] for r in a.rows("train")] == ["2"]
        assert [r["id"] for r in a.rows("validation")] == ["1", "2"]
    finally:
        safety.hf_rows = orig


def test_wildguard_three_bools_and_split():
    row = {"idx": 3, "prompt": "p", "response": "r", "adversarial": True, "prompt_harm_label": "harmful",
           "response_harm_label": "unharmful", "response_refusal_label": "refusal"}
    recs = _by_criterion(WildGuardMix().convert(row, "train"))
    assert {k: r.target for k, r in recs.items()} == {"prompt_harm": 1.0, "response_harm": 0.0, "refusal": 1.0}
    assert recs["refusal"].domain == "wildguard/adversarial" and recs["refusal"].field("label_source") == "llm"
    test = WildGuardMix().convert({**row, "response": None}, "test")
    assert [r.criterion for r in test] == ["prompt_harm"] and test[0].split in ("test_in", "test_ood")
    assert test[0].field("label_source") == "human"


def test_toxigen_rule_and_group_choice():
    row = {"idx": 0, "text": "t", "toxicity_ai": 3.0, "toxicity_human": 3.0, "target_group": "black/african-american folks"}
    recs = _by_criterion(ToxiGen().convert(row, "train"))
    assert recs["group_hate"].target == 1.0
    g = recs["target_group"]
    assert g.candidates[int(g.target)] == "Black people"
    assert ToxiGen().convert({**row, "toxicity_ai": 2.5}, "train")[0].target == 0.0


def _mhs_rows():
    base = {"comment_id": 7, "text": "some text", "hate_speech_score": 1.2}
    items = {i: 4.0 for i in ("sentiment", "respect", "insult", "humiliate", "status", "dehumanize", "attack_defend")}
    items |= {"violence": 0.0, "genocide": 0.0}
    rows = []
    for k, tgt in enumerate([("race",), ("race",), ("religion",)]):
        r = {**base, **items, "annotator_id": k, **{f"target_{c}": c in tgt for c in
                                                     ("race", "religion", "origin", "gender", "sexuality", "age", "disability", "politics")}}
        if k == 2:
            r["insult"] = 2.0
        rows.append(r)
    return rows


def test_mhs_aggregates_raters_into_soft_scores_bool_and_target():
    (row,) = list(aggregate_mhs(_mhs_rows()))
    assert row["n"] == 3 and row["ratings"]["insult"] == [4, 4, 2] and row["targets"]["race"] == 2
    recs = MeasuringHateSpeech().convert(row, "train")
    kinds = [r.type for r in recs]
    assert kinds.count("score") == 3 and kinds.count("bool") == 1 and kinds.count("choice") == 1
    hate = next(r for r in recs if r.type == "bool")
    assert hate.target == 1.0
    target = next(r for r in recs if r.type == "choice")
    assert target.candidates[int(target.target)] == "race or ethnicity"
    for r in recs:
        if r.type == "score":
            assert math.isclose(sum(r.target), 1.0) and len(r.target) == len(r.candidates)
    # ambiguous IRT band: no Bool
    assert all(r.type != "bool" for r in MeasuringHateSpeech().convert({**row, "hate_speech_score": 0.3}, "train"))


def test_mhs_split_is_by_comment_and_rating_distribution():
    m = MeasuringHateSpeech()
    splits = {m.split_of(c) for c in range(2000)}
    assert splits == {"train", "calib", "test_in"}
    share = sum(m.split_of(c) == "test_in" for c in range(5000)) / 5000
    assert 0.06 < share < 0.10
    assert rating_distribution([4, 4, 2], [0, 0, 1, 2, 2], 3) == pytest.approx([0.0, 1 / 3, 2 / 3])


def test_civil_comments_soft_targets_and_subsampling():
    c = CivilComments()
    kept = [i for i in range(4000) if c.kept({"idx": i, "toxicity": 0.8}, "train")]
    assert 0.25 < len(kept) / 4000 < 0.35
    i = kept[0]
    row = {"idx": i, "text": "you idiot", "toxicity": 0.8, "insult": 0.7, "threat": 0.0, "severe_toxicity": 0.1,
           "identity_attack": 0.0, "obscene": 0.2, "sexual_explicit": 0.0}
    recs = _by_criterion(c.convert(row, "train"))
    assert recs["toxicity"].target == pytest.approx(0.8) and recs["insult"].target == pytest.approx(0.7)


def test_eval_only_sources_go_to_test_ood():
    tc = ToxicChat().convert({"conv_id": "c", "user_input": "u", "toxicity": 1, "human_annotation": True}, "test")
    assert tc[0].split == "test_ood" and tc[0].target == 1.0
    assert ToxicChat().convert({"conv_id": "c", "user_input": "u", "toxicity": 1, "human_annotation": False}, "test") == []
    bt = BeaverTails().convert({"idx": 0, "prompt": "p", "response": "r", "is_safe": False}, "30k_test")
    assert bt[0].split == "test_ood" and bt[0].target == 1.0
    om = OpenAIModeration().convert({"idx": 0, "prompt": "t", "S": 0, "H": 0, "V": 1, "HR": 0, "SH": 0, "S3": 0, "H2": 0, "V2": 0}, "train")
    assert [r.target for r in om] == [0.0, 1.0]


def test_aegis_repeated_ids_give_unique_records():
    import jevlite.data.public.safety as safety

    base = {"id": "x", "prompt": "how to hurt someone", "prompt_label": "unsafe", "prompt_label_source": "human",
            "violated_categories": "Violence"}
    rows = [
        {**base, "response": "Here is how…", "response_label": "unsafe", "response_label_source": "llm_jury"},
        {**base, "response": "Here is how…", "response_label": "unsafe", "response_label_source": "llm_jury"},  # exact repeat
        {**base, "response": "I can't help.", "response_label": "safe", "response_label_source": "refusal_data_augmentation"},
        {**base, "response": "Another harmful answer", "response_label": "unsafe", "response_label_source": "human"},
    ]
    a = Aegis2()
    a._eval_prompts = set()
    orig = safety.hf_rows
    safety.hf_rows = lambda *x, **k: iter(rows)
    try:
        recs = [r for row in a.rows("train") for r in a.convert(row, "train")]
    finally:
        safety.hf_rows = orig
    ids = [r.id for r in recs]
    assert len(ids) == len(set(ids))
    kinds = sorted(r.criterion for r in recs)
    assert kinds == ["harm_category", "prompt_harm", "response_harm", "response_harm"]
