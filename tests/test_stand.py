"""Eval-stand rows → records (``jevlite.data.stand``), on made-up rows and criteria (no company data)."""

import pytest

from jevlite.data.mix import load_mix_config
from jevlite.data.stand import (
    CriteriaSpec, assign_splits, convert_rows, history_turns, merge_rows, mode_score, soft_target,
)
from jevlite.model import parse_hub_spec

SPEC = CriteriaSpec.from_dict({
    "criteria": {
        "grounded": {"context": True, "prompts": ["Is it grounded?", "How grounded is it?"]},
        "tone": {"context": False, "prompts": ["How is the tone?"]},
    },
    "train_schemes": {"digits": 1.0},
    "eval_scheme": "words",
    "faithfulness": {"from": "grounded", "pos_min": 4, "neg_max": 2},
})

TRANSCRIPT = (
    "[turn 1]\nUSER: How do I reset my key?\nASSISTANT: Open settings.\n"
    "[turn 2 (judged)]\nUSER: And then?\nASSISTANT: Press reset."
)


def row(i, **kw):
    base = {"id": f"r{i}", "question": f"Question {i}?", "answer": f"Answer {i}.", "guard_is_allowed": "True",
            "retrieved_context": "Chunk A.\n---\nChunk B.", "grounded": "4.0", "tone": "5.0", "category": "cat_a",
            "models": "ASKER_MODEL=x; JUDGE_MODEL=judge-1"}
    return base | kw


def test_history_and_states():
    assert history_turns(TRANSCRIPT) == ["User: How do I reset my key?\nAssistant: Open settings."]
    rows, _ = merge_rows([row(1, question="And then?", answer="Press reset.", transcript=TRANSCRIPT)], SPEC)
    r = rows[0]
    assert r.dialogue().startswith("Guard verdict on the question: ALLOWED\n\nUser: How do I reset my key?")
    assert r.dialogue().endswith("User: And then?\n\nAssistant: Press reset.")
    assert r.with_documents().startswith("RETRIEVED CONTEXT:\nDocument [1]: Chunk A.\n\nDocument [2]: Chunk B.")
    assert r.dialogue(0).count("User:") == 1


def test_records_per_row_and_fields():
    records, report = convert_rows([row(1)], SPEC)
    by_crit = {r.criterion: r for r in records}
    assert set(by_crit) == {"grounded", "tone", "faithfulness_full"}
    g, t, f = by_crit["grounded"], by_crit["tone"], by_crit["faithfulness_full"]
    assert g.state.startswith("RETRIEVED CONTEXT") and not t.state.startswith("RETRIEVED CONTEXT")
    assert g.candidates == ["1", "2", "3", "4", "5"] and g.target == 3        # train: digits, score 4 → index 3
    assert g.prompt in SPEC.criteria[0].prompts and g.field("judge_model") == "judge-1"
    assert g.field("category") == "cat_a" and g.group == t.group == f.group
    assert f.type == "bool" and f.target == 1.0 and f.state == ["Chunk A.", "Chunk B."] and f.field("label_source") == "judge_weak"
    assert report["rows"] == 1 and report["records"] == 3


def test_skips_and_missing_context():
    rows = [row(1, error="Timeout"), row(2, answer=""), row(3, grounded="", tone=""),
            row(4, retrieved_context=""),                                  # allowed, chunks not logged
            row(5, retrieved_context="", guard_is_allowed="False", grounded="2")]  # blocked: no chunks needed
    records, report = convert_rows(rows, SPEC)
    assert report["skipped"] == {"error": 1, "no question or answer": 1, "no judge scores": 1}
    crit = {(r.field("stand_row"), r.criterion) for r in records}
    assert ("r4", "grounded") not in crit and ("r4", "tone") in crit
    assert ("r5", "grounded") in crit and ("r5", "faithfulness_full") not in crit     # no chunks → no faithfulness
    assert "RETRIEVED CONTEXT:\n(none)" in next(r.state for r in records if r.id == "stand:r5:grounded")


def test_judge_repeats_soft_in_train_mode_in_eval():
    assert soft_target([4, 4, 5]) == pytest.approx([0, 0, 0, 2 / 3, 1 / 3])
    assert mode_score([4, 4, 5]) == 4 and mode_score([3, 5]) == 3 and mode_score([2, 4, 4, 2, 5]) == 2
    rows = [row(1, grounded="4"), row(1, grounded="4"), row(1, grounded="5")]
    train, _ = convert_rows(rows, SPEC)
    assert next(r for r in train if r.criterion == "grounded").target == pytest.approx([0, 0, 0, 2 / 3, 1 / 3])
    test, _ = convert_rows(rows, SPEC, explicit_splits={"r1": "test_in"})
    g = next(r for r in test if r.criterion == "grounded")
    assert g.target == 3 and g.candidates == SPEC.level_schemes["words"] and g.field("judge_repeats") == 3


def test_judge_only_repeats_without_id_merge_by_question_id():
    # the stand's judge-only output drops ``id`` but keeps it as ``question_id`` (turn blank)
    orig = row(1, question_id="r1")
    rep = {k: v for k, v in row(1, question_id="r1", grounded="5", turn="").items() if k != "id"}
    rows, _ = merge_rows([orig, rep], SPEC)
    assert len(rows) == 1 and rows[0].id == "r1" and rows[0].scores["grounded"] == [4, 5]
    dlg = {k: v for k, v in row(2, question_id="d7", turn="2.0").items() if k != "id"}
    assert merge_rows([dlg], SPEC)[0][0].id == "d7_t2"


def test_live_records_match_stand_test_records():
    from jevlite.data.stand import live_records

    stand, _ = convert_rows([row(1, question="And then?", answer="Press reset.", transcript=TRANSCRIPT)], SPEC,
                            explicit_splits={"r1": "test_in"})
    hist = ["User: How do I reset my key?\nAssistant: Open settings."]
    live = live_records("And then?", "Press reset.", True, ["Chunk A.", "Chunk B."], SPEC, history=hist)
    by = lambda recs: {r.criterion: r for r in recs if r.type == "score"}  # noqa: E731
    s, lv = by(stand), by(live)
    assert set(lv) == {"grounded", "tone"}
    for c in lv:
        assert lv[c].state == s[c].state and lv[c].candidates == s[c].candidates == SPEC.level_schemes["words"]
        assert lv[c].prompt in SPEC.criteria[[x.name for x in SPEC.criteria].index(c)].prompts
    # allowed but no chunks: the criteria that read them are left out, as in the converter
    assert {r.criterion for r in live_records("Q?", "A.", True, [], SPEC)} == {"tone"}
    blocked = {r.criterion: r.state for r in live_records("Q?", "A.", False, [], SPEC)}  # blocked: no chunks needed
    assert blocked["grounded"].startswith("RETRIEVED CONTEXT:\n(none)\n\nGuard verdict on the question: BLOCKED")
    assert blocked["tone"].startswith("Guard verdict on the question: BLOCKED")


def test_history_trimmed_to_fit():
    rows = [row(1, question="And then?", answer="Press reset.", transcript=TRANSCRIPT)]
    count = len  # characters as "tokens"
    full = merge_rows(rows, SPEC)[0][0]
    records, report = convert_rows(rows, SPEC, count_tokens=count, max_len=len(full.with_documents(0)) + 256)
    g = next(r for r in records if r.criterion == "grounded")
    t = next(r for r in records if r.criterion == "tone")
    assert g.field("history_trimmed") == 1 and "reset my key" not in g.state
    assert t.field("history_trimmed") == 0 and "reset my key" in t.state      # the shorter state keeps its history
    assert report["history_trimmed_records"] == 1


def test_splits_keep_dialogues_whole_and_respect_eligibility():
    turn1 = row(1, question="How do I reset my key?")
    turn2 = row(2, question="And then?", transcript=TRANSCRIPT)       # same dialogue: same first user message
    rows, _ = merge_rows([turn1, turn2, *[row(i) for i in range(3, 400)]], SPEC)
    assert rows[0].group == rows[1].group
    splits = assign_splits(rows, shares={"test_in": 0.3, "calib": 0.2}, eval_eligible=lambda r: r.id != "r3")
    assert splits["r1"] == splits["r2"] and splits["r3"] == "train"
    assert {"train", "test_in", "calib"} <= set(splits.values())
    with pytest.raises(ValueError, match="one split"):
        assign_splits(rows, explicit={"r1": "test_in"})
    with pytest.raises(ValueError, match="shares"):
        assign_splits(rows, shares={"test": 0.5})


def test_spec_validation():
    with pytest.raises(ValueError, match="levels"):
        CriteriaSpec.from_dict({"criteria": {"a": {"prompts": ["?"]}}, "level_schemes": {"x": ["1", "2"]}})
    with pytest.raises(ValueError, match="faithfulness"):
        CriteriaSpec.from_dict({"criteria": {"a": {"prompts": ["?"]}}, "faithfulness": {"from": "b"}})


def test_stage_c_mix_config_loads():  # order B → D → C: C replays D
    cfg = load_mix_config("configs/mixes/stage_c.yaml")
    assert cfg.max_len == 8192 and {s.name for s in cfg.sources} == {"stand", "stage_d_replay", "stage_d_eval"}


def test_parse_hub_spec():
    assert parse_hub_spec("hub:Nefelim584/jevlite-runs/stage_a-full-s42/final") == ("Nefelim584/jevlite-runs", None, "stage_a-full-s42/final")
    assert parse_hub_spec("hub:Nefelim584/jev-lite@v0.2.0") == ("Nefelim584/jev-lite", "v0.2.0", "")
    assert parse_hub_spec("hub:o/r@main/sub") == ("o/r", "main", "sub")
    with pytest.raises(ValueError):
        parse_hub_spec("hub:only-owner")
