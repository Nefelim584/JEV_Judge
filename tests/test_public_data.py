from collections import Counter

import pytest

from jevlite.data.public import CONVERTERS, get_converter
from jevlite.data.public.base import (
    CALIB_SHARE,
    CONTRADICTED,
    ENTAILED,
    NOT_MENTIONED,
    NLI_CHOICE,
    Template,
    nli_records,
    pick_template,
    sample_candidates,
)
from jevlite.data.public.factcheck import evidence_window, table_to_text
from jevlite.schema import MAX_CANDIDATES

# One handmade source row per converter, in the shape its ``rows`` yields.
ROWS = {
    "vitaminc": ("train", {"unique_id": "u1", "claim": "X has 5 staff.", "evidence": "X has five staff.", "label": "SUPPORTS", "revision_type": "real"}),
    "tabfact": ("train", {"table_id": "t1.csv", "idx": 0, "table_text": "year#team\n1999#a\n2000#b", "caption": "seasons", "statement": "team a played in 1999", "label": 1}),
    "multi_nli": ("train", {"pairID": "p1", "premise": "A man sleeps.", "hypothesis": "A man is awake.", "label": 2, "genre": "fiction"}),
    "wanli": ("train", {"id": "w1", "premise": "It rained.", "hypothesis": "The ground is wet.", "gold": "neutral"}),
    "snli": ("train", {"premise": "A dog runs.", "hypothesis": "An animal moves.", "label": 0}),
    "qnli": ("train", {"idx": 1, "question": "When did it start?", "sentence": "It started in 1990.", "label": 0}),
    "paws": ("train", {"id": 1, "sentence1": "A met B in Paris.", "sentence2": "In Paris, A met B.", "label": 1}),
    "boolq": ("train", {"question": "is the sky blue", "answer": True, "passage": "The sky is blue."}),
    "squad_v2": ("train", {"id": "s1", "context": "Paris is in France.", "question": "Where is Rome?", "answers": {"text": [], "answer_start": []}}),
    "clapnq": ("train", {"id": "c1", "input": "who wrote it", "passages": [{"title": "Book", "text": "It was written by Ann."}], "output": [{"answer": "Ann wrote it."}]}),
    "arc": ("train", {"id": "a1", "question": "What melts ice?", "choices": {"text": ["heat", "cold"], "label": ["A", "B"]}, "answerKey": "A", "config": "ARC-Easy"}),
    "openbookqa": ("train", {"id": "o1", "question_stem": "The sun is", "choices": {"text": ["a star", "a planet"], "label": ["A", "B"]}, "answerKey": "A"}),
    "commonsense_qa": ("train", {"id": "q1", "question": "Where do fish live?", "choices": {"label": ["A", "B", "C"], "text": ["water", "sky", "desk"]}, "answerKey": "A"}),
    "cosmos_qa": ("train", {"id": "k1", "context": "I saw a band.", "question": "What did I see?", "answer0": "a band", "answer1": "a car", "answer2": "a dog", "answer3": "a tree", "label": "0"}),
    "social_iqa": ("train", {"idx": 0, "context": "Ann helped Bo.", "question": "How does Bo feel?", "answerA": "grateful", "answerB": "angry", "answerC": "bored", "label": 1}),
    "clinc150": ("train", {"idx": 0, "text": "book me a flight", "intent": 2, "label_names": ["oos", "card_declined", "book_flight", "translate"]}),
    "massive": ("train", {"id": "m1", "utt": "wake me up at nine", "intent": "alarm_set", "label_names": ["alarm_set", "play_music", "weather_query"]}),
    "dbpedia14": ("train", {"idx": 0, "title": "Abbott Ltd", "content": "A coachbuilding firm.", "label": 0, "label_names": ["Company", "Artist", "Village", "Animal"]}),
    "go_emotions": ("train", {"id": "g1", "text": "I love it!", "labels": [1], "label_names": ["anger", "love", "joy", "fear"]}),
    "helpsteer2": ("train", {"idx": 0, "prompt": "c#", "response": "C# is a language.", "helpfulness": 3, "correctness": 4, "coherence": 4, "complexity": 2, "verbosity": 1}),
    "helpsteer3": ("train", {"idx": 0, "domain": "general", "context": [{"role": "user", "content": "hi"}], "response1": "Hello!", "response2": "Hey.", "overall_preference": -3}),
    "summarize_from_feedback": ("train", {"idx": 0, "info": {"post": "Long post."}, "summaries": [{"text": "short a"}, {"text": "short b"}], "choice": 1}),
    "contract_nli": ("test", {"doc_id": 7, "text": "NDA text. No copies.", "spans": [[0, 9], [10, 20]], "hyp_id": "nda-1", "hypothesis": "Party shall not copy.", "choice": "NotMentioned", "evidence": []}),
    "banking77": ("test", {"idx": 0, "text": "where is my card", "category": "card_arrival", "label_names": ["card_arrival", "top_up_failed", "pin_blocked"]}),
}


def test_every_converter_has_a_sample_row():
    assert set(ROWS) == set(CONVERTERS)


@pytest.mark.parametrize("name", sorted(ROWS))
def test_convert_gives_valid_deterministic_records(name):
    split, row = ROWS[name]
    conv = get_converter(name)
    records = list(conv.convert(row, split))
    assert records, f"{name}: no records from a valid row"
    again = list(conv.convert(row, split))
    assert [r.model_dump() for r in records] == [r.model_dump() for r in again]
    for r in records:
        assert r.source == name and r.id.startswith(f"{name}:")
        assert r.field("family") and r.field("template")
        if r.candidates:
            assert len(r.candidates) <= MAX_CANDIDATES


def test_held_out_datasets_go_to_test_ood_only():
    for name in ("contract_nli", "banking77"):
        split, row = ROWS[name]
        assert {r.split for r in get_converter(name).convert(row, split)} == {"test_ood"}


def test_train_rows_carve_out_calib():
    conv = get_converter("vitaminc")
    splits = Counter(conv.target_split("train", i) for i in range(20_000))
    assert set(splits) == {"train", "calib"}
    assert splits["calib"] / 20_000 == pytest.approx(CALIB_SHARE, abs=0.01)


def test_held_out_template_never_used_in_train_and_moves_test_to_ood():
    templates = (Template("a", "A"), Template("b", "B"), Template("held", "H"))
    train = Counter(pick_template(templates, "train", i)[0].id for i in range(2000))
    assert set(train) == {"a", "b"}
    test = [pick_template(templates, "test_in", i) for i in range(2000)]
    assert all(split == "test_ood" for t, split in test if t.id == "held")
    assert all(split == "test_in" for t, split in test if t.id != "held")
    assert 0.4 < sum(t.id == "held" for t, _ in test) / 2000 < 0.6


def test_nli_rows_split_about_evenly_between_bool_and_choice():
    conv = get_converter("vitaminc")
    types = Counter()
    for i in range(2000):
        (r,) = nli_records(conv, key=("train", i), source_split="train", state="s", claim="c", label=CONTRADICTED)
        types[r.type] += 1
        if r.type == "bool":
            assert r.target == 0.0
        else:
            assert r.target == CONTRADICTED and len(r.candidates) == 3
    assert 0.45 < types["bool"] / 2000 < 0.55


def test_nli_label_maps_to_bool_and_choice():
    conv = get_converter("vitaminc")
    for label, bool_target in ((ENTAILED, 1.0), (CONTRADICTED, 0.0), (NOT_MENTIONED, 0.0)):
        for i in range(50):
            (r,) = nli_records(conv, key=("train", label, i), source_split="train", state="s", claim="c", label=label)
            assert r.target == (bool_target if r.type == "bool" else label)
    choice_labels = {t.candidates[NOT_MENTIONED] for t in NLI_CHOICE}
    assert "not mentioned" in choice_labels


def test_two_class_sets_stay_bool():
    conv = get_converter("tabfact")
    for i in range(200):
        (r,) = nli_records(conv, key=("train", i), source_split="train", state="s", claim="c", label=ENTAILED, three_way=False)
        assert r.type == "bool"


def test_sample_candidates_keeps_gold_and_bounds():
    labels = [f"l{i}" for i in range(151)]
    ks = set()
    for i in range(500):
        cands, gold = sample_candidates(labels, 42, "clinc", i)
        assert cands[gold] == "l42" and len(set(cands)) == len(cands)
        assert 3 <= len(cands) <= 16
        ks.add(len(cands))
    assert len(ks) > 5  # the number of options varies
    assert sample_candidates(labels, 42, "clinc", 1) == sample_candidates(labels, 42, "clinc", 1)
    small, gold = sample_candidates(["a", "b"], 1, "x")
    assert sorted(small) == ["a", "b"] and small[gold] == "b"


def test_evidence_window():
    text = "s0. s1. s2. s3. s4. s5. s6."
    spans = [[i * 4, i * 4 + 3] for i in range(7)]
    assert evidence_window(text, spans, [1]) == "s0.\ns1.\ns2."
    # Two separate evidence spans: neighbours of each, with an ellipsis over the gap.
    assert evidence_window(text, spans, [0, 5]) == "s0.\ns1.\n…\ns4.\ns5.\ns6."
    window = evidence_window(text, spans, [], "k")
    assert window == evidence_window(text, spans, [], "k") and "…" not in window


def test_table_to_text():
    text = table_to_text("year#team\n1999#a\n", "seasons")
    assert text == "Table: seasons\n| year | team |\n|---|---|\n| 1999 | a |"


def test_helpsteer2_gives_one_score_per_attribute_in_one_split():
    split, row = ROWS["helpsteer2"]
    records = list(get_converter("helpsteer2").convert(row, split))
    assert [r.criterion for r in records] == ["helpfulness", "correctness", "coherence", "complexity", "verbosity"]
    assert len({r.split for r in records}) == 1
    for r in records:
        level = r.candidates[r.target]
        if len(r.candidates) == 3:  # coarse scheme: 0–1 low, 2 medium, 3–4 high
            assert level == {3: "high", 4: "high", 2: "medium", 1: "low"}[row[r.criterion]]
        else:
            assert r.target == row[r.criterion]


def test_helpsteer3_maps_preference_and_skips_multilingual():
    conv = get_converter("helpsteer3")
    split, row = ROWS["helpsteer3"]
    (r,) = conv.convert(row, split)
    assert r.target == 0  # −3: response 1 much better, i.e. response 2 is the lowest level
    assert list(conv.convert({**row, "domain": "multilingual"}, split)) == []


def test_skipped_rows():
    assert list(get_converter("go_emotions").convert({**ROWS["go_emotions"][1], "labels": [1, 2]}, "train")) == []
    assert list(get_converter("multi_nli").convert({**ROWS["multi_nli"][1], "label": -1}, "train")) == []
    dup = {**ROWS["arc"][1], "choices": {"text": ["heat", "heat"], "label": ["A", "B"]}}
    assert list(get_converter("arc").convert(dup, "train")) == []


def test_multi_nli_mismatched_is_ood():
    (r,) = get_converter("multi_nli").convert(ROWS["multi_nli"][1], "validation_mismatched")
    assert r.split == "test_ood"


def test_social_iqa_labels_are_one_based():
    (r,) = get_converter("social_iqa").convert(ROWS["social_iqa"][1], "train")
    assert r.candidates[r.target] == "grateful"


def test_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        get_converter("nope")
