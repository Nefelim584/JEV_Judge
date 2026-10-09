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
    unique_pairs,
)
from jevlite.data.public.factcheck import evidence_window, table_to_text
from jevlite.schema import MAX_CANDIDATES

# One handmade source row per converter, in the shape its ``rows`` yields.
ROWS = {
    "vitaminc": ("train", {"unique_id": "u1", "claim": "X has 5 staff.", "evidence": "X has five staff.", "label": "SUPPORTS", "revision_type": "real"}),
    "tabfact": ("train", {"table_id": "t1.csv", "idx": 0, "table_text": "year#team\n1999#a\n2000#b", "caption": "seasons", "statement": "team a played in 1999", "label": 1}),
    "multi_nli": ("train", {"idx": 0, "pairID": "p1", "premise": "A man sleeps.", "hypothesis": "A man is awake.", "label": 2, "genre": "fiction"}),
    "wanli": ("train", {"id": "w1", "premise": "It rained.", "hypothesis": "The ground is wet.", "gold": "neutral"}),
    "snli": ("train", {"idx": 0, "premise": "A dog runs.", "hypothesis": "An animal moves.", "label": 0}),
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
    "fever": ("train", {"id": 1, "claim": "Oslo is in Norway.", "label": "SUPPORTS", "pages": [{"title": "Oslo", "sentences": ["Oslo is a city.", "It is the capital of Norway."], "evidence": [1]}]}),
    "hover": ("train", {"uid": "h1", "claim": "A and B are dogs.", "label": "SUPPORTED", "num_hops": 2, "pages": [{"title": "A", "text": "A is a dog."}, {"title": "B", "text": "B is a dog breed."}]}),
    "feverous": ("train", {"id": 2, "claim": "X won in 1990.", "label": "NOT ENOUGH INFO", "pages": [{"title": "X", "sentences": ["X is a team.", "", "X played in 1990."], "evidence": [2]}]}),
    "nq": ("train", {"id": 3, "question": "who wrote the book", "title": "The Book", "passage": "The Book is a novel by Ann Lee.", "answerable": True}),
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


def test_unique_pairs_collapses_repeats_and_drops_conflicts():
    rows = [
        {"p": "a", "h": "x", "label": 0},
        {"p": "a", "h": "x", "label": 0},  # repeat, same label: collapsed
        {"p": "b", "h": "y", "label": 0},
        {"p": "b", "h": "y", "label": 2},  # conflicting labels: both dropped
        {"p": "c", "h": "z", "label": 1},
    ]
    kept = list(unique_pairs(rows, ("p", "h"), "label"))
    assert [(r["p"], r["idx"]) for r in kept] == [("a", 0), ("c", 4)]
    streamed = list(unique_pairs(iter(rows), ("p", "h"), "label", streaming=True))
    assert [r["idx"] for r in streamed] == [0, 2, 4]


def test_multi_nli_mismatched_is_ood():
    (r,) = get_converter("multi_nli").convert(ROWS["multi_nli"][1], "validation_mismatched")
    assert r.split == "test_ood"


def test_social_iqa_labels_are_one_based():
    (r,) = get_converter("social_iqa").convert(ROWS["social_iqa"][1], "train")
    assert r.candidates[r.target] == "grateful"


def test_unknown_dataset():
    with pytest.raises(ValueError, match="unknown dataset"):
        get_converter("nope")


def _mcqa(i, n=4, target=0, split="train"):
    from jevlite.data.unified import Record

    return Record(id=f"m:{i}", source="m", split=split, state=f"question {i}", type="choice", prompt="p",
                  candidates=[f"opt {i}-{k}" for k in range(n)], target=target)


def test_borrow_distractors_keeps_gold_and_varies_k():
    from jevlite.data.public.mcqa import BORROW_MAX, borrow_distractors

    records = [_mcqa(i, target=i % 4) for i in range(200)]
    out = borrow_distractors(records, "m", "train")
    assert [r.id for r in out] == [r.id for r in records]
    assert out == borrow_distractors(records, "m", "train")  # deterministic
    added = [r.field("borrowed_distractors") for r in out]
    assert 0.35 < sum(a > 0 for a in added) / len(added) < 0.65
    assert set(added) <= set(range(BORROW_MAX + 1))
    for before, after in zip(records, out):
        assert after.candidates[after.target] == before.candidates[before.target]  # gold kept
        assert len(after.candidates) == len(before.candidates) + after.field("borrowed_distractors")
        assert len(set(after.candidates)) == len(after.candidates)
        assert [c for c in after.candidates if c in before.candidates] == before.candidates  # own order kept
        borrowed = [c for c in after.candidates if c not in before.candidates]
        assert all(not c.startswith(f"opt {before.id[2:]}-") for c in borrowed)  # from other questions


def test_borrow_distractors_respects_max_candidates_and_duplicates():
    from jevlite.data.public.mcqa import borrow_distractors

    full = _mcqa(0, n=MAX_CANDIDATES)
    same = [_mcqa(1), _mcqa(1).model_copy(update={"id": "m:dup"})]  # the pool holds only their own options
    out = borrow_distractors([full, *same], "m", "train", share=1.0)
    assert len(out[0].candidates) == MAX_CANDIDATES
    pool_other = set(full.candidates)
    for r in out[1:]:
        assert len(set(r.candidates)) == len(r.candidates)
        assert set(r.candidates) - set(_mcqa(1).candidates) <= pool_other


def test_mcqa_records_borrow_within_a_source_split(monkeypatch):
    conv = get_converter("commonsense_qa")
    rows = {
        "train": [{"id": f"t{i}", "question": f"q{i}?", "choices": {"label": ["A", "B", "C"], "text": [f"a{i}", f"b{i}", f"c{i}"]}, "answerKey": "A"} for i in range(30)],
        "validation": [{"id": f"v{i}", "question": f"v{i}?", "choices": {"label": ["A", "B", "C"], "text": [f"x{i}", f"y{i}", f"z{i}"]}, "answerKey": "B"} for i in range(30)],
    }
    monkeypatch.setattr(type(conv), "rows", lambda self, split: iter(rows[split]))
    records = list(conv.records())
    assert any(r.field("borrowed_distractors") for r in records)
    for r in records:
        own_split_letters = "abc" if r.id.split(":")[1] == "train" else "xyz"
        assert all(c[0] in own_split_letters for c in r.candidates)  # never borrowed across source splits
    assert len(list(conv.records(limit=5))) == 10


def test_share_alike_flags():
    sa = {n for n, c in CONVERTERS.items() if c.share_alike}
    assert sa == {"boolq", "snli", "qnli", "vitaminc", "squad_v2", "clapnq", "arc", "dbpedia14", "fever", "hover", "feverous", "nq"}
    multi_nli = CONVERTERS["multi_nli"]
    assert multi_nli.is_share_alike("fiction") and not multi_nli.is_share_alike("travel")
