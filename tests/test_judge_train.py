"""Judge training converters (todo Phase 4b): labels, negatives, one split per source document."""

from collections import Counter

import pytest

from jevlite.data.public import get_converter
from jevlite.data.public.judge_bench import RAG_CLAIM
from jevlite.data.public.judge_train import (
    clean_finqa,
    finqa_text,
    format_like,
    numeric_negative,
    parse_number,
    ragbench_sentences,
    rng_for,
)

RAGBENCH_ROW = {
    "subset": "tatqa", "idx": 3, "id": "tatqa_7_1", "question": "What was revenue in 2019?",
    "documents": ["Revenue was $1.2m in 2019.", "Costs were $0.8m in 2019.", "The firm has 40 staff."],
    "response": "Revenue was $1.2m in 2019. Profit was $0.9m.", "adherence_score": False,
    "response_sentences": [["a", "Revenue was $1.2m in 2019."], ["b", "Profit was $0.9m in 2019."]],
    "sentence_support_information": [
        {"response_sentence_key": "a", "supporting_sentence_keys": ["0a"], "fully_supported": True},
        {"response_sentence_key": "b", "supporting_sentence_keys": [], "fully_supported": False},
    ],
    "unsupported_response_sentence_keys": ["b"],
}

HOTPOT_ROW = {
    "id": "h2", "question": "The band that recorded X is from which city?", "answer": "Leeds", "type": "bridge", "level": "hard",
    "supporting_facts": {"title": ["X (song)", "The Band"], "sent_id": [0, 0]},
    "context": {"title": ["X (song)", "The Band", "Paris", "Rome", "Oslo", "Bern"],
                "sentences": [["X is a song by The Band."], ["The Band is a group from Leeds."], ["Paris is a city in France."],
                              ["Rome is a city."], ["Oslo is a city."], ["Bern is a city."]]},
}


def test_numbers_parse_and_format_like_the_answer():
    assert parse_number("$1,234.5") == 1234.5
    assert parse_number("(5,580)") == -5580
    assert parse_number("14.1%") == 14.1
    assert parse_number("Revenue") is None
    assert format_like(20.0, "14.1%") == "20.0%"
    assert format_like(1500, "$1,200") == "$1,500"
    assert format_like(-3, "7") == "-3"


def test_numeric_negative_is_wrong_and_looks_alike():
    for i in range(200):
        text, kind = numeric_negative(rng_for("t", i), "14.1%", ["14.1", "9.5", "$300"])
        assert kind in ("other_value", "shift") and text.endswith("%") and text != "14.1%"
        assert abs(parse_number(text) - 14.1) > 0.05
    assert numeric_negative(rng_for("t"), "Leeds", ["1"]) is None
    for i in range(50):
        year, _ = numeric_negative(rng_for("y", i), "1979", ["1981", "300"])
        assert year != "1979" and 1960 < int(year) < 2000


def test_ragbench_sentence_labels():
    sents = {s["key"]: s for s in ragbench_sentences(RAGBENCH_ROW)}
    assert sents["a"]["supported"] and sents["a"]["docs"] == [0]
    assert not sents["b"]["supported"]
    # backed only by a "general" key, or contradictory: left out
    row = {**RAGBENCH_ROW, "sentence_support_information": [
        {"response_sentence_key": "a", "supporting_sentence_keys": ["general"], "fully_supported": True},
        {"response_sentence_key": "b", "supporting_sentence_keys": [], "fully_supported": True}]}
    assert ragbench_sentences(row) == []


def test_ragbench_train_records():
    conv = get_converter("ragbench_train")
    records = list(conv.convert(RAGBENCH_ROW, "train"))
    by_crit = Counter(r.criterion for r in records)
    assert by_crit["faithfulness_full"] == 2 and by_crit["faithfulness"] <= 1
    answer = records[0]
    assert answer.target == 0.0 and answer.state == RAGBENCH_ROW["documents"] and answer.domain == "ragbench/tatqa"
    assert {r.split for r in records} <= {"train", "calib"} and len({r.split for r in records}) == 1
    for r in records:
        assert r.field("label_source") == "llm" and r.user_question == RAGBENCH_ROW["question"]
        if r.criterion == "faithfulness":
            assert isinstance(r.state, str)
            # a supported sentence is 1 only against its own document
            assert r.target == float(r.prompt.find("Revenue") >= 0 and r.state == RAGBENCH_ROW["documents"][0])


def test_ragbench_train_skips_the_eval_subsets():
    assert set(get_converter("ragbench_train").SUBSETS).isdisjoint({"covidqa", "cuad", "emanual", "delucionqa", "msmarco"})


def test_templates_avoid_the_eval_wording():
    eval_prefix = RAG_CLAIM.prompt.split("{")[0]
    for name, row in (("ragbench_train", RAGBENCH_ROW), ("hotpotqa_judge", HOTPOT_ROW)):
        for i in range(30):
            for r in get_converter(name).convert({**row, "id": f"{row['id']}-{i}"}, "train"):
                assert not r.prompt.startswith(eval_prefix)


def test_hotpot_answer_pair_and_answerability_share_the_paragraph_pool():
    records = {(r.criterion, r.target): r for r in get_converter("hotpotqa_judge").convert(HOTPOT_ROW, "train")}
    gold, neg = records[("faithfulness_full", 1.0)], records[("faithfulness_full", 0.0)]
    assert gold.state == neg.state and "Leeds" in gold.prompt and "Leeds" not in neg.prompt
    # no one-word name in the gold paragraphs but sentence-initial ones: a distractor's name instead
    assert neg.field("perturbation") == "distractor_entity" and neg.prompt.endswith(": France")
    assert gold.field("template") == neg.field("template")
    full, minus = records[("answerable", 1.0)], records[("answerable", 0.0)]
    assert len(full.state) == len(minus.state)                 # the count of chunks is not a cue
    assert any("Leeds" in c for c in full.state) and not any("Leeds" in c for c in minus.state)
    assert minus.field("removed") == "The Band"
    assert len({r.split for r in records.values()}) == 1


def test_names_stop_at_a_full_stop():
    from jevlite.data.public.judge_train import _NAME, names_in

    text = "It is in Scotland. Approximately 5 km. B sold F.E.A.R. to the Academy of Motion Picture Arts."
    assert _NAME.findall(text) == ["It", "Scotland", "Approximately", "B", "F.E.A.R.", "Academy of Motion Picture Arts"]
    assert names_in(text) == ["Scotland", "F.E.A.R.", "Academy of Motion Picture Arts"]


def test_hotpot_comparison_and_yes_no_negatives():
    conv = get_converter("hotpotqa_judge")
    from test_public_data import ROWS

    comp = {r.target: r for r in conv.convert(ROWS["hotpotqa_judge"][1], "train") if r.criterion == "faithfulness_full"}
    assert comp[0.0].field("perturbation") == "other_entity" and comp[0.0].prompt.endswith("B")
    yes = {**ROWS["hotpotqa_judge"][1], "answer": "yes"}
    neg = [r for r in conv.convert(yes, "train") if r.criterion == "faithfulness_full" and r.target == 0.0]
    assert neg[0].prompt.endswith("no")


def test_tatqa_negatives_keep_the_scale_or_break_it():
    from test_public_data import ROWS

    conv = get_converter("tatqa_judge")
    kinds = Counter()
    for i in range(100):
        row = {**ROWS["tatqa_judge"][1], "uid": f"t{i}"}
        gold, neg = conv.convert(row, "train")
        assert gold.target == 1.0 and gold.prompt.endswith("1,200 thousand")
        assert neg.target == 0.0 and not neg.prompt.endswith("1,200 thousand")
        kinds[neg.field("perturbation")] += 1
    assert set(kinds) <= {"other_value", "shift", "scale"} and kinds["scale"] > 0
    # questions on one table share the split
    splits = {r.split for i in range(200) for r in conv.convert({**ROWS["tatqa_judge"][1], "uid": f"u{i}"}, "train")}
    assert len(splits) == 1


def test_finqa_text_cleanup_and_window():
    assert clean_finqa("rate of 2.05% ( 2.05 % ) as of october 31 , 2009 .") == "rate of 2.05% as of october 31, 2009."
    assert clean_finqa("expense of $ 3.8 million .") == "expense of $3.8 million."
    sents = [f"s{i} " + "x" * 50 for i in range(40)]
    text = finqa_text(sents, {20}, budget=200)
    assert text.split()[0] in ("s19", "s20", "s18") and "s20" in text and len(text) < 260


@pytest.mark.parametrize("name", ["ragbench_train", "hotpotqa_judge", "tatqa_judge", "finqa_judge"])
def test_dev_rows_are_test_in(name):
    from test_public_data import ROWS

    split, row = ROWS[name]
    dev = {"ragbench_train": "validation", "hotpotqa_judge": "validation"}.get(name, "dev")
    assert {r.split for r in get_converter(name).convert(row, dev)} <= {"test_in", "test_ood"}
