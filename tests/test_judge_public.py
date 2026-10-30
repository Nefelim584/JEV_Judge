"""New public judge sources and the ClapNQ rebuild (todo Phase 4b, DATASETS.md 8.2)."""

from collections import Counter

from test_public_data import ROWS

from jevlite.data.public import get_converter
from jevlite.data.public import judge_public
from jevlite.data.public.judge_public import sentence_labels, trust_prompt


def convert(name, **override):
    split, row = ROWS[name]
    return list(get_converter(name).convert({**row, **override}, split))


def test_clapnq_judge_answerability_is_a_property_of_the_pair():
    recs = [r for r in convert("clapnq_judge") if r.criterion == "answerable"]
    assert sorted(r.target for r in recs) == [0.0, 1.0]
    assert len({r.state for r in recs}) == 1          # the same passage carries both labels
    assert {r.split for r in convert("clapnq_judge")} <= {"train", "calib"} and len({r.split for r in convert("clapnq_judge")}) == 1


def test_clapnq_judge_relevance_levels():
    rel = {r.id.rsplit(":", 1)[1]: r for r in convert("clapnq_judge") if r.criterion == "relevance"}
    assert rel["own"].target == 3 and rel["other"].target == 0 and rel["joined"].target == 1
    assert rel["own"].candidates[rel["own"].target] == "fully addresses"
    assert rel["joined"].state["question"].endswith("when did the war end?")


def test_clapnq_judge_length_is_no_cue():
    recs = [r for r in convert("clapnq_judge") if r.criterion in ("conciseness", "completeness")]
    by = Counter((r.criterion, r.target) for r in recs)
    assert by == {("conciseness", 0): 1, ("conciseness", 1): 1, ("completeness", 0): 1, ("completeness", 1): 1}
    for r in recs:
        a, b = r.state.split("Answer A:\n")[1].split("\n\nAnswer B:\n")
        better, worse = (a, b) if r.target == 0 else (b, a)
        assert (len(better) < len(worse)) == (r.criterion == "conciseness")


def test_faithfulness_qa_pair_and_filter():
    gold, orig = convert("faithfulness_qa")
    assert gold.target == 1.0 and gold.prompt.endswith("Bo") and orig.target == 0.0 and orig.prompt.endswith("Ann")
    assert gold.field("template") == orig.field("template") and orig.field("perturbation") == "world_true_answer"
    assert convert("faithfulness_qa", modified_context="X was founded by Bo, not Ann.") == []   # the edit is incomplete
    assert convert("faithfulness_qa", original_entity="2005", original_answer="2005", faithful_answer="42 years",
                   modified_context="In 42 years, X was founded.") == []                         # a year for a non-year


def test_minicheck_fails_closed_without_aggrefact(monkeypatch):
    def fake_rows(hf_id, config, split, streaming=False, parquet_export=False):
        if hf_id == "lytang/LLM-AggreFact":
            raise PermissionError("gated")
        yield {"claim": "c", "doc": "d", "label": 1}

    monkeypatch.setattr(judge_public, "hf_rows", fake_rows)
    conv = get_converter("minicheck")
    assert list(conv.rows("c2d")) == [] and len(list(conv.rows("d2c"))) == 1


def test_minicheck_drops_aggrefact_claims(monkeypatch):
    def fake_rows(hf_id, config, split, streaming=False, parquet_export=False):
        if hf_id == "lytang/LLM-AggreFact":
            yield {"claim": "The  bridge opened in 1932."}
        else:
            yield from ({"claim": "The bridge opened in 1932.", "doc": "d", "label": 1}, {"claim": "other", "doc": "d", "label": 0})

    monkeypatch.setattr(judge_public, "hf_rows", fake_rows)
    assert [r["claim"] for r in get_converter("minicheck").rows("c2d")] == ["other"]


PROMPT = """Instruction: Write an accurate, engaging, and concise answer for the given question using only the provided search results.

Question: Who wrote X?

Document [1](Title: X (book)): X was written by Ann.
Document [2](Title: Bo): Bo is a painter.

Answer:"""


def test_trust_prompt_parsing_and_eli5_whitelist(monkeypatch):
    kind, question, docs = trust_prompt(PROMPT)
    assert kind == "long" and question == "Who wrote X?" and docs == ["X (book)\nX was written by Ann.", "Bo\nBo is a painter."]
    assert trust_prompt("Instruction: Provide a list of accurate answers.\n\n" + PROMPT.split("\n\n", 1)[1])[0] == "qampari"

    def fake_rows(hf_id, config, split, streaming=False, parquet_export=False):
        if hf_id == "din0s/asqa":
            yield {"ambiguous_question": "who wrote x"}
        else:
            yield {"prompt": PROMPT, "chosen": "Ann wrote X [1].", "rejected": "Bo wrote X [2]."}
            yield {"prompt": PROMPT.replace("Who wrote X?", "Why is the sky blue?"), "chosen": "a", "rejected": "b"}  # ELI5-like

    monkeypatch.setattr(judge_public, "hf_rows", fake_rows)
    rows = list(get_converter("trust_data").rows("train"))
    assert [r["source"] for r in rows] == ["asqa"]


def test_trust_data_records():
    recs = convert("trust_data")
    pairs = [r for r in recs if r.type == "choice"]
    assert sorted(r.target for r in pairs) == [0, 1] and all(r.candidates == ["answer A", "answer B"] for r in pairs)
    (ans,) = [r for r in recs if r.criterion == "answerable"]
    assert ans.target == 1.0
    refusal = judge_public._REFUSAL
    (ans,) = [r for r in convert("trust_data", chosen=refusal) if r.criterion == "answerable"]
    assert ans.target == 0.0
    assert convert("trust_data", chosen=refusal, rejected=refusal) == []


def test_sentence_labels_and_psiloqa():
    text = "Acrijos is in Soria. It has a famous castle."
    assert sentence_labels(text, [(21, 44)]) == [("Acrijos is in Soria.", True), ("It has a famous castle.", False)]
    answer, *rest = convert("psiloqa")
    assert answer.target == 0.0 and answer.state == ROWS["psiloqa"][1]["wiki_passage"]
    assert all(r.criterion == "faithfulness" for r in [answer, *rest])
    (clean,) = [r for r in convert("psiloqa", labels="[]") if r.id.endswith("answer:llm")]
    assert clean.target == 1.0
    (golden,) = [r for r in convert("psiloqa") if r.id.endswith("answer:golden")]
    assert golden.target == 1.0 and golden.state == answer.state and golden.field("template") == answer.field("template")


def test_qasper_answerability_on_one_paper():
    recs = convert("qasper")
    ans = {r.id.rsplit(":", 1)[1]: r for r in recs if r.criterion == "answerable"}
    assert ans["with_evidence"].target == 1.0 and ans["without_evidence"].target == 0.0 and ans["unanswerable"].target == 0.0
    assert len(ans["with_evidence"].state) == len(ans["without_evidence"].state)
    assert any("15 words" in c for c in ans["with_evidence"].state) and not any("15 words" in c for c in ans["without_evidence"].state)
    (gold,) = [r for r in recs if r.criterion == "faithfulness_full"]
    assert gold.target == 1.0 and gold.prompt.endswith("15 words")


def test_ragtruth_summary_and_sentence():
    summary, sentence = convert("ragtruth")
    assert summary.target == 0.0 and summary.field("hallucination_types") == ["Evident Baseless Info"]
    assert sentence.target == (0.0 if sentence.prompt.endswith("It cost $5 million.") else 1.0)
    assert {r.split for r in convert("ragtruth")} <= {"train", "calib"}


def test_repeated_ids_keep_the_first_row(monkeypatch):
    def fake_rows(hf_id, config, split, streaming=False, parquet_export=False):
        row = {**ROWS["faithfulness_qa"][1], "source": "triviaqa"}
        yield from (row, {**row, "modified_context": "X was founded by Cy.", "faithful_answer": "Cy"}, {**row, "source": "squad"})

    monkeypatch.setattr(judge_public, "hf_rows", fake_rows)
    rows = list(get_converter("faithfulness_qa").rows("train"))
    assert [(r["source"], r["faithful_answer"]) for r in rows] == [("triviaqa", "Bo"), ("squad", "Bo")]
    ids = [r.id for row in rows for r in get_converter("faithfulness_qa").convert(row, "train")]
    assert len(ids) == len(set(ids))
