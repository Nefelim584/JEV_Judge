"""FEVER, HoVer, FEVEROUS and Natural Questions: the pure parts of the converters (no dumps)."""

from jevlite.data.public import get_converter
from jevlite.data.public.factcheck import (
    clean_abstract,
    claim_grams,
    detokenize,
    fever_lines,
    fever_title,
    feverous_sentences,
    sentence_window,
    strip_wiki_links,
    subject_page,
    wiki_chunks,
)
from jevlite.data.public.qa import join_tokens, nq_example


def test_detokenize_ptb():
    assert detokenize("Savages -LRB- 2012 film -RRB- is a film , directed by Oliver Stone .") == (
        "Savages (2012 film) is a film, directed by Oliver Stone."
    )
    assert detokenize("He did n't know -COLON- `` yes '' .") == 'He didn\'t know: " yes ".'
    assert fever_title("Savages_-LRB-2012_film-RRB-") == "Savages (2012 film)"


def test_fever_lines_index_by_sentence_number():
    lines = "0\tOslo is a city .\tOslo\n1\t\n2\tIt is the capital of Norway .\tNorway"
    assert fever_lines(lines) == ["Oslo is a city.", "", "It is the capital of Norway."]


def test_sentence_window_with_and_without_evidence():
    s = [f"S{i}." for i in range(10)]
    assert sentence_window(s, [5]) == "S4. S5. S6."
    assert sentence_window(s, [1, 8]) == "S0. S1. S2. … S7. S8. S9."
    # empty sentences are skipped, and a gap made only of them is not a gap
    assert sentence_window(["A.", "", "B."], [0]) == "A."
    assert sentence_window(["A.", "", "B."], [0, 2]) == "A. B."
    rand = sentence_window(s, [], "k")
    assert rand == sentence_window(s, [], "k") and 4 <= len(rand.split()) <= 10


def test_wiki_chunks_one_per_page():
    pages = [{"title": "A", "text": "A is a dog."}, {"title": "B", "sentences": ["B1.", "B2."], "evidence": [1]}]
    assert wiki_chunks(pages, "k") == ["A\nA is a dog.", "B\nB1. B2."]
    assert wiki_chunks(pages[:1], "k") == "A\nA is a dog."


def test_subject_page_prefers_longest_match_and_plain_title():
    by_key = {"Colin Kaepernick": ["Colin_Kaepernick"], "Savages": ["Savages_-LRB-2012_film-RRB-", "Savages"]}
    assert subject_page("Colin Kaepernick became a quarterback.", by_key) == "Colin_Kaepernick"
    assert subject_page("Savages was released in 2012.", by_key) == "Savages"
    assert subject_page("nothing matches here.", by_key) is None
    assert "The" not in claim_grams("The film was released.")
    assert "Kaepernick" in claim_grams("Kaepernick's team won.")


def test_strip_wiki_links_and_feverous_sentences():
    assert strip_wiki_links("[[Algebraic_logic|algebraic logic]] in [[Paris]].") == "algebraic logic in Paris."
    page = {"title": "X", "order": ["sentence_0", "table_0", "sentence_2"], "sentence_0": "[[X_team|X]] is a team.",
            "sentence_2": "It played in 1990.", "table_0": {"table": []}}
    assert feverous_sentences(page) == ["X is a team.", "", "It played in 1990."]


def test_hover_is_bool_only():
    conv = get_converter("hover")
    row = {"uid": "h", "claim": "c", "label": "NOT_SUPPORTED", "num_hops": 3, "pages": [{"title": "A", "text": "a"}, {"title": "B", "text": "b"}]}
    for i in range(50):
        (r,) = conv.convert({**row, "uid": f"h{i}"}, "train")
        assert r.type == "bool" and r.target == 0.0 and isinstance(r.state, list)


def test_fever_nei_maps_to_not_mentioned():
    conv = get_converter("fever")
    row = {"id": 0, "claim": "c", "label": "NOT ENOUGH INFO", "pages": [{"title": "A", "sentences": ["a.", "b."], "evidence": []}]}
    for i in range(50):
        (r,) = conv.convert({**row, "id": i}, "train")
        assert r.target == (0.0 if r.type == "bool" else 2)


def _nq_row(chosen: list[int], first_tag: str = "<P>") -> dict:
    words = [first_tag, "Ann", "Lee", "wrote", "the", "book", "in", "the", "year", "1990", ".", "</P>", "<P>", "It", "is", "a", "long", "novel", "about", "the", "sea", ".", "</P>"]
    return {
        "id": 7,
        "question": {"text": "who wrote the book"},
        "document": {"title": "The Book", "tokens": {"token": words, "is_html": [w.startswith("<") for w in words]}},
        "long_answer_candidates": {"start_token": [0, 12], "end_token": [12, 23], "top_level": [True, True]},
        "annotations": [{"long_answer": {"candidate_index": c}} for c in chosen],
    }


def test_nq_example_answerable_and_rules():
    ex = nq_example(_nq_row([0]), "k")
    assert ex == {"id": 7, "question": "who wrote the book", "title": "The Book", "passage": "Ann Lee wrote the book in the year 1990.", "answerable": True}
    # dev: 5 annotators, one long answer is ambiguous, two are enough
    assert nq_example(_nq_row([0, -1, -1, -1, -1]), "k") is None
    assert nq_example(_nq_row([1, 1, -1, -1, -1]), "k")["passage"] == "It is a long novel about the sea."
    # tables and lists are dropped
    assert nq_example(_nq_row([0], first_tag="<Table>"), "k") is None


def test_nq_example_unanswerable_draws_a_paragraph():
    ex = nq_example(_nq_row([-1]), "k")
    assert ex["answerable"] is False and ex["passage"] in {"Ann Lee wrote the book in the year 1990.", "It is a long novel about the sea."}
    # struct-of-lists layout of the annotations works too
    row = {**_nq_row([0]), "annotations": {"long_answer": [{"candidate_index": 0}]}}
    assert nq_example(row, "k")["answerable"] is True


def test_join_tokens():
    assert join_tokens(["Paris", "(", "France", ")", "is", "big", ",", "is", "n't", "it", "?"]) == "Paris (France) is big, isn't it?"
    assert join_tokens(["Kore", "(", "``", "the", "maiden", "''", ")"]) == 'Kore ("the maiden")'
    assert join_tokens(['"', "The", "Birds", '"', "was", "a", "1964", "single", ",", "1950s", "'", "hits"]) == '"The Birds" was a 1964 single, 1950s\' hits'
    assert join_tokens(["Pokhran", "-", "II", ",", "code", "-", "named", "NewYork", "--", "Presbyterian"]) == "Pokhran-II, code-named NewYork – Presbyterian"


def test_clean_abstract():
    assert clean_abstract("Nicotiana ( ) is a genus .") == "Nicotiana is a genus."
    assert clean_abstract("Rickson Gracie (] ; born November 21, 1958) is") == "Rickson Gracie (born November 21, 1958) is"
