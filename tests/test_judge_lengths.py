import pytest

from jevlite.judge.claims import split_sentences
from jevlite.judge.export import JudgeSample
from jevlite.judge.lengths import length_stats, to_markdown


def test_split_sentences():
    text = 'Files stay for 30 days. You can restore them [1]. "Really?" Yes!\n- first item\n- second item'
    assert split_sentences(text) == ["Files stay for 30 days.", "You can restore them [1].", '"Really?"', "Yes!", "first item", "second item"]
    assert split_sentences("version 2.5 is out") == ["version 2.5 is out"]
    assert split_sentences("") == []


def test_judge_sample_accepts_chunk_objects():
    s = JudgeSample.model_validate({"id": "a", "question": "q", "chunks": [{"id": "c1", "text": "t1"}, "t2"], "answer": "x"})
    assert s.chunks == ["t1", "t2"]


def test_length_stats(tokenizer):
    samples = [
        JudgeSample(id=str(i), question="How long are files kept?", chunks=["Files are kept for 30 days. " * (10 * i + 1)] * 4, answer="They are kept for 30 days. Hope this helps!")
        for i in range(5)
    ]
    stats = length_stats(samples, tokenizer)
    assert stats["n_samples"] == 5
    assert stats["chunk"]["n"] == 20 and stats["n_chunks"]["mean"] == 4
    assert stats["answer_sentences"]["mean"] == 2
    # Per-chunk sequences are longer than the chunk itself (header + special tokens), the full pass longer still.
    assert stats["faithfulness_per_chunk"]["mean"] > stats["chunk"]["mean"]
    assert stats["faithfulness_full"]["max"] > stats["faithfulness_per_chunk"]["max"]
    assert 0 < stats["faithfulness_full"]["share_gt_512"] <= 1
    assert "| faithfulness_full |" in to_markdown(stats)
