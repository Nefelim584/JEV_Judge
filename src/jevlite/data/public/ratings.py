"""Human rubric scores and preferences: the Score primitive (``DATASETS.md`` section 0).

Score has little variety in public data (HelpSteer2/3), so every record also draws a level scheme:
numbers, words, or a coarser 3-level merge of the same scale.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from ..unified import Record
from .base import Converter, Template, hf_rows, pick_template, unit

# Level schemes for a 0–4 scale: (id, levels, map from the source value to a level index).
_FIVE = [
    ("digits", ["0", "1", "2", "3", "4"], [0, 1, 2, 3, 4]),
    ("words", ["very low", "low", "medium", "high", "very high"], [0, 1, 2, 3, 4]),
    ("coarse", ["low", "medium", "high"], [0, 0, 1, 2, 2]),
]
# Response 2 compared with response 1, for HelpSteer3's −3…+3 preference.
_SEVEN = [
    ("words", ["much worse", "worse", "slightly worse", "about the same", "slightly better", "better", "much better"],
     [0, 1, 2, 3, 4, 5, 6]),
    ("coarse", ["worse", "about the same", "better"], [0, 0, 0, 1, 2, 2, 2]),
]
_SCHEME_WEIGHTS = {"digits": 0.4, "words": 0.3, "coarse": 0.3}


def pick_scheme(schemes: Sequence[tuple], *key) -> tuple:
    """A level scheme by stable weighted draw (weights renormalised over the schemes given)."""
    total = sum(_SCHEME_WEIGHTS[s[0]] for s in schemes)
    x = unit("scheme", *key) * total
    for s in schemes:
        x -= _SCHEME_WEIGHTS[s[0]]
        if x < 0:
            return s
    return schemes[-1]


def chat_text(turns: Sequence[dict]) -> str:
    return "\n\n".join(f"{t['role'].capitalize()}: {t['content'].strip()}" for t in turns)


# HelpSteer2 attributes (rubric wording from the dataset card), each with its own templates.
HELPSTEER2 = {
    "helpfulness": (
        Template("overall", "How helpful is the assistant's response overall?"),
        Template("useful", "Rate how useful the response is for the user's request."),
        Template("meets_need", "How well does the response meet the user's need?"),
    ),
    "correctness": (
        Template("facts", "How correct is the response: does it include all pertinent facts without errors?"),
        Template("accurate", "Rate the factual accuracy and completeness of the response."),
        Template("errors", "How free of factual errors and omissions is the response?"),
    ),
    "coherence": (
        Template("clear", "How clear and consistent is the response?"),
        Template("coherent", "Rate the coherence of the response."),
        Template("readable", "How logically organised and easy to follow is the response?"),
    ),
    "complexity": (
        Template("depth", "How much expertise does it take to write this response?"),
        Template("sophisticated", "Rate how sophisticated the language and content of the response are."),
        Template("expert", "How much domain knowledge does the response require from its writer?"),
    ),
    "verbosity": (
        Template("detail", "How verbose is the response relative to what the user asked for?"),
        Template("length", "Rate the amount of detail in the response."),
        Template("wordy", "How long and detailed is the response compared with the request?"),
    ),
}


class HelpSteer2(Converter):
    """Each (prompt, response) gives one Score record per attribute (5 per row)."""

    name = "helpsteer2"
    domain = "assistant_chat"
    family = "rating"
    license = "CC BY 4.0"
    origin = "hf:nvidia/HelpSteer2"

    def rows(self, source_split: str) -> Iterable[dict]:
        for i, row in enumerate(hf_rows("nvidia/HelpSteer2", None, source_split, self.streaming)):
            yield {**row, "idx": i}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        # Multi-turn prompts separate turns with "<extra_id_1>User" / "<extra_id_1>Assistant".
        prompt = row["prompt"].replace("<extra_id_1>", "\n").strip()
        state = f"User: {prompt}\n\nAssistant: {row['response'].strip()}"
        out = []
        for attr, templates in HELPSTEER2.items():
            key = (source_split, row["idx"], attr)
            scheme_id, levels, to_level = pick_scheme(_FIVE, self.name, *key)
            tpl, split = pick_template(templates, self.target_split(source_split, row["idx"]), self.name, *key)
            out.append(
                self.record(
                    key=key, split=split, state=state, type="score", prompt=tpl.prompt, candidates=list(levels),
                    target=to_level[row[attr]], criterion=attr, template=f"helpsteer2/{attr}/{tpl.id}/{scheme_id}",
                )
            )
        return out


PREFERENCE = (
    Template("compare", "How does response 2 compare with response 1?"),
    Template("quality", "Compared with response 1, how good is response 2?"),
    Template("judge", "Judge response 2 against response 1."),
)


class HelpSteer3(Converter):
    """Pairwise preference (−3…+3) as a Score of response 2 relative to response 1. The multilingual
    domain is dropped (English only for v1)."""

    name = "helpsteer3"
    domain = "assistant_chat"
    family = "preference"
    license = "CC BY 4.0"
    origin = "hf:nvidia/HelpSteer3/preference"

    def rows(self, source_split: str) -> Iterable[dict]:
        for i, row in enumerate(hf_rows("nvidia/HelpSteer3", "preference", source_split, self.streaming)):
            yield {**row, "idx": i}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        if row["domain"] == "multilingual":
            return []
        key = (source_split, row["idx"])
        state = (
            f"{chat_text(row['context'])}\n\nResponse 1:\n{row['response1'].strip()}"
            f"\n\nResponse 2:\n{row['response2'].strip()}"
        )
        scheme_id, levels, to_level = pick_scheme(_SEVEN, self.name, *key)
        tpl, split = pick_template(PREFERENCE, self.target_split(source_split, *key), self.name, *key)
        return [
            self.record(
                key=key, split=split, state=state, type="score", prompt=tpl.prompt, candidates=list(levels),
                target=to_level[row["overall_preference"] + 3], domain=f"assistant_{row['domain']}",
                criterion="preference", template=f"preference/{tpl.id}/{scheme_id}",
            )
        ]


SUMMARY_CHOICE = (
    Template("better", "Which summary is better?"),
    Template("prefer", "Which summary would a careful reader prefer?"),
    Template("captures", "Which summary captures the text more accurately and concisely?"),
)


class SummarizeFromFeedback(Converter):
    """Pairwise human comparisons of summaries (TL;DR posts and CNN/DM articles): Choice K = 2."""

    name = "summarize_from_feedback"
    domain = "reddit"
    family = "preference"
    license = "Modified MIT (labels); TL;DR posts CC BY 4.0"
    origin = "hf:openai/summarize_from_feedback/comparisons"

    def rows(self, source_split: str) -> Iterable[dict]:
        for i, row in enumerate(hf_rows(
            "openai/summarize_from_feedback", "comparisons", source_split, self.streaming, parquet_export=True
        )):
            yield {**row, "idx": i}

    def convert(self, row: dict, source_split: str) -> Iterable[Record]:
        info = row["info"]
        text = info.get("post") or info.get("article")
        summaries = [s["text"].strip() for s in row["summaries"]]
        if not text or len(summaries) != 2 or not all(summaries) or summaries[0] == summaries[1]:
            return []
        key = (source_split, row["idx"])
        state = f"Text:\n{text.strip()}\n\nSummary 1:\n{summaries[0]}\n\nSummary 2:\n{summaries[1]}"
        tpl, split = pick_template(SUMMARY_CHOICE, self.target_split(source_split, *key), self.name, *key)
        return [
            self.record(
                key=key, split=split, state=state, type="choice", prompt=tpl.prompt,
                candidates=["summary 1", "summary 2"], target=int(row["choice"]),
                domain="reddit" if info.get("post") else "news", criterion="preference",
                template=f"summary_choice/{tpl.id}",
            )
        ]
