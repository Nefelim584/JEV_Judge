"""Reply schemas, one per prompt. llama-server turns them into a grammar, so a reply is always
well-formed JSON of this shape; content checks live in ``checks.py`` and the stages."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Label = Literal["supports", "contradicts", "silent"]


class Topic(BaseModel):
    name: str
    summary: str
    aspects: list[str] = Field(min_length=3, max_length=6)


class Topics(BaseModel):
    topics: list[Topic] = Field(min_length=1)


class Section(BaseModel):
    heading: str
    text: str


class Document(BaseModel):
    title: str
    sections: list[Section] = Field(min_length=1)


class Question(BaseModel):
    question: str


class Passage(BaseModel):
    text: str


class Evidence(BaseModel):
    chunk: int
    quote: str


class AnswerSentence(BaseModel):
    text: str
    factual: bool
    evidence: list[Evidence]


class Answer(BaseModel):
    sentences: list[AnswerSentence] = Field(min_length=1)


class Refusal(BaseModel):
    sentences: list[str] = Field(min_length=1, max_length=3)


class OffTopic(BaseModel):
    neighbour_question: str
    sentences: list[AnswerSentence] = Field(min_length=1)


class Rewrite(BaseModel):
    sentence: str


class SentenceLabel(BaseModel):
    id: int
    label: Label
    quote: str


class Labels(BaseModel):
    labels: list[SentenceLabel]


class FullLabel(BaseModel):
    id: int
    supported_together: bool


class FullLabels(BaseModel):
    labels: list[FullLabel]


class Verdict(BaseModel):
    id: int
    verdict: Literal["TRUE", "FALSE", "NOT_STATED"]


class Verdicts(BaseModel):
    verdicts: list[Verdict]


class Answerable(BaseModel):
    answerable: bool


VERDICT_TO_LABEL = {"TRUE": "supports", "FALSE": "contradicts", "NOT_STATED": "silent"}
