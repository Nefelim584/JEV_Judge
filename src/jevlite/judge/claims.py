"""Rule-based sentence splitting for the judge (TODO 2.12, step 1).

First version, used for length statistics. Phase 8 refines it (abbreviations, lists, quotes) and
adds the non-factual filter. Training data must be split by the same function as inference.
"""

from __future__ import annotations

import re

# A sentence ends at . ! ? (optionally followed by a closing quote or bracket) before whitespace and an
# uppercase letter, digit, quote or bracket. List items and blank lines also end a sentence.
_SENTENCE_END = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"')\]]))\s+(?=[A-Z0-9\"'(\[])")
_LINE_BREAK = re.compile(r"\n\s*(?:[-*•]|\d+[.)])?\s*")


def split_sentences(text: str) -> list[str]:
    sentences = []
    for line in _LINE_BREAK.split(text):
        sentences += [s.strip() for s in _SENTENCE_END.split(line) if s.strip()]
    return sentences
