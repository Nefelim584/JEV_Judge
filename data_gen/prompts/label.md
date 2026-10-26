You check sentences of an assistant's answer against ONE passage. Judge only by this passage and the user's question; ignore what you know about the world.

User question: $question

Passage:
$passage

Sentences:
$sentences

For each sentence choose:
- "supports": someone who has only this passage and the question would agree the whole sentence is true. Paraphrase, resolving "it" or "they" from the question, simple arithmetic on numbers in the passage, and a weaker statement ("usually" when the passage says "always") count as support.
- "contradicts": the passage states something incompatible with the sentence (another number, date, name in the same role, or the opposite fact).
- "silent": anything else, including: true in the real world but not in the passage; only part of the sentence is supported; the sentence is stronger than the passage ("will" vs "may", a dropped condition or exception).

For "supports" and "contradicts" copy the exact words from the passage that decide it into "quote"; for "silent" use an empty quote.

JSON: {"labels": [{"id": 1, "label": "silent", "quote": ""}]}
