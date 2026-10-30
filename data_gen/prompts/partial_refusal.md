You are a support assistant. The user's question has more than one part.

User question: $question

Passages:
$passages

Answer only ONE part of the question, using the passages. For the other part, write exactly one sentence saying that the documents do not contain that information (even though they might). $n_sentences sentences in total.

For each sentence give "factual" and "evidence" (passage number and an exact quote from that passage). The sentence that says the information is missing is factual and has an empty evidence list. Non-factual sentences (pleasantries) have factual false and empty evidence. Do not use brackets like [1].

JSON: {"sentences": [{"text": "...", "factual": true, "evidence": [{"chunk": 1, "quote": "..."}]}]}
