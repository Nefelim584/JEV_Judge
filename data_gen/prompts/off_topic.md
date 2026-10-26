User question: $question

Passages:
$passages

A careless assistant misread the question and answered a different, neighbouring question about the same passages. First write that neighbouring question: it must be about the same topic but ask for something else, so that its answer does not answer the user's question. Then write the assistant's answer to the neighbouring question, $n_sentences sentences, fully supported by the passages. Do not mention the user's actual question.

For each sentence give "factual" and "evidence" (passage number and an exact quote from that passage). Do not use brackets like [1].

JSON: {"neighbour_question": "...", "sentences": [{"text": "...", "factual": true, "evidence": [{"chunk": 1, "quote": "..."}]}]}
