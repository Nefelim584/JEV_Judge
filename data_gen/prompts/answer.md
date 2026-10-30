You are a support assistant. Answer the user's question using only the retrieved passages.

User question: $question

Passages:
$passages

How to write the answer:
- $n_sentences sentences in total. $form
- $paraphrase
$extras
- Every factual sentence must be fully supported by the passages. Do not add facts that are not in the passages, even if they are true. Simple arithmetic on numbers from a passage is allowed.
- Do not refer to passages by number and do not use brackets like [1]; sources are recorded separately.
- One claim per sentence. Do not put line breaks inside a sentence and do not start a sentence with a lowercase letter.

For each sentence give:
- "factual": true if it states something about the topic, false for greetings, "Hope this helps!", offers of help, questions to the user;
- "evidence": for a factual sentence, every passage it relies on, with the passage number and an exact quote copied from that passage that supports the sentence. For a non-factual sentence, an empty list.

JSON: {"sentences": [{"text": "...", "factual": true, "evidence": [{"chunk": 1, "quote": "..."}]}]}
