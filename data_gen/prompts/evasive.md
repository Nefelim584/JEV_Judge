A user of $docs asks:

$question

Write an evasive reply of $n_sentences sentences: polite, on the general topic, but it never actually answers the question and gives no specific details (no numbers, dates or conditions). Generic advice like "it depends on your situation" is typical.

For each sentence give "factual": true if it states something about the topic (even vaguely), false for pure pleasantries; and an empty "evidence" list.

JSON: {"sentences": [{"text": "...", "factual": false, "evidence": []}]}
