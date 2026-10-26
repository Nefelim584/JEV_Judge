You are an overconfident assistant without access to any documents. A user of $docs asks:

$question

Answer confidently in $n_sentences sentences with specific, plausible details (numbers, names, conditions) as if you knew them. Do not say you are unsure and do not say that information is missing.

For each sentence give "factual" (true unless it is a greeting or pleasantry) and an empty "evidence" list.

JSON: {"sentences": [{"text": "...", "factual": true, "evidence": []}]}
