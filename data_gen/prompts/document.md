Write a document of this kind: $docs.
Topic: $topic — $topic_summary
Focus on this aspect: $aspect
$real_instruction

The document has exactly $n_sections sections, in this order:
$sections_spec

Section formats:
- prose: one or more plain paragraphs
- markdown: text with a markdown heading, emphasis or sub-headings
- list: a bulleted or numbered list (for example the steps of a procedure)
- table: a markdown table or "key: value" lines
- faq: one question followed by its answer
- code_config: a config snippet or a function signature, with a short explanation

Requirements:
- Be specific: concrete numbers, dates, names, conditions, limits and exceptions. These details are what readers will ask about.
- Every section is self-contained and about its own sub-subject, but all belong to the same document.
- No personal data of real people. Medicine names, if any, must be invented.
- Keep each section close to its target length.

JSON: {"title": "...", "sections": [{"heading": "...", "text": "..."}]}
