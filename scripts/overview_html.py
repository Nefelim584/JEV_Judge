"""One self-contained HTML file of the overview: all five parts, charts embedded (base64), styles inline,
a table of contents. Opens in any browser, survives forwarding by mail or chat, and prints to PDF.

    python scripts/inline_md_images.py                    # first: docs/overview/dist/OVERVIEW.md
    uv run --extra reports python scripts/overview_html.py  # then: docs/overview/dist/OVERVIEW.html

Heading ids follow ``inline_md_images.anchor``, so the links between parts of OVERVIEW.md keep working.
"""

from __future__ import annotations

import argparse
import html
from pathlib import Path

from inline_md_images import anchor
from markdown_it import MarkdownIt

CSS = """
:root { --text: #1f2328; --muted: #59636e; --line: #d1d9e0; --soft: #f6f8fa; --accent: #2a78d6; }
* { box-sizing: border-box; }
body { margin: 0; background: #fff; color: var(--text);
       font: 16px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
.layout { display: flex; max-width: 1280px; margin: 0 auto; }
nav { position: sticky; top: 0; align-self: flex-start; width: 270px; max-height: 100vh; overflow-y: auto;
      padding: 24px 16px; font-size: 13.5px; border-right: 1px solid var(--line); }
nav a { display: block; color: var(--muted); text-decoration: none; padding: 2px 0; }
nav a:hover { color: var(--accent); }
nav .h1 { font-weight: 600; color: var(--text); margin-top: 10px; }
nav .h3 { padding-left: 12px; }
main { flex: 1; min-width: 0; padding: 24px 40px 80px; max-width: 980px; }
h1 { font-size: 1.9em; border-bottom: 1px solid var(--line); padding-bottom: .3em; margin-top: 1.6em; }
h2 { font-size: 1.45em; border-bottom: 1px solid var(--line); padding-bottom: .25em; margin-top: 1.5em; }
h3 { font-size: 1.15em; margin-top: 1.4em; }
a { color: var(--accent); }
blockquote { margin: 1em 0; padding: .2em 1em; color: var(--muted); border-left: 4px solid var(--line); }
table { border-collapse: collapse; display: block; overflow-x: auto; margin: 1em 0; font-size: 14px; }
th, td { border: 1px solid var(--line); padding: 5px 10px; vertical-align: top; }
th { background: var(--soft); text-align: left; }
tr:nth-child(2n) td { background: #fbfcfd; }
code { background: var(--soft); padding: .1em .35em; border-radius: 4px; font-size: 88%; }
pre { background: var(--soft); padding: 14px; border-radius: 6px; overflow-x: auto; line-height: 1.35; }
pre code { background: none; padding: 0; font-size: 13px; }
img { max-width: 100%; height: auto; display: block; margin: 1em auto; border: 1px solid var(--line); border-radius: 6px; }
hr { border: 0; border-top: 2px solid var(--line); margin: 3em 0 1em; }
@media (max-width: 900px) { nav { display: none; } main { padding: 16px; } }
@media print {
  nav { display: none; } main { max-width: none; padding: 0; }
  hr { page-break-after: always; border: 0; margin: 0; }
  img, table, pre { page-break-inside: avoid; }
  table { display: table; }
}
"""


def render(md_text: str) -> tuple[str, list[tuple[int, str, str]]]:
    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    tokens = md.parse(md_text)
    toc = []
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open":
            title = tokens[i + 1].content
            tok.attrSet("id", anchor(title))
            level = int(tok.tag[1])
            if level <= 3:
                toc.append((level, title, anchor(title)))
    return md.renderer.render(tokens, md.options, {}), toc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, default=Path("docs/overview/dist/OVERVIEW.md"))
    ap.add_argument("--out", type=Path, default=Path("docs/overview/dist/OVERVIEW.html"))
    args = ap.parse_args(argv)

    text = args.src.read_text()
    body, toc = render(text)
    title = text.splitlines()[0].lstrip("# ")
    nav = "\n".join(f'<a class="h{lvl}" href="#{a}">{html.escape(t)}</a>' for lvl, t, a in toc)
    page = (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<title>{html.escape(title)}</title>\n'
            f"<style>{CSS}</style>\n</head>\n<body>\n<div class=\"layout\">\n<nav>{nav}</nav>\n<main>\n{body}</main>\n</div>\n"
            "</body>\n</html>\n")
    args.out.write_text(page)
    print(f"{args.out}: {len(page) / 1e6:.2f} MB, {body.count('<img')} images, {len(toc)} headings")


if __name__ == "__main__":
    main()
