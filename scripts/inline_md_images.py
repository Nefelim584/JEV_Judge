"""Build self-contained copies of the overview with every image embedded as a base64 data URI.

Reads ``docs/overview/0*.md``, writes ``docs/overview/dist/<part>.md`` (one per part) and
``docs/overview/dist/OVERVIEW.md`` (all parts, with links between parts turned into anchors).
Links to other files of the repo are left as plain text, since they do not travel with the file.
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import re
from pathlib import Path

IMAGE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
PART_LINK = re.compile(r"\[([^\]]+)\]\((0\d_[a-z_]+\.md|README\.md)\)")


def embed_images(text: str, base: Path) -> str:
    def repl(m: re.Match) -> str:
        path = (base / m.group(2)).resolve()
        mime = mimetypes.guess_type(path.name)[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"![{m.group(1)}](data:{mime};base64,{data})"

    return IMAGE.sub(repl, text)


def anchor(heading: str) -> str:
    """GitHub-style anchor of a markdown heading."""
    slug = re.sub(r"[^\w\- ]", "", heading.strip().lower())
    return slug.replace(" ", "-")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=Path("docs/overview"))
    args = ap.parse_args()
    src: Path = args.src
    out = src / "dist"
    out.mkdir(exist_ok=True)

    parts = sorted(src.glob("0*.md"))
    titles = {p.name: p.read_text().splitlines()[0].lstrip("# ") for p in parts}

    combined = [(src / "README.md").read_text().split("**For sending.**")[0].rstrip(), ""]
    for p in parts:
        text = embed_images(p.read_text(), p.parent)
        (out / p.name).write_text(text)
        print(f"{out / p.name}: {len(text) / 1e6:.2f} MB")

        def to_anchor(m: re.Match) -> str:
            target = m.group(2)
            if target == "README.md":
                return f"[{m.group(1)}](#{anchor(combined[0].splitlines()[0].lstrip('# '))})"
            return f"[{m.group(1)}](#{anchor(titles[target])})"

        combined += ["---", "", PART_LINK.sub(to_anchor, text).rstrip(), ""]

    index = PART_LINK.sub(lambda m: f"[{m.group(1)}](#{anchor(titles[m.group(2)])})", combined[0])
    combined[0] = index
    (out / "OVERVIEW.md").write_text("\n".join(combined))
    print(f"{out / 'OVERVIEW.md'}: {len(''.join(combined)) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
