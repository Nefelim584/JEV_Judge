"""Markdown reports with charts for comparisons (eval reports of several runs) and training runs.

Static charts are matplotlib PNGs embedded in the markdown; each has an interactive plotly twin
(HTML, with hover) linked under it. Every chart is paired with a table holding the same numbers.
Needs the ``reports`` extra: ``uv sync --extra reports``.
"""
