"""Logging for notebooks and scripts, via loguru.

    from jevlite.log import logger, setup_logging
    setup_logging(run_dir / "run.log")   # once, in the first notebook cell
    logger.info("step {} loss {:.4f}", step, loss)

Library modules log through the same ``logger``; nothing is shown until ``setup_logging`` (or loguru's
default stderr sink) is in place.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

__all__ = ["logger", "setup_logging"]

_CONSOLE_FORMAT = "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <level>{message}</level>"
_FILE_FORMAT = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <7} | {name}:{function}:{line} | {message}"


def setup_logging(file: str | Path | None = None, level: str = "INFO", file_level: str = "DEBUG") -> None:
    """Replace loguru's sinks with a console sink and, optionally, a file sink.

    The console sink writes to stdout: Jupyter renders stderr with a red background, which makes
    every info line look like an error. Safe to call again (e.g. when a cell is re-run).
    """
    logger.remove()
    logger.add(sys.stdout, level=level, format=_CONSOLE_FORMAT, colorize=True)
    if file is not None:
        path = Path(file)
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(path, level=file_level, format=_FILE_FORMAT, encoding="utf-8")
