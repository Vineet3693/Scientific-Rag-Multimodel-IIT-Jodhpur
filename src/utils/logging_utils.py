"""
Logging Utilities for Scientific Multimodal RAG.

Provides a configured logger factory with colored console output
using colorlog. Every module in the project should call
``get_logger(__name__)`` at the top of the file to obtain a
module-level logger with consistent formatting.

Example:
    >>> from src.utils.logging_utils import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Pipeline started")
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

try:
    import colorlog
    _HAS_COLORLOG = True
except ImportError:
    _HAS_COLORLOG = False

# ---------------------------------------------------------------------------
# Default format strings
# ---------------------------------------------------------------------------
_CONSOLE_FORMAT = (
    "%(log_color)s[%(levelname)s]%(reset)s "
    "%(asctime)s | %(name)s | %(message)s"
)
_PLAIN_FORMAT = (
    "[%(levelname)s] %(asctime)s | %(name)s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Track handlers we have already attached so that repeated calls to
# ``get_logger`` with the same name do not duplicate handlers.
_registered_loggers: set[str] = set()


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Return a configured logger with optional colored console output.

    Args:
        name: Logger name — typically ``__name__`` of the calling module.
        level: Logging threshold (default ``logging.INFO``).
        log_file: Optional path to a file handler.  When *None*, only
            console output is produced.

    Returns:
        A ``logging.Logger`` instance ready for use.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on repeated calls.
    if name in _registered_loggers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    # ── Console handler ──
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if _HAS_COLORLOG:
        formatter = colorlog.ColoredFormatter(
            _CONSOLE_FORMAT,
            datefmt=_DATE_FORMAT,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )
    else:
        formatter = logging.Formatter(_PLAIN_FORMAT, datefmt=_DATE_FORMAT)

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── Optional file handler ──
    if log_file is not None:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(_PLAIN_FORMAT, datefmt=_DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    _registered_loggers.add(name)
    return logger
