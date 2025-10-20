"""Utilities for configuring consistent logging across Passivbot."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional

TRACE_LEVEL = 5
TRACE_LEVEL_NAME = "TRACE"

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%S"


def _ensure_trace_level() -> None:
    """Register the TRACE log level on the logging module if missing."""
    if logging.getLevelName(TRACE_LEVEL) != TRACE_LEVEL_NAME:
        logging.addLevelName(TRACE_LEVEL, TRACE_LEVEL_NAME)
    if getattr(logging, TRACE_LEVEL_NAME, None) != TRACE_LEVEL:
        setattr(logging, TRACE_LEVEL_NAME, TRACE_LEVEL)

    if not hasattr(logging.Logger, "trace"):

        def trace(self: logging.Logger, msg: str, *args, **kwargs) -> None:
            if self.isEnabledFor(TRACE_LEVEL):
                self._log(TRACE_LEVEL, msg, args, **kwargs)

        logging.Logger.trace = trace  # type: ignore[attr-defined]


def _normalize_debug(debug: Optional[int | str]) -> int:
    if debug is None:
        return 1
    if isinstance(debug, bool):
        return 1 if debug else 0
    try:
        return int(debug)
    except (TypeError, ValueError):
        return 1


def _debug_to_level(debug: int) -> int:
    if debug <= 0:
        return logging.WARNING
    if debug == 1:
        return logging.INFO
    if debug == 2:
        return logging.DEBUG
    return TRACE_LEVEL


def configure_logging(
    debug: Optional[int | str] = 1,
    *,
    log_file: Optional[str] = None,
    rotation: bool = False,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    stream: bool = True,
    fmt: str = DEFAULT_FORMAT,
    datefmt: str = DEFAULT_DATEFMT,
) -> None:
    """Initialise the root logger based on Passivbot's debug settings."""
    _ensure_trace_level()
    debug_level = _normalize_debug(debug)
    numeric_level = _debug_to_level(debug_level)

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handlers: list[logging.Handler] = []

    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(numeric_level)
        handlers.append(stream_handler)

    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        if rotation:
            file_handler = RotatingFileHandler(path, maxBytes=max_bytes, backupCount=backup_count)
        else:
            file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(numeric_level)
        handlers.append(file_handler)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    for existing in list(root.handlers):
        root.removeHandler(existing)
        existing.close()

    for handler in handlers:
        root.addHandler(handler)
