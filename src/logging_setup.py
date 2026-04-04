"""Utilities for configuring consistent logging across Passivbot."""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, Sequence

TRACE_LEVEL = 5
TRACE_LEVEL_NAME = "TRACE"

DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(message)s"
DEFAULT_FORMAT_WITH_PREFIX = "%(asctime)s %(levelname)-8s [%(log_prefix)s] %(message)s"
DEFAULT_DATEFMT = "%Y-%m-%dT%H:%M:%SZ"
_LAST_LOG_ACTIVITY_MONOTONIC = time.monotonic()
DEFAULT_LOG_FILENAME_MAX_LEN = 100


class PrefixFilter(logging.Filter):
    """Filter that adds a log_prefix attribute to log records."""

    def __init__(self, prefix: str = ""):
        super().__init__()
        self.prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        record.log_prefix = self.prefix
        return True


class ActivityFilter(logging.Filter):
    """Filter that tracks the most recent emitted log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        mark_log_activity()
        return True


def mark_log_activity() -> None:
    global _LAST_LOG_ACTIVITY_MONOTONIC
    _LAST_LOG_ACTIVITY_MONOTONIC = time.monotonic()


def get_last_log_activity_monotonic() -> float:
    return _LAST_LOG_ACTIVITY_MONOTONIC


_LOG_LEVEL_ALIASES = {
    "warning": 0,
    "warn": 0,
    "w": 0,
    "info": 1,
    "i": 1,
    "debug": 2,
    "d": 2,
    "trace": 3,
    "t": 3,
}


def normalize_log_level(value, default=None):
    """Return normalized log level 0-3 or default when invalid/missing."""
    if value is None:
        return default
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in _LOG_LEVEL_ALIASES:
            return _LOG_LEVEL_ALIASES[cleaned]
        try:
            value = float(cleaned)
        except ValueError:
            return default
    try:
        level = int(float(value))
    except (TypeError, ValueError):
        return default
    return max(0, min(level, 3))


def resolve_log_level(cli_value, config_value, fallback=1):
    """Resolve final log level from CLI override and config value."""
    cli_level = normalize_log_level(cli_value, None)
    if cli_level is not None:
        return cli_level
    cfg_level = normalize_log_level(config_value, None)
    if cfg_level is not None:
        return cfg_level
    return fallback


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
    level = normalize_log_level(debug, None)
    if level is None:
        return 1
    return level


def _debug_to_level(debug: int) -> int:
    if debug <= 0:
        return logging.WARNING
    if debug == 1:
        return logging.INFO
    if debug == 2:
        return logging.DEBUG
    return TRACE_LEVEL


def sanitize_log_filename(text: str, *, max_len: int = DEFAULT_LOG_FILENAME_MAX_LEN) -> str:
    """Return a filesystem-safe filename fragment."""
    sanitized = re.sub(r"[\s/\\]", "_", text)
    sanitized = re.sub(r'[<>:"|?*]', "", sanitized)
    sanitized = sanitized.strip(". ")
    if len(sanitized) > max_len:
        sanitized = sanitized[:max_len]
    return sanitized or "log"


def create_command_log_filename(
    command_args: Sequence[object], *, timestamp: Optional[datetime] = None
) -> str:
    """Return a timestamped log filename for a command invocation."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
    elif timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    command_str = " ".join(str(part) for part in command_args)
    sanitized_command = sanitize_log_filename(command_str)
    prefix = timestamp.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{sanitized_command}.log"


def build_command_log_path(
    command_args: Sequence[object], log_dir: str | Path, *, timestamp: Optional[datetime] = None
) -> Path:
    """Return the log file path for a command invocation under the given directory."""
    return Path(log_dir).expanduser() / create_command_log_filename(command_args, timestamp=timestamp)


def resolve_command_logging_options(
    logging_config: dict,
    command_args: Sequence[object],
    *,
    default_persist: bool = False,
) -> tuple[Optional[str], bool, int, int]:
    """Resolve file-logging options from config for a command invocation."""
    persist_to_file = bool(logging_config.get("persist_to_file", default_persist))
    rotation_enabled = bool(logging_config.get("rotation_enabled", False))

    try:
        rotation_max_mb = float(logging_config.get("rotation_max_mb", 10))
    except (TypeError, ValueError) as exc:
        raise ValueError("logging.rotation_max_mb must be a positive number") from exc
    if rotation_max_mb <= 0:
        raise ValueError("logging.rotation_max_mb must be greater than 0")

    try:
        rotation_backups = int(logging_config.get("rotation_backups", 5))
    except (TypeError, ValueError) as exc:
        raise ValueError("logging.rotation_backups must be a non-negative integer") from exc
    if rotation_backups < 0:
        raise ValueError("logging.rotation_backups must be a non-negative integer")

    log_file = None
    if persist_to_file:
        log_dir = logging_config.get("dir", "logs")
        if not isinstance(log_dir, str) or not log_dir.strip():
            raise ValueError("logging.dir must be a non-empty string when file logging is enabled")
        log_file = str(build_command_log_path(command_args, log_dir.strip()))

    return log_file, rotation_enabled, int(rotation_max_mb * 1024 * 1024), rotation_backups


def configure_logging(
    debug: Optional[int | str] = 1,
    *,
    log_file: Optional[str] = None,
    rotation: bool = False,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    stream: bool = True,
    fmt: Optional[str] = None,
    datefmt: str = DEFAULT_DATEFMT,
    prefix: Optional[str] = None,
) -> None:
    """Initialise the root logger based on Passivbot's debug settings.

    Args:
        debug: Logging level (0=warning, 1=info, 2=debug, 3=trace)
        log_file: Optional path to log file
        rotation: Enable log rotation
        max_bytes: Max bytes per log file before rotation
        backup_count: Number of backup files to keep
        stream: Enable console output
        fmt: Custom log format (defaults based on prefix)
        datefmt: Date format string
        prefix: Optional prefix to add to all log messages (e.g., exchange name)
    """
    _ensure_trace_level()
    debug_level = _normalize_debug(debug)
    numeric_level = _debug_to_level(debug_level)

    # Choose format based on prefix
    if fmt is None:
        fmt = DEFAULT_FORMAT_WITH_PREFIX if prefix else DEFAULT_FORMAT

    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    formatter.converter = time.gmtime
    handlers: list[logging.Handler] = []

    # Create prefix filter if needed
    prefix_filter = PrefixFilter(prefix or "") if prefix else None
    activity_filter = ActivityFilter()

    if stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(numeric_level)
        stream_handler.addFilter(activity_filter)
        if prefix_filter:
            stream_handler.addFilter(prefix_filter)
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
        file_handler.addFilter(activity_filter)
        if prefix_filter:
            file_handler.addFilter(prefix_filter)
        handlers.append(file_handler)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    for existing in list(root.handlers):
        root.removeHandler(existing)
        existing.close()

    for handler in handlers:
        root.addHandler(handler)

    # Configure CCXT logger to only log at TRACE level.
    # CCXT logs full API request/response payloads at DEBUG, which is too noisy.
    # These payloads belong at TRACE (level 3) per log_analysis_prompt.md guidelines.
    ccxt_logger = logging.getLogger("ccxt")
    if debug_level >= 3:
        # TRACE mode: allow CCXT logs through
        ccxt_logger.setLevel(TRACE_LEVEL)
    else:
        # DEBUG and below: suppress CCXT's noisy API payloads
        # Set to WARNING so only actual warnings/errors from CCXT are shown
        ccxt_logger.setLevel(logging.WARNING)
