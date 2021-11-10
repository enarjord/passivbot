import logging
import sys
from logging import handlers
from typing import Optional

LOG_LEVELS = {
    "all": logging.NOTSET,
    "debug": logging.DEBUG,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "info": logging.INFO,
    "warning": logging.WARNING,
}
logging.root.setLevel(logging.DEBUG)
logging.getLogger("telegram").setLevel(logging.CRITICAL)


def setup_cli_logging(
    log_level: str, fmt: Optional[str] = None, datefmt: Optional[str] = None
) -> None:
    if fmt is None:
        fmt = "[%(asctime)s][%(levelname)-7s] - %(message)s"
    if datefmt is None:
        datefmt = "%H:%M:%S"
    handler_fmt = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setLevel(level=LOG_LEVELS.get(log_level) or logging.WARNING)
    handler.setFormatter(handler_fmt)
    logging.root.addHandler(handler)


def setup_logfile_logging(
    logfile, log_level: str, fmt: Optional[str] = None, datefmt: Optional[str] = None
) -> None:
    if fmt is None:
        fmt = "%(asctime)s,%(msecs)03d [%(name)-17s:%(lineno)-4d][%(levelname)-7s] %(message)s"
    if datefmt is None:
        datefmt = "%Y-%m-%d %H:%M:%S"
    handler_fmt = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handler = handlers.WatchedFileHandler(logfile, mode="a", encoding="utf-8", delay=False)
    handler.setLevel(level=LOG_LEVELS.get(log_level) or logging.WARNING)
    handler.setFormatter(handler_fmt)
    logging.root.addHandler(handler)
