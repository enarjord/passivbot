import logging
import sys
from collections import deque
from logging import handlers
from typing import Any
from typing import Dict
from typing import Optional
from typing import Type

from typing_extensions import Deque

LOG_LEVELS = {
    "all": logging.NOTSET,
    "debug": logging.DEBUG,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
    "info": logging.INFO,
    "warning": logging.WARNING,
}
logging.root.setLevel(logging.DEBUG)

# Store an instance of the current logging logger class
LOGGING_LOGGER_CLASS: Type[logging.Logger] = logging.getLoggerClass()


class LogRecord(logging.LogRecord):
    wipe_line: bool


class TemporaryLoggingHandler(logging.NullHandler):
    """
    This logging handler will store all the log records up to its maximum
    queue size at which stage the first messages stored will be dropped.
    Should only be used as a temporary logging handler, while the logging
    system is not fully configured.
    Once configured, pass any logging handlers that should have received the
    initial log messages to the function
    :func:`TemporaryLoggingHandler.sync_with_handlers` and all stored log
    records will be dispatched to the provided handlers.
    """

    def __init__(self, level=logging.NOTSET, max_queue_size=10000):
        self.__max_queue_size: int = max_queue_size
        super().__init__(level=level)
        self.__messages: Deque[logging.LogRecord] = deque(maxlen=max_queue_size)

    def handle(self, record):
        self.acquire()
        self.__messages.append(record)
        self.release()

    def sync_with_handlers(self, handlers=()):
        """
        Sync the stored log records to the provided log handlers.
        """
        if not handlers:
            return

        while self.__messages:
            record = self.__messages.popleft()
            for handler in handlers:
                if handler.level > record.levelno:
                    # If the handler's level is higher than the log record one,
                    # it should not handle the log record
                    continue
                handler.handle(record)


LOGGING_TEMP_HANDLER = TemporaryLoggingHandler(logging.WARNING)


class ConsoleHandler(logging.StreamHandler):

    _previous_record_wiped: bool = False

    def format(self, record):
        msg = super().format(record)
        previous_record_wiped = self._previous_record_wiped
        self._previous_record_wiped = record.wipe_line
        if record.wipe_line and previous_record_wiped:
            msg = f"\r{msg}"
        elif previous_record_wiped:
            msg = f"\n{msg}"
        return msg

    def emit(self, record):
        """
        Emit a record.
        If a formatter is specified, it is used to format the record.
        The record is then written to the stream with a trailing newline.  If
        exception information is present, it is formatted using
        traceback.print_exception and appended to the stream.  If the stream
        has an 'encoding' attribute, it is used to determine how to do the
        output to the stream.
        """
        try:
            msg = self.format(record)
            stream = self.stream
            # issue 35046: merged two stream.writes into one.
            if record.wipe_line is False:
                msg = f"{msg}{self.terminator}"
            stream.write(msg)
            self.flush()
        except RecursionError:  # See issue 36272
            raise
        except Exception:
            self.handleError(record)


class PassivBotLoggingClass(LOGGING_LOGGER_CLASS):  # type: ignore[valid-type,misc]
    def _log(
        self,
        level,
        msg,
        args,
        exc_info=None,
        extra: Optional[Dict[Any, Any]] = None,
        stack_info: bool = False,
        stacklevel: int = 1,
        wipe_line: bool = False,
    ):
        if extra is None:
            extra = {}

        extra["wipe_line"] = wipe_line

        super()._log(
            level,
            msg,
            args,
            exc_info=exc_info,
            extra=extra,
            stack_info=stack_info,
            stacklevel=stacklevel,
        )

    def makeRecord(
        self,
        name,
        level,
        fn,
        lno,
        msg,
        args,
        exc_info,
        func=None,
        extra=None,
        sinfo=None,
    ):
        # Let's remove wipe_line from extra
        wipe_line = extra.pop("wipe_line")

        if not extra:
            # If nothing else is in extra, make it None
            extra = None

        logrecord: LogRecord = super().makeRecord(
            name, level, fn, lno, msg, args, exc_info, func=func, extra=extra, sinfo=sinfo
        )
        logrecord.wipe_line = wipe_line
        return logrecord


def set_logger_class() -> None:
    """
    Override python's logging logger class. This should be called as soon as possible
    """
    if logging.getLoggerClass() is not PassivBotLoggingClass:

        logging.setLoggerClass(PassivBotLoggingClass)
        logging.setLogRecordFactory(LogRecord)

        logging.root.addHandler(LOGGING_TEMP_HANDLER)


def reset_logging_handlers() -> None:
    """
    Remove any logging handlers attached to python's root logger
    """
    for handler in list(logging.root.handlers):
        logging.root.removeHandler(handler)


def setup_cli_logging(
    log_level: str, fmt: Optional[str] = None, datefmt: Optional[str] = None
) -> None:
    """
    Setup CLI logging.

    Should be called before ``setup_logfile_logging``
    """
    if fmt is None:
        fmt = "[%(asctime)s][%(levelname)-7s] - %(message)s"
    if datefmt is None:
        datefmt = "%H:%M:%S"

    reset_logging_handlers()

    handler_fmt = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handler = ConsoleHandler(stream=sys.stderr)
    handler.setLevel(level=LOG_LEVELS.get(log_level) or logging.WARNING)
    handler.setFormatter(handler_fmt)
    logging.root.addHandler(handler)

    LOGGING_TEMP_HANDLER.sync_with_handlers(logging.root.handlers)


def setup_logfile_logging(
    logfile, log_level: str, fmt: Optional[str] = None, datefmt: Optional[str] = None
) -> None:
    """
    Setup log file logging.

    Should be called after ``setup_logfile_logging``
    """
    if fmt is None:
        fmt = "%(asctime)s,%(msecs)03d [%(name)-17s:%(lineno)-4d][%(levelname)-7s] %(message)s"
    if datefmt is None:
        datefmt = "%Y-%m-%d %H:%M:%S"
    handler_fmt = logging.Formatter(fmt=fmt, datefmt=datefmt)
    handler = handlers.WatchedFileHandler(logfile, mode="a", encoding="utf-8", delay=False)
    handler.setLevel(level=LOG_LEVELS.get(log_level) or logging.WARNING)
    handler.setFormatter(handler_fmt)
    logging.root.addHandler(handler)
