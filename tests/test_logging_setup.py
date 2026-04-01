import logging
import time

from logging_setup import DEFAULT_DATEFMT, configure_logging


def test_configure_logging_uses_utc_formatter_and_z_suffix():
    configure_logging(debug=1, stream=True)

    handler = logging.getLogger().handlers[0]
    formatter = handler.formatter

    assert formatter is not None
    assert formatter.datefmt == DEFAULT_DATEFMT
    assert formatter.converter is time.gmtime
