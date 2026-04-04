import logging
import time

from logging_setup import DEFAULT_DATEFMT, configure_logging, resolve_live_log_file_settings


def test_configure_logging_uses_utc_formatter_and_z_suffix():
    configure_logging(debug=1, stream=True)

    handler = logging.getLogger().handlers[0]
    formatter = handler.formatter

    assert formatter is not None
    assert formatter.datefmt == DEFAULT_DATEFMT
    assert formatter.converter is time.gmtime


def test_resolve_live_log_file_settings_defaults_to_user_logfile():
    settings = resolve_live_log_file_settings(
        {"logging": {"persist_to_file": True, "dir": "logs", "rotation": False}},
        user="bitget_01",
    )

    assert settings["log_file"] == "logs/bitget_01.log"
    assert settings["rotation"] is False
    assert settings["max_bytes"] == 10 * 1024 * 1024
    assert settings["backup_count"] == 5


def test_resolve_live_log_file_settings_disables_file_handler_when_requested():
    settings = resolve_live_log_file_settings(
        {"logging": {"persist_to_file": False, "dir": "logs"}},
        user="bitget_01",
    )

    assert settings["log_file"] is None
