import os
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from logging_setup import (
    DEFAULT_DATEFMT,
    build_command_log_path,
    configure_logging,
    create_command_log_filename,
    resolve_live_log_file_settings,
)


def test_configure_logging_uses_utc_formatter_and_z_suffix():
    configure_logging(debug=1, stream=True)

    handler = logging.getLogger().handlers[0]
    formatter = handler.formatter

    assert formatter is not None
    assert formatter.datefmt == DEFAULT_DATEFMT
    assert formatter.converter is time.gmtime


def test_create_command_log_filename_uses_utc_timestamp_and_sanitized_command():
    timestamp = datetime(2026, 4, 4, 18, 7, 6, tzinfo=timezone.utc)

    filename = create_command_log_filename(
        ["passivbot live", "-u", "bybit_01", "configs/live/my config.json"],
        timestamp=timestamp,
    )

    assert filename == "20260404_180706_passivbot_live_-u_bybit_01_configs_live_my_config.json.log"


def test_build_command_log_path_places_file_under_requested_dir():
    timestamp = datetime(2026, 4, 4, 18, 7, 6, tzinfo=timezone.utc)

    path = build_command_log_path(["passivbot live", "-u", "bybit_01"], "logs", timestamp=timestamp)

    assert path == Path("logs/20260404_180706_passivbot_live_-u_bybit_01.log")


def test_resolve_live_log_file_settings_defaults_to_timestamped_archive_and_stable_alias():
    settings = resolve_live_log_file_settings(
        {"logging": {"persist_to_file": True, "dir": "logs", "rotation": False}},
        user="bitget_01",
        command_args=["passivbot live", "-u", "bitget_01", "-lm", "graceful_stop"],
    )

    assert settings["log_file"].startswith("logs/")
    assert settings["log_file"].endswith(
        "_passivbot_live_-u_bitget_01_-lm_graceful_stop.log"
    )
    assert settings["current_log_file"] == "logs/bitget_01.log"
    assert settings["rotation"] is False
    assert settings["max_bytes"] == 10 * 1024 * 1024
    assert settings["backup_count"] == 5


def test_resolve_live_log_file_settings_disables_file_handler_when_requested():
    settings = resolve_live_log_file_settings(
        {"logging": {"persist_to_file": False, "dir": "logs"}},
        user="bitget_01",
    )

    assert settings["log_file"] is None
    assert settings["current_log_file"] is None
    assert settings["rotation"] is False
    assert settings["max_bytes"] == 10 * 1024 * 1024
    assert settings["backup_count"] == 5


def test_configure_logging_with_file_handler_writes_log_file(tmp_path):
    log_path = tmp_path / "logs" / "live.log"

    configure_logging(debug=1, stream=False, log_file=str(log_path))
    logging.getLogger().info("hello file logging")

    assert log_path.exists()
    text = log_path.read_text(encoding="utf-8")
    assert "hello file logging" in text


@pytest.mark.skipif(os.name == "nt", reason="symlink alias behavior is exercised on POSIX")
def test_configure_logging_updates_stable_alias_to_current_run_log(tmp_path):
    archived_log = tmp_path / "logs" / "20260406_140000_passivbot_live_-u_bitget_01.log"
    current_log = tmp_path / "logs" / "bitget_01.log"

    configure_logging(
        debug=1,
        stream=False,
        log_file=str(archived_log),
        current_log_file=str(current_log),
    )
    logging.getLogger().info("line for stable alias")

    assert archived_log.exists()
    assert current_log.is_symlink()
    assert current_log.resolve() == archived_log.resolve()
    assert "line for stable alias" in current_log.read_text(encoding="utf-8")
