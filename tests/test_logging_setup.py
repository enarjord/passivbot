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
    resolve_command_logging_options,
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


def test_resolve_command_logging_options_defaults_to_file_logging_when_requested():
    log_file, rotation_enabled, max_bytes, backup_count = resolve_command_logging_options(
        {},
        ["passivbot live", "-u", "bybit_01"],
        default_persist=True,
    )

    assert log_file is not None
    assert log_file.startswith("logs/")
    assert rotation_enabled is False
    assert max_bytes == 10 * 1024 * 1024
    assert backup_count == 5


def test_resolve_command_logging_options_can_disable_file_logging():
    log_file, rotation_enabled, max_bytes, backup_count = resolve_command_logging_options(
        {"persist_to_file": False, "rotation_enabled": True, "rotation_max_mb": 3, "rotation_backups": 2},
        ["passivbot live"],
        default_persist=True,
    )

    assert log_file is None
    assert rotation_enabled is True
    assert max_bytes == 3 * 1024 * 1024
    assert backup_count == 2


def test_resolve_command_logging_options_rejects_invalid_dir():
    with pytest.raises(ValueError, match="logging.dir"):
        resolve_command_logging_options(
            {"persist_to_file": True, "dir": ""},
            ["passivbot live"],
            default_persist=True,
        )


def test_resolve_command_logging_options_rejects_invalid_rotation_values():
    with pytest.raises(ValueError, match="rotation_max_mb"):
        resolve_command_logging_options(
            {"persist_to_file": True, "rotation_max_mb": 0},
            ["passivbot live"],
            default_persist=True,
        )

    with pytest.raises(ValueError, match="rotation_backups"):
        resolve_command_logging_options(
            {"persist_to_file": True, "rotation_backups": -1},
            ["passivbot live"],
            default_persist=True,
        )


def test_configure_logging_with_file_handler_writes_log_file(tmp_path):
    log_path = tmp_path / "logs" / "live.log"

    configure_logging(debug=1, stream=False, log_file=str(log_path))
    logging.getLogger().info("hello file logging")

    assert log_path.exists()
    text = log_path.read_text(encoding="utf-8")
    assert "hello file logging" in text
