"""Pure-Python tests for the per-run-only enforcement of
live.hsl_accept_incomplete_history (no passivbot_rust required).

The flag waives the HSL fail-closed coverage contract, so a value persisted
in a config file must never survive a restart: load_input_config strips it
(with a critical log) before CLI overrides are applied, and only the CLI
flag of the current invocation can re-enable it.
"""

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config.load import (  # noqa: E402
    load_input_config,
    strip_persisted_hsl_incomplete_history_override,
)


def test_persisted_nested_true_is_stripped_with_critical_log(caplog):
    source = {"live": {"hsl_accept_incomplete_history": True, "leverage": 10}}
    with caplog.at_level(logging.CRITICAL):
        strip_persisted_hsl_incomplete_history_override(source, "cfg.json")
    assert "hsl_accept_incomplete_history" not in source["live"]
    assert source["live"]["leverage"] == 10
    assert "per-run CLI-only" in caplog.text
    assert "cfg.json" in caplog.text


def test_persisted_flat_true_is_stripped_with_critical_log(caplog):
    source = {"hsl_accept_incomplete_history": True}
    with caplog.at_level(logging.CRITICAL):
        strip_persisted_hsl_incomplete_history_override(source, "cfg.json")
    assert "hsl_accept_incomplete_history" not in source
    assert "per-run CLI-only" in caplog.text


def test_persisted_false_is_left_untouched_silently(caplog):
    # A persisted False matches the schema default and poses no risk;
    # leaving it in place keeps _raw snapshots faithful to the file.
    source = {"live": {"hsl_accept_incomplete_history": False}}
    with caplog.at_level(logging.CRITICAL):
        strip_persisted_hsl_incomplete_history_override(source, "cfg.json")
    assert source["live"]["hsl_accept_incomplete_history"] is False
    assert caplog.text == ""


def test_load_input_config_strips_persisted_override(tmp_path, caplog):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps({"live": {"hsl_accept_incomplete_history": True}})
    )
    with caplog.at_level(logging.CRITICAL):
        source, base_path, raw_snapshot = load_input_config(
            str(cfg_path), log_info=False
        )
    assert "hsl_accept_incomplete_history" not in source["live"]
    assert "hsl_accept_incomplete_history" not in raw_snapshot["live"]
    assert "per-run CLI-only" in caplog.text


def test_cli_override_survives_because_it_is_applied_after_load(tmp_path):
    from config_utils import update_config_with_args

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps({"live": {"hsl_accept_incomplete_history": True}})
    )
    source, _, _ = load_input_config(str(cfg_path), log_info=False)
    assert "hsl_accept_incomplete_history" not in source["live"]
    args = SimpleNamespace(**{"live.hsl_accept_incomplete_history": True})
    update_config_with_args(
        source, args, allowed_keys={"live.hsl_accept_incomplete_history"}
    )
    assert source["live"]["hsl_accept_incomplete_history"] is True
