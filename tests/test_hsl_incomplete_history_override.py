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

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config.load import (  # noqa: E402
    load_input_config,
    load_prepared_config,
    prepare_config,
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


@pytest.mark.parametrize("shape", ["current", "live_only", "nested_current"])
def test_prepared_config_cannot_restore_persisted_override(tmp_path, caplog, shape):
    from config.schema import get_template_config

    config = get_template_config()
    config["live"]["hsl_accept_incomplete_history"] = True
    if shape == "live_only":
        config = {key: config[key] for key in ("config_version", "bot", "live")}
    document = {"config": config} if shape == "nested_current" else config
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document))

    with caplog.at_level(logging.CRITICAL):
        prepared = load_prepared_config(
            str(path), live_only=True, target="live", verbose=False, log_info=False
        )

    assert prepared["live"]["hsl_accept_incomplete_history"] is False
    source, _, raw = load_input_config(str(path), log_info=False)
    for snapshot in (source, raw):
        effective = snapshot["config"] if shape == "nested_current" else snapshot
        assert "hsl_accept_incomplete_history" not in effective["live"]
    assert "per-run CLI-only" in caplog.text


def test_serialized_per_run_override_is_stripped_when_wrapped_and_reloaded(tmp_path):
    from config.schema import get_template_config
    from config_utils import update_config_with_args

    path = tmp_path / "config.json"
    path.write_text(json.dumps(get_template_config()))
    source, _, _ = load_input_config(str(path), log_info=False)
    update_config_with_args(
        source,
        SimpleNamespace(**{"live.hsl_accept_incomplete_history": True}),
        allowed_keys={"live.hsl_accept_incomplete_history"},
    )
    assert source["live"]["hsl_accept_incomplete_history"] is True

    path.write_text(json.dumps({"config": source}))
    reloaded = load_prepared_config(str(path), verbose=False, log_info=False)
    assert reloaded["live"]["hsl_accept_incomplete_history"] is False


@pytest.mark.parametrize("persisted_waiver", [False, True])
def test_wrapped_config_applies_explicit_cli_waiver_before_preparation(
    tmp_path, persisted_waiver
):
    from config.schema import get_template_config
    from config_utils import update_config_with_args

    config = get_template_config()
    config["live"]["hsl_accept_incomplete_history"] = persisted_waiver
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"config": config}))
    source, base_path, raw = load_input_config(str(path), log_info=False)
    update_config_with_args(
        source,
        SimpleNamespace(**{"live.hsl_accept_incomplete_history": True, "live.leverage": 3}),
        allowed_keys={"live.hsl_accept_incomplete_history", "live.leverage"},
    )

    prepared = prepare_config(
        source, base_config_path=base_path, raw_snapshot=raw,
        live_only=True, target="live", verbose=False,
    )

    assert prepared["live"]["hsl_accept_incomplete_history"] is True
    assert prepared["live"]["leverage"] == 3
    assert not raw["config"]["live"].get("hsl_accept_incomplete_history", False)
    assert "live" not in source
    assert any(step.get("step") == "update_config_with_args"
               for step in prepared["_transform_log"])


@pytest.mark.parametrize("wrapped", [False, True])
def test_cli_overrides_do_not_load_strategy_metadata(monkeypatch, wrapped):
    import config_utils

    def unavailable():
        raise AssertionError("override processing must not load Rust metadata")

    monkeypatch.setattr(config_utils, "get_template_config", unavailable)
    payload = {"bot": {}, "live": {"leverage": 1}, "backtest": {}, "optimize": {}}
    source = {"config": payload} if wrapped else payload
    config_utils.update_config_with_args(source, SimpleNamespace(**{"live.leverage": 3}))
    assert payload["live"]["leverage"] == 3
    assert source["_transform_log"][-1]["step"] == "update_config_with_args"
