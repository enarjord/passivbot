"""Retired legacy recovery controls cannot authorize or change current HSL."""
import json
import logging

import pytest

from config.load import load_input_config, load_prepared_config, prepare_config
from config.schema import get_template_config


@pytest.mark.parametrize("field", ["hsl_accept_incomplete_history", "hsl_unavailable_grace_seconds", "risk_input_max_attempts"])
@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("shape", ["current", "live_only", "nested_current"])
def test_retired_recovery_controls_are_removed_at_canonical_boundary(tmp_path, caplog, field, value, shape):
    config = get_template_config()
    config["live"][field] = value
    if shape == "live_only":
        config = {key: config[key] for key in ("config_version", "bot", "live")}
    document = {"config": config} if shape == "nested_current" else config
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document))
    source, _, raw = load_input_config(str(path), log_info=False)
    # Raw input remains faithful; normalization owns the warning and retirement.
    assert source == raw == document
    with caplog.at_level(logging.WARNING):
        prepared = load_prepared_config(str(path), live_only=True, target="live", verbose=True, log_info=False)
    assert field not in prepared["live"]
    assert f"live.{field} is removed" in caplog.text
    assert "per-run CLI-only" not in caplog.text
    # Serialization or a repeated normalization cannot resurrect a waiver.
    assert field not in prepare_config(prepared, target="canonical", runtime=None, verbose=False)["live"]
