from __future__ import annotations

import json
from pathlib import Path

from config_utils import get_template_config
from optimize_suite import ensure_suite_config


def test_optimizer_suite_reads_backtest_scenarios(tmp_path: Path):
    """Test that optimizer reads suite config from new backtest.scenarios structure."""
    base_cfg = get_template_config()
    base_cfg["backtest"]["scenarios"] = [{"label": "test_scenario"}]
    base_cfg["backtest"]["aggregate"] = {"default": "mean"}
    config_path = tmp_path / "base.json"
    config_path.write_text(json.dumps(base_cfg))

    suite_cfg = ensure_suite_config(config_path, None)
    assert suite_cfg["enabled"] is True  # enabled derived from scenarios presence
    assert suite_cfg["aggregate"]["default"] == "mean"
    assert suite_cfg["scenarios"] == [{"label": "test_scenario"}]


def test_optimizer_suite_config_override_uses_backtest_scenarios(tmp_path: Path):
    """Test that suite config override properly merges with base config."""
    base_cfg = get_template_config()
    base_cfg["backtest"]["scenarios"] = [{"label": "base_scenario"}]
    base_cfg["backtest"]["aggregate"] = {"default": "mean"}

    # Override config needs to be a full valid config
    override_cfg = get_template_config()
    override_cfg["backtest"]["scenarios"] = [{"label": "override_scenario"}]
    override_cfg["backtest"]["aggregate"] = {"default": "median"}

    config_path = tmp_path / "base.json"
    suite_path = tmp_path / "override.json"
    config_path.write_text(json.dumps(base_cfg))
    suite_path.write_text(json.dumps(override_cfg))

    suite_cfg = ensure_suite_config(config_path, suite_path)
    assert suite_cfg["enabled"] is True
    assert suite_cfg["aggregate"]["default"] == "median"
    assert suite_cfg["scenarios"] == [{"label": "override_scenario"}]


def test_optimizer_suite_legacy_override_format(tmp_path: Path):
    """Test that legacy backtest.suite format in override config still works."""
    base_cfg = get_template_config()
    base_cfg["backtest"]["scenarios"] = [{"label": "base_scenario"}]

    # Legacy format with suite wrapper - needs to be a full valid config
    override_cfg = get_template_config()
    override_cfg["backtest"]["suite"] = {
        "scenarios": [{"label": "legacy_override"}],
        "aggregate": {"default": "max"},
    }

    config_path = tmp_path / "base.json"
    suite_path = tmp_path / "override.json"
    config_path.write_text(json.dumps(base_cfg))
    suite_path.write_text(json.dumps(override_cfg))

    suite_cfg = ensure_suite_config(config_path, suite_path)
    assert suite_cfg["scenarios"] == [{"label": "legacy_override"}]
    assert suite_cfg["aggregate"]["default"] == "max"
