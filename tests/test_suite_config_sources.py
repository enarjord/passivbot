from __future__ import annotations

import json
from pathlib import Path

from config_utils import get_template_config
from optimize_suite import ensure_suite_config


def test_optimizer_suite_reads_backtest_suite(tmp_path: Path):
    base_cfg = get_template_config()
    base_cfg["backtest"]["suite"]["enabled"] = True
    base_cfg["backtest"]["suite"]["aggregate"]["default"] = "mean"
    base_cfg["backtest"]["suite"]["scenarios"] = []
    config_path = tmp_path / "base.json"
    config_path.write_text(json.dumps(base_cfg))

    suite_cfg = ensure_suite_config(config_path, None)
    assert suite_cfg["enabled"] is True
    assert suite_cfg["aggregate"]["default"] == "mean"


def test_optimizer_suite_config_override_uses_backtest_suite(tmp_path: Path):
    base_cfg = get_template_config()
    base_cfg["backtest"]["suite"]["enabled"] = False
    base_cfg["backtest"]["suite"]["aggregate"]["default"] = "mean"

    override_cfg = get_template_config()
    override_cfg["backtest"]["suite"]["enabled"] = True
    override_cfg["backtest"]["suite"]["aggregate"]["default"] = "median"

    config_path = tmp_path / "base.json"
    suite_path = tmp_path / "override.json"
    config_path.write_text(json.dumps(base_cfg))
    suite_path.write_text(json.dumps(override_cfg))

    suite_cfg = ensure_suite_config(config_path, suite_path)
    assert suite_cfg["enabled"] is True
    assert suite_cfg["aggregate"]["default"] == "median"
