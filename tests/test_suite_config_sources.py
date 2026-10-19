from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from config_utils import get_template_config
from optimize_suite import ensure_suite_config
from suite_runner import load_suite_override_config


def test_optimizer_suite_reads_backtest_scenarios(tmp_path: Path):
    """Test that optimizer reads suite config from new backtest.scenarios structure."""
    base_cfg = get_template_config()
    base_cfg["backtest"]["suite_enabled"] = True
    base_cfg["backtest"]["scenarios"] = [{"label": "test_scenario"}]
    base_cfg["backtest"]["reducer"] = {"default": "mean"}
    config_path = tmp_path / "base.json"
    config_path.write_text(json.dumps(base_cfg))

    suite_cfg = ensure_suite_config(config_path, None)
    assert suite_cfg["enabled"] is True
    assert suite_cfg["reducer"]["default"] == "mean"
    assert suite_cfg["scenarios"] == [{"label": "test_scenario"}]


def test_optimizer_suite_config_override_uses_backtest_scenarios(tmp_path: Path):
    """Test that suite config override properly merges with base config."""
    base_cfg = get_template_config()
    base_cfg["backtest"]["suite_enabled"] = True
    base_cfg["backtest"]["scenarios"] = [{"label": "base_scenario"}]
    base_cfg["backtest"]["reducer"] = {"default": "mean"}

    # Override config needs to be a full valid config
    override_cfg = get_template_config()
    override_cfg["backtest"]["scenarios"] = [{"label": "override_scenario"}]
    override_cfg["backtest"]["reducer"] = {"default": "median"}

    config_path = tmp_path / "base.json"
    suite_path = tmp_path / "override.json"
    config_path.write_text(json.dumps(base_cfg))
    suite_path.write_text(json.dumps(override_cfg))

    suite_cfg = ensure_suite_config(config_path, suite_path)
    assert suite_cfg["enabled"] is True
    assert suite_cfg["reducer"]["default"] == "median"
    assert suite_cfg["scenarios"] == [{"label": "override_scenario"}]


def test_optimizer_suite_config_override_accepts_partial_backtest_scenarios(tmp_path: Path):
    """Suite override files do not need enough bot config fields for flavor detection."""
    base_cfg = get_template_config()
    base_cfg["backtest"]["suite_enabled"] = True
    base_cfg["backtest"]["scenarios"] = [{"label": "base_scenario"}]

    override_cfg = {
        "backtest": {
            "scenarios": [
                {
                    "label": "stress_window",
                    "start_date": "2025-09-19",
                    "end_date": "2026-01-08",
                    "coins": ["BTC", "ETH"],
                }
            ],
            "reducer": {"default": "median"},
        }
    }

    config_path = tmp_path / "base.json"
    suite_path = tmp_path / "partial_suite.hjson"
    config_path.write_text(json.dumps(base_cfg))
    suite_path.write_text(json.dumps(override_cfg))

    suite_cfg = ensure_suite_config(config_path, suite_path)
    assert suite_cfg["enabled"] is True
    assert suite_cfg["reducer"]["default"] == "median"
    assert suite_cfg["scenarios"] == override_cfg["backtest"]["scenarios"]


def test_partial_backtest_suite_config_preserves_base_reducer(tmp_path: Path):
    base_cfg = get_template_config()
    base_cfg["backtest"]["suite_enabled"] = True
    base_cfg["backtest"]["scenarios"] = [{"label": "base_scenario"}]
    base_cfg["backtest"]["reducer"] = {"default": "max", "adg": "median"}

    override_cfg = {
        "backtest": {
            "scenarios": [{"label": "stress_window"}],
        }
    }

    config_path = tmp_path / "base.json"
    suite_path = tmp_path / "partial_suite.hjson"
    config_path.write_text(json.dumps(base_cfg))
    suite_path.write_text(json.dumps(override_cfg))

    suite_cfg = ensure_suite_config(config_path, suite_path)
    assert suite_cfg["reducer"] == {"default": "max", "adg": "median"}
    assert suite_cfg["scenarios"] == override_cfg["backtest"]["scenarios"]


def test_partial_top_level_suite_config_preserves_base_reducer(tmp_path: Path):
    base_cfg = get_template_config()
    base_cfg["backtest"]["suite_enabled"] = True
    base_cfg["backtest"]["scenarios"] = [{"label": "base_scenario"}]
    base_cfg["backtest"]["reducer"] = {"default": "max", "adg": "median"}

    override_cfg = {"scenarios": [{"label": "top_level_stress"}]}

    config_path = tmp_path / "base.json"
    suite_path = tmp_path / "partial_suite.hjson"
    config_path.write_text(json.dumps(base_cfg))
    suite_path.write_text(json.dumps(override_cfg))

    suite_cfg = ensure_suite_config(config_path, suite_path)
    assert suite_cfg["reducer"] == {"default": "max", "adg": "median"}
    assert suite_cfg["scenarios"] == override_cfg["scenarios"]


def test_optimizer_suite_rejects_legacy_override_format(tmp_path: Path):
    """External files obey the same final-v9 structural boundary."""
    base_cfg = get_template_config()
    base_cfg["backtest"]["suite_enabled"] = True
    base_cfg["backtest"]["scenarios"] = [{"label": "base_scenario"}]

    override_cfg = get_template_config()
    override_cfg["backtest"]["suite"] = {
        "scenarios": [{"label": "legacy_override"}],
        "aggregate": {"default": "max"},
    }

    config_path = tmp_path / "base.json"
    suite_path = tmp_path / "override.json"
    config_path.write_text(json.dumps(base_cfg))
    suite_path.write_text(json.dumps(override_cfg))

    with pytest.raises(ValueError, match="unknown configuration field"):
        ensure_suite_config(config_path, suite_path)


def test_optimizer_suite_rejects_partial_legacy_backtest_suite(tmp_path: Path):
    base_cfg = get_template_config()
    base_cfg["backtest"]["suite_enabled"] = True
    base_cfg["backtest"]["scenarios"] = [{"label": "base_scenario"}]

    override_cfg = {
        "backtest": {
            "suite": {
                "scenarios": [{"label": "legacy_partial"}],
                "aggregate": {"default": "min"},
            }
        }
    }

    config_path = tmp_path / "base.json"
    suite_path = tmp_path / "legacy_partial_suite.hjson"
    config_path.write_text(json.dumps(base_cfg))
    suite_path.write_text(json.dumps(override_cfg))

    with pytest.raises(ValueError, match="unknown configuration field"):
        ensure_suite_config(config_path, suite_path)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("selector", ["long.unstuck.ema_span_0", "bot.long.strategy.entry.ema_span_0"])
def test_external_suite_rejects_authored_selector_shorthand_before_projection(tmp_path, wrapped, selector):
    section = {"scenarios": [{"label": "bad", "overrides": {selector: 17.5}}]}
    document = {"backtest": section} if wrapped else section
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unknown configuration field"):
        load_suite_override_config(path, source_config=get_template_config())


@pytest.mark.parametrize("document", [
    {"backtest": {"scenarios": [{"label": "base"}]}, "unknown_root": 1},
    {"backtest": {"scenarios": [{"label": "base"}], "unknown_section": 1}},
    {"scenarios": [{"label": "base"}], "unknown_section": 1},
    {"backtest": {"scenarios": [{"label": "base"}]}, "live": {"unknown": 1}},
    {"scenarios": [{"label": "base"}], "aggregate": {"default": "median"}},
    {"suite": {"scenarios": [{"label": "base"}]}},
])
def test_external_suite_validates_unprojected_authored_fields(tmp_path, document):
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="unknown configuration field"):
        load_suite_override_config(path)


def test_external_suite_canonical_coin_patch_uses_base_strategy_context(tmp_path):
    base = get_template_config()
    base["live"]["strategy_kind"] = "ema_anchor"
    section = {"scenarios": [{"label": "anchor", "overrides": {"coin_overrides": {
        "BTC": {"bot": {"long": {"strategy": {"ema_anchor": {"offset": 0.013}}}}},
    }}}]}
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(section))
    original = deepcopy(base)
    assert load_suite_override_config(path, source_config=base) == section
    assert base == original


def test_external_suite_canonical_coin_patch_uses_base_scope_context(tmp_path):
    base = get_template_config()
    base["bot"]["risk"]["hsl"]["scope"] = "pside"
    section = {"scenarios": [{"label": "scope", "overrides": {"coin_overrides": {
        "BTC": {"bot": {"long": {"hsl": {"enabled": True}}}},
    }}}]}
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(section))
    with pytest.raises(ValueError, match="not overridable|overridable only"):
        load_suite_override_config(path, source_config=base)


@pytest.mark.parametrize("version", ["v8.6.0", "v9.1.0", None])
def test_external_suite_never_erases_an_authored_version_marker(tmp_path, version):
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"config_version": version,
                                "backtest": {"scenarios": [{"label": "base"}]}}))
    with pytest.raises(ValueError, match="config_version"):
        load_suite_override_config(path, source_config=get_template_config())


def test_external_suite_rejects_retired_fields_outside_projected_section(tmp_path):
    source = get_template_config()
    source["backtest"]["scenarios"] = [{"label": "base"}]
    source["live"]["auto_gs"] = True
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="retired"):
        load_suite_override_config(path, source_config=get_template_config())


def test_external_full_config_without_base_uses_its_own_policy_context(tmp_path):
    source = get_template_config()
    source["live"]["strategy_kind"] = "ema_anchor"
    source["backtest"]["scenarios"] = [{"label": "anchor", "overrides": {"coin_overrides": {
        "BTC": {"bot": {"long": {"strategy": {"ema_anchor": {"offset": 0.013}}}}},
    }}}]
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(source))
    loaded = load_suite_override_config(path)
    assert loaded["scenarios"] == source["backtest"]["scenarios"]
