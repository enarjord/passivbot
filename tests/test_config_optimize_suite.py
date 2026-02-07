from copy import deepcopy

from config_utils import format_config, get_template_config


def test_optimize_suite_is_ignored_and_removed(caplog):
    """Test that optimize.suite is removed and warning is logged."""
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    # Use new flattened structure for backtest scenarios
    base["backtest"]["scenarios"] = [{"label": "s1", "start_date": "2022-01-01"}]
    base["backtest"]["aggregate"] = {"default": "mean"}
    # Add optimize.suite which should be removed with a warning
    base["optimize"]["suite"] = {"enabled": True, "aggregate": {"default": "median"}}

    formatted = format_config(deepcopy(base), verbose=False)

    # optimize.suite should be removed
    assert "suite" not in formatted.get("optimize", {})
    # backtest scenarios should remain
    assert formatted["backtest"]["scenarios"] == [{"label": "s1", "start_date": "2022-01-01"}]
    # Warning should have been logged about optimize.suite
    assert any("optimize.suite" in rec.message for rec in caplog.records)


def test_suite_aggregate_default_preserved():
    """Test that aggregate default setting is preserved in new flattened structure."""
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    # Modify the default aggregate mode
    base["backtest"]["aggregate"]["default"] = "median"
    formatted = format_config(deepcopy(base), verbose=False)
    assert formatted["backtest"]["aggregate"]["default"] == "median"


def test_legacy_suite_migration():
    """Test that legacy backtest.suite structure is migrated to new format."""
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    # Inject legacy suite structure
    base["backtest"]["suite"] = {
        "enabled": True,
        "include_base_scenario": True,
        "base_label": "combined",
        "aggregate": {"default": "median"},
        "scenarios": [
            {"label": "binance_only", "exchanges": ["binance"]},
            {"label": "bybit_only", "exchanges": ["bybit"]},
        ],
    }
    # Remove new-style keys to simulate old config
    base["backtest"].pop("scenarios", None)
    base["backtest"].pop("aggregate", None)

    formatted = format_config(deepcopy(base), verbose=True)

    # suite wrapper should be removed
    assert "suite" not in formatted["backtest"]
    # scenarios should be at top level with base scenario prepended
    assert len(formatted["backtest"]["scenarios"]) == 3
    assert formatted["backtest"]["scenarios"][0]["label"] == "combined"  # base scenario
    assert formatted["backtest"]["scenarios"][1]["label"] == "binance_only"
    assert formatted["backtest"]["scenarios"][2]["label"] == "bybit_only"
    # aggregate should be at top level
    assert formatted["backtest"]["aggregate"]["default"] == "median"


def test_legacy_combine_ohlcvs_removed():
    """Test that legacy combine_ohlcvs is removed during migration."""
    base = get_template_config()
    base["_raw"] = deepcopy(base)
    base["backtest"]["combine_ohlcvs"] = True

    formatted = format_config(deepcopy(base), verbose=True)

    # combine_ohlcvs should be removed
    assert "combine_ohlcvs" not in formatted["backtest"]
    # volume_normalization should exist
    assert "volume_normalization" in formatted["backtest"]
