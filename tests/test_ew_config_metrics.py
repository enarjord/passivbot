"""One CAGR EW selector and a native campaign default; no Python metric formula."""

import pytest
from config.ew import (
    EW_METRICS,
    is_ew_metric,
    validate_ew_half_life_days,
    validate_ew_config,
)
from config.metrics import DEFAULT_EW_HALF_LIFE_DAYS, validate_core_metric_requests
from config.load import prepare_config
from config.schema import CONFIG_SCHEMA_VERSION, get_template_config


@pytest.mark.parametrize(
    "value", [True, False, "180", -1, 0, float("nan"), float("inf")]
)
def test_half_life_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="ew_half_life_days"):
        validate_ew_half_life_days(value)


@pytest.mark.parametrize("value", [0.01, 0.99, 1, 180])
def test_positive_half_lives_are_preserved(value):
    assert validate_ew_half_life_days(value) == value


def test_omitted_half_life_uses_profile_and_explicit_null_uses_native_fallback():
    assert DEFAULT_EW_HALF_LIFE_DAYS == 180.0
    source = {"config_version": CONFIG_SCHEMA_VERSION}
    assert prepare_config(source, verbose=False)["backtest"]["ew_half_life_days"] == 360.0
    source["backtest"] = {"ew_half_life_days": None}
    assert (
        prepare_config(source, verbose=False)["backtest"]["ew_half_life_days"]
        == DEFAULT_EW_HALF_LIFE_DAYS
    )


def test_one_ew_metric_selector():
    assert EW_METRICS == {"cagr_fraction_strategy_equity_ew"}
    assert is_ew_metric("cagr_fraction_strategy_equity_ew_max")
    assert not is_ew_metric("sharpe_ratio_strategy_equity_ew")


def test_minute_selected_metrics_fail_coarser_replay_without_blocking_other_quantities():
    config = get_template_config()
    config["backtest"]["candle_interval_minutes"] = 5
    with pytest.raises(ValueError, match="one-minute"):
        validate_core_metric_requests(config, ["cagr_fraction_strategy_equity_ew"])
    validate_core_metric_requests(
        config, ["cagr_fraction_strategy_equity", "omega_ratio_strategy_equity"]
    )


def test_non_utc_compatible_daily_cadence_is_rejected():
    config = get_template_config()
    config["backtest"]["candle_interval_minutes"] = 7
    with pytest.raises(ValueError, match="UTC"):
        validate_core_metric_requests(config, ["omega_ratio_strategy_equity"])


def test_campaign_half_life_cannot_be_overridden_by_a_scenario():
    config = get_template_config()
    config["backtest"]["scenarios"] = [
        {"label": "other", "overrides": {"backtest.ew_half_life_days": 12}}
    ]
    with pytest.raises(ValueError, match="Scenario"):
        validate_ew_config(config)
