"""Terminal projection uses canonical facts without changing persisted metrics."""

from copy import deepcopy

import pytest

from backtest_display import format_backtest_metrics
from config.metrics import CORE_METRIC_CATALOG


CAGR = "cagr_fraction_strategy_equity"
OMEGA = "omega_ratio_strategy_equity"
LONG = "position_exposure_utilization_rms_ratio_long"
SHORT = "position_exposure_utilization_rms_ratio_short"
SHARE = "short_net_profit_contribution_fraction"
HSL = "hsl_panic_episode_loss_max_fraction"


def _analysis(**updates):
    return {
        **dict.fromkeys(CORE_METRIC_CATALOG, 0.0),
        "effective_start_date": "2024-01-01T00:00:00Z",
        "effective_end_date": "2024-01-07T00:00:00Z",
        "liquidated": False,
        "metric_status": {},
        "metric_metadata": {
            "hsl_enabled": False, "hsl_trigger_count": 0,
            "hsl_execution_episode_count": 0, "daily_return_count": 5,
            "equity_observation_count": 8641, "fill_count": 100,
            "holding_episode_count": 10, "ew_half_life_days": 180.0,
            "peak_recovery_tail_censored": True,
        },
        **updates,
    }


def _config(long=True, short=False):
    return {"bot": {"long": {"enabled": long}, "short": {"enabled": short}}}


def _row(table, metric):
    return next(line for line in table.splitlines() if line.startswith(metric + " "))


def test_five_significant_digits_percentage_units_and_aligned_columns():
    analysis = _analysis(**{
        CAGR: 0.123456789, OMEGA: 1.23456789,
        "gross_turnover_ratio_per_day": 0.678912345,
        "peak_recovery_max_days_strategy_equity": 12.3456789,
        "ulcer_index_fraction_strategy_equity": 0.00123456789,
    })
    before = deepcopy(analysis)
    table = format_backtest_metrics(analysis, _config())
    assert _row(table, CAGR).split() == [CAGR, "12.346%"]
    assert _row(table, OMEGA).split() == [OMEGA, "1.2346"]
    assert _row(table, "gross_turnover_ratio_per_day").split()[-1] == "0.67891"
    assert _row(table, "ulcer_index_fraction_strategy_equity").split()[-1] == "0.12346%"
    recovery = _row(table, "peak_recovery_max_days_strategy_equity")
    assert "12.346" in recovery and "unfinished tail" in recovery
    assert "half-life 180 days" in table
    header = next(line for line in table.splitlines() if line.startswith("Metric "))
    value_end = header.index("Value") + len("Value")
    assert _row(table, CAGR).index("12.346%") + len("12.346%") == value_end
    assert _row(table, OMEGA).index("1.2346") + len("1.2346") == value_end
    assert "Period: 2024-01-01T00:00:00Z to 2024-01-07T00:00:00Z" in table
    assert "Liquidated: no" in table
    assert "Support: 5 daily returns, 8,641 equity observations, 100 fills, 10 holding episodes" in table
    assert "metric_metadata" not in table and "metric_status" not in table
    assert analysis == before


@pytest.mark.parametrize("value,expected", [
    (-0.123456789, "-12.346%"), (1.23456789, "123.46%"),
    (0.000000123456789, "1.2346e-05%"),
    (1e308, "1e+310%"),
])
def test_fraction_percentages_preserve_sign_and_unbounded_range(value, expected):
    assert _row(format_backtest_metrics({CAGR: value}, {}), CAGR).split()[-1] == expected


@pytest.mark.parametrize("tag,value,reason", [
    ("positive_infinity", "+inf", ""),
    ("negative_infinity", "-inf", ""),
    ("undefined", "undefined", "nonpositive_combined_net_profit"),
    ("invalid", "invalid", "missing_daily_support"),
])
def test_nonfinite_status_is_not_presented_as_zero_or_none(tag, value, reason):
    analysis = {OMEGA: None, "metric_status": {OMEGA: {"status": tag, "reason": reason}}}
    row = _row(format_backtest_metrics(analysis, {}), OMEGA)
    assert row.split()[1] == value
    assert reason.replace("_", " ") in row
    assert "None" not in row
    assert analysis[OMEGA] is None


def test_disabled_long_short_and_hsl_hide_only_zero_inactive_rows():
    long_only = format_backtest_metrics(_analysis(), _config())
    assert LONG in long_only
    assert SHORT not in long_only and SHARE not in long_only
    assert "hsl_" not in long_only and "HSL:" not in long_only
    short_only = format_backtest_metrics(_analysis(), _config(False, True))
    assert LONG not in short_only and SHORT in short_only and SHARE in short_only


def test_enabled_zero_activity_and_unknown_applicability_remain_visible():
    analysis = _analysis()
    analysis["metric_metadata"]["hsl_enabled"] = True
    enabled = format_backtest_metrics(analysis, _config(True, True))
    assert all(name in enabled for name in CORE_METRIC_CATALOG)
    assert "HSL: 0 triggers, 0 execution episodes" in enabled
    unknown = format_backtest_metrics(dict.fromkeys(CORE_METRIC_CATALOG, 0.0), {})
    assert all(name in unknown for name in CORE_METRIC_CATALOG)


@pytest.mark.parametrize("side,metric", [("long", LONG), ("short", SHORT)])
def test_disabled_side_retains_observed_fills_even_with_zero_metric(side, metric):
    table = format_backtest_metrics(_analysis(), _config(False, False),
                                    fill_types=[f"close_ema_anchor_{side}"])
    assert metric in table


@pytest.mark.parametrize("metric", [SHORT, SHARE, HSL])
def test_disabled_feature_retains_nonzero_or_unavailable_metrics(metric):
    for value in (0.125, None):
        analysis = _analysis(**{metric: value})
        if value is None:
            analysis["metric_status"][metric] = {"status": "invalid", "reason": "fixture"}
        assert metric in format_backtest_metrics(analysis, _config())


@pytest.mark.parametrize("metadata,fill_types", [
    ({"hsl_trigger_count": 1}, ()),
    ({"hsl_execution_episode_count": 1}, ()),
    ({}, ("close_panic_long",)),
])
def test_disabled_hsl_retains_observed_counts_or_panic_execution(metadata, fill_types):
    analysis = _analysis()
    analysis["metric_metadata"].update(metadata)
    assert HSL in format_backtest_metrics(analysis, _config(), fill_types=fill_types)


def test_selected_metrics_do_not_invent_unrequested_rows_or_support():
    table = format_backtest_metrics({OMEGA: 1.25, "liquidated": True}, {})
    assert OMEGA in table and CAGR not in table
    assert "Liquidated: yes" in table
    assert "Support:" not in table and "HSL:" not in table
