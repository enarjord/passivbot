import pytest

from analysis_visibility import (
    filter_analysis_for_visibility,
    resolve_visible_metric_names,
    validate_visible_metrics_config,
)
from config_utils import get_template_config


def _make_analysis():
    return {
        "adg_fraction_usd": 0.01,
        "adg_fraction_btc": 0.02,
        "drawdown_max_fraction_usd": 0.3,
        "drawdown_max_fraction_btc": 0.4,
        "loss_profit_ratio": 0.5,
        "drawdown_max_fraction_strategy_equity": 0.24,
        "drawdown_max_fraction_strategy_equity_long": 0.12,
        "adg_fraction_strategy_equity": 0.014,
        "recovery_max_hours_strategy_equity": 12.0,
        "recovery_max_hours_strategy_equity_short": 6.0,
        "hsl_trigger_count_long": 2,
        "hsl_restart_count_short": 1,
    }


def test_visible_metrics_none_uses_scoring_and_limits():
    cfg = get_template_config()
    cfg["backtest"]["visible_metrics"] = None
    cfg["optimize"]["scoring"] = ["adg_fraction_strategy_equity", "adg_fraction"]
    cfg["optimize"]["limits"] = [
        {"metric": "drawdown_max_fraction"},
        {"metric": "loss_profit_ratio"},
    ]

    result = filter_analysis_for_visibility(_make_analysis(), cfg)

    assert list(result.analysis) == [
        "adg_fraction_strategy_equity",
        "adg_fraction_usd",
        "adg_fraction_btc",
        "drawdown_max_fraction_usd",
        "drawdown_max_fraction_btc",
        "loss_profit_ratio",
    ]
    assert result.shown_count == 6
    assert result.total_count == 12


def test_visible_metrics_empty_list_shows_all():
    cfg = get_template_config()
    cfg["backtest"]["visible_metrics"] = []

    result = filter_analysis_for_visibility(_make_analysis(), cfg)

    assert result.analysis == _make_analysis()
    assert result.shown_count == result.total_count == 12


def test_visible_metrics_explicit_list_unions_with_optimize_metrics():
    cfg = get_template_config()
    cfg["backtest"]["visible_metrics"] = ["drawdown_max_fraction_strategy_equity"]
    cfg["optimize"]["scoring"] = ["adg_fraction"]
    cfg["optimize"]["limits"] = [{"metric": "loss_profit_ratio"}]

    resolved = resolve_visible_metric_names(cfg, _make_analysis().keys())

    assert resolved == [
        "adg_fraction_usd",
        "adg_fraction_btc",
        "loss_profit_ratio",
        "drawdown_max_fraction_strategy_equity",
    ]


def test_visible_metrics_unknown_explicit_entry_raises():
    cfg = get_template_config()
    cfg["backtest"]["visible_metrics"] = ["not_a_metric"]

    with pytest.raises(ValueError, match="unknown backtest.visible_metrics entries"):
        filter_analysis_for_visibility(_make_analysis(), cfg)


def test_visible_metrics_rejects_invalid_config_type():
    cfg = get_template_config()
    cfg["backtest"]["visible_metrics"] = "adg_fraction"

    with pytest.raises(ValueError, match="backtest.visible_metrics must be null, \\[\\], or"):
        filter_analysis_for_visibility(_make_analysis(), cfg)


def test_validate_visible_metrics_config_rejects_unknown_metric_early():
    cfg = get_template_config()
    cfg["backtest"]["visible_metrics"] = ["not_a_metric"]

    with pytest.raises(ValueError, match="unknown backtest.visible_metrics entries"):
        validate_visible_metrics_config(cfg)


def test_validate_visible_metrics_config_accepts_explicit_hsl_and_hard_stop_metrics():
    cfg = get_template_config()
    cfg["backtest"]["visible_metrics"] = [
        "drawdown_max_fraction_strategy_equity_long",
        "recovery_max_hours_strategy_equity_short",
        "hsl_trigger_count_long",
        "hsl_restart_count_short",
    ]

    validate_visible_metrics_config(cfg)


def test_validate_visible_metrics_config_accepts_fill_activity_metrics():
    cfg = get_template_config()
    cfg["backtest"]["visible_metrics"] = [
        "backtest_completion_fraction",
        "fills_active_days_count",
        "fills_active_day_fraction",
        "fills_active_symbols_count",
        "analysis_duration_days",
        "fills_count",
        "fills_close_count",
        "fills_entry_count",
        "fills_count_long",
        "fills_count_short",
        "entry_to_close_fill_count_ratio",
        "fills_gap_max_days",
        "fills_gap_mean_hours",
        "fills_gap_median_hours",
        "fills_gap_p95_hours",
        "fills_gap_p99_hours",
        "fills_gap_time_weighted_mean_hours",
        "position_held_time_weighted_mean_hours",
        "fills_per_day",
        "fills_close_per_day",
        "fills_entry_per_day",
        "fills_per_day_long",
        "fills_per_day_per_position_slot",
        "fills_per_day_per_position_slot_long",
        "fills_per_day_per_position_slot_short",
        "fills_per_day_short",
        "fills_top_symbol_fraction",
    ]

    validate_visible_metrics_config(cfg)


def test_validate_visible_metrics_config_accepts_gain_quality_metrics():
    cfg = get_template_config()
    cfg["backtest"]["visible_metrics"] = [
        "adg_rolling_hmean_fraction_strategy_equity",
        "adg_time_integrated_fraction_strategy_equity",
        "positive_log_return_participation_score_strategy_equity",
    ]

    validate_visible_metrics_config(cfg)


def test_visible_metrics_exact_hsl_and_hard_stop_names_resolve():
    cfg = get_template_config()
    cfg["backtest"]["visible_metrics"] = [
        "drawdown_max_fraction_strategy_equity_long",
        "recovery_max_hours_strategy_equity_short",
        "hsl_trigger_count_long",
        "hsl_restart_count_short",
    ]
    cfg["optimize"]["scoring"] = []
    cfg["optimize"]["limits"] = []

    resolved = resolve_visible_metric_names(cfg, _make_analysis().keys())

    assert resolved == [
        "drawdown_max_fraction_strategy_equity_long",
        "recovery_max_hours_strategy_equity_short",
        "hsl_trigger_count_long",
        "hsl_restart_count_short",
    ]


def test_visible_metrics_prefix_match_supports_metric_families():
    cfg = get_template_config()
    cfg["backtest"]["visible_metrics"] = ["drawdown_max_fraction"]
    cfg["optimize"]["scoring"] = []
    cfg["optimize"]["limits"] = []

    resolved = resolve_visible_metric_names(cfg, _make_analysis().keys())

    assert resolved == [
        "drawdown_max_fraction_usd",
        "drawdown_max_fraction_btc",
    ]
