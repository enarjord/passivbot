import pytest

from analysis_visibility import filter_analysis_for_visibility, resolve_visible_metric_names
from config_utils import get_template_config


def _make_analysis():
    return {
        "adg_usd": 0.01,
        "adg_btc": 0.02,
        "drawdown_worst_usd": 0.3,
        "drawdown_worst_btc": 0.4,
        "loss_profit_ratio": 0.5,
        "drawdown_worst_hsl": 0.24,
        "adg_strategy_pnl_rebased": 0.014,
        "peak_recovery_hours_hsl": 12.0,
    }


def test_visible_metrics_none_uses_scoring_and_limits():
    cfg = get_template_config()
    cfg["backtest"]["visible_metrics"] = None
    cfg["optimize"]["scoring"] = ["adg_strategy_pnl_rebased", "adg"]
    cfg["optimize"]["limits"] = [
        {"metric": "drawdown_worst"},
        {"metric": "loss_profit_ratio"},
    ]

    result = filter_analysis_for_visibility(_make_analysis(), cfg)

    assert list(result.analysis) == [
        "adg_strategy_pnl_rebased",
        "adg_usd",
        "adg_btc",
        "drawdown_worst_usd",
        "drawdown_worst_btc",
        "loss_profit_ratio",
    ]
    assert result.shown_count == 6
    assert result.total_count == 8


def test_visible_metrics_empty_list_shows_all():
    cfg = get_template_config()
    cfg["backtest"]["visible_metrics"] = []

    result = filter_analysis_for_visibility(_make_analysis(), cfg)

    assert result.analysis == _make_analysis()
    assert result.shown_count == result.total_count == 8


def test_visible_metrics_explicit_list_unions_with_optimize_metrics():
    cfg = get_template_config()
    cfg["backtest"]["visible_metrics"] = ["drawdown_worst_hsl"]
    cfg["optimize"]["scoring"] = ["adg"]
    cfg["optimize"]["limits"] = [{"metric": "loss_profit_ratio"}]

    resolved = resolve_visible_metric_names(cfg, _make_analysis().keys())

    assert resolved == [
        "adg_usd",
        "adg_btc",
        "loss_profit_ratio",
        "drawdown_worst_hsl",
    ]


def test_visible_metrics_unknown_explicit_entry_raises():
    cfg = get_template_config()
    cfg["backtest"]["visible_metrics"] = ["not_a_metric"]

    with pytest.raises(ValueError, match="unknown backtest.visible_metrics entries"):
        filter_analysis_for_visibility(_make_analysis(), cfg)


def test_visible_metrics_rejects_invalid_config_type():
    cfg = get_template_config()
    cfg["backtest"]["visible_metrics"] = "adg"

    with pytest.raises(ValueError, match="backtest.visible_metrics must be null, \\[\\], or"):
        filter_analysis_for_visibility(_make_analysis(), cfg)


def test_visible_metrics_prefix_match_supports_metric_families():
    cfg = get_template_config()
    cfg["backtest"]["visible_metrics"] = ["drawdown_worst"]
    cfg["optimize"]["scoring"] = []
    cfg["optimize"]["limits"] = []

    resolved = resolve_visible_metric_names(cfg, _make_analysis().keys())

    assert resolved == [
        "drawdown_worst_usd",
        "drawdown_worst_btc",
    ]
