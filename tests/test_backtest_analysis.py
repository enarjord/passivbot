import logging

import numpy as np
import pandas as pd
import pytest

import backtest as bt
import plotting
from backtest import (
    BacktestPayload,
    DISABLE_PLOTTING_HELP,
    calc_backtest_completion_fraction,
    execute_backtest,
    expand_analysis,
    parse_disabled_plot_groups,
    process_forager_fills,
)
from plotting import (
    create_forager_balance_figures,
    create_forager_hard_stop_drawdown_figure,
)


def _make_analysis_entry(value):
    base_keys = [
        "adg_fraction",
        "mdg_fraction",
        "gain_multiplier",
        "position_episodes_per_day",
        "position_held_time_weighted_mean_hours",
        "position_held_mean_hours",
        "position_held_max_hours",
        "position_held_median_hours",
        "position_unchanged_max_hours",
        "position_held_mean_days",
        "position_held_max_days",
        "position_held_median_days",
        "position_unchanged_max_days",
        "trade_win_fraction",
        "trade_loss_max_fraction",
        "trade_loss_mean_fraction",
        "trade_loss_median_fraction",
        "adg_to_paper_loss_max_ratio",
        "adg_to_paper_loss_mean_ratio",
        "exposure_ratio",
        "adg_to_absolute_net_exposure_mean_ratio",
        "loss_profit_ratio",
        "active_day_turnover_proxy_mean_ratio",
        "peak_recovery_max_hours_realized_pnl",
        "peak_recovery_max_days_realized_pnl",
        "absolute_net_wallet_exposure_max_ratio",
        "absolute_net_wallet_exposure_mean_ratio",
        "absolute_net_wallet_exposure_median_ratio",
        "high_exposure_mean_hours_long",
        "high_exposure_max_hours_long",
        "high_exposure_mean_hours_short",
        "high_exposure_max_hours_short",
        "high_exposure_mean_days_long",
        "high_exposure_max_days_long",
        "high_exposure_mean_days_short",
        "high_exposure_max_days_short",
        "initial_entry_balance_fraction_long",
        "initial_entry_balance_fraction_short",
        "entry_interval_p95_hours",
        "entry_interval_p99_hours",
        "recovery_mean_days_strategy_equity",
        "recovery_median_days_strategy_equity",
        "recovery_p95_days_strategy_equity",
        "recovery_p99_days_strategy_equity",
        "recovery_worst_5pct_mean_days_strategy_equity",
        "recovery_worst_1pct_mean_days_strategy_equity",
        "recovery_max_days_strategy_equity",
    ]
    analysis = {key: value for key in base_keys}
    analysis.update(
        {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "omega_ratio": 0.0,
            "expected_shortfall_1pct_fraction": 0.0,
            "calmar_ratio": 0.0,
            "sterling_ratio": 0.0,
            "drawdown_max_fraction": 0.0,
            "drawdown_daily_worst_1pct_mean_fraction": 0.0,
            "pain_index_fraction_strategy_equity": 0.0,
            "ulcer_index_fraction_strategy_equity": 0.0,
            "equity_balance_gap_negative_max_fraction": 0.0,
            "equity_balance_gap_negative_mean_fraction": 0.0,
            "equity_balance_gap_positive_max_fraction": 0.0,
            "equity_balance_gap_positive_mean_fraction": 0.0,
            "peak_recovery_max_hours": 0.0,
            "peak_recovery_max_days": 0.0,
            "equity_choppiness_ratio": 0.0,
            "equity_jerkiness_ratio": 0.0,
            "exponential_fit_mse_log_ratio_squared": 0.0,
            "adg_fraction_per_twel_ratio_long": 0.0,
            "adg_fraction_per_twel_ratio_short": 0.0,
        }
    )
    return analysis


def test_backtest_hsl_scope_requires_canonical_global_config():
    assert (
        bt._resolve_backtest_hsl_signal_mode({"bot": {"portfolio": {'hsl': {'scope': 'coin'}}}})
        == "coin"
    )

    with pytest.raises(KeyError, match="bot.portfolio.hsl.scope"):
        bt._resolve_backtest_hsl_signal_mode({"bot": {"portfolio": {'hsl': {}}}})


def test_expand_analysis_includes_entry_balance_pct():
    analysis_usd = _make_analysis_entry(0.123)
    analysis_btc = _make_analysis_entry(0.456)
    config = {
        "bot": {
            "long": {"total_wallet_exposure_limit_ratio": 1.0},
            "short": {"total_wallet_exposure_limit_ratio": 1.0},
        }
    }
    result = expand_analysis(
        analysis_usd,
        analysis_btc,
        fills=np.empty((0, 0)),
        equities_array=np.empty((0, 3)),
        config=config,
    )
    assert "initial_entry_balance_fraction_long" in result
    assert "initial_entry_balance_fraction_short" in result
    assert result["initial_entry_balance_fraction_long"] == 0.123
    assert result["initial_entry_balance_fraction_short"] == 0.123


def test_expand_analysis_includes_high_exposure_hours():
    analysis_usd = _make_analysis_entry(0.5)
    analysis_btc = _make_analysis_entry(0.5)
    config = {
        "bot": {
            "long": {"total_wallet_exposure_limit_ratio": 1.0},
            "short": {"total_wallet_exposure_limit_ratio": 1.0},
        }
    }
    result = expand_analysis(
        analysis_usd,
        analysis_btc,
        fills=np.empty((0, 0)),
        equities_array=np.empty((0, 3)),
        config=config,
    )
    for side in ("long", "short"):
        assert f"high_exposure_mean_hours_{side}" in result
        assert f"high_exposure_max_hours_{side}" in result
        assert result[f"high_exposure_mean_hours_{side}"] == 0.5
        assert result[f"high_exposure_max_hours_{side}"] == 0.5


def test_expand_analysis_includes_day_duration_metrics():
    analysis_usd = _make_analysis_entry(2.0)
    analysis_btc = _make_analysis_entry(4.0)
    analysis_usd["peak_recovery_max_days"] = 1.25
    analysis_btc["peak_recovery_max_days"] = 2.5
    config = {
        "bot": {
            "long": {"total_wallet_exposure_limit_ratio": 1.0},
            "short": {"total_wallet_exposure_limit_ratio": 1.0},
        }
    }
    result = expand_analysis(
        analysis_usd,
        analysis_btc,
        fills=np.empty((0, 0)),
        equities_array=np.empty((0, 3)),
        config=config,
    )

    assert result["position_held_max_days"] == 2.0
    assert result["position_unchanged_max_days"] == 2.0
    assert result["peak_recovery_max_days_realized_pnl"] == 2.0
    assert result["high_exposure_max_days_long"] == 2.0
    assert result["high_exposure_max_days_short"] == 2.0
    assert result["peak_recovery_max_days_usd"] == 1.25
    assert result["peak_recovery_max_days_btc"] == 2.5


def test_expand_analysis_includes_trade_level_metrics():
    analysis_usd = _make_analysis_entry(0.25)
    analysis_btc = _make_analysis_entry(0.75)
    config = {
        "bot": {
            "long": {"total_wallet_exposure_limit_ratio": 1.0},
            "short": {"total_wallet_exposure_limit_ratio": 1.0},
        }
    }

    result = expand_analysis(
        analysis_usd,
        analysis_btc,
        fills=np.empty((0, 0)),
        equities_array=np.empty((0, 3)),
        config=config,
    )

    assert result["trade_win_fraction"] == 0.25
    assert result["trade_loss_max_fraction"] == 0.25
    assert result["trade_loss_mean_fraction"] == 0.25
    assert result["trade_loss_median_fraction"] == 0.25
    assert "win_rate_usd" not in result
    assert "trade_loss_max_btc" not in result


def test_expand_analysis_currency_suffixes_new_ratio_metrics():
    analysis_usd = _make_analysis_entry(0.25)
    analysis_btc = _make_analysis_entry(0.75)
    config = {
        "bot": {
            "long": {"total_wallet_exposure_limit_ratio": 1.0},
            "short": {"total_wallet_exposure_limit_ratio": 1.0},
        }
    }

    result = expand_analysis(
        analysis_usd,
        analysis_btc,
        fills=np.empty((0, 0)),
        equities_array=np.empty((0, 3)),
        config=config,
    )

    for metric in (
        "adg_to_paper_loss_max_ratio",
        "adg_to_paper_loss_mean_ratio",
        "exposure_ratio",
        "adg_to_absolute_net_exposure_mean_ratio",
    ):
        assert result[f"{metric}_usd"] == 0.25
        assert result[f"{metric}_btc"] == 0.75
        assert metric not in result


def test_calc_backtest_completion_ratio_is_full_when_final_candle_reaches_end():
    config = {
        "backtest": {
            "start_date": "2026-01-01",
            "end_date": "2026-01-02",
            "candle_interval_minutes": 1,
        }
    }
    equities = np.array(
        [
            [1767225600000, 1000.0, 1.0],
            [1767311940000, 1000.0, 1.0],
        ],
        dtype=np.float64,
    )

    assert calc_backtest_completion_fraction(equities, config) == pytest.approx(1.0)


def test_calc_backtest_completion_ratio_detects_early_stop():
    config = {
        "backtest": {
            "start_date": "2026-01-01",
            "end_date": "2026-01-03",
            "candle_interval_minutes": 1,
        }
    }
    equities = np.array(
        [
            [1767225600000, 1000.0, 1.0],
            [1767311940000, 50.0, 0.05],
        ],
        dtype=np.float64,
    )

    assert calc_backtest_completion_fraction(equities, config) == pytest.approx(0.5)


def test_expand_analysis_includes_backtest_completion_ratio_when_available():
    analysis_usd = _make_analysis_entry(0.25)
    analysis_btc = _make_analysis_entry(0.75)
    config = {
        "backtest": {
            "start_date": "2026-01-01",
            "end_date": "2026-01-02",
            "candle_interval_minutes": 1,
        },
        "bot": {
            "long": {"total_wallet_exposure_limit_ratio": 1.0},
            "short": {"total_wallet_exposure_limit_ratio": 1.0},
        },
    }
    equities = np.array(
        [
            [1767225600000, 1000.0, 1.0],
            [1767311940000, 1000.0, 1.0],
        ],
        dtype=np.float64,
    )

    result = expand_analysis(
        analysis_usd,
        analysis_btc,
        fills=np.empty((0, 0)),
        equities_array=equities,
        config=config,
    )

    assert result["backtest_completion_fraction"] == pytest.approx(1.0)


def test_execute_backtest_metrics_only_keeps_completion_ratio(monkeypatch):
    analysis_usd = _make_analysis_entry(0.25)
    analysis_btc = _make_analysis_entry(0.25)
    equities = np.array(
        [
            [1767225600000],
            [1767311940000],
        ],
        dtype=np.float64,
    )

    def fake_run_backtest_bundle(*_args):
        return None, equities, analysis_usd, analysis_btc

    monkeypatch.setattr(bt.pbr, "run_backtest_bundle", fake_run_backtest_bundle, raising=False)
    payload = BacktestPayload(
        bundle=object(),
        bot_params_list=[],
        strategy_params_list=[],
        exchange_params=[],
        backtest_params={"metrics_only": True},
    )
    config = {
        "backtest": {
            "start_date": "2026-01-01",
            "end_date": "2026-01-02",
            "candle_interval_minutes": 1,
        },
        "bot": {
            "long": {"total_wallet_exposure_limit_ratio": 1.0},
            "short": {"total_wallet_exposure_limit_ratio": 1.0},
        },
    }

    fills, equities_array, analysis = execute_backtest(payload, config)

    assert fills is None
    assert equities_array is None
    assert analysis["backtest_completion_fraction"] == pytest.approx(1.0)


def test_execute_backtest_captures_metrics_only_rust_profile(monkeypatch):
    analysis_usd = _make_analysis_entry(0.25)
    analysis_btc = _make_analysis_entry(0.25)
    rust_profile = {"rust_simulation_ms": 12.5, "rust_analysis_pair_ms": 3.25}
    equities = np.array(
        [
            [1767225600000],
            [1767311940000],
        ],
        dtype=np.float64,
    )

    def fake_run_backtest_bundle(*_args):
        return (
            None,
            equities,
            analysis_usd,
            analysis_btc,
            {"_rust_profile": rust_profile},
        )

    monkeypatch.setattr(bt.pbr, "run_backtest_bundle", fake_run_backtest_bundle, raising=False)
    payload = BacktestPayload(
        bundle=object(),
        bot_params_list=[],
        strategy_params_list=[],
        exchange_params=[],
        backtest_params={"metrics_only": True},
    )
    config = {
        "backtest": {
            "start_date": "2026-01-01",
            "end_date": "2026-01-02",
            "candle_interval_minutes": 1,
        },
        "bot": {
            "long": {"total_wallet_exposure_limit_ratio": 1.0},
            "short": {"total_wallet_exposure_limit_ratio": 1.0},
        },
    }

    fills, equities_array, _analysis = execute_backtest(payload, config)

    assert fills is None
    assert equities_array is None
    assert payload.rust_profile == rust_profile


def test_expand_analysis_allows_skipped_btc_analysis():
    analysis_usd = _make_analysis_entry(0.25)
    result = expand_analysis(
        analysis_usd,
        {},
        fills=None,
        equities_array=None,
        config={
            "bot": {
                "long": {"total_wallet_exposure_limit_ratio": 1.0},
                "short": {"total_wallet_exposure_limit_ratio": 1.0},
            }
        },
    )

    assert result["adg_fraction_usd"] == pytest.approx(0.25)
    assert result["adg_fraction_per_twel_ratio_usd_long"] == pytest.approx(0.25)
    assert "adg_fraction_btc" not in result
    assert "adg_fraction_per_twel_ratio_btc_long" not in result


def test_legacy_make_table_includes_trade_metrics():
    table = plotting.make_table(
        {
            "exchange": "binance",
            "market_type": "futures",
            "symbol": "BTC/USDT:USDT",
            "passivbot_mode": "recursive_grid",
            "adg_n_subdivisions": 10,
            "long": {"enabled": False},
            "short": {"enabled": False},
            "result": {
                "n_days": 7.0,
                "starting_balance": 1000.0,
                "win_rate": 0.625,
                "trade_loss_max": 0.015,
            },
        }
    ).get_string()

    assert "Win rate" in table
    assert "62.5%" in table
    assert "Worst trade loss" in table
    assert "1.5%" in table


def test_expand_analysis_deduplicates_hard_stop_metrics():
    analysis_usd = _make_analysis_entry(0.5)
    analysis_btc = _make_analysis_entry(0.5)
    analysis_usd.update(
        {
            "hsl_trigger_count": 3,
            "hsl_triggers_per_year": 36.5,
            "hsl_restart_count": 2,
            "hsl_panic_loss_starting_balance_fraction": 0.125,
            "hsl_restarts_per_year": 24.3333333333,
        }
    )
    analysis_btc.update(
        {
            "hsl_trigger_count": 3,
            "hsl_triggers_per_year": 36.5,
            "hsl_restart_count": 2,
            "hsl_panic_loss_starting_balance_fraction": 0.125,
            "hsl_restarts_per_year": 24.3333333333,
        }
    )
    config = {
        "bot": {
            "long": {"total_wallet_exposure_limit_ratio": 1.0},
            "short": {"total_wallet_exposure_limit_ratio": 1.0},
        }
    }

    result = expand_analysis(
        analysis_usd,
        analysis_btc,
        fills=np.empty((0, 0)),
        equities_array=np.empty((0, 3)),
        config=config,
    )

    assert result["hsl_trigger_count"] == 3
    assert result["hsl_triggers_per_year"] == 36.5
    assert result["hsl_restart_count"] == 2
    assert result["hsl_panic_loss_starting_balance_fraction"] == 0.125
    assert result["hsl_restarts_per_year"] == 24.3333333333
    assert "hard_stop_triggers_usd" not in result
    assert "hard_stop_triggers_btc" not in result
    assert "hard_stop_triggers_per_year_usd" not in result
    assert "hard_stop_triggers_per_year_btc" not in result
    assert "hard_stop_restarts_usd" not in result
    assert "hard_stop_restarts_btc" not in result
    assert "hsl_panic_loss_starting_balance_fraction_usd" not in result
    assert "hsl_panic_loss_starting_balance_fraction_btc" not in result
    assert "hard_stop_restarts_per_year_usd" not in result
    assert "hard_stop_restarts_per_year_btc" not in result


def test_process_forager_fills_handles_zero_pnl_division():
    """Zero-PnL inputs should not raise and should return stable neutral ratios."""
    equities_array = np.array([[1704067200000, 1000.0, 0.02]], dtype=np.float64)

    _fdf, analysis_appendix, _bal_eq = process_forager_fills(
        fills=[],
        coins=[],
        hlcvs=np.empty((0, 0), dtype=np.float64),
        equities_array=equities_array,
        balance_sample_divider=1,
    )

    assert analysis_appendix["loss_profit_ratio_long"] == 1.0
    assert analysis_appendix["loss_profit_ratio_short"] == 1.0
    assert analysis_appendix["long_pnl_to_total_pnl_ratio"] == 0.5


def test_process_forager_fills_no_fills_keeps_datetime_index_for_resample():
    """No-fill balance/equity joins should stay resample-safe on a DatetimeIndex."""
    t0 = 1_740_000_000_000
    equities_array = np.array(
        [
            [t0, 1000.0, 0.02],
            [t0 + 3_600_000, 1000.5, 0.02],
        ],
        dtype=np.float64,
    )

    fdf, _analysis_appendix, bal_eq = process_forager_fills(
        fills=[],
        coins=[],
        hlcvs=np.empty((0, 0), dtype=np.float64),
        equities_array=equities_array,
        balance_sample_divider=60,
    )

    assert fdf.empty
    assert isinstance(bal_eq.index, pd.DatetimeIndex)
    assert not bal_eq.empty


def test_expand_analysis_keeps_rust_fill_activity_metrics_shared():
    analysis_usd = _make_analysis_entry(0.25)
    analysis_btc = _make_analysis_entry(0.75)
    fill_metrics = {
        "fills_active_days_count": 1.0,
        "fills_active_day_fraction": 1.0,
        "fills_active_symbols_count": 2.0,
        "analysis_duration_days": 4.0 / 24.0,
        "fills_count": 2.0,
        "fills_close_count": 1.0,
        "fills_entry_count": 1.0,
        "fills_count_long": 2.0,
        "fills_count_short": 0.0,
        "entry_to_close_fill_count_ratio": 1.0,
        "fills_gap_max_days": 2.0 / 24.0,
        "fills_gap_mean_hours": 4.0 / 3.0,
        "fills_gap_median_hours": 1.0,
        "fills_gap_p95_hours": 1.9,
        "fills_gap_p99_hours": 1.98,
        "fills_gap_time_weighted_mean_hours": 1.5,
        "fills_per_day": 12.0,
        "fills_close_per_day": 6.0,
        "fills_entry_per_day": 6.0,
        "fills_per_day_long": 12.0,
        "fills_per_day_per_position_slot": 6.0,
        "fills_per_day_per_position_slot_long": 6.0,
        "fills_per_day_per_position_slot_short": 0.0,
        "fills_per_day_short": 0.0,
        "fills_top_symbol_fraction": 0.5,
    }
    analysis_usd.update(fill_metrics)
    analysis_btc.update(fill_metrics)

    result = expand_analysis(
        analysis_usd,
        analysis_btc,
        fills=np.empty((0, 0)),
        equities_array=np.empty((0, 3)),
        config={
            "bot": {
                "long": {"total_wallet_exposure_limit_ratio": 1.0},
                "short": {"total_wallet_exposure_limit_ratio": 0.0},
            }
        },
    )

    for key, value in fill_metrics.items():
        assert result[key] == pytest.approx(value)
        assert f"{key}_usd" not in result
        assert f"{key}_btc" not in result


def test_process_forager_fills_no_fills_leaves_fill_activity_to_rust_analysis():
    start = 1_740_000_000_000
    one_hour = 3_600_000
    equities_array = np.array(
        [
            [start, 1000.0, 0.02],
            [start + one_hour, 1001.0, 0.02],
            [start + 2 * one_hour, 1002.0, 0.02],
        ],
        dtype=np.float64,
    )

    _fdf, analysis_appendix, _bal_eq = process_forager_fills(
        fills=[],
        coins=[],
        hlcvs=np.empty((0, 0), dtype=np.float64),
        equities_array=equities_array,
        balance_sample_divider=1,
    )

    assert "fills_per_day" not in analysis_appendix


def test_process_forager_fills_includes_strategy_equity_column():
    t0 = 1_740_000_000_000
    equities_array = np.array(
        [
            [t0, 1000.0, 0.02, 1000.0],
            [t0 + 60_000, 990.0, 0.0198, 995.0],
        ],
        dtype=np.float64,
    )

    _fdf, _analysis_appendix, bal_eq = process_forager_fills(
        fills=[],
        coins=[],
        hlcvs=np.empty((0, 0), dtype=np.float64),
        equities_array=equities_array,
        balance_sample_divider=1,
    )

    assert "strategy_equity" in bal_eq.columns
    assert np.isclose(bal_eq["strategy_equity"].iloc[0], 1000.0)
    assert np.isclose(bal_eq["strategy_equity"].iloc[-1], 995.0)


def test_create_forager_balance_figures_adds_strategy_equity_drawdown():
    idx = pd.date_range("2025-01-01", periods=4, freq="1h")
    bal_eq = pd.DataFrame(
        {
            "usd_total_equity": [100.0, 90.0, 120.0, 96.0],
            "strategy_equity": [100.0, 90.0, 120.0, 96.0],
            "btc_total_equity": [1.0, 0.9, 1.2, 0.96],
        },
        index=idx,
    )

    figures = create_forager_balance_figures(
        bal_eq,
        autoplot=False,
        return_figures=True,
    )

    assert "drawdown" in figures
    plotted_drawdown = figures["drawdown"].axes[0].lines[0].get_ydata()
    np.testing.assert_allclose(plotted_drawdown, [0.0, 0.1, 0.0, 0.2])
    plotting.plt.close("all")


def test_post_process_disable_plotting_skips_all_figure_generation(
    tmp_path, monkeypatch
):
    calls = {"balance": 0, "twe": 0, "pnl": 0, "save": 0, "coin": 0}

    def _fake_process_forager_fills(*args, **kwargs):
        fdf = pd.DataFrame(columns=["coin", "pnl"])
        bal_eq = pd.DataFrame({"balance": [1000.0], "equity": [1000.0]})
        return fdf, {}, bal_eq

    monkeypatch.setattr(bt, "process_forager_fills", _fake_process_forager_fills)
    monkeypatch.setattr(bt, "format_config", lambda config, verbose=False: config)
    monkeypatch.setattr(bt, "strip_config_metadata", lambda config: config)
    monkeypatch.setattr(
        bt, "dump_backtest_dataset_metadata", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        bt,
        "dump_config",
        lambda config, path: open(path, "w", encoding="utf-8").write("{}"),
    )

    monkeypatch.setattr(
        bt,
        "create_forager_balance_figures",
        lambda *args, **kwargs: calls.__setitem__("balance", calls["balance"] + 1)
        or {},
    )
    monkeypatch.setattr(
        bt,
        "create_forager_twe_figure",
        lambda *args, **kwargs: calls.__setitem__("twe", calls["twe"] + 1) or {},
    )
    monkeypatch.setattr(
        bt,
        "create_forager_pnl_figure",
        lambda *args, **kwargs: calls.__setitem__("pnl", calls["pnl"] + 1) or {},
    )
    monkeypatch.setattr(
        bt,
        "save_figures",
        lambda *args, **kwargs: calls.__setitem__("save", calls["save"] + 1) or {},
    )
    monkeypatch.setattr(
        bt,
        "create_forager_coin_figures",
        lambda *args, **kwargs: calls.__setitem__("coin", calls["coin"] + 1) or {},
    )

    config = {
        "disable_plotting": True,
        "backtest": {"balance_sample_divider": 60, "coins": {"binance": ["BTC"]}},
        "bot": {
            "long": {"risk": {"total_wallet_exposure_limit_ratio": 1.0}},
            "short": {"risk": {"total_wallet_exposure_limit_ratio": 0.0}},
        },
        "live": {},
    }

    bt.post_process(
        plot_context=bt.BacktestPlotContext(
            hard_stop_plot_data={
                "hsl": {
                    "schema_version": 1,
                    "engine": "hsl",
                    "mode": "coin",
                    "detailed": False,
                    "scopes": [],
                    "samples": [],
                    "events": [],
                }
            }
        ),
        config=config,
        hlcvs=np.zeros((1, 1, 3), dtype=np.float64),
        fills=[],
        equities_array=np.array([[1704067200000, 1000.0, 1000.0]], dtype=np.float64),
        btc_usd_prices=np.array([]),
        analysis={"gain_multiplier_usd": 1.0},
        results_path=str(tmp_path),
        exchange="binance",
    )

    assert calls == {"balance": 0, "twe": 0, "pnl": 0, "save": 0, "coin": 0}


def test_post_process_writes_original_config_for_dataset_override(
    tmp_path, monkeypatch
):
    dumped = []

    def _fake_process_forager_fills(*args, **kwargs):
        fdf = pd.DataFrame(columns=["coin", "pnl"])
        bal_eq = pd.DataFrame({"balance": [1000.0], "equity": [1000.0]})
        return fdf, {}, bal_eq

    def _fake_dump_config(config, path):
        dumped.append((config, path))
        open(path, "w", encoding="utf-8").write("{}")

    monkeypatch.setattr(bt, "process_forager_fills", _fake_process_forager_fills)
    monkeypatch.setattr(bt, "format_config", lambda config, verbose=False: config)
    monkeypatch.setattr(bt, "dump_config", _fake_dump_config)
    monkeypatch.setattr(
        bt, "create_forager_balance_figures", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(bt, "create_forager_twe_figure", lambda *args, **kwargs: {})
    monkeypatch.setattr(bt, "create_forager_pnl_figure", lambda *args, **kwargs: {})
    monkeypatch.setattr(bt, "save_figures", lambda *args, **kwargs: {})
    monkeypatch.setattr(bt, "create_forager_coin_figures", lambda *args, **kwargs: {})

    original_config = {
        "backtest": {"start_date": "2025-01-01", "coins": {"binance": ["ETH"]}},
        "live": {},
        "bot": {
            "long": {"universe": {"approved_coins": ["ETH"]}},
            "short": {"universe": {"approved_coins": []}},
        },
    }
    config = {
        "disable_plotting": True,
        "backtest": {
            "balance_sample_divider": 60,
            "coins": {"binance": ["BTC"]},
            "hlcvs_data_dir": "caches/hlcvs_data/custom__abc",
        },
        "bot": {
            "long": {
                "risk": {"total_wallet_exposure_limit_ratio": 1.0},
                "universe": {"approved_coins": ["BTC"]},
            },
            "short": {
                "risk": {"total_wallet_exposure_limit_ratio": 0.0},
                "universe": {"approved_coins": []},
            },
        },
        "live": {},
        "_original_backtest_config": original_config,
        "metrics": {"stale": True},
        "suite_metrics": {"stale": True},
    }

    bt.post_process(
        plot_context=bt.BacktestPlotContext(
            hard_stop_plot_data={
                "hsl": {
                    "schema_version": 1,
                    "engine": "hsl",
                    "mode": "coin",
                    "detailed": False,
                    "scopes": [],
                    "samples": [],
                    "events": [],
                }
            }
        ),
        config=config,
        hlcvs=np.zeros((1, 1, 3), dtype=np.float64),
        fills=[],
        equities_array=np.array([[1704067200000, 1000.0, 1000.0]], dtype=np.float64),
        btc_usd_prices=np.array([]),
        analysis={"gain_multiplier_usd": 1.0},
        results_path=str(tmp_path),
        exchange="binance",
    )

    dumped_by_name = {path.split("/")[-1]: cfg for cfg, path in dumped}
    assert dumped_by_name["config.original.json"]["backtest"]["coins"]["binance"] == [
        "ETH"
    ]
    assert dumped_by_name["config.original.json"]["bot"]["long"]["universe"]["approved_coins"] == [
        "ETH"
    ]
    assert dumped_by_name["config.json"]["backtest"]["coins"]["binance"] == ["BTC"]
    assert dumped_by_name["config.json"]["metrics"]["stats"]["gain_multiplier_usd"]["mean"] == 1.0
    assert "suite_metrics" not in dumped_by_name["config.json"]
    assert "metrics" not in dumped_by_name["config.original.json"]


def test_post_process_disable_plotting_coin_fills_only(tmp_path, monkeypatch):
    calls = {"balance": 0, "twe": 0, "pnl": 0, "save": 0, "coin": 0}

    def _fake_process_forager_fills(*args, **kwargs):
        fdf = pd.DataFrame(columns=["coin", "pnl"])
        bal_eq = pd.DataFrame({"balance": [1000.0], "equity": [1000.0]})
        return fdf, {}, bal_eq

    monkeypatch.setattr(bt, "process_forager_fills", _fake_process_forager_fills)
    monkeypatch.setattr(bt, "format_config", lambda config, verbose=False: config)
    monkeypatch.setattr(bt, "strip_config_metadata", lambda config: config)
    monkeypatch.setattr(
        bt, "dump_backtest_dataset_metadata", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        bt,
        "dump_config",
        lambda config, path: open(path, "w", encoding="utf-8").write("{}"),
    )

    monkeypatch.setattr(
        bt,
        "create_forager_balance_figures",
        lambda *args, **kwargs: calls.__setitem__("balance", calls["balance"] + 1)
        or {"balance": object()},
    )
    monkeypatch.setattr(
        bt,
        "create_forager_twe_figure",
        lambda *args, **kwargs: calls.__setitem__("twe", calls["twe"] + 1)
        or {"twe": object()},
    )
    monkeypatch.setattr(
        bt,
        "create_forager_pnl_figure",
        lambda *args, **kwargs: calls.__setitem__("pnl", calls["pnl"] + 1)
        or {"pnl": object()},
    )
    monkeypatch.setattr(
        bt,
        "save_figures",
        lambda *args, **kwargs: calls.__setitem__("save", calls["save"] + 1) or {},
    )
    monkeypatch.setattr(
        bt,
        "create_forager_coin_figures",
        lambda *args, **kwargs: calls.__setitem__("coin", calls["coin"] + 1) or {},
    )

    config = {
        "disable_plotting": "coin_fills",
        "backtest": {"balance_sample_divider": 60, "coins": {"binance": ["BTC"]}},
        "bot": {
            "long": {"risk": {"total_wallet_exposure_limit_ratio": 1.0}},
            "short": {"risk": {"total_wallet_exposure_limit_ratio": 0.0}},
        },
        "live": {},
    }

    bt.post_process(
        plot_context=bt.BacktestPlotContext(
            hard_stop_plot_data={
                "hsl": {
                    "schema_version": 1,
                    "engine": "hsl",
                    "mode": "coin",
                    "detailed": False,
                    "scopes": [],
                    "samples": [],
                    "events": [],
                }
            }
        ),
        config=config,
        hlcvs=np.zeros((1, 1, 3), dtype=np.float64),
        fills=[],
        equities_array=np.array([[1704067200000, 1000.0, 1000.0]], dtype=np.float64),
        btc_usd_prices=np.array([]),
        analysis={"gain_multiplier_usd": 1.0},
        results_path=str(tmp_path),
        exchange="binance",
    )

    assert calls == {"balance": 1, "twe": 1, "pnl": 1, "save": 3, "coin": 0}


def test_post_process_visible_metrics_filters_logged_output_only(
    tmp_path, monkeypatch, caplog
):
    def _fake_process_forager_fills(*args, **kwargs):
        fdf = pd.DataFrame(columns=["coin", "pnl"])
        bal_eq = pd.DataFrame({"balance": [1000.0], "equity": [1000.0]})
        return fdf, {}, bal_eq

    monkeypatch.setattr(bt, "process_forager_fills", _fake_process_forager_fills)
    monkeypatch.setattr(bt, "format_config", lambda config, verbose=False: config)
    monkeypatch.setattr(bt, "strip_config_metadata", lambda config: config)
    monkeypatch.setattr(
        bt, "dump_backtest_dataset_metadata", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        bt,
        "dump_config",
        lambda config, path: open(path, "w", encoding="utf-8").write("{}"),
    )
    monkeypatch.setattr(
        bt, "create_forager_balance_figures", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(bt, "create_forager_twe_figure", lambda *args, **kwargs: {})
    monkeypatch.setattr(bt, "create_forager_pnl_figure", lambda *args, **kwargs: {})
    monkeypatch.setattr(bt, "save_figures", lambda *args, **kwargs: {})
    monkeypatch.setattr(bt, "create_forager_coin_figures", lambda *args, **kwargs: {})

    config = {
        "disable_plotting": True,
        "backtest": {
            "balance_sample_divider": 60,
            "coins": {"binance": ["BTC"]},
            "visible_metrics": None,
        },
        "bot": {
            "long": {"risk": {"total_wallet_exposure_limit_ratio": 1.0}},
            "short": {"risk": {"total_wallet_exposure_limit_ratio": 0.0}},
        },
        "live": {},
        "optimize": {
            "scoring": ["adg_fraction"],
            "limits": [{"metric": "loss_profit_ratio"}],
        },
    }

    caplog.set_level(logging.INFO)
    bt.post_process(
        plot_context=bt.BacktestPlotContext(
            hard_stop_plot_data={
                "hsl": {
                    "schema_version": 1,
                    "engine": "hsl",
                    "mode": "coin",
                    "detailed": False,
                    "scopes": [],
                    "samples": [],
                    "events": [],
                }
            }
        ),
        config=config,
        hlcvs=np.zeros((1, 1, 3), dtype=np.float64),
        fills=[],
        equities_array=np.array([[1704067200000, 1000.0, 1000.0]], dtype=np.float64),
        btc_usd_prices=np.array([]),
        analysis={
            "adg_fraction_usd": 0.01,
            "adg_fraction_btc": 0.02,
            "loss_profit_ratio": 0.5,
            "recovery_max_hours_strategy_equity": 12.0,
        },
        results_path=str(tmp_path),
        exchange="binance",
    )

    logged = caplog.text
    assert "Showing 3 of 4 metrics" in logged
    assert "adg_fraction_usd" in logged
    assert "adg_fraction_btc" in logged
    assert "loss_profit_ratio" in logged
    assert "recovery_max_hours_strategy_equity" not in logged
    analysis_path = next(tmp_path.glob("*/analysis.json"))
    assert "recovery_max_hours_strategy_equity" in analysis_path.read_text(encoding="utf-8")


def test_parse_disabled_plot_groups_accepts_summary_alias_and_commas():
    assert parse_disabled_plot_groups("summary,coin_fills") == {
        "balance",
        "twe",
        "pnl",
        "hard_stop",
        "coin_fills",
    }


def test_disable_plotting_help_lists_every_plot_group():
    assert "hard_stop" in DISABLE_PLOTTING_HELP
    assert "summary disables balance, twe, pnl, and hard_stop" in DISABLE_PLOTTING_HELP
    assert "coin_fills disables per-coin fill plots only" in DISABLE_PLOTTING_HELP


@pytest.mark.parametrize("columns", [1, 4])
@pytest.mark.parametrize("count", [0, 1, 3])
def test_analysis_period_uses_actual_equity_timestamps(columns, count):
    timestamps = np.array([1704153600000, 1704196800000, 1704240000000])[:count]
    equities = np.ones((count, columns))
    equities[:, 0] = timestamps
    duration = 1.0 if count == 3 else 0.0
    result = expand_analysis(
        {"analysis_duration_days": duration},
        {},
        None,
        equities,
        {
            "bot": {"long": {}, "short": {}},
            "backtest": {"start_date": "2020-01-01", "end_date": "2025-01-01"},
        },
    )
    assert result["analysis_duration_days"] == result["analysis_duration_days"] == duration
    assert result["effective_start_date"] == ("2024-01-02T00:00:00Z" if count else None)
    assert result["effective_end_date"] == (
        "2024-01-03T00:00:00Z"
        if count == 3
        else "2024-01-02T00:00:00Z" if count else None
    )


def test_post_process_persists_artifacts_with_infinite_diagnostic(
    tmp_path, monkeypatch
):
    import json
    from copy import deepcopy

    fdf = pd.DataFrame(columns=["coin", "pnl"])
    bal_eq = pd.DataFrame({"balance": [1000.0, 1000.0], "equity": [1000.0, 1000.0]})
    monkeypatch.setattr(
        bt, "process_forager_fills", lambda *args, **kwargs: (fdf, {}, bal_eq)
    )
    monkeypatch.setattr(bt, "sanitize_prepared_config_for_dump", deepcopy)
    monkeypatch.setattr(
        bt, "dump_backtest_dataset_metadata", lambda *args, **kwargs: None
    )
    plot_calls = []
    monkeypatch.setattr(
        bt,
        "create_forager_balance_figures",
        lambda *args, **kwargs: plot_calls.append("balance") or {},
    )
    monkeypatch.setattr(
        bt, "save_figures", lambda *args, **kwargs: plot_calls.append("save")
    )
    config = {
        "disable_plotting": ["twe", "pnl", "hard_stop", "coin_fills"],
        "backtest": {"balance_sample_divider": 60, "coins": {"binance": ["BTC"]}},
        "bot": {
            "long": {"total_wallet_exposure_limit_ratio": 1.0},
            "short": {"total_wallet_exposure_limit_ratio": 0.0},
        },
        "live": {},
    }
    bt.post_process(
        plot_context=bt.BacktestPlotContext(
            hard_stop_plot_data={
                "hsl": {
                    "schema_version": 1,
                    "engine": "hsl",
                    "mode": "coin",
                    "detailed": False,
                    "scopes": [],
                    "samples": [],
                    "events": [],
                }
            }
        ),
        config=config,
        hlcvs=np.zeros((2, 1, 3), dtype=np.float64),
        fills=[],
        equities_array=np.array(
            [[1704067200000, 1000.0, 1000.0], [1704240000000, 1000.0, 1000.0]]
        ),
        btc_usd_prices=np.array([]),
        analysis={
            "gain_multiplier_usd": 1.0,
            "analysis_duration_days": 2.0,
            "equity_choppiness_ratio": float("inf"),
            "effective_start_date": "2024-01-01T00:00:00Z",
            "effective_end_date": "2024-01-03T00:00:00Z",
        },
        results_path=str(tmp_path),
        exchange="binance",
    )
    result_dir = next(tmp_path.iterdir())
    raw = json.loads((result_dir / "analysis.json").read_text())
    saved = json.loads((result_dir / "config.json").read_text())
    assert np.isinf(raw["equity_choppiness_ratio"])
    assert saved["metrics"]["stats"]["analysis_duration_days"]["mean"] == 2.0
    assert saved["metrics"]["nonfinite_diagnostics"]["equity_choppiness_ratio"] == "inf"
    assert saved["metrics"]["effective_end_date"] == "2024-01-03T00:00:00Z"
    assert (result_dir / "fills.csv").exists()
    assert (result_dir / "balance_and_equity.csv.gz").exists()
    assert plot_calls == ["balance", "save"]


def test_expand_analysis_keeps_position_held_time_weighted_metric_shared():
    result = expand_analysis(
        _make_analysis_entry(2.5),
        _make_analysis_entry(9.0),
        fills=np.empty((0, 0)),
        equities_array=np.empty((0, 3)),
        config={
            "bot": {
                side: {"total_wallet_exposure_limit_ratio": 1.0} for side in ("long", "short")
            }
        },
    )
    assert result["position_held_time_weighted_mean_hours"] == 2.5
    assert "position_held_time_weighted_mean_hours_usd" not in result
    assert "position_held_time_weighted_mean_hours_btc" not in result
