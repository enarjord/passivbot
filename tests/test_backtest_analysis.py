import numpy as np
import pandas as pd

import backtest as bt
import plotting
from backtest import expand_analysis, parse_disabled_plot_groups, process_forager_fills
from plotting import create_forager_hard_stop_drawdown_figure


def _make_analysis_entry(value):
    base_keys = [
        "adg",
        "adg_w",
        "mdg",
        "mdg_w",
        "gain",
        "positions_held_per_day",
        "position_held_hours_mean",
        "position_held_hours_max",
        "position_held_hours_median",
        "position_unchanged_hours_max",
        "loss_profit_ratio",
        "loss_profit_ratio_long",
        "loss_profit_ratio_short",
        "loss_profit_ratio_w",
        "volume_pct_per_day_avg",
        "volume_pct_per_day_avg_w",
        "peak_recovery_hours_pnl",
        "total_wallet_exposure_max",
        "total_wallet_exposure_mean",
        "total_wallet_exposure_median",
        "high_exposure_hours_mean_long",
        "high_exposure_hours_max_long",
        "high_exposure_hours_mean_short",
        "high_exposure_hours_max_short",
        "entry_initial_balance_pct_long",
        "entry_initial_balance_pct_short",
    ]
    analysis = {key: value for key in base_keys}
    analysis.update(
        {
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "omega_ratio": 0.0,
            "expected_shortfall_1pct": 0.0,
            "calmar_ratio": 0.0,
            "sterling_ratio": 0.0,
            "drawdown_worst": 0.0,
            "drawdown_worst_mean_1pct": 0.0,
            "gain_strategy_pnl_rebased": 0.0,
            "adg_strategy_pnl_rebased": 0.0,
            "mdg_strategy_pnl_rebased": 0.0,
            "sharpe_ratio_strategy_pnl_rebased": 0.0,
            "sortino_ratio_strategy_pnl_rebased": 0.0,
            "omega_ratio_strategy_pnl_rebased": 0.0,
            "expected_shortfall_1pct_strategy_pnl_rebased": 0.0,
            "calmar_ratio_strategy_pnl_rebased": 0.0,
            "sterling_ratio_strategy_pnl_rebased": 0.0,
            "adg_strategy_pnl_rebased_w": 0.0,
            "mdg_strategy_pnl_rebased_w": 0.0,
            "sharpe_ratio_strategy_pnl_rebased_w": 0.0,
            "sortino_ratio_strategy_pnl_rebased_w": 0.0,
            "omega_ratio_strategy_pnl_rebased_w": 0.0,
            "calmar_ratio_strategy_pnl_rebased_w": 0.0,
            "sterling_ratio_strategy_pnl_rebased_w": 0.0,
            "drawdown_worst_hsl": 0.0,
            "drawdown_worst_hsl_long": 0.0,
            "drawdown_worst_hsl_short": 0.0,
            "drawdown_worst_ema_hsl": 0.0,
            "drawdown_worst_ema_hsl_long": 0.0,
            "drawdown_worst_ema_hsl_short": 0.0,
            "drawdown_worst_mean_1pct_hsl": 0.0,
            "drawdown_worst_mean_1pct_hsl_long": 0.0,
            "drawdown_worst_mean_1pct_hsl_short": 0.0,
            "drawdown_worst_mean_1pct_ema_hsl": 0.0,
            "drawdown_worst_mean_1pct_ema_hsl_long": 0.0,
            "drawdown_worst_mean_1pct_ema_hsl_short": 0.0,
            "peak_recovery_hours_hsl": 0.0,
            "peak_recovery_hours_hsl_long": 0.0,
            "peak_recovery_hours_hsl_short": 0.0,
            "equity_balance_diff_neg_max": 0.0,
            "equity_balance_diff_neg_mean": 0.0,
            "equity_balance_diff_pos_max": 0.0,
            "equity_balance_diff_pos_mean": 0.0,
            "peak_recovery_hours_equity": 0.0,
            "equity_choppiness": 0.0,
            "equity_jerkiness": 0.0,
            "exponential_fit_error": 0.0,
            "equity_choppiness_w": 0.0,
            "equity_jerkiness_w": 0.0,
            "exponential_fit_error_w": 0.0,
            "adg_per_exposure_long": 0.0,
            "adg_per_exposure_short": 0.0,
            "pnl_ratio_long_short": 0.5,
            "long_short_profit_ratio": 0.5,
        }
    )
    return analysis


def test_expand_analysis_includes_entry_balance_pct():
    analysis_usd = _make_analysis_entry(0.123)
    analysis_btc = _make_analysis_entry(0.456)
    config = {
        "bot": {
            "long": {"total_wallet_exposure_limit": 1.0},
            "short": {"total_wallet_exposure_limit": 1.0},
        }
    }
    result = expand_analysis(
        analysis_usd,
        analysis_btc,
        fills=np.empty((0, 0)),
        equities_array=np.empty((0, 3)),
        config=config,
    )
    assert "entry_initial_balance_pct_long" in result
    assert "entry_initial_balance_pct_short" in result
    assert result["entry_initial_balance_pct_long"] == 0.123
    assert result["entry_initial_balance_pct_short"] == 0.123


def test_expand_analysis_includes_high_exposure_hours():
    analysis_usd = _make_analysis_entry(0.5)
    analysis_btc = _make_analysis_entry(0.5)
    config = {
        "bot": {
            "long": {"total_wallet_exposure_limit": 1.0},
            "short": {"total_wallet_exposure_limit": 1.0},
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
        assert f"high_exposure_hours_mean_{side}" in result
        assert f"high_exposure_hours_max_{side}" in result
        assert result[f"high_exposure_hours_mean_{side}"] == 0.5
        assert result[f"high_exposure_hours_max_{side}"] == 0.5


def test_expand_analysis_deduplicates_hard_stop_metrics():
    analysis_usd = _make_analysis_entry(0.5)
    analysis_btc = _make_analysis_entry(0.5)
    analysis_usd.update(
        {
            "hard_stop_triggers": 3,
            "hard_stop_triggers_per_year": 36.5,
            "hard_stop_triggers_long": 2,
            "hard_stop_triggers_short": 1,
            "hard_stop_restarts": 2,
            "hard_stop_restarts_per_year": 24.3333333333,
            "hard_stop_restarts_per_year_long": 12.16666666665,
            "hard_stop_restarts_per_year_short": 12.16666666665,
            "hard_stop_restarts_long": 1,
            "hard_stop_restarts_short": 1,
            "hard_stop_halt_to_restart_equity_loss_pct": 0.125,
            "hard_stop_time_in_yellow_pct": 0.15,
            "hard_stop_time_in_orange_pct": 0.1,
            "hard_stop_time_in_red_pct": 0.2,
            "hard_stop_duration_minutes_mean": 180.0,
            "hard_stop_duration_minutes_max": 360.0,
            "hard_stop_trigger_drawdown_mean": 0.27,
            "hard_stop_panic_close_loss_sum": 1250.0,
            "hard_stop_panic_close_loss_max": 800.0,
            "hard_stop_flatten_time_minutes_mean": 12.0,
            "hard_stop_post_restart_retrigger_pct": 0.5,
        }
    )
    analysis_btc.update(
        {
            "hard_stop_triggers": 3,
            "hard_stop_triggers_per_year": 36.5,
            "hard_stop_triggers_long": 2,
            "hard_stop_triggers_short": 1,
            "hard_stop_restarts": 2,
            "hard_stop_restarts_per_year": 24.3333333333,
            "hard_stop_restarts_per_year_long": 12.16666666665,
            "hard_stop_restarts_per_year_short": 12.16666666665,
            "hard_stop_restarts_long": 1,
            "hard_stop_restarts_short": 1,
            "hard_stop_halt_to_restart_equity_loss_pct": 0.125,
            "hard_stop_time_in_yellow_pct": 0.15,
            "hard_stop_time_in_orange_pct": 0.1,
            "hard_stop_time_in_red_pct": 0.2,
            "hard_stop_duration_minutes_mean": 180.0,
            "hard_stop_duration_minutes_max": 360.0,
            "hard_stop_trigger_drawdown_mean": 0.27,
            "hard_stop_panic_close_loss_sum": 1250.0,
            "hard_stop_panic_close_loss_max": 800.0,
            "hard_stop_flatten_time_minutes_mean": 12.0,
            "hard_stop_post_restart_retrigger_pct": 0.5,
        }
    )
    config = {
        "bot": {
            "long": {"total_wallet_exposure_limit": 1.0},
            "short": {"total_wallet_exposure_limit": 1.0},
        }
    }

    result = expand_analysis(
        analysis_usd,
        analysis_btc,
        fills=np.empty((0, 0)),
        equities_array=np.empty((0, 3)),
        config=config,
    )

    assert result["hard_stop_triggers"] == 3
    assert result["hard_stop_triggers_per_year"] == 36.5
    assert result["hard_stop_triggers_long"] == 2
    assert result["hard_stop_triggers_short"] == 1
    assert result["hard_stop_restarts"] == 2
    assert result["hard_stop_restarts_per_year"] == 24.3333333333
    assert result["hard_stop_restarts_per_year_long"] == 12.16666666665
    assert result["hard_stop_restarts_per_year_short"] == 12.16666666665
    assert result["hard_stop_restarts_long"] == 1
    assert result["hard_stop_restarts_short"] == 1
    assert result["hard_stop_halt_to_restart_equity_loss_pct"] == 0.125
    assert result["hard_stop_time_in_yellow_pct"] == 0.15
    assert result["hard_stop_time_in_orange_pct"] == 0.1
    assert result["hard_stop_time_in_red_pct"] == 0.2
    assert result["hard_stop_duration_minutes_mean"] == 180.0
    assert result["hard_stop_duration_minutes_max"] == 360.0
    assert result["hard_stop_trigger_drawdown_mean"] == 0.27
    assert result["hard_stop_panic_close_loss_sum"] == 1250.0
    assert result["hard_stop_panic_close_loss_max"] == 800.0
    assert result["hard_stop_flatten_time_minutes_mean"] == 12.0
    assert result["hard_stop_post_restart_retrigger_pct"] == 0.5
    assert "hard_stop_triggers_usd" not in result
    assert "hard_stop_triggers_btc" not in result
    assert "hard_stop_triggers_per_year_usd" not in result
    assert "hard_stop_triggers_per_year_btc" not in result
    assert "hard_stop_restarts_usd" not in result
    assert "hard_stop_restarts_btc" not in result
    assert "hard_stop_restarts_per_year_usd" not in result
    assert "hard_stop_restarts_per_year_btc" not in result
    assert "hard_stop_halt_to_restart_equity_loss_pct_usd" not in result
    assert "hard_stop_halt_to_restart_equity_loss_pct_btc" not in result


def test_expand_analysis_keeps_strategy_pnl_rebased_and_hsl_metrics_shared():
    analysis_usd = _make_analysis_entry(0.5)
    analysis_btc = _make_analysis_entry(0.5)
    analysis_usd["adg_strategy_pnl_rebased"] = 0.19
    analysis_usd["drawdown_worst_hsl"] = 0.21
    analysis_usd["drawdown_worst_hsl_long"] = 0.11
    analysis_usd["drawdown_worst_hsl_short"] = 0.31
    analysis_usd["drawdown_worst_ema_hsl"] = 0.18
    analysis_usd["drawdown_worst_ema_hsl_long"] = 0.10
    analysis_usd["drawdown_worst_ema_hsl_short"] = 0.18
    analysis_usd["drawdown_worst_mean_1pct_ema_hsl"] = 0.17
    analysis_usd["drawdown_worst_mean_1pct_ema_hsl_long"] = 0.09
    analysis_usd["drawdown_worst_mean_1pct_ema_hsl_short"] = 0.17
    analysis_usd["peak_recovery_hours_hsl"] = 17.0
    analysis_usd["peak_recovery_hours_hsl_long"] = 12.0
    analysis_usd["peak_recovery_hours_hsl_short"] = 21.0
    analysis_btc["adg_strategy_pnl_rebased"] = 0.19
    analysis_btc["drawdown_worst_hsl"] = 0.21
    analysis_btc["drawdown_worst_hsl_long"] = 0.11
    analysis_btc["drawdown_worst_hsl_short"] = 0.31
    analysis_btc["drawdown_worst_ema_hsl"] = 0.18
    analysis_btc["drawdown_worst_ema_hsl_long"] = 0.10
    analysis_btc["drawdown_worst_ema_hsl_short"] = 0.18
    analysis_btc["drawdown_worst_mean_1pct_ema_hsl"] = 0.17
    analysis_btc["drawdown_worst_mean_1pct_ema_hsl_long"] = 0.09
    analysis_btc["drawdown_worst_mean_1pct_ema_hsl_short"] = 0.17
    analysis_btc["peak_recovery_hours_hsl"] = 17.0
    analysis_btc["peak_recovery_hours_hsl_long"] = 12.0
    analysis_btc["peak_recovery_hours_hsl_short"] = 21.0
    config = {
        "bot": {
            "long": {"total_wallet_exposure_limit": 1.0},
            "short": {"total_wallet_exposure_limit": 1.0},
        }
    }

    result = expand_analysis(
        analysis_usd,
        analysis_btc,
        fills=np.empty((0, 0)),
        equities_array=np.empty((0, 3)),
        config=config,
    )

    assert result["adg_strategy_pnl_rebased"] == 0.19
    assert result["drawdown_worst_hsl"] == 0.21
    assert result["drawdown_worst_hsl_long"] == 0.11
    assert result["drawdown_worst_hsl_short"] == 0.31
    assert result["drawdown_worst_ema_hsl"] == 0.18
    assert result["drawdown_worst_ema_hsl_long"] == 0.10
    assert result["drawdown_worst_ema_hsl_short"] == 0.18
    assert result["drawdown_worst_mean_1pct_ema_hsl"] == 0.17
    assert result["drawdown_worst_mean_1pct_ema_hsl_long"] == 0.09
    assert result["drawdown_worst_mean_1pct_ema_hsl_short"] == 0.17
    assert result["peak_recovery_hours_hsl"] == 17.0
    assert result["peak_recovery_hours_hsl_long"] == 12.0
    assert result["peak_recovery_hours_hsl_short"] == 21.0
    assert "adg_strategy_pnl_rebased_usd" not in result
    assert "adg_strategy_pnl_rebased_btc" not in result


def test_expand_analysis_keeps_long_short_profit_ratio_shared():
    analysis_usd = _make_analysis_entry(0.5)
    analysis_btc = _make_analysis_entry(0.5)
    analysis_usd["pnl_ratio_long_short"] = 0.6
    analysis_usd["long_short_profit_ratio"] = 0.6
    analysis_btc["pnl_ratio_long_short"] = 0.6
    analysis_btc["long_short_profit_ratio"] = 0.6
    config = {
        "bot": {
            "long": {"total_wallet_exposure_limit": 1.0},
            "short": {"total_wallet_exposure_limit": 1.0},
        }
    }

    result = expand_analysis(
        analysis_usd,
        analysis_btc,
        fills=np.empty((0, 0)),
        equities_array=np.empty((0, 3)),
        config=config,
    )

    assert result["pnl_ratio_long_short"] == 0.6
    assert result["long_short_profit_ratio"] == 0.6
    assert "pnl_ratio_long_short_usd" not in result
    assert "pnl_ratio_long_short_btc" not in result
    assert "long_short_profit_ratio_usd" not in result
    assert "long_short_profit_ratio_btc" not in result


def test_expand_analysis_keeps_side_loss_profit_ratios_shared():
    analysis_usd = _make_analysis_entry(0.5)
    analysis_btc = _make_analysis_entry(0.5)
    analysis_usd["loss_profit_ratio_long"] = 0.2
    analysis_usd["loss_profit_ratio_short"] = 0.8
    analysis_btc["loss_profit_ratio_long"] = 0.2
    analysis_btc["loss_profit_ratio_short"] = 0.8
    config = {
        "bot": {
            "long": {"total_wallet_exposure_limit": 1.0},
            "short": {"total_wallet_exposure_limit": 1.0},
        }
    }

    result = expand_analysis(
        analysis_usd,
        analysis_btc,
        fills=np.empty((0, 0)),
        equities_array=np.empty((0, 3)),
        config=config,
    )

    assert result["loss_profit_ratio_long"] == 0.2
    assert result["loss_profit_ratio_short"] == 0.8
    assert "loss_profit_ratio_long_usd" not in result
    assert "loss_profit_ratio_long_btc" not in result
    assert "loss_profit_ratio_short_usd" not in result
    assert "loss_profit_ratio_short_btc" not in result


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
    assert analysis_appendix["pnl_ratio_long_short"] == 0.5
    assert analysis_appendix["long_short_profit_ratio"] == 0.5


def test_process_forager_fills_reports_long_short_profit_ratio():
    fills = [
        {
            "minute": 0,
            "index": 0,
            "symbol": "BTCUSDT",
            "type": "close_long",
            "pnl": 60.0,
            "fee_paid": 0.0,
            "timestamp": 1704067200000,
            "usd_total_balance": 1000.0,
            "btc_cash_wallet": 0.0,
            "usd_cash_wallet": 1000.0,
            "btc_price": 20_000.0,
            "qty": 0.0,
            "price": 0.0,
            "psize": 0.0,
            "pprice": 0.0,
            "wallet_exposure": 0.0,
            "twe_long": 0.0,
            "twe_short": 0.0,
            "twe_net": 0.0,
        },
        {
            "minute": 1,
            "index": 1,
            "symbol": "ETHUSDT",
            "type": "close_short",
            "pnl": 40.0,
            "fee_paid": 0.0,
            "timestamp": 1704067260000,
            "usd_total_balance": 1100.0,
            "btc_cash_wallet": 0.0,
            "usd_cash_wallet": 1100.0,
            "btc_price": 20_000.0,
            "qty": 0.0,
            "price": 0.0,
            "psize": 0.0,
            "pprice": 0.0,
            "wallet_exposure": 0.0,
            "twe_long": 0.0,
            "twe_short": 0.0,
            "twe_net": 0.0,
        },
    ]
    equities_array = np.array([[1704067200000, 1000.0, 0.05]], dtype=np.float64)

    _fdf, analysis_appendix, _bal_eq = process_forager_fills(
        fills=fills,
        coins=["BTC", "ETH"],
        hlcvs=np.empty((0, 0), dtype=np.float64),
        equities_array=equities_array,
        balance_sample_divider=1,
    )

    assert analysis_appendix["pnl_ratio_long_short"] == 0.6
    assert analysis_appendix["long_short_profit_ratio"] == 0.6


def test_post_process_disable_plotting_skips_all_figure_generation(tmp_path, monkeypatch):
    calls = {"balance": 0, "twe": 0, "pnl": 0, "hard_stop": 0, "save": 0, "coin": 0}

    def _fake_process_forager_fills(*args, **kwargs):
        fdf = pd.DataFrame(columns=["coin", "pnl"])
        bal_eq = pd.DataFrame({"balance": [1000.0], "equity": [1000.0]})
        return fdf, {}, bal_eq

    monkeypatch.setattr(bt, "process_forager_fills", _fake_process_forager_fills)
    monkeypatch.setattr(bt, "format_config", lambda config, verbose=False: config)
    monkeypatch.setattr(bt, "strip_config_metadata", lambda config: config)
    monkeypatch.setattr(
        bt, "dump_config", lambda config, path: open(path, "w", encoding="utf-8").write("{}")
    )

    monkeypatch.setattr(
        bt,
        "create_forager_balance_figures",
        lambda *args, **kwargs: calls.__setitem__("balance", calls["balance"] + 1) or {},
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
        "create_forager_hard_stop_drawdown_figure",
        lambda *args, **kwargs: calls.__setitem__("hard_stop", calls["hard_stop"] + 1) or {},
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
            "long": {"total_wallet_exposure_limit": 1.0},
            "short": {"total_wallet_exposure_limit": 0.0},
        },
        "live": {},
    }

    bt.post_process(
        config=config,
        hlcvs=np.zeros((1, 1, 3), dtype=np.float64),
        fills=[],
        equities_array=np.array([[1704067200000, 1000.0, 1000.0]], dtype=np.float64),
        btc_usd_prices=np.array([]),
        analysis={"gain_usd": 1.0},
        results_path=str(tmp_path),
        exchange="binance",
    )

    assert calls == {"balance": 0, "twe": 0, "pnl": 0, "hard_stop": 0, "save": 0, "coin": 0}


def test_post_process_disable_plotting_coin_fills_only(tmp_path, monkeypatch):
    calls = {"balance": 0, "twe": 0, "pnl": 0, "hard_stop": 0, "save": 0, "coin": 0}

    def _fake_process_forager_fills(*args, **kwargs):
        fdf = pd.DataFrame(columns=["coin", "pnl"])
        bal_eq = pd.DataFrame({"balance": [1000.0], "equity": [1000.0]})
        return fdf, {}, bal_eq

    monkeypatch.setattr(bt, "process_forager_fills", _fake_process_forager_fills)
    monkeypatch.setattr(bt, "format_config", lambda config, verbose=False: config)
    monkeypatch.setattr(bt, "strip_config_metadata", lambda config: config)
    monkeypatch.setattr(
        bt, "dump_config", lambda config, path: open(path, "w", encoding="utf-8").write("{}")
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
        lambda *args, **kwargs: calls.__setitem__("twe", calls["twe"] + 1) or {"twe": object()},
    )
    monkeypatch.setattr(
        bt,
        "create_forager_pnl_figure",
        lambda *args, **kwargs: calls.__setitem__("pnl", calls["pnl"] + 1) or {"pnl": object()},
    )
    monkeypatch.setattr(
        bt,
        "create_forager_hard_stop_drawdown_figure",
        lambda *args, **kwargs: calls.__setitem__("hard_stop", calls["hard_stop"] + 1)
        or {"hard_stop": object()},
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
            "long": {"total_wallet_exposure_limit": 1.0},
            "short": {"total_wallet_exposure_limit": 0.0},
        },
        "live": {},
    }

    bt.post_process(
        config=config,
        hlcvs=np.zeros((1, 1, 3), dtype=np.float64),
        fills=[],
        equities_array=np.array([[1704067200000, 1000.0, 1000.0]], dtype=np.float64),
        btc_usd_prices=np.array([]),
        analysis={"gain_usd": 1.0},
        results_path=str(tmp_path),
        exchange="binance",
    )

    assert calls == {"balance": 1, "twe": 1, "pnl": 1, "hard_stop": 1, "save": 4, "coin": 0}


def test_parse_disabled_plot_groups_accepts_summary_alias_and_commas():
    assert parse_disabled_plot_groups("summary,coin_fills") == {
        "balance",
        "twe",
        "pnl",
        "hard_stop",
        "coin_fills",
    }


def test_create_forager_hard_stop_drawdown_figure_returns_plot_when_enabled(monkeypatch):
    class _Axis:
        def plot(self, *args, **kwargs):
            return None

        def axhline(self, *args, **kwargs):
            return None

        def axvline(self, *args, **kwargs):
            return None

        def axvspan(self, *args, **kwargs):
            return None

        def set_title(self, *args, **kwargs):
            return None

        def set_ylabel(self, *args, **kwargs):
            return None

        def set_xlabel(self, *args, **kwargs):
            return None

        def grid(self, *args, **kwargs):
            return None

        def legend(self, *args, **kwargs):
            return None

        def fill_between(self, *args, **kwargs):
            return None

    class _Figure:
        def tight_layout(self):
            return None

    monkeypatch.setattr(
        plotting.plt,
        "subplots",
        lambda *args, **kwargs: (_Figure(), [_Axis(), _Axis()]),
    )

    idx = pd.date_range("2021-01-01", periods=6, freq="1h")
    bal_eq = pd.DataFrame(
        {
            "usd_total_balance": [1000.0, 1000.0, 980.0, 970.0, 990.0, 995.0],
            "usd_total_equity": [1000.0, 990.0, 950.0, 940.0, 980.0, 992.0],
        },
        index=idx,
    )
    config = {
        "live": {"pnls_max_lookback_days": 30.0},
        "bot": {
            "long": {
                "hsl_enabled": True,
                "hsl_red_threshold": 0.1,
                "hsl_ema_span_minutes": 60.0,
                "hsl_tier_ratios": {"yellow": 0.5, "orange": 0.75},
            },
            "short": {
                "hsl_enabled": True,
                "hsl_red_threshold": 0.12,
                "hsl_ema_span_minutes": 90.0,
                "hsl_tier_ratios": {"yellow": 0.5, "orange": 0.75},
            },
        },
    }
    hard_stop_plot_data = {
        "timestamps_ms_long": (idx.view("int64") // 10**6).tolist(),
        "drawdown_raw_long": [0.0, 0.01, 0.05, 0.06, 0.02, 0.01],
        "drawdown_ema_long": [0.0, 0.005, 0.03, 0.045, 0.03, 0.02],
        "drawdown_score_long": [0.0, 0.005, 0.03, 0.045, 0.02, 0.01],
        "events_long": [
            {"kind": "red_enter", "timestamp_ms": int(idx[2].value // 10**6), "terminal": False},
            {
                "kind": "halt",
                "timestamp_ms": int(idx[3].value // 10**6),
                "cooldown_until_ms": int(idx[5].value // 10**6),
                "terminal": False,
            },
            {"kind": "restart", "timestamp_ms": int(idx[5].value // 10**6), "terminal": False},
        ],
        "timestamps_ms_short": (idx.view("int64") // 10**6).tolist(),
        "drawdown_raw_short": [0.0, 0.0, 0.01, 0.015, 0.01, 0.0],
        "drawdown_ema_short": [0.0, 0.0, 0.005, 0.01, 0.009, 0.004],
        "drawdown_score_short": [0.0, 0.0, 0.005, 0.01, 0.009, 0.0],
        "events_short": [],
    }

    figs = create_forager_hard_stop_drawdown_figure(
        bal_eq,
        config,
        hard_stop_plot_data=hard_stop_plot_data,
        autoplot=False,
        return_figures=True,
    )
    assert "hard_stop_drawdown" in figs
