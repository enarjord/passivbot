import numpy as np
import pandas as pd

import backtest as bt
from backtest import expand_analysis, parse_disabled_plot_groups, process_forager_fills


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
            "hard_stop_restarts": 2,
            "hard_stop_total_loss_pct": 0.125,
            "hard_stop_restarts_per_year": 24.3333333333,
        }
    )
    analysis_btc.update(
        {
            "hard_stop_triggers": 3,
            "hard_stop_triggers_per_year": 36.5,
            "hard_stop_restarts": 2,
            "hard_stop_total_loss_pct": 0.125,
            "hard_stop_restarts_per_year": 24.3333333333,
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
    assert result["hard_stop_restarts"] == 2
    assert result["hard_stop_total_loss_pct"] == 0.125
    assert result["hard_stop_restarts_per_year"] == 24.3333333333
    assert "hard_stop_triggers_usd" not in result
    assert "hard_stop_triggers_btc" not in result
    assert "hard_stop_triggers_per_year_usd" not in result
    assert "hard_stop_triggers_per_year_btc" not in result
    assert "hard_stop_restarts_usd" not in result
    assert "hard_stop_restarts_btc" not in result
    assert "hard_stop_total_loss_pct_usd" not in result
    assert "hard_stop_total_loss_pct_btc" not in result
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
    assert analysis_appendix["pnl_ratio_long_short"] == 0.5


def test_post_process_disable_plotting_skips_all_figure_generation(tmp_path, monkeypatch):
    calls = {"balance": 0, "twe": 0, "pnl": 0, "save": 0, "coin": 0}

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
        "bot": {"long": {"total_wallet_exposure_limit": 1.0}, "short": {"total_wallet_exposure_limit": 0.0}},
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

    assert calls == {"balance": 0, "twe": 0, "pnl": 0, "save": 0, "coin": 0}


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
        bt, "dump_config", lambda config, path: open(path, "w", encoding="utf-8").write("{}")
    )

    monkeypatch.setattr(
        bt,
        "create_forager_balance_figures",
        lambda *args, **kwargs: calls.__setitem__("balance", calls["balance"] + 1) or {"balance": object()},
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
        "bot": {"long": {"total_wallet_exposure_limit": 1.0}, "short": {"total_wallet_exposure_limit": 0.0}},
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

    assert calls == {"balance": 1, "twe": 1, "pnl": 1, "save": 3, "coin": 0}


def test_parse_disabled_plot_groups_accepts_summary_alias_and_commas():
    assert parse_disabled_plot_groups("summary,coin_fills") == {
        "balance",
        "twe",
        "pnl",
        "coin_fills",
    }
