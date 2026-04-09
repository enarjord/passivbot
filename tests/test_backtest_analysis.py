import numpy as np
import pandas as pd
import pytest

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
        "win_rate",
        "win_rate_w",
        "trade_loss_max",
        "trade_loss_mean",
        "trade_loss_median",
        "paper_loss_ratio",
        "paper_loss_mean_ratio",
        "exposure_ratio",
        "exposure_mean_ratio",
        "paper_loss_ratio_w",
        "paper_loss_mean_ratio_w",
        "exposure_ratio_w",
        "exposure_mean_ratio_w",
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


def test_expand_analysis_includes_trade_level_metrics():
    analysis_usd = _make_analysis_entry(0.25)
    analysis_btc = _make_analysis_entry(0.75)
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

    assert result["win_rate"] == 0.25
    assert result["win_rate_w"] == 0.25
    assert result["trade_loss_max"] == 0.25
    assert result["trade_loss_mean"] == 0.25
    assert result["trade_loss_median"] == 0.25
    assert "win_rate_usd" not in result
    assert "trade_loss_max_btc" not in result


def test_expand_analysis_currency_suffixes_new_ratio_metrics():
    analysis_usd = _make_analysis_entry(0.25)
    analysis_btc = _make_analysis_entry(0.75)
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

    for metric in (
        "paper_loss_ratio",
        "paper_loss_mean_ratio",
        "exposure_ratio",
        "exposure_mean_ratio",
        "paper_loss_ratio_w",
        "paper_loss_mean_ratio_w",
        "exposure_ratio_w",
        "exposure_mean_ratio_w",
    ):
        assert result[f"{metric}_usd"] == 0.25
        assert result[f"{metric}_btc"] == 0.75
        assert metric not in result


def test_make_table_includes_trade_metrics():
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


def test_process_forager_fills_adds_fill_activity_metrics_from_trading_window():
    start = 1_740_000_000_000
    one_hour = 3_600_000
    equities_array = np.array(
        [
            [start, 1000.0, 0.02],
            [start + one_hour, 1001.0, 0.02],
            [start + 2 * one_hour, 1002.0, 0.02],
            [start + 3 * one_hour, 1003.0, 0.02],
            [start + 4 * one_hour, 1004.0, 0.02],
        ],
        dtype=np.float64,
    )
    fills = [
        [1, start + one_hour, "BTC", 1.0, 0.1, 1000.0, 0.0, 1000.0, 50000.0, 0.1, 100.0, 0.1, 100.0, "entry_ema_anchor_long", "maker", 0.1, 0.0, 0.0, 0.0],
        [3, start + 3 * one_hour, "BTC", -0.5, 0.1, 1005.0, 0.0, 1005.0, 50000.0, -0.1, 110.0, 0.0, 0.0, "close_ema_anchor_long", "maker", 0.0, 0.0, 0.0, 0.0],
    ]

    _fdf, analysis_appendix, _bal_eq = process_forager_fills(
        fills=fills,
        coins=["BTC"],
        hlcvs=np.empty((0, 0), dtype=np.float64),
        equities_array=equities_array,
        balance_sample_divider=1,
    )

    assert analysis_appendix["fills_per_day"] == pytest.approx(12.0)
    assert analysis_appendix["hours_no_fills_max"] == pytest.approx(2.0)
    assert analysis_appendix["hours_no_fills_mean"] == pytest.approx(4.0 / 3.0)
    assert analysis_appendix["hours_no_fills_median"] == pytest.approx(1.0)


def test_process_forager_fills_no_fills_penalizes_whole_trading_window():
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

    assert analysis_appendix["fills_per_day"] == pytest.approx(0.0)
    assert analysis_appendix["hours_no_fills_max"] == pytest.approx(2.0)
    assert analysis_appendix["hours_no_fills_mean"] == pytest.approx(2.0)
    assert analysis_appendix["hours_no_fills_median"] == pytest.approx(2.0)


def test_post_process_disable_plotting_skips_all_figure_generation(tmp_path, monkeypatch):
    calls = {"balance": 0, "twe": 0, "pnl": 0, "save": 0, "coin": 0}

    def _fake_process_forager_fills(*args, **kwargs):
        fdf = pd.DataFrame(columns=["coin", "pnl"])
        bal_eq = pd.DataFrame({"balance": [1000.0], "equity": [1000.0]})
        return fdf, {}, bal_eq

    monkeypatch.setattr(bt, "process_forager_fills", _fake_process_forager_fills)
    monkeypatch.setattr(bt, "format_config", lambda config, verbose=False: config)
    monkeypatch.setattr(bt, "strip_config_metadata", lambda config: config)
    monkeypatch.setattr(bt, "dump_backtest_dataset_metadata", lambda *args, **kwargs: None)
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
    monkeypatch.setattr(bt, "dump_backtest_dataset_metadata", lambda *args, **kwargs: None)
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


def test_post_process_visible_metrics_filters_terminal_output_only(tmp_path, monkeypatch, capsys):
    def _fake_process_forager_fills(*args, **kwargs):
        fdf = pd.DataFrame(columns=["coin", "pnl"])
        bal_eq = pd.DataFrame({"balance": [1000.0], "equity": [1000.0]})
        return fdf, {}, bal_eq

    monkeypatch.setattr(bt, "process_forager_fills", _fake_process_forager_fills)
    monkeypatch.setattr(bt, "format_config", lambda config, verbose=False: config)
    monkeypatch.setattr(bt, "strip_config_metadata", lambda config: config)
    monkeypatch.setattr(bt, "dump_backtest_dataset_metadata", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        bt, "dump_config", lambda config, path: open(path, "w", encoding="utf-8").write("{}")
    )
    monkeypatch.setattr(bt, "create_forager_balance_figures", lambda *args, **kwargs: {})
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
            "long": {"total_wallet_exposure_limit": 1.0},
            "short": {"total_wallet_exposure_limit": 0.0},
        },
        "live": {},
        "optimize": {
            "scoring": ["adg"],
            "limits": [{"metric": "loss_profit_ratio"}],
        },
    }

    bt.post_process(
        config=config,
        hlcvs=np.zeros((1, 1, 3), dtype=np.float64),
        fills=[],
        equities_array=np.array([[1704067200000, 1000.0, 1000.0]], dtype=np.float64),
        btc_usd_prices=np.array([]),
        analysis={
            "adg_usd": 0.01,
            "adg_btc": 0.02,
            "loss_profit_ratio": 0.5,
            "peak_recovery_hours_hsl": 12.0,
        },
        results_path=str(tmp_path),
        exchange="binance",
    )

    captured = capsys.readouterr().out
    assert "Showing 3 of 4 metrics" in captured
    assert "adg_usd" in captured
    assert "adg_btc" in captured
    assert "loss_profit_ratio" in captured
    assert "peak_recovery_hours_hsl" not in captured
    analysis_path = next(tmp_path.glob("*/analysis.json"))
    assert "peak_recovery_hours_hsl" in analysis_path.read_text(encoding="utf-8")


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
                "n_positions": 1,
                "total_wallet_exposure_limit": 1.0,
            },
            "short": {
                "hsl_enabled": False,
                "n_positions": 0,
                "total_wallet_exposure_limit": 0.0,
            },
        },
    }
    hard_stop_plot_data = {
        "timestamps_ms": (idx.view("int64") // 10**6).tolist(),
        "drawdown_raw": [0.0, 0.01, 0.05, 0.06, 0.02, 0.01],
    }

    figs = create_forager_hard_stop_drawdown_figure(
        bal_eq,
        config,
        hard_stop_plot_data=hard_stop_plot_data,
        autoplot=False,
        return_figures=True,
    )
    assert "hard_stop_drawdown" in figs
