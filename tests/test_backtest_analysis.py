import json
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import backtest as bt
import plotting
from backtest import (
    BacktestPayload, DISABLE_PLOTTING_HELP, build_backtest_frames,
    decode_core_analysis, execute_backtest, parse_disabled_plot_groups,
)
from plotting import create_forager_balance_figures

START = 1704067200000
CAGR = "cagr_fraction_strategy_equity"
OMEGA = "omega_ratio_strategy_equity"


def _native_analysis(values=None, *, statuses=None, end=START + 120000):
    return {
        **(values or {}),
        "metric_status": statuses or {},
        "metric_metadata": {
            "definition_version": 1, "start_timestamp_ms": START,
            "actual_end_timestamp_ms": end, "intended_end_timestamp_ms": end,
            "observation_interval_ms": 60000, "year_days": 365.25,
        },
        "liquidated": False,
    }


def _analysis(values=None, **kwargs):
    return decode_core_analysis(_native_analysis(values, **kwargs))


def test_decode_core_analysis_keeps_requested_quantities_and_actual_dates():
    raw = _native_analysis({CAGR: .25, "backtest_completion_fraction": .4})
    decoded = decode_core_analysis(raw)
    assert decoded[CAGR] == .25
    assert decoded["backtest_completion_fraction"] == .4
    assert decoded["effective_start_date"] == "2024-01-01T00:00:00Z"
    assert decoded["effective_end_date"] == "2024-01-01T00:02:00Z"
    assert not any(name.endswith(("_usd", "_btc")) for name in decoded)
    assert raw == _native_analysis({CAGR: .25, "backtest_completion_fraction": .4})


@pytest.mark.parametrize("status", ["positive_infinity", "negative_infinity", "undefined", "invalid"])
def test_decode_core_analysis_preserves_json_safe_mathematical_status(status):
    result = decode_core_analysis(_native_analysis(
        {OMEGA: None}, statuses={OMEGA: {"status": status, "reason": "fixture"}},
    ))
    assert json.loads(json.dumps(result, allow_nan=False))["metric_status"][OMEGA]["status"] == status


@pytest.mark.parametrize("mutation", [
    lambda raw: raw.pop("metric_metadata"),
    lambda raw: raw["metric_metadata"].update(definition_version=0),
    lambda raw: raw["metric_metadata"].update(definition_version=True),
    lambda raw: raw["metric_metadata"].update(peak_recovery_tail_censored=1),
    lambda raw: raw.update(adg_fraction_usd=.1),
    lambda raw: raw.update({CAGR: float("inf")}),
    lambda raw: raw.update({CAGR: None}),
    lambda raw: raw.update({CAGR: True}),
    lambda raw: raw["metric_status"].update({CAGR: {"status": "invalid"}}),
    lambda raw: raw["metric_status"].update({OMEGA: {"status": "undefined"}}),
    lambda raw: raw.update(liquidated=1),
    lambda raw: raw["metric_metadata"].update(actual_end_timestamp_ms=-1),
])
def test_decode_core_analysis_rejects_malformed_or_unversioned_producers(mutation):
    raw = _native_analysis({CAGR: .1})
    mutation(raw)
    with pytest.raises(ValueError):
        decode_core_analysis(raw)


def test_execute_metrics_only_keeps_native_values_and_profile(monkeypatch):
    raw = _native_analysis({CAGR: .25, "backtest_completion_fraction": .5})
    monkeypatch.setattr(bt.pbr, "run_backtest_bundle", lambda *args: (
        None, None, raw, {"_rust_profile": {"rust_simulation_ms": 12.0}},
    ))
    payload = BacktestPayload(None, [], [], [], {"metrics_only": True})
    fills, equities, analysis = execute_backtest(payload, {"backtest": {"start_date": "2000-01-01"}})
    assert fills is None and equities is None
    assert analysis["backtest_completion_fraction"] == .5
    assert analysis[CAGR] == .25
    assert payload.rust_profile == {"rust_simulation_ms": 12.0}
    assert payload.hard_stop_plot_data == {}


def test_execute_rejects_old_native_tuple_shape(monkeypatch):
    monkeypatch.setattr(bt.pbr, "run_backtest_bundle", lambda *args: (None, None, {}, {}, {}))
    with pytest.raises(ValueError, match="four-value core result"):
        execute_backtest(BacktestPayload(None, [], [], [], {}), {})


def test_frames_preserve_native_precision_and_do_not_manufacture_profit_ratios():
    values = [1000.123456789123, 1000.234567891234, 1000.345678912345]
    equities = np.column_stack((START + np.arange(3) * 60000, values, np.ones(3), values))
    fills, frame = build_backtest_frames([], equities)
    assert fills.empty
    assert len(frame) == 3 and isinstance(frame.index, pd.DatetimeIndex)
    np.testing.assert_array_equal(frame["strategy_equity"], values)
    assert frame["strategy_equity"].dtype == np.float64
    assert frame["usd_total_balance"].isna().all()
    assert not any("profit" in name or "ratio" in name for name in frame)


def test_frames_keep_equity_clock_and_never_backfill_future_cash():
    equities = np.array([[START, 1000., 1., 1000.], [START + 60000, 1001., 1., 1001.]])
    fills = [[1, START + 30000, "BTC", 0., -.2, 999.8, 0., 999.8, 50000.,
              1., 100., 1., 100., "entry_long", "maker", .1, .1, 0., .1]]
    fdf, frame = build_backtest_frames(fills, equities)
    assert len(frame) == 2 and len(fdf) == 1
    assert np.isnan(frame["usd_total_balance"].iloc[0])
    assert frame["usd_total_balance"].iloc[1] == 999.8
    assert frame.index[-1] == pd.Timestamp(START + 60000, unit="ms")


def test_post_process_exports_full_equity_precision_and_status_without_detailed_hsl(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "sanitize_prepared_config_for_dump", deepcopy)
    monkeypatch.setattr(bt, "dump_backtest_dataset_metadata", lambda *args, **kwargs: None)
    values = np.array([1000.123456789123, 1000.234567891234, 1000.345678912345])
    equities = np.column_stack((START + np.arange(3) * 60000, values, np.ones(3), values))
    analysis = _analysis({CAGR: .1, OMEGA: None}, statuses={OMEGA: {"status": "positive_infinity"}})
    config = {
        "disable_plotting": True,
        "backtest": {"balance_sample_divider": 60, "coins": {"binance": ["BTC"]}},
        "bot": {"long": {}, "short": {}}, "live": {},
    }
    bt.post_process(config, np.zeros((3, 1, 4)), [], equities, np.array([]),
                    analysis, str(tmp_path), "binance")
    directory = next(tmp_path.iterdir())
    raw = json.loads((directory / "analysis.json").read_text())
    saved = json.loads((directory / "config.json").read_text())
    assert raw == analysis
    json.dumps(saved, allow_nan=False)
    frame = pd.read_csv(directory / "balance_and_equity.csv.gz", index_col=0,
                        parse_dates=[0], float_precision="round_trip")
    assert len(frame) == 3
    np.testing.assert_array_equal(frame["strategy_equity"], values)
    assert (directory / "fills.csv").exists()


def test_backtest_hsl_scope_requires_canonical_global_config():
    assert (
        bt._resolve_backtest_hsl_signal_mode({"bot": {"portfolio": {'hsl': {'scope': 'coin'}}}})
        == "coin"
    )

    with pytest.raises(KeyError, match="bot.portfolio.hsl.scope"):
        bt._resolve_backtest_hsl_signal_mode({"bot": {"portfolio": {'hsl': {}}}})

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

    def _fake_build_backtest_frames(*args, **kwargs):
        fdf = pd.DataFrame(columns=["coin", "pnl"])
        bal_eq = pd.DataFrame({"balance": [1000.0], "equity": [1000.0]})
        bal_eq.index = pd.date_range("2024-01-01", periods=len(bal_eq), freq="min")
        return fdf, bal_eq

    monkeypatch.setattr(bt, "build_backtest_frames", _fake_build_backtest_frames)
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
        analysis=_analysis({"cagr_fraction_strategy_equity": 1.0}),
        results_path=str(tmp_path),
        exchange="binance",
    )

    assert calls == {"balance": 0, "twe": 0, "pnl": 0, "save": 0, "coin": 0}

def test_post_process_writes_original_config_for_dataset_override(
    tmp_path, monkeypatch
):
    dumped = []

    def _fake_build_backtest_frames(*args, **kwargs):
        fdf = pd.DataFrame(columns=["coin", "pnl"])
        bal_eq = pd.DataFrame({"balance": [1000.0], "equity": [1000.0]})
        bal_eq.index = pd.date_range("2024-01-01", periods=len(bal_eq), freq="min")
        return fdf, bal_eq

    def _fake_dump_config(config, path):
        dumped.append((config, path))
        open(path, "w", encoding="utf-8").write("{}")

    monkeypatch.setattr(bt, "build_backtest_frames", _fake_build_backtest_frames)
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
        analysis=_analysis({"cagr_fraction_strategy_equity": 1.0}),
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
    assert dumped_by_name["config.json"]["metrics"]["stats"]["cagr_fraction_strategy_equity"]["mean"] == 1.0
    assert "suite_metrics" not in dumped_by_name["config.json"]
    assert "metrics" not in dumped_by_name["config.original.json"]

def test_post_process_disable_plotting_coin_fills_only(tmp_path, monkeypatch):
    calls = {"balance": 0, "twe": 0, "pnl": 0, "save": 0, "coin": 0}

    def _fake_build_backtest_frames(*args, **kwargs):
        fdf = pd.DataFrame(columns=["coin", "pnl"])
        bal_eq = pd.DataFrame({"balance": [1000.0], "equity": [1000.0]})
        bal_eq.index = pd.date_range("2024-01-01", periods=len(bal_eq), freq="min")
        return fdf, bal_eq

    monkeypatch.setattr(bt, "build_backtest_frames", _fake_build_backtest_frames)
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
        analysis=_analysis({"cagr_fraction_strategy_equity": 1.0}),
        results_path=str(tmp_path),
        exchange="binance",
    )

    assert calls == {"balance": 1, "twe": 1, "pnl": 1, "save": 3, "coin": 0}

def test_post_process_formats_console_without_filtering_or_rounding_exports(
    tmp_path, monkeypatch, caplog
):
    def _fake_build_backtest_frames(*args, **kwargs):
        fdf = pd.DataFrame(columns=["coin", "pnl"])
        bal_eq = pd.DataFrame({"balance": [1000.0], "equity": [1000.0]})
        bal_eq.index = pd.date_range("2024-01-01", periods=len(bal_eq), freq="min")
        return fdf, bal_eq

    monkeypatch.setattr(bt, "build_backtest_frames", _fake_build_backtest_frames)
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
        },
        "bot": {
            "long": {"enabled": True, "risk": {"total_wallet_exposure_limit_ratio": 1.0}},
            "short": {"enabled": False, "risk": {"total_wallet_exposure_limit_ratio": 0.0}},
        },
        "live": {},
        "optimize": {
            "scoring": ["adg_fraction"],
            "limits": [{"metric": "loss_profit_ratio"}],
        },
    }

    analysis = _analysis({
        "cagr_fraction_strategy_equity": 0.123456789,
        "omega_ratio_strategy_equity": 0.02,
        "loss_profit_ratio": 0.5,
        "peak_recovery_max_days_strategy_equity": 12.3456789,
        "position_exposure_utilization_rms_ratio_short": 0.0,
        "short_net_profit_contribution_fraction": 0.0,
        "hsl_red_time_fraction": 0.0,
    })
    analysis["metric_metadata"]["hsl_enabled"] = False
    before = deepcopy(analysis)
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
        analysis=analysis,
        results_path=str(tmp_path),
        exchange="binance",
    )

    logged = caplog.text
    assert "Backtest metrics" in logged
    assert "Metric" in logged and "Value" in logged
    assert "12.346%" in logged
    assert "0.123456789" not in logged
    assert "position_exposure_utilization_rms_ratio_short" not in logged
    assert "short_net_profit_contribution_fraction" not in logged
    assert "hsl_red_time_fraction" not in logged
    assert analysis == before
    assert "Showing " not in logged
    assert "cagr_fraction_strategy_equity" in logged
    assert "omega_ratio_strategy_equity" in logged
    assert "loss_profit_ratio" in logged
    assert "peak_recovery_max_days_strategy_equity" in logged
    analysis_path = next(tmp_path.glob("*/analysis.json"))
    assert json.loads(analysis_path.read_text(encoding="utf-8")) == analysis

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
