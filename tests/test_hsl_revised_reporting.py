"""Native report to plots and saved artifacts; no network or account access."""
from copy import deepcopy
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from backtest import BacktestPlotContext, execute_backtest, post_process
from backtest_artifacts import load_backtest_artifact
from hsl_revised_reporting import revised_report
from plotting import create_forager_hard_stop_drawdown_figure
from test_backtest_artifacts import _write_artifact
from test_hsl_revised_backtest_config import inputs, payload as base_payload, prepared_payload, run


def payload(*args, **kwargs):
    result = base_payload(*args, **kwargs)
    result[-1]["hsl_detailed_report"] = True
    return result


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_native_scope_values_and_effective_threshold_are_plotted(mode):
    cfg, markets, _ = inputs(mode)
    if mode == "coin":
        cfg["coin_overrides"] = {"AAA": {"bot": {"long": {"hsl": {
            "red_threshold": .025, "ema_span_minutes": 2.75}}}}}
    report_data = run(payload(mode, cfg=cfg, mss=markets))[4]
    report = report_data["revised"]
    before = deepcopy(report)
    figures = create_forager_hard_stop_drawdown_figure(
        pd.DataFrame({"usd_total_equity": [1000., 1.]}), cfg,
        hard_stop_plot_data=report_data, autoplot=False, return_figures=True)
    assert len(figures) == 1  # Disabled side policies do not create fictitious signals.
    try:
        fig, = figures.values()
        signal_ax, state_ax = fig.axes
        scope, = report["scopes"]
        rows = report["samples"]
        lines = {line.get_label(): line for line in signal_ax.lines}
        np.testing.assert_equal(lines["Raw drawdown"].get_ydata(),
                                np.array([r["raw"] for r in rows], dtype=float))
        np.testing.assert_equal(lines["EMA drawdown"].get_ydata(),
                                np.array([r["ema"] for r in rows], dtype=float))
        assert lines["RED threshold"].get_ydata() == [scope["policy"]["red_threshold"]] * 2
        assert not any("orange" in key.lower() or "yellow" in key.lower() for key in lines)
        assert [t.get_text() for t in state_ax.get_yticklabels()] == ["GREEN", "RED"]
        transitions = sorted(
            [(r["sequence"], r["timestamp"], int(r["action"] != "normal")) for r in rows]
            + [(e["sequence"], e["observed_at"], int(e["kind"] != "restart"))
               for e in report["events"]])
        np.testing.assert_equal(state_ax.lines[0].get_ydata(), [r[2] for r in transitions])
        np.testing.assert_equal(state_ax.lines[0].get_xdata(),
                                pd.to_datetime([r[1] for r in transitions], unit="ms").to_numpy())
        for line, event in zip(state_ax.lines[1:], report["events"]):
            assert line.get_xdata()[0] == pd.to_datetime(event["observed_at"], unit="ms")
        if mode == "coin":
            assert scope["policy"]["red_threshold"] == .025
            assert "AAA long" in signal_ax.get_title()
        if mode == "unified":
            assert scope["side"] is None and scope["coin"] is None
            assert "portfolio" in signal_ax.get_title()
        fig.canvas.draw()
    finally:
        for fig in figures.values():
            plt.close(fig)
    assert report == before


def test_missing_native_report_never_synthesizes_revised_signal(caplog):
    cfg, _, _ = inputs()
    assert create_forager_hard_stop_drawdown_figure(
        pd.DataFrame({"usd_total_equity": [1000., 1.]}), cfg,
        autoplot=False, return_figures=True) == {}
    assert "native report was not supplied" in caplog.text


def test_coin_plots_keep_each_coin_and_side_separate():
    args = list(payload())
    args[0] = np.repeat(args[0], 2, axis=1)
    for index in (2, 3, 4):
        args[index] = [deepcopy(args[index][0]), deepcopy(args[index][0])]
    params = args[-1]
    params["coins"] = ["AAA", "BBB"]
    for key in ("first_valid_indices", "last_valid_indices", "warmup_minutes", "trade_start_indices"):
        params[key] *= 2
    for pair in args[2]:
        pair["short"]["entry_eligible"] = True
        pair["short"]["wallet_exposure_limit"] = -1.0
    policies = params["equity_hard_stop_loss"]["coins"]
    policies["BBB"] = deepcopy(policies["AAA"])
    for coin_policies in policies.values():
        coin_policies[1] = deepcopy(coin_policies[0])
    policies["BBB"][0]["red_threshold"] = .04
    data = run(args)[4]
    figures = create_forager_hard_stop_drawdown_figure(
        pd.DataFrame(), {}, hard_stop_plot_data=data, autoplot=False, return_figures=True)
    try:
        assert len(figures) == 4
        for scope in data["revised"]["scopes"]:
            side, coin = scope["side"], scope["coin"]
            fig = figures[f"hard_stop_drawdown_coin_{coin}_{('long', 'short')[side]}"]
            rows = [row for row in data["revised"]["samples"]
                    if (row["side"], row["coin"]) == (side, coin)]
            np.testing.assert_equal(fig.axes[0].lines[0].get_ydata(),
                                    np.array([row["raw"] for row in rows], dtype=float))
            assert fig.axes[0].lines[2].get_ydata() == [scope["policy"]["red_threshold"]] * 2
    finally:
        for fig in figures.values():
            plt.close(fig)


def test_native_report_saved_even_when_all_plotting_disabled(tmp_path):
    cfg, _, candles = inputs("unified")
    cfg["disable_plotting"] = True
    args = payload("unified", cfg=cfg)
    prepared = prepared_payload(args)
    fills, equities, analysis = execute_backtest(prepared, cfg)
    post_process(cfg, candles, fills, equities, args[1], analysis, str(tmp_path), "binance",
                 plot_context=BacktestPlotContext.from_payload(prepared))
    saved, = tmp_path.rglob("hsl_report.json")
    report = json.loads(saved.read_text())
    assert report == prepared.hard_stop_plot_data["revised"]
    assert report["detailed"] is True
    assert report["schema_version"] == 1 and report["engine"] == "revised"
    assert report["scopes"][0]["policy"]["red_threshold"] == .01
    assert report["samples"]
    assert not list(tmp_path.rglob("*.png"))


def test_compact_native_report_is_explicit_and_has_no_plot_trace():
    cfg, _, _ = inputs("unified")
    args = payload("unified")
    args[-1]["metrics_only"] = True
    prepared = prepared_payload(args)
    execute_backtest(prepared, cfg)
    report = prepared.hard_stop_plot_data["revised"]
    assert report["detailed"] is False
    assert report["samples"] == []
    assert report["scopes"][0]["policy"]["red_threshold"] == .01
    assert create_forager_hard_stop_drawdown_figure(
        pd.DataFrame(), cfg, hard_stop_plot_data={"revised": report},
        autoplot=False, return_figures=True) == {}


def test_artifact_workspace_preserves_native_report_and_legacy_absence(tmp_path):
    folder = _write_artifact(tmp_path)
    assert load_backtest_artifact(folder).hsl_report is None
    report = run(payload("unified"))[4]["revised"]
    (folder / "hsl_report.json").write_text(json.dumps(report))
    artifact = load_backtest_artifact(folder)
    assert artifact.hsl_report == report
    assert artifact.workspace()["hsl_report"] == report


def test_unsupported_report_version_is_not_interpreted_as_legacy():
    with pytest.raises(ValueError, match="unsupported revised HSL report schema"):
        revised_report({"revised": {"schema_version": 999, "engine": "revised"}})


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_zero_cooldown_red_is_visible_between_same_timestamp_transitions(mode):
    cfg, markets, _ = inputs(mode)
    policy = cfg["bot"]["hsl"] if mode == "unified" else cfg["bot"]["long"]["hsl"]
    policy["cooldown_minutes_after_red"] = 0
    data = run(payload(mode, cfg=cfg, mss=markets))[4]
    report = data["revised"]
    sequences = [r["sequence"] for r in report["samples"] + report["events"]]
    assert len(sequences) == len(set(sequences))
    red = next(e for e in report["events"] if e["kind"] == "red")
    restart = next(e for e in report["events"] if e["kind"] == "restart")
    assert red["observed_at"] == restart["observed_at"]
    assert red["sequence"] < restart["sequence"]
    figures = create_forager_hard_stop_drawdown_figure(
        pd.DataFrame(), cfg, hard_stop_plot_data=data, autoplot=False, return_figures=True)
    try:
        fig, = figures.values()
        line = fig.axes[1].lines[0]
        at_boundary = line.get_xdata() == pd.to_datetime(red["observed_at"], unit="ms")
        values = line.get_ydata()[at_boundary].tolist()
        assert 1 in values and values[-1] == 0
    finally:
        for fig in figures.values():
            plt.close(fig)


def test_inactive_coin_scope_has_no_native_permission_or_green_plot():
    args = payload()
    args[2][0]["long"]["n_positions"] = 0
    data = run(args)[4]
    rows = data["revised"]["samples"]
    assert rows
    assert all(row["action"] is None and row["raw"] is None and row["ema"] is None for row in rows)
    assert all("inactive_scope" in row["reasons"] for row in rows)
    assert create_forager_hard_stop_drawdown_figure(
        pd.DataFrame(), {}, hard_stop_plot_data=data, autoplot=False, return_figures=True) == {}


def test_report_rows_and_reason_lists_are_independent():
    report = run(payload("coin"))[4]["revised"]
    before = deepcopy(report)
    assert len(report["samples"]) > 2
    report["samples"][0]["raw"] = -123.0
    report["samples"][0]["reasons"].append("caller annotation")
    assert report["samples"][1:] == before["samples"][1:]
    assert report["events"] == before["events"]
    assert run(payload("coin"))[4]["revised"] == before


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_default_report_preserves_trading_metrics_and_events_without_samples(mode):
    # Exercise canonical default transport rather than the plotting fixture opt-in.
    args = base_payload(mode)
    assert args[-1]["hsl_detailed_report"] is False
    compact = run(args)
    args[-1]["hsl_detailed_report"] = True
    detailed = run(args)
    for actual, expected in zip(compact[:4], detailed[:4]):
        np.testing.assert_equal(actual, expected)
    compact_report = compact[4]["revised"]
    detailed_report = detailed[4]["revised"]
    assert compact_report["detailed"] is False
    assert compact_report["samples"] == []
    assert detailed_report["detailed"] is True
    assert detailed_report["samples"]
    assert compact_report["events"]
    assert {k: v for k, v in compact_report.items() if k not in {"detailed", "samples"}} == {
        k: v for k, v in detailed_report.items() if k not in {"detailed", "samples"}}
    args[-1]["metrics_only"] = True  # Opt-in cannot inflate optimizer results.
    metrics = run(args)
    assert metrics[2:4] == detailed[2:4]
    assert metrics[4]["revised"]["samples"] == []
    assert metrics[4]["revised"]["events"] == []
    assert metrics[4]["revised"]["detailed"] is False


@pytest.mark.parametrize("detailed", [False, True])
def test_report_option_roundtrips_and_is_strictly_boolean(detailed):
    from config import prepare_config
    from config_utils import strip_config_metadata
    cfg, markets, _ = inputs()
    cfg["backtest"]["hsl_detailed_report"] = detailed
    normalized = prepare_config(cfg, verbose=False, target="canonical", runtime=None)
    persisted = json.loads(json.dumps(strip_config_metadata(normalized)))
    reloaded = prepare_config(persisted, verbose=False, target="canonical", runtime=None)
    assert reloaded["backtest"]["hsl_detailed_report"] is detailed
    reloaded["backtest"]["coins"] = {"binance": ["AAA"]}
    assert base_payload(cfg=reloaded, mss=markets)[-1]["hsl_detailed_report"] is detailed
    for bad in ["false", 0, None]:
        cfg["backtest"]["hsl_detailed_report"] = bad
        with pytest.raises(ValueError, match="hsl_detailed_report must be a boolean"):
            prepare_config(cfg, verbose=False, target="canonical", runtime=None)
    del cfg["backtest"]["hsl_detailed_report"]
    assert prepare_config(cfg, verbose=False, target="canonical", runtime=None)["backtest"]["hsl_detailed_report"] is False
