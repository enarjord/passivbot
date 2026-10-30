"""Exact facts survive storage; view reductions never erase risk extrema."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from backtest_results import (
    ResultStore,
    extrema_indices,
    write_equity,
    write_hsl,
    validate_trace,
)
from backtest_reports import (
    ReportOptions,
    create_equity_overview,
    equity_view,
    hsl_view,
    render_reports,
)
from tools.backtest_dash import BacktestAPI


def result(tmp_path, values=(100.0, 99.0, 120.0, 60.0, 100.0)):
    frame = pd.DataFrame(
        {
            "strategy_equity": values,
            "usd_total_equity": values,
            "btc_total_equity": np.asarray(values) / 10,
            "usd_total_balance": np.full(len(values), 100.0),
        },
        index=pd.date_range("2025-01-01", periods=len(values), freq="min"),
    )
    write_equity(tmp_path, frame)
    (tmp_path / "analysis.json").write_text(
        json.dumps({"drawdown_max_fraction_strategy_equity": 0.5})
    )
    (tmp_path / "dataset.json").write_text(json.dumps({"coins": ["AAA"]}))
    pd.DataFrame(
        {
            "timestamp": [frame.index[0]],
            "coin": ["AAA"],
            "type": ["entry_initial_normal_long"],
            "pnl": [0.0],
            "fee_paid": [-0.1],
            "price": [10.0],
            "qty": [1.0],
        }
    ).to_csv(tmp_path / "fills.csv", index=False)
    return ResultStore(tmp_path), frame


def test_numeric_archive_is_lossless_including_negative_zero_and_missing_cash(tmp_path):
    store, frame = result(tmp_path, (1000.123456789123, -0.0, 1000.234567891234))
    frame["usd_cash_wallet"] = [np.nan, 999.123456789123, 999.234567891234]
    write_equity(tmp_path, frame)
    loaded = ResultStore(tmp_path).equity()
    np.testing.assert_array_equal(loaded.index, frame.index)
    for key in frame:
        assert loaded[key].dtype == np.float64
        assert loaded[key].to_numpy().tobytes() == frame[key].to_numpy().tobytes()
    assert store._arrays == {}  # Metadata-only construction did not inflate columns.


@pytest.mark.parametrize("version", [True, 2, "1", np.array([1])])
def test_unknown_archive_versions_do_not_fall_back_to_csv(tmp_path, version):
    store, frame = result(tmp_path)
    frame.to_csv(tmp_path / "balance_and_equity.csv.gz")
    np.savez(tmp_path / "balance_and_equity.npz", schema_version=version)
    with pytest.raises(ValueError, match="schema version"):
        store.equity()


def test_object_archives_are_never_unpickled(tmp_path):
    store, _ = result(tmp_path)
    np.savez(
        tmp_path / "balance_and_equity.npz",
        schema_version=np.int64(1),
        timestamp_ms=np.array([1], dtype=np.int64),
        strategy_equity=np.array(["bad"], dtype=object),
    )
    with pytest.raises(ValueError, match="allow_pickle=False"):
        store.equity(columns=["strategy_equity"])


def test_archive_columns_must_match_their_clocks(tmp_path):
    store, _ = result(tmp_path)
    np.savez(
        tmp_path / "balance_and_equity.npz",
        schema_version=np.int64(1),
        timestamp_ms=np.array([1, 2], dtype=np.int64),
        strategy_equity=np.array([1.0]),
    )
    with pytest.raises(ValueError, match="different lengths"):
        store.equity(columns=["strategy_equity"])


@pytest.mark.parametrize("timestamps", [[2, 1], [1, 1], [-1, 1]])
def test_invalid_equity_clocks_rejected_before_writing(tmp_path, timestamps):
    _, frame = result(tmp_path, (100.0, 100.0))
    frame.index = pd.to_datetime(timestamps, unit="ms")
    with pytest.raises(ValueError):
        write_equity(tmp_path, frame)


def test_later_window_keeps_the_original_peak_and_intrahour_trough(tmp_path):
    # Hourly-last resampling would hide the 70% intrahour fall completely.
    values = np.full(5000, 100.0)
    values[2000], values[2001] = 200.0, 60.0
    store, frame = result(tmp_path, values)
    view = equity_view(store, max_points=100)
    assert view["drawdown_fraction"].max() == 0.7
    assert frame.index[2001].value // 1_000_000 in view["timestamp"]
    later = equity_view(store, start=int(frame.index[2002].value // 1_000_000))
    assert later["drawdown_fraction"][0] == 0.5
    assert view["observations"] == 5000


def test_extrema_selection_keeps_duplicate_timestamp_order_and_forced_boundaries():
    values = np.zeros((10000, 2))
    values[500, 0], values[501, 0], values[9000, 1] = 10, -20, 50
    chosen = extrema_indices(values, max_points=100, forced=[505, 506])
    assert {500, 501, 9000, 505, 506} <= set(chosen)
    assert np.all(np.diff(chosen) > 0)
    assert len(chosen) < 110


@pytest.mark.parametrize("scale", ["log", "linear"])
def test_default_equity_view_and_selected_scale_preserve_the_peak(tmp_path, scale):
    import matplotlib.pyplot as plt

    store, _ = result(tmp_path)
    fig = create_equity_overview(store, ReportOptions(equity_scale=scale))
    try:
        assert fig.axes[0].get_yscale() == scale
        assert "50.000%" in fig.axes[1].get_title()
        assert len(fig.axes) == 2
    finally:
        plt.close(fig)


def test_nonpositive_equity_remains_visible_with_an_explicit_linear_fallback(tmp_path):
    import matplotlib.pyplot as plt

    store, _ = result(tmp_path, (100.0, 0.0, -10.0))
    fig = create_equity_overview(store, ReportOptions())
    try:
        assert fig.axes[0].get_yscale() == "linear"
        assert "nonpositive" in fig.axes[0].get_title()
        assert fig.axes[0].lines[0].get_ydata().min() == -10.0
        assert fig.axes[1].lines[0].get_ydata().max() == pytest.approx(110.0)
    finally:
        plt.close(fig)


def test_report_failure_keeps_facts_and_records_the_failure(tmp_path, monkeypatch):
    store, _ = result(tmp_path)
    monkeypatch.setattr(
        "backtest_reports.create_equity_overview",
        lambda *args: (_ for _ in ()).throw(RuntimeError("fixture")),
    )
    with pytest.raises(RuntimeError):
        render_reports(store)
    status = json.loads((tmp_path / "reports/report.json").read_text())
    assert status["status"] == "failed" and status["error_type"] == "RuntimeError"
    assert len(store.equity()) == 5


def test_reports_default_to_equity_and_hsl_and_keep_csv_an_explicit_export(tmp_path):
    store, frame = result(tmp_path)
    status = render_reports(store)
    assert status["artifacts"] == ["equity.png"]  # HSL has no producer in this fixture.
    assert not list(tmp_path.rglob("*pnl*"))
    assert not (tmp_path / "balance_and_equity.csv.gz").exists()
    status = render_reports(store, options=ReportOptions((), equity_csv=True))
    loaded = pd.read_csv(
        tmp_path / "reports/balance_and_equity.csv.gz",
        index_col=0,
        float_precision="round_trip",
    )
    np.testing.assert_equal(
        loaded.strategy_equity.to_numpy(), frame.strategy_equity.to_numpy()
    )


def test_browser_api_loads_metadata_lazily_and_pages_position_sides(tmp_path):
    store, _ = result(tmp_path)
    api = BacktestAPI(tmp_path)
    summary = api.request("/api/summary")
    assert summary["dataset"]["coins"] == ["AAA"]
    assert not api.store._arrays and api.store._fills is None
    rows = api.request("/api/fills?side=long&limit=1")
    assert rows["total"] == 1 and len(rows["rows"]) == 1
    assert api.request("/api/fills?side=short")["total"] == 0
    assert not api.store._arrays


@pytest.mark.parametrize(
    "path",
    [
        "/api/equity?start=2&end=1",
        "/api/equity?points=999999",
        "/api/fills?limit=0",
        "/api/fills?side=buy",
        "/api/fills?coin=UNKNOWN",
        "/api/equity?start=1&start=2",
    ],
)
def test_browser_api_rejects_ambiguous_or_unbounded_requests(tmp_path, path):
    result(tmp_path)
    with pytest.raises(ValueError):
        BacktestAPI(tmp_path).request(path)


@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
def test_normal_native_hsl_reports_have_a_separate_lossless_plot_packet(tmp_path, mode):
    from test_hsl_backtest_config import payload, run

    native = run(payload(mode))
    report, trace = native[3]["hsl"], native[3]["hsl_trace"]
    json.dumps(report, allow_nan=False)  # Metadata remains a plain JSON envelope.
    columns = validate_trace(trace, report)
    assert len(columns["timestamp"]) > 0 and report["samples"] == []
    metadata = write_hsl(tmp_path, report, trace=trace)
    store = ResultStore(tmp_path)
    np.testing.assert_equal(store.hsl_trace(), trace)
    assert metadata["trace"]["rows"] == len(columns["timestamp"])
    view = hsl_view(store)
    assert len(view["scopes"]) == 1
    assert report["execution_episodes"]
    loss = [e["loss_fraction"] for e in report["execution_episodes"]]
    assert max(loss) == native[2]["hsl_panic_episode_loss_max_fraction"]


@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
def test_compact_trace_matches_verbose_native_observations_exactly(mode):
    from test_hsl_backtest_config import payload, run

    args = payload(mode)
    args[-1]["hsl_detailed_report"] = True
    output = run(args)[3]
    trace, report = output["hsl_trace"], output["hsl"]
    samples = {row["sequence"]: row for row in report["samples"]}
    for key in ("signal_raw_fraction", "signal_ema_fraction"):
        assert trace[key].dtype == np.float64
        expected = np.array([samples[int(seq)][key] for seq in trace["sequence"]])
        assert expected.tobytes() == trace[key].tobytes()


def test_cached_trace_cannot_be_corrupted_by_a_view(tmp_path):
    from test_hsl_backtest_config import payload, run

    output = run(payload("coin"))[3]
    write_hsl(tmp_path, output["hsl"], trace=output["hsl_trace"])
    store = ResultStore(tmp_path)
    trace = store.hsl_trace()
    with pytest.raises(ValueError, match="read-only"):
        trace["action"][0] = 2
    trace.pop("action")
    assert "action" in store.hsl_trace()


def test_nonpositive_equity_trace_code_must_be_boolean():
    from test_hsl_backtest_config import payload, run

    trace = run(payload("coin"))[3]["hsl_trace"]
    trace["nonpositive_equity"][0] = 2
    with pytest.raises(ValueError, match="state code"):
        validate_trace(trace)


@pytest.mark.parametrize(
    "mode,scopes", [("coin", 3), ("position_side", 2), ("portfolio", 1)]
)
def test_two_coin_long_short_native_reports_keep_controller_scopes(
    tmp_path, mode, scopes
):
    from copy import deepcopy
    from backtest import build_backtest_frames, prep_backtest_args
    from config.universe import set_side_universe_sources
    from test_hsl_backtest_config import inputs, payload, run

    cfg, markets, candles = inputs(mode)
    set_side_universe_sources(
        cfg,
        "approved_coins",
        {
            "long": ["AAA", "BBB"],
            "short": ["AAA", "BBB"],
        },
    )
    cfg["backtest"]["coins"]["binance"] = ["AAA", "BBB"]
    for side in ("long", "short"):
        cfg["bot"][side]["risk"]["max_positions"] = 2
    cfg["bot"]["short"]["hsl"] = deepcopy(cfg["bot"]["long"]["hsl"])
    markets["BBB"] = deepcopy(markets["AAA"])
    bot, strategy, exchange, params = prep_backtest_args(cfg, markets, "binance")
    for key in (
        "first_timestamp_ms",
        "requested_start_timestamp_ms",
        "global_warmup_bars",
        "candle_interval_minutes",
    ):
        params[key] = payload(mode)[-1][key]
    for key, value in (
        ("first_valid_indices", 0),
        ("last_valid_indices", len(candles) - 1),
        ("warmup_minutes", 1),
        ("trade_start_indices", 1),
    ):
        params[key] = [value, value]
    opposite = candles.copy()
    opposite[:, :, :3] = 10000.0 / candles[:, :, :3][:, :, [1, 0, 2]]
    args = [
        np.concatenate([candles, opposite], axis=1),
        np.full(len(candles), 50000.0),
        bot,
        strategy,
        exchange,
        params,
    ]
    fills, equity, analysis, extra = run(args)
    assert {"long", "short"} <= {str(row[13]).rsplit("_", 1)[-1] for row in fills}
    _, frame = build_backtest_frames(fills, equity)
    write_equity(tmp_path, frame)
    write_hsl(tmp_path, extra["hsl"], trace=extra["hsl_trace"])
    store = ResultStore(tmp_path)
    view = hsl_view(store)
    assert len(view["scopes"]) == scopes
    assert all(group["observations"] > 0 for group in view["scopes"])
    assert {"equity.png", "hsl.png"} <= set(render_reports(store)["artifacts"])
    params["metrics_only"] = True
    assert run(args)[2] == analysis


def test_no_fill_btc_result_has_an_explicit_empty_hsl_report(tmp_path):
    from backtest import build_backtest_frames
    from test_hsl_backtest_config import payload, run

    args = list(payload("portfolio"))
    args[-1]["btc_collateral_target_ratio"] = 0.5
    args[1] = np.linspace(50000.0, 90000.0, len(args[0]))
    args[0][:] = [100.0, 100.0, 100.0, 1000.0]
    fills, equity, _, extra = run(args)
    assert len(fills) == 0
    _, frame = build_backtest_frames(fills, equity)
    write_equity(tmp_path, frame)
    write_hsl(tmp_path, extra["hsl"], trace=extra["hsl_trace"])
    store = ResultStore(tmp_path)
    assert hsl_view(store)["scopes"] == []
    status = render_reports(store, options=ReportOptions(btc=True))
    assert {"equity.png", "hsl.png", "hsl_episodes.csv"} <= set(status["artifacts"])
    assert store.equity().usd_total_balance.isna().all()


@pytest.mark.parametrize(
    "path,host,origin,status",
    [
        ("/", "127.0.0.1:8765", None, 200),
        ("/app.js", "localhost:8765", "http://localhost:8765", 200),
        ("/api/summary", "attacker.invalid:8765", None, 403),
        ("/api/summary", "127.0.0.1:8765", "https://attacker.invalid", 403),
        ("/../config.json", "127.0.0.1:8765", None, 404),
        ("/api/equity?start=2&end=1", "127.0.0.1:8765", None, 400),
    ],
)
def test_browser_http_boundary_and_bundled_assets(tmp_path, path, host, origin, status):
    from io import BytesIO
    from tools.backtest_dash import handler_for

    result(tmp_path)
    handler = object.__new__(handler_for(BacktestAPI(tmp_path)))
    handler.path = path
    handler.headers = {"Host": host}
    if origin is not None:
        handler.headers["Origin"] = origin
    handler.server = SimpleNamespace(server_port=8765)
    handler.request_version = "HTTP/1.1"
    handler.requestline = "GET " + path + " HTTP/1.1"
    handler.wfile = BytesIO()
    handler.do_GET()
    response = handler.wfile.getvalue()
    assert response.startswith(f"HTTP/1.0 {status} ".encode())
    assert b"Content-Security-Policy:" in response
    assert b"Access-Control-Allow-Origin:" not in response
    assert str(tmp_path).encode() not in response


def test_zero_cooldown_lifecycle_is_not_erased_by_columnar_rendering(tmp_path):
    from test_hsl_backtest_config import payload, run

    args = payload("coin")
    args[-1]["equity_hard_stop_loss"]["sides"][0]["cooldown_minutes_after_red"] = 0
    for pair in args[-1]["equity_hard_stop_loss"]["coins"].values():
        pair[0]["cooldown_minutes_after_red"] = 0
    output = run(args)[3]
    write_hsl(tmp_path, output["hsl"], trace=output["hsl_trace"])
    group = hsl_view(ResultStore(tmp_path))["scopes"][0]
    event = next(e for e in output["hsl"]["events"] if e["kind"] == "red")
    restart = next(e for e in output["hsl"]["events"] if e["kind"] == "restart")
    assert event["observed_at"] == restart["observed_at"]
    same_time = group["state_action"][group["state_timestamp"] == event["observed_at"]]
    assert 1 in same_time and same_time[-1] == 0


def test_hsl_display_decimation_does_not_create_idle_gaps_in_continuous_exposure(
    tmp_path,
):
    from test_hsl_backtest_config import payload, run

    output = run(payload("coin"))[3]
    write_hsl(tmp_path, output["hsl"], trace=output["hsl_trace"])
    view = hsl_view(ResultStore(tmp_path), max_points=10)
    for group in view["scopes"]:
        assert np.count_nonzero(group["gap_before"]) < len(group["timestamp"]) // 2


def test_idle_gaps_keep_integer_utc_clocks_with_strict_numpy_errors():
    from backtest_reports import _gap_lines

    timestamps = np.array([1704067200000, 1704067260000, 1704153600000], dtype=np.int64)
    with np.errstate(all="raise"):
        x, y = _gap_lines(timestamps, [0.1, 0.2, 0.3], [False, False, True])
        converted = pd.to_datetime(x, unit="ms")
    assert x.dtype == np.int64 and not converted.isna().any()
    assert np.isnan(y[2]) and x[2] == timestamps[2]
