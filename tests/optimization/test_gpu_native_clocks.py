"""Native equity clocks retain exact candle indices through host decoding."""

from datetime import datetime, timezone

import numpy as np
import pytest


@pytest.mark.parametrize("interval_ms", [60_000, 300_000])
def test_native_equity_clocks_decode_indices_before_float64_scaling(interval_ms):
    torch = pytest.importorskip("torch")
    from optimization.gpu.mps_kernel import _decode_outputs
    from optimization.gpu.metrics import _with_native_equity_timestamps

    first, last = 54_001, 98_639
    daily = torch.zeros((1, 1, 9), dtype=torch.float32)
    scalars = torch.zeros((1, 68), dtype=torch.float32)
    scalars[0, 7], scalars[0, 8] = first, last
    out = _decode_outputs(daily, scalars, torch.zeros((1, 512)),
                          native_factual_hsl=True, interval_ms=interval_ms)
    out = _with_native_equity_timestamps(out, interval_ms)
    for name, expected in (("first_eq", first), ("last_eq", last)):
        assert out[f"{name}_ts"].dtype == torch.float64
        assert out[f"{name}_ts"].item() == expected * interval_ms
        assert out[f"{name}_step"].item() == expected
    assert (out["last_eq_ts"] - out["first_eq_ts"]).item() == (last - first) * interval_ms




def test_native_equity_clock_missing_sentinels_remain_missing():
    torch = pytest.importorskip("torch")
    from optimization.gpu.mps_kernel import _decode_outputs
    from optimization.gpu.metrics import _with_native_equity_timestamps

    daily = torch.zeros((1, 1, 9), dtype=torch.float32)
    scalars = torch.zeros((1, 68), dtype=torch.float32)
    scalars[0, 7:9] = -1
    out = _decode_outputs(daily, scalars, torch.zeros((1, 512)),
                          native_factual_hsl=True)
    out = _with_native_equity_timestamps(out, 60_000)
    assert torch.isnan(out["first_eq_ts"]).all()
    assert torch.isnan(out["last_eq_ts"]).all()
    assert out["first_eq_step"].item() == out["last_eq_step"].item() == -1


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("coin_count", [1, 2])
def test_native_long_history_equity_clock_transport(strategy, sides, coin_count):
    torch = pytest.importorskip("torch")
    if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
        pytest.skip("GPU required")
    from optimization.gpu.service import SharedAccountReplay
    from tools.gpu_parity import build_parser, fixture_inputs
    from test_gpu_hsl_multicoin import raw

    config, _, markets, _, _ = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", sides, "--coins", str(coin_count),
        "--bars", "128", "--hsl", "disabled",
    ]))
    count, first = 98_640, 54_001
    timestamps = 1_700_000_000_000 + np.arange(count, dtype=np.int64) * 60_000
    candles = np.full((count, coin_count, 4), 100.0)
    candles[:, :, 3] = 1.0
    config['bot']['global']['warmup']['max_minutes'] = 1
    config["backtest"].update(
        start_date=datetime.fromtimestamp(int(timestamps[first]) / 1000,
                                          timezone.utc).isoformat(),
        end_date=datetime.fromtimestamp((int(timestamps[-1]) + 60_000) / 1000,
                                        timezone.utc).isoformat(),
    )
    for coin, market in markets.items():
        if coin != "__meta__":
            market.update(first_valid_index=0, last_valid_index=count - 1,
                          warmup_minutes=1, warmup_minutes_source="history")
    markets["__meta__"] = {"requested_start_ts": int(timestamps[first])}
    proxy = SharedAccountReplay(
        config=config, hlcvs=candles, mss=markets,
        btc=np.full(count, 50_000.0), timestamps=timestamps,
        exchange="binance", batch_size=1,
        needed_metrics={"adg_strategy_eq"}, factual_hsl=True,
    )
    _, device_out = raw(proxy, [{}])
    from optimization.gpu.metrics import _with_native_equity_timestamps
    out = _with_native_equity_timestamps(
        {key: value.cpu() if isinstance(value, torch.Tensor) else value
         for key, value in device_out.items()}, 60_000
    )
    assert out["first_eq_step"].item() == first
    # Rust Backtest::run iterates 1..(n - 1); the final candle is lookahead.
    last = count - 2
    assert out["last_eq_step"].item() == last
    assert out["first_eq_ts"].item() == first * 60_000
    assert out["last_eq_ts"].item() == last * 60_000
    assert (out["last_eq_ts"] - out["first_eq_ts"]).item() == (last - first) * 60_000


@pytest.mark.parametrize("first, last", [(-2, -1), (1, -1), (4, 3)])
def test_native_equity_clock_pairs_reject_invalid_producer_values(first, last):
    torch = pytest.importorskip("torch")
    from optimization.gpu.metrics import _with_native_equity_timestamps

    out = {"first_eq_step": torch.tensor([first], dtype=torch.int64),
           "last_eq_step": torch.tensor([last], dtype=torch.int64)}
    with pytest.raises(RuntimeError, match="equity observation clocks"):
        _with_native_equity_timestamps(out, 60_000)


@pytest.mark.parametrize("origin", [0, 1_700_000_000_000])
def test_native_objectives_preserve_common_fill_and_equity_clock_origin(origin):
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from optimization.gpu.mps_kernel import _decode_outputs
    from optimization.gpu.metrics import compute_objectives
    from optimization.gpu.model import GAP_BINS

    interval = 60_000
    daily = torch.zeros((1, 1, 9))
    daily[0, 0, :2] = 1000
    daily[0, 0, 4] = 1
    scalars = torch.zeros((1, 68))
    scalars[0, 3:5] = 5 * interval
    scalars[0, 7:9] = torch.tensor([0, 10])
    scalars[0, 9] = -1
    scalars[0, 24] = 1
    out = _decode_outputs(daily, scalars, torch.zeros((1, GAP_BINS)),
                          native_factual_hsl=True, interval_ms=interval)
    # Services place all timestamp fields on the prepared dataset's Unix clock.
    for key in ("first_fill_ts", "last_fill_ts", "last_high_ts", "first_eq_ts", "last_eq_ts"):
        out[key] = out[key].to(torch.float64) + origin
    run = SimpleNamespace(interval_ms=interval, requested_start_ts_ms=origin, guard_ts_ms=origin)
    data = {"ts0": origin, "n": 11}
    metrics = compute_objectives(out, run, data,
                                needed={"fills_gap_longest_days", "fills_gap_p95_hours"})
    assert metrics["fills_gap_longest_days"].item() == pytest.approx(5 / 1440)
    assert metrics["fills_gap_p95_hours"].item() == pytest.approx(5 / 60)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("coin_count", [1, 2])
def test_native_service_fill_gap_clocks_match_exact_rust(side, coin_count):
    torch = pytest.importorskip("torch")
    if not (torch.backends.mps.is_available() or torch.cuda.is_available()):
        pytest.skip("GPU required")
    from test_gpu_entry_sizing_parity import _fixture
    from optimization.gpu.parameters import prepare_candidate_parameters
    from optimization.gpu.service import SharedAccountReplay
    from backtest import run_backtest
    from config.metrics import resolve_metric_value
    from rust_utils import verify_loaded_runtime_extension

    assert not verify_loaded_runtime_extension().get("skipped")
    config, candles, markets, btc, timestamps = _fixture(side, coin_count, "initial")
    requested = {"fills_gap_longest_days", "fills_gap_p95_hours"}
    _, _, analysis = run_backtest(candles, markets, config, "bybit", btc, timestamps)
    proxy = SharedAccountReplay(config=config, hlcvs=candles, mss=markets, btc=btc,
        timestamps=timestamps, exchange="bybit", batch_size=1,
        needed_metrics=requested, factual_hsl=True)
    actual = proxy.evaluate([prepare_candidate_parameters(config, markets, "bybit")])[0]
    for name in requested:
        assert actual[name] == pytest.approx(resolve_metric_value(analysis, name), abs=1e-8)
