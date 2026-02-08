"""Tests for configurable candle interval feature."""
from pathlib import Path

import numpy as np
import pytest

try:
    import passivbot_rust as pbr
except Exception:  # pragma: no cover - exercised when the extension is unavailable
    pbr = None

pbr_is_stub = bool(getattr(pbr, "__is_stub__", False)) if pbr is not None else False


def test_aggregate_candles_basic():
    """Test that aggregate_candles produces correct HLCV values."""
    from backtest import aggregate_candles

    # Create 10 1m candles for 2 coins
    # Shape: (10, 2, 4) for HLCV (high, low, close, volume)
    candles = np.zeros((10, 2, 4), dtype=np.float64)

    # Coin 0: high=100+i+0.5, low=100+i-0.5, close=100+i+0.1, volume=1 each
    for i in range(10):
        candles[i, 0, :] = [100 + i + 0.5, 100 + i - 0.5, 100 + i + 0.1, 1.0]

    # Coin 1: high=200+i+1.0, low=200+i-1.0, close=200+i+0.2, volume=2 each
    for i in range(10):
        candles[i, 1, :] = [200 + i + 1.0, 200 + i - 1.0, 200 + i + 0.2, 2.0]

    result = aggregate_candles(candles, 5)

    assert result.shape == (2, 2, 4), f"Expected (2, 2, 4), got {result.shape}"

    # First 5m candle for coin 0 (indices 0-4):
    # high=max(100.5,101.5,102.5,103.5,104.5)=104.5, low=min(99.5,...,103.5)=99.5
    # close=104.1, volume=5
    assert result[0, 0, 0] == 104.5, "High should be max of interval"
    assert result[0, 0, 1] == 99.5, "Low should be min of interval"
    assert abs(result[0, 0, 2] - 104.1) < 0.01, "Close should be last candle's close"
    assert result[0, 0, 3] == 5.0, "Volume should be sum"


def test_aggregate_candles_interval_1():
    """Test that interval=1 returns unchanged array."""
    from backtest import aggregate_candles

    candles = np.random.rand(100, 3, 4)
    result = aggregate_candles(candles, 1)

    assert result is candles, "interval=1 should return same array"


def test_aggregate_candles_truncates():
    """Test that incomplete final interval is dropped."""
    from backtest import aggregate_candles

    candles = np.random.rand(17, 2, 4)  # 17 candles, interval 5 -> 3 complete intervals
    result = aggregate_candles(candles, 5)

    assert result.shape[0] == 3, f"Expected 3 intervals, got {result.shape[0]}"


def test_aggregate_candles_error_on_insufficient():
    """Test that error is raised when not enough candles."""
    from backtest import aggregate_candles

    candles = np.random.rand(3, 2, 4)  # Only 3 candles

    with pytest.raises(ValueError, match="Not enough candles"):
        aggregate_candles(candles, 5)


@pytest.mark.skipif(pbr is None or pbr_is_stub, reason="passivbot_rust extension not available")
def test_backtest_with_candle_interval():
    from backtest import build_backtest_payload, execute_backtest
    from config_utils import load_config

    root = Path(__file__).resolve().parents[1]
    config = load_config(str(root / "configs" / "template.json"), verbose=False)
    config["backtest"]["exchanges"] = ["binance"]
    config["backtest"]["coins"] = {"binance": ["BTC"]}
    config["backtest"]["candle_interval_minutes"] = 5
    config["backtest"]["filter_by_min_effective_cost"] = False
    config["backtest"]["start_date"] = "2021-01-01"
    config["backtest"]["end_date"] = "2021-01-02"
    config["live"]["warmup_ratio"] = 0.0
    config["live"]["max_warmup_minutes"] = 0
    config["live"]["hedge_mode"] = False

    n_minutes = 60
    start_ts = 1609459200000  # 2021-01-01 00:00:00 UTC
    timestamps = np.arange(start_ts, start_ts + n_minutes * 60_000, 60_000, dtype=np.int64)
    hlcvs = np.zeros((n_minutes, 1, 4), dtype=np.float64)
    for i in range(n_minutes):
        base = 100 + i * 0.1
        hlcvs[i, 0, 0] = base + 0.5  # high
        hlcvs[i, 0, 1] = base - 0.5  # low
        hlcvs[i, 0, 2] = base  # close
        hlcvs[i, 0, 3] = 1.0  # volume
    btc_usd_prices = np.full(n_minutes, 20_000.0, dtype=np.float64)
    mss = {
        "BTC": {
            "qty_step": 0.001,
            "price_step": 0.1,
            "min_qty": 0.0,
            "min_cost": 0.0,
            "c_mult": 1.0,
            "maker": 0.0002,
            "taker": 0.0005,
            "exchange": "binance",
        },
        "__meta__": {
            "requested_start_ts": int(timestamps[0]),
            "requested_start_date": "2021-01-01",
            "warmup_minutes_requested": 0,
            "warmup_minutes_provided": 0,
        },
    }

    payload = build_backtest_payload(
        hlcvs,
        mss,
        config,
        "binance",
        btc_usd_prices,
        timestamps,
    )
    assert payload.bundle.hlcvs.shape[0] == n_minutes // 5
    assert payload.backtest_params["candle_interval_minutes"] == 5
    aggregated_timestamps = np.asarray(payload.bundle.timestamps)
    assert aggregated_timestamps.shape[0] == n_minutes // 5
    assert np.all(np.diff(aggregated_timestamps) == 5 * 60_000)

    fills, equities_array, analysis = execute_backtest(payload, config)
    assert equities_array.shape[1] == 3
    assert equities_array.shape[0] <= n_minutes // 5
    assert np.isfinite(analysis["positions_held_per_day"])
