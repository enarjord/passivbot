"""Tests for configurable candle interval feature."""
import numpy as np
import pytest


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
