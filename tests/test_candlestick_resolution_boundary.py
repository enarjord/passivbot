import numpy as np
import pytest

from candlestick_manager import (
    CANDLE_DTYPE,
    fetch_candles_with_resolution_ladder,
)

MINUTE = 60_000


def make_candles(minutes):
    arr = np.zeros(len(minutes), dtype=CANDLE_DTYPE)
    arr["ts"] = np.asarray(minutes, dtype=np.int64) * MINUTE
    return arr


@pytest.mark.asyncio
async def test_bridges_unaligned_retention_boundary_with_coarse_bucket():
    # Exact 1m starts at minute 4, inside the coarse 0..4 5m bucket.
    # The coarse bucket may provide only minutes 0..3.
    exact = make_candles(range(4, 10))
    coarse = make_candles([0])

    async def fetch_candles(*, timeframe, start_ts, end_ts):
        if timeframe == "1m":
            arr = exact
        elif timeframe == "5m":
            arr = coarse
        else:
            return np.empty((0,), dtype=CANDLE_DTYPE)

        return arr[(arr["ts"] >= start_ts) & (arr["ts"] <= end_ts)]

    result = await fetch_candles_with_resolution_ladder(
        fetch_candles,
        start_ts=0,
        end_ts=9 * MINUTE,
        supported_timeframes={"1m", "5m"},
    )

    assert list(result.candles["ts"].astype(int)) == [
        minute * MINUTE for minute in range(10)
    ]
    assert result.source_counts["5m"] == 4
    assert result.source_counts["1m"] == 6


@pytest.mark.asyncio
async def test_does_not_bridge_when_exact_overlap_has_gap():
    # Exact history starts at minute 1, but minute 3 is missing.
    # The overlapping 5m candle must not conceal that genuine gap.
    exact = make_candles([1, 2, 4, 5, 6, 7, 8, 9])
    coarse = make_candles([0])

    async def fetch_candles(*, timeframe, start_ts, end_ts):
        if timeframe == "1m":
            arr = exact
        elif timeframe == "5m":
            arr = coarse
        else:
            return np.empty((0,), dtype=CANDLE_DTYPE)

        return arr[(arr["ts"] >= start_ts) & (arr["ts"] <= end_ts)]

    result = await fetch_candles_with_resolution_ladder(
        fetch_candles,
        start_ts=0,
        end_ts=9 * MINUTE,
        supported_timeframes={"1m", "5m"},
    )

    timestamps = set(result.candles["ts"].astype(int))

    assert 0 not in timestamps
    assert 3 * MINUTE not in timestamps
    assert result.source_counts.get("5m", 0) == 0
