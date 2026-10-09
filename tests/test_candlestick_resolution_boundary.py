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


BOUNDARIES = [
    pytest.param(timeframe, minutes, boundary, id=f"{timeframe}-offset-{boundary}")
    for timeframe, minutes in (("5m", 5), ("15m", 15), ("1h", 60))
    for boundary in range(1, minutes)
]


def priced_candles(minutes, *, opening=100.0, high=100.0, low=100.0, close=100.0):
    arr = make_candles(minutes)
    arr["o"] = opening
    arr["h"] = high
    arr["l"] = low
    arr["c"] = close
    arr["bv"] = 1.0
    return arr


async def read_boundary(exact, coarse, timeframe, minutes, *, start_minute=0, end_minute=None):
    if end_minute is None:
        end_minute = minutes + 4

    async def fetch(*, timeframe, start_ts, end_ts):
        arr = exact if timeframe == "1m" else coarse
        return arr[(arr["ts"] >= start_ts) & (arr["ts"] <= end_ts)]

    return await fetch_candles_with_resolution_ladder(
        fetch,
        start_ts=start_minute * MINUTE,
        end_ts=end_minute * MINUTE,
        supported_timeframes={"1m", timeframe},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("timeframe,minutes,boundary", BOUNDARIES)
@pytest.mark.parametrize("close", [100.0, 110.0])
async def test_boundary_preserves_extrema_that_must_belong_to_prefix(
    timeframe, minutes, boundary, close
):
    # The exact suffix has neither extreme: both must belong to the unknown prefix.
    exact = priced_candles(
        range(boundary, minutes + 5), opening=close, high=close, low=close, close=close
    )
    coarse = priced_candles([0], high=200.0, low=50.0, close=close)
    coarse["bv"] = minutes
    result = await read_boundary(exact, coarse, timeframe, minutes)

    prefix = result.candles[result.candles["ts"] < boundary * MINUTE]
    assert result.candles["ts"].tolist() == [m * MINUTE for m in range(minutes + 5)]
    assert np.array_equal(result.candles[boundary:], exact)
    assert result.source_counts == {"1m": len(exact), timeframe: boundary}
    assert float(prefix["h"].max()) == 200.0
    assert float(prefix["l"].min()) == 50.0
    assert float(prefix[0]["o"]) == 100.0
    assert float(prefix[-1]["c"]) == close
    assert float(prefix["bv"].sum()) == pytest.approx(boundary)

    # Exercise the actual coverage admission consumer without invoking a live bot.
    from types import SimpleNamespace
    from passivbot import Passivbot

    bot = SimpleNamespace(_completed_candle_health_now_ms=lambda: (minutes + 5) * MINUTE)
    subset, projection = Passivbot._completed_trailing_candle_subset(bot, result.candles, -1)
    assert subset is not None
    assert projection is None
    assert float(subset["h"].max()) == 200.0
    assert float(subset["l"].min()) == 50.0


@pytest.mark.asyncio
@pytest.mark.parametrize("timeframe,minutes,boundary", BOUNDARIES)
@pytest.mark.parametrize("prefix_extreme", ["high", "low", "neither"])
async def test_boundary_does_not_move_exact_overlap_extrema_into_prefix(
    timeframe, minutes, boundary, prefix_extreme
):
    exact = priced_candles(range(boundary, minutes + 5), high=150.0, low=75.0)
    coarse = priced_candles(
        [0],
        high=200.0 if prefix_extreme == "high" else 150.0,
        low=50.0 if prefix_extreme == "low" else 75.0,
    )
    result = await read_boundary(exact, coarse, timeframe, minutes)
    prefix = result.candles[result.candles["ts"] < boundary * MINUTE]

    assert np.array_equal(result.candles[boundary:], exact)
    assert float(prefix["h"].max()) == (200.0 if prefix_extreme == "high" else 100.0)
    assert float(prefix["l"].min()) == (50.0 if prefix_extreme == "low" else 100.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeframe,minutes", [("5m", 5), ("15m", 15), ("1h", 60)])
async def test_boundary_requires_complete_overlap_through_bucket_end(timeframe, minutes):
    exact = priced_candles(range(1, minutes + 5))
    exact = exact[exact["ts"] != (minutes - 1) * MINUTE]
    coarse = priced_candles([0], high=200.0, low=50.0)
    result = await read_boundary(exact, coarse, timeframe, minutes)

    assert result.source_counts == {"1m": len(exact)}
    assert np.array_equal(result.candles, exact)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeframe,minutes", [("5m", 5), ("15m", 15), ("1h", 60)])
async def test_boundary_does_not_use_overlap_outside_requested_end(timeframe, minutes):
    exact = priced_candles(range(1, minutes + 5))
    coarse = priced_candles([0], high=200.0, low=50.0)
    result = await read_boundary(exact, coarse, timeframe, minutes, end_minute=minutes - 2)

    assert result.source_counts.get(timeframe, 0) == 0
    assert result.candles["ts"].tolist() == [m * MINUTE for m in range(1, minutes - 1)]


@pytest.mark.asyncio
@pytest.mark.parametrize("timeframe,minutes", [("5m", 5), ("15m", 15), ("1h", 60)])
async def test_boundary_never_repairs_later_exact_gap(timeframe, minutes):
    exact = priced_candles(range(1, minutes + 5))
    exact = exact[exact["ts"] != (minutes + 1) * MINUTE]
    coarse = priced_candles([0], high=200.0, low=50.0)
    result = await read_boundary(exact, coarse, timeframe, minutes)

    assert result.source_counts[timeframe] == 1
    assert (minutes + 1) * MINUTE not in result.candles["ts"]
    assert np.array_equal(result.candles[1:], exact)


@pytest.mark.asyncio
async def test_boundary_finer_source_wins_over_straddling_coarser_source():
    exact = priced_candles(range(6, 20))
    five_minute = priced_candles([5], high=120.0, low=80.0)
    fifteen_minute = priced_candles([0], high=200.0, low=50.0)

    async def fetch(*, timeframe, start_ts, end_ts):
        return {"1m": exact, "5m": five_minute, "15m": fifteen_minute}[timeframe]

    result = await fetch_candles_with_resolution_ladder(
        fetch,
        start_ts=0,
        end_ts=19 * MINUTE,
        supported_timeframes={"1m", "5m", "15m"},
    )
    assert result.source_counts == {"1m": 14, "5m": 1, "15m": 5}
    assert np.array_equal(result.candles[6:], exact)
    # Minute 5 already belongs to the finer source and cannot be overwritten.
    assert float(result.candles[5]["h"]) == 120.0
    assert float(result.candles[5]["l"]) == 80.0


@pytest.mark.asyncio
async def test_boundary_keeps_missing_extrema_when_finer_prefix_rows_win():
    exact = priced_candles(range(10, 20))
    five_minute = priced_candles([5], high=120.0, low=80.0)
    fifteen_minute = priced_candles([0], high=200.0, low=50.0)

    async def fetch(*, timeframe, start_ts, end_ts):
        return {"1m": exact, "5m": five_minute, "15m": fifteen_minute}[timeframe]

    result = await fetch_candles_with_resolution_ladder(
        fetch,
        start_ts=0,
        end_ts=19 * MINUTE,
        supported_timeframes={"1m", "5m", "15m"},
    )
    assert result.source_counts == {"1m": 10, "5m": 5, "15m": 5}
    assert float(result.candles["h"].max()) == 200.0
    assert float(result.candles["l"].min()) == 50.0
    assert float(result.candles[5:10]["h"].max()) == 120.0
    assert float(result.candles[5:10]["l"].min()) == 80.0
    assert np.array_equal(result.candles[10:], exact)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeframe,minutes", [("5m", 5), ("15m", 15), ("1h", 60)])
async def test_boundary_reads_local_shards_without_changing_sources(tmp_path, timeframe, minutes):
    from candlestick_manager import CandlestickManager

    manager = CandlestickManager(exchange=None, cache_dir=str(tmp_path))
    manager._now_ms_callback = lambda: (minutes + 5) * MINUTE
    symbol = "TEST/USDT"
    exact = priced_candles(range(1, minutes + 5))
    coarse = priced_candles([0], high=200.0, low=50.0)
    manager._persist_batch(symbol, exact, timeframe="1m")
    manager._persist_batch(symbol, coarse, timeframe=timeframe)

    async def fetch(*, timeframe, start_ts, end_ts):
        return await manager.get_candles(
            symbol,
            timeframe=timeframe,
            start_ts=start_ts,
            end_ts=end_ts,
            standardize=False,
            allow_remote_fetch=False,
        )

    result = await fetch_candles_with_resolution_ladder(
        fetch,
        start_ts=0,
        end_ts=(minutes + 4) * MINUTE,
        supported_timeframes={"1m", timeframe},
    )
    assert result.failures == {}
    assert result.source_counts == {"1m": len(exact), timeframe: 1}
    assert float(result.candles["h"].max()) == 200.0
    assert float(result.candles["l"].min()) == 50.0
    assert np.array_equal(result.candles[1:], exact)
    assert np.array_equal(await fetch(timeframe=timeframe, start_ts=0, end_ts=0), coarse)
    assert np.array_equal(
        await fetch(timeframe="1m", start_ts=0, end_ts=(minutes + 4) * MINUTE), exact
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["o", "h", "l", "c", "bv"])
async def test_boundary_does_not_hide_nonfinite_coarse_values(field):
    exact = priced_candles(range(1, 10))
    coarse = priced_candles([0], high=200.0, low=50.0)
    coarse[field] = np.nan
    with pytest.raises(ValueError, match="finite"):
        await read_boundary(exact, coarse, "5m", 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("timeframe,minutes", [("5m", 5), ("15m", 15), ("1h", 60)])
@pytest.mark.parametrize(
    "opening,high,low,expected_open",
    [(100.0, 50.0, 200.0, 100.0),
     (300.0, 200.0, 50.0, 200.0),
     (25.0, 200.0, 50.0, 50.0),
     (300.0, 50.0, 200.0, 200.0)],
)
async def test_boundary_normalizes_coarse_ohlc_like_full_bucket(
    timeframe, minutes, opening, high, low, expected_open
):
    exact = priced_candles(range(1, minutes + 5))
    raw = priced_candles([0], opening=opening, high=high, low=low)
    canonical = raw.copy()
    canonical["o"] = expected_open
    canonical["h"] = 200.0
    canonical["l"] = 50.0

    result = await read_boundary(exact, raw, timeframe, minutes)
    reference = await read_boundary(exact, canonical, timeframe, minutes)
    assert np.array_equal(result.candles, reference.candles)
    assert np.array_equal(result.candles[1:], exact)
    assert float(result.candles[0]["o"]) == expected_open
    assert float(result.candles["h"].max()) == 200.0
    assert float(result.candles["l"].min()) == 50.0


@pytest.mark.asyncio
@pytest.mark.parametrize("timeframe,minutes", [("5m", 5), ("15m", 15), ("1h", 60)])
@pytest.mark.parametrize("exact_open,expected_close", [(300.0, 200.0), (25.0, 50.0)])
async def test_boundary_clamps_synthetic_endpoint_to_normalized_coarse_range(
    timeframe, minutes, exact_open, expected_close
):
    exact = priced_candles(
        range(1, minutes + 5), opening=exact_open, high=exact_open,
        low=exact_open, close=exact_open,
    )
    coarse = priced_candles([0], high=200.0, low=50.0)
    result = await read_boundary(exact, coarse, timeframe, minutes)
    assert float(result.candles[0]["c"]) == expected_close
    assert float(result.candles[0]["h"]) <= 200.0
    assert float(result.candles[0]["l"]) >= 50.0
    assert np.array_equal(result.candles[1:], exact)
