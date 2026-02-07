import os
import numpy as np
import pytest

from candlestick_manager import CandlestickManager, ONE_MIN_MS, CANDLE_DTYPE, _sanitize_symbol


@pytest.mark.asyncio
async def test_loads_legacy_historical_data_shard_when_primary_missing(tmp_path, monkeypatch):
    # Work in an isolated cwd so relative historical_data/ paths are inside tmp_path
    monkeypatch.chdir(tmp_path)

    # Legacy downloader format: 2D array [timestamp, open, high, low, close, volume]
    day = "2021-03-01"
    start_ts = 1614556800000  # 2021-03-01 00:00:00 UTC
    legacy_dir = tmp_path / "historical_data" / "ohlcvs_binanceusdm" / "BTC"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = legacy_dir / f"{day}.npy"

    legacy = np.array(
        [
            [start_ts, 1.0, 2.0, 0.5, 1.5, 10.0],
            [start_ts + ONE_MIN_MS, 1.5, 2.5, 1.0, 2.0, 12.0],
        ],
        dtype=np.float64,
    )
    np.save(legacy_path, legacy)

    cm = CandlestickManager(
        exchange=None, exchange_name="binanceusdm", cache_dir=str(tmp_path / "caches")
    )
    out = await cm.get_candles(
        "BTC/USDT:USDT",
        start_ts=start_ts,
        end_ts=start_ts + ONE_MIN_MS,
        max_age_ms=None,
        strict=True,
    )

    assert out.dtype == CANDLE_DTYPE
    assert out.shape[0] == 2
    assert int(out[0]["ts"]) == start_ts
    assert float(out[0]["o"]) == pytest.approx(1.0)
    assert float(out[0]["c"]) == pytest.approx(1.5)


@pytest.mark.asyncio
async def test_legacy_leading_minutes_are_not_backfilled_by_default(tmp_path, monkeypatch):
    """By default, leading gaps (before the first real candle) are NOT filled to avoid fake data."""
    monkeypatch.chdir(tmp_path)
    day = "2021-03-01"
    day_start = 1614556800000
    legacy_dir = tmp_path / "historical_data" / "ohlcvs_binanceusdm" / "BTC"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = legacy_dir / f"{day}.npy"

    # Only one real candle at minute 2; leading minutes should NOT be backfilled.
    legacy = np.array(
        [
            [day_start + 2 * ONE_MIN_MS, 10.0, 10.0, 10.0, 10.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.save(legacy_path, legacy)

    cm = CandlestickManager(
        exchange=None, exchange_name="binanceusdm", cache_dir=str(tmp_path / "caches")
    )
    out = await cm.get_candles(
        "BTC/USDT:USDT",
        start_ts=day_start,
        end_ts=day_start + 2 * ONE_MIN_MS,
        max_age_ms=None,
        strict=False,
    )
    # With fill_leading_gaps=False (default), only the real candle is returned
    assert out.shape[0] == 1
    assert int(out[0]["ts"]) == day_start + 2 * ONE_MIN_MS
    assert float(out[0]["c"]) == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_legacy_leading_minutes_are_backfilled_when_requested(tmp_path, monkeypatch):
    """When fill_leading_gaps=True is explicitly passed to standardize_gaps, leading gaps are filled."""
    monkeypatch.chdir(tmp_path)
    day = "2021-03-01"
    day_start = 1614556800000
    legacy_dir = tmp_path / "historical_data" / "ohlcvs_binanceusdm" / "BTC"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = legacy_dir / f"{day}.npy"

    legacy = np.array(
        [
            [day_start + 2 * ONE_MIN_MS, 10.0, 10.0, 10.0, 10.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.save(legacy_path, legacy)

    cm = CandlestickManager(
        exchange=None, exchange_name="binanceusdm", cache_dir=str(tmp_path / "caches")
    )

    # First get the raw candles
    raw = await cm.get_candles(
        "BTC/USDT:USDT",
        start_ts=day_start,
        end_ts=day_start + 2 * ONE_MIN_MS,
        max_age_ms=None,
        strict=True,
    )

    # Then call standardize_gaps with fill_leading_gaps=True
    out = cm.standardize_gaps(
        raw,
        start_ts=day_start,
        end_ts=day_start + 2 * ONE_MIN_MS,
        strict=False,
        fill_leading_gaps=True,
    )
    assert out.shape[0] == 3
    assert list(out["ts"]) == [day_start, day_start + ONE_MIN_MS, day_start + 2 * ONE_MIN_MS]
    assert float(out[0]["c"]) == pytest.approx(10.0)
    assert float(out[1]["c"]) == pytest.approx(10.0)
    assert float(out[0]["bv"]) == pytest.approx(0.0)
    assert float(out[1]["bv"]) == pytest.approx(0.0)


def test_partial_but_continuous_legacy_day_does_not_block_primary_overlay_write(
    tmp_path, monkeypatch
):
    """A partial legacy day (continuous, but not full 00:00-23:59 coverage) must not be treated as complete.

    Otherwise `_save_shard()` would skip writing the primary overlay shard and the missing minutes
    would be re-downloaded on every run.
    """
    monkeypatch.chdir(tmp_path)

    day = "2021-03-01"
    day_start = 1614556800000  # 2021-03-01 00:00:00 UTC

    # Create a partial legacy shard: continuous minutes, but only 2 candles (not a full day).
    legacy_dir = tmp_path / "historical_data" / "ohlcvs_binanceusdm" / "BTC"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = legacy_dir / f"{day}.npy"
    legacy = np.array(
        [
            [day_start + 10 * ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 1.0],
            [day_start + 11 * ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.save(legacy_path, legacy)

    cm = CandlestickManager(
        exchange=None, exchange_name="binanceusdm", cache_dir=str(tmp_path / "caches")
    )
    symbol = "BTC/USDT:USDT"

    # Attempt to write a primary shard for this day.
    primary = np.empty((1,), dtype=CANDLE_DTYPE)
    primary[0]["ts"] = day_start
    primary[0]["o"] = 1.0
    primary[0]["h"] = 1.0
    primary[0]["l"] = 1.0
    primary[0]["c"] = 1.0
    primary[0]["bv"] = 0.0

    cm._save_shard(symbol, day, primary, tf="1m")

    # With strict legacy completeness, this MUST be written.
    assert os.path.exists(cm._shard_path(symbol, day, tf="1m"))


def test_global_legacy_migration_covers_all_exchanges(tmp_path, monkeypatch):
    """Global migration should migrate legacy shards for all exchanges on first init."""
    monkeypatch.chdir(tmp_path)

    day = "2021-03-01"
    start_ts = 1614556800000  # 2021-03-01 00:00:00 UTC

    legacy_binance = tmp_path / "historical_data" / "ohlcvs_binanceusdm" / "BTC"
    legacy_binance.mkdir(parents=True, exist_ok=True)
    legacy_bybit = tmp_path / "historical_data" / "ohlcvs_bybit" / "ETH"
    legacy_bybit.mkdir(parents=True, exist_ok=True)

    legacy = np.array(
        [
            [start_ts, 1.0, 2.0, 0.5, 1.5, 10.0],
            [start_ts + ONE_MIN_MS, 1.5, 2.5, 1.0, 2.0, 12.0],
        ],
        dtype=np.float64,
    )
    np.save(legacy_binance / f"{day}.npy", legacy)
    np.save(legacy_bybit / f"{day}.npy", legacy)

    CandlestickManager(
        exchange=None, exchange_name="binanceusdm", cache_dir=str(tmp_path / "caches")
    )

    binance_symbol = _sanitize_symbol("BTC/USDT:USDT")
    bybit_symbol = _sanitize_symbol("ETH/USDT:USDT")
    binance_target = (
        tmp_path / "caches" / "ohlcv" / "binance" / "1m" / binance_symbol / f"{day}.npy"
    )
    bybit_target = tmp_path / "caches" / "ohlcv" / "bybit" / "1m" / bybit_symbol / f"{day}.npy"

    assert binance_target.exists()
    assert bybit_target.exists()
