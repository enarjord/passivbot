from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from candlestick_manager import CANDLE_DTYPE, CandlestickManager
from live import candle_ws
from passivbot import Passivbot


ONE_MIN_MS = 60_000


def _candles(*rows) -> np.ndarray:
    return np.array(list(rows), dtype=CANDLE_DTYPE)


def _manager(tmp_path, *, now_ms: int) -> CandlestickManager:
    cm = CandlestickManager(
        exchange=None,
        exchange_name="fake",
        cache_dir=str(tmp_path),
        archive_enabled=False,
    )
    cm._now_ms_callback = lambda: int(now_ms)
    return cm


async def _prime_and_finalize_ws_row(
    cm: CandlestickManager,
    symbol: str,
    row: list,
) -> int:
    """Observe one open bucket, then seal it with a fresh successor."""
    ts = int(row[0])
    assert (
        await cm.ingest_live_ws_ohlcv(
            symbol,
            [row],
            now_ms=ts + ONE_MIN_MS - 10_000,
        )
        == 0
    )
    close = float(row[4])
    return await cm.ingest_live_ws_ohlcv(
        symbol,
        [[ts + ONE_MIN_MS, close, close, close, close, 0.0]],
        now_ms=ts + ONE_MIN_MS + 1_000,
    )


@pytest.mark.asyncio
async def test_ws_requires_canonical_basis_then_persists_only_finalized_rows(
    tmp_path,
):
    now_ms = 5 * ONE_MIN_MS + 10_000
    cm = _manager(tmp_path, now_ms=now_ms)
    symbol = "BTC/USDT:USDT"

    assert (
        await cm.ingest_live_ws_ohlcv(
            symbol,
            [[4 * ONE_MIN_MS, 101, 103, 100, 102, 4]],
            now_ms=5 * ONE_MIN_MS - 10_000,
        )
        == 0
    )
    assert cm.get_last_final_ts(symbol) == 0

    cm._persist_batch(
        symbol,
        _candles(
            (2 * ONE_MIN_MS, 100, 101, 99, 100, 2),
            (3 * ONE_MIN_MS, 100, 102, 99, 101, 3),
        ),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=now_ms,
    )

    changed = await cm.ingest_live_ws_ohlcv(
        symbol,
        [
            [4 * ONE_MIN_MS, 101, 103, 100, 102, 4],
            [5 * ONE_MIN_MS, 102, 104, 101, 103, 5],
        ],
        now_ms=now_ms,
    )

    assert changed == 1
    assert cm.get_last_live_ws_ohlcv_ts(symbol) == 4 * ONE_MIN_MS
    assert cm.get_last_final_ts(symbol) == 4 * ONE_MIN_MS
    disk = cm._load_from_disk(
        symbol,
        2 * ONE_MIN_MS,
        5 * ONE_MIN_MS,
        timeframe="1m",
    )
    assert list(disk["ts"]) == [
        2 * ONE_MIN_MS,
        3 * ONE_MIN_MS,
        4 * ONE_MIN_MS,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_first_write", [False, True])
async def test_ws_finalizes_first_candle_of_new_utc_day(
    tmp_path, monkeypatch, fail_first_write
):
    midnight = 24 * 60 * ONE_MIN_MS
    symbol = "BTC/USDT:USDT"
    cm = _manager(tmp_path, now_ms=midnight + ONE_MIN_MS + 1_000)
    cm._persist_batch(
        symbol,
        _candles((midnight - 2 * ONE_MIN_MS, 100, 101, 99, 100, 2)),
        timeframe="1m",
        merge_cache=True,
    )
    for ts, expected in [(midnight - ONE_MIN_MS, 0), (midnight, 1)]:
        assert await cm.ingest_live_ws_ohlcv(
            symbol, [[ts, 100, 101, 99, 100, 2]], now_ms=ts + 1_000
        ) == expected

    # Yesterday's canonical basis exists, but today's shard does not yet exist.
    assert cm._load_from_disk(symbol, midnight, midnight, timeframe="1m") is None
    assert cm.get_last_final_ts(symbol) == midnight - ONE_MIN_MS
    successor = [[midnight + ONE_MIN_MS, 100, 102, 99, 101, 3]]
    previous_frame = cm._live_ws_ohlcv_observations[symbol].copy()
    if fail_first_write:
        with monkeypatch.context() as patch:
            def fail_save(*_args, **_kwargs):
                raise OSError("disk unavailable")

            patch.setattr(cm, "_save_range_incremental", fail_save)
            with pytest.raises(OSError, match="disk unavailable"):
                await cm.ingest_live_ws_ohlcv(symbol, successor)
        assert cm.get_last_final_ts(symbol) == midnight - ONE_MIN_MS
        assert cm.get_last_live_ws_ohlcv_ts(symbol) == midnight - ONE_MIN_MS
        assert cm._live_ws_ohlcv_observations[symbol] == previous_frame
        assert not np.any(cm._ensure_symbol_cache(symbol)["ts"] >= midnight)
        assert cm._load_from_disk(symbol, midnight, midnight, timeframe="1m") is None

    assert await cm.ingest_live_ws_ohlcv(symbol, successor) == 1
    expected = _candles((midnight, 100, 101, 99, 100, 2))
    disk = cm._load_from_disk(
        symbol, midnight, midnight, timeframe="1m", merge_memory_cache=False
    )
    np.testing.assert_array_equal(disk, expected)
    assert cm.get_last_final_ts(symbol) == midnight
    assert cm.get_last_live_ws_ohlcv_ts(symbol) == midnight
    assert not np.any(cm._ensure_symbol_cache(symbol)["ts"] > midnight)

    # A fresh manager reconstructs the same candle from disk after restart.
    restarted = _manager(tmp_path, now_ms=midnight + ONE_MIN_MS + 1_000)
    np.testing.assert_array_equal(
        restarted._load_from_disk(
            symbol, midnight, midnight, timeframe="1m", merge_memory_cache=False
        ),
        expected,
    )


@pytest.mark.asyncio
async def test_cached_forager_ema_uses_contiguous_ws_tail(tmp_path):
    now_ms = 5 * ONE_MIN_MS + 10_000
    cm = _manager(tmp_path, now_ms=now_ms)
    symbol = "BTC/USDT:USDT"
    cm._persist_batch(
        symbol,
        _candles(
            (2 * ONE_MIN_MS, 100, 100, 100, 100, 1),
            (3 * ONE_MIN_MS, 100, 101, 100, 101, 2),
        ),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=now_ms,
    )
    assert (
        await _prime_and_finalize_ws_row(
            cm,
            symbol,
            [4 * ONE_MIN_MS, 101, 102, 101, 102, 3],
        )
        == 1
    )

    metrics = await cm.get_latest_cached_ema_metric_spans(
        symbol,
        {"close": [3], "volume": [3], "log_range": [3]},
        max_staleness_ms=10 * ONE_MIN_MS,
        window_candles=3,
        timeframe="1m",
    )

    assert set(metrics) == {"close", "volume", "log_range"}
    assert set(metrics["close"]) == {3.0}
    assert metrics["close"][3.0] > 100.0
    assert metrics["volume"][3.0] > 1.0
    assert metrics["log_range"][3.0] >= 0.0
    primary = await cm.get_latest_ema_metric_spans(
        symbol,
        {"close": [3], "qv": [3], "log_range": [3]},
        max_age_ms=365 * 24 * 60 * ONE_MIN_MS,
        timeframe="1m",
        allow_remote_fetch=False,
        allow_provisional_internal_gaps=False,
    )
    assert set(primary["close"]) == {3.0}
    assert set(primary["qv"]) == {3.0}
    assert set(primary["log_range"]) == {3.0}


@pytest.mark.asyncio
async def test_persisted_ws_tail_is_reproducible_after_manager_restart(tmp_path):
    now_ms = 5 * ONE_MIN_MS + 10_000
    symbol = "BTC/USDT:USDT"
    first = _manager(tmp_path, now_ms=now_ms)
    first._persist_batch(
        symbol,
        _candles(
            (2 * ONE_MIN_MS, 100, 100, 100, 100, 1),
            (3 * ONE_MIN_MS, 100, 101, 100, 101, 2),
        ),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=3 * ONE_MIN_MS,
    )
    assert (
        await _prime_and_finalize_ws_row(
            first,
            symbol,
            [4 * ONE_MIN_MS, 101, 102, 101, 102, 3],
        )
        == 1
    )
    expected = await first.get_latest_cached_ema_metrics(
        symbol,
        {"close": 3},
        max_staleness_ms=10 * ONE_MIN_MS,
        timeframe="1m",
    )

    restarted = _manager(tmp_path, now_ms=now_ms)
    actual = await restarted.get_latest_cached_ema_metrics(
        symbol,
        {"close": 3},
        max_staleness_ms=10 * ONE_MIN_MS,
        timeframe="1m",
    )

    assert restarted.get_last_final_ts(symbol) == 4 * ONE_MIN_MS
    assert restarted.get_last_live_ws_ohlcv_ts(symbol) == 4 * ONE_MIN_MS
    assert actual["close"] == pytest.approx(expected["close"])


@pytest.mark.asyncio
async def test_ws_reconnect_gap_and_silence_remain_missing(tmp_path):
    now_ms = 4 * ONE_MIN_MS + 5_000
    cm = _manager(tmp_path, now_ms=now_ms)
    symbol = "BTC/USDT:USDT"
    cm._persist_batch(
        symbol,
        _candles((ONE_MIN_MS, 100, 100, 100, 100, 1)),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=now_ms,
    )

    assert await cm.ingest_live_ws_ohlcv(symbol, [], now_ms=now_ms) == 0
    assert (
        await _prime_and_finalize_ws_row(
            cm,
            symbol,
            [3 * ONE_MIN_MS, 101, 101, 101, 101, 1],
        )
        == 1
    )
    report = cm.get_completed_candle_health(
        symbol,
        {"1m": 3},
        now_ms=now_ms,
    )["timeframes"]["1m"]

    assert report["coverage_ok"] is False
    assert report["missing_spans"] == [(2 * ONE_MIN_MS, 2 * ONE_MIN_MS)]
    assert report["runtime_synthetic_count"] == 0


@pytest.mark.asyncio
async def test_rest_correction_replaces_persisted_ws_row_and_invalidates_ema(tmp_path):
    now_ms = 3 * ONE_MIN_MS + 5_000
    cm = _manager(tmp_path, now_ms=now_ms)
    symbol = "BTC/USDT:USDT"
    cm._persist_batch(
        symbol,
        _candles((ONE_MIN_MS, 100, 100, 100, 100, 1)),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=now_ms,
    )
    assert (
        await _prime_and_finalize_ws_row(
            cm,
            symbol,
            [2 * ONE_MIN_MS, 100, 102, 99, 101, 2],
        )
        == 1
    )
    before = await cm.get_latest_cached_ema_metrics(
        symbol,
        {"close": 1},
        max_staleness_ms=10 * ONE_MIN_MS,
        timeframe="1m",
    )
    assert before["close"] == pytest.approx(101.0)

    cm._persist_batch(
        symbol,
        _candles((2 * ONE_MIN_MS, 100, 103, 99, 102, 2)),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=now_ms + 1,
    )
    after = await cm.get_latest_cached_ema_metrics(
        symbol,
        {"close": 1},
        max_staleness_ms=10 * ONE_MIN_MS,
        timeframe="1m",
    )

    assert cm.get_last_live_ws_ohlcv_ts(symbol) == 2 * ONE_MIN_MS
    assert cm.get_last_live_ws_persist_ms(symbol) < cm.get_last_refresh_ms(symbol)
    assert after["close"] == pytest.approx(102.0)


@pytest.mark.asyncio
async def test_successor_timestamp_proves_unchanged_preceding_candle_final(
    tmp_path,
):
    symbol = "BTC/USDT:USDT"
    cm = _manager(tmp_path, now_ms=4 * ONE_MIN_MS)
    cm._persist_batch(
        symbol,
        _candles((ONE_MIN_MS, 100, 100, 100, 100, 1)),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=ONE_MIN_MS,
    )
    row = [2 * ONE_MIN_MS, 100, 101, 99, 100.5, 2]

    assert (
        await cm.ingest_live_ws_ohlcv(
            symbol,
            [row],
            now_ms=2 * ONE_MIN_MS + 50_000,
        )
        == 0
    )
    assert (
        await cm.ingest_live_ws_ohlcv(
            symbol,
            [[3 * ONE_MIN_MS, 100.5, 101, 100, 100.7, 0.2]],
            now_ms=3 * ONE_MIN_MS + 1_000,
        )
        == 1
    )
    assert cm.get_last_final_ts(symbol) == 2 * ONE_MIN_MS


@pytest.mark.asyncio
async def test_changed_open_row_processed_after_boundary_cannot_extend_tail(
    tmp_path,
):
    symbol = "BTC/USDT:USDT"
    cm = _manager(tmp_path, now_ms=4 * ONE_MIN_MS)
    cm._persist_batch(
        symbol,
        _candles((ONE_MIN_MS, 100, 100, 100, 100, 1)),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=ONE_MIN_MS,
    )
    partial = [2 * ONE_MIN_MS, 100, 101, 99, 100.5, 2]
    changed = [2 * ONE_MIN_MS, 100, 102, 99, 101.5, 3]

    assert (
        await cm.ingest_live_ws_ohlcv(
            symbol,
            [partial],
            now_ms=2 * ONE_MIN_MS + 50_000,
        )
        == 0
    )
    # This payload could have been emitted while the bucket was open and then
    # delayed in the consumer queue until after the boundary. A value change
    # alone is therefore insufficient to extend canonical history.
    assert (
        await cm.ingest_live_ws_ohlcv(
            symbol,
            [changed],
            now_ms=3 * ONE_MIN_MS + 1_000,
        )
        == 0
    )
    assert cm.get_last_final_ts(symbol) == ONE_MIN_MS
    assert (
        await cm.ingest_live_ws_ohlcv(
            symbol,
            [[3 * ONE_MIN_MS, 101.5, 102, 101, 101.7, 0.2]],
            now_ms=3 * ONE_MIN_MS + 2_000,
        )
        == 1
    )
    disk = cm._load_from_disk(
        symbol,
        2 * ONE_MIN_MS,
        2 * ONE_MIN_MS,
        timeframe="1m",
    )
    persisted = disk[disk["ts"] == 2 * ONE_MIN_MS]
    assert persisted.shape[0] == 1
    assert float(persisted[0]["c"]) == pytest.approx(101.5)


@pytest.mark.asyncio
async def test_unchanged_open_row_processed_after_boundary_cannot_extend_tail(
    tmp_path,
):
    symbol = "BTC/USDT:USDT"
    cm = _manager(tmp_path, now_ms=4 * ONE_MIN_MS)
    cm._persist_batch(
        symbol,
        _candles((ONE_MIN_MS, 100, 100, 100, 100, 1)),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=ONE_MIN_MS,
    )
    partial = [2 * ONE_MIN_MS, 100, 101, 99, 100.5, 2]

    assert (
        await cm.ingest_live_ws_ohlcv(
            symbol,
            [partial],
            now_ms=2 * ONE_MIN_MS + 50_000,
        )
        == 0
    )
    # Processing time is not transport provenance: this unchanged payload may
    # have been emitted before close and resumed late by the event loop.
    assert (
        await cm.ingest_live_ws_ohlcv(
            symbol,
            [partial],
            now_ms=3 * ONE_MIN_MS + 1_000,
        )
        == 0
    )
    assert cm.get_last_final_ts(symbol) == ONE_MIN_MS
    assert (
        await cm.ingest_live_ws_ohlcv(
            symbol,
            [[3 * ONE_MIN_MS, 100.5, 101, 100, 100.7, 0.2]],
            now_ms=3 * ONE_MIN_MS + 2_000,
        )
        == 1
    )
    assert cm.get_last_final_ts(symbol) == 2 * ONE_MIN_MS


def test_ws_legacy_shard_noop_cannot_expose_ram_only_correction(
    tmp_path,
    monkeypatch,
):
    symbol = "BTC/USDT:USDT"
    cm = _manager(tmp_path, now_ms=24 * 60 * ONE_MIN_MS)
    day_key = "1970-01-01"
    legacy_path = tmp_path / "legacy" / f"{day_key}.npy"
    legacy_path.parent.mkdir(parents=True)
    legacy = _candles(
        *[
            (minute * ONE_MIN_MS, 100, 101, 99, 100, 1)
            for minute in range(24 * 60)
        ]
    )
    np.save(legacy_path, legacy, allow_pickle=False)
    monkeypatch.setattr(
        cm,
        "_get_legacy_shard_paths",
        lambda _symbol, _tf, **_kwargs: {day_key: str(legacy_path)},
    )
    monkeypatch.setattr(
        cm,
        "_legacy_day_is_complete",
        lambda _symbol, _tf, _day: True,
    )
    final_ts = (24 * 60 - 1) * ONE_MIN_MS
    cm._load_from_disk(symbol, final_ts, final_ts, timeframe="1m")
    ema_key = ("close", 3.0, str(ONE_MIN_MS))
    ema_value = (100.0, final_ts, final_ts)
    cm._ema_cache[symbol] = {ema_key: ema_value}

    correction = _candles((final_ts, 100, 102, 99, 101, 2))
    with pytest.raises(OSError, match="persistence verification failed"):
        cm._persist_batch(
            symbol,
            correction,
            timeframe="1m",
            merge_cache=True,
            source="ws",
        )

    cached = cm._ensure_symbol_cache(symbol)
    canonical = cached[cached["ts"] == final_ts]
    assert canonical.shape[0] == 1
    assert float(canonical[0]["c"]) == pytest.approx(100.0)
    assert cm._ema_cache[symbol] == {ema_key: ema_value}
    assert not Path(cm._shard_path(symbol, day_key, timeframe="1m")).exists()
    assert cm.get_last_live_ws_ohlcv_ts(symbol) == 0


@pytest.mark.asyncio
async def test_ws_persistence_failure_does_not_expose_ram_only_candle(
    tmp_path,
    monkeypatch,
):
    symbol = "BTC/USDT:USDT"
    cm = _manager(tmp_path, now_ms=4 * ONE_MIN_MS)
    cm._persist_batch(
        symbol,
        _candles((ONE_MIN_MS, 100, 100, 100, 100, 1)),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=ONE_MIN_MS,
    )
    partial = [2 * ONE_MIN_MS, 100, 101, 99, 100.5, 2]
    assert (
        await cm.ingest_live_ws_ohlcv(
            symbol,
            [partial],
            now_ms=2 * ONE_MIN_MS + 50_000,
        )
        == 0
    )
    ema_key = ("close", 3.0, str(ONE_MIN_MS))
    ema_value = (100.0, ONE_MIN_MS, ONE_MIN_MS)
    projected_value = {("sentinel",): {"close": 100.0}}
    cm._ema_cache[symbol] = {ema_key: ema_value}
    cm._projected_open_tail_ema_cache[symbol] = projected_value

    def fail_save(*_args, **_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(cm, "_save_range_incremental", fail_save)
    with pytest.raises(OSError, match="disk unavailable"):
        await cm.ingest_live_ws_ohlcv(
            symbol,
            [[3 * ONE_MIN_MS, 100.5, 101, 100, 100.7, 0.2]],
            now_ms=3 * ONE_MIN_MS + 1_000,
        )

    assert cm.get_last_final_ts(symbol) == ONE_MIN_MS
    assert not np.any(cm._ensure_symbol_cache(symbol)["ts"] == 2 * ONE_MIN_MS)
    disk = cm._load_from_disk(
        symbol,
        2 * ONE_MIN_MS,
        2 * ONE_MIN_MS,
        timeframe="1m",
    )
    assert not np.any(disk["ts"] == 2 * ONE_MIN_MS)
    assert cm.get_last_live_ws_ohlcv_ts(symbol) == 0
    assert cm._ema_cache[symbol] == {ema_key: ema_value}
    assert cm._projected_open_tail_ema_cache[symbol] == projected_value


@pytest.mark.asyncio
async def test_ws_unavailable_disk_reads_cannot_publish_unverified_rows(tmp_path, monkeypatch):
    symbol = "BTC/USDT:USDT"
    cm = _manager(tmp_path, now_ms=4 * ONE_MIN_MS)
    basis = _candles((ONE_MIN_MS, 100, 100, 100, 100, 1))
    cm._persist_batch(symbol, basis, timeframe="1m", merge_cache=True)

    # A failed readback cannot prove persistence even when the write succeeds.
    load_from_disk = cm._load_from_disk

    def unavailable_readback(*args, **kwargs):
        if not kwargs.get("merge_memory_cache", True):
            return None
        return load_from_disk(*args, **kwargs)

    monkeypatch.setattr(cm, "_load_from_disk", unavailable_readback)
    with pytest.raises(OSError, match="persistence verification failed"):
        await _prime_and_finalize_ws_row(
            cm, symbol, [2 * ONE_MIN_MS, 100, 101, 99, 100.5, 2]
        )
    np.testing.assert_array_equal(cm._ensure_symbol_cache(symbol), basis)
    assert cm.get_last_live_ws_ohlcv_ts(symbol) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("storage", ["primary", "legacy", "mixed_primary", "mixed_legacy"])
@pytest.mark.parametrize("failure", ["corrupt", "unreadable"])
async def test_ws_failed_existing_shard_read_preserves_history(
    tmp_path, monkeypatch, storage, failure
):
    symbol = "BTC/USDT:USDT"
    midnight = 24 * 60 * ONE_MIN_MS
    cm = _manager(tmp_path, now_ms=midnight + 4 * ONE_MIN_MS)
    basis = _candles(
        (ONE_MIN_MS, 100, 100, 100, 100, 1),
        (midnight + ONE_MIN_MS, 100, 100, 100, 100, 1),
    )
    cm._persist_batch(symbol, basis, timeframe="1m", merge_cache=True)
    primary = Path(cm._shard_path(symbol, "1970-01-02", timeframe="1m"))
    shard = primary
    legacy = None
    if storage != "primary":
        legacy_dir = tmp_path / "legacy"
        legacy_dir.mkdir()
        legacy = legacy_dir / primary.name
        if storage == "legacy":
            primary.rename(legacy)
        else:
            np.save(legacy, _candles((midnight, 99, 100, 98, 99, 1)))
        shard = primary if storage == "mixed_primary" else legacy
        monkeypatch.setattr(cm, "_legacy_shard_dirs", lambda *_args: [str(legacy_dir)])
        cm._legacy_shard_paths_cache.clear()
        cm._invalidate_shard_paths_cache(symbol, tf="1m")
    if failure == "corrupt":
        shard.write_bytes(b"invalid numpy shard")
    else:
        load = np.load

        def unreadable(file, *args, **kwargs):
            if Path(file.name) == shard:
                raise PermissionError("unreadable fixture")
            return load(file, *args, **kwargs)

        monkeypatch.setattr(np, "load", unreadable)
    original_bytes = {path: path.read_bytes() for path in [primary, legacy] if path and path.exists()}
    expected_error = ValueError if failure == "corrupt" else PermissionError
    with pytest.raises(expected_error):
        await _prime_and_finalize_ws_row(
            cm, symbol, [midnight + 2 * ONE_MIN_MS, 100, 101, 99, 100.5, 2]
        )

    assert {path: path.read_bytes() for path in original_bytes} == original_bytes
    if storage == "legacy":
        assert not primary.exists()
    np.testing.assert_array_equal(cm._ensure_symbol_cache(symbol), basis)
    assert cm.get_last_live_ws_ohlcv_ts(symbol) == 0


@pytest.mark.asyncio
async def test_ws_merge_read_failure_after_successful_canonical_read(tmp_path, monkeypatch):
    symbol = "BTC/USDT:USDT"
    cm = _manager(tmp_path, now_ms=4 * ONE_MIN_MS)
    basis = _candles((ONE_MIN_MS, 100, 100, 100, 100, 1))
    cm._persist_batch(symbol, basis, timeframe="1m", merge_cache=True)
    shard = Path(cm._shard_path(symbol, "1970-01-01", timeframe="1m"))
    original_bytes = shard.read_bytes()
    load = np.load
    reads = 0

    def fail_second_read(file, *args, **kwargs):
        nonlocal reads
        reads += 1
        if reads == 2:
            raise PermissionError("merge read unavailable")
        return load(file, *args, **kwargs)

    monkeypatch.setattr(np, "load", fail_second_read)
    with pytest.raises(PermissionError):
        await _prime_and_finalize_ws_row(
            cm, symbol, [2 * ONE_MIN_MS, 100, 101, 99, 100.5, 2]
        )
    assert reads == 2
    assert shard.read_bytes() == original_bytes
    np.testing.assert_array_equal(cm._ensure_symbol_cache(symbol), basis)
    assert cm.get_last_live_ws_ohlcv_ts(symbol) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("unreadable", [False, True])
async def test_ws_merge_reads_actual_shard_after_another_manager_publishes_day(
    tmp_path, monkeypatch, unreadable
):
    midnight = 24 * 60 * ONE_MIN_MS
    symbol = "BTC/USDT:USDT"
    cm = _manager(tmp_path, now_ms=midnight + 3 * ONE_MIN_MS)
    basis = _candles((midnight - ONE_MIN_MS, 100, 101, 99, 100, 1))
    cm._persist_batch(symbol, basis, timeframe="1m", merge_cache=True)
    # This manager caches discovery before the new day exists.
    assert "1970-01-02" not in cm._iter_shard_paths(symbol, tf="1m")
    other = _manager(tmp_path, now_ms=midnight + 3 * ONE_MIN_MS)
    first_row = _candles((midnight, 100, 102, 99, 101, 2))
    other._persist_batch(symbol, first_row, timeframe="1m", merge_cache=True)
    assert "1970-01-02" not in cm._iter_shard_paths(symbol, tf="1m")
    shard = Path(cm._shard_path(symbol, "1970-01-02", timeframe="1m"))
    original_bytes = shard.read_bytes()
    row = [midnight + ONE_MIN_MS, 101, 103, 100, 102, 3]
    if unreadable:
        load = np.load

        def fail_read(file, *args, **kwargs):
            if Path(file.name) == shard:
                raise PermissionError("newly discovered target unavailable")
            return load(file, *args, **kwargs)

        monkeypatch.setattr(np, "load", fail_read)
        with pytest.raises(PermissionError):
            await _prime_and_finalize_ws_row(cm, symbol, row)
        assert shard.read_bytes() == original_bytes
        assert not np.any(cm._ensure_symbol_cache(symbol)["ts"] > midnight)
        assert cm.get_last_live_ws_ohlcv_ts(symbol) == 0
    else:
        assert await _prime_and_finalize_ws_row(cm, symbol, row) == 1
        expected = np.concatenate([first_row, _candles(tuple(row))])
        np.testing.assert_array_equal(np.load(shard, allow_pickle=False), expected)
        restarted = _manager(tmp_path, now_ms=midnight + 3 * ONE_MIN_MS)
        disk = restarted._load_from_disk(
            symbol, midnight, midnight + ONE_MIN_MS,
            timeframe="1m", merge_memory_cache=False,
        )
        np.testing.assert_array_equal(disk, expected)


@pytest.mark.asyncio
async def test_ws_refreshes_missing_legacy_discovery_before_verifying_correction(
    tmp_path, monkeypatch
):
    midnight = 24 * 60 * ONE_MIN_MS
    symbol = "BTC/USDT:USDT"
    cm = _manager(tmp_path, now_ms=midnight + 2 * ONE_MIN_MS)
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    monkeypatch.setattr(cm, "_legacy_shard_dirs", lambda *_args: [str(legacy_dir)])
    cm._persist_batch(
        symbol, _candles((midnight - ONE_MIN_MS, 100, 101, 99, 100, 1)),
        timeframe="1m", merge_cache=True,
    )
    assert cm._get_legacy_shard_paths(symbol, "1m") == {}
    canonical = _candles((midnight, 90, 91, 89, 90, 2))
    legacy = legacy_dir / "1970-01-02.npy"
    np.save(legacy, canonical)
    original_bytes = legacy.read_bytes()

    # A conflicting correction cannot become canonical in the running manager
    # while a newly published legacy source wins on a fresh manager's read.
    with pytest.raises(OSError, match="persistence verification failed"):
        await _prime_and_finalize_ws_row(
            cm, symbol, [midnight, 100, 101, 99, 100, 2]
        )
    assert legacy.read_bytes() == original_bytes
    np.testing.assert_array_equal(cm._ensure_symbol_cache(symbol)[-1:], canonical)
    assert cm.get_last_live_ws_ohlcv_ts(symbol) == 0
    restarted = _manager(tmp_path, now_ms=midnight + 2 * ONE_MIN_MS)
    monkeypatch.setattr(restarted, "_legacy_shard_dirs", lambda *_args: [str(legacy_dir)])
    np.testing.assert_array_equal(
        restarted._load_from_disk(
            symbol, midnight, midnight, timeframe="1m", merge_memory_cache=False
        ),
        canonical,
    )


@pytest.mark.asyncio
async def test_ws_legacy_discovery_failure_propagates(tmp_path, monkeypatch):
    symbol = "BTC/USDT:USDT"
    cm = _manager(tmp_path, now_ms=4 * ONE_MIN_MS)
    basis = _candles((ONE_MIN_MS, 100, 100, 100, 100, 1))
    cm._persist_batch(symbol, basis, timeframe="1m", merge_cache=True)
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    monkeypatch.setattr(cm, "_legacy_shard_dirs", lambda *_args: [str(legacy_dir)])
    iterdir = Path.iterdir

    def fail_legacy_scan(path):
        if path == legacy_dir:
            raise PermissionError("legacy discovery unavailable")
        return iterdir(path)

    monkeypatch.setattr(Path, "iterdir", fail_legacy_scan)
    with pytest.raises(PermissionError):
        await _prime_and_finalize_ws_row(
            cm, symbol, [2 * ONE_MIN_MS, 100, 101, 99, 100.5, 2]
        )
    np.testing.assert_array_equal(cm._ensure_symbol_cache(symbol), basis)
    assert cm.get_last_live_ws_ohlcv_ts(symbol) == 0


@pytest.mark.asyncio
async def test_delayed_ws_correction_overwrites_persisted_row(tmp_path):
    symbol = "BTC/USDT:USDT"
    cm = _manager(tmp_path, now_ms=4 * ONE_MIN_MS)
    cm._persist_batch(
        symbol,
        _candles((ONE_MIN_MS, 100, 100, 100, 100, 1)),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=ONE_MIN_MS,
    )

    assert (
        await _prime_and_finalize_ws_row(
            cm,
            symbol,
            [2 * ONE_MIN_MS, 100, 101, 99, 100.5, 2],
        )
        == 1
    )
    assert (
        await cm.ingest_live_ws_ohlcv(
            symbol,
            [[2 * ONE_MIN_MS, 100, 102, 99, 101.5, 3]],
            now_ms=3 * ONE_MIN_MS + 2_000,
        )
        == 1
    )

    disk = cm._load_from_disk(
        symbol,
        2 * ONE_MIN_MS,
        2 * ONE_MIN_MS,
        timeframe="1m",
    )
    corrected = disk[disk["ts"] == 2 * ONE_MIN_MS]
    assert corrected.shape[0] == 1
    assert float(corrected[0]["c"]) == pytest.approx(101.5)


@pytest.mark.asyncio
async def test_rest_omission_does_not_delete_validated_ws_candle(tmp_path):
    symbol = "BTC/USDT:USDT"
    now_ms = 5 * ONE_MIN_MS + 1_000
    cm = _manager(tmp_path, now_ms=now_ms)
    cm._persist_batch(
        symbol,
        _candles(
            (ONE_MIN_MS, 100, 100, 100, 100, 1),
            (2 * ONE_MIN_MS, 100, 100, 100, 100, 1),
        ),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=2 * ONE_MIN_MS,
    )
    assert (
        await _prime_and_finalize_ws_row(
            cm,
            symbol,
            [3 * ONE_MIN_MS, 100, 101, 99, 100.5, 1],
        )
        == 1
    )

    cm._persist_batch(
        symbol,
        _candles((4 * ONE_MIN_MS, 100.5, 101, 100, 100.7, 1)),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=now_ms,
    )
    disk = cm._load_from_disk(
        symbol,
        ONE_MIN_MS,
        4 * ONE_MIN_MS,
        timeframe="1m",
    )
    assert list(disk["ts"]) == [
        ONE_MIN_MS,
        2 * ONE_MIN_MS,
        3 * ONE_MIN_MS,
        4 * ONE_MIN_MS,
    ]


@pytest.mark.asyncio
async def test_first_post_reconnect_snapshot_cannot_persist_replayed_partial_row(
    tmp_path,
):
    symbol = "BTC/USDT:USDT"
    cm = _manager(tmp_path, now_ms=4 * ONE_MIN_MS)
    cm._persist_batch(
        symbol,
        _candles((ONE_MIN_MS, 100, 100, 100, 100, 1)),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=ONE_MIN_MS,
    )
    partial = [2 * ONE_MIN_MS, 100, 101, 99, 100.5, 2]
    current = [3 * ONE_MIN_MS, 100.5, 101, 100, 100.7, 0.2]

    assert (
        await cm.ingest_live_ws_ohlcv(
            symbol,
            [partial],
            now_ms=3 * ONE_MIN_MS - 10_000,
        )
        == 0
    )
    cm.clear_live_ws_ohlcv_state(symbol)
    assert (
        await cm.ingest_live_ws_ohlcv(
            symbol,
            [partial, current],
            now_ms=3 * ONE_MIN_MS + 1_000,
        )
        == 0
    )
    changed_current = [3 * ONE_MIN_MS, 100.5, 102, 100, 101, 0.5]
    assert (
        await cm.ingest_live_ws_ohlcv(
            symbol,
            [partial, changed_current],
            now_ms=3 * ONE_MIN_MS + 2_000,
        )
        == 0
    )

    disk = cm._load_from_disk(
        symbol,
        2 * ONE_MIN_MS,
        2 * ONE_MIN_MS,
        timeframe="1m",
    )
    assert not np.any(disk["ts"] == 2 * ONE_MIN_MS)
    assert cm.get_last_final_ts(symbol) == ONE_MIN_MS


class _BlockingCCP:
    def __init__(self):
        self.has = {"watchOHLCV": True}
        self.started: list[str] = []
        self.unwatched: list[str] = []
        self.events: dict[str, asyncio.Event] = {}

    async def watch_ohlcv(self, symbol, timeframe):
        assert timeframe == "1m"
        self.started.append(symbol)
        event = self.events.setdefault(symbol, asyncio.Event())
        await event.wait()
        return []

    async def un_watch_ohlcv(self, symbol, timeframe):
        assert timeframe == "1m"
        self.unwatched.append(symbol)
        self.events.setdefault(symbol, asyncio.Event()).set()


@pytest.mark.asyncio
async def test_watcher_trims_large_exchange_snapshot_to_recent_tail():
    rows = [[index * ONE_MIN_MS, 1, 1, 1, 1, 1] for index in range(20)]

    class _SnapshotCCP:
        async def watch_ohlcv(self, symbol, timeframe):
            assert symbol == "BTC/USDT:USDT"
            assert timeframe == "1m"
            return rows

        async def un_watch_ohlcv(self, symbol, timeframe):
            assert symbol == "BTC/USDT:USDT"
            assert timeframe == "1m"

    received = []
    bot = SimpleNamespace(
        ccp=_SnapshotCCP(),
        cm=None,
        stop_signal_received=False,
    )

    def ingest(symbol, incoming):
        received.append((symbol, incoming))
        bot.stop_signal_received = True

    bot.cm = SimpleNamespace(ingest_live_ws_ohlcv=ingest)

    await candle_ws.watch_forager_ws_symbol(bot, "BTC/USDT:USDT")

    assert received == [("BTC/USDT:USDT", rows[-3:])]


def test_zero_ohlcv_budget_disables_forager_ws_network_path():
    bot = SimpleNamespace(
        config={
            "live": {
                "enable_forager_ws_candles": True,
                "max_ohlcv_fetches_per_minute": 0,
            }
        },
        ws_enabled=True,
        ccp=SimpleNamespace(has={"watchOHLCV": True}, watch_ohlcv=lambda: None),
        is_forager_mode=lambda: True,
    )

    assert candle_ws.forager_ws_candles_enabled(bot) is False


@pytest.mark.parametrize(
    ("feature_enabled", "ws_enabled", "capability"),
    [
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ],
)
def test_ws_disabled_or_unsupported_preserves_rest_only_mode(
    feature_enabled, ws_enabled, capability
):
    bot = SimpleNamespace(
        config={
            "live": {
                "enable_forager_ws_candles": feature_enabled,
                "max_ohlcv_fetches_per_minute": 30,
            }
        },
        ws_enabled=ws_enabled,
        ccp=SimpleNamespace(
            has={"watchOHLCV": capability},
            watch_ohlcv=(lambda *_args: None),
        ),
        is_forager_mode=lambda _pside=None: True,
    )

    assert candle_ws.forager_ws_candles_enabled(bot) is False


def test_repeated_ws_failures_enter_rest_fallback_cooldown():
    assert candle_ws.reconnect_delay_seconds(1) == 1.0
    assert candle_ws.reconnect_delay_seconds(4) == 8.0
    assert candle_ws.reconnect_delay_seconds(5) == 300.0
    assert candle_ws.reconnect_delay_seconds(50) == 300.0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["receive", "ingest"])
async def test_ws_failures_identify_stage_and_preserve_rest_fallback_cooldown(
    failure_stage, caplog
):
    delays = []

    class _SuccessfulWatcher:
        has = {"watchOHLCV": True}

        async def watch_ohlcv(self, _symbol, _timeframe):
            if failure_stage == "receive":
                raise OSError("sensitive transport details")
            return [[ONE_MIN_MS, 1, 1, 1, 1, 1]]

    class _FailingManager:
        async def ingest_live_ws_ohlcv(self, _symbol, _rows):
            raise OSError("sensitive persistence details")

        def clear_live_ws_ohlcv_state(self, _symbol):
            return None

    bot = SimpleNamespace(
        ccp=_SuccessfulWatcher(),
        cm=_FailingManager(),
        stop_signal_received=False,
        get_exchange_time=lambda: 10_000_000,
    )

    async def record_sleep(delay_s, *, stage):
        assert stage == "forager_ws_candle_reconnect"
        delays.append(delay_s)
        if len(delays) == 5:
            bot.stop_signal_received = True

    bot._sleep_unless_shutdown = record_sleep

    await candle_ws.watch_forager_ws_symbol(bot, "BTC/USDT:USDT")

    assert delays == [1.0, 2.0, 4.0, 8.0, 300.0]
    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "forager websocket candle update failed" in warnings[0].message
    assert f"stage={failure_stage}" in warnings[0].message
    assert "error_type=OSError" in warnings[0].message
    assert "action=rest_fallback" in warnings[0].message
    assert "sensitive" not in caplog.text


@pytest.mark.asyncio
async def test_dynamic_subscriptions_follow_flat_forager_universe():
    ccp = _BlockingCCP()
    bot = SimpleNamespace(
        config={"live": {"enable_forager_ws_candles": True}},
        ws_enabled=True,
        ccp=ccp,
        cm=SimpleNamespace(ingest_live_ws_ohlcv=lambda *_args: 0),
        stop_signal_received=False,
        approved_coins_minus_ignored_coins={
            "long": {"A/USDT:USDT", "ACTIVE/USDT:USDT"},
            "short": {"SHORT_ONLY/USDT:USDT"},
        },
        is_forager_mode=lambda pside=None: pside in {None, "long"},
        _urgent_active_candle_symbols=lambda: ["ACTIVE/USDT:USDT"],
    )
    tasks: dict[str, asyncio.Task] = {}

    added, removed = await candle_ws.reconcile_forager_ws_tasks(bot, tasks)
    await asyncio.sleep(0)

    assert added == {"A/USDT:USDT"}
    assert removed == set()
    assert set(tasks) == {"A/USDT:USDT"}
    assert ccp.started == ["A/USDT:USDT"]

    bot.approved_coins_minus_ignored_coins = {
        "long": {"B/USDT:USDT"},
        "short": set(),
    }
    added, removed = await candle_ws.reconcile_forager_ws_tasks(bot, tasks)
    await asyncio.sleep(0)

    assert added == {"B/USDT:USDT"}
    assert removed == {"A/USDT:USDT"}
    assert set(tasks) == {"B/USDT:USDT"}
    assert ccp.unwatched == ["A/USDT:USDT"]

    for task in tasks.values():
        task.cancel()
    await asyncio.gather(*tasks.values(), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("bulk_supported", [False, True])
async def test_watcher_retirement_propagates_owner_cancellation(bulk_supported):
    unsubscribe_started = asyncio.Event()

    class _CancellationBlockingCCP:
        def __init__(self):
            self.single_calls = 0
            self.bulk_calls = 0
            if not bulk_supported:
                self.un_watch_ohlcv_for_symbols = None

        async def un_watch_ohlcv(self, _symbol, _timeframe):
            self.single_calls += 1
            unsubscribe_started.set()
            await asyncio.Event().wait()

        async def un_watch_ohlcv_for_symbols(self, _subscriptions):
            self.bulk_calls += 1
            unsubscribe_started.set()
            await asyncio.Event().wait()

    symbol = "CANCEL/USDT:USDT"
    watcher = asyncio.create_task(asyncio.Event().wait())
    bot = SimpleNamespace(ccp=_CancellationBlockingCCP())
    retirement = asyncio.create_task(
        candle_ws._retire_watchers(bot, {symbol: watcher})
    )

    try:
        await asyncio.wait_for(unsubscribe_started.wait(), timeout=1.0)
        retirement.cancel()
        with pytest.raises(asyncio.CancelledError):
            await retirement

        assert not watcher.done()
        assert not candle_ws._watcher_is_retiring(bot, symbol, watcher)
        if bulk_supported:
            assert bot.ccp.bulk_calls == 1
            assert bot.ccp.single_calls == 0
        else:
            assert bot.ccp.bulk_calls == 0
            assert bot.ccp.single_calls == 1
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("bulk_supported", [False, True])
async def test_unsubscribe_timeout_is_hard_when_connector_suppresses_cancellation(
    bulk_supported, monkeypatch
):
    monkeypatch.setattr(candle_ws, "_UNWATCH_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(candle_ws, "_UNWATCH_MANY_TIMEOUT_SECONDS", 0.01)
    started = asyncio.Event()
    release = asyncio.Event()

    async def cancellation_resistant_unsubscribe(*_args):
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue

    ccp = SimpleNamespace(un_watch_ohlcv=cancellation_resistant_unsubscribe)
    if bulk_supported:
        ccp.un_watch_ohlcv_for_symbols = cancellation_resistant_unsubscribe
    bot = SimpleNamespace(ccp=ccp)

    if bulk_supported:
        result = await asyncio.wait_for(
            candle_ws._best_effort_unwatch_many(bot, ["A/USDT:USDT"]),
            timeout=0.2,
        )
    else:
        result = await asyncio.wait_for(
            candle_ws._best_effort_unwatch(bot, "A/USDT:USDT"),
            timeout=0.2,
        )

    assert result is False
    await asyncio.wait_for(started.wait(), timeout=0.2)
    abandoned = candle_ws._abandoned_unsubscribe_tasks(bot)
    assert len(abandoned) == 1
    abandoned_task = next(iter(abandoned))
    release.set()
    await asyncio.wait_for(abandoned_task, timeout=1.0)
    await asyncio.sleep(0)
    assert candle_ws._abandoned_unsubscribe_tasks(bot) == set()


@pytest.mark.asyncio
async def test_reconcile_cancellation_keeps_removed_watcher_owned():
    unsubscribe_started = asyncio.Event()

    class _BlockingBulkCCP:
        async def un_watch_ohlcv_for_symbols(self, _subscriptions):
            unsubscribe_started.set()
            await asyncio.Event().wait()

        async def un_watch_ohlcv(self, _symbol, _timeframe):
            raise AssertionError("single fallback must not start after owner cancellation")

    symbol = "OWNED/USDT:USDT"
    watcher = asyncio.create_task(asyncio.Event().wait())
    tasks = {symbol: watcher}
    bot = SimpleNamespace(
        config={"live": {"enable_forager_ws_candles": True}},
        ws_enabled=True,
        ccp=_BlockingBulkCCP(),
        cm=SimpleNamespace(clear_live_ws_ohlcv_state=lambda _symbol: None),
        approved_coins_minus_ignored_coins={"long": set(), "short": set()},
        is_forager_mode=lambda pside=None: pside in {None, "long"},
        _urgent_active_candle_symbols=lambda: [],
    )
    reconciliation = asyncio.create_task(
        candle_ws.reconcile_forager_ws_tasks(bot, tasks)
    )

    try:
        await asyncio.wait_for(unsubscribe_started.wait(), timeout=1.0)
        reconciliation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await reconciliation

        assert tasks == {symbol: watcher}
        assert not watcher.done()
    finally:
        watcher.cancel()
        await asyncio.gather(watcher, return_exceptions=True)


@pytest.mark.asyncio
async def test_unsubscribe_wakeup_exception_is_consumed_before_watcher_retirement():
    class _UnsubscribeWakeupCCP:
        has = {"watchOHLCV": True}

        def __init__(self):
            self.future = None

        async def watch_ohlcv(self, _symbol, _timeframe):
            self.future = asyncio.get_running_loop().create_future()
            return await self.future

        async def un_watch_ohlcv(self, _symbol, _timeframe):
            self.future.set_exception(RuntimeError("unsubscribe wakeup"))

    ccp = _UnsubscribeWakeupCCP()
    bot = SimpleNamespace(
        ccp=ccp,
        cm=SimpleNamespace(ingest_live_ws_ohlcv=lambda *_args: 0),
        stop_signal_received=False,
    )
    task = asyncio.create_task(
        candle_ws.watch_forager_ws_symbol(bot, "BTC/USDT:USDT")
    )
    await asyncio.sleep(0)

    await candle_ws._retire_watcher(bot, "BTC/USDT:USDT", task)

    assert task.done()
    assert task.cancelled() is False
    assert ccp.future.done()
    assert ccp.future.exception() is not None


@pytest.mark.asyncio
async def test_retiring_watcher_does_not_resubscribe_after_inflight_ingestion():
    ingest_started = asyncio.Event()
    release_ingest = asyncio.Event()
    unsubscribe_called = asyncio.Event()

    class _CountingCCP:
        def __init__(self):
            self.watch_calls = 0

        async def watch_ohlcv(self, _symbol, _timeframe):
            self.watch_calls += 1
            if self.watch_calls > 1:
                raise AssertionError("retired watcher recreated its subscription")
            return [[ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 1.0]]

        async def un_watch_ohlcv(self, _symbol, _timeframe):
            unsubscribe_called.set()

    class _BlockingManager:
        async def ingest_live_ws_ohlcv(self, _symbol, _rows):
            ingest_started.set()
            await release_ingest.wait()

    ccp = _CountingCCP()
    symbol = "BTC/USDT:USDT"
    bot = SimpleNamespace(
        ccp=ccp,
        cm=_BlockingManager(),
        stop_signal_received=False,
    )
    watcher = asyncio.create_task(
        candle_ws.watch_forager_ws_symbol(bot, symbol)
    )
    await asyncio.wait_for(ingest_started.wait(), timeout=1.0)
    retirement = asyncio.create_task(
        candle_ws._retire_watcher(bot, symbol, watcher)
    )
    await asyncio.wait_for(unsubscribe_called.wait(), timeout=1.0)

    release_ingest.set()
    await asyncio.wait_for(retirement, timeout=1.0)

    assert watcher.done()
    assert watcher.cancelled() is False
    assert ccp.watch_calls == 1
    assert not candle_ws._watcher_is_retiring(bot, symbol, watcher)


@pytest.mark.asyncio
async def test_bulk_watcher_retirement_uses_one_supported_unsubscribe():
    class _BulkWakeupCCP:
        def __init__(self):
            self.futures = {}
            self.bulk_calls = []
            self.single_calls = []

        async def watch_ohlcv(self, symbol, _timeframe):
            future = asyncio.get_running_loop().create_future()
            self.futures[symbol] = future
            return await future

        async def un_watch_ohlcv_for_symbols(self, subscriptions):
            self.bulk_calls.append(subscriptions)
            for symbol, _timeframe in subscriptions:
                self.futures[symbol].set_exception(RuntimeError("unsubscribe wakeup"))

        async def un_watch_ohlcv(self, symbol, _timeframe):
            self.single_calls.append(symbol)

    ccp = _BulkWakeupCCP()
    bot = SimpleNamespace(
        ccp=ccp,
        cm=SimpleNamespace(ingest_live_ws_ohlcv=lambda *_args: 0),
        stop_signal_received=False,
    )
    tasks = {
        symbol: asyncio.create_task(candle_ws.watch_forager_ws_symbol(bot, symbol))
        for symbol in ("A/USDT:USDT", "B/USDT:USDT")
    }
    await asyncio.sleep(0)

    await candle_ws._retire_watchers(bot, tasks)

    assert ccp.bulk_calls == [
        [["A/USDT:USDT", "1m"], ["B/USDT:USDT", "1m"]]
    ]
    assert ccp.single_calls == []
    assert all(task.done() and not task.cancelled() for task in tasks.values())


@pytest.mark.asyncio
async def test_watcher_retirement_abandons_cancellation_resistant_task(
    caplog, monkeypatch
):
    monkeypatch.setattr(candle_ws, "_WATCHER_RETIRE_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(candle_ws, "_WATCHER_CANCEL_GRACE_SECONDS", 0.01)
    release = asyncio.Event()

    async def resistant_watcher():
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                continue

    symbol = "RESISTANT/USDT:USDT"
    task = asyncio.create_task(resistant_watcher())
    bot = SimpleNamespace(ccp=SimpleNamespace())

    try:
        with caplog.at_level(logging.WARNING):
            await asyncio.wait_for(
                candle_ws._retire_watchers(bot, {symbol: task}), timeout=0.2
            )
        assert not task.done()
        assert candle_ws._watcher_is_retiring(bot, symbol, task)
        assert "websocket watcher cancellation grace expired" in caplog.text
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=1.0)
        await asyncio.sleep(0)

    assert not candle_ws._watcher_is_retiring(bot, symbol, task)


@pytest.mark.asyncio
async def test_resistant_old_watcher_does_not_retire_replacement(monkeypatch):
    monkeypatch.setattr(candle_ws, "_WATCHER_RETIRE_GRACE_SECONDS", 0.01)
    monkeypatch.setattr(candle_ws, "_WATCHER_CANCEL_GRACE_SECONDS", 0.01)
    release_old = asyncio.Event()
    replacement_waiting = asyncio.Event()
    ingested = asyncio.Event()

    async def resistant_old_watcher():
        while not release_old.is_set():
            try:
                await release_old.wait()
            except asyncio.CancelledError:
                continue

    class _ReplacementCCP:
        def __init__(self):
            self.calls = 0

        async def watch_ohlcv(self, _symbol, _timeframe):
            self.calls += 1
            if self.calls == 1:
                return [[ONE_MIN_MS, 1.0, 1.0, 1.0, 1.0, 1.0]]
            await replacement_waiting.wait()
            return []

        async def un_watch_ohlcv(self, _symbol, _timeframe):
            return None

    symbol = "REENTERED/USDT:USDT"
    old_task = asyncio.create_task(resistant_old_watcher())
    bot = SimpleNamespace(
        ccp=_ReplacementCCP(),
        cm=SimpleNamespace(
            ingest_live_ws_ohlcv=lambda *_args: ingested.set(),
            clear_live_ws_ohlcv_state=lambda _symbol: None,
        ),
        stop_signal_received=False,
    )

    try:
        await candle_ws._retire_watchers(bot, {symbol: old_task})
        replacement = asyncio.create_task(
            candle_ws.watch_forager_ws_symbol(bot, symbol)
        )
        await asyncio.wait_for(ingested.wait(), timeout=1.0)
        await asyncio.sleep(0)

        assert candle_ws._watcher_is_retiring(bot, symbol, old_task)
        assert not candle_ws._watcher_is_retiring(bot, symbol, replacement)
        assert not replacement.done()
    finally:
        if "replacement" in locals():
            replacement.cancel()
            await asyncio.gather(replacement, return_exceptions=True)
        release_old.set()
        await asyncio.wait_for(old_task, timeout=1.0)
        await asyncio.sleep(0)

    assert not candle_ws._watcher_is_retiring(bot, symbol, old_task)


@pytest.mark.asyncio
async def test_reconcile_retires_removed_watchers_in_one_bulk_unsubscribe():
    class _BulkWakeupCCP:
        def __init__(self):
            self.futures = {}
            self.bulk_calls = []
            self.single_calls = []

        async def watch_ohlcv(self, symbol, _timeframe):
            future = asyncio.get_running_loop().create_future()
            self.futures[symbol] = future
            return await future

        async def un_watch_ohlcv_for_symbols(self, subscriptions):
            self.bulk_calls.append(subscriptions)
            for symbol, _timeframe in subscriptions:
                self.futures[symbol].set_exception(RuntimeError("unsubscribe wakeup"))

        async def un_watch_ohlcv(self, symbol, _timeframe):
            self.single_calls.append(symbol)

    ccp = _BulkWakeupCCP()
    bot = SimpleNamespace(
        config={"live": {"enable_forager_ws_candles": True}},
        ws_enabled=True,
        ccp=ccp,
        cm=SimpleNamespace(
            ingest_live_ws_ohlcv=lambda *_args: 0,
            clear_live_ws_ohlcv_state=lambda _symbol: None,
        ),
        stop_signal_received=False,
        approved_coins_minus_ignored_coins={"long": set(), "short": set()},
        is_forager_mode=lambda pside=None: pside in {None, "long"},
        _urgent_active_candle_symbols=lambda: [],
    )
    tasks = {
        symbol: asyncio.create_task(candle_ws.watch_forager_ws_symbol(bot, symbol))
        for symbol in ("A/USDT:USDT", "B/USDT:USDT")
    }
    await asyncio.sleep(0)

    added, removed = await candle_ws.reconcile_forager_ws_tasks(bot, tasks)

    assert added == set()
    assert removed == {"A/USDT:USDT", "B/USDT:USDT"}
    assert tasks == {}
    assert ccp.bulk_calls == [
        [["A/USDT:USDT", "1m"], ["B/USDT:USDT", "1m"]]
    ]
    assert ccp.single_calls == []


@pytest.mark.asyncio
async def test_ws_maintainer_starts_before_forager_mode_becomes_active():
    bot = Passivbot.__new__(Passivbot)
    bot.config = {
        "live": {
            "enable_forager_ws_candles": True,
            "max_ohlcv_fetches_per_minute": 30,
        }
    }
    bot.ws_enabled = True
    bot.ccp = SimpleNamespace(
        has={"watchOHLCV": True},
        watch_ohlcv=lambda *_args: None,
    )
    bot.monitor_publisher = None
    bot.is_forager_mode = lambda _pside=None: False
    blocker = asyncio.Event()

    async def wait_for_stop():
        await blocker.wait()

    bot.maintain_hourly_cycle = wait_for_stop
    bot.watch_orders = wait_for_stop
    bot.maintain_forager_ws_candles = wait_for_stop

    await bot.start_data_maintainers()

    assert "maintain_forager_ws_candles" in bot.maintainers
    tasks = list(bot.maintainers.values())
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def test_ws_tail_rest_audit_is_periodic_and_one_minute_only():
    bot = SimpleNamespace(
        config={"live": {"forager_ws_candle_rest_audit_minutes": 30}}
    )
    now_ms = 100 * ONE_MIN_MS
    health = {
        "ws_persisted_contributed_to_tail": True,
        "last_ws_persist_ms": now_ms - ONE_MIN_MS,
        "last_refresh_ms": now_ms - 29 * ONE_MIN_MS,
    }

    assert (
        Passivbot._forager_ws_rest_audit_due(
            bot, "1m", health, now_ms=now_ms
        )
        is False
    )
    health["last_refresh_ms"] = now_ms - 30 * ONE_MIN_MS
    assert (
        Passivbot._forager_ws_rest_audit_due(
            bot, "1m", health, now_ms=now_ms
        )
        is True
    )
    assert (
        Passivbot._forager_ws_rest_audit_due(
            bot, "1h", health, now_ms=now_ms
        )
        is False
    )
    health["last_refresh_ms"] = now_ms
    assert (
        Passivbot._forager_ws_rest_audit_due(
            bot, "1m", health, now_ms=now_ms
        )
        is False
    )


@pytest.mark.asyncio
async def test_forager_due_ws_audit_forces_rest_overlap(monkeypatch):
    import passivbot as passivbot_module

    symbol = "BTC/USDT:USDT"
    now_ms = 100 * ONE_MIN_MS
    monkeypatch.setattr(passivbot_module, "utc_ms", lambda: now_ms)
    monkeypatch.setattr(
        passivbot_module,
        "compute_live_warmup_windows",
        lambda *args, **kwargs: ({symbol: 3}, {symbol: 0}, {symbol: True}),
    )
    monkeypatch.setattr(
        Passivbot,
        "_urgent_active_candle_symbols",
        lambda _bot: [],
    )

    class _AuditCM:
        default_window_candles = 120

        def __init__(self):
            self.refresh_calls = []

        async def refresh(self, called_symbol, **kwargs):
            self.refresh_calls.append((called_symbol, kwargs))

        async def get_candles(self, *_args, **_kwargs):
            raise AssertionError("due WS audit must bypass ordinary get_candles")

    class _AuditBot:
        config = {
            "live": {
                "max_ohlcv_fetches_per_minute": 4,
                "max_forager_candle_refresh_seconds": 0,
                "forager_ws_candle_rest_audit_minutes": 30,
            }
        }
        approved_coins_minus_ignored_coins = {"long": {symbol}, "short": set()}
        stop_signal_received = False
        _shutdown_in_progress = False
        start_time_ms = now_ms
        cm = _AuditCM()

        def is_forager_mode(self, pside=None):
            return pside in (None, "long")

        def get_max_n_positions(self, pside):
            return 1 if pside == "long" else 0

        def get_current_n_positions(self, _pside):
            return 0

        def bp(self, *_args, **_kwargs):
            return 0.0

        def _get_fetch_delay_seconds(self):
            return 0.0

        def _forager_refresh_budget(self, *_args, **_kwargs):
            return 1

        def _forager_target_staleness_ms(self, *_args, **_kwargs):
            return 0

        _shutdown_requested = Passivbot._shutdown_requested

    health = {
        "coverage_ok": True,
        "age_ms": 0,
        "last_cached_ts": now_ms - ONE_MIN_MS,
        "missing_candles": 0,
        "tail_gap_candles": 0,
        "tail_only": False,
        "leading_gap_only": False,
        "no_basis": False,
        "last_refresh_ms": now_ms - 31 * ONE_MIN_MS,
        "last_ws_persist_ms": now_ms - ONE_MIN_MS,
        "ws_persisted_contributed_to_tail": True,
    }
    monkeypatch.setattr(
        Passivbot,
        "_candidate_candle_surface_health",
        lambda *_args, **_kwargs: dict(health),
    )

    bot = _AuditBot()
    await Passivbot._refresh_forager_candidate_candles(bot)

    assert bot.cm.refresh_calls == [
        (
            symbol,
            {
                "through_ts": now_ms - ONE_MIN_MS,
                "force_overlap": True,
            },
        )
    ]


@pytest.mark.asyncio
async def test_force_overlap_refresh_fetches_when_canonical_tail_is_fresh(
    tmp_path,
):
    now_ms = 5 * ONE_MIN_MS + 10_000
    symbol = "BTC/USDT:USDT"
    cm = _manager(tmp_path, now_ms=now_ms)
    cm.exchange = object()
    cm._persist_batch(
        symbol,
        _candles(
            (3 * ONE_MIN_MS, 100, 100, 100, 100, 1),
            (4 * ONE_MIN_MS, 100, 101, 99, 100.5, 2),
        ),
        timeframe="1m",
        merge_cache=True,
        last_refresh_ms=ONE_MIN_MS,
    )
    calls = []

    async def fetch(symbol_arg, start_ts, end_ts, **kwargs):
        calls.append((symbol_arg, start_ts, end_ts))
        corrected = _candles(
            (4 * ONE_MIN_MS, 100, 102, 99, 101.5, 3),
        )
        on_batch = kwargs.get("on_batch")
        if on_batch is not None:
            on_batch(corrected)
        return corrected

    cm._fetch_ohlcv_paginated = fetch

    await cm.refresh(symbol, through_ts=4 * ONE_MIN_MS)
    assert calls == []

    await cm.refresh(
        symbol,
        through_ts=4 * ONE_MIN_MS,
        force_overlap=True,
    )

    assert len(calls) == 1
    disk = cm._load_from_disk(
        symbol,
        4 * ONE_MIN_MS,
        4 * ONE_MIN_MS,
        timeframe="1m",
    )
    assert float(disk[-1]["c"]) == pytest.approx(101.5)
    assert cm.get_last_refresh_ms(symbol) > ONE_MIN_MS


def test_forager_staleness_uses_persisted_ws_canonical_tail():
    bot = SimpleNamespace(
        cm=SimpleNamespace(
            get_last_final_ts=lambda _symbol: 12 * ONE_MIN_MS,
            get_last_refresh_ms=lambda _symbol: 9 * ONE_MIN_MS,
        )
    )
    now_ms = 13 * ONE_MIN_MS + 10_000

    assert Passivbot._candle_staleness_ms(bot, "BTC", now_ms=now_ms) == 0


@pytest.mark.asyncio
async def test_shared_network_failure_preserves_each_symbol_and_recovery(monkeypatch, caplog):
    from ccxt.base.errors import NetworkError
    from live.candle_ws_health import observe_receive_status
    from live.event_bus import EventTypes
    events = []
    monkeypatch.setattr('live.candle_ws_health.emit_event', lambda bot, event: events.append(event))
    bot = SimpleNamespace()
    with caplog.at_level(logging.INFO):
        for symbol in ('BTC', 'ETH', 'SOL'):
            observe_receive_status(bot, symbol, error=NetworkError('secret'), retry_s=1.)
        for symbol in ('BTC', 'ETH'):
            observe_receive_status(bot, symbol, recovered=True)
        assert bot._console_candle_receive is not None
        observe_receive_status(bot, 'SOL', recovered=True)
    warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert 'action=rest_fallback' in warnings[0]
    assert 'receive recovered symbols=3 failures=3' in caplog.text
    assert 'secret' not in caplog.text
    assert [e['status'] for e in events] == ['failed']*3 + ['recovered']*3
    assert all(e['event_type'] == EventTypes.CANDLE_WEBSOCKET_STATUS for e in events)
    assert bot._console_candle_receive is None


@pytest.mark.asyncio
async def test_network_coalescing_preserves_retry_schedule_and_private_error_boundary(monkeypatch, caplog):
    from ccxt.base.errors import NetworkError
    events, delays = [], []
    monkeypatch.setattr('live.candle_ws_health.emit_event', lambda bot, event: events.append(event))
    async def watch(*args):
        raise NetworkError('private exchange request')
    bot = SimpleNamespace(ccp=SimpleNamespace(watch_ohlcv=watch),
        cm=SimpleNamespace(clear_live_ws_ohlcv_state=lambda symbol: None),
        get_exchange_time=lambda: 1_000_000, stop_signal_received=False)
    async def sleep(delay, **kwargs):
        delays.append(delay)
        if len(delays) == 5:
            bot.stop_signal_received = True
    bot._sleep_unless_shutdown = sleep
    with caplog.at_level(logging.INFO):
        await candle_ws.watch_forager_ws_symbol(bot, 'BTC/USDT:USDT')
    assert delays == [1., 2., 4., 8., 300.]
    assert sum(r.levelno == logging.WARNING for r in caplog.records) == 1
    assert [e['data']['receive_status'] for e in events] == ['failed'] * 5 + ['retired']
    assert 'private' not in caplog.text
    assert 'watchers cleared' in caplog.text


def test_shared_receive_details_reach_real_event_pipeline():
    from types import SimpleNamespace
    from ccxt.base.errors import NetworkError
    from live.event_bus import ListEventSink, LiveEventPipeline, EventTypes
    from live.candle_ws_health import observe_receive_status
    durable = ListEventSink()
    pipeline = LiveEventPipeline(structured_sinks=[durable])
    bot = SimpleNamespace(_live_event_pipeline=pipeline)
    try:
        observe_receive_status(bot, 'BTC', error=NetworkError('private'))
        observe_receive_status(bot, 'BTC', retired=True)
        assert pipeline.flush()
        assert len(durable.events) == 2
        assert all(e.event_type == EventTypes.CANDLE_WEBSOCKET_STATUS for e in durable.events)
        assert durable.events[-1].status == 'skipped'
        assert durable.events[-1].data['receive_status'] == 'retired'
        assert pipeline.health_snapshot()['event_sink_error_total'] == 0
    finally:
        pipeline.close()


@pytest.mark.asyncio
async def test_receive_recovery_precedes_failed_ingestion(monkeypatch, caplog):
    from ccxt.base.errors import NetworkError
    events, calls = [], []
    monkeypatch.setattr('live.candle_ws_health.emit_event', lambda bot, event: events.append(event))
    async def watch(*args):
        calls.append('receive')
        if len(calls) == 1:
            raise NetworkError('private')
        return []
    bot = SimpleNamespace(ccp=SimpleNamespace(watch_ohlcv=watch), stop_signal_received=False,
                          get_exchange_time=lambda: 1000000)
    async def ingest(*args):
        assert bot._console_candle_receive is None
        assert events[-1]['data']['receive_status'] == 'recovered'
        raise ValueError('invalid candle')
    bot.cm = SimpleNamespace(ingest_live_ws_ohlcv=ingest, clear_live_ws_ohlcv_state=lambda symbol: None)
    async def sleep(*args, **kwargs):
        if len(calls) == 2:
            bot.stop_signal_received = True
    bot._sleep_unless_shutdown = sleep
    with caplog.at_level(logging.INFO):
        await candle_ws.watch_forager_ws_symbol(bot, 'BTC')
    assert [event['data']['receive_status'] for event in events] == ['failed', 'recovered']
    assert 'stage=ingest error_type=ValueError' in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize('terminal', ['recovered', 'retired'])
async def test_overflow_watcher_terminal_event_survives_console_aggregate_clear(monkeypatch, terminal):
    from ccxt.base.errors import NetworkError
    from live.candle_ws_health import observe_receive_status
    events, calls = [], []
    monkeypatch.setattr('live.candle_ws_health.emit_event', lambda bot, event: events.append(event))
    bot = SimpleNamespace(stop_signal_received=False, get_exchange_time=lambda: 1000000)
    for i in range(256):
        observe_receive_status(bot, str(i), error=NetworkError('private'))
    async def watch(*args):
        calls.append('receive')
        if len(calls) == 1:
            raise NetworkError('private')
        bot.stop_signal_received = True
        return []
    bot.ccp = SimpleNamespace(watch_ohlcv=watch)
    bot.cm = SimpleNamespace(clear_live_ws_ohlcv_state=lambda symbol: None)
    async def sleep(*args, **kwargs):
        assert len(bot._console_candle_receive['active']) == 256
        for i in range(256):
            observe_receive_status(bot, str(i), recovered=True)
        assert bot._console_candle_receive is None
        if terminal == 'retired':
            bot.stop_signal_received = True
    bot._sleep_unless_shutdown = sleep
    await candle_ws.watch_forager_ws_symbol(bot, 'OVERFLOW')
    assert [e['data']['receive_status'] for e in events if e.get('symbol') == 'OVERFLOW'] == [
        'failed', terminal]
    assert bot._console_candle_receive is None
