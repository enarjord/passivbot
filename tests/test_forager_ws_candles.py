from __future__ import annotations

import asyncio
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
async def test_ingestion_failures_advance_to_rest_fallback_cooldown():
    delays = []

    class _SuccessfulWatcher:
        has = {"watchOHLCV": True}

        async def watch_ohlcv(self, _symbol, _timeframe):
            return [[ONE_MIN_MS, 1, 1, 1, 1, 1]]

    class _FailingManager:
        async def ingest_live_ws_ohlcv(self, _symbol, _rows):
            raise OSError("persistence unavailable")

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
