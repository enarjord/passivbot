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


def test_ws_overlay_accepts_only_finalized_rows_and_is_not_persisted(tmp_path):
    now_ms = 5 * ONE_MIN_MS + 10_000
    cm = _manager(tmp_path, now_ms=now_ms)
    symbol = "BTC/USDT:USDT"
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

    changed = cm.ingest_live_ws_ohlcv(
        symbol,
        [
            [4 * ONE_MIN_MS, 101, 103, 100, 102, 4],
            [5 * ONE_MIN_MS, 102, 104, 101, 103, 5],
        ],
        now_ms=now_ms,
    )

    assert changed == 1
    assert cm.get_last_live_ws_ohlcv_ts(symbol) == 4 * ONE_MIN_MS
    assert cm.get_last_final_ts(symbol) == 3 * ONE_MIN_MS
    disk = cm._load_from_disk(
        symbol,
        2 * ONE_MIN_MS,
        5 * ONE_MIN_MS,
        timeframe="1m",
    )
    assert list(disk["ts"]) == [2 * ONE_MIN_MS, 3 * ONE_MIN_MS]


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
    cm.ingest_live_ws_ohlcv(
        symbol,
        [[4 * ONE_MIN_MS, 101, 102, 101, 102, 3]],
        now_ms=now_ms,
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


def test_ws_reconnect_gap_and_silence_remain_missing(tmp_path):
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

    assert cm.ingest_live_ws_ohlcv(symbol, [], now_ms=now_ms) == 0
    cm.ingest_live_ws_ohlcv(
        symbol,
        [[3 * ONE_MIN_MS, 101, 101, 101, 101, 1]],
        now_ms=now_ms,
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
async def test_authoritative_rest_row_replaces_ws_overlay_and_ema(tmp_path):
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
    cm.ingest_live_ws_ohlcv(
        symbol,
        [[2 * ONE_MIN_MS, 100, 102, 99, 101, 2]],
        now_ms=now_ms,
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

    assert cm.get_last_live_ws_ohlcv_ts(symbol) == 0
    assert after["close"] == pytest.approx(102.0)


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
            "short": set(),
        },
        is_forager_mode=lambda: True,
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


def test_ws_tail_rest_audit_is_periodic_and_one_minute_only():
    bot = SimpleNamespace(
        config={"live": {"forager_ws_candle_rest_audit_minutes": 30}}
    )
    now_ms = 100 * ONE_MIN_MS
    health = {
        "ws_overlay_contributed_to_tail": True,
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


def test_forager_staleness_uses_ws_tail_without_changing_rest_index():
    bot = SimpleNamespace(
        cm=SimpleNamespace(
            get_last_final_ts=lambda _symbol: 10 * ONE_MIN_MS,
            get_last_live_ws_ohlcv_ts=lambda _symbol: 12 * ONE_MIN_MS,
            get_last_refresh_ms=lambda _symbol: 9 * ONE_MIN_MS,
        )
    )
    now_ms = 13 * ONE_MIN_MS + 10_000

    assert Passivbot._candle_staleness_ms(bot, "BTC", now_ms=now_ms) == 0
