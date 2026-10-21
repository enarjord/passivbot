"""Slow admission facts cannot hold up management of existing positions."""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import passivbot as live
from config.schema import get_template_config


FIRST = 1_704_067_200_000
DAY = 86_400_000
KEY = "first_timestamp_refresh"
FLAT = "FLAT/USDT:USDT"
HELD = "HELD/USDT:USDT"


def _bot(side="long", exchange="binance"):
    bot = object.__new__(live.Passivbot)
    bot.config = get_template_config()
    for pside in ("long", "short"):
        bot.config["bot"][pside]["risk"].update(
            max_positions=1, total_wallet_exposure_limit=1.0 if side == pside else 0.0,
        )
        bot.config["bot"][pside]["universe"]["minimum_coin_age_days"] = 7.25
    bot.exchange = exchange
    bot.stop_signal_received = False
    bot.first_timestamps = {HELD: FIRST}
    bot.approved_coins_minus_ignored_coins = {
        pside: {FLAT} if pside == side else set() for pside in ("long", "short")
    }
    bot.positions = {HELD: {
        pside: {"size": 1.0 if pside == side else 0.0, "price": 1.0}
        for pside in ("long", "short")
    }}
    bot.open_orders = {}
    bot.coin_overrides = {}
    bot.markets_dict = {FLAT: {}, HELD: {}}
    bot.coin_to_symbol = lambda value, **_kwargs: value
    bot.get_exchange_time = lambda: FIRST + 10 * DAY
    bot.get_forced_PB_mode = lambda _side: ""
    bot.is_approved = lambda pside, symbol: (
        symbol in bot.approved_coins_minus_ignored_coins[pside]
        and bot.is_old_enough(pside, symbol)
    )
    bot.maintainers = {}
    bot.WS_ohlcvs_1m_tasks = {}
    bot.update_effective_min_cost = AsyncMock()
    bot.refresh_approved_ignored_coins_lists = lambda: None
    bot.set_wallet_exposure_limits = lambda: None
    bot._assert_supported_live_state = lambda: None
    bot.refresh_authoritative_state = AsyncMock(return_value=True)
    bot.refresh_market_state_if_needed = AsyncMock(return_value=True)
    return bot


async def _stop(bot):
    bot.stop_signal_received = True
    bot.stop_data_maintainers(verbose=False)
    await asyncio.gather(*bot.maintainers.values(), return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("exchange", ["binance", "fake"])
async def test_slow_flat_inception_does_not_delay_held_planning_and_protection(
    monkeypatch, side, exchange,
):
    bot = _bot(side, exchange)
    started, release = asyncio.Event(), asyncio.Event()

    async def slow_lookup(symbols, **_kwargs):
        assert symbols == [FLAT]
        started.set()
        await release.wait()
        return {FLAT: FIRST}

    async def slow_candles(symbol, **_kwargs):
        assert symbol == FLAT
        started.set()
        await release.wait()
        return [[FIRST, 1, 1, 1, 1, 1]]

    lookup = AsyncMock(side_effect=slow_lookup)
    monkeypatch.setattr(live, "get_first_timestamps_unified", lookup)
    bot.cca = SimpleNamespace(fetch_ohlcv=AsyncMock(side_effect=slow_candles))
    protective_plan = AsyncMock()

    async def held_cycle():
        if await bot.update_pos_oos_pnls_ohlcvs():
            await protective_plan(bot.active_symbols)

    try:
        await asyncio.wait_for(held_cycle(), timeout=0.2)
        await asyncio.wait_for(started.wait(), timeout=1.0)
        protective_plan.assert_awaited_once_with([HELD])
        bot.refresh_market_state_if_needed.assert_awaited_once()
        assert not bot.is_old_enough(side, FLAT)
        task = bot.maintainers[KEY]
        assert not task.done()

        for _ in range(5):
            await asyncio.wait_for(bot.prepare_planning_universe(), timeout=0.2)
            assert bot.maintainers[KEY] is task
            assert bot.active_symbols == [HELD]
        assert (bot.cca.fetch_ohlcv.await_count if exchange == "fake" else lookup.await_count) == 1

        release.set()
        await asyncio.wait_for(task, timeout=1.0)
        await bot.prepare_planning_universe()
        assert bot.is_old_enough(side, FLAT)
        assert bot.active_symbols == [FLAT, HELD]
        assert bot.first_timestamps[HELD] == FIRST
    finally:
        release.set()
        await _stop(bot)


@pytest.mark.asyncio
async def test_worker_timeout_retries_after_backoff_without_blocking_planning(monkeypatch, caplog):
    bot = _bot()
    started = asyncio.Event()

    async def stalled_lookup(*_args, **_kwargs):
        started.set()
        await asyncio.Event().wait()

    lookup = AsyncMock(side_effect=stalled_lookup)
    monkeypatch.setattr(live, "get_first_timestamps_unified", lookup)
    monkeypatch.setattr(live, "FIRST_TIMESTAMP_REFRESH_TIMEOUT_SECONDS", 0.01)
    try:
        with caplog.at_level(logging.WARNING):
            await bot.prepare_planning_universe()
            await asyncio.wait_for(started.wait(), timeout=1.0)
            first = bot.maintainers[KEY]
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(asyncio.shield(first), timeout=1.0)
        assert "inception acquisition timed out" in caplog.text
        assert not bot.is_old_enough("long", FLAT)
        await bot.prepare_planning_universe()
        assert bot.maintainers[KEY] is first
        assert lookup.await_count == 1

        bot._first_timestamp_refresh_retry_after = 0.0
        lookup.side_effect = None
        lookup.return_value = {FLAT: FIRST}
        await bot.prepare_planning_universe()
        await asyncio.wait_for(bot.maintainers[KEY], timeout=1.0)
        assert lookup.await_count == 2
        assert bot.is_old_enough("long", FLAT)
    finally:
        await _stop(bot)


@pytest.mark.asyncio
async def test_failed_lookup_is_harvested_and_retry_is_bounded(monkeypatch, caplog):
    bot = _bot()
    secret = "token=private-age-fact https://private.invalid/inception"
    lookup = AsyncMock(side_effect=live.NetworkError(secret))
    monkeypatch.setattr(live, "get_first_timestamps_unified", lookup)
    try:
        with caplog.at_level(logging.WARNING):
            await bot.prepare_planning_universe()
            first = bot.maintainers[KEY]
            await first
            await asyncio.sleep(0)
        assert first.exception() is None
        assert "inception acquisition failed" in caplog.text
        assert secret not in caplog.text
        assert not bot.is_old_enough("long", FLAT)
        for _ in range(5):
            await bot.prepare_planning_universe()
        assert bot.maintainers[KEY] is first
        assert lookup.await_count == 1
        bot._first_timestamp_refresh_retry_after = 0.0
        lookup.side_effect = None
        lookup.return_value = {FLAT: FIRST}
        await bot.prepare_planning_universe()
        await bot.maintainers[KEY]
        assert bot.is_old_enough("long", FLAT)
    finally:
        await _stop(bot)


@pytest.mark.asyncio
async def test_resistant_worker_remains_owned_through_replacement_and_restart(monkeypatch, caplog):
    bot = _bot()
    started, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def resistant_lookup(*_args, **_kwargs):
        started.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
        return {FLAT: FIRST}

    lookup = AsyncMock(side_effect=resistant_lookup)
    monkeypatch.setattr(live, "get_first_timestamps_unified", lookup)
    monkeypatch.setattr(live, "FIRST_TIMESTAMP_REFRESH_TIMEOUT_SECONDS", 0.01)
    bot.ws_enabled = False
    bot.maintain_hourly_cycle = AsyncMock()
    bot.ccp = SimpleNamespace(close=AsyncMock())
    bot.cca = SimpleNamespace(close=AsyncMock())
    bot.monitor_publisher = None
    bot._close_live_event_pipeline = lambda **_kwargs: True
    try:
        await bot.prepare_planning_universe()
        await asyncio.wait_for(started.wait(), timeout=1.0)
        task = bot.maintainers[KEY]
        await asyncio.wait_for(cancelled.wait(), timeout=1.0)
        for _ in range(5):
            await bot.prepare_planning_universe()
        assert bot.maintainers[KEY] is task
        assert lookup.await_count == 1
        assert not bot.is_old_enough("long", FLAT)

        await bot.start_data_maintainers()
        assert bot.maintainers[KEY] is task
        assert lookup.await_count == 1
        with caplog.at_level(logging.WARNING):
            await asyncio.wait_for(bot.cleanup_for_restart(timeout_seconds=0.0), timeout=1.5)
        assert "maintainer cancellation grace expired" in caplog.text
        assert bot.maintainers[KEY] is task
        assert not task.done()
        assert bot.cca is None and bot.ccp is None
    finally:
        release.set()
        await _stop(bot)


@pytest.mark.asyncio
async def test_resistant_expired_batch_cannot_dispatch_a_second_batch(monkeypatch):
    bot = _bot()
    first, second = "AAA/USDT:USDT", "BBB/USDT:USDT"
    bot.approved_coins_minus_ignored_coins["long"] = {first, second}
    bot.markets_dict.update({first: {}, second: {}})
    cancelled, release = asyncio.Event(), asyncio.Event()

    async def resistant_lookup(symbols, **_kwargs):
        assert symbols == [first]
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
        return {first: FIRST}

    lookup = AsyncMock(side_effect=resistant_lookup)
    monkeypatch.setattr(live, "get_first_timestamps_unified", lookup)
    monkeypatch.setattr(live, "FIRST_TIMESTAMP_REFRESH_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(live, "FIRST_TIMESTAMP_REFRESH_BATCH_SIZE", 1)
    try:
        await bot.prepare_planning_universe()
        task = bot.maintainers[KEY]
        await asyncio.wait_for(cancelled.wait(), timeout=1.0)
        assert not task.done()
        release.set()
        await asyncio.wait_for(task, timeout=1.0)
        lookup.assert_awaited_once_with([first], exchange="binance")
        assert bot.first_timestamps[first] == FIRST
        assert not bot.is_old_enough("long", second)
    finally:
        release.set()
        await _stop(bot)


@pytest.mark.asyncio
async def test_startup_account_configuration_defers_inception_to_tracked_maintainer(monkeypatch):
    bot = _bot()
    lookup = AsyncMock(side_effect=AssertionError("startup account preparation must not await inception"))
    monkeypatch.setattr(live, "get_first_timestamps_unified", lookup)
    bot._exchange_config_write_ready = AsyncMock(return_value=True)
    bot.update_exchange_config = AsyncMock()
    bot.set_market_specific_settings = lambda: None
    bot.is_forager_mode = lambda *_args: True
    await asyncio.wait_for(
        bot._init_markets_account_config(verbose=False, metadata_loaded=True), timeout=0.2,
    )
    lookup.assert_not_awaited()
    assert KEY not in bot.maintainers
    bot.ws_enabled = False
    bot.maintain_hourly_cycle = AsyncMock()
    lookup.side_effect = None
    lookup.return_value = {FLAT: FIRST}
    try:
        await bot.start_data_maintainers()
        assert KEY in bot.maintainers
        await bot.maintainers[KEY]
        lookup.assert_awaited_once_with([FLAT], exchange="binance")
    finally:
        await _stop(bot)


@pytest.mark.asyncio
async def test_stopped_bot_never_starts_inception_work(monkeypatch):
    bot = _bot()
    bot.stop_signal_received = True
    lookup = AsyncMock(side_effect=AssertionError("stopped bot must not acquire facts"))
    monkeypatch.setattr(live, "get_first_timestamps_unified", lookup)
    bot._schedule_first_timestamp_refresh()
    assert KEY not in bot.maintainers
    lookup.assert_not_awaited()
