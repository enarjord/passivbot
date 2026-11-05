"""Offline candidate repair remains live when ordinary execution cannot be admitted."""
import asyncio
import math
from types import SimpleNamespace

import pytest

import passivbot as pb
from candlestick_manager import CandlestickManager, ONE_MIN_MS
from live.hsl_live import Owner


@pytest.mark.asyncio
async def test_candidate_native_surfaces_recover_without_admitted_ordinary_plan(tmp_path, monkeypatch):
    now = 1_800_000_000_000
    symbols = ['A/USDT:USDT', 'B/USDT:USDT']
    monkeypatch.setattr(pb, 'utc_ms', lambda: now)
    monkeypatch.setattr(pb, 'compute_live_warmup_windows', lambda *a, **k: (
        {symbol: 6 for symbol in symbols}, {symbol: 4 for symbol in symbols}, {}
    ))
    monkeypatch.setattr(pb.Passivbot, '_live_required_h1_log_range_span', lambda *a: 2.0)

    class PublicFixture:
        id = 'fixture'

        def __init__(self):
            self.calls = []

        async def fetch_ohlcv(self, symbol, timeframe='1m', since=None, limit=None, params=None):
            period = ONE_MIN_MS if timeframe == '1m' else 60 * ONE_MIN_MS
            self.calls.append((symbol, timeframe))
            end = now // period * period - period
            return [[ts, 10.0, 11.0, 9.0, 10.0, 2.0]
                    for ts in range(int(since), end + 1, period)][:limit]

    exchange = PublicFixture()
    bot = pb.Passivbot.__new__(pb.Passivbot)
    bot.config = {'live': {'candles': {'max_ohlcv_fetches_per_minute': 8}},
                  'bot': {'portfolio': {'warmup': {'max_minutes': None}}}}
    bot.stop_signal_received = bot._shutdown_in_progress = False
    bot.ws_enabled = False
    bot.monitor_publisher = None
    bot.positions = bot.open_orders = {}
    bot.PB_modes = {'long': {symbol: 'graceful_stop' for symbol in symbols}, 'short': {}}
    bot.approved_coins_minus_ignored_coins = {'long': set(symbols), 'short': set()}
    bot.start_time_ms = now
    bot.cm = CandlestickManager(exchange=exchange, exchange_name='fixture',
                                cache_dir=str(tmp_path), default_window_candles=6)
    bot.cm._now_ms_callback = lambda: now
    bot.is_forager_mode = lambda side=None: side in (None, 'long')
    bot.get_max_n_positions = lambda side: 1 if side == 'long' else 0
    bot.get_current_n_positions = lambda side: 0
    bot.bp = lambda *args: 0.0
    bot._get_fetch_delay_seconds = lambda: 0.0
    bot._forager_target_staleness_ms = lambda *args: 0
    bot._schedule_first_timestamp_refresh = lambda: None
    blocker = asyncio.Event()
    bot.maintain_hourly_cycle = blocker.wait

    # The real HSL preparation owner rejects every ordinary attempt before
    # execute_order_plan can run its historical candidate-refresh hook.
    ordinary = Owner(bot)
    fact = SimpleNamespace(payload='unchanged', generation=1)
    ordinary.account_facts = lambda: (fact, (), 100.0, None)
    bot.prepare_planning_universe = lambda: asyncio.sleep(0)

    async def ready():
        return True

    bot.refresh_market_state_if_needed = ready
    bot._staged_execution_ready_state = lambda **kwargs: (False, {})

    async def forbidden_execution(*args, **kwargs):
        pytest.fail('unavailable ordinary work must not execute exchange writes')

    bot.execute_order_plan_to_exchange = forbidden_execution
    passes = 0

    async def interval(seconds, *, stage):
        nonlocal passes, now
        assert stage == 'forager_candidate_refresh_interval'
        assert seconds == 15.0
        await asyncio.wait_for(bot.maintainers['forager_candidate_candle_refresh'], 2.0)
        assert await ordinary._ordinary_plan() is None
        if passes == 0:
            assert math.isnan(await bot.cm.get_latest_ema_log_range(
                symbols[0], 2.0, timeframe='1h', allow_remote_fetch=False
            ))
            assert bot.approved_coins_minus_ignored_coins['long'] == set(symbols)
        passes += 1
        now += 15_000
        if passes == 2:
            bot.stop_signal_received = True

    bot._sleep_unless_shutdown = interval
    before = await bot.cm.get_latest_ema_log_range(symbols[0], 2.0, allow_remote_fetch=False)
    assert math.isnan(before)
    tasks = []
    try:
        await bot.start_data_maintainers()
        tasks = list(bot.maintainers.values())
        assert 'maintain_forager_candidate_candles' in bot.maintainers
        await asyncio.wait_for(bot.maintainers['maintain_forager_candidate_candles'], 5.0)
        assert passes == 2
        assert set(exchange.calls) == {(symbol, tf) for symbol in symbols for tf in ('1m', '1h')}
        # Every member survives the acquisition cohort. Strict completed-candle
        # features become available only after real source rows cover the window.
        for symbol in symbols:
            for tf in ('1m', '1h'):
                value = await bot.cm.get_latest_ema_log_range(
                    symbol, 2.0, timeframe=tf, allow_remote_fetch=False,
                    allow_provisional_internal_gaps=False,
                )
                assert math.isfinite(value) and value > 0.0
    finally:
        bot.stop_signal_received = True
        for task in set(tasks + list(getattr(bot, 'maintainers', {}).values())):
            task.cancel()
        await asyncio.gather(*set(tasks + list(getattr(bot, 'maintainers', {}).values())), return_exceptions=True)
        bot.cm.close()


@pytest.mark.asyncio
async def test_candidate_maintainer_does_not_spawn_when_tokens_are_exhausted():
    bot = pb.Passivbot.__new__(pb.Passivbot)
    bot.config = {'live': {'candles': {'max_ohlcv_fetches_per_minute': 8}}}
    bot.maintainers = {}
    bot.stop_signal_received = bot._shutdown_in_progress = False
    bot.is_forager_mode = lambda: True
    peeks = []

    def exhausted(*args, **kwargs):
        peeks.append(kwargs)
        return 0

    bot._forager_refresh_budget = exhausted
    bot._schedule_forager_candidate_candle_refresh = lambda: pytest.fail('empty bucket spawned work')

    async def interval(*args, **kwargs):
        bot.stop_signal_received = True

    bot._sleep_unless_shutdown = interval
    await bot.maintain_forager_candidate_candles()
    assert len(peeks) == 1 and peeks[0]['consume'] is False
    assert bot.maintainers == {}


@pytest.mark.asyncio
async def test_candidate_maintainer_cancellation_drains_only_its_registry_worker():
    bot = pb.Passivbot.__new__(pb.Passivbot)
    bot.config = {'live': {'candles': {'max_ohlcv_fetches_per_minute': 8}}}
    registry = bot.maintainers = {}
    bot.stop_signal_received = bot._shutdown_in_progress = False
    bot.is_forager_mode = lambda side=None: True
    bot._forager_refresh_budget = lambda *args, **kwargs: 1
    started, drained = asyncio.Event(), asyncio.Event()
    attempts = 0

    async def refresh():
        nonlocal attempts
        attempts += 1
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            drained.set()

    async def interval(*args, **kwargs):
        await asyncio.Event().wait()

    bot._refresh_forager_candidate_candles = refresh
    bot._sleep_unless_shutdown = interval
    maintainer = asyncio.create_task(bot.maintain_forager_candidate_candles())
    replacement = None
    try:
        await asyncio.wait_for(started.wait(), 1.0)
        original = registry['forager_candidate_candle_refresh']
        bot._schedule_forager_candidate_candle_refresh()
        assert registry['forager_candidate_candle_refresh'] is original and attempts == 1
        # Restart may replace the registry before the canceled old owner runs
        # its finally block. It must not cancel the replacement owner's task.
        replacement = asyncio.create_task(asyncio.Event().wait())
        bot.maintainers = {'forager_candidate_candle_refresh': replacement}
        maintainer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(maintainer, 1.0)
        assert drained.is_set() and original.done()
        assert not replacement.done()
    finally:
        maintainer.cancel()
        if replacement is not None:
            replacement.cancel()
        await asyncio.gather(maintainer, *([replacement] if replacement else []), return_exceptions=True)
