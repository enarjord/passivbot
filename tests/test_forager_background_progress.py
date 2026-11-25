"""Offline candidate repair remains live when ordinary execution cannot be admitted."""
import asyncio
import math
from types import SimpleNamespace

import numpy as np
import pytest

import passivbot as pb
from candlestick_manager import CANDLE_DTYPE, CandlestickManager, ONE_MIN_MS
from config.schema import get_expanded_config_prototype
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
    bot.config = get_expanded_config_prototype()
    bot.config['live']['candles']['max_ohlcv_fetches_per_minute'] = 8
    bot.config['bot']['portfolio']['warmup']['max_minutes'] = None
    bot.config['bot']['portfolio']['risk']['hsl']['scope'] = 'coin'
    for side in ('long', 'short'):
        bot.config['bot'][side]['risk']['hsl']['enabled'] = False
    bot.coin_overrides = {}
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

    async def ready(*, trailing_acquisition_only):
        assert trailing_acquisition_only is True
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


@pytest.mark.asyncio
@pytest.mark.parametrize("current_span_longer", [False, True])
async def test_candidate_rms_repairs_history_in_pages_with_fresh_ws_tail_and_restart(
    tmp_path, monkeypatch, current_span_longer
):
    from live.rms_directionality import read_required

    now = 1_800_000_000_000
    symbol = "APT/USDT:USDT"
    span = 65.0
    adverse_span = 70.0 if current_span_longer else span
    count = CandlestickManager.rms_history_candle_count(adverse_span)
    end = now - ONE_MIN_MS
    start = end - (count - 1) * ONE_MIN_MS
    gap_start = start + 30 * ONE_MIN_MS
    gap_end = gap_start + 1169 * ONE_MIN_MS

    def row(ts):
        price = 10 + (ts - start) / ONE_MIN_MS * 0.01
        return [ts, price, price, price, price, 2.0]

    class PublicFixture:
        id = "fixture"

        def __init__(self):
            self.calls = []

        async def fetch_ohlcv(self, symbol, timeframe="1m", since=None, limit=None, params=None):
            self.calls.append((since, limit))
            return [row(ts) for ts in range(int(since), end + 1, ONE_MIN_MS)][:limit]

    exchange = PublicFixture()
    monkeypatch.setattr(pb, "utc_ms", lambda: now)
    monkeypatch.setattr(pb, "compute_live_warmup_windows", lambda *a, **k: ({symbol: 6}, {}, {}))

    def make_bot():
        bot = pb.Passivbot.__new__(pb.Passivbot)
        bot.config = get_expanded_config_prototype()
        bot.config["live"]["candles"].update(max_ohlcv_fetches_per_minute=4,
                                             max_forager_candle_refresh_seconds=0)
        bot.config["bot"]["long"]["forager"]["score_weights"]["rms_directionality"] = 1.0
        bot.cm = CandlestickManager(exchange=exchange, exchange_name="fixture",
                                    cache_dir=str(tmp_path), default_window_candles=6)
        bot.cm._now_ms_callback = lambda: now
        bot.stop_signal_received = bot._shutdown_in_progress = False
        bot.approved_coins_minus_ignored_coins = {"long": {symbol}, "short": set()}
        bot.positions = bot.open_orders = {}
        bot.is_forager_mode = lambda side=None: side in (None, "long")
        bot.is_pside_enabled = lambda side: side == "long" or current_span_longer
        bot.is_approved = lambda side, coin: bot.is_pside_enabled(side)
        bot.get_max_n_positions = lambda side: 1
        bot.get_current_n_positions = lambda side: 0
        bot.get_exchange_time = lambda: now
        bot.bot_value = lambda side, key: (
            {"rms_directionality": float(side == "long")}
            if key == "forager_score_weights"
            else span if side == "long" else adverse_span
        )
        cooldown = {
            "entry_cooldown_weights_minutes": {"adverse_directionality": 1.0},
            "risk_entry_cooldown_minutes": 0.0,
            "entry_cooldown_min_duration_minutes": 0.0,
            "entry_cooldown_max_duration_minutes": 60.0,
        }
        bot.bp = lambda side, key, coin: (
            {"adverse_directionality": 0.0}
            if side == "long" and key == "entry_cooldown_weights_minutes"
            else cooldown.get(key, 0.0)
        )
        bot._get_fetch_delay_seconds = lambda: 0.0
        def target_staleness(surface_count, max_calls):
            # One ordinary 1m surface sets the freshness target. The separate
            # RMS page below still needs to run and consume the fetch token.
            assert (surface_count, max_calls) == (1, 4)
            return 0

        bot._forager_target_staleness_ms = target_staleness
        bot._forager_refresh_budget = lambda *a, **k: 1
        return bot

    bot = make_bot()
    try:
        data = [row(ts) for ts in range(start, end, ONE_MIN_MS) if not gap_start <= ts <= gap_end]
        bot.cm._persist_batch(symbol, np.array([tuple(r) for r in data], dtype=CANDLE_DTYPE), last_refresh_ms=now)
        bot.cm._persist_batch(symbol, np.array([tuple(row(end))], dtype=CANDLE_DTYPE), source="ws")
        assert not (await read_required(bot, [symbol], {symbol}))[1][symbol]

        # Scanning can discover an unsatisfied history job before the worker
        # can reserve a fetch token. Ranking is still deferred in that case.
        bot._forager_refresh_budget = lambda *a, consume, **k: int(not consume)
        await bot._refresh_forager_candidate_candles()
        assert not exchange.calls
        assert bot._forager_startup_ready_sides == set()

        bot._forager_refresh_budget = lambda *a, **k: 1
        await bot._refresh_forager_candidate_candles()
        assert bot._forager_startup_ready_sides == {"long"}
        health = bot.cm.get_completed_candle_health(symbol, {"1m": count}, now_ms=now)["timeframes"]["1m"]
        assert health["missing_spans"] == [(gap_start, gap_end - 1000 * ONE_MIN_MS)]
        assert exchange.calls == [(gap_end - 999 * ONE_MIN_MS, 1000)]
        assert bot.cm.get_last_refresh_ms(symbol) == now
        assert not (await read_required(bot, [symbol], {symbol}))[1][symbol]
    finally:
        bot.cm.close()

    restarted = make_bot()
    try:
        await restarted._refresh_forager_candidate_candles()
        before = len(exchange.calls)
        assert exchange.calls[-1] == (gap_start, 1000)
        assert before == 2
        current, ranking, missing = await read_required(restarted, [symbol], {symbol})
        assert set(current[symbol]) == ({adverse_span} if current_span_longer else set())
        assert ranking[symbol][span] > 0.99
        assert not missing
        await restarted._refresh_forager_candidate_candles()
        assert len(exchange.calls) == before
    finally:
        restarted.cm.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gap_reason", ["no_trades", "fetch_failed", "proof_after_unknown", "verified_tail"]
)
async def test_rms_uses_only_contiguous_verified_no_trade_minutes(tmp_path, gap_reason):
    import passivbot_rust as pbr

    symbol = "APT/USDT:USDT"
    span = 1.0
    base = 1_800_000_000_000
    end = base + 20 * ONE_MIN_MS
    prices = [100.0 + i for i in range(21)]
    missing_start = 19 if gap_reason == "verified_tail" else 8
    prices[missing_start:missing_start + 2] = [prices[missing_start - 1]] * 2
    data = np.array(
        [
            (base + i * ONE_MIN_MS, price, price, price, price, 1.0)
            for i, price in enumerate(prices)
            if i not in (missing_start, missing_start + 1)
        ],
        dtype=CANDLE_DTYPE,
    )
    with CandlestickManager(exchange_name="fixture", cache_dir=str(tmp_path)) as manager:
        manager._now_ms_callback = lambda: end + ONE_MIN_MS
        manager._persist_batch(symbol, data, timeframe="1m")
        gap_start = base + missing_start * ONE_MIN_MS
        gap_end = gap_start + ONE_MIN_MS
        if gap_reason in ("no_trades", "verified_tail"):
            manager._record_verified_gap(symbol, gap_start, gap_end)
        elif gap_reason == "proof_after_unknown":
            manager._record_verified_gap(symbol, gap_end, gap_end)
        else:
            manager._add_known_gap(symbol, gap_start, gap_end, reason=gap_reason)
        scores = await manager.get_rms_emas(
            symbol, {"forager": {span}, "current": {span}},
            end_ts=end, forager_max_age_ms=0, allow_remote_fetch=False,
        )
        if gap_reason in ("no_trades", "verified_tail"):
            expected = (pbr.calc_signed_rms_directionality(prices, span), end)
            assert scores == {"forager": {span: expected}, "current": {span: expected}}
            raw = await manager.get_candles(
                symbol, start_ts=base, end_ts=end, timeframe="1m",
                allow_remote_fetch=False, standardize=False,
            )
            assert len(raw) == len(data)
        else:
            assert scores == {"forager": {}, "current": {}}


@pytest.mark.asyncio
async def test_rms_ranking_only_memo_cannot_satisfy_current_at_same_cutoff(tmp_path, monkeypatch):
    symbol = "APT/USDT:USDT"
    base = 1_800_000_000_000
    end = base + 20 * ONE_MIN_MS
    data = np.array(
        [(base + i * ONE_MIN_MS, 100 + i, 100 + i, 100 + i, 100 + i, 1.0) for i in range(21)],
        dtype=CANDLE_DTYPE,
    )
    read_options = {"end_ts": end, "forager_max_age_ms": ONE_MIN_MS, "allow_remote_fetch": False}
    with CandlestickManager(exchange_name="fixture", cache_dir=str(tmp_path)) as manager:
        manager._now_ms_callback = lambda: end + ONE_MIN_MS
        manager._persist_batch(symbol, data, timeframe="1m")
        ranking = await manager.get_rms_emas(symbol, {"forager": {1.0}}, **read_options)
        assert ranking["forager"][1.0][1] == end

        async def incomplete_read(*args, **kwargs):
            return data[:-1]

        monkeypatch.setattr(manager, "get_candles", incomplete_read)
        current = await manager.get_rms_emas(symbol, {"current": {1.0}}, **read_options)
        assert current == {"current": {}}
        assert manager.cached_rms_ema(symbol, "forager", 1.0) == ranking["forager"][1.0]


@pytest.mark.asyncio
async def test_rms_late_arrival_displaces_stale_ranking_memo_at_same_cutoff(tmp_path):
    symbol = "APT/USDT:USDT"
    base = 1_800_000_000_000
    end = base + 21 * ONE_MIN_MS
    data = np.array(
        [(base + i * ONE_MIN_MS, 100 + i, 100 + i, 100 + i, 100 + i, 1.0) for i in range(21)],
        dtype=CANDLE_DTYPE,
    )
    late = np.array([(end, 115.0, 115.0, 115.0, 115.0, 1.0)], dtype=CANDLE_DTYPE)
    read_options = {"end_ts": end, "forager_max_age_ms": ONE_MIN_MS, "allow_remote_fetch": False}
    with CandlestickManager(exchange_name="fixture", cache_dir=str(tmp_path)) as manager:
        manager._now_ms_callback = lambda: end + ONE_MIN_MS
        manager._persist_batch(symbol, data, timeframe="1m")
        before = await manager.get_rms_emas(symbol, {"forager": {1.0}}, **read_options)
        assert before["forager"][1.0][1] == end - ONE_MIN_MS
        manager._persist_batch(symbol, late, timeframe="1m")
        after = await manager.get_rms_emas(symbol, {"forager": {1.0}}, **read_options)
        assert after["forager"][1.0][1] == end
        assert after["forager"][1.0][0] != before["forager"][1.0][0]

    with CandlestickManager(exchange_name="fixture", cache_dir=str(tmp_path)) as restarted:
        restarted._now_ms_callback = lambda: end + ONE_MIN_MS
        assert await restarted.get_rms_emas(symbol, {"forager": {1.0}}, **read_options) == after


@pytest.mark.asyncio
async def test_rms_cache_only_reuses_unchanged_window_until_age_or_new_candles(tmp_path, monkeypatch):
    symbol = "APT/USDT:USDT"
    base = 1_800_000_000_000
    end = base + 20 * ONE_MIN_MS
    data = np.array(
        [(base + i * ONE_MIN_MS, 100 + i, 100 + i, 100 + i, 100 + i, 1.0) for i in range(21)],
        dtype=CANDLE_DTYPE,
    )
    clock = [end + ONE_MIN_MS]
    with CandlestickManager(exchange_name="fixture", cache_dir=str(tmp_path)) as manager:
        manager._now_ms_callback = lambda: clock[0]
        manager._persist_batch(symbol, data, timeframe="1m")
        read = manager.get_candles
        reads = []

        async def counted_read(*args, **kwargs):
            reads.append(kwargs["end_ts"])
            return await read(*args, **kwargs)

        monkeypatch.setattr(manager, "get_candles", counted_read)

        async def score(cutoff):
            clock[0] = cutoff + ONE_MIN_MS
            return await manager.get_rms_emas(
                symbol, {"forager": {1.0}}, end_ts=cutoff,
                forager_max_age_ms=2 * ONE_MIN_MS, allow_remote_fetch=False,
            )

        initial = await score(end)
        assert await score(end + ONE_MIN_MS) == initial
        assert await score(end + 2 * ONE_MIN_MS) == initial
        assert reads == [end]
        assert await score(end + 3 * ONE_MIN_MS) == {"forager": {}}
        assert reads == [end, end + 3 * ONE_MIN_MS]

        # The same index is shared by independent owners. Its new finalized
        # row must be considered even though this manager has a memo in RAM.
        next_ts = end + ONE_MIN_MS
        late = np.array([(next_ts, 115, 115, 115, 115, 1.0)], dtype=CANDLE_DTYPE)
        with CandlestickManager(exchange_name="fixture", cache_dir=str(tmp_path)) as writer:
            writer._now_ms_callback = lambda: clock[0]
            writer._persist_batch(symbol, late, timeframe="1m")
        refreshed = await score(end + 2 * ONE_MIN_MS)
        assert refreshed["forager"][1.0][1] == next_ts
        assert refreshed["forager"][1.0][0] != initial["forager"][1.0][0]


@pytest.mark.asyncio
async def test_rms_cache_only_reconsiders_verified_proof_beyond_prior_cutoff(tmp_path):
    symbol = "APT/USDT:USDT"
    base = 1_800_000_000_000
    end = base + 20 * ONE_MIN_MS
    data = np.array(
        [(base + i * ONE_MIN_MS, 100 + i, 100 + i, 100 + i, 100 + i, 1.0) for i in range(21)],
        dtype=CANDLE_DTYPE,
    )
    next_ts = end + ONE_MIN_MS
    with CandlestickManager(exchange_name="fixture", cache_dir=str(tmp_path)) as manager:
        manager._now_ms_callback = lambda: next_ts + ONE_MIN_MS
        manager._persist_batch(symbol, data, timeframe="1m")
        manager._record_verified_gap(symbol, next_ts, next_ts)
        initial = await manager.get_rms_emas(
            symbol, {"forager": {1.0}}, end_ts=end,
            forager_max_age_ms=ONE_MIN_MS, allow_remote_fetch=False,
        )
        assert initial["forager"][1.0][1] == end
        updated = await manager.get_rms_emas(
            symbol, {"forager": {1.0}}, end_ts=next_ts,
            forager_max_age_ms=ONE_MIN_MS, allow_remote_fetch=False,
        )
        assert updated["forager"][1.0][1] == next_ts
        manager.clear_known_gaps(symbol, date_range=(next_ts, next_ts))
        after_clear = await manager.get_rms_emas(
            symbol, {"forager": {1.0}, "current": {1.0}},
            end_ts=next_ts, forager_max_age_ms=ONE_MIN_MS, allow_remote_fetch=False,
        )
        assert after_clear == {"forager": initial["forager"], "current": {}}


@pytest.mark.asyncio
async def test_rms_history_repair_fetches_missing_seed_outside_verified_window(tmp_path, monkeypatch):
    import passivbot_rust as pbr

    symbol = "APT/USDT:USDT"
    base = 1_800_000_000_000
    end = base + 20 * ONE_MIN_MS
    proof_start = base - ONE_MIN_MS
    seed_ts = proof_start - ONE_MIN_MS
    prices = [99.0] * 2 + [100.0 + i for i in range(2, 21)]
    candles = np.array(
        [
            (base + i * ONE_MIN_MS, price, price, price, price, 1.0)
            for i, price in enumerate(prices)
            if i >= 2
        ],
        dtype=CANDLE_DTYPE,
    )
    seed = np.array([(seed_ts, 99, 99, 99, 99, 1.0)], dtype=CANDLE_DTYPE)
    with CandlestickManager(exchange_name="fixture", cache_dir=str(tmp_path)) as manager:
        manager._now_ms_callback = lambda: end + ONE_MIN_MS
        manager._persist_batch(symbol, candles, timeframe="1m")
        # The proof crosses the requested window start. The only priced
        # predecessor is two minutes before that start.
        manager._record_verified_gap(symbol, proof_start, base + ONE_MIN_MS)
        assert manager.plan_rms_history_repair(symbol, 1.0, now_ms=end + ONE_MIN_MS) == (
            seed_ts, seed_ts,
        )
        assert await manager.get_rms_emas(
            symbol, {"current": {1.0}}, end_ts=end,
            forager_max_age_ms=0, allow_remote_fetch=False,
        ) == {"current": {}}

        async def fetch_page(sym, start, end_exclusive, *, timeframe, single_page):
            assert (sym, start, end_exclusive, timeframe, single_page) == (
                symbol, seed_ts, seed_ts + ONE_MIN_MS, "1m", True,
            )
            return seed

        manager.exchange = SimpleNamespace()
        monkeypatch.setattr(manager, "_fetch_ohlcv_paginated", fetch_page)
        assert await manager.repair_completed_minute_page(symbol, seed_ts, seed_ts) == 1
        assert manager.plan_rms_history_repair(symbol, 1.0, now_ms=end + ONE_MIN_MS) is None
        assert await manager.get_rms_emas(
            symbol, {"current": {1.0}}, end_ts=end,
            forager_max_age_ms=0, allow_remote_fetch=False,
        ) == {"current": {1.0: (pbr.calc_signed_rms_directionality(prices, 1.0), end)}}
