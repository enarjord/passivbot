import asyncio
from dataclasses import replace
from functools import partial

import pytest

from live import hsl_runtime
from live.hsl_candles import Sources
from passivbot import Passivbot
from test_hsl_ordinary_freshness import planning_bot
from test_hsl_runtime import NOW, SYMBOL


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [
        "raw_balance",
        "position",
        "fills",
        "fill_receipt",
        "candle_receipt",
        "no_publication",
        "slow_quote",
    ],
)
async def test_hsl_disabled_ordinary_waits_for_existing_account_refresh(monkeypatch, change):
    fixture = planning_bot(monkeypatch, stage="ticker")
    bot, owner = fixture.bot, fixture.owner
    # Retain an earlier receipt so identity changes can be exercised even
    # though disabled HSL will no longer acquire one itself.
    bot._hsl_fill_observation = hsl_runtime.observe_fills(bot, (NOW, NOW))
    bot.config["bot"]["long"]["hsl"]["enabled"] = False
    assert not hsl_runtime.enabled(bot)
    owner.sources[SYMBOL] = Sources((), (), 0)
    ledger = bot._ensure_freshness_ledger()
    quotes = bot._get_orchestrator_market_snapshots
    quote_epochs = []
    quoted = asyncio.Event()

    async def track_quotes(symbols):
        quote_epochs.append(ledger.epoch)
        if change == "slow_quote" and len(quote_epochs) == 2:
            fixture.clock[0] += 11_000
        result = await quotes(symbols)
        quoted.set()
        return result

    bot._get_orchestrator_market_snapshots = track_quotes
    task = asyncio.create_task(owner._ordinary_plan())
    try:
        await asyncio.wait_for(fixture.entered.wait(), 1)
        async with owner._refresh_lock:
            # A read starts after quote acquisition begins, with the old raw
            # balance still published when that quote is ready for planning.
            fixture.clock[0] += 15_000
            ledger.begin_epoch(now_ms=fixture.clock[0])
            fixture.release.set()
            await asyncio.wait_for(quoted.wait(), 1)
            assert not fixture.captures and not task.done()
            if change != "no_publication":
                for surface in ("balance", "positions", "open_orders"):
                    ledger.stamp(surface, now_ms=fixture.clock[0])
            if change == "raw_balance":
                bot.get_raw_balance = lambda: 1000.001
            elif change == "position":
                bot.positions[SYMBOL]["long"]["size"] += 1.0
            elif change == "fills":
                ledger.stamp("fills", ("new_fill",), now_ms=fixture.clock[0])
            elif change == "fill_receipt":
                bot._hsl_fill_observation = replace(bot._hsl_fill_observation)
            elif change == "candle_receipt":
                owner.sources[SYMBOL] = replace(owner.sources[SYMBOL])
        plan = await asyncio.wait_for(task, 1)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert not fixture.calls, "the ordinary task must not start another account read"
    if change == "raw_balance":
        assert plan is not None and plan[3].balance == 1000.001
        assert owner._account_matches(plan[3], fixture.clock[0])
        assert quote_epochs == [0, 1]
        assert not fixture.defers
        bot.get_raw_balance = lambda: 1000.002
        assert owner._account_mismatch_reason(plan[3], fixture.clock[0]) == "raw_balance_changed"
    else:
        assert plan is None
        expected = {
            "position": "account_facts_changed_during_preparation",
            "fills": "account_facts_changed_during_preparation",
            "fill_receipt": "hsl_fill_receipt_replaced",
            "candle_receipt": "hsl_candle_receipt_replaced",
            "no_publication": "current_account_inputs_unavailable",
            "slow_quote": "current_account_inputs_unavailable",
        }
        assert fixture.defers[-1]["defer_reason"] == expected[change]


@pytest.mark.asyncio
async def test_hsl_disabled_ordinary_without_refresh_preserves_existing_preparation(monkeypatch):
    fixture = planning_bot(monkeypatch)
    fixture.bot.config["bot"]["long"]["hsl"]["enabled"] = False
    fixture.release.set()
    plan = await fixture.owner._ordinary_plan()
    assert plan is not None and fixture.owner._account_matches(plan[3], fixture.clock[0])
    assert not fixture.calls and not fixture.defers


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "collision, outcome, expected_reason",
    [
        ("quotes", "success", None),
        ("candles", "success", None),
        ("quotes", "no_publication", "current_account_inputs_unavailable"),
        ("quotes", "expired", "current_account_inputs_unavailable"),
        ("quotes", "pending", "current_account_inputs_unavailable"),
        ("quotes", "position", "account_facts_changed_during_preparation"),
        ("quotes", "timeout", "account_refresh_in_progress"),
        ("quotes", "third_refresh", "account_refresh_in_progress"),
        ("candles", "epoch_advanced", "account_cohort_changed_during_consumption"),
        ("candles", "universe_changed", "prepared_universe_changed"),
    ],
)
async def test_hsl_disabled_rejoins_account_refresh_once_during_final_preparation(
    monkeypatch, collision, outcome, expected_reason
):
    fixture = planning_bot(monkeypatch)
    bot, owner = fixture.bot, fixture.owner
    bot.config["bot"]["long"]["hsl"]["enabled"] = False
    assert not hsl_runtime.enabled(bot)
    fixture.release.set()
    ledger = bot._ensure_freshness_ledger()
    quote_reads, candle_reads = [], []
    refresh_started = [asyncio.Event(), asyncio.Event()]
    release = [asyncio.Event(), asyncio.Event()]
    joined = asyncio.Queue()
    surfaces = ("balance", "positions", "open_orders")

    class ObservableLock(asyncio.Lock):
        async def acquire(self):
            if self.locked():
                joined.put_nowait(None)
            return await super().acquire()

    owner._refresh_lock = ObservableLock()
    bot._market_snapshot_signature = partial(Passivbot._market_snapshot_signature, bot)
    bot._record_market_snapshot_surface = partial(Passivbot._record_market_snapshot_surface, bot)
    bot._market_snapshot_signature_invalid = partial(Passivbot._market_snapshot_signature_invalid, bot)

    async def refresh(number):
        async with owner._refresh_lock:
            ledger.begin_epoch()
            refresh_started[number].set()
            await release[number].wait()
            if number == 1 and outcome == "no_publication":
                return
            for surface in surfaces:
                ledger.stamp(surface, now_ms=fixture.clock[0])
            if number == 1 and outcome == "expired":
                fixture.clock[0] += 10_001
            elif number == 1 and outcome == "pending":
                bot._authoritative_pending_confirmations = {"open_orders": ledger.epoch + 1}
            elif number == 1 and outcome == "position":
                bot.positions[SYMBOL]["long"]["size"] += 1.0

    refresh_tasks = []

    async def start_refresh(number):
        refresh_tasks.append(asyncio.create_task(refresh(number)))
        await refresh_started[number].wait()

    original_quotes = bot._get_orchestrator_market_snapshots

    async def quotes(symbols, *, record_surface=True):
        quote_reads.append(ledger.epoch)
        count = len(quote_reads)
        if count == 1:
            await start_refresh(0)
        if count == 2 and collision == "quotes":
            if outcome == "timeout":
                bot._live_market_snapshot_max_age_ms = lambda: 10
            await start_refresh(1)
        if count == 3 and outcome == "third_refresh":
            await owner._refresh_lock.acquire()
            ledger.begin_epoch()
        result = await original_quotes(symbols)
        if record_surface:
            bot._record_market_snapshot_surface(symbols, result)
        return result

    async def preload(*, allow_remote_fetch):
        assert allow_remote_fetch is False
        candle_reads.append(ledger.epoch)
        if collision == "candles" and len(candle_reads) == 1:
            await start_refresh(1)
        elif outcome == "epoch_advanced" and len(candle_reads) == 2:
            ledger.begin_epoch()
            for surface in surfaces:
                ledger.stamp(surface, now_ms=fixture.clock[0])
        return {}

    original_assert = bot._assert_staged_planner_preconditions

    def assert_preconditions(*, include_market_snapshot, **kwargs):
        if include_market_snapshot:
            assert set(surfaces) <= ledger.surfaces_at_epoch()
            assert ledger.surface_epoch("market_snapshot") == ledger.epoch
            assert bot._market_snapshot_signature_invalid([SYMBOL]) == []
        original_assert(include_market_snapshot=include_market_snapshot, **kwargs)

    bot._get_orchestrator_market_snapshots = quotes
    bot._preload_trailing_candles = preload
    bot._assert_staged_planner_preconditions = assert_preconditions
    bot._orchestrator_preparation_is_current = lambda _: outcome != "universe_changed"

    async def derive_trailing(**_kwargs):
        assert outcome != "universe_changed", "check the universe before deriving trailing inputs"

    bot.update_trailing_data = derive_trailing

    task = asyncio.create_task(owner._ordinary_plan())
    try:
        await asyncio.wait_for(joined.get(), 1)
        release[0].set()
        await asyncio.wait_for(joined.get(), 1)
        if outcome != "timeout":
            release[1].set()
        plan = await asyncio.wait_for(task, 1)
        if outcome == "timeout":
            assert owner._refresh_lock.locked() and not refresh_tasks[-1].done()
        if outcome == "third_refresh":
            assert owner._refresh_lock.locked()
            owner._refresh_lock.release()
    finally:
        for event in release:
            event.set()
        task.cancel()
        await asyncio.gather(task, *refresh_tasks, return_exceptions=True)

    assert fixture.calls == [], "the retry must use the existing refresh"
    if expected_reason is None:
        assert plan is not None and owner._account_matches(plan[3], fixture.clock[0])
        assert quote_reads == [0, 1, 2]
        assert candle_reads == ([1, 2] if collision == "candles" else [2])
        assert len(fixture.captures) == 1 and not fixture.defers
    else:
        assert plan is None and not fixture.captures
        assert fixture.defers[-1]["defer_reason"] == expected_reason
        reached_quote_retry = outcome in {"third_refresh", "epoch_advanced", "universe_changed"}
        assert len(quote_reads) == (3 if reached_quote_retry else 2)
        assert ledger.surfaces["market_snapshot"].revision == 2
