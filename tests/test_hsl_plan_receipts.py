"""An ordinary plan retains the acquisitions consumed by its final calculation."""

import asyncio
from dataclasses import replace

import pytest

from live import hsl_live, hsl_runtime
from live.hsl_candles import Sources
from test_hsl_runtime import NOW, SYMBOL, bot as make_bot, quotes


def planning_bot(*, empty):
    bot = make_bot()
    bot.get_exchange_time = lambda: NOW
    bot._live_market_snapshot_max_age_ms = lambda: 10_000
    bot._staged_planner_required_surfaces = lambda **kwargs: set()
    bot._staged_execution_ready_state = lambda **kwargs: (True, {})
    bot._begin_live_event_cycle = lambda **kwargs: None
    ledger = bot._ensure_freshness_ledger()
    for surface in ("balance", "positions", "open_orders"):
        ledger.stamp(surface, now_ms=NOW)
    instance = hsl_live.owner(bot)
    instance.sources[SYMBOL] = Sources((), (), 0)
    bot._hsl_fill_observation = hsl_runtime.observe_fills(bot, (NOW, NOW))
    bot._current_planning_snapshot = object()

    async def ready(**kwargs):
        return True

    bot._log_staged_execution_defer = lambda details: None
    bot._after_ordinary_consumption = ready

    async def reconcile():
        bot._hsl_planning_wave = hsl_live.Wave(
            NOW, NOW, (), (), (), hsl_runtime.observe_positions(bot).payload,
            hsl_runtime.observe_open_orders(bot), bot.get_raw_balance(), 0,
            ordinary_receipts=instance.ordinary_receipts(),
        )
        await bot._after_ordinary_consumption()
        return [], [] if empty else [dict(symbol=SYMBOL, qty=1.0)]

    bot.prepare_planning_universe = ready
    bot.refresh_market_state_if_needed = ready
    bot.calc_orders_to_cancel_and_create = reconcile
    return bot, instance


def change_receipt(bot, instance, change):
    if change == "source":
        old = instance.sources[SYMBOL]
        instance.sources[SYMBOL] = replace(old)
        assert instance.sources[SYMBOL] == old and instance.sources[SYMBOL] is not old
    elif change == "fill":
        old = bot._hsl_fill_observation
        bot._hsl_fill_observation = replace(old)
        assert bot._hsl_fill_observation == old and bot._hsl_fill_observation is not old
    elif change == "quote":
        # A planner's own quote refresh must not invalidate its preparation.
        instance.quotes.update(quotes())


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("stage", ["universe", "market", "reconcile", "after_capture"])
@pytest.mark.parametrize("change", ["source", "fill", "quote", "unchanged"])
async def test_ordinary_consumption_retains_acquisition_identity(empty, stage, change):
    bot, instance = planning_bot(empty=empty)
    entered, release = asyncio.Event(), asyncio.Event()
    attribute = {
        "universe": "prepare_planning_universe",
        "market": "refresh_market_state_if_needed",
        "reconcile": "calc_orders_to_cancel_and_create",
        "after_capture": "_after_ordinary_consumption",
    }[stage]
    original = getattr(bot, attribute)

    async def pause(**kwargs):
        entered.set()
        await release.wait()
        return await original(**kwargs)

    setattr(bot, attribute, pause)
    task = asyncio.create_task(instance._ordinary_plan())
    await asyncio.wait_for(entered.wait(), 1.0)
    change_receipt(bot, instance, change)
    release.set()
    result = await task
    allowed = stage != "after_capture" or change in {"quote", "unchanged"}
    assert (result is not None) == allowed
    # Only the returned plan owns this receipt, never the per-order risk wave.
    if result is not None:
        assert len(result) == 4
        assert getattr(bot._hsl_planning_wave, "ordinary_receipts", None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("stage", ["completed", "protection"])
@pytest.mark.parametrize("change", ["source", "fill", "quote", "unchanged"])
async def test_completed_ordinary_plan_rechecks_receipts_after_protection(
    monkeypatch, empty, stage, change
):
    import utils

    monkeypatch.setattr(utils, "utc_ms", lambda: NOW)
    bot, instance = planning_bot(empty=empty)
    plan = await instance._ordinary_plan()

    async def prepared():
        return plan

    instance._ordinary = asyncio.create_task(prepared())
    await instance._ordinary
    if stage == "completed":
        change_receipt(bot, instance, change)

    async def protect(**kwargs):
        if stage == "protection":
            change_receipt(bot, instance, change)
        return False

    async def unavailable(**kwargs):
        return False

    writes = []

    async def execute(cancels, creates):
        writes.append((cancels, creates))

    instance.protect = protect
    bot.refresh_protective_authoritative_state = unavailable
    bot.execute_order_plan_to_exchange = execute
    result = await instance.cycle()
    allowed = change in {"quote", "unchanged"}
    assert result["ordinary_completed"]
    assert result["ordinary_executed"] is allowed
    assert len(writes) == int(allowed)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["unchanged", "fill", "source"])
async def test_ordinary_plan_finishing_during_protection_is_serviced_in_same_cycle(
    monkeypatch, change
):
    import utils

    monkeypatch.setattr(utils, "utc_ms", lambda: NOW)
    bot, instance = planning_bot(empty=False)
    plan = await instance._ordinary_plan()
    release = asyncio.Event()
    calls = []
    defers = []
    bot._log_staged_execution_defer = defers.append

    async def prepare():
        await release.wait()
        return plan

    pending = asyncio.create_task(prepare())
    instance._ordinary = pending

    async def refresh(**kwargs):
        calls.append("refresh")
        assert not pending.done()
        return True

    async def protect(**kwargs):
        calls.append("protection")
        release.set()
        await pending
        change_receipt(bot, instance, change)
        return False

    async def execute(cancels, creates):
        calls.append("ordinary")
        assert (cancels, creates) == plan[:2]

    instance.remember_position = lambda: None
    instance.schedule_history = lambda: None
    instance.schedule_sources = lambda: None
    instance.schedule_account_reporting = lambda: None
    instance.protect = protect
    bot.refresh_protective_authoritative_state = refresh
    bot.execute_order_plan_to_exchange = execute
    try:
        result = await instance.cycle()
        accepted = change == "unchanged"
        assert result["ordinary_completed"]
        assert result["ordinary_executed"] is accepted
        assert calls == ["refresh", "protection"] + (["ordinary"] if accepted else [])
        if not accepted:
            receipt = "candle" if change == "source" else "fill"
            assert defers[0]["defer_reason"] == f"hsl_{receipt}_receipt_replaced"
        assert plan[3].ordinary_receipts in instance._retired_ordinary_receipts
    finally:
        release.set()
        tasks = [pending]
        if instance._ordinary is not None:
            tasks.append(instance._ordinary)
        instance.cancel_inputs()
        await asyncio.gather(*tasks, return_exceptions=True)
