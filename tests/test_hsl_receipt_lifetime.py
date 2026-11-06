"""Consumed ordinary inputs stay strict only for the active plan lifetime."""

import asyncio
import gc
import weakref
from dataclasses import replace

import pytest

from live import hsl_live
from test_hsl_plan_receipts import change_receipt, planning_bot
from test_hsl_runtime import NOW, SYMBOL, quotes


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["source", "fill"])
async def test_bound_connector_wave_rejects_equal_receipt_replacement(monkeypatch, change):
    import utils

    monkeypatch.setattr(utils, "utc_ms", lambda: NOW)
    bot, instance = planning_bot(empty=False)
    bot.approved_coins_minus_ignored_coins = {"long": {SYMBOL}, "short": set()}
    original = bot.calc_orders_to_cancel_and_create

    async def native_capture():
        result = await original()
        bot._hsl_planning_wave = instance.capture(
            quotes(), ordinary_receipts=bot._hsl_planning_wave.ordinary_receipts
        )
        return result

    bot.calc_orders_to_cancel_and_create = native_capture
    plan = await instance._ordinary_plan()
    wave = plan[3]
    order = dict(symbol=SYMBOL, position_side="long")
    instance.bind(wave, (), (order,), ordinary=True)
    bound = instance._waves[order["_hsl_wave"]]
    assert bound.ordinary_receipts is None
    assert bound.ordinary_receipt_ref() is wave.ordinary_receipts
    assert instance._account_matches(bound, NOW)
    instance.quotes.update(quotes())
    assert instance.admit(order)
    change_receipt(bot, instance, change)
    assert not instance.admit(order)
    assert instance._account_mismatch_reason(bound, NOW) == (
        f"hsl_{'candle' if change == 'source' else 'fill'}_receipt_replaced"
    )


@pytest.mark.asyncio
async def test_retired_plan_cannot_keep_history_or_authorize_old_order(monkeypatch):
    import utils

    monkeypatch.setattr(utils, "utc_ms", lambda: NOW)
    bot, instance = planning_bot(empty=False)
    plan = await instance._ordinary_plan()
    receipt = weakref.ref(plan[3].ordinary_receipts)
    order = dict(symbol=SYMBOL, position_side="long")
    instance.bind(plan[3], (), (order,), ordinary=True)
    bound = instance._waves[order["_hsl_wave"]]
    assert bot._hsl_planning_wave.ordinary_receipts is None
    assert receipt() is not None
    del plan
    gc.collect()
    assert receipt() is None
    assert instance._account_mismatch_reason(bound, NOW) == "ordinary_plan_retired"
    assert not instance.admit(order)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["rejected", "error", "cancelled"])
async def test_failed_attempt_releases_consumed_receipt(outcome):
    bot, instance = planning_bot(empty=False)
    original = bot.calc_orders_to_cancel_and_create
    entered = asyncio.Event()
    references, orders = [], []

    async def reconcile():
        result = await original()
        wave = bot._hsl_planning_wave
        references.append(weakref.ref(wave.ordinary_receipts))
        order = dict(symbol=SYMBOL, position_side="long")
        orders.append(order)
        instance.bind(wave, (), (order,), ordinary=True)
        if outcome == "rejected":
            change_receipt(bot, instance, "fill")
        elif outcome == "error":
            raise ValueError("producer failure")
        else:
            entered.set()
            await asyncio.Event().wait()
        return result

    bot.calc_orders_to_cancel_and_create = reconcile
    task = asyncio.create_task(instance._ordinary_plan())
    if outcome == "cancelled":
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    elif outcome == "error":
        with pytest.raises(ValueError):
            await task
    else:
        assert await task is None
    assert bot._hsl_planning_wave.ordinary_receipts is None
    del task
    gc.collect()
    # A retained exception traceback may still own the source references, but
    # neither the bot nor wave history owns them and retirement is explicit.
    bound = instance._waves[orders[0]["_hsl_wave"]]
    assert instance._account_mismatch_reason(bound, NOW) == "ordinary_plan_retired"


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["source", "fill"])
async def test_disabled_hsl_preserves_early_preparation_receipt(change):
    bot, instance = planning_bot(empty=True)
    for side in ("long", "short"):
        bot.config["bot"][side]["hsl"]["enabled"] = False
    original = bot.prepare_planning_universe

    async def prepare():
        result = await original()
        change_receipt(bot, instance, change)
        return result

    bot.prepare_planning_universe = prepare
    assert await instance._ordinary_plan() is None


@pytest.mark.asyncio
async def test_disabled_hsl_active_plan_owns_exact_bound_receipt(monkeypatch):
    import utils

    monkeypatch.setattr(utils, "utc_ms", lambda: NOW)
    bot, instance = planning_bot(empty=False)
    bot.approved_coins_minus_ignored_coins = {"long": {SYMBOL}, "short": set()}
    for side in ("long", "short"):
        bot.config["bot"][side]["hsl"]["enabled"] = False
    original = bot.calc_orders_to_cancel_and_create
    orders = []

    async def reconcile():
        result = await original()
        order = dict(symbol=SYMBOL, position_side="long")
        orders.append(order)
        instance.bind(bot._hsl_planning_wave, (), (order,), ordinary=True)
        return result

    bot.calc_orders_to_cancel_and_create = reconcile
    plan = await instance._ordinary_plan()
    bound = instance._waves[orders[0]["_hsl_wave"]]
    gc.collect()
    assert plan is not None
    assert bound.ordinary_receipt_ref() is plan[3].ordinary_receipts
    assert instance._account_matches(bound, NOW)
    assert instance.admit(orders[0])


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["returned", "error", "cancelled"])
async def test_cycle_retires_exact_plan_after_executor_lifetime(monkeypatch, outcome):
    import utils

    monkeypatch.setattr(utils, "utc_ms", lambda: NOW)
    bot, instance = planning_bot(empty=False)
    plan = await instance._ordinary_plan()
    order = dict(symbol=SYMBOL, position_side="long")
    instance.bind(plan[3], (), (order,), ordinary=True)
    bound = instance._waves[order["_hsl_wave"]]

    async def prepared():
        return plan

    async def protect(**kwargs):
        return False

    async def unavailable(**kwargs):
        return False

    entered = asyncio.Event()

    async def execute(cancels, creates):
        # The original plan owns the exact consumed receipt across an await;
        # no bot or historical wave needs a strong reference to its tape.
        await asyncio.sleep(0)
        gc.collect()
        assert bound.ordinary_receipt_ref() is plan[3].ordinary_receipts
        assert instance._account_matches(bound, NOW)
        entered.set()
        if outcome == "error":
            raise ValueError("executor failed")
        if outcome == "cancelled":
            await asyncio.Event().wait()

    instance.protect = protect
    bot.refresh_protective_authoritative_state = unavailable
    bot.execute_order_plan_to_exchange = execute
    instance._ordinary = asyncio.create_task(prepared())
    await instance._ordinary
    cycle = asyncio.create_task(instance.cycle())
    await entered.wait()
    if outcome == "cancelled":
        cycle.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cycle
    elif outcome == "error":
        with pytest.raises(ValueError):
            await cycle
    else:
        assert (await cycle)["ordinary_executed"]
    # Even an external observer retaining the original plan cannot extend its
    # execution lifetime or authorize an order from its retired receipt.
    assert bound.ordinary_receipt_ref() is plan[3].ordinary_receipts
    assert instance._account_mismatch_reason(bound, NOW) == "ordinary_plan_retired"
    assert instance._account_mismatch_reason(plan[3], NOW) == "ordinary_plan_retired"


@pytest.mark.asyncio
@pytest.mark.parametrize("sink_fails", [False, True])
async def test_discard_reason_is_bounded_and_diagnostic_failure_cannot_admit(sink_fails):
    bot, instance = planning_bot(empty=False)
    original = bot.calc_orders_to_cancel_and_create
    observed = []

    async def reconcile():
        result = await original()
        change_receipt(bot, instance, "fill")
        return result

    def observe(details):
        observed.append(details)
        if sink_fails:
            raise OSError("unavailable diagnostic sink")

    bot.calc_orders_to_cancel_and_create = reconcile
    bot._log_staged_execution_defer = observe
    assert await instance._ordinary_plan() is None
    assert observed[-1]["defer_reason"] == "hsl_fill_receipt_replaced"
    assert observed[-1]["context"] == "ordinary after_reconciliation: hsl_fill_receipt_replaced"
    assert observed[-1]["invalid"] == {
        "ordinary_plan": [{"reason": "hsl_fill_receipt_replaced"}]
    }
