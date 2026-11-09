import asyncio
from types import SimpleNamespace

import pytest

from live.event_bus import EventTypes, LiveEvent
from live.event_emitters import emit_live_cycle_completed
from live.freshness import FreshnessLedger
from live.ordinary_diagnostics import describe_ordinary_admission
from test_hsl_plan_receipts import change_receipt, planning_bot
from test_hsl_runtime import NOW


def test_ordinary_admission_preserves_account_clock_categories_in_monitor_event():
    now_ms = 20_000
    ledger = FreshnessLedger()
    ledger.stamp("balance", "private raw balance", now_ms=9_999)
    ledger.stamp("positions", ["private position"], now_ms=10_000)
    ledger.stamp("open_orders", ["private order"], now_ms=20_001)
    events = []
    bot = SimpleNamespace(
        freshness_ledger=ledger,
        _ensure_freshness_ledger=lambda: ledger,
        _live_market_snapshot_max_age_ms=lambda: 10_000,
        _authoritative_pending_confirmations={"positions": 1},
        _emit_live_event=lambda kind, **kwargs: events.append(LiveEvent(kind, **kwargs)),
    )

    detail = describe_ordinary_admission(
        bot, now_ms=now_ms, handoff="same_cycle", reason="current_account_inputs_unavailable",
        create_count=2, cancel_count=1,
    )
    emit_live_cycle_completed(
        bot, cycle_id="cy_1", loop_start_ms=now_ms, timings_ms={}, execution_owner="hsl",
        ordinary_admission=detail,
    )
    kind, _, payload = events[0].to_monitor_event()
    assert kind == EventTypes.CYCLE_COMPLETED
    assert payload["ordinary_admission"] == {
        "handoff": "same_cycle",
        "reason": "current_account_inputs_unavailable",
        "create_count": 2,
        "cancel_count": 1,
        "max_age_ms": 10_000,
        "receipts": [
            {"surface": "balance", "age_ms": 10_001, "clock_status": "stale", "confirmation_pending": False},
            {"surface": "positions", "age_ms": 10_000, "clock_status": "fresh", "confirmation_pending": True},
            {"surface": "open_orders", "age_ms": None, "clock_status": "invalid", "confirmation_pending": False},
        ],
    }
    assert "private" not in events[0].to_json()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handoff, age_account, replace_fill",
    [
        ("ready_at_entry", False, False),
        ("ready_at_entry", False, True),
        ("same_cycle", False, False),
        ("same_cycle", False, True),
        ("ready_at_entry", True, False),
        ("after_refresh_retry", True, True),
    ],
)
async def test_completed_cycle_records_rejected_ready_orders(
    monkeypatch, handoff, age_account, replace_fill,
):
    monkeypatch.setattr("utils.utc_ms", lambda: NOW)
    bot, owner = planning_bot(empty=False)
    _, _, snapshot, wave = await owner._ordinary_plan()
    cancels = [{"symbol": "private cancellation"}]
    creates = [{"symbol": "private creation"}] * 2
    plan = (cancels, creates, snapshot, wave)
    pending = asyncio.get_running_loop().create_future()
    owner._ordinary = pending
    if handoff != "same_cycle":
        pending.set_result(plan)
    ledger = bot._ensure_freshness_ledger()
    surfaces = ("balance", "positions", "open_orders")
    if age_account:
        for surface in surfaces:
            ledger.stamp(surface, now_ms=NOW - 10_001)

    async def protect(**kwargs):
        if not pending.done():
            pending.set_result(plan)
        if replace_fill and not age_account:
            change_receipt(bot, owner, "fill")
        return False

    async def refresh(**kwargs):
        if age_account:
            for surface in surfaces:
                ledger.stamp(surface, now_ms=NOW)
            if replace_fill:
                change_receipt(bot, owner, "fill")
        return True

    writes = []

    async def execute(*orders):
        writes.append(orders)

    events = []
    bot._begin_live_event_cycle = lambda **kwargs: setattr(bot, "_live_event_current_cycle_id", "cy_1")
    bot._emit_live_event = lambda kind, **kwargs: events.append(LiveEvent(kind, **kwargs))
    bot.refresh_protective_authoritative_state = refresh
    bot.execute_order_plan_to_exchange = execute
    owner.protect = protect
    owner.remember_position = lambda: None
    owner.schedule_history = lambda: None
    owner.schedule_sources = lambda: None
    owner.schedule_account_reporting = lambda: None

    try:
        result = await owner.cycle()
        assert result["ordinary_executed"] is (not replace_fill)
        assert writes == ([] if replace_fill else [(cancels, creates)])
        [completed] = [event for event in events if event.event_type == EventTypes.CYCLE_COMPLETED]
        _, _, payload = completed.to_monitor_event()
        if replace_fill or age_account:
            detail = payload["ordinary_admission"]
            assert detail["handoff"] == handoff
            assert detail["reason"] == (
                "hsl_fill_receipt_replaced" if replace_fill else "current_account_inputs_unavailable"
            )
            assert detail["create_count"] == 2
            assert detail["cancel_count"] == 1
            assert {row["clock_status"] for row in detail["receipts"]} == {
                "stale" if age_account and not replace_fill else "fresh"
            }
            assert "private" not in completed.to_json()
        else:
            assert "ordinary_admission" not in payload
    finally:
        task = owner._ordinary
        owner.cancel_inputs()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
