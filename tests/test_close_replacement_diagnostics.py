"""Operator visibility for proven close replacements using offline account facts."""

import logging
from types import SimpleNamespace

import pytest

import passivbot
from live import close_replacement_diagnostics, executor
from live.freshness import FreshnessLedger
from live.reconciler import annotate_order_deltas
from passivbot import Passivbot


_SYMBOL = "BTC/USDT:USDT"
_SCOPE = (_SYMBOL, "long")
_SURFACES = {"positions", "open_orders"}


@pytest.fixture
def scenario(monkeypatch):
    bot = SimpleNamespace(
        freshness_ledger=FreshnessLedger(),
        positions={_SYMBOL: {"long": {"size": 2.0}}},
        _authoritative_pending_confirmations={},
        _canonical_open_order_reduce_only=lambda row: row["reduce_only"],
        _health_orders_cancelled=0,
        debug_mode=False,
        state_change_detected_by_symbol=set(),
    )
    bot._ensure_freshness_ledger = lambda: bot.freshness_ledger
    bot._request_authoritative_confirmation = lambda surfaces: bot._authoritative_pending_confirmations.update(
        (name, bot.freshness_ledger.epoch + 1) for name in surfaces
    )
    bot._cancel_result_requires_full_authoritative_confirmation = (
        lambda row: row.get("requires_confirmation", False)
    )
    bot.did_cancel_order = lambda response, _: bool(response.get("id"))
    bot.live_value = lambda _: 10
    bot._resolve_pb_order_type = lambda _: "close_grid_long"
    bot.remove_order = lambda *_args, **_kwargs: None
    bot._monitor_record_event = lambda *_args, **_kwargs: None
    bot._monitor_order_payload = lambda *_args, **_kwargs: {}
    monkeypatch.setattr(Passivbot, "_emit_execution_order_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(Passivbot, "_live_event_console_available", lambda *_args: False)

    old = dict(id="old", symbol=_SYMBOL, position_side="long", side="sell", type="limit",
               qty=2.0, price=101.0, reduce_only=True)
    new = {**old, "price": 102.0}
    del new["id"]
    annotate_order_deltas(bot, [old], [new])
    bot.open_orders = {_SYMBOL: [old]}
    wave = {}
    return bot, old, new, wave


async def _cancel(bot, old, wave, monkeypatch, *, response=None):
    async def cancel(orders):
        assert orders == [old]
        return [dict(id=old["id"]) if response is None else response]

    bot.execute_cancellations = cancel
    bot._order_wave_in_progress = wave
    monkeypatch.setattr(passivbot, "utc_ms", lambda: 1000)
    return await executor.execute_cancellations_parent(bot, [old])


def _observe(bot, monkeypatch, *, now_ms, resting=(), pending=(), surfaces=_SURFACES):
    ledger = bot.freshness_ledger
    ledger.begin_epoch()
    # Stable signature isolates the existing position-change confirmation path.
    for surface in surfaces:
        ledger.stamp(surface, signature=None, now_ms=now_ms)
    bot._authoritative_pending_confirmations.update(
        (surface, ledger.epoch + 1) for surface in pending
    )
    bot.fetched_open_orders = list(resting)
    monkeypatch.setattr(passivbot, "utc_ms", lambda: now_ms)
    Passivbot._finalize_authoritative_refresh_consistency(bot, surfaces)


@pytest.mark.asyncio
async def test_delay_requires_current_uncovered_cohort_and_reports_restored_quantity(
    scenario, monkeypatch, caplog
):
    bot, old, new, wave = scenario
    close_replacement_diagnostics.observe_plan(bot, wave, [old], [new], 900)
    await _cancel(bot, old, wave, monkeypatch)
    # The original order is still visible, only one surface refreshed, or the
    # next epoch is required. None is proof of a delayed close.
    with caplog.at_level(logging.INFO):
        _observe(bot, monkeypatch, now_ms=60_999)
        _observe(bot, monkeypatch, now_ms=61_000, resting=[old])
        _observe(bot, monkeypatch, now_ms=61_010, surfaces={"positions"})
        _observe(bot, monkeypatch, now_ms=61_020, pending={"open_orders"})
        assert not caplog.records

        bot._console_ordinary_last_defer = {"missing_surfaces": ["fills"]}
        _observe(bot, monkeypatch, now_ms=61_030)
        _observe(bot, monkeypatch, now_ms=62_000)
        unrelated = {**new, "id": "short", "position_side": "short", "side": "buy"}
        _observe(bot, monkeypatch, now_ms=63_000, resting=[unrelated])
        _observe(bot, monkeypatch, now_ms=63_500, resting=[{**new, "id": "new", "remaining": 0}])
        replacement = {**new, "id": "new", "remaining": 1.5}
        _observe(bot, monkeypatch, now_ms=64_000, resting=[replacement])
        _observe(bot, monkeypatch, now_ms=65_000, resting=[replacement])

    assert [record.levelname for record in caplog.records] == ["WARNING", "INFO"]
    assert "close replacement delayed" in caplog.messages[0]
    assert "elapsed=60.0s" in caplog.messages[0]
    assert "resting_close_qty=0/2" in caplog.messages[0]
    assert "last_planning_defer=fills" in caplog.messages[0]
    assert "resting close observed" in caplog.messages[1]
    assert "resting_close_qty=1.5/2" in caplog.messages[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", ["retire", "other_side", "multiple_resting", "ambiguous", "rejected"]
)
async def test_never_arms_without_matched_intent_and_unambiguous_ack(
    case, scenario, monkeypatch, caplog
):
    bot, old, new, wave = scenario
    response = None
    if case == "retire":
        new = None
    elif case == "other_side":
        new = {**new, "position_side": "short", "side": "buy"}
    elif case == "multiple_resting":
        bot.open_orders[_SYMBOL].append({**old, "id": "another"})
    elif case == "ambiguous":
        response = dict(id=old["id"], requires_confirmation=True)
    else:
        response = {}
    close_replacement_diagnostics.observe_plan(bot, wave, [old], [new] if new else [], 900)
    await _cancel(bot, old, wave, monkeypatch, response=response)
    _observe(bot, monkeypatch, now_ms=70_000)
    assert not getattr(bot, "_console_close_replacements", {})
    assert "close replacement delayed" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resolution", ["position closed", "latest plan has no close to submit"]
)
async def test_flat_or_changed_intent_clears_warning_without_claiming_coverage(
    resolution, scenario, monkeypatch, caplog
):
    bot, old, new, wave = scenario
    close_replacement_diagnostics.observe_plan(bot, wave, [old], [new], 900)
    await _cancel(bot, old, wave, monkeypatch)
    executor._observe_close_create_defer(bot, [new], "state_change_detected")
    other_scope = {**new, "position_side": "short", "side": "buy"}
    executor._observe_close_create_defer(bot, [other_scope], "account_cancel_first_barrier")
    bot._console_ordinary_last_defer = {"missing_surfaces": ["fills"]}
    with caplog.at_level(logging.INFO):
        _observe(bot, monkeypatch, now_ms=61_000)
        if resolution == "position closed":
            bot.positions = {}
            _observe(bot, monkeypatch, now_ms=64_000)
        else:
            close_replacement_diagnostics.observe_plan(bot, None, [], [], 64_000)
        _observe(bot, monkeypatch, now_ms=65_000)

    assert "close replacement delayed" in caplog.messages[0]
    assert "last_create_defer=state_change_detected" in caplog.messages[0]
    assert "last_planning_defer=fills" in caplog.messages[0]
    assert resolution in caplog.messages[1]
    assert len(caplog.records) == 2
    assert not bot._console_close_replacements


@pytest.mark.asyncio
async def test_prompt_replacement_and_broken_log_handler_do_not_emit_a_recovery(
    scenario, monkeypatch, caplog
):
    bot, old, new, wave = scenario
    close_replacement_diagnostics.observe_plan(bot, wave, [old], [new], 900)
    await _cancel(bot, old, wave, monkeypatch)
    with caplog.at_level(logging.INFO):
        _observe(bot, monkeypatch, now_ms=9000, resting=[{**new, "id": "new"}])
        assert not caplog.records
        assert not bot._console_close_replacements

        close_replacement_diagnostics.observe_plan(bot, wave, [old], [new], 10_000)
        await _cancel(bot, old, wave, monkeypatch)

        def broken_handler(*args, **kwargs):
            raise RuntimeError("unsafe exception contents")

        monkeypatch.setattr(logging, "warning", broken_handler)
        _observe(bot, monkeypatch, now_ms=61_000)
        assert bot._console_close_replacements[_SCOPE].warned is False
        assert "unsafe exception contents" not in caplog.text


@pytest.mark.asyncio
async def test_real_plan_records_cancel_first_blocker_without_submitting_a_create(
    scenario, monkeypatch, caplog
):
    bot, old, new, _ = scenario
    actions = []

    async def cancel(orders):
        actions.append(("cancel", list(orders)))
        return [{"id": old["id"]}]

    async def create(orders):
        actions.append(("create", list(orders)))
        return []

    async def keep_market_snapshot(_bot, orders, **_kwargs):
        return orders

    bot.execute_cancellations = cancel
    bot.execute_cancellations_parent = lambda orders: executor.execute_cancellations_parent(bot, orders)
    bot.execute_orders_parent = create
    bot.get_raw_balance = lambda: 100.0
    bot._authoritative_full_confirmation_surfaces = lambda: _SURFACES
    monkeypatch.setattr(passivbot, "utc_ms", lambda: 1000)
    monkeypatch.setattr(executor, "connector_supports_order_churn_gate", lambda _: True)
    monkeypatch.setattr(Passivbot, "_filter_fresh_market_snapshot_creations", keep_market_snapshot)
    monkeypatch.setattr(Passivbot, "_shutdown_requested", lambda *_args: False)
    monkeypatch.setattr(Passivbot, "_track_order_wave_confirmation", lambda *_args: None)
    monkeypatch.setattr(Passivbot, "_log_order_wave_summary", lambda *_args: None)

    await executor.execute_order_plan(bot, [old], [new])
    _observe(bot, monkeypatch, now_ms=61_000)

    assert actions == [("cancel", [old])]
    assert "last_create_defer=account_cancel_first_barrier" in caplog.text
