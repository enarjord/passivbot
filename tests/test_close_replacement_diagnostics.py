"""Operator visibility for proven close replacements using offline account facts."""

import logging
from types import SimpleNamespace

import pytest

import passivbot
from live import close_replacement_diagnostics, executor
from live.event_bus import EventTypes
from live.freshness import FreshnessLedger
from live.order_churn_gate import OrderChurnGateState
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


def _record_timings(bot):
    timings = []

    def record(kind, _tags, payload, **envelope):
        if kind == EventTypes.ORDER_CLOSE_REPLACEMENT_TIMING:
            timings.append((payload, envelope))

    bot._monitor_record_event = record
    return timings


@pytest.mark.asyncio
async def test_replacement_timing_requires_post_cancel_fills_and_the_admitted_order(
    scenario, monkeypatch
):
    bot, old, new, cancel_wave = scenario
    cancel_wave["id"] = 4
    timings = _record_timings(bot)
    ledger = bot.freshness_ledger
    ledger.begin_epoch()
    ledger.stamp("fills", now_ms=900)
    close_replacement_diagnostics.observe_plan(bot, cancel_wave, [old], [new], 900)
    await _cancel(bot, old, cancel_wave, monkeypatch)

    # Position/order refresh and then one fills refresh still leave the
    # required post-cancel fills epoch outstanding.
    bot._authoritative_pending_confirmations["fills"] = ledger.epoch + 3
    _observe(bot, monkeypatch, now_ms=1400, surfaces=_SURFACES | {"balance"})
    _observe(bot, monkeypatch, now_ms=1700, surfaces={"fills"})
    close_replacement_diagnostics.observe_confirmation(bot, 1700)
    _observe(bot, monkeypatch, now_ms=2100, surfaces={"fills"})
    close_replacement_diagnostics.observe_confirmation(bot, 2100)

    create_wave = {"id": 7}
    amended = {**new, "price": 103.0}
    alternative = {**new, "price": 104.0}
    # More than one candidate makes the association ambiguous. This also
    # ensures the reported first plan excludes the ambiguous candidate set.
    close_replacement_diagnostics.observe_plan(
        bot, {"id": 6}, [], [amended, alternative], 2400
    )
    close_replacement_diagnostics.observe_admission(bot, [amended], 2450)
    close_replacement_diagnostics.observe_create_sent(bot, {"id": 6}, amended, 2500)
    assert not timings

    close_replacement_diagnostics.observe_plan(bot, create_wave, [], [amended], 2700)
    close_replacement_diagnostics.observe_create_sent(bot, create_wave, amended, 2750)
    close_replacement_diagnostics.observe_admission(bot, [dict(amended)], 2800)
    close_replacement_diagnostics.observe_create_sent(
        bot, create_wave, dict(amended), 2850
    )
    assert not timings

    close_replacement_diagnostics.observe_admission(bot, [amended], 2900)
    close_replacement_diagnostics.observe_create_sent(bot, create_wave, amended, 3100)
    close_replacement_diagnostics.observe_create_sent(bot, create_wave, amended, 3200)
    _observe(bot, monkeypatch, now_ms=4000, resting=[{**amended, "id": "new"}])

    [(timing, envelope)] = timings
    assert timing == {
        "outcome": "create_sent",
        "elapsed_ms": 2100,
        "account_confirmed_ms": 1100,
        "first_plan_ms": 1700,
        "create_admitted_ms": 1900,
        "cancel_wave_id": 4,
        "create_wave_id": 7,
    }
    assert envelope["symbol"] == _SYMBOL
    assert envelope["pside"] == "long"


@pytest.mark.asyncio
@pytest.mark.parametrize("sink_fails", [False, True])
async def test_replacement_timing_uses_guarded_send_and_isolates_sink_failure(
    sink_fails, scenario, monkeypatch, caplog
):
    bot, old, new, cancel_wave = scenario
    cancel_wave["id"] = 4
    close_replacement_diagnostics.observe_plan(bot, cancel_wave, [old], [new], 900)
    await _cancel(bot, old, cancel_wave, monkeypatch)
    _observe(bot, monkeypatch, now_ms=1500, surfaces=_SURFACES | {"balance", "fills"})

    timing_calls = []
    order_events = []
    transport_calls = []
    fills_settling = True
    now_ms = 2300
    close_replacement_diagnostics.observe_reconciled_plan(bot, [new], 1900)

    def record(kind, _tags, payload, **_envelope):
        if kind == EventTypes.ORDER_CLOSE_REPLACEMENT_TIMING:
            timing_calls.append(payload)
            if sink_fails:
                raise RuntimeError("private sink failure detail")

    async def cancel(orders):
        assert not orders
        return []

    async def keep_market_snapshot(_bot, orders, **_kwargs):
        return orders

    async def create(orders):
        nonlocal now_ms
        [order] = orders
        assert order is new
        assert not timing_calls
        # This callback marks the final guarded transport admission. Entering
        # the connector alone has not sent an order or emitted its timing.
        now_ms = 2700
        executor.record_create_connector_admission(bot, order)
        transport_calls.append(order)
        return [{**order, "id": "new"}]

    bot._monitor_record_event = record
    bot.execute_cancellations_parent = cancel
    bot.execute_orders_parent = lambda orders: executor.execute_orders_parent(bot, orders)
    bot.execute_orders = create
    bot.log_order_action = lambda *_a, **_kw: None
    bot._is_market_execution_order = lambda _order: False
    bot._log_order_action_summary = lambda *_a: None
    bot.did_create_order = lambda order: bool(order.get("id"))
    bot.add_to_recent_order_executions = lambda _order: None
    bot.add_new_order = lambda *_a, **_kw: None
    bot._health_orders_placed = 0
    bot.get_raw_balance = lambda: 100.0
    bot.order_was_recently_updated = lambda _order: 0
    bot._order_requires_exchange_config_before_create = lambda _order: False
    monkeypatch.setattr(passivbot, "utc_ms", lambda: now_ms)
    monkeypatch.setattr(Passivbot, "_filter_fresh_market_snapshot_creations", keep_market_snapshot)
    monkeypatch.setattr(Passivbot, "_record_emitted_order_custom_id", lambda *_a, **_kw: None)
    monkeypatch.setattr(Passivbot, "_shutdown_requested", lambda *_args: False)
    monkeypatch.setattr(Passivbot, "_track_order_wave_confirmation", lambda *_args: None)
    monkeypatch.setattr(Passivbot, "_log_order_wave_summary", lambda *_args: None)
    monkeypatch.setattr(
        executor.position_fill_sync, "permits", lambda *_args: not fills_settling
    )
    monkeypatch.setattr(
        Passivbot, "_emit_execution_order_event",
        lambda _bot, **event: order_events.append(event["event_type"]),
    )

    with caplog.at_level(logging.DEBUG):
        await executor.execute_order_plan(bot, [], [new])
        assert not transport_calls
        assert not timing_calls
        fills_settling = False
        now_ms = 2400
        await executor.execute_order_plan(bot, [], [new])

    assert transport_calls == [new]
    assert order_events == [EventTypes.EXECUTION_CREATE_SENT, EventTypes.EXECUTION_CREATE_SUCCEEDED]
    [timing] = timing_calls
    assert timing["outcome"] == "create_sent"
    assert timing["account_confirmed_ms"] == 500
    assert timing["first_plan_ms"] == 900
    assert timing["create_admitted_ms"] == 1400
    assert timing["create_action_index"] == 0
    assert timing["last_create_defer"] == "position_fill_settling"
    assert timing["elapsed_ms"] == 1700
    assert "private sink failure detail" not in caplog.text


@pytest.mark.asyncio
async def test_delay_requires_current_uncovered_cohort_and_reports_restored_quantity(
    scenario, monkeypatch, caplog
):
    bot, old, new, wave = scenario
    timings = _record_timings(bot)
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
    assert [
        (timing["outcome"], timing["elapsed_ms"])
        for timing, _envelope in timings
    ] == [("still_waiting", 60_030), ("resting_close_observed", 63_000)]
    assert timings[0][0]["last_missing_surfaces"] == ["fills"]
    assert all("account_confirmed_ms" not in timing for timing, _envelope in timings)


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
    timings = _record_timings(bot)
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
    outcome = "position_closed" if resolution == "position closed" else "no_close_in_plan"
    assert [timing["outcome"] for timing, _envelope in timings] == [
        "still_waiting",
        outcome,
    ]
    assert timings[0][0]["last_create_defer"] == "state_change_detected"
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "distance, reason",
    [
        (0.02, "order_churn_allowance_exhausted"),
        (None, "order_churn_market_data_unavailable"),
    ],
)
async def test_churn_admission_reports_current_blocker_for_delayed_close(
    distance, reason, scenario, monkeypatch, caplog
):
    bot, old, new, wave = scenario
    close_replacement_diagnostics.observe_plan(bot, wave, [old], [new], 900)
    await _cancel(bot, old, wave, monkeypatch)
    executor._observe_close_create_defer(bot, [new], "account_cancel_first_barrier")

    settings = {
        "order_replacement_churn_gate_activation_count": 1,
        "order_replacement_churn_gate_window_minutes": 10,
        "order_replacement_churn_gate_market_distance_fraction": 0.005,
    }
    bot.live_value = settings.__getitem__
    bot._is_market_execution_order = lambda order: order["type"] == "market"
    bot._order_churn_gate_state = OrderChurnGateState()
    bot._order_churn_gate_state.action_attempt_timestamps.append(executor.time.monotonic())
    monkeypatch.setattr(Passivbot, "_emit_execution_create_filter_event", lambda *a, **kw: None)
    close = {**new, "_churn_evidence": True}
    other_scope = {**new, "side": "buy", "position_side": "short", "_churn_evidence": True}
    if distance is None:
        other_scope["_churn_gate_market_distance"] = 0.02
    else:
        close["_churn_gate_market_distance"] = distance
    unaffected = {**new, "symbol": "ETH/USDT:USDT"}

    assert executor._apply_order_churn_admission(
        bot, [close, other_scope, unaffected]
    ) == [unaffected]
    _observe(bot, monkeypatch, now_ms=61_000)

    assert "close replacement delayed" in caplog.text
    assert f"last_create_defer={reason}" in caplog.text
    assert "last_create_defer=account_cancel_first_barrier" not in caplog.text
