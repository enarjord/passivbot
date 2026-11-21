import json
from types import SimpleNamespace

import pytest
from ccxt.base.errors import InvalidOrder, RequestTimeout

from live.create_failure_guard import CreateFailureGuard
from live.event_bus import EventTypes, LiveEvent, ReasonCodes, format_console_event
from passivbot_exceptions import FatalOrderWriteError


SYMBOL = "BTC/USDC:USDC"


@pytest.fixture
def guard_setup():
    clock = [0]
    events = []
    def emit(kind, *, require_enqueue=False, **data):
        event = LiveEvent(kind, **data)
        events.append(event)
        return event

    bot = SimpleNamespace(
        exchange="hyperliquid",
        did_create_order=lambda result: bool(result and result.get("id")),
        _emit_live_event=emit,
    )
    return CreateFailureGuard(lambda: clock[0]), bot, clock, events


def attempt(guard, bot, client_id, *, error=None, result=None):
    order = {"symbol": SYMBOL, "custom_id": client_id}
    guard.submitted(order)
    guard.completed(bot, order, error=error, result=result)


def test_slow_failures_stop_below_hourly_restart_budget(guard_setup):
    guard, bot, clock, events = guard_setup
    for minute in (0, 12, 24, 36, 48, 59):
        clock[0] = minute * 60
        attempt(guard, bot, str(minute), error=InvalidOrder("unknown order rejection"))
        guard.check(bot)
    clock[0] = 3600
    guard.check(bot)  # The clock alone cannot turn inactivity into a failure.
    attempt(guard, bot, "deadline", error=InvalidOrder("unknown order rejection"))
    with pytest.raises(FatalOrderWriteError):
        guard.check(bot)
    assert [e.status for e in events] == ["degraded", "failed"]
    event = events[-1]
    assert event.event_type == EventTypes.EXECUTION_CREATE_HEALTH
    assert event.data["failed_creates"] == 7
    assert event.data["unconfirmed_creates"] == 0
    assert "next_action=stop" in format_console_event(event)


def test_five_is_inclusive_and_never_submitted_or_cancellations_do_not_count(guard_setup):
    guard, bot, clock, _ = guard_setup
    for index, sec in enumerate((0, 901, 1801, 2701, 3600)):
        clock[0] = sec
        guard.completed(bot, {"symbol": SYMBOL}, error=InvalidOrder())
        guard.check_write_error(bot, "cancel", InvalidOrder())
        guard.check(bot)
        attempt(guard, bot, str(index), result={"status": "rejected"})
    assert guard.failures == 5
    with pytest.raises(FatalOrderWriteError):
        guard.check(bot)


@pytest.mark.parametrize("ack", [{"id": "resting", "status": "open"}, {"id": "fill", "status": "closed"}])
def test_acknowledged_sibling_prevents_stop_and_resets_streak(guard_setup, ack):
    guard, bot, clock, events = guard_setup
    for index in range(4):
        attempt(guard, bot, str(index), error=InvalidOrder())
    sibling = {"symbol": SYMBOL, "custom_id": "sibling"}
    guard.submitted(sibling)
    clock[0] = 3600
    attempt(guard, bot, "fifth", error=InvalidOrder())
    guard.check(bot)
    guard.completed(bot, sibling, result=ack)
    guard.check(bot)
    assert guard.failures == 0
    assert [e.status for e in events] == ["degraded", "succeeded"]
    assert "error_type" not in events[-1].data


@pytest.mark.parametrize("confirmed", [True, False])
def test_unconfirmed_results_get_bounded_grace_and_exact_authoritative_proof(guard_setup, confirmed):
    guard, bot, clock, events = guard_setup
    for index in range(4):
        attempt(guard, bot, str(index), error=RequestTimeout())
    clock[0] = 3600
    attempt(guard, bot, "submitted-id", error=RequestTimeout())
    guard.confirm(bot, [("ETH/USDC:USDC", "submitted-id"), (SYMBOL, "different-id")])
    clock[0] = 3659
    attempt(guard, bot, "later", error=RequestTimeout())
    guard.check(bot)
    clock[0] = 3660
    if confirmed:
        guard.confirm(bot, [(SYMBOL, "submitted-id")])
        guard.check(bot)
        assert events[-1].status == "succeeded"
    else:
        with pytest.raises(FatalOrderWriteError):
            guard.check(bot)
        assert events[-1].data["unconfirmed_creates"] == 6


@pytest.mark.parametrize("action", ["create", "cancel"])
def test_only_exact_hyperliquid_wallet_rejection_stops_immediately(guard_setup, action):
    guard, bot, _, events = guard_setup
    guard.check_write_error(bot, action, InvalidOrder('hyperliquid {"label":"HYPERLIQUID_UNRECOGNIZED_WALLET"}'))
    guard.check_write_error(bot, action, InvalidOrder("Error: invalid order"))
    address = "0x" + "a" * 40
    error = InvalidOrder("hyperliquid " + json.dumps({
        "status": "err",
        "response": f"User or API Wallet {address} does not exist.",
    }))
    bot.exchange = "fake"
    guard.check_write_error(bot, action, error)
    bot.exchange = "hyperliquid"
    with pytest.raises(FatalOrderWriteError):
        guard.check_write_error(bot, action, error)
    event, = events
    assert event.reason_code == ReasonCodes.ORDER_WRITE_SIGNER_REJECTED
    assert event.data["action"] == action
    assert address not in format_console_event(event)


def test_emitter_failure_cannot_prevent_stop(guard_setup, monkeypatch):
    guard, bot, clock, _ = guard_setup
    for index in range(4):
        attempt(guard, bot, str(index), error=InvalidOrder())
    clock[0] = 3600
    def fail(*_, **__):
        raise RuntimeError("sink unavailable")
    monkeypatch.setattr(bot, "_emit_live_event", fail)
    attempt(guard, bot, "deadline", error=InvalidOrder())
    with pytest.raises(FatalOrderWriteError):
        guard.check(bot)


@pytest.mark.asyncio
async def test_live_create_counts_admitted_connector_outcome(monkeypatch, guard_setup):
    import asyncio

    from live import hsl_live, executor
    from passivbot import Passivbot

    guard, bot, _, _ = guard_setup
    bot._create_failure_guard = guard
    bot._build_order_params = lambda order: {}
    bot._request_authoritative_confirmation = lambda surfaces: None
    bot._emit_execution_connector_call_started_event = lambda **_: None
    monkeypatch.setattr(
        hsl_live, "owner",
        lambda _: SimpleNamespace(_write_lock=asyncio.Lock(), admit=lambda order: True),
    )
    monkeypatch.setattr(
        executor, "_pb_attr",
        lambda _: SimpleNamespace(
            _record_emitted_order_custom_id=lambda *a, **kw: None,
            _record_order_churn_allowance_attempts=lambda *a, **kw: None,
            _emit_execution_order_event=lambda *a, **kw: None,
        ),
    )

    class OfflineConnector:
        async def create_order(self, **kwargs):
            return await self.fetch("offline", method="POST")

        async def fetch(self, url, method):
            raise InvalidOrder("synthetic rejection")

    bot.cca = OfflineConnector()
    with pytest.raises(InvalidOrder):
        await Passivbot.execute_order(
            bot, {"symbol": SYMBOL, "position_side": "long", "side": "buy",
                  "price": 1, "qty": 1, "custom_id": "submitted"},
        )
    assert guard.failures == 1
    assert not guard.in_flight
