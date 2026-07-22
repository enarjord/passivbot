from __future__ import annotations

import pytest

from live import executor
from live.event_bus import EventTypes, ReasonCodes
from live.order_churn_gate import OrderChurnGateState
from passivbot import Passivbot


def _order(name: str, *, execution_type: str = "limit", panic: bool = False) -> dict:
    return {
        "name": name,
        "symbol": "BTC/USDT:USDT",
        "position_side": "long",
        "side": "sell" if panic else "buy",
        "qty": 1.0,
        "price": 100.0,
        "reduce_only": panic,
        "type": execution_type,
        "pb_order_type": "close_panic_long" if panic else "entry_grid_normal_long",
        "execution_priority": "risk_critical" if panic else "ordinary",
    }


class _PlanBot:
    debug_mode = False
    balance_threshold = 0.0
    quote = "USDT"
    state_change_detected_by_symbol = set()
    _equity_hard_stop_coin_replay_pending_pairs = set()
    _order_churn_gate_state = None

    def __init__(self, *, cancel_error: Exception | None = None):
        self.cancel_error = cancel_error
        self.cancelled: list[dict] = []
        self.created: list[dict] = []
        self.confirmations: list[set[str]] = []
        self.execution_scheduled = False
        self._fresh_entry_eligibility_trace = None
        self._order_wave_in_progress = None

    def get_raw_balance(self):
        return 100.0

    def _is_market_execution_order(self, order):
        return order.get("type") == "market"

    async def execute_cancellations_parent(self, orders):
        self.cancelled = list(orders)
        if self.cancel_error is not None:
            raise self.cancel_error
        return list(orders)

    async def execute_orders_parent(self, orders):
        self.created = list(orders)
        return list(orders)

    def _request_authoritative_confirmation(self, surfaces):
        self.confirmations.append(set(surfaces))

    def _authoritative_full_confirmation_surfaces(self):
        return {"balance", "positions", "open_orders", "fills"}

    def order_was_recently_updated(self, _order):
        return 0

    def _shutdown_requested(self):
        return False

    async def update_exchange_configs(self, symbols):
        return set(symbols)


@pytest.fixture
def execution_shell(monkeypatch):
    wave = {
        "event_id": "ow_test",
        "skipped_create": 0,
        "deferred_create": 0,
        "cancel_posted": 0,
        "create_posted": 0,
    }
    events = []

    async def keep_market_snapshot(_bot, orders):
        return list(orders)

    monkeypatch.setattr(Passivbot, "_begin_order_wave", lambda *args, **kwargs: wave)
    monkeypatch.setattr(
        Passivbot,
        "_emit_execution_create_filter_event",
        lambda _bot, **kwargs: events.append(kwargs),
    )
    monkeypatch.setattr(
        Passivbot, "_filter_fresh_market_snapshot_creations", keep_market_snapshot
    )
    monkeypatch.setattr(Passivbot, "_shutdown_requested", lambda *args: False)
    monkeypatch.setattr(Passivbot, "_track_order_wave_confirmation", lambda *args: None)
    monkeypatch.setattr(Passivbot, "_log_order_wave_summary", lambda *args: None)
    monkeypatch.setattr(Passivbot, "_live_event_console_available", lambda *args: False)
    return wave, events


@pytest.mark.asyncio
async def test_any_stale_actual_defers_all_normal_plan_creates_account_wide(
    execution_shell,
):
    _wave, events = execution_shell
    bot = _PlanBot()
    stale = _order("stale")
    desired = _order("desired")

    await executor.execute_order_plan(bot, [stale], [desired])

    assert bot.cancelled == [stale]
    assert bot.created == []
    assert bot.confirmations == [{"balance", "positions", "open_orders", "fills"}]
    [barrier] = [
        event
        for event in events
        if event["event_type"] == EventTypes.EXECUTION_CANCEL_FIRST_BARRIER
    ]
    assert barrier["reason_code"] == ReasonCodes.ACCOUNT_CANCEL_FIRST_BARRIER
    assert barrier["order_count"] == 1


@pytest.mark.asyncio
async def test_failed_cancellation_still_arms_full_confirmation_and_never_creates(
    execution_shell,
):
    bot = _PlanBot(cancel_error=RuntimeError("exchange cancellation failed"))

    with pytest.raises(RuntimeError, match="cancellation failed"):
        await executor.execute_order_plan(bot, [_order("stale")], [_order("desired")])

    assert bot.created == []
    assert bot.confirmations == [{"balance", "positions", "open_orders", "fills"}]


@pytest.mark.asyncio
async def test_only_dedicated_protective_market_panic_bypasses_cancel_first(
    execution_shell,
):
    bot = _PlanBot()
    market_panic = _order("market_panic", execution_type="market", panic=True)
    limit_panic = _order("limit_panic", execution_type="limit", panic=True)

    await executor.execute_order_plan(
        bot,
        [_order("stale")],
        [limit_panic, market_panic],
        configure_creations=False,
    )

    assert bot.created == [market_panic]
    assert bot.confirmations == [{"balance", "positions", "open_orders", "fills"}]


@pytest.mark.asyncio
async def test_unsupported_generic_connector_keeps_legacy_same_wave_execution(
    execution_shell,
):
    bot = _PlanBot()
    bot._order_churn_gate_enabled_for_connector = False
    stale = _order("stale")
    desired = _order("desired")

    await executor.execute_order_plan(bot, [stale], [desired])

    assert bot.cancelled == [stale]
    assert bot.created == [desired]
    assert bot.confirmations == []


@pytest.mark.asyncio
async def test_local_create_deferral_consumes_no_churn_attempt(execution_shell):
    bot = _PlanBot()
    bot._order_churn_gate_state = OrderChurnGateState()
    bot.order_was_recently_updated = lambda _order: 1_000

    await executor.execute_order_plan(bot, [], [_order("desired")])

    assert bot.created == []
    assert list(bot._order_churn_gate_state.create_attempt_timestamps) == []
