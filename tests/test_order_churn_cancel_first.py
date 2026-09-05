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
    _config_hedge_mode = True
    hedge_mode = True
    state_change_detected_by_symbol = set()
    _equity_hard_stop_coin_replay_pending_pairs = set()
    _order_churn_gate_state = None

    def __init__(self, *, cancel_error: Exception | None = None):
        self.cancel_error = cancel_error
        self.state_change_detected_by_symbol = set()
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
async def test_stale_actual_defers_normal_create_for_same_symbol_and_pside(
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
    assert barrier["data"]["cancel_scope_count"] == 1
    assert barrier["data"]["unscoped_cancel_count"] == 0
    assert barrier["data"]["dedicated_market_panic_bypass_count"] == 0
    assert barrier["data"]["unaffected_scope_create_count"] == 0


@pytest.mark.asyncio
async def test_cancel_first_allows_creates_for_unaffected_position_scopes(
    execution_shell,
):
    _wave, events = execution_shell
    bot = _PlanBot()
    stale = _order("stale")
    same_scope = _order("same_scope")
    other_pside = {**_order("other_pside"), "position_side": "short"}
    other_symbol = {**_order("other_symbol"), "symbol": "ETH/USDT:USDT"}

    await executor.execute_order_plan(
        bot,
        [stale],
        [same_scope, other_pside, other_symbol],
    )

    assert bot.cancelled == [stale]
    assert bot.created == [other_pside, other_symbol]
    assert bot.confirmations == [{"balance", "positions", "open_orders", "fills"}]
    [barrier] = [
        event
        for event in events
        if event["event_type"] == EventTypes.EXECUTION_CANCEL_FIRST_BARRIER
    ]
    assert barrier["order_count"] == 1
    assert barrier["data"]["cancel_scope_count"] == 1
    assert barrier["data"]["unscoped_cancel_count"] == 0
    assert barrier["data"]["dedicated_market_panic_bypass_count"] == 0
    assert barrier["data"]["unaffected_scope_create_count"] == 2


@pytest.mark.asyncio
async def test_cancel_first_defers_opposite_pside_on_same_symbol_in_one_way_mode(
    execution_shell,
):
    _wave, events = execution_shell
    bot = _PlanBot()
    bot._config_hedge_mode = False
    stale = _order("stale")
    opposite_pside = {**_order("opposite_pside"), "position_side": "short"}
    other_symbol = {**_order("other_symbol"), "symbol": "ETH/USDT:USDT"}

    await executor.execute_order_plan(
        bot,
        [stale],
        [opposite_pside, other_symbol],
    )

    assert bot.cancelled == [stale]
    assert bot.created == [other_symbol]
    [barrier] = [
        event
        for event in events
        if event["event_type"] == EventTypes.EXECUTION_CANCEL_FIRST_BARRIER
    ]
    assert barrier["order_count"] == 1
    assert barrier["data"]["cancel_scope_count"] == 1
    assert barrier["data"]["unscoped_cancel_count"] == 0
    assert barrier["data"]["unaffected_scope_create_count"] == 1


@pytest.mark.asyncio
async def test_unscoped_cancel_conservatively_defers_all_normal_creates(
    execution_shell,
):
    _wave, events = execution_shell
    bot = _PlanBot()
    stale = _order("stale")
    stale.pop("position_side")
    desired = {**_order("desired"), "symbol": "ETH/USDT:USDT"}

    await executor.execute_order_plan(bot, [stale], [desired])

    assert bot.cancelled == [stale]
    assert bot.created == []
    [barrier] = [
        event
        for event in events
        if event["event_type"] == EventTypes.EXECUTION_CANCEL_FIRST_BARRIER
    ]
    assert barrier["order_count"] == 1
    assert barrier["data"]["cancel_scope_count"] == 0
    assert barrier["data"]["unscoped_cancel_count"] == 1


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
async def test_dedicated_market_panic_survives_ambiguous_cancel_state_filter(
    execution_shell,
):
    bot = _PlanBot()
    market_panic = _order("market_panic", execution_type="market", panic=True)

    async def ambiguous_cancel(orders):
        bot.cancelled = list(orders)
        bot.state_change_detected_by_symbol.add(orders[0]["symbol"])
        return []

    bot.execute_cancellations_parent = ambiguous_cancel

    await executor.execute_order_plan(
        bot,
        [_order("stale")],
        [market_panic],
        configure_creations=False,
    )

    assert bot.created == [market_panic]
    assert bot.confirmations == [{"balance", "positions", "open_orders", "fills"}]


@pytest.mark.asyncio
async def test_exposure_increasing_market_panic_never_bypasses_cancel_first(
    execution_shell,
):
    bot = _PlanBot()
    malformed_panic = _order("market_panic", execution_type="market", panic=True)
    malformed_panic["reduce_only"] = False

    await executor.execute_order_plan(
        bot,
        [_order("stale")],
        [malformed_panic],
        configure_creations=False,
    )

    assert bot.created == []
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
    assert list(bot._order_churn_gate_state.action_attempt_timestamps) == []


@pytest.mark.asyncio
async def test_final_churn_distance_recheck_runs_after_exchange_config_writes(
    execution_shell, monkeypatch
):
    bot = _PlanBot()
    bot._order_churn_gate_state = OrderChurnGateState()
    bot._order_churn_gate_state.record_action_attempts(
        1, now_monotonic=executor.time.monotonic()
    )
    bot.configured = []

    def live_value(key):
        return {
            "order_replacement_churn_gate_activation_count": 1,
            "order_replacement_churn_gate_window_minutes": 10.0,
            "max_n_creations_per_batch": 20,
            "order_replacement_churn_gate_market_dist_pct": 0.005,
        }[key]

    async def update_configs(symbols):
        bot.configured.append(list(symbols))
        return set(symbols)

    async def final_market_recheck(_bot, orders):
        assert bot.configured == [[orders[0]["symbol"]]]
        orders[0]["_churn_gate_market_distance"] = 0.01
        return list(orders)

    bot.live_value = live_value
    bot.update_exchange_configs = update_configs
    monkeypatch.setattr(
        Passivbot,
        "_filter_fresh_market_snapshot_creations",
        final_market_recheck,
    )
    desired = _order("desired")
    desired["_churn_evidence"] = True

    await executor.execute_order_plan(bot, [], [desired])

    assert bot.configured == [[desired["symbol"]]]
    assert bot.created == []
    assert desired["_churn_gate_reason"] == "allowance_exhausted"


@pytest.mark.asyncio
async def test_exchange_config_uses_precreate_eligibility_snapshot(
    execution_shell, monkeypatch
):
    bot = _PlanBot()
    observed = {}

    async def update_configs(symbols, *, eligibility_now_ms):
        observed["symbols"] = list(symbols)
        observed["eligibility_now_ms"] = eligibility_now_ms
        return set(symbols)

    monkeypatch.setattr(executor, "_utc_ms", lambda: 12_345)
    bot.update_exchange_configs = update_configs
    desired = _order("desired")

    await executor.execute_order_plan(bot, [], [desired])

    assert observed == {
        "symbols": [desired["symbol"]],
        "eligibility_now_ms": 12_345,
    }
    assert bot.created == [desired]


@pytest.mark.asyncio
async def test_risk_first_capacity_applies_after_exchange_config_writes(
    execution_shell,
):
    bot = _PlanBot()
    bot.configured = []

    def live_value(key):
        return {
            "max_n_creations_per_batch": 1,
        }[key]

    async def update_configs(symbols):
        bot.configured.append(list(symbols))
        return set(symbols)

    bot.live_value = live_value
    bot.update_exchange_configs = update_configs
    ordinary = _order("ordinary")
    ordinary["symbol"] = "ETH/USDT:USDT"
    critical = _order("critical", panic=True)

    await executor.execute_order_plan(bot, [], [ordinary, critical])

    assert bot.configured == [[critical["symbol"], ordinary["symbol"]]]
    assert bot.created == [critical]
    assert ordinary["_churn_gate_reason"] == "batch_capacity"


@pytest.mark.asyncio
async def test_capacity_selects_later_admissible_order_after_churn_deferral(
    execution_shell,
):
    bot = _PlanBot()
    bot._order_churn_gate_state = OrderChurnGateState()
    bot._order_churn_gate_state.record_action_attempts(
        1, now_monotonic=executor.time.monotonic()
    )
    bot.configured = []

    def live_value(key):
        return {
            "max_n_creations_per_batch": 1,
            "order_replacement_churn_gate_activation_count": 1,
            "order_replacement_churn_gate_window_minutes": 10.0,
            "order_replacement_churn_gate_market_dist_pct": 0.005,
        }[key]

    async def update_configs(symbols):
        bot.configured.append(list(symbols))
        return set(symbols)

    bot.live_value = live_value
    bot.update_exchange_configs = update_configs
    far_churn = _order("far_churn")
    far_churn["symbol"] = "ETH/USDT:USDT"
    far_churn["_churn_evidence"] = True
    far_churn["_churn_gate_market_distance"] = 0.01
    stable = _order("stable")
    stable["_churn_evidence"] = False

    await executor.execute_order_plan(bot, [], [far_churn, stable])

    assert bot.configured == [[stable["symbol"], far_churn["symbol"]]]
    assert bot.created == [stable]
    assert far_churn["_churn_gate_reason"] == "allowance_exhausted"


@pytest.mark.asyncio
@pytest.mark.parametrize("dirty_stage", ["before_execution", "configuration", "market_refresh"])
async def test_account_invalidation_after_planning_defers_all_ordinary_creates(
    execution_shell, monkeypatch, dirty_stage
):
    from types import SimpleNamespace
    from live import reconciler
    from live.freshness import FreshnessLedger

    wave, events = execution_shell
    bot = _PlanBot()
    ledger = FreshnessLedger()
    ledger.begin_epoch()
    bot._ensure_freshness_ledger = lambda: ledger
    bot._request_authoritative_confirmation = (
        lambda surfaces, **kwargs: bot.confirmations.append(set(surfaces))
    )
    bot._current_planning_snapshot = SimpleNamespace(account_invalidation_generation=0)
    desired = _order("desired")

    def invalidate():
        # A different symbol's fill still changes account balance/exposure.
        reconciler.mark_account_critical_state_dirty(
            bot, reason="order_ws_fill", symbols=["ETH/USDT:USDT"], source="order_ws"
        )

    async def configure(symbols):
        if dirty_stage == "configuration":
            invalidate()
        return set(symbols)

    async def market_refresh(_bot, orders):
        if dirty_stage == "market_refresh":
            invalidate()
        return orders

    bot.update_exchange_configs = configure
    monkeypatch.setattr(Passivbot, "_filter_fresh_market_snapshot_creations", market_refresh)
    if dirty_stage == "before_execution":
        invalidate()

    await executor.execute_order_plan(bot, [], [desired])

    assert bot.created == []
    assert wave["skipped_create"] == 1
    assert bot.confirmations == [{"balance", "positions", "open_orders", "fills"}]
    assert any(event["reason_code"] == ReasonCodes.STATE_CHANGE_DETECTED for event in events)

    # A newly planned authoritative cohort may trade again; invalidations do not latch.
    bot._current_planning_snapshot.account_invalidation_generation = (
        bot._account_invalidation_generation
    )
    bot.update_exchange_configs = _PlanBot.update_exchange_configs.__get__(bot)

    async def fresh_market(_bot, orders):
        return orders

    monkeypatch.setattr(Passivbot, "_filter_fresh_market_snapshot_creations", fresh_market)
    await executor.execute_order_plan(bot, [], [desired])
    assert bot.created == [desired]


@pytest.mark.asyncio
async def test_account_invalidation_preserves_dedicated_market_panic_bypass(
    execution_shell, monkeypatch
):
    from types import SimpleNamespace
    from live import reconciler
    from live.freshness import FreshnessLedger

    bot = _PlanBot()
    ledger = FreshnessLedger()
    ledger.begin_epoch()
    bot._ensure_freshness_ledger = lambda: ledger
    bot._request_authoritative_confirmation = (
        lambda surfaces, **kwargs: bot.confirmations.append(set(surfaces))
    )
    bot._current_planning_snapshot = SimpleNamespace(account_invalidation_generation=0)
    desired = _order("panic", execution_type="market", panic=True)

    async def market_refresh(_bot, orders):
        reconciler.mark_account_critical_state_dirty(
            bot, reason="order_ws_fill", symbols=[desired["symbol"]], source="order_ws"
        )
        return orders

    monkeypatch.setattr(Passivbot, "_filter_fresh_market_snapshot_creations", market_refresh)
    await executor.execute_order_plan(bot, [], [desired], configure_creations=False)
    assert bot.created == [desired]


@pytest.mark.asyncio
@pytest.mark.parametrize("batch_route", ["ccxt", "base", "hyperliquid"])
@pytest.mark.parametrize("invalidate_at", ["batch_yield", "first_connector"])
@pytest.mark.parametrize("order_kind", ["entry", "market_entry", "normal_panic", "dedicated_panic"])
async def test_account_invalidation_rechecked_inside_each_connector_task(
    execution_shell, monkeypatch, batch_route, invalidate_at, order_kind
):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import Mock
    from exchanges.ccxt_bot import CCXTBot
    from exchanges.hyperliquid import HyperliquidBot
    from live import reconciler
    from live.freshness import FreshnessLedger

    class Bot(_PlanBot, Passivbot):
        execute_orders_parent = Passivbot.execute_orders_parent
        _execute_order_and_record_ack = HyperliquidBot._execute_order_and_record_ack
        did_create_order = Passivbot.did_create_order

    _wave, filter_events = execution_shell
    dedicated_panic = order_kind == "dedicated_panic"
    bot = Bot()
    bot._current_planning_snapshot = SimpleNamespace(account_invalidation_generation=0)
    bot._order_churn_gate_state = OrderChurnGateState()
    bot._health_orders_placed = 0
    bot.get_exchange_time = lambda: 1_000_000
    bot.live_value = lambda key: {
        "max_n_creations_per_batch": 10,
        "order_replacement_churn_gate_activation_count": 10,
        "order_replacement_churn_gate_window_minutes": 10.0,
        "order_replacement_churn_gate_market_dist_pct": 0.005,
    }[key]
    ledger = FreshnessLedger()
    ledger.begin_epoch()
    bot._ensure_freshness_ledger = lambda: ledger
    bot._request_authoritative_confirmation = lambda surfaces, **kwargs: None
    for name in (
        "log_order_action",
        "_log_order_action_summary",
        "_log_market_execution_notice",
        "add_to_recent_order_executions",
        "add_new_order",
        "_monitor_record_event",
        "_monitor_order_payload",
        "_emit_execution_connector_call_started_event",
        "_schedule_forager_candidate_candle_refresh",
    ):
        setattr(bot, name, Mock())
    recorded_orders = Mock()
    monkeypatch.setattr(Passivbot, "_record_emitted_order_custom_id", recorded_orders)
    order_events = []
    monkeypatch.setattr(
        Passivbot,
        "_emit_execution_order_event",
        lambda *args, **kwargs: order_events.append(kwargs),
    )
    failures = []

    async def handle_failures(items):
        failures.extend(items)

    bot._handle_order_write_failures = handle_failures

    def invalidate():
        reconciler.mark_account_critical_state_dirty(
            bot, reason="order_ws_fill", symbols=["ETH/USDT:USDT"], source="order_ws"
        )

    calls = []

    async def create_order(**params):
        calls.append(params)
        if invalidate_at == "first_connector" and len(calls) == 1:
            invalidate()
        return {"id": str(len(calls)), "status": "open"}

    bot.cca = SimpleNamespace(create_order=create_order)
    route = {
        "ccxt": CCXTBot.execute_orders,
        "base": Passivbot.execute_orders,
        "hyperliquid": HyperliquidBot.execute_orders,
    }[batch_route]

    async def execute_orders(orders):
        # The coroutine is entered only after the executor's batch-level check.
        if invalidate_at == "batch_yield":
            asyncio.get_running_loop().call_soon(invalidate)
        return await route(bot, orders)

    bot.execute_orders = execute_orders
    orders = [
        _order(
            str(i),
            execution_type="limit" if order_kind == "entry" else "market",
            panic=order_kind.endswith("panic"),
        )
        for i in range(2)
    ]
    await executor.execute_order_plan(
        bot, [], orders, configure_creations=not dedicated_panic
    )

    expected_calls = 2 if dedicated_panic else int(invalidate_at == "first_connector")
    assert len(calls) == expected_calls
    assert len(bot._order_churn_gate_state.action_attempt_timestamps) == expected_calls
    assert sum(
        call.kwargs.get("status") == "submitted" for call in recorded_orders.call_args_list
    ) == expected_calls
    assert failures == []
    assert not any(
        event["event_type"] == EventTypes.EXECUTION_AMBIGUOUS for event in order_events
    )
    assert (
        sum(
            event["event_type"] == EventTypes.EXECUTION_CREATE_SKIPPED
            for event in filter_events
        )
        == 2 - expected_calls
    )
