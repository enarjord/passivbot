from types import SimpleNamespace

import pytest
from ccxt.base.errors import InvalidOrder, NetworkError, OrderNotFound, RateLimitExceeded

from live import realized_close_admission as admission
from live.event_bus import LiveEvent, format_console_event
from live.freshness import ACCOUNT_SURFACES, FreshnessLedger
from live.reconciler import apply_order_match_tolerance, extract_order_reduce_only


@pytest.fixture
def native():
    from rust_utils import verify_loaded_runtime_extension

    if getattr(admission.native, "__is_stub__", False):
        pytest.skip("requires the real Rust close projector")
    verify_loaded_runtime_extension()
    return admission.native


def _close(symbol, *, pside, identifier, qty=1.0, order_type="grid"):
    return {
        "symbol": symbol,
        "position_side": pside,
        "side": "sell" if pside == "long" else "buy",
        "qty": qty,
        "price": 99.0 if pside == "long" else 101.0,
        "reduce_only": True,
        "type": "limit",
        "custom_id": identifier,
        "pb_order_type": f"close_{order_type}_{pside}",
    }


def _bot():
    events = []
    ledger = FreshnessLedger()
    return SimpleNamespace(
        cca=SimpleNamespace(),
        _current_planning_snapshot=object(),
        _canonical_open_order_reduce_only=extract_order_reduce_only,
        _create_failure_guard=SimpleNamespace(),
        _ensure_freshness_ledger=lambda: ledger,
        _loss_gate_enabled=lambda: True,
        _pnls_manager=SimpleNamespace(
            get_events=lambda start_ms, end_ms: [
                event for event in events if start_ms <= event.timestamp <= end_ms
            ]
        ),
        _realized_loss_memory_diagnostics={"now": 1_000},
        open_orders={},
        live_value=lambda key: {
            "time_in_force": "post_only", "order_match_tolerance_fraction": 0.01,
        }[key],
        get_exchange_time=lambda: 1_000,
    ), events


def _bind(bot, orders, *, cap=0.001, enabled=True, balance=1000.0, drawdown=0.0, symbols=("A", "B")):
    exchange = {
        "qty_step": 0.1, "price_step": 0.1, "min_qty": 0.1, "min_cost": 0.0,
        "c_mult": 1.0, "maker_fee_fraction": 0.0, "taker_fee_fraction": 0.0,
    }
    indices = {symbol: index for index, symbol in enumerate(symbols)}
    symbols = [
        {
            "symbol": symbol, "symbol_idx": index, "exchange": exchange,
            "order_book": {"bid": 98.0, "ask": 102.0},
            "long": {"position": {"size": 5.0, "price": 100.0}},
            "short": {"position": {"size": -5.0, "price": 100.0}},
        }
        for symbol, index in indices.items()
    ]
    rows = [
        {
            "outcome": "admitted", "symbol_idx": indices[order["symbol"]],
            "pside": order["position_side"], "order_type": order["pb_order_type"],
            "final_qty": order["qty"], "price": order["price"],
            "execution_type": order["type"], "estimated_loss": order.get("loss", order["qty"]),
        }
        for order in orders
    ]
    if not hasattr(bot._current_planning_snapshot, "invalid_details"):
        bot._current_planning_snapshot = SimpleNamespace(
            symbols=tuple(indices), invalid_details=lambda *, now_ms: [],
        )
    admission.bind(
        bot,
        {
            "global": {
                "loss_gate_enabled": enabled, "loss_gate_max_drawdown_fraction": cap,
                "realized_pnl_drawdown": drawdown, "limit_orders_post_only": True,
                "auto_unstuck_allowed": True,
            },
            "balance_raw": balance,
            "symbols": symbols,
        },
        rows,
        bot._current_planning_snapshot,
    )


def test_resting_close_reserves_other_symbol_and_side_until_fresh_plan(native):
    bot, _ = _bot()
    resting = _close("A", pside="long", identifier="old-a")
    replacement = _close("B", pside="short", identifier="new-b")
    bot.open_orders = {"A": [resting]}
    _bind(bot, [replacement])
    assert not admission.permits(bot, replacement)

    bot.open_orders = {}
    bot._current_planning_snapshot = object()
    assert not admission.permits(bot, replacement)
    _bind(bot, [replacement])
    assert admission.permits(bot, replacement)
    assert not admission.permits(bot, {**replacement, "type": "market"})
    assert not admission.permits(bot, {**replacement, "qty": 1.1})


def test_unresolved_cross_side_write_counts_once_when_exact_order_is_observed(native):
    bot, events = _bot()
    first = _close("A", pside="long", identifier="first")
    second = _close("B", pside="short", identifier="second")
    _bind(bot, [first])
    assert admission.permits(bot, first)
    admission.submitted(bot, first)
    _bind(bot, [second])
    assert not admission.permits(bot, second)

    events.append(SimpleNamespace(
        symbol="A", position_side="long", qty=-0.4,
        client_order_id="first", timestamp=1_001, id="partial-fill",
    ))
    bot.open_orders = {"A": [{**first, "amount": 1.0, "filled": 0.4, "remaining": 0.6}]}
    bot._realized_loss_memory_diagnostics = {"now": 1_002}
    replan = {**second, "qty": 0.1}
    _bind(bot, [replan], cap=0.0012, balance=999.6, drawdown=0.4)
    assert admission.permits(bot, replan)
    assert not bot._create_failure_guard.realized_close_attempts


@pytest.mark.parametrize("cap", [None, 0.0, 0.03])
def test_disabled_gate_preserves_normal_creates_and_owns_one_unstuck(native, cap):
    bot, _ = _bot()
    bot._loss_gate_enabled = lambda: False
    old = _close("A", pside="long", identifier="old-unstuck", order_type="unstuck")
    new = _close("B", pside="short", identifier="new-unstuck", order_type="unstuck")
    ordinary = _close("B", pside="short", identifier="ordinary")
    _bind(bot, [new], cap=cap, enabled=False)
    bot.open_orders = {"A": [old]}
    assert admission.permits(bot, ordinary)
    assert not admission.permits(bot, new)
    bot.open_orders = {}
    _bind(bot, [new], cap=cap, enabled=False)
    assert admission.permits(bot, new)


def test_bitunix_durable_transport_can_recheck_its_own_exact_attempt(native):
    bot, _ = _bot()
    order = _close("A", pside="long", identifier="submitted")
    _bind(bot, [order])
    bot.cca.ordinary_create_recovery = SimpleNamespace(error=False, attempts={"submitted": object()})
    assert admission.permits(bot, order)
    bot.cca.ordinary_create_recovery.attempts["other"] = object()
    assert not admission.permits(bot, order)


def test_native_overrun_funds_only_the_order_that_rounded_above_budget(native):
    bot, _ = _bot()
    resting = _close("A", pside="long", identifier="old", qty=0.9)
    first = _close("A", pside="long", identifier="first", qty=0.5)
    overrun = _close("B", pside="short", identifier="oversized", qty=10.0)
    _bind(bot, [first, overrun])
    bot.open_orders = {"A": [resting]}
    assert not admission.permits(bot, first)
    assert not admission.permits(bot, overrun)


@pytest.mark.asyncio
async def test_fake_connector_cancels_cross_symbol_reservation_before_replanning(native):
    import asyncio

    from exchanges.fake import FakeCCXTClient
    from live import hsl_live
    from live.executor import DeferredOrderCreation

    bot, _ = _bot()
    a, b = "A/USDT:USDT", "B/USDT:USDT"
    old = _close(a, pside="long", identifier="old")
    replacement = _close(b, pside="short", identifier="new")
    bot.cca = FakeCCXTClient({
        "name": "cross_symbol_loss_reservation",
        "start_time": "2026-01-01T00:00:00Z",
        "symbols": {a: {}, b: {}},
        "timeline": [{"t": 0, "prices": {a: 98.0, b: 102.0}}],
        "account": {
            "balance": 1000.0,
            "positions": [
                {"symbol": a, "position_side": "long", "qty": 5.0, "price": 100.0},
                {"symbol": b, "position_side": "short", "qty": 5.0, "price": 100.0},
            ],
            "open_orders": [{**old, "id": "1", "amount": old["qty"]}],
        },
    })
    bot._hsl_live = SimpleNamespace(_write_lock=asyncio.Lock(), admit=lambda order: True)
    bot.add_to_recent_order_cancellations = lambda *_: None
    bot.log_order_action = lambda *_, **__: None
    bot._log_order_action_summary = lambda *_: None
    bot._orchestrator_uses_realized_pnl = lambda: True
    bot.get_exchange_time = lambda: bot.cca.now_ms
    bot._realized_loss_memory_diagnostics = {"now": bot.cca.now_ms}
    confirmations = []

    def require_confirmation(surfaces):
        confirmations.append(set(surfaces))
        bot._current_planning_snapshot = object()

    bot._request_authoritative_confirmation = require_confirmation

    async def observe_orders():
        bot.open_orders = {}
        for order in await bot.cca.fetch_open_orders():
            order["position_side"] = order["info"]["positionSide"].lower()
            bot.open_orders.setdefault(order["symbol"], []).append(order)

    @hsl_live.connector_write("create")
    async def create(bot, order):
        return await bot.cca.create_order(
            order["symbol"], order["type"], order["side"], order["qty"], order["price"],
            params={
                "positionSide": order["position_side"].upper(),
                "reduceOnly": True, "clientOrderId": order["custom_id"],
            },
        )

    @hsl_live.connector_write("cancel")
    async def cancel(bot, order):
        return await bot.cca.cancel_order(order["id"], symbol=order["symbol"])

    await observe_orders()
    _bind(bot, [replacement], symbols=(a, b))
    assert isinstance(await create(bot, replacement), DeferredOrderCreation)
    assert not any(row["method"] == "create_order" for row in bot.cca.request_log)
    await cancel(bot, bot.open_orders[a][0])
    assert confirmations == [{"balance", "positions", "open_orders", "fills"}]
    assert isinstance(await create(bot, replacement), DeferredOrderCreation)
    await observe_orders()
    _bind(bot, [replacement], symbols=(a, b))
    assert (await create(bot, replacement))["clientOrderId"] == "new"
    writes = [
        (row["method"], row["symbol"])
        for row in bot.cca.request_log if row["method"] in ("create_order", "cancel_order")
    ]
    assert writes == [("cancel_order", a), ("create_order", b)]


def test_gate_rechecks_current_quote_receipt_at_transport(native):
    bot, _ = _bot()
    order = _close("A", pside="long", identifier="pending")
    _bind(bot, [order])
    assert admission.permits(bot, order)
    bot._current_planning_snapshot.invalid_details = lambda *, now_ms: [
        {"surface": "market", "reason": "expired_quote"},
    ]
    assert not admission.permits(bot, order)


@pytest.mark.parametrize("gate_enabled", [True, False])
@pytest.mark.parametrize("venue", ["binance", "bybit"])
@pytest.mark.asyncio
async def test_accepted_close_cancelled_before_open_order_observation(native, gate_enabled, venue):
    bot, _ = _bot()
    bot._loss_gate_enabled = lambda: gate_enabled
    initial = _close("A", pside="long", identifier="initial", order_type="unstuck")
    subsequent = _close("B", pside="short", identifier="next", order_type="unstuck")
    requests = []
    bot._trailing_fill_refresh_started_generation = 1

    async def lookup(order_id, symbol):
        requests.append((order_id, symbol))
        return {"id": order_id, "symbol": symbol, "status": "canceled", "filled": 0}

    async def terminal_history(symbol, since, limit, params):
        assert since is None and limit == 2
        return [await lookup(params["orderId"], symbol)]

    if venue == "bybit":
        bot.cca = SimpleNamespace(
            id=venue, has={"fetchCanceledAndClosedOrders": True},
            fetch_canceled_and_closed_orders=terminal_history,
        )
    else:
        bot.cca = SimpleNamespace(id=venue, has={"fetchOrder": True}, fetch_order=lookup)
    confirmations = []
    bot._request_authoritative_confirmation = (
        lambda surfaces, *, min_epoch: confirmations.append((surfaces, min_epoch))
    )
    _bind(bot, [initial], enabled=gate_enabled)
    assert admission.permits(bot, initial)
    admission.submitted(bot, initial)
    admission.result(bot, initial, response={"id": "123", "status": "open"})
    _bind(bot, [subsequent], enabled=gate_enabled)
    assert not admission.permits(bot, subsequent)

    await admission.recover(bot)
    assert requests == [("123", "A")]
    assert confirmations == [(ACCOUNT_SURFACES, 1)]
    assert not admission.permits(bot, subsequent)
    ledger = bot._ensure_freshness_ledger()
    ledger.begin_epoch()
    for surface in ACCOUNT_SURFACES - {"fills"}:
        ledger.stamp(surface, now_ms=1_001)
    assert not admission.permits(bot, subsequent)
    ledger.stamp("fills", now_ms=1_001)
    bot._hsl_fill_capture_interval = (1_000, 1_001)
    bot._trailing_fill_refresh_generation = 1
    assert not admission.permits(bot, subsequent)
    bot._trailing_fill_refresh_generation = 2
    assert admission.permits(bot, subsequent)
    assert not bot._create_failure_guard.realized_close_attempts


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [TimeoutError(), InvalidOrder("client order index already exists")])
async def test_ambiguous_close_history_requires_exact_identity_and_complete_fills(
    native, monkeypatch, error
):
    bot, events = _bot()
    initial = _close("A", pside="long", identifier="initial")
    subsequent = _close("B", pside="short", identifier="next")
    now = [0.0]
    monkeypatch.setattr(admission, "time", SimpleNamespace(monotonic=lambda: now[0]))
    confirmations = []
    bot._request_authoritative_confirmation = (
        lambda surfaces, *, min_epoch: confirmations.append((surfaces, min_epoch))
    )
    diagnostics = []
    bot._emit_live_event = lambda event_type, **event: diagnostics.append((event_type, event))
    requests = []

    async def lookup(order_id, symbol, params):
        requests.append((order_id, symbol, params))
        raise OrderNotFound("not seen")

    async def history(symbol, since, limit):
        requests.append((symbol, since, limit))
        return [
            {"id": "unrelated", "symbol": "A", "clientOrderId": "other",
             "status": "canceled", "filled": 0.0},
            {"id": "123", "symbol": "A", "clientOrderId": "initial",
             "status": "canceled", "filled": 0.3},
        ]

    bot.cca = SimpleNamespace(
        id="okx", has={"fetchOrder": True, "fetchClosedOrders": True},
        fetch_order=lookup, fetch_closed_orders=history,
    )
    _bind(bot, [initial])
    admission.submitted(bot, initial)
    admission.result(bot, initial, error=error)
    _bind(bot, [subsequent])
    await admission.recover(bot)
    await admission.recover(bot)
    assert requests == [(None, "A", {"clientOrderId": "initial"})]
    assert not confirmations
    assert len(diagnostics) == 1
    event_type, event = diagnostics[0]
    assert event_type == "risk.realized_close_recovery"
    assert event["reason_code"] == "realized_close_order_evidence_unavailable"
    assert event["data"] == {"pending_count": 1}
    console = format_console_event(LiveEvent(event_type, **event))
    assert console.startswith("[risk]")
    assert "pending=1" in console
    assert not admission.permits(bot, subsequent)

    now[0] += 5.0
    await admission.recover(bot)
    assert requests[-1] == ("A", 0, 50)
    assert confirmations == [(ACCOUNT_SURFACES, 1)]
    ledger = bot._ensure_freshness_ledger()
    ledger.begin_epoch()
    for surface in ACCOUNT_SURFACES:
        ledger.stamp(surface, now_ms=1_002)
    bot._hsl_fill_capture_interval = (1_001, 1_002)
    bot._trailing_fill_refresh_generation = 1
    assert not admission.permits(bot, subsequent)

    events.append(SimpleNamespace(
        symbol="A", position_side="long", qty=-(0.1 + 0.2), client_order_id="",
        raw=[{"data": {"id": "trade-1", "order": "123"}}],
        timestamp=995, id="trade-1",
    ))
    bot._realized_loss_memory_diagnostics = {"now": 1_002}
    _bind(bot, [subsequent], cap=0.002)
    assert admission.permits(bot, subsequent)
    assert not bot._create_failure_guard.realized_close_attempts


def test_unstuck_working_by_venue_id_retains_attribution_when_client_id_is_missing(native):
    bot, _ = _bot()
    bot._loss_gate_enabled = lambda: False
    initial = _close("A", pside="long", identifier="initial", order_type="unstuck")
    subsequent = _close("B", pside="short", identifier="next", order_type="unstuck")
    _bind(bot, [initial], enabled=False)
    admission.submitted(bot, initial)
    admission.result(bot, initial, response={"id": "123", "status": "open"})
    working = {**initial, "id": "123", "custom_id": "", "pb_order_type": ""}
    bot.open_orders = {"A": [working]}
    _bind(bot, [subsequent], enabled=False)
    assert not admission.permits(bot, subsequent)
    assert not admission.permits(bot, subsequent)
    assert bot._create_failure_guard.realized_close_attempts


@pytest.mark.parametrize("error_type", [RateLimitExceeded, NetworkError])
@pytest.mark.asyncio
async def test_recovery_lookup_preserves_owner_backpressure(native, error_type):
    bot, _ = _bot()
    initial = _close("A", pside="long", identifier="initial")
    _bind(bot, [initial])
    admission.submitted(bot, initial)
    admission.result(bot, initial, error=TimeoutError())

    async def lookup(*args):
        error = error_type("busy")
        if error_type is NetworkError:
            error.retry_after_seconds = 10.0
        raise error

    bot.cca = SimpleNamespace(id="okx", has={"fetchOrder": True}, fetch_order=lookup)
    with pytest.raises(error_type, match="busy"):
        await admission.recover(bot)
    assert bot._create_failure_guard.realized_close_attempts


@pytest.mark.parametrize(
    "ideal_price,resting_price,order_type,expected_matches",
    [
        (100.2, 100.1, "grid", 1),
        (100.2, 99.9, "grid", 0),
        (99.9, 100.2, "grid", 0),
        (100.2, 100.1, "unstuck", 0),
    ],
)
def test_order_match_tolerance_uses_both_close_projections(
    native, ideal_price, resting_price, order_type, expected_matches
):
    bot, _ = _bot()
    ideal = {
        **_close("A", pside="long", identifier="ideal", order_type=order_type),
        "price": ideal_price, "loss": max(0, 100.0 - ideal_price),
    }
    resting = {
        **_close("A", pside="long", identifier="resting", order_type=order_type),
        "price": resting_price, "remaining": 1.0,
    }
    _bind(bot, [ideal])
    cancels, creates, matches = apply_order_match_tolerance(bot, [resting], [ideal])
    assert matches == expected_matches
    assert cancels == ([] if expected_matches else [resting])
    assert creates == ([] if expected_matches else [ideal])


def test_allocation_validation_uses_global_market_slippage(native):
    exchange = {
        "qty_step": 0.1, "price_step": 0.1, "min_qty": 0.1, "min_cost": 0.0,
        "c_mult": 1.0, "maker_fee_fraction": 0.0, "taker_fee_fraction": 0.0,
    }
    order_book = {"bid": 99.0, "ask": 101.0}
    position = {"size": 5.0, "price": 100.0}
    symbol = {
        "symbol_idx": 0, "order_book": order_book, "exchange": exchange,
        "long": {"position": position},
    }
    order = {
        "symbol_idx": 0, "pside": "long", "order_type": "close_grid_long",
        "qty": -1.0, "price": 99.0, "execution_type": "market",
    }
    policy = {
        "loss_gate_enabled": True, "loss_gate_max_drawdown_fraction": 0.03,
        "limit_orders_post_only": False, "market_order_slippage_fraction": 0.01,
    }
    row = {
        **order, "proposed_qty": -1.0, "final_qty": -1.0, "execution_price": 98.0,
        "proposed_loss": 2.0, "estimated_loss": 2.0,
        "outcome": "admitted", "binding_cap": None, "headroom_before": None,
        "headroom_after": None, "overshoot": 0.0, "reason": "admitted",
        "loss_gate_enabled": True, "loss_gate_max_drawdown_fraction": 0.03,
    }
    admission.validate_allocations([row], [order], {"global": policy, "symbols": [symbol]})
