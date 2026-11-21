"""Account reservation for close creates at the guarded transport boundary."""

from dataclasses import dataclass
from fractions import Fraction
import json
import logging
import math

from ccxt.base.errors import InsufficientFunds, InvalidOrder
import passivbot_rust as native

from live import reconciler
from passivbot_exceptions import FatalBotException


def _order_key(order):
    return (
        order["symbol"], order["position_side"], order["pb_order_type"],
        abs(order["qty"]), order["price"], order["type"],
    )


@dataclass(frozen=True)
class Allocation:
    """Economic authority bound to the exact Rust output and account cohort."""

    snapshot: object
    cap: float | None
    headroom: Fraction | None
    overruns: dict
    post_only: bool
    symbols: dict
    losses: dict
    priority: dict
    fill_cutoff_ms: int


def bind(bot, inputs, allocations, snapshot):
    """Preserve the budget and accepted close identities from the Rust batch."""
    policy = inputs["global"]
    bot._realized_close_allocation = None
    gate_enabled = policy["loss_gate_enabled"]
    if not gate_enabled and not policy["auto_unstuck_allowed"]:
        return
    cap = policy["loss_gate_max_drawdown_fraction"]
    headroom = (
        Fraction(native.calc_loss_gate_headroom_v9(
            inputs["balance_raw"], True, cap, policy["realized_pnl_drawdown"]
        ))
        if gate_enabled else None
    )
    symbols_by_idx = {row["symbol_idx"]: row for row in inputs["symbols"]}
    losses = {}
    priorities = {}
    overruns = {}
    reserved = Fraction()
    for index, row in enumerate(allocations):
        if row["outcome"] == "blocked":
            continue
        symbol = symbols_by_idx[row["symbol_idx"]]["symbol"]
        key = (
            symbol, row["pside"], row["order_type"], abs(row["final_qty"]),
            row["price"], row["execution_type"],
        )
        if key in losses:
            raise FatalBotException("Native loss allocation repeated a close identity")
        loss = Fraction(row["estimated_loss"])
        losses[key] = loss
        priorities[key] = index
        after = reserved + loss
        if headroom is not None and reserved < headroom < after:
            overruns[key] = after - headroom
        reserved = after
    bot._realized_close_allocation = Allocation(
        snapshot, cap, headroom, overruns,
        policy["limit_orders_post_only"],
        {row["symbol"]: row for row in inputs["symbols"]},
        losses, priorities, bot._realized_loss_memory_diagnostics["now"],
    )


@dataclass
class SubmittedClose:
    symbol: str
    pside: str
    client_id: str
    qty: Fraction
    loss: Fraction
    price: float
    route: str
    post_only: bool
    unstuck: bool
    timestamp: int
    terminal_fill_qty: Fraction | None = None


def _attempts(bot):
    # The write guard outlives a bot's in-process restart. Exact observed
    # attempts remain unresolved until current account and fill truth settle.
    guard = bot._create_failure_guard
    if not hasattr(guard, "realized_close_attempts"):
        guard.realized_close_attempts = {}
    return guard.realized_close_attempts


def _native_order_type(order):
    from passivbot import custom_id_has_explicit_passivbot_marker, custom_id_to_snake

    custom_id = reconciler.extract_order_custom_id(order)
    if custom_id and custom_id_has_explicit_passivbot_marker(custom_id):
        return custom_id_to_snake(custom_id)
    return str(order.get("pb_order_type") or "")


def _identity(order):
    return (
        order["symbol"],
        reconciler.canonical_passivbot_custom_id(reconciler.extract_order_custom_id(order)),
    )


def _project_loss(allocation, symbol, pside, remaining, price, route, *, post_only):
    inputs = allocation.symbols[symbol]
    position = inputs[pside]["position"]
    size = position["size"]
    if size == 0.0:
        # A reduce-only order cannot fill while this position is flat.
        return Fraction()
    projected = json.loads(native.project_close_reservation_json(json.dumps({
        "pside": pside,
        "qty": math.copysign(min(float(remaining), abs(size)), -size),
        "price": price,
        "execution_type": route,
        "post_only": post_only,
        "position": position,
        "order_book": inputs["order_book"],
        "exchange": inputs["exchange"],
        "market_order_slippage_fraction": 0.0,
        "loss_gate_enabled": True,
        "loss_gate_max_drawdown_fraction": allocation.cap,
    }, allow_nan=False)))
    return Fraction(projected["estimated_loss"])


def _project_resting(allocation, order, remaining):
    pside = order["position_side"]
    if pside not in ("long", "short") or order["side"] != ("sell" if pside == "long" else "buy"):
        raise ValueError("resting close side contradicts its position")
    if order["type"].lower() != "limit":
        raise ValueError("resting close execution type is unavailable")
    # A limit still on the open-order surface is resting liquidity.
    return _project_loss(
        allocation, order["symbol"], pside, remaining, order["price"], "limit",
        post_only=True,
    )


def _reservations(bot, allocation):
    """Account for exact working orders and outstanding transport attempts once."""
    reserved = Fraction()
    resting_unstuck = False
    observed = {}
    for orders in bot.open_orders.values():
        for order in orders:
            reduce_only = bot._canonical_open_order_reduce_only(order)
            if reduce_only is False:
                continue
            if reduce_only is not True:
                raise ValueError("open order has unknown reduce-only status")
            order_type = _native_order_type(order)
            if order_type.startswith("close_panic_"):
                continue
            resting_unstuck |= order_type.startswith("close_unstuck_")
            remaining = reconciler.extract_order_remaining_qty(order)
            if remaining is None:
                raise ValueError("resting close has no authoritative remaining quantity")
            key = _identity(order)
            if key[1]:
                if key in observed:
                    raise ValueError("duplicate close client order ID")
                observed[key] = order["position_side"]
            if allocation.headroom is not None and remaining:
                reserved += _project_resting(allocation, order, remaining)

    attempts = _attempts(bot)
    if not attempts:
        return reserved, resting_unstuck
    events = bot._pnls_manager.get_events(
        start_ms=min(attempt.timestamp for attempt in attempts.values()),
        end_ms=allocation.fill_cutoff_ms,
    )
    filled = {key: Fraction() for key in attempts}
    seen_fills = set()
    for fill in events:
        key = (fill.symbol, reconciler.canonical_passivbot_custom_id(fill.client_order_id))
        if key in attempts and fill.id not in seen_fills:
            attempt = attempts[key]
            if fill.position_side == attempt.pside and fill.timestamp >= attempt.timestamp:
                filled[key] += Fraction(str(abs(fill.qty)))
                seen_fills.add(fill.id)
    for key, attempt in list(attempts.items()):
        remaining = attempt.qty - filled[key]
        if (
            observed.get(key) == attempt.pside
            or remaining <= 0
            or (attempt.terminal_fill_qty is not None and filled[key] >= attempt.terminal_fill_qty)
        ):
            del attempts[key]
            continue
        resting_unstuck |= attempt.unstuck
        if allocation.headroom is not None:
            latest = _project_loss(
                allocation, attempt.symbol, attempt.pside, remaining, attempt.price,
                attempt.route, post_only=attempt.post_only,
            )
            reserved += max(latest, attempt.loss * remaining / attempt.qty)
    return reserved, resting_unstuck


def permits(bot, order):
    """Check the next write under the ordinary connector's account lock."""
    if not order.get("reduce_only") or _native_order_type(order).startswith("close_panic_"):
        return True
    unstuck = _native_order_type(order).startswith("close_unstuck_")
    gate_enabled = bot._loss_gate_enabled()
    if not gate_enabled and not unstuck:
        return True

    from live import hsl_live

    if hsl_live.active_ordinary_create_wave(bot) is not None:
        # Standalone unstuck uses standard admission after fresh fill facts.
        return False
    recovery = getattr(bot.cca, "ordinary_create_recovery", None)
    if recovery is not None and (
        recovery.error or recovery.attempts.keys() - {reconciler.extract_order_custom_id(order)}
    ):
        return False
    allocation = getattr(bot, "_realized_close_allocation", None)
    if (
        allocation is None
        or allocation.snapshot is not getattr(bot, "_current_planning_snapshot", None)
        or gate_enabled != (allocation.headroom is not None)
        or allocation.post_only != (bot.live_value("time_in_force") == "post_only")
        or _order_key(order) not in allocation.losses
    ):
        return False
    from live.planning_gates import current_planning_snapshot_invalid_for_creations

    if current_planning_snapshot_invalid_for_creations(
        bot, (order["symbol"],), snapshot=allocation.snapshot
    ):
        return False
    loss = allocation.losses[_order_key(order)]
    if not loss and not unstuck:
        return True
    try:
        reserved, resting_unstuck = _reservations(bot, allocation)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        logging.debug("[risk] close reservation inputs unavailable | error_type=%s", type(exc).__name__)
        return False

    if unstuck and resting_unstuck:
        return False
    if allocation.headroom is None:
        return True
    if not loss:
        return True
    overrun = allocation.overruns.get(_order_key(order), Fraction())
    return reserved < allocation.headroom and (
        reserved + loss <= allocation.headroom + overrun
    )


def prioritize(bot, orders):
    """Retain the native close priority through connector batch capacity."""
    allocation = getattr(bot, "_realized_close_allocation", None)
    if (
        allocation is None or allocation.headroom is None
        or allocation.snapshot is not getattr(bot, "_current_planning_snapshot", None)
    ):
        return orders
    return sorted(
        orders,
        key=lambda order: allocation.priority.get(_order_key(order), len(allocation.priority)),
    )


def submitted(bot, order):
    order_type = _native_order_type(order)
    if not order_type.startswith("close_") or order_type.startswith("close_panic_"):
        return
    unstuck = order_type.startswith("close_unstuck_")
    gate_enabled = bot._loss_gate_enabled()
    if not gate_enabled and not unstuck:
        return
    # Bitunix owns durably tracked ambiguity and its authoritative release.
    recovery = getattr(bot.cca, "ordinary_create_recovery", None)
    if recovery is not None:
        return
    allocation = bot._realized_close_allocation if gate_enabled else None
    loss = allocation.losses[_order_key(order)] if allocation is not None else Fraction()
    key = _identity(order)
    if not key[1]:
        raise FatalBotException("Reserved close creation requires an exact client order ID")
    attempts = _attempts(bot)
    if key in attempts:
        raise FatalBotException("Unresolved close submission reused its client order ID")
    attempts[key] = SubmittedClose(
        key[0], order["position_side"], key[1], Fraction(str(abs(order["qty"]))), loss,
        order["price"], order["type"], allocation.post_only if allocation is not None else
        bot.live_value("time_in_force") == "post_only", unstuck, int(bot.get_exchange_time()),
    )


def result(bot, order, *, response=None, error=None):
    if not order.get("reduce_only") or _native_order_type(order).startswith("close_panic_"):
        return
    attempts = getattr(getattr(bot, "_create_failure_guard", None), "realized_close_attempts", {})
    attempt = attempts.get(_identity(order))
    if attempt is None:
        return
    status = response.get("status") if isinstance(response, dict) else None
    if isinstance(error, (InvalidOrder, InsufficientFunds)) and "duplicate" not in str(error).lower():
        attempt.terminal_fill_qty = Fraction()
    elif status in ("failed", "rejected"):
        attempt.terminal_fill_qty = Fraction()
    elif status in ("closed", "filled", "expired", "canceled", "cancelled"):
        filled = response.get("filled")
        if type(filled) in (int, float) and math.isfinite(filled) and 0 <= filled <= attempt.qty:
            attempt.terminal_fill_qty = Fraction(str(filled))


def wave_allowed(bot):
    """Use the same account policy at Bitunix selection and continuation."""
    if bot._loss_gate_enabled():
        return False
    recovery = getattr(bot.cca, "ordinary_create_recovery", None)
    if recovery is None:
        return True
    return not recovery.error and not any(
        _native_order_type({"clientOrderId": client_id}).startswith("close_unstuck_")
        for client_id in recovery.attempts
    )


def validate_allocations(rows, orders, inputs):
    """Validate projected reservations against the exact native order envelope."""
    policy = inputs["global"]
    enabled = policy["loss_gate_enabled"]
    cap = policy["loss_gate_max_drawdown_fraction"]
    if not isinstance(rows, list):
        raise FatalBotException("Rust orchestrator close_allocations must be a list")
    symbols = {row["symbol_idx"]: row for row in inputs["symbols"]}
    expected = {
        (row["symbol_idx"], row["pside"], row["order_type"], row["qty"], row["price"], row["execution_type"])
        for row in orders
        if row["order_type"].startswith("close_") and not row["order_type"].startswith("close_panic_")
    }
    seen = set()
    try:
        for row in rows:
            symbol_idx = row["symbol_idx"]
            if type(symbol_idx) is not int:
                raise ValueError("invalid close-allocation symbol index")
            symbol = symbols[symbol_idx]
            pside = row["pside"]
            order_type = row["order_type"]
            route = row["execution_type"]
            proposed_qty = row["proposed_qty"]
            final_qty = row["final_qty"]
            price = row["price"]
            loss = row["estimated_loss"]
            outcome = row["outcome"]
            binding = row["binding_cap"]
            before, after, overrun = row["headroom_before"], row["headroom_after"], row["overshoot"]
            for value in (proposed_qty, final_qty, price, row["execution_price"], loss, row["proposed_loss"], overrun):
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError("non-finite close-allocation value")
            if (
                pside not in ("long", "short")
                or not order_type.startswith("close_") or order_type.startswith("close_panic_")
                or not order_type.endswith("_" + pside)
                or route not in ("limit", "market")
                or price <= 0.0 or row["execution_price"] <= 0.0
                or proposed_qty == 0 or (proposed_qty < 0) != (pside == "long")
                or (final_qty != 0 and (final_qty < 0) != (pside == "long"))
                or abs(final_qty) > abs(proposed_qty)
                or loss < 0 or row["proposed_loss"] < loss
                or binding not in (None, "loss_gate", "unstuck")
                or (binding == "loss_gate" and not enabled)
                or row["loss_gate_enabled"] is not enabled
                or row["loss_gate_max_drawdown_fraction"] != cap
                or isinstance(row["loss_gate_max_drawdown_fraction"], bool)
                or not isinstance(row["reason"], str) or not row["reason"]
            ):
                raise ValueError("invalid close-allocation policy or order")
            if binding is None:
                if before is not None or after is not None or overrun != 0:
                    raise ValueError("unexpected close-allocation headroom")
            elif (
                type(before) not in (int, float) or not math.isfinite(before) or before < 0
                or type(after) not in (int, float) or not math.isfinite(after)
                or after != max(0.0, before - loss) or overrun != max(0.0, loss - before)
            ):
                raise ValueError("invalid close-allocation headroom")
            if outcome == "blocked":
                if final_qty or loss:
                    raise ValueError("blocked close reserved quantity or loss")
                continue
            if (
                outcome not in ("admitted", "trimmed")
                or (outcome == "admitted") != (final_qty == proposed_qty)
                or final_qty == 0
            ):
                raise ValueError("invalid close-allocation outcome")
            key = (symbol_idx, pside, order_type, final_qty, price, route)
            if key not in expected or key in seen:
                raise ValueError("close allocation disagrees with submitted order batch")
            seen.add(key)
            projection = json.loads(native.project_close_reservation_json(json.dumps({
                "pside": pside,
                "qty": final_qty,
                "price": price,
                "execution_type": route,
                "post_only": policy["limit_orders_post_only"],
                "position": symbol[pside]["position"],
                "order_book": symbol["order_book"],
                "exchange": symbol["exchange"],
                "market_order_slippage_fraction": inputs.get("market_order_slippage_fraction", 0.0),
                "loss_gate_enabled": enabled,
                "loss_gate_max_drawdown_fraction": cap,
            }, allow_nan=False)))
            if loss != projection["estimated_loss"] or row["execution_price"] != projection["execution_price"]:
                raise ValueError("close allocation disagrees with native projection")
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
        raise FatalBotException(f"Malformed Rust close allocation: {exc}") from exc
    required = expected if enabled else {
        order for order in expected if order[2].startswith("close_unstuck_")
    }
    if required - seen:
        raise FatalBotException("Rust close allocation did not cover the finalized close batch")
