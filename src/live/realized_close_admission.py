"""Account reservation for close creates at the guarded transport boundary."""

import asyncio
from dataclasses import dataclass
from fractions import Fraction
import json
import logging
import math
import time

from ccxt.base.errors import (
    DDoSProtection,
    ExchangeError,
    InsufficientFunds,
    InvalidNonce,
    NetworkError,
    NotSupported,
    OnMaintenance,
    OrderNotFound,
    RateLimitExceeded,
)
import passivbot_rust as native

from live import reconciler
from live.event_bus import EventTags, EventTypes, ReasonCodes
from live.events import run_diagnostic_step
from live.freshness import ACCOUNT_SURFACES, FreshnessLedger
from passivbot_exceptions import FatalBotException

LOOKUP_INTERVAL_SECONDS = 5.0
LOOKUP_TIMEOUT_SECONDS = 3.0
RECOVERY_DIAGNOSTIC_INTERVAL_SECONDS = 300.0
LOOKUP_HISTORY_LIMIT = 50
LOOKUP_HISTORY_OVERLAP_MS = 60_000


class _ReservationEvidenceError(ValueError):
    def __init__(self, reason_code):
        super().__init__(reason_code)
        self.reason_code = reason_code


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
    market_slippage_fraction: float
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
        policy.get("market_order_slippage_fraction", 0.0),
        {row["symbol"]: row for row in inputs["symbols"]},
        losses, priorities, bot._realized_loss_memory_diagnostics["now"],
    )


@dataclass(frozen=True)
class TerminalConfirmation:
    ledger: FreshnessLedger
    epoch: int
    previous_fill_generation: int


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
    wire_client_id: str
    exchange_id: str = ""
    submission_complete: bool = False
    terminal_fill_qty: Fraction | None = None
    confirmation: TerminalConfirmation | None = None
    next_lookup: float = 0.0
    lookup_index: int = 0


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
        "market_order_slippage_fraction": allocation.market_slippage_fraction,
        "loss_gate_enabled": True,
        "loss_gate_max_drawdown_fraction": allocation.cap,
    }, allow_nan=False)))
    return Fraction(projected["estimated_loss"])


def _project_resting(allocation, order, remaining):
    pside = order.get("position_side")
    if pside not in ("long", "short") or order.get("side") != ("sell" if pside == "long" else "buy"):
        raise _ReservationEvidenceError(ReasonCodes.REALIZED_CLOSE_CONFLICTING_RESTING_SIDE)
    execution_type = order.get("type")
    if not isinstance(execution_type, str) or execution_type.lower() != "limit":
        raise _ReservationEvidenceError(ReasonCodes.REALIZED_CLOSE_UNKNOWN_RESTING_EXECUTION_TYPE)
    # A limit still on the open-order surface is resting liquidity.
    return _project_loss(
        allocation, order["symbol"], pside, remaining, order["price"], "limit",
        post_only=True,
    )


def spends_headroom(bot, order, *, resting=False):
    """Treat a close with missing projection authority as budget-consuming."""
    allocation = getattr(bot, "_realized_close_allocation", None)
    if (
        allocation is None or allocation.headroom is None
        or allocation.snapshot is not getattr(bot, "_current_planning_snapshot", None)
    ):
        return True
    try:
        if resting:
            remaining = reconciler.extract_order_remaining_qty(order)
            return remaining is None or bool(_project_resting(allocation, order, remaining))
        return bool(allocation.losses[_order_key(order)])
    except (KeyError, TypeError, ValueError, OverflowError):
        return True


def _matches_attempt(attempt, order):
    if not isinstance(order, dict) or order.get("symbol") != attempt.symbol:
        return False
    client_id = _identity(order)[1]
    exchange_id = reconciler.extract_order_exchange_id(order)
    if client_id and client_id != attempt.client_id:
        return False
    if exchange_id and attempt.exchange_id and exchange_id != attempt.exchange_id:
        return False
    return (
        client_id == attempt.client_id
        or bool(attempt.exchange_id and exchange_id == attempt.exchange_id)
    )


def _fill_exchange_order_ids(fill):
    # A trade's id is its execution identity, separate from its order identity.
    for source in getattr(fill, "raw", ()) or ():
        data = source.get("data") or {}
        if not isinstance(data, dict):
            continue
        for row in (data, data.get("info") or {}):
            if not isinstance(row, dict):
                continue
            for key in ("order", "orderId", "ordId", "order_id", "oid"):
                if row.get(key) not in (None, ""):
                    yield str(row[key])


def _require_terminal_confirmation(bot, attempt):
    ledger = bot._ensure_freshness_ledger()
    target_epoch = ledger.epoch + 1
    attempt.confirmation = TerminalConfirmation(
        ledger, target_epoch, getattr(bot, "_trailing_fill_refresh_started_generation", 0)
    )
    bot._request_authoritative_confirmation(ACCOUNT_SURFACES, min_epoch=target_epoch)


def _same_fill_qty(allocation, attempt, observed, expected):
    if observed == expected:
        return True
    if min(observed, expected) <= 0 or attempt.symbol not in allocation.symbols:
        return False
    # Canonical fills can sum venue quantities in binary floats. Accept only
    # representation error well below one executable lot.
    step = allocation.symbols[attempt.symbol]["exchange"]["qty_step"]
    precision = min(math.ulp(float(expected)) * 4.0, step / 1024.0)
    return abs(observed - expected) <= Fraction(precision)


def _terminal_confirmed(bot, allocation, attempt, filled):
    if attempt.terminal_fill_qty is None or not _same_fill_qty(
        allocation, attempt, filled, attempt.terminal_fill_qty
    ):
        return False
    confirmation = attempt.confirmation
    ledger = confirmation.ledger
    return (
        ledger is bot._ensure_freshness_ledger()
        and all(ledger.surface_epoch(surface) >= confirmation.epoch for surface in ACCOUNT_SURFACES)
        and getattr(bot, "_hsl_fill_capture_interval", None) is not None
        and getattr(bot, "_trailing_fill_refresh_generation", 0) > confirmation.previous_fill_generation
    )


def _terminal_quantity(bot, attempt, row):
    if not _matches_attempt(attempt, row):
        return None
    if str(row.get("status") or "").lower() not in (
        "closed", "filled", "expired", "canceled", "cancelled", "failed", "rejected",
    ):
        return None
    if row.get("side") not in (None, "", "sell" if attempt.pside == "long" else "buy"):
        return None
    if reconciler.extract_order_reduce_only(row) is False:
        try:
            if bot._canonical_open_order_reduce_only(row) is not True:
                return None
        except (KeyError, TypeError, ValueError):
            return None
    filled = row.get("filled")
    if type(filled) not in (int, float) or not math.isfinite(filled):
        return None
    filled = Fraction(str(filled))
    return filled if 0 <= filled <= attempt.qty else None


async def recover(bot):
    """Look for positive terminal evidence outside the ordinary write lock."""
    guard = getattr(bot, "_create_failure_guard", None)
    attempts = getattr(guard, "realized_close_attempts", {})
    if not attempts or getattr(bot.cca, "ordinary_create_recovery", None) is not None:
        return
    ledger = bot._ensure_freshness_ledger()
    for attempt in attempts.values():
        if attempt.confirmation is not None and attempt.confirmation.ledger is not ledger:
            _require_terminal_confirmation(bot, attempt)
    now = time.monotonic()
    if now < getattr(guard, "next_realized_close_lookup", 0.0):
        return
    ready = [
        attempt for attempt in attempts.values()
        if attempt.submission_complete
        and attempt.terminal_fill_qty is None
        and attempt.next_lookup <= now
        and not any(
            _matches_attempt(attempt, row)
            for row in bot.open_orders.get(attempt.symbol, ())
        )
    ]
    if not ready:
        return
    attempt = min(ready, key=lambda attempt: attempt.next_lookup)
    guard.next_realized_close_lookup = attempt.next_lookup = now + LOOKUP_INTERVAL_SECONDS
    routes = []
    lookup_routes = getattr(bot, "realized_close_lookup_routes", None)
    if callable(lookup_routes):
        routes = list(lookup_routes(
            attempt.symbol, attempt.exchange_id, attempt.wire_client_id,
            since_ms=max(0, attempt.timestamp - LOOKUP_HISTORY_OVERLAP_MS),
            limit=LOOKUP_HISTORY_LIMIT,
        ))
    reason = ReasonCodes.REALIZED_CLOSE_LOOKUP_UNSUPPORTED
    if routes:
        method, args, collection = routes[attempt.lookup_index % len(routes)]
        attempt.lookup_index += 1
        try:
            async with asyncio.timeout(LOOKUP_TIMEOUT_SECONDS):
                response = await method(*args)
        except (DDoSProtection, InvalidNonce, OnMaintenance, RateLimitExceeded):
            raise
        except (OrderNotFound, NotSupported):
            reason = ReasonCodes.REALIZED_CLOSE_ORDER_EVIDENCE_UNAVAILABLE
        except (ExchangeError, NetworkError, TimeoutError) as exc:
            if isinstance(exc, NetworkError) and getattr(exc, "retry_after_seconds", 0.0) > 0.0:
                raise
            reason = ReasonCodes.REALIZED_CLOSE_LOOKUP_UNAVAILABLE
        else:
            rows = response if collection else [response]
            if isinstance(rows, list):
                matches = [
                    (row, filled) for row in rows
                    if (filled := _terminal_quantity(bot, attempt, row)) is not None
                ]
                if len(matches) == 1:
                    row, attempt.terminal_fill_qty = matches[0]
                    attempt.exchange_id = reconciler.extract_order_exchange_id(row) or attempt.exchange_id
                    _require_terminal_confirmation(bot, attempt)
                    return
            reason = ReasonCodes.REALIZED_CLOSE_ORDER_EVIDENCE_UNAVAILABLE
    if now >= getattr(guard, "next_realized_close_recovery_log", 0.0):
        guard.next_realized_close_recovery_log = now + RECOVERY_DIAGNOSTIC_INTERVAL_SECONDS
        run_diagnostic_step(
            "emit realized close recovery",
            lambda: _report_recovery(
                bot, reason, len(attempts), message="awaiting close submission evidence"
            ),
        )


def _report_recovery(bot, reason, pending_count, *, message):
    message = f"{message} | pending={pending_count}"
    emit = getattr(bot, "_emit_live_event", None)
    if callable(emit) and emit(
        EventTypes.REALIZED_CLOSE_RECOVERY,
        level="warning", component="risk", tags=(EventTags.RISK, EventTags.ORDER),
        status="deferred", reason_code=reason, message=message,
        data={"pending_count": pending_count},
    ) is not None:
        return
    logging.warning("[risk] %s reason=%s", message, reason)


def _report_reservation_evidence_error(bot, reason):
    guard = bot._create_failure_guard
    deadlines = getattr(guard, "realized_close_evidence_deadlines", None)
    if deadlines is None:
        guard.realized_close_evidence_deadlines = deadlines = {}
    now = time.monotonic()
    if now < deadlines.get(reason, 0.0):
        return
    deadlines[reason] = now + RECOVERY_DIAGNOSTIC_INTERVAL_SECONDS
    _report_recovery(
        bot, reason, len(getattr(guard, "realized_close_attempts", ())),
        message="unresolved close reservation evidence",
    )


def _reservations(bot, allocation):
    """Account for exact working orders and outstanding transport attempts once."""
    reserved = Fraction()
    resting_unstuck = False
    observed = []
    observed_client_ids = set()
    for orders in bot.open_orders.values():
        for order in orders:
            reduce_only = bot._canonical_open_order_reduce_only(order)
            if reduce_only is False:
                continue
            if reduce_only is not True:
                raise _ReservationEvidenceError(ReasonCodes.REALIZED_CLOSE_UNKNOWN_REDUCE_ONLY)
            order_type = _native_order_type(order)
            if order_type.startswith("close_panic_"):
                continue
            resting_unstuck |= order_type.startswith("close_unstuck_")
            remaining = reconciler.extract_order_remaining_qty(order)
            if remaining is None:
                raise _ReservationEvidenceError(ReasonCodes.REALIZED_CLOSE_MISSING_REMAINING_QTY)
            key = _identity(order)
            if key[1]:
                if key in observed_client_ids:
                    raise _ReservationEvidenceError(ReasonCodes.REALIZED_CLOSE_DUPLICATE_CLIENT_ORDER_ID)
                observed_client_ids.add(key)
            observed.append(order)
            if allocation.headroom is not None and remaining:
                reserved += _project_resting(allocation, order, remaining)

    attempts = _attempts(bot)
    if not attempts:
        return reserved, resting_unstuck
    events = bot._pnls_manager.get_events(
        start_ms=max(0, min(attempt.timestamp for attempt in attempts.values()) - LOOKUP_HISTORY_OVERLAP_MS),
        end_ms=allocation.fill_cutoff_ms,
    )
    filled = {key: Fraction() for key in attempts}
    attempts_by_order_id = {
        (attempt.symbol, attempt.exchange_id): key
        for key, attempt in attempts.items() if attempt.exchange_id
    }
    if len(attempts_by_order_id) != sum(bool(attempt.exchange_id) for attempt in attempts.values()):
        raise _ReservationEvidenceError(ReasonCodes.REALIZED_CLOSE_DUPLICATE_EXCHANGE_ORDER_ID)
    seen_fills = set()
    for fill in events:
        client_id = reconciler.canonical_passivbot_custom_id(fill.client_order_id)
        key = (fill.symbol, client_id)
        if attempts_by_order_id and (not client_id or key in attempts):
            order_ids = set(_fill_exchange_order_ids(fill))
            if not client_id and len(order_ids) == 1:
                key = attempts_by_order_id.get((fill.symbol, next(iter(order_ids))))
            elif key in attempts and attempts[key].exchange_id and order_ids:
                if attempts[key].exchange_id not in order_ids:
                    raise _ReservationEvidenceError(
                        ReasonCodes.REALIZED_CLOSE_CONFLICTING_FILL_IDENTITY
                    )
                if len(order_ids) > 1:
                    # Canonical fills may coalesce multiple real orders into
                    # one quantity and a representative client ID.
                    raise _ReservationEvidenceError(
                        ReasonCodes.REALIZED_CLOSE_AMBIGUOUS_FILL_IDENTITY
                    )
        if key in attempts and fill.id not in seen_fills:
            attempt = attempts[key]
            if (
                fill.position_side == attempt.pside
                and fill.timestamp >= attempt.timestamp - LOOKUP_HISTORY_OVERLAP_MS
            ):
                filled[key] += Fraction(str(abs(fill.qty)))
                seen_fills.add(fill.id)
    for key, attempt in list(attempts.items()):
        remaining = attempt.qty - filled[key]
        fully_filled = _same_fill_qty(allocation, attempt, filled[key], attempt.qty)
        working = next(
            (
                order for order in observed
                if order["position_side"] == attempt.pside and _matches_attempt(attempt, order)
            ),
            None,
        )
        if working is not None and attempt.terminal_fill_qty is None:
            if attempt.unstuck and not _native_order_type(working).startswith("close_unstuck_"):
                # The order is budgeted above. Keep the attempt as the only
                # source of its one-at-a-time unstuck classification.
                resting_unstuck = True
            else:
                del attempts[key]
            continue
        if (
            working is None
            and (
                (attempt.terminal_fill_qty is None and fully_filled)
                or _terminal_confirmed(bot, allocation, attempt, filled[key])
            )
        ):
            del attempts[key]
            continue
        resting_unstuck |= attempt.unstuck
        if (
            remaining < 0 and not fully_filled
            and (allocation.headroom is not None or attempt.unstuck)
        ):
            raise _ReservationEvidenceError(ReasonCodes.REALIZED_CLOSE_CONFLICTING_FILL_QUANTITY)
        if allocation.headroom is not None:
            if remaining <= 0:
                raise ValueError("full close fill awaits reservation confirmation")
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
    except _ReservationEvidenceError as exc:
        reason = exc.reason_code
        run_diagnostic_step(
            "emit close reservation evidence warning",
            lambda: _report_reservation_evidence_error(bot, reason),
        )
        return False
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
        reconciler.extract_order_custom_id(order),
    )


def result(bot, order, *, response=None, error=None):
    if not order.get("reduce_only") or _native_order_type(order).startswith("close_panic_"):
        return
    attempts = getattr(getattr(bot, "_create_failure_guard", None), "realized_close_attempts", {})
    attempt = attempts.get(_identity(order))
    if attempt is None:
        return
    attempt.submission_complete = True
    # Some connectors report duplicate submission identities as InvalidOrder.
    # A generic invalid-order error requires exact order evidence.
    if isinstance(error, InsufficientFunds):
        attempt.terminal_fill_qty = Fraction()
    elif isinstance(response, dict):
        # The result belongs to this transport call. Check any echoed client ID
        # before adopting its exchange identity.
        client_id = reconciler.canonical_passivbot_custom_id(
            reconciler.extract_order_custom_id(response)
        )
        if client_id and client_id != attempt.client_id:
            return
        if response.get("symbol") not in (None, "", attempt.symbol):
            return
        attempt.exchange_id = reconciler.extract_order_exchange_id(response)
        evidence = {**response, "symbol": attempt.symbol}
        if response.get("status") in ("failed", "rejected") and response.get("filled") is None:
            evidence["filled"] = 0
        evidence["clientOrderId"] = attempt.wire_client_id
        attempt.terminal_fill_qty = _terminal_quantity(bot, attempt, evidence)
    if attempt.terminal_fill_qty is not None:
        _require_terminal_confirmation(bot, attempt)


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
                "market_order_slippage_fraction": policy.get("market_order_slippage_fraction", 0.0),
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
