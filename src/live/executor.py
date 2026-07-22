from __future__ import annotations

import logging
import math
import sys
import time
from collections import Counter, defaultdict

from passivbot_exceptions import RestartBotException
from live.event_bus import EventTypes, ReasonCodes
from live.fresh_entry_eligibility import FreshEntryEligibilityTrace
from live.order_churn_gate import connector_supports_order_churn_gate
from pure_funcs import shorten_custom_id
from utils import utc_ms as _utils_utc_ms


def _passivbot_module():
    module = sys.modules.get("passivbot")
    if module is None:
        import passivbot as module  # type: ignore
    return module


def _pb_attr(name: str):
    return getattr(_passivbot_module(), name)


def _utc_ms() -> int:
    module = sys.modules.get("passivbot")
    if module is not None and hasattr(module, "utc_ms"):
        return int(module.utc_ms())
    return int(_utils_utc_ms())


def _live_event_console_available(bot, passivbot_cls) -> bool:
    checker = getattr(passivbot_cls, "_live_event_console_available", None)
    return bool(checker(bot)) if callable(checker) else False


def _authoritative_full_confirmation_surfaces(bot, passivbot_cls) -> set[str]:
    getter = getattr(bot, "_authoritative_full_confirmation_surfaces", None)
    if callable(getter):
        return set(getter())
    return set(passivbot_cls._authoritative_full_confirmation_surfaces(bot))


def _request_authoritative_confirmation(bot, passivbot_cls, surfaces) -> None:
    requester = getattr(bot, "_request_authoritative_confirmation", None)
    if callable(requester):
        requester(surfaces)
    else:
        passivbot_cls._request_authoritative_confirmation(bot, surfaces)


def _order_is_reduce_only(order: dict) -> bool:
    if not isinstance(order, dict):
        return False
    for key in ("reduce_only", "reduceOnly"):
        if key in order:
            return bool(order[key])
    info = order.get("info")
    if isinstance(info, dict):
        for key in ("reduceOnly", "reduce_only"):
            if key in info:
                return bool(info[key])
    return False


def _order_pb_type(order: dict) -> str:
    if not isinstance(order, dict):
        return ""
    pb_type = str(order.get("pb_order_type") or "")
    if pb_type:
        return pb_type
    custom_id = str(order.get("custom_id") or "")
    if not custom_id:
        info = order.get("info")
        if isinstance(info, dict):
            custom_id = str(
                info.get("clientOrderId")
                or info.get("clientOid")
                or info.get("clOrdId")
                or ""
            )
    if not custom_id:
        return ""
    return str(_pb_attr("custom_id_to_snake")(custom_id))


def _order_is_panic(order: dict) -> bool:
    return "panic" in _order_pb_type(order)


def _order_is_risk_critical(order: dict) -> bool:
    return str(order.get("execution_priority") or "ordinary").lower() == "risk_critical"


def _is_dedicated_protective_market_panic(
    bot, order: dict, *, configure_creations: bool
) -> bool:
    return (
        not configure_creations
        and _order_is_panic(order)
        and bot._is_market_execution_order(order)
    )


def _order_is_protective_create(order: dict) -> bool:
    return _order_is_reduce_only(order) or _order_is_panic(order)


def _filter_hsl_replay_pending_creates(
    bot, passivbot_cls, orders: list[dict], order_wave
) -> list[dict]:
    pending_pairs = set(
        getattr(bot, "_equity_hard_stop_coin_replay_pending_pairs", set()) or set()
    )
    if not pending_pairs:
        return orders
    blocked = [
        order
        for order in orders
        if not _order_is_protective_create(order)
        and (
            str(order.get("position_side") or order.get("positionSide") or "").lower(),
            str(order.get("symbol") or ""),
        )
        in pending_pairs
    ]
    if not blocked:
        return orders
    blocked_ids = {id(order) for order in blocked}
    if order_wave is not None:
        order_wave["skipped_create"] += len(blocked)
    passivbot_cls._emit_execution_create_filter_event(
        bot,
        event_type=EventTypes.EXECUTION_CREATE_SKIPPED,
        status="skipped",
        reason_code=ReasonCodes.HSL_REPLAY_PENDING,
        order_count=len(blocked),
        symbols=_symbols_from_orders(blocked),
        wave=order_wave,
        message="initial-entry creates skipped until coin HSL replay is ready",
        data={"pending_pairs_count": len(pending_pairs)},
    )
    return [order for order in orders if id(order) not in blocked_ids]


def _symbols_from_orders(orders: list[dict]) -> list[str]:
    return sorted(str(order["symbol"]) for order in orders if order.get("symbol"))


def _orders_removed_by_identity(before: list[dict], after: list[dict]) -> list[dict]:
    remaining = Counter(id(order) for order in after)
    removed = []
    for order in before:
        order_id = id(order)
        if remaining[order_id] > 0:
            remaining[order_id] -= 1
        else:
            removed.append(order)
    return removed


async def _apply_order_churn_final_admission(bot, orders: list[dict]) -> list[dict]:
    if not orders:
        return []
    if not connector_supports_order_churn_gate(bot):
        return orders
    state = getattr(bot, "_order_churn_gate_state", None)
    if state is None:
        return orders
    activation_count = int(bot.live_value("order_replacement_churn_gate_activation_count"))
    max_batch = int(bot.live_value("max_n_creations_per_batch"))
    if max_batch <= 0:
        return []
    prioritized = sorted(
        enumerate(orders),
        key=lambda item: (0 if _order_is_risk_critical(item[1]) else 1, item[0]),
    )
    prioritized_orders = [order for _index, order in prioritized]
    if activation_count <= 0:
        selected = prioritized_orders[:max_batch]
        for order in selected:
            order["_churn_gate_reason"] = "disabled"
        for order in prioritized_orders[max_batch:]:
            order["_churn_gate_reason"] = "batch_capacity"
        emitter = getattr(bot, "_emit_order_churn_admission_event", None)
        if callable(emitter):
            try:
                emitter(
                    orders=prioritized_orders,
                    rolling_count=0,
                    activation_count=activation_count,
                    market_distance_threshold=float(
                        bot.live_value("order_replacement_churn_gate_market_dist_pct")
                    ),
                    action_headroom=None,
                    wave=getattr(bot, "_order_wave_in_progress", None),
                )
            except Exception as exc:
                logging.debug(
                    "[event] order churn admission emitter failed | error_type=%s",
                    type(exc).__name__,
                )
        return selected
    window_seconds = (
        float(bot.live_value("order_replacement_churn_gate_window_minutes")) * 60.0
    )
    now_monotonic = time.monotonic()
    projected_usage = state.create_attempt_count(
        now_monotonic=now_monotonic, window_seconds=window_seconds
    )
    threshold = float(
        bot.live_value("order_replacement_churn_gate_market_dist_pct")
    )
    churn_limit_orders = [
        order
        for order in prioritized_orders
        if bool(order.get("_churn_evidence"))
        and not bot._is_market_execution_order(order)
        and not _order_is_risk_critical(order)
    ]
    market_prices = {}
    if churn_limit_orders:
        try:
            market_prices = await bot._fetch_fresh_order_churn_market_prices(
                {
                    str(order.get("symbol"))
                    for order in churn_limit_orders
                    if order.get("symbol")
                }
            )
        except Exception as exc:
            logging.warning(
                "[order] fresh market data unavailable for churn-gate final admission | error_type=%s",
                type(exc).__name__,
            )
            market_prices = {}
    admission: dict[int, tuple[str, bool]] = {}
    for order in prioritized_orders:
        churn_evidenced = bool(order.get("_churn_evidence"))
        always_allowed = (
            bot._is_market_execution_order(order)
            or _order_is_risk_critical(order)
            or not churn_evidenced
        )
        if always_allowed:
            admission[id(order)] = ("ready", True)
            continue
        market_price = market_prices.get(str(order.get("symbol")))
        if market_price is None:
            admission[id(order)] = ("market_price_unavailable", False)
            continue
        try:
            market_price = float(market_price)
            if not math.isfinite(market_price) or market_price <= 0.0:
                raise ValueError("invalid market price")
            signed_distance = float(
                _pb_attr("order_market_diff")(
                    str(order["side"]), float(order["price"]), market_price
                )
            )
            if not math.isfinite(signed_distance):
                raise ValueError("invalid market distance")
        except (KeyError, TypeError, ValueError):
            admission[id(order)] = ("market_price_invalid", False)
            continue
        order["_churn_gate_market_distance"] = signed_distance
        admission[id(order)] = ("ready", signed_distance <= threshold)
    selected: list[dict] = []
    deferred: list[dict] = []
    capacity_deferred: list[dict] = []
    action_headroom: float | int | None = None
    action_headroom_checked = False
    projected_signed_actions = 0
    for order_index, order in enumerate(prioritized_orders):
        if len(selected) >= max_batch:
            capacity_deferred.append(order)
            order["_churn_gate_reason"] = "batch_capacity"
            continue
        churn_evidenced = bool(order.get("_churn_evidence"))
        admission_status, exempt = admission[id(order)]
        if admission_status != "ready":
            deferred.append(order)
            order["_churn_gate_reason"] = admission_status
            continue
        if (
            activation_count > 0
            and churn_evidenced
            and not exempt
            and projected_usage >= activation_count
        ):
            deferred.append(order)
            order["_churn_gate_reason"] = "allowance_exhausted"
            continue
        if churn_evidenced and not exempt:
            if not action_headroom_checked:
                action_headroom_checked = True
                try:
                    action_headroom = await bot._order_churn_far_create_headroom()
                    if action_headroom is not None:
                        normalized_headroom = float(action_headroom)
                        if math.isnan(normalized_headroom) or normalized_headroom < 0.0:
                            action_headroom = None
                        else:
                            action_headroom = normalized_headroom
                except Exception as exc:
                    action_headroom = None
                    logging.warning(
                        "[order] connector churn headroom unavailable | error_type=%s",
                        type(exc).__name__,
                    )
            if action_headroom is None:
                deferred.append(order)
                order["_churn_gate_reason"] = "action_headroom_unavailable"
                continue
            remaining_slots = max(0, max_batch - len(selected))
            future_always_allowed = sum(
                1
                for future in prioritized_orders[order_index + 1 :]
                if admission[id(future)] == ("ready", True)
            )
            reserved_headroom = min(remaining_slots, future_always_allowed)
            if not math.isinf(float(action_headroom)) and (
                float(action_headroom)
                - projected_signed_actions
                - reserved_headroom
                <= 0.0
            ):
                deferred.append(order)
                order["_churn_gate_reason"] = "action_headroom_exhausted"
                continue
        if not churn_evidenced:
            reason = "no_churn_evidence"
        elif bot._is_market_execution_order(order):
            reason = "market_order_exempt"
        elif _order_is_risk_critical(order):
            reason = "risk_critical_exempt"
        elif exempt:
            reason = "market_distance_exempt"
        else:
            reason = "allowance"
        order["_churn_gate_reason"] = reason
        selected.append(order)
        projected_usage += 1
        projected_signed_actions += 1
    if deferred:
        logging.info(
            "[order] churn gate deferred %d far unstable creates | rolling_usage=%d activation_count=%d",
            len(deferred),
            state.create_attempt_count(
                now_monotonic=now_monotonic, window_seconds=window_seconds
            ),
            activation_count,
        )
        _record_fresh_entry_orders(bot, "record_blocked_orders", deferred, "order_churn_gate")
        reason_codes = {
            "allowance_exhausted": ReasonCodes.ORDER_CHURN_ALLOWANCE_EXHAUSTED,
            "market_price_unavailable": ReasonCodes.ORDER_CHURN_MARKET_DATA_UNAVAILABLE,
            "market_price_invalid": ReasonCodes.ORDER_CHURN_MARKET_DATA_UNAVAILABLE,
            "action_headroom_unavailable": ReasonCodes.ORDER_CHURN_ACTION_HEADROOM_UNAVAILABLE,
            "action_headroom_exhausted": ReasonCodes.ORDER_CHURN_ACTION_HEADROOM_EXHAUSTED,
        }
        for reason in sorted({str(order.get("_churn_gate_reason")) for order in deferred}):
            grouped = [
                order
                for order in deferred
                if str(order.get("_churn_gate_reason")) == reason
            ]
            _pb_attr("Passivbot")._emit_execution_create_filter_event(
                bot,
                event_type=EventTypes.EXECUTION_CREATE_DEFERRED,
                status="deferred",
                reason_code=reason_codes.get(reason, ReasonCodes.ORDER_CHURN_ADMISSION),
                order_count=len(grouped),
                symbols=_symbols_from_orders(grouped),
                wave=getattr(bot, "_order_wave_in_progress", None),
                data={
                    "rolling_count": state.create_attempt_count(
                        now_monotonic=now_monotonic, window_seconds=window_seconds
                    ),
                    "activation_count": activation_count,
                    "market_distance_threshold_pct": threshold * 100.0,
                },
            )
    if capacity_deferred:
        _record_fresh_entry_orders(
            bot, "record_blocked_orders", capacity_deferred, "batch_capacity"
        )
        _pb_attr("Passivbot")._emit_execution_create_filter_event(
            bot,
            event_type=EventTypes.EXECUTION_CREATE_DEFERRED,
            status="deferred",
            reason_code=ReasonCodes.BATCH_CAPACITY,
            order_count=len(capacity_deferred),
            symbols=_symbols_from_orders(capacity_deferred),
            wave=getattr(bot, "_order_wave_in_progress", None),
            data={"max_n_creations_per_batch": max_batch},
        )
    emitter = getattr(bot, "_emit_order_churn_admission_event", None)
    if callable(emitter):
        try:
            emitter(
                orders=prioritized_orders,
                rolling_count=state.create_attempt_count(
                    now_monotonic=now_monotonic, window_seconds=window_seconds
                ),
                activation_count=activation_count,
                market_distance_threshold=threshold,
                action_headroom=action_headroom if action_headroom_checked else None,
                wave=getattr(bot, "_order_wave_in_progress", None),
            )
        except Exception as exc:
            logging.debug(
                "[event] order churn admission emitter failed | error_type=%s",
                type(exc).__name__,
            )
    return selected


def _fresh_entry_trace(bot) -> FreshEntryEligibilityTrace | None:
    trace = getattr(bot, "_fresh_entry_eligibility_trace", None)
    return trace if isinstance(trace, FreshEntryEligibilityTrace) else None


def _record_fresh_entry_orders(
    bot, method: str, orders: list[dict], reason: str | None = None
) -> None:
    """Record one existing executor decision without allowing diagnostics to affect it."""
    trace = _fresh_entry_trace(bot)
    if trace is None:
        return
    try:
        recorder = getattr(trace, method)
        if reason is None:
            recorder(orders)
        else:
            recorder(orders, reason)
    except Exception as exc:
        bot._fresh_entry_eligibility_trace = None
        logging.debug(
            "[entry] fresh-entry eligibility trace disabled during execution | "
            "method=%s error_type=%s",
            method,
            type(exc).__name__,
        )


def _emit_fresh_entry_eligibility(bot, passivbot_cls, wave) -> None:
    """Consume and emit the cycle trace exactly once on a completed local plan."""
    trace = _fresh_entry_trace(bot)
    bot._fresh_entry_eligibility_trace = None
    if trace is None:
        return
    try:
        data = trace.to_event_data()
    except Exception as exc:
        logging.debug(
            "[entry] fresh-entry eligibility payload build failed | error_type=%s",
            type(exc).__name__,
        )
        return
    try:
        passivbot_cls._emit_initial_entry_eligibility_event(bot, data=data, wave=wave)
    except Exception as exc:
        logging.debug(
            "[entry] fresh-entry eligibility event emission failed | error_type=%s",
            type(exc).__name__,
        )


def _create_rejection_reason(result) -> str:
    if isinstance(result, BaseException):
        return "result_exception"
    if not isinstance(result, dict):
        return "create_not_acknowledged"
    status = str(result.get("status") or "").lower()
    info = result.get("info")
    info_status = ""
    if isinstance(info, dict):
        info_status = str(
            info.get("status") or info.get("ordStatus") or info.get("state") or ""
        ).lower()
    terminal = {
        "canceled",
        "cancelled",
        "closed",
        "expired",
        "failed",
        "rejected",
    }
    if status in terminal or info_status in terminal:
        return "terminal_rejection"
    if not (result.get("id") or result.get("order_id")):
        return "missing_exchange_order_id"
    return "create_not_acknowledged"


def _remember_ambiguous_create(
    bot,
    passivbot_cls,
    order: dict,
    emitted_ts: int,
    *,
    status: str,
    reason: str,
    error: BaseException | None = None,
) -> None:
    bot.add_to_recent_order_executions(order)
    passivbot_cls._record_emitted_order_custom_id(
        bot,
        order,
        emitted_ts=emitted_ts,
        status=status,
    )
    logging.debug(
        "[order] remembered ambiguous create | symbol=%s type=%s custom_id=%s status=%s reason=%s error_type=%s",
        passivbot_cls._log_symbol(order.get("symbol")),
        bot._resolve_pb_order_type(order),
        shorten_custom_id(str(order.get("custom_id", ""))),
        status,
        reason,
        type(error).__name__ if error is not None else "",
    )


async def execute_to_exchange(bot, *, prepare_cycle: bool = True):
    """Run one execution cycle including config sync and order placement/cancellation."""
    bot._fresh_entry_eligibility_trace = None
    try:
        if prepare_cycle:
            await bot.execution_cycle()
        to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()
        return await execute_order_plan(bot, to_cancel, to_create)
    finally:
        bot._fresh_entry_eligibility_trace = None


async def execute_order_plan(
    bot,
    to_cancel: list[dict],
    to_create: list[dict],
    *,
    configure_creations: bool = True,
):
    """Execute a precomputed order plan against the exchange."""
    passivbot_cls = _pb_attr("Passivbot")
    order_wave = passivbot_cls._begin_order_wave(bot, to_cancel, to_create)
    cancel_first_barrier = (
        bool(to_cancel)
        and not bot.debug_mode
        and connector_supports_order_churn_gate(bot)
    )

    seen = set()
    for elm in to_cancel:
        key = str(elm["price"]) + str(elm["qty"])
        if key in seen:
            logging.debug(
                "[order] duplicate cancel candidate | symbol=%s type=%s",
                passivbot_cls._log_symbol(elm.get("symbol")),
                _order_pb_type(elm),
            )
        seen.add(key)

    seen = set()
    for elm in to_create:
        key = str(elm["price"]) + str(elm["qty"])
        if key in seen:
            logging.debug(
                "[order] duplicate create candidate | symbol=%s type=%s",
                passivbot_cls._log_symbol(elm.get("symbol")),
                _order_pb_type(elm),
            )
        seen.add(key)
    low_balance = False
    if not bot.debug_mode:
        raw_balance = float(bot.get_raw_balance())
        if not math.isfinite(raw_balance):
            raise RuntimeError(
                f"invalid raw balance for order execution: {raw_balance!r}"
            )
        balance_threshold = float(getattr(bot, "balance_threshold", 0.0) or 0.0)
        low_balance = raw_balance < balance_threshold
        if low_balance and (to_cancel or to_create):
            blocked_creates = [
                order for order in to_create if not _order_is_protective_create(order)
            ]
            to_create = [
                order for order in to_create if _order_is_protective_create(order)
            ]
            if order_wave is not None:
                order_wave["skipped_create"] += len(blocked_creates)
            if blocked_creates:
                _record_fresh_entry_orders(
                    bot, "record_blocked_orders", blocked_creates, "low_balance"
                )
                passivbot_cls._emit_execution_create_filter_event(
                    bot,
                    event_type=EventTypes.EXECUTION_CREATE_SKIPPED,
                    status="skipped",
                    reason_code=ReasonCodes.LOW_BALANCE,
                    order_count=len(blocked_creates),
                    symbols=_symbols_from_orders(blocked_creates),
                    wave=order_wave,
                    message="exposure-increasing creates skipped because balance is below threshold",
                    data={
                        "raw_balance": raw_balance,
                        "balance_threshold": balance_threshold,
                        "quote": bot.quote,
                        "allowed_cancel": len(to_cancel),
                        "allowed_protective_create": len(to_create),
                    },
                )
            if not _live_event_console_available(bot, passivbot_cls):
                logging.info(
                    "[balance] too low: %.2f %s; skipped %d exposure-increasing order creates; "
                    "allowing %d cancellations and %d protective creates",
                    raw_balance,
                    bot.quote,
                    len(blocked_creates),
                    len(to_cancel),
                    len(to_create),
                )
    before_hsl_filter = list(to_create)
    to_create = _filter_hsl_replay_pending_creates(
        bot, passivbot_cls, to_create, order_wave
    )
    _record_fresh_entry_orders(
        bot,
        "record_blocked_orders",
        _orders_removed_by_identity(before_hsl_filter, to_create),
        "hsl_replay_pending",
    )
    if bot.debug_mode:
        if to_cancel:
            logging.info(
                "[order] debug mode would cancel orders | count=%d", len(to_cancel)
            )
    else:
        cancel_started_ms = _utc_ms()
        bot._order_wave_in_progress = order_wave
        try:
            res = await bot.execute_cancellations_parent(to_cancel)
        finally:
            if cancel_first_barrier:
                _request_authoritative_confirmation(
                    bot,
                    passivbot_cls,
                    _authoritative_full_confirmation_surfaces(bot, passivbot_cls),
                )
            bot._order_wave_in_progress = None
        if order_wave is not None:
            order_wave["cancel_ms"] = int(max(0, _utc_ms() - cancel_started_ms))
            order_wave["cancel_posted"] = len(res or [])
    if cancel_first_barrier:
        barrier_deferred = [
            order
            for order in to_create
            if not _is_dedicated_protective_market_panic(
                bot, order, configure_creations=configure_creations
            )
        ]
        bypass_creates = [
            order
            for order in to_create
            if _is_dedicated_protective_market_panic(
                bot, order, configure_creations=configure_creations
            )
        ]
        _record_fresh_entry_orders(
            bot,
            "record_blocked_orders",
            barrier_deferred,
            "account_cancel_first_barrier",
        )
        if order_wave is not None:
            order_wave["deferred_create"] += len(barrier_deferred)
        if barrier_deferred:
            logging.info(
                "[order] account-wide cancel-first barrier deferred %d creates until full confirmation and replanning",
                len(barrier_deferred),
            )
            passivbot_cls._emit_execution_create_filter_event(
                bot,
                event_type=EventTypes.EXECUTION_CANCEL_FIRST_BARRIER,
                status="deferred",
                reason_code=ReasonCodes.ACCOUNT_CANCEL_FIRST_BARRIER,
                order_count=len(barrier_deferred),
                symbols=_symbols_from_orders(barrier_deferred),
                wave=order_wave,
                data={
                    "cancel_count": len(to_cancel),
                    "dedicated_market_panic_bypass_count": len(bypass_creates),
                    "required_surfaces": sorted(
                        _authoritative_full_confirmation_surfaces(bot, passivbot_cls)
                    ),
                },
            )
        to_create = bypass_creates
    if bot.debug_mode:
        if to_create:
            _record_fresh_entry_orders(
                bot, "record_blocked_orders", to_create, "debug_mode"
            )
            logging.info(
                "[order] debug mode would create orders | count=%d", len(to_create)
            )
    else:
        to_create_mod = []
        recent_execution_deferred: list[tuple[dict, float]] = []
        for order in to_create:
            xf_log = (
                f"{passivbot_cls._log_symbol(order['symbol'])} {order['side']} "
                f"{order['position_side']} {order['qty']} @ {order['price']}"
            )
            if delay_time_ms := bot.order_was_recently_updated(order):
                recent_execution_deferred.append((order, float(delay_time_ms)))
                logging.info(
                    "[order] recent execution found; delaying for up to %.1f secs: %s",
                    delay_time_ms / 1000,
                    xf_log,
                )
            else:
                to_create_mod.append(order)
        if recent_execution_deferred:
            _record_fresh_entry_orders(
                bot,
                "record_blocked_orders",
                [order for order, _delay in recent_execution_deferred],
                "recent_execution",
            )
            passivbot_cls._emit_execution_create_filter_event(
                bot,
                event_type=EventTypes.EXECUTION_CREATE_DEFERRED,
                status="deferred",
                reason_code=ReasonCodes.RECENT_EXECUTION,
                order_count=len(recent_execution_deferred),
                symbols=_symbols_from_orders(
                    [order for order, _delay in recent_execution_deferred]
                ),
                wave=order_wave,
                message="create orders deferred because a matching recent execution exists",
                data={
                    "max_delay_ms": int(
                        max(delay for _order, delay in recent_execution_deferred)
                    ),
                    "delays_sample_ms": [
                        int(delay)
                        for _order, delay in recent_execution_deferred[:8]
                    ],
                },
            )
        if order_wave is not None:
            order_wave["deferred_create"] += max(0, len(to_create) - len(to_create_mod))
        if bot.state_change_detected_by_symbol:
            state_filtered_orders = [
                order
                for order in to_create_mod
                if order["symbol"] in bot.state_change_detected_by_symbol
            ]
            logging.info(
                "[order] state change detected; skipping order creation for %s until next cycle",
                passivbot_cls._log_symbols(
                    sorted(bot.state_change_detected_by_symbol), limit=12
                ),
            )
            if state_filtered_orders:
                _record_fresh_entry_orders(
                    bot,
                    "record_blocked_orders",
                    state_filtered_orders,
                    "state_change_detected",
                )
                passivbot_cls._emit_execution_create_filter_event(
                    bot,
                    event_type=EventTypes.EXECUTION_CREATE_SKIPPED,
                    status="skipped",
                    reason_code=ReasonCodes.STATE_CHANGE_DETECTED,
                    order_count=len(state_filtered_orders),
                    symbols=_symbols_from_orders(state_filtered_orders),
                    wave=order_wave,
                    message="create orders skipped until authoritative state settles",
                    data={
                        "blocked_symbols_count": len(
                            bot.state_change_detected_by_symbol
                        )
                    },
                )
            before_state_filter = len(to_create_mod)
            to_create_mod = [
                order
                for order in to_create_mod
                if order["symbol"] not in bot.state_change_detected_by_symbol
            ]
            if order_wave is not None:
                order_wave["skipped_create"] += max(
                    0, before_state_filter - len(to_create_mod)
                )
        if to_create_mod and configure_creations:
            creation_symbols = sorted({order["symbol"] for order in to_create_mod})
            configured_symbols = await bot.update_exchange_configs(creation_symbols)
            if bot._shutdown_requested():
                bot._order_wave_in_progress = None
                bot._fresh_entry_eligibility_trace = None
                return None
            pending_config = sorted(
                set(creation_symbols) - set(configured_symbols or set())
            )
            if pending_config:
                pending_config_orders = [
                    order for order in to_create_mod if order["symbol"] in pending_config
                ]
                logging.warning(
                    "[config] skipping order creation for symbols pending exchange config: %s",
                    passivbot_cls._log_symbols(pending_config, limit=12),
                )
                if pending_config_orders:
                    _record_fresh_entry_orders(
                        bot,
                        "record_blocked_orders",
                        pending_config_orders,
                        "pending_exchange_config",
                    )
                    passivbot_cls._emit_execution_create_filter_event(
                        bot,
                        event_type=EventTypes.EXECUTION_CREATE_SKIPPED,
                        status="skipped",
                        reason_code=ReasonCodes.PENDING_EXCHANGE_CONFIG,
                        order_count=len(pending_config_orders),
                        symbols=_symbols_from_orders(pending_config_orders),
                        wave=order_wave,
                        level="warning",
                        message="create orders skipped while exchange config update is pending",
                        data={
                            "configured_symbols_count": len(configured_symbols or set()),
                            "pending_symbols_count": len(pending_config),
                        },
                    )
                before_config_filter = len(to_create_mod)
                to_create_mod = [
                    order
                    for order in to_create_mod
                    if order["symbol"] not in pending_config
                ]
                if order_wave is not None:
                    order_wave["skipped_create"] += max(
                        0, before_config_filter - len(to_create_mod)
                    )
        elif to_create_mod:
            logging.info(
                "[risk] executing protective order plan without exchange config sync | symbols=%s orders=%d",
                passivbot_cls._log_symbols(
                    sorted({order["symbol"] for order in to_create_mod}), limit=12
                ),
                len(to_create_mod),
            )
        before_market_filter = len(to_create_mod)
        to_create_mod = await passivbot_cls._filter_fresh_market_snapshot_creations(
            bot, to_create_mod
        )
        if order_wave is not None:
            order_wave["skipped_create"] += max(
                0, before_market_filter - len(to_create_mod)
            )
        before_churn_admission = list(to_create_mod)
        to_create_mod = await _apply_order_churn_final_admission(bot, to_create_mod)
        if order_wave is not None:
            order_wave["deferred_create"] += max(
                0, len(before_churn_admission) - len(to_create_mod)
            )
        if to_create_mod:
            res = None
            try:
                create_started_ms = _utc_ms()
                bot._order_wave_in_progress = order_wave
                try:
                    res = await bot.execute_orders_parent(to_create_mod)
                finally:
                    bot._order_wave_in_progress = None
                if order_wave is not None:
                    order_wave["create_ms"] = int(max(0, _utc_ms() - create_started_ms))
                    order_wave["create_posted"] = len(res or [])
            except RestartBotException:
                raise
            except Exception as exc:
                if not _live_event_console_available(bot, passivbot_cls):
                    logging.error(
                        "[order] create batch raised before completion | count=%d error_type=%s",
                        len(to_create_mod),
                        type(exc).__name__,
                    )
                await bot.restart_bot_on_too_many_errors()
    _emit_fresh_entry_eligibility(bot, passivbot_cls, order_wave)
    if to_cancel or to_create:
        bot.execution_scheduled = True
    if not passivbot_cls._shutdown_requested(bot):
        schedule_forager_refresh = getattr(
            bot, "_schedule_forager_candidate_candle_refresh", None
        )
        if callable(schedule_forager_refresh):
            schedule_forager_refresh()
    passivbot_cls._track_order_wave_confirmation(bot, order_wave)
    passivbot_cls._log_order_wave_summary(bot, order_wave)
    if bot.debug_mode:
        return to_cancel, to_create


async def execute_orders_parent(bot, orders: list[dict]) -> list[dict]:
    """Submit a batch of orders after throttling and bookkeeping."""
    passivbot_cls = _pb_attr("Passivbot")
    requested_orders = list(orders)
    orders = requested_orders[: int(bot.live_value("max_n_creations_per_batch"))]
    _record_fresh_entry_orders(
        bot,
        "record_blocked_orders",
        _orders_removed_by_identity(requested_orders, orders),
        "batch_capacity",
    )
    grouped_orders: dict[str, list[dict]] = defaultdict(list)
    emitted_ts = (
        int(bot.get_exchange_time()) if hasattr(bot, "get_exchange_time") else _utc_ms()
    )
    for order in orders:
        passivbot_cls._record_emitted_order_custom_id(
            bot, order, emitted_ts=emitted_ts, status="submitted"
        )
        bot.log_order_action(
            order,
            "posting order",
            context=order.get("_context", "plan_sync"),
            level=logging.DEBUG,
            delta=order.get("_delta"),
        )
        if bot._is_market_execution_order(order):
            bot._log_market_execution_notice(
                order, context=order.get("_context", "plan_sync")
            )
        grouped_orders[order["symbol"]].append(order)
    bot._log_order_action_summary(grouped_orders, "post")
    wave = getattr(bot, "_order_wave_in_progress", None)
    for idx, order in enumerate(orders):
        passivbot_cls._emit_execution_order_event(
            bot,
            event_type=EventTypes.EXECUTION_CREATE_SENT,
            order=order,
            action="create",
            status="started",
            reason_code=ReasonCodes.SUBMITTED_TO_EXCHANGE,
            index=idx,
            wave=wave,
        )
    _record_fresh_entry_orders(bot, "record_eligible_orders", orders)
    _emit_fresh_entry_eligibility(bot, passivbot_cls, wave)
    connector_call_context = {
        "action": "create",
        "orders": orders,
        "wave": wave,
    }
    bot._execution_connector_call_context = connector_call_context
    churn_state = getattr(bot, "_order_churn_gate_state", None)
    connector_enabled = connector_supports_order_churn_gate(bot)
    churn_activation_count = (
        int(bot.live_value("order_replacement_churn_gate_activation_count"))
        if churn_state is not None and connector_enabled
        else 0
    )
    if churn_activation_count > 0:
        now_monotonic = time.monotonic()
        churn_state.record_create_attempts(
            len(orders), now_monotonic=now_monotonic
        )
        window_seconds = (
            float(bot.live_value("order_replacement_churn_gate_window_minutes"))
            * 60.0
        )
        emitter = getattr(bot, "_emit_order_churn_actions_accounted_event", None)
        if callable(emitter):
            try:
                emitter(
                    action_count=len(orders),
                    rolling_count=churn_state.create_attempt_count(
                        now_monotonic=now_monotonic, window_seconds=window_seconds
                    ),
                    wave=wave,
                )
            except Exception as exc:
                logging.debug(
                    "[event] order churn accounting emitter failed | error_type=%s",
                    type(exc).__name__,
                )
    record_signed_actions = getattr(
        bot, "_record_order_churn_signed_action_attempts", None
    )
    if callable(record_signed_actions):
        record_signed_actions(len(orders))
    try:
        res = await bot.execute_orders(orders)
    except RestartBotException:
        raise
    except Exception as exc:
        for idx, order in enumerate(orders):
            passivbot_cls._emit_execution_order_event(
                bot,
                event_type=EventTypes.EXECUTION_AMBIGUOUS,
                order=order,
                action="create",
                status="degraded",
                reason_code=ReasonCodes.EXCHANGE_EXCEPTION,
                level="warning",
                index=idx,
                wave=wave,
                error=exc,
            )
            _remember_ambiguous_create(
                bot,
                passivbot_cls,
                order,
                emitted_ts,
                status="create_error_ambiguous",
                reason=ReasonCodes.EXCHANGE_EXCEPTION,
                error=exc,
            )
        raise
    finally:
        if (
            getattr(bot, "_execution_connector_call_context", None)
            is connector_call_context
        ):
            bot._execution_connector_call_context = None
    if not res:
        for idx, order in enumerate(orders):
            passivbot_cls._emit_execution_order_event(
                bot,
                event_type=EventTypes.EXECUTION_AMBIGUOUS,
                order=order,
                action="create",
                status="degraded",
                reason_code="empty_response",
                level="warning",
                index=idx,
                wave=wave,
                extra={"response_count": 0, "request_count": len(orders)},
            )
            _remember_ambiguous_create(
                bot,
                passivbot_cls,
                order,
                emitted_ts,
                status="create_response_missing",
                reason="empty_response",
            )
        return []
    if len(orders) != len(res):
        for idx, order in enumerate(orders):
            passivbot_cls._emit_execution_order_event(
                bot,
                event_type=EventTypes.EXECUTION_AMBIGUOUS,
                order=order,
                action="create",
                status="degraded",
                reason_code=ReasonCodes.LENGTH_MISMATCH,
                level="warning",
                index=idx,
                wave=wave,
                extra={"response_count": len(res), "request_count": len(orders)},
            )
            _remember_ambiguous_create(
                bot,
                passivbot_cls,
                order,
                emitted_ts,
                status="create_response_partial",
                reason=ReasonCodes.LENGTH_MISMATCH,
            )
        if not _live_event_console_available(bot, passivbot_cls):
            logging.warning(
                "[order] create response count mismatch | requested=%d returned=%d",
                len(orders),
                len(res),
            )
        return []
    to_return = []
    for idx, (ex, order) in enumerate(zip(res, orders)):
        if not bot.did_create_order(ex):
            if isinstance(ex, Exception):
                reason_code = "result_exception"
                passivbot_cls._emit_execution_order_event(
                    bot,
                    event_type=EventTypes.EXECUTION_AMBIGUOUS,
                    order=order,
                    action="create",
                    status="degraded",
                    reason_code=reason_code,
                    level="warning",
                    index=idx,
                    wave=wave,
                    result=ex,
                )
                _remember_ambiguous_create(
                    bot,
                    passivbot_cls,
                    order,
                    emitted_ts=emitted_ts,
                    status="create_error_ambiguous",
                    reason=reason_code,
                    error=ex,
                )
            else:
                reason_code = _create_rejection_reason(ex)
                event_type = (
                    EventTypes.EXECUTION_CREATE_REJECTED
                    if reason_code == "terminal_rejection"
                    else EventTypes.EXECUTION_CREATE_FAILED
                )
                passivbot_cls._emit_execution_order_event(
                    bot,
                    event_type=event_type,
                    order=order,
                    action="create",
                    status="failed",
                    reason_code=reason_code,
                    level="warning",
                    index=idx,
                    wave=wave,
                    result=ex if isinstance(ex, dict) else None,
                )
            if not _live_event_console_available(bot, passivbot_cls):
                logging.warning(
                    "[order] create not acknowledged | symbol=%s type=%s reason=%s error_type=%s",
                    passivbot_cls._log_symbol(order.get("symbol")),
                    bot._resolve_pb_order_type(order),
                    reason_code,
                    type(ex).__name__ if isinstance(ex, BaseException) else "",
                )
            continue
        normalized_fields: dict[str, list[str]] = {}
        for key in order:
            if key not in ex:
                normalized_fields.setdefault("missing", []).append(str(key))
                ex[key] = order[key]
            elif ex[key] is None:
                normalized_fields.setdefault("is_none", []).append(str(key))
                ex[key] = order[key]
        if normalized_fields and bot.debug_mode:
            logging.debug(
                "[order] normalized create response fields | missing_keys=%s none_keys=%s",
                sorted(normalized_fields.get("missing", []))[:12],
                sorted(normalized_fields.get("is_none", []))[:12],
            )
        passivbot_cls._record_emitted_order_custom_id(bot, ex, emitted_ts=emitted_ts)
        bot.add_to_recent_order_executions(ex)
        passivbot_cls._emit_execution_order_event(
            bot,
            event_type=EventTypes.EXECUTION_CREATE_SUCCEEDED,
            order=order,
            action="create",
            status="succeeded",
            reason_code=ReasonCodes.EXCHANGE_ACKNOWLEDGED,
            index=idx,
            wave=wave,
            result=ex,
        )
        to_return.append(ex)
    if to_return:
        for elm in to_return:
            bot.add_new_order(elm, source="POST")
            bot._monitor_record_event(
                "order.opened",
                ("order", "open"),
                bot._monitor_order_payload(elm, source="POST"),
                symbol=elm.get("symbol"),
                pside=elm.get("position_side"),
            )
        bot._health_orders_placed += len(to_return)
        _request_authoritative_confirmation(bot, passivbot_cls, {"open_orders"})
    return to_return


async def execute_cancellations_parent(bot, orders: list[dict]) -> list[dict]:
    """Submit a batch of cancellations, prioritising reduce-only orders."""
    passivbot_cls = _pb_attr("Passivbot")
    max_cancellations = int(bot.live_value("max_n_cancellations_per_batch"))
    requested_orders = list(orders)
    if len(orders) > max_cancellations:
        try:
            reduce_only_orders = [x for x in orders if _order_is_reduce_only(x)]
            rest = [x for x in orders if not _order_is_reduce_only(x)]
            orders = (reduce_only_orders + rest)[:max_cancellations]
        except Exception as exc:
            logging.error(
                "[order] cancellation priority filtering failed; using input order | error_type=%s",
                type(exc).__name__,
            )
            orders = orders[:max_cancellations]
        deferred = _orders_removed_by_identity(requested_orders, orders)
        emitter = getattr(bot, "_emit_execution_cancel_filter_event", None)
        if callable(emitter):
            try:
                emitter(
                    event_type=EventTypes.EXECUTION_CANCEL_DEFERRED,
                    status="deferred",
                    reason_code=ReasonCodes.CANCEL_BATCH_CAPACITY,
                    order_count=len(deferred),
                    symbols=_symbols_from_orders(deferred),
                    wave=getattr(bot, "_order_wave_in_progress", None),
                    data={
                        "requested_count": len(requested_orders),
                        "submitted_count": len(orders),
                        "max_n_cancellations_per_batch": max_cancellations,
                    },
                )
            except Exception as exc:
                logging.debug(
                    "[event] cancellation-capacity emitter failed | error_type=%s",
                    type(exc).__name__,
                )
    grouped_orders: dict[str, list[dict]] = defaultdict(list)
    for order in orders:
        bot.add_to_recent_order_cancellations(order)
        bot.log_order_action(
            order,
            "cancelling order",
            context=order.get("_context", "plan_sync"),
            level=logging.DEBUG,
            delta=order.get("_delta"),
        )
        grouped_orders[order["symbol"]].append(order)
    bot._log_order_action_summary(grouped_orders, "cancel")
    wave = getattr(bot, "_order_wave_in_progress", None)
    for idx, order in enumerate(orders):
        passivbot_cls._emit_execution_order_event(
            bot,
            event_type=EventTypes.EXECUTION_CANCEL_SENT,
            order=order,
            action="cancel",
            status="started",
            reason_code=ReasonCodes.SUBMITTED_TO_EXCHANGE,
            index=idx,
            wave=wave,
        )
    connector_call_context = {
        "action": "cancel",
        "orders": orders,
        "wave": wave,
    }
    bot._execution_connector_call_context = connector_call_context
    record_signed_actions = getattr(
        bot, "_record_order_churn_signed_action_attempts", None
    )
    if callable(record_signed_actions):
        record_signed_actions(len(orders))
    try:
        res = await bot.execute_cancellations(orders)
    except RestartBotException:
        raise
    except Exception as exc:
        for idx, order in enumerate(orders):
            passivbot_cls._emit_execution_order_event(
                bot,
                event_type=EventTypes.EXECUTION_CANCEL_FAILED,
                order=order,
                action="cancel",
                status="failed",
                reason_code=ReasonCodes.EXCHANGE_EXCEPTION,
                level="warning",
                index=idx,
                wave=wave,
                error=exc,
            )
        raise
    finally:
        if (
            getattr(bot, "_execution_connector_call_context", None)
            is connector_call_context
        ):
            bot._execution_connector_call_context = None
    to_return = []
    if len(orders) != len(res):
        bot.execution_scheduled = True
        for idx, order in enumerate(orders):
            passivbot_cls._emit_execution_order_event(
                bot,
                event_type=EventTypes.EXECUTION_AMBIGUOUS,
                order=order,
                action="cancel",
                status="degraded",
                reason_code=ReasonCodes.LENGTH_MISMATCH,
                level="warning",
                index=idx,
                wave=wave,
                extra={"response_count": len(res), "request_count": len(orders)},
            )
            bot.state_change_detected_by_symbol.add(order["symbol"])
        if not _live_event_console_available(bot, passivbot_cls):
            logging.warning(
                "[order] cancel response count mismatch | requested=%d returned=%d",
                len(orders),
                len(res),
            )
        return []
    for idx, (ex, order) in enumerate(zip(res, orders)):
        if not bot.did_cancel_order(ex, order):
            bot.state_change_detected_by_symbol.add(order["symbol"])
            passivbot_cls._emit_execution_order_event(
                bot,
                event_type=EventTypes.EXECUTION_CANCEL_FAILED,
                order=order,
                action="cancel",
                status="failed",
                reason_code="cancel_not_acknowledged",
                level="warning",
                index=idx,
                wave=wave,
                result=ex if isinstance(ex, dict) else None,
                error=ex if isinstance(ex, BaseException) else None,
            )
            if not _live_event_console_available(bot, passivbot_cls):
                logging.warning(
                    "[order] cancel not acknowledged | symbol=%s type=%s error_type=%s",
                    passivbot_cls._log_symbol(order.get("symbol")),
                    bot._resolve_pb_order_type(order),
                    type(ex).__name__ if isinstance(ex, BaseException) else "",
                )
            continue
        ambiguous_terminal_state = (
            bot._cancel_result_requires_full_authoritative_confirmation(ex)
        )
        if ambiguous_terminal_state:
            bot.state_change_detected_by_symbol.add(order["symbol"])
        normalized_fields: dict[str, list[str]] = {}
        for key in order:
            if key not in ex:
                normalized_fields.setdefault("missing", []).append(str(key))
                ex[key] = order[key]
            elif ex[key] is None:
                normalized_fields.setdefault("is_none", []).append(str(key))
                ex[key] = order[key]
        if normalized_fields and bot.debug_mode:
            logging.debug(
                "[order] normalized cancel response fields | missing_keys=%s none_keys=%s",
                sorted(normalized_fields.get("missing", []))[:12],
                sorted(normalized_fields.get("is_none", []))[:12],
            )
        if ambiguous_terminal_state:
            passivbot_cls._emit_execution_order_event(
                bot,
                event_type=EventTypes.EXECUTION_CANCEL_AMBIGUOUS_TERMINAL,
                order=order,
                action="cancel",
                status="degraded",
                reason_code="requires_full_authoritative_confirmation",
                level="warning",
                index=idx,
                wave=wave,
                result=ex,
            )
        else:
            passivbot_cls._emit_execution_order_event(
                bot,
                event_type=EventTypes.EXECUTION_CANCEL_SUCCEEDED,
                order=order,
                action="cancel",
                status="succeeded",
                reason_code=ReasonCodes.EXCHANGE_ACKNOWLEDGED,
                index=idx,
                wave=wave,
                result=ex,
            )
        to_return.append(ex)
    if to_return:
        for elm in to_return:
            bot.remove_order(elm, source="POST")
            bot._monitor_record_event(
                "order.canceled",
                ("order", "cancel"),
                bot._monitor_order_payload(elm, source="POST"),
                symbol=elm.get("symbol"),
                pside=elm.get("position_side"),
            )
        bot._health_orders_cancelled += len(to_return)
        confirmation_surfaces = {"open_orders"}
        ambiguous_symbols = sorted(
            {
                elm.get("symbol")
                for elm in to_return
                if bot._cancel_result_requires_full_authoritative_confirmation(elm)
                and elm.get("symbol")
            }
        )
        if ambiguous_symbols:
            confirmation_surfaces = _authoritative_full_confirmation_surfaces(
                bot, passivbot_cls
            )
            if not (
                _live_event_console_available(bot, passivbot_cls)
                and getattr(getattr(bot, "_live_event_pipeline", None), "console_sink", None)
                is not None
            ):
                logging.info(
                    "[order] ambiguous cancel terminal state; forcing full account confirmation "
                    "before next cycle | symbols=%s",
                    passivbot_cls._log_symbols(ambiguous_symbols, limit=12),
                )
        _request_authoritative_confirmation(
            bot, passivbot_cls, confirmation_surfaces
        )
    return to_return
