from __future__ import annotations

import logging
import math
import sys
import traceback
from collections import defaultdict

from passivbot_exceptions import RestartBotException
from procedures import print_async_exception
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
    side = str(order.get("side") or "").lower()
    pside = str(order.get("position_side") or order.get("positionSide") or "").lower()
    return (pside == "long" and side == "sell") or (pside == "short" and side == "buy")


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
    try:
        return str(_pb_attr("custom_id_to_snake")(custom_id))
    except Exception:
        return ""


def _order_is_panic(order: dict) -> bool:
    return "panic" in _order_pb_type(order)


def _order_is_protective_create(order: dict) -> bool:
    return _order_is_reduce_only(order) or _order_is_panic(order)


async def execute_to_exchange(bot, *, prepare_cycle: bool = True):
    """Run one execution cycle including config sync and order placement/cancellation."""
    if prepare_cycle:
        await bot.execution_cycle()
    to_cancel, to_create = await bot.calc_orders_to_cancel_and_create()
    return await execute_order_plan(bot, to_cancel, to_create)


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

    seen = set()
    for elm in to_cancel:
        key = str(elm["price"]) + str(elm["qty"])
        if key in seen:
            logging.debug("duplicate cancel candidate: %s", elm)
        seen.add(key)

    seen = set()
    for elm in to_create:
        key = str(elm["price"]) + str(elm["qty"])
        if key in seen:
            logging.debug("duplicate create candidate: %s", elm)
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
            logging.info(
                "[balance] too low: %.2f %s; skipped %d exposure-increasing order creates; "
                "allowing %d cancellations and %d protective creates",
                raw_balance,
                bot.quote,
                len(blocked_creates),
                len(to_cancel),
                len(to_create),
            )
    if bot.debug_mode:
        if to_cancel:
            print(
                f"would cancel {len(to_cancel)} order{'s' if len(to_cancel) > 1 else ''}"
            )
    else:
        cancel_started_ms = _utc_ms()
        bot._order_wave_in_progress = order_wave
        try:
            res = await bot.execute_cancellations_parent(to_cancel)
        finally:
            bot._order_wave_in_progress = None
        if order_wave is not None:
            order_wave["cancel_ms"] = int(max(0, _utc_ms() - cancel_started_ms))
            order_wave["cancel_posted"] = len(res or [])
    if bot.debug_mode:
        if to_create:
            print(
                f"would create {len(to_create)} order{'s' if len(to_create) > 1 else ''}"
            )
    else:
        to_create_mod = []
        for order in to_create:
            xf_log = (
                f"{passivbot_cls._log_symbol(order['symbol'])} {order['side']} "
                f"{order['position_side']} {order['qty']} @ {order['price']}"
            )
            if _pb_attr("order_has_match")(order, to_cancel):
                logging.debug(
                    "matching order cancellation found; will be delayed until next cycle: %s",
                    xf_log,
                )
            elif delay_time_ms := bot.order_was_recently_updated(order):
                logging.info(
                    "[order] recent execution found; delaying for up to %.1f secs: %s",
                    delay_time_ms / 1000,
                    xf_log,
                )
            else:
                to_create_mod.append(order)
        if order_wave is not None:
            order_wave["deferred_create"] += max(0, len(to_create) - len(to_create_mod))
        if bot.state_change_detected_by_symbol:
            logging.info(
                "[order] state change detected; skipping order creation for %s until next cycle",
                passivbot_cls._log_symbols(
                    sorted(bot.state_change_detected_by_symbol), limit=12
                ),
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
                return None
            pending_config = sorted(
                set(creation_symbols) - set(configured_symbols or set())
            )
            if pending_config:
                logging.warning(
                    "[config] skipping order creation for symbols pending exchange config: %s",
                    passivbot_cls._log_symbols(pending_config, limit=12),
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
                logging.error(f"error executing orders {to_create_mod} {exc}")
                print_async_exception(res)
                traceback.print_exc()
                await bot.restart_bot_on_too_many_errors()
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
    orders = orders[: int(bot.live_value("max_n_creations_per_batch"))]
    grouped_orders: dict[str, list[dict]] = defaultdict(list)
    emitted_ts = (
        int(bot.get_exchange_time()) if hasattr(bot, "get_exchange_time") else _utc_ms()
    )
    for order in orders:
        bot.add_to_recent_order_executions(order)
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
    res = await bot.execute_orders(orders)
    if not res:
        return
    if len(orders) != len(res):
        print(
            f"debug unequal lengths execute_orders_parent: "
            f"{len(orders)} orders, {len(res)} executions",
            res,
        )
        return []
    to_return = []
    for ex, order in zip(res, orders):
        if not bot.did_create_order(ex):
            if isinstance(ex, Exception):
                passivbot_cls._record_emitted_order_custom_id(
                    bot,
                    order,
                    emitted_ts=emitted_ts,
                    status="create_error_ambiguous",
                )
                logging.debug(
                    "[order] remembered ambiguous create after exchange error | symbol=%s type=%s custom_id=%s error_type=%s",
                    passivbot_cls._log_symbol(order.get("symbol")),
                    bot._resolve_pb_order_type(order),
                    shorten_custom_id(str(order.get("custom_id", ""))),
                    type(ex).__name__,
                )
            print(f"debug did_create_order false {ex}")
            continue
        debug_prints = {}
        for key in order:
            if key not in ex:
                debug_prints.setdefault("missing", []).append((key, order[key]))
                ex[key] = order[key]
            elif ex[key] is None:
                debug_prints.setdefault("is_none", []).append((key, order[key]))
                ex[key] = order[key]
        if debug_prints and bot.debug_mode:
            print("debug create_orders", debug_prints)
        passivbot_cls._record_emitted_order_custom_id(bot, ex, emitted_ts=emitted_ts)
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
        if hasattr(bot, "_request_authoritative_confirmation"):
            bot._request_authoritative_confirmation({"open_orders"})
        else:
            passivbot_cls._request_authoritative_confirmation(bot, {"open_orders"})
    return to_return


async def execute_cancellations_parent(bot, orders: list[dict]) -> list[dict]:
    """Submit a batch of cancellations, prioritising reduce-only orders."""
    passivbot_cls = _pb_attr("Passivbot")
    max_cancellations = int(bot.live_value("max_n_cancellations_per_batch"))
    if len(orders) > max_cancellations:
        try:
            reduce_only_orders = [
                x for x in orders if x.get("reduce_only") or x.get("reduceOnly")
            ]
            rest = [x for x in orders if not x["reduce_only"]]
            orders = (reduce_only_orders + rest)[:max_cancellations]
        except Exception as exc:
            logging.error(f"debug filter cancellations {exc}")
            orders = orders[:max_cancellations]
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
    res = await bot.execute_cancellations(orders)
    to_return = []
    if len(orders) != len(res):
        bot.execution_scheduled = True
        for order in orders:
            bot.state_change_detected_by_symbol.add(order["symbol"])
        print(
            f"debug unequal lengths execute_cancellations_parent: "
            f"{len(orders)} orders, {len(res)} executions",
            res,
        )
        return []
    for ex, order in zip(res, orders):
        if not bot.did_cancel_order(ex, order):
            bot.state_change_detected_by_symbol.add(order["symbol"])
            print(f"debug did_cancel_order false {ex} {order}")
            continue
        ambiguous_terminal_state = (
            bot._cancel_result_requires_full_authoritative_confirmation(ex)
        )
        if ambiguous_terminal_state:
            bot.state_change_detected_by_symbol.add(order["symbol"])
        debug_prints = {}
        for key in order:
            if key not in ex:
                debug_prints.setdefault("missing", []).append((key, order[key]))
                ex[key] = order[key]
            elif ex[key] is None:
                debug_prints.setdefault("is_none", []).append((key, order[key]))
                ex[key] = order[key]
        if debug_prints and bot.debug_mode:
            print("debug cancel_orders", debug_prints)
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
            confirmation_surfaces = bot._authoritative_full_confirmation_surfaces()
            logging.info(
                "[order] ambiguous cancel terminal state; forcing full account confirmation "
                "before next cycle | symbols=%s",
                passivbot_cls._log_symbols(ambiguous_symbols, limit=12),
            )
        if hasattr(bot, "_request_authoritative_confirmation"):
            bot._request_authoritative_confirmation(confirmation_surfaces)
        else:
            passivbot_cls._request_authoritative_confirmation(
                bot, confirmation_surfaces
            )
    return to_return
