from __future__ import annotations

import json
import logging
import math
import sys
import time
from collections import Counter
from typing import Iterable, Optional

from live.diagnostic_safety import bounded_exception_type
from live.fresh_entry_eligibility import FreshEntryEligibilityTrace
from live.freshness import ACCOUNT_SURFACES
from live.order_churn_gate import (
    ChurnDecision,
    OrderChurnGateState,
    connector_supports_order_churn_gate,
    deterministic_one_to_one_matches,
    normalize_ideal_orders,
)
from passivbot_exceptions import FatalBotException
from pure_funcs import determine_side_from_order_tuple, filter_orders, shorten_custom_id
from utils import symbol_to_coin, ts_to_date, utc_ms as _utils_utc_ms


def _passivbot_module():
    module = sys.modules.get("passivbot")
    if module is None:
        import passivbot as module  # type: ignore
    return module


def _pb_attr(name: str):
    return getattr(_passivbot_module(), name)


def _pb_const(name: str):
    return getattr(_passivbot_module(), name)


def _utc_ms() -> int:
    module = sys.modules.get("passivbot")
    if module is not None and hasattr(module, "utc_ms"):
        return int(module.utc_ms())
    return int(_utils_utc_ms())


def _order_churn_max_generation_gap_seconds(bot) -> float:
    """Return a cadence gap which includes the execution loop's quiet wait."""
    execution_delay = float(bot.live_value("execution_delay_seconds"))
    scheduled_wait = float(_pb_const("EXECUTION_SCHEDULED_WAIT_SECONDS"))
    return max(10.0, 3.0 * (execution_delay + scheduled_wait))


def _orders_removed_by_identity(before: list[dict], after: list[dict]) -> list[dict]:
    """Return objects removed by an existing filter without comparing mutable payloads."""
    remaining = Counter(id(order) for order in after)
    removed = []
    for order in before:
        order_id = id(order)
        if remaining[order_id] > 0:
            remaining[order_id] -= 1
        else:
            removed.append(order)
    return removed


def _trace_record(
    trace: FreshEntryEligibilityTrace | None,
    method: str,
    *args,
    **kwargs,
) -> FreshEntryEligibilityTrace | None:
    """Best-effort diagnostic recording which can never affect order reconciliation."""
    if trace is None:
        return None
    try:
        getattr(trace, method)(*args, **kwargs)
        return trace
    except Exception as exc:
        logging.debug(
            "[entry] fresh-entry eligibility trace disabled during reconciliation | "
            "method=%s error_type=%s",
            method,
            bounded_exception_type(exc),
        )
        return None


def _initialize_fresh_entry_trace(
    bot,
    ideal_orders: dict,
    actual_orders: dict[str, list[dict]],
) -> FreshEntryEligibilityTrace | None:
    """Build cycle-local diagnostic scope from already evaluated live symbols and orders."""
    trace: FreshEntryEligibilityTrace | None = FreshEntryEligibilityTrace()
    try:
        flat_ideal = [
            order
            for orders in ideal_orders.values()
            if isinstance(orders, list)
            for order in orders
            if isinstance(order, dict)
        ]
        flat_actual = [
            order
            for orders in actual_orders.values()
            if isinstance(orders, list)
            for order in orders
            if isinstance(order, dict)
        ]
        observed_pairs = {
            (str(order.get("symbol") or ""), str(order.get("position_side") or ""))
            for order in flat_ideal + flat_actual
            if order.get("symbol") and order.get("position_side") in {"long", "short"}
        }
        active_symbols = set(getattr(bot, "active_symbols", ()) or ())
        planning_snapshot = getattr(bot, "_current_planning_snapshot", None)
        active_symbols.update(getattr(planning_snapshot, "symbols", ()) or ())
        enabled = getattr(bot, "is_pside_enabled", None)
        if callable(enabled):
            for pside in ("long", "short"):
                try:
                    pside_enabled = bool(enabled(pside))
                except Exception:
                    pside_enabled = False
                if pside_enabled:
                    observed_pairs.update(
                        (str(symbol), pside) for symbol in active_symbols if symbol
                    )
        for symbol, pside in sorted(observed_pairs):
            trace.record_evaluated(symbol, pside)
        trace.record_ideal_orders(flat_ideal)
        trace.record_protective_orders(flat_ideal)
        for pair, count in dict(
            getattr(bot, "_fresh_entry_conversion_blocked_counts", {}) or {}
        ).items():
            if not isinstance(pair, tuple) or len(pair) != 2:
                continue
            trace.record_count(
                pair[0],
                pair[1],
                "blocked",
                count=int(count),
                reason="conversion_zero_or_duplicate",
            )
        return trace
    except Exception as exc:
        logging.debug(
            "[entry] fresh-entry eligibility trace initialization failed | error_type=%s",
            bounded_exception_type(exc),
        )
        return None


def add_to_recent_order_cancellations(bot, order):
    """Record a recently cancelled order to throttle repeated cancellations."""
    bot.recent_order_cancellations.append(
        {**order, **{"execution_timestamp": _utc_ms()}}
    )


def order_was_recently_cancelled(bot, order, max_age_ms=15_000) -> float:
    """Return remaining throttle delay if the order was cancelled within `max_age_ms`."""
    age_limit = _utc_ms() - max_age_ms
    bot.recent_order_cancellations = [
        x
        for x in bot.recent_order_cancellations
        if x["execution_timestamp"] > age_limit
    ]
    if matching := _pb_attr("order_has_match")(
        order,
        bot.recent_order_cancellations,
        tolerance_price=0.0,
        tolerance_qty=0.0,
    ):
        return max(0.0, (matching["execution_timestamp"] + max_age_ms) - _utc_ms())
    return 0.0


def order_matches_bot_cancellation(bot, order, max_age_ms=180_000) -> bool:
    """Return True when an exact recent bot cancellation strongly explains the disappearance."""
    age_limit = _utc_ms() - max_age_ms
    bot.recent_order_cancellations = [
        x
        for x in bot.recent_order_cancellations
        if x["execution_timestamp"] > age_limit
    ]
    return bool(
        _pb_attr("order_has_match")(
            order,
            bot.recent_order_cancellations,
            tolerance_price=0.0,
            tolerance_qty=0.0,
        )
    )


def add_to_recent_order_executions(bot, order):
    """Track newly created orders to limit duplicate submissions."""
    bot.recent_order_executions.append({**order, **{"execution_timestamp": _utc_ms()}})


def order_matches_recent_execution(bot, order, max_age_ms=180_000) -> bool:
    """Return True when an exact recent bot creation strongly explains a new open order."""
    age_limit = _utc_ms() - max_age_ms
    if not hasattr(bot, "recent_order_executions"):
        bot.recent_order_executions = []
    bot.recent_order_executions = [
        x for x in bot.recent_order_executions if x["execution_timestamp"] > age_limit
    ]
    return bool(
        _pb_attr("order_has_match")(
            order,
            bot.recent_order_executions,
            tolerance_price=0.0,
            tolerance_qty=0.0,
        )
    )


def local_order_open_orders_confirmed(bot, max_age_ms=15_000) -> bool:
    """Return True when recent local creates/cancels are reflected in the current open-orders view."""
    age_limit = _utc_ms() - max_age_ms
    if not hasattr(bot, "recent_order_cancellations"):
        bot.recent_order_cancellations = []
    if not hasattr(bot, "recent_order_executions"):
        bot.recent_order_executions = []
    bot.recent_order_cancellations = [
        x
        for x in bot.recent_order_cancellations
        if x["execution_timestamp"] > age_limit
    ]
    bot.recent_order_executions = [
        x for x in bot.recent_order_executions if x["execution_timestamp"] > age_limit
    ]
    order_has_match = _pb_attr("order_has_match")
    current_open_orders = [
        elm for sublist in bot.open_orders.values() for elm in sublist
    ]
    for cancelled in bot.recent_order_cancellations:
        if order_has_match(
            cancelled, current_open_orders, tolerance_price=0.0, tolerance_qty=0.0
        ):
            return False
    for created in bot.recent_order_executions:
        if order_has_match(
            created,
            bot.recent_order_cancellations,
            tolerance_price=0.0,
            tolerance_qty=0.0,
        ):
            continue
        if not order_has_match(
            created, current_open_orders, tolerance_price=0.0, tolerance_qty=0.0
        ):
            return False
    return True


def order_was_recently_updated(bot, order, max_age_ms=15_000) -> float:
    """Return throttle delay if the order was placed within `max_age_ms`."""
    age_limit = _utc_ms() - max_age_ms
    bot.recent_order_executions = [
        x for x in bot.recent_order_executions if x["execution_timestamp"] > age_limit
    ]
    if matching := _pb_attr("order_has_match")(order, bot.recent_order_executions):
        return max(0.0, (matching["execution_timestamp"] + max_age_ms) - _utc_ms())
    return 0.0


def extract_order_custom_id(order: dict) -> str:
    """Return the first normalized client/custom order id from unified or raw fields."""
    if not isinstance(order, dict):
        return ""
    candidates = (
        "custom_id",
        "customId",
        "client_order_id",
        "clientOrderId",
        "client_oid",
        "clientOid",
        "order_link_id",
        "orderLinkId",
        "clOrdId",
        "cloid",
        "text",
    )
    for source in (order, order.get("info", {})):
        if not isinstance(source, dict):
            continue
        for key in candidates:
            value = source.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def extract_order_exchange_id(order: dict) -> str:
    """Return the exchange-assigned order id from unified or raw fields."""
    if not isinstance(order, dict):
        return ""
    candidates = ("id", "order_id", "orderId", "orderID", "ordId")
    for source in (order, order.get("info", {})):
        if not isinstance(source, dict):
            continue
        for key in candidates:
            value = source.get(key)
            if value not in (None, ""):
                return str(value)
    return ""


def ema_entry_cancellation_order_key(order: dict) -> Optional[tuple[str, str, str, str]]:
    """Return a strong identity for an already-proven resting entry.

    Degraded-mode cancellation is an exception to manual ownership, so price/qty
    similarity is deliberately insufficient: an exchange or client order id must
    prove that the cancellation still targets the same order.
    """
    if not isinstance(order, dict):
        return None
    symbol = str(order.get("symbol") or "")
    pside = str(order.get("position_side") or order.get("positionSide") or "")
    if not symbol or pside not in ("long", "short"):
        return None
    exchange_id = extract_order_exchange_id(order)
    if exchange_id:
        return (symbol, pside, "exchange_id", exchange_id)
    custom_id = canonical_passivbot_custom_id(extract_order_custom_id(order))
    if custom_id:
        return (symbol, pside, "client_id", custom_id)
    return None


def ema_entry_cancellation_order_keys(
    order: dict,
) -> set[tuple[str, str, str, str]]:
    """Return every strong identity alias for an already-proven resting entry."""
    if not isinstance(order, dict):
        return set()
    symbol = str(order.get("symbol") or "")
    pside = str(order.get("position_side") or order.get("positionSide") or "")
    if not symbol or pside not in ("long", "short"):
        return set()
    keys: set[tuple[str, str, str, str]] = set()
    exchange_id = extract_order_exchange_id(order)
    if exchange_id:
        keys.add((symbol, pside, "exchange_id", exchange_id))
    custom_id = canonical_passivbot_custom_id(extract_order_custom_id(order))
    if custom_id:
        keys.add((symbol, pside, "client_id", custom_id))
    return keys


def canonical_passivbot_custom_id(custom_id: str) -> str:
    """Normalize broker/exchange wrappers around Passivbot custom ids."""
    if not custom_id:
        return ""
    custom_id = str(custom_id)
    marker = _pb_attr("_TYPE_MARKER_RE").search(custom_id)
    if marker:
        return custom_id[marker.start() :]
    return custom_id


def extract_order_reduce_only(order: dict) -> Optional[bool]:
    if not isinstance(order, dict):
        return None
    for source in (order, order.get("info", {})):
        if not isinstance(source, dict):
            continue
        for key in ("reduce_only", "reduceOnly"):
            if key not in source:
                continue
            value = source[key]
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "y"}:
                    return True
                if normalized in {"false", "0", "no", "n"}:
                    return False
                return None
            if isinstance(value, (int, float)) and value in {0, 1}:
                return bool(value)
            return None
    return None


def extract_order_remaining_qty(order: dict) -> Optional[float]:
    """Return authoritative remaining open quantity, or None when unproven."""
    if not isinstance(order, dict):
        return None

    def get_number(key: str) -> Optional[float]:
        for source in (order, order.get("info", {})):
            if not isinstance(source, dict) or key not in source:
                continue
            value = source.get(key)
            if value in (None, ""):
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                return None
            return parsed if math.isfinite(parsed) and parsed >= 0.0 else None
        return None

    remaining = get_number("remaining")
    amount = get_number("amount")
    filled = get_number("filled")
    if remaining is not None:
        if amount is not None and filled is not None:
            if filled > amount or remaining > amount:
                return None
            derived = max(0.0, amount - filled)
            scale = max(1.0, remaining, amount, filled)
            if abs(remaining - derived) > scale * 1e-10:
                return None
        return remaining
    if amount is not None and filled is not None:
        return max(0.0, amount - filled) if filled <= amount else None
    # Locally normalized/fake order fixtures may expose only qty. Treat it as
    # authoritative only when no original-amount fields are present.
    if amount is None and filled is None:
        return get_number("qty")
    return None


def extract_order_float(order: dict, candidates: tuple[str, ...]) -> Optional[float]:
    if not isinstance(order, dict):
        return None
    for source in (order, order.get("info", {})):
        if not isinstance(source, dict):
            continue
        for key in candidates:
            value = source.get(key)
            if value in (None, ""):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return None


def order_identity_fingerprint(order: dict, pb_type: str) -> Optional[dict]:
    if not isinstance(order, dict) or not pb_type or pb_type == "unknown":
        return None
    reduce_only = extract_order_reduce_only(order)
    qty = extract_order_float(order, ("qty", "amount", "size"))
    price = extract_order_float(order, ("price",))
    symbol = order.get("symbol")
    side = order.get("side")
    position_side = order.get("position_side") or order.get("positionSide")
    if any(
        x in (None, "") for x in (symbol, side, position_side, reduce_only, qty, price)
    ):
        return None
    return {
        "symbol": str(symbol),
        "side": str(side).lower(),
        "position_side": str(position_side).lower(),
        "reduce_only": bool(reduce_only),
        "pb_type": str(pb_type),
        "qty": round(abs(float(qty)), 12),
        "price": round(float(price), 12),
    }


def _record_status_rank(status: str) -> int:
    return {
        "submitted": 0,
        "legacy": 1,
        "create_error_ambiguous": 2,
        "open_snapshot_confirmed": 3,
        "acknowledged": 4,
    }.get(str(status or ""), 0)


def _records_refer_to_same_order(a: dict, b: dict) -> bool:
    a_exchange_id = str(a.get("exchange_id") or "")
    b_exchange_id = str(b.get("exchange_id") or "")
    if a_exchange_id and b_exchange_id and a_exchange_id == b_exchange_id:
        return True
    a_custom_id = str(a.get("canonical_custom_id") or "")
    b_custom_id = str(b.get("canonical_custom_id") or "")
    if a_custom_id and b_custom_id and a_custom_id == b_custom_id:
        return True
    if (a_exchange_id and b_exchange_id) or (a_custom_id and b_custom_id):
        return False
    a_fingerprint = a.get("fingerprint")
    b_fingerprint = b.get("fingerprint")
    return bool(a_fingerprint and b_fingerprint and a_fingerprint == b_fingerprint)


def _merge_emitted_order_record(existing: dict, incoming: dict) -> dict:
    merged = dict(existing)
    for key, value in incoming.items():
        if value not in (None, "", []):
            merged[key] = value
    if existing.get("timestamp") not in (None, ""):
        merged["timestamp"] = int(existing["timestamp"])
    if _record_status_rank(existing.get("status")) > _record_status_rank(
        incoming.get("status")
    ):
        merged["status"] = existing.get("status")
    return merged


def build_emitted_order_record(
    bot, order: dict, emitted_ts: int, *, status: str = "acknowledged"
) -> Optional[dict]:
    custom_id = extract_order_custom_id(order)
    custom_id_to_snake = _pb_attr("custom_id_to_snake")
    pb_type = (
        custom_id_to_snake(custom_id)
        if custom_id
        else bot._resolve_pb_order_type(order)
    )
    if not pb_type or pb_type == "unknown":
        pb_type = bot._resolve_pb_order_type(order)
    record = {
        "timestamp": int(emitted_ts),
        "exchange_id": extract_order_exchange_id(order),
        "custom_id": custom_id,
        "canonical_custom_id": canonical_passivbot_custom_id(custom_id),
        "pb_type": pb_type if pb_type and pb_type != "unknown" else "",
        "status": str(status or "acknowledged"),
    }
    record["fingerprint"] = order_identity_fingerprint(order, record["pb_type"])
    if record["fingerprint"]:
        record.update(record["fingerprint"])
    order_ts = order.get("timestamp") if isinstance(order, dict) else None
    if order_ts not in (None, ""):
        try:
            record["order_timestamp"] = int(float(order_ts))
        except (TypeError, ValueError):
            pass
    if not (
        record["exchange_id"] or record["canonical_custom_id"] or record["fingerprint"]
    ):
        return None
    return record


def emitted_order_records(bot) -> list[dict]:
    """Return recent emitted order records, upgrading legacy custom-id maps if needed."""
    records = getattr(bot, "orders_emitted_to_exchange", [])
    if isinstance(records, dict):
        upgraded = []
        for custom_id, timestamp in records.items():
            custom_id = str(custom_id)
            upgraded.append(
                {
                    "timestamp": int(timestamp),
                    "exchange_id": "",
                    "custom_id": custom_id,
                    "canonical_custom_id": canonical_passivbot_custom_id(custom_id),
                    "pb_type": _pb_attr("custom_id_to_snake")(custom_id),
                    "status": "legacy",
                    "fingerprint": None,
                }
            )
        bot.orders_emitted_to_exchange = upgraded
        return upgraded
    if not isinstance(records, list):
        bot.orders_emitted_to_exchange = []
        return []
    return records


def _emitted_record_matches_open_order(record: dict, order: dict) -> bool:
    """Return whether overlapping durable identities agree for an open order."""
    if not isinstance(record, dict) or not isinstance(order, dict):
        return False
    record_exchange_id = str(record.get("exchange_id") or "")
    record_custom_id = str(record.get("canonical_custom_id") or "")
    order_exchange_id = extract_order_exchange_id(order)
    order_custom_id = canonical_passivbot_custom_id(extract_order_custom_id(order))
    comparisons = []
    if record_exchange_id and order_exchange_id:
        comparisons.append(record_exchange_id == order_exchange_id)
    if record_custom_id and order_custom_id:
        comparisons.append(record_custom_id == order_custom_id)
    return bool(comparisons) and all(comparisons)


def prune_emitted_order_custom_ids(bot, now_ts: int) -> None:
    """Drop expired emitted records while retaining identities of open orders."""
    now_ts = int(now_ts)
    acknowledged_cutoff_ts = now_ts - _pb_const("FOREIGN_PASSIVBOT_LOOKBACK_MS")
    ambiguous_cutoff_ts = now_ts - _pb_const(
        "FOREIGN_PASSIVBOT_AMBIGUOUS_CREATE_LOOKBACK_MS"
    )
    short_lived_statuses = {"submitted", "create_error_ambiguous"}
    open_orders_by_symbol = getattr(bot, "open_orders", {}) or {}
    active_open_orders = []
    if isinstance(open_orders_by_symbol, dict):
        active_open_orders = [
            order
            for orders in open_orders_by_symbol.values()
            if isinstance(orders, list)
            for order in orders
            if isinstance(order, dict)
        ]
    kept = []
    for record in emitted_order_records(bot):
        status = str(record.get("status") or "")
        cutoff_ts = (
            ambiguous_cutoff_ts
            if status in short_lived_statuses
            else acknowledged_cutoff_ts
        )
        still_open = status not in short_lived_statuses and any(
            _emitted_record_matches_open_order(record, order)
            for order in active_open_orders
        )
        if int(record.get("timestamp", 0)) >= cutoff_ts or still_open:
            kept.append(record)
    bot.orders_emitted_to_exchange = kept


def prune_foreign_passivbot_seen(bot, now_ts: int) -> None:
    """Drop old foreign Passivbot detections outside the rolling stop window."""
    cutoff_ts = int(now_ts) - _pb_const("FOREIGN_PASSIVBOT_WINDOW_MS")
    bot.foreign_passivbot_seen = {
        cid: ts
        for cid, ts in getattr(bot, "foreign_passivbot_seen", {}).items()
        if int(ts) >= cutoff_ts
    }


def record_emitted_order_custom_id(
    bot,
    order: dict,
    emitted_ts: Optional[int] = None,
    *,
    status: str = "acknowledged",
) -> None:
    """Remember an acknowledged or ambiguous create so later refreshes can adopt it."""
    if emitted_ts is None:
        emitted_ts = (
            int(bot.get_exchange_time())
            if hasattr(bot, "get_exchange_time")
            else _utc_ms()
        )
    record = build_emitted_order_record(bot, order, emitted_ts, status=status)
    if record is None:
        return
    if not hasattr(bot, "orders_emitted_to_exchange"):
        bot.orders_emitted_to_exchange = []
    records = emitted_order_records(bot)
    for idx, existing in enumerate(records):
        if _records_refer_to_same_order(existing, record):
            records[idx] = _merge_emitted_order_record(existing, record)
            return
    records.append(record)


def foreign_passivbot_detection_key(
    bot, order: dict, custom_id: str, pb_type: str
) -> str:
    exchange_id = extract_order_exchange_id(order)
    if exchange_id:
        return f"id:{exchange_id}"
    canonical_custom_id = canonical_passivbot_custom_id(custom_id)
    if canonical_custom_id:
        return f"cid:{canonical_custom_id}"
    fingerprint = order_identity_fingerprint(order, pb_type)
    if fingerprint:
        return "fp:" + json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return f"unknown:{custom_id}"


def order_matches_recent_emitted_record(
    bot,
    order: dict,
    custom_id: str,
    pb_type: str,
    order_ts: int,
    consumed_record_indices: set[int],
) -> bool:
    exchange_id = extract_order_exchange_id(order)
    canonical_custom_id = canonical_passivbot_custom_id(custom_id)
    fingerprint = order_identity_fingerprint(order, pb_type)
    for idx, record in enumerate(emitted_order_records(bot)):
        if idx in consumed_record_indices:
            continue
        record_exchange_id = record.get("exchange_id") or ""
        if exchange_id and record_exchange_id and exchange_id == record_exchange_id:
            consumed_record_indices.add(idx)
            adopt_open_order_as_emitted_record(bot, idx, order, order_ts)
            return True
        record_custom_id = record.get("canonical_custom_id") or ""
        if (
            canonical_custom_id
            and record_custom_id
            and canonical_custom_id == record_custom_id
        ):
            consumed_record_indices.add(idx)
            adopt_open_order_as_emitted_record(bot, idx, order, order_ts)
            return True
        if exchange_id and record_exchange_id:
            continue
        if canonical_custom_id and record_custom_id:
            continue
        record_fingerprint = record.get("fingerprint")
        record_ts = int(record.get("timestamp", 0))
        if (
            fingerprint
            and record_fingerprint
            and fingerprint == record_fingerprint
            and abs(int(order_ts) - record_ts)
            <= _pb_const("FOREIGN_PASSIVBOT_FINGERPRINT_MATCH_MS")
        ):
            consumed_record_indices.add(idx)
            adopt_open_order_as_emitted_record(bot, idx, order, order_ts)
            return True
    return False


def adopt_open_order_as_emitted_record(
    bot, record_idx: int, order: dict, order_ts: int
) -> None:
    """Upgrade a submitted/ambiguous record once an open-order snapshot confirms it."""
    records = emitted_order_records(bot)
    if record_idx < 0 or record_idx >= len(records):
        return
    incoming = build_emitted_order_record(
        bot, order, order_ts, status="open_snapshot_confirmed"
    )
    if incoming is None:
        return
    records[record_idx] = _merge_emitted_order_record(records[record_idx], incoming)


def emitted_order_match_diagnostics(
    bot, order: dict, custom_id: str, pb_type: str, order_ts: int
) -> str:
    """Return compact diagnostics for an unmatched Passivbot-marked open order."""
    exchange_id = extract_order_exchange_id(order)
    canonical_custom_id = canonical_passivbot_custom_id(custom_id)
    fingerprint = order_identity_fingerprint(order, pb_type)
    records = emitted_order_records(bot)
    status_counts: dict[str, int] = {}
    same_exchange_id = False
    same_custom_id = False
    same_fingerprint = False
    for record in records:
        status = str(record.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        record_exchange_id = str(record.get("exchange_id") or "")
        record_custom_id = str(record.get("canonical_custom_id") or "")
        if exchange_id and record_exchange_id and exchange_id == record_exchange_id:
            same_exchange_id = True
        if (
            canonical_custom_id
            and record_custom_id
            and canonical_custom_id == record_custom_id
        ):
            same_custom_id = True
        if fingerprint and record.get("fingerprint") == fingerprint:
            same_fingerprint = True
    status_summary = ",".join(
        f"{status}:{count}" for status, count in sorted(status_counts.items())
    )
    reduce_only = extract_order_reduce_only(order)
    qty = extract_order_float(order, ("qty", "amount", "size"))
    price = extract_order_float(order, ("price",))
    pside = order.get("position_side") or order.get("positionSide")
    return (
        f"reason=unmatched_passivbot_custom_id order_id={exchange_id or ''} "
        f"side={order.get('side') or ''} pside={pside or ''} qty={qty} price={price} "
        f"reduce_only={reduce_only} order_ts={ts_to_date(order_ts)} "
        f"emitted_records={len(records)} statuses={status_summary or 'none'} "
        f"match_id={same_exchange_id} match_custom_id={same_custom_id} "
        f"match_fingerprint={same_fingerprint}"
    )


async def stop_for_foreign_passivbot_orders(
    bot, detections: list[tuple[dict, str, str, int]], unique_count: int
) -> None:
    """Stop the bot after repeated evidence of a competing Passivbot writer."""
    if getattr(bot, "_foreign_passivbot_stop_requested", False):
        return
    bot._foreign_passivbot_stop_requested = True
    orders_summary = ", ".join(
        f"{symbol_to_coin(order.get('symbol'), verbose=False) or order.get('symbol')}"
        f":{pb_type}:{shorten_custom_id(custom_id)}"
        for order, pb_type, custom_id, _ in detections
    )
    logging.critical(
        "[safety] detected %s unique foreign Passivbot orders in the last %.1f minutes; "
        "stopping bot to avoid competing writers | latest=%s",
        unique_count,
        _pb_const("FOREIGN_PASSIVBOT_WINDOW_MS") / (60 * 1000),
        orders_summary,
    )
    bot.stop_signal_received = True
    if hasattr(bot, "stop_data_maintainers"):
        try:
            bot.stop_data_maintainers(verbose=False)
        except Exception as exc:
            logging.error(
                "[safety] failed to stop data maintainers | error_type=%s "
                "action=continue_foreign_writer_stop",
                bounded_exception_type(exc),
            )
    raise Exception("foreign Passivbot writer detected; stopping bot")


async def detect_foreign_passivbot_orders(bot, open_orders: list[dict]) -> None:
    """Detect newer Passivbot-managed open orders not emitted by this running bot instance."""
    if not hasattr(bot, "orders_emitted_to_exchange"):
        bot.orders_emitted_to_exchange = []
    if not hasattr(bot, "foreign_passivbot_seen"):
        bot.foreign_passivbot_seen = {}
    if not hasattr(bot, "_foreign_passivbot_stop_requested"):
        bot._foreign_passivbot_stop_requested = False
    now_ts = int(bot.get_exchange_time())
    bot_start_ts = int(getattr(bot, "bot_start_exchange_ts", now_ts))
    bot._prune_emitted_order_custom_ids(now_ts)
    bot._prune_foreign_passivbot_seen(now_ts)
    if not open_orders:
        return
    cutoff_ts = max(
        bot_start_ts + _pb_const("FOREIGN_PASSIVBOT_GRACE_MS"),
        now_ts - _pb_const("FOREIGN_PASSIVBOT_LOOKBACK_MS"),
    )
    custom_id_has_explicit_passivbot_marker = _pb_attr(
        "custom_id_has_explicit_passivbot_marker"
    )
    custom_id_to_snake = _pb_attr("custom_id_to_snake")
    new_detections: list[tuple[dict, str, str, int]] = []
    consumed_emitted_records: set[int] = set()
    for order in open_orders:
        ts_raw = order.get("timestamp")
        if ts_raw is None:
            continue
        try:
            order_ts = int(float(ts_raw))
        except Exception:
            continue
        if order_ts < cutoff_ts:
            continue
        custom_id = bot._extract_order_custom_id(order)
        if not custom_id:
            continue
        if not custom_id_has_explicit_passivbot_marker(custom_id):
            continue
        pb_type = custom_id_to_snake(custom_id)
        if not pb_type or pb_type == "unknown":
            continue
        if bot._order_matches_recent_emitted_record(
            order, custom_id, pb_type, order_ts, consumed_emitted_records
        ):
            continue
        detection_key = bot._foreign_passivbot_detection_key(order, custom_id, pb_type)
        if detection_key in bot.foreign_passivbot_seen:
            continue
        bot.foreign_passivbot_seen[detection_key] = order_ts
        new_detections.append((order, pb_type, custom_id, order_ts))
    if not new_detections:
        return
    passivbot_cls = _pb_attr("Passivbot")
    for order, pb_type, custom_id, order_ts in new_detections:
        diagnostics = emitted_order_match_diagnostics(
            bot, order, custom_id, pb_type, order_ts
        )
        logging.error(
            "[safety] detected foreign Passivbot order candidate | symbol=%s type=%s "
            "custom_id=%s ts=%s | %s",
            passivbot_cls._log_symbol(order.get("symbol")),
            pb_type,
            shorten_custom_id(custom_id),
            ts_to_date(order_ts),
            diagnostics,
        )
    if len(bot.foreign_passivbot_seen) >= _pb_const(
        "FOREIGN_PASSIVBOT_MAX_UNIQUE_PER_WINDOW"
    ):
        await bot._stop_for_foreign_passivbot_orders(
            new_detections, unique_count=len(bot.foreign_passivbot_seen)
        )


def mark_account_critical_state_dirty(
    bot,
    *,
    reason: str,
    symbols: Iterable[str] | None = None,
    source: str = "unknown",
    level: int = logging.DEBUG,
) -> None:
    """Force a coherent account-state refresh before the next execution cycle."""
    min_epoch = int(bot._ensure_freshness_ledger().epoch) + 1
    bot._request_authoritative_confirmation(ACCOUNT_SURFACES, min_epoch=min_epoch)
    bot.execution_scheduled = True
    normalized_symbols = sorted({str(symbol) for symbol in (symbols or []) if symbol})
    if normalized_symbols:
        if not hasattr(bot, "state_change_detected_by_symbol"):
            bot.state_change_detected_by_symbol = set()
        bot.state_change_detected_by_symbol.update(normalized_symbols)
    log_key = (
        str(source),
        str(reason),
        tuple(normalized_symbols[:8]),
        len(normalized_symbols),
        min_epoch,
    )
    now_ms = _utc_ms()
    last_key = getattr(bot, "_account_dirty_last_log_key", None)
    last_ms = int(getattr(bot, "_account_dirty_last_log_ms", 0) or 0)
    if log_key == last_key and now_ms - last_ms < 5_000:
        return
    bot._account_dirty_last_log_key = log_key
    bot._account_dirty_last_log_ms = now_ms
    passivbot_cls = _pb_attr("Passivbot")
    symbol_preview = (
        passivbot_cls._log_symbols(normalized_symbols, limit=6)
        if normalized_symbols
        else "unknown"
    )
    logging.log(
        level,
        "[state] account-critical refresh requested | source=%s | reason=%s | symbols=%s | required=%s | min_epoch=%s",
        source,
        reason,
        symbol_preview,
        ",".join(sorted(ACCOUNT_SURFACES)),
        min_epoch,
    )


async def calc_orders_to_cancel_and_create(bot):
    """Determine which existing orders to cancel and which new ones to place."""
    bot._fresh_entry_eligibility_trace = None
    connector_enabled = connector_supports_order_churn_gate(bot)
    state = getattr(bot, "_order_churn_gate_state", None)
    if not isinstance(state, OrderChurnGateState):
        state = OrderChurnGateState()
        bot._order_churn_gate_state = state
    generation = state.begin_generation()
    try:
        ideal_orders = await bot.calc_ideal_orders()
    except Exception:
        if connector_enabled:
            reset = state.clear_history()
            _emit_order_churn_evidence_summary(
                bot,
                state=state,
                generation=generation,
                reset=reset,
                decisions=[ChurnDecision(False, "planning_failed")],
                symbols=getattr(bot, "active_symbols", []) or [],
                snapshot_status="skipped",
            )
        raise
    validate_rust_ideal_orders(ideal_orders)
    if connector_enabled:
        prepare_order_churn_evidence(bot, ideal_orders, generation=generation)
    else:
        state.clear_history()
    return await calc_orders_to_cancel_and_create_from_ideal(bot, ideal_orders)


def validate_rust_ideal_orders(ideal_orders: object) -> None:
    """Reject malformed Rust intent before reconciliation or exchange actions."""
    if not isinstance(ideal_orders, dict):
        raise FatalBotException("Rust ideal orders must be a symbol-to-orders mapping")
    for symbol, orders in ideal_orders.items():
        normalized_symbol = str(symbol)
        if not normalized_symbol:
            raise FatalBotException("Rust ideal orders contain an empty symbol")
        if not isinstance(orders, list):
            raise FatalBotException(
                f"Rust ideal orders for {normalized_symbol} must be a list"
            )
        try:
            normalize_ideal_orders(orders)
            if any(str(order["symbol"]) != normalized_symbol for order in orders):
                raise ValueError("ideal order symbol does not match its mapping key")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FatalBotException(
                f"Rust emitted malformed ideal orders for {normalized_symbol}"
            ) from exc


def _validated_rust_finite_number(value: object, context: str) -> float:
    error = f"Rust orchestrator {context}"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FatalBotException(error)
    try:
        value_f = float(value)
    except (OverflowError, TypeError, ValueError) as exc:
        raise FatalBotException(error) from exc
    if not math.isfinite(value_f):
        raise FatalBotException(error)
    return value_f


def _validated_rust_u64(value: object, context: str) -> int:
    error = f"Rust orchestrator {context}"
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > (1 << 64) - 1
    ):
        raise FatalBotException(error)
    return value


def _rust_representation_tolerance(value: float, expected: float) -> float:
    """Return a small tolerance for arithmetic representation noise only."""
    return sys.float_info.epsilon * max(abs(value), abs(expected)) * 4.0


def _canonical_rust_order_book_value(value: float, price_step: float) -> float:
    """Snap only representation-noisy aligned books to Rust's submitted tick."""
    step_count = value / price_step
    if not math.isfinite(step_count):
        return value
    nearest = round(step_count) * price_step
    tolerance = _rust_representation_tolerance(value, nearest)
    return nearest if abs(value - nearest) <= tolerance else value


_PROTECTIVE_REDUCE_ONLY_FAMILIES = frozenset(
    {
        "close_panic",
        "close_auto_reduce_twel",
        "close_auto_reduce_wel",
        "close_unstuck",
    }
)

_SUBMITTED_GATED_REDUCER_FAMILIES = frozenset(
    {
        "close_auto_reduce_twel",
        "close_auto_reduce_wel",
        "close_unstuck",
    }
)

_LEGACY_UNEMITTABLE_RUST_ORDER_TYPES = frozenset(
    {
        "entry_grid_inflated_long",
        "entry_grid_inflated_short",
    }
)


def _rust_order_requires_risk_critical_priority(order_type: str) -> bool:
    return order_type.rsplit("_", 1)[0] in _PROTECTIVE_REDUCE_ONLY_FAMILIES


def _validate_rust_reducer_enablement(
    order_type: str,
    pair: tuple[int, str],
    submitted_reducer_family_enablement: dict[tuple[int, str], frozenset[str]],
    submitted_auto_unstuck_allowed: bool,
    context: str,
) -> None:
    protective_family = order_type.rsplit("_", 1)[0]
    if (
        protective_family in _SUBMITTED_GATED_REDUCER_FAMILIES
        and protective_family not in submitted_reducer_family_enablement[pair]
    ):
        raise FatalBotException(
            f"{context} contradicts submitted reducer enablement"
        )
    if protective_family == "close_unstuck" and not submitted_auto_unstuck_allowed:
        raise FatalBotException(
            f"{context} contradicts submitted auto-unstuck gate"
        )


def _expected_rust_execution_priority(order_type: str, input_mode: object) -> str:
    if _rust_order_requires_risk_critical_priority(order_type) or (
        input_mode == "graceful_stop" and order_type.startswith("close_")
    ):
        return "risk_critical"
    return "ordinary"


def _validate_intrinsic_rust_execution_priority(
    order_type: str, execution_priority: str, context: str
) -> None:
    if _rust_order_requires_risk_critical_priority(order_type):
        expected_priority = "risk_critical"
    elif order_type.startswith("entry_"):
        expected_priority = "ordinary"
    else:
        return
    if execution_priority != expected_priority:
        raise FatalBotException(
            f"{context} has execution_priority inconsistent with its order_type"
        )


def _validate_rust_order_family_for_submitted_mode(
    order_type: str,
    input_mode: object,
    submitted_position_size: float,
    submitted_flat_side_eligible: bool,
    context: str,
) -> None:
    """Reject order families Rust cannot emit for the submitted mode and eligibility."""
    mode = "normal" if input_mode is None else input_mode
    is_entry = order_type.startswith("entry_")
    is_close = order_type.startswith("close_")
    is_panic_close = order_type.rsplit("_", 1)[0] == "close_panic"
    if not (is_entry or is_close):
        raise FatalBotException(f"{context} has unsupported order family")
    invalid = (
        mode == "manual"
        or (mode == "panic" and not is_panic_close)
        or (mode != "panic" and is_panic_close)
        or (mode == "tp_only" and is_entry)
        or (
            is_entry
            and submitted_position_size == 0.0
            and not submitted_flat_side_eligible
        )
        or (
            mode == "graceful_stop"
            and submitted_position_size == 0.0
        )
    )
    if invalid:
        raise FatalBotException(
            f"{context} has order family inconsistent with its submitted mode or eligibility"
        )


def _validate_rust_order_family_for_submitted_strategy(
    order_type: str,
    strategy_kind: str,
    entry_retracement_enabled: bool,
    close_retracement_enabled: bool,
    context: str,
) -> None:
    """Reject ordinary strategy families the submitted Rust strategy cannot emit."""
    if _rust_order_requires_risk_critical_priority(order_type):
        return
    is_ema_anchor_order = order_type.rsplit("_", 1)[0] in {
        "entry_ema_anchor",
        "close_ema_anchor",
    }
    if (strategy_kind == "ema_anchor") != is_ema_anchor_order:
        raise FatalBotException(
            f"{context} has order family inconsistent with submitted strategy"
        )
    if strategy_kind != "trailing_martingale":
        return
    order_family = order_type.rsplit("_", 1)[0]
    grid_reentry = order_family in {"entry_grid_normal", "entry_grid_cropped"}
    trailing_reentry = order_family in {
        "entry_trailing_normal",
        "entry_trailing_cropped",
    }
    if (grid_reentry and entry_retracement_enabled) or (
        trailing_reentry and not entry_retracement_enabled
    ):
        raise FatalBotException(
            f"{context} has entry family inconsistent with submitted retracement mode"
        )
    grid_close = order_family == "close_grid"
    trailing_close = order_family == "close_trailing"
    if (grid_close and close_retracement_enabled) or (
        trailing_close and not close_retracement_enabled
    ):
        raise FatalBotException(
            f"{context} has close family inconsistent with submitted retracement mode"
        )


def _validate_rust_flat_entry_batch(
    order_types: list[str], strategy_kind: str, pside: str, context: str
) -> None:
    """Reject flat-side entry batches no submitted Rust strategy can emit."""
    if strategy_kind == "ema_anchor":
        valid = order_types == [f"entry_ema_anchor_{pside}"]
    else:
        initial_type = f"entry_initial_normal_{pside}"
        allowed_types = {
            initial_type,
            f"entry_grid_normal_{pside}",
            f"entry_grid_cropped_{pside}",
        }
        valid = (
            order_types.count(initial_type) == 1
            and all(order_type in allowed_types for order_type in order_types)
        )
    if not valid:
        raise FatalBotException(
            f"{context} has entry families inconsistent with a flat submitted side"
        )


def _validate_rust_entry_exchange_constraints(
    qty: float,
    cost_price: float,
    exchange: tuple[float, float, float, float, float],
    context: str,
) -> None:
    """Reject entry quantities impossible under the submitted exchange constraints."""
    qty_step, _price_step, min_qty, min_cost, c_mult = exchange
    qty_abs = abs(qty)
    qty_steps = qty_abs / qty_step
    entry_cost = qty_abs * cost_price * c_mult
    if not math.isfinite(qty_steps) or not math.isfinite(entry_cost):
        raise FatalBotException(f"{context} has invalid exchange-constrained quantity")
    rounded_qty = round(qty_steps) * qty_step
    qty_tolerance = _rust_representation_tolerance(qty_abs, rounded_qty)
    if not math.isclose(qty_abs, rounded_qty, rel_tol=0.0, abs_tol=qty_tolerance):
        raise FatalBotException(
            f"{context} quantity is inconsistent with submitted qty_step"
        )
    minimum_qty_tolerance = _rust_representation_tolerance(qty_abs, min_qty)
    cost_tolerance = 8.0 * max(math.ulp(entry_cost), math.ulp(min_cost))
    if (
        qty_abs + minimum_qty_tolerance < min_qty
        or entry_cost + cost_tolerance < min_cost
    ):
        raise FatalBotException(
            f"{context} quantity is below submitted effective entry minimum"
        )


def _validate_rust_limit_price_exchange_constraints(
    price: float,
    exchange: tuple[float, float, float, float, float],
    context: str,
) -> None:
    """Reject limit prices impossible under the submitted exchange constraints."""
    _qty_step, price_step, _min_qty, _min_cost, _c_mult = exchange
    price_steps = price / price_step
    if not math.isfinite(price_steps):
        raise FatalBotException(f"{context} has invalid exchange-constrained price")
    rounded_price = round(price_steps) * price_step
    price_tolerance = _rust_representation_tolerance(price, rounded_price)
    if not math.isclose(price, rounded_price, rel_tol=0.0, abs_tol=price_tolerance):
        raise FatalBotException(
            f"{context} price is inconsistent with submitted price_step: "
            f"price={price!r} price_step={price_step!r} "
            f"nearest_price={rounded_price!r} "
            f"delta={abs(price - rounded_price)!r} tolerance={price_tolerance!r}"
        )


def _rust_tolerant_touch_step_count(
    value: float, price_step: float, *, round_up: bool
) -> int | float:
    """Return Rust's representation-tolerant directional touch tick."""
    step_count = value / price_step
    if not math.isfinite(step_count):
        return math.inf
    nearest_step_count = round(step_count)
    nearest_price = nearest_step_count * price_step
    representation_tolerance = (
        sys.float_info.epsilon * max(abs(value), abs(nearest_price)) * 4.0
    )
    if math.isclose(
        value,
        nearest_price,
        rel_tol=0.0,
        abs_tol=representation_tolerance,
    ):
        return nearest_step_count
    return math.ceil(step_count) if round_up else math.floor(step_count)


def _validate_rust_panic_limit_price(
    price: float,
    pside: str,
    order_book: tuple[float, float],
    exchange: tuple[float, float, float, float, float],
    context: str,
) -> None:
    """Reject panic-limit prices inconsistent with Rust's submitted-book formula."""
    _qty_step, price_step, _min_qty, _min_cost, _c_mult = exchange
    bid, ask = order_book
    if pside == "long":
        expected_step_count = max(
            _rust_tolerant_touch_step_count(ask, price_step, round_up=False) - 1,
            1,
        )
    else:
        expected_step_count = (
            _rust_tolerant_touch_step_count(bid, price_step, round_up=True) + 1
        )
    expected_price = expected_step_count * price_step
    representation_tolerance = (
        sys.float_info.epsilon * max(abs(price), abs(expected_price)) * 4.0
    )
    if not math.isclose(
        price,
        expected_price,
        rel_tol=0.0,
        abs_tol=representation_tolerance,
    ):
        raise FatalBotException(
            f"{context} panic limit price is inconsistent with submitted order book"
        )


def _rust_effective_min_qty(
    price: float, exchange: tuple[float, float, float, float, float]
) -> float:
    """Mirror Rust's compact effective-minimum calculation for boundary checks."""
    qty_step, _price_step, min_qty, min_cost, c_mult = exchange
    raw_min = max(min_qty, min_cost / price / c_mult)
    raw_min_steps = raw_min / qty_step
    if not math.isfinite(raw_min_steps):
        return math.inf
    nearest_step_count = round(raw_min_steps)
    nearest_step = nearest_step_count * qty_step
    representation_tolerance = (
        sys.float_info.epsilon * max(abs(raw_min), abs(nearest_step)) * 4.0
    )
    if raw_min == 0.0:
        return 0.0
    if (
        nearest_step_count > 0
        and abs(raw_min_steps - nearest_step_count) <= 1e-8
        and (
            nearest_step >= raw_min
            or raw_min - nearest_step <= representation_tolerance
        )
    ):
        return max(nearest_step, raw_min)
    return math.ceil(raw_min_steps) * qty_step


def _validate_rust_close_exchange_constraints(
    qty: float,
    position_size: float,
    order_book: tuple[float, float],
    exchange: tuple[float, float, float, float, float],
    context: str,
    *,
    minimum_price: float | None = None,
) -> None:
    """Reject close quantities Rust trimming cannot emit under submitted constraints."""
    qty_step, _price_step, _min_qty, _min_cost, _c_mult = exchange
    qty_abs = abs(qty)
    position_abs = abs(position_size)
    if position_abs <= 1e-12:
        raise FatalBotException(
            f"{context} cannot close a submitted position at or below Rust's dust threshold"
        )
    market_price = order_book[1] if qty > 0.0 else order_book[0]
    effective_min_qty = _rust_effective_min_qty(
        market_price if minimum_price is None else minimum_price, exchange
    )
    closes_exact_remaining_position = math.isclose(
        qty_abs,
        position_abs,
        rel_tol=0.0,
        abs_tol=_rust_representation_tolerance(qty_abs, position_abs),
    )
    if closes_exact_remaining_position:
        return
    qty_steps = qty_abs / qty_step
    if not math.isfinite(qty_steps):
        raise FatalBotException(f"{context} has invalid exchange-constrained quantity")
    rounded_qty = round(qty_steps) * qty_step
    qty_tolerance = _rust_representation_tolerance(qty_abs, rounded_qty)
    if not math.isclose(qty_abs, rounded_qty, rel_tol=0.0, abs_tol=qty_tolerance):
        raise FatalBotException(
            f"{context} quantity is inconsistent with submitted qty_step"
        )
    minimum_qty_tolerance = _rust_representation_tolerance(
        qty_abs, effective_min_qty
    )
    if qty_abs + minimum_qty_tolerance < effective_min_qty:
        raise FatalBotException(
            f"{context} quantity is below submitted effective close minimum"
        )


def _expected_rust_effective_mode(
    input_mode: str | None, has_position: bool, globally_enabled: bool
) -> str:
    """Return Rust's deterministic generation mode for one submitted side."""
    if not globally_enabled:
        return "manual"
    if input_mode == "graceful_stop" and has_position:
        return "normal"
    return "normal" if input_mode is None else input_mode


def _expected_rust_panic_execution_type(
    global_input: dict,
    symbol_side_hsl_execution: tuple[bool, str],
    order_type: str,
) -> str | None:
    if order_type.rsplit("_", 1)[0] != "close_panic":
        return None
    # HSL panic execution is an explicit protective override. Rust evaluates
    # this before market_orders_allowed, so Python must preserve the same rule.
    if global_input.get("panic_close_market", False) is True:
        return "market"
    hsl_enabled, panic_close_order_type = symbol_side_hsl_execution
    return (
        "market"
        if hsl_enabled and panic_close_order_type == "market"
        else "limit"
    )


def _expected_rust_execution_type(
    global_input: dict,
    symbol_side_hsl_execution: tuple[bool, str],
    order_book: tuple[float, float],
    order_type: str,
    qty: float,
    price: float,
) -> str:
    panic_execution_type = _expected_rust_panic_execution_type(
        global_input, symbol_side_hsl_execution, order_type
    )
    if panic_execution_type is not None:
        return panic_execution_type
    if global_input.get("market_orders_allowed", False) is not True:
        return "limit"
    market_price = (order_book[0] + order_book[1]) * 0.5
    if not math.isfinite(market_price) or market_price <= 0.0:
        return "limit"
    if (qty > 0.0 and price >= market_price) or (
        qty < 0.0 and price <= market_price
    ):
        return "market"
    threshold = _validated_rust_finite_number(
        global_input.get("market_order_near_touch_threshold", 0.001),
        "global input has invalid market_order_near_touch_threshold",
    )
    price_diff = abs(price / market_price - 1.0)
    return "market" if price_diff <= max(threshold, 0.0) else "limit"


def _submitted_rust_input_context(
    orchestrator_input: dict, expected_symbol_idxs: set[int]
) -> tuple[
    dict[tuple[int, str], object],
    dict[int, tuple[float, float]],
    dict[tuple[int, str], float],
    dict[tuple[int, str], bool],
    dict[str, bool],
    dict[str, int],
    bool,
    str,
    dict[tuple[int, str], frozenset[str]],
    dict[int, tuple[float, float, float, float, float]],
    dict[tuple[int, str], bool],
    dict[tuple[int, str], bool],
    dict[tuple[int, str], bool],
    dict[tuple[int, str], bool],
    dict[tuple[int, str], tuple[bool, str]],
]:
    symbols = orchestrator_input.get("symbols")
    if not isinstance(symbols, list):
        raise FatalBotException(
            "Rust orchestrator validation missing corresponding symbol inputs"
        )
    valid_modes = {"normal", "panic", "graceful_stop", "tp_only", "manual"}
    modes: dict[tuple[int, str], object] = {}
    order_books: dict[int, tuple[float, float]] = {}
    position_sizes: dict[tuple[int, str], float] = {}
    symbol_side_eligibility: dict[tuple[int, str], bool] = {}
    exchange_constraints: dict[int, tuple[float, float, float, float, float]] = {}
    entry_cooldown_active: dict[tuple[int, str], bool] = {}
    entry_cooldown_positive: dict[tuple[int, str], bool] = {}
    entry_sequential_staging: dict[tuple[int, str], bool] = {}
    close_retracement_enabled: dict[tuple[int, str], bool] = {}
    hsl_execution_policy: dict[tuple[int, str], tuple[bool, str]] = {}
    timestamp_ms = _validated_rust_u64(
        orchestrator_input.get("timestamp_ms", 0), "input has invalid timestamp_ms"
    )
    global_input = orchestrator_input.get("global")
    if not isinstance(global_input, dict):
        raise FatalBotException(
            "Rust orchestrator validation missing corresponding global input"
        )
    global_bot_params = global_input.get("global_bot_params")
    if not isinstance(global_bot_params, dict):
        raise FatalBotException("Rust orchestrator global input has invalid bot params")
    global_side_enablement: dict[str, bool] = {}
    global_side_n_positions: dict[str, int] = {}
    global_twel_enforcer_enablement: dict[str, bool] = {}
    for pside in ("long", "short"):
        side_params = global_bot_params.get(pside)
        if not isinstance(side_params, dict):
            raise FatalBotException(
                f"Rust orchestrator global input has invalid {pside} bot params"
            )
        total_wallet_exposure_limit = _validated_rust_finite_number(
            side_params.get("total_wallet_exposure_limit"),
            f"global input has invalid {pside} total_wallet_exposure_limit",
        )
        n_positions = side_params.get("n_positions")
        if (
            isinstance(n_positions, bool)
            or not isinstance(n_positions, int)
            or n_positions < 0
        ):
            raise FatalBotException(
                f"global input has invalid {pside} n_positions"
            )
        global_side_enablement[pside] = (
            total_wallet_exposure_limit > 0.0 and n_positions > 0
        )
        global_side_n_positions[pside] = n_positions
        twel_enabled = side_params.get("risk_twel_enforcer_enabled", True)
        if not isinstance(twel_enabled, bool):
            raise FatalBotException(
                f"Rust orchestrator global input has invalid {pside} TWEL enforcer flag"
            )
        twel_threshold = _validated_rust_finite_number(
            side_params.get("risk_twel_enforcer_threshold", 0.0),
            f"global input has invalid {pside} TWEL enforcer threshold",
        )
        global_twel_enforcer_enablement[pside] = (
            twel_enabled and twel_threshold > 0.0
        )
    hedge_mode = global_input.get("hedge_mode", True)
    if not isinstance(hedge_mode, bool):
        raise FatalBotException("Rust orchestrator global input has invalid hedge_mode")
    strategy_kind = global_input.get("strategy_kind", "trailing_martingale")
    if not isinstance(strategy_kind, str) or strategy_kind not in {
        "trailing_martingale",
        "trailing_grid_v7",
        "ema_anchor",
    }:
        raise FatalBotException("Rust orchestrator global input has invalid strategy_kind")
    seen_symbol_idxs: set[int] = set()
    reducer_family_enablement: dict[tuple[int, str], frozenset[str]] = {}
    for input_idx, row in enumerate(symbols):
        if not isinstance(row, dict):
            raise FatalBotException(
                f"Rust orchestrator symbol input {input_idx} must be a mapping"
            )
        symbol_idx = row.get("symbol_idx")
        if (
            isinstance(symbol_idx, bool)
            or not isinstance(symbol_idx, int)
            or symbol_idx not in expected_symbol_idxs
            or symbol_idx in seen_symbol_idxs
        ):
            raise FatalBotException(
                f"Rust orchestrator symbol input {input_idx} has invalid symbol_idx"
            )
        seen_symbol_idxs.add(symbol_idx)
        order_book = row.get("order_book")
        if not isinstance(order_book, dict):
            raise FatalBotException(
                f"Rust orchestrator symbol input {input_idx} has invalid order_book"
            )
        bid = _validated_rust_finite_number(
            order_book.get("bid"),
            f"symbol input {input_idx} has invalid order_book bid",
        )
        ask = _validated_rust_finite_number(
            order_book.get("ask"),
            f"symbol input {input_idx} has invalid order_book ask",
        )
        if bid <= 0.0 or ask <= 0.0:
            raise FatalBotException(
                f"Rust orchestrator symbol input {input_idx} has invalid order_book"
            )
        exchange = row.get("exchange")
        if not isinstance(exchange, dict):
            raise FatalBotException(
                f"Rust orchestrator symbol input {input_idx} has invalid exchange"
            )
        qty_step = _validated_rust_finite_number(
            exchange.get("qty_step"),
            f"symbol input {input_idx} has invalid exchange qty_step",
        )
        price_step = _validated_rust_finite_number(
            exchange.get("price_step"),
            f"symbol input {input_idx} has invalid exchange price_step",
        )
        min_qty = _validated_rust_finite_number(
            exchange.get("min_qty"),
            f"symbol input {input_idx} has invalid exchange min_qty",
        )
        min_cost = _validated_rust_finite_number(
            exchange.get("min_cost"),
            f"symbol input {input_idx} has invalid exchange min_cost",
        )
        c_mult = _validated_rust_finite_number(
            exchange.get("c_mult"),
            f"symbol input {input_idx} has invalid exchange c_mult",
        )
        if (
            qty_step <= 0.0
            or price_step <= 0.0
            or min_qty < 0.0
            or min_cost < 0.0
            or c_mult <= 0.0
        ):
            raise FatalBotException(
                f"Rust orchestrator symbol input {input_idx} has invalid exchange"
            )
        exchange_constraints[symbol_idx] = (
            qty_step,
            price_step,
            min_qty,
            min_cost,
            c_mult,
        )
        order_books[symbol_idx] = (
            _canonical_rust_order_book_value(bid, price_step),
            _canonical_rust_order_book_value(ask, price_step),
        )
        tradable = row.get("tradable")
        if not isinstance(tradable, bool):
            raise FatalBotException(
                f"Rust orchestrator symbol input {input_idx} has invalid tradable"
            )
        for pside in ("long", "short"):
            side_input = row.get(pside)
            if not isinstance(side_input, dict):
                raise FatalBotException(
                    f"Rust orchestrator symbol input {input_idx} has invalid {pside} input"
                )
            mode = side_input.get("mode") if "mode" in side_input else object()
            if mode is not None and (
                not isinstance(mode, str) or mode not in valid_modes
            ):
                raise FatalBotException(
                    f"Rust orchestrator symbol input {input_idx} has invalid {pside} mode"
                )
            modes[(symbol_idx, pside)] = mode
            position = side_input.get("position")
            if not isinstance(position, dict):
                raise FatalBotException(
                    f"Rust orchestrator symbol input {input_idx} has invalid {pside} position"
                )
            position_size = _validated_rust_finite_number(
                position.get("size"),
                f"symbol input {input_idx} has invalid {pside} position size",
            )
            position_sizes[(symbol_idx, pside)] = abs(position_size)
            bot_params = side_input.get("bot_params")
            if not isinstance(bot_params, dict):
                raise FatalBotException(
                    f"Rust orchestrator symbol input {input_idx} has invalid {pside} bot_params"
                )
            hsl_enabled = bot_params.get("hsl_enabled", False)
            panic_close_order_type = bot_params.get(
                "hsl_panic_close_order_type", "market"
            )
            if not isinstance(hsl_enabled, bool) or panic_close_order_type not in {
                "limit",
                "market",
            }:
                raise FatalBotException(
                    f"Rust orchestrator symbol input {input_idx} has invalid {pside} HSL execution policy"
                )
            hsl_execution_policy[(symbol_idx, pside)] = (
                hsl_enabled,
                panic_close_order_type,
            )
            cooldown_minutes = _validated_rust_finite_number(
                bot_params.get("risk_entry_cooldown_minutes", 0.0),
                f"symbol input {input_idx} has invalid {pside} risk_entry_cooldown_minutes",
            )
            if cooldown_minutes < 0.0:
                raise FatalBotException(
                    f"Rust orchestrator symbol input {input_idx} has invalid {pside} risk_entry_cooldown_minutes"
                )
            entry_cooldown_positive[(symbol_idx, pside)] = cooldown_minutes > 0.0
            entry_sequential_staging[(symbol_idx, pside)] = False
            close_retracement_enabled[(symbol_idx, pside)] = False
            strategy_params = side_input.get("strategy_params")
            if strategy_kind == "trailing_martingale" and isinstance(
                strategy_params, dict
            ):
                entry_params = strategy_params.get("entry")
                if isinstance(entry_params, dict):
                    retracement_base_pct = _validated_rust_finite_number(
                        entry_params.get("retracement_base_pct", 0.0),
                        f"symbol input {input_idx} has invalid {pside} entry retracement_base_pct",
                    )
                    entry_sequential_staging[(symbol_idx, pside)] = (
                        retracement_base_pct > 0.0
                    )
                close_params = strategy_params.get("close")
                if isinstance(close_params, dict):
                    close_retracement_base_pct = _validated_rust_finite_number(
                        close_params.get("retracement_base_pct", 0.0),
                        f"symbol input {input_idx} has invalid {pside} close retracement_base_pct",
                    )
                    close_retracement_enabled[(symbol_idx, pside)] = (
                        close_retracement_base_pct > 0.0
                    )
            last_fill_timestamp_ms = side_input.get(
                "last_increase_fill_timestamp_ms"
            )
            if last_fill_timestamp_ms is not None:
                last_fill_timestamp_ms = _validated_rust_u64(
                    last_fill_timestamp_ms,
                    f"symbol input {input_idx} has invalid {pside} last_increase_fill_timestamp_ms",
                )
            if cooldown_minutes > 0.0 and last_fill_timestamp_ms is not None:
                delay_float = cooldown_minutes * 60_000.0
                delay_ms = (
                    (1 << 64) - 1
                    if not math.isfinite(delay_float)
                    else min(math.ceil(delay_float), (1 << 64) - 1)
                )
                cooldown_until_ms = min(
                    last_fill_timestamp_ms + delay_ms, (1 << 64) - 1
                )
                entry_cooldown_active[(symbol_idx, pside)] = (
                    timestamp_ms < cooldown_until_ms
                )
            else:
                entry_cooldown_active[(symbol_idx, pside)] = False
            wallet_exposure_limit = _validated_rust_finite_number(
                bot_params.get("wallet_exposure_limit"),
                f"symbol input {input_idx} has invalid {pside} wallet_exposure_limit",
            )
            symbol_side_eligibility[(symbol_idx, pside)] = (
                tradable and wallet_exposure_limit != 0.0
            )
            wel_enabled = bot_params.get("risk_wel_enforcer_enabled", True)
            if not isinstance(wel_enabled, bool):
                raise FatalBotException(
                    f"Rust orchestrator symbol input {input_idx} has invalid {pside} WEL enforcer flag"
                )
            wel_threshold = _validated_rust_finite_number(
                bot_params.get("risk_wel_enforcer_threshold", 0.0),
                f"symbol input {input_idx} has invalid {pside} WEL enforcer threshold",
            )
            unstuck_enabled = bot_params.get("unstuck_enabled", True)
            if not isinstance(unstuck_enabled, bool):
                raise FatalBotException(
                    f"Rust orchestrator symbol input {input_idx} has invalid {pside} unstuck flag"
                )
            unstuck_values = {
                field: _validated_rust_finite_number(
                    bot_params.get(field, 0.0),
                    f"symbol input {input_idx} has invalid {pside} {field}",
                )
                for field in (
                    "unstuck_loss_allowance_pct",
                    "unstuck_close_pct",
                    "unstuck_threshold",
                )
            }
            enabled_families: set[str] = set()
            if wel_enabled and wel_threshold > 0.0:
                enabled_families.add("close_auto_reduce_wel")
            if global_twel_enforcer_enablement[pside]:
                enabled_families.add("close_auto_reduce_twel")
            if unstuck_enabled and all(value > 0.0 for value in unstuck_values.values()):
                enabled_families.add("close_unstuck")
            reducer_family_enablement[(symbol_idx, pside)] = frozenset(
                enabled_families
            )
    if seen_symbol_idxs != expected_symbol_idxs:
        raise FatalBotException(
            "Rust orchestrator symbol inputs do not cover the requested symbols"
        )
    return (
        modes,
        order_books,
        position_sizes,
        symbol_side_eligibility,
        global_side_enablement,
        global_side_n_positions,
        hedge_mode,
        strategy_kind,
        reducer_family_enablement,
        exchange_constraints,
        entry_cooldown_active,
        entry_cooldown_positive,
        entry_sequential_staging,
        close_retracement_enabled,
        hsl_execution_policy,
    )


def _submitted_auto_unstuck_allowed(global_input: dict) -> bool:
    explicit = global_input.get("auto_unstuck_allowed")
    if explicit is not None:
        if not isinstance(explicit, bool):
            raise FatalBotException(
                "Rust orchestrator global input has invalid auto_unstuck_allowed"
            )
        return explicit
    allowances = []
    for pside in ("long", "short"):
        field = f"unstuck_allowance_{pside}"
        allowance = _validated_rust_finite_number(
            global_input.get(field, 0.0),
            f"global input has invalid {field}",
        )
        if allowance < 0.0:
            raise FatalBotException(
                f"Rust orchestrator global input has invalid {field}"
            )
        allowances.append(allowance)
    return any(allowance > 0.0 for allowance in allowances)


def rust_order_conversion_identity(
    symbol_key: object, qty: object, price: object, order_type: object
) -> tuple[object, float, float, str]:
    """Return the identity for one order at the Rust-to-exchange conversion boundary."""
    qty_f = _validated_rust_finite_number(
        qty, "order conversion identity has invalid qty"
    )
    price_f = _validated_rust_finite_number(
        price, "order conversion identity has invalid price"
    )
    return (
        symbol_key,
        abs(qty_f),
        price_f,
        str(order_type),
    )


def validate_rust_orchestrator_output(
    out: object,
    idx_to_symbol: dict[int, str],
    orchestrator_input: object,
) -> list[dict]:
    """Validate the required Rust output envelope before any result is consumed."""
    if not isinstance(orchestrator_input, dict) or not isinstance(
        orchestrator_input.get("global"), dict
    ):
        raise FatalBotException(
            "Rust orchestrator validation missing corresponding global input"
        )
    global_input = orchestrator_input["global"]
    submitted_auto_unstuck_allowed = _submitted_auto_unstuck_allowed(global_input)
    if not isinstance(out, dict):
        raise FatalBotException("Rust orchestrator output must be a mapping")
    if "orders" not in out:
        raise FatalBotException("Rust orchestrator output missing required orders field")
    orders = out["orders"]
    if not isinstance(orders, list):
        raise FatalBotException("Rust orchestrator orders must be a list")
    expected_symbol_idxs = set(idx_to_symbol)
    (
        submitted_input_modes,
        submitted_order_books,
        submitted_position_sizes,
        submitted_symbol_side_eligibility,
        submitted_global_side_enablement,
        submitted_global_side_n_positions,
        submitted_hedge_mode,
        submitted_strategy_kind,
        submitted_reducer_family_enablement,
        submitted_exchange_constraints,
        submitted_entry_cooldown_active,
        submitted_entry_cooldown_positive,
        submitted_entry_sequential_staging,
        submitted_close_retracement_enabled,
        submitted_hsl_execution_policy,
    ) = _submitted_rust_input_context(orchestrator_input, expected_symbol_idxs)
    seen_conversion_identities: dict[tuple[object, float, float, str], int] = {}
    aggregate_close_qty: dict[tuple[int, str], float] = {}
    ema_anchor_entry_count: dict[tuple[int, str], int] = {}
    ema_anchor_close_count: dict[tuple[int, str], int] = {}
    trailing_martingale_close_count: dict[tuple[int, str], int] = {}
    initial_partial_entry_count: dict[tuple[int, str], int] = {}
    entry_order_count: dict[tuple[int, str], int] = {}
    panic_close_pairs: set[tuple[int, str]] = set()
    held_initial_normal_orders: list[tuple[int, str]] = []
    protective_reducer_order_indices: dict[tuple[int, str], int] = {}
    flat_entry_pairs: set[tuple[int, str]] = set()
    flat_entry_order_types: dict[tuple[int, str], list[str]] = {}
    for order_idx, order in enumerate(orders):
        if not isinstance(order, dict):
            raise FatalBotException(
                f"Rust orchestrator order {order_idx} must be a mapping"
            )
        symbol_idx = order.get("symbol_idx")
        if isinstance(symbol_idx, bool) or not isinstance(symbol_idx, int):
            raise FatalBotException(
                f"Rust orchestrator order {order_idx} has invalid symbol_idx"
            )
        if symbol_idx not in idx_to_symbol:
            raise FatalBotException(
                f"Rust orchestrator order {order_idx} has unknown symbol_idx {symbol_idx}"
            )
        pside = order.get("pside")
        if not isinstance(pside, str) or pside not in {"long", "short"}:
            raise FatalBotException(
                f"Rust orchestrator order {order_idx} has invalid pside"
            )
        qty = _validated_rust_finite_number(
            order.get("qty"), f"order {order_idx} has invalid qty"
        )
        if qty == 0.0:
            raise FatalBotException(
                f"Rust orchestrator order {order_idx} has invalid qty"
            )
        price = _validated_rust_finite_number(
            order.get("price"), f"order {order_idx} has invalid price"
        )
        if price <= 0.0:
            raise FatalBotException(
                f"Rust orchestrator order {order_idx} has invalid price"
            )
        order_type = order.get("order_type")
        if not isinstance(order_type, str) or not order_type:
            raise FatalBotException(
                f"Rust orchestrator order {order_idx} has invalid order_type"
            )
        try:
            order_type_id = _pb_attr("pbr").order_type_snake_to_id(order_type)
            if _pb_attr("pbr").order_type_id_to_snake(order_type_id) != order_type:
                raise ValueError("order type lookup did not round-trip")
            if order_type in _LEGACY_UNEMITTABLE_RUST_ORDER_TYPES:
                raise ValueError("legacy order type has no Rust producer")
            order_side = determine_side_from_order_tuple((qty, price, order_type))
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FatalBotException(
                f"Rust orchestrator order {order_idx} has invalid order_type"
            ) from exc
        if not order_type.endswith(f"_{pside}"):
            raise FatalBotException(
                f"Rust orchestrator order {order_idx} order_type disagrees with pside"
            )
        if (order_side == "buy") != (qty > 0.0):
            raise FatalBotException(
                f"Rust orchestrator order {order_idx} qty sign disagrees with order_type"
            )
        pair = (symbol_idx, pside)
        if not submitted_global_side_enablement[pside]:
            raise FatalBotException(
                f"Rust orchestrator order {order_idx} is inconsistent with globally disabled {pside}"
            )
        _validate_rust_order_family_for_submitted_mode(
            order_type,
            submitted_input_modes[pair],
            submitted_position_sizes[pair],
            submitted_symbol_side_eligibility[pair]
            and submitted_global_side_enablement[pside],
            f"Rust orchestrator order {order_idx}",
        )
        _validate_rust_order_family_for_submitted_strategy(
            order_type,
            submitted_strategy_kind,
            submitted_entry_sequential_staging[pair],
            submitted_close_retracement_enabled[pair],
            f"Rust orchestrator order {order_idx}",
        )
        if order_type.startswith("entry_"):
            entry_order_count[pair] = entry_order_count.get(pair, 0) + 1
            if (
                submitted_entry_cooldown_positive[pair]
                and entry_order_count[pair] > 1
            ):
                raise FatalBotException(
                    f"Rust orchestrator {pside} entry batch for symbol_idx {symbol_idx} "
                    "contains more than one entry with positive submitted cooldown"
                )
            if (
                order_type == f"entry_initial_normal_{pside}"
                and submitted_position_sizes[pair] != 0.0
            ):
                held_initial_normal_orders.append((order_idx, pside))
            if order_type == f"entry_initial_partial_{pside}":
                initial_partial_entry_count[pair] = (
                    initial_partial_entry_count.get(pair, 0) + 1
                )
                if initial_partial_entry_count[pair] > 1:
                    raise FatalBotException(
                        f"Rust orchestrator {pside} initial-partial entry batch for "
                        f"symbol_idx {symbol_idx} contains more than one entry"
                    )
            if order_type == f"entry_ema_anchor_{pside}":
                ema_anchor_entry_count[pair] = ema_anchor_entry_count.get(pair, 0) + 1
                if ema_anchor_entry_count[pair] > 1:
                    raise FatalBotException(
                        f"Rust orchestrator {pside} EMA Anchor entry batch for symbol_idx "
                        f"{symbol_idx} contains more than one entry"
                    )
            if submitted_entry_cooldown_active[pair]:
                raise FatalBotException(
                    f"Rust orchestrator order {order_idx} is inconsistent with submitted entry cooldown"
                )
        if order_type.startswith("entry_") and submitted_position_sizes[pair] == 0.0:
            flat_entry_pairs.add(pair)
            flat_entry_order_types.setdefault(pair, []).append(order_type)
        if _rust_order_requires_risk_critical_priority(order_type):
            if pair in protective_reducer_order_indices:
                previous_idx = protective_reducer_order_indices[pair]
                raise FatalBotException(
                    "Rust orchestrator orders "
                    f"{previous_idx} and {order_idx} contain competing protective reducers"
                )
            protective_reducer_order_indices[pair] = order_idx
        if order_type.startswith("close_"):
            if order_type == f"close_ema_anchor_{pside}":
                ema_anchor_close_count[pair] = (
                    ema_anchor_close_count.get(pair, 0) + 1
                )
                if ema_anchor_close_count[pair] > 1:
                    raise FatalBotException(
                        f"Rust orchestrator {pside} EMA Anchor close batch for symbol_idx "
                        f"{symbol_idx} contains more than one close"
                    )
            if (
                submitted_strategy_kind == "trailing_martingale"
                and order_type == f"close_trailing_{pside}"
            ):
                trailing_martingale_close_count[pair] = (
                    trailing_martingale_close_count.get(pair, 0) + 1
                )
                if trailing_martingale_close_count[pair] > 1:
                    raise FatalBotException(
                        f"Rust orchestrator {pside} trailing-martingale close batch for "
                        f"symbol_idx {symbol_idx} contains more than one trailing close"
                    )
            if order_type.startswith("close_panic_"):
                panic_close_pairs.add(pair)
            if order_type.startswith("close_panic_") and not math.isclose(
                abs(qty),
                abs(submitted_position_sizes[pair]),
                rel_tol=0.0,
                abs_tol=_rust_representation_tolerance(
                    qty, submitted_position_sizes[pair]
                ),
            ):
                raise FatalBotException(
                    f"Rust orchestrator order {order_idx} panic quantity does not equal submitted position"
                )
            aggregate_close_qty[pair] = aggregate_close_qty.get(pair, 0.0) + abs(qty)
            aggregate_tolerance = _rust_representation_tolerance(
                aggregate_close_qty[pair], submitted_position_sizes[pair]
            )
            if (
                aggregate_close_qty[pair]
                > submitted_position_sizes[pair] + aggregate_tolerance
            ):
                raise FatalBotException(
                    f"Rust orchestrator close quantity for symbol_idx {symbol_idx} "
                    f"{pside} exceeds submitted position"
                )
        execution_type = order.get("execution_type")
        if not isinstance(execution_type, str) or execution_type not in {
            "limit",
            "market",
        }:
            raise FatalBotException(
                f"Rust orchestrator order {order_idx} has invalid execution_type"
            )
        if execution_type == "limit":
            _validate_rust_limit_price_exchange_constraints(
                price,
                submitted_exchange_constraints[symbol_idx],
                f"Rust orchestrator order {order_idx} symbol_idx={symbol_idx} "
                f"order_type={order_type}",
            )
        expected_execution_type = _expected_rust_execution_type(
            global_input,
            submitted_hsl_execution_policy[pair],
            submitted_order_books[symbol_idx],
            order_type,
            qty,
            price,
        )
        if execution_type != expected_execution_type:
            raise FatalBotException(
                f"Rust orchestrator order {order_idx} has execution_type "
                "inconsistent with its submitted input"
            )
        if order_type.startswith("close_"):
            _validate_rust_close_exchange_constraints(
                qty,
                submitted_position_sizes[pair],
                submitted_order_books[symbol_idx],
                submitted_exchange_constraints[symbol_idx],
                f"Rust orchestrator order {order_idx}",
                minimum_price=price if execution_type == "limit" else None,
            )
        if order_type.startswith("entry_"):
            bid, ask = submitted_order_books[symbol_idx]
            entry_cost_price = (
                ask if qty > 0.0 else bid
            ) if execution_type == "market" else price
            _validate_rust_entry_exchange_constraints(
                qty,
                entry_cost_price,
                submitted_exchange_constraints[symbol_idx],
                f"Rust orchestrator order {order_idx}",
            )
        if order_type.startswith("close_panic_") and execution_type == "limit":
            _validate_rust_panic_limit_price(
                price,
                pside,
                submitted_order_books[symbol_idx],
                submitted_exchange_constraints[symbol_idx],
                f"Rust orchestrator order {order_idx}",
            )
        execution_priority = order.get("execution_priority")
        if not isinstance(execution_priority, str) or execution_priority not in {
            "ordinary",
            "risk_critical",
        }:
            raise FatalBotException(
                f"Rust orchestrator order {order_idx} has invalid execution_priority"
            )
        _validate_intrinsic_rust_execution_priority(
            order_type,
            execution_priority,
            f"Rust orchestrator order {order_idx}",
        )
        _validate_rust_reducer_enablement(
            order_type,
            pair,
            submitted_reducer_family_enablement,
            submitted_auto_unstuck_allowed,
            f"Rust orchestrator order {order_idx}",
        )
        conversion_identity = rust_order_conversion_identity(
            idx_to_symbol[symbol_idx], qty, price, order_type
        )
        if conversion_identity in seen_conversion_identities:
            previous_idx = seen_conversion_identities[conversion_identity]
            raise FatalBotException(
                "Rust orchestrator orders "
                f"{previous_idx} and {order_idx} collide under conversion identity"
            )
        seen_conversion_identities[conversion_identity] = order_idx
        if (
            order_type.startswith("entry_")
            and submitted_entry_sequential_staging[pair]
            and entry_order_count[pair] > 1
        ):
            raise FatalBotException(
                f"Rust orchestrator {pside} entry batch for symbol_idx {symbol_idx} "
                "contains more than one entry with positive submitted retracement"
            )

    if held_initial_normal_orders:
        order_idx, pside = held_initial_normal_orders[0]
        raise FatalBotException(
            f"Rust orchestrator order {order_idx} {pside} initial-normal entry "
            "requires a flat submitted side"
        )

    for (symbol_idx, pside), order_types in flat_entry_order_types.items():
        _validate_rust_flat_entry_batch(
            order_types,
            submitted_strategy_kind,
            pside,
            f"Rust orchestrator flat {pside} entry batch for symbol_idx {symbol_idx}",
        )

    diagnostics = out.get("diagnostics")
    if not isinstance(diagnostics, dict):
        raise FatalBotException("Rust orchestrator output missing valid diagnostics")
    if "warnings" not in diagnostics:
        raise FatalBotException("Rust orchestrator diagnostics missing required warnings")
    warnings = diagnostics["warnings"]
    if not isinstance(warnings, list):
        raise FatalBotException("Rust orchestrator warnings must be a list")
    warning_shapes = {
        "disabled_pside_has_position": {"symbol_idx", "pside"},
        "non_tradable_has_position": {"symbol_idx", "pside"},
        "strategy_input_unavailable": {"symbol_idx", "pside", "scope"},
        "twel_repair_blocked_by_loss_gate": {
            "pside",
            "current_twe",
            "twel_repair_target",
            "policy",
            "candidate_count",
            "blocked_order_count",
            "projected_twe_after_allowed_reductions",
        },
    }
    for warning_idx, warning in enumerate(warnings):
        context = f"warning {warning_idx}"
        if not isinstance(warning, dict) or len(warning) != 1:
            raise FatalBotException(
                f"Rust orchestrator {context} must contain exactly one warning variant"
            )
        variant, details = next(iter(warning.items()))
        if variant not in warning_shapes or not isinstance(details, dict):
            raise FatalBotException(
                f"Rust orchestrator {context} has invalid warning variant"
            )
        if set(details) != warning_shapes[variant]:
            raise FatalBotException(
                f"Rust orchestrator {context} has invalid warning fields"
            )
        pside = details.get("pside")
        if not isinstance(pside, str) or pside not in {"long", "short"}:
            raise FatalBotException(
                f"Rust orchestrator {context} has invalid pside"
            )
        if variant in {
            "disabled_pside_has_position",
            "non_tradable_has_position",
            "strategy_input_unavailable",
        }:
            symbol_idx = details.get("symbol_idx")
            if (
                isinstance(symbol_idx, bool)
                or not isinstance(symbol_idx, int)
                or symbol_idx not in expected_symbol_idxs
            ):
                raise FatalBotException(
                    f"Rust orchestrator {context} has invalid symbol_idx"
                )
            if variant == "strategy_input_unavailable":
                scope = details.get("scope")
                if not isinstance(scope, str) or scope not in {
                    "forager_selection",
                    "one_way_arbitration",
                    "strategy_orders",
                    "unstuck",
                }:
                    raise FatalBotException(
                        f"Rust orchestrator {context} has invalid scope"
                    )
            continue
        for field in (
            "current_twe",
            "twel_repair_target",
            "projected_twe_after_allowed_reductions",
        ):
            _validated_rust_finite_number(
                details.get(field), f"{context} has invalid {field}"
            )
        policy = details.get("policy")
        if not isinstance(policy, str) or policy not in {
            "reduce_overweight",
            "reduce_portfolio",
        }:
            raise FatalBotException(
                f"Rust orchestrator {context} has invalid policy"
            )
        for field in ("candidate_count", "blocked_order_count"):
            value = details.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise FatalBotException(
                    f"Rust orchestrator {context} has invalid {field}"
                )
    if "symbol_states" not in diagnostics:
        raise FatalBotException(
            "Rust orchestrator diagnostics missing required symbol_states"
        )
    symbol_states = diagnostics["symbol_states"]
    if not isinstance(symbol_states, list):
        raise FatalBotException("Rust orchestrator symbol_states must be a list")
    seen_symbol_idxs: set[int] = set()
    submitted_symbol_states: dict[tuple[int, str], dict] = {}
    valid_modes = {"normal", "panic", "graceful_stop", "tp_only", "manual"}
    for state_idx, row in enumerate(symbol_states):
        if not isinstance(row, dict):
            raise FatalBotException(
                f"Rust orchestrator symbol_state {state_idx} must be a mapping"
            )
        symbol_idx = row.get("symbol_idx")
        if (
            isinstance(symbol_idx, bool)
            or not isinstance(symbol_idx, int)
            or symbol_idx not in expected_symbol_idxs
            or symbol_idx in seen_symbol_idxs
        ):
            raise FatalBotException(
                f"Rust orchestrator symbol_state {state_idx} has invalid symbol_idx"
            )
        seen_symbol_idxs.add(symbol_idx)
        for pside in ("long", "short"):
            side_state = row.get(pside)
            if not isinstance(side_state, dict):
                raise FatalBotException(
                    f"Rust orchestrator symbol_state {state_idx} has invalid {pside} state"
                )
            input_mode = (
                side_state.get("input_mode")
                if "input_mode" in side_state
                else object()
            )
            if input_mode is not None and (
                not isinstance(input_mode, str) or input_mode not in valid_modes
            ):
                raise FatalBotException(
                    f"Rust orchestrator symbol_state {state_idx} has invalid {pside} input_mode"
                )
            if input_mode != submitted_input_modes[(symbol_idx, pside)]:
                raise FatalBotException(
                    f"Rust orchestrator symbol_state {state_idx} has {pside} input_mode "
                    "inconsistent with its submitted input"
                )
            effective_mode = side_state.get("effective_mode")
            if not isinstance(effective_mode, str) or effective_mode not in valid_modes:
                raise FatalBotException(
                    f"Rust orchestrator symbol_state {state_idx} has invalid {pside} effective_mode"
                )
            expected_effective_mode = _expected_rust_effective_mode(
                submitted_input_modes[(symbol_idx, pside)],
                submitted_position_sizes[(symbol_idx, pside)] != 0.0,
                submitted_global_side_enablement[pside],
            )
            if effective_mode != expected_effective_mode:
                raise FatalBotException(
                    f"Rust orchestrator symbol_state {state_idx} has {pside} effective_mode "
                    "inconsistent with its submitted input"
                )
            for field in ("active", "allow_initial"):
                if not isinstance(side_state.get(field), bool):
                    raise FatalBotException(
                        f"Rust orchestrator symbol_state {state_idx} has invalid {pside} {field}"
                    )
            submitted_symbol_states[(symbol_idx, pside)] = side_state
            if (
                side_state["active"]
                and (
                    not submitted_symbol_side_eligibility[(symbol_idx, pside)]
                    or (
                        submitted_position_sizes[(symbol_idx, pside)] == 0.0
                        and not submitted_global_side_enablement[pside]
                    )
                )
            ):
                raise FatalBotException(
                    f"Rust orchestrator symbol_state {state_idx} has {pside} active "
                    "inconsistent with submitted eligibility"
                )
            if (
                not side_state["active"]
                and submitted_symbol_side_eligibility[(symbol_idx, pside)]
                and submitted_position_sizes[(symbol_idx, pside)] != 0.0
                and submitted_input_modes[(symbol_idx, pside)] != "manual"
            ):
                raise FatalBotException(
                    f"Rust orchestrator symbol_state {state_idx} has {pside} inactive "
                    "inconsistent with submitted managed position"
                )
    if seen_symbol_idxs != expected_symbol_idxs:
        raise FatalBotException(
            "Rust orchestrator symbol_states do not cover the requested symbols"
        )

    submitted_symbol_idx_order = [
        int(row["symbol_idx"]) for row in orchestrator_input["symbols"]
    ]
    for pside in ("long", "short"):
        eligible_count = sum(
            submitted_symbol_side_eligibility[(symbol_idx, pside)]
            for symbol_idx in expected_symbol_idxs
        )
        forced_normal_count = sum(
            submitted_symbol_side_eligibility[(symbol_idx, pside)]
            and submitted_input_modes[(symbol_idx, pside)] == "normal"
            for symbol_idx in expected_symbol_idxs
        )
        effective_n_positions = max(
            min(submitted_global_side_n_positions[pside], eligible_count),
            forced_normal_count,
        )
        held_workspace_count = sum(
            submitted_position_sizes[(symbol_idx, pside)] != 0.0
            and submitted_input_modes[(symbol_idx, pside)] != "manual"
            for symbol_idx in expected_symbol_idxs
        )
        workspace_active_idxs = {
            symbol_idx
            for symbol_idx in expected_symbol_idxs
            if submitted_position_sizes[(symbol_idx, pside)] != 0.0
            and submitted_input_modes[(symbol_idx, pside)] != "manual"
        }
        if submitted_global_side_enablement[pside]:
            opposite_pside = "short" if pside == "long" else "long"
            for symbol_idx in submitted_symbol_idx_order:
                if len(workspace_active_idxs) >= effective_n_positions:
                    break
                pair = (symbol_idx, pside)
                if (
                    symbol_idx in workspace_active_idxs
                    or not submitted_symbol_side_eligibility[pair]
                    or submitted_input_modes[pair] != "normal"
                    or (
                        not submitted_hedge_mode
                        and submitted_position_sizes[(symbol_idx, opposite_pside)]
                        != 0.0
                    )
                ):
                    continue
                workspace_active_idxs.add(symbol_idx)
                if not submitted_symbol_states[pair]["active"]:
                    raise FatalBotException(
                        f"Rust orchestrator symbol_state for symbol_idx {symbol_idx} "
                        f"has {pside} inactive inconsistent with submitted forced-normal capacity"
                    )
        max_flat_active = max(effective_n_positions - held_workspace_count, 0)
        flat_active_count = sum(
            submitted_position_sizes[(symbol_idx, pside)] == 0.0
            and submitted_symbol_states[(symbol_idx, pside)]["active"]
            for symbol_idx in expected_symbol_idxs
        )
        if flat_active_count > max_flat_active:
            raise FatalBotException(
                f"Rust orchestrator {pside} flat active set exceeds submitted position cap"
            )

    for pair in flat_entry_pairs:
        symbol_idx, pside = pair
        side_state = submitted_symbol_states[pair]
        if not side_state["active"] or not side_state["allow_initial"]:
            raise FatalBotException(
                f"Rust orchestrator flat {pside} entry for symbol_idx {symbol_idx} "
                "contradicts submitted symbol state"
            )

    if not submitted_hedge_mode:
        for symbol_idx in expected_symbol_idxs:
            long_allow_initial = submitted_symbol_states[(symbol_idx, "long")][
                "allow_initial"
            ]
            short_allow_initial = submitted_symbol_states[(symbol_idx, "short")][
                "allow_initial"
            ]
            if long_allow_initial and (
                submitted_position_sizes[(symbol_idx, "short")] != 0.0
                or short_allow_initial
            ):
                raise FatalBotException(
                    f"Rust orchestrator symbol_state for symbol_idx {symbol_idx} "
                    "violates submitted one-way position-side exclusion"
                )
            if short_allow_initial and submitted_position_sizes[(symbol_idx, "long")] != 0.0:
                raise FatalBotException(
                    f"Rust orchestrator symbol_state for symbol_idx {symbol_idx} "
                    "violates submitted one-way position-side exclusion"
                )

    for order_idx, order in enumerate(orders):
        order_type = str(order["order_type"])
        input_mode = submitted_input_modes[
            (int(order["symbol_idx"]), str(order["pside"]))
        ]
        expected_priority = _expected_rust_execution_priority(
            order_type,
            input_mode,
        )
        if order["execution_priority"] != expected_priority:
            raise FatalBotException(
                f"Rust orchestrator order {order_idx} has execution_priority "
                "inconsistent with its order_type and input mode"
            )

    if "loss_gate_blocks" not in diagnostics:
        raise FatalBotException(
            "Rust orchestrator diagnostics missing required loss_gate_blocks"
        )
    loss_gate_blocks = diagnostics["loss_gate_blocks"]
    if not isinstance(loss_gate_blocks, list):
        raise FatalBotException("Rust orchestrator loss_gate_blocks must be a list")
    submitted_max_realized_loss_pct: float | None = None
    if loss_gate_blocks:
        submitted_max_realized_loss_pct = _validated_rust_finite_number(
            global_input.get("max_realized_loss_pct"),
            "submitted global input has invalid max_realized_loss_pct",
        )
        if submitted_max_realized_loss_pct < 0.0:
            raise FatalBotException(
                "Rust orchestrator submitted global input has invalid max_realized_loss_pct"
            )
        if submitted_max_realized_loss_pct >= 1.0:
            raise FatalBotException(
                "Rust orchestrator loss_gate_blocks present while submitted "
                "realized-loss gate is disabled"
            )
    finite_fields = (
        "qty",
        "price",
        "projected_pnl",
        "balance_before",
        "projected_balance_after",
        "balance_peak",
        "balance_floor",
        "max_realized_loss_pct",
    )
    for block_idx, block in enumerate(loss_gate_blocks):
        if not isinstance(block, dict):
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} must be a mapping"
            )
        symbol_idx = block.get("symbol_idx")
        if (
            isinstance(symbol_idx, bool)
            or not isinstance(symbol_idx, int)
            or symbol_idx not in expected_symbol_idxs
        ):
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} has invalid symbol_idx"
            )
        pside = block.get("pside")
        if not isinstance(pside, str) or pside not in {"long", "short"}:
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} has invalid pside"
            )
        order_type = block.get("order_type")
        if not isinstance(order_type, str) or not order_type.endswith(f"_{pside}"):
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} has invalid order_type"
            )
        try:
            _pb_attr("pbr").order_type_snake_to_id(order_type)
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError) as exc:
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} has invalid order_type"
            ) from exc
        if not order_type.startswith("close_"):
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} order_type must be a close order"
            )
        if order_type.startswith("close_panic_"):
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} panic order_type "
                "bypasses the realized-loss gate"
            )
        pair = (symbol_idx, pside)
        if not submitted_global_side_enablement[pside]:
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} is inconsistent "
                f"with globally disabled {pside}"
            )
        if submitted_position_sizes[pair] == 0.0:
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} requires a submitted position"
            )
        _validate_rust_order_family_for_submitted_mode(
            order_type,
            submitted_input_modes[pair],
            submitted_position_sizes[pair],
            False,
            f"Rust orchestrator loss_gate_block {block_idx}",
        )
        _validate_rust_order_family_for_submitted_strategy(
            order_type,
            submitted_strategy_kind,
            submitted_entry_sequential_staging[pair],
            submitted_close_retracement_enabled[pair],
            f"Rust orchestrator loss_gate_block {block_idx}",
        )
        _validate_rust_reducer_enablement(
            order_type,
            pair,
            submitted_reducer_family_enablement,
            submitted_auto_unstuck_allowed,
            f"Rust orchestrator loss_gate_block {block_idx}",
        )
        finite_values: dict[str, float] = {}
        for field in finite_fields:
            value = _validated_rust_finite_number(
                block.get(field),
                f"loss_gate_block {block_idx} has invalid {field}",
            )
            finite_values[field] = value
            if field == "qty" and value == 0.0:
                raise FatalBotException(
                    f"Rust orchestrator loss_gate_block {block_idx} has invalid qty"
                )
            if field == "price" and value <= 0.0:
                raise FatalBotException(
                    f"Rust orchestrator loss_gate_block {block_idx} has invalid price"
                )
        qty = finite_values["qty"]
        _validate_rust_limit_price_exchange_constraints(
            finite_values["price"],
            submitted_exchange_constraints[symbol_idx],
            f"Rust orchestrator loss_gate_block {block_idx}",
        )
        if (pside == "long" and qty >= 0.0) or (pside == "short" and qty <= 0.0):
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} qty sign disagrees with pside"
            )
        block_execution_type = _expected_rust_execution_type(
            global_input,
            submitted_hsl_execution_policy[pair],
            submitted_order_books[symbol_idx],
            order_type,
            qty,
            finite_values["price"],
        )
        _validate_rust_close_exchange_constraints(
            qty,
            submitted_position_sizes[pair],
            submitted_order_books[symbol_idx],
            submitted_exchange_constraints[symbol_idx],
            f"Rust orchestrator loss_gate_block {block_idx}",
            minimum_price=(
                finite_values["price"]
                if block_execution_type == "limit"
                else None
            ),
        )
        qty_tolerance = _rust_representation_tolerance(
            abs(qty), submitted_position_sizes[pair]
        )
        if abs(qty) > submitted_position_sizes[pair] + qty_tolerance:
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} qty exceeds submitted position"
            )
        projected_pnl = finite_values["projected_pnl"]
        balance_before = finite_values["balance_before"]
        projected_balance_after = finite_values["projected_balance_after"]
        balance_peak = finite_values["balance_peak"]
        balance_floor = finite_values["balance_floor"]
        max_realized_loss_pct = finite_values["max_realized_loss_pct"]
        if projected_pnl >= 0.0:
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} must have negative projected_pnl"
            )
        if balance_before <= 0.0 or balance_peak <= 0.0 or balance_floor <= 0.0:
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} has invalid balance state"
            )
        if not 0.0 <= max_realized_loss_pct < 1.0:
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} has invalid max_realized_loss_pct"
            )
        if max_realized_loss_pct != submitted_max_realized_loss_pct:
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} max_realized_loss_pct "
                "is inconsistent with submitted policy"
            )
        if not math.isclose(
            projected_balance_after,
            balance_before + projected_pnl,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} has inconsistent projected balance"
            )
        if not math.isclose(
            balance_floor,
            balance_peak * (1.0 - max_realized_loss_pct),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} has inconsistent balance floor"
            )
        if projected_balance_after >= balance_floor - 1e-12:
            raise FatalBotException(
                f"Rust orchestrator loss_gate_block {block_idx} does not cross balance floor"
            )

    def validate_diagnostic_symbol_idx(value: object, context: str) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value not in expected_symbol_idxs
        ):
            raise FatalBotException(
                f"Rust orchestrator {context} has invalid symbol_idx"
            )

    def validate_diagnostic_pside(value: object, context: str) -> None:
        if not isinstance(value, str) or value not in {"long", "short"}:
            raise FatalBotException(f"Rust orchestrator {context} has invalid pside")

    def validate_diagnostic_finite_fields(
        item: dict, fields: tuple[str, ...], context: str
    ) -> None:
        for field in fields:
            _validated_rust_finite_number(
                item.get(field), f"{context} has invalid {field}"
            )

    def validate_diagnostic_symbol_idx_list(value: object, context: str) -> None:
        if not isinstance(value, list):
            raise FatalBotException(f"Rust orchestrator {context} must be a list")
        for symbol_idx in value:
            validate_diagnostic_symbol_idx(symbol_idx, context)

    if "min_effective_cost_blocks" not in diagnostics:
        raise FatalBotException(
            "Rust orchestrator diagnostics missing required min_effective_cost_blocks"
        )
    min_effective_cost_blocks = diagnostics["min_effective_cost_blocks"]
    if not isinstance(min_effective_cost_blocks, list):
        raise FatalBotException(
            "Rust orchestrator min_effective_cost_blocks must be a list"
        )
    min_cost_finite_fields = (
        "balance",
        "effective_limit",
        "entry_initial_qty_pct",
        "projected_initial_cost",
        "effective_min_cost",
    )
    for block_idx, block in enumerate(min_effective_cost_blocks):
        context = f"min_effective_cost_block {block_idx}"
        if not isinstance(block, dict):
            raise FatalBotException(f"Rust orchestrator {context} must be a mapping")
        validate_diagnostic_symbol_idx(block.get("symbol_idx"), context)
        validate_diagnostic_pside(block.get("pside"), context)
        validate_diagnostic_finite_fields(block, min_cost_finite_fields, context)

    if "forager_selections" not in diagnostics:
        raise FatalBotException(
            "Rust orchestrator diagnostics missing required forager_selections"
        )
    forager_selections = diagnostics["forager_selections"]
    if not isinstance(forager_selections, list):
        raise FatalBotException("Rust orchestrator forager_selections must be a list")
    score_finite_fields = (
        "score",
        "volume_component",
        "ema_readiness_component",
        "volatility_component",
    )
    event_finite_fields = (
        "incumbent_score",
        "challenger_score",
        "score_gap",
    )
    forager_selected_pairs: set[tuple[int, str]] = set()
    for selection_idx, selection in enumerate(forager_selections):
        context = f"forager_selection {selection_idx}"
        if not isinstance(selection, dict):
            raise FatalBotException(f"Rust orchestrator {context} must be a mapping")
        selection_pside = selection.get("pside")
        validate_diagnostic_pside(selection_pside, context)
        slots_to_fill = selection.get("slots_to_fill")
        if (
            isinstance(slots_to_fill, bool)
            or not isinstance(slots_to_fill, int)
            or slots_to_fill < 0
        ):
            raise FatalBotException(
                f"Rust orchestrator {context} has invalid slots_to_fill"
            )
        if not isinstance(selection.get("ranking_required"), bool):
            raise FatalBotException(
                f"Rust orchestrator {context} has invalid ranking_required"
            )
        validate_diagnostic_finite_fields(selection, ("score_hysteresis_pct",), context)
        for field in ("selected_symbol_indices", "incumbent_symbol_indices"):
            validate_diagnostic_symbol_idx_list(
                selection.get(field), f"{context} {field}"
            )
        selected_symbol_indices = selection["selected_symbol_indices"]
        for symbol_idx in selected_symbol_indices:
            pair = (symbol_idx, selection_pside)
            side_state = submitted_symbol_states[pair]
            if (
                submitted_position_sizes[pair] != 0.0
                or not side_state["active"]
            ):
                raise FatalBotException(
                    f"Rust orchestrator {context} selected_symbol_indices "
                    "disagree with submitted flat active symbol states"
                )
            forager_selected_pairs.add(pair)

        top_scores = selection.get("top_scores")
        if not isinstance(top_scores, list):
            raise FatalBotException(
                f"Rust orchestrator {context} top_scores must be a list"
            )
        for score_idx, score in enumerate(top_scores):
            score_context = f"{context} top_score {score_idx}"
            if not isinstance(score, dict):
                raise FatalBotException(
                    f"Rust orchestrator {score_context} must be a mapping"
                )
            validate_diagnostic_symbol_idx(score.get("symbol_idx"), score_context)
            rank = score.get("rank")
            if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
                raise FatalBotException(
                    f"Rust orchestrator {score_context} has invalid rank"
                )
            validate_diagnostic_finite_fields(score, score_finite_fields, score_context)
            for field in ("selected", "incumbent"):
                if not isinstance(score.get(field), bool):
                    raise FatalBotException(
                        f"Rust orchestrator {score_context} has invalid {field}"
                    )

        hysteresis_events = selection.get("hysteresis_events")
        if not isinstance(hysteresis_events, list):
            raise FatalBotException(
                f"Rust orchestrator {context} hysteresis_events must be a list"
            )
        for event_idx, event in enumerate(hysteresis_events):
            event_context = f"{context} hysteresis_event {event_idx}"
            if not isinstance(event, dict):
                raise FatalBotException(
                    f"Rust orchestrator {event_context} must be a mapping"
                )
            for field in ("incumbent_symbol_idx", "challenger_symbol_idx"):
                validate_diagnostic_symbol_idx(
                    event.get(field), f"{event_context} {field}"
                )
            validate_diagnostic_finite_fields(event, event_finite_fields, event_context)
            if not isinstance(event.get("kept_incumbent"), bool):
                raise FatalBotException(
                    f"Rust orchestrator {event_context} has invalid kept_incumbent"
                )
    for pair, side_state in submitted_symbol_states.items():
        if (
            submitted_position_sizes[pair] == 0.0
            and side_state["active"]
            and submitted_input_modes[pair] != "normal"
            and pair not in forager_selected_pairs
        ):
            symbol_idx, pside = pair
            raise FatalBotException(
                f"Rust orchestrator flat active {pside} symbol_idx {symbol_idx} "
                "is missing from submitted forager selections"
            )
    for pair, position_size in submitted_position_sizes.items():
        symbol_idx, pside = pair
        if (
            submitted_global_side_enablement[pside]
            and submitted_input_modes[pair] == "panic"
            and position_size > 1e-12
            and pair not in panic_close_pairs
        ):
            raise FatalBotException(
                f"Rust orchestrator panic {pside} batch for symbol_idx {symbol_idx} "
                "is missing required full-position panic close"
            )
    return orders


def _reject_duplicate_json_object_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(constant: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant {constant!r}")


def _parse_finite_json_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON float {value!r}")
    return parsed


def parse_and_validate_rust_orchestrator_output(
    out_json: object,
    idx_to_symbol: dict[int, str],
    orchestrator_input: object,
) -> tuple[dict, list[dict]]:
    """Decode and validate Rust output, preserving malformed-output fatality."""
    try:
        out = json.loads(
            out_json,
            object_pairs_hook=_reject_duplicate_json_object_keys,
            parse_constant=_reject_nonstandard_json_constant,
            parse_float=_parse_finite_json_float,
        )
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise FatalBotException("Rust orchestrator returned malformed JSON") from exc
    orders = validate_rust_orchestrator_output(
        out, idx_to_symbol, orchestrator_input
    )
    return out, orders


def order_churn_risk_active_pairs_from_rust_output(
    out: dict, idx_to_symbol: dict[int, str]
) -> tuple[tuple[str, str], ...]:
    """Return symbol/pside pairs whose Rust plan is under an active risk phase.

    Raw balance is intentionally not an epoch scalar: quote-valued collateral may
    move on every market tick.  Rust remains authoritative for raw-balance risk
    behavior, and its emitted risk-critical orders plus realized-loss blocks are
    the behavioral phase boundary needed by churn reconciliation.
    """
    if not isinstance(out, dict):
        raise ValueError("Rust orchestrator output must be a dict")
    pairs: set[tuple[str, str]] = set()

    def add_pair(item: dict, *, context: str) -> None:
        if not isinstance(item, dict):
            raise ValueError(f"Rust {context} item must be a dict")
        try:
            symbol_idx = int(item["symbol_idx"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Rust {context} item missing valid symbol_idx") from exc
        symbol = idx_to_symbol.get(symbol_idx)
        if not symbol:
            raise ValueError(f"Rust {context} item has unknown symbol_idx {symbol_idx}")
        pside = str(item.get("pside") or "").lower()
        if pside not in {"long", "short"}:
            raise ValueError(f"Rust {context} item missing valid pside")
        pairs.add((str(symbol), pside))

    orders = out["orders"]
    if not isinstance(orders, list):
        raise ValueError("Rust orchestrator orders must be a list")
    for order in orders:
        if not isinstance(order, dict):
            raise ValueError("Rust orchestrator order must be a dict")
        if str(order.get("execution_priority") or "").lower() == "risk_critical":
            add_pair(order, context="risk-critical order")

    diagnostics = out["diagnostics"]
    if not isinstance(diagnostics, dict):
        raise ValueError("Rust orchestrator diagnostics must be a dict")
    loss_gate_blocks = diagnostics.get("loss_gate_blocks", [])
    if not isinstance(loss_gate_blocks, list):
        raise ValueError("Rust loss_gate_blocks must be a list")
    for block in loss_gate_blocks:
        add_pair(block, context="loss-gate block")
    return tuple(sorted(pairs))


def _emit_order_churn_evidence_summary(
    bot,
    *,
    state: OrderChurnGateState,
    generation: int,
    reset: bool,
    decisions: Iterable[ChurnDecision],
    symbols: Iterable[str],
    snapshot_status: str = "observed",
) -> None:
    emitter = getattr(bot, "_emit_order_churn_evidence_event", None)
    if not callable(emitter):
        return
    decisions = list(decisions)
    reason_counts = Counter(decision.reason for decision in decisions)
    try:
        emitter(
            generation=generation,
            reset=reset,
            reset_count=state.reset_count,
            reason_counts=reason_counts,
            churn_count=sum(decision.churn_evidenced for decision in decisions),
            order_count=len(decisions),
            symbols=symbols,
            history_symbol_count=len(state.history_by_symbol),
            snapshot_status=snapshot_status,
        )
    except Exception as exc:
        logging.debug(
            "[event] order churn evidence emitter failed | error_type=%s",
            bounded_exception_type(exc),
        )


def prepare_order_churn_evidence(
    bot, ideal_orders: dict, *, generation: int
) -> None:
    """Annotate Rust ideals from recent behavior without gating reconciliation."""
    state = bot._order_churn_gate_state
    activation_count = int(bot.live_value("order_replacement_churn_gate_activation_count"))
    reset = False
    if not state.history_started:
        state.history_started = True
        reset = True
        logging.info(
            "[order] churn evidence history initialized empty | reason=process_start"
        )
    current_universe = set(ideal_orders)
    current_universe.update(getattr(bot, "active_symbols", []) or [])
    current_universe.update((getattr(bot, "open_orders", {}) or {}).keys())
    current_universe.update(
        symbol
        for symbol in (getattr(bot, "positions", {}) or {})
        if _symbol_position_state(bot, symbol) != "flat"
    )
    history_symbols = state.symbols_with_history()
    if activation_count > 0:
        for symbol in history_symbols - current_universe:
            reset = state.clear_symbol_history(symbol) or reset
    complete_ideals = {
        str(symbol): list(ideal_orders.get(symbol, [])) for symbol in current_universe
    }
    if activation_count <= 0:
        reset = state.clear_history() or reset
        for orders in complete_ideals.values():
            for order in orders:
                order["_churn_evidence"] = False
                order["_churn_reason"] = "disabled"
        _emit_order_churn_evidence_summary(
            bot,
            state=state,
            generation=generation,
            reset=reset,
            decisions=[
                ChurnDecision(False, "disabled")
                for orders in complete_ideals.values()
                for _order in orders
            ],
            symbols=complete_ideals,
        )
        return
    window_seconds = float(bot.live_value("order_replacement_churn_gate_window_minutes")) * 60.0
    stability_seconds = (
        float(bot.live_value("order_replacement_churn_gate_stability_minutes")) * 60.0
    )
    decisions = state.evaluate_and_record(
        complete_ideals,
        now_monotonic=time.monotonic(),
        tolerance=float(bot.live_value("order_match_tolerance_pct")),
        stability_seconds=stability_seconds,
        window_seconds=window_seconds,
        max_sample_gap_seconds=_order_churn_max_generation_gap_seconds(bot),
    )
    reset = state.history_reset_during_evaluation or reset
    risk_active_pairs = set(
        getattr(bot, "_order_churn_risk_active_pairs", ()) or ()
    )
    for orders in complete_ideals.values():
        for order in orders:
            decision = decisions.get(id(order), ChurnDecision(False, "unavailable"))
            pair = (str(order.get("symbol") or ""), str(order.get("position_side") or ""))
            if pair in risk_active_pairs:
                # Raw-balance risk behavior remains owned by Rust.  Never let
                # economy-only churn admission delay any order sharing the
                # affected symbol/pside while that risk phase is active.
                decision = ChurnDecision(False, "rust_risk_phase_active")
                decisions[id(order)] = decision
            order["_churn_evidence"] = bool(decision.churn_evidenced)
            order["_churn_reason"] = decision.reason
    _emit_order_churn_evidence_summary(
        bot,
        state=state,
        generation=generation,
        reset=reset,
        decisions=decisions.values(),
        symbols=complete_ideals,
    )


async def calc_orders_to_cancel_and_create_from_ideal(
    bot,
    ideal_orders,
    *,
    actual_symbols: Optional[Iterable[str]] = None,
    actual_psides_by_symbol: Optional[dict[str, Iterable[str]]] = None,
    apply_mode_filters: bool = True,
    collect_fresh_entry_eligibility: bool = True,
):
    """Reconcile exchange orders against a supplied ideal order map."""
    bot._fresh_entry_eligibility_trace = None
    if not hasattr(bot, "_last_plan_detail"):
        bot._last_plan_detail = {}

    if actual_symbols is None:
        actual_symbols = sorted(
            set(getattr(bot, "active_symbols", []) or [])
            | set(ideal_orders if isinstance(ideal_orders, dict) else {})
            | set((getattr(bot, "open_orders", {}) or {}).keys())
            | set((getattr(bot, "positions", {}) or {}).keys())
        )
    actual_orders = bot._snapshot_actual_orders(
        actual_symbols, psides_by_symbol=actual_psides_by_symbol
    )
    trace = (
        _initialize_fresh_entry_trace(bot, ideal_orders, actual_orders)
        if collect_fresh_entry_eligibility and isinstance(ideal_orders, dict)
        else None
    )
    malformed_actual_symbols = set(
        getattr(bot, "_malformed_actual_order_symbols", set()) or set()
    )
    malformed_actual_counts = dict(
        getattr(bot, "_malformed_actual_order_counts", {}) or {}
    )
    connector_enabled = connector_supports_order_churn_gate(bot)
    keys = (
        (
            "symbol",
            "side",
            "position_side",
            "reduce_only",
            "type",
            "pb_order_type",
            "qty",
            "price",
        )
        if connector_enabled
        else ("symbol", "side", "position_side", "qty", "price")
    )
    to_cancel, to_create = [], []
    plan_summaries = []
    for symbol, symbol_orders in actual_orders.items():
        ideal_list = (
            ideal_orders.get(symbol, []) if isinstance(ideal_orders, dict) else []
        )
        if symbol in malformed_actual_symbols:
            trace = _trace_record(
                trace,
                "record_blocked_orders",
                ideal_list,
                "malformed_actual_orders",
            )
            blocked_actual = len(symbol_orders) + int(
                malformed_actual_counts.get(symbol, 0) or 0
            )
            plan_summaries.append(
                (
                    symbol,
                    blocked_actual,
                    0,
                    len(ideal_list),
                    0,
                    blocked_actual + len(ideal_list),
                )
            )
            continue
        if trace is not None:
            cancel_, create_, raw_create = bot._reconcile_symbol_orders(
                symbol,
                symbol_orders,
                ideal_list,
                keys,
                apply_mode_filters=apply_mode_filters,
                return_unfiltered_create=True,
            )
            trace = _trace_record(
                trace,
                "record_satisfied_orders",
                _orders_removed_by_identity(ideal_list, raw_create),
                "exact_reconciliation_match",
            )
            trace = _trace_record(
                trace,
                "record_blocked_orders",
                _orders_removed_by_identity(raw_create, create_),
                "mode_filter",
            )
        else:
            cancel_, create_ = bot._reconcile_symbol_orders(
                symbol,
                symbol_orders,
                ideal_list,
                keys,
                apply_mode_filters=apply_mode_filters,
            )
        pre_cancel = len(cancel_)
        pre_create = len(create_)
        cancel_, create_ = bot._annotate_order_deltas(cancel_, create_)
        before_tolerance_create = list(create_)
        cancel_, create_, skipped = bot._apply_order_match_tolerance(cancel_, create_)
        trace = _trace_record(
            trace,
            "record_satisfied_orders",
            _orders_removed_by_identity(before_tolerance_create, create_),
            "order_match_tolerance",
        )
        plan_summaries.append(
            (symbol, pre_cancel, len(cancel_), pre_create, len(create_), skipped)
        )
        to_cancel += cancel_
        to_create += create_

    if malformed_actual_symbols and connector_enabled:
        blocked = list(to_create)
        blocked_cancellations = list(to_cancel)
        to_cancel = []
        to_create = []
        trace = _trace_record(
            trace,
            "record_blocked_orders",
            blocked,
            "malformed_actual_orders",
        )
        logging.error(
            "[order] blocking all exchange actions because the account-critical open-orders "
            "snapshot is malformed | symbols=%s | blocked_cancellations=%d | blocked_creations=%d",
            _pb_attr("Passivbot")._log_symbols(sorted(malformed_actual_symbols), limit=8),
            len(blocked_cancellations),
            len(blocked),
        )
    to_cancel = await bot._sort_orders_by_market_diff(to_cancel, "to_cancel")
    to_create = await bot._sort_orders_by_market_diff(to_create, "to_create")
    if plan_summaries:
        total_pre_cancel = sum(p[1] for p in plan_summaries)
        total_cancel = sum(p[2] for p in plan_summaries)
        total_pre_create = sum(p[3] for p in plan_summaries)
        total_create = len(to_create)
        total_skipped = sum(p[5] for p in plan_summaries)
        detail_parts = []
        untouched_cancel = total_pre_cancel - total_cancel
        untouched_create = total_pre_create - total_create
        passivbot_cls = _pb_attr("Passivbot")
        for symbol, pre_c, c, pre_cr, cr, skipped in plan_summaries:
            prev = bot._last_plan_detail.get(symbol)
            current = (c, cr, skipped)
            bot._last_plan_detail[symbol] = current
            if c or cr or skipped:
                if prev != current:
                    detail_parts.append(
                        f"{passivbot_cls._log_symbol(symbol)}:c{pre_c}->{c} "
                        f"cr{pre_cr}->{cr} skip{skipped}"
                    )
        detail = " | ".join(detail_parts[:6])
        summary_key = (
            total_pre_cancel,
            total_cancel,
            total_pre_create,
            total_create,
            total_skipped,
            untouched_cancel,
            untouched_create,
            detail,
        )
        if summary_key != getattr(bot, "_last_order_plan_summary", None):
            bot._last_order_plan_summary = summary_key
            if total_cancel or total_create or total_skipped:
                extra = []
                if untouched_cancel:
                    extra.append(f"unchanged_cancel={untouched_cancel}")
                if untouched_create:
                    extra.append(f"unchanged_create={untouched_create}")
                log_level = (
                    logging.INFO
                    if bot._order_plan_summary_is_interesting(
                        total_pre_cancel=total_pre_cancel,
                        total_cancel=total_cancel,
                        total_pre_create=total_pre_create,
                        total_create=total_create,
                        total_skipped=total_skipped,
                    )
                    else logging.DEBUG
                )
                logging.log(
                    log_level,
                    "[order] order plan summary | cancel %d->%d | create %d->%d | skipped=%d%s%s",
                    total_pre_cancel,
                    total_cancel,
                    total_pre_create,
                    total_create,
                    total_skipped,
                    f" | {' '.join(extra)}" if extra else "",
                    f" | details: {detail}" if detail else "",
                )
    if collect_fresh_entry_eligibility:
        bot._fresh_entry_eligibility_trace = trace
    return to_cancel, to_create


def snapshot_actual_orders(
    bot,
    symbols: Optional[Iterable[str]] = None,
    *,
    psides_by_symbol: Optional[dict[str, Iterable[str]]] = None,
) -> dict[str, list[dict]]:
    """Return a normalized snapshot of currently open orders keyed by symbol."""
    connector_enabled = connector_supports_order_churn_gate(bot)
    actual_orders: dict[str, list[dict]] = {}
    malformed_symbols: set[str] = set()
    malformed_counts: dict[str, int] = {}
    open_orders_by_symbol = getattr(bot, "open_orders", {}) or {}
    if not isinstance(open_orders_by_symbol, dict):
        open_orders_by_symbol = {}
        malformed_symbols.add("<unknown>")
        malformed_counts["<unknown>"] = 1
        logging.error(
            "[order] malformed open-orders snapshot container; "
            "marking account-critical open-orders surface unavailable"
        )
    for raw_symbol, bucket_orders in open_orders_by_symbol.items():
        bucket_symbol = str(raw_symbol or "").strip()
        diagnostic_symbol = bucket_symbol or "<unknown>"
        valid_bucket_symbol = isinstance(raw_symbol, str) and bool(bucket_symbol)
        if not valid_bucket_symbol or not isinstance(bucket_orders, list):
            malformed_symbols.add(diagnostic_symbol)
            malformed_counts[diagnostic_symbol] = max(
                1, len(bucket_orders) if isinstance(bucket_orders, list) else 1
            )
            logging.error(
                "[order] malformed open-orders snapshot bucket; "
                "marking account-critical open-orders surface unavailable | symbol=%s | reason=%s",
                _pb_attr("Passivbot")._log_symbol(diagnostic_symbol),
                (
                    "missing or non-string symbol"
                    if not valid_bucket_symbol
                    else "orders bucket is not a list"
                ),
            )
    if symbols is None:
        requested_symbols = set(
            set(getattr(bot, "active_symbols", []) or [])
            | {
                symbol
                for symbol in open_orders_by_symbol
                if isinstance(symbol, str) and symbol.strip()
            }
            | set((getattr(bot, "positions", {}) or {}).keys())
        )
    else:
        requested_symbols = {str(symbol) for symbol in symbols if symbol}
    # A scoped protective reconciliation may return only selected symbols or
    # psides, but the open-orders surface remains account-critical. Validate
    # every currently known resting order before authorizing any account write.
    validation_symbols = requested_symbols | set(
        symbol
        for symbol in open_orders_by_symbol
        if isinstance(symbol, str) and symbol.strip()
    )
    pside_filter = None
    if psides_by_symbol is not None:
        pside_filter = {
            str(symbol): {str(pside) for pside in psides if pside}
            for symbol, psides in psides_by_symbol.items()
        }
    for symbol in sorted(str(symbol) for symbol in validation_symbols if symbol):
        return_symbol = symbol in requested_symbols
        symbol_orders = []
        allowed_psides = (
            pside_filter.get(symbol, set()) if pside_filter is not None else None
        )
        if allowed_psides == set() and not connector_enabled:
            actual_orders[symbol] = symbol_orders
            continue
        bucket_orders = open_orders_by_symbol.get(symbol, [])
        if not isinstance(bucket_orders, list):
            bucket_orders = []
        for order in bucket_orders:
            try:
                if not isinstance(order, dict):
                    raise TypeError(f"expected dict, got {type(order).__name__}")
                exchange_order_id = extract_order_exchange_id(order)
                if connector_enabled and not exchange_order_id:
                    raise ValueError("missing authoritative exchange order id")
                missing = [
                    key
                    for key in ("symbol", "side", "position_side", "price")
                    if key not in order
                ]
                if missing:
                    raise ValueError(f"missing required fields {','.join(missing)}")
                if connector_enabled:
                    remaining_qty = extract_order_remaining_qty(order)
                else:
                    remaining_qty = (
                        abs(float(order["qty"])) if "qty" in order else None
                    )
                if remaining_qty is None or remaining_qty <= 0.0:
                    raise ValueError("missing or contradictory authoritative remaining quantity")
                qty = float(remaining_qty)
                price = float(order["price"])
                if (
                    not math.isfinite(qty)
                    or not math.isfinite(price)
                    or qty <= 0.0
                    or price <= 0.0
                ):
                    raise ValueError("non-positive or non-finite qty or price")
                raw_symbol = order["symbol"]
                raw_side = order["side"]
                raw_position_side = order["position_side"]
                if raw_symbol is None or raw_side is None or raw_position_side is None:
                    raise ValueError("null symbol, side, or position_side")
                order_symbol = str(raw_symbol).strip()
                side = str(raw_side).strip().lower()
                position_side = str(raw_position_side).strip().lower()
                if not order_symbol or side not in {"buy", "sell"}:
                    raise ValueError("empty symbol or invalid side")
                if connector_enabled and order_symbol != symbol:
                    raise ValueError(
                        "open-order symbol contradicts its authoritative snapshot bucket"
                    )
                if position_side not in {"long", "short"}:
                    raise ValueError("invalid position_side")
                if connector_enabled:
                    close_only_resolver = getattr(
                        bot, "_canonical_open_order_reduce_only", None
                    )
                    reduce_only = (
                        close_only_resolver(order)
                        if callable(close_only_resolver)
                        else extract_order_reduce_only(order)
                    )
                    if not isinstance(reduce_only, bool):
                        raise ValueError("missing authoritative close-only semantics")
                else:
                    reduce_only = (
                        position_side == "long" and side == "sell"
                    ) or (position_side == "short" and side == "buy")
                raw_execution_type = str(
                    order.get("type") or order.get("execution_type") or ""
                ).lower()
                execution_type = (
                    "limit" if raw_execution_type == "limit" else "unknown"
                )
                custom_id = extract_order_custom_id(order)
                pb_order_type = (
                    str(_pb_attr("custom_id_to_snake")(custom_id)).lower()
                    if custom_id
                    and _pb_attr("custom_id_has_explicit_passivbot_marker")(
                        custom_id
                    )
                    else "unknown"
                )
                if connector_enabled and pb_order_type != "unknown":
                    if pb_order_type.startswith("close_") and not reduce_only:
                        raise ValueError(
                            "close pb_order_type contradicts authoritative close-only semantics"
                        )
                    if pb_order_type.startswith("entry_") and reduce_only:
                        raise ValueError(
                            "entry pb_order_type contradicts authoritative close-only semantics"
                        )
                if not return_symbol or (
                    allowed_psides is not None and position_side not in allowed_psides
                ):
                    continue
                symbol_orders.append(
                    {
                        "symbol": order_symbol,
                        "side": side,
                        "position_side": position_side,
                        "qty": qty,
                        "price": price,
                        "reduce_only": reduce_only,
                        "type": execution_type,
                        "pb_order_type": pb_order_type,
                        "id": exchange_order_id or order.get("id"),
                        "custom_id": custom_id,
                    }
                )
            except (TypeError, KeyError, ValueError, OverflowError) as exc:
                malformed_symbols.add(symbol)
                malformed_counts[symbol] = malformed_counts.get(symbol, 0) + 1
                logging.error(
                    "[order] malformed open order snapshot; "
                    "marking account-critical open-orders surface unavailable | symbol=%s | "
                    "error_type=%s",
                    _pb_attr("Passivbot")._log_symbol(symbol),
                    bounded_exception_type(exc),
                )
        if return_symbol:
            actual_orders[symbol] = symbol_orders
    bot._malformed_actual_order_symbols = malformed_symbols
    bot._malformed_actual_order_counts = malformed_counts
    if malformed_symbols:
        if hasattr(bot, "_mark_account_critical_state_dirty"):
            bot._mark_account_critical_state_dirty(
                reason="malformed_open_order_snapshot",
                symbols=malformed_symbols,
                source="snapshot_actual_orders",
                level=logging.ERROR,
            )
        else:
            mark_account_critical_state_dirty(
                bot,
                reason="malformed_open_order_snapshot",
                symbols=malformed_symbols,
                source="snapshot_actual_orders",
                level=logging.ERROR,
            )
    return actual_orders


def reconcile_symbol_orders(
    bot,
    symbol: str,
    actual_orders: list[dict],
    ideal_orders: list,
    keys: tuple[str, ...],
    *,
    apply_mode_filters: bool = True,
    return_unfiltered_create: bool = False,
) -> tuple[list[dict], list[dict]] | tuple[list[dict], list[dict], list[dict]]:
    """Return cancel/create lists for a single symbol after mode filtering."""
    to_cancel, to_create = filter_orders(actual_orders, ideal_orders, keys)
    raw_create = list(to_create)
    if apply_mode_filters:
        to_cancel, to_create = bot._apply_mode_filters(symbol, to_cancel, to_create)
    if return_unfiltered_create:
        return to_cancel, to_create, raw_create
    return to_cancel, to_create


def _order_is_reduce_only(order: dict) -> bool:
    if not isinstance(order, dict):
        return False
    reduced = extract_order_reduce_only(order)
    if reduced is not None:
        return bool(reduced)
    side = str(order.get("side") or "").lower()
    pside = str(order.get("position_side") or order.get("positionSide") or "").lower()
    return (pside == "long" and side == "sell") or (pside == "short" and side == "buy")


def _order_is_panic(order: dict) -> bool:
    if not isinstance(order, dict):
        return False
    pb_type = str(order.get("pb_order_type") or "")
    if pb_type:
        return "panic" in pb_type
    custom_id = str(order.get("custom_id") or "")
    if not custom_id:
        return False
    try:
        return "panic" in str(_pb_attr("custom_id_to_snake")(custom_id))
    except Exception:
        return False


def _order_is_market_panic(order: dict) -> bool:
    execution_type = str(
        order.get("type") or order.get("execution_type") or ""
    ).lower()
    return (
        execution_type == "market"
        and _order_is_panic(order)
        and _order_is_reduce_only(order)
    )


def _symbol_position_state(bot, symbol: str) -> str:
    positions = getattr(bot, "positions", None)
    if not isinstance(positions, dict) or symbol not in positions:
        return "unproven"
    sides = positions.get(symbol)
    if not isinstance(sides, dict):
        return "unproven"
    has_nonzero = False
    for pside in ("long", "short"):
        if pside not in sides:
            return "unproven"
        position = sides.get(pside)
        if not isinstance(position, dict) or "size" not in position:
            return "unproven"
        raw_size = position["size"]
        if isinstance(raw_size, bool) or raw_size is None:
            return "unproven"
        try:
            size = float(raw_size)
        except (TypeError, ValueError):
            return "unproven"
        if not math.isfinite(size):
            return "unproven"
        if size != 0.0:
            has_nonzero = True
    return "nonzero" if has_nonzero else "flat"


def _order_pb_type(order: dict) -> str:
    if not isinstance(order, dict):
        return ""
    pb_type = str(order.get("pb_order_type") or "")
    if pb_type:
        return pb_type
    custom_id = str(order.get("custom_id") or "")
    if not custom_id:
        return ""
    try:
        return str(_pb_attr("custom_id_to_snake")(custom_id))
    except Exception:
        return ""


def _reduce_only_order_family(order: dict) -> tuple[str, str] | None:
    if not _order_is_reduce_only(order):
        return None
    pb_type = _order_pb_type(order)
    if not pb_type:
        return None
    parts = pb_type.split("_")
    pside = str(order.get("position_side") or order.get("positionSide") or "").lower()
    if parts and parts[-1] in {"long", "short"}:
        pside = pside or parts[-1]
        parts = parts[:-1]
    family = "_".join(parts) if parts else pb_type
    return pside, family


def _order_is_protective_reducer(order: dict) -> bool:
    family = _reduce_only_order_family(order)
    return family is not None and family[1] in _PROTECTIVE_REDUCE_ONLY_FAMILIES


def annotate_order_deltas(
    bot, to_cancel: list[dict], to_create: list[dict]
) -> tuple[list[dict], list[dict]]:
    """
    Attach best-effort delta info between existing and desired orders to aid logging.

    Matches orders by symbol/side/position_side and closest price distance.
    """
    remaining_create = list(to_create)
    for order in to_create:
        order.setdefault("_context", "new")
        order.setdefault("_reason", "new")
    for cancel_order in to_cancel:
        cancel_order.setdefault("_context", "retire")
        cancel_order.setdefault("_reason", "retire")

    def pct(a: float, b: float) -> float:
        if a == 0 and b == 0:
            return 0.0
        if a == 0:
            return float("inf")
        return abs(b - a) / abs(a) * 100.0

    for cancel_order in to_cancel:
        candidates = [
            (idx, co)
            for idx, co in enumerate(remaining_create)
            if co.get("symbol") == cancel_order.get("symbol")
            and co.get("side") == cancel_order.get("side")
            and co.get("position_side") == cancel_order.get("position_side")
        ]
        if not candidates:
            continue
        best_idx, best_order = min(
            candidates,
            key=lambda c: abs(
                float(c[1].get("price", 0.0)) - float(cancel_order.get("price", 0.0))
            ),
        )
        raw_price_diff = pct(
            float(cancel_order.get("price", 0.0)),
            float(best_order.get("price", 0.0)),
        )
        raw_qty_diff = pct(
            float(cancel_order.get("qty", 0.0)), float(best_order.get("qty", 0.0))
        )
        price_diff = (
            round(raw_price_diff, 4)
            if math.isfinite(raw_price_diff)
            else raw_price_diff
        )
        qty_diff = (
            round(raw_qty_diff, 4) if math.isfinite(raw_qty_diff) else raw_qty_diff
        )
        reason_parts = []
        if price_diff > 0:
            reason_parts.append("price")
        if qty_diff > 0:
            reason_parts.append("qty")
        reason = "+".join(reason_parts) if reason_parts else "adjustment"
        cancel_order["_delta"] = {
            "price_old": cancel_order.get("price"),
            "price_new": best_order.get("price"),
            "price_pct_diff": price_diff,
            "qty_old": cancel_order.get("qty"),
            "qty_new": best_order.get("qty"),
            "qty_pct_diff": qty_diff,
        }
        cancel_order["_context"] = "replace"
        cancel_order["_reason"] = reason
        best_order["_delta"] = {
            "price_old": cancel_order.get("price"),
            "price_new": best_order.get("price"),
            "price_pct_diff": price_diff,
            "qty_old": cancel_order.get("qty"),
            "qty_new": best_order.get("qty"),
            "qty_pct_diff": qty_diff,
        }
        best_order["_context"] = "replace"
        best_order["_reason"] = reason
        remaining_create.pop(best_idx)

    for order in remaining_create:
        order.setdefault("_context", "new")
        order.setdefault("_reason", "fresh")
    return to_cancel, to_create


def apply_order_match_tolerance(
    bot, to_cancel: list[dict], to_create: list[dict]
) -> tuple[list[dict], list[dict], int]:
    """Drop cancel/create pairs that are within tolerance to avoid churn.

    Returns (remaining_cancel, remaining_create, skipped_pairs)
    """
    tolerance = float(bot.live_value("order_match_tolerance_pct"))
    if tolerance <= 0.0:
        return to_cancel, to_create, 0

    def pct_diff(a: float, b: float) -> float:
        if b == 0:
            return 0.0 if a == 0 else float("inf")
        return abs(a - b) / abs(b) * 100.0

    if not connector_supports_order_churn_gate(bot):
        used_cancel: set[int] = set()
        kept_create: list[dict] = []
        skipped = 0
        for order in to_create:
            match_idx = None
            for idx, existing in enumerate(to_cancel):
                if idx in used_cancel:
                    continue
                try:
                    if _pb_attr("orders_matching")(
                        order,
                        existing,
                        tolerance_qty=tolerance,
                        tolerance_price=tolerance,
                    ):
                        match_idx = idx
                        break
                except Exception:
                    continue
            if match_idx is None:
                kept_create.append(order)
                continue
            used_cancel.add(match_idx)
            skipped += 1
            try:
                logging.debug(
                    "skipped_recreate | %s | tolerance=%.4f%% price_diff=%.4f%% qty_diff=%.4f%%",
                    order.get("symbol", "?"),
                    tolerance * 100.0,
                    pct_diff(float(order["price"]), float(to_cancel[match_idx]["price"])),
                    pct_diff(float(order["qty"]), float(to_cancel[match_idx]["qty"])),
                )
            except Exception:
                logging.debug(
                    "skipped_recreate | %s | tolerance=%.4f%%",
                    order.get("symbol", "?"),
                    tolerance * 100.0,
                )
        return (
            [order for idx, order in enumerate(to_cancel) if idx not in used_cancel],
            kept_create,
            skipped,
        )

    try:
        current = normalize_ideal_orders(to_create)
    except (KeyError, TypeError, ValueError):
        # Current ideals are expected to be complete. Preserve the pre-existing
        # fail-open behavior here; validation at the producer boundary owns the
        # malformed-ideal failure contract.
        return to_cancel, to_create, 0

    matchable_cancel: list[dict] = []
    matchable_cancel_indices: list[int] = []
    for idx, order in enumerate(to_cancel):
        try:
            normalize_ideal_orders([order])
        except (KeyError, TypeError, ValueError):
            continue
        matchable_cancel.append(order)
        matchable_cancel_indices.append(idx)
    previous = normalize_ideal_orders(matchable_cancel)
    matches = deterministic_one_to_one_matches(current, previous, tolerance)
    used_cancel = {
        matchable_cancel_indices[previous[previous_idx].source_index]
        for previous_idx in matches.values()
    }
    matched_create = {current[current_idx].source_index for current_idx in matches}
    kept_create = [
        order for idx, order in enumerate(to_create) if idx not in matched_create
    ]
    skipped = len(matches)

    for current_idx, previous_idx in sorted(matches.items()):
        create_idx = current[current_idx].source_index
        order = to_create[create_idx]
        match_idx = matchable_cancel_indices[previous[previous_idx].source_index]
        try:
            price_diff = pct_diff(
                float(order["price"]), float(to_cancel[match_idx]["price"])
            )
            qty_diff = pct_diff(
                float(order["qty"]), float(to_cancel[match_idx]["qty"])
            )
            logging.debug(
                "skipped_recreate | %s | tolerance=%.4f%% price_diff=%.4f%% qty_diff=%.4f%%",
                order.get("symbol", "?"),
                tolerance * 100.0,
                price_diff,
                qty_diff,
            )
        except Exception:
            logging.debug(
                "skipped_recreate | %s | tolerance=%.4f%%",
                order.get("symbol", "?"),
                tolerance * 100.0,
            )

    remaining_cancel = [o for i, o in enumerate(to_cancel) if i not in used_cancel]
    return remaining_cancel, kept_create, skipped


def apply_mode_filters(
    bot,
    symbol: str,
    to_cancel: list[dict],
    to_create: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Apply mode-specific cancel/create filtering rules."""
    authorized_ema_entry_cancellation_order_keys = set(
        getattr(bot, "_orchestrator_ema_entry_cancellation_order_keys", set())
        or set()
    )
    for pside in ["long", "short"]:
        mode = bot.PB_modes[pside].get(symbol)
        if mode == "manual":
            if authorized_ema_entry_cancellation_order_keys:
                # This pair was dynamically managed by forager when its required
                # EMA became unavailable. Preserve cancellation only for the
                # exact proven resting entry; do not broaden manual mode to
                # closes, creations, replacements, or operator-owned orders.
                to_cancel = [
                    x
                    for x in to_cancel
                    if x["position_side"] != pside
                    or (
                        not x["reduce_only"]
                        and bool(
                            ema_entry_cancellation_order_keys(x)
                            & authorized_ema_entry_cancellation_order_keys
                        )
                    )
                ]
            else:
                to_cancel = [x for x in to_cancel if x["position_side"] != pside]
            to_create = [x for x in to_create if x["position_side"] != pside]
        elif mode == "tp_only":
            to_cancel = [
                x
                for x in to_cancel
                if (
                    x["position_side"] != pside
                    or (x["position_side"] == pside and x["reduce_only"])
                )
            ]
            to_create = [
                x
                for x in to_create
                if (
                    x["position_side"] != pside
                    or (x["position_side"] == pside and x["reduce_only"])
                )
            ]
        elif mode == "tp_only_with_active_entry_cancellation":
            to_create = [
                x
                for x in to_create
                if (
                    x["position_side"] != pside
                    or (x["position_side"] == pside and x["reduce_only"])
                )
            ]
    return to_cancel, to_create


def to_executable_orders(
    bot, ideal_orders: dict, last_prices: dict[str, float]
) -> tuple[dict[str, list], set[str]]:
    """Convert raw order tuples into api-ready dicts and find WEL-restricted symbols."""
    bot._fresh_entry_conversion_blocked_counts = {}
    ideal_orders_f: dict[str, list] = {}
    wel_blocked_symbols: set[str] = set()
    conversion_blocked_counts: Counter = Counter()
    order_market_diff = _pb_attr("order_market_diff")
    snake_of = _pb_attr("snake_of")

    for symbol, orders in ideal_orders.items():
        ideal_orders_f[symbol] = []
        last_mprice = last_prices[symbol]
        seen: set[tuple[object, float, float, str]] = set()
        with_mprice_diff = []
        for order in orders:
            side = determine_side_from_order_tuple(order)
            diff = order_market_diff(side, order[1], last_mprice)
            with_mprice_diff.append((diff, order, side))
            if (
                isinstance(order, tuple)
                and isinstance(order[2], str)
                and "close_auto_reduce_wel" in order[2]
            ):
                wel_blocked_symbols.add(symbol)
        for mprice_diff, order, order_side in sorted(
            with_mprice_diff, key=lambda item: item[0]
        ):
            position_side = "long" if "long" in order[2] else "short"
            if order[0] == 0.0:
                if str(order[2]).startswith("entry_initial_"):
                    conversion_blocked_counts[(symbol, position_side)] += 1
                continue
            seen_key = rust_order_conversion_identity(
                symbol, order[0], order[1], order[2]
            )
            if seen_key in seen:
                raise FatalBotException(
                    f"Rust ideal orders for {symbol} collide under conversion identity"
                )
            pb_order_type = snake_of(order[3])
            # The Rust orchestrator is the single source of execution-type
            # truth (every live path builds 6-tuples from its execution_type
            # and execution_priority fields); a short tuple here means a broken producer, and
            # silently defaulting could downgrade a panic market close.
            if len(order) < 6:
                raise ValueError(
                    f"ideal order for {symbol} missing execution_type or execution_priority: {order!r}; "
                    "the Rust orchestrator must supply both"
                )
            execution_type = str(order[4]).lower()
            if execution_type not in {"limit", "market"}:
                raise ValueError(
                    f"ideal order for {symbol} has invalid execution_type "
                    f"{order[4]!r}; expected limit or market"
                )
            execution_priority = str(order[5]).lower()
            if execution_priority not in {"ordinary", "risk_critical"}:
                raise ValueError(
                    f"ideal order for {symbol} has invalid execution_priority "
                    f"{order[5]!r}; expected ordinary or risk_critical"
                )
            _validate_intrinsic_rust_execution_priority(
                str(order[2]),
                execution_priority,
                f"Rust ideal order for {symbol}",
            )
            ideal_orders_f[symbol].append(
                {
                    "symbol": symbol,
                    "side": order_side,
                    "position_side": position_side,
                    "qty": abs(order[0]),
                    "price": order[1],
                    "reduce_only": "close" in order[2],
                    "custom_id": bot.format_custom_id_single(order[3]),
                    "type": execution_type,
                    "pb_order_type": pb_order_type,
                    "execution_priority": execution_priority,
                }
            )
            seen.add(seen_key)
    bot._fresh_entry_conversion_blocked_counts = dict(conversion_blocked_counts)
    return (
        bot._finalize_reduce_only_orders(ideal_orders_f, last_prices),
        wel_blocked_symbols,
    )


def finalize_reduce_only_orders(
    bot, orders_by_symbol: dict[str, list], last_prices: dict[str, float]
) -> dict[str, list]:
    """Bound reduce-only quantities so they never exceed the current position size."""
    order_market_diff = _pb_attr("order_market_diff")
    for symbol, orders in orders_by_symbol.items():
        market_price = float(last_prices.get(symbol, 0.0))

        for order in orders:
            if not order.get("reduce_only"):
                continue
            pos = bot.positions.get(order["symbol"], {}).get(order["position_side"], {})
            pos_size_abs = abs(float(pos.get("size", 0.0)))
            if abs(order["qty"]) > pos_size_abs:
                logging.warning(
                    "trimmed reduce-only qty to position size | order=%s | position=%s",
                    order,
                    pos,
                )
                order["qty"] = pos_size_abs

        for pside in ("long", "short"):
            pos_size_abs = abs(
                float(bot.positions.get(symbol, {}).get(pside, {}).get("size", 0.0))
            )
            if pos_size_abs <= 0.0:
                continue
            ro = [
                o
                for o in orders
                if o.get("reduce_only") and o.get("position_side") == pside
            ]
            if not ro:
                continue
            total = sum(float(o.get("qty", 0.0)) for o in ro)
            aggregate_tolerance = _rust_representation_tolerance(total, pos_size_abs)
            if total <= pos_size_abs + aggregate_tolerance:
                continue
            excess = total - pos_size_abs
            # Rust already caps a planned wave, but the position may shrink before live
            # reconciliation. Preserve the selected protective reducer and trim ordinary closes
            # first when this final exchange-state cap has to intervene.
            ro_sorted = sorted(
                ro,
                key=lambda o: (
                    0 if _order_is_protective_reducer(o) else 1,
                    order_market_diff(
                        o.get("side", ""), float(o.get("price", 0.0)), market_price
                    ),
                ),
                reverse=True,
            )
            for order in ro_sorted:
                if excess <= 0.0:
                    break
                qty = float(order.get("qty", 0.0))
                if qty <= 0.0:
                    continue
                reduce_by = min(qty, excess)
                new_qty = qty - reduce_by
                order["qty"] = float(round(new_qty, 12))
                excess -= reduce_by
            orders_by_symbol[symbol] = [
                order
                for order in orders_by_symbol[symbol]
                if not (
                    order.get("reduce_only") and float(order.get("qty", 0.0)) <= 0.0
                )
            ]

    return orders_by_symbol
