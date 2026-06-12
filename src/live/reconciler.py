from __future__ import annotations

import json
import logging
import math
import sys
from typing import Iterable, Optional

from live.freshness import ACCOUNT_SURFACES
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
                return value.strip().lower() in {"true", "1", "yes", "y"}
            return bool(value)
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


def prune_emitted_order_custom_ids(bot, now_ts: int) -> None:
    """Drop emitted order records outside the foreign-writer lookback window."""
    now_ts = int(now_ts)
    acknowledged_cutoff_ts = now_ts - _pb_const("FOREIGN_PASSIVBOT_LOOKBACK_MS")
    ambiguous_cutoff_ts = now_ts - _pb_const(
        "FOREIGN_PASSIVBOT_AMBIGUOUS_CREATE_LOOKBACK_MS"
    )
    short_lived_statuses = {"submitted", "create_error_ambiguous"}
    kept = []
    for record in emitted_order_records(bot):
        cutoff_ts = (
            ambiguous_cutoff_ts
            if record.get("status") in short_lived_statuses
            else acknowledged_cutoff_ts
        )
        if int(record.get("timestamp", 0)) >= cutoff_ts:
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
            logging.error("[safety] failed to stop data maintainers: %s", exc)
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


def order_matches_known_self_emission(bot, order: dict, max_age_ms: int) -> bool:
    """Return True when an open order is known to have been emitted by this process."""
    if not isinstance(order, dict):
        return False
    if bot.order_matches_recent_execution(order, max_age_ms=max_age_ms):
        return True
    now_ts = (
        int(bot.get_exchange_time()) if hasattr(bot, "get_exchange_time") else _utc_ms()
    )
    cutoff = now_ts - int(max_age_ms)
    exchange_id = extract_order_exchange_id(order)
    custom_id = extract_order_custom_id(order)
    canonical_custom_id = canonical_passivbot_custom_id(custom_id)
    pb_type = (
        _pb_attr("custom_id_to_snake")(custom_id)
        if custom_id
        else bot._resolve_pb_order_type(order)
    )
    fingerprint = order_identity_fingerprint(order, pb_type)
    for record in emitted_order_records(bot):
        record_ts_raw = record.get("timestamp")
        if record_ts_raw is None:
            continue
        try:
            record_ts = int(record_ts_raw)
        except (TypeError, ValueError):
            continue
        if record_ts < cutoff:
            continue
        record_exchange_id = str(record.get("exchange_id") or "")
        if (
            exchange_id
            and record_exchange_id
            and str(exchange_id) == record_exchange_id
        ):
            return True
        record_custom_id = str(record.get("canonical_custom_id") or "")
        if (
            canonical_custom_id
            and record_custom_id
            and canonical_custom_id == record_custom_id
        ):
            return True
        if (
            fingerprint
            and record.get("fingerprint")
            and record.get("fingerprint") == fingerprint
        ):
            return True
    return False


def flag_disappeared_self_order_guardrail(bot, order: dict) -> None:
    """Block creates for a symbol until account surfaces refresh after a self order vanishes."""
    symbol = str(order.get("symbol") or "")
    if not symbol:
        return
    ledger = bot._ensure_freshness_ledger()
    min_epoch = int(getattr(bot, "_authoritative_refresh_epoch", 0) or 0) + 1
    details = {
        "order_id": str(order.get("id") or ""),
        "side": str(order.get("side") or ""),
        "position_side": str(order.get("position_side") or ""),
        "price": order.get("price"),
        "qty": order.get("qty", order.get("amount")),
    }
    ledger.flag_symbol_block(
        symbol,
        reason="self_order_disappeared_position_may_be_stale",
        required_surfaces=ACCOUNT_SURFACES,
        min_epoch=min_epoch,
        detected_ms=_utc_ms(),
        details=details,
    )
    bot.execution_scheduled = True
    if not hasattr(bot, "state_change_detected_by_symbol"):
        bot.state_change_detected_by_symbol = set()
    bot.state_change_detected_by_symbol.add(symbol)
    bot._request_authoritative_confirmation(ACCOUNT_SURFACES, min_epoch=min_epoch)
    logging.debug(
        "[state] freshness guardrail armed | symbol=%s | reason=self_order_disappeared_position_may_be_stale | required=%s | min_epoch=%s | order_id=%s",
        _pb_attr("Passivbot")._log_symbol(symbol),
        ",".join(sorted(ACCOUNT_SURFACES)),
        min_epoch,
        details["order_id"],
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
    min_epoch = int(getattr(bot, "_authoritative_refresh_epoch", 0) or 0) + 1
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
    if not hasattr(bot, "_last_plan_detail"):
        bot._last_plan_detail = {}
    ideal_orders = await bot.calc_ideal_orders()

    actual_orders = bot._snapshot_actual_orders()
    malformed_actual_symbols = set(
        getattr(bot, "_malformed_actual_order_symbols", set()) or set()
    )
    trailing_unavailable_symbols = set(
        getattr(bot, "_orchestrator_trailing_unavailable_symbols", set()) or set()
    )
    malformed_actual_counts = dict(
        getattr(bot, "_malformed_actual_order_counts", {}) or {}
    )
    keys = ("symbol", "side", "position_side", "qty", "price")
    to_cancel, to_create = [], []
    plan_summaries = []
    for symbol, symbol_orders in actual_orders.items():
        ideal_list = (
            ideal_orders.get(symbol, []) if isinstance(ideal_orders, dict) else []
        )
        if symbol in malformed_actual_symbols:
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
        cancel_, create_ = bot._reconcile_symbol_orders(
            symbol, symbol_orders, ideal_list, keys
        )
        pre_cancel = len(cancel_)
        pre_create = len(create_)
        trailing_skipped = 0
        if symbol in trailing_unavailable_symbols:
            cancel_, create_, trailing_skipped, fully_blocked = (
                filter_trailing_unavailable_reconciliation(
                    bot, symbol, cancel_, create_, ideal_list
                )
            )
            if fully_blocked:
                blocked_total = len(symbol_orders) + len(ideal_list)
                plan_summaries.append(
                    (
                        symbol,
                        len(symbol_orders),
                        0,
                        len(ideal_list),
                        0,
                        blocked_total,
                    )
                )
                continue
        cancel_, create_ = bot._annotate_order_deltas(cancel_, create_)
        cancel_, create_, skipped = bot._apply_order_match_tolerance(cancel_, create_)
        skipped += trailing_skipped
        plan_summaries.append(
            (symbol, pre_cancel, len(cancel_), pre_create, len(create_), skipped)
        )
        to_cancel += cancel_
        to_create += create_

    to_create, initial_entry_gate_skipped = (
        await bot._apply_initial_entry_distance_gate(to_create)
    )
    to_cancel = await bot._sort_orders_by_market_diff(to_cancel, "to_cancel")
    to_create = await bot._sort_orders_by_market_diff(to_create, "to_create")
    to_create, freshness_skipped = bot._apply_freshness_creation_guardrails(to_create)
    if plan_summaries:
        total_pre_cancel = sum(p[1] for p in plan_summaries)
        total_cancel = sum(p[2] for p in plan_summaries)
        total_pre_create = sum(p[3] for p in plan_summaries)
        total_create = len(to_create)
        total_skipped = (
            sum(p[5] for p in plan_summaries)
            + freshness_skipped
            + initial_entry_gate_skipped
        )
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
                        f"{passivbot_cls._log_symbol(symbol)}:c{pre_c}->{c} cr{pre_cr}->{cr} skip{skipped}"
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
    return to_cancel, to_create


def snapshot_actual_orders(bot) -> dict[str, list[dict]]:
    """Return a normalized snapshot of currently open orders keyed by symbol."""
    actual_orders: dict[str, list[dict]] = {}
    malformed_symbols: set[str] = set()
    malformed_counts: dict[str, int] = {}
    for symbol in bot.active_symbols:
        symbol_orders = []
        for order in bot.open_orders.get(symbol, []):
            try:
                if not isinstance(order, dict):
                    raise TypeError(f"expected dict, got {type(order).__name__}")
                missing = [
                    key
                    for key in ("symbol", "side", "position_side", "qty", "price")
                    if key not in order
                ]
                if missing:
                    raise ValueError(f"missing required fields {','.join(missing)}")
                qty = abs(float(order["qty"]))
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
                side = str(raw_side).strip()
                position_side = str(raw_position_side).strip()
                if not order_symbol or side not in {"buy", "sell"}:
                    raise ValueError("empty symbol or invalid side")
                if position_side not in {"long", "short"}:
                    raise ValueError("invalid position_side")
                symbol_orders.append(
                    {
                        "symbol": order_symbol,
                        "side": side,
                        "position_side": position_side,
                        "qty": qty,
                        "price": price,
                        "reduce_only": (
                            position_side == "long" and side == "sell"
                        )
                        or (
                            position_side == "short" and side == "buy"
                        ),
                        "id": order.get("id"),
                        "custom_id": order.get("custom_id"),
                    }
                )
            except (TypeError, KeyError, ValueError) as exc:
                malformed_symbols.add(symbol)
                malformed_counts[symbol] = malformed_counts.get(symbol, 0) + 1
                order_id = order.get("id") if isinstance(order, dict) else None
                logging.error(
                    "[order] malformed open order snapshot; "
                    "blocking order planning for symbol | symbol=%s | "
                    "order_id=%s | reason=%s",
                    _pb_attr("Passivbot")._log_symbol(symbol),
                    order_id or "unknown",
                    exc,
                )
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
) -> tuple[list[dict], list[dict]]:
    """Return cancel/create lists for a single symbol after mode filtering."""
    to_cancel, to_create = filter_orders(actual_orders, ideal_orders, keys)
    to_cancel, to_create = bot._apply_mode_filters(symbol, to_cancel, to_create)
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


def _trailing_unavailable_reasons(bot, symbol: str) -> set[str]:
    by_symbol = getattr(bot, "_orchestrator_trailing_unavailable_reasons", {}) or {}
    reasons = by_symbol.get(symbol, []) if isinstance(by_symbol, dict) else []
    if isinstance(reasons, str):
        return {reasons}
    return {str(reason) for reason in reasons if reason}


def filter_trailing_unavailable_reconciliation(
    bot,
    symbol: str,
    to_cancel: list[dict],
    to_create: list[dict],
    ideal_orders: list[dict],
) -> tuple[list[dict], list[dict], int, bool]:
    """Constrain reconciliation when trailing data is unavailable.

    Missing anchors/fetch failures can make regular trailing closes unsafe, so keep
    preserving the symbol. The short post-fill window with no newer candle is softer:
    block new entries, but allow entry cleanup and reduce-only/panic exits.
    """
    reasons = _trailing_unavailable_reasons(bot, symbol)
    has_panic_plan = any(_order_is_panic(order) for order in ideal_orders) or any(
        _order_is_panic(order) for order in to_create
    )
    soft_missing_candles_only = reasons == {"missing_trailing_candles"}
    if not soft_missing_candles_only and not has_panic_plan:
        return [], [], len(to_cancel) + len(to_create), True

    filtered_create = [
        order
        for order in to_create
        if _order_is_reduce_only(order) or _order_is_panic(order)
    ]
    dropped_create = len(to_create) - len(filtered_create)

    if has_panic_plan:
        return to_cancel, filtered_create, dropped_create, False

    ideal_has_reduce_only = any(_order_is_reduce_only(order) for order in ideal_orders)
    filtered_cancel = []
    dropped_cancel = 0
    for order in to_cancel:
        if _order_is_reduce_only(order) and not ideal_has_reduce_only:
            dropped_cancel += 1
            continue
        filtered_cancel.append(order)
    return filtered_cancel, filtered_create, dropped_cancel + dropped_create, False


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

    used_cancel: set[int] = set()
    kept_create: list[dict] = []
    skipped = 0

    def pct_diff(a: float, b: float) -> float:
        if b == 0:
            return 0.0 if a == 0 else float("inf")
        return abs(a - b) / abs(b) * 100.0

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
        else:
            used_cancel.add(match_idx)
            skipped += 1
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
    for pside in ["long", "short"]:
        mode = bot.PB_modes[pside].get(symbol)
        if mode == "manual":
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
    ideal_orders_f: dict[str, list] = {}
    wel_blocked_symbols: set[str] = set()
    order_market_diff = _pb_attr("order_market_diff")
    snake_of = _pb_attr("snake_of")

    for symbol, orders in ideal_orders.items():
        ideal_orders_f[symbol] = []
        last_mprice = last_prices[symbol]
        seen = set()
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
                continue
            seen_key = str(abs(order[0])) + str(order[1]) + order[2]
            if seen_key in seen:
                logging.debug("duplicate ideal order for %s skipped: %s", symbol, order)
                continue
            pb_order_type = snake_of(order[3])
            if len(order) >= 5:
                execution_type = str(order[4]).lower()
            else:
                execution_type = "limit"
                panic_close_pref = bot._equity_hard_stop_panic_close_order_type(
                    position_side
                )
                if "panic" in pb_order_type:
                    execution_type = (
                        "market" if panic_close_pref == "market" else "limit"
                    )
            if execution_type not in {"limit", "market"}:
                execution_type = "limit"
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
                }
            )
            seen.add(seen_key)
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
            if total <= pos_size_abs + 1e-12:
                continue
            excess = total - pos_size_abs
            ro_sorted = sorted(
                ro,
                key=lambda o: order_market_diff(
                    o.get("side", ""), float(o.get("price", 0.0)), market_price
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
