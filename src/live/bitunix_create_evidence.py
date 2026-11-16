"""Positive order evidence for unresolved Bitunix creates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Literal

from ccxt.base.errors import OrderNotFound

if TYPE_CHECKING:
    from live.bitunix_create_journal import BitunixCreateAttempt


_ORDER_DETAIL = "/api/v1/futures/trade/get_order_detail"
_ORDER_HISTORY = "/api/v1/futures/trade/get_history_orders"
_DAY_MS = 86_400_000
# Bitunix permits up to 60 seconds of clock difference on signed requests.
_CLOCK_SKEW_MS = 60_000
_WORKING_STATUSES = {"NEW", "PART_FILLED"}
_TERMINAL_STATUSES = {"FILLED", "CANCELED", "PART_FILLED_CANCELED", "EXPIRED"}


@dataclass(frozen=True)
class BitunixCreateEvidence:
    order_id: str
    disposition: Literal["working", "terminal"]
    filled_qty: Decimal
    venue_status: str


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        result = Decimal(str(value))
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


def parse_bitunix_create_evidence(
    row: Any,
    attempt: BitunixCreateAttempt,
    *,
    confirmed_working_order_id: str | None = None,
) -> BitunixCreateEvidence | None:
    """Validate a raw detail, pending, or historical order against the attempt."""
    if not isinstance(row, dict):
        return None
    order_id = row.get("orderId")
    client_id = row.get("clientId")
    if (
        not isinstance(order_id, str)
        or not order_id
        or order_id != order_id.strip()
        or (
            client_id != attempt.client_id
            and not (client_id in (None, "") and order_id == confirmed_working_order_id)
        )
        or row.get("symbol") != attempt.symbol
        or row.get("side") != attempt.side.upper()
        or row.get("reduceOnly") is not attempt.reduce_only
        or row.get("positionMode") != "HEDGE"
    ):
        return None

    # The order lookup exposes the actual action side for a hedge-mode close.
    # Additional position metadata, when present, must agree with the request.
    expected_type = attempt.order_type.upper()
    type_fields = [row[key] for key in ("orderType", "type") if key in row]
    if not type_fields or any(value != expected_type for value in type_fields):
        return None
    if "tradeSide" in row and row["tradeSide"] != (
        "CLOSE" if attempt.reduce_only else "OPEN"
    ):
        return None
    if (
        attempt.position_id is not None
        and "positionId" in row
        and str(row["positionId"]) != attempt.position_id
    ):
        return None
    if attempt.price is not None and (
        _decimal(row.get("price")) != Decimal(attempt.price)
        or row.get("effect") != attempt.effect
    ):
        return None

    qty = Decimal(attempt.qty)
    if _decimal(row.get("qty")) != qty:
        return None
    created_ms = _decimal(row.get("ctime"))
    if (
        created_ms is None
        or created_ms <= 0
        or created_ms != created_ms.to_integral_value()
        or created_ms < attempt.submitted_at_ms - _CLOCK_SKEW_MS
    ):
        return None
    # The ordinary adapter can default a missing trade quantity to zero.
    # Recovery needs the venue's explicit cumulative executed quantity.
    fill_quantities = [
        _decimal(row[key])
        for key in ("tradeQty", "dealAmount")
        if row.get(key) not in (None, "")
    ]
    filled_qty = fill_quantities[0] if fill_quantities else None
    if (
        filled_qty is None
        or not 0 <= filled_qty <= qty
        or any(value != filled_qty for value in fill_quantities)
    ):
        return None
    # Bitunix has also emitted NEW_ on live REST. Reject contradictory aliases.
    statuses = [
        row[key].rstrip("_") if isinstance(row[key], str) else None
        for key in ("status", "orderStatus")
        if key in row
    ]
    if not statuses or any(value != statuses[0] for value in statuses):
        return None
    status = statuses[0]
    if status is None:
        return None
    if status in {"PART_FILLED", "PART_FILLED_CANCELED"} and filled_qty == 0:
        return None
    if (status == "FILLED") != (filled_qty == qty):
        return None
    if status in _WORKING_STATUSES:
        return BitunixCreateEvidence(order_id, "working", filled_qty, status)
    if status in _TERMINAL_STATUSES:
        return BitunixCreateEvidence(order_id, "terminal", filled_qty, status)
    return None


async def lookup_bitunix_create_evidence(
    request: Callable[..., Awaitable[Any]],
    attempt: BitunixCreateAttempt,
    *,
    now_ms: int | None = None,
) -> BitunixCreateEvidence | None:
    """Look for positive exact-ID evidence. None leaves the attempt unresolved."""
    try:
        detail = await request(
            "GET", _ORDER_DETAIL, params={"clientId": attempt.client_id}, private=True
        )
    except OrderNotFound:
        # Bitunix 20007 means to try again later.
        detail = None
    if detail is not None and detail != []:
        if isinstance(detail, list) and len(detail) == 1:
            detail = detail[0]
        return parse_bitunix_create_evidence(detail, attempt)

    if now_ms is None:
        now_ms = time.time_ns() // 1_000_000
    # Exact ID and one page per mode. The canceled-only query has a shorter
    # venue retention window. Out-of-retention attempts still remain pending.
    for canceled, retention_days in ((False, 90), (True, 3)):
        oldest_ms = now_ms - retention_days * _DAY_MS + _CLOCK_SKEW_MS
        if attempt.submitted_at_ms + _CLOCK_SKEW_MS < oldest_ms:
            continue
        try:
            history = await request(
                "GET",
                _ORDER_HISTORY,
                params={
                    "clientId": attempt.client_id,
                    "symbol": attempt.symbol,
                    "startTime": max(
                        0, oldest_ms, attempt.submitted_at_ms - _CLOCK_SKEW_MS
                    ),
                    "queryCanceled": "true" if canceled else "false",
                    "limit": 2,
                },
                private=True,
            )
        except OrderNotFound:
            continue
        rows = history.get("orderList") if isinstance(history, dict) else None
        if not isinstance(rows, list) or len(rows) > 1:
            return None
        total = history.get("total")
        if total is not None and str(total) != str(len(rows)):
            return None
        if not rows:
            continue
        evidence = parse_bitunix_create_evidence(rows[0], attempt)
        return (
            evidence
            if evidence is not None and evidence.disposition == "terminal"
            else None
        )
    return None
