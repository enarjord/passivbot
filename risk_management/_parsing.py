"""Helpers for parsing exchange payloads into dashboard-friendly structures."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional

from ._utils import first_float

__all__ = ["extract_balance", "parse_position", "parse_order"]


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_balance(balance: Mapping[str, Any], settle_currency: str) -> float:
    """Extract a numeric balance from ccxt balance payloads."""

    if not isinstance(balance, Mapping):
        return 0.0

    aggregate_keys = (
        "totalMarginBalance",
        "totalEquity",
        "totalWalletBalance",
        "marginBalance",
        "totalBalance",
    )

    def _find_nested_aggregate(value: Any) -> Optional[float]:
        if isinstance(value, Mapping):
            for key in aggregate_keys:
                candidate = _to_float(value.get(key))
                if candidate is not None:
                    return candidate
            for child in value.values():
                result = _find_nested_aggregate(child)
                if result is not None:
                    return result
        elif isinstance(value, (list, tuple)):
            for child in value:
                result = _find_nested_aggregate(child)
                if result is not None:
                    return result
        return None

    for key in (*aggregate_keys, "equity"):
        candidate = _to_float(balance.get(key))
        if candidate is not None:
            return candidate

    info = balance.get("info")
    if isinstance(info, Mapping):
        for key in (*aggregate_keys, "equity"):
            candidate = _to_float(info.get(key))
            if candidate is not None:
                return candidate
        nested = _find_nested_aggregate(info)
        if nested is not None:
            return nested

    total = balance.get("total")
    if isinstance(total, Mapping) and total:
        if settle_currency in total:
            candidate = _to_float(total.get(settle_currency))
            if candidate is not None:
                return candidate
        summed = 0.0
        found_value = False
        for value in total.values():
            candidate = _to_float(value)
            if candidate is None:
                continue
            summed += candidate
            found_value = True
        if found_value:
            return summed

    for currency_key in (settle_currency, "USDT"):
        entry = balance.get(currency_key)
        if isinstance(entry, Mapping):
            for key in ("total", "free", "used"):
                candidate = _to_float(entry.get(key))
                if candidate is not None:
                    return candidate
        else:
            candidate = _to_float(entry)
            if candidate is not None:
                return candidate

    return 0.0


def parse_position(position: Mapping[str, Any], balance: float) -> Optional[Dict[str, Any]]:
    size = first_float(
        position.get("contracts"),
        position.get("size"),
        position.get("amount"),
        position.get("info", {}).get("positionAmt") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("positionAmt")
        if isinstance(position.get("info"), Mapping) and position["info"].get("positionAmt") not in (None, "")
        else None,
    ) or 0.0
    entry_price = first_float(
        position.get("entryPrice"),
        position.get("entry_price"),
        position.get("avgPrice"),
        position.get("info", {}).get("entryPrice") if isinstance(position.get("info"), Mapping) else None,
    )
    mark_price = first_float(
        position.get("markPrice"),
        position.get("mark_price"),
        position.get("info", {}).get("markPrice") if isinstance(position.get("info"), Mapping) else None,
    )
    liquidation_price = first_float(
        position.get("liquidationPrice"),
        position.get("liq_price"),
        position.get("liquidation_price"),
        position.get("info", {}).get("liquidationPrice") if isinstance(position.get("info"), Mapping) else None,
    )
    side = str(position.get("side") or "").upper()
    if not side:
        side = str(position.get("positionSide") or position.get("position_side") or "").upper()
        if not side:
            side = "LONG" if size >= 0 else "SHORT"
    unrealized = first_float(
        position.get("unrealizedPnl"),
        position.get("unrealized_pnl"),
        position.get("info", {}).get("unrealisedPnl") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("unrealizedPnl") if isinstance(position.get("info"), Mapping) else None,
    ) or 0.0
    realized = first_float(
        position.get("realizedPnl"),
        position.get("realized_pnl"),
        position.get("info", {}).get("realisedPnl") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("realizedPnl") if isinstance(position.get("info"), Mapping) else None,
        position.get("daily_realized_pnl"),
    ) or 0.0
    mark_price = mark_price or entry_price or 0.0
    contract_size = first_float(
        position.get("contractSize"),
        position.get("info", {}).get("contractSize") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("ctVal") if isinstance(position.get("info"), Mapping) else None,
    ) or 1.0
    notional = first_float(
        position.get("notional"),
        position.get("notionalValue"),
        position.get("info", {}).get("notionalValue") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("notionalUsd") if isinstance(position.get("info"), Mapping) else None,
    )
    if notional is None:
        reference_price = mark_price or entry_price or 0.0
        notional = abs(size) * contract_size * reference_price
    notional_value = float(notional or 0.0)
    if size < 0 and notional_value > 0:
        signed_notional = -abs(notional_value)
    elif size > 0 and notional_value < 0:
        signed_notional = abs(notional_value)
    else:
        signed_notional = notional_value
    abs_notional = abs(signed_notional)
    take_profit = first_float(
        position.get("takeProfitPrice"),
        position.get("tpPrice"),
        position.get("info", {}).get("takeProfitPrice") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("tpTriggerPx") if isinstance(position.get("info"), Mapping) else None,
    )
    stop_loss = first_float(
        position.get("stopLossPrice"),
        position.get("slPrice"),
        position.get("info", {}).get("stopLossPrice") if isinstance(position.get("info"), Mapping) else None,
        position.get("info", {}).get("slTriggerPx") if isinstance(position.get("info"), Mapping) else None,
    )
    wallet_exposure = None
    if balance:
        wallet_exposure = abs_notional / balance if balance else None
    return {
        "symbol": str(position.get("symbol") or position.get("id") or "unknown"),
        "side": side,
        "notional": abs_notional,
        "entry_price": float(entry_price or 0.0),
        "mark_price": float(mark_price or 0.0),
        "liquidation_price": float(liquidation_price) if liquidation_price is not None else None,
        "wallet_exposure_pct": float(wallet_exposure) if wallet_exposure is not None else None,
        "unrealized_pnl": float(unrealized),
        "daily_realized_pnl": float(realized),
        "max_drawdown_pct": None,
        "take_profit_price": float(take_profit) if take_profit is not None else None,
        "stop_loss_price": float(stop_loss) if stop_loss is not None else None,
        "size": float(size),
        "signed_notional": signed_notional,
    }


def parse_order(order: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(order, Mapping):
        return None
    symbol = order.get("symbol") or order.get("id")
    if not symbol:
        return None
    price = first_float(
        order.get("price"),
        order.get("triggerPrice"),
        order.get("stopPrice"),
        order.get("info", {}).get("price") if isinstance(order.get("info"), Mapping) else None,
    )
    amount = first_float(
        order.get("amount"),
        order.get("contracts"),
        order.get("size"),
        order.get("info", {}).get("origQty") if isinstance(order.get("info"), Mapping) else None,
    )
    if amount is None:
        return None
    remaining = first_float(
        order.get("remaining"),
        order.get("remainingAmount"),
        order.get("info", {}).get("leavesQty") if isinstance(order.get("info"), Mapping) else None,
    )
    reduce_only_raw = order.get("reduceOnly")
    if isinstance(order.get("info"), Mapping):
        reduce_only_raw = reduce_only_raw or order["info"].get("reduceOnly")
    reduce_only = bool(reduce_only_raw)
    stop_price = first_float(
        order.get("stopPrice"),
        order.get("triggerPrice"),
        order.get("info", {}).get("stopPrice") if isinstance(order.get("info"), Mapping) else None,
    )
    timestamp_raw = order.get("timestamp")
    created_at = None
    if isinstance(timestamp_raw, (int, float)):
        created_at = datetime.fromtimestamp(float(timestamp_raw) / 1000, timezone.utc).isoformat()
    else:
        datetime_str = order.get("datetime")
        if isinstance(datetime_str, str) and datetime_str:
            created_at = datetime_str
    notional = price * amount if price is not None else None
    return {
        "order_id": str(order.get("id") or order.get("clientOrderId") or ""),
        "symbol": str(symbol),
        "side": str(order.get("side") or "").lower(),
        "type": str(order.get("type") or "").lower(),
        "price": price,
        "amount": amount,
        "remaining": remaining,
        "status": str(order.get("status") or ""),
        "reduce_only": reduce_only,
        "stop_price": stop_price,
        "notional": notional,
        "created_at": created_at,
    }

