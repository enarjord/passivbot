import json
from typing import Callable, Dict, List

from utils import ts_to_date


def bitget_payload_context(payload: object) -> str:
    if isinstance(payload, dict):
        keys = sorted(str(key) for key in payload.keys())
        data = payload.get("data")
        data_keys = sorted(str(key) for key in data.keys()) if isinstance(data, dict) else []
        code = payload.get("code")
        msg = payload.get("msg")
        return f"code={code!r} msg={msg!r} keys={keys!r} data_keys={data_keys!r}"
    return f"type={type(payload).__name__}"


def require_uta_field(fill: dict, field: str) -> object:
    if field not in fill or fill[field] in (None, ""):
        raise ValueError(
            f"bitget UTA fill missing required field {field!r}; "
            f"context={bitget_payload_context(fill)}"
        )
    return fill[field]


def require_uta_float(fill: dict, field: str, *, positive: bool = False) -> float:
    raw = require_uta_field(fill, field)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"bitget UTA fill field {field!r} is not numeric; "
            f"value={raw!r} context={bitget_payload_context(fill)}"
        ) from exc
    if positive and value <= 0.0:
        raise ValueError(
            f"bitget UTA fill field {field!r} must be positive; "
            f"value={raw!r} context={bitget_payload_context(fill)}"
        )
    return value


def require_uta_timestamp(fill: dict) -> int:
    raw = require_uta_field(fill, "createdTime")
    try:
        timestamp = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "bitget UTA fill createdTime is not an integer millisecond timestamp; "
            f"value={raw!r} context={bitget_payload_context(fill)}"
        ) from exc
    if timestamp <= 0:
        raise ValueError(
            f"bitget UTA fill createdTime must be positive; "
            f"value={raw!r} context={bitget_payload_context(fill)}"
        )
    return timestamp


def _bitget_fee_paid_from_fee(raw_fee: object) -> float:
    fee = float(raw_fee)
    return fee if fee < 0.0 else -abs(fee)


def normalize_bitget_fee_detail(fee_detail: object) -> object:
    """Normalize Bitget feeDetail rows into canonical signed fee entries."""
    if fee_detail is None:
        return fee_detail
    if isinstance(fee_detail, dict):
        entries = [fee_detail]
    elif isinstance(fee_detail, str):
        try:
            decoded = json.loads(fee_detail)
        except Exception:
            return fee_detail
        if isinstance(decoded, dict):
            entries = [decoded]
        elif isinstance(decoded, list):
            entries = [entry for entry in decoded if isinstance(entry, dict)]
        else:
            return fee_detail
    else:
        try:
            entries = [entry for entry in list(fee_detail) if isinstance(entry, dict)]
        except Exception:
            return fee_detail
    if not entries:
        return fee_detail

    normalized: List[Dict[str, object]] = []
    for entry in entries:
        item = dict(entry)
        currency = (
            item.get("currency")
            or item.get("feeCoin")
            or item.get("coin")
            or item.get("asset")
            or item.get("feeCurrency")
        )
        if currency:
            item["currency"] = str(currency).upper()
        try:
            if item.get("fee_paid") not in (None, ""):
                item["fee_paid"] = float(item["fee_paid"])
            elif item.get("totalFee") not in (None, ""):
                item["fee_paid"] = float(item["totalFee"])
            elif item.get("fee") not in (None, ""):
                item["fee_paid"] = _bitget_fee_paid_from_fee(item["fee"])
            elif item.get("totalDeductionFee") not in (None, ""):
                item["fee_paid"] = float(item["totalDeductionFee"])
        except (TypeError, ValueError):
            pass
        normalized.append(item)
    return normalized


def deduce_uta_side_pside(fill: dict) -> tuple[str, str]:
    """Infer standard ``(side, pside)`` for Bitget UTA v3 fill payloads."""

    raw_side = str(fill.get("side", "")).lower()
    trade_side = str(fill.get("tradeSide", "")).lower()
    pos_side = str(fill.get("posSide", "")).lower()

    if pos_side not in ("long", "short"):
        if trade_side == "close":
            if raw_side == "buy":
                pos_side = "short"
            elif raw_side == "sell":
                pos_side = "long"
        elif trade_side == "open":
            if raw_side == "buy":
                pos_side = "long"
            elif raw_side == "sell":
                pos_side = "short"

    if pos_side not in ("long", "short"):
        raise ValueError(
            "bitget UTA fill cannot infer position side; "
            f"context={bitget_payload_context(fill)}"
        )

    if raw_side not in ("buy", "sell"):
        if trade_side == "close":
            raw_side = "sell" if pos_side == "long" else "buy"
        elif trade_side == "open":
            raw_side = "buy" if pos_side == "long" else "sell"
    if raw_side not in ("buy", "sell"):
        raise ValueError(
            "bitget UTA fill cannot infer order side; "
            f"context={bitget_payload_context(fill)}"
        )

    return raw_side, pos_side


def normalize_uta_fill_payload(
    fill: dict,
    symbol_resolver: Callable[[str], str],
    custom_id_parser: Callable[[object], str] | None = None,
) -> dict:
    """Validate and normalize a Bitget UTA v3 fill payload."""

    exec_id = str(require_uta_field(fill, "execId"))
    order_id = str(require_uta_field(fill, "orderId"))
    timestamp = require_uta_timestamp(fill)
    symbol_external = str(require_uta_field(fill, "symbol"))
    side, position_side = deduce_uta_side_pside(fill)
    symbol = symbol_resolver(symbol_external)
    if not symbol:
        raise ValueError(
            "bitget UTA fill symbol resolver returned an empty symbol; "
            f"symbol={symbol_external!r} context={bitget_payload_context(fill)}"
        )
    cid = fill.get("clientOid")
    return {
        "id": exec_id,
        "order_id": order_id,
        "timestamp": timestamp,
        "datetime": ts_to_date(timestamp),
        "symbol": symbol,
        "symbol_external": symbol_external,
        "side": side,
        "qty": require_uta_float(fill, "execQty", positive=True),
        "price": require_uta_float(fill, "execPrice", positive=True),
        "pnl": require_uta_float(fill, "execPnl"),
        "fees": normalize_bitget_fee_detail(fill.get("feeDetail")),
        "pb_order_type": custom_id_parser(cid) if cid and custom_id_parser is not None else "",
        "position_side": position_side,
        "client_order_id": cid,
    }


def deduce_side_pside(fill: dict) -> tuple[str, str]:
    """Infer standard ``(side, pside)`` for a Bitget fill payload."""

    trade_side = str(fill.get("tradeSide", "") or "").strip().lower()
    raw_side = str(fill.get("side", "") or "").strip().lower()
    pos_mode = str(fill.get("posMode", "") or "").strip().lower()
    raw_pside = _explicit_position_side(fill)

    def _canonical(side: str, pside: str) -> tuple[str, str]:
        if side not in {"buy", "sell"} or pside not in {"long", "short"}:
            raise ValueError(_ambiguous_bitget_fill_message(fill))
        return side, pside

    def _open_side(pside: str) -> str:
        return "buy" if pside == "long" else "sell"

    def _close_side(pside: str) -> str:
        return "sell" if pside == "long" else "buy"

    def _pside_from_open_side(side: str) -> str:
        return "long" if side == "buy" else "short"

    def _pside_from_close_side(side: str) -> str:
        return "short" if side == "buy" else "long"

    # Normalize hedge mode strings first.
    if pos_mode == "hedge_mode":
        if "close_long" in trade_side:
            return _canonical("sell", "long")
        if "close_short" in trade_side:
            return _canonical("buy", "short")
        if trade_side == "open":
            if raw_side in {"buy", "sell"}:
                return _canonical(raw_side, _pside_from_open_side(raw_side))
            if raw_pside:
                return _canonical(_open_side(raw_pside), raw_pside)
            raise ValueError(_ambiguous_bitget_fill_message(fill))
        if trade_side == "close":
            if raw_side in {"buy", "sell"}:
                return _canonical(raw_side, _pside_from_close_side(raw_side))
            if raw_pside:
                return _canonical(_close_side(raw_pside), raw_pside)
            raise ValueError(_ambiguous_bitget_fill_message(fill))
        if "long" in trade_side:
            return _canonical("buy", "long")
        if "short" in trade_side:
            return _canonical("sell", "short")
        if raw_pside and raw_side in {"buy", "sell"}:
            return _canonical(raw_side, raw_pside)
        raise ValueError(_ambiguous_bitget_fill_message(fill))

    # One-way mode ("single") encodes direction explicitly.
    if "reduce_buy_single" in trade_side:
        return _canonical("buy", "long")
    if "reduce_sell_single" in trade_side:
        return _canonical("sell", "short")
    if "burst_buy_single" in trade_side:
        return _canonical("buy", "long")
    if "burst_sell_single" in trade_side:
        return _canonical("sell", "short")
    if "delivery_buy_single" in trade_side:
        return _canonical("buy", "long")
    if "delivery_sell_single" in trade_side:
        return _canonical("sell", "short")
    if "dte_sys_adl_buy_in_single_side_mode" in trade_side:
        return _canonical("buy", "long")
    if "dte_sys_adl_sell_in_single_side_mode" in trade_side:
        return _canonical("sell", "short")
    if "buy_single" in trade_side:
        return _canonical("buy", "long")
    if "sell_single" in trade_side:
        return _canonical("sell", "short")

    # Generic tradeSide fallback: require direction in tradeSide, not merely raw side.
    if "close_long" in trade_side:
        return _canonical("sell", "long")
    if "close_short" in trade_side:
        return _canonical("buy", "short")
    if "buy" in trade_side:
        return _canonical("buy", "long")
    if "sell" in trade_side:
        return _canonical("sell", "short")

    if raw_pside and raw_side in {"buy", "sell"}:
        return _canonical(raw_side, raw_pside)
    if pos_mode in {"one_way_mode", "single_hold", "single_side_mode"} and raw_side in {
        "buy",
        "sell",
    }:
        return _canonical(raw_side, _pside_from_open_side(raw_side))

    raise ValueError(_ambiguous_bitget_fill_message(fill))


def _explicit_position_side(fill: dict) -> str:
    for key in ("posSide", "holdSide", "positionSide", "position_side", "pside"):
        value = str(fill.get(key, "") or "").strip().lower()
        if value in {"long", "short"}:
            return value
    return ""


def _ambiguous_bitget_fill_message(fill: dict) -> str:
    return (
        "cannot infer Bitget fill side/position_side from "
        f"tradeSide={fill.get('tradeSide')!r} side={fill.get('side')!r} "
        f"posMode={fill.get('posMode')!r} posSide={fill.get('posSide')!r} "
        f"holdSide={fill.get('holdSide')!r}"
    )
