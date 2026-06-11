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
