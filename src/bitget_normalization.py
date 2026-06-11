def deduce_side_pside(fill: dict) -> tuple[str, str]:
    """Infer standard ``(side, pside)`` for a Bitget fill payload."""

    trade_side = str(fill.get("tradeSide", "")).lower()
    raw_side = str(fill.get("side", "")).lower()
    pos_mode = str(fill.get("posMode", "")).lower()

    def _canonical(side: str, pside: str) -> tuple[str, str]:
        side = side or ("buy" if pside == "long" else "sell")
        return side, pside

    # Normalize hedge mode strings first.
    if pos_mode == "hedge_mode":
        if "close_long" in trade_side:
            return _canonical("sell", "long")
        if "close_short" in trade_side:
            return _canonical("buy", "short")
        if trade_side == "open":
            if raw_side == "sell":
                return _canonical("sell", "short")
            return _canonical("buy", "long")
        if trade_side == "close":
            if raw_side == "buy":
                return _canonical("buy", "short")
            if raw_side == "sell":
                return _canonical("sell", "long")
            return _canonical("sell", "long")
        if "long" in trade_side:
            return _canonical("buy", "long")
        if "short" in trade_side:
            return _canonical("sell", "short")

    # One-way mode ("single") encodes direction explicitly.
    if "buy_single" in trade_side:
        return _canonical("buy", "long")
    if "sell_single" in trade_side:
        return _canonical("sell", "short")
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

    # Generic fallback: look for keywords.
    if "close_long" in trade_side:
        return _canonical("sell", "long")
    if "close_short" in trade_side:
        return _canonical("buy", "short")
    if "buy" in trade_side:
        return _canonical("buy", "long")
    if "sell" in trade_side:
        return _canonical("sell", "short")

    if raw_side == "sell":
        return _canonical("sell", "long")
    if raw_side == "buy":
        return _canonical("buy", "long")

    return _canonical(raw_side or "buy", "long")
