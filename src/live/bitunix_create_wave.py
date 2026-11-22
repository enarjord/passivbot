"""Certify a pair of independent ordinary Bitunix close submissions."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from math import isfinite
from typing import Mapping

from live.bitunix_create_journal import CLIENT_ORDER_ID_PATTERN, WIRE_TOKEN_PATTERN
from live.market_snapshot import MarketSnapshot


_CLOSE_SIDES = {"long": "sell", "short": "buy"}
_QUOTE_SOURCES = {"fetch_tickers", "fetch_tickers_symbols", "bitunix_depth_mid"}


def _decimal(value) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        return None
    try:
        number = Decimal(str(value))
    except InvalidOperation:
        return None
    if not number.is_finite():
        return None
    # Account and book observations have already crossed an f64 boundary.
    # Values outside that domain cannot certify a quantity or price.
    binary_value = float(number)
    if not isfinite(binary_value) or (number and binary_value == 0.0):
        return None
    return number


def _valid_id(value) -> bool:
    return isinstance(value, str) and WIRE_TOKEN_PATTERN.fullmatch(value) is not None


@dataclass(frozen=True)
class OrdinaryCloseWaveMember:
    order: dict
    symbol: str
    position_side: str
    position_id: str
    client_id: str
    qty: Decimal
    price: Decimal

    def matches_wire(self, body: dict, *, venue_symbol: str) -> bool:
        """Verify that the guarded request equals the certified close."""
        return (
            body.get("clientId") == self.client_id
            and body.get("symbol") == venue_symbol
            and body.get("positionId") == self.position_id
            and body.get("side") == ("BUY" if self.position_side == "long" else "SELL")
            and body.get("tradeSide") == "CLOSE"
            and body.get("reduceOnly") is True
            and body.get("orderType") == "LIMIT"
            and body.get("effect") == "POST_ONLY"
            and _decimal(body.get("qty")) == self.qty
            and _decimal(body.get("price")) == self.price
        )


def close_is_passive(
    member: OrdinaryCloseWaveMember,
    quote: MarketSnapshot | None,
    *,
    now_ms: int,
    max_age_ms: int,
) -> bool:
    if (
        not isinstance(quote, MarketSnapshot)
        or quote.symbol != member.symbol
        or quote.source not in _QUOTE_SOURCES
        or not quote.is_valid()
        or not (
            0 < quote.fetched_ms <= now_ms and now_ms - quote.fetched_ms <= max_age_ms
        )
    ):
        return False
    # The provider can restamp a ticker when it reads the connector cache.
    # Require the actual quote timestamp to fit the same freshness bound.
    quote_ms = quote.exchange_timestamp_ms
    if quote_ms is None or not (
        0 < quote_ms <= now_ms and now_ms - quote_ms <= max_age_ms
    ):
        return False
    if member.position_side == "long":
        return member.price > _decimal(quote.bid)
    return member.price < _decimal(quote.ask)


def _position_for_close(bot, symbol: str, position_side: str):
    rows = [
        row
        for row in bot.fetched_positions
        if row.get("symbol") == symbol and row.get("position_side") == position_side
    ]
    if len(rows) != 1:
        return None
    info = rows[0].get("info") or {}
    position_id = info.get("positionId")
    authoritative_qty = _decimal(info.get("qty"))
    normalized_qty = _decimal(
        bot.positions.get(symbol, {}).get(position_side, {}).get("size")
    )
    if (
        not _valid_id(position_id)
        or bot.cca._position_ids.get((symbol, position_side)) != position_id
        or authoritative_qty is None
        or authoritative_qty <= 0
        or normalized_qty is None
        or normalized_qty
        != (authoritative_qty if position_side == "long" else -authoritative_qty)
    ):
        return None
    return position_id, authoritative_qty


def _remaining_qty(info: dict) -> Fraction | None:
    qty = _decimal(info.get("qty"))
    filled = [
        _decimal(info[key])
        for key in ("tradeQty", "dealAmount")
        if info.get(key) not in (None, "")
    ]
    if qty is None or qty <= 0 or not filled or filled[0] is None:
        return None
    if any(value != filled[0] for value in filled) or not 0 <= filled[0] < qty:
        return None
    return Fraction(qty) - Fraction(filled[0])


def _resting_close_qty(bot, member: OrdinaryCloseWaveMember) -> Fraction | None:
    total = Fraction()
    for order in bot.fetched_open_orders:
        if order["symbol"] != member.symbol:
            continue
        position_side = order.get("position_side")
        if position_side not in _CLOSE_SIDES:
            return None
        if position_side != member.position_side:
            continue
        info = order.get("info") or {}
        reduce_only = info.get("reduceOnly")
        if type(reduce_only) is not bool:
            return None
        if not reduce_only:
            continue
        if order.get("side") != _CLOSE_SIDES[position_side] or (
            info.get("positionId") not in (None, "", member.position_id)
        ):
            return None
        remaining = _remaining_qty(info)
        if remaining is None:
            return None
        total += remaining
    return total


def _certify_close(bot, order, quotes, *, now_ms, max_age_ms):
    position_side = order.get("position_side")
    symbol = order.get("symbol")
    client_id = order.get("custom_id")
    qty = _decimal(order.get("qty"))
    price = _decimal(order.get("price"))
    if (
        position_side not in _CLOSE_SIDES
        or order.get("side") != _CLOSE_SIDES[position_side]
        or order.get("reduce_only") is not True
        or order.get("type") != "limit"
        or order.get("execution_priority") != "ordinary"
        or order.get("pb_order_type")
        not in {f"close_grid_{position_side}", f"close_trailing_{position_side}"}
        or not isinstance(symbol, str)
        or not symbol
        or not isinstance(client_id, str)
        or CLIENT_ORDER_ID_PATTERN.fullmatch(client_id) is None
        or qty is None
        or qty <= 0
        or price is None
        or price <= 0
    ):
        return None
    position = _position_for_close(bot, symbol, position_side)
    if position is None:
        return None
    position_id, capacity = position
    member = OrdinaryCloseWaveMember(
        order, symbol, position_side, position_id, client_id, qty, price
    )
    if not close_is_passive(
        member, quotes.get(symbol), now_ms=now_ms, max_age_ms=max_age_ms
    ):
        return None
    reserved = _resting_close_qty(bot, member)
    if reserved is None or reserved + Fraction(qty) > Fraction(capacity):
        return None
    return member


def select_ordinary_close_wave(
    bot,
    orders: list[dict],
    quotes: Mapping[str, MarketSnapshot],
    *,
    now_ms: int,
    max_age_ms: int,
) -> tuple[OrdinaryCloseWaveMember, ...]:
    """Select the leading two admitted closes when all joint checks pass."""
    if len(orders) < 2 or bot.live_value("time_in_force") != "post_only":
        return ()
    from live.realized_close_admission import wave_allowed

    if not wave_allowed(bot):
        # Existing venue closes may differ from the current Rust ideal within
        # tolerance. The live cohort has no certificate of their loss budget.
        return ()
    if len(bot.fetched_open_orders) + 2 > bot.MAX_OPEN_ORDERS:
        return ()
    ids = [order.get("id") for order in bot.fetched_open_orders]
    if any(not _valid_id(id_) for id_ in ids) or len(ids) != len(set(ids)):
        return ()
    first, second = (
        _certify_close(bot, order, quotes, now_ms=now_ms, max_age_ms=max_age_ms)
        for order in orders[:2]
    )
    if (
        first is None
        or second is None
        or first.symbol == second.symbol
        or first.position_id == second.position_id
        or first.client_id == second.client_id
    ):
        return ()
    client_ids = {order.get("clientOrderId") for order in bot.fetched_open_orders}
    if first.client_id in client_ids or second.client_id in client_ids:
        return ()
    return first, second
