"""Lighter USDC perpetuals through CCXT, using an existing L2 API key."""

from __future__ import annotations

import asyncio
import math
import re
import secrets

import ccxt.async_support as ccxt_async
import ccxt.pro as ccxt_pro

from config.access import require_live_value
from exchanges.ccxt_bot import CCXTBot
from exchanges.lighter_balance import validate_balance

from exchanges.lighter_credentials import SIGNER_REVISION, SIGNER_SHA256, client_config


def finite(value, name: str, *, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"Lighter {name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or (positive and number <= 0):
        raise ValueError(f"Lighter invalid {name}")
    return number


def encode_client_id(value: str) -> int:
    # Four namespace bits, twelve order-type bits, thirty-two random bits.
    if not re.fullmatch(r"0x0[0-9a-f]{11}", value):
        raise ValueError("Lighter requires a generated Passivbot client order ID")
    return (0xB << 44) | int(value[2:], 16)


def decode_client_id(value) -> str:
    text = str(value or "")
    if text.isdecimal():
        number = int(text)
        if 0 <= number < 2**48 and number >> 44 == 0xB:
            return f"0x{number & ((1 << 44) - 1):012x}"
    return text


class _LighterMixin:
    def _validate_order_owner(self, order):
        if str(order["owner_account_index"]) != str(self.options["accountIndex"]):
            raise ValueError("Lighter order belongs to an unexpected account")

    def create_order_request(self, symbol, type, side, amount, price=None, params=None):
        requests = super().create_order_request(
            symbol, type, side, amount, price, params or {}
        )
        # CCXT 4.5.66 omits these with builderFee=False but its ctypes signer
        # still requires them. Zero is the official SDK's no-integrator value.
        for request in requests:
            request.update(
                integrator_account_index=0,
                integrator_taker_fee=0,
                integrator_maker_fee=0,
            )
        return requests

    async def prepare_api_key(self):
        account, key = str(self.options["accountIndex"]), str(
            self.options["apiKeyIndex"]
        )
        signer = await self.load_account(
            self.options["chainId"],
            self.get_lighter_private_key(account, key),
            key,
            account,
        )
        if signer is None:
            raise ValueError("Lighter existing API key signer is unavailable")

    async def fetch_ohlcv(
        self, symbol, timeframe="1h", since=None, limit=None, params=None
    ):
        # The API tail-anchors over-wide ranges and returns at most 500 candles.
        # A bounded half-open range prevents an old since silently losing its
        # first half when CCXT/the candle manager requests 1,000 rows.
        params = dict(params or {})
        count = min(500, int(limit)) if limit is not None else 500
        if count <= 0:
            raise ValueError("Lighter candle limit must be positive")
        duration = self.parse_timeframe(timeframe) * 1000
        end = int(params.get("until", self.milliseconds()))
        if since is None:
            since = end // duration * duration - count * duration
        params["until"] = min(end, int(since) + count * duration)
        return await super().fetch_ohlcv(symbol, timeframe, int(since), count, params)

    async def fetch_first_candle(self, symbol):
        """Find the first available daily candle without an epoch-sized range."""
        day = 86_400_000
        until = self.milliseconds() // day * day
        first = None
        while until > 0:
            since = max(0, until - 500 * day)
            rows = await self.fetch_ohlcv(
                symbol, "1d", since=since, limit=500, params={"until": until}
            )
            if not rows:
                return first
            timestamp = int(rows[0][0])
            if not since <= timestamp < until:
                raise ValueError("Lighter first-candle pagination did not progress")
            first = rows[0]
            until = timestamp
        return first

    async def fetch_open_orders(self, symbol=None, since=None, limit=None, params=None):
        await self.load_markets()
        await self.prepare_api_key()
        request = {"account_index": self.options["accountIndex"], "market_type": "perp"}
        if symbol is not None:
            request["market_id"] = self.market(symbol)["id"]
        request.update(params or {})
        response = await self.privateGetAccountActiveOrders(request)
        rows = response["orders"]
        if not isinstance(rows, list):
            raise ValueError("Lighter active orders must be a complete list")
        for row in rows:
            self._validate_order_owner(row)
            amount = finite(row["initial_base_amount"], "order amount", positive=True)
            remaining = finite(
                row["remaining_base_amount"], "remaining order amount", positive=True
            )
            finite(row["price"], "order price", positive=True)
            if (
                remaining > amount
                or not isinstance(row["reduce_only"], bool)
                or not isinstance(row["is_ask"], bool)
            ):
                raise ValueError("Lighter invalid active order semantics")
        orders = self.parse_orders(rows, None, since, limit)
        ids = [o["id"] for o in orders]
        if any(not value for value in ids) or len(set(ids)) != len(ids):
            raise ValueError(
                "Lighter active order identities are missing or duplicated"
            )
        return orders

    async def fetch_closed_orders(
        self, symbol=None, since=None, limit=None, params=None
    ):
        orders = await super().fetch_closed_orders(symbol, since, limit, params or {})
        for order in orders:
            self._validate_order_owner(order["info"])
        return orders

    async def create_order(self, symbol, type, side, amount, price=None, params=None):
        params = dict(params or {})
        client_id = decode_client_id(params["clientOrderId"])
        # A sendTx receipt is not an order acknowledgement. Resolve the exact
        # client identity from exchange state; never retry a possibly sent write.
        await super().create_order(symbol, type, side, amount, price, params)
        for attempt in range(5):
            await asyncio.sleep(1)
            active = await self.fetch_open_orders(symbol)
            matches = [o for o in active if o["clientOrderId"] == client_id]
            if not matches:
                closed = await self.fetch_closed_orders(symbol, limit=100)
                matches = [o for o in closed if o["clientOrderId"] == client_id]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                raise ValueError("Lighter duplicate client order identity")
        raise ccxt_async.RequestTimeout(
            "Lighter submitted order has no confirmed exchange acknowledgement"
        )

    async def cancel_order(self, id, symbol=None, params=None):
        await super().cancel_order(id, symbol, params or {})
        # A cancel transaction may still be pending; absence from authoritative
        # active orders confirms removal without claiming cancellation over a fill.
        for attempt in range(5):
            await asyncio.sleep(1)
            active = await self.fetch_open_orders(symbol)
            if all(o["id"] != str(id) for o in active):
                return {
                    "id": str(id),
                    "symbol": symbol,
                    "status": "success",
                    "_passivbot_cancel_requires_full_authoritative_confirmation": True,
                }
        raise ccxt_async.RequestTimeout(
            "Lighter cancellation has not left active orders"
        )

    def parse_order(self, order, market=None):
        parsed = super().parse_order(order, market)
        parsed["clientOrderId"] = decode_client_id(parsed["clientOrderId"])
        return parsed

    async def fetch_balance(self, params=None):
        result = await super().fetch_balance(params or {})
        return validate_balance(result, self.options["accountIndex"])

    async def fetch_positions(self, symbols=None, params=None):
        await self.load_markets()
        response = await self.publicGetAccount(
            {"by": "index", "value": self.options["accountIndex"], **(params or {})}
        )
        rows = response["accounts"]
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or str(rows[0]["account_index"]) != str(self.options["accountIndex"])
        ):
            raise ValueError("Lighter returned an unexpected account")
        positions = rows[0]["positions"]
        if not isinstance(positions, list):
            raise ValueError("Lighter positions must be a list")
        market_ids = set()
        for row in positions:
            market_id = str(row["market_id"])
            if market_id in market_ids:
                raise ValueError("Lighter duplicate one-way position market")
            market_ids.add(market_id)
            size = finite(row["position"], "position size")
            if (
                size < 0
                or type(row["sign"]) is not int
                or row["sign"] not in (-1, 1)
                or type(row["margin_mode"]) is not int
                or row["margin_mode"] not in (0, 1)
            ):
                raise ValueError("Lighter invalid position size, sign, or margin mode")
            if size:
                finite(row["avg_entry_price"], "entry price", positive=True)
        return self.parse_positions(positions, symbols)


class AsyncLighter(_LighterMixin, ccxt_async.lighter):
    pass


class ProLighter(_LighterMixin, ccxt_pro.lighter):
    async def watch_orders(self, symbol=None, since=None, limit=None, params=None):
        orders = await super().watch_orders(symbol, since, limit, params or {})
        for order in orders:
            self._validate_order_owner(order["info"])
        return orders


class LighterBot(CCXTBot):
    def __init__(self, config):
        super().__init__(config)
        self.quote = "USDC"
        self.hedge_mode = False
        self._lighter_write_lock = asyncio.Lock()

    def create_ccxt_sessions(self):
        config = client_config(self.user_info)
        self.cca = AsyncLighter(config)
        self._apply_endpoint_override(self.cca)
        self.ccp = (
            ProLighter(client_config(self.user_info)) if self.ws_enabled else None
        )
        if self.ccp is not None:
            self._apply_endpoint_override(self.ccp)

    def format_custom_id_single(self, order_type_id):
        if not 0 <= order_type_id < 4096:
            raise ValueError("Lighter order type exceeds the client-ID encoding range")
        return f"0x{order_type_id:04x}{secrets.randbits(32):08x}"

    def _market_snapshot_ticker_strategy(self):
        return "symbols"

    async def fetch_tickers_for_symbols(self, symbols):
        symbols = list(dict.fromkeys(symbols))
        if not symbols:
            return {}
        # Lighter's ticker has last price but no bid/ask. Read actual depth for
        # requested symbols; never substitute a last trade for the spread.
        tickers = await self.cca.fetch_tickers(symbols)

        async def fetch_book(symbol):
            if self.ccp is not None:
                return await asyncio.wait_for(
                    self.ccp.watch_order_book(symbol, limit=10), timeout=15
                )
            return await self.cca.fetch_order_book(symbol, limit=10)

        if self.ccp is not None:
            # CCXT owns subscription reuse and sequence validation. Consume a
            # fresh update for each requested book, never an untimed cached row.
            books = await asyncio.gather(
                *(fetch_book(s) for s in symbols), return_exceptions=True
            )
            for book in books:
                if isinstance(book, BaseException):
                    raise book
        else:
            books = [await fetch_book(symbol) for symbol in symbols]
        result = {}
        for symbol, book in zip(symbols, books):
            bid = finite(book["bids"][0][0], "best bid", positive=True)
            ask = finite(book["asks"][0][0], "best ask", positive=True)
            last = finite(tickers[symbol]["last"], "last trade", positive=True)
            if bid > ask:
                raise ValueError("Lighter crossed order book")
            result[symbol] = {
                "bid": bid,
                "ask": ask,
                "last": last,
                "source": "lighter_order_book_and_ticker",
            }
        return result

    async def fetch_tickers(self):
        return await self.fetch_tickers_for_symbols(sorted(self.markets_dict))

    def _get_position_side_for_order(self, order):
        return self._normalize_one_way_position_side(order)

    def _build_order_params(self, order):
        side, pside = order["side"], order["position_side"]
        if side not in {"buy", "sell"} or pside not in {"long", "short"}:
            raise ValueError("Lighter order requires explicit side and position_side")
        reduce_only = (side == "sell" and pside == "long") or (
            side == "buy" and pside == "short"
        )
        params = {
            "clientOrderId": encode_client_id(order["custom_id"]),
            "reduceOnly": reduce_only,
        }
        if order.get("type", "limit") == "limit":
            params["timeInForce"] = "GTT"
            if require_live_value(self.config, "time_in_force") == "post_only":
                params.update(postOnly=True, orderExpiry=-1)
        return params

    async def update_exchange_config(self):
        # Lighter positions are intrinsically one-way; there is no hedge-mode mutation.
        await self.cca.prepare_api_key()

    async def update_exchange_config_by_symbols(self, symbols):
        async with self._lighter_write_lock:
            for symbol in symbols:
                mode = self._get_margin_mode_for_symbol(symbol)
                await self.cca.set_leverage(
                    self._calc_leverage_for_symbol(symbol), symbol, {"marginMode": mode}
                )
                self._record_live_margin_mode(symbol, mode)

    def set_market_specific_settings(self):
        super().set_market_specific_settings()
        for symbol in self.symbols_requiring_market_sizing():
            # CCXT's quote_multiplier is a wire scaling factor, not contract size.
            self.c_mults[symbol] = 1.0
            fraction = finite(
                self.markets_dict[symbol]["info"]["min_initial_margin_fraction"],
                "minimum margin fraction",
                positive=True,
            )
            self.max_leverage[symbol] = 10000 / fraction

    async def execute_order(self, order):
        async with self._lighter_write_lock:
            return await super().execute_order(order)

    async def execute_cancellation(self, order):
        async with self._lighter_write_lock:
            return await super().execute_cancellation(order)
