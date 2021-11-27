from __future__ import annotations

import asyncio
import logging
import operator
from typing import Any

import numpy as np

from passivbot.bot import Bot
from passivbot.datastructures import Asset
from passivbot.datastructures import Candle
from passivbot.datastructures import Fill
from passivbot.datastructures import Order
from passivbot.datastructures import Position
from passivbot.datastructures import Tick
from passivbot.datastructures.config import NamedConfig
from passivbot.datastructures.runtime import RuntimeSpotConfig
from passivbot.utils.funcs.njit import calc_diff
from passivbot.utils.funcs.njit import calc_long_pnl
from passivbot.utils.funcs.njit import calc_min_entry_qty
from passivbot.utils.funcs.njit import calc_upnl
from passivbot.utils.funcs.njit import qty_to_cost
from passivbot.utils.funcs.njit import round_dn
from passivbot.utils.funcs.njit import round_up
from passivbot.utils.funcs.pure import calc_long_pprice
from passivbot.utils.funcs.pure import get_position_fills
from passivbot.utils.funcs.pure import ts_to_date
from passivbot.utils.httpclient import BinanceHTTPClient
from passivbot.utils.httpclient import HTTPRequestError
from passivbot.utils.procedures import log_async_exception

log = logging.getLogger(__name__)


class BinanceBotSpot(Bot):

    rtc: RuntimeSpotConfig
    httpclient: BinanceHTTPClient

    def __bot_init__(self):
        """
        Subclass initialization routines
        """
        self.exchange = "binance_spot"
        self.balance: dict[str, Asset] = {}
        self.force_update_interval = 40

    @staticmethod
    def get_initial_runtime_config(config: NamedConfig) -> RuntimeSpotConfig:
        return RuntimeSpotConfig(market_type=config.market_type, long=config.long)

    @staticmethod
    async def get_exchange_info() -> dict[str, Any]:
        response: dict[str, Any] = await BinanceHTTPClient.onetime_get(
            "https://api.binance.com/api/v3/exchangeInfo"
        )
        return response

    @staticmethod
    async def get_httpclient(config: NamedConfig) -> BinanceHTTPClient:
        websocket_url = "wss://stream.binance.com/ws"
        httpclient = BinanceHTTPClient(
            "https://api.binance.com",
            config.api_key.key,
            config.api_key.secret,
            endpoints={
                "balance": "/api/v3/account",
                "exchange_info": "/api/v3/exchangeInfo",
                "open_orders": "/api/v3/openOrders",
                "ticker": "/api/v3/ticker/bookTicker",
                "fills": "/api/v3/myTrades",
                "create_order": "/api/v3/order",
                "cancel_order": "/api/v3/order",
                "ticks": "/api/v3/aggTrades",
                "ohlcvs": "/api/v3/klines",
                "websocket": websocket_url,
                "websocket_market": f"{websocket_url}/{config.symbol.name.lower()}@aggTrade",
                "websocket_user": websocket_url,
                "listen_key": "/api/v3/userDataStream",
                "transfer": "/sapi/v1/asset/transfer",
                "account": "/api/v3/account",
            },
        )
        return httpclient

    @staticmethod
    async def init_market_type(config: NamedConfig, rtc: RuntimeSpotConfig):  # type: ignore[override]
        log.info("spot market")
        if "spot" not in rtc.market_type:
            rtc.market_type += "_spot"
        rtc.inverse = False
        rtc.hedge_mode = False
        rtc.pair = config.symbol.name
        exchange_info: dict[str, Any] = await BinanceBotSpot.get_exchange_info()
        for e in exchange_info["symbols"]:
            if e["symbol"] == config.symbol.name:
                rtc.coin = e["baseAsset"]
                rtc.quote = e["quoteAsset"]
                rtc.margin_coin = e["quoteAsset"]
                for q in e["filters"]:
                    if q["filterType"] == "LOT_SIZE":
                        rtc.min_qty = float(q["minQty"])
                        rtc.qty_step = float(q["stepSize"])
                    elif q["filterType"] == "PRICE_FILTER":
                        rtc.price_step = float(q["tickSize"])
                        rtc.min_price = float(q["minPrice"])
                        rtc.max_price = float(q["maxPrice"])
                    elif q["filterType"] == "PERCENT_PRICE":
                        rtc.price_multiplier_up = float(q["multiplierUp"])
                        rtc.price_multiplier_dn = float(q["multiplierDown"])
                    elif q["filterType"] == "MIN_NOTIONAL":
                        rtc.min_cost = float(q["minNotional"])
                break

    async def _init(self):
        self.httpclient = await self.get_httpclient(self.config)
        await self.init_market_type(self.config, self.rtc)
        await super()._init()
        await self.init_order_book()
        await self.update_position()

    def calc_orders(self) -> list[Order]:
        default_orders = super().calc_orders()
        orders = []
        assert self.rtc.quote is not None
        remaining_cost = self.balance[self.rtc.quote].onhand
        for order in sorted(default_orders, key=lambda x: calc_diff(x.price, self.rtc.price)):  # type: ignore[no-any-return]
            if order.price > min(
                self.rtc.max_price,
                round_dn(self.rtc.price * self.rtc.price_multiplier_up, self.rtc.price_step),
            ):
                log.warning(f"price {order.price} too high")
                continue
            if order.price < max(
                self.rtc.min_price,
                round_up(self.rtc.price * self.rtc.price_multiplier_dn, self.rtc.price_step),
            ):
                log.warning(f"price {order.price} too low")
                continue
            if order.side == "buy":
                cost = qty_to_cost(order.qty, order.price, self.rtc.inverse, self.rtc.c_mult)
                if cost > remaining_cost:
                    adjusted_qty = round_dn(remaining_cost / order.price, self.rtc.qty_step)
                    min_entry_qty = calc_min_entry_qty(
                        order.price,
                        self.rtc.inverse,
                        self.rtc.qty_step,
                        self.rtc.min_qty,
                        self.rtc.min_cost,
                    )
                    if adjusted_qty >= min_entry_qty:
                        order.qty = adjusted_qty
                        orders.append(order)
                        remaining_cost = 0.0
                else:
                    orders.append(order)
                    remaining_cost -= cost
            else:
                orders.append(order)
        return orders

    async def check_if_other_positions(self, abort=True):
        pass
        # todo...

    async def execute_leverage_change(self):
        pass

    async def init_exchange_config(self):
        await self.check_if_other_positions()

    async def init_order_book(self):
        ticker: dict[str, Any] = await self.httpclient.get(
            "ticker", params={"symbol": self.config.symbol.name}
        )
        self.ob = [float(ticker["bidPrice"]), float(ticker["askPrice"])]
        self.rtc.price = np.random.choice(self.ob)

    async def fetch_open_orders(self) -> list[Order]:
        return [
            Order.from_binance_payload(e)
            for e in await self.httpclient.get(
                "open_orders", signed=True, params={"symbol": self.config.symbol.name}
            )
        ]

    async def fetch_position(self) -> Position:
        balances: dict[str, Any] = await self.httpclient.get("balance", signed=True)
        await self.update_fills()
        balance: dict[str, Asset] = {}
        for elm in balances["balances"]:
            free = float(elm["free"])
            locked = float(elm["locked"])
            onhand = free + locked
            balance[elm["asset"]] = Asset(free=free, locked=locked, onhand=onhand)
        if "BNB" in balance:
            balance["BNB"].onhand = max(0.0, balance["BNB"].onhand - 0.01)
        self.balance = balance
        return self.calc_simulated_position(self.balance, self.fills)

    def calc_simulated_position(
        self, balance: dict[str, Asset], long_fills: list[Fill]
    ) -> Position:
        assert self.rtc.coin is not None
        assert self.rtc.quote is not None
        long_psize = round_dn(balance[self.rtc.coin].onhand, self.rtc.qty_step)
        long_pfills, short_pfills = get_position_fills(long_psize, 0.0, self.fills)
        long_pprice = calc_long_pprice(long_psize, long_pfills) if long_psize else 0.0
        if long_psize * long_pprice < self.rtc.min_cost:
            long_psize, long_pprice, long_pfills = 0.0, 0.0, []
        position = {
            "long": {"size": long_psize, "price": long_pprice, "liquidation_price": 0.0},
            "short": {"size": 0.0, "price": 0.0, "liquidation_price": 0.0},
            "wallet_balance": (
                balance[self.rtc.quote].onhand + balance[self.rtc.coin].onhand * long_pprice
            ),
        }
        return Position.parse_obj(position)

    async def execute_order(self, order: Order) -> Order:
        params = order.to_binance_payload(futures=False)
        o = await self.httpclient.post("create_order", params=params)
        o["symbol"] = self.config.symbol.name
        return Order.from_binance_payload(o, futures=False)

    async def execute_cancellation(self, order: Order) -> Order | None:
        cancellation = None
        try:
            cancellation = await self.httpclient.delete(
                "cancel_order",
                params={"symbol": self.config.symbol.name, "orderId": order.order_id},
            )
            cancellation["symbol"] = order.symbol
            return Order.from_binance_payload(cancellation, futures=False)
        except HTTPRequestError as exc:
            log.error("API Error code=%s; message=%s", exc.code, exc.msg)
            self.ts_released["force_update"] = 0.0
        except Exception as exc:
            log.info("error cancelling order %r: %s", order, exc, exc_info=True)
            log_async_exception(cancellation)
            self.ts_released["force_update"] = 0.0
        return None

    async def get_all_fills(
        self, symbol: str | None = None, start_time: int | None = None
    ) -> list[Fill]:
        fills: list[Fill] = []
        i = 0
        while True:
            i += 1
            if i >= 15:
                log.info("Warning: more than 15 calls to fetch_fills(), breaking")
                break
            fetched = await self.fetch_fills(symbol=symbol, start_time=start_time)
            log.info("fetched fills %s", fetched[0].dt)
            if fetched == fills[-len(fetched) :]:
                break
            fills += fetched
            if len(fetched) < 1000:
                break
            start_time = fills[-1].timestamp
        fills_d = {e.id: e for e in fills}
        return sorted(fills_d.values(), key=operator.attrgetter("timestamp"))

    async def get_all_income(
        self,
        symbol: str | None = None,
        start_time: int | None = None,
        income_type: str = "realized_pnl",
        end_time: int | None = None,
    ):
        fills = await self.get_all_fills(symbol=symbol, start_time=start_time)

        income: list[dict[str, Any]] = []
        psize, pprice = 0.0, 0.0
        for fill in fills:
            if fill.side == "buy":
                new_psize = psize + fill.qty
                pprice = pprice * (psize / new_psize) + fill.price * (fill.qty / new_psize)
                psize = new_psize
            elif psize > 0.0:
                income.append(
                    {
                        "symbol": fill.symbol,
                        "income_type": "realized_pnl",
                        "income": calc_long_pnl(pprice, fill.price, fill.qty, False, 1.0),
                        "token": self.rtc.quote,
                        "timestamp": fill.timestamp,
                        "info": 0,
                        "transaction_id": fill.id,
                        "trade_id": fill.id,
                    }
                )
                psize = max(0.0, psize - fill.qty)
        return income

    async def fetch_fills(
        self,
        symbol: str | None = None,
        limit: int = 1000,
        from_id: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
    ) -> list[Fill]:
        params = {
            "symbol": (self.config.symbol.name if symbol is None else symbol),
            "limit": min(1000, max(500, limit)),
        }
        if from_id is not None:
            params["fromId"] = max(0, from_id)
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None and start_time is not None:
            params["endTime"] = int(min(end_time, start_time + 1000 * 60 * 60 * 23.99))
        try:
            fetched: list[dict[str, Any]] = await self.httpclient.get(  # type: ignore[assignment]
                "fills", signed=True, params=params
            )
            return [
                Fill.from_binance_payload(x, futures=False, inverse=self.rtc.inverse)
                for x in fetched
            ]
        except HTTPRequestError as exc:
            log.error("API Error code=%s; message=%s", exc.code, exc.msg)
        except Exception as e:
            log.error("error fetching fills a: %s", e, exc_info=True)
        return []

    async def fetch_income(
        self,
        symbol: str | None = None,
        limit: int = 1000,
        start_time: int | None = None,
        end_time: int | None = None,
    ):
        log.info("fetch income not implemented in spot")
        return []

    async def fetch_account(self):
        try:
            return await self.httpclient.get("balance", signed=True)
        except HTTPRequestError as exc:
            log.error("API Error code=%s; message=%s", exc.code, exc.msg)
        except Exception as e:
            log.error("error fetching account: %s", e)
        return {"balances": []}

    async def fetch_ticks(
        self,
        from_id: int | None = None,
        start_time: int | None = None,
        end_time: int | None = None,
        do_print: bool = True,
    ):
        params = {"symbol": self.config.symbol.name, "limit": 1000}
        if from_id is not None:
            params["fromId"] = max(0, from_id)
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        try:
            fetched: list[dict[str, Any]] = await self.httpclient.get("ticks", params=params)  # type: ignore[assignment]
        except HTTPRequestError as exc:
            log.error("API Error code=%s; message=%s", exc.code, exc.msg)
            return []
        except Exception as e:
            log.error("error fetching ticks a: %s", e)
            return []
        try:
            ticks = [Tick.from_binance_payload(t) for t in fetched]
            if do_print:
                log.info(
                    "fetched ticks for symbold %r %s %s",
                    self.config.symbol.name,
                    ticks[0].trade_id,
                    ts_to_date(float(ticks[0].timestamp) / 1000),
                )
        except Exception as e:
            log.info("error fetching ticks b: %s - %s", e, fetched)
            ticks = []
            if do_print:
                log.info("fetched no new ticks %s", self.config.symbol.name)
        return ticks

    async def fetch_ticks_time(
        self, start_time: int, end_time: int | None = None, do_print: bool = True
    ):
        return await self.fetch_ticks(start_time=start_time, end_time=end_time, do_print=do_print)

    async def fetch_ohlcvs(
        self,
        symbol: str | None = None,
        start_time: int | None = None,
        interval: str = "1m",
        limit: int = 1000,
    ) -> list[Candle]:
        # m -> minutes; h -> hours; d -> days; w -> weeks; M -> months
        interval_map = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "12h": 720,
            "1d": 60 * 60 * 24,
            "1w": 60 * 60 * 24 * 7,
            "1M": 60 * 60 * 24 * 30,
        }
        assert interval in interval_map
        if symbol is None:
            symbol = self.config.symbol.name
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            start_time = int(start_time)
            params["startTime"] = start_time
            interval_time: int = interval_map[interval]
            params["endTime"] = start_time + interval_time * 60 * 1000 * limit

        fetched = None
        try:
            fetched = await self.httpclient.get("ohlcvs", params=params)
            return [Candle.parse_obj(e) for e in fetched]
        except HTTPRequestError as exc:
            log.error("API Error code=%s; message=%s", exc.code, exc.msg)
        except Exception as e:
            log.error("error fetching ohlcvs: %s %s", fetched, e, exc_info=True)
        return []

    async def transfer(self, type_: str, amount: float, asset: str = "USDT"):
        log.info("transfer not implemented in spot")
        return

    def standardize_market_stream_event(self, data: dict[str, Any]) -> list[Tick]:
        try:
            return [Tick.from_binance_payload(data)]
        except Exception as e:
            log.error("error in websocket tick: %s", e)
        return []

    async def beat_heart_user_stream(self, ws) -> None:
        while True:
            await asyncio.sleep(60 + np.random.randint(60 * 9, 60 * 14))
            await self.init_user_stream()

    async def init_user_stream(self) -> None:
        try:
            response = await self.httpclient.post("listen_key", signed=False)
            self.listen_key = response["listenKey"]
            self.httpclient.endpoints[
                "websocket_user"
            ] = f"{self.httpclient.endpoints['websocket']}/{self.listen_key}"
        except HTTPRequestError as exc:
            log.error("API Error code=%s; message=%s", exc.code, exc.msg)
        except Exception as e:
            log.error("error fetching listen key: %s", e, exc_info=True)

    async def on_user_stream_event(self, event: dict[str, Any]) -> None:
        try:
            pos_change = False
            if "balance" in event:
                onhand_change = False
                for token in event["balance"]:
                    self.balance[token].free = event["balance"][token]["free"]
                    self.balance[token].locked = event["balance"][token]["locked"]
                    onhand = self.balance[token].free + self.balance[token].locked
                    if (
                        token in (self.rtc.quote, self.rtc.coin)
                        and self.balance[token].onhand != onhand
                    ):
                        onhand_change = True
                    if token == "BNB":
                        onhand = max(0.0, onhand - 0.01)
                    self.balance[token].onhand = onhand
                if onhand_change:
                    self.position = self.calc_simulated_position(self.balance, self.fills)
                    self.position.wallet_balance = self.adjust_wallet_balance(
                        self.position.wallet_balance
                    )
                    self.position = self.add_wallet_exposures_to_pos(self.position)
                    pos_change = True
            if "filled" in event:
                if event["filled"].order_id not in {fill.order_id for fill in self.fills}:
                    self.fills[:] = sorted(
                        self.fills + [event["filled"]], key=operator.attrgetter("order_id")
                    )
                self.position = self.calc_simulated_position(self.balance, self.fills)
                self.position.wallet_balance = self.adjust_wallet_balance(
                    self.position.wallet_balance
                )
                self.position = self.add_wallet_exposures_to_pos(self.position)
                pos_change = True
            elif "partially_filled" in event:
                await asyncio.sleep(0.01)
                await asyncio.gather(self.update_position(), self.update_open_orders())
                pos_change = True
            if "new_open_order" in event:
                if event["new_open_order"]["order_id"] not in {
                    x.order_id for x in self.open_orders
                }:
                    self.open_orders.append(
                        Order.from_binance_payload(event["new_open_order"], futures=False)
                    )
            elif "deleted_order_id" in event:
                for i, o in enumerate(self.open_orders):
                    if o.order_id == event["deleted_order_id"]:
                        self.open_orders = self.open_orders[:i] + self.open_orders[i + 1 :]
                        break
            if pos_change:
                self.position.equity = self.position.wallet_balance + calc_upnl(
                    self.position.long.size,
                    self.position.long.price,
                    self.position.short.size,
                    self.position.short.price,
                    self.rtc.price,
                    self.rtc.inverse,
                    self.rtc.c_mult,
                )
                await asyncio.sleep(
                    0.01
                )  # sleep 10 ms to catch both pos update and open orders update
                await self.cancel_and_create()
        except Exception as e:
            log.error("error handling user stream event: %s", e, exc_info=True)

    def standardize_user_stream_event(self, event: dict[str, Any]) -> dict[str, Any]:
        standardized: dict[str, Any] = {}
        if "e" in event:
            if event["e"] == "outboundAccountPosition":
                standardized["balance"] = {}
                for e in event["B"]:
                    standardized["balance"][e["a"]] = {
                        "free": float(e["f"]),
                        "locked": float(e["l"]),
                    }
            elif event["e"] == "executionReport":
                if event["X"] == "NEW":
                    if event["s"] == self.config.symbol.name:
                        standardized["new_open_order"] = {
                            "order_id": int(event["i"]),
                            "symbol": event["s"],
                            "price": float(event["p"]),
                            "qty": float(event["q"]),
                            "type": event["o"].lower(),
                            "side": event["S"].lower(),
                            "position_side": "long",
                            "timestamp": int(event["T"]),
                        }
                    else:
                        standardized["other_symbol"] = event["s"]
                        standardized["other_type"] = "new_open_order"
                elif event["X"] in ["CANCELED", "EXPIRED", "REJECTED"]:
                    if event["s"] == self.config.symbol.name:
                        standardized["deleted_order_id"] = int(event["i"])
                    else:
                        standardized["other_symbol"] = event["s"]
                        standardized["other_type"] = event["X"].lower()
                elif event["X"] == "FILLED":
                    if event["s"] == self.config.symbol.name:
                        price = fp if (fp := float(event["p"])) != 0.0 else float(event["L"])
                        standardized["filled"] = Fill.parse_obj(
                            {
                                "id": int(event["g"]),
                                "order_id": int(event["i"]),
                                "symbol": event["s"],
                                "price": price,
                                "qty": float(event["q"]),
                                "cost": float(event["Y"]),
                                "realized_pnl": 0.0,
                                "fee_paid": float(event["n"]),
                                "fee_token": event["N"],
                                "type": event["o"].lower(),
                                "side": event["S"].lower(),
                                "position_side": "long",
                                "timestamp": int(event["T"]),
                                "is_maker": event["m"],
                                "dt": ts_to_date(int(event["T"])),
                            }
                        )
                        standardized["deleted_order_id"] = standardized["filled"].order_id
                    else:
                        standardized["other_symbol"] = event["s"]
                        standardized["other_type"] = "filled"
                elif event["X"] == "PARTIALLY_FILLED":
                    if event["s"] == self.config.symbol.name:
                        price = fp if (fp := float(event["p"])) != 0.0 else float(event["L"])
                        standardized["filled"] = Fill.parse_obj(
                            {
                                "id": int(event["g"]),
                                "order_id": int(event["i"]),
                                "symbol": event["s"],
                                "price": price,
                                "qty": float(event["q"]),
                                "cost": float(event["Y"]),
                                "realized_pnl": 0.0,
                                "fee_paid": int(event["n"]),
                                "fee_token": event["N"],
                                "type": event["o"].lower(),
                                "side": event["S"].lower(),
                                "position_side": "long",
                                "timestamp": int(event["T"]),
                                "is_maker": event["m"],
                                "dt": ts_to_date(int(event["T"])),
                            }
                        )
                        standardized["deleted_order_id"] = standardized["partially_filled"].order_id
                    else:
                        standardized["other_symbol"] = event["s"]
                        standardized["other_type"] = "partially_filled"

        return standardized
