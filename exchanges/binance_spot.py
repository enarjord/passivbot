import asyncio
import hashlib
import hmac
import json
import traceback
from time import time
from urllib.parse import urlencode

import aiohttp
import numpy as np

from njit_funcs import (
    round_dn,
    round_up,
    calc_pnl_long,
    calc_min_entry_qty,
    qty_to_cost,
    calc_upnl,
    calc_diff,
)
from passivbot import Bot
from procedures import print_, print_async_exception
from pure_funcs import (
    ts_to_date,
    sort_dict_keys,
    calc_pprice_long,
    format_float,
    get_position_fills,
    spotify_config,
)


class BinanceBotSpot(Bot):
    def __init__(self, config: dict):
        if config["exchange"] == "binance_us":
            self.exchange = "binance_us"
        else:
            self.exchange = "binance_spot"
        self.balance = {}
        super().__init__(spotify_config(config))
        self.spot = self.config["spot"] = True
        self.inverse = self.config["inverse"] = False
        self.hedge_mode = self.config["hedge_mode"] = False
        self.do_short = self.config["do_short"] = self.config["short"]["enabled"] = False
        self.session = aiohttp.ClientSession()
        self.headers = {"X-MBX-APIKEY": self.key}
        self.base_endpoint = ""
        self.force_update_interval = 40
        self.max_n_orders_per_batch = 5
        self.max_n_cancellations_per_batch = 10

    async def public_get(self, url: str, params: dict = {}) -> dict:
        async with self.session.get(self.base_endpoint + url, params=params) as response:
            result = await response.text()
        return json.loads(result)

    async def private_(self, type_: str, base_endpoint: str, url: str, params: dict = {}) -> dict:
        timestamp = int(time() * 1000)
        params.update({"timestamp": timestamp, "recvWindow": 5000})
        for k in params:
            if type(params[k]) == bool:
                params[k] = "true" if params[k] else "false"
            elif type(params[k]) == float:
                params[k] = format_float(params[k])
        params = sort_dict_keys(params)
        params["signature"] = hmac.new(
            self.secret.encode("utf-8"),
            urlencode(params).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        async with getattr(self.session, type_)(
            base_endpoint + url, params=params, headers=self.headers
        ) as response:
            result = await response.text()
        return json.loads(result)

    async def post_listen_key(self):
        async with self.session.post(
            self.base_endpoint + self.endpoints["listen_key"],
            params={},
            headers=self.headers,
        ) as response:
            result = await response.text()
        return json.loads(result)

    async def private_get(self, url: str, params: dict = {}) -> dict:
        return await self.private_("get", self.base_endpoint, url, params)

    async def private_post(self, url: str, params: dict = {}) -> dict:
        return await self.private_("post", self.base_endpoint, url, params)

    async def private_delete(self, url: str, params: dict = {}) -> dict:
        return await self.private_("delete", self.base_endpoint, url, params)

    def init_market_type(self):
        print("spot market")
        if "spot" not in self.market_type:
            self.market_type += "_spot"
        self.inverse = self.config["inverse"] = False
        self.spot = True
        self.hedge_mode = False
        self.pair = self.symbol
        self.endpoints = {
            "balance": "/api/v3/account",
            "exchange_info": "/api/v3/exchangeInfo",
            "open_orders": "/api/v3/openOrders",
            "ticker": "/api/v3/ticker/bookTicker",
            "fills": "/api/v3/myTrades",
            "fills_detailed": "/api/v3/allOrders",
            "create_order": "/api/v3/order",
            "cancel_order": "/api/v3/order",
            "ticks": "/api/v3/aggTrades",
            "ohlcvs": "/api/v3/klines",
            "listen_key": "/api/v3/userDataStream",
        }
        self.endpoints["transfer"] = "/sapi/v1/asset/transfer"
        self.endpoints["account"] = "/api/v3/account"
        if self.exchange == "binance_us":
            self.base_endpoint = "https://api.binance.us"
            self.endpoints["websocket"] = "wss://stream.binance.us:9443/ws/"
        else:
            self.base_endpoint = "https://api.binance.com"
            self.endpoints["websocket"] = "wss://stream.binance.com/ws/"
        self.endpoints["websocket_market"] = (
            self.endpoints["websocket"] + f"{self.symbol.lower()}@aggTrade"
        )
        self.endpoints["websocket_user"] = self.endpoints["websocket"]

    async def _init(self):
        self.init_market_type()
        exchange_info = await self.public_get(self.endpoints["exchange_info"])
        for e in exchange_info["symbols"]:
            if e["symbol"] == self.symbol:
                self.coin = e["baseAsset"]
                self.quote = self.margin_coin = e["quoteAsset"]
                for q in e["filters"]:
                    if q["filterType"] == "LOT_SIZE":
                        self.min_qty = self.config["min_qty"] = float(q["minQty"])
                        self.qty_step = self.config["qty_step"] = float(q["stepSize"])
                    elif q["filterType"] == "PRICE_FILTER":
                        self.price_step = self.config["price_step"] = float(q["tickSize"])
                        self.min_price = float(q["minPrice"])
                        self.max_price = float(q["maxPrice"])
                    elif q["filterType"] == "PERCENT_PRICE_BY_SIDE":
                        self.price_multiplier_up = min(
                            float(q["bidMultiplierUp"]), float(q["askMultiplierUp"])
                        )
                        self.price_multiplier_dn = max(
                            float(q["bidMultiplierDown"]), float(q["askMultiplierDown"])
                        )
                    elif q["filterType"] == "MIN_NOTIONAL":
                        self.min_cost = self.config["min_cost"] = float(q["minNotional"])
                try:
                    z = self.min_cost
                except AttributeError:
                    self.min_cost = self.config["min_cost"] = 0.0
                break

        await super()._init()
        await self.init_order_book()
        await self.update_position()

    def calc_orders(self):
        default_orders = super().calc_orders()
        orders = []
        remaining_cost = self.balance[self.quote]["onhand"]
        for order in sorted(default_orders, key=lambda x: calc_diff(x["price"], self.price)):
            if order["price"] > min(
                self.max_price,
                round_dn(self.price * self.price_multiplier_up, self.price_step),
            ):
                print(f'price {order["price"]} too high')
                continue
            if order["price"] < max(
                self.min_price,
                round_up(self.price * self.price_multiplier_dn, self.price_step),
            ):
                print(f'price {order["price"]} too low')
                continue
            if order["side"] == "buy":
                cost = qty_to_cost(order["qty"], order["price"], self.inverse, self.c_mult)
                if cost > remaining_cost:
                    adjusted_qty = round_dn(remaining_cost / order["price"], self.qty_step)
                    min_entry_qty = calc_min_entry_qty(
                        order["price"],
                        self.inverse,
                        self.qty_step,
                        self.min_qty,
                        self.min_cost,
                    )
                    if adjusted_qty >= min_entry_qty:
                        orders.append({**order, **{"qty": adjusted_qty}})
                        remaining_cost = 0.0
                else:
                    orders.append(order)
                    remaining_cost -= cost
            else:
                # TODO: ensure sell qty is greater than min qty
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
        ticker = None
        try:
            ticker = await self.public_get(self.endpoints["ticker"], {"symbol": self.symbol})
            self.ob = [float(ticker["bidPrice"]), float(ticker["askPrice"])]
            self.price = np.random.choice(self.ob)
            return True
        except Exception as e:
            logging.error(f"error updating order book {e}")
            print_async_exception(ticker)
            return False

    async def fetch_open_orders(self) -> [dict]:
        return [
            {
                "order_id": int(e["orderId"]),
                "custom_id": e["clientOrderId"],
                "symbol": e["symbol"],
                "price": float(e["price"]),
                "qty": float(e["origQty"]),
                "type": e["type"].lower(),
                "side": e["side"].lower(),
                "position_side": "long",
                "timestamp": int(e["time"]),
            }
            for e in await self.private_get(self.endpoints["open_orders"], {"symbol": self.symbol})
        ]

    async def fetch_position(self) -> dict:
        balances, _ = await asyncio.gather(
            self.private_get(self.endpoints["balance"]), self.update_fills()
        )
        balance = {}
        for elm in balances["balances"]:
            balance[elm["asset"]] = {"free": float(elm["free"])}
            balance[elm["asset"]]["locked"] = float(elm["locked"])
            balance[elm["asset"]]["onhand"] = (
                balance[elm["asset"]]["free"] + balance[elm["asset"]]["locked"]
            )
        if "BNB" in balance:
            balance["BNB"]["onhand"] = max(0.0, balance["BNB"]["onhand"] - 0.01)
        self.balance = balance
        return self.calc_simulated_position(self.balance, self.fills)

    def calc_simulated_position(self, balance: dict, long_fills: [dict]) -> dict:
        """
        balance = {'BTC': {'free': float, 'locked': float, 'onhand': float}, ...}
        long_pfills = [{order...}, ...]
        """
        psize_long = round_dn(balance[self.coin]["onhand"], self.qty_step)
        long_pfills, short_pfills = get_position_fills(psize_long, 0.0, self.fills)
        pprice_long = calc_pprice_long(psize_long, long_pfills) if psize_long else 0.0
        if psize_long * pprice_long < self.min_cost:
            psize_long, pprice_long, long_pfills = 0.0, 0.0, []
        position = {
            "long": {
                "size": psize_long,
                "price": pprice_long,
                "liquidation_price": 0.0,
            },
            "short": {"size": 0.0, "price": 0.0, "liquidation_price": 0.0},
            "wallet_balance": balance[self.quote]["onhand"]
            + balance[self.coin]["onhand"] * pprice_long,
        }
        return position

    async def execute_orders(self, orders: [dict]) -> [dict]:
        if not orders:
            return []
        creations = []
        for order in sorted(orders, key=lambda x: calc_diff(x["price"], self.price)):
            creation = None
            try:
                creation = asyncio.create_task(self.execute_order(order))
                creations.append((order, creation))
            except Exception as e:
                print(f"error creating order {order} {e}")
                print_async_exception(creation)
                traceback.print_exc()
        results = []
        for creation in creations:
            result = None
            try:
                result = await creation[1]
                results.append(result)
            except Exception as e:
                print(f"error creating order {creation} {e}")
                print_async_exception(result)
                traceback.print_exc()
        return results

    async def execute_cancellations(self, orders: [dict]) -> [dict]:
        if not orders:
            return []
        cancellations = []
        for order in sorted(orders, key=lambda x: calc_diff(x["price"], self.price)):
            cancellation = None
            try:
                cancellation = asyncio.create_task(self.execute_cancellation(order))
                cancellations.append((order, cancellation))
            except Exception as e:
                print(f"error cancelling order {order} {e}")
                print_async_exception(cancellation)
                traceback.print_exc()
        results = []
        for cancellation in cancellations:
            result = None
            try:
                result = await cancellation[1]
                results.append(result)
            except Exception as e:
                print(f"error cancelling order {cancellation} {e}")
                print_async_exception(result)
                traceback.print_exc()
        return results

    async def execute_order(self, order: dict) -> dict:
        params = {
            "symbol": self.symbol,
            "side": order["side"].upper(),
            "type": order["type"].upper(),
            "quantity": format_float(order["qty"]),
        }
        if params["type"] == "LIMIT":
            params["timeInForce"] = "GTC"
            params["price"] = format_float(order["price"])
        if "custom_id" in order:
            params[
                "newClientOrderId"
            ] = f"{order['custom_id']}_{str(int(time() * 1000))[8:]}_{int(np.random.random() * 1000)}"
        o = await self.private_post(self.endpoints["create_order"], params)
        if "side" in o:
            return {
                "symbol": self.symbol,
                "side": o["side"].lower(),
                "position_side": "long",
                "type": o["type"].lower(),
                "qty": float(o["origQty"]),
                "order_id": int(o["orderId"]),
                "price": float(o["price"]),
            }
        else:
            return o

    async def execute_cancellation(self, order: dict) -> [dict]:
        cancellation = None
        try:
            cancellation = await self.private_delete(
                self.endpoints["cancel_order"],
                {"symbol": self.symbol, "orderId": order["order_id"]},
            )
            return {
                "symbol": self.symbol,
                "side": cancellation["side"].lower(),
                "position_side": "long",
                "order_id": int(cancellation["orderId"]),
                "qty": float(cancellation["origQty"]),
                "price": float(cancellation["price"]),
            }
        except Exception as e:
            print(f"error cancelling order {order} {e}")
            print_async_exception(cancellation)
            traceback.print_exc()
            self.ts_released["force_update"] = 0.0
            return {}

    async def get_all_fills(self, symbol: str = None, start_time: int = None):
        fills = []
        i = 0
        while True:
            i += 1
            if i >= 15:
                print("\nWarning: more than 15 calls to fetch_fills(), breaking")
                break
            fetched = await self.fetch_fills(symbol=symbol, start_time=start_time)
            print_(["fetched fills", ts_to_date(fetched[0]["timestamp"])])
            if fetched == fills[-len(fetched) :]:
                break
            fills += fetched
            if len(fetched) < 1000:
                break
            start_time = fills[-1]["timestamp"]
        fills_d = {e["id"]: e for e in fills}
        return sorted(fills_d.values(), key=lambda x: x["timestamp"])

    async def get_all_income(
        self,
        symbol: str = None,
        start_time: int = None,
        income_type: str = "realized_pnl",
        end_time: int = None,
    ):
        fills = await self.get_all_fills(symbol=symbol, start_time=start_time)

        income = []
        psize, pprice = 0.0, 0.0
        for fill in fills:
            if fill["side"] == "buy":
                new_psize = psize + fill["qty"]
                pprice = pprice * (psize / new_psize) + fill["price"] * (fill["qty"] / new_psize)
                psize = new_psize
            elif psize > 0.0:
                income.append(
                    {
                        "symbol": fill["symbol"],
                        "income_type": "realized_pnl",
                        "income": calc_pnl_long(pprice, fill["price"], fill["qty"], False, 1.0),
                        "token": self.quote,
                        "timestamp": fill["timestamp"],
                        "info": 0,
                        "transaction_id": fill["id"],
                        "trade_id": fill["id"],
                    }
                )
                psize = max(0.0, psize - fill["qty"])
        return income

    async def fetch_fills(
        self,
        symbol: str = None,
        limit: int = 1000,
        from_id: int = None,
        start_time: int = None,
        end_time: int = None,
    ):
        params = {
            "symbol": (self.symbol if symbol is None else symbol),
            "limit": min(1000, max(500, limit)),
        }
        if from_id is not None:
            params["fromId"] = max(0, from_id)
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(min(end_time, start_time + 1000 * 60 * 60 * 23.99))
        try:
            fetched = await self.private_get(self.endpoints["fills"], params)
            fills = [
                {
                    "symbol": x["symbol"],
                    "id": int(x["id"]),
                    "order_id": int(x["orderId"]),
                    "side": "buy" if x["isBuyer"] else "sell",
                    "price": float(x["price"]),
                    "qty": float(x["qty"]),
                    "realized_pnl": 0.0,
                    "cost": float(x["quoteQty"]),
                    "fee_paid": float(x["commission"]),
                    "fee_token": x["commissionAsset"],
                    "timestamp": (ts := int(x["time"])),
                    "position_side": "long",
                    "datetime": ts_to_date(ts),
                    "is_maker": x["isMaker"],
                }
                for x in fetched
            ]
        except Exception as e:
            print("error fetching fills a", e)
            traceback.print_exc()
            return []
        return fills

    async def fetch_latest_fills(self):
        params = {"symbol": self.symbol, "limit": 100}
        fetched = None
        try:
            fetched = await self.private_get(self.endpoints["fills_detailed"], params)
            fills = [
                {
                    "order_id": elm["orderId"],
                    "symbol": elm["symbol"],
                    "status": elm["status"].lower(),
                    "custom_id": elm["clientOrderId"],
                    "price": float(elm["price"]),
                    "qty": float(elm["executedQty"]),
                    "original_qty": float(elm["origQty"]),
                    "type": elm["type"].lower(),
                    "reduce_only": None,
                    "side": elm["side"].lower(),
                    "position_side": "long",
                    "timestamp": elm["time"],
                }
                for elm in fetched
                if "FILLED" in elm["status"]
            ]
        except Exception as e:
            print("error fetching latest fills", e)
            print_async_exception(fetched)
            traceback.print_exc()
            return []
        return fills

    async def fetch_income(
        self,
        symbol: str = None,
        limit: int = 1000,
        start_time: int = None,
        end_time: int = None,
    ):
        print("fetch income not implemented in spot")
        return []

    async def fetch_account(self):
        try:
            return await self.private_get(self.endpoints["balance"])
        except Exception as e:
            print("error fetching account: ", e)
            return {"balances": []}

    async def fetch_ticks(
        self,
        from_id: int = None,
        start_time: int = None,
        end_time: int = None,
        do_print: bool = True,
    ):
        params = {"symbol": self.symbol, "limit": 1000}
        if from_id is not None:
            params["fromId"] = max(0, from_id)
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time
        try:
            fetched = await self.public_get(self.endpoints["ticks"], params)
        except Exception as e:
            print("error fetching ticks a", e)
            return []
        try:
            ticks = [
                {
                    "trade_id": int(t["a"]),
                    "price": float(t["p"]),
                    "qty": float(t["q"]),
                    "timestamp": int(t["T"]),
                    "is_buyer_maker": t["m"],
                }
                for t in fetched
            ]
            if do_print:
                print_(
                    [
                        "fetched ticks",
                        self.symbol,
                        ticks[0]["trade_id"],
                        ts_to_date(float(ticks[0]["timestamp"]) / 1000),
                    ]
                )
        except Exception as e:
            print("error fetching ticks b", e, fetched)
            ticks = []
            if do_print:
                print_(["fetched no new ticks", self.symbol])
        return ticks

    async def fetch_ticks_time(self, start_time: int, end_time: int = None, do_print: bool = True):
        return await self.fetch_ticks(start_time=start_time, end_time=end_time, do_print=do_print)

    async def fetch_ohlcvs(
        self, symbol: str = None, start_time: int = None, interval="1m", limit=1000
    ):
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
        params = {
            "symbol": self.symbol if symbol is None else symbol,
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
            params["endTime"] = params["startTime"] + interval_map[interval] * 60 * 1000 * limit
        try:
            fetched = await self.public_get(self.endpoints["ohlcvs"], params)
            return [
                {
                    **{"timestamp": int(e[0])},
                    **{
                        k: float(e[i + 1])
                        for i, k in enumerate(["open", "high", "low", "close", "volume"])
                    },
                }
                for e in fetched
            ]
        except Exception as e:
            print("error fetching ohlcvs", fetched, e)
            traceback.print_exc()

    async def transfer(self, type_: str, amount: float, asset: str = "USDT"):
        print("transfer not implemented in spot")
        return

    def standardize_market_stream_event(self, data: dict) -> [dict]:
        try:
            return [
                {
                    "timestamp": int(data["T"]),
                    "price": float(data["p"]),
                    "qty": float(data["q"]),
                    "is_buyer_maker": data["m"],
                }
            ]
        except Exception as e:
            print("error in websocket tick", e)
        return []

    async def beat_heart_user_stream(self) -> None:
        while True:
            await asyncio.sleep(60 + np.random.randint(60 * 9, 60 * 14))
            await self.init_user_stream()

    async def init_user_stream(self) -> None:
        try:
            response = await self.post_listen_key()
            self.listen_key = response["listenKey"]
            self.endpoints["websocket_user"] = self.endpoints["websocket"] + self.listen_key
        except Exception as e:
            traceback.print_exc()
            print_(["error fetching listen key", e])

    async def on_user_stream_event(self, event: dict) -> None:
        try:
            pos_change = False
            if "balance" in event:
                onhand_change = False
                for token in event["balance"]:
                    self.balance[token]["free"] = event["balance"][token]["free"]
                    self.balance[token]["locked"] = event["balance"][token]["locked"]
                    onhand = event["balance"][token]["free"] + event["balance"][token]["locked"]
                    if token in [self.quote, self.coin] and (
                        "onhand" not in self.balance[token] or self.balance[token]["onhand"] != onhand
                    ):
                        onhand_change = True
                    if token == "BNB":
                        onhand = max(0.0, onhand - 0.01)
                    self.balance[token]["onhand"] = onhand
                if onhand_change:
                    self.position = self.calc_simulated_position(self.balance, self.fills)
                    self.position["wallet_balance"] = self.adjust_wallet_balance(
                        self.position["wallet_balance"]
                    )
                    self.position = self.add_wallet_exposures_to_pos(self.position)
                    pos_change = True
            if "filled" in event:
                if event["filled"]["order_id"] not in {fill["order_id"] for fill in self.fills}:
                    self.fills = sorted(self.fills + [event["filled"]], key=lambda x: x["order_id"])
                self.position = self.calc_simulated_position(self.balance, self.fills)
                self.position["wallet_balance"] = self.adjust_wallet_balance(
                    self.position["wallet_balance"]
                )
                self.position = self.add_wallet_exposures_to_pos(self.position)
                pos_change = True
            elif "partially_filled" in event:
                await asyncio.sleep(0.01)
                await asyncio.gather(self.update_position(), self.update_open_orders())
                pos_change = True
            if "new_open_order" in event:
                if event["new_open_order"]["order_id"] not in {
                    x["order_id"] for x in self.open_orders
                }:
                    self.open_orders.append(event["new_open_order"])
            elif "deleted_order_id" in event:
                for i, o in enumerate(self.open_orders):
                    if o["order_id"] == event["deleted_order_id"]:
                        self.open_orders = self.open_orders[:i] + self.open_orders[i + 1 :]
                        break
            if pos_change:
                self.position["equity"] = self.position["wallet_balance"] + calc_upnl(
                    self.position["long"]["size"],
                    self.position["long"]["price"],
                    self.position["short"]["size"],
                    self.position["short"]["price"],
                    self.price,
                    self.inverse,
                    self.c_mult,
                )
                await asyncio.sleep(
                    0.01
                )  # sleep 10 ms to catch both pos update and open orders update
                await self.cancel_and_create()
        except Exception as e:
            print(["error handling user stream event", e])
            traceback.print_exc()

    def standardize_user_stream_event(self, event: dict) -> dict:
        standardized = {}
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
                    if event["s"] == self.symbol:
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
                    if event["s"] == self.symbol:
                        standardized["deleted_order_id"] = int(event["i"])
                    else:
                        standardized["other_symbol"] = event["s"]
                        standardized["other_type"] = event["X"].lower()
                elif event["X"] == "FILLED":
                    if event["s"] == self.symbol:
                        price = fp if (fp := float(event["p"])) != 0.0 else float(event["L"])
                        standardized["filled"] = {
                            "order_id": int(event["i"]),
                            "symbol": event["s"],
                            "price": price,
                            "qty": float(event["q"]),
                            "type": event["o"].lower(),
                            "side": event["S"].lower(),
                            "position_side": "long",
                            "timestamp": int(event["T"]),
                        }
                        standardized["deleted_order_id"] = standardized["filled"]["order_id"]
                    else:
                        standardized["other_symbol"] = event["s"]
                        standardized["other_type"] = "filled"
                elif event["X"] == "PARTIALLY_FILLED":
                    if event["s"] == self.symbol:
                        price = fp if (fp := float(event["p"])) != 0.0 else float(event["L"])
                        standardized["partially_filled"] = {
                            "order_id": int(event["i"]),
                            "symbol": event["s"],
                            "price": price,
                            "qty": float(event["q"]),
                            "type": event["o"].lower(),
                            "side": event["S"].lower(),
                            "position_side": "long",
                            "timestamp": int(event["T"]),
                        }
                        standardized["deleted_order_id"] = standardized["partially_filled"][
                            "order_id"
                        ]
                    else:
                        standardized["other_symbol"] = event["s"]
                        standardized["other_type"] = "partially_filled"

        return standardized
