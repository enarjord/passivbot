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
    calc_diff,
    round_dn,
    round_up,
    qty_to_cost,
    calc_min_entry_qty,
    calc_pnl_long,
)
from passivbot import Bot
from procedures import print_async_exception, print_
from pure_funcs import (
    sort_dict_keys,
    spotify_config,
    get_position_fills,
    calc_pprice_long,
    format_float,
    ts_to_date,
)


class BybitBotSpot(Bot):
    def __init__(self, config: dict):
        self.exchange = "bybit_spot"
        self.balance = {}
        super().__init__(spotify_config(config))
        self.spot = self.config["spot"] = True
        self.inverse = self.config["inverse"] = False
        self.hedge_mode = self.config["hedge_mode"] = False
        self.do_short = self.config["do_short"] = self.config["short"]["enabled"] = False
        self.session = aiohttp.ClientSession(
            headers=({"referer": self.broker_code} if self.broker_code else {}),
            connector=aiohttp.TCPConnector(resolver=aiohttp.AsyncResolver()),
        )
        self.base_endpoint = "https://api.bybit.com"
        self.force_update_interval = 40
        self.max_n_orders_per_batch = 5
        self.max_n_cancellations_per_batch = 10
        self.endpoints = None

    async def public_get(self, url: str, params=None) -> dict:
        if params is None:
            params = {}
        async with self.session.get(self.base_endpoint + url, params=params) as response:
            result = await response.text()
        return json.loads(result)

    async def private_(
        self,
        type_: str,
        base_endpoint: str,
        url: str,
        params: dict = None,
        json_: bool = False,
    ) -> dict:
        if params is None:
            params = {}
        timestamp = int(time() * 1000)
        params.update({"api_key": self.key, "timestamp": timestamp, "recv_window": 10000})
        for k in params:
            if type(params[k]) == bool:
                params[k] = "true" if params[k] else "false"
            elif type(params[k]) == float:
                params[k] = str(params[k])
        params["sign"] = hmac.new(
            self.secret.encode("utf-8"),
            urlencode(sort_dict_keys(params)).encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if json_:
            async with getattr(self.session, type_)(base_endpoint + url, json=params) as response:
                result = await response.text()
        else:
            async with getattr(self.session, type_)(base_endpoint + url, params=params) as response:
                result = await response.text()
        result_dict = json.loads(result)
        return result_dict

    async def private_get(self, url: str, params=None, base_endpoint: str = None) -> dict:
        if params is None:
            params = {}
        return await self.private_(
            "get",
            self.base_endpoint if base_endpoint is None else base_endpoint,
            url,
            params,
        )

    async def private_post(self, url: str, params=None, base_endpoint: str = None) -> dict:
        if params is None:
            params = {}
        return await self.private_(
            "post",
            self.base_endpoint if base_endpoint is None else base_endpoint,
            url,
            params,
        )

    def init_market_type(self):
        print("Spot market")
        if "spot" not in self.market_type:
            self.market_type += "_spot"
        self.inverse = self.config["inverse"] = False
        self.spot = True
        self.hedge_mode = False
        self.endpoints = {
            "balance": "/spot/v3/private/account",
            "exchange_info": "/spot/v3/public/symbols",
            "open_orders": "/spot/v3/private/open-orders",
            "ticker": "/spot/v3/public/quote/ticker/bookTicker",
            "fills": "/spot/v3/private/my-trades",
            "fills_detailed": "/spot/v3/private/my-trades",
            "create_order": "/spot/v3/private/order",
            "cancel_order": "/spot/v3/private/cancel-order",
            "ticks": "/spot/v3/public/quote/trades",
            "ohlcvs": "/spot/v3/public/quote/kline",
        }
        self.endpoints["transfer"] = ""
        self.endpoints["account"] = "/spot/v3/private/account"

    async def _init(self):
        self.init_market_type()
        "here"
        exchange_info = await self.public_get(self.endpoints["exchange_info"])
        for e in exchange_info["result"]["list"]:
            if e["name"] == self.symbol:
                self.coin = e["baseCoin"]
                self.quote = self.margin_coin = e["quoteCoin"]
                self.min_qty = self.config["min_qty"] = float(e["minTradeQty"])
                self.qty_step = self.config["qty_step"] = float(e["basePrecision"])
                self.price_step = self.config["price_step"] = float(e["minPricePrecision"])
                self.min_price = 0  # TODO Fix
                self.max_price = 1000000000  # TODO Fix
                self.min_cost = self.config["min_cost"] = 0.0
                self.price_multiplier_up = 5
                self.price_multiplier_dn = 0.2

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

    async def execute_leverage_change(self):
        pass

    async def init_exchange_config(self):
        await self.check_if_other_positions()

    async def init_order_book(self):
        ticker = None
        try:
            ticker = await self.public_get(self.endpoints["ticker"], {"symbol": self.symbol})
            ticker = ticker["result"]
            self.ob = [float(ticker["bidPrice"]), float(ticker["askPrice"])]
            self.price = np.random.choice(self.ob)
            return True
        except Exception as e:
            print(f"{self.symbol} error updating order book {e}")
            print_async_exception(ticker)
            return False

    async def fetch_open_orders(self) -> [dict]:
        try:
            open_orders = await self.private_get(
                self.endpoints["open_orders"], {"symbol": self.symbol}
            )
            open_orders = open_orders["result"]["list"]
        except KeyError:
            open_orders = []
        return [
            {
                "order_id": int(e["orderId"]),
                "custom_id": e["orderLinkId"],
                "symbol": e["symbol"],
                "price": float(e["orderPrice"]),
                "qty": float(e["orderQty"]),
                "type": e["orderType"].lower(),  # TODO
                "side": e["side"].lower(),
                "position_side": "long",
                "timestamp": int(e["createTime"]),
            }
            for e in open_orders
        ]

    async def fetch_position(self) -> dict:
        balances, _ = await asyncio.gather(
            self.private_get(self.endpoints["balance"]), self.update_fills()
        )
        balances = balances["result"]["balances"]
        balance = {}
        for elm in balances:
            balance[elm["coin"]] = {"free": float(elm["free"])}
            balance[elm["coin"]]["locked"] = float(elm["locked"])
            balance[elm["coin"]]["onhand"] = (
                balance[elm["coin"]]["free"] + balance[elm["coin"]]["locked"]
            )
        self.balance = balance
        return self.calc_simulated_position(self.balance, self.fills)

    def calc_simulated_position(self, balance: dict, long_fills: [dict]) -> dict:
        """
        balance = {'BTC': {'free': float, 'locked': float, 'onhand': float}, ...}
        long_pfills = [{order...}, ...]
        """
        if self.coin in balance:
            psize_long = round_dn(balance[self.coin]["onhand"], self.qty_step)
            long_pfills, short_pfills = get_position_fills(psize_long, 0.0, self.fills)
            pprice_long = calc_pprice_long(psize_long, long_pfills) if psize_long else 0.0
            if psize_long * pprice_long < self.min_cost:
                psize_long, pprice_long, long_pfills = 0.0, 0.0, []
            wallet_balance = (
                balance[self.quote]["onhand"] + balance[self.coin]["onhand"] * pprice_long
            )
        else:
            psize_long = 0.0
            pprice_long = 0.0
            wallet_balance = balance[self.quote]["onhand"]

        position = {
            "long": {
                "size": psize_long,
                "price": pprice_long,
                "liquidation_price": 0.0,
            },
            "short": {"size": 0.0, "price": 0.0, "liquidation_price": 0.0},
            "wallet_balance": wallet_balance,
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
            "orderType": order["type"].upper(),
            "orderQty": format_float(order["qty"]),
        }
        if params["orderType"] == "LIMIT":
            params["timeInForce"] = "GTC"
            params["orderPrice"] = format_float(order["price"])
        if "custom_id" in order:
            params[
                "orderLinkId"
            ] = f"{order['custom_id']}_{str(int(time() * 1000))[8:]}_{int(np.random.random() * 1000)}"
        o = await self.private_post(self.endpoints["create_order"], params)
        o = o["result"]
        if "side" in o:
            return {
                "symbol": self.symbol,
                "side": o["side"].lower(),
                "position_side": "long",
                "type": o["orderType"].lower(),
                "qty": float(o["orderQty"]),
                "order_id": int(o["orderId"]),
                "price": float(o["orderPrice"]),
            }
        else:
            return o

    async def execute_cancellation(self, order: dict) -> [dict]:
        cancellation = None
        try:
            cancellation = await self.private_post(
                self.endpoints["cancel_order"],
                {"symbol": self.symbol, "orderId": order["order_id"]},
            )
            cancellation = cancellation["result"]
            return {
                "symbol": self.symbol,
                "side": cancellation["side"].lower(),
                "position_side": "long",
                "order_id": int(cancellation["orderId"]),
                "qty": float(cancellation["orderQty"]),
                "price": float(cancellation["orderPrice"]),
            }
        except Exception as e:
            print(f"error cancelling order {order} {e}")
            print_async_exception(cancellation)
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
        # income_type: str = "realized_pnl",
        # end_time: int = None,
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
            fetched = fetched["result"]["list"]
            fills = [
                {
                    "symbol": x["symbol"],
                    "id": int(x["id"]),
                    "order_id": int(x["orderId"]),
                    "side": "buy" if x["isBuyer"] == "0" else "sell",
                    "price": float(x["orderPrice"]),
                    "qty": float(x["orderQty"]),
                    "realized_pnl": 0.0,
                    "cost": float(x["orderQty"]) * float(x["orderPrice"]),
                    "fee_paid": float(x["execFee"]),
                    "fee_token": x["feeTokenId"],
                    "timestamp": (ts := int(x["executionTime"])),
                    "position_side": "long",
                    "datetime": ts_to_date(ts),
                    "is_maker": x["isMaker"],
                }
                for x in fetched
            ]
        except Exception as e:
            print(f"error fetching fills a: {e}")
            return []
        return fills

    async def fetch_latest_fills(self):
        params = {"symbol": self.symbol, "limit": 100}
        fetched = None
        fills = []
        try:
            fetched = await self.private_get(self.endpoints["fills_detailed"], params)
            fills = [
                {
                    "order_id": elm["orderId"],
                    "symbol": elm["symbol"],
                    "status": "filled",
                    "custom_id": elm["orderId"],
                    "price": elm["orderPrice"],
                    "qty": float(elm["orderQty"]),
                    "original_qty": float(elm["orderQty"]),
                    "type": "limit",
                    "reduce_only": None,
                    "side": "buy" if elm["isBuyer"] == "0" else "sell",
                    "position_side": "long",
                    "timestamp": elm["creatTime"],
                }
                for elm in fetched["result"]["list"]
            ]
        except Exception as e:
            print("error fetching latest fills", e)
            print_async_exception(fetched)
            traceback.print_exc()
        return fills

    async def fetch_income(
        self,
        # symbol: str = None,
        # limit: int = 1000,
        # start_time: int = None,
        # end_time: int = None,
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
                    "trade_id": int(t["time"]),
                    "price": float(t["price"]),
                    "qty": float(t["qty"]),
                    "timestamp": int(t["time"]),
                    "is_buyer_maker": t["isBuyerMaker"],
                }
                for t in fetched["results"]["list"]
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
                    **{"timestamp": int(e["t"])},
                    **{"open": float(e["o"])},
                    **{"high": float(e["h"])},
                    **{"low": float(e["l"])},
                    **{"close": float(e["c"])},
                    **{"volume": float(e["v"])},
                }
                for e in fetched["result"]["list"]
            ]
        except Exception as e:
            traceback.print_exc()
            print(f"error fetching ohlcvs: {e}")

    async def transfer(self, type_: str, amount: float, asset: str = "USDT"):
        print("transfer not implemented in spot")
        return

    def standardize_market_stream_event(self, data: dict) -> [dict]:
        raise NotImplementedError
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
        raise NotImplementedError
        while True:
            await asyncio.sleep(60 + np.random.randint(60 * 9, 60 * 14))
            await self.init_user_stream()

    async def init_user_stream(self) -> None:
        raise NotImplementedError

    async def on_user_stream_event(self, event: dict) -> None:
        raise NotImplementedError

    def standardize_user_stream_event(self, event: dict) -> dict:
        raise NotImplementedError
