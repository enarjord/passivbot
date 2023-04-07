import asyncio
import hashlib
import hmac
import json
import traceback
from time import time
from typing import Union, List, Dict
from urllib.parse import urlencode
from uuid import uuid4

import aiohttp
import numpy as np

from njit_funcs import round_, calc_diff
from passivbot import Bot, logging
from procedures import print_async_exception, print_
from pure_funcs import ts_to_date, sort_dict_keys, date_to_ts


def first_capitalized(s: str):
    return s[0].upper() + s[1:].lower()


def determine_pos_side(o: dict) -> str:
    if o["side"].lower() == "buy":
        if "reduce_only" in o:
            if o["reduce_only"]:
                return "short"
            else:
                return "long"
        if "closed_size" in o:
            if o["closed_size"] != 0.0:
                return "short"
            else:
                return "long"
        if "entry" in o["order_link_id"]:
            return "long"
        elif "close" in o["order_link_id"]:
            return "short"
        else:
            return "both"
    else:
        if "reduce_only" in o:
            if o["reduce_only"]:
                return "long"
            else:
                return "short"
        if "closed_size" in o:
            if o["closed_size"] != 0.0:
                return "long"
            else:
                return "short"
        if "entry" in o["order_link_id"]:
            return "short"
        elif "close" in o["order_link_id"]:
            return "long"
        else:
            return "both"


class BybitBot(Bot):
    def __init__(self, config: dict):
        self.exchange = "bybit"
        self.min_notional = 1.0
        self.max_n_orders_per_batch = 5
        self.max_n_cancellations_per_batch = 10
        super().__init__(config)
        self.base_endpoint = "https://api.bybit.com"
        if self.test_mode:
            self.base_endpoint = "https://api-testnet.bybit.com"
        self.endpoints = {
            "balance": "/v2/private/wallet/balance",
            "exchange_info": "/v2/public/symbols",
            "ticker": "/v2/public/tickers",
            "funds_transfer": "/asset/v1/private/transfer",
        }
        self.session = aiohttp.ClientSession(
            headers=({"referer": self.broker_code} if self.broker_code else {})
        )

    def init_market_type(self):
        websockets_base_endpoint = "wss://stream.bybit.com"
        if self.test_mode:
            websockets_base_endpoint = "wss://stream-testnet.bybit.com"

        if self.symbol.endswith("USDT"):
            print("linear perpetual")
            self.market_type += "_linear_perpetual"
            self.inverse = self.config["inverse"] = False
            self.endpoints = {
                "position": "/private/linear/position/list",
                "open_orders": "/private/linear/order/search",
                "create_order": "/private/linear/order/create",
                "cancel_order": "/private/linear/order/cancel",
                "ticks": "/public/linear/recent-trading-records",
                "fills": "/private/linear/trade/execution/list",
                "ohlcvs": "/public/linear/kline",
                "websocket_market": f"{websockets_base_endpoint}/realtime_public",
                "websocket_user": f"{websockets_base_endpoint}/realtime_private",
                "income": "/private/linear/trade/closed-pnl/list",
                "created_at_key": "created_time",
            }

        else:
            self.inverse = self.config["inverse"] = True
            if self.symbol.endswith("USD"):
                print("inverse perpetual")
                self.market_type += "_inverse_perpetual"
                self.endpoints = {
                    "position": "/v2/private/position/list",
                    "open_orders": "/v2/private/order",
                    "create_order": "/v2/private/order/create",
                    "cancel_order": "/v2/private/order/cancel",
                    "ticks": "/v2/public/trading-records",
                    "fills": "/v2/private/execution/list",
                    "ohlcvs": "/v2/public/kline/list",
                    "websocket_market": f"{websockets_base_endpoint}/realtime",
                    "websocket_user": f"{websockets_base_endpoint}/realtime",
                    "income": "/v2/private/trade/closed-pnl/list",
                    "created_at_key": "created_at",
                }

                self.hedge_mode = self.config["hedge_mode"] = False
            else:
                print("inverse futures")
                self.market_type += "_inverse_futures"
                self.endpoints = {
                    "position": "/futures/private/position/list",
                    "open_orders": "/futures/private/order",
                    "create_order": "/futures/private/order/create",
                    "cancel_order": "/futures/private/order/cancel",
                    "ticks": "/v2/public/trading-records",
                    "fills": "/futures/private/execution/list",
                    "ohlcvs": "/v2/public/kline/list",
                    "websocket_market": f"{websockets_base_endpoint}/realtime",
                    "websocket_user": f"{websockets_base_endpoint}/realtime",
                    "income": "/futures/private/trade/closed-pnl/list",
                    "created_at_key": "created_at",
                }

        self.spot_base_endpoint = "https://api.bybit.com"
        if self.test_mode:
            self.spot_base_endpoint = "https://api-testnet.bybit.com"

        self.endpoints["spot_balance"] = "/spot/v1/account"
        self.endpoints["balance"] = "/v2/private/wallet/balance"
        self.endpoints["exchange_info"] = "/v2/public/symbols"
        self.endpoints["ticker"] = "/v2/public/tickers"
        self.endpoints["funds_transfer"] = "/asset/v1/private/transfer"

    async def _init(self):
        info = await self.public_get(self.endpoints["exchange_info"])
        for e in info["result"]:
            if e["name"] == self.symbol:
                break
        else:
            raise Exception(f"symbol missing {self.symbol}")
        self.max_leverage = e["leverage_filter"]["max_leverage"]
        self.coin = e["base_currency"]
        self.quote = e["quote_currency"]
        self.price_step = self.config["price_step"] = float(e["price_filter"]["tick_size"])
        self.qty_step = self.config["qty_step"] = float(e["lot_size_filter"]["qty_step"])
        self.min_qty = self.config["min_qty"] = float(e["lot_size_filter"]["min_trading_qty"])
        self.min_cost = self.config["min_cost"] = 1.0
        self.init_market_type()
        self.margin_coin = self.coin if self.inverse else self.quote
        await super()._init()
        await self.init_order_book()
        await self.update_position()

    async def init_order_book(self):
        ticker = None
        try:
            ticker = await self.private_get(self.endpoints["ticker"], {"symbol": self.symbol})
            self.ob = [
                float(ticker["result"][0]["bid_price"]),
                float(ticker["result"][0]["ask_price"]),
            ]
            self.price = float(ticker["result"][0]["last_price"])
            return True
        except Exception as e:
            logging.error(f"error updating order book {e}")
            print_async_exception(ticker)
            return False

    async def fetch_open_orders(self) -> [dict]:
        fetched = await self.private_get(self.endpoints["open_orders"], {"symbol": self.symbol})
        return [
            {
                "order_id": elm["order_id"],
                "custom_id": elm["order_link_id"],
                "symbol": elm["symbol"],
                "price": float(elm["price"]),
                "qty": float(elm["qty"]),
                "side": elm["side"].lower(),
                "position_side": determine_pos_side(elm),
                "timestamp": date_to_ts(elm[self.endpoints["created_at_key"]]),
            }
            for elm in fetched["result"]
        ]

    async def public_get(self, url: str, params: dict = {}) -> dict:
        async with self.session.get(self.base_endpoint + url, params=params) as response:
            result = await response.text()
        return json.loads(result)

    async def private_(
        self, type_: str, base_endpoint: str, url: str, params: dict = {}, json_: bool = False
    ) -> dict:
        timestamp = int(time() * 1000)
        params.update({"api_key": self.key, "timestamp": timestamp})
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
        return json.loads(result)

    async def private_get(self, url: str, params: dict = {}, base_endpoint: str = None) -> dict:
        return await self.private_(
            "get",
            self.base_endpoint if base_endpoint is None else base_endpoint,
            url,
            params,
        )

    async def private_post(self, url: str, params: dict = {}, base_endpoint: str = None) -> dict:
        return await self.private_(
            "post",
            self.base_endpoint if base_endpoint is None else base_endpoint,
            url,
            params,
        )

    async def transfer_from_derivatives_to_spot(self, coin: str, amount: float):
        params = {
            "coin": coin,
            "amount": str(amount),
            "from_account_type": "CONTRACT",
            "to_account_type": "SPOT",
            "transfer_id": str(uuid4()),
        }
        return await self.private_(
            "post", self.base_endpoint, self.endpoints["funds_transfer"], params=params, json_=True
        )

    async def get_server_time(self):
        now = await self.public_get("/v2/public/time")
        return float(now["time_now"]) * 1000

    async def fetch_position(self) -> dict:
        position = {}
        fetched, bal = None, None
        try:
            if "linear_perpetual" in self.market_type:
                fetched, bal = await asyncio.gather(
                    self.private_get(self.endpoints["position"], {"symbol": self.symbol}),
                    self.private_get(self.endpoints["balance"], {"coin": self.quote}),
                )
                long_pos = [e for e in fetched["result"] if e["side"] == "Buy"][0]
                short_pos = [e for e in fetched["result"] if e["side"] == "Sell"][0]
                position["wallet_balance"] = float(bal["result"][self.quote]["wallet_balance"])
            else:
                fetched, bal = await asyncio.gather(
                    self.private_get(self.endpoints["position"], {"symbol": self.symbol}),
                    self.private_get(self.endpoints["balance"], {"coin": self.coin}),
                )
                position["wallet_balance"] = float(bal["result"][self.coin]["wallet_balance"])
                if "inverse_perpetual" in self.market_type:
                    if fetched["result"]["side"] == "Buy":
                        long_pos = fetched["result"]
                        short_pos = {"size": 0.0, "entry_price": 0.0, "liq_price": 0.0}
                    else:
                        long_pos = {"size": 0.0, "entry_price": 0.0, "liq_price": 0.0}
                        short_pos = fetched["result"]
                elif "inverse_futures" in self.market_type:
                    long_pos = [
                        e["data"] for e in fetched["result"] if e["data"]["position_idx"] == 1
                    ][0]
                    short_pos = [
                        e["data"] for e in fetched["result"] if e["data"]["position_idx"] == 2
                    ][0]
                else:
                    raise Exception("unknown market type")

            position["long"] = {
                "size": round_(float(long_pos["size"]), self.qty_step),
                "price": float(long_pos["entry_price"]),
                "liquidation_price": float(long_pos["liq_price"]),
            }
            position["short"] = {
                "size": -round_(float(short_pos["size"]), self.qty_step),
                "price": float(short_pos["entry_price"]),
                "liquidation_price": float(short_pos["liq_price"]),
            }
            return position
        except Exception as e:
            logging.error(f"error fetching pos or balance {e}")
            print_async_exception(fetched)
            print_async_exception(bal)
            traceback.print_exc()
            return None

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

    async def execute_order(self, order: dict) -> dict:
        o = None
        try:
            params = {
                "symbol": self.symbol,
                "side": first_capitalized(order["side"]),
                "order_type": first_capitalized(order["type"]),
                "qty": float(order["qty"])
                if "linear_perpetual" in self.market_type
                else int(order["qty"]),
                "close_on_trigger": False,
            }
            if self.hedge_mode:
                params["position_idx"] = 1 if order["position_side"] == "long" else 2
                if "linear_perpetual" in self.market_type:
                    params["reduce_only"] = "close" in order["custom_id"]
            else:
                params["position_idx"] = 0
                params["reduce_only"] = "close" in order["custom_id"]
            if params["order_type"] == "Limit":
                params["time_in_force"] = "PostOnly"
                params["price"] = str(order["price"])
            else:
                params["time_in_force"] = "GoodTillCancel"
            params[
                "order_link_id"
            ] = f"{order['custom_id']}_{str(int(time() * 1000))[8:]}_{int(np.random.random() * 1000)}"
            o = await self.private_post(self.endpoints["create_order"], params)
            if o["result"]:
                return {
                    "symbol": o["result"]["symbol"],
                    "side": o["result"]["side"].lower(),
                    "order_id": o["result"]["order_id"],
                    "position_side": order["position_side"],
                    "type": o["result"]["order_type"].lower(),
                    "qty": o["result"]["qty"],
                    "price": o["result"]["price"],
                }
            else:
                return o, order
        except Exception as e:
            print(f"error executing order {order} {e}")
            print_async_exception(o)
            traceback.print_exc()
            return {}

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

    async def execute_cancellation(self, order: dict) -> dict:
        cancellation = None
        try:
            cancellation = await self.private_post(
                self.endpoints["cancel_order"],
                {"symbol": self.symbol, "order_id": order["order_id"]},
            )
            return {
                "symbol": self.symbol,
                "side": order["side"],
                "order_id": cancellation["result"]["order_id"],
                "position_side": order["position_side"],
                "qty": order["qty"],
                "price": order["price"],
            }
        except Exception as e:
            if (
                cancellation is not None
                and "ret_code" in cancellation
                and cancellation["ret_code"] == 20001
            ):
                error_cropped = {
                    k: v for k, v in cancellation.items() if k in ["ret_msg", "ret_code"]
                }
                logging.error(
                    f"error cancelling order {error_cropped} {order}"
                )  # neater error message
            else:
                print(f"error cancelling order {order} {e}")
                print_async_exception(cancellation)
                traceback.print_exc()
            self.ts_released["force_update"] = 0.0
            return {}

    async def fetch_account(self):
        try:
            resp = await self.private_get(
                self.endpoints["spot_balance"], base_endpoint=self.spot_base_endpoint
            )
            return resp["result"]
        except Exception as e:
            print("error fetching account: ", e)
            return {"balances": []}

    async def fetch_ticks(self, from_id: int = None, do_print: bool = True):
        params = {"symbol": self.symbol, "limit": 1000}
        if from_id is not None:
            params["from"] = max(0, from_id)
        try:
            ticks = await self.public_get(self.endpoints["ticks"], params)
        except Exception as e:
            print("error fetching ticks", e)
            return []
        try:
            trades = [
                {
                    "trade_id": int(tick["id"]),
                    "price": float(tick["price"]),
                    "qty": float(tick["qty"]),
                    "timestamp": date_to_ts(tick["time"]),
                    "is_buyer_maker": tick["side"] == "Sell",
                }
                for tick in ticks["result"]
            ]
            if do_print:
                print_(
                    [
                        "fetched trades",
                        self.symbol,
                        trades[0]["trade_id"],
                        ts_to_date(float(trades[0]["timestamp"]) / 1000),
                    ]
                )
        except:
            trades = []
            if do_print:
                print_(["fetched no new trades", self.symbol])
        return trades

    async def fetch_ohlcvs(
        self, symbol: str = None, start_time: int = None, interval="1m", limit=200
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
            "1d": "D",
            "1w": "W",
            "1M": "M",
        }
        assert interval in interval_map
        params = {
            "symbol": self.symbol if symbol is None else symbol,
            "interval": interval_map[interval],
            "limit": limit,
        }
        if start_time is None:
            server_time = await self.public_get("/v2/public/time")
            if type(interval_map[interval]) == str:
                minutes = {"D": 1, "W": 7, "M": 30}[interval_map[interval]] * 60 * 24
            else:
                minutes = interval_map[interval]
            params["from"] = int(round(float(server_time["time_now"]))) - 60 * minutes * limit
        else:
            params["from"] = int(start_time / 1000)
        fetched = await self.public_get(self.endpoints["ohlcvs"], params)
        return [
            {
                **{"timestamp": e["open_time"] * 1000},
                **{k: float(e[k]) for k in ["open", "high", "low", "close", "volume"]},
            }
            for e in fetched["result"]
        ]

    async def get_all_income(
        self,
        symbol: str = None,
        start_time: int = None,
        income_type: str = "Trade",
        end_time: int = None,
    ):
        if symbol is None:
            all_income = []
            all_positions = await self.private_get(self.endpoints["position"], params={"symbol": ""})
            symbols = sorted(
                set(
                    [
                        x["data"]["symbol"]
                        for x in all_positions["result"]
                        if float(x["data"]["size"]) > 0
                    ]
                )
            )
            for symbol in symbols:
                all_income += await self.get_all_income(
                    symbol=symbol, start_time=start_time, income_type=income_type, end_time=end_time
                )
            return sorted(all_income, key=lambda x: x["timestamp"])
        limit = 50
        income = []
        page = 1
        while True:
            fetched = await self.fetch_income(
                symbol=symbol,
                start_time=start_time,
                income_type=income_type,
                limit=limit,
                page=page,
            )
            if len(fetched) == 0:
                break
            print_(["fetched income", symbol, ts_to_date(fetched[0]["timestamp"])])
            if fetched == income[-len(fetched) :]:
                break
            income += fetched
            if len(fetched) < limit:
                break
            page += 1
        income_d = {e["transaction_id"]: e for e in income}
        return sorted(income_d.values(), key=lambda x: x["timestamp"])

    async def fetch_income(
        self,
        symbol: str = None,
        income_type: str = None,
        limit: int = 50,
        start_time: int = None,
        end_time: int = None,
        page=None,
    ):
        params = {"limit": limit, "symbol": self.symbol if symbol is None else symbol}
        if start_time is not None:
            params["start_time"] = int(start_time / 1000)
        if end_time is not None:
            params["end_time"] = int(end_time / 1000)
        if income_type is not None:
            params["exec_type"] = first_capitalized(income_type)
        if page is not None:
            params["page"] = page
        fetched = None
        try:
            fetched = await self.private_get(self.endpoints["income"], params)
            if fetched["result"]["data"] is None:
                return []
            return sorted(
                [
                    {
                        "symbol": e["symbol"],
                        "income_type": e["exec_type"].lower(),
                        "income": float(e["closed_pnl"]),
                        "token": self.margin_coin,
                        "timestamp": float(e["created_at"]) * 1000,
                        "info": {"page": fetched["result"]["current_page"]},
                        "transaction_id": float(e["id"]),
                        "trade_id": e["order_id"],
                    }
                    for e in fetched["result"]["data"]
                ],
                key=lambda x: x["timestamp"],
            )
        except Exception as e:
            print("error fetching income: ", e)
            traceback.print_exc()
            print_async_exception(fetched)
            return []

    async def fetch_latest_fills(self):
        fetched = None
        try:
            fetched = await self.private_get(
                self.endpoints["fills"],
                {
                    "symbol": self.symbol,
                    "limit": 200,
                    "start_time": int((time() - 60 * 60 * 24) * 1000),
                },
            )
            if "inverse_perpetual" in self.market_type:
                fetched_data = fetched["result"]["trade_list"]
            elif "linear_perpetual" in self.market_type:
                fetched_data = fetched["result"]["data"]
            if fetched_data is None and fetched["ret_code"] == 0 and fetched["ret_msg"] == "OK":
                return []
            fills = [
                {
                    "order_id": elm["order_id"],
                    "symbol": elm["symbol"],
                    "status": elm["exec_type"].lower(),
                    "custom_id": elm["order_link_id"],
                    "price": float(elm["exec_price"]),
                    "qty": float(elm["exec_qty"]),
                    "original_qty": float(elm["order_qty"]),
                    "type": elm["order_type"].lower(),
                    "reduce_only": None,
                    "side": elm["side"].lower(),
                    "position_side": determine_pos_side(elm),
                    "timestamp": elm["trade_time_ms"],
                }
                for elm in fetched_data
                if elm["exec_type"] == "Trade"
            ]
        except Exception as e:
            print("error fetching latest fills", e)
            print_async_exception(fetched)
            traceback.print_exc()
            return []
        return fills

    async def fetch_fills(
        self,
        limit: int = 200,
        from_id: int = None,
        start_time: int = None,
        end_time: int = None,
    ):
        fetched = None
        try:
            if start_time is None:
                start_time = int((time() - 60 * 60 * 24) * 1000)
            if end_time is None:
                end_time = int((time() + 60 * 60 * 2) * 1000)
            fills = []
            page = 1
            last_order = None
            while True:
                fetched = await self.private_get(
                    self.endpoints["fills"],
                    {
                        "symbol": self.symbol,
                        "limit": 200,
                        "start_time": start_time,
                        "end_time": end_time,
                        "exec_type": "Trade",
                        "page": page,
                    },
                )
                fetched = fetched["result"][
                    "data" if "linear_perpetual" in self.market_type else "trade_list"
                ]
                if fetched is None:
                    break
                if fetched == []:
                    break
                if fetched[-1] == last_order:
                    break
                fills += fetched
                if fetched[-1]["trade_time_ms"] >= end_time:
                    break
                last_order = fetched[-1]
                page += 1
                print(last_order)
            fills_f = []
            for k, elm in {x["order_id"]: x for x in fills}.items():
                fills_f.append(
                    {
                        "order_id": elm["order_id"],
                        "symbol": elm["symbol"],
                        "status": elm["exec_type"].lower(),
                        "custom_id": elm["order_link_id"],
                        "price": float(elm["exec_price"]),
                        "qty": float(elm["exec_qty"]),
                        "original_qty": float(elm["order_qty"]),
                        "type": elm["order_type"].lower(),
                        "reduce_only": None,
                        "side": elm["side"].lower(),
                        "position_side": determine_pos_side(elm),
                        "timestamp": elm["trade_time_ms"],
                    }
                )
            return sorted(fills_f, key=lambda x: x["timestamp"], reverse=False)
        except Exception as e:
            print("error fetching latest fills", e)
            print_async_exception(fetched)
            traceback.print_exc()
            return []
        return fills

    async def init_exchange_config(self):
        try:
            # set cross mode
            if "inverse_futures" in self.market_type:
                res = await asyncio.gather(
                    self.private_post(
                        "/futures/private/position/leverage/save",
                        {
                            "symbol": self.symbol,
                            "buy_leverage": self.leverage,
                            "sell_leverage": self.leverage,
                        },
                    ),
                    self.private_post(
                        "/futures/private/position/switch-isolated",
                        {
                            "symbol": self.symbol,
                            "is_isolated": False,
                            "buy_leverage": self.leverage,
                            "sell_leverage": self.leverage,
                        },
                    ),
                )
                print(res)
                res = await self.private_post(
                    "/futures/private/position/switch-mode",
                    {"symbol": self.symbol, "mode": 3},
                )
                print(res)
            elif "linear_perpetual" in self.market_type:
                res = await self.private_post(
                    "/private/linear/position/switch-mode",
                    {
                        "symbol": self.symbol,
                        "mode": "BothSide",
                    },
                )
                print(res)
                res = await self.private_post(
                    "/private/linear/position/switch-isolated",
                    {
                        "symbol": self.symbol,
                        "is_isolated": False,
                        "buy_leverage": self.leverage,
                        "sell_leverage": self.leverage,
                    },
                )
                print(res)
                res = await self.private_post(
                    "/private/linear/position/set-leverage",
                    {
                        "symbol": self.symbol,
                        "buy_leverage": self.leverage,
                        "sell_leverage": self.leverage,
                    },
                )
                print(res)
            elif "inverse_perpetual" in self.market_type:
                res = await self.private_post(
                    "/v2/private/position/switch-isolated",
                    {
                        "symbol": self.symbol,
                        "is_isolated": False,
                        "buy_leverage": self.leverage,
                        "sell_leverage": self.leverage,
                    },
                )
                print("1", res)
                res = await self.private_post(
                    "/v2/private/position/leverage/save",
                    {"symbol": self.symbol, "leverage": self.leverage, "leverage_only": True},
                )
                print("2", res)
        except Exception as e:
            print(e)

    def standardize_market_stream_event(self, data: dict) -> [dict]:
        ticks = []
        for e in data["data"]:
            try:
                ticks.append(
                    {
                        "timestamp": int(e["trade_time_ms"]),
                        "price": float(e["price"]),
                        "qty": float(e["size"]),
                        "is_buyer_maker": e["side"] == "Sell",
                    }
                )
            except Exception as ex:
                print("error in websocket tick", e, ex)
        return ticks

    async def beat_heart_user_stream(self) -> None:
        while True:
            await asyncio.sleep(27)
            try:
                await self.ws_user.send(json.dumps({"op": "ping"}))
            except Exception as e:
                traceback.print_exc()
                print_(["error sending heartbeat user", e])

    async def subscribe_to_market_stream(self, ws):
        await ws.send(json.dumps({"op": "subscribe", "args": ["trade." + self.symbol]}))

    async def subscribe_to_user_stream(self, ws):
        expires = int((time() + 1) * 1000)
        signature = str(
            hmac.new(
                bytes(self.secret, "utf-8"),
                bytes(f"GET/realtime{expires}", "utf-8"),
                digestmod="sha256",
            ).hexdigest()
        )
        await ws.send(json.dumps({"op": "auth", "args": [self.key, expires, signature]}))
        await asyncio.sleep(1)
        await ws.send(
            json.dumps(
                {
                    "op": "subscribe",
                    "args": ["position", "execution", "wallet", "order"],
                }
            )
        )

    async def transfer(self, type_: str, amount: float, asset: str = "USDT"):
        return {"code": "-1", "msg": "Transferring funds not supported for Bybit"}

    def standardize_user_stream_event(
        self, event: Union[List[Dict], Dict]
    ) -> Union[List[Dict], Dict]:
        events = []
        if "topic" in event:
            if event["topic"] == "order":
                for elm in event["data"]:
                    if elm["symbol"] == self.symbol:
                        if elm["order_status"] == "Created":
                            pass
                        elif elm["order_status"] == "Rejected":
                            pass
                        elif elm["order_status"] == "New":
                            new_open_order = {
                                "order_id": elm["order_id"],
                                "symbol": elm["symbol"],
                                "price": float(elm["price"]),
                                "qty": float(elm["qty"]),
                                "type": elm["order_type"].lower(),
                                "side": (side := elm["side"].lower()),
                                "timestamp": date_to_ts(
                                    elm["timestamp" if self.inverse else "update_time"]
                                ),
                            }
                            if "inverse_perpetual" in self.market_type:
                                if self.position["long"]["size"] == 0.0:
                                    if self.position["short"]["size"] == 0.0:
                                        new_open_order["position_side"] = (
                                            "long" if new_open_order["side"] == "buy" else "short"
                                        )
                                    else:
                                        new_open_order["position_side"] = "short"
                                else:
                                    new_open_order["position_side"] = "long"
                            elif "inverse_futures" in self.market_type:
                                new_open_order["position_side"] = determine_pos_side(elm)
                            else:
                                new_open_order["position_side"] = (
                                    "long"
                                    if (
                                        (
                                            new_open_order["side"] == "buy"
                                            and elm["create_type"] == "CreateByUser"
                                        )
                                        or (
                                            new_open_order["side"] == "sell"
                                            and elm["create_type"] == "CreateByClosing"
                                        )
                                    )
                                    else "short"
                                )
                            events.append({"new_open_order": new_open_order})
                        elif elm["order_status"] == "PartiallyFilled":
                            events.append(
                                {
                                    "deleted_order_id": elm["order_id"],
                                    "partially_filled": True,
                                }
                            )
                        elif elm["order_status"] == "Filled":
                            events.append({"deleted_order_id": elm["order_id"], "filled": True})
                        elif elm["order_status"] == "Cancelled":
                            events.append({"deleted_order_id": elm["order_id"]})
                        elif elm["order_status"] == "PendingCancel":
                            pass
                    else:
                        events.append(
                            {
                                "other_symbol": elm["symbol"],
                                "other_type": event["topic"],
                            }
                        )
            elif event["topic"] == "execution":
                for elm in event["data"]:
                    if elm["symbol"] == self.symbol:
                        if elm["exec_type"] == "Trade":
                            # already handled by "order"
                            pass
                    else:
                        events.append(
                            {
                                "other_symbol": elm["symbol"],
                                "other_type": event["topic"],
                            }
                        )
            elif event["topic"] == "position":
                for elm in event["data"]:
                    if elm["symbol"] == self.symbol:
                        standardized = {}
                        if elm["side"] == "Buy":
                            standardized["psize_long"] = round_(float(elm["size"]), self.qty_step)
                            standardized["pprice_long"] = float(elm["entry_price"])
                        elif elm["side"] == "Sell":
                            standardized["psize_short"] = -round_(
                                abs(float(elm["size"])), self.qty_step
                            )
                            standardized["pprice_short"] = float(elm["entry_price"])

                        events.append(standardized)
                        if self.inverse:
                            events.append({"wallet_balance": float(elm["wallet_balance"])})
                    else:
                        events.append(
                            {
                                "other_symbol": elm["symbol"],
                                "other_type": event["topic"],
                            }
                        )
            elif not self.inverse and event["topic"] == "wallet":
                for elm in event["data"]:
                    events.append({"wallet_balance": float(elm["wallet_balance"])})
        return events
