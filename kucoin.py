import asyncio
import base64
import hashlib
import hmac
import json
import logging
import traceback
from time import time
from typing import Union, List, Dict
from uuid import uuid4

import aiohttp
import numpy as np

from njit_funcs import round_, calc_diff
from passivbot import Bot
from procedures import print_async_exception, print_
from pure_funcs import ts_to_date


def first_capitalized(s: str):
    return s[0].upper() + s[1:].lower()


def determine_pos_side(o: dict) -> str:
    if o["side"].lower() == "buy":
        if "entry" in o["clientOid"]:
            return "long"
        elif "close" in o["clientOid"]:
            return "short"
        else:
            return "both"
    else:
        if "entry" in o["clientOid"]:
            return "short"
        elif "close" in o["clientOid"]:
            return "long"
        else:
            return "both"


class KuCoinBot(Bot):
    def __init__(self, config: dict):
        self.exchange = "kucoin"
        self.min_notional = 1.0
        self.max_n_orders_per_batch = 5
        self.max_n_cancellations_per_batch = 10
        super().__init__(config)
        self.base_endpoint = "https://api-futures.kucoin.com"
        self.endpoints = {
            "balance": "/api/v1/account-overview",
            "exchange_info": "/api/v1/contracts/active",
            "ticker": "/api/v1/ticker",
            "funds_transfer": "/api/v3/transfer-out",
        }
        self.session = aiohttp.ClientSession()

    def init_market_type(self):
        if self.symbol.endswith("USDTM"):
            print("linear perpetual")
            self.market_type += "_linear_perpetual"
            self.inverse = self.config["inverse"] = False
            self.endpoints = {
                "position": "/api/v1/position",
                "open_orders": "/api/v1/orders",
                "create_order": "/api/v1/orders",
                "cancel_order": "/api/v1/orders/{order-id}",
                "ticks": "/public/linear/recent-trading-records",
                "fills": "/private/linear/trade/execution/list",
                "ohlcvs": "/api/v1/kline/query",
                "public_token_ws": "/api/v1/bullet-public",
                "private_token_ws": "/api/v1/bullet-private",
                "income": "/api/v1/recentFills",
                "created_at_key": "createdAt",
            }
            self.hedge_mode = self.config["hedge_mode"] = False
        else:
            raise "Not implemented"
            self.inverse = self.config["inverse"] = True
            if self.symbol.endswith("USDM"):
                print("inverse perpetual")
                self.market_type += "_inverse_perpetual"
                self.endpoints = {
                    "position": "/api/v1/position",
                    "open_orders": "/api/v1/orders",
                    "create_order": "/api/v1/orders",
                    "cancel_order": "/api/v1/orders/{order-id}",
                    "ticks": "/public/linear/recent-trading-records",
                    "fills": "/private/linear/trade/execution/list",
                    "ohlcvs": "/api/v1/kline/query",
                    "income": "/api/v1/recentFills",
                    "created_at_key": "createdAt",
                }

                self.hedge_mode = self.config["hedge_mode"] = False
            else:
                print("inverse futures")
                self.market_type += "_inverse_futures"
                self.endpoints = {
                    "position": "/api/v1/position",
                    "open_orders": "/api/v1/orders",
                    "create_order": "/api/v1/orders",
                    "cancel_order": "/api/v1/orders",
                    "ticks": "/public/linear/recent-trading-records",
                    "fills": "/private/linear/trade/execution/list",
                    "ohlcvs": "/api/v1/kline/query",
                    "income": "/api/v1/recentFills",
                    "created_at_key": "createdAt",
                }

        self.endpoints["spot_balance"] = "/api/v1/accounts"
        self.endpoints["balance"] = "/api/v1/account-overview"
        self.endpoints["exchange_info"] = "/api/v1/contracts/active"
        self.endpoints["ticker"] = "/api/v1/ticker"
        self.endpoints["funds_transfer"] = "/asset/v1/private/transfer"  # TODO

    async def _init(self):
        info = await self.public_get(self.endpoints["exchange_info"])
        for e in info["data"]:
            if e["symbol"] == self.symbol:
                break
        else:
            raise Exception(f"symbol missing {self.symbol}")
        self.max_leverage = e["maxLeverage"]
        self.coin = e["baseCurrency"]
        self.quot = e["quoteCurrency"]
        self.price_step = self.config["price_step"] = float(e["tickSize"])
        self.qty_step = self.config["qty_step"] = float(e["lotSize"])
        self.min_qty = self.config["min_qty"] = float(e["lotSize"])
        self.min_cost = self.config["min_cost"] = 0.0
        self.c_mult = self.config["c_mult"] = float(e["multiplier"])
        self.init_market_type()
        self.margin_coin = self.coin if self.inverse else self.quot
        await super()._init()
        await self.init_order_book()
        await self.update_position()

    async def init_order_book(self):
        ticker = await self.private_get(
            self.endpoints["ticker"], {"symbol": self.symbol}
        )
        print()
        self.ob = [
            float(ticker["data"]["bestBidPrice"]),
            float(ticker["data"]["bestAskPrice"]),
        ]
        self.price = float(ticker["data"]["price"])

    async def fetch_open_orders(self) -> [dict]:
        fetched = await self.private_get(
            self.endpoints["open_orders"], {"symbol": self.symbol, "status": "active"}
        )
        return [
            {
                "order_id": elm["id"],
                "custom_id": elm["clientOid"],
                "symbol": elm["symbol"],
                "price": float(elm["price"]),
                "qty": float(elm["size"]),
                "side": elm["side"].lower(),
                "position_side": determine_pos_side(elm),
                "timestamp": elm[self.endpoints["created_at_key"]],
            }
            for elm in fetched["data"]["items"]
        ]

    async def public_get(self, url: str, params=None) -> dict:
        if params is None:
            params = {}
        async with self.session.get(
            self.base_endpoint + url, params=params
        ) as response:
            result = await response.text()
        return json.loads(result)

    async def private_(
        self,
        type_: str,
        base_endpoint: str,
        url: str,
        params=None,
    ) -> dict:
        if params is None:
            params = {}
        data_json = None
        timestamp = int(time() * 1000)
        if type_ == "get" or "delete" in type_:
            if len(params) > 0:
                url += "?"
                for param in params:
                    url += f"{param}={params[param]}&"
            str_to_sign = f"{str(timestamp)}{type_.upper()}{url}"
        elif type_ == "post":
            data_json = json.dumps(params, separators=(",", ":"), ensure_ascii=False)
            str_to_sign = f"{str(timestamp)}{type_.upper()}{url}{data_json}"
        else:
            print(f"not implemented")
            return

        signature = base64.b64encode(
            hmac.new(
                self.secret.encode("utf-8"), str_to_sign.encode("utf-8"), hashlib.sha256
            ).digest()
        )

        passphrase = base64.b64encode(
            hmac.new(
                self.secret.encode("utf-8"),
                self.passphrase.encode("utf-8"),
                hashlib.sha256,
            ).digest()
        )
        headers = {
            "KC-API-SIGN": signature.decode("utf-8"),
            "KC-API-TIMESTAMP": str(timestamp),
            "KC-API-KEY": self.key,
            "KC-API-PASSPHRASE": passphrase.decode("utf-8"),
            "KC-API-KEY-VERSION": "2",
        }

        if "get" in type_ or "delete" in type_:
            async with getattr(self.session, type_)(
                base_endpoint + url, headers=headers
            ) as response:
                result = await response.text()
                return json.loads(result)

        elif "post" in type_ and data_json:
            headers["Content-Type"] = "application/json"
            async with getattr(self.session, type_)(
                base_endpoint + url, headers=headers, data=data_json
            ) as response:
                result = await response.text()
                return json.loads(result)

    async def private_get(
        self, url: str, params=None, base_endpoint: str = None
    ) -> dict:
        if params is None:
            params = {}
        return await self.private_(
            "get",
            self.base_endpoint if base_endpoint is None else base_endpoint,
            url,
            params,
        )

    async def private_delete(
        self, url: str, params=None, base_endpoint: str = None
    ) -> dict:
        if params is None:
            params = {}
        return await self.private_(
            "delete",
            self.base_endpoint if base_endpoint is None else base_endpoint,
            url,
            params,
        )

    async def private_post(
        self, url: str, params=None, base_endpoint: str = None
    ) -> dict:
        if params is None:
            params = {}
        return await self.private_(
            "post",
            self.base_endpoint if base_endpoint is None else base_endpoint,
            url,
            params,
        )

    async def transfer_from_derivatives_to_spot(self, coin: str, amount: float):
        raise "Not implemented"
        params = {
            "coin": coin,
            "amount": str(amount),
            "from_account_type": "CONTRACT",
            "to_account_type": "SPOT",
            "transfer_id": str(uuid4()),
        }
        return await self.private_(
            "post",
            self.base_endpoint,
            self.endpoints["funds_transfer"],
            params=params,
            json_=True,
        )

    async def get_server_time(self):
        now = await self.public_get("/v2/public/time")
        return float(now["time_now"]) * 1000

    async def fetch_position(self) -> dict:
        position = {}
        long_pos = None
        short_pos = None
        if "linear_perpetual" in self.market_type:
            fetched, bal = await asyncio.gather(
                self.private_get(self.endpoints["position"], {"symbol": self.symbol}),
                self.private_get(self.endpoints["balance"], {"currency": self.quot}),
            )
            if fetched["data"]["isOpen"]:
                if fetched["data"]["currentQty"] > 0:
                    long_pos = fetched["data"]
                else:
                    short_pos = fetched["data"]
            position["wallet_balance"] = float(bal["data"]["accountEquity"])
        # TODO
        # else:
        # fetched, bal = await asyncio.gather(
        #     self.private_get(self.endpoints["position"], {"symbol": self.symbol}),
        #     self.private_get(self.endpoints["balance"], {"currency": self.coin}),
        # )
        # position["wallet_balance"] = float(bal["data"]["availableBalance"])
        # if "inverse_perpetual" in self.market_type:
        #     if fetched["result"]["side"] == "Buy":
        #         long_pos = fetched["data"]
        #         short_pos = {"size": 0.0, "entry_price": 0.0, "liq_price": 0.0}
        #     else:
        #         long_pos = {"size": 0.0, "entry_price": 0.0, "liq_price": 0.0}
        #         short_pos = fetched["data"]
        # elif "inverse_futures" in self.market_type:
        #     long_pos = [
        #         e["data"]
        #         for e in fetched["result"]
        #         if e["data"]["position_idx"] == 1
        #     ][0]
        #     short_pos = [
        #         e["data"]
        #         for e in fetched["result"]
        #         if e["data"]["position_idx"] == 2
        #     ][0]
        # else:
        #     raise Exception("unknown market type")

        if long_pos is not None and long_pos["currentQty"] > 0:
            position["long"] = {
                "size": round_(float(long_pos["currentQty"]), self.qty_step),
                "price": float(long_pos["avgEntryPrice"]),
                "liquidation_price": float(long_pos["liquidationPrice"]),
            }
        else:
            position["long"] = {
                "size": round_(float(0), self.qty_step),
                "price": float(0),
                "liquidation_price": float(0),
            }

        if short_pos is not None and short_pos["currentQty"] < 0:
            position["short"] = {
                "size": round_(float(short_pos["currentQty"]), self.qty_step),
                "price": float(short_pos["avgEntryPrice"]),
                "liquidation_price": float(short_pos["liquidationPrice"]),
            }
        else:
            position["short"] = {
                "size": round_(float(0), self.qty_step),
                "price": float(0),
                "liquidation_price": float(0),
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

    async def execute_order(self, order: dict) -> dict:
        o = None
        try:
            params = {
                "symbol": self.symbol,
                "side": order["side"],
                "type": order["type"],
                "leverage": str(self.leverage),
            }
            size = int(order["qty"])
            params["size"] = size
            params["reduceOnly"] = "close" in order["custom_id"]

            if params["type"] == "limit":
                params["timeInForce"] = "GTC"
                # params["postOnly"] = False
                params["price"] = "{:.8f}".format(order["price"])

            params[
                "clientOid"
            ] = f"{order['custom_id']}_{str(int(time() * 1000))[8:]}_{int(np.random.random() * 1000)}"
            o = await self.private_post(self.endpoints["create_order"], params)
            if o["data"]["orderId"]:
                return {
                    "symbol": self.symbol,
                    "side": order["side"].lower(),
                    "order_id": o["data"]["orderId"],
                    "position_side": order["position_side"],
                    "type": order["type"].lower(),
                    "qty": int(order["qty"]),
                    "price": order["price"],
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
            cancellation = await self.private_delete(
                f"{self.endpoints['cancel_order']}/{order['order_id']}", {}
            )
            print(cancellation)
            if (
                cancellation is not None
                and "code" in cancellation
                and cancellation["code"] == 100004
                or cancellation["code"] == 404
            ):
                raise
            return {
                "symbol": self.symbol,
                "side": order["side"],
                "order_id": order["order_id"],
                "position_side": order["position_side"],
                "qty": order["qty"],
                "price": order["price"],
            }
        except Exception as e:
            if (
                cancellation is not None
                and "code" in cancellation
                and cancellation["code"] == 100004
                or cancellation["code"] == 404
            ):
                error_cropped = {
                    k: v
                    for k, v in cancellation.items()
                    if k in ["ret_msg", "ret_code"]
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
            "granularity": interval_map[interval],
        }
        fetched = await self.public_get(self.endpoints["ohlcvs"], params)
        ticks = []
        for e in fetched["data"]:
            tick = {
                "timestamp": float(e[0]),
                "high": float(e[1]),
                "low": float(e[2]),
                "close": float(e[3]),
                "volume": float(e[4]),
            }
            ticks.append(tick)
        return ticks

    async def get_all_income(
        self,
        symbol: str = None,
        start_time: int = None,
        income_type: str = "Trade",
        end_time: int = None,
    ):
        raise "Not implemented"
        if symbol is None:
            all_income = []
            all_positions = await self.private_get(
                self.endpoints["position"], params={"symbol": ""}
            )
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
                    symbol=symbol,
                    start_time=start_time,
                    income_type=income_type,
                    end_time=end_time,
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
        raise "Not implemented"
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

    async def fetch_fills(
        self,
        limit: int = 200,
        from_id: int = None,
        start_time: int = None,
        end_time: int = None,
    ):
        return []
        ffills, fpnls = await asyncio.gather(
            self.private_get(
                self.endpoints["fills"], {"symbol": self.symbol, "limit": limit}
            ),
            self.private_get(
                self.endpoints["pnls"], {"symbol": self.symbol, "limit": 50}
            ),
        )
        return ffills, fpnls
        try:
            fills = []
            for x in fetched["result"]["data"][::-1]:
                qty, price = float(x["order_qty"]), float(x["price"])
                if not qty or not price:
                    continue
                fill = {
                    "symbol": x["symbol"],
                    "id": str(x["exec_id"]),
                    "order_id": str(x["order_id"]),
                    "side": x["side"].lower(),
                    "price": price,
                    "qty": qty,
                    "realized_pnl": 0.0,
                    "cost": (cost := qty / price if self.inverse else qty * price),
                    "fee_paid": float(x["exec_fee"]),
                    "fee_token": self.margin_coin,
                    "timestamp": int(x["trade_time_ms"]),
                    "position_side": determine_pos_side(x),
                    "is_maker": x["fee_rate"] < 0.0,
                }
                fills.append(fill)
            return fills
        except Exception as e:
            print("error fetching fills", e)
            return []
        return fetched
        print("fetch_fills not implemented for KuCoin")
        return []

    async def init_exchange_config(self):
        try:
            # set cross mode
            if "inverse_futures" in self.market_type:
                raise "Not implemented"
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
                self.leverage = 5
                res = await self.private_post(
                    "/api/v1/position/risk-limit-level/change",
                    {
                        "symbol": self.symbol,
                        "level": self.leverage,
                    },
                )
                print(res)

                res = await self.private_post(self.endpoints["public_token_ws"], {})
                print(res)

                self.endpoints[
                    "websocket_market"
                ] = f"{res['data']['instanceServers'][0]['endpoint']}?token={res['data']['token']}"

                res = await self.private_post(self.endpoints["private_token_ws"], {})
                print(res)
                self.endpoints[
                    "websocket_user"
                ] = f"{res['data']['instanceServers'][0]['endpoint']}?token={res['data']['token']}"

            elif "inverse_perpetual" in self.market_type:
                raise "Not implemented"
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
                    {
                        "symbol": self.symbol,
                        "leverage": self.leverage,
                        "leverage_only": True,
                    },
                )
                print("2", res)
        except Exception as e:
            print(e)

    def standardize_market_stream_event(self, data: dict) -> [dict]:
        ticks = []
        try:
            price = data["data"]["change"].split(",")[0]
            side = data["data"]["change"].split(",")[1]
            quantity = data["data"]["change"].split(",")[2]
            ticks.append(
                {
                    "timestamp": int(data["data"]["timestamp"]),
                    "price": float(price),
                    "qty": float(quantity),
                    "is_buyer_maker": side == "sell",
                }
            )
        except Exception as ex:
            print("error in websocket tick", ex)
        return ticks

    async def beat_heart_user_stream(self) -> None:
        while True:
            await asyncio.sleep(27)
            try:
                await self.ws_user.send(json.dumps({"type": "ping"}))
            except Exception as e:
                traceback.print_exc()
                print_(["error sending heartbeat user", e])

    async def subscribe_to_market_stream(self, ws):
        await ws.send(
            json.dumps(
                {
                    "type": "subscribe",
                    "topic": f"/contractMarket/level2:{self.symbol}",
                    "response": True,
                }
            )
        )

    async def subscribe_to_user_stream(self, ws):
        await ws.send(
            json.dumps({"type": "openTunnel", "newTunnelId": "order", "response": True})
        )
        await ws.send(
            json.dumps(
                {"type": "openTunnel", "newTunnelId": "wallet", "response": True}
            )
        )
        await ws.send(
            json.dumps(
                {"type": "openTunnel", "newTunnelId": "position", "response": True}
            )
        )

        await ws.send(
            json.dumps(
                {
                    "type": "subscribe",
                    "topic": f"/contractMarket/tradeOrders:{self.symbol}",
                    "response": True,
                    "privateChannel": True,
                    "tunnelId": "order",
                }
            )
        )
        await ws.send(
            json.dumps(
                {
                    "type": "subscribe",
                    "topic": f"/contractAccount/wallet",
                    "response": True,
                    "privateChannel": True,
                    "tunnelId": "wallet",
                }
            )
        )
        await ws.send(
            json.dumps(
                {
                    "type": "subscribe",
                    "topic": f"/contract/position:{self.symbol}",
                    "response": True,
                    "privateChannel": True,
                    "tunnelId": "position",
                }
            )
        )

    async def transfer(self, type_: str, amount: float, asset: str = "USDT"):
        raise "Not implemented"

    def standardize_user_stream_event(
        self, event: Union[List[Dict], Dict]
    ) -> Union[List[Dict], Dict]:
        events = []
        if "tunnelId" not in event:
            return events
        if "tunnelId" in event:
            if event["tunnelId"] == "order":
                if event["data"]["type"] == "open":
                    new_open_order = {
                        "order_id": event["data"]["orderId"],
                        "symbol": event["data"]["symbol"],
                        "price": float(event["data"]["price"]),
                        "qty": float(int(event["data"]["size"])),
                        "type": "limit",
                        "side": event["data"]["side"].lower(),
                        "timestamp": event["data"]["orderTime"],
                    }
                    if self.position["long"]["size"] == 0.0:
                        if self.position["short"]["size"] == 0.0:
                            new_open_order["position_side"] = "long"
                        else:
                            new_open_order["position_side"] = "short"
                    else:
                        new_open_order["position_side"] = "long"
                    events.append({"new_open_order": new_open_order})

                elif event["data"]["type"] == "match":
                    events.append(
                        {"deleted_order_id": event["data"]["orderId"], "filled": True}
                    )
                elif event["data"]["type"] == "filled":
                    print(f"order---filled{event}")
                    events.append(
                        {"deleted_order_id": event["data"]["orderId"], "filled": True}
                    )
                elif event["data"]["type"] == "canceled":
                    events.append({"deleted_order_id": event["data"]["orderId"]})
                elif event["data"]["type"] == "update":
                    print(f"order---update{event}")
                    events.append(
                        {
                            "other_symbol": event["data"]["symbol"],
                            "other_type": "order_update",
                        }
                    )

            elif event["tunnelId"] == "position":
                if (
                    event["data"]["symbol"] == self.symbol
                    and event["data"]["changeReason"] != "markPriceChange"
                ):
                    standardized = {}
                    if event["data"]["changeReason"] == "positionChange":
                        if self.position["long"]["size"] == 0.0:
                            if self.position["short"]["size"] == 0.0:
                                standardized["psize_long"] = round_(
                                    float(int(event["data"]["currentQty"])),
                                    self.qty_step,
                                )
                                standardized["pprice_long"] = float(
                                    event["data"]["avgEntryPrice"]
                                )
                            else:
                                standardized["psize_short"] = -round_(
                                    abs(
                                        float(event["data"]["currentQty"]),
                                    ),
                                    self.qty_step,
                                )
                                standardized["pprice_short"] = float(
                                    event["data"]["avgEntryPrice"]
                                )
                        else:
                            standardized["psize_long"] = round_(
                                float(event["data"]["currentQty"]),
                                self.qty_step,
                            )
                            standardized["pprice_long"] = float(
                                event["data"]["avgEntryPrice"]
                            )
                    events.append(standardized)

            elif event["tunnelId"] == "account":
                wallet_balance = float(
                    event["data"]["holdBalancexecute_orderse"]
                ) + float(event["data"]["availableBalance"])
                events.append({"wallet_balance": wallet_balance})

        return events
