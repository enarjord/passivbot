import asyncio
import base64
import hashlib
import hmac
import json
import traceback
from time import time
from typing import Union, List, Dict
import uuid

import aiohttp

from njit_funcs import round_, calc_diff
from passivbot import Bot, logging
from procedures import print_async_exception, print_


def determine_pos_side(o: dict) -> str:
    if o["side"] == "buy":
        if "reduceOnly" in o:
            if o["reduceOnly"]:
                return "short"
            else:
                return "long"
        if "closeOrder" in o:
            if o["closeOrder"]:
                return "short"
            else:
                return "long"
        if "entry" in o["clientOid"]:
            return "long"
        elif "close" in o["clientOid"]:
            return "short"
        else:
            return "both"
    else:
        if "reduceOnly" in o:
            if o["reduceOnly"]:
                return "long"
            else:
                return "short"
        if "closeOrder" in o:
            if o["closeOrder"]:
                return "long"
            else:
                return "short"
        if "entry" in o["clientOid"]:
            return "short"
        elif "close" in o["clientOid"]:
            return "long"
        else:
            return "both"


class KuCoinBot(Bot):
    def __init__(self, config: dict):
        self.exchange = "kucoin"
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
        if self.symbol.endswith("USDT"):
            self.symbol += "M"
            logging.info("linear perpetual")
            self.market_type += "_linear_perpetual"
            self.inverse = self.config["inverse"] = False
            self.endpoints = {
                "position": "/api/v1/position",
                "open_orders": "/api/v1/orders",
                "create_order": "/api/v1/orders",
                "cancel_order": "/api/v1/orders",
                "ohlcvs": "/api/v1/kline/query",
                "public_token_ws": "/api/v1/bullet-public",
                "private_token_ws": "/api/v1/bullet-private",
                "income": "/api/v1/recentFills",
                "recent_orders": "/api/v1/recentDoneOrders",
            }
            self.hedge_mode = self.config["hedge_mode"] = False
        else:
            raise "Not implemented"

        self.endpoints["spot_balance"] = "/api/v1/accounts"
        self.endpoints["balance"] = "/api/v1/account-overview"
        self.endpoints["exchange_info"] = "/api/v1/contracts/active"
        self.endpoints["ticker"] = "/api/v1/ticker"
        self.endpoints["funds_transfer"] = "/asset/v1/private/transfer"  # TODO

    async def _init(self):
        self.init_market_type()
        info = await self.public_get(self.endpoints["exchange_info"])
        for elm in info["data"]:
            if elm["symbol"] == self.symbol:
                break
        else:
            raise Exception(f"symbol missing {self.symbol}")
        self.coin = elm["baseCurrency"]
        self.quote = elm["quoteCurrency"]
        self.price_step = self.config["price_step"] = float(elm["tickSize"])
        self.qty_step = self.config["qty_step"] = float(elm["lotSize"])
        self.min_qty = self.config["min_qty"] = 1.0
        self.min_cost = self.config["min_cost"] = 0.0
        self.c_mult = self.config["c_mult"] = float(elm["multiplier"])
        self.leverage = 5  # cannot be greater than 5
        await super()._init()
        await self.init_order_book()
        await self.update_position()

    async def init_order_book(self):
        ticker = None
        try:
            ticker = await self.private_get(self.endpoints["ticker"], {"symbol": self.symbol})
            self.ob = [
                float(ticker["data"]["bestBidPrice"]),
                float(ticker["data"]["bestAskPrice"]),
            ]
            self.price = float(ticker["data"]["price"])
            return True
        except Exception as e:
            logging.error(f"error updating order book {e}")
            print_async_exception(ticker)
            return False

    async def fetch_open_orders(self) -> [dict]:
        open_orders = None
        try:
            open_orders = await self.private_get(
                self.endpoints["open_orders"], {"symbol": self.symbol, "status": "active"}
            )
            return [
                {
                    "order_id": elm["id"],
                    "custom_id": elm["clientOid"],
                    "symbol": elm["symbol"],
                    "price": float(elm["price"]),
                    "qty": float(elm["size"]),
                    "side": elm["side"],
                    "position_side": determine_pos_side(elm),
                    "timestamp": elm["createdAt"],
                }
                for elm in open_orders["data"]["items"]
            ]
        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            print_async_exception(open_orders)
            traceback.print_exc()
            return False

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
            logging.error(f"not implemented")
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
            async with getattr(self.session, type_)(base_endpoint + url, headers=headers) as response:
                result = await response.text()
                return json.loads(result)

        elif "post" in type_ and data_json:
            headers["Content-Type"] = "application/json"
            async with getattr(self.session, type_)(
                base_endpoint + url, headers=headers, data=data_json
            ) as response:
                result = await response.text()
                return json.loads(result)

    async def private_get(self, url: str, params=None, base_endpoint: str = None) -> dict:
        if params is None:
            params = {}
        return await self.private_(
            "get",
            self.base_endpoint if base_endpoint is None else base_endpoint,
            url,
            params,
        )

    async def private_delete(self, url: str, params=None, base_endpoint: str = None) -> dict:
        if params is None:
            params = {}
        return await self.private_(
            "delete",
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

    async def transfer_from_derivatives_to_spot(self, coin: str, amount: float):
        raise "Not implemented"

    async def get_server_time(self):
        raise "Not implemented"

    async def fetch_position(self) -> dict:
        positions, balance = None, None
        try:

            positions, balance = await asyncio.gather(
                self.private_get(self.endpoints["position"], {"symbol": self.symbol}),
                self.private_get(self.endpoints["balance"], {"currency": self.quote}),
            )
            position = {
                "long": {"size": 0.0, "price": 0.0, "liquidation_price": 0.0},
                "short": {"size": 0.0, "price": 0.0, "liquidation_price": 0.0},
                "wallet_balance": 0.0,
                "equity": 0.0,
            }
            if positions["data"]["currentQty"] > 0.0:
                position["long"]["size"] = positions["data"]["currentQty"]
                position["long"]["price"] = positions["data"]["avgEntryPrice"]
                position["long"]["liquidation_price"] = positions["data"]["liquidationPrice"]
            elif positions["data"]["currentQty"] < 0.0:
                position["short"]["size"] = positions["data"]["currentQty"]
                position["short"]["price"] = positions["data"]["avgEntryPrice"]
                position["short"]["liquidation_price"] = positions["data"]["liquidationPrice"]
            position["wallet_balance"] = balance["data"]["marginBalance"]
            # if false, enable auto margin deposit
            if not positions["data"]["autoDeposit"]:
                logging.info("enabling auto margin deposit")
                ret = None
                try:
                    ret = await self.private_post(
                        "/api/v1/position/margin/auto-deposit-status",
                        {"symbol": self.symbol, "status": True},
                    )
                    logging.info(f"{ret}")
                except Exception as exx:
                    logging.error(f"failed to enable auto margin deposit {exx}")
                    print_async_exception(ret)
                    traceback.print_exc()
            return position
        except Exception as e:
            logging.error(f"error fetching pos or balance {e}")
            print_async_exception(positions)
            print_async_exception(balance)
            traceback.print_exc()

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
                longging.error(f"error creating order {order} {e}")
                print_async_exception(creation)
                traceback.print_exc()
        results = []
        for creation in creations:
            result = None
            try:
                result = await creation[1]
                results.append(result)
            except Exception as e:
                logging.error(f"error creating order {creation} {e}")
                print_async_exception(result)
                traceback.print_exc()
        return results

    async def execute_order(self, order: dict) -> dict:
        executed = None
        try:
            params = {
                "symbol": self.symbol,
                "side": order["side"],
                "type": order["type"],
                "leverage": str(self.leverage),
                "size": int(order["qty"]),
                "reduceOnly": order["reduce_only"],
            }
            if order["type"] == "limit":
                params["postOnly"] = True
                params["price"] = str(order["price"])
            params[
                "clientOid"
            ] = f"{(order['custom_id'] if 'custom_id' in order else '')}{uuid.uuid4().hex}"[:32]
            executed = await self.private_post(self.endpoints["create_order"], params)
            if "code" in executed and executed["code"] == "200000":
                return {
                    "symbol": self.symbol,
                    "side": order["side"],
                    "order_id": executed["data"]["orderId"],
                    "position_side": order["position_side"],
                    "type": order["type"],
                    "qty": int(order["qty"]),
                    "price": order["price"],
                }
            raise Exception
        except Exception as e:
            logging.error(f"error executing order {executed} {order} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return None

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
                logging.error(f"error cancelling order {order} {e}")
                print_async_exception(cancellation)
                traceback.print_exc()
        results = []
        for cancellation in cancellations:
            result = None
            try:
                result = await cancellation[1]
                results.append(result)
            except Exception as e:
                logging.error(f"error cancelling order {cancellation} {e}")
                print_async_exception(result)
                traceback.print_exc()
        return results

    async def execute_cancellation(self, order: dict) -> dict:
        cancellation = None
        try:
            cancellation = await self.private_delete(
                f"{self.endpoints['cancel_order']}/{order['order_id']}", {}
            )
            return {
                "symbol": self.symbol,
                "side": order["side"],
                "order_id": cancellation["data"]["cancelledOrderIds"][0],
                "position_side": order["position_side"],
                "qty": order["qty"],
                "price": order["price"],
            }
        except Exception as e:
            logging.error(f"error cancelling order {order} {e}")
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
        ohlcvs = []
        for e in fetched["data"]:
            ohlcv = {
                "timestamp": float(e[0]),
                "high": float(e[1]),
                "low": float(e[2]),
                "close": float(e[3]),
                "volume": float(e[4]),
            }
            ohlcvs.append(ohlcv)
        return ohlcvs

    async def get_all_income(
        self,
        symbol: str = None,
        start_time: int = None,
        income_type: str = "Trade",
        end_time: int = None,
    ):
        raise "Not implemented"

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

    async def fetch_fills(
        self,
        limit: int = 200,
        from_id: int = None,
        start_time: int = None,
        end_time: int = None,
    ):
        return []

    async def fetch_latest_fills(self):
        fetched = None
        try:
            fetched = await self.private_get(self.endpoints["recent_orders"])
            return [
                {
                    "order_id": elm["id"],
                    "symbol": elm["symbol"],
                    "type": elm["type"],
                    "status": elm["status"],
                    "custom_id": elm["clientOid"],
                    "price": float(elm["price"]),
                    "qty": float(elm["filledSize"]),
                    "original_qty": float(elm["size"]),
                    "reduce_only": elm["reduceOnly"],
                    "side": elm["side"],
                    "position_side": determine_pos_side(elm),
                    "timestamp": float(elm["updatedAt"]),
                }
                for elm in fetched["data"]
                if elm["symbol"] == self.symbol and not elm["cancelExist"] and not elm["isActive"]
            ]
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
                raise "Not implemented"
            elif "linear_perpetual" in self.market_type:
                res = await self.private_post(
                    "/api/v1/position/risk-limit-level/change",
                    {
                        "symbol": self.symbol,
                        "level": self.leverage,
                    },
                )
                logging.info(f"setting risk level {res}")

            elif "inverse_perpetual" in self.market_type:
                raise "Not implemented"
        except Exception as e:
            logging.error(f"error with init_exchange_config {e}")

    def standardize_market_stream_event(self, data: dict) -> [dict]:
        try:
            return [
                {
                    "timestamp": int(data["data"]["ts"]),
                    "price": float(data["data"]["price"]),
                    "qty": float(data["data"]["size"]),
                    "is_buyer_maker": data["data"]["side"] == "sell",
                }
            ]
        except Exception as ex:
            logging.error(f"error in websocket tick {ex}")
        return []

    async def beat_heart_user_stream(self) -> None:
        while True:
            await asyncio.sleep(27)
            try:
                await self.ws_user.send(json.dumps({"type": "ping"}))
            except Exception as e:
                traceback.print_exc()
                logging.error(f"error sending heartbeat user {e}")

    async def init_user_stream(self) -> None:
        res = await self.private_post(self.endpoints["private_token_ws"], {})
        logging.info(f"init user stream {res}")
        self.endpoints[
            "websocket_user"
        ] = f"{res['data']['instanceServers'][0]['endpoint']}?token={res['data']['token']}"

    async def init_market_stream(self):
        res = await self.private_post(self.endpoints["public_token_ws"], {})
        logging.info(f"init market stream {res}")
        self.endpoints[
            "websocket_market"
        ] = f"{res['data']['instanceServers'][0]['endpoint']}?token={res['data']['token']}"

    async def subscribe_to_market_stream(self, ws):
        await ws.send(
            json.dumps(
                {
                    "type": "subscribe",
                    "topic": f"/contractMarket/execution:{self.symbol}",
                    "response": True,
                }
            )
        )

    async def subscribe_to_user_stream(self, ws):
        await ws.send(json.dumps({"type": "openTunnel", "newTunnelId": "order", "response": True}))
        await ws.send(json.dumps({"type": "openTunnel", "newTunnelId": "wallet", "response": True}))
        await ws.send(json.dumps({"type": "openTunnel", "newTunnelId": "position", "response": True}))

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
                if new_open_order["side"] == "buy":
                    if self.position["long"]["size"] == 0.0:
                        if self.position["short"]["size"] == 0.0:
                            new_open_order["position_side"] = "long"
                        else:
                            new_open_order["position_side"] = "short"
                    else:
                        new_open_order["position_side"] = "long"
                elif new_open_order["side"] == "sell":
                    if self.position["short"]["size"] == 0.0:
                        if self.position["long"]["size"] == 0.0:
                            new_open_order["position_side"] = "short"
                        else:
                            new_open_order["position_side"] = "long"
                    else:
                        new_open_order["position_side"] = "short"
                else:
                    raise Exception(f"unknown pos side {event}")
                events.append({"new_open_order": new_open_order})
            elif event["data"]["type"] in ["match", "filled"]:
                events.append({"deleted_order_id": event["data"]["orderId"], "filled": True})
            elif event["data"]["type"] in ["canceled", "update"]:
                events.append({"deleted_order_id": event["data"]["orderId"]})
        elif event["tunnelId"] == "position":
            if (
                event["data"]["symbol"] == self.symbol
                and event["data"]["changeReason"] != "markPriceChange"
            ):
                standardized = {}
                if event["data"]["changeReason"] == "positionChange":
                    standardized["psize_long"] = 0.0
                    standardized["pprice_long"] = 0.0
                    standardized["psize_short"] = 0.0
                    standardized["pprice_short"] = 0.0
                    if event["data"]["currentQty"] > 0.0:
                        standardized["psize_long"] = event["data"]["currentQty"]
                        standardized["pprice_long"] = event["data"]["avgEntryPrice"]
                    elif event["data"]["currentQty"] < 0.0:
                        standardized["psize_short"] = event["data"]["currentQty"]
                        standardized["pprice_short"] = event["data"]["avgEntryPrice"]
                events.append(standardized)

        elif event["tunnelId"] == "wallet" and event["subject"] == "availableBalance.change":

            """
            events.append(
                {
                    "wallet_balance": float(event["data"]["availableBalance"])
                    + float(event["data"]["holdBalance"])
                }
            )
            """
            # updates too often, would cause spam to exchange, will work without
            pass

        return events
