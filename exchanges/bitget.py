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
import base64
import numpy as np
import pprint
import os

from njit_funcs import round_
from passivbot import Bot, logging
from procedures import print_async_exception, print_, utc_ms, make_get_filepath
from pure_funcs import ts_to_date, sort_dict_keys, date_to_ts, format_float, flatten


def first_capitalized(s: str):
    return s[0].upper() + s[1:].lower()


def truncate_float(x: float, d: int) -> float:
    if x is None:
        return 0.0
    multiplier = 10 ** d
    return int(x * multiplier) / multiplier


class BitgetBot(Bot):
    def __init__(self, config: dict):
        self.is_logged_into_user_stream = False
        self.exchange = "bitget"
        self.max_n_orders_per_batch = 50
        self.max_n_cancellations_per_batch = 60
        super().__init__(config)
        self.base_endpoint = "https://api.bitget.com"
        self.endpoints = {
            "exchange_info": "/api/mix/v1/market/contracts",
            "funds_transfer": "/api/spot/v1/wallet/transfer-v2",
            "position": "/api/mix/v1/position/singlePosition",
            "balance": "/api/mix/v1/account/accounts",
            "ticker": "/api/mix/v1/market/ticker",
            "tickers": "/api/mix/v1/market/tickers",
            "open_orders": "/api/mix/v1/order/current",
            "create_order": "/api/mix/v1/order/placeOrder",
            "batch_orders": "/api/mix/v1/order/batch-orders",
            "batch_cancel_orders": "/api/mix/v1/order/cancel-batch-orders",
            "cancel_order": "/api/mix/v1/order/cancel-order",
            "ticks": "/api/mix/v1/market/fills",
            "fills": "/api/mix/v1/order/fills",
            "fills_detailed": "/api/mix/v1/order/history",
            "ohlcvs": "/api/mix/v1/market/candles",
            "websocket_market": "wss://ws.bitget.com/mix/v1/stream",
            "websocket_user": "wss://ws.bitget.com/mix/v1/stream",
            "set_margin_mode": "/api/mix/v1/account/setMarginMode",
            "set_leverage": "/api/mix/v1/account/setLeverage",
        }
        self.order_side_map = {
            "buy": {"long": "open_long", "short": "close_short"},
            "sell": {"long": "close_long", "short": "open_short"},
        }
        self.fill_side_map = {
            "burst_close_long": "sell",
            "burst_close_short": "buy",
            "close_long": "sell",
            "open_long": "buy",
            "close_short": "buy",
            "open_short": "sell",
        }
        self.interval_map = {
            "1m": "60",
            "5m": "300",
            "15m": "900",
            "30m": "1800",
            "1h": "3600",
            "4h": "14400",
            "12h": "43200",
            "1d": "86400",
            "1w": "604800",
        }
        self.session = aiohttp.ClientSession()

    def init_market_type(self):
        self.symbol_stripped = self.symbol
        if self.symbol.endswith("USDT"):
            print("linear perpetual")
            self.symbol += "_UMCBL"
            self.market_type += "_linear_perpetual"
            self.product_type = "umcbl"
            self.inverse = self.config["inverse"] = False
            self.min_cost = self.config["min_cost"] = 5.5
        elif self.symbol.endswith("USD"):
            print("inverse perpetual")
            self.symbol += "_DMCBL"
            self.market_type += "_inverse_perpetual"
            self.product_type = "dmcbl"
            self.inverse = self.config["inverse"] = False
            self.min_cost = self.config[
                "min_cost"
            ] = 6.0  # will complain with $5 even if order cost > $5
        else:
            raise NotImplementedError("not yet implemented")

    async def _init(self):
        self.init_market_type()
        info = await self.fetch_exchange_info()
        for e in info["data"]:
            if e["symbol"] == self.symbol:
                break
        else:
            raise Exception(f"symbol missing {self.symbol}")
        self.coin = e["baseCoin"]
        self.quote = e["quoteCoin"]
        self.price_step = self.config["price_step"] = round_(
            (10 ** (-int(e["pricePlace"]))) * int(e["priceEndStep"]), 1e-12
        )
        self.price_rounding = int(e["pricePlace"])
        self.qty_step = self.config["qty_step"] = round_(10 ** (-int(e["volumePlace"])), 1e-12)
        self.min_qty = self.config["min_qty"] = float(e["minTradeNum"])
        self.margin_coin = self.coin if self.product_type == "dmcbl" else self.quote
        await super()._init()
        await self.init_order_book()
        await self.update_position()

    async def fetch_exchange_info(self):
        info = await self.public_get(
            self.endpoints["exchange_info"], params={"productType": self.product_type}
        )
        return info

    async def fetch_ticker(self, symbol=None):
        ticker = await self.public_get(
            self.endpoints["ticker"], params={"symbol": self.symbol if symbol is None else symbol}
        )
        return {
            "symbol": ticker["data"]["symbol"],
            "bid": float(ticker["data"]["bestBid"]),
            "ask": float(ticker["data"]["bestAsk"]),
            "last": float(ticker["data"]["last"]),
        }

    async def fetch_tickers(self, product_type=None):
        tickers = await self.public_get(
            self.endpoints["tickers"],
            params={"productType": self.product_type if product_type is None else product_type},
        )
        return [
            {
                "symbol": ticker["symbol"],
                "bid": 0.0 if ticker["bestBid"] is None else float(ticker["bestBid"]),
                "ask": 0.0 if ticker["bestAsk"] is None else float(ticker["bestAsk"]),
                "last": 0.0 if ticker["last"] is None else float(ticker["last"]),
            }
            for ticker in tickers["data"]
        ]

    async def init_order_book(self):
        ticker = None
        try:
            ticker = await self.fetch_ticker()
            self.ob = [
                ticker["bid"],
                ticker["ask"],
            ]
            self.price = ticker["last"]
            return True
        except Exception as e:
            logging.error(f"error updating order book {e}")
            print_async_exception(ticker)
            return False

    async def fetch_open_orders(self) -> [dict]:
        fetched = await self.private_get(self.endpoints["open_orders"], {"symbol": self.symbol})
        return [
            {
                "order_id": elm["orderId"],
                "custom_id": elm["clientOid"],
                "symbol": elm["symbol"],
                "price": float(elm["price"]),
                "qty": float(elm["size"]),
                "side": "buy" if elm["side"] in ["close_short", "open_long"] else "sell",
                "position_side": elm["posSide"],
                "timestamp": float(elm["cTime"]),
            }
            for elm in fetched["data"]
        ]

    async def public_get(self, url: str, params: dict = {}) -> dict:
        async with self.session.get(self.base_endpoint + url, params=params) as response:
            result = await response.text()
        return json.loads(result)

    async def private_(
        self, type_: str, base_endpoint: str, url: str, params: dict = {}, json_: bool = False
    ) -> dict:
        def stringify(x):
            if type(x) == bool:
                return "true" if x else "false"
            elif type(x) == float:
                return format_float(x)
            elif type(x) == int:
                return str(x)
            elif type(x) == list:
                return [stringify(y) for y in x]
            elif type(x) == dict:
                return {k: stringify(v) for k, v in x.items()}
            else:
                return x

        timestamp = int(time() * 1000)
        params = {k: stringify(v) for k, v in params.items()}
        if type_ == "get":
            url = url + "?" + urlencode(sort_dict_keys(params))
            to_sign = str(timestamp) + type_.upper() + url
        elif type_ == "post":
            to_sign = str(timestamp) + type_.upper() + url + json.dumps(params)
        signature = base64.b64encode(
            hmac.new(
                self.secret.encode("utf-8"),
                to_sign.encode("utf-8"),
                digestmod="sha256",
            ).digest()
        ).decode("utf-8")
        header = {
            "Content-Type": "application/json",
            "locale": "en-US",
            "ACCESS-KEY": self.key,
            "ACCESS-SIGN": signature,
            "ACCESS-TIMESTAMP": str(timestamp),
            "ACCESS-PASSPHRASE": self.passphrase,
        }
        if type_ == "post":
            async with getattr(self.session, type_)(
                base_endpoint + url, headers=header, data=json.dumps(params)
            ) as response:
                result = await response.text()
        elif type_ == "get":
            async with getattr(self.session, type_)(base_endpoint + url, headers=header) as response:
                result = await response.text()
        return json.loads(result)

    async def private_get(self, url: str, params: dict = {}, base_endpoint: str = None) -> dict:
        return await self.private_(
            type_="get",
            base_endpoint=self.base_endpoint if base_endpoint is None else base_endpoint,
            url=url,
            params=params,
        )

    async def private_post(self, url: str, params: dict = {}, base_endpoint: str = None) -> dict:
        return await self.private_(
            type_="post",
            base_endpoint=self.base_endpoint if base_endpoint is None else base_endpoint,
            url=url,
            params=params,
        )

    async def transfer_from_derivatives_to_spot(self, coin: str, amount: float):
        params = {
            "coin": "USDT",
            "amount": str(amount),
            "from_account_type": "mix_usdt",
            "to_account_type": "spot",
        }
        return await self.private_(
            "post", self.base_endpoint, self.endpoints["funds_transfer"], params=params, json_=True
        )

    async def get_server_time(self):
        now = await self.public_get("/api/spot/v1/public/time")
        return float(now["data"])

    async def fetch_position(self) -> dict:
        """
        returns {"long": {"size": float, "price": float, "liquidation_price": float},
                 "short": {...},
                 "wallet_balance": float}
        """
        position = {
            "long": {"size": 0.0, "price": 0.0, "liquidation_price": 0.0},
            "short": {"size": 0.0, "price": 0.0, "liquidation_price": 0.0},
            "wallet_balance": 0.0,
        }
        fetched_pos, fetched_balance = await asyncio.gather(
            self.private_get(
                self.endpoints["position"], {"symbol": self.symbol, "marginCoin": self.margin_coin}
            ),
            self.private_get(self.endpoints["balance"], {"productType": self.product_type}),
        )
        for elm in fetched_pos["data"]:
            if elm["holdSide"] == "long":
                position["long"] = {
                    "size": round_(float(elm["total"]), self.qty_step),
                    "price": 0.0
                    if elm["averageOpenPrice"] is None
                    else float(elm["averageOpenPrice"]),
                    "liquidation_price": 0.0
                    if elm["liquidationPrice"] is None
                    else float(elm["liquidationPrice"]),
                }

            elif elm["holdSide"] == "short":
                position["short"] = {
                    "size": -abs(round_(float(elm["total"]), self.qty_step)),
                    "price": 0.0
                    if elm["averageOpenPrice"] is None
                    else float(elm["averageOpenPrice"]),
                    "liquidation_price": 0.0
                    if elm["liquidationPrice"] is None
                    else float(elm["liquidationPrice"]),
                }
        for elm in fetched_balance["data"]:
            if elm["marginCoin"] == self.margin_coin:
                if self.product_type == "dmcbl":
                    # convert balance to usd using mean of emas as price
                    all_emas = list(self.emas_long) + list(self.emas_short)
                    if any(ema == 0.0 for ema in all_emas):
                        # catch case where any ema is zero
                        all_emas = self.ob
                    position["wallet_balance"] = float(elm["available"]) * np.mean(all_emas)
                else:
                    position["wallet_balance"] = float(elm["available"])
                break

        return position

    async def execute_orders(self, orders: [dict]) -> [dict]:
        if len(orders) == 0:
            return []
        if len(orders) == 1:
            return [await self.execute_order(orders[0])]
        return await self.execute_batch_orders(orders)

    async def execute_order(self, order: dict) -> dict:
        o = None
        try:
            params = {
                "symbol": self.symbol,
                "marginCoin": self.margin_coin,
                "size": str(order["qty"]),
                "side": self.order_side_map[order["side"]][order["position_side"]],
                "orderType": order["type"],
                "presetTakeProfitPrice": "",
                "presetStopLossPrice": "",
            }
            if params["orderType"] == "limit":
                params["timeInForceValue"] = "post_only"
                params["price"] = str(order["price"])
            else:
                params["timeInForceValue"] = "normal"
            random_str = f"{str(int(time() * 1000))[-6:]}_{int(np.random.random() * 10000)}"
            custom_id = order["custom_id"] if "custom_id" in order else "0"
            params["clientOid"] = f"{self.broker_code}#{custom_id}_{random_str}"
            o = await self.private_post(self.endpoints["create_order"], params)
            # print('debug create order', o, order)
            if o["data"]:
                # print('debug execute order', o)
                return {
                    "symbol": self.symbol,
                    "side": order["side"],
                    "order_id": o["data"]["orderId"],
                    "position_side": order["position_side"],
                    "type": order["type"],
                    "qty": order["qty"],
                    "price": order["price"],
                }
            else:
                return o, order
        except Exception as e:
            print(f"error executing order {order} {e}")
            print_async_exception(o)
            traceback.print_exc()
            return {}

    async def execute_batch_orders(self, orders: [dict]) -> [dict]:
        executed = None
        try:
            to_execute = []
            orders_with_custom_ids = []
            for order in orders:
                params = {
                    "size": str(order["qty"]),
                    "side": self.order_side_map[order["side"]][order["position_side"]],
                    "orderType": order["type"],
                    "presetTakeProfitPrice": "",
                    "presetStopLossPrice": "",
                }
                if params["orderType"] == "limit":
                    params["timeInForceValue"] = "post_only"
                    params["price"] = str(order["price"])
                else:
                    params["timeInForceValue"] = "normal"
                random_str = f"{str(int(time() * 1000))[-6:]}_{int(np.random.random() * 10000)}"
                custom_id = order["custom_id"] if "custom_id" in order else "0"
                params["clientOid"] = order[
                    "custom_id"
                ] = f"{self.broker_code}#{custom_id}_{random_str}"
                orders_with_custom_ids.append({**order, **{"symbol": self.symbol}})
                to_execute.append(params)
            executed = await self.private_post(
                self.endpoints["batch_orders"],
                {"symbol": self.symbol, "marginCoin": self.margin_coin, "orderDataList": to_execute},
            )
            formatted = []
            for ex in executed["data"]["orderInfo"]:
                to_add = {"order_id": ex["orderId"], "custom_id": ex["clientOid"]}
                for elm in orders_with_custom_ids:
                    if elm["custom_id"] == ex["clientOid"]:
                        to_add.update(elm)
                        formatted.append(to_add)
                        break
            # print('debug execute batch orders', executed, orders, formatted)
            return formatted
        except Exception as e:
            print(f"error executing order {executed} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return []

    async def execute_cancellations(self, orders: [dict]) -> [dict]:
        if not orders:
            return []
        cancellations = []
        symbol = orders[0]["symbol"] if "symbol" in orders[0] else self.symbol
        try:
            cancellations = await self.private_post(
                self.endpoints["batch_cancel_orders"],
                {
                    "symbol": symbol,
                    "marginCoin": self.margin_coin,
                    "orderIds": [order["order_id"] for order in orders],
                },
            )

            formatted = []
            for oid in cancellations["data"]["order_ids"]:
                to_add = {"order_id": oid}
                for order in orders:
                    if order["order_id"] == oid:
                        to_add.update(order)
                        formatted.append(to_add)
                        break
            # print('debug cancel batch orders', cancellations, orders, formatted)
            return formatted
        except Exception as e:
            logging.error(f"error cancelling orders {orders} {e}")
            print_async_exception(cancellations)
            traceback.print_exc()
            return []

    async def fetch_account(self):
        raise NotImplementedError("not implemented")
        try:
            resp = await self.private_get(
                self.endpoints["spot_balance"], base_endpoint=self.spot_base_endpoint
            )
            return resp["result"]
        except Exception as e:
            print("error fetching account: ", e)
            return {"balances": []}

    async def fetch_ticks(self, from_id: int = None, do_print: bool = True):
        params = {"symbol": self.symbol, "limit": 100}
        try:
            ticks = await self.public_get(self.endpoints["ticks"], params)
        except Exception as e:
            print("error fetching ticks", e)
            return []
        try:
            trades = [
                {
                    "trade_id": int(tick["tradeId"]),
                    "price": float(tick["price"]),
                    "qty": float(tick["size"]),
                    "timestamp": float(tick["timestamp"]),
                    "is_buyer_maker": tick["side"] == "sell",
                }
                for tick in ticks["data"]
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

    async def fetch_ohlcvs(self, symbol: str = None, start_time: int = None, interval="1m"):
        # m -> minutes, h -> hours, d -> days, w -> weeks
        assert interval in self.interval_map, f"unsupported interval {interval}"
        params = {
            "symbol": self.symbol if symbol is None else symbol,
            "granularity": self.interval_map[interval],
        }
        limit = 100
        seconds = float(self.interval_map[interval])
        if start_time is None:
            server_time = await self.get_server_time()
            params["startTime"] = int(round(float(server_time)) - 1000 * seconds * limit)
        else:
            params["startTime"] = int(round(start_time))
        params["endTime"] = int(round(params["startTime"] + 1000 * seconds * limit))
        fetched = await self.public_get(self.endpoints["ohlcvs"], params)
        return [
            {
                "timestamp": float(e[0]),
                "open": float(e[1]),
                "high": float(e[2]),
                "low": float(e[3]),
                "close": float(e[4]),
                "volume": float(e[5]),
            }
            for e in fetched
        ]

    async def get_all_income(
        self,
        symbol: str = None,
        start_time: int = None,
        end_time: int = None,
    ):
        if end_time is None:
            end_time = utc_ms() + 1000 * 60 * 60 * 6
        if start_time is None:
            start_time = utc_ms() - 1000 * 60 * 60 * 24 * 3
        all_fills = []
        all_fills_ids = set()
        while True:
            fills = await self.fetch_fills(symbol=symbol, start_time=start_time, end_time=end_time)
            # latest fills returned first
            if not fills:
                break
            new_fills = []
            for elm in fills:
                if elm["id"] not in all_fills_ids:
                    new_fills.append(elm)
                    all_fills_ids.add(elm["id"])
            if not new_fills:
                break
            end_time = fills[-1]["timestamp"]
            all_fills += new_fills
        income = [
            {
                "symbol": elm["symbol"],
                "transaction_id": elm["id"],
                "income": elm["realized_pnl"],
                "token": self.quote,
                "timestamp": elm["timestamp"],
            }
            for elm in all_fills
        ]
        return sorted([elm for elm in income if elm["income"] != 0.0], key=lambda x: x["timestamp"])

    async def fetch_income(
        self,
        symbol: str = None,
        start_time: int = None,
        end_time: int = None,
    ):
        raise NotImplementedError

    async def fetch_latest_fills_new(self):
        cached = None
        fname = make_get_filepath(f"logs/fills_cached_bitget/{self.user}_{self.symbol}.json")
        try:
            if os.path.exists(fname):
                cached = json.load(open(fname))
            else:
                cached = []
        except Exception as e:
            logging.error("error loading fills cache", e)
            traceback.print_exc()
            cached = []
        fetched = None
        lookback_since = int(
            utc_ms() - max(flatten([v for k, v in self.xk.items() if "delay_between_fills_ms" in k]))
        )
        try:
            params = {
                "symbol": self.symbol,
                "startTime": lookback_since,
                "endTime": int(utc_ms() + 1000 * 60 * 60 * 2),
                "pageSize": 100,
            }
            fetched = await self.private_get(self.endpoints["fills_detailed"], params)
            if (
                fetched["code"] == "00000"
                and fetched["msg"] == "success"
                and fetched["data"]["orderList"] is None
            ):
                return []
            fetched = fetched["data"]["orderList"]
            k = 0
            while fetched and float(fetched[-1]["cTime"]) > utc_ms() - 1000 * 60 * 60 * 24 * 3:
                k += 1
                if k > 5:
                    break
                params["endTime"] = int(float(fetched[-1]["cTime"]))
                fetched2 = (await self.private_get(self.endpoints["fills_detailed"], params))["data"][
                    "orderList"
                ]
                if fetched2[-1] == fetched[-1]:
                    break
                fetched_d = {x["orderId"]: x for x in fetched + fetched2}
                fetched = sorted(fetched_d.values(), key=lambda x: float(x["cTime"]), reverse=True)
            fills = [
                {
                    "order_id": elm["orderId"],
                    "symbol": elm["symbol"],
                    "status": elm["state"],
                    "custom_id": elm["clientOid"],
                    "price": float(elm["priceAvg"]),
                    "qty": float(elm["filledQty"]),
                    "original_qty": float(elm["size"]),
                    "type": elm["orderType"],
                    "reduce_only": elm["reduceOnly"],
                    "side": "buy" if elm["side"] in ["close_short", "open_long"] else "sell",
                    "position_side": elm["posSide"],
                    "timestamp": float(elm["cTime"]),
                }
                for elm in fetched
                if "filled" in elm["state"]
            ]
        except Exception as e:
            print("error fetching latest fills", e)
            print_async_exception(fetched)
            traceback.print_exc()
            return []
        return fills

    async def fetch_latest_fills(self):
        fetched = None
        try:
            params = {
                "symbol": self.symbol,
                "startTime": int(utc_ms() - 1000 * 60 * 60 * 24 * 6),
                "endTime": int(utc_ms() + 1000 * 60 * 60 * 2),
                "pageSize": 100,
            }
            fetched = await self.private_get(self.endpoints["fills_detailed"], params)
            if (
                fetched["code"] == "00000"
                and fetched["msg"] == "success"
                and fetched["data"]["orderList"] is None
            ):
                return []
            fetched = fetched["data"]["orderList"]
            k = 0
            while fetched and float(fetched[-1]["cTime"]) > utc_ms() - 1000 * 60 * 60 * 24 * 3:
                k += 1
                if k > 5:
                    break
                params["endTime"] = int(float(fetched[-1]["cTime"]))
                fetched2 = (await self.private_get(self.endpoints["fills_detailed"], params))["data"][
                    "orderList"
                ]
                if fetched2[-1] == fetched[-1]:
                    break
                fetched_d = {x["orderId"]: x for x in fetched + fetched2}
                fetched = sorted(fetched_d.values(), key=lambda x: float(x["cTime"]), reverse=True)
            fills = [
                {
                    "order_id": elm["orderId"],
                    "symbol": elm["symbol"],
                    "status": elm["state"],
                    "custom_id": elm["clientOid"],
                    "price": float(elm["priceAvg"]),
                    "qty": float(elm["filledQty"]),
                    "original_qty": float(elm["size"]),
                    "type": elm["orderType"],
                    "reduce_only": elm["reduceOnly"],
                    "side": "buy" if elm["side"] in ["close_short", "open_long"] else "sell",
                    "position_side": elm["posSide"],
                    "timestamp": float(elm["cTime"]),
                }
                for elm in fetched
                if "filled" in elm["state"]
            ]
        except Exception as e:
            print("error fetching latest fills", e)
            print_async_exception(fetched)
            traceback.print_exc()
            return []
        return fills

    async def fetch_fills(
        self,
        symbol=None,
        limit: int = 100,
        from_id: int = None,
        start_time: int = None,
        end_time: int = None,
    ):
        params = {"symbol": self.symbol if symbol is None else symbol}
        if from_id is not None:
            params["lastEndId"] = max(0, from_id - 1)
        if start_time is None:
            server_time = await self.get_server_time()
            params["startTime"] = int(round(server_time - 1000 * 60 * 60 * 24))
        else:
            params["startTime"] = int(round(start_time))

        if end_time is None:
            params["endTime"] = int(round(time() + 60 * 60 * 24) * 1000)
        else:
            params["endTime"] = int(round(end_time + 1))
        try:
            fetched = await self.private_get(self.endpoints["fills"], params)
            fills = [
                {
                    "symbol": x["symbol"],
                    "id": int(x["tradeId"]),
                    "order_id": int(x["orderId"]),
                    "side": self.fill_side_map[x["side"]],
                    "price": float(x["price"]),
                    "qty": float(x["sizeQty"]),
                    "realized_pnl": float(x["profit"]),
                    "cost": float(x["fillAmount"]),
                    "fee_paid": float(x["fee"]),
                    "fee_token": self.quote,
                    "timestamp": int(x["cTime"]),
                    "position_side": "long" if "long" in x["side"] else "short",
                    "is_maker": "unknown",
                }
                for x in fetched["data"]
            ]
        except Exception as e:
            print("error fetching fills", e)
            traceback.print_exc()
            return []
        return fills

    async def init_exchange_config(self):
        try:
            # set margin mode
            res = await self.private_post(
                self.endpoints["set_margin_mode"],
                params={
                    "symbol": self.symbol,
                    "marginCoin": self.margin_coin,
                    "marginMode": "crossed",
                },
            )
            print(res)
            # set leverage
            res = await self.private_post(
                self.endpoints["set_leverage"],
                params={
                    "symbol": self.symbol,
                    "marginCoin": self.margin_coin,
                    "leverage": self.leverage,
                },
            )
            print(res)
        except Exception as e:
            print("error initiating exchange config", e)

    def standardize_market_stream_event(self, data: dict) -> [dict]:
        if "action" not in data or data["action"] != "update":
            return []
        ticks = []
        for e in data["data"]:
            try:
                ticks.append(
                    {
                        "timestamp": int(e[0]),
                        "price": float(e[1]),
                        "qty": float(e[2]),
                        "is_buyer_maker": e[3] == "sell",
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

    async def beat_heart_market_stream(self) -> None:
        while True:
            await asyncio.sleep(27)
            try:
                await self.ws_market.send(json.dumps({"op": "ping"}))
            except Exception as e:
                traceback.print_exc()
                print_(["error sending heartbeat market", e])

    async def subscribe_to_market_stream(self, ws):
        res = await ws.send(
            json.dumps(
                {
                    "op": "subscribe",
                    "args": [
                        {
                            "instType": "mc",
                            "channel": "trade",
                            "instId": self.symbol_stripped,
                        }
                    ],
                }
            )
        )

    async def subscribe_to_user_streams(self, ws):
        res = await ws.send(
            json.dumps(
                {
                    "op": "subscribe",
                    "args": [
                        {
                            "instType": self.product_type.upper(),
                            "channel": "account",
                            "instId": "default",
                        }
                    ],
                }
            )
        )
        print(res)
        res = await ws.send(
            json.dumps(
                {
                    "op": "subscribe",
                    "args": [
                        {
                            "instType": self.product_type.upper(),
                            "channel": "positions",
                            "instId": "default",
                        }
                    ],
                }
            )
        )
        print(res)
        res = await ws.send(
            json.dumps(
                {
                    "op": "subscribe",
                    "args": [
                        {
                            "channel": "orders",
                            "instType": self.product_type.upper(),
                            "instId": "default",
                        }
                    ],
                }
            )
        )
        print(res)

    async def subscribe_to_user_stream(self, ws):
        if self.is_logged_into_user_stream:
            await self.subscribe_to_user_streams(ws)
        else:
            await self.login_to_user_stream(ws)

    async def login_to_user_stream(self, ws):
        timestamp = int(time())
        signature = base64.b64encode(
            hmac.new(
                self.secret.encode("utf-8"),
                f"{timestamp}GET/user/verify".encode("utf-8"),
                digestmod="sha256",
            ).digest()
        ).decode("utf-8")
        res = await ws.send(
            json.dumps(
                {
                    "op": "login",
                    "args": [
                        {
                            "apiKey": self.key,
                            "passphrase": self.passphrase,
                            "timestamp": timestamp,
                            "sign": signature,
                        }
                    ],
                }
            )
        )
        print(res)

    async def transfer(self, type_: str, amount: float, asset: str = "USDT"):
        return {"code": "-1", "msg": "Transferring funds not supported for Bitget"}

    def standardize_user_stream_event(
        self, event: Union[List[Dict], Dict]
    ) -> Union[List[Dict], Dict]:

        events = []
        if "event" in event and event["event"] == "login":
            self.is_logged_into_user_stream = True
            return {"logged_in": True}
        # logging.info(f"debug 0 {event}")
        if "arg" in event and "data" in event and "channel" in event["arg"]:
            if event["arg"]["channel"] == "orders":
                for elm in event["data"]:
                    if elm["instId"] == self.symbol and "status" in elm:
                        standardized = {}
                        if elm["status"] == "cancelled":
                            standardized["deleted_order_id"] = elm["ordId"]
                        elif elm["status"] == "new":
                            standardized["new_open_order"] = {
                                "order_id": elm["ordId"],
                                "symbol": elm["instId"],
                                "price": float(elm["px"]),
                                "qty": float(elm["sz"]),
                                "type": elm["ordType"],
                                "side": elm["side"],
                                "position_side": elm["posSide"],
                                "timestamp": elm["uTime"],
                            }
                        elif elm["status"] == "partial-fill":
                            standardized["deleted_order_id"] = elm["ordId"]
                            standardized["partially_filled"] = True
                        elif elm["status"] == "full-fill":
                            standardized["deleted_order_id"] = elm["ordId"]
                            standardized["filled"] = True
                        events.append(standardized)
            if event["arg"]["channel"] == "positions":
                long_pos = {"psize_long": 0.0, "pprice_long": 0.0}
                short_pos = {"psize_short": 0.0, "pprice_short": 0.0}
                for elm in event["data"]:
                    if elm["instId"] == self.symbol and "averageOpenPrice" in elm:
                        if elm["holdSide"] == "long":
                            long_pos["psize_long"] = round_(abs(float(elm["total"])), self.qty_step)
                            long_pos["pprice_long"] = truncate_float(
                                elm["averageOpenPrice"], self.price_rounding
                            )
                        elif elm["holdSide"] == "short":
                            short_pos["psize_short"] = -abs(
                                round_(abs(float(elm["total"])), self.qty_step)
                            )
                            short_pos["pprice_short"] = truncate_float(
                                elm["averageOpenPrice"], self.price_rounding
                            )
                # absence of elemet means no pos
                events.append(long_pos)
                events.append(short_pos)

            if event["arg"]["channel"] == "account":
                for elm in event["data"]:
                    if elm["marginCoin"] == self.quote:
                        events.append({"wallet_balance": float(elm["available"])})
        return events
