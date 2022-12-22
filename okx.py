import asyncio
import hashlib
import hmac
import json
import traceback
from time import time, time_ns
from urllib.parse import urlencode

import aiohttp
import numpy as np
import ccxt.async_support as ccxt

from passivbot import Bot, logging
from procedures import print_, print_async_exception
from pure_funcs import ts_to_date, sort_dict_keys, format_float


class OKXBot(Bot):
    def __init__(self, config: dict):
        self.exchange = "okx"
        super().__init__(config)
        self.okx = getattr(ccxt, "okx")(
            {"apiKey": self.key, "secret": self.secret, "password": self.passphrase}
        )

    async def init_market_type(self):
        self.markets = None
        try:
            self.markets = await self.okx.fetch_markets()
            self.market_type = "linear perpetual swap"
            if self.symbol.endswith("USDT"):
                self.symbol = f"{self.symbol[:-4]}/USDT:USDT"
            else:
                # TODO implement inverse
                raise NotImplementedError(f"not implemented for {self.symbol}")
        except Exception as e:
            logging.error(f"error initiating market type {e}")
            print_async_exception(self.exchange_info)
            traceback.print_exc()
            raise Exception("stopping bot")

    async def _init(self):
        await self.init_market_type()
        for elm in self.markets:
            if elm["symbol"] == self.symbol:
                break
        else:
            raise Exception(f"symbol {self.symbol} not found")
        self.coin = elm["base"]
        self.quote = elm["quote"]
        self.margin_coin = elm["quote"]
        self.c_mult = self.config["c_mult"] = elm["contractSize"]
        self.min_qty = self.config["min_qty"] = (
            elm["limits"]["amount"]["min"] if elm["limits"]["amount"]["min"] else 0.0
        )
        self.min_cost = self.config["min_cost"] = (
            elm["limits"]["cost"]["min"] if elm["limits"]["cost"]["min"] else 0.0
        )
        self.qty_step = self.config["qty_step"] = elm["precision"]["amount"]
        self.price_step = self.config["price_step"] = elm["precision"]["price"]
        self.inverse = self.config["inverse"] = False
        await super()._init()
        await self.init_order_book()
        await self.update_position()

    async def get_server_time(self):
        # millis
        return await self.okx.fetch_time()

    async def transfer_from_derivatives_to_spot(self, coin: str, amount: float):
        return
        return await self.private_post(
            self.endpoints["futures_transfer"],
            {"asset": coin, "amount": amount, "type": 2},
            base_endpoint=self.spot_base_endpoint,
        )

    async def execute_leverage_change(self):
        return await self.okx.set_leverage(self.leverage, symbol=self.symbol)

    async def init_exchange_config(self) -> bool:
        try:
            logging.info(
                str(
                    await self.okx.set_margin_mode(
                        "cross", symbol=self.symbol, params={"lever": self.leverage}
                    )
                )
            )
        except Exception as e:
            print(e)
        try:
            logging.info(str(await self.execute_leverage_change()))
        except Exception as e:
            print(e)
        try:
            logging.info(str(await self.okx.set_position_mode(True)))
        except Exception as e:
            print(e)

    async def init_order_book(self):
        ticker = None
        try:
            ticker = await self.okx.fetch_ticker(symbol=self.symbol)
            self.ob = [ticker["bid"], ticker["ask"]]
            self.price = np.random.choice(self.ob)
            return True
        except Exception as e:
            logging.error(f"error updating order book {e}")
            print_async_exception(ticker)
            return False

    async def fetch_open_orders(self) -> [dict]:
        open_orders = None
        try:
            open_orders = await self.okx.fetch_open_orders(symbol=self.symbol)
            return [
                {
                    "order_id": e["id"],
                    "symbol": e["symbol"],
                    "price": e["price"],
                    "qty": e["amount"],
                    "type": e["type"],
                    "side": e["side"],
                    "position_side": e["info"]["posSide"],
                    "timestamp": e["timestamp"],
                }
                for e in open_orders
            ]
        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            print_async_exception(open_orders)
            traceback.print_exc()
            return False

    async def fetch_position(self) -> dict:
        positions, balance = None, None
        try:
            positions, balance = await asyncio.gather(
                self.okx.fetch_positions(),
                self.okx.fetch_balance(),
            )
            positions = [e for e in positions if e["symbol"] == self.symbol]
            position = {
                "long": {"size": 0.0, "price": 0.0, "liquidation_price": 0.0},
                "short": {"size": 0.0, "price": 0.0, "liquidation_price": 0.0},
                "wallet_balance": 0.0,
                "equity": 0.0,
            }
            if positions:
                for p in positions:
                    if p["side"] == "long":
                        position["long"] = {
                            "size": p["contracts"],
                            "price": p["entryPrice"],
                            "liquidation_price": p["liquidationPrice"]
                            if p["liquidationPrice"]
                            else 0.0,
                        }
                    elif p["positionSide"] == "SHORT":
                        position["short"] = {
                            "size": p["contracts"],
                            "price": p["entryPrice"],
                            "liquidation_price": p["liquidationPrice"]
                            if p["liquidationPrice"]
                            else 0.0,
                        }
            position["wallet_balance"] = balance[self.quote]["total"]
            return position
        except Exception as e:
            logging.error(f"error fetching pos or balance {e}")
            print_async_exception(positions)
            print_async_exception(balance)
            traceback.print_exc()

    async def execute_orders(self, orders: [dict]) -> [dict]:
        if len(orders) == 0:
            return []
        if len(orders) == 1:
            return [await self.execute_order(orders[0])]
        return await self.execute_batch_orders(orders)

    async def execute_order(self, order: dict) -> dict:
        executed = None
        try:
            executed = await getattr(self.okx, f"create_limit {order['side']}_order")(
                symbol=self.symbol,
                price=order["price"],
                amount=order["qty"],
                params={
                    "posSide": order["position_side"],
                    "ordType": "post_only",
                    "reduceOnly": "close" in order["custom_id"],
                    "clOrdId": f"{order['custom_id']}_{str(int(time() * 1000))[8:]}_{int(np.random.random() * 1000)}",
                },
            )
            return executed
            return {
                "symbol": self.symbol,
                "side": executed["side"].lower(),
                "position_side": executed["positionSide"].lower().replace("short", "short"),
                "type": executed["type"].lower(),
                "qty": float(executed["origQty"]),
                "order_id": int(executed["orderId"]),
                "price": float(executed["price"]),
            }
        except Exception as e:
            print(f"error executing order {order} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return {}

    async def execute_batch_orders(self, orders: [dict]) -> [dict]:
        executed = None
        try:
            to_execute = []
            for order in orders:
                params = {
                    "symbol": self.symbol,
                    "side": order["side"].upper(),
                    "positionSide": order["position_side"].replace("short", "short").upper(),
                    "type": order["type"].upper(),
                    "quantity": str(order["qty"]),
                }
                if params["type"] == "LIMIT":
                    params["timeInForce"] = "GTX"
                    params["price"] = order["price"]
                if "custom_id" in order:
                    params[
                        "newClientOrderId"
                    ] = f"{order['custom_id']}_{str(int(time() * 1000))[8:]}_{int(np.random.random() * 1000)}"
                to_execute.append(params)
            executed = await self.private_post(
                self.endpoints["batch_orders"], {"batchOrders": to_execute}, data_=True
            )
            return [
                {
                    "symbol": self.symbol,
                    "side": ex["side"].lower(),
                    "position_side": ex["positionSide"].lower().replace("short", "short"),
                    "type": ex["type"].lower(),
                    "qty": float(ex["origQty"]),
                    "order_id": int(ex["orderId"]),
                    "price": float(ex["price"]),
                }
                for ex in executed
            ]
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
            cancellations = await self.private_delete(
                self.endpoints["batch_orders"],
                {"symbol": symbol, "orderIdList": [order["order_id"] for order in orders]},
                data_=True,
            )
            return [
                {
                    "symbol": symbol,
                    "side": cancellation["side"].lower(),
                    "order_id": int(cancellation["orderId"]),
                    "position_side": cancellation["positionSide"].lower().replace("short", "short"),
                    "qty": float(cancellation["origQty"]),
                    "price": float(cancellation["price"]),
                }
                for cancellation in cancellations
            ]
        except Exception as e:
            logging.error(f"error cancelling orders {orders} {e}")
            print_async_exception(cancellations)
            traceback.print_exc()
            return []

    async def fetch_fills(
        self,
        symbol=None,
        limit: int = 1000,
        from_id: int = None,
        start_time: int = None,
        end_time: int = None,
    ):
        return []
        params = {
            "symbol": self.symbol if symbol is None else symbol,
            "limit": min(100, limit) if self.inverse else limit,
        }
        if from_id is not None:
            params["fromId"] = max(0, from_id)
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(min(end_time, start_time + 1000 * 60 * 60 * 24 * 6.99))
        try:
            fetched = await self.private_get(self.endpoints["fills"], params)
            fills = [
                {
                    "symbol": x["symbol"],
                    "id": int(x["id"]),
                    "order_id": int(x["orderId"]),
                    "side": x["side"].lower(),
                    "price": float(x["price"]),
                    "qty": float(x["qty"]),
                    "realized_pnl": float(x["realizedPnl"]),
                    "cost": float(x["baseQty"]) if self.inverse else float(x["quoteQty"]),
                    "fee_paid": float(x["commission"]),
                    "fee_token": x["commissionAsset"],
                    "timestamp": int(x["time"]),
                    "position_side": x["positionSide"].lower().replace("short", "short"),
                    "is_maker": x["maker"],
                }
                for x in fetched
            ]
        except Exception as e:
            print("error fetching fills", e)
            traceback.print_exc()
            return []
        return fills

    async def get_all_income(
        self,
        symbol: str = None,
        start_time: int = None,
        income_type: str = "realized_pnl",
        end_time: int = None,
    ):
        income = []
        while True:
            fetched = await self.fetch_income(
                symbol=symbol,
                start_time=start_time,
                income_type=income_type,
                limit=1000,
            )
            print_(["fetched income", ts_to_date(fetched[0]["timestamp"])])
            if fetched == income[-len(fetched) :]:
                break
            income += fetched
            if len(fetched) < 1000:
                break
            start_time = income[-1]["timestamp"]
        income_d = {e["transaction_id"]: e for e in income}
        return sorted(income_d.values(), key=lambda x: x["timestamp"])

    async def fetch_income(
        self,
        symbol: str = None,
        income_type: str = None,
        limit: int = 1000,
        start_time: int = None,
        end_time: int = None,
    ):
        params = {"limit": limit}
        if symbol is not None:
            params["symbol"] = symbol
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        if income_type is not None:
            params["incomeType"] = income_type.upper()
        try:
            fetched = await self.private_get(self.endpoints["income"], params)
            return [
                {
                    "symbol": e["symbol"],
                    "income_type": e["incomeType"].lower(),
                    "income": float(e["income"]),
                    "token": e["asset"],
                    "timestamp": float(e["time"]),
                    "info": e["info"],
                    "transaction_id": float(e["tranId"]),
                    "trade_id": float(e["tradeId"]) if e["tradeId"] != "" else 0,
                }
                for e in fetched
            ]
        except Exception as e:
            print("error fetching income: ", e)
            traceback.print_exc()
            return []
        return income

    async def fetch_account(self):
        try:
            return await self.private_get(
                self.endpoints["account"], base_endpoint=self.spot_base_endpoint
            )
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
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)
        try:
            fetched = await self.public_get(self.endpoints["ticks"], params)
        except Exception as e:
            print("error fetching ticks a", e)
            traceback.print_exc()
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
        self, symbol: str = None, start_time: int = None, interval="1m", limit=300
    ):

        intervals = ['1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H']
        assert interval in intervals
        params={'bar': interval, 'limit': limit}
        if start_time is not None:
            params['after'] = str(start_time)
        print(params)
        fetched = await self.okx.fetch_ohlcv(symbol=self.symbol, params=params)
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
        params = {"type": type_.upper(), "amount": amount, "asset": asset}
        return await self.private_post(
            self.endpoints["transfer"], params, base_endpoint=self.spot_base_endpoint
        )

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
            print("error in websocket tick", e, data)
        return []

    async def beat_heart_user_stream(self) -> None:
        while True:
            await asyncio.sleep(60 + np.random.randint(60 * 9, 60 * 14))
            await self.init_user_stream()

    async def init_user_stream(self) -> None:
        try:
            response = await self.private_post(self.endpoints["listen_key"])
            self.listen_key = response["listenKey"]
            self.endpoints["websocket_user"] = self.endpoints["websocket"] + self.listen_key
        except Exception as e:
            traceback.print_exc()
            print_(["error fetching listen key", e])

    def standardize_user_stream_event(self, event: dict) -> dict:
        standardized = {}
        if "e" in event:
            if event["e"] == "ACCOUNT_UPDATE":
                if "a" in event and "B" in event["a"]:
                    for x in event["a"]["B"]:
                        if x["a"] == self.margin_coin:
                            standardized["wallet_balance"] = float(x["cw"])
                if event["a"]["m"] == "ORDER":
                    for x in event["a"]["P"]:
                        if x["s"] != self.symbol:
                            standardized["other_symbol"] = x["s"]
                            standardized["other_type"] = "account_update"
                            continue
                        if x["ps"] == "LONG":
                            standardized["psize_long"] = float(x["pa"])
                            standardized["pprice_long"] = float(x["ep"])
                        elif x["ps"] == "SHORT":
                            standardized["psize_short"] = float(x["pa"])
                            standardized["pprice_short"] = float(x["ep"])
            elif event["e"] == "ORDER_TRADE_UPDATE":
                if event["o"]["s"] == self.symbol:
                    if event["o"]["X"] == "NEW":
                        standardized["new_open_order"] = {
                            "order_id": int(event["o"]["i"]),
                            "symbol": event["o"]["s"],
                            "price": float(event["o"]["p"]),
                            "qty": float(event["o"]["q"]),
                            "type": event["o"]["o"].lower(),
                            "side": event["o"]["S"].lower(),
                            "position_side": event["o"]["ps"].lower().replace("short", "short"),
                            "timestamp": int(event["o"]["T"]),
                        }
                    elif event["o"]["X"] in ["CANCELED", "EXPIRED"]:
                        standardized["deleted_order_id"] = int(event["o"]["i"])
                    elif event["o"]["X"] == "FILLED":
                        standardized["deleted_order_id"] = int(event["o"]["i"])
                        standardized["filled"] = True
                    elif event["o"]["X"] == "PARTIALLY_FILLED":
                        standardized["deleted_order_id"] = int(event["o"]["i"])
                        standardized["partially_filled"] = True
                else:
                    standardized["other_symbol"] = event["o"]["s"]
                    standardized["other_type"] = "order_update"
        return standardized
