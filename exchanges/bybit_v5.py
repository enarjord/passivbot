import asyncio
import json
import traceback
from time import time
from typing import Union, List, Dict
from uuid import uuid4

import numpy as np

from njit_funcs import round_, calc_diff
from passivbot import Bot, logging
from procedures import print_async_exception, print_
from pure_funcs import ts_to_date, sort_dict_keys, date_to_ts, determine_pos_side_ccxt


import ccxt.async_support as ccxt

assert (
    ccxt.__version__ == "4.0.57"
), f"Currently ccxt {ccxt.__version__} is installed. Please pip reinstall requirements.txt or install ccxt v4.0.57 manually"


class BybitBot(Bot):
    def __init__(self, config: dict):
        self.exchange = "bybit"
        self.market_type = config["market_type"] = "linear_perpetual"
        self.inverse = config["inverse"] = False

        self.max_n_orders_per_batch = 20

        super().__init__(config)
        self.cc = getattr(ccxt, "bybit")(
            {
                "apiKey": self.key,
                "secret": self.secret,
                "headers": {"referer": self.broker_code} if self.broker_code else {},
            }
        )

    def init_market_type(self):
        if not self.symbol.endswith("USDT"):
            raise Exception(f"unsupported symbol {self.symbol}")

    async def _init(self):
        info = await self.cc.fetch_markets()
        self.symbol_id = self.symbol
        for elm in info:
            if elm["id"] == self.symbol_id and elm["type"] == "swap":
                break
        else:
            raise Exception(f"unsupported symbol {self.symbol}")
        self.symbol = elm["symbol"]
        self.max_leverage = elm["limits"]["leverage"]["max"]
        self.coin = elm["base"]
        self.quote = elm["quote"]
        self.price_step = self.config["price_step"] = elm["precision"]["price"]
        self.qty_step = self.config["qty_step"] = elm["precision"]["amount"]
        self.min_qty = self.config["min_qty"] = elm["limits"]["amount"]["min"]
        self.min_cost = self.config["min_cost"] = (
            0.1 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
        )
        self.margin_coin = self.quote
        await super()._init()

    async def fetch_ticker(self, symbol=None):
        ticker = None
        try:
            ticker = await self.cc.fetch_ticker(symbol=self.symbol if symbol is None else symbol)
            return ticker
        except Exception as e:
            logging.error(f"error fetching ticker {e}")
            print_async_exception(ticker)
            return None

    async def init_order_book(self):
        await self.update_ticker()

    async def fetch_open_orders(self) -> [dict]:
        open_orders = None
        try:
            open_orders = await self.cc.fetch_open_orders(symbol=self.symbol)
            return [
                {
                    "order_id": e["id"],
                    "custom_id": e["clientOrderId"],
                    "symbol": e["symbol"],
                    "price": e["price"],
                    "qty": e["amount"],
                    "type": e["type"],
                    "side": e["side"],
                    "position_side": determine_pos_side_ccxt(e),
                    "timestamp": e["timestamp"],
                }
                for e in open_orders
            ]
        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            print_async_exception(open_orders)
            traceback.print_exc()
            return False

    async def transfer_from_derivatives_to_spot(self, coin: str, amount: float):
        return

    async def get_server_time(self):
        server_time = None
        try:
            server_time = await self.cc.fetch_time()
            return server_time
        except Exception as e:
            logging.error(f"error fetching server time {e}")
            print_async_exception(server_time)
            traceback.print_exc()

    async def fetch_position(self) -> dict:
        positions, balance = None, None
        try:
            positions, balance = await asyncio.gather(
                self.cc.fetch_positions(self.symbol), self.cc.fetch_balance()
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
                            "price": 0.0 if p["entryPrice"] is None else p["entryPrice"],
                            "liquidation_price": p["liquidationPrice"]
                            if p["liquidationPrice"]
                            else 0.0,
                        }
                    elif p["side"] == "short":
                        position["short"] = {
                            "size": -abs(p["contracts"]),
                            "price": 0.0 if p["entryPrice"] is None else p["entryPrice"],
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
        return

        positions, balance = None, None
        try:
            positions, balance = await asyncio.gather(
                self.cc.fetch_positions(),
                self.cc.fetch_balance(),
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
                    elif p["side"] == "short":
                        position["short"] = {
                            "size": p["contracts"],
                            "price": p["entryPrice"],
                            "liquidation_price": p["liquidationPrice"]
                            if p["liquidationPrice"]
                            else 0.0,
                        }
            if balance:
                for elm in balance["info"]["data"]:
                    for elm2 in elm["details"]:
                        if elm2["ccy"] == self.quote:
                            position["wallet_balance"] = float(elm2["cashBal"])
                            break
            return position
        except Exception as e:
            logging.error(f"error fetching pos or balance {e}")
            print_async_exception(positions)
            print_async_exception(balance)
            traceback.print_exc()

    async def execute_orders(self, orders: [dict]) -> [dict]:
        return

    async def execute_order(self, order: dict) -> dict:
        executed = None
        try:
            executed = await self.cc.create_limit_order(
                symbol=order['symbol'] if 'symbol' in order else self.symbol,
                side=order['side'],
                amount=abs(order['qty']),
                price=order['price'],
                params={
                    'positionIdx': 1 if order['position_side'] == 'long' else 2,
                    'timeInForce': 'postOnly',
                    'orderLinkId': order['custom_id'] + str(uuid4()),
                }
            )
            return executed
        except Exception as e:
            logging.error(f"error executing order {order} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return {}

    async def execute_cancellations(self, orders: [dict]) -> [dict]:
        return

    async def execute_cancellation(self, order: dict) -> dict:
        return

    async def fetch_account(self):
        return

    async def fetch_ticks(self, from_id: int = None, do_print: bool = True):
        return

    async def fetch_ohlcvs(
        self, symbol: str = None, start_time: int = None, interval="1m", limit=1000
    ):
        ohlcvs = None
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
        assert interval in interval_map, f"unsupported timeframe {interval}"
        try:
            ohlcvs = await self.cc.fetch_ohlcv(
                self.symbol if symbol is None else symbol,
                timeframe=interval_map[interval],
                limit=limit,
                params={} if start_time is None else {"startTime": int(start_time)},
            )
            keys = ["timestamp", "open", "high", "low", "close", "volume"]
            return [{k: elm[i] for i, k in enumerate(keys)} for elm in ohlcvs]
        except:
            logging.error(f"error fetching ohlcv {e}")
            print_async_exception(ohlcvs)
            traceback.print_exc()

    async def get_all_income(
        self,
        symbol: str = None,
        start_time: int = None,
        income_type: str = "Trade",
        end_time: int = None,
    ):
        return

    async def fetch_income(
        self,
        symbol: str = None,
        income_type: str = None,
        limit: int = 50,
        start_time: int = None,
        end_time: int = None,
        page=None,
    ):
        return

    async def fetch_latest_fills(self):
        fetched = None
        try:
            fetched = await self.cc.fetch_my_trades(symbol=self.symbol)
            fills = [
                {
                    "order_id": elm["id"],
                    "symbol": elm["symbol"],
                    "custom_id": elm["info"]["orderLinkId"],
                    "price": elm["price"],
                    "qty": elm["amount"],
                    "type": elm["type"],
                    "reduce_only": None,
                    "side": elm["side"].lower(),
                    "position_side": determine_pos_side_ccxt(elm),
                    "timestamp": elm["timestamp"],
                }
                for elm in fetched
                if elm["amount"] != 0.0 and elm["type"] is not None
            ]
            return sorted(fills, key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching latest fills {e}")
            print_async_exception(fetched)
            traceback.print_exc()

    async def fetch_fills(
        self,
        limit: int = 200,
        from_id: int = None,
        start_time: int = None,
        end_time: int = None,
    ):
        return []

    async def init_exchange_config(self):
        try:
            res = await self.cc.set_derivatives_margin_mode(
                marginMode="cross", symbol=self.symbol, params={"leverage": self.leverage}
            )
            logging.info(f"cross mode set {res}")
        except Exception as e:
            logging.error(f"error setting cross mode: {e}")
        try:
            res = await self.cc.set_position_mode(hedged=True)
            logging.info(f"hedge mode set {res}")
        except Exception as e:
            logging.error(f"error setting hedge mode: {e}")
        try:
            res = await self.cc.set_leverage(int(self.leverage), symbol=self.symbol)
            logging.info(f"leverage set {res}")
        except Exception as e:
            logging.error(f"error setting leverage: {e}")
