from passivbot import Passivbot, logging
from uuid import uuid4
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import asyncio
import traceback
import numpy as np
from pure_funcs import (
    floatify,
    ts_to_date_utc,
    calc_hash,
    shorten_custom_id,
    hysteresis_rounding,
)
from procedures import print_async_exception, utc_ms, assert_correct_ccxt_version

assert_correct_ccxt_version(ccxt=ccxt_async)


class DefxBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.custom_id_max_length = 36  # adjust if needed
        self.quote = "USDC"
        self.hedge_mode = False

    def create_ccxt_sessions(self):
        self.ccp = getattr(ccxt_pro, self.exchange)({
            "apiKey": self.user_info["key"],
            "secret": self.user_info["secret"],
        })
        self.cca = getattr(ccxt_async, self.exchange)({
            "apiKey": self.user_info["key"],
            "secret": self.user_info["secret"],
        })
        self.ccp.options["defaultType"] = "swap"
        self.cca.options["defaultType"] = "swap"

    def set_market_specific_settings(self):
        super().set_market_specific_settings()
        for symbol in self.markets_dict:
            elm = self.markets_dict[symbol]
            self.symbol_ids[symbol] = elm["id"]
            self.min_costs[symbol] = 0.1 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
            self.min_qtys[symbol] = elm["limits"]["amount"]["min"]
            self.qty_steps[symbol] = elm["precision"]["amount"]
            self.price_steps[symbol] = elm["precision"]["price"]
            self.c_mults[symbol] = elm["contractSize"]

    async def watch_balance(self):
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_balance()
                self.handle_balance_update(res)
            except Exception as e:
                logging.error(f"exception watch_balance {e}")
                traceback.print_exc()
                await asyncio.sleep(1)

    async def watch_orders(self):
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_orders()
                for order in res:
                    order["position_side"] = order.get("info", {}).get("positionSide", "unknown").lower()
                    order["qty"] = order["amount"]
                self.handle_order_update(res)
            except Exception as e:
                logging.error(f"exception watch_orders {e}")
                traceback.print_exc()
                await asyncio.sleep(1)

    async def fetch_open_orders(self, symbol: str = None):
        fetched = None
        open_orders = []
        try:
            fetched = await self.cca.fetch_open_orders(symbol=symbol)
            for order in fetched:
                order["position_side"] = order.get("info", {}).get("positionSide", "unknown").lower()
                order["qty"] = order["amount"]
            return sorted(fetched, key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_positions(self):
        fetched_positions, fetched_balance = None, None
        try:
            fetched_positions, fetched_balance = await asyncio.gather(
                self.cca.fetch_positions(),
                self.cca.fetch_balance(),
            )
            positions = []
            for p in fetched_positions:
                positions.append({
                    "symbol": self.get_symbol_id_inv(p["symbol"]),
                    "position_side": p.get("side", "unknown"),
                    "size": float(p["contracts"]),
                    "price": float(p["entryPrice"]),
                })
            balance = float(fetched_balance[self.quote]["total"])
            return positions, balance
        except Exception as e:
            logging.error(f"error fetching positions and balance {e}")
            print_async_exception(fetched_positions)
            print_async_exception(fetched_balance)
            traceback.print_exc()
            return False

    async def fetch_tickers(self):
        fetched = None
        try:
            fetched = await self.cca.fetch_tickers()
            return fetched
        except Exception as e:
            logging.error(f"error fetching tickers {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_ohlcv(self, symbol: str, timeframe="1m"):
        try:
            return await self.cca.fetch_ohlcv(symbol, timeframe=timeframe, limit=1000)
        except Exception as e:
            logging.error(f"error fetching ohlcv for {symbol} {e}")
            traceback.print_exc()
            return False

    async def fetch_pnls(self, start_time=None, end_time=None, limit=None):
        # Placeholder: implement properly if Defx has separate PNL and fills
        return []

    async def execute_orders(self, orders: dict) -> dict:
        return await self.execute_multiple(
            orders, "execute_order", self.config["live"]["max_n_creations_per_batch"]
        )

    async def execute_order(self, order: dict) -> dict:
        order_type = order["type"] if "type" in order else "limit"
        executed = await self.cca.create_order(
            symbol=order["symbol"],
            type=order_type,
            side=order["side"],
            amount=abs(order["qty"]),
            price=order["price"],
            params={
                "timeInForce": "GTC",
                "reduceOnly": order["reduce_only"],
            },
        )
        if "info" in executed and "orderId" in executed["info"]:
            for k in ["price", "id", "side", "position_side"]:
                if k not in executed or executed[k] is None:
                    executed[k] = order[k]
            executed["qty"] = executed["amount"] if executed["amount"] else order["qty"]
            executed["timestamp"] = (
                executed["timestamp"] if executed["timestamp"] else self.get_exchange_time()
            )
        return executed

    async def execute_cancellation(self, order: dict) -> dict:
        # Placeholder for actual cancellation implementation
        return {}

    async def determine_utc_offset(self, verbose=True):
        # Placeholder for custom UTC offset logic
        self.utc_offset = 0

