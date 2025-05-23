from passivbot import Passivbot, logging
from uuid import uuid4
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import asyncio
import traceback
import numpy as np
import passivbot_rust as pbr
from pure_funcs import (
    floatify,
    ts_to_date_utc,
    calc_hash,
    shorten_custom_id,
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
        self.ccp = getattr(ccxt_pro, self.exchange)(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
            }
        )
        self.cca = getattr(ccxt_async, self.exchange)(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
            }
        )
        self.ccp.options["defaultType"] = "swap"
        self.cca.options["defaultType"] = "swap"

    def set_market_specific_settings(self):
        super().set_market_specific_settings()
        for symbol in self.markets_dict:
            elm = self.markets_dict[symbol]
            self.symbol_ids[symbol] = elm["id"]
            self.min_costs[symbol] = (
                0.1 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
            )
            self.min_qtys[symbol] = elm["limits"]["amount"]["min"]
            self.qty_steps[symbol] = elm["precision"]["amount"]
            self.price_steps[symbol] = elm["precision"]["price"]
            self.c_mults[symbol] = elm["contractSize"]
            self.max_leverage[symbol] = int(elm["limits"]["leverage"]["max"])

    async def watch_balance(self):
        # TODO
        return
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_balance()
                print("debug kkkk", res)
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
                    order["position_side"] = self.determine_pos_side(order)
                    order["qty"] = order["amount"]
                self.handle_order_update(res)
            except Exception as e:
                logging.error(f"exception watch_orders {e}")
                traceback.print_exc()
                await asyncio.sleep(1)

    def determine_pos_side(self, order):
        # non hedge mode
        if self.has_position("long", order["symbol"]):
            return "long"
        elif self.has_position("short", order["symbol"]):
            return "short"
        elif order["side"] == "buy":
            return "long"
        elif order["side"] == "sell":
            return "short"
        raise Exception(f"unknown side {order['side']}")

    async def fetch_open_orders(self, symbol: str = None):
        fetched = None
        open_orders = []
        try:
            fetched = await self.cca.fetch_open_orders(symbol=symbol)
            for order in fetched:
                order["position_side"] = self.determine_pos_side(order)
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
                positions.append(
                    {
                        **p,
                        **{
                            "symbol": p["symbol"],
                            "position_side": p["info"]["positionSide"].lower(),
                            "size": float(p["contracts"]),
                            "price": float(p["entryPrice"]),
                        },
                    }
                )
            balance = float(fetched_balance[self.quote]["total"]) + sum(
                [float(p["info"]["marginAmount"]) for p in fetched_positions]
            )
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

    async def fetch_ohlcvs_1m(self, symbol: str, limit=None):
        n_candles_limit = 1000 if limit is None else limit
        result = await self.cca.fetch_ohlcv(
            symbol,
            timeframe="1m",
            limit=n_candles_limit,
        )
        return result

    async def fetch_pnls(self, start_time=None, end_time=None, limit=None):
        # TODO: impl start_time and end_time
        res = await self.cca.fetch_my_trades()
        for i in range(len(res)):
            res[i]["qty"] = res[i]["amount"]
            res[i]["pnl"] = float(res[i]["info"]["pnl"])
            if res[i]["side"] == "buy":
                res[i]["position_side"] = "long" if res[i]["pnl"] == 0.0 else "short"
            elif res[i]["side"] == "sell":
                res[i]["position_side"] = "short" if res[i]["pnl"] == 0.0 else "long"
            else:
                raise Exception(f"invalid side {res[i]}")
        return res

    async def execute_orders(self, orders: dict) -> dict:
        return await self.execute_multiple(orders, "execute_order")

    async def execute_order(self, order: dict) -> dict:
        # order_type = order["type"] if "type" in order else "limit"
        order_type = "limit"  # only limit orders
        reduce_only = False  # reduceOnly=True gives server error
        params = {
            "symbol": order["symbol"],
            "type": order_type,
            "side": order["side"],
            "amount": abs(order["qty"]),
            "price": order["price"],
            "params": {
                "timeInForce": "GTC",
                "reduceOnly": reduce_only,
            },
        }
        print(params)
        executed = await self.cca.create_order(**params)
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
        executed = None
        try:
            executed = await self.cca.cancel_order(order["id"], symbol=order["symbol"])
            return {
                "symbol": executed["symbol"],
                "side": order["side"],
                "id": executed["id"],
                "position_side": order["position_side"],
                "qty": order["qty"],
                "price": order["price"],
            }
        except Exception as e:
            logging.error(f"error cancelling order {order} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return {}

    async def execute_cancellations(self, orders: [dict]) -> [dict]:
        if len(orders) == 0:
            return []
        if len(orders) == 1:
            return [await self.execute_cancellation(orders[0])]
        return await self.execute_multiple(orders, "execute_cancellation")

    async def determine_utc_offset(self, verbose=True):
        # returns millis to add to utc to get exchange timestamp
        # call some endpoint which includes timestamp for exchange's server
        # if timestamp is not included in self.cca.fetch_balance(),
        # implement method in exchange child class
        result = await self.cca.fetch_ticker("BTC/USDC:USDC")
        self.utc_offset = round((result["timestamp"] - utc_ms()) / (1000 * 60 * 60)) * (
            1000 * 60 * 60
        )
        if verbose:
            logging.info(f"Exchange time offset is {self.utc_offset}ms compared to UTC")

    async def update_exchange_config_by_symbols(self, symbols):
        coros_to_call_leverage = {}
        for symbol in symbols:
            try:
                params = {
                    "leverage": int(
                        min(
                            self.max_leverage[symbol],
                            self.live_configs[symbol]["leverage"],
                            pbr.round_up(
                                max(
                                    self.get_wallet_exposure_limit("long", symbol),
                                    self.get_wallet_exposure_limit("short", symbol),
                                )
                                * 1.1,
                                1,
                            ),
                        )
                    ),
                    "symbol": symbol,
                }
                print("debug update_exchange_config_by_symbols", params)
                coros_to_call_leverage[symbol] = asyncio.create_task(self.cca.set_leverage(**params))
            except Exception as e:
                logging.error(f"{symbol}: error setting leverage {e}")
        for symbol in symbols:
            res = None
            to_print = ""
            try:
                res = await coros_to_call_leverage[symbol]
                to_print += f"set leverage {res}"
            except Exception as e:
                if '"code":"59107"' in e.args[0]:
                    to_print += f" cross mode and leverage: {res} {e}"
                else:
                    logging.error(f"{symbol} error setting leverage {res} {e}")
            if to_print:
                logging.info(f"{symbol}: {to_print}")
        return
