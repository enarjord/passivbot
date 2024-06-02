from passivbot_multi import Passivbot, logging
from uuid import uuid4
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import pprint
import asyncio
import traceback
import numpy as np
from pure_funcs import (
    multi_replace,
    floatify,
    ts_to_date_utc,
    calc_hash,
    determine_pos_side_ccxt,
)
from njit_funcs import calc_diff, round_
from procedures import print_async_exception, utc_ms, assert_correct_ccxt_version

assert_correct_ccxt_version(ccxt=ccxt_async)


class BingXBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.ccp = getattr(ccxt_pro, self.exchange)(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
                "headers": {"X-SOURCE-KEY": self.broker_code} if self.broker_code else {},
            }
        )
        self.ccp.options["defaultType"] = "swap"
        self.cca = getattr(ccxt_async, self.exchange)(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
                "headers": {"X-SOURCE-KEY": self.broker_code} if self.broker_code else {},
            }
        )
        self.cca.options["defaultType"] = "swap"
        self.custom_id_max_length = 40

    def set_market_specific_settings(self):
        super().set_market_specific_settings()
        for symbol in self.markets_dict:
            elm = self.markets_dict[symbol]
            self.symbol_ids[symbol] = elm["id"]
            self.price_steps[symbol] = round(1.0 / (10 ** elm["precision"]["price"]), 12)
            self.qty_steps[symbol] = round(1.0 / (10 ** elm["precision"]["amount"]), 12)
            self.min_qtys[symbol] = elm["limits"]["amount"]["min"]
            self.min_costs[symbol] = elm["limits"]["cost"]["min"]
            self.c_mults[symbol] = 1.0

    async def start_websockets(self):
        await asyncio.gather(
            self.watch_balance(),
            self.watch_orders(),
            self.watch_tickers(),
        )

    async def watch_balance(self):
        res = None
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_balance()
                if "info" in res:
                    if "data" in res["info"] and "balance" in res["info"]["data"]:
                        res[self.quote]["total"] = float(res["info"]["data"]["balance"]["balance"])
                    else:
                        for elm in res["info"]:
                            if elm["a"] == self.quote:
                                res[self.quote]["total"] = float(elm["wb"])
                                break
                self.handle_balance_update(res)
            except Exception as e:
                print(f"exception watch_balance", e)
                print_async_exception(res)
                traceback.print_exc()

    async def watch_orders(self):
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_orders()
                for i in range(len(res)):
                    res[i]["position_side"] = res[i]["info"]["ps"].lower()
                    res[i]["qty"] = res[i]["amount"]
                self.handle_order_update(res)
            except Exception as e:
                print(f"exception watch_orders", e)
                traceback.print_exc()

    async def watch_tickers(self):
        self.WS_ticker_tasks = {}
        while not self.stop_websocket:
            current_symbols = set(self.active_symbols)
            started_symbols = set(self.WS_ticker_tasks.keys())

            # Start watch_ticker tasks for new symbols
            for symbol in current_symbols - started_symbols:
                task = asyncio.create_task(self.watch_ticker(symbol))
                self.WS_ticker_tasks[symbol] = task
                logging.info(f"Started watching ticker for symbol: {symbol}")

            # Cancel tasks for symbols that are no longer active
            for symbol in started_symbols - current_symbols:
                self.WS_ticker_tasks[symbol].cancel()
                del self.WS_ticker_tasks[symbol]
                logging.info(f"Stopped watching ticker for symbol: {symbol}")

            # Wait a bit before checking again
            await asyncio.sleep(1)  # Adjust sleep time as needed

    async def watch_ticker(self, symbol):
        while not self.stop_websocket and symbol in self.active_symbols:
            try:
                res = await self.ccp.watch_ticker(symbol)
                self.handle_ticker_update(res)
            except Exception as e:
                logging.error(f"exception watch_ticker {symbol} {str(e)}")
                traceback.print_exc()
                await asyncio.sleep(1)
            await asyncio.sleep(0.1)

    async def fetch_open_orders(self, symbol: str = None):
        fetched = None
        open_orders = []
        try:
            fetched = await self.cca.swap_v2_private_get_trade_openorders()
            fetched = self.cca.parse_orders(
                fetched["data"]["orders"], market={"spot": False, "quote": self.quote, "symbol": None}
            )
            for i in range(len(fetched)):
                fetched[i]["position_side"] = fetched[i]["info"]["positionSide"].lower()
                fetched[i]["qty"] = float(fetched[i]["amount"])
                fetched[i]["price"] = float(fetched[i]["price"])
                fetched[i]["timestamp"] = float(fetched[i]["timestamp"])
                fetched[i]["reduce_only"] = (
                    fetched[i]["side"] == "sell" and fetched[i]["position_side"] == "long"
                ) or (fetched[i]["side"] == "buy" and fetched[i]["position_side"] == "short")
            return sorted(fetched, key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_positions(self) -> ([dict], float):
        # also fetches balance
        fetched_positions, fetched_balance = None, None
        try:
            fetched_positions, fetched_balance = await asyncio.gather(
                self.cca.fetch_positions(),
                self.cca.fetch_balance(),
            )
            balance = float(fetched_balance["info"]["data"]["balance"]["balance"])
            fetched_positions = [x for x in fetched_positions if x["marginMode"] == "cross"]
            for i in range(len(fetched_positions)):
                fetched_positions[i]["position_side"] = fetched_positions[i]["side"]
                fetched_positions[i]["size"] = fetched_positions[i]["contracts"]
                fetched_positions[i]["price"] = fetched_positions[i]["entryPrice"]
            return fetched_positions, balance
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
        # intervals: 1,3,5,15,30,60,120,240,360,720,D,M,W
        fetched = None
        try:
            fetched = await self.cca.fetch_ohlcv(symbol, timeframe=timeframe, limit=1000)
            return fetched
        except Exception as e:
            logging.error(f"error fetching ohlcv for {symbol} {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_pnls(
        self,
        start_time: int = None,
        end_time: int = None,
    ):
        limit = 1000
        if start_time is None and end_time is None:
            return await self.fetch_pnl()
        all_fetched = {}
        while True:
            fetched = await self.fetch_pnl(start_time=start_time, end_time=end_time)
            if fetched == []:
                break
            for elm in fetched:
                all_fetched[elm["id"]] = elm
            if fetched[0]["timestamp"] <= start_time:
                break
            if len(fetched) < limit:
                break
            logging.info(f"debug fetching income {ts_to_date_utc(fetched[-1]['timestamp'])}")
            end_time = fetched[0]["timestamp"]
        return sorted(all_fetched.values(), key=lambda x: x["timestamp"])

    async def fetch_pnl(
        self,
        start_time: int = None,
        end_time: int = None,
    ):
        fetched = None
        # if there are more fills in timeframe than 100, it will fetch latest
        try:
            if end_time is None:
                end_time = utc_ms() + 1000 * 60 * 60 * 24
            if start_time is None:
                start_time = end_time - 1000 * 60 * 60 * 24 * 6.99
            start_time = max(start_time, end_time - 1000 * 60 * 60 * 24 * 6.99)  # max 7 days fetch
            params = {"startTime": int(start_time), "endTime": int(end_time), "limit": 1000}
            fetched = await self.cca.swap_v2_private_get_trade_allorders(params=params)
            fetched = floatify(fetched["data"]["orders"])
            for i in range(len(fetched)):
                fetched[i]["pnl"] = fetched[i]["profit"]
                fetched[i]["timestamp"] = fetched[i]["updateTime"]
                fetched[i]["id"] = fetched[i]["orderId"]
                fetched[i]["symbol_id"] = fetched[i]["symbol"]
                fetched[i]["symbol"] = self.symbol_ids_inv[fetched[i]["symbol"]]
            return sorted(fetched, key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching pnl {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def execute_cancellation(self, order: dict) -> dict:
        executed = None
        try:
            executed = await self.cca.cancel_order(order["id"], symbol=order["symbol"])
            for key in ["symbol", "side", "position_side", "qty", "price"]:
                if key not in executed or executed[key] is None:
                    executed[key] = order[key]
            return executed
        except Exception as e:
            if '"sCode":"51400"' in e.args[0]:
                logging.info(e.args[0])
                return {}
            logging.error(f"error cancelling order {order} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return {}

    async def execute_cancellations(self, orders: [dict]) -> [dict]:
        if len(orders) > self.config["max_n_cancellations_per_batch"]:
            # prioritize cancelling reduce-only orders
            try:
                reduce_only_orders = [x for x in orders if x["reduce_only"]]
                rest = [x for x in orders if not x["reduce_only"]]
                orders = (reduce_only_orders + rest)[: self.config["max_n_cancellations_per_batch"]]
            except Exception as e:
                logging.error(f"debug filter cancellations {e}")
        return await self.execute_multiple(
            orders, "execute_cancellation", self.config["max_n_cancellations_per_batch"]
        )

    async def execute_order(self, order: dict) -> dict:
        executed = None
        try:
            executed = await self.cca.create_limit_order(
                symbol=order["symbol"],
                side=order["side"],
                amount=order["qty"],
                price=order["price"],
                params={
                    "positionSide": order["position_side"].upper(),
                    "clientOrderID": order["custom_id"],
                    "timeInForce": "PostOnly",
                },
            )
            if "symbol" not in executed or executed["symbol"] is None:
                executed["symbol"] = order["symbol"]
            for key in ["side", "position_side", "qty", "price"]:
                if key not in executed or executed[key] is None:
                    executed[key] = order[key]
            return executed
        except Exception as e:
            if '"code":101400' in str(e):
                sym = order["symbol"]
                new_min_qty = round_(
                    max(self.min_qtys[sym], order["qty"]) + self.qty_steps[sym], self.qty_steps[sym]
                )
                logging.info(
                    f"successfully caught order size error, code 101400. Adjusting min_qty from {self.min_qtys[sym]} to {new_min_qty}..."
                )
                self.min_qtys[sym] = new_min_qty
                logging.error(f"{order} {e}")
                return {}
            logging.error(f"error executing order {order} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return {}

    async def execute_orders(self, orders: [dict]) -> [dict]:
        return await self.execute_multiple(
            orders, "execute_order", self.config["max_n_creations_per_batch"]
        )

    async def update_exchange_config_by_symbols(self, symbols):

        coros_to_call_lev, coros_to_call_margin_mode = {}, {}
        for symbol in symbols:
            try:
                coros_to_call_margin_mode[symbol] = asyncio.create_task(
                    self.cca.set_margin_mode(
                        "cross",
                        symbol=symbol,
                    )
                )
            except Exception as e:
                logging.error(f"{symbol}: error setting cross mode {e}")
            try:
                coros_to_call_lev[symbol] = asyncio.create_task(
                    self.cca.set_leverage(
                        int(self.live_configs[symbol]["leverage"]),
                        symbol=symbol,
                        params={"side": "LONG"},
                    )
                )
            except Exception as e:
                logging.error(f"{symbol}: a error setting leverage long {e}")
            try:
                coros_to_call_lev[symbol] = asyncio.create_task(
                    self.cca.set_leverage(
                        int(self.live_configs[symbol]["leverage"]),
                        symbol=symbol,
                        params={"side": "SHORT"},
                    )
                )
            except Exception as e:
                logging.error(f"{symbol}: a error setting leverage short {e}")
        for symbol in symbols:
            res = None
            to_print = ""
            try:
                res = await coros_to_call_lev[symbol]
                to_print += f" set leverage {res} "
            except Exception as e:
                if '"retCode":110043' in e.args[0]:
                    to_print += f" leverage: {e}"
                else:
                    logging.error(f"{symbol} error setting leverage {e}")
            try:
                res = await coros_to_call_margin_mode[symbol]
                to_print += f"set cross mode {res}"
            except Exception as e:
                if '"retCode":110026' in e.args[0]:
                    to_print += f" set cross mode: {res} {e}"
                else:
                    logging.error(f"{symbol} error setting cross mode {res} {e}")
            if to_print:
                logging.info(f"{symbol}: {to_print}")

    async def update_exchange_config(self):
        pass
