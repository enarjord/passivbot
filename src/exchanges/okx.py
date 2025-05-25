from passivbot import Passivbot, logging
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
    shorten_custom_id,
)
from njit_funcs import calc_diff
from procedures import print_async_exception, utc_ms, assert_correct_ccxt_version

assert_correct_ccxt_version(ccxt=ccxt_async)


class OKXBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.order_side_map = {
            "buy": {"long": "open_long", "short": "close_short"},
            "sell": {"long": "close_long", "short": "open_short"},
        }
        self.custom_id_max_length = 32

    def create_ccxt_sessions(self):
        self.ccp = getattr(ccxt_pro, self.exchange)(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
                "password": self.user_info["passphrase"],
            }
        )
        self.ccp.options["defaultType"] = "swap"
        self.cca = getattr(ccxt_async, self.exchange)(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
                "password": self.user_info["passphrase"],
            }
        )
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

    async def watch_balance(self):
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_balance()
                res["USDT"]["total"] = float(
                    [x for x in res["info"]["data"][0]["details"] if x["ccy"] == self.quote][0][
                        "cashBal"
                    ]
                )
                self.handle_balance_update(res)
            except Exception as e:
                print(f"exception watch_balance", e)
                traceback.print_exc()
                await asyncio.sleep(1)

    async def watch_orders(self):
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_orders()
                for i in range(len(res)):
                    res[i]["position_side"] = res[i]["info"]["posSide"]
                    res[i]["qty"] = res[i]["amount"]
                self.handle_order_update(res)
            except Exception as e:
                print(f"exception watch_orders", e)
                traceback.print_exc()
                await asyncio.sleep(1)

    async def fetch_open_orders(self, symbol: str = None):
        fetched = None
        open_orders = []
        try:
            fetched = await self.cca.fetch_open_orders()
            for i in range(len(fetched)):
                fetched[i]["position_side"] = fetched[i]["info"]["posSide"]
                fetched[i]["qty"] = fetched[i]["amount"]
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
            for elm in fetched_balance["info"]["data"]:
                for elm2 in elm["details"]:
                    if elm2["ccy"] == self.quote:
                        balance = float(elm2["cashBal"])
                        break
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

    async def fetch_ohlcvs_1m(self, symbol: str, since: float = None, limit=None):
        n_candles_limit = 300 if limit is None else limit
        if since is None:
            result = await self.cca.fetch_ohlcv(symbol, timeframe="1m", limit=n_candles_limit)
            return result
        since = since // 60000 * 60000
        max_n_fetches = 5000 // n_candles_limit
        all_fetched = []
        for i in range(max_n_fetches):
            fetched = await self.cca.fetch_ohlcv(
                symbol, timeframe="1m", since=int(since), limit=n_candles_limit
            )
            all_fetched += fetched
            if len(fetched) < n_candles_limit:
                break
            since = fetched[-1][0]
        all_fetched_d = {x[0]: x for x in all_fetched}
        return sorted(all_fetched_d.values(), key=lambda x: x[0])

    async def fetch_pnls(self, start_time: int = None, end_time: int = None, limit=None):
        if limit is None:
            limit = 100
        if start_time is None and end_time is None:
            return await self.fetch_pnl()
        all_fetched = {}
        while True:
            fetched = await self.fetch_pnl(start_time=start_time, end_time=end_time)
            if fetched == []:
                break
            for elm in fetched:
                all_fetched[elm["id"]] = elm
            if len(fetched) < limit:
                break
            logging.info(f"debug fetching income {ts_to_date_utc(fetched[-1]['timestamp'])}")
            end_time = fetched[0]["timestamp"]
        return sorted(all_fetched.values(), key=lambda x: x["timestamp"])
        return sorted(
            [x for x in all_fetched.values() if x["pnl"] != 0.0], key=lambda x: x["timestamp"]
        )

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
                start_time = end_time - 1000 * 60 * 60 * 24 * 7
            fetched = await self.cca.fetch_my_trades(
                since=int(start_time), params={"until": int(end_time)}
            )
            for i in range(len(fetched)):
                fetched[i]["pnl"] = float(fetched[i]["info"]["fillPnl"])
                fetched[i]["position_side"] = fetched[i]["info"]["posSide"]
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
        return await self.execute_multiple(orders, "execute_cancellation")

    async def execute_order(self, order: dict) -> dict:
        return self.execute_orders([order])

    async def execute_orders(self, orders: [dict]) -> [dict]:
        if len(orders) == 0:
            return []
        to_execute = []
        orders = orders
        for order in orders:
            to_execute.append(
                {
                    "type": "limit",
                    "symbol": order["symbol"],
                    "side": order["side"],
                    "ordType": (
                        "post_only"
                        if self.config["live"]["time_in_force"] == "post_only"
                        else "limit"
                    ),
                    "amount": abs(order["qty"]),
                    "tdMode": "cross",
                    "price": order["price"],
                    "params": {
                        "tag": self.broker_code,
                        "posSide": order["position_side"],
                        "clOrdId": order["custom_id"],
                    },
                }
            )
        executed = None
        try:
            executed = await self.cca.create_orders(to_execute)
            return executed
        except Exception as e:
            logging.error(f"error executing orders {orders} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return []

        to_return = []
        for order, res in zip(orders, executed):
            try:
                if "status" in res and res["status"] == "rejected":
                    logging.info(f"order rejected: {res}")
                for key in order:
                    if key not in res or res[key] is None:
                        res[key] = order[key]
                to_return.append(res)
            except Exception as e:
                logging.error(f"error executing order {res} {e}")
                traceback.print_exc()
        return to_return

    async def update_exchange_config_by_symbols(self, symbols: [str]):
        coros_to_call_margin_mode = {}
        for symbol in symbols:
            try:
                coros_to_call_margin_mode[symbol] = asyncio.create_task(
                    self.cca.set_margin_mode(
                        "cross",
                        symbol=symbol,
                        params={"lever": int(self.live_configs[symbol]["leverage"])},
                    )
                )
            except Exception as e:
                logging.error(f"{symbol}: error setting cross mode and leverage {e}")
        for symbol in symbols:
            res = None
            to_print = ""
            try:
                res = await coros_to_call_margin_mode[symbol]
                to_print += f"set cross mode {res}"
            except Exception as e:
                if '"code":"59107"' in e.args[0]:
                    to_print += f" cross mode and leverage: {res} {e}"
                else:
                    logging.error(f"{symbol} error setting cross mode {res} {e}")
            if to_print:
                logging.info(f"{symbol}: {to_print}")

    async def update_exchange_config(self):
        try:
            res = await self.cca.set_position_mode(True)
            logging.info(f"set hedge mode {res}")
        except Exception as e:
            if '"code":"59000"' in e.args[0]:
                logging.info(f"margin mode: {e}")
            else:
                logging.error(f"error setting hedge mode {e}")

    def calc_ideal_orders(self):
        # okx has max 100 open orders. Drop orders whose pprice diff is greatest.
        ideal_orders = super().calc_ideal_orders()
        ideal_orders_tmp = []
        for s in ideal_orders:
            for x in ideal_orders[s]:
                ideal_orders_tmp.append({**x, **{"symbol": s}})
        ideal_orders_tmp = sorted(
            ideal_orders_tmp,
            key=lambda x: calc_diff(x["price"], self.get_last_price(x["symbol"])),
        )[:100]
        ideal_orders = {symbol: [] for symbol in self.active_symbols}
        for x in ideal_orders_tmp:
            ideal_orders[x["symbol"]].append(x)
        return ideal_orders

    def format_custom_ids(self, orders: [dict]) -> [dict]:
        # okx needs broker code at the beginning of the custom_id
        new_orders = []
        for order in orders:
            order["custom_id"] = (self.broker_code + uuid4().hex)[: self.custom_id_max_length]
            new_orders.append(order)
        return new_orders
