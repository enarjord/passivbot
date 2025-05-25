from passivbot import Passivbot, logging
from uuid import uuid4
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import pprint
import asyncio
import traceback
import json
import numpy as np
from pure_funcs import (
    multi_replace,
    floatify,
    ts_to_date_utc,
    calc_hash,
    shorten_custom_id,
    coin2symbol,
    symbol_to_coin,
)
from njit_funcs import (
    calc_diff,
    round_,
    round_up,
    round_dn,
    round_dynamic,
    round_dynamic_up,
    round_dynamic_dn,
)
from procedures import print_async_exception, utc_ms, assert_correct_ccxt_version
from sortedcontainers import SortedDict

assert_correct_ccxt_version(ccxt=ccxt_async)


class GateIOBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.ohlcvs_1m_init_duration_seconds = (
            120  # gateio has stricter rate limiting on fetching ohlcvs
        )
        self.hedge_mode = False
        self.config["live"]["max_n_cancellations_per_batch"] = min(
            self.config["live"]["max_n_cancellations_per_batch"], 20
        )
        self.config["live"]["max_n_creations_per_batch"] = min(
            self.config["live"]["max_n_creations_per_batch"], 10
        )

    def create_ccxt_sessions(self):
        self.ccp = getattr(ccxt_pro, self.exchange)(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
                "headers": {"X-Gate-Channel-Id": self.broker_code} if self.broker_code else {},
            }
        )
        self.ccp.options["defaultType"] = "swap"
        self.cca = getattr(ccxt_async, self.exchange)(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
                "headers": {"X-Gate-Channel-Id": self.broker_code} if self.broker_code else {},
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
            self.min_qtys[symbol] = (
                elm["precision"]["amount"]
                if elm["limits"]["amount"]["min"] is None
                else elm["limits"]["amount"]["min"]
            )
            self.qty_steps[symbol] = elm["precision"]["amount"]
            self.price_steps[symbol] = elm["precision"]["price"]
            self.c_mults[symbol] = elm["contractSize"]
            self.max_leverage[symbol] = elm["limits"]["leverage"]["max"]

    async def determine_utc_offset(self, verbose=True):
        # returns millis to add to utc to get exchange timestamp
        # call some endpoint which includes timestamp for exchange's server
        # if timestamp is not included in self.cca.fetch_balance(),
        # implement method in exchange child class
        result = await self.cca.fetch_ohlcv("BTC/USDT:USDT", timeframe="1m")
        self.utc_offset = round((result[-1][0] - utc_ms()) / (1000 * 60 * 60)) * (1000 * 60 * 60)
        if verbose:
            logging.info(f"Exchange time offset is {self.utc_offset}ms compared to UTC")

    async def watch_balance(self):
        # hyperliquid ccxt watch balance not supported.
        # relying instead on periodic REST updates
        res = None
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.cca.fetch_balance()
                res[self.quote]["total"] = float(res["info"]["marginSummary"]["accountValue"]) - sum(
                    [float(x["position"]["unrealizedPnl"]) for x in res["info"]["assetPositions"]]
                )
                self.handle_balance_update(res)
                await asyncio.sleep(10)
            except Exception as e:
                logging.error(f"exception watch_balance {res} {e}")
                traceback.print_exc()
                await asyncio.sleep(1)

    async def watch_orders(self):
        res = None
        while not self.stop_signal_received:
            if not self.ccp.uid:
                await asyncio.sleep(1)
                continue
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_orders()
                for i in range(len(res)):
                    res[i]["position_side"] = self.determine_pos_side(res[i])
                    res[i]["qty"] = res[i]["amount"]
                self.handle_order_update(res)
            except Exception as e:
                logging.error(f"exception watch_orders {res} {e}")
                traceback.print_exc()
                await asyncio.sleep(1)

    def determine_pos_side(self, order):
        if order["side"] == "buy":
            return "short" if order["reduceOnly"] else "long"
        if order["side"] == "sell":
            return "long" if order["reduceOnly"] else "short"
        raise Exception(f"unsupported order side {order['side']}")

    async def fetch_open_orders(self, symbol: str = None):
        fetched = None
        open_orders = []
        try:
            fetched = await self.cca.fetch_open_orders()
            for i in range(len(fetched)):
                fetched[i]["position_side"] = self.determine_pos_side(fetched[i])
                fetched[i]["qty"] = fetched[i]["amount"]
            return sorted(fetched, key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_positions(self) -> ([dict], float):
        positions, balance = None, None
        try:
            positions_fetched, balance = await asyncio.gather(
                self.cca.fetch_positions(), self.cca.fetch_balance()
            )
            if not hasattr(self, "uid") or not self.uid:
                self.uid = balance["info"][0]["user"]
                self.cca.uid = self.uid
                self.ccp.uid = self.uid
            balance = balance[self.quote]["total"]
            positions = []
            for x in positions_fetched:
                if x["contracts"] != 0.0:
                    x["size"] = x["contracts"]
                    x["price"] = x["entryPrice"]
                    x["position_side"] = x["side"]
                    positions.append(x)
            return positions, balance
        except Exception as e:
            logging.error(f"error fetching positions and balance {e}")
            print_async_exception(positions)
            print_async_exception(balance)
            traceback.print_exc()
            return False

    async def fetch_tickers(self):
        fetched = None
        try:
            fetched = await self.cca.fetch(
                "https://api.hyperliquid.xyz/info",
                method="POST",
                headers={"Content-Type": "application/json"},
                body=json.dumps({"type": "allMids"}),
            )
            return {
                coin2symbol(coin, self.quote): {
                    "bid": float(fetched[coin]),
                    "ask": float(fetched[coin]),
                    "last": float(fetched[coin]),
                }
                for coin in fetched
            }
        except Exception as e:
            logging.error(f"error fetching tickers {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_ohlcv(self, symbol: str, timeframe="1m"):
        # intervals: 1,3,5,15,30,60,120,240,360,720,D,M,W
        # fetches latest ohlcvs
        fetched = None
        str2int = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 60 * 4}
        n_candles = 480
        try:
            since = int(utc_ms() - 1000 * 60 * str2int[timeframe] * n_candles)
            fetched = await self.cca.fetch_ohlcv(symbol, timeframe=timeframe, since=since)
            return fetched
        except Exception as e:
            logging.error(f"error fetching ohlcv for {symbol} {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_ohlcvs_1m(self, symbol: str, limit=None):
        n_candles_limit = 1440 if limit is None else limit
        result = await self.cca.fetch_ohlcv(
            symbol,
            timeframe="1m",
            limit=n_candles_limit,
        )
        return result

    async def fetch_pnls(
        self,
        start_time: int = None,
        end_time: int = None,
        limit=None,
    ):
        if start_time is None:
            return await self.fetch_pnl(limit=limit)
        all_fetched = {}
        if limit is None:
            limit = 1000
        offset = 0
        while True:
            fetched = await self.fetch_pnl(offset=offset, limit=limit)
            if not fetched:
                break
            for elm in fetched:
                all_fetched[elm["id"]] = elm
            if len(fetched) < limit:
                break
            if fetched[0]["timestamp"] <= start_time:
                break
            logging.info(f"debug fetching pnls {ts_to_date_utc(fetched[-1]['timestamp'])}")
            offset += limit
        return sorted(all_fetched.values(), key=lambda x: x["timestamp"])

    async def fetch_pnl(
        self,
        offset=0,
        limit=None,
    ):
        fetched = None
        n_pnls_limit = 1000 if limit is None else limit
        try:
            fetched = await self.cca.fetch_closed_orders(
                limit=n_pnls_limit, params={"offset": offset}
            )
            for i in range(len(fetched)):
                fetched[i]["pnl"] = float(fetched[i]["info"]["pnl"])
                fetched[i]["position_side"] = self.determine_pos_side(fetched[i])
            return sorted(fetched, key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching pnl {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def execute_cancellation(self, order: dict) -> dict:
        return await self.execute_cancellations([order])

    async def execute_cancellations(self, orders: [dict]) -> [dict]:
        if not orders:
            return []
        res = None
        try:
            res = await self.cca.cancel_orders([x["id"] for x in orders])
            return res
            cancellations = []
            for order, elm in zip(orders, res):
                if elm["status"] != "rejected":
                    joined = order.copy()
                    for k, v in elm.items():
                        if k not in joined or not joined[k]:
                            joined[k] = v
                    cancellations.append(joined)
            return cancellations
        except Exception as e:
            logging.error(f"error executing cancellations {e} {orders}")
            print_async_exception(res)
            traceback.print_exc()

    def did_cancel_order(self, executed):
        try:
            return "status" in executed and executed["status"] != "rejected"
        except:
            return False

    async def execute_order(self, order: dict) -> dict:
        return await self.execute_orders([order])

    async def execute_orders(self, orders: [dict]) -> [dict]:
        if len(orders) == 0:
            return []
        to_execute = []
        for order in orders:
            order_type = order["type"] if "type" in order else "limit"
            params = {
                "reduce_only": order["reduce_only"],
            }
            if order_type == "limit":
                params["timeInForce"] = (
                    "poc" if self.config["live"]["time_in_force"] == "post_only" else "gtc"
                )
            to_execute.append(
                {
                    "symbol": order["symbol"],
                    "type": order_type,
                    "side": order["side"],
                    "amount": order["qty"],
                    "price": order["price"],
                    "params": params,
                }
            )
        res = await self.cca.create_orders(to_execute)
        return res
        executed = []
        for ex, order in zip(res, orders):
            if "info" in ex and ex["status"] in ["closed", "open"]:
                executed.append({**ex, **order})
        return executed

    def did_create_order(self, executed):
        try:
            return "status" in executed and executed["status"] != "rejected"
        except:
            return False

    async def update_exchange_config_by_symbols(self, symbols):
        return
        coros_to_call_margin_mode = {}
        for symbol in symbols:
            try:
                params = {
                    "leverage": int(
                        min(
                            self.max_leverage[symbol],
                            self.live_configs[symbol]["leverage"],
                        )
                    )
                }
                if self.user_info["is_vault"]:
                    params["vaultAddress"] = self.user_info["wallet_address"]
                coros_to_call_margin_mode[symbol] = asyncio.create_task(
                    self.cca.set_margin_mode("cross", symbol=symbol, params=params)
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
        pass

    def calc_ideal_orders(self):
        # hyperliquid needs custom price rounding
        ideal_orders = super().calc_ideal_orders()
        return ideal_orders
