from passivbot import Passivbot, logging
from uuid import uuid4
import passivbot_rust as pbr

round_ = pbr.round_
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import pprint
import asyncio
import traceback
import numpy as np
import json
import passivbot_rust as pbr
from copy import deepcopy
from utils import ts_to_date_utc, utc_ms
from pure_funcs import (
    floatify,
    calc_hash,
    determine_pos_side_ccxt,
    flatten,
    shorten_custom_id,
)
from procedures import print_async_exception, assert_correct_ccxt_version, load_broker_code

assert_correct_ccxt_version(ccxt=ccxt_async)


class BinanceBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.custom_id_max_length = 36

    def create_ccxt_sessions(self):
        self.broker_code_spot = load_broker_code("binance_spot")
        for ccx, ccxt_module in [("cca", ccxt_async), ("ccp", ccxt_pro)]:
            exchange_class = getattr(ccxt_module, "binanceusdm")
            setattr(
                self,
                ccx,
                exchange_class(
                    {
                        "apiKey": self.user_info["key"],
                        "secret": self.user_info["secret"],
                        "password": self.user_info["passphrase"],
                    }
                ),
            )
            getattr(self, ccx).options["defaultType"] = "swap"
            if self.broker_code:
                for key in ["future", "delivery", "swap", "option"]:
                    getattr(self, ccx).options["broker"][key] = "x-" + self.broker_code
            if self.broker_code_spot:
                for key in ["spot", "margin"]:
                    getattr(self, ccx).options["broker"][key] = "x-" + self.broker_code_spot

    async def print_new_user_suggestion(self):
        between_print_wait_ms = 1000 * 60 * 60 * 4
        if hasattr(self, "previous_user_suggestion_print_ts"):
            if utc_ms() - self.previous_user_suggestion_print_ts < between_print_wait_ms:
                return
        self.previous_user_suggestion_print_ts = utc_ms()

        res = None
        try:
            res = await self.cca.fapiprivate_get_apireferral_ifnewuser(
                params={"brokerid": self.broker_code}
            )
        except Exception as e:
            logging.error(f"failed to fetch fapiprivate_get_apireferral_ifnewuser {e}")
            print_async_exception(res)
            return
        if res["ifNewUser"] and res["rebateWorking"]:
            return
        lines = [
            "To support continued Passivbot development, please use a Binance account which",
            "1) was created after 2024-09-21 and",
            "2) either:",
            "  a) was created without a referral link, or",
            '  b) was created with referral ID: "TII4B07C".',
            " ",
            "Passivbot receives commissions from trades only for accounts meeting these criteria.",
            " ",
            json.dumps(res),
        ]
        front_pad = " " * 8 + "##"
        back_pad = "##"
        max_len = max([len(line) for line in lines])
        print("\n\n")
        print(front_pad + "#" * (max_len + 2) + back_pad)
        for line in lines:
            print(front_pad + " " + line + " " * (max_len - len(line) + 1) + back_pad)
        print(front_pad + "#" * (max_len + 2) + back_pad)
        print("\n\n")

    async def execute_to_exchange(self):
        res = await super().execute_to_exchange()
        await self.print_new_user_suggestion()
        return res

    def set_market_specific_settings(self):
        super().set_market_specific_settings()
        for symbol in self.markets_dict:
            elm = self.markets_dict[symbol]
            self.min_costs[symbol] = (
                0.1 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
            )
            self.min_qtys[symbol] = elm["limits"]["amount"]["min"]
            self.price_steps[symbol] = elm["precision"]["price"]
            self.qty_steps[symbol] = elm["precision"]["amount"]
            self.c_mults[symbol] = elm["contractSize"]

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
                if "Abnormal closure of client" not in str(e):
                    logging.error(f"exception watch_orders {e}")
                    traceback.print_exc()
                await asyncio.sleep(1)

    async def fetch_open_orders(self, symbol: str = None, all=False) -> [dict]:
        fetched = None
        open_orders = {}
        try:
            # binance has expensive fetch_open_orders without specified symbol
            if all:
                self.cca.options["warnOnFetchOpenOrdersWithoutSymbol"] = False
                logging.info(f"fetching all open orders for binance")
                fetched = await self.cca.fetch_open_orders()
                self.cca.options["warnOnFetchOpenOrdersWithoutSymbol"] = True
            else:
                symbols_ = set()
                symbols_.update([s for s in self.open_orders if self.open_orders[s]])
                symbols_.update([s for s in self.get_symbols_with_pos()])
                if hasattr(self, "active_symbols") and self.active_symbols:
                    symbols_.update(list(self.active_symbols))
                fetched = await asyncio.gather(
                    *[self.cca.fetch_open_orders(symbol=symbol) for symbol in sorted(symbols_)]
                )
                fetched = [x for sublist in fetched for x in sublist]
            for elm in fetched:
                elm["position_side"] = elm["info"]["positionSide"].lower()
                elm["qty"] = elm["amount"]
                open_orders[elm["id"]] = elm
            return sorted(open_orders.values(), key=lambda x: x["timestamp"])
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
                self.cca.fapiprivatev3_get_positionrisk(), self.cca.fetch_balance()
            )
            positions = []
            for elm in fetched_positions:
                if float(elm["positionAmt"]) != 0.0:
                    positions.append(
                        {
                            "symbol": self.get_symbol_id_inv(elm["symbol"]),
                            "position_side": elm["positionSide"].lower(),
                            "size": float(elm["positionAmt"]),
                            "price": float(elm["entryPrice"]),
                        }
                    )
            balance = float(fetched_balance["info"]["totalCrossWalletBalance"])
            if not hasattr(self, "previous_rounded_balance"):
                self.previous_rounded_balance = balance
            self.previous_rounded_balance = pbr.hysteresis_rounding(
                balance,
                self.previous_rounded_balance,
                self.hyst_rounding_balance_pct,
                self.hyst_rounding_balance_h,
            )
            return positions, self.previous_rounded_balance
        except Exception as e:
            logging.error(f"error fetching positions {e}")
            print_async_exception(fetched_positions)
            print_async_exception(fetched_balance)
            traceback.print_exc()
            return False

    async def fetch_tickers(self):
        fetched = None
        try:
            fetched = await self.cca.fapipublic_get_ticker_bookticker()
            tickers = {
                self.get_symbol_id_inv(elm["symbol"]): {
                    "bid": float(elm["bidPrice"]),
                    "ask": float(elm["askPrice"]),
                }
                for elm in fetched
            }
            for sym in tickers:
                tickers[sym]["last"] = np.random.choice([tickers[sym]["bid"], tickers[sym]["ask"]])
            return tickers
        except Exception as e:
            logging.error(f"error fetching tickers {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            if "bybit does not have market symbol" in str(e):
                # ccxt is raising bad symbol error
                # restart might help...
                raise Exception("ccxt gives bad symbol error... attempting bot restart")
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
        limit: int = None,
    ):
        pnls = await self.fetch_pnls_sub(start_time, end_time, limit)
        symbols = sorted(set(self.positions) | set([x["symbol"] for x in pnls]))
        tasks = {}
        for symbol in symbols:
            tasks[symbol] = asyncio.create_task(
                self.fetch_fills_sub(symbol, start_time, end_time, limit)
            )
        fills = {}
        for symbol in tasks:
            fills[symbol] = await tasks[symbol]
        fills = flatten(fills.values())
        if start_time:
            pnls = [x for x in pnls if x["timestamp"] >= start_time]
            fills = [x for x in fills if x["timestamp"] >= start_time]
        unified = {x["id"]: x for x in pnls}
        for x in fills:
            if x["id"] in unified:
                unified[x["id"]].update(x)
            else:
                unified[x["id"]] = x
        result = []
        for x in sorted(unified.values(), key=lambda x: x["timestamp"]):
            if "position_side" not in x:
                logging.info(f"debug: pnl without corresponding fill {x}")
                x["position_side"] = "unknown"
            result.append(x)
        return result

    async def fetch_pnls_sub(
        self,
        start_time: int = None,
        end_time: int = None,
        limit: int = None,
    ):
        # binance needs symbol specified for fetch fills
        # but can fetch pnls for all symbols
        # fetch fills for all symbols with pos
        # fetch pnls for all symbols
        # fills only needed for symbols with pos for trailing orders
        # binance returns at most 7 days worth of pnls per fetch unless both start_time and end_time are given
        if limit is None:
            limit = 1000
        else:
            limit = min(limit, 1000)
        if end_time is None:
            if start_time is None:
                return await self.fetch_pnl(limit=limit)
            end_time = self.get_exchange_time() + 1000 * 60 * 60
        all_fetched = {}
        week = 1000 * 60 * 60 * 24 * 7
        while True:
            fetched = await self.fetch_pnl(start_time, end_time, limit)
            if fetched == []:
                break
            if fetched[0]["tradeId"] in all_fetched and fetched[-1]["tradeId"] in all_fetched:
                break
            for elm in fetched:
                all_fetched[elm["tradeId"]] = elm
            if len(fetched) < limit:
                if start_time:
                    if end_time:
                        if end_time - start_time < week:
                            break
                    else:
                        if self.get_exchange_time() - start_time < week:
                            break
            logging.info(
                f"fetched {len(fetched)} pnls from {ts_to_date_utc(fetched[0]['timestamp'])[:19]} until {ts_to_date_utc(fetched[-1]['timestamp'])[:19]}"
            )
            start_time = fetched[-1]["timestamp"]
        return sorted(all_fetched.values(), key=lambda x: x["timestamp"])

    async def fetch_fills_sub(self, symbol, start_time=None, end_time=None, limit=None):
        try:
            if symbol not in self.markets_dict:
                return []
            # limit is max 1000
            # fetches at most 7 days worth
            max_limit = 1000
            limit = min(max_limit, limit) if limit else max_limit
            if start_time is None and end_time is None:
                fills = await self.cca.fetch_my_trades(symbol, limit=limit)
                all_fills = {x["id"]: x for x in fills}
            elif start_time is None:
                fills = await self.cca.fetch_my_trades(
                    symbol, limit=limit, params={"endTime": int(end_time)}
                )
                all_fills = {x["id"]: x for x in fills}
            else:
                if end_time is None:
                    end_time = self.get_exchange_time() + 1000 * 60 * 60
                all_fills = {}
                params = {}
                week = 1000 * 60 * 60 * 24 * 7.0
                start_time_sub = start_time
                while True:
                    param_start_time = int(min(start_time_sub, self.get_exchange_time() - 1000 * 60))
                    param_end_time = max(
                        param_start_time, int(min(end_time, start_time_sub + week * 0.999))
                    )
                    fills = await self.cca.fetch_my_trades(
                        symbol,
                        limit=limit,
                        params={
                            "startTime": param_start_time,
                            "endTime": param_end_time,
                        },
                    )
                    if not fills:
                        if end_time - start_time_sub < week * 0.9:
                            self.debug_print("debug fetch_fills_sub a", symbol)
                            break
                        else:
                            logging.info(
                                f"fetched 0 fills for {symbol} between {ts_to_date_utc(start_time_sub)[:19]} and {ts_to_date_utc(end_time)[:19]}"
                            )
                            start_time_sub += week
                            continue
                    if fills[0]["id"] in all_fills and fills[-1]["id"] in all_fills:
                        if end_time - start_time_sub < week * 0.9:
                            self.debug_print("debug fetch_fills_sub b", symbol)
                            break
                        else:
                            logging.info(
                                f"fetched 0 new fills for {symbol} between {ts_to_date_utc(start_time_sub)[:19]} and {ts_to_date_utc(end_time)[:19]}"
                            )
                            start_time_sub += week
                            continue
                    else:
                        for x in fills:
                            all_fills[x["id"]] = x
                    if end_time - start_time_sub < week * 0.9 and len(fills) < limit:
                        self.debug_print("debug fetch_fills_sub c", symbol)
                        break
                    start_time_sub = fills[-1]["timestamp"]
                    logging.info(
                        f"fetched {len(fills)} fill{'s' if len(fills) > 1 else ''} for {symbol} {ts_to_date_utc(fills[0]['timestamp'])[:19]}"
                    )
            all_fills = sorted(all_fills.values(), key=lambda x: x["timestamp"])
            for i in range(len(all_fills)):
                all_fills[i]["pnl"] = float(all_fills[i]["info"]["realizedPnl"])
                all_fills[i]["position_side"] = all_fills[i]["info"]["positionSide"].lower()
            return all_fills
        except Exception as e:
            logging.error(f"error with fetch_fills_sub {symbol} {e}")
            return []

    async def fetch_pnl(
        self,
        start_time: int = None,
        end_time: int = None,
        limit: int = None,
    ):
        # will fetch from start_time until end_time, earliest first
        # if start_time is None and end_time is None, will only fetch for last 7 days
        # if end_time is None, will fetch for more than 7 days
        # if start_time is None, will only fetch for last 7 days
        fetched = None
        max_limit = 1000
        if limit is None:
            limit = max_limit
        try:
            params = {"incomeType": "REALIZED_PNL", "limit": min(max_limit, limit)}
            if start_time is not None:
                params["startTime"] = int(start_time)
            if end_time is not None:
                params["endTime"] = int(end_time)
            fetched = await self.cca.fapiprivate_get_income(params=params)
            for i in range(len(fetched)):
                fetched[i]["symbol"] = self.get_symbol_id_inv(fetched[i]["symbol"])
                fetched[i]["pnl"] = float(fetched[i]["income"])
                fetched[i]["timestamp"] = float(fetched[i]["time"])
                fetched[i]["id"] = fetched[i]["tradeId"]
            return sorted(fetched, key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error with fetch_pnl {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    def get_order_execution_params(self, order: dict) -> dict:
        # defined for each exchange
        order_type = order.get("type", "limit")
        params = {
            "positionSide": order["position_side"].upper(),
            "newClientOrderId": order["custom_id"],
        }
        if order_type == "limit":
            params["timeInForce"] = (
                "GTX" if self.config["live"]["time_in_force"] == "post_only" else "GTC"
            )
        return params

    async def update_exchange_config_by_symbols(self, symbols):
        coros_to_call_lev, coros_to_call_margin_mode = {}, {}
        for symbol in symbols:
            try:
                coros_to_call_margin_mode[symbol] = asyncio.create_task(
                    self.cca.set_margin_mode("cross", symbol=symbol)
                )
            except Exception as e:
                logging.error(f"{symbol}: error setting cross mode {e}")
            try:
                coros_to_call_lev[symbol] = asyncio.create_task(
                    self.cca.set_leverage(
                        int(self.config_get(["live", "leverage"], symbol=symbol)), symbol=symbol
                    )
                )
            except Exception as e:
                logging.error(f"{symbol}: a error setting leverage {e}")
        for symbol in symbols:
            res = None
            to_print = ""
            try:
                res = await coros_to_call_lev[symbol]
                to_print += f"set leverage {res} "
            except Exception as e:
                logging.error(f"{symbol}: b error setting leverage {e}")
            try:
                res = await coros_to_call_margin_mode[symbol]
                to_print += f"set cross mode {res}"
            except:
                logging.error(f"error setting cross mode {res}")
            if to_print:
                logging.info(f"{symbol}: {to_print}")

    async def update_exchange_config(self):
        try:
            res = await self.cca.set_position_mode(True)
            logging.info(f"set hedge mode {res}")
        except Exception as e:
            if '"code":-4059' in e.args[0]:
                logging.info(f"hedge mode: {e}")
            else:
                logging.error(f"error setting hedge mode {e}")

    async def determine_utc_offset(self, verbose=True):
        # returns millis to add to utc to get exchange timestamp
        # call some endpoint which includes timestamp for exchange's server
        # if timestamp is not included in self.cca.fetch_balance(),
        # implement method in exchange child class
        result = await self.cca.fetch_ticker("BTC/USDT:USDT")
        self.utc_offset = round((result["timestamp"] - utc_ms()) / (1000 * 60 * 60)) * (
            1000 * 60 * 60
        )
        if verbose:
            logging.info(f"Exchange time offset is {self.utc_offset}ms compared to UTC")

    async def fetch_ohlcvs_1m(self, symbol: str, since: float = None, limit=None):
        n_candles_limit = 1500 if limit is None else limit
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

    def format_custom_id_single(self, order_type_id: int) -> str:
        formatted = super().format_custom_id_single(order_type_id)
        return ("x-" + self.broker_code + formatted)[: self.custom_id_max_length]
