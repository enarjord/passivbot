from passivbot import Passivbot, logging
from uuid import uuid4
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import pprint
import asyncio
import traceback
import numpy as np
from utils import utc_ms, ts_to_date
from config_utils import require_live_value
from pure_funcs import (
    multi_replace,
    floatify,
    calc_hash,
    shorten_custom_id,
)
import passivbot_rust as pbr

calc_diff = pbr.calc_diff
from procedures import print_async_exception, assert_correct_ccxt_version
import passivbot_rust as pbr

assert_correct_ccxt_version(ccxt=ccxt_async)


class BitgetBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.position_side_map = {
            "buy": {"open": "long", "close": "short"},
            "sell": {"open": "short", "close": "long"},
        }
        self.custom_id_max_length = 64

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

    def set_market_specific_settings(self):
        super().set_market_specific_settings()
        for symbol in self.markets_dict:
            elm = self.markets_dict[symbol]
            self.symbol_ids[symbol] = elm["id"]
            self.min_costs[symbol] = max(
                5.1, 0.1 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
            )
            self.min_qtys[symbol] = elm["limits"]["amount"]["min"]
            self.qty_steps[symbol] = elm["precision"]["amount"]
            self.price_steps[symbol] = elm["precision"]["price"]
            self.c_mults[symbol] = elm["contractSize"]

    async def watch_orders(self):
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_orders()
                for i in range(len(res)):
                    res[i]["position_side"] = res[i]["info"]["posSide"]
                    res[i]["qty"] = res[i]["amount"]
                    res[i]["side"] = self.determine_side(res[i])
                self.handle_order_update(res)
            except Exception as e:
                print(f"exception watch_orders", e)
                traceback.print_exc()
                await asyncio.sleep(1)

    def determine_side(self, order: dict) -> str:
        if "info" in order:
            if all([x in order["info"] for x in ["tradeSide", "reduceOnly", "posSide"]]):
                if order["info"]["tradeSide"] == "close":
                    if order["info"]["posSide"] == "long":
                        return "sell"
                    elif order["info"]["posSide"] == "short":
                        return "buy"
                elif order["info"]["tradeSide"] == "open":
                    if order["info"]["posSide"] == "long":
                        return "buy"
                    elif order["info"]["posSide"] == "short":
                        return "sell"
        raise Exception(f"failed to determine side {order}")

    async def fetch_open_orders(self, symbol: str = None):
        fetched = None
        open_orders = []
        try:
            fetched = await self.cca.fetch_open_orders()
            for i in range(len(fetched)):
                fetched[i]["position_side"] = fetched[i]["info"]["posSide"]
                fetched[i]["qty"] = fetched[i]["amount"]
                fetched[i]["custom_id"] = fetched[i]["clientOrderId"]
                fetched[i]["side"] = self.determine_side(fetched[i])
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
            balance_info = [x for x in fetched_balance["info"] if x["marginCoin"] == self.quote][0]
            if (
                "assetMode" in balance_info
                and "unionTotalMargin" in balance_info
                and balance_info["assetMode"] == "union"
            ):
                balance = float(balance_info["unionTotalMargin"]) - float(
                    balance_info["unrealizedPL"]
                )
                if not hasattr(self, "previous_rounded_balance"):
                    self.previous_rounded_balance = balance
                self.previous_rounded_balance = pbr.hysteresis_rounding(
                    balance,
                    self.previous_rounded_balance,
                    self.hyst_rounding_balance_pct,
                    self.hyst_rounding_balance_h,
                )
                balance = self.previous_rounded_balance
            else:
                balance = float(balance_info["available"])
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
            tickers = await self.cca.fetch_tickers()
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

    async def fetch_ohlcvs_1m(self, symbol: str, limit=None):
        n_candles_limit = 1000 if limit is None else limit
        result = await self.cca.fetch_ohlcv(
            symbol,
            timeframe="1m",
            limit=n_candles_limit,
        )
        return result

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

    async def fetch_pnls(self, start_time=None, end_time=None, limit=None):
        params = {"productType": "USDT-FUTURES"}
        if start_time:
            start_time = int(start_time)
        if end_time:
            params["endTime"] = int(end_time)
        if limit:
            params["limit"] = min(100, limit)
        side_pos_side_map = {"buy": "long", "sell": "short"}
        data_d = {}
        while True:
            fetched = await self.cca.private_mix_get_v2_mix_order_fill_history(params)
            end_id = fetched["data"]["endId"]
            data = fetched["data"]["fillList"]
            if data is None:
                # print("debug a")
                break
            if not data:
                # print("debug b")
                break
            with_hashes = {calc_hash(x): x for x in data}
            if all([h in data_d for h in with_hashes]):
                # print("debug c")
                break
            for h, x in with_hashes.items():
                data_d[h] = x
                data_d[h]["pnl"] = float(x["profit"])
                data_d[h]["price"] = float(x["price"])
                data_d[h]["amount"] = float(x["baseVolume"])
                data_d[h]["id"] = x["tradeId"]
                data_d[h]["timestamp"] = float(x["cTime"])
                data_d[h]["datetime"] = ts_to_date(data_d[h]["timestamp"])
                data_d[h]["position_side"] = side_pos_side_map[x["side"]]
                data_d[h]["symbol"] = self.get_symbol_id_inv(x["symbol"])
            if start_time is None:
                # print("debug d")
                break
            last_ts = float(data[-1]["cTime"])
            if last_ts < start_time:
                # print("debug e")
                break
            logging.info(f"fetched {len(data)} fills until {ts_to_date(last_ts)[:19]}")
            params["endTime"] = int(last_ts)
        return sorted(data_d.values(), key=lambda x: x["timestamp"])

    async def fetch_pnls_old(self, start_time=None, end_time=None, limit=None):
        wait_between_fetches_minimum_seconds = 0.5
        all_res = {}
        until = int(end_time) if end_time else None
        since = int(start_time) if start_time else None
        retry_count = 0
        first_fetch = True
        while True:
            if since and until and since >= until:
                # print("debug fetch_pnls g")
                break
            sts = utc_ms()
            res = await (
                self.cca.fetch_closed_orders(since=since)
                if until is None
                else self.cca.fetch_closed_orders(since=since, params={"until": until})
            )
            if first_fetch:
                if not res:
                    # print("debug fetch_pnls e")
                    break
                first_fetch = False
            if not res:
                # print("debug fetch_pnls a retry_count:", retry_count)
                if retry_count >= 10:
                    break
                retry_count += 1
                until = int(until - 1000 * 60 * 60 * 4)
                continue
            resd = {elm["id"]: elm for elm in res}
            # if len(resd) != len(res):
            #    print("debug fetch_pnls b", len(resd), len(res))
            if all(id_ in all_res for id_ in resd):
                # print("debug fetch_pnls c retry_count:", retry_count)
                if retry_count >= 10:
                    break
                retry_count += 1
                until = int(until - 1000 * 60 * 60 * 4)
                continue
            retry_count = 0
            for k, v in resd.items():
                all_res[k] = v
                all_res[k]["pnl"] = float(v["info"]["totalProfits"])
                all_res[k]["position_side"] = v["info"]["posSide"]
            if start_time is None and end_time is None:
                break
            if since and res[0]["timestamp"] <= since:
                # print("debug fetch_pnls e")
                break
            until = int(res[0]["timestamp"])
            # print(
            #    "debug fetch_pnls d len(res):",
            #    len(res),
            #    res[0]["datetime"],
            #    res[-1]["datetime"],
            #    (res[-1]["timestamp"] - res[0]["timestamp"]) / (1000 * 60 * 60),
            # )
            wait_time_seconds = max(
                0.0, wait_between_fetches_minimum_seconds - (utc_ms() - sts) / 1000
            )
            await asyncio.sleep(wait_time_seconds)
        all_res_list = sorted(all_res.values(), key=lambda x: x["timestamp"])
        return all_res_list

    async def fetch_fills_with_types(self, start_time=None, end_time=None):
        fills = await self.fetch_pnls(start_time=start_time, end_time=end_time)
        print("n fills", len(fills))
        order_details_tasks = []
        for fill in fills:
            order_details_tasks.append(
                asyncio.create_task(
                    self.cca.fetch_order(fill.get("orderId", fill.get("id")), fill["symbol"])
                )
            )
        order_details_results = {}
        for task in order_details_tasks:
            result = await task
            order_details_results[result.get("orderId", result.get("id"))] = result
        fills_by_id = {fill.get("orderId", result.get("id")): fill for fill in fills}
        for id_ in order_details_results:
            if id_ in fills_by_id:
                fills_by_id[id_].update(order_details_results[id_])
            else:
                logging.warning(f"fetch_fills_with_types id missing {id_}")
        return sorted(fills_by_id.values(), key=lambda x: x["timestamp"])

    def get_order_execution_params(self, order: dict) -> dict:
        # defined for each exchange
        return {
            "timeInForce": (
                "PO" if require_live_value(self.config, "time_in_force") == "post_only" else "GTC"
            ),
            "holdSide": order["position_side"],
            "reduceOnly": order["reduce_only"],
            "oneWayMode": False,
            "clientOid": order["custom_id"],
        }

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
                to_print += f" set leverage {res} "
            except Exception as e:
                logging.error(f"{symbol} error setting leverage {e} {res}")
            res = None
            try:
                res = await coros_to_call_margin_mode[symbol]
                to_print += f"set cross mode {res}"
            except Exception as e:
                logging.error(f"{symbol} error setting cross mode {e} {res}")
            if to_print:
                logging.info(f"{symbol}: {to_print}")

    async def calc_ideal_orders(self, allow_unstuck: bool = True):
        # Bitget returns max 100 open orders per fetch_open_orders.
        # Only create 100 open orders.
        # Drop orders whose pprice diff is greatest.
        ideal_orders = await super().calc_ideal_orders(allow_unstuck=allow_unstuck)
        ideal_orders_tmp = []
        for s in ideal_orders:
            for x in ideal_orders[s]:
                ideal_orders_tmp.append(
                    (
                        calc_diff(
                            x["price"], (await self.cm.get_current_close(s, max_age_ms=10_000))
                        ),
                        {**x, **{"symbol": s}},
                    )
                )
        ideal_orders_tmp = [x[1] for x in sorted(ideal_orders_tmp, key=lambda item: item[0])][:100]
        ideal_orders = {symbol: [] for symbol in self.active_symbols}
        for x in ideal_orders_tmp:
            ideal_orders[x["symbol"]].append(x)
        return ideal_orders

    async def update_exchange_config(self):
        res = None
        try:
            res = await self.cca.set_position_mode(True)
            logging.info(f"set hedge mode {res}")
        except Exception as e:
            logging.error(f"error setting hedge mode {e} {res}")

    def format_custom_id_single(self, order_type_id: int) -> str:
        formatted = super().format_custom_id_single(order_type_id)
        return (self.broker_code + "#" + formatted)[: self.custom_id_max_length]
