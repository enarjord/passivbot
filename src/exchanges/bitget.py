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
    hysteresis_rounding,
)
from njit_funcs import calc_diff
from procedures import print_async_exception, utc_ms, assert_correct_ccxt_version

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
                self.previous_rounded_balance = hysteresis_rounding(
                    balance, self.previous_rounded_balance, 0.02, 0.5
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
        if len(orders) > self.config["live"]["max_n_cancellations_per_batch"]:
            # prioritize cancelling reduce-only orders
            try:
                reduce_only_orders = [x for x in orders if x["reduce_only"]]
                rest = [x for x in orders if not x["reduce_only"]]
                orders = (reduce_only_orders + rest)[
                    : self.config["live"]["max_n_cancellations_per_batch"]
                ]
            except Exception as e:
                logging.error(f"debug filter cancellations {e}")
        return await self.execute_multiple(
            orders, "execute_cancellation", self.config["live"]["max_n_cancellations_per_batch"]
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
                "timeInForce": "PO" if self.config["live"]["time_in_force"] == "post_only" else "GTC",
                "holdSide": order["position_side"],
                "reduceOnly": order["reduce_only"],
                "oneWayMode": False,
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
            executed["custom_id"] = executed["clientOrderId"]
        return executed
        if "msg" in executed and executed["msg"] == "success":
            for key in ["symbol", "side", "position_side", "qty", "price"]:
                executed[key] = order[key]
            executed["timestamp"] = float(executed["requestTime"])
            executed["id"] = executed["data"]["orderId"]
            executed["custom_id"] = executed["data"]["clientOid"]
        return executed

    async def execute_orders(self, orders: [dict]) -> [dict]:
        return await self.execute_multiple(
            orders, "execute_order", self.config["live"]["max_n_creations_per_batch"]
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
                    self.cca.set_leverage(int(self.live_configs[symbol]["leverage"]), symbol=symbol)
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

    def calc_ideal_orders(self):
        # Bitget returns max 100 open orders per fetch_open_orders.
        # Only create 100 open orders.
        # Drop orders whose pprice diff is greatest.
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

    async def update_exchange_config(self):
        res = None
        try:
            res = await self.cca.set_position_mode(True)
            logging.info(f"set hedge mode {res}")
        except Exception as e:
            logging.error(f"error setting hedge mode {e} {res}")

    def format_custom_ids(self, orders: [dict]) -> [dict]:
        # bitget needs broker code plus '#' at the beginning of the custom_id
        new_orders = []
        for order in orders:
            order["custom_id"] = (
                self.broker_code
                + "#"
                + shorten_custom_id(order["custom_id"] if "custom_id" in order else "")
                + uuid4().hex
            )[: self.custom_id_max_length]
            new_orders.append(order)
        return new_orders
