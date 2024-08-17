from passivbot_forager import Passivbot, logging
from uuid import uuid4
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import pprint
import asyncio
import traceback
import numpy as np
from collections import defaultdict
from pure_funcs import (
    multi_replace,
    floatify,
    ts_to_date_utc,
    calc_hash,
    determine_pos_side_ccxt,
    symbol_to_coin,
)
from procedures import print_async_exception, utc_ms, assert_correct_ccxt_version

assert_correct_ccxt_version(ccxt=ccxt_async)


class BybitBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.ccp = getattr(ccxt_pro, self.exchange)(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
                "password": self.user_info["passphrase"],
                "headers": {"referer": self.broker_code} if self.broker_code else {},
            }
        )
        self.cca = getattr(ccxt_async, self.exchange)(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
                "password": self.user_info["passphrase"],
                "headers": {"referer": self.broker_code} if self.broker_code else {},
            }
        )

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

    async def start_websockets(self):
        await asyncio.gather(
            self.watch_balance(),
            self.watch_orders(),
            self.watch_tickers(),
        )

    async def watch_balance(self):
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_balance()
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
                    res[i]["position_side"] = determine_pos_side_ccxt(res[i])
                    res[i]["qty"] = res[i]["amount"]
                self.handle_order_update(res)
            except Exception as e:
                print(f"exception watch_orders", e)
                traceback.print_exc()
                await asyncio.sleep(1)

    async def watch_tickers(self, symbols=None):
        self.prev_active_symbols = set()
        while not self.stop_websocket:
            try:
                if (actives := set(self.active_symbols)) != self.prev_active_symbols:
                    for symbol in actives - self.prev_active_symbols:
                        logging.info(f"Started watching ticker for symbol: {symbol}")
                    for symbol in self.prev_active_symbols - actives:
                        logging.info(f"Stopped watching ticker for symbol: {symbol}")
                    self.prev_active_symbols = actives
                res = await self.ccp.watch_tickers(self.active_symbols)
                self.handle_ticker_update(res)
                await asyncio.sleep(0.1)
            except Exception as e:
                logging.error(
                    f"Exception in watch_tickers: {e}, active symbols: {len(self.active_symbols)}"
                )
                traceback.print_exc()
                await asyncio.sleep(1)

    async def watch_ohlcvs_1m(self):
        if not hasattr(self, "ohlcvs_1m"):
            self.ohlcvs_1m = {}
        symbols_and_timeframes = [[s, "1m"] for s in sorted(self.eligible_symbols)]
        coins = [symbol_to_coin(x[0]) for x in symbols_and_timeframes]
        if coins:
            logging.info(f"Started watching ohlcv_1m for {','.join(coins)}")
        while not self.stop_websocket:
            try:
                res = await self.ccp.watch_ohlcv_for_symbols(symbols_and_timeframes)
                symbol = next(iter(res))
                self.handle_ohlcv_1m_update(symbol, res[symbol]["1m"])
            except Exception as e:
                logging.error(f"Exception in watch_ohlcvs_1m: {e}")
                traceback.print_exc()
                await asyncio.sleep(1)

    async def fetch_open_orders(self, symbol: str = None) -> [dict]:
        fetched = None
        open_orders = {}
        limit = 50
        try:
            fetched = await self.cca.fetch_open_orders(symbol=symbol, limit=limit)
            while True:
                if all([elm["id"] in open_orders for elm in fetched]):
                    break
                next_page_cursor = None
                for elm in fetched:
                    elm["position_side"] = determine_pos_side_ccxt(elm)
                    elm["qty"] = elm["amount"]
                    open_orders[elm["id"]] = elm
                    if "nextPageCursor" in elm["info"]:
                        next_page_cursor = elm["info"]["nextPageCursor"]
                if len(fetched) < limit:
                    break
                if next_page_cursor is None:
                    break
                # fetch more
                fetched = await self.cca.fetch_open_orders(
                    symbol=symbol, limit=limit, params={"cursor": next_page_cursor}
                )
            return sorted(open_orders.values(), key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_positions(self):
        fetched_positions, fetched_balance = None, None
        positions = {}
        limit = 200
        try:
            fetched_positions, fetched_balance = await asyncio.gather(
                self.cca.fetch_positions(params={"limit": limit}), self.cca.fetch_balance()
            )
            balance = fetched_balance[self.quote]["total"]
            while True:
                if all([elm["symbol"] + elm["side"] in positions for elm in fetched_positions]):
                    break
                next_page_cursor = None
                for elm in fetched_positions:
                    elm["position_side"] = determine_pos_side_ccxt(elm)
                    elm["size"] = float(elm["contracts"])
                    elm["price"] = float(elm["entryPrice"])
                    positions[elm["symbol"] + elm["side"]] = elm
                    if "nextPageCursor" in elm["info"]:
                        next_page_cursor = elm["info"]["nextPageCursor"]
                    positions[elm["symbol"] + elm["side"]] = elm
                if len(fetched_positions) < limit:
                    break
                if next_page_cursor is None:
                    break
                # fetch more
                fetched_positions = await self.cca.fetch_positions(
                    params={"cursor": next_page_cursor, "limit": limit}
                )
            return sorted(positions.values(), key=lambda x: x["timestamp"]), balance
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

    async def fetch_pnls_old(
        self,
        symbol: str = None,
        start_time: int = None,
        end_time: int = None,
    ):
        if start_time is not None:
            week = 1000 * 60 * 60 * 24 * 7
            income = []
            if end_time is None:
                end_time = int(utc_ms() + 1000 * 60 * 60 * 24)
            # bybit has limit of 7 days per pageinated fetch
            # fetch multiple times
            i = 1
            while i < 52:  # limit n fetches to 52 (one year)
                sts = end_time - week * i
                ets = sts + week
                sts = max(sts, start_time)
                fetched = await self.fetch_pnl(symbol=symbol, start_time=sts, end_time=ets)
                income.extend(fetched)
                if sts <= start_time:
                    break
                i += 1
                logging.debug(f"fetching income for more than a week {ts_to_date_utc(sts)}")
            return sorted({elm["id"]: elm for elm in income}.values(), key=lambda x: x["timestamp"])
        else:
            return await self.fetch_pnl(symbol=symbol, start_time=start_time, end_time=end_time)

    async def fetch_pnl(
        self,
        symbol: str = None,
        start_time: int = None,
        end_time: int = None,
    ):
        fetched = None
        income_d = {}
        limit = 100
        try:
            params = {"category": "linear", "limit": limit}
            if symbol is not None:
                params["symbol"] = symbol
            if start_time is not None:
                params["startTime"] = int(start_time)
            if end_time is not None:
                params["endTime"] = int(end_time)
            fetched = await self.cca.private_get_v5_position_closed_pnl(params)
            fetched["result"]["list"] = sorted(
                floatify(fetched["result"]["list"]), key=lambda x: x["updatedTime"]
            )
            while True:
                if fetched["result"]["list"] == []:
                    break
                logging.debug(
                    f"fetching income {ts_to_date_utc(fetched['result']['list'][-1]['updatedTime'])}"
                )
                if (
                    fetched["result"]["list"][0]["orderId"] in income_d
                    and fetched["result"]["list"][-1]["orderId"] in income_d
                ):
                    break
                for elm in fetched["result"]["list"]:
                    income_d[elm["orderId"]] = elm
                if start_time is None:
                    break
                if fetched["result"]["list"][0]["updatedTime"] <= start_time:
                    break
                if not fetched["result"]["nextPageCursor"]:
                    break
                params["cursor"] = fetched["result"]["nextPageCursor"]
                fetched = await self.cca.private_get_v5_position_closed_pnl(params)
                fetched["result"]["list"] = sorted(
                    floatify(fetched["result"]["list"]), key=lambda x: x["updatedTime"]
                )
            for k in income_d:
                income_d[k]["pnl"] = income_d[k]["closedPnl"]
                income_d[k]["timestamp"] = income_d[k]["updatedTime"]
                income_d[k]["id"] = str(income_d[k]["orderId"]) + str(income_d[k]["qty"])
            return sorted(income_d.values(), key=lambda x: x["updatedTime"])
        except Exception as e:
            logging.error(f"error fetching income {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return []

    async def fetch_history_sub(self, func, end_time=None):
        all_fetched_d = {}
        params = {"limit": 100}
        if end_time:
            params["paginate"] = True
            params["endTime"] = int(end_time)
        result = await getattr(self.cca, func)(params=params)
        return sorted(result, key=lambda x: x["timestamp"])

    async def fetch_fills_sub(self, start_time=None, end_time=None):
        return await self.fetch_history_sub("fetch_my_trades", end_time)

    async def fetch_pnls_sub(self, start_time=None, end_time=None):
        return await self.fetch_history_sub("fetch_positions_history", end_time)

    async def fetch_pnls(self, start_time=None, end_time=None):
        # fetches both fills and pnls (bybit has them in separate endpoints)
        if start_time is None:
            all_fetched_fills, all_fetched_pnls = await asyncio.gather(
                self.fetch_fills_sub(), self.fetch_pnls_sub()
            )
        else:
            all_fetched_fills = []
            for _ in range(100):
                fills = await self.fetch_fills_sub(None, end_time)
                if not fills:
                    break
                all_fetched_fills += fills
                if fills[0]["timestamp"] <= start_time:
                    break
                logging.info(
                    f"fetched fills: {fills[0]['datetime']} {fills[-1]['datetime']} {len(fills)}"
                )
                end_time = fills[0]["timestamp"]
            else:
                logging.error(f"more than 100 calls to ccxt fetch_my_trades")
        if not all_fetched_fills:
            return []
        all_fetched_fills.sort(key=lambda y: y["timestamp"])
        fillsd = defaultdict(list)
        execIds_seen = set()
        closes_ids = set()
        for x in all_fetched_fills:
            if float(x["info"]["closedSize"]) != 0.0:
                closes_ids.add(x["info"]["orderId"])
            if x["info"]["execId"] not in execIds_seen:
                fillsd[x["info"]["orderId"]].append(
                    {**x, **{"pnl": 0.0, "position_side": self.determine_pos_side(x)}}
                )
                execIds_seen.add(x["info"]["execId"])
        if start_time is not None:
            start_time = all_fetched_fills[0]["timestamp"]
            end_time = all_fetched_fills[-1]["timestamp"] + 1
            all_fetched_pnls = []
            for _ in range(100):
                pnls = await self.fetch_pnls_sub(None, end_time)
                if not pnls:
                    break
                all_fetched_pnls += pnls
                if pnls[0]["timestamp"] <= start_time:
                    break
                if all([x["info"]["orderId"] in closes_ids for x in all_fetched_pnls]):
                    break
                logging.info(
                    f"fetched pnls: {pnls[0]['datetime']} {pnls[-1]['datetime']} {len(pnls)}"
                )
                end_time = pnls[0]["timestamp"]
        for x in all_fetched_pnls:
            if x["timestamp"] < all_fetched_fills[0]["timestamp"]:
                continue
            if x["info"]["orderId"] in fillsd:
                fillsd[x["info"]["orderId"]][-1]["pnl"] = float(x["info"]["closedPnl"])
            else:
                logging.info(f"debug: pnl order id {x['info']['orderId']} not in fills")
        result = []
        for x in fillsd.values():
            result += x
        return sorted(result, key=lambda x: x["timestamp"])

    def determine_pos_side(self, x):
        if x["side"] == "buy":
            return "short" if float(x["info"]["closedSize"]) != 0.0 else "long"
        return "long" if float(x["info"]["closedSize"]) != 0.0 else "short"

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
        executed = None
        try:
            executed = await self.cca.create_limit_order(
                symbol=order["symbol"],
                side=order["side"],
                amount=abs(order["qty"]),
                price=order["price"],
                params={
                    "positionIdx": 1 if order["position_side"] == "long" else 2,
                    "timeInForce": (
                        "postOnly" if self.config["live"]["time_in_force"] == "post_only" else "GTC"
                    ),
                    "orderLinkId": order["custom_id"],
                },
            )
            if "symbol" not in executed or executed["symbol"] is None:
                executed["symbol"] = order["symbol"]
            for key in ["side", "position_side", "qty", "price"]:
                if key not in executed or executed[key] is None:
                    executed[key] = order[key]
            return executed
        except Exception as e:
            logging.error(f"error executing order {order} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return {}

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
                        params={"leverage": int(self.live_configs[symbol]["leverage"])},
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
        try:
            res = await self.cca.set_position_mode(True)
            logging.info(f"set hedge mode {res}")
        except Exception as e:
            logging.error(f"error setting hedge mode {e}")

    async def fetch_ohlcvs_1m(self, symbol: str, since: float = None):
        n_candles_limit = (
            20 if symbol in self.ohlcvs_1m and len(self.ohlcvs_1m[symbol]) > 100 else 1000
        )
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
