from passivbot_multi import Passivbot, logging
from uuid import uuid4
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import pprint
import asyncio
import traceback
import numpy as np
from pure_funcs import multi_replace, floatify, ts_to_date_utc, calc_hash, determine_pos_side_ccxt
from procedures import print_async_exception, utc_ms


class BybitBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.ccp = getattr(ccxt_pro, self.exchange)(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
                "password": self.user_info["passphrase"],
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
        self.max_n_cancellations_per_batch = 20
        self.max_n_creations_per_batch = 12

    async def init_bot(self):
        # require symbols to be formatted to ccxt standard COIN/USDT:USDT
        self.markets = await self.cca.fetch_markets()
        self.markets_dict = {elm["symbol"]: elm for elm in self.markets}
        self.symbols = {}
        for symbol_ in sorted(set(self.config["symbols"])):
            symbol = symbol_
            if not symbol.endswith("/USDT:USDT"):
                coin_extracted = multi_replace(
                    symbol_, [("/", ""), (":", ""), ("USDT", ""), ("BUSD", ""), ("USDC", "")]
                )
                symbol_reformatted = coin_extracted + "/USDT:USDT"
                logging.info(
                    f"symbol {symbol_} is wrongly formatted. Trying to reformat to {symbol_reformatted}"
                )
                symbol = symbol_reformatted
            if symbol not in self.markets_dict:
                logging.info(f"{symbol} missing from {self.exchange}")
            else:
                elm = self.markets_dict[symbol]
                if elm["type"] != "swap":
                    logging.info(f"wrong market type for {symbol}: {elm['type']}")
                elif not elm["active"]:
                    logging.info(f"{symbol} not active")
                elif not elm["linear"]:
                    logging.info(f"{symbol} is not a linear market")
                else:
                    self.symbols[symbol] = self.config["symbols"][symbol_]
        self.quote = "USDT"
        self.inverse = False
        for symbol in self.symbols:
            elm = self.markets_dict[symbol]
            self.symbol_ids[symbol] = elm["id"]
            self.min_costs[symbol] = (
                0.1 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
            )
            self.min_qtys[symbol] = elm["limits"]["amount"]["min"]
            self.qty_steps[symbol] = elm["precision"]["amount"]
            self.price_steps[symbol] = elm["precision"]["price"]
            self.c_mults[symbol] = elm["contractSize"]
            self.coins[symbol] = symbol.replace("/USDT:USDT", "")
            self.tickers[symbol] = {"bid": 0.0, "ask": 0.0, "last": 0.0}
            self.open_orders[symbol] = []
            self.positions[symbol] = {
                "long": {"size": 0.0, "price": 0.0},
                "short": {"size": 0.0, "price": 0.0},
            }
            self.upd_timestamps["open_orders"][symbol] = 0.0
            self.upd_timestamps["tickers"][symbol] = 0.0
            self.upd_timestamps["positions"][symbol] = 0.0
        await super().init_bot()

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
                await self.handle_balance_update(res)
            except Exception as e:
                print(f"exception watch_balance", e)
                traceback.print_exc()

    async def watch_orders(self):
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_orders()
                for i in range(len(res)):
                    res[i]["position_side"] = determine_pos_side_ccxt(res[i])
                    res[i]["qty"] = res[i]["amount"]
                await self.handle_order_update(res)
            except Exception as e:
                print(f"exception watch_orders", e)
                traceback.print_exc()

    async def watch_tickers(self, symbols=None):
        symbols = list(self.symbols if symbols is None else symbols)
        while True:
            try:
                if self.stop_websocket:
                    break
                res = await self.ccp.watch_tickers(symbols)
                if res["last"] is None:
                    res["last"] = np.random.choice([res["bid"], res["ask"]])
                await self.handle_ticker_update(res)
            except Exception as e:
                print(f"exception watch_tickers {symbols}", e)
                traceback.print_exc()

    async def fetch_open_orders(self, symbol: str = None):
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

    async def fetch_pnls(
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
            return sorted(
                {elm["orderId"]: elm for elm in income}.values(), key=lambda x: x["updatedTime"]
            )
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
            return sorted(income_d.values(), key=lambda x: x["updatedTime"])
        except Exception as e:
            logging.error(f"error fetching income {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return []

    async def execute_multiple(self, orders: [dict], type_: str, max_n_executions: int):
        if not orders:
            return []
        executions = []
        for order in orders[:max_n_executions]:  # sorted by PA dist
            execution = None
            try:
                execution = asyncio.create_task(getattr(self, type_)(order))
                executions.append((order, execution))
            except Exception as e:
                logging.error(f"error executing {type_} {order} {e}")
                print_async_exception(execution)
                traceback.print_exc()
        results = []
        for execution in executions:
            result = None
            try:
                result = await execution[1]
                results.append(result)
            except Exception as e:
                logging.error(f"error executing {type_} {execution} {e}")
                print_async_exception(result)
                traceback.print_exc()
        return results

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
        if len(orders) > self.max_n_cancellations_per_batch:
            # prioritize cancelling reduce-only orders
            try:
                reduce_only_orders = [x for x in orders if x["reduce_only"]]
                rest = [x for x in orders if not x["reduce_only"]]
                orders = (reduce_only_orders + rest)[: self.max_n_cancellations_per_batch]
            except Exception as e:
                logging.error(f"debug filter cancellations {e}")
        return await self.execute_multiple(
            orders, "execute_cancellation", self.max_n_cancellations_per_batch
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
                    "timeInForce": "postOnly",
                    "orderLinkId": order["custom_id"] + str(uuid4()),
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
        return await self.execute_multiple(orders, "execute_order", self.max_n_creations_per_batch)
