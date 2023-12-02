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


class BinanceBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.ccp = getattr(ccxt_pro, "binanceusdm")(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
                "password": self.user_info["passphrase"],
            }
        )
        self.cca = getattr(ccxt_async, "binanceusdm")(
            {
                "apiKey": self.user_info["key"],
                "secret": self.user_info["secret"],
                "password": self.user_info["passphrase"],
                "headers": {"referer": self.broker_code} if self.broker_code else {},
            }
        )
        self.max_n_cancellations_per_batch = 10
        self.max_n_creations_per_batch = 5

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
        self.symbol_ids = {
            symbol: self.markets_dict[symbol]["id"]
            for symbol in self.markets_dict
            if symbol.endswith(f":{self.quote}")
        }
        self.symbol_ids_inv = {v: k for k, v in self.symbol_ids.items()}
        for symbol in self.symbols:
            elm = self.markets_dict[symbol]
            self.min_costs[symbol] = (
                0.1 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
            )
            self.min_qtys[symbol] = elm["limits"]["amount"]["min"]
            for felm in elm["info"]["filters"]:
                if felm["filterType"] == "PRICE_FILTER":
                    self.price_steps[symbol] = float(felm["tickSize"])
                elif felm["filterType"] == "MARKET_LOT_SIZE":
                    self.qty_steps[symbol] = float(felm["stepSize"])
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
        try:
            fetched = await asyncio.gather(
                *[self.cca.fetch_open_orders(symbol=symbol) for symbol in self.symbols]
            )
            for elm in [x for sublist in fetched for x in sublist]:
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
        fetched = None
        try:
            fetched = await self.cca.fetch_balance()
            fetched = floatify(fetched)
            positions = []
            for elm in fetched["info"]["positions"]:
                if elm["positionAmt"] == 0.0 or elm["symbol"] not in self.symbol_ids_inv:
                    continue
                positions.append(
                    {
                        "symbol": self.symbol_ids_inv[elm["symbol"]],
                        "position_side": elm["positionSide"].lower(),
                        "size": elm["positionAmt"],
                        "price": elm["entryPrice"],
                    }
                )
            balance = [x for x in fetched["info"]["assets"] if x["asset"] == self.quote][0][
                "walletBalance"
            ]
            return positions, balance
        except Exception as e:
            logging.error(f"error fetching tickers {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def fetch_tickers(self):
        fetched = None
        try:
            fetched = await self.cca.fapipublic_get_ticker_bookticker()
            tickers = {
                self.symbol_ids_inv[elm["symbol"]]: {
                    "bid": float(elm["bidPrice"]),
                    "ask": float(elm["askPrice"]),
                }
                for elm in fetched
                if elm["symbol"] in self.symbol_ids_inv
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
        symbol: str = None,
        start_time: int = None,
        end_time: int = None,
    ):
        limit = 1000
        if start_time is None and end_time is None:
            return await self.fetch_pnl(symbol=symbol)
        all_fetched = {}
        while True:
            fetched = await self.fetch_pnl(symbol=symbol, start_time=start_time, end_time=end_time)
            if fetched == []:
                break
            for elm in fetched:
                all_fetched[elm["tranId"]] = elm
            if len(fetched) < limit:
                break
            logging.info(f"debug fetching income {ts_to_date_utc(fetched[-1]['timestamp'])}")
            start_time = fetched[-1]["timestamp"]
        return sorted(all_fetched.values(), key=lambda x: x["timestamp"])

    async def fetch_pnl(
        self,
        symbol: str = None,
        start_time: int = None,
        end_time: int = None,
    ):
        fetched = None
        try:
            params = {"incomeType": "REALIZED_PNL", "limit": 1000}
            if symbol is not None:
                params["symbol"] = symbol
            if start_time is not None:
                params["startTime"] = int(start_time)
            if end_time is not None:
                params["endTime"] = int(end_time)
            fetched = floatify(await self.cca.fapiprivate_get_income(params=params))
            for i in range(len(fetched)):
                fetched[i]["pnl"] = fetched[i]["income"]
                fetched[i]["timestamp"] = fetched[i]["time"]
                fetched[i]["id"] = fetched[i]["tranId"]
            return sorted(fetched, key=lambda x: x["time"])
        except Exception as e:
            logging.error(f"error fetching income {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

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
            if "code" in executed and executed["code"] == -2011:
                logging.info(f"{executed}")
                return {}
            return {
                "symbol": executed["symbol"],
                "side": executed["side"],
                "id": executed["id"],
                "position_side": executed["info"]["positionSide"].lower(),
                "qty": executed["amount"],
                "price": executed["price"],
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
                    "positionSide": order["position_side"].upper(),
                    "newClientOrderId": (order["custom_id"] + str(uuid4()))[:36],
                    "timeInForce": "GTX",
                },
            )
            if (
                "info" in executed
                and "code" in executed["info"]
                and executed["info"]["code"] == "-5022"
            ):
                logging.info(f"{executed['info']['msg']}")
                return {}
            elif "status" in executed and executed["status"] == "open":
                executed["position_side"] = executed["info"]["positionSide"].lower()
                executed["qty"] = executed["amount"]
                executed["reduce_only"] = executed["reduceOnly"]
                return executed
        except Exception as e:
            logging.error(f"error executing order {order} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return {}

    async def execute_orders(self, orders: [dict]) -> [dict]:
        if len(orders) == 0:
            return []
        if len(orders) == 1:
            return [await self.execute_order(orders[0])]
        to_execute = []
        for order in orders[: self.max_n_creations_per_batch]:
            to_execute.append(
                {
                    "type": "limit",
                    "symbol": order["symbol"],
                    "side": order["side"],
                    "amount": abs(order["qty"]),
                    "price": order["price"],
                    "params": {
                        "positionSide": order["position_side"].upper(),
                        "newClientOrderId": (order["custom_id"] + str(uuid4()))[:36],
                        "timeInForce": "GTX",
                    },
                }
            )
        executed = None
        try:
            executed = await self.cca.create_orders(to_execute)
            for i in range(len(executed)):
                if (
                    "info" in executed[i]
                    and "code" in executed[i]["info"]
                    and executed[i]["info"]["code"] == "-5022"
                ):
                    logging.info(f"{executed[i]['info']['msg']}")
                    executed[i] = {}
                elif "status" in executed[i] and executed[i]["status"] == "open":
                    executed[i]["position_side"] = executed[i]["info"]["positionSide"].lower()
                    executed[i]["qty"] = executed[i]["amount"]
                    executed[i]["reduce_only"] = executed[i]["reduceOnly"]
            return executed
        except Exception as e:
            logging.error(f"error executing orders {orders} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return {}
