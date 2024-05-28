import asyncio
import traceback
import os
import json
import numpy as np

from uuid import uuid4
from njit_funcs import calc_diff, round_
from passivbot import Bot, logging
from procedures import print_async_exception, utc_ms, make_get_filepath
from pure_funcs import determine_pos_side_ccxt, floatify, calc_hash, ts_to_date_utc, date_to_ts2

import ccxt.async_support as ccxt

from procedures import load_ccxt_version

ccxt_version_req = load_ccxt_version()
assert (
    ccxt.__version__ == ccxt_version_req
), f"Currently ccxt {ccxt.__version__} is installed. Please pip reinstall requirements.txt or install ccxt v{ccxt_version_req} manually"


class BingXBot(Bot):
    def __init__(self, config: dict):
        self.exchange = "bingx"
        self.market_type = config["market_type"] = "linear_perpetual"
        self.inverse = config["inverse"] = False

        self.max_n_orders_per_batch = 7
        self.max_n_cancellations_per_batch = 10

        super().__init__(config)
        self.cc = getattr(ccxt, "bingx")(
            {
                "apiKey": self.key,
                "secret": self.secret,
                "headers": {"X-SOURCE-KEY": self.broker_code} if self.broker_code else {},
            }
        )
        self.custom_id_max_length = 40

    def init_market_type(self):
        if not self.symbol.endswith("USDT"):
            raise Exception(f"unsupported symbol {self.symbol}")

    async def fetch_market_info_from_cache(self):
        fname = make_get_filepath(f"caches/bingx_market_info.json")
        info = None
        try:
            if os.path.exists(fname):
                info = json.load(open(fname))
                logging.info("loaded market info from cache")
            if info is None or utc_ms() - info["dump_ts"] > 1000 * 60 * 60 * 24:
                info = {"info": await self.cc.fetch_markets(), "dump_ts": utc_ms()}
                json.dump(info, open(fname, "w"))
                logging.info("dumped market info to cache")
        except Exception as e:
            logging.error(f"failed to load market info from cache {e}")
            traceback.print_exc()
            print_async_exception(info)
            if info is None:
                info = {"info": await self.cc.fetch_markets(), "dump_ts": utc_ms()}
        return info["info"]

    async def _init(self):
        info = await self.fetch_market_info_from_cache()
        self.symbol_id = self.symbol
        self.symbol_id_map = {elm["id"]: elm["symbol"] for elm in info if elm["type"] == "swap"}
        for elm in info:
            if elm["baseId"] + elm["quoteId"] == self.symbol_id and elm["type"] == "swap":
                break
        else:
            raise Exception(f"unsupported symbol {self.symbol}")
        self.symbol_id = elm["id"]
        self.symbol = elm["symbol"]
        self.max_leverage = elm["limits"]["leverage"]["max"]
        self.coin = elm["base"]
        self.quote = elm["quote"]
        self.price_step = self.config["price_step"] = round(
            1.0 / (10 ** elm["precision"]["price"]), 12
        )
        self.qty_step = self.config["qty_step"] = round(1.0 / (10 ** elm["precision"]["amount"]), 12)
        self.min_qty = self.config["min_qty"] = elm["contractSize"]
        self.min_cost = self.config["min_cost"] = (
            2.2 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
        )
        min_qty_overrides = [("OCEAN", 8)]
        for coin, qty in min_qty_overrides:
            if coin in self.symbol:
                self.min_qty = self.config["min_qty"] = max(qty, self.min_qty)
        self.margin_coin = self.quote
        self.cache_path_open_orders = make_get_filepath(
            f"caches/bingx_cache/open_orders_{self.user}_{self.symbol_id}.json"
        )
        self.cache_path_fills = make_get_filepath(
            f"caches/bingx_cache/fills_{self.user}_{self.symbol_id}.json"
        )
        await super()._init()

    async def fetch_ticker(self, symbol=None):
        fetched = None
        try:
            fetched = await self.cc.swap_v2_public_get_quote_depth(
                params={"symbol": self.symbol_id, "limit": 5}
            )
            ticker = {
                "bid": sorted(floatify(fetched["data"]["bids"]))[-1][0],
                "ask": sorted(floatify(fetched["data"]["asks"]))[0][0],
            }
            ticker["last"] = np.random.choice([ticker["bid"], ticker["ask"]])
            return ticker
        except Exception as e:
            logging.error(f"error fetching ticker {e}")
            print_async_exception(fetched)
            return None

    async def init_order_book(self):
        return await self.update_ticker()

    async def fetch_order_details(self, order_ids, cache_path, cache=True, max_n_fetches=10):
        # fetch order details for given order_ids, using cache
        if len(order_ids) == 0:
            return []
        if isinstance(order_ids[0], dict):
            for key in ["id", "orderId", "order_id"]:
                if key in order_ids[0]:
                    break
            order_ids = [elm[key] for elm in order_ids]
        cached_orders = []
        if cache:
            try:
                if os.path.exists(cache_path):
                    cached_orders = json.load(open(cache_path))
            except Exception as e:
                logging.error(f"error loading cache {cache_path} {e}")
        ids_cached = set([elm["orderId"] for elm in cached_orders])
        ids_to_fetch = [id_ for id_ in order_ids if id_ not in ids_cached]

        # split into multiple tasks
        sublists = [
            ids_to_fetch[i : i + max_n_fetches] for i in range(0, len(ids_to_fetch), max_n_fetches)
        ]
        all_fetched = []
        order_details = [x for x in cached_orders if x["orderId"] in order_ids]
        order_details = sorted(
            {x["orderId"]: x for x in order_details}.values(), key=lambda x: float(x["orderId"])
        )
        for sublist in sublists:
            logging.info(
                f"debug fetch order details sublist {cache_path} n fetches: {len(ids_to_fetch) - len(all_fetched)}"
            )
            fetched = await asyncio.gather(
                *[
                    self.cc.swap_v2_private_get_trade_order(
                        params={"symbol": self.symbol_id, "orderId": id_}
                    )
                    for id_ in sublist
                ]
            )
            fetched = [x["data"]["order"] for x in fetched]
            fetched = [{**floatify(x), **{"orderId": x["orderId"]}} for x in fetched]
            all_fetched += fetched
            order_details = [x for x in order_details + all_fetched if x["orderId"] in order_ids]
            # dedup
            order_details = sorted(
                {x["orderId"]: x for x in order_details}.values(), key=lambda x: float(x["orderId"])
            )
            if cache:
                try:
                    json.dump(order_details, open(cache_path, "w"))
                except Exception as e:
                    logging.error(f"error dumping cache {cache_path} {e}")
        return order_details

    async def fetch_open_orders(self) -> [dict]:
        open_orders = None
        try:
            open_orders = await self.cc.fetch_open_orders(symbol=self.symbol, limit=50)
            if len(open_orders) == 50:
                # fetch more
                pass
            order_details = await self.fetch_order_details(open_orders, self.cache_path_open_orders)
            order_details = {elm["orderId"]: elm for elm in order_details}
            for i in range(len(open_orders)):
                try:
                    open_orders[i]["clientOrderId"] = order_details[open_orders[i]["id"]][
                        "clientOrderId"
                    ]
                except:
                    return order_details, open_orders
            return [
                {
                    "order_id": elm["id"],
                    "custom_id": elm["clientOrderId"],
                    "symbol": elm["symbol"],
                    "price": elm["price"],
                    "qty": elm["amount"],
                    "type": elm["type"],
                    "side": elm["side"],
                    "position_side": elm["info"]["positionSide"].lower(),
                    "timestamp": elm["timestamp"],
                }
                for elm in open_orders
            ]
        except Exception as e:
            logging.error(f"error fetching open orders {e}")
            print_async_exception(open_orders)
            traceback.print_exc()
            return False

    async def transfer_from_derivatives_to_spot(self, coin: str, amount: float):
        return

    async def fetch_server_time(self):
        return self.get_server_time()

    async def get_server_time(self):
        server_time = None
        try:
            server_time = await self.cc.fetch_time()
            return server_time
        except Exception as e:
            logging.error(f"error fetching server time {e}")
            print_async_exception(server_time)
            traceback.print_exc()

    async def fetch_position(self) -> dict:
        positions, balance = None, None
        try:
            positions, balance = await asyncio.gather(
                self.cc.fetch_positions(params={"symbol": self.symbol_id}),
                self.cc.swap_v2_private_get_user_balance(),
            )
            positions = floatify([e for e in positions if e["symbol"] == self.symbol])
            position = {
                "long": {"size": 0.0, "price": 0.0, "liquidation_price": 0.0},
                "short": {"size": 0.0, "price": 0.0, "liquidation_price": 0.0},
                "wallet_balance": 0.0,
                "equity": 0.0,
            }
            if positions:
                for p in positions:
                    if p["side"] == "long":
                        position["long"] = {
                            "size": p["notional"],
                            "price": 0.0 if p["entryPrice"] is None else p["entryPrice"],
                            "liquidation_price": (
                                p["liquidationPrice"] if p["liquidationPrice"] else 0.0
                            ),
                        }
                    elif p["side"] == "short":
                        position["short"] = {
                            "size": -abs(p["notional"]),
                            "price": 0.0 if p["entryPrice"] is None else p["entryPrice"],
                            "liquidation_price": (
                                p["liquidationPrice"] if p["liquidationPrice"] else 0.0
                            ),
                        }
            position["wallet_balance"] = float(balance["data"]["balance"]["balance"])
            return position
        except Exception as e:
            logging.error(f"error fetching pos or balance {e}")
            print_async_exception(positions)
            print_async_exception(balance)
            traceback.print_exc()
            return None

    async def execute_orders(self, orders: [dict]) -> [dict]:
        executed_orders = await self.execute_multiple(
            orders, self.execute_order, "creations", self.max_n_orders_per_batch
        )
        # dump to open orders cache
        to_dump = []
        for elm in executed_orders:
            if "info" in elm and "clientOrderID" in elm["info"]:
                # bingx inconsistent naming: "ID" and "Id"
                elm["info"]["clientOrderId"] = elm["info"]["clientOrderID"]
                to_dump.append(elm["info"].copy())
        if to_dump:
            try:
                cached_orders = json.load(open(self.cache_path_open_orders))
            except Exception as e:
                logging.error(f"error loading cached open orders {e}")
                traceback.print_exc()
                cached_orders = []
            cached_orders.extend(to_dump)
            try:
                json.dump(cached_orders, open(self.cache_path_open_orders, "w"))
            except Exception as e:
                logging.error(f"error dumping cached open orders {e}")
                traceback.print_exc()
        return executed_orders

    async def execute_order(self, order: dict) -> dict:
        executed = None
        try:
            executed = await self.cc.create_limit_order(
                symbol=order["symbol"] if "symbol" in order else self.symbol,
                side=order["side"].upper(),
                amount=abs(order["qty"]),
                price=order["price"],
                params={
                    "positionSide": order["position_side"].upper(),
                    "clientOrderID": order["custom_id"],
                },
            )
            if "symbol" not in executed or executed["symbol"] is None:
                executed["symbol"] = order["symbol"] if "symbol" in order else self.symbol
            for key in ["side", "position_side", "qty", "price"]:
                if key not in executed or executed[key] is None:
                    executed[key] = order[key]
            return executed
        except Exception as e:
            if '"code":101400' in str(e):
                new_min_qty = round_(max(self.min_qty, order["qty"]) + self.qty_step, self.qty_step)
                logging.info(
                    f"successfully caught order size error, code 101400. Adjusting min_qty from {self.min_qty} to {new_min_qty}..."
                )
                self.min_qty = self.xk["min_qty"] = self.config["min_qty"] = new_min_qty
                logging.error(f"{order} {e}")
                return {}
            logging.error(f"error executing order {order} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return {}

    async def execute_multiple(self, orders: [dict], func, type_: str, max_n_executions: int):
        if not orders:
            return []
        executions = []
        for order in sorted(orders, key=lambda x: calc_diff(x["price"], self.price))[
            :max_n_executions
        ]:
            execution = None
            try:
                execution = asyncio.create_task(func(order))
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

    async def execute_cancellations(self, orders: [dict]) -> [dict]:
        if len(orders) > self.max_n_cancellations_per_batch:
            # prioritize cancelling reduce-only orders
            try:
                reduce_only_orders = [x for x in orders if x["reduce_only"]]
                rest = [x for x in orders if not x["reduce_only"]]
                orders = (reduce_only_orders + rest)[:max_n_cancellations_per_batch]
            except Exception as e:
                logging.error("debug filter cancellations {e}")
        return await self.execute_multiple(
            orders, self.execute_cancellation, "cancellations", self.max_n_cancellations_per_batch
        )

    async def execute_cancellation(self, order: dict) -> dict:
        executed = None
        try:
            executed = await self.cc.cancel_order(id=order["order_id"], symbol=self.symbol)
            return {
                "symbol": executed["symbol"],
                "side": order["side"],
                "order_id": executed["id"],
                "position_side": order["position_side"],
                "qty": order["qty"],
                "price": order["price"],
            }
        except Exception as e:
            logging.error(f"error cancelling order {order} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return {}

    async def fetch_account(self):
        return

    async def fetch_ticks(self, from_id: int = None, do_print: bool = True):
        return

    async def fetch_ohlcvs(
        self, symbol: str = None, start_time: int = None, interval="1m", limit=1440
    ):
        ohlcvs = None
        # m -> minutes; h -> hours; d -> days; w -> weeks; M -> months
        interval_set = {
            "1m",
            "3m",
            "5m",
            "15m",
            "30m",
            "1h",
            "2h",
            "4h",
            "6h",
            "8h",
            "12h",
            "1d",
            "3d",
            "1w",
            "1M",
        }
        # endTime is respected first
        assert interval in interval_set, f"unsupported timeframe {interval}"
        end_time = int(await self.get_server_time() + 1000 * 60)
        params = {"endTime": end_time}
        if start_time is not None:
            params["startTime"] = int(start_time)
        try:
            ohlcvs = await self.cc.fetch_ohlcv(
                symbol=self.symbol if symbol is None else symbol,
                timeframe=interval,
                limit=limit,
                params=params,
            )
            keys = ["timestamp", "open", "high", "low", "close", "volume"]
            return [{k: elm[i] for i, k in enumerate(keys)} for elm in ohlcvs]
        except Exception as e:
            logging.error(f"error fetching ohlcv {e}")
            print_async_exception(ohlcvs)
            traceback.print_exc()

    async def get_all_income(
        self,
        symbol: str = None,
        start_time: int = None,
        income_type: str = "Trade",
        end_time: int = None,
    ):
        return await self.fetch_income(symbol=symbol, start_time=start_time, end_time=end_time)

    async def transfer_from_derivatives_to_spot(self, coin: str, amount: float):
        transferred = None
        try:
            transferred = await self.cc.transfer(coin, amount, "CONTRACT", "SPOT")
            return transferred
        except:
            logging.error(f"error transferring from derivatives to spot {e}")
            print_async_exception(transferred)
            traceback.print_exc()

    async def fetch_income(
        self,
        symbol: str = None,
        start_time: int = None,
        end_time: int = None,
    ):
        fetched = None
        incomed = {}
        try:
            limit = 100
            params = {"category": "linear", "limit": limit}
            if symbol is not None:
                params["symbol"] = symbol
            if end_time is not None:
                params["endTime"] = int(end_time)
            fetched = await self.cc.private_get_v5_position_closed_pnl(params)
            fetched["result"]["list"] = floatify(fetched["result"]["list"])
            while True:
                if fetched["result"]["list"] == []:
                    break
                for elm in fetched["result"]["list"]:
                    incomed[calc_hash(elm)] = elm
                if start_time is None:
                    break
                if fetched["result"]["list"][-1]["updatedTime"] <= start_time:
                    break
                params["cursor"] = fetched["result"]["nextPageCursor"]
                fetched = await self.cc.private_get_v5_position_closed_pnl(params)
                fetched["result"]["list"] = floatify(fetched["result"]["list"])
                logging.debug(
                    f"fetching income {ts_to_date_utc(fetched['result']['list'][-1]['updatedTime'])}"
                )
            return [
                {
                    "symbol": elm["symbol"],
                    "income": elm["closedPnl"],
                    "token": "USDT",
                    "timestamp": elm["updatedTime"],
                    "info": elm,
                    "transaction_id": elm["orderId"],
                    "trade_id": elm["orderId"],
                }
                for elm in sorted(incomed.values(), key=lambda x: x["updatedTime"])
            ]
            return sorted(incomed.values(), key=lambda x: x["updatedTime"])
        except Exception as e:
            logging.error(f"error fetching income {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return []

    async def fetch_latest_fills(self, cache=True):
        fetched = None
        try:
            age_limit = int(utc_ms() - 1000 * 60 * 60 * 48)
            fetched = await self.cc.fetch_my_trades(symbol=self.symbol, since=age_limit)
            fetched = [
                elm
                for elm in fetched
                if date_to_ts2(elm["info"]["filledTm"]) > age_limit
                and elm["info"]["symbol"] == self.symbol_id
            ]
            order_details = await self.fetch_order_details(
                [elm["info"]["orderId"] for elm in fetched], self.cache_path_fills
            )
            fills = [
                {
                    "order_id": elm["orderId"],
                    "symbol": self.symbol_id_map[elm["symbol"]],
                    "custom_id": elm["clientOrderId"],
                    "price": elm["avgPrice"],
                    "qty": elm["executedQty"],
                    "type": elm["type"].lower(),
                    "reduce_only": None,
                    "side": elm["side"].lower(),
                    "position_side": elm["positionSide"].lower(),
                    "timestamp": float(elm["updateTime"]),
                }
                for elm in order_details
            ]
            return sorted(fills, key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching latest fills {e}")
            print_async_exception(fetched)
            traceback.print_exc()

    async def fetch_fills(
        self,
        limit: int = 200,
        from_id: int = None,
        start_time: int = None,
        end_time: int = None,
    ):
        return []

    async def init_exchange_config(self):
        try:
            res = await self.cc.swap_v2_private_post_trade_margintype(
                params={"symbol": self.symbol_id, "marginType": "CROSSED"}
            )
            logging.info(f"cross mode set {res}")
        except Exception as e:
            logging.error(f"error setting cross mode: {e}")
        """
        # no hedge mode with bingx
        try:
            res = await self.cc.set_position_mode(hedged=True)
            logging.info(f"hedge mode set {res}")
        except Exception as e:
            logging.error(f"error setting hedge mode: {e}")
        """
        try:
            res = await self.cc.swap_v2_private_post_trade_leverage(
                params={"symbol": self.symbol_id, "side": "LONG", "leverage": 7}
            )
            logging.info(f"leverage set long {res}")
        except Exception as e:
            logging.error(f"error setting leverage long: {e}")
        try:
            res = await self.cc.swap_v2_private_post_trade_leverage(
                params={"symbol": self.symbol_id, "side": "SHORT", "leverage": 7}
            )
            logging.info(f"leverage set short {res}")
        except Exception as e:
            logging.error(f"error setting leverage short: {e}")
