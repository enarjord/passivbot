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
    shorten_custom_id,
)
from njit_funcs import calc_diff, round_, round_up, round_dn, round_dynamic
from procedures import print_async_exception, utc_ms, assert_correct_ccxt_version

assert_correct_ccxt_version(ccxt=ccxt_async)


class HyperliquidBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.ccp = getattr(ccxt_pro, self.exchange)(
            {
                "walletAddress": self.user_info["wallet_address"],
                "privateKey": self.user_info["private_key"],
            }
        )
        self.ccp.options["defaultType"] = "swap"
        self.cca = getattr(ccxt_async, self.exchange)(
            {
                "walletAddress": self.user_info["wallet_address"],
                "privateKey": self.user_info["private_key"],
            }
        )
        self.cca.options["defaultType"] = "swap"
        self.max_n_cancellations_per_batch = 8
        self.max_n_creations_per_batch = 4
        self.quote = "USDC"
        self.hedge_mode = False
        self.significant_digits = {}
        self.szDecimals = {}

    async def init_bot(self):
        await self.init_symbols()
        for symbol in self.symbols:
            elm = self.markets_dict[symbol]
            self.symbol_ids[symbol] = elm["id"]
            self.min_costs[symbol] = (
                10.0 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
            )
            self.min_qtys[symbol] = (
                elm["precision"]["amount"]
                if elm["limits"]["amount"]["min"] is None
                else elm["limits"]["amount"]["min"]
            )
            self.qty_steps[symbol] = elm["precision"]["amount"]
            self.price_steps[symbol] = round_(10 ** -elm["precision"]["price"], 0.0000000001)
            self.szDecimals[symbol] = elm["info"]["szDecimals"]
            self.c_mults[symbol] = elm["contractSize"]
            self.coins[symbol] = symbol.replace(f"/{self.quote}:{self.quote}", "")
            self.tickers[symbol] = {"bid": 0.0, "ask": 0.0, "last": 0.0}
            self.open_orders[symbol] = []
            self.positions[symbol] = {
                "long": {"size": 0.0, "price": 0.0},
                "short": {"size": 0.0, "price": 0.0},
            }
            self.upd_timestamps["open_orders"][symbol] = 0.0
            self.upd_timestamps["tickers"][symbol] = 0.0
            self.upd_timestamps["positions"][symbol] = 0.0
        self.n_decimal_places = 6
        self.n_significant_digits = 5
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
                res[self.quote]["total"] = float(
                    [x for x in res["info"]["data"][0]["details"] if x["ccy"] == self.quote][0][
                        "cashBal"
                    ]
                )
                self.handle_balance_update(res)
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
                    res[i]["position_side"] = res[i]["info"]["posSide"]
                    res[i]["qty"] = res[i]["amount"]
                self.handle_order_update(res)
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
                for k in res:
                    self.handle_ticker_update(res[k])
            except Exception as e:
                print(f"exception watch_tickers {symbols}", e)
                traceback.print_exc()

    def determine_pos_side(self, order):
        # hyperliquid is not hedge mode
        if order["symbol"] in self.positions:
            if self.positions[order["symbol"]]["long"]["size"] != 0.0:
                return "long"
            elif self.positions[order["symbol"]]["short"]["size"] != 0.0:
                return "short"
            else:
                return "long" if order["side"] == "buy" else "short"
        else:
            return "long" if order["side"] == "buy" else "short"

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
        info = None
        try:
            info = await self.cca.fetch_balance()
            balance = float(info["info"]["marginSummary"]["accountValue"]) - sum(
                [float(x["position"]["unrealizedPnl"]) for x in info["info"]["assetPositions"]]
            )
            positions = [
                {
                    "symbol": x["position"]["coin"] + "/USDC:USDC",
                    "position_side": "long"
                    if (size := float(x["position"]["szi"])) > 0.0
                    else "short",
                    "size": size,
                    "price": float(x["position"]["entryPx"]),
                }
                for x in info["info"]["assetPositions"]
            ]

            return positions, balance
            # also fetches balance
            fetched_positions, fetched_balance = None, None
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
            print_async_exception(info)
            traceback.print_exc()
            return False

    async def fetch_tickers(self):
        fetched = None
        try:
            fetched = await asyncio.gather(*[self.cca.fetch_order_book(s) for s in self.symbols])
            return {
                x["symbol"]: {
                    "bid": x["bids"][0][0],
                    "ask": x["asks"][0][0],
                    "last": np.random.choice([x["bids"][0][0], x["asks"][0][0]]),
                }
                for x in fetched
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

    async def fetch_pnls(
        self,
        start_time: int = None,
        end_time: int = None,
    ):
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
                since=int(start_time), params={"endTime": int(end_time)}
            )
            for i in range(len(fetched)):
                fetched[i]["pnl"] = float(fetched[i]["info"]["closedPnl"])
            return sorted(fetched, key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error fetching pnl {e}")
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
        if len(orders) > self.max_n_cancellations_per_batch:
            # prioritize cancelling reduce-only orders
            try:
                reduce_only_orders = [x for x in orders if x["reduce_only"]]
                rest = [x for x in orders if not x["reduce_only"]]
                orders = (reduce_only_orders + rest)[: self.max_n_cancellations_per_batch]
            except Exception as e:
                logging.error(f"debug filter cancellations {e}")
        by_symbol = {}
        for order in orders:
            if order["symbol"] not in by_symbol:
                by_symbol[order["symbol"]] = []
            by_symbol[order["symbol"]].append(order)
        syms = sorted(by_symbol)
        res = await asyncio.gather(
            *[self.cca.cancel_orders([x["id"] for x in by_symbol[sym]], symbol=sym) for sym in syms]
        )
        cancellations = []
        for sym, elm in zip(syms, res):
            if "status" in elm and elm["status"] == "ok":
                for status, order in zip(elm["response"]["data"]["statuses"], by_symbol[sym]):
                    if status == "success":
                        cancellations.append(order)
        return cancellations
        return await self.execute_multiple(
            orders, "execute_cancellation", self.max_n_cancellations_per_batch
        )

    async def execute_order(self, order: dict) -> dict:
        res = await self.cca.create_limit_order(
            order["symbol"], order["side"], order["qty"], order["price"]
        )
        return res

    async def execute_orders(self, orders: [dict]) -> [dict]:
        if len(orders) == 0:
            return []
        to_execute = []
        for order in orders:
            to_execute.append(
                {
                    "symbol": order["symbol"],
                    "type": "limit",
                    "side": order["side"],
                    "amount": order["qty"],
                    "price": order["price"],
                    # "params": {
                    #    "orderType": {"limit": {"tif": "Alo"}},
                    #    "reduceOnly": order["reduce_only"],
                    # },
                }
            )
        print(to_execute)
        res = await self.cca.create_orders(to_execute)
        print(res)
        executed = []
        for i, order in enumerate(orders):
            if "info" in res[i] and "filled" in res[i]["info"] or "resting" in res[i]["info"]:
                executed.append({**res[i], **order})
        return executed

        custom_ids_map = {}
        for order in orders[: self.max_n_creations_per_batch]:
            to_execute.append(
                {
                    "type": "limit",
                    "symbol": order["symbol"],
                    "side": order["side"],
                    "ordType": "post_only",
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
            custom_ids_map[to_execute[-1]["params"]["clOrdId"]] = {**to_execute[-1], **order}
        executed = None
        try:
            executed = await self.cca.create_orders(to_execute)
        except Exception as e:
            logging.error(f"error executing orders {orders} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return []

        to_return = []
        for res in executed:
            try:
                if "status" in res and res["status"] == "rejected":
                    logging.info(f"order rejected: {res} {custom_ids_map[res['clientOrderId']]}")
                elif "clientOrderId" in res and res["clientOrderId"] in custom_ids_map:
                    for key in ["side", "position_side", "qty", "price", "symbol", "reduce_only"]:
                        res[key] = custom_ids_map[res["clientOrderId"]][key]
                    to_return.append(res)
            except Exception as e:
                logging.error(f"error executing order {res} {e}")
                traceback.print_exc()
        return to_return

    async def update_exchange_config(self):
        try:
            res = await self.cca.set_position_mode(True)
            logging.info(f"set hedge mode {res}")
        except Exception as e:
            if '"code":"59000"' in e.args[0]:
                logging.info(f"margin mode: {e}")
            else:
                logging.error(f"error setting hedge mode {e}")

        coros_to_call_margin_mode = {}
        for symbol in self.symbols:
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
        for symbol in self.symbols:
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

    def calc_ideal_orders(self):
        # okx has max 100 open orders. Drop orders whose pprice diff is greatest.
        ideal_orders = super().calc_ideal_orders()
        ideal_orders_tmp = []
        for s in ideal_orders:
            for x in ideal_orders[s]:
                ideal_orders_tmp.append({**x, **{"symbol": s}})
        ideal_orders_tmp = sorted(
            ideal_orders_tmp,
            key=lambda x: calc_diff(x["price"], self.tickers[x["symbol"]]["last"]),
        )[:100]
        ideal_orders = {symbol: [] for symbol in self.symbols}
        for x in ideal_orders_tmp:
            ideal_orders[x["symbol"]].append(x)
        return ideal_orders

    def format_custom_ids(self, orders: [dict]) -> [dict]:
        # okx needs broker code at the beginning of the custom_id
        new_orders = []
        for order in orders:
            order["custom_id"] = (
                self.broker_code
                + shorten_custom_id(order["custom_id"] if "custom_id" in order else "")
                + uuid4().hex
            )[: self.custom_id_max_length]
            new_orders.append(order)
        return new_orders

    def px_round(self, val, symbol, direction=""):
        if direction == "up":
            return round_dynamic(round_up(val, self.price_steps[symbol]), self.n_significant_digits)
        elif direction in ["dn", "down"]:
            return round_dynamic(round_dn(val, self.price_steps[symbol]), self.n_significant_digits)
        else:
            return round_dynamic(round_(val, self.price_steps[symbol]), self.n_significant_digits)

    def sz_round(self, val, symbol, direction=""):
        if direction == "up":
            return round_up(val, self.qty_steps[symbol])
        elif direction in ["dn", "down"]:
            return round_dn(val, self.qty_steps[symbol])
        else:
            return round_(val, self.qty_steps[symbol])
