from passivbot import Passivbot, logging, get_function_name
from uuid import uuid4
import ccxt.pro as ccxt_pro
import ccxt.async_support as ccxt_async
import pprint
import asyncio
import traceback
import json
import numpy as np
import passivbot_rust as pbr
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


class HyperliquidBot(Passivbot):
    def __init__(self, config: dict):
        super().__init__(config)
        self.quote = "USDC"
        self.hedge_mode = False
        self.significant_digits = {}
        if "is_vault" not in self.user_info or self.user_info["is_vault"] == "":
            logging.info(
                f"parameter 'is_vault' missing from api-keys.json for user {self.user}. Setting to false"
            )
            self.user_info["is_vault"] = False
        self.max_n_concurrent_ohlcvs_1m_updates = 2

    def create_ccxt_sessions(self):
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

    def set_market_specific_settings(self):
        super().set_market_specific_settings()
        for symbol in self.markets_dict:
            elm = self.markets_dict[symbol]
            self.symbol_ids[symbol] = elm["id"]
            self.min_costs[symbol] = (
                10.0 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
            )
            self.min_costs[symbol] = pbr.round_(self.min_costs[symbol] * 1.01, 0.01)
            self.qty_steps[symbol] = elm["precision"]["amount"]
            self.min_qtys[symbol] = (
                self.qty_steps[symbol]
                if elm["limits"]["amount"]["min"] is None
                else elm["limits"]["amount"]["min"]
            )
            self.price_steps[symbol] = elm["precision"]["price"]
            self.c_mults[symbol] = elm["contractSize"]
            self.max_leverage[symbol] = (
                int(elm["info"]["maxLeverage"]) if "maxLeverage" in elm["info"] else 0
            )
        self.n_decimal_places = 6
        self.n_significant_figures = 5

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
        while True:
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
        # hyperliquid is not hedge mode
        if order["symbol"] in self.positions:
            if self.positions[order["symbol"]]["long"]["size"] != 0.0:
                return "long"
            elif self.positions[order["symbol"]]["short"]["size"] != 0.0:
                return "short"
            else:
                return "long" if order["side"] == "buy" else "short"
        else:
            if "reduceOnly" in order:
                if order["side"] == "buy":
                    return "short" if order["reduceOnly"] else "long"
                if order["side"] == "sell":
                    return "long" if order["reduceOnly"] else "short"
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
                    "position_side": (
                        "long" if (size := float(x["position"]["szi"])) > 0.0 else "short"
                    ),
                    "size": size,
                    "price": float(x["position"]["entryPx"]),
                }
                for x in info["info"]["assetPositions"]
            ]

            return positions, balance
        except Exception as e:
            logging.error(f"error fetching positions and balance {e}")
            print_async_exception(info)
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

    async def fetch_ohlcvs_1m(self, symbol: str, since: float = None, limit=None):
        n_candles_limit = 5000 if limit is None else limit
        result = await self.cca.fetch_ohlcv(
            symbol,
            timeframe="1m",
            limit=n_candles_limit,
            since=int(self.get_exchange_time() - 1000 * 60 * n_candles_limit * 0.95),
        )
        return result

    async def fetch_pnls(
        self,
        start_time: int = None,
        end_time: int = None,
        limit=None,
    ):
        # hyperliquid fetches from past to future
        if limit is None:
            limit = 2000
        if start_time is None:
            # hyperliquid returns latest trades if no time frame is passed
            return await self.fetch_pnl(limit=limit)
        all_fetched = {}
        prev_hash = ""
        while True:
            fetched = await self.fetch_pnl(start_time=start_time, limit=limit)
            if fetched == []:
                break
            for elm in fetched:
                all_fetched[elm["id"]] = elm
            if len(fetched) < limit:
                break
            if end_time and fetched[-1]["timestamp"] >= end_time:
                break
            new_hash = calc_hash(fetched)
            if prev_hash == new_hash:
                print("debug pnls hash", prev_hash, new_hash)
                break
            prev_hash = new_hash
            logging.info(
                f"debug fetching pnls {ts_to_date_utc(fetched[-1]['timestamp'])} len {len(fetched)}"
            )
            start_time = fetched[-1]["timestamp"] - 1000
            limit = 2000
        return sorted(all_fetched.values(), key=lambda x: x["timestamp"])

    async def fetch_pnl(
        self,
        start_time: int = None,
        limit=None,
    ):
        fetched = None
        try:
            if start_time is None:
                fetched = await self.cca.fetch_my_trades(limit=limit)
            else:
                fetched = await self.cca.fetch_my_trades(since=max(1, int(start_time)), limit=limit)
            for i in range(len(fetched)):
                fetched[i]["pnl"] = float(fetched[i]["info"]["closedPnl"])
                fetched[i]["position_side"] = (
                    "long" if "long" in fetched[i]["info"]["dir"].lower() else "short"
                )
            return sorted(fetched, key=lambda x: x["timestamp"])
        except Exception as e:
            logging.error(f"error with {get_function_name()} {e}")
            print_async_exception(fetched)
            traceback.print_exc()
            return False

    async def execute_cancellation(self, order: dict) -> dict:
        return await self.execute_cancellations([order])

    async def execute_cancellations(self, orders: [dict]) -> [dict]:
        res = None
        try:
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
            by_symbol = {}
            for order in orders:
                if order["symbol"] not in by_symbol:
                    by_symbol[order["symbol"]] = []
                by_symbol[order["symbol"]].append(order)
            syms = sorted(by_symbol)
            res = await asyncio.gather(
                *[
                    self.cca.cancel_orders(
                        [x["id"] for x in by_symbol[sym]],
                        symbol=sym,
                        params=(
                            {"vaultAddress": self.user_info["wallet_address"]}
                            if self.user_info["is_vault"]
                            else {}
                        ),
                    )
                    for sym in syms
                ]
            )
            cancellations = []
            for sym, elm in zip(syms, res):
                if "status" in elm and elm["status"] == "ok":
                    for status, order in zip(elm["response"]["data"]["statuses"], by_symbol[sym]):
                        if status == "success":
                            cancellations.append(order)
            return cancellations
        except Exception as e:
            logging.error(f"error executing cancellations {e}")
            print_async_exception(res)
            traceback.print_exc()

    async def execute_order(self, order: dict) -> dict:
        return await self.execute_orders([order])

    async def execute_orders(self, orders: [dict]) -> [dict]:
        if len(orders) == 0:
            return []
        to_execute = []
        for order in orders:
            to_execute.append(
                {
                    "symbol": order["symbol"],
                    "type": order["type"] if "type" in order else "limit",
                    "side": order["side"],
                    "amount": order["qty"],
                    "price": order["price"],
                    "params": {
                        # "orderType": {"limit": {"tif": "Alo"}},
                        # "cloid": order["custom_id"],
                        "reduceOnly": order["reduce_only"],
                        "timeInForce": (
                            "Alo" if self.config["live"]["time_in_force"] == "post_only" else "Gtc"
                        ),
                    },
                }
            )
        try:
            res = await self.cca.create_orders(
                to_execute,
                params=(
                    {"vaultAddress": self.user_info["wallet_address"]}
                    if self.user_info["is_vault"]
                    else {}
                ),
            )
        except Exception as e:
            if self.adjust_min_cost_on_error(e):
                return []
            else:
                raise
        executed = []
        for ex, order in zip(res, orders):
            if "info" in ex and "filled" in ex["info"] or "resting" in ex["info"]:
                executed.append({**ex, **order})
        return executed

    def adjust_min_cost_on_error(self, error):
        any_adjusted = False
        successful_orders = []
        str_e = error.args[0]
        error_json = json.loads(str_e[str_e.find("{") :])
        if (
            "response" in error_json
            and "data" in error_json["response"]
            and "statuses" in error_json["response"]["data"]
        ):
            for elm in error_json["response"]["data"]["statuses"]:
                if "error" in elm:
                    if "Order must have minimum value of $10" in elm["error"]:
                        asset_id = int(elm["error"][elm["error"].find("asset=") + 6 :])
                        for symbol in self.markets_dict:
                            if (
                                "baseId" in self.markets_dict[symbol]["info"]
                                and self.markets_dict[symbol]["info"]["baseId"] == asset_id
                            ):
                                break
                        else:
                            raise Exception(f"No symbol match for asset_id={asset_id}")
                        new_min_cost = pbr.round_(self.min_costs[symbol] * 1.1, 0.1)
                        logging.info(
                            f"caught {elm['error']} {symbol}. Upping min_cost from {self.min_costs[symbol]} to {new_min_cost}"
                        )
                        self.min_costs[symbol] = new_min_cost
                        any_adjusted = True
        return any_adjusted

    def symbol_is_eligible(self, symbol):
        try:
            if (
                "onlyIsolated" in self.markets_dict[symbol]["info"]
                and self.markets_dict[symbol]["info"]["onlyIsolated"]
            ):
                return False
            if float(self.markets_dict[symbol]["info"]["openInterest"]) == 0.0:
                return False
        except Exception as e:
            logging.error(f"error with symbol_is_eligible {e} {symbol}")
            return False
        return True

    async def update_exchange_config_by_symbols(self, symbols):
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
        for sym in ideal_orders:
            for i in range(len(ideal_orders[sym])):
                if ideal_orders[sym][i]["side"] == "sell":
                    ideal_orders[sym][i]["price"] = round_dynamic_up(
                        round(ideal_orders[sym][i]["price"], self.n_decimal_places),
                        self.n_significant_figures,
                    )
                elif ideal_orders[sym][i]["side"] == "buy":
                    ideal_orders[sym][i]["price"] = round_dynamic_dn(
                        round(ideal_orders[sym][i]["price"], self.n_decimal_places),
                        self.n_significant_figures,
                    )
                else:
                    ideal_orders[sym][i]["price"] = round_dynamic(
                        round(ideal_orders[sym][i]["price"], self.n_decimal_places),
                        self.n_significant_figures,
                    )
                ideal_orders[sym][i]["price"] = round_(
                    ideal_orders[sym][i]["price"], self.price_steps[sym]
                )
        return ideal_orders
