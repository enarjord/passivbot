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
from utils import ts_to_date, symbol_to_coin, coin_to_symbol, utc_ms
from config_utils import require_live_value
from pure_funcs import (
    multi_replace,
    floatify,
    calc_hash,
    shorten_custom_id,
)

calc_diff = pbr.calc_diff
round_ = pbr.round_
round_up = pbr.round_up
round_dn = pbr.round_dn
round_dynamic = pbr.round_dynamic
round_dynamic_up = pbr.round_dynamic_up
round_dynamic_dn = pbr.round_dynamic_dn
from procedures import print_async_exception, assert_correct_ccxt_version
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
        self.custom_id_max_length = 34

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
                    "symbol": self.coin_to_symbol(x["position"]["coin"]),
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
                self.coin_to_symbol(coin): {
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
                f"debug fetching pnls {ts_to_date(fetched[-1]['timestamp'])} len {len(fetched)}"
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
        executed = None
        try:
            executed = await self.cca.cancel_order(
                order["id"],
                symbol=order["symbol"],
                params=(
                    {"vaultAddress": self.user_info["wallet_address"]}
                    if self.user_info["is_vault"]
                    else {}
                ),
            )
            return executed
        except Exception as e:
            logging.error(f"error cancelling order {order} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return {}

    def did_cancel_order(self, executed, order=None) -> bool:
        if isinstance(executed, list) and len(executed) == 1:
            return self.did_cancel_order(executed[0])
        try:
            return "status" in executed and executed["status"] == "success"
        except:
            return False

    def get_order_execution_params(self, order: dict) -> dict:
        # defined for each exchange
        params = {
            "reduceOnly": order["reduce_only"],
            "timeInForce": (
                "Alo" if require_live_value(self.config, "time_in_force") == "post_only" else "Gtc"
            ),
            "clientOrderId": order["custom_id"],  # TODO
        }
        if self.user_info["is_vault"]:
            params["vaultAddress"] = self.user_info["wallet_address"]
        return params

    async def execute_order(self, order: dict) -> dict:
        executed = None
        try:
            executed = await super().execute_order(order)
            return executed
        except Exception as e:
            try:
                if self.adjust_min_cost_on_error(e):
                    return {}
            except Exception as e0:
                logging.error(f"error with adjust_min_cost_on_error {e0}")
            logging.error(f"error executing order {order} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return {}

    async def execute_orders(self, orders: [dict]) -> [dict]:
        return await self.execute_multiple(orders, "execute_order")

    def did_create_order(self, executed) -> bool:
        did_create = super().did_create_order(executed)
        try:
            return did_create and (
                "info" in executed and ("filled" in executed["info"] or "resting" in executed["info"])
            )
        except:
            return False

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
                            self.config_get(["live", "leverage"], symbol=symbol),
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

    async def calc_ideal_orders(self, allow_unstuck: bool = True):
        # hyperliquid needs custom price rounding
        ideal_orders = await super().calc_ideal_orders(allow_unstuck=allow_unstuck)
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

    def format_custom_id_single(self, order_type_id: int) -> str:
        formatted = super().format_custom_id_single(order_type_id)
        return (formatted)[: self.custom_id_max_length]
