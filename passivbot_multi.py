import os

if "NOJIT" not in os.environ:
    os.environ["NOJIT"] = "true"

import traceback
import argparse
import asyncio
import json
import hjson
import pprint
import numpy as np
from uuid import uuid4

from procedures import load_broker_code, load_user_info, utc_ms, make_get_filepath, load_live_config
from njit_funcs_recursive_grid import calc_recursive_entries_long, calc_recursive_entries_short
from njit_funcs import (
    calc_samples,
    calc_emas_last,
    calc_ema,
    calc_close_grid_long,
    calc_close_grid_short,
    calc_diff,
    qty_to_cost,
    cost_to_qty,
    calc_min_entry_qty,
    round_,
    round_up,
    round_dn,
    calc_pnl_long,
    calc_pnl_short,
)
from njit_multisymbol import calc_AU_allowance
from pure_funcs import (
    numpyize,
    filter_orders,
    multi_replace,
    shorten_custom_id,
    determine_side_from_order_tuple,
    str2bool,
)

import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S",
)


class Passivbot:
    def __init__(self, config: dict):
        self.config = config
        for key, default_val in [("auto_gs", True), ("long_enabled", True), ("short_enabled", True)]:
            if key not in self.config:
                self.config[key] = default_val
        self.user = config["user"]
        self.user_info = load_user_info(config["user"])
        self.exchange = self.user_info["exchange"]
        self.broker_code = load_broker_code(self.user_info["exchange"])
        self.custom_id_max_length = 36
        self.sym_padding = 17
        self.stop_websocket = False
        self.balance = 1e-12
        self.upnls = {}
        self.upd_timestamps = {
            "pnls": 0.0,
            "open_orders": {},
            "positions": {},
            "tickers": {},
        }
        self.hedge_mode = True
        self.positions = {}
        self.open_orders = {}
        self.pnls = []
        self.tickers = {}
        self.emas_long = {}
        self.emas_short = {}
        self.ema_minute = None
        self.symbol_ids = {}
        self.min_costs = {}
        self.min_qtys = {}
        self.qty_steps = {}
        self.price_steps = {}
        self.c_mults = {}
        self.coins = {}
        self.live_configs = {}
        self.stop_bot = False
        self.pnls_cache_filepath = make_get_filepath(f"caches/{self.exchange}/{self.user}_pnls.json")
        self.previous_execution_ts = 0
        self.recent_fill = False
        self.execution_delay_millis = max(3000.0, self.config["execution_delay_seconds"] * 1000)
        self.force_update_age_millis = 60 * 1000  # force update once a minute

    async def init_bot(self):
        max_len_symbol = max([len(s) for s in self.symbols])
        self.sym_padding = max(self.sym_padding, max_len_symbol + 1)

        # this argparser is used only internally
        parser = argparse.ArgumentParser(prog="passivbot", description="run passivbot")
        parser.add_argument("-sm", type=str, required=False, dest="short_mode", default=None)
        parser.add_argument("-lm", type=str, required=False, dest="long_mode", default=None)
        parser.add_argument(
            "-pp", type=float, required=False, dest="price_precision_multiplier", default=None
        )
        parser.add_argument("-ps", type=float, required=False, dest="price_step_custom", default=None)
        parser.add_argument("-lw", type=float, required=False, dest="WE_limit_long", default=None)
        parser.add_argument("-sw", type=float, required=False, dest="WE_limit_short", default=None)
        parser.add_argument("-lev", type=float, required=False, dest="leverage", default=None)
        parser.add_argument("-lc", type=str, required=False, dest="live_config_path", default=None)

        if os.path.isdir(self.config["live_configs_dir"]):
            live_configs_fnames = sorted(
                [f for f in os.listdir(self.config["live_configs_dir"]) if f.endswith(".json")]
            )
        else:
            live_configs_fnames = []
        self.args = {}
        for symbol in self.symbols:
            # look for an exact match first
            live_config_fname_l = [
                x for x in live_configs_fnames if x == self.coins[symbol] + "USDT.json"
            ]
            if not live_config_fname_l:
                live_config_fname_l = [x for x in live_configs_fnames if self.coins[symbol] in x]
            live_configs_dir_fname = (
                None
                if live_config_fname_l == []
                else os.path.join(self.config["live_configs_dir"], live_config_fname_l[0])
            )
            args = parser.parse_args(self.symbols[symbol].split())
            self.args[symbol] = args
            for path in [
                args.live_config_path,
                live_configs_dir_fname,
                self.config["default_config_path"],
            ]:
                if path is not None and os.path.exists(path):
                    try:
                        self.live_configs[symbol] = load_live_config(path)
                        logging.info(f"{symbol: <{max_len_symbol}} loaded live config: {path}")
                        break
                    except Exception as e:
                        logging.error(f"failed to load live config {symbol} {path} {e}")
            else:
                raise Exception(f"no usable live config found for {symbol}")

            if args.leverage is None:
                self.live_configs[symbol]["leverage"] = 10.0
            else:
                self.live_configs[symbol]["leverage"] = max(1.0, float(args.leverage))

            for pside in ["long", "short"]:
                if getattr(args, f"{pside}_mode") is None:
                    self.live_configs[symbol][pside]["enabled"] = self.config[f"{pside}_enabled"]
                    self.live_configs[symbol][pside]["mode"] = (
                        "normal"
                        if self.config[f"{pside}_enabled"]
                        else ("graceful_stop" if self.config["auto_gs"] else "manual")
                    )
                else:
                    if getattr(args, f"{pside}_mode") == "gs":
                        self.live_configs[symbol][pside]["enabled"] = False
                        self.live_configs[symbol][pside]["mode"] = "graceful_stop"
                    elif getattr(args, f"{pside}_mode") == "m":
                        self.live_configs[symbol][pside]["enabled"] = False
                        self.live_configs[symbol][pside]["mode"] = "manual"
                    elif getattr(args, f"{pside}_mode") == "n":
                        self.live_configs[symbol][pside]["enabled"] = True
                        self.live_configs[symbol][pside]["mode"] = "normal"
                    elif getattr(args, f"{pside}_mode") == "p":
                        self.live_configs[symbol][pside]["enabled"] = False
                        self.live_configs[symbol][pside]["mode"] = "panic"
                    elif getattr(args, f"{pside}_mode").lower() == "t":
                        self.live_configs[symbol][pside]["enabled"] = False
                        self.live_configs[symbol][pside]["mode"] = "tp_only"
                    else:
                        raise Exception(f"unknown {pside} mode: {getattr(args,f'{pside}_mode')}")
        modes = ["normal", "manual", "graceful_stop", "tp_only", "panic"]
        for mode in modes:
            for pside in ["long", "short"]:
                syms_ = [
                    s.replace("/USDT:USDT", "")
                    for s in self.symbols
                    if self.live_configs[s][pside]["mode"] == mode
                ]
                if len(syms_) > 0:
                    logging.info(
                        f"{pside: <5} mode: {mode: <{max([len(x) for x in modes])}}: {', '.join(syms_)}"
                    )
        for pside in ["long", "short"]:
            for symbol in self.symbols:
                # disable AU
                # if self.config["loss_allowance_pct"] != 0.0:
                # possible TODO: single coin auto unstuck in multi symbol mode
                if True:
                    for key in [
                        "auto_unstuck_delay_minutes",
                        "auto_unstuck_ema_dist",
                        "auto_unstuck_qty_pct",
                        "auto_unstuck_wallet_exposure_threshold",
                    ]:
                        self.live_configs[symbol][pside][key] = 0.0

        for f in ["exchange_config", "emas", "positions", "open_orders", "pnls"]:
            res = await getattr(self, f"update_{f}")()
            logging.info(f"initiating {f} {res}")
        self.set_wallet_exposure_limits()

    async def get_active_symbols(self):
        # get symbols with open orders and/or positions
        positions, balance = await self.fetch_positions()
        open_orders = await self.fetch_open_orders()
        return sorted(set([elm["symbol"] for elm in positions + open_orders]))

    async def init_symbols(self):
        # require symbols to be formatted to ccxt standard COIN/USDT:USDT

        self.markets_dict = await self.cca.load_markets()
        self.symbols = {}
        for symbol_ in sorted(set(self.config["symbols"])):
            symbol = symbol_
            if not symbol.endswith("/USDT:USDT"):
                coin_extracted = multi_replace(
                    symbol_, [("/", ""), (":", ""), ("USDT", ""), ("BUSD", ""), ("USDC", "")]
                )
                symbol_reformatted = coin_extracted + "/USDT:USDT"
                logging.info(f"Trying to reformat symbol {symbol_} to {symbol_reformatted}")
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
                    self.symbols[symbol] = (
                        self.config["symbols"][symbol_]
                        if isinstance(self.config["symbols"], dict)
                        else ""
                    )
        self.quote = "USDT"
        self.inverse = False
        self.symbol_ids = {
            symbol: self.markets_dict[symbol]["id"]
            for symbol in self.markets_dict
            if symbol.endswith(f":{self.quote}")
        }
        self.symbol_ids_inv = {v: k for k, v in self.symbol_ids.items()}
        active_symbols = await self.get_active_symbols()
        for symbol in active_symbols:
            if symbol not in self.symbols:
                if self.config["auto_gs"]:
                    logging.info(f"{symbol: <{self.sym_padding}} will be set to graceful stop mode")
                    self.symbols[symbol] = "-lm gs -sm gs"
                else:
                    logging.info(f"{symbol: <{self.sym_padding}} will be set to manual mode")

                    self.symbols[symbol] = "-lm m -sm m"

    def get_approved_symbols(self):
        pass

    def set_wallet_exposure_limits(self):
        # an active bot has normal mode or graceful stop mode with position
        for pside in ["long", "short"]:
            n_actives = 0
            for sym in self.live_configs:
                if self.live_configs[sym][pside]["mode"] == "normal" or (
                    self.live_configs[sym][pside]["mode"] == "graceful_stop"
                    and self.positions[sym][pside]["size"] != 0.0
                ):
                    n_actives += 1
            if not hasattr(self, "prev_n_actives"):
                self.prev_n_actives = {"long": 0, "short": 0}
            if self.prev_n_actives[pside] != n_actives:
                logging.info(f"n active {pside} bots: {self.prev_n_actives[pside]} -> {n_actives}")
                self.prev_n_actives[pside] = n_actives
            new_WE_limit = round_(
                self.config[f"TWE_{pside}"] / n_actives if n_actives > 0 else 0.01, 0.0001
            )
            for symbol in self.symbols:
                if getattr(self.args[symbol], f"WE_limit_{pside}") is None:
                    if self.live_configs[symbol][pside]["wallet_exposure_limit"] != new_WE_limit:
                        logging.info(
                            f"changing WE limit for {pside} {symbol}: {self.live_configs[symbol][pside]['wallet_exposure_limit']} -> {new_WE_limit}"
                        )
                        self.live_configs[symbol][pside]["wallet_exposure_limit"] = new_WE_limit
                else:
                    self.live_configs[symbol][pside]["wallet_exposure_limit"] = getattr(
                        self.args[symbol], f"WE_limit_{pside}"
                    )
                self.live_configs[symbol][pside]["wallet_exposure_limit"] = max(
                    self.live_configs[symbol][pside]["wallet_exposure_limit"], 0.01
                )

    def add_new_order(self, order, source="WS"):
        try:
            if not order or "id" not in order:
                return False
            if order["id"] not in {x["id"] for x in self.open_orders[order["symbol"]]}:
                self.open_orders[order["symbol"]].append(order)
                logging.info(
                    f"  created {order['symbol']: <{self.sym_padding}} {order['side']} {order['qty']} {order['position_side']} @ {order['price']} source: {source}"
                )
                return True
        except Exception as e:
            logging.error(f"failed to add order to self.open_orders {order} {e}")
            traceback.print_exc()
            return False

    def remove_cancelled_order(self, order: dict, source="WS"):
        try:
            if not order or "id" not in order:
                return False
            if order["id"] in {x["id"] for x in self.open_orders[order["symbol"]]}:
                self.open_orders[order["symbol"]] = [
                    x for x in self.open_orders[order["symbol"]] if x["id"] != order["id"]
                ]
                logging.info(
                    f"cancelled {order['symbol']: <{self.sym_padding}} {order['side']} {order['qty']} {order['position_side']} @ {order['price']} source: {source}"
                )
                return True
        except Exception as e:
            logging.error(f"failed to remove order from self.open_orders {order} {e}")
            traceback.print_exc()
            return False

    def handle_order_update(self, upd_list):
        try:
            for upd in upd_list:
                if upd["symbol"] not in self.symbols:
                    return
                if "filled" in upd and upd["filled"] is not None and upd["filled"] > 0.0:
                    # There was a fill, partial or full. Schedule update of open orders, pnls, position.
                    logging.info(
                        f"   filled {upd['symbol']: <{self.sym_padding}} {upd['side']} {upd['qty']} {upd['position_side']} @ {upd['price']} source: WS"
                    )
                    self.recent_fill = True
                elif upd["status"] in ["canceled", "expired"]:
                    # remove order from open_orders
                    self.remove_cancelled_order(upd)
                    self.upd_timestamps["open_orders"][upd["symbol"]] = utc_ms()
                elif upd["status"] == "open":
                    # add order to open_orders
                    self.add_new_order(upd)
                    self.upd_timestamps["open_orders"][upd["symbol"]] = utc_ms()
                else:
                    print("debug open orders unknown type", upd)
        except Exception as e:
            logging.error(f"error updating open orders from websocket {upd_list} {e}")
            traceback.print_exc()

    def handle_balance_update(self, upd):
        try:
            if self.balance != upd["USDT"]["total"]:
                logging.info(
                    f"balance changed: {self.balance} -> {upd['USDT']['total']} equity: {(upd['USDT']['total'] + self.calc_upnl_sum()):.4f} source: WS"
                )
            self.balance = max(upd["USDT"]["total"], 1e-12)
        except Exception as e:
            logging.error(f"error updating balance from websocket {upd} {e}")
            traceback.print_exc()

    def handle_ticker_update(self, upd):
        self.upd_timestamps["tickers"][upd["symbol"]] = utc_ms()  # update timestamp
        if (
            upd["bid"] != self.tickers[upd["symbol"]]["bid"]
            or upd["ask"] != self.tickers[upd["symbol"]]["ask"]
        ):
            ticker_new = {k: upd[k] for k in ["bid", "ask", "last"]}
            # print(f"ticker changed {upd['symbol']: <16} {self.tickers[upd['symbol']]} -> {ticker_new}")
            self.tickers[upd["symbol"]] = ticker_new

    def calc_upnl_sum(self):
        try:
            self.upnls = {}
            for sym in self.positions:
                self.upnls[sym] = (
                    calc_pnl_long(
                        self.positions[sym]["long"]["price"],
                        self.tickers[sym]["last"],
                        self.positions[sym]["long"]["size"],
                        self.inverse,
                        self.c_mults[sym],
                    )
                ) + calc_pnl_short(
                    self.positions[sym]["short"]["price"],
                    self.tickers[sym]["last"],
                    self.positions[sym]["short"]["size"],
                    self.inverse,
                    self.c_mults[sym],
                )
            return sum(self.upnls.values())
        except Exception as e:
            logging.error(f"error calculating upnl sum {e}")
            traceback.print_exc()
            return 0.0

    async def update_pnls(self):
        # fetch latest pnls
        # dump new pnls to cache
        age_limit = utc_ms() - 1000 * 60 * 60 * 24 * self.config["pnls_max_lookback_days"]
        missing_pnls = []
        if len(self.pnls) == 0:
            # load pnls from cache
            pnls_cache = []
            try:
                if os.path.exists(self.pnls_cache_filepath):
                    pnls_cache = json.load(open(self.pnls_cache_filepath))
            except Exception as e:
                logging.error(f"error loading {self.pnls_cache_filepath} {e}")
            # fetch pnls since latest timestamp
            if len(pnls_cache) > 0:
                if pnls_cache[0]["timestamp"] > age_limit + 1000 * 60 * 60 * 4:
                    # fetch missing pnls
                    res = await self.fetch_pnls(
                        start_time=age_limit - 1000, end_time=pnls_cache[0]["timestamp"]
                    )
                    if res in [None, False]:
                        return False
                    missing_pnls = res
                    pnls_cache = sorted(
                        {
                            elm["id"]: elm
                            for elm in pnls_cache + missing_pnls
                            if elm["timestamp"] >= age_limit
                        }.values(),
                        key=lambda x: x["timestamp"],
                    )
            self.pnls = pnls_cache
        start_time = self.pnls[-1]["timestamp"] if self.pnls else age_limit
        res = await self.fetch_pnls(start_time=start_time)
        if res in [None, False]:
            return False
        new_pnls = [x for x in res if x["id"] not in {elm["id"] for elm in self.pnls}]
        self.pnls = sorted(
            {elm["id"]: elm for elm in self.pnls + new_pnls if elm["timestamp"] > age_limit}.values(),
            key=lambda x: x["timestamp"],
        )
        if new_pnls:
            new_income = sum([x["pnl"] for x in new_pnls])
            if new_income != 0.0:
                logging.info(
                    f"{len(new_pnls)} new pnl{'s' if len(new_pnls) > 1 else ''} {new_income} {self.quote}"
                )
            try:
                json.dump(self.pnls, open(self.pnls_cache_filepath, "w"))
            except Exception as e:
                logging.error(f"error dumping pnls to {self.pnls_cache_filepath} {e}")
        self.upd_timestamps["pnls"] = utc_ms()
        return True

    async def update_open_orders(self):
        res = await self.fetch_open_orders()
        if res in [None, False]:
            return False
        open_orders = res
        oo_ids_old = {elm["id"] for sublist in self.open_orders.values() for elm in sublist}
        created_prints, cancelled_prints = [], []
        for oo in open_orders:
            if oo["id"] not in oo_ids_old:
                # there was a new open order not caught by websocket
                created_prints.append(
                    f"new order {oo['symbol']: <{self.sym_padding}} {oo['side']} {oo['qty']} {oo['position_side']} @ {oo['price']} source: REST"
                )
        oo_ids_new = {elm["id"] for elm in open_orders}
        for oo in [elm for sublist in self.open_orders.values() for elm in sublist]:
            if oo["id"] not in oo_ids_new:
                # there was an order cancellation not caught by websocket
                cancelled_prints.append(
                    f"cancelled {oo['symbol']: <{self.sym_padding}} {oo['side']} {oo['qty']} {oo['position_side']} @ {oo['price']} source: REST"
                )
        self.open_orders = {symbol: [] for symbol in self.open_orders}
        for elm in open_orders:
            if elm["symbol"] in self.open_orders:
                self.open_orders[elm["symbol"]].append(elm)
            else:
                logging.debug(
                    f"{elm['symbol']: <{self.sym_padding}} has open order {elm['position_side']} {elm['id']}, but is not under passivbot management"
                )
                logging.info(
                    f"debug {elm['symbol']: <{self.sym_padding}} has open order {elm['position_side']} {elm['id']}, but is not under passivbot management"
                )
        if len(created_prints) > 12:
            logging.info(f"{len(created_prints)} new open orders")
        else:
            for line in created_prints:
                logging.info(line)
        for line in cancelled_prints:
            logging.info(line)
        now = utc_ms()
        self.upd_timestamps["open_orders"] = {k: now for k in self.upd_timestamps["open_orders"]}
        return True

    async def update_positions(self):
        res = await self.fetch_positions()
        if res in [None, False]:
            return False
        positions_list_new, balance_new = res
        balance_old, self.balance = self.balance, max(balance_new, 1e-12)
        positions_new = {
            symbol: {"long": {"size": 0.0, "price": 0.0}, "short": {"size": 0.0, "price": 0.0}}
            for symbol in self.positions
        }
        for elm in positions_list_new:
            if elm["symbol"] not in self.positions:
                print(
                    f"debug {elm['symbol']: <{self.sym_padding}} has a {elm['position_side']} position, but is not under passivbot management"
                )
                logging.debug(
                    f"debug {elm['symbol']: <{self.sym_padding}} has a {elm['position_side']} position, but is not under passivbot management"
                )
            else:
                positions_new[elm["symbol"]][elm["position_side"]] = {
                    "size": abs(elm["size"]) * (-1.0 if elm["position_side"] == "short" else 1.0),
                    "price": elm["price"],
                }

        for symbol in self.positions:
            for pside in self.positions[symbol]:
                if self.positions[symbol][pside] != positions_new[symbol][pside]:
                    wallet_exposure = (
                        qty_to_cost(
                            positions_new[symbol][pside]["size"],
                            positions_new[symbol][pside]["price"],
                            self.inverse,
                            self.c_mults[symbol],
                        )
                        / self.balance
                    )
                    WE_ratio = (
                        wallet_exposure / self.live_configs[symbol][pside]["wallet_exposure_limit"]
                    )
                    if pside == "long":
                        pprice_diff = (
                            (
                                1.0
                                - self.tickers[symbol]["last"]
                                / positions_new[symbol]["long"]["price"]
                            )
                            if positions_new[symbol]["long"]["price"] > 0.0
                            else 0.0
                        )
                        upnl = calc_pnl_long(
                            positions_new[symbol][pside]["price"],
                            self.tickers[symbol]["last"],
                            positions_new[symbol][pside]["size"],
                            self.inverse,
                            self.c_mults[symbol],
                        )
                    else:
                        pprice_diff = (
                            self.tickers[symbol]["last"] / positions_new[symbol]["short"]["price"]
                            - 1.0
                            if positions_new[symbol]["short"]["price"] > 0.0
                            else 0.0
                        )
                        upnl = calc_pnl_short(
                            positions_new[symbol][pside]["price"],
                            self.tickers[symbol]["last"],
                            positions_new[symbol][pside]["size"],
                            self.inverse,
                            self.c_mults[symbol],
                        )
                    logging.info(
                        f"{symbol: <{self.sym_padding}} {pside} changed: {self.positions[symbol][pside]} -> {positions_new[symbol][pside]} WE ratio: {WE_ratio:.3f} pprice diff: {pprice_diff:.4f} upnl: {upnl:.4f}"
                    )
        self.positions = positions_new
        now = utc_ms()
        self.upd_timestamps["positions"] = {k: now for k in self.upd_timestamps["positions"]}
        if balance_old != balance_new:
            logging.info(
                f"balance changed: {balance_old} -> {balance_new} equity: {(balance_new + self.calc_upnl_sum()):.4f} source: REST"
            )
        return True

    async def update_tickers(self):
        res = await self.fetch_tickers()
        if res in [None, False]:
            return False
        tickers_new = res
        for symbol in self.symbols:
            if symbol not in tickers_new:
                raise Exception(f"{symbol} missing from tickers")
            ticker_new = {k: tickers_new[symbol][k] for k in ["bid", "ask", "last"]}
            if self.tickers[symbol] != ticker_new:
                # logging.info(f"{symbol} ticker changed: {self.tickers[symbol]} -> {ticker_new}")
                pass
            self.tickers[symbol] = ticker_new
            self.upd_timestamps["tickers"][symbol] = utc_ms()
        return True

    async def update_emas(self):
        if len(self.emas_long) == 0 or self.ema_minute is None:
            await self.init_emas()
            return True
        now_minute = int(utc_ms() // (1000 * 60) * (1000 * 60))
        if now_minute <= self.ema_minute:
            return True
        while self.ema_minute < int(round(now_minute - 1000 * 60)):
            for symbol in self.symbols:
                self.emas_long[symbol] = calc_ema(
                    self.alphas_long[symbol],
                    self.alphas__long[symbol],
                    self.emas_long[symbol],
                    self.prev_prices[symbol],
                )
                self.emas_short[symbol] = calc_ema(
                    self.alphas_short[symbol],
                    self.alphas__short[symbol],
                    self.emas_short[symbol],
                    self.prev_prices[symbol],
                )
            self.ema_minute += 1000 * 60
        for symbol in self.symbols:
            self.emas_long[symbol] = calc_ema(
                self.alphas_long[symbol],
                self.alphas__long[symbol],
                self.emas_long[symbol],
                self.tickers[symbol]["last"],
            )
            self.emas_short[symbol] = calc_ema(
                self.alphas_short[symbol],
                self.alphas__short[symbol],
                self.emas_short[symbol],
                self.tickers[symbol]["last"],
            )
            self.prev_prices[symbol] = self.tickers[symbol]["last"]

        self.ema_minute = now_minute
        return True

    async def init_emas(self):
        self.ema_spans_long, self.alphas_long, self.alphas__long, self.emas_long = {}, {}, {}, {}
        self.ema_spans_short, self.alphas_short, self.alphas__short, self.emas_short = {}, {}, {}, {}
        self.ema_minute = int(utc_ms() // (1000 * 60) * (1000 * 60))
        self.prev_prices = {}
        for sym in self.symbols:
            self.ema_spans_long[sym] = [
                self.live_configs[sym]["long"]["ema_span_0"],
                self.live_configs[sym]["long"]["ema_span_1"],
            ]
            self.ema_spans_long[sym] = numpyize(
                sorted(
                    self.ema_spans_long[sym]
                    + [(self.ema_spans_long[sym][0] * self.ema_spans_long[sym][1]) ** 0.5]
                )
            )
            self.ema_spans_short[sym] = [
                self.live_configs[sym]["short"]["ema_span_0"],
                self.live_configs[sym]["short"]["ema_span_1"],
            ]
            self.ema_spans_short[sym] = numpyize(
                sorted(
                    self.ema_spans_short[sym]
                    + [(self.ema_spans_short[sym][0] * self.ema_spans_short[sym][1]) ** 0.5]
                )
            )

            self.alphas_long[sym] = 2 / (self.ema_spans_long[sym] + 1)
            self.alphas__long[sym] = 1 - self.alphas_long[sym]
            self.alphas_short[sym] = 2 / (self.ema_spans_short[sym] + 1)
            self.alphas__short[sym] = 1 - self.alphas_short[sym]
        if self.tickers[next(iter(self.symbols))]["last"] == 0.0:
            logging.info(f"updating tickers...")
            await self.update_tickers()
        for sym in self.symbols:
            self.emas_long[sym] = np.repeat(self.tickers[sym]["last"], 3)
            self.emas_short[sym] = np.repeat(self.tickers[sym]["last"], 3)
            self.prev_prices[sym] = self.tickers[sym]["last"]
        ohs = None
        try:
            logging.info(f"fetching 15 min ohlcv for all symbols, initiating EMAs.")
            sym_list = list(self.symbols)
            ohs = await asyncio.gather(
                *[self.fetch_ohlcv(symbol, timeframe="15m") for symbol in sym_list]
            )
            samples_1m = [
                calc_samples(numpyize(oh)[:, [0, 5, 4]], sample_size_ms=60000) for oh in ohs
            ]
            for i in range(len(sym_list)):
                self.emas_long[sym_list[i]] = calc_emas_last(
                    samples_1m[i][:, 2], self.ema_spans_long[sym_list[i]]
                )
                self.emas_short[sym_list[i]] = calc_emas_last(
                    samples_1m[i][:, 2], self.ema_spans_short[sym_list[i]]
                )
            return True
        except Exception as e:
            logging.error(
                f"error fetching ohlcvs to initiate EMAs {e}. Using latest prices as starting EMAs"
            )
            traceback.print_exc()

    def calc_ideal_orders(self):
        unstuck_close_order = None
        stuck_positions = []
        for symbol in self.symbols:
            # check for stuck position
            if self.config["loss_allowance_pct"] == 0.0:
                # no auto unstuck
                break
            for pside in ["long", "short"]:
                if self.live_configs[symbol][pside]["mode"] in ["manual", "panic", "tp_only"]:
                    # no auto unstuck in these modes
                    continue
                if self.live_configs[symbol][pside]["wallet_exposure_limit"] == 0.0:
                    continue
                wallet_exposure = (
                    qty_to_cost(
                        self.positions[symbol][pside]["size"],
                        self.positions[symbol][pside]["price"],
                        self.inverse,
                        self.c_mults[symbol],
                    )
                    / self.balance
                )
                if (
                    wallet_exposure / self.live_configs[symbol][pside]["wallet_exposure_limit"]
                    > self.config["stuck_threshold"]
                ):
                    pprice_diff = (
                        1.0 - self.tickers[symbol]["last"] / self.positions[symbol]["long"]["price"]
                        if pside == "long"
                        else self.tickers[symbol]["last"] / self.positions[symbol]["short"]["price"]
                        - 1.0
                    )
                    if pprice_diff > 0.0:
                        # don't unstuck if position is in profit
                        stuck_positions.append((symbol, pside, pprice_diff))
        if stuck_positions:
            # logging.info(f"debug unstucking {sorted(stuck_positions, key=lambda x: x[2])}")
            sym, pside, pprice_diff = sorted(stuck_positions, key=lambda x: x[2])[0]
            AU_allowance = (
                calc_AU_allowance(
                    np.array([x["pnl"] for x in self.pnls]),
                    self.balance,
                    loss_allowance_pct=self.config["loss_allowance_pct"],
                )
                if len(self.pnls) > 0
                else 0.0
            )
            if AU_allowance > 0.0:
                close_price = (
                    max(
                        self.tickers[sym]["ask"],
                        round_up(self.emas_long[sym].max(), self.price_steps[sym]),
                    )
                    if pside == "long"
                    else min(
                        self.tickers[sym]["bid"],
                        round_dn(self.emas_short[sym].min(), self.price_steps[sym]),
                    )
                )
                upnl = (
                    calc_pnl_long(
                        self.positions[sym][pside]["price"],
                        self.tickers[sym]["last"],
                        self.positions[sym][pside]["size"],
                        self.inverse,
                        self.c_mults[sym],
                    )
                    if pside == "long"
                    else calc_pnl_short(
                        self.positions[sym][pside]["price"],
                        self.tickers[sym]["last"],
                        self.positions[sym][pside]["size"],
                        self.inverse,
                        self.c_mults[sym],
                    )
                )
                AU_allowance_pct = 1.0 if upnl >= 0.0 else min(1.0, AU_allowance / abs(upnl))
                AU_allowance_qty = round_(
                    abs(self.positions[sym][pside]["size"]) * AU_allowance_pct, self.qty_steps[sym]
                )
                close_qty = max(
                    calc_min_entry_qty(
                        close_price,
                        self.inverse,
                        self.c_mults[sym],
                        self.qty_steps[sym],
                        self.min_qtys[sym],
                        self.min_costs[sym],
                    ),
                    min(
                        abs(AU_allowance_qty),
                        round_(
                            cost_to_qty(
                                self.balance
                                * self.live_configs[sym][pside]["wallet_exposure_limit"]
                                * self.config["unstuck_close_pct"],
                                close_price,
                                self.inverse,
                                self.c_mults[sym],
                            ),
                            self.qty_steps[sym],
                        ),
                    ),
                )
                unstuck_close_order = {
                    "symbol": sym,
                    "position_side": pside,
                    "order": (
                        close_qty * (-1.0 if pside == "long" else 1.0),
                        close_price,
                        f"unstuck_close_{pside}",
                    ),
                }
                try:
                    if utc_ms() - self.prev_AU_print_ms > 1000 * 60:
                        line = f"Auto unstuck allowance: {AU_allowance:.3f} {self.quote}. Placing {pside} unstucking order for {sym} at {close_price}. Last price: {self.tickers[sym]['last']}"
                        logging.info(line)
                        self.prev_AU_print_ms = utc_ms()
                except:
                    self.prev_AU_print_ms = 0.0

        ideal_orders = {symbol: [] for symbol in self.symbols}
        for symbol in self.symbols:
            if self.hedge_mode:
                do_long = (
                    self.live_configs[symbol]["long"]["enabled"]
                    or self.positions[symbol]["long"]["size"] != 0.0
                )
                do_short = (
                    self.live_configs[symbol]["short"]["enabled"]
                    or self.positions[symbol]["short"]["size"] != 0.0
                )
            else:
                no_pos = (
                    self.positions[symbol]["long"]["size"] == 0.0
                    and self.positions[symbol]["short"]["size"] == 0.0
                )
                do_long = (no_pos and self.live_configs[symbol]["long"]["enabled"]) or self.positions[
                    symbol
                ]["long"]["size"] != 0.0
                do_short = (
                    no_pos and self.live_configs[symbol]["short"]["enabled"]
                ) or self.positions[symbol]["short"]["size"] != 0.0
            if self.live_configs[symbol]["long"]["mode"] == "panic":
                if self.positions[symbol]["long"]["size"] != 0.0:
                    # if in panic mode, only one close order at current market price
                    ideal_orders[symbol].append(
                        (
                            -abs(self.positions[symbol]["long"]["size"]),
                            self.tickers[symbol]["ask"],
                            "panic_close_long",
                        )
                    )
                # otherwise, no orders
            elif (
                self.live_configs[symbol]["long"]["mode"] == "graceful_stop"
                and self.positions[symbol]["long"]["size"] == 0.0
            ):
                # if graceful stop and no pos, don't open new pos
                pass
            elif do_long:
                entries_long = calc_recursive_entries_long(
                    self.balance,
                    self.positions[symbol]["long"]["size"],
                    self.positions[symbol]["long"]["price"],
                    self.tickers[symbol]["bid"],
                    self.emas_long[symbol].min(),
                    self.inverse,
                    self.qty_steps[symbol],
                    self.price_steps[symbol],
                    self.min_qtys[symbol],
                    self.min_costs[symbol],
                    self.c_mults[symbol],
                    self.live_configs[symbol]["long"]["initial_qty_pct"],
                    self.live_configs[symbol]["long"]["initial_eprice_ema_dist"],
                    self.live_configs[symbol]["long"]["ddown_factor"],
                    self.live_configs[symbol]["long"]["rentry_pprice_dist"],
                    self.live_configs[symbol]["long"]["rentry_pprice_dist_wallet_exposure_weighting"],
                    self.live_configs[symbol]["long"]["wallet_exposure_limit"],
                    self.live_configs[symbol]["long"]["auto_unstuck_ema_dist"],
                    self.live_configs[symbol]["long"]["auto_unstuck_wallet_exposure_threshold"],
                    self.live_configs[symbol]["long"]["auto_unstuck_delay_minutes"]
                    or self.live_configs[symbol]["long"]["auto_unstuck_qty_pct"],
                )
                if (
                    unstuck_close_order is not None
                    and unstuck_close_order["symbol"] == symbol
                    and unstuck_close_order["position_side"] == "long"
                ):
                    ideal_orders[symbol].append(unstuck_close_order["order"])
                    psize_ = max(
                        0.0,
                        round_(
                            abs(self.positions[symbol]["long"]["size"])
                            - abs(unstuck_close_order["order"][0]),
                            self.qty_steps[symbol],
                        ),
                    )
                    logging.debug(
                        f"creating unstucking order for {symbol} long: {unstuck_close_order['order']}"
                    )
                else:
                    psize_ = self.positions[symbol]["long"]["size"]
                closes_long = calc_close_grid_long(
                    self.live_configs[symbol]["long"]["backwards_tp"],
                    self.balance,
                    psize_,
                    self.positions[symbol]["long"]["price"],
                    self.tickers[symbol]["ask"],
                    self.emas_long[symbol].max(),
                    0,
                    0,
                    self.inverse,
                    self.qty_steps[symbol],
                    self.price_steps[symbol],
                    self.min_qtys[symbol],
                    self.min_costs[symbol],
                    self.c_mults[symbol],
                    self.live_configs[symbol]["long"]["wallet_exposure_limit"],
                    self.live_configs[symbol]["long"]["min_markup"],
                    self.live_configs[symbol]["long"]["markup_range"],
                    self.live_configs[symbol]["long"]["n_close_orders"],
                    self.live_configs[symbol]["long"]["auto_unstuck_wallet_exposure_threshold"],
                    self.live_configs[symbol]["long"]["auto_unstuck_ema_dist"],
                    self.live_configs[symbol]["long"]["auto_unstuck_delay_minutes"],
                    self.live_configs[symbol]["long"]["auto_unstuck_qty_pct"],
                )
                ideal_orders[symbol] += entries_long + closes_long
            if self.live_configs[symbol]["short"]["mode"] == "panic":
                if self.positions[symbol]["short"]["size"] != 0.0:
                    # if in panic mode, only one close order at current market price
                    ideal_orders[symbol].append(
                        (
                            abs(self.positions[symbol]["short"]["size"]),
                            self.tickers[symbol]["bid"],
                            "panic_close_short",
                        )
                    )
            elif (
                self.live_configs[symbol]["short"]["mode"] == "graceful_stop"
                and self.positions[symbol]["short"]["size"] == 0.0
            ):
                # if graceful stop and no pos, don't open new pos
                pass
            elif do_short:
                entries_short = calc_recursive_entries_short(
                    self.balance,
                    self.positions[symbol]["short"]["size"],
                    self.positions[symbol]["short"]["price"],
                    self.tickers[symbol]["ask"],
                    self.emas_short[symbol].max(),
                    self.inverse,
                    self.qty_steps[symbol],
                    self.price_steps[symbol],
                    self.min_qtys[symbol],
                    self.min_costs[symbol],
                    self.c_mults[symbol],
                    self.live_configs[symbol]["short"]["initial_qty_pct"],
                    self.live_configs[symbol]["short"]["initial_eprice_ema_dist"],
                    self.live_configs[symbol]["short"]["ddown_factor"],
                    self.live_configs[symbol]["short"]["rentry_pprice_dist"],
                    self.live_configs[symbol]["short"][
                        "rentry_pprice_dist_wallet_exposure_weighting"
                    ],
                    self.live_configs[symbol]["short"]["wallet_exposure_limit"],
                    self.live_configs[symbol]["short"]["auto_unstuck_ema_dist"],
                    self.live_configs[symbol]["short"]["auto_unstuck_wallet_exposure_threshold"],
                    self.live_configs[symbol]["short"]["auto_unstuck_delay_minutes"]
                    or self.live_configs[symbol]["short"]["auto_unstuck_qty_pct"],
                )
                if (
                    unstuck_close_order is not None
                    and unstuck_close_order["symbol"] == symbol
                    and unstuck_close_order["position_side"] == "short"
                ):
                    ideal_orders[symbol].append(unstuck_close_order["order"])
                    psize_ = -max(
                        0.0,
                        round_(
                            abs(self.positions[symbol]["short"]["size"])
                            - abs(unstuck_close_order["order"][0]),
                            self.qty_steps[symbol],
                        ),
                    )
                    logging.debug(
                        f"creating unstucking order for {symbol} short: {unstuck_close_order['order']}"
                    )
                else:
                    psize_ = self.positions[symbol]["short"]["size"]
                closes_short = calc_close_grid_short(
                    self.live_configs[symbol]["short"]["backwards_tp"],
                    self.balance,
                    psize_,
                    self.positions[symbol]["short"]["price"],
                    self.tickers[symbol]["bid"],
                    self.emas_short[symbol].min(),
                    0,
                    0,
                    self.inverse,
                    self.qty_steps[symbol],
                    self.price_steps[symbol],
                    self.min_qtys[symbol],
                    self.min_costs[symbol],
                    self.c_mults[symbol],
                    self.live_configs[symbol]["short"]["wallet_exposure_limit"],
                    self.live_configs[symbol]["short"]["min_markup"],
                    self.live_configs[symbol]["short"]["markup_range"],
                    self.live_configs[symbol]["short"]["n_close_orders"],
                    self.live_configs[symbol]["short"]["auto_unstuck_wallet_exposure_threshold"],
                    self.live_configs[symbol]["short"]["auto_unstuck_ema_dist"],
                    self.live_configs[symbol]["short"]["auto_unstuck_delay_minutes"],
                    self.live_configs[symbol]["short"]["auto_unstuck_qty_pct"],
                )
                ideal_orders[symbol] += entries_short + closes_short

        ideal_orders = {
            symbol: sorted(
                [x for x in ideal_orders[symbol] if x[0] != 0.0],
                key=lambda x: calc_diff(x[1], self.tickers[symbol]["last"]),
            )
            for symbol in ideal_orders
        }
        return {
            symbol: [
                {
                    "symbol": symbol,
                    "side": determine_side_from_order_tuple(x),
                    "position_side": "long" if "long" in x[2] else "short",
                    "qty": abs(x[0]),
                    "price": x[1],
                    "reduce_only": "close" in x[2],
                    "custom_id": x[2],
                }
                for x in ideal_orders[symbol]
            ]
            for symbol in ideal_orders
        }

    def calc_orders_to_cancel_and_create(self):
        ideal_orders = self.calc_ideal_orders()
        actual_orders = {}
        for symbol in self.open_orders:
            actual_orders[symbol] = []
            for x in self.open_orders[symbol]:
                actual_orders[symbol].append(
                    {
                        "symbol": x["symbol"],
                        "side": x["side"],
                        "position_side": x["position_side"],
                        "qty": abs(x["amount"]),
                        "price": x["price"],
                        "reduce_only": (x["position_side"] == "long" and x["side"] == "sell")
                        or (x["position_side"] == "short" and x["side"] == "buy"),
                        "id": x["id"],
                    }
                )
        keys = ("symbol", "side", "position_side", "qty", "price")
        to_cancel, to_create = [], []
        for symbol in actual_orders:
            to_cancel_, to_create_ = filter_orders(actual_orders[symbol], ideal_orders[symbol], keys)
            for pside in ["long", "short"]:
                if self.live_configs[symbol][pside]["mode"] == "manual":
                    # neither create nor cancel orders
                    to_cancel_ = [x for x in to_cancel_ if x["position_side"] != pside]
                    to_create_ = [x for x in to_create_ if x["position_side"] != pside]
                elif self.live_configs[symbol][pside]["mode"] == "tp_only":
                    # if take profit only mode, remove same pside entry orders
                    to_cancel_ = [
                        x
                        for x in to_cancel_
                        if (
                            x["position_side"] != pside
                            or (x["position_side"] == pside and x["reduce_only"])
                        )
                    ]
                    to_create_ = [
                        x
                        for x in to_create_
                        if (
                            x["position_side"] != pside
                            or (x["position_side"] == pside and x["reduce_only"])
                        )
                    ]
            to_cancel += to_cancel_
            to_create += to_create_
        return sorted(
            to_cancel, key=lambda x: calc_diff(x["price"], self.tickers[x["symbol"]]["last"])
        ), sorted(to_create, key=lambda x: calc_diff(x["price"], self.tickers[x["symbol"]]["last"]))

    async def force_update(self):
        # if some information has not been updated in a while, force update via REST
        coros_to_call = []
        now = utc_ms()
        for key in self.upd_timestamps:
            if isinstance(self.upd_timestamps[key], dict):
                for sym in self.upd_timestamps[key]:
                    if now - self.upd_timestamps[key][sym] > self.force_update_age_millis:
                        # logging.info(f"forcing update {key} {sym}")
                        coros_to_call.append((key, getattr(self, f"update_{key}")()))
                        break
            else:
                if now - self.upd_timestamps[key] > self.force_update_age_millis:
                    # logging.info(f"forcing update {key}")
                    coros_to_call.append((key, getattr(self, f"update_{key}")()))
        if any(coros_to_call):
            self.set_wallet_exposure_limits()
        res = await asyncio.gather(*[x[1] for x in coros_to_call])
        return res

    async def execute_to_exchange(self):
        # cancels wrong orders and creates missing orders
        # check whether to call any self.update_*()
        if utc_ms() - self.execution_delay_millis < self.previous_execution_ts:
            return True
        self.previous_execution_ts = utc_ms()
        try:
            if self.recent_fill:
                self.upd_timestamps["positions"] = {k: 0.0 for k in self.upd_timestamps["positions"]}
                self.upd_timestamps["open_orders"] = {
                    k: 0.0 for k in self.upd_timestamps["positions"]
                }
                self.upd_timestamps["pnls"] = 0.0
                self.recent_fill = False
            update_res = await self.force_update()
            if not all(update_res):
                print("debug", update_res)
                for i, key in enumerate(self.upd_timestamps):
                    if not update_res[i]:
                        logging.error(f"error with {key}")
                return
            to_cancel, to_create = self.calc_orders_to_cancel_and_create()

            # debug duplicates
            seen = set()
            for elm in to_cancel:
                key = str(elm["price"]) + str(elm["qty"])
                if key in seen:
                    print("debug duplicate", elm)
                seen.add(key)

            # format custom_id
            to_create = self.format_custom_ids(to_create)

            res = await self.execute_cancellations(to_cancel)
            for elm in res:
                self.remove_cancelled_order(elm, source="POST")
            res = await self.execute_orders(to_create)
            for elm in res:
                self.add_new_order(elm, source="POST")
            if to_cancel or to_create:
                await asyncio.gather(self.update_open_orders(), self.update_positions())
        except Exception as e:
            logging.error(f"error executing to exchange {e}")
            traceback.print_exc()
        finally:
            self.previous_execution_ts = utc_ms()

    def format_custom_ids(self, orders: [dict]) -> [dict]:
        new_orders = []
        for order in orders:
            order["custom_id"] = (
                shorten_custom_id(order["custom_id"] if "custom_id" in order else "") + uuid4().hex
            )[: self.custom_id_max_length]
            new_orders.append(order)
        return new_orders

    async def execution_loop(self):
        while True:
            if self.stop_websocket:
                break
            await self.update_emas()
            if utc_ms() - self.execution_delay_millis > self.previous_execution_ts:
                await self.execute_to_exchange()
            await asyncio.sleep(1.0)

    async def start_bot(self):
        await self.init_bot()
        logging.info("done initiating bot")
        logging.info("starting websockets")
        await asyncio.gather(self.execution_loop(), self.start_websockets())


async def main():
    parser = argparse.ArgumentParser(prog="passivbot", description="run passivbot")
    parser.add_argument("hjson_config_path", type=str, help="path to hjson passivbot meta config")
    parser_items = [
        ("s", "symbols", "symbols", str, ", comma separated (SYM1USDT,SYM2USDT,...)"),
        ("le", "long_enabled", "long_enabled", str2bool, " (y/n or t/f)"),
        ("se", "short_enabled", "short_enabled", str2bool, " (y/n or t/f)"),
        ("tl", "total_wallet_exposure_long", "TWE_long", float, ""),
        ("ts", "total_wallet_exposure_short", "TWE_short", float, ""),
        ("u", "user", "user", str, ""),
        ("lap", "loss_allowance_pct", "loss_allowance_pct", float, " (set to 0.0 to disable)"),
        ("pml", "pnls_max_lookback_days", "pnls_max_lookback_days", float, ""),
        ("st", "stuck_threshold", "stuck_threshold", float, ""),
        ("ucp", "unstuck_close_pct", "unstuck_close_pct", float, ""),
        ("eds", "execution_delay_seconds", "execution_delay_seconds", float, ""),
        ("lcd", "live_configs_dir", "live_configs_dir", str, ""),
        ("dcp", "default_config_path", "default_config_path", str, ""),
        ("ag", "auto_gs", "auto_gs", str2bool, " enabled (y/n or t/f)"),
    ]
    for k0, k1, d, t, h in parser_items:
        parser.add_argument(
            *[f"-{k0}", f"--{k1}"] + ([f"--{k1.replace('_', '-')}"] if "_" in k1 else []),
            type=t,
            required=False,
            dest=d,
            default=None,
            help=f"specify {k1}{h}, overriding value from live hjson config.",
        )
    max_n_restarts_per_day = 5
    cooldown_secs = 60
    restarts = []
    while True:
        args = parser.parse_args()
        config = hjson.load(open(args.hjson_config_path))
        for key in [x[2] for x in parser_items]:
            if getattr(args, key) is not None:
                if key == "symbols":
                    old_value = sorted(set(config["symbols"]))
                    new_value = sorted(set(args.symbols.split(",")))
                else:
                    old_value = config[key]
                    new_value = getattr(args, key)
                logging.info(f"changing {key}: {old_value} -> {new_value}")
                config[key] = new_value
        user_info = load_user_info(config["user"])
        if user_info["exchange"] == "bybit":
            from exchanges_multi.bybit import BybitBot

            bot = BybitBot(config)
        elif user_info["exchange"] == "binance":
            from exchanges_multi.binance import BinanceBot

            bot = BinanceBot(config)
        elif user_info["exchange"] == "bitget":
            from exchanges_multi.bitget import BitgetBot

            bot = BitgetBot(config)
        elif user_info["exchange"] == "okx":
            from exchanges_multi.okx import OKXBot

            bot = OKXBot(config)
        elif user_info["exchange"] == "bingx":
            from exchanges_multi.bingx import BingXBot

            bot = BingXBot(config)
        else:
            raise Exception(f"unknown exchange {user_info['exchange']}")
        try:
            await bot.start_bot()
        except Exception as e:
            logging.error(f"passivbot error {e}")
            traceback.print_exc()
        finally:
            try:
                await bot.ccp.close()
                await bot.cca.close()
            except:
                pass
        logging.info(f"restarting bot...")
        print()
        for z in range(cooldown_secs, -1, -1):
            print(f"\rcountdown {z}...  ")
            await asyncio.sleep(1)
        print()
        restarts.append(utc_ms())
        restarts = [x for x in restarts if x > utc_ms() - 1000 * 60 * 60 * 24]
        if len(restarts) > max_n_restarts_per_day:
            break


if __name__ == "__main__":
    asyncio.run(main())
