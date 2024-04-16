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
from copy import deepcopy

from procedures import (
    load_broker_code,
    load_user_info,
    utc_ms,
    make_get_filepath,
    load_live_config,
    get_file_mod_utc,
    get_first_ohlcv_timestamps,
    load_hjson_config,
)
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
    round_dynamic,
    calc_pnl,
    calc_pnl_long,
    calc_pnl_short,
    calc_pprice_diff,
)
from njit_multisymbol import calc_AU_allowance
from pure_funcs import (
    numpyize,
    denumpyize,
    filter_orders,
    multi_replace,
    shorten_custom_id,
    determine_side_from_order_tuple,
    str2bool,
    symbol2coin,
    add_missing_params_to_hjson_live_multi_config,
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
        self.user = config["user"]
        self.user_info = load_user_info(config["user"])
        self.exchange = self.user_info["exchange"]
        self.broker_code = load_broker_code(self.user_info["exchange"])
        self.custom_id_max_length = 36
        self.sym_padding = 17
        self.stop_websocket = False
        self.balance = 1e-12
        self.upd_timestamps = {
            "pnls": 0.0,
            "open_orders": 0.0,
            "positions": 0.0,
            "tickers": 0.0,
        }
        self.hedge_mode = True
        self.inverse = False
        self.active_symbols = []
        self.fetched_positions = []
        self.fetched_open_orders = []
        self.open_orders = {}
        self.positions = {}
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
        self.live_configs = {}
        self.stop_bot = False
        self.pnls_cache_filepath = make_get_filepath(f"caches/{self.exchange}/{self.user}_pnls.json")
        self.ohlcvs_cache_dirpath = make_get_filepath(f"caches/{self.exchange}/ohlcvs/")
        self.previous_execution_ts = 0
        self.recent_fill = False
        self.execution_delay_millis = max(3000.0, self.config["execution_delay_seconds"] * 1000)
        self.force_update_age_millis = 60 * 1000  # force update once a minute
        self.quote = "USDT"

    def set_live_configs(self):
        # live configs priority:
        # 1) -lc path from hjson multi config
        # 2) live config from live configs dir, matching name or coin
        # 3) live config from default config path
        # 4) universal live config given in hjson multi config

        if os.path.isdir(self.config["live_configs_dir"]):
            # live config candidates from live configs dir
            live_configs_fnames = sorted(
                [f for f in os.listdir(self.config["live_configs_dir"]) if f.endswith(".json")]
            )
        else:
            live_configs_fnames = []
        for symbol in self.approved_symbols:
            # look for an exact match first
            coin = symbol2coin(symbol)
            live_config_fname_l = [
                x for x in live_configs_fnames if x == coin + "USDT.json" or x == coin + ".json"
            ]
            if not live_config_fname_l:
                # then look if coin name in filename
                live_config_fname_l = [x for x in live_configs_fnames if coin in x]
            live_configs_dir_fname = (
                None
                if live_config_fname_l == []
                else os.path.join(self.config["live_configs_dir"], live_config_fname_l[0])
            )
            for path in [
                self.args[symbol].live_config_path,
                live_configs_dir_fname,
                self.config["default_config_path"],
            ]:
                if path is not None and os.path.exists(path):
                    try:
                        self.live_configs[symbol] = deepcopy(load_live_config(path))
                        logging.info(f"{symbol: <{self.max_len_symbol}} loaded live config: {path}")
                        break
                    except Exception as e:
                        logging.error(f"failed to load live config {symbol} {path} {e}")
            else:
                try:
                    self.live_configs[symbol] = deepcopy(self.config["universal_live_config"])
                    logging.info(
                        f"{symbol: <{self.max_len_symbol}} loaded universal live config from hjson config"
                    )
                except Exception as e:
                    logging.error(f"failed to apply universal_live_config {e}")
                    raise Exception(f"no usable live config found for {symbol}")

            if self.args[symbol].leverage is None:
                self.live_configs[symbol]["leverage"] = 10.0
            else:
                self.live_configs[symbol]["leverage"] = max(1.0, float(self.args[symbol].leverage))

            for pside in ["long", "short"]:
                if getattr(self.args[symbol], f"{pside}_mode") is None:
                    self.live_configs[symbol][pside]["enabled"] = self.config[f"{pside}_enabled"]
                    self.live_configs[symbol][pside]["mode"] = (
                        "normal"
                        if self.config[f"{pside}_enabled"]
                        else ("graceful_stop" if self.config["auto_gs"] else "manual")
                    )
                else:
                    if getattr(self.args[symbol], f"{pside}_mode") == "gs":
                        self.live_configs[symbol][pside]["enabled"] = False
                        self.live_configs[symbol][pside]["mode"] = "graceful_stop"
                    elif getattr(self.args[symbol], f"{pside}_mode") == "m":
                        self.live_configs[symbol][pside]["enabled"] = False
                        self.live_configs[symbol][pside]["mode"] = "manual"
                    elif getattr(self.args[symbol], f"{pside}_mode") == "n":
                        self.live_configs[symbol][pside]["enabled"] = True
                        self.live_configs[symbol][pside]["mode"] = "normal"
                    elif getattr(self.args[symbol], f"{pside}_mode") == "p":
                        self.live_configs[symbol][pside]["enabled"] = False
                        self.live_configs[symbol][pside]["mode"] = "panic"
                    elif getattr(self.args[symbol], f"{pside}_mode").lower() == "t":
                        self.live_configs[symbol][pside]["enabled"] = False
                        self.live_configs[symbol][pside]["mode"] = "tp_only"
                    else:
                        raise Exception(
                            f"unknown {pside} mode: {getattr(self.args[symbol],f'{pside}_mode')}"
                        )
                # disable AU and set backwards TP
                for key, val in [
                    ("auto_unstuck_delay_minutes", 0.0),
                    ("auto_unstuck_qty_pct", 0.0),
                    ("auto_unstuck_wallet_exposure_threshold", 0.0),
                    ("auto_unstuck_ema_dist", 0.0),
                    ("backwards_tp", True),
                ]:
                    self.live_configs[symbol][pside][key] = val

        # print symbols and modes
        modes = ["normal", "manual", "graceful_stop", "tp_only", "panic"]
        for mode in modes:
            for pside in ["long", "short"]:
                syms_ = [
                    symbol2coin(s)
                    for s in self.live_configs
                    if self.live_configs[s][pside]["mode"] == mode
                ]
                if len(syms_) > 0:
                    logging.info(
                        f"{pside: <5} mode: {mode: <{max([len(x) for x in modes])}}: {', '.join(syms_)}"
                    )

    def pad_sym(self, symbol):
        return f"{symbol: <{self.sym_padding}}"

    def find_file_mod_utc_time_diff(self):
        try:
            fname = f"{self.user}_testfile_{uuid4().hex}.txt"
            with open(fname, "w") as f:
                f.write("\n")
            now = utc_ms()
            fmod_time = get_file_mod_utc(fname)
            os.remove(fname)
            return now - fmod_time
        except Exception as e:
            logging.error(f"error with find_file_mod_utc_time_diff {e}")

    async def init_bot(self):
        logging.info(f"initiating markets...")
        await self.init_market_dict()
        logging.info(f"initiating tickers...")
        await self.update_tickers()
        logging.info(f"initiating balance, positions...")
        await self.update_positions()
        logging.info(f"initiating open orders...")
        await self.update_open_orders()
        await self.update_approved_symbols()
        self.set_live_configs()
        self.ohlcv_maintainer = asyncio.create_task(self.maintain_ohlcvs())
        for i in range(10000):
            await asyncio.sleep(1)
            upd_timestamps = [v for v in self.ohlcv_upd_timestamps.values()]
            if all(upd_timestamps):
                break
            logging.info(
                f"updating ohlcvs... {len([x for x in upd_timestamps if x])} / {len(upd_timestamps)}"
            )

        for f in ["exchange_config", "emas", "pnls"]:
            res = await getattr(self, f"update_{f}")()
            logging.info(f"initiating {f} {res}")
        self.set_wallet_exposure_limits()
        if not self.forager_mode:
            res = self.ohlcv_maintainer.cancel()
            logging.info(f"not in foarger mode; cancelled ohlcv maintainer {res}")

    async def get_active_symbols(self):
        # get symbols with open orders and/or positions
        positions, balance = await self.fetch_positions()
        open_orders = await self.fetch_open_orders()
        return sorted(set([elm["symbol"] for elm in positions + open_orders]))

    def format_symbol(self, symbol: str, suppress_log=False) -> str:
        formatted = f"{symbol2coin(symbol)}/{self.quote}:{self.quote}"
        if not suppress_log and symbol != formatted:
            logging.info(f"formatted {symbol} -> {formatted}")
        return formatted

    async def init_market_dict(self):
        self.markets_dict = await self.cca.load_markets()
        self.all_symbols = []  # all active symbols on exchange
        for symbol in self.markets_dict:
            if all(
                [
                    self.markets_dict[symbol]["active"],
                    self.markets_dict[symbol]["swap"],
                    self.markets_dict[symbol]["linear"],
                    symbol.endswith(f"/{self.quote}:{self.quote}"),
                ]
            ):
                self.all_symbols.append(symbol)
        self.all_symbols = sorted(set(self.all_symbols))
        self.set_market_specific_settings()

    def set_market_specific_settings(self):
        # set min cost, min qty, price step, qty step, c_mult
        # defined individually for each exchange
        self.symbol_ids = {symbol: self.markets_dict[symbol]["id"] for symbol in self.markets_dict}
        self.symbol_ids_inv = {v: k for k, v in self.symbol_ids.items()}

    def update_active_symbols(self):
        symbols_with_pos_or_open_orders = sorted(
            set([x["symbol"] for x in self.fetched_positions + self.fetched_open_orders])
        )
        # put on gs/manual mode
        for symbol in symbols_with_pos_or_open_orders:
            if symbol not in self.approved_symbols:
                if self.config["auto_gs"]:
                    logging.info(f"{self.pad_sym(symbol)} will be set to graceful stop mode")
                    self.approved_symbols[symbol] = "-lm gs -sm gs"
                else:
                    logging.info(f"{self.pad_sym(symbol)} will be set to manual mode")
                    self.approved_symbols[symbol] = "-lm m -sm m"

        if self.config["n_longs"] == 0 and self.config["n_shorts"] == 0:
            # forager is disabled
            # use static symbol list
            # all approved symbols plus symbols with position on graceful stop
            self.active_symbols = sorted(set(self.approved_symbols))
            self.forager_mode = False
        else:
            # forager mode
            # active symbols are symbols with position plus symbols with high noisiness
            self.active_symbols = symbols_with_pos_or_open_orders
            self.forager_mode = True
            # ... TBC
        for symbol in self.active_symbols:
            if symbol not in self.positions:
                self.positions[symbol] = {
                    "long": {"size": 0.0, "price": 0.0},
                    "short": {"size": 0.0, "price": 0.0},
                }
            if symbol not in self.open_orders:
                self.open_orders[symbol] = []

    async def update_approved_symbols(self):
        # symbols are formatted to ccxt standard COIN/QUOTE:QUOTE
        self.ignored_symbols = [
            self.format_symbol(x, suppress_log=True) for x in self.config["ignored_symbols"]
        ]

        approved_symbols = {}  # all symbols approved according to various conditions, with flags
        if self.config["approved_symbols"]:
            for symbol in self.config["approved_symbols"]:
                approved_symbols[self.format_symbol(symbol)] = (
                    self.config["approved_symbols"][symbol]
                    if isinstance(self.config["approved_symbols"], dict)
                    else ""
                )
        else:
            # all symbols are approved
            for symbol in self.all_symbols:
                approved_symbols[symbol] = ""

        if self.config["minimum_market_age_days"] > 0:
            first_timestamps = await get_first_ohlcv_timestamps(cc=self.cca)
            for symbol in sorted(first_timestamps):
                first_timestamps[self.format_symbol(symbol)] = first_timestamps[symbol]
        else:
            first_timestamps = None

        self.approved_symbols = {}
        for symbol in sorted(set(approved_symbols)):
            if symbol not in self.markets_dict:
                logging.info(f"{symbol} missing from {self.exchange}")
            elif not self.markets_dict[symbol]["active"]:
                logging.info(f"{symbol} not active")
            elif not self.markets_dict[symbol]["swap"]:
                logging.info(f"wrong market type for {symbol}: {self.market_dict[symbol]['type']}")
            elif not self.markets_dict[symbol]["linear"]:
                logging.info(f"{symbol} is not a linear market")
            elif not symbol.endswith(f"/{self.quote}:{self.quote}"):
                logging.info(f"{symbol} has wrong formatting")
            elif self.format_symbol(symbol) in self.ignored_symbols:
                logging.info(f"{symbol} is ignored")
            elif first_timestamps:
                if symbol not in first_timestamps:
                    logging.info(f"{self.pad_sym(symbol)} missing from first timestamps")
                elif utc_ms() - first_timestamps[symbol] < self.config["minimum_market_age_days"]:
                    logging.info(
                        f"{symbol} is younger than {self.config['minimum_market_age_days']} days"
                    )
                else:
                    self.approved_symbols[symbol] = approved_symbols[symbol]
            else:
                self.approved_symbols[symbol] = approved_symbols[symbol]

        # add symbols on gs
        self.update_active_symbols()

        # this argparser is used only internally
        parser = argparse.ArgumentParser(prog="passivbot", description="run passivbot")
        parser.add_argument("-sm", type=str, required=False, dest="short_mode", default=None)
        parser.add_argument("-lm", type=str, required=False, dest="long_mode", default=None)
        parser.add_argument("-lw", type=float, required=False, dest="WE_limit_long", default=None)
        parser.add_argument("-sw", type=float, required=False, dest="WE_limit_short", default=None)
        parser.add_argument("-lev", type=float, required=False, dest="leverage", default=None)
        parser.add_argument("-lc", type=str, required=False, dest="live_config_path", default=None)
        self.args = {
            symbol: parser.parse_args(self.approved_symbols[symbol].split())
            for symbol in self.approved_symbols
        }
        # for prettier printing
        self.max_len_symbol = max([len(s) for s in self.approved_symbols])
        self.sym_padding = max(self.sym_padding, self.max_len_symbol + 1)

    def set_wallet_exposure_limits(self):
        # an active bot has normal mode or graceful stop mode with position
        for pside in ["long", "short"]:
            n_actives = 0
            for sym in self.live_configs:
                if self.live_configs[sym][pside]["mode"] == "normal" or (
                    self.live_configs[sym][pside]["mode"] == "graceful_stop"
                    and (sym in self.positions and self.positions[sym][pside]["size"] != 0.0)
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
            for symbol in self.active_symbols:
                if "wallet_exposure_limit" not in self.live_configs[symbol][pside]:
                    self.live_configs[symbol][pside]["wallet_exposure_limit"] = 0.0
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
                    f"  created {self.pad_sym(order['symbol'])} {order['side']} {order['qty']} {order['position_side']} @ {order['price']} source: {source}"
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
                    f"cancelled {self.pad_sym(order['symbol'])} {order['side']} {order['qty']} {order['position_side']} @ {order['price']} source: {source}"
                )
                return True
        except Exception as e:
            logging.error(f"failed to remove order from self.open_orders {order} {e}")
            traceback.print_exc()
            return False

    def handle_order_update(self, upd_list):
        try:
            for upd in upd_list:
                if upd["symbol"] not in self.approved_symbols:
                    return
                if upd["status"] == "closed" or (
                    "filled" in upd and upd["filled"] is not None and upd["filled"] > 0.0
                ):
                    # There was a fill, partial or full. Schedule update of open orders, pnls, position.
                    logging.info(
                        f"   filled {self.pad_sym(upd['symbol'])} {upd['side']} {upd['qty']} {upd['position_side']} @ {upd['price']} source: WS"
                    )
                    self.recent_fill = True
                elif upd["status"] in ["canceled", "expired", "rejected"]:
                    # remove order from open_orders
                    self.remove_cancelled_order(upd)
                elif upd["status"] == "open":
                    # add order to open_orders
                    self.add_new_order(upd)
                else:
                    print("debug open orders unknown type", upd)
        except Exception as e:
            logging.error(f"error updating open orders from websocket {upd_list} {e}")
            traceback.print_exc()

    def handle_balance_update(self, upd, source="WS"):
        try:
            upd[self.quote]["total"] = round_dynamic(upd[self.quote]["total"], 10)
            equity = upd[self.quote]["total"] + self.calc_upnl_sum()
            if self.balance != upd[self.quote]["total"]:
                logging.info(
                    f"balance changed: {self.balance} -> {upd[self.quote]['total']} equity: {equity:.4f} source: {source}"
                )
            self.balance = max(upd[self.quote]["total"], 1e-12)
        except Exception as e:
            logging.error(f"error updating balance from websocket {upd} {e}")
            traceback.print_exc()

    def handle_ticker_update(self, upd):
        if isinstance(upd, list):
            for x in upd:
                self.handle_ticker_update(x)
        elif isinstance(upd, dict):
            if len(upd) == 1:
                # sometimes format is {symbol: {ticker}}
                upd = upd[next(iter(upd))]
            if "bid" not in upd and "bids" in upd and "ask" not in upd and "asks" in upd:
                # order book, not ticker
                upd["bid"], upd["ask"] = upd["bids"][0][0], upd["asks"][0][0]
            if all([key in upd for key in ["bid", "ask", "symbol"]]):
                if "last" not in upd or upd["last"] is None:
                    upd["last"] = np.random.choice([upd["bid"], upd["ask"]])
                for key in ["bid", "ask", "last"]:
                    if upd[key] is not None:
                        self.tickers[upd["symbol"]][key] = upd[key]
                    else:
                        logging.info(f"ticker {upd['symbol']} {key} is None")

                # if upd['bid'] is not None:
                #    if upd['ask'] is not None:
                #        if upd['last'] is not None:
                #            self.tickers[upd['symbol']] = {k: upd[k] for k in ["bid", "ask", "last"]}
                #            return
                #        self.tickers[upd['symbol']] = {'bid': upd['bid'], 'ask': upd['ask'], 'last': np.random.choice([upd['bid'], upd['ask']])}
                #        return
                #    self.tickers[upd['symbol']] = {'bid': upd['bid'], 'ask': upd['bid'], 'last': upd['bid']}
                #    return
            else:
                logging.info(f"unexpected WS ticker formatting: {upd}")

    def calc_upnl_sum(self):
        try:
            upnl_sum = 0.0
            for elm in self.fetched_positions:
                upnl_sum += calc_pnl(
                    elm["position_side"],
                    elm["price"],
                    self.tickers[elm["symbol"]]["last"],
                    elm["size"],
                    self.inverse,
                    self.c_mults[elm["symbol"]],
                )
            return upnl_sum
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
        if not hasattr(self, "open_orders"):
            self.open_orders = {}
        res = await self.fetch_open_orders()
        if res in [None, False]:
            return False
        self.fetched_open_orders = res
        open_orders = res
        oo_ids_old = {elm["id"] for sublist in self.open_orders.values() for elm in sublist}
        created_prints, cancelled_prints = [], []
        for oo in open_orders:
            if oo["id"] not in oo_ids_old:
                # there was a new open order not caught by websocket
                created_prints.append(
                    f"new order {self.pad_sym(oo['symbol'])} {oo['side']} {oo['qty']} {oo['position_side']} @ {oo['price']} source: REST"
                )
        oo_ids_new = {elm["id"] for elm in open_orders}
        for oo in [elm for sublist in self.open_orders.values() for elm in sublist]:
            if oo["id"] not in oo_ids_new:
                # there was an order cancellation not caught by websocket
                cancelled_prints.append(
                    f"cancelled {self.pad_sym(oo['symbol'])} {oo['side']} {oo['qty']} {oo['position_side']} @ {oo['price']} source: REST"
                )
        self.open_orders = {symbol: [] for symbol in self.open_orders}
        self.open_orders = {}
        for elm in open_orders:
            if elm["symbol"] not in self.open_orders:
                self.open_orders[elm["symbol"]] = []
            self.open_orders[elm["symbol"]].append(elm)
        if len(created_prints) > 12:
            logging.info(f"{len(created_prints)} new open orders")
        else:
            for line in created_prints:
                logging.info(line)
        if len(cancelled_prints) > 12:
            logging.info(f"{len(created_prints)} cancelled open orders")
        else:
            for line in cancelled_prints:
                logging.info(line)
        self.upd_timestamps["open_orders"] = utc_ms()
        return True

    async def update_positions(self):
        # also updates balance
        if not hasattr(self, "positions"):
            self.positions = {}
        res = await self.fetch_positions()
        if all(x in [None, False] for x in res):
            return False
        positions_list_new, balance_new = res
        self.fetched_positions = positions_list_new
        self.handle_balance_update({self.quote: {"total": balance_new}})
        positions_new = {
            sym: {
                "long": {"size": 0.0, "price": 0.0},
                "short": {"size": 0.0, "price": 0.0},
            }
            for sym in set(list(self.positions) + self.active_symbols)
        }
        for elm in positions_list_new:
            symbol, pside, pprice = elm["symbol"], elm["position_side"], elm["price"]
            psize = abs(elm["size"]) * (-1.0 if elm["position_side"] == "short" else 1.0)
            if symbol not in positions_new:
                positions_new[symbol] = {
                    "long": {"size": 0.0, "price": 0.0},
                    "short": {"size": 0.0, "price": 0.0},
                }
            positions_new[symbol][pside] = {"size": psize, "price": pprice}
            # check if changed
            if symbol not in self.positions or self.positions[symbol][pside]["size"] != psize:
                wallet_exposure = (
                    qty_to_cost(
                        positions_new[symbol][pside]["size"],
                        positions_new[symbol][pside]["price"],
                        self.inverse,
                        self.c_mults[symbol],
                    )
                    / self.balance
                )
                try:
                    WE_ratio = (
                        wallet_exposure / self.live_configs[symbol][pside]["wallet_exposure_limit"]
                    )
                except:
                    WE_ratio = 0.0
                try:
                    pprice_diff = calc_pprice_diff(pside, pprice, self.tickers[symbol]["last"])
                except:
                    pprice_diff = 0.0
                try:
                    upnl = calc_pnl(
                        pside,
                        pprice,
                        self.tickers[symbol]["last"],
                        psize,
                        self.inverse,
                        self.c_mults[symbol],
                    )
                except:
                    upnl = 0.0
                line = f"{self.pad_sym(symbol)} {pside} changed:"
                if symbol in self.positions:
                    line += f" {self.positions[symbol][pside]}"
                line += f" -> {positions_new[symbol][pside]}"
                line += f" WE: {wallet_exposure:.4f}"
                if WE_ratio:
                    line += f" WE ratio: {WE_ratio:.3f}"
                if pprice_diff:
                    line += f" pprice diff: {pprice_diff:.4f} upnl: {upnl:.4f}"
                logging.info(line)
        self.positions = positions_new
        self.upd_timestamps["positions"] = utc_ms()
        return True

    async def update_tickers(self):
        res = await self.fetch_tickers()
        if res in [None, False]:
            return False
        tickers_new = res
        for symbol in tickers_new:
            ticker_new = {k: tickers_new[symbol][k] for k in ["bid", "ask", "last"]}
            self.tickers[symbol] = ticker_new
        self.upd_timestamps["tickers"] = utc_ms()
        return True

    async def update_emas(self):
        if len(self.emas_long) == 0 or self.ema_minute is None:
            await self.init_emas()
            return True
        now_minute = int(utc_ms() // (1000 * 60) * (1000 * 60))
        if now_minute <= self.ema_minute:
            return True
        while self.ema_minute < int(round(now_minute - 1000 * 60)):
            for symbol in self.approved_symbols:
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
        for symbol in self.approved_symbols:
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
        for sym in self.approved_symbols:
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
        if self.tickers[next(iter(self.tickers))]["last"] == 0.0:
            logging.info(f"updating tickers...")
            await self.update_tickers()
        for sym in self.approved_symbols:
            self.emas_long[sym] = np.repeat(self.tickers[sym]["last"], 3)
            self.emas_short[sym] = np.repeat(self.tickers[sym]["last"], 3)
            self.prev_prices[sym] = self.tickers[sym]["last"]
        ohs = None
        try:
            for symbol in self.approved_symbols:
                samples1m = calc_samples(
                    numpyize(self.ohlcvs[symbol])[:, [0, 5, 4]], sample_size_ms=60000
                )
                self.emas_long[symbol] = calc_emas_last(samples1m[:, 2], self.ema_spans_long[symbol])
                self.emas_short[symbol] = calc_emas_last(
                    samples1m[:, 2], self.ema_spans_short[symbol]
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
        for symbol in self.active_symbols:
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
                upnl = calc_pnl(
                    pside,
                    self.positions[sym][pside]["price"],
                    self.tickers[sym]["last"],
                    self.positions[sym][pside]["size"],
                    self.inverse,
                    self.c_mults[sym],
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
                    if utc_ms() - self.prev_AU_print_ms > 1000 * 300:
                        line = f"Auto unstuck allowance: {AU_allowance:.3f} {self.quote}. Will place {pside} unstucking order for {sym} at {close_price}. Last price: {self.tickers[sym]['last']}"
                        logging.info(line)
                        self.prev_AU_print_ms = utc_ms()
                except:
                    self.prev_AU_print_ms = 0.0

        ideal_orders = {symbol: [] for symbol in self.active_symbols}
        for symbol in self.active_symbols:
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
                    and abs(
                        calc_pprice_diff(
                            unstuck_close_order["position_side"],
                            unstuck_close_order["order"][1],
                            self.tickers[unstuck_close_order["symbol"]]["last"],
                        )
                    )
                    < 0.002
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
                    and abs(
                        calc_pprice_diff(
                            unstuck_close_order["position_side"],
                            unstuck_close_order["order"][1],
                            self.tickers[unstuck_close_order["symbol"]]["last"],
                        )
                    )
                    < 0.002
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

    async def force_update(self, force=False):
        # if some information has not been updated in a while, force update via REST
        coros_to_call = []
        now = utc_ms()
        for key in self.upd_timestamps:
            if force or now - self.upd_timestamps[key] > self.force_update_age_millis:
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
            self.update_active_symbols()
            if self.recent_fill:
                self.upd_timestamps["positions"] = 0.0
                self.upd_timestamps["open_orders"] = 0.0
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
            res = await self.execute_cancellations(
                to_cancel[: self.config["max_n_cancellations_per_batch"]]
            )
            if res:
                for elm in res:
                    self.remove_cancelled_order(elm, source="POST")
            res = await self.execute_orders(to_create[: self.config["max_n_creations_per_batch"]])
            if res:
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
            # self.debug_dump_bot_state_to_disk()

    def debug_dump_bot_state_to_disk(self):
        if not hasattr(self, "tmp_debug_ts"):
            self.tmp_debug_ts = 0
            self.tmp_debug_cache = make_get_filepath(f"caches/{self.exchange}/{self.user}_debug/")
        if utc_ms() - self.tmp_debug_ts > 1000 * 60 * 3:
            logging.info(f"debug dumping bot state to disk")
            for k, v in vars(self).items():
                try:
                    json.dump(
                        denumpyize(v), open(os.path.join(self.tmp_debug_cache, k + ".json"), "w")
                    )
                except Exception as e:
                    logging.error(f"debug failed to dump to disk {k} {e}")
            self.tmp_debug_ts = utc_ms()

    async def start_bot(self):
        await self.init_bot()
        logging.info("done initiating bot")
        logging.info("starting websockets")
        await asyncio.gather(self.execution_loop(), self.start_websockets())

    def get_ohlcv_fpath(self, symbol) -> str:
        return os.path.join(
            self.ohlcvs_cache_dirpath, symbol.replace(f"/{self.quote}:{self.quote}", "") + ".json"
        )

    def load_ohlcv_from_cache(self, symbol, suppress_error_log=False):
        fpath = self.get_ohlcv_fpath(symbol)
        try:
            ohlcvs = json.load(open(fpath))
            return ohlcvs
        except Exception as e:
            if not suppress_error_log:
                logging.error(f"failed to load ohlcvs from cache for {symbol}")
                traceback.print_exc()

    def dump_ohlcv_to_cache(self, symbol, ohlcv):
        fpath = self.get_ohlcv_fpath(symbol)
        try:
            json.dump(ohlcv, open(fpath, "w"))
            self.ohlcv_upd_timestamps[symbol] = get_file_mod_utc(self.get_ohlcv_fpath(symbol))
        except Exception as e:
            logging.error(f"failed to dump ohlcvs to cache for {symbol}")
            traceback.print_exc()

    def get_oldest_updated_ohlcv_symbol(self):
        return sorted(self.approved_symbols, key=lambda x: self.ohlcv_upd_timestamps[x])[0]

    def calc_noisiness(self, symbol=None):
        if not hasattr(self, "noisiness"):
            self.noisiness = {}
        symbols = self.approved_symbols if symbol is None else [symbol]
        for symbol in symbols:
            if symbol in self.ohlcvs and len(self.ohlcvs[symbol]) > 0:
                self.noisiness[symbol] = np.mean([(x[2] - x[3]) / x[4] for x in self.ohlcvs[symbol]])
            else:
                self.noisiness[symbol] = 0.0

    async def maintain_ohlcvs(self, timeframe="15m", sleep_interval=15):
        self.ohlcvs = {}
        self.ohlcv_upd_timestamps = {symbol: 0 for symbol in self.approved_symbols}
        time_diff = self.find_file_mod_utc_time_diff()
        if time_diff > 1000:
            logging.info(
                f"time diff between utc_ms() and get_file_mod_utc() is greater than one second"
            )
        force_update_syms = []
        for symbol in self.approved_symbols:
            ohlcvs = self.load_ohlcv_from_cache(symbol, suppress_error_log=True)
            if ohlcvs:
                self.ohlcvs[symbol] = ohlcvs
                self.ohlcv_upd_timestamps[symbol] = get_file_mod_utc(self.get_ohlcv_fpath(symbol))
                if utc_ms() - self.ohlcv_upd_timestamps[symbol] > 1000 * 60 * 15:
                    force_update_syms.append(symbol)
                    self.ohlcv_upd_timestamps[symbol] = 0
                    del self.ohlcvs[symbol]
        if force_update_syms:
            logging.info(
                f"ohlcvs too old; forcing update for {','.join([symbol2coin(x) for x in force_update_syms])}"
            )
        while True:
            missing_symbols = [s for s in self.approved_symbols if s not in self.ohlcvs]
            if missing_symbols:
                sleep_interval_ = 0.1
                symbol = missing_symbols[0]
            else:
                sleep_interval_ = sleep_interval
                for _ in range(100):
                    symbol = self.get_oldest_updated_ohlcv_symbol()
                    # check if has been modified by other PB instance
                    self.ohlcv_upd_timestamps[symbol] = get_file_mod_utc(self.get_ohlcv_fpath(symbol))
                    if symbol == self.get_oldest_updated_ohlcv_symbol():
                        break
                else:
                    logging.error(
                        f"more than 100 retries for getting most recently modified ohlcv symbol"
                    )
            self.ohlcvs[symbol] = await self.fetch_ohlcv(symbol, timeframe=timeframe)
            self.dump_ohlcv_to_cache(symbol, self.ohlcvs[symbol])
            # logging.info(f"updated ohlcvs for {symbol}")
            await asyncio.sleep(sleep_interval_)

    async def close(self):
        try:
            self.ohlcv_maintainer.cancel()
        except Exception as e:
            logging.error(f"error stopping ohlcvs maintainer {e}")
        await self.cca.close()
        await self.ccp.close()


async def main():
    parser = argparse.ArgumentParser(prog="passivbot", description="run passivbot")
    parser.add_argument("hjson_config_path", type=str, help="path to hjson passivbot meta config")
    parser_items = [
        (
            "s",
            "approved_symbols",
            "approved_symbols",
            str,
            ", comma separated (SYM1USDT,SYM2USDT,...)",
        ),
        ("i", "ignored_symbols", "ignored_symbols", str, ", comma separated (SYM1USDT,SYM2USDT,...)"),
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
        ("nca", "max_n_cancellations_per_batch", "max_n_cancellations_per_batch", int, ""),
        ("ncr", "max_n_creations_per_batch", "max_n_creations_per_batch", int, ""),
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
        config = load_hjson_config(args.hjson_config_path)
        config, logging_lines = add_missing_params_to_hjson_live_multi_config(config)
        for line in logging_lines:
            logging.info(line)

        for key in [x[2] for x in parser_items]:
            if getattr(args, key) is not None:
                if key.endswith("symbols"):
                    old_value = sorted(set(config[key]))
                    new_value = sorted(set(getattr(args, key).split(",")))
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
        elif user_info["exchange"] == "hyperliquid":
            from exchanges_multi.hyperliquid import HyperliquidBot

            bot = HyperliquidBot(config)
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
