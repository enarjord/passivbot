import os

if "NOJIT" not in os.environ:
    os.environ["NOJIT"] = "true"


import logging
import traceback
import argparse
import asyncio
import json
import hjson
import pprint
import numpy as np

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
from pure_funcs import numpyize, filter_orders


class Passivbot:
    def __init__(self, config: dict):
        self.config = config
        self.user = config["user"]
        self.user_info = load_user_info(config["user"])
        self.exchange = self.user_info["exchange"]
        self.broker_code = load_broker_code(self.user_info["exchange"])

        self.stop_websocket = False
        self.balance = 1e-12
        self.upd_timestamps = {
            "balance": 0.0,
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
        self.any_stuck = False
        self.previous_execution_ts = 0
        self.recent_fill = False
        self.execution_delay_millis = max(3000.0, self.config["execution_delay_seconds"] * 1000)
        self.force_update_age_millis = 60 * 1000  # force update once a minute
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            level=logging.INFO,
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    async def init_bot(self):
        # load live configs
        live_configs_fnames = sorted(
            [f for f in os.listdir(self.config["live_configs_dir"]) if f.endswith(".json")]
        )
        for symbol in self.symbols:
            # check self.config['live_configs_map'] for matches
            if symbol in self.config["live_configs_map"]:
                if os.path.exists(self.config["live_configs_map"][symbol]):
                    try:
                        self.live_configs[symbol] = load_live_config(
                            self.config["live_configs_map"][symbol]
                        )
                        logging.info(
                            f"loaded live config for {symbol}: {self.config['live_configs_map'][symbol]}"
                        )
                    except Exception as e:
                        logging.error(
                            f"failed to load {symbol} live config from {self.config['live_configs_map'][symbol]} {e}"
                        )
            if symbol not in self.live_configs:
                # find matches in live_configs_dir
                for fname in live_configs_fnames:
                    if self.coins[symbol] in fname:
                        ffpath = os.path.join(self.config["live_configs_dir"], fname)
                        try:
                            self.live_configs[symbol] = load_live_config(ffpath)
                            logging.info(f"loaded live config for {symbol}: {ffpath}")
                        except Exception as e:
                            logging.error(f"failed to load {symbol} live config from {ffpath} {e}")
            if symbol not in self.live_configs:
                try:
                    self.live_configs[symbol] = load_live_config(self.config["default_config_path"])
                    logging.info(
                        f"loaded live config for {symbol}: {self.config['default_config_path']}"
                    )
                except Exception as e:
                    logging.error(
                        f"failed to load {symbol} live config from {self.config['default_config_path']} {e}"
                    )
                    raise Exception(f"no usable live config found for {symbol}")
            # disable AU
            if self.config["multisym_auto_unstuck_enabled"]:
                for side in ["long", "short"]:
                    for key in [
                        "auto_unstuck_delay_minutes",
                        "auto_unstuck_ema_dist",
                        "auto_unstuck_qty_pct",
                        "auto_unstuck_wallet_exposure_threshold",
                    ]:
                        self.live_configs[symbol][side][key] = 0.0
                    if symbol in getattr(self, f"approved_symbols_{side}"):
                        if getattr(self, f"approved_symbols_{side}")[symbol] is None:
                            self.live_configs[symbol][side]["enabled"] = True
                            self.live_configs[symbol][side]["wallet_exposure_limit"] = (
                                self.config[f"TWE_{side}"]
                                / len(getattr(self, f"approved_symbols_{side}"))
                                if len(getattr(self, f"approved_symbols_{side}")) > 0
                                else 0.0
                            )
                        elif getattr(self, f"approved_symbols_{side}")[symbol] > 0.0:
                            self.live_configs[symbol][side]["enabled"] = True
                            self.live_configs[symbol][side]["wallet_exposure_limit"] = getattr(
                                self, f"approved_symbols_{side}"
                            )[symbol]
                        else:
                            self.live_configs[symbol][side]["enabled"] = False
                    else:
                        self.live_configs[symbol][side]["enabled"] = False

        for f in ["positions", "emas", "open_orders", "pnls", "balance"]:
            res = await getattr(self, f"update_{f}")()
            logging.info(f"initiating {f} {res}")

    async def handle_order_update(self, upd_list):
        try:
            for upd in upd_list:
                if upd["symbol"] not in self.symbols:
                    return
                if upd["filled"] > 0.0:
                    # There was a fill, partial or full. Schedule update of open orders, pnls, position.
                    logging.info(
                        f"   filled {upd['symbol']} {upd['side']} {upd['qty']} {upd['position_side']} @ {upd['price']}"
                    )
                    self.recent_fill = True
                elif upd["status"] == "canceled":
                    # remove order from open_orders
                    if upd["id"] in {x["id"] for x in self.open_orders[upd["symbol"]]}:
                        logging.info(
                            f"cancelled {upd['symbol']} {upd['side']} {upd['qty']} {upd['position_side']} @ {upd['price']}"
                        )
                        self.open_orders[upd["symbol"]] = [
                            elm for elm in self.open_orders[upd["symbol"]] if elm["id"] != upd["id"]
                        ]
                    self.upd_timestamps["open_orders"][upd["symbol"]] = utc_ms()
                elif upd["status"] == "open":
                    # add order to open_orders
                    if upd["id"] not in {x["id"] for x in self.open_orders[upd["symbol"]]}:
                        logging.info(
                            f"  created {upd['symbol']} {upd['side']} {upd['qty']} {upd['position_side']} @ {upd['price']}"
                        )
                        self.open_orders[upd["symbol"]].append(upd)
                    self.upd_timestamps["open_orders"][upd["symbol"]] = utc_ms()
                else:
                    print("debug open orders unknown type", upd)
        except Exception as e:
            logging.error(f"error updating open orders from websocket {upd_list} {e}")
            traceback.print_exc()

    async def handle_balance_update(self, upd):
        try:
            if self.balance != upd["USDT"]["total"]:
                logging.info(f"balance changed: {self.balance} -> {upd['USDT']['total']}")
            self.balance = max(upd["USDT"]["total"], 1e-12)
            self.upd_timestamps["balance"] = utc_ms()
        except Exception as e:
            logging.error(f"error updating balance from websocket {upd} {e}")
            traceback.print_exc()

    async def handle_ticker_update(self, upd):
        self.upd_timestamps["tickers"][upd["symbol"]] = utc_ms()  # update timestamp
        if (
            upd["bid"] != self.tickers[upd["symbol"]]["bid"]
            or upd["ask"] != self.tickers[upd["symbol"]]["ask"]
        ):
            ticker_new = {k: upd[k] for k in ["bid", "ask", "last"]}
            self.tickers[upd["symbol"]] = ticker_new

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
                if pnls_cache[0]["updatedTime"] > age_limit + 1000 * 60 * 60 * 4:
                    # fetch missing pnls
                    missing_pnls = await self.fetch_pnls(
                        start_time=age_limit - 1000, end_time=pnls_cache[0]["updatedTime"]
                    )
                    pnls_cache = sorted(
                        {
                            elm["orderId"] + str(elm["qty"]): elm
                            for elm in pnls_cache + missing_pnls
                            if elm["updatedTime"] >= age_limit
                        }.values(),
                        key=lambda x: x["updatedTime"],
                    )
            self.pnls = pnls_cache
        start_time = self.pnls[-1]["updatedTime"] if self.pnls else age_limit
        new_pnls = await self.fetch_pnls(start_time=start_time)
        len_pnls = len(self.pnls)
        self.pnls = sorted(
            {
                elm["orderId"] + str(elm["qty"]): elm
                for elm in self.pnls + new_pnls
                if elm["updatedTime"] > age_limit
            }.values(),
            key=lambda x: x["updatedTime"],
        )
        if len(self.pnls) > len_pnls or len(missing_pnls) > 0:
            n_new_pnls = len(self.pnls) - len_pnls
            logging.debug(f"{n_new_pnls} new pnl{'s' if n_new_pnls > 1 else ''}")
            print(f"{len(self.pnls) - len_pnls} new pnls")
            try:
                json.dump(self.pnls, open(self.pnls_cache_filepath, "w"))
            except Exception as e:
                logging.error(f"error dumping pnls to {self.pnls_cache_filepath} {e}")
        self.upd_timestamps["pnls"] = utc_ms()
        return True

    async def update_open_orders(self):
        open_orders = await self.fetch_open_orders()
        oo_ids_old = {elm["id"] for sublist in self.open_orders.values() for elm in sublist}
        for oo in open_orders:
            if oo["id"] not in oo_ids_old:
                # there was a new open order not caught by websocket
                logging.info(f"new open order {oo['symbol']} {oo['position_side']} {oo['id']}")
        oo_ids_new = {elm["id"] for elm in open_orders}
        for oo in [elm for sublist in self.open_orders.values() for elm in sublist]:
            if oo["id"] not in oo_ids_new:
                # there was an order cancellation not caught by websocket
                logging.info(f"cancelled open order {oo['symbol']} {oo['position_side']} {oo['id']}")
        self.open_orders = {symbol: [] for symbol in self.open_orders}
        for elm in open_orders:
            if elm["symbol"] in self.open_orders:
                self.open_orders[elm["symbol"]].append(elm)
            else:
                logging.debug(
                    f"{elm['symbol']} has open order {elm['position_side']} {elm['id']}, but is not under passivbot management"
                )
                print(
                    f"debug {elm['symbol']} has open order {elm['position_side']} {elm['id']}, but is not under passivbot management"
                )
        now = utc_ms()
        self.upd_timestamps["open_orders"] = {k: now for k in self.upd_timestamps["open_orders"]}
        return True

    async def update_positions(self):
        positions_list_new = await self.fetch_positions()
        positions_new = {
            symbol: {"long": {"size": 0.0, "price": 0.0}, "short": {"size": 0.0, "price": 0.0}}
            for symbol in self.positions
        }
        for elm in positions_list_new:
            if elm["symbol"] not in self.positions:
                print(
                    f"debug {elm['symbol']} has a {elm['position_side']} position, but is not under passivbot management"
                )
                logging.debug(
                    f"debug {elm['symbol']} has a {elm['position_side']} position, but is not under passivbot management"
                )
            else:
                positions_new[elm["symbol"]][elm["position_side"]] = {
                    "size": abs(elm["contracts"])
                    * (-1.0 if elm["position_side"] == "short" else 1.0),
                    "price": elm["entryPrice"],
                }

        for symbol in self.positions:
            for side in self.positions[symbol]:
                if self.positions[symbol][side] != positions_new[symbol][side]:
                    logging.info(
                        f"{symbol} {side} changed: {self.positions[symbol][side]} -> {positions_new[symbol][side]}"
                    )
        self.positions = positions_new
        now = utc_ms()
        self.upd_timestamps["positions"] = {k: now for k in self.upd_timestamps["positions"]}
        return True

    async def update_balance(self):
        balance_new = await self.fetch_balance()
        if self.balance != balance_new:
            logging.info(f"balance changed: {self.balance} -> {balance_new}")
        self.balance = max(balance_new, 1e-12)
        self.upd_timestamps["balance"] = utc_ms()
        return True

    async def update_tickers(self):
        tickers_new = await self.fetch_tickers()
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
                print("debug", symbol, emas_long[symbol])
                print(
                    self.alphas_long[symbol],
                    self.alphas__long[symbol],
                    self.emas_long[symbol],
                    self.prev_prices[symbol],
                )
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
        if self.tickers[self.symbols[0]]["last"] == 0.0:
            logging.info(f"updating tickers...")
            await self.update_tickers()
        for sym in self.symbols:
            self.emas_long[sym] = np.repeat(self.tickers[sym]["last"], 3)
            self.emas_short[sym] = np.repeat(self.tickers[sym]["last"], 3)
            self.prev_prices[sym] = self.tickers[sym]["last"]
        ohs = None
        try:
            logging.info(f"fetching 15 min ohlcv for all symbols, initiating EMAs.")
            ohs = await asyncio.gather(
                *[self.fetch_ohlcv(symbol, timeframe="15m") for symbol in self.symbols]
            )
            samples_1m = [
                calc_samples(numpyize(oh)[:, [0, 5, 4]], sample_size_ms=60000) for oh in ohs
            ]
            for i in range(len(self.symbols)):
                self.emas_long[self.symbols[i]] = calc_emas_last(
                    samples_1m[i][:, 2], self.ema_spans_long[self.symbols[i]]
                )
                self.emas_short[self.symbols[i]] = calc_emas_last(
                    samples_1m[i][:, 2], self.ema_spans_short[self.symbols[i]]
                )
            return True
        except Exception as e:
            logging.error(
                f"error fetching ohlcvs to initiate EMAs {e}. Using latest prices as starting EMAs"
            )
            traceback.print_exc()

    def calc_ideal_orders(self):
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
            if do_long:
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
                closes_long = calc_close_grid_long(
                    self.live_configs[symbol]["long"]["backwards_tp"],
                    self.balance,
                    self.positions[symbol]["long"]["size"],
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
            if do_short:
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
                closes_short = calc_close_grid_short(
                    self.live_configs[symbol]["short"]["backwards_tp"],
                    self.balance,
                    self.positions[symbol]["short"]["size"],
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
        stuck_positions = []
        for symbol in self.symbols:
            if not self.config["multisym_auto_unstuck_enabled"]:
                break
            for side in ["long", "short"]:
                if (
                    not self.live_configs[symbol][side]["enabled"]
                    or self.live_configs[symbol][side]["wallet_exposure_limit"] == 0.0
                ):
                    continue
                wallet_exposure = (
                    qty_to_cost(
                        self.positions[symbol][side]["size"],
                        self.positions[symbol][side]["price"],
                        self.inverse,
                        self.c_mults[symbol],
                    )
                    / self.balance
                )
                if (
                    wallet_exposure / self.live_configs[symbol][side]["wallet_exposure_limit"]
                    > self.config["stuck_threshold"]
                ):
                    pprice_diff = (
                        1.0 - self.tickers[symbol]["last"] / self.positions[symbol]["long"]["price"]
                        if side == "long"
                        else self.tickers[symbol]["last"] / self.positions[symbol]["short"]["price"]
                        - 1.0
                    )
                    print("pprice_diff", pprice_diff)
                    if pprice_diff > 0.0:
                        # don't unstuck if position is in profit
                        stuck_positions.append((symbol, side, pprice_diff))
        if stuck_positions:
            self.any_stuck = True
            sym, side, pprice_diff = sorted(stuck_positions, key=lambda x: x[2])[0]
            AU_allowance = calc_AU_allowance(
                np.array([x["closedPnl"] for x in self.pnls]),
                self.balance,
                loss_allowance_pct=self.config["loss_allowance_pct"],
            )
            if AU_allowance > 0.0:
                close_price = (
                    round_up(self.emas_short[sym].max(), self.price_steps[sym])
                    if side == "long"
                    else round_dn(self.emas_short[sym].min(), self.price_steps[sym])
                )
                upnl = (
                    calc_pnl_long(
                        self.positions[sym][side]["price"],
                        self.tickers[sym]["last"],
                        self.positions[sym][side]["size"],
                        self.inverse,
                        self.c_mults[sym],
                    )
                    if side == "long"
                    else calc_pnl_short(
                        self.positions[sym][side]["price"],
                        self.tickers[sym]["last"],
                        self.positions[sym][side]["size"],
                        self.inverse,
                        self.c_mults[sym],
                    )
                )
                AU_allowance_pct = 1.0 if upnl >= 0.0 else min(1.0, AU_allowance / abs(upnl))
                AU_allowance_qty = round_(
                    self.positions[sym][side]["size"] * AU_allowance_pct, self.qty_steps[sym]
                )
                print(
                    "upnl, AU_allowance_pct, AU_allowance_qty",
                    upnl,
                    AU_allowance_pct,
                    AU_allowance_qty,
                )
                close_qty = max(
                    calc_min_entry_qty(
                        close_price,
                        self.inverse,
                        self.qty_steps[sym],
                        self.min_qtys[sym],
                        self.min_costs[sym],
                    ),
                    min(
                        abs(AU_allowance_qty),
                        round_(
                            cost_to_qty(
                                self.balance
                                * self.live_configs[sym][side]["wallet_exposure_limit"]
                                * self.config["unstuck_close_pct"],
                                close_price,
                                self.inverse,
                                self.c_mults[sym],
                            ),
                            self.qty_steps[sym],
                        ),
                    ),
                )
                unstuck_close_order = (
                    close_qty * (-1.0 if side == "long" else 1.0),
                    close_price,
                    f"unstuck_close_{side}",
                )
                if unstuck_close_order[0] != 0.0:
                    print(ideal_orders[sym])
                    ideal_orders[sym] = [
                        x for x in ideal_orders[sym] if not (side in x[2] and "close" in x[2])
                    ] + [unstuck_close_order]
                    logging.debug(f"creating unstucking order for {sym}: {unstuck_close_order}")
                    print(f"creating unstucking order for {sym}: {unstuck_close_order}")
        else:
            self.any_stuck = False
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
                    "side": "buy" if x[0] > 0.0 else "sell",
                    "position_side": "long" if "long" in x[2] else "short",
                    "qty": x[0],
                    "price": x[1],
                    "reduce_only": "close" in x[2],
                    "custom_id": x[2],
                }
                for x in ideal_orders[symbol]
            ]
            for symbol in ideal_orders
        }

    def calc_orders_to_create_and_cancel(self):
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
                        "qty": abs(x["amount"]) * (-1.0 if x["side"] == "sell" else 1.0),
                        "price": x["price"],
                        "id": x["id"],
                    }
                )
        keys = ("symbol", "side", "position_side", "qty", "price")
        to_cancel, to_create = [], []
        for symbol in actual_orders:
            to_cancel_, to_create_ = filter_orders(actual_orders[symbol], ideal_orders[symbol], keys)
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
        res = await asyncio.gather(*[x[1] for x in coros_to_call])
        await self.update_emas()
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
                self.upd_timestamps["balance"] = 0.0
                self.upd_timestamps["pnls"] = 0.0
                self.recent_fill = False
            await self.force_update()
            to_cancel, to_create = self.calc_orders_to_create_and_cancel()
            if to_create:
                pass
            res = await self.execute_cancellations(to_cancel)
            for elm in res:
                try:
                    logging.info(
                        f"cancelled {elm['symbol']} {elm['side']} {elm['qty']} {elm['position_side']} @ {elm['price']}"
                    )
                    self.open_orders[elm["symbol"]] = [
                        x for x in self.open_orders[elm["symbol"]] if x["id"] != elm["id"]
                    ]
                except Exception as e:
                    logging.error(f"error cancelling order {elm}")
            res = await self.execute_orders(to_create)
            for elm in res:
                try:
                    logging.info(
                        f"  created {elm['symbol']} {elm['side']} {elm['qty']} {elm['position_side']} @ {elm['price']}"
                    )
                    elm["amount"] = elm["qty"]
                    self.open_orders[elm["symbol"]].append(elm)
                except Exception as e:
                    logging.error(f"error creating order {elm}")
            await asyncio.gather(self.update_open_orders(), self.update_positions())
        except Exception as e:
            logging.error(f"error executing to exchange {e}")
            traceback.print_exc()
        finally:
            self.previous_execution_ts = utc_ms()

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
    args = parser.parse_args()
    config = hjson.load(open(args.hjson_config_path))
    user_info = load_user_info(config["user"])
    pprint.pprint(config)
    if user_info["exchange"] == "bybit":
        from exchanges_multi.bybit import BybitBot

        bot = BybitBot(config)
    await bot.start_bot()


if __name__ == "__main__":
    asyncio.run(main())
