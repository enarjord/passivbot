import os

if "NOJIT" not in os.environ:
    os.environ["NOJIT"] = "true"

import traceback
import argparse
import asyncio
import json
import sys
import signal
import hjson
import pprint
import numpy as np
import inspect
import passivbot_rust as pbr
import logging
from prettytable import PrettyTable
from uuid import uuid4
from copy import deepcopy
from collections import defaultdict
from sortedcontainers import SortedDict

from procedures import (
    load_broker_code,
    load_user_info,
    utc_ms,
    make_get_filepath,
    get_file_mod_utc,
    get_first_ohlcv_timestamps,
    get_first_ohlcv_timestamps_new,
    load_config,
    add_arguments_recursively,
    update_config_with_args,
    format_config,
    print_async_exception,
    coin_to_symbol,
    read_external_coins_lists,
)
from njit_funcs_recursive_grid import calc_recursive_entries_long, calc_recursive_entries_short
from njit_funcs import (
    calc_samples,
    calc_emas_last,
    calc_ema,
    calc_close_grid_long,
    calc_close_grid_short,
    calc_diff,
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
from pure_funcs import (
    numpyize,
    denumpyize,
    filter_orders,
    multi_replace,
    shorten_custom_id,
    determine_side_from_order_tuple,
    str2bool,
    symbol_to_coin,
    add_missing_params_to_hjson_live_multi_config,
    expand_PB_mode,
    ts_to_date_utc,
    get_template_live_config,
    flatten,
    log_dict_changes,
)


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S",
)


def signal_handler(sig, frame):
    print("\nReceived shutdown signal. Stopping bot...")
    if "bot" in globals():
        bot.stop_signal_received = True
    else:
        sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)


def get_function_name():
    return inspect.currentframe().f_back.f_code.co_name


def or_default(f, *args, default=None, **kwargs):
    try:
        return f(*args, **kwargs)
    except:
        return default


class Passivbot:
    def __init__(self, config: dict):
        self.config = config
        self.user = config["live"]["user"]
        self.user_info = load_user_info(self.user)
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
        }
        self.hedge_mode = True
        self.inverse = False
        self.active_symbols = []
        self.fetched_positions = []
        self.fetched_open_orders = []
        self.open_orders = {}
        self.positions = {}
        self.pnls = []
        self.symbol_ids = {}
        self.min_costs = {}
        self.min_qtys = {}
        self.qty_steps = {}
        self.price_steps = {}
        self.c_mults = {}
        self.max_leverage = {}
        self.live_configs = {}
        self.PB_modes = {"long": {}, "short": {}}
        self.pnls_cache_filepath = make_get_filepath(f"caches/{self.exchange}/{self.user}_pnls.json")
        self.ohlcvs_1m_cache_dirpath = make_get_filepath(f"caches/{self.exchange}/ohlcvs_1m/")
        self.previous_REST_update_ts = 0
        self.recent_fill = False
        self.execution_delay_millis = max(
            3000.0, self.config["live"]["execution_delay_seconds"] * 1000
        )
        self.quote = "USDT"

        self.minimum_market_age_millis = (
            self.config["live"]["minimum_coin_age_days"] * 24 * 60 * 60 * 1000
        )
        self.emas = {"long": {}, "short": {}}
        self.ema_alphas = {"long": {}, "short": {}}
        self.upd_minute_emas = {}
        self.ineligible_symbols_with_pos = set()
        self.ohlcvs_1m_update_after_minutes = config["live"]["ohlcvs_1m_update_after_minutes"]
        self.ohlcvs_1m_rolling_window_days = config["live"]["ohlcvs_1m_rolling_window_days"]
        self.n_symbols_missing_ohlcvs_1m = 1000
        self.ohlcvs_1m_update_timestamps = {}
        self.max_n_concurrent_ohlcvs_1m_updates = 3
        self.stop_signal_received = False
        self.ohlcvs_1m_update_timestamps_WS = {}
        self.PB_mode_stop = {
            "long": "graceful_stop" if self.config["live"]["auto_gs"] else "manual",
            "short": "graceful_stop" if self.config["live"]["auto_gs"] else "manual",
        }
        self.create_ccxt_sessions()
        self.debug_mode = False

    async def start_bot(self):
        logging.info(f"Starting bot...")
        await self.init_markets()
        await asyncio.sleep(1)
        logging.info(f"Starting data maintainers...")
        await self.start_data_maintainers()
        await self.wait_for_ohlcvs_1m_to_update()
        logging.info(f"starting websocket...")
        self.previous_REST_update_ts = utc_ms()
        await self.prepare_for_execution()

        logging.info(f"starting execution loop...")
        if not self.debug_mode:
            await self.run_execution_loop()

    async def init_markets(self, verbose=True):
        # called at bot startup and once an hour thereafter
        self.init_markets_last_update_ms = utc_ms()
        await self.update_exchange_config()  # set hedge mode
        self.markets_dict = await self.cca.load_markets(True)
        await self.determine_utc_offset(verbose)
        # ineligible symbols cannot open new positions
        self.ineligible_symbols = {}
        self.eligible_symbols = set()
        for symbol in list(self.markets_dict):
            if not self.markets_dict[symbol]["active"]:
                self.ineligible_symbols[symbol] = "not active"
            elif not self.markets_dict[symbol]["swap"]:
                self.ineligible_symbols[symbol] = "wrong market type"
            elif not self.markets_dict[symbol]["linear"]:
                self.ineligible_symbols[symbol] = "not linear"
            elif not symbol.endswith(f"/{self.quote}:{self.quote}"):
                self.ineligible_symbols[symbol] = "wrong quote"
            elif not self.symbol_is_eligible(symbol):
                self.ineligible_symbols[symbol] = f"not eligible on {self.exchange}"
            elif not self.symbol_is_eligible(symbol):
                self.ineligible_symbols[symbol] = f"not eligible on {self.exchange}"
            else:
                self.eligible_symbols.add(symbol)
        if verbose:
            for line in set(self.ineligible_symbols.values()):
                syms_ = [s for s in self.ineligible_symbols if self.ineligible_symbols[s] == line]
                if len(syms_) > 12:
                    logging.info(f"{line}: {len(syms_)} symbols")
                elif len(syms_) > 0:
                    logging.info(f"{line}: {','.join(sorted(set([s for s in syms_])))}")
        self.set_market_specific_settings()
        # for prettier printing
        self.max_len_symbol = max([len(s) for s in self.markets_dict])
        self.sym_padding = max(self.sym_padding, self.max_len_symbol + 1)
        await self.init_flags()
        await self.update_tickers()
        self.refresh_approved_ignored_coins_lists()
        self.set_live_configs()
        await self.update_positions()
        await self.update_open_orders()
        self.update_effective_min_cost()
        self.n_symbols_missing_ohlcvs_1m = len(self.get_symbols_approved_or_has_pos())
        if self.is_forager_mode():
            await self.update_first_timestamps()

    async def init_flags(self):
        self.flags = {}
        for k, v in self.config["live"]["coin_flags"].items():
            if kr := self.coin_to_symbol(k):
                logging.info(f"setting flag for {kr}: {v}")
                self.flags[kr] = v

        # this argparser is used only internally
        parser = argparse.ArgumentParser(prog="passivbot", description="run passivbot")
        parser.add_argument(
            "-sm", type=expand_PB_mode, required=False, dest="short_mode", default=None
        )
        parser.add_argument(
            "-lm", type=expand_PB_mode, required=False, dest="long_mode", default=None
        )
        parser.add_argument("-lw", type=float, required=False, dest="WE_limit_long", default=None)
        parser.add_argument("-sw", type=float, required=False, dest="WE_limit_short", default=None)
        parser.add_argument("-lev", type=float, required=False, dest="leverage", default=None)
        parser.add_argument("-lc", type=str, required=False, dest="live_config_path", default=None)
        self.forced_modes = {"long": "", "short": ""}
        for pside in self.forced_modes:
            if fmode := self.config["live"][f"forced_mode_{pside}"]:
                try:
                    self.forced_modes[pside] = expand_PB_mode(fmode)
                    if self.forced_modes[pside] == "normal":
                        self.forced_modes[pside] = ""
                    else:
                        logging.info(f"Set forced mode {pside} {self.forced_modes[pside]}")
                        self.PB_mode_stop[pside] = self.forced_modes[pside]
                except Exception as e:
                    logging.info(f"Error setting {pside} forced mode {fmode} {e}")
        self.flagged_modes = {"long": {}, "short": {}}
        for symbol in self.flags:
            self.flags[symbol] = parser.parse_args(self.flags[symbol].split())
            for pside in ["long", "short"]:
                if (mode := getattr(self.flags[symbol], f"{pside}_mode")) is not None:
                    self.flagged_modes[pside][symbol] = mode
                elif not self.markets_dict[symbol]["active"]:
                    self.flagged_modes[pside][symbol] = "tp_only"

    async def update_first_timestamps(self, symbols=[]):
        if not hasattr(self, "first_timestamps"):
            self.first_timestamps = {}
        symbols = sorted(set(symbols + flatten(self.approved_coins_minus_ignored_coins.values())))
        if all([s in self.first_timestamps for s in symbols]):
            return
        first_timestamps = await get_first_ohlcv_timestamps_new(
            symbols=symbols, exchange=self.exchange
        )
        self.first_timestamps.update(first_timestamps)
        for symbol in sorted(self.first_timestamps):
            symbolf = self.coin_to_symbol(symbol)
            if symbolf not in self.first_timestamps:
                self.first_timestamps[symbolf] = self.first_timestamps[symbol]
        for symbol in symbols:
            if symbol not in self.first_timestamps:
                logging.info(f"warning: unable to get first timestamp for {symbol}. Setting to zero.")
                self.first_timestamps[symbol] = 0.0

    def get_first_timestamp(self, symbol):
        if symbol not in self.first_timestamps:
            logging.info(f"warning: {symbol} missing from first_timestamps. Setting to zero.")
            return 0.0
        return self.first_timestamps[symbol]

    def coin_to_symbol(self, coin):
        if not hasattr(self, "coin_to_symbol_map"):
            self.coin_to_symbol_map = {}
        if coin in self.coin_to_symbol_map:
            return self.coin_to_symbol_map[coin]
        coinf = symbol_to_coin(coin)
        if coinf in self.coin_to_symbol_map:
            self.coin_to_symbol_map[coin] = self.coin_to_symbol_map[coinf]
            return self.coin_to_symbol_map[coinf]
        result = coin_to_symbol(
            coin,
            eligible_symbols=self.eligible_symbols,
            quote=self.quote,
        )
        self.coin_to_symbol_map[coin] = result
        return result

    async def run_execution_loop(self):
        while not self.stop_signal_received:
            try:
                now = utc_ms()
                if now - self.previous_REST_update_ts > 1000 * 60:
                    self.previous_REST_update_ts = utc_ms()
                    await self.prepare_for_execution()
                await self.execute_to_exchange()
                await asyncio.sleep(
                    max(0.0, self.config["live"]["execution_delay_seconds"] - (utc_ms() - now) / 1000)
                )
            except Exception as e:
                logging.error(f"error with {get_function_name()} {e}")
                traceback.print_exc()
                await asyncio.sleep(1.0)

    async def prepare_for_execution(self):
        await asyncio.gather(
            self.update_open_orders(),
            self.update_positions(),
            self.update_pnls(),
        )
        await self.update_ohlcvs_1m_for_actives()

    async def execute_to_exchange(self):
        await self.execution_cycle()
        await self.update_EMAs()
        await self.update_exchange_configs()
        to_cancel, to_create = self.calc_orders_to_cancel_and_create()

        # debug duplicates
        seen = set()
        for elm in to_cancel:
            key = str(elm["price"]) + str(elm["qty"])
            if key in seen:
                logging.info(f"debug duplicate order cancel {elm}")
            seen.add(key)

        seen = set()
        for elm in to_create:
            key = str(elm["price"]) + str(elm["qty"])
            if key in seen:
                logging.info(f"debug duplicate order create {elm}")
            seen.add(key)

        # format custom_id
        to_create = self.format_custom_ids(to_create)
        if self.debug_mode:
            if to_cancel:
                print("would cancel:")
            for x in to_cancel[: self.config["live"]["max_n_cancellations_per_batch"]]:
                pprint.pprint(x)
        else:
            res = await self.execute_cancellations(
                to_cancel[: self.config["live"]["max_n_cancellations_per_batch"]]
            )
            if res:
                for elm in res:
                    self.remove_cancelled_order(elm, source="POST")
        if self.debug_mode:
            if to_create:
                print("would create:")
            for x in to_create[: self.config["live"]["max_n_creations_per_batch"]]:
                pprint.pprint(x)
        else:
            res = None
            try:
                res = await self.execute_orders(
                    to_create[: self.config["live"]["max_n_creations_per_batch"]]
                )
                if res:
                    for elm in res:
                        self.add_new_order(elm, source="POST")
            except Exception as e:
                logging.error(f"error executing orders {to_create} {e}")
                print_async_exception(res)
                traceback.print_exc()
                await self.restart_bot_on_too_many_errors()
        if to_cancel or to_create:
            self.previous_REST_update_ts = 0

    def is_forager_mode(self, pside=None):
        if pside is None:
            return self.is_forager_mode("long") or self.is_forager_mode("short")
        if self.config["bot"][pside]["total_wallet_exposure_limit"] <= 0.0:
            return False
        if self.forced_modes[pside]:
            return False
        n_positions = self.get_max_n_positions(pside)
        if n_positions == 0:
            return False
        if n_positions >= len(self.approved_coins[pside]):
            return False
        return True

    def set_live_configs(self):
        skip = {
            "n_positions",
            "total_wallet_exposure_limit",
            "unstuck_loss_allowance_pct",
            "unstuck_close_pct",
            "filter_rolling_window",
            "filter_relative_volume_clip_pct",
        }  # skip parameters affecting global behavior
        for pside in ["long", "short"]:
            self.config["bot"][pside]["n_positions"] = min(
                len(self.eligible_symbols), int(round(self.config["bot"][pside]["n_positions"]))
            )
        for symbol in self.markets_dict:
            if symbol in self.ineligible_symbols:
                if any(
                    [x in self.ineligible_symbols[symbol] for x in ["quote", "market type", "linear"]]
                ):
                    continue
            self.live_configs[symbol] = deepcopy(self.config["bot"])
            self.live_configs[symbol]["leverage"] = self.config["live"]["leverage"]
            if symbol in self.flags and self.flags[symbol].live_config_path is not None:
                try:
                    loaded = load_config(self.flags[symbol].live_config_path)
                    logging.info(
                        f"successfully loaded {self.flags[symbol].live_config_path} for {symbol}"
                    )
                    for pside in loaded["bot"]:
                        for k, v in loaded["bot"][pside].items():
                            if k not in skip:
                                self.live_configs[symbol][pside][k] = v
                except Exception as e:
                    logging.error(
                        f"failed to load config {self.flags[symbol].live_config_path} for {symbol} {e}. Using default config."
                    )
        self.set_wallet_exposure_limits()

    def pad_sym(self, symbol):
        return f"{symbol: <{self.sym_padding}}"

    def stop_data_maintainers(self, verbose=True):
        if not hasattr(self, "maintainers"):
            return
        res = {}
        for key in self.maintainers:
            try:
                res[key] = self.maintainers[key].cancel()
            except Exception as e:
                logging.error(f"error stopping maintainer {key} {e}")
        if hasattr(self, "WS_ohlcvs_1m_tasks"):
            res0s = {}
            for key in self.WS_ohlcvs_1m_tasks:
                try:
                    res0 = self.WS_ohlcvs_1m_tasks[key].cancel()
                    res0s[key] = res0
                except Exception as e:
                    logging.error(f"error stopping WS_ohlcvs_1m_tasks {key} {e}")
            if res0s:
                if verbose:
                    logging.info(f"stopped ohlcvs watcher tasks {res0s}")
        if verbose:
            logging.info(f"stopped data maintainers: {res}")
        return res

    def has_position(self, pside=None, symbol=None):
        if pside is None:
            return self.has_position("long", symbol) or self.has_position("short", symbol)
        if symbol is None:
            return any([self.has_position(pside, s) for s in self.positions])
        return symbol in self.positions and self.positions[symbol][pside]["size"] != 0.0

    def is_trailing(self, symbol, pside=None):
        if pside is None:
            return self.is_trailing(symbol, "long") or self.is_trailing(symbol, "short")
        return symbol in self.live_configs and any(
            [
                self.live_configs[symbol][pside][f"{x}_trailing_grid_ratio"] != 0.0
                for x in ["entry", "close"]
            ]
        )

    def get_last_position_changes(self, symbol=None):
        last_position_changes = defaultdict(dict)
        for symbol in self.positions:
            for pside in ["long", "short"]:
                if self.has_position(pside, symbol) and self.is_trailing(symbol, pside):
                    last_position_changes[symbol][pside] = utc_ms() - 1000 * 60 * 60 * 24 * 7
                    for fill in self.pnls[::-1]:
                        try:
                            if fill["symbol"] == symbol and fill["position_side"] == pside:
                                last_position_changes[symbol][pside] = fill["timestamp"]
                                break
                        except Exception as e:
                            logging.error(
                                f"Error with get_last_position_changes. Faulty element: {fill}"
                            )
        return last_position_changes

    async def wait_for_ohlcvs_1m_to_update(self):
        await asyncio.sleep(1.0)
        prev_print_ts = utc_ms() - 5000.0
        while (
            not self.stop_signal_received
            and self.n_symbols_missing_ohlcvs_1m > self.max_n_concurrent_ohlcvs_1m_updates - 1
        ):
            if utc_ms() - prev_print_ts > 1000 * 10:
                logging.info(
                    f"Waiting for ohlcvs to be refreshed. Number of symbols with "
                    f"out-of-date ohlcvs: {self.n_symbols_missing_ohlcvs_1m}"
                )
                prev_print_ts = utc_ms()
            await asyncio.sleep(0.1)

    def get_ohlcvs_1m_filepath(self, symbol):
        try:
            return self.ohlcvs_1m_filepaths[symbol]
        except:
            if not hasattr(self, "filepath"):
                self.ohlcvs_1m_filepaths = {}
            filepath = f"{self.ohlcvs_1m_cache_dirpath}{symbol_to_coin(symbol)}.npy"
            self.ohlcvs_1m_filepaths[symbol] = filepath
            return filepath

    def trim_ohlcvs_1m(self, symbol):
        try:
            if not hasattr(self, "ohlcvs_1m"):
                return
            if symbol not in self.ohlcvs_1m:
                return
            age_limit = (
                self.get_exchange_time() - 1000 * 60 * 60 * 24 * self.ohlcvs_1m_rolling_window_days
            )
            for i in range(len(self.ohlcvs_1m[symbol])):
                ts = self.ohlcvs_1m[symbol].peekitem(0)[0]
                if ts < age_limit:
                    del self.ohlcvs_1m[symbol][ts]
                else:
                    break
            return True
        except Exception as e:
            logging.error(f"error with {get_function_name()} {symbol} {e}")
            traceback.print_exc()
            return False

    def dump_ohlcvs_1m_to_cache(self, symbol):
        try:
            self.trim_ohlcvs_1m(symbol)
            to_dump = np.array([x for x in self.ohlcvs_1m[symbol].values()])
            np.save(self.get_ohlcvs_1m_filepath(symbol), to_dump)
            return True
        except Exception as e:
            logging.error(f"error with {get_function_name()} for {symbol}: {e}")
            traceback.print_exc()
            return False

    def update_trailing_data(self):
        if not hasattr(self, "trailing_prices"):
            self.trailing_prices = {}
        last_position_changes = self.get_last_position_changes()
        symbols = set(self.trailing_prices) | set(last_position_changes) | set(self.active_symbols)
        for symbol in symbols:
            self.trailing_prices[symbol] = {
                "long": {
                    "max_since_open": 0.0,
                    "min_since_max": np.inf,
                    "min_since_open": np.inf,
                    "max_since_min": 0.0,
                },
                "short": {
                    "max_since_open": 0.0,
                    "min_since_max": np.inf,
                    "min_since_open": np.inf,
                    "max_since_min": 0.0,
                },
            }
            if symbol not in last_position_changes:
                continue
            for pside in last_position_changes[symbol]:
                if symbol not in self.ohlcvs_1m:
                    logging.info(f"debug: {symbol} missing from self.ohlcvs_1m")
                    continue
                for ts in self.ohlcvs_1m[symbol]:
                    if ts <= last_position_changes[symbol][pside]:
                        continue
                    x = self.ohlcvs_1m[symbol][ts]
                    if x[2] > self.trailing_prices[symbol][pside]["max_since_open"]:
                        self.trailing_prices[symbol][pside]["max_since_open"] = x[2]
                        self.trailing_prices[symbol][pside]["min_since_max"] = x[4]
                    else:
                        self.trailing_prices[symbol][pside]["min_since_max"] = min(
                            self.trailing_prices[symbol][pside]["min_since_max"], x[3]
                        )
                    if x[3] < self.trailing_prices[symbol][pside]["min_since_open"]:
                        self.trailing_prices[symbol][pside]["min_since_open"] = x[3]
                        self.trailing_prices[symbol][pside]["max_since_min"] = x[4]
                    else:
                        self.trailing_prices[symbol][pside]["max_since_min"] = max(
                            self.trailing_prices[symbol][pside]["max_since_min"], x[2]
                        )

    def format_symbol(self, symbol: str) -> str:
        try:
            return self.formatted_symbols_map[symbol]
        except (KeyError, AttributeError):
            pass
        if not hasattr(self, "formatted_symbols_map"):
            self.formatted_symbols_map = {}
            self.formatted_symbols_map_inv = defaultdict(set)
        formatted = f"{symbol_to_coin(symbol.replace(',', ''))}/{self.quote}:{self.quote}"
        self.formatted_symbols_map[symbol] = formatted
        self.formatted_symbols_map_inv[formatted].add(symbol)
        return formatted

    def symbol_is_eligible(self, symbol):
        # defined for each child class
        return True

    def set_market_specific_settings(self):
        # set min cost, min qty, price step, qty step, c_mult
        # defined individually for each exchange
        self.symbol_ids = {symbol: self.markets_dict[symbol]["id"] for symbol in self.markets_dict}
        self.symbol_ids_inv = {v: k for k, v in self.symbol_ids.items()}

    def get_symbol_id(self, symbol):
        try:
            return self.symbol_ids[symbol]
        except:
            logging.info(f"debug: symbol {symbol} missing from self.symbol_ids. Using {symbol}")
            self.symbol_ids[symbol] = symbol
            return symbol

    def get_symbol_id_inv(self, symbol):
        try:
            return self.symbol_ids_inv[symbol]
        except:
            logging.info(f"debug: symbol {symbol} missing from self.symbol_ids_inv. Using {symbol}")
            self.symbol_ids_inv[symbol] = symbol
            return symbol

    def is_approved(self, pside, symbol) -> bool:
        if symbol not in self.approved_coins[pside]:
            return False
        if symbol in self.ignored_coins[pside]:
            return False
        if not self.is_old_enough(pside, symbol):
            return False
        return True

    async def update_exchange_configs(self):
        if not hasattr(self, "already_updated_exchange_config_symbols"):
            self.already_updated_exchange_config_symbols = set()
        symbols_not_done = [
            x for x in self.active_symbols if x not in self.already_updated_exchange_config_symbols
        ]
        if symbols_not_done:
            await self.update_exchange_config_by_symbols(symbols_not_done)
            self.already_updated_exchange_config_symbols.update(symbols_not_done)

    async def update_exchange_config_by_symbols(self, symbols):
        # defined by each exchange child class
        pass

    async def update_exchange_config(self):
        # defined by each exchange child class
        pass

    def reformat_symbol(self, symbol: str, verbose=False) -> str:
        # tries to reformat symbol to correct variant for exchange
        # (e.g. BONK -> 1000BONK/USDT:USDT, PEPE - kPEPE/USDC:USDC)
        # if no reformatting is possible, return empty string
        fsymbol = self.format_symbol(symbol)
        if fsymbol in self.markets_dict:
            return fsymbol
        else:
            if verbose:
                logging.info(f"{symbol} missing from {self.exchange}")
            if fsymbol in self.formatted_symbols_map_inv:
                for x in self.formatted_symbols_map_inv[fsymbol]:
                    if x in self.markets_dict:
                        if verbose:
                            logging.info(f"changing {symbol} -> {x}")
                        return x
        return ""

    def is_old_enough(self, pside, symbol):
        if self.is_forager_mode(pside) and self.minimum_market_age_millis > 0:
            first_timestamp = self.get_first_timestamp(symbol)
            if first_timestamp:
                return utc_ms() - first_timestamp > self.minimum_market_age_millis
            else:
                return False
        else:
            return True

    async def update_tickers(self):
        if not hasattr(self, "tickers"):
            self.tickers = {}
        tickers = None
        try:
            tickers = await self.cca.fetch_tickers()
            for symbol in tickers:
                if tickers[symbol]["last"] is None:
                    if tickers[symbol]["bid"] is not None and tickers[symbol]["ask"] is not None:
                        tickers[symbol]["last"] = np.mean(
                            [tickers[symbol]["bid"], tickers[symbol]["ask"]]
                        )
                else:
                    for oside in ["bid", "ask"]:
                        if tickers[symbol][oside] is None and tickers[symbol]["last"] is not None:
                            tickers[symbol][oside] = tickers[symbol]["last"]
            self.tickers = tickers
        except Exception as e:
            logging.error(f"Error with {get_function_name()} {e}")

    async def execution_cycle(self):
        # called before every execution to exchange
        # assumes positions, open orders are up to date
        # determine coins with position and open orders
        # determine eligible/ineligible coins
        # determine approved/ignored coins
        #   from external ignored/approved coins files
        #   from coin age
        #   from effective min cost (only if has updated price info)
        # determine and set special t,p,m modes and forced modes
        # determine ideal coins from noisiness and volume
        # determine coins with pos for normal or gs modes
        # determine coins from ideal coins for normal modes

        self.update_effective_min_cost()
        self.refresh_approved_ignored_coins_lists()
        self.set_wallet_exposure_limits()
        previous_PB_modes = deepcopy(self.PB_modes) if hasattr(self, "PB_modes") else None
        self.PB_modes = {"long": {}, "short": {}}
        for pside, other_pside in [("long", "short"), ("short", "long")]:
            if self.is_forager_mode(pside):
                await self.update_first_timestamps()
            for symbol in self.flagged_modes[pside]:
                self.PB_modes[pside][symbol] = self.flagged_modes[pside][symbol]
            ideal_coins = self.get_filtered_coins(pside)
            slots_filled = {
                k for k, v in self.PB_modes[pside].items() if v in ["normal", "graceful_stop"]
            }
            max_n_positions = self.get_max_n_positions(pside)
            symbols_with_pos = self.get_symbols_with_pos(pside)
            for symbol in symbols_with_pos:
                if symbol in self.PB_modes[pside]:
                    continue
                elif self.forced_modes[pside]:
                    self.PB_modes[pside][symbol] = self.forced_modes[pside]
                else:
                    if symbol in self.ineligible_symbols:
                        if self.ineligible_symbols[symbol] == "not active":
                            self.PB_modes[pside][symbol] = "tp_only"
                        else:
                            self.PB_modes[pside][symbol] = "manual"
                    elif len(symbols_with_pos) > max_n_positions:
                        self.PB_modes[pside][symbol] = self.PB_mode_stop[pside]
                    elif symbol in ideal_coins:
                        self.PB_modes[pside][symbol] = "normal"
                    else:
                        self.PB_modes[pside][symbol] = self.PB_mode_stop[pside]
                    slots_filled.add(symbol)
            for symbol in ideal_coins:
                if len(slots_filled) >= max_n_positions:
                    break
                if symbol in self.PB_modes[pside]:
                    continue
                if not self.hedge_mode and self.has_position(other_pside, symbol):
                    continue
                self.PB_modes[pside][symbol] = "normal"
                slots_filled.add(symbol)
            for symbol in self.open_orders:
                if symbol in self.PB_modes[pside]:
                    continue
                self.PB_modes[pside][symbol] = self.PB_mode_stop[pside]
        self.active_symbols = sorted(
            {s for subdict in self.PB_modes.values() for s in subdict.keys()}
        )
        for symbol in self.active_symbols:
            for pside in self.PB_modes:
                if symbol not in self.PB_modes[pside]:
                    self.PB_modes[pside][symbol] = self.PB_mode_stop[pside]
            if symbol not in self.positions:
                self.positions[symbol] = {
                    "long": {"size": 0.0, "price": 0.0},
                    "short": {"size": 0.0, "price": 0.0},
                }
            if symbol not in self.open_orders:
                self.open_orders[symbol] = []
        self.set_wallet_exposure_limits()
        self.update_trailing_data()
        res = log_dict_changes(previous_PB_modes, self.PB_modes)
        for k, v in res.items():
            for elm in v:
                logging.info(f"{k} {elm}")

    def get_filtered_coins(self, pside):
        # filter coins by age
        # filter coins by min effective cost
        # filter coins by relative volume
        # filter coins by noisiness
        if self.is_forager_mode(pside):
            candidates = self.approved_coins_minus_ignored_coins[pside]
            candidates = [s for s in candidates if self.is_old_enough(pside, s)]
            candidates = [s for s in candidates if self.effective_min_cost_is_low_enough(pside, s)]
            if candidates == []:
                self.warn_on_high_effective_min_cost(pside)
            # filter coins by relative volume and noisiness
            clip_pct = self.config["bot"][pside]["filter_relative_volume_clip_pct"]
            max_n_positions = self.get_max_n_positions(pside)
            if clip_pct > 0.0:
                volumes = self.calc_volumes(pside, symbols=candidates)
                # filter by relative volume
                n_eligible = round(len(volumes) * (1 - clip_pct))
                candidates = sorted(volumes, key=lambda x: volumes[x], reverse=True)
                candidates = candidates[: int(max(n_eligible, max_n_positions))]
            # ideal symbols are high noise symbols
            noisiness = self.calc_noisiness(pside, eligible_symbols=candidates)
            noisiness = {k: v for k, v in sorted(noisiness.items(), key=lambda x: x[1], reverse=True)}
            ideal_coins = [k for k in noisiness.keys()][:max_n_positions]
        elif self.forced_modes[pside]:
            return []
        else:
            # all approved coins are selected, no filtering
            ideal_coins = sorted(self.approved_coins_minus_ignored_coins[pside])
        return ideal_coins

    def warn_on_high_effective_min_cost(self, pside):
        if not self.config["live"]["filter_by_min_effective_cost"]:
            return
        eligible_symbols_filtered = [
            x for x in self.eligible_symbols if self.effective_min_cost_is_low_enough(pside, x)
        ]
        if len(eligible_symbols_filtered) == 0:
            logging.info(
                f"Warning: No {pside} symbols are approved due to min effective cost too high. "
                + f"Suggestions: 1) increase account balance, 2) "
                + f"set 'filter_by_min_effective_cost' to false, 3) reduce n_{pside}s"
            )

    def get_max_n_positions(self, pside):
        max_n_positions = min(
            self.config["bot"][pside]["n_positions"],
            len(self.approved_coins_minus_ignored_coins[pside]),
        )
        return max(0, max_n_positions)

    def get_current_n_positions(self, pside):
        n_positions = 0
        for symbol in self.positions:
            if self.positions[symbol][pside]["size"] != 0.0:
                if symbol in self.flagged_modes[pside]:
                    if self.flagged_modes[pside][symbol] in ["normal", "graceful_stop"]:
                        n_positions += 1
                else:
                    n_positions += 1
        return n_positions

    def set_wallet_exposure_limits(self):
        for symbol in self.live_configs:
            for pside in ["long", "short"]:
                self.live_configs[symbol][pside]["wallet_exposure_limit"] = (
                    self.get_wallet_exposure_limit(pside, symbol)
                )

    def get_wallet_exposure_limit(self, pside, symbol):
        if (
            symbol in self.flags
            and (fwel := getattr(self.flags[symbol], f"WE_limit_{pside}")) is not None
        ):
            return fwel
        else:
            twel = self.config["bot"][pside]["total_wallet_exposure_limit"]
            if twel == 0.0:
                return 0.0
            n_positions = max(self.get_max_n_positions(pside), self.get_current_n_positions(pside))
            if n_positions == 0:
                return 0.0
            return round(twel / n_positions, 8)

    def effective_min_cost_is_low_enough(self, pside, symbol):
        if not self.config["live"]["filter_by_min_effective_cost"]:
            return True
        return (
            self.balance
            * self.get_wallet_exposure_limit(pside, symbol)
            * self.live_configs[symbol][pside]["entry_initial_qty_pct"]
            >= self.effective_min_cost[symbol]
        )

    def add_new_order(self, order, source="WS"):
        try:
            if not order or "id" not in order:
                return False
            if "symbol" not in order or order["symbol"] is None:
                logging.info(f"symbol not in order. Source: {source} {order}")
                return False
            if order["symbol"] not in self.open_orders:
                self.open_orders[order["symbol"]] = []
            if order["id"] not in {x["id"] for x in self.open_orders[order["symbol"]]}:
                self.open_orders[order["symbol"]].append(order)
                logging.info(
                    f"  created {self.pad_sym(order['symbol'])} {order['side']} {order['qty']} {order['position_side']} @ {order['price']} source: {source}"
                )
                return True
        except Exception as e:
            logging.error(f"failed to add order to self.open_orders {source} {order} {e}")
            traceback.print_exc()
            return False

    def remove_cancelled_order(self, order: dict, source="WS"):
        try:
            if not order or "id" not in order:
                return False
            if "symbol" not in order or order["symbol"] is None:
                logging.info(f"{order}")
                return False
            if order["symbol"] not in self.open_orders:
                self.open_orders[order["symbol"]] = []
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
                if upd["status"] == "closed" or (
                    "filled" in upd and upd["filled"] is not None and upd["filled"] > 0.0
                ):
                    # There was a fill, partial or full. Schedule update of open orders, pnls, position.
                    logging.info(
                        f"   filled {self.pad_sym(upd['symbol'])} {upd['side']} {upd['qty']} {upd['position_side']} @ {upd['price']} source: WS"
                    )
                    self.recent_fill = True
                    self.previous_REST_update_ts = 0
                elif upd["status"] in ["canceled", "expired", "rejected"]:
                    # remove order from open_orders
                    self.remove_cancelled_order(upd, source="WS")
                elif upd["status"] == "open":
                    # add order to open_orders
                    self.add_new_order(upd, source="WS")
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

    def handle_ohlcv_1m_update(self, symbol, upd):
        if symbol not in self.ohlcvs_1m:
            self.ohlcvs_1m[symbol] = SortedDict()
        for elm in upd:
            self.ohlcvs_1m[symbol][int(elm[0])] = elm
            self.ohlcvs_1m_update_timestamps_WS[symbol] = utc_ms()

    def calc_upnl_sum(self):
        upnl_sum = 0.0
        for elm in self.fetched_positions:
            try:
                upnl = calc_pnl(
                    elm["position_side"],
                    elm["price"],
                    self.get_last_price(elm["symbol"]),
                    elm["size"],
                    self.inverse,
                    self.c_mults[elm["symbol"]],
                )
                if upnl:
                    upnl_sum += upnl
            except Exception as e:
                logging.error(f"error calculating upnl sum {e}")
                traceback.print_exc()
                return 0.0
        return upnl_sum

    async def init_pnls(self):
        if not hasattr(self, "pnls"):
            self.pnls = []
        elif self.pnls:
            return  # pnls already initiated; abort
        logging.info(f"initiating pnls...")
        age_limit = (
            self.get_exchange_time()
            - 1000 * 60 * 60 * 24 * self.config["live"]["pnls_max_lookback_days"]
        )
        pnls_cache = []
        if os.path.exists(self.pnls_cache_filepath):
            try:
                pnls_cache = json.load(open(self.pnls_cache_filepath))
            except Exception as e:
                logging.error(f"error loading {self.pnls_cache_filepath} {e}")
        if pnls_cache:
            newest_pnls = await self.fetch_pnls(start_time=pnls_cache[-1]["timestamp"])
            if pnls_cache[0]["timestamp"] > age_limit + 1000 * 60 * 60 * 4:
                # might be older missing pnls
                logging.info(
                    f"fetching missing pnls from before {ts_to_date_utc(pnls_cache[0]['timestamp'])}"
                )
                missing_pnls = await self.fetch_pnls(
                    start_time=age_limit, end_time=pnls_cache[0]["timestamp"]
                )
                pnls_cache = sorted(
                    {
                        elm["id"]: elm
                        for elm in pnls_cache + missing_pnls + newest_pnls
                        if elm["timestamp"] >= age_limit
                    }.values(),
                    key=lambda x: x["timestamp"],
                )
        else:
            pnls_cache = await self.fetch_pnls(start_time=age_limit)
        self.pnls = pnls_cache

    async def update_pnls(self):
        # fetch latest pnls
        # dump new pnls to cache
        age_limit = (
            self.get_exchange_time()
            - 1000 * 60 * 60 * 24 * self.config["live"]["pnls_max_lookback_days"]
        )
        if not hasattr(self, "pnls"):
            self.pnls = []
        old_ids = {elm["id"] for elm in self.pnls}
        if len(self.pnls) == 0:
            await self.init_pnls()
        start_time = self.pnls[-1]["timestamp"] - 1000 if self.pnls else age_limit
        res = await self.fetch_pnls(start_time=start_time, limit=100)
        if res in [None, False]:
            return False
        new_pnls = [x for x in res if x["id"] not in old_ids]
        self.pnls = sorted(
            {
                elm["id"]: elm for elm in self.pnls + new_pnls if elm["timestamp"] >= age_limit
            }.values(),
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
        res = None
        try:
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
        except Exception as e:
            logging.error(f"error with {get_function_name()} {e}")
            print_async_exception(res)
            traceback.print_exc()
            return False

    async def determine_utc_offset(self, verbose=True):
        # returns millis to add to utc to get exchange timestamp
        # call some endpoint which includes timestamp for exchange's server
        # if timestamp is not included in self.cca.fetch_balance(),
        # implement method in exchange child class
        result = await self.cca.fetch_balance()
        self.utc_offset = round((result["timestamp"] - utc_ms()) / (1000 * 60 * 60)) * (
            1000 * 60 * 60
        )
        if verbose:
            logging.info(f"Exchange time offset is {self.utc_offset}ms compared to UTC")

    def get_exchange_time(self):
        return utc_ms() + self.utc_offset

    async def update_positions(self):
        # also updates balance
        if not hasattr(self, "positions"):
            self.positions = {}
        res = await self.fetch_positions()
        if not res or all(x in [None, False] for x in res):
            return False
        positions_list_new, balance_new = res
        self.fetched_positions = positions_list_new
        self.handle_balance_update({self.quote: {"total": balance_new}}, source="REST")
        positions_new = {
            sym: {
                "long": {"size": 0.0, "price": 0.0},
                "short": {"size": 0.0, "price": 0.0},
            }
            for sym in set(list(self.positions) + list(self.active_symbols))
        }
        position_changes = []
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
                position_changes.append((symbol, pside))
        try:
            self.log_position_changes(position_changes, positions_new)
        except Exception as e:
            logging.error(f"error printing position changes {e}")
        self.positions = positions_new
        self.upd_timestamps["positions"] = utc_ms()
        return True

    def get_last_price(self, symbol, null_replace=0.0):
        if not hasattr(self, "ohlcvs_1m") or symbol not in self.ohlcvs_1m:
            try:
                if hasattr(self, "tickers") and symbol in self.tickers:
                    res = self.tickers[symbol]["last"]
                    if res is None:
                        logging.info(f"debug get_last_price {symbol} price from tickers is null")
                        return null_replace
                    return res
            except Exception as e:
                logging.error(f"Error fetching last price from tickers")
        try:
            if symbol in self.ohlcvs_1m and self.ohlcvs_1m[symbol]:
                res = self.ohlcvs_1m[symbol].peekitem(-1)[1][4]
                if res is None:
                    logging.info(f"debug get_last_price {symbol} price from ohlcvs_1m is null")
                    return null_replace
                return res
        except Exception as e:
            logging.error(f"error with {get_function_name()} for {symbol}: {e}")
            traceback.print_exc()
        logging.info(f"debug get_last_price {symbol} failed")
        return null_replace

    def log_position_changes(self, position_changes, positions_new, rd=6) -> str:
        if not position_changes:
            return ""
        table = PrettyTable()
        table.border = False
        table.header = False
        table.padding_width = 0  # Reduces padding between columns to zero
        for symbol, pside in position_changes:
            wallet_exposure = (
                pbr.qty_to_cost(
                    positions_new[symbol][pside]["size"],
                    positions_new[symbol][pside]["price"],
                    self.c_mults[symbol],
                )
                / self.balance
            )
            try:
                wel = self.live_configs[symbol][pside]["wallet_exposure_limit"]
                WE_ratio = wallet_exposure / wel if wel > 0.0 else 0.0
            except Exception as e:
                logging.error(f"error with log_position_changes {e}")
                WE_ratio = 0.0
            last_price = or_default(self.get_last_price, symbol, default=0.0)
            try:
                pprice_diff = (
                    calc_pprice_diff(pside, positions_new[symbol][pside]["price"], last_price)
                    if last_price
                    else 0.0
                )
            except:
                pprice_diff = 0.0
            try:
                upnl = (
                    calc_pnl(
                        pside,
                        positions_new[symbol][pside]["price"],
                        self.get_last_price(symbol),
                        positions_new[symbol][pside]["size"],
                        self.inverse,
                        self.c_mults[symbol],
                    )
                    if last_price
                    else 0.0
                )
            except Exception as e:
                upnl = 0.0
            table.add_row(
                [
                    symbol + " ",
                    pside + " ",
                    (
                        round_dynamic(self.positions[symbol][pside]["size"], rd)
                        if symbol in self.positions
                        else 0.0
                    ),
                    " @ ",
                    (
                        round_dynamic(self.positions[symbol][pside]["price"], rd)
                        if symbol in self.positions
                        else 0.0
                    ),
                    " -> ",
                    round_dynamic(positions_new[symbol][pside]["size"], rd),
                    " @ ",
                    round_dynamic(positions_new[symbol][pside]["price"], rd),
                    " WE: ",
                    round_dynamic(wallet_exposure, max(3, rd - 2)),
                    " WE ratio: ",
                    round(WE_ratio, 3),
                    " PA dist: ",
                    round(pprice_diff, 4),
                    " upnl: ",
                    round_dynamic(upnl, max(3, rd - 1)),
                ]
            )
        string = table.get_string()
        for line in string.splitlines():
            logging.info(line)
        return string

    def update_effective_min_cost(self, symbol=None):
        if not hasattr(self, "effective_min_cost"):
            self.effective_min_cost = {}
        if symbol is None:
            symbols = sorted(self.eligible_symbols)
        else:
            symbols = [symbol]
        for symbol in symbols:
            try:
                self.effective_min_cost[symbol] = max(
                    pbr.qty_to_cost(
                        self.min_qtys[symbol],
                        self.get_last_price(symbol),
                        self.c_mults[symbol],
                    ),
                    self.min_costs[symbol],
                )
            except Exception as e:
                logging.error(f"error with {get_function_name()} for {symbol}: {e}")
                traceback.print_exc()

    def calc_ideal_orders(self):
        ideal_orders = {symbol: [] for symbol in self.active_symbols}
        for pside in self.PB_modes:
            for symbol in self.PB_modes[pside]:
                if self.PB_modes[pside][symbol] == "panic":
                    if self.has_position(pside, symbol):
                        # if in panic mode, only one close order at current market price
                        qmul = -1 if pside == "long" else 1
                        ideal_orders[symbol].append(
                            (
                                abs(self.positions[symbol][pside]["size"]) * qmul,
                                self.get_last_price(symbol),
                                f"panic_close_{pside}",
                            )
                        )
                elif self.PB_modes[pside][symbol] in [
                    "graceful_stop",
                    "tp_only",
                ] and not self.has_position(pside, symbol):
                    pass
                elif self.PB_modes[pside][symbol] == "manual":
                    pass
                else:
                    entries = getattr(pbr, f"calc_entries_{pside}_py")(
                        self.qty_steps[symbol],
                        self.price_steps[symbol],
                        self.min_qtys[symbol],
                        self.min_costs[symbol],
                        self.c_mults[symbol],
                        self.live_configs[symbol][pside]["entry_grid_double_down_factor"],
                        self.live_configs[symbol][pside]["entry_grid_spacing_weight"],
                        self.live_configs[symbol][pside]["entry_grid_spacing_pct"],
                        self.live_configs[symbol][pside]["entry_initial_ema_dist"],
                        self.live_configs[symbol][pside]["entry_initial_qty_pct"],
                        self.live_configs[symbol][pside]["entry_trailing_grid_ratio"],
                        self.live_configs[symbol][pside]["entry_trailing_retracement_pct"],
                        self.live_configs[symbol][pside]["entry_trailing_threshold_pct"],
                        self.live_configs[symbol][pside]["wallet_exposure_limit"],
                        self.balance,
                        self.positions[symbol][pside]["size"],
                        self.positions[symbol][pside]["price"],
                        self.trailing_prices[symbol][pside]["min_since_open"],
                        self.trailing_prices[symbol][pside]["max_since_min"],
                        self.emas[pside][symbol].min(),
                        self.get_last_price(symbol),
                    )
                    closes = getattr(pbr, f"calc_closes_{pside}_py")(
                        self.qty_steps[symbol],
                        self.price_steps[symbol],
                        self.min_qtys[symbol],
                        self.min_costs[symbol],
                        self.c_mults[symbol],
                        self.live_configs[symbol][pside]["close_grid_markup_range"],
                        self.live_configs[symbol][pside]["close_grid_min_markup"],
                        self.live_configs[symbol][pside]["close_grid_qty_pct"],
                        self.live_configs[symbol][pside]["close_trailing_grid_ratio"],
                        self.live_configs[symbol][pside]["close_trailing_qty_pct"],
                        self.live_configs[symbol][pside]["close_trailing_retracement_pct"],
                        self.live_configs[symbol][pside]["close_trailing_threshold_pct"],
                        self.live_configs[symbol][pside]["wallet_exposure_limit"],
                        self.balance,
                        self.positions[symbol][pside]["size"],
                        self.positions[symbol][pside]["price"],
                        self.trailing_prices[symbol][pside]["max_since_open"],
                        self.trailing_prices[symbol][pside]["min_since_max"],
                        self.get_last_price(symbol),
                    )
                    ideal_orders[symbol] += entries + closes

        unstucking_symbol, unstucking_close = self.calc_unstucking_close(ideal_orders)
        if unstucking_close[0] != 0.0:
            ideal_orders[unstucking_symbol] = [
                x for x in ideal_orders[unstucking_symbol] if not "close" in x[2]
            ]
            ideal_orders[unstucking_symbol].append(unstucking_close)

        ideal_orders_f = {}
        for symbol in ideal_orders:
            ideal_orders_f[symbol] = []
            with_pprice_diff = [
                (calc_diff(x[1], self.get_last_price(symbol)), x) for x in ideal_orders[symbol]
            ]
            seen = set()
            any_partial = any(["partial" in order[2] for _, order in with_pprice_diff])
            for pprice_diff, order in sorted(with_pprice_diff):
                position_side = "long" if "long" in order[2] else "short"
                if order[0] == 0.0:
                    continue
                if pprice_diff > self.config["live"]["price_distance_threshold"]:
                    if any_partial and "entry" in order[2]:
                        continue
                    if any([x in order[2] for x in ["initial", "unstuck"]]):
                        continue
                    if not self.has_position(position_side, symbol):
                        continue
                seen_key = str(abs(order[0])) + str(order[1]) + order[2]
                if seen_key in seen:
                    logging.info(f"debug duplicate ideal order {symbol} {order}")
                    continue
                ideal_orders_f[symbol].append(
                    {
                        "symbol": symbol,
                        "side": determine_side_from_order_tuple(order),
                        "position_side": position_side,
                        "qty": abs(order[0]),
                        "price": order[1],
                        "reduce_only": "close" in order[2],
                        "custom_id": order[2],
                    }
                )
                seen.add(seen_key)
        return ideal_orders_f

    def calc_unstucking_close(self, ideal_orders):
        stuck_positions = []
        pnls_cumsum = np.array([x["pnl"] for x in self.pnls]).cumsum()
        unstuck_allowances = {"long": 0.0, "short": 0.0}
        for symbol in self.positions:
            for pside in ["long", "short"]:
                if (
                    self.has_position(pside, symbol)
                    and self.live_configs[symbol][pside]["unstuck_loss_allowance_pct"] > 0.0
                ):
                    wallet_exposure = pbr.calc_wallet_exposure(
                        self.c_mults[symbol],
                        self.balance,
                        self.positions[symbol][pside]["size"],
                        self.positions[symbol][pside]["price"],
                    )
                    if (
                        self.live_configs[symbol][pside]["wallet_exposure_limit"] == 0.0
                        or wallet_exposure / self.live_configs[symbol][pside]["wallet_exposure_limit"]
                        > self.live_configs[symbol][pside]["unstuck_threshold"]
                    ):
                        unstuck_allowance = (
                            pbr.calc_auto_unstuck_allowance(
                                self.balance,
                                self.config["bot"][pside]["unstuck_loss_allowance_pct"]
                                * self.config["bot"][pside]["total_wallet_exposure_limit"],
                                pnls_cumsum.max(),
                                pnls_cumsum[-1],
                            )
                            if len(pnls_cumsum) > 0
                            else 0.0
                        )
                        unstuck_allowances[pside] = unstuck_allowance
                        if unstuck_allowance > 0.0:
                            pprice_diff = calc_pprice_diff(
                                pside,
                                self.positions[symbol][pside]["price"],
                                self.get_last_price(symbol),
                            )
                            stuck_positions.append((symbol, pside, pprice_diff))
        if not stuck_positions:
            return "", (0.0, 0.0, "")
        stuck_positions.sort(key=lambda x: x[2])
        for symbol, pside, _ in stuck_positions:
            if pside == "long":
                close_price = max(
                    self.get_last_price(symbol),
                    pbr.round_up(
                        self.emas[pside][symbol].max()
                        * (1.0 + self.live_configs[symbol][pside]["unstuck_ema_dist"]),
                        self.price_steps[symbol],
                    ),
                )
                ideal_closes = (
                    [x for x in ideal_orders[symbol] if "close" in x[2] and pside in x[2]]
                    if symbol in ideal_orders
                    else []
                )
                if ideal_closes and close_price >= ideal_closes[0][1]:
                    continue
                min_entry_qty = calc_min_entry_qty(
                    close_price,
                    False,
                    self.c_mults[symbol],
                    self.qty_steps[symbol],
                    self.min_qtys[symbol],
                    self.min_costs[symbol],
                )
                close_qty = -min(
                    self.positions[symbol][pside]["size"],
                    max(
                        min_entry_qty,
                        pbr.round_dn(
                            pbr.cost_to_qty(
                                self.balance
                                * self.live_configs[symbol][pside]["wallet_exposure_limit"]
                                * self.live_configs[symbol][pside]["unstuck_close_pct"],
                                close_price,
                                self.c_mults[symbol],
                            ),
                            self.qty_steps[symbol],
                        ),
                    ),
                )
                if close_qty != 0.0:
                    pnl_if_closed = getattr(pbr, f"calc_pnl_{pside}")(
                        self.positions[symbol][pside]["price"],
                        close_price,
                        close_qty,
                        self.c_mults[symbol],
                    )
                    pnl_if_closed_abs = abs(pnl_if_closed)
                    if pnl_if_closed < 0.0 and pnl_if_closed_abs > unstuck_allowances[pside]:
                        close_qty = -min(
                            self.positions[symbol][pside]["size"],
                            max(
                                min_entry_qty,
                                pbr.round_dn(
                                    abs(close_qty) * (unstuck_allowances[pside] / pnl_if_closed_abs),
                                    self.qty_steps[symbol],
                                ),
                            ),
                        )
                    return symbol, (close_qty, close_price, "unstuck_close_long")
            elif pside == "short":
                close_price = min(
                    self.get_last_price(symbol),
                    pbr.round_dn(
                        self.emas[pside][symbol].min()
                        * (1.0 - self.live_configs[symbol][pside]["unstuck_ema_dist"]),
                        self.price_steps[symbol],
                    ),
                )
                ideal_closes = (
                    [x for x in ideal_orders[symbol] if "close" in x[2] and pside in x[2]]
                    if symbol in ideal_orders
                    else []
                )
                if ideal_closes and close_price <= ideal_closes[0][1]:
                    continue
                min_entry_qty = calc_min_entry_qty(
                    close_price,
                    False,
                    self.c_mults[symbol],
                    self.qty_steps[symbol],
                    self.min_qtys[symbol],
                    self.min_costs[symbol],
                )
                close_qty = min(
                    abs(self.positions[symbol][pside]["size"]),
                    max(
                        min_entry_qty,
                        pbr.round_dn(
                            pbr.cost_to_qty(
                                self.balance
                                * self.live_configs[symbol][pside]["wallet_exposure_limit"]
                                * self.live_configs[symbol][pside]["unstuck_close_pct"],
                                close_price,
                                self.c_mults[symbol],
                            ),
                            self.qty_steps[symbol],
                        ),
                    ),
                )
                if close_qty != 0.0:
                    pnl_if_closed = getattr(pbr, f"calc_pnl_{pside}")(
                        self.positions[symbol][pside]["price"],
                        close_price,
                        close_qty,
                        self.c_mults[symbol],
                    )
                    pnl_if_closed_abs = abs(pnl_if_closed)
                    if pnl_if_closed < 0.0 and pnl_if_closed_abs > unstuck_allowances[pside]:
                        close_qty = min(
                            abs(self.positions[symbol][pside]["size"]),
                            max(
                                min_entry_qty,
                                pbr.round_dn(
                                    close_qty * (unstuck_allowances[pside] / pnl_if_closed_abs),
                                    self.qty_steps[symbol],
                                ),
                            ),
                        )
                    return symbol, (close_qty, close_price, "unstuck_close_short")
        return "", (0.0, 0.0, "")

    def calc_orders_to_cancel_and_create(self):
        ideal_orders = self.calc_ideal_orders()
        actual_orders = {}
        for symbol in self.active_symbols:
            actual_orders[symbol] = []
            for x in self.open_orders[symbol] if symbol in self.open_orders else []:
                try:
                    actual_orders[symbol].append(
                        {
                            "symbol": x["symbol"],
                            "side": x["side"],
                            "position_side": x["position_side"],
                            "qty": abs(x["qty"]),
                            "price": x["price"],
                            "reduce_only": (x["position_side"] == "long" and x["side"] == "sell")
                            or (x["position_side"] == "short" and x["side"] == "buy"),
                            "id": x["id"],
                        }
                    )
                except Exception as e:
                    logging.error(f"error in calc_orders_to_cancel_and_create {e}")
                    traceback.print_exc()
                    print(x)
        keys = ("symbol", "side", "position_side", "qty", "price")
        to_cancel, to_create = [], []
        for symbol in actual_orders:
            to_cancel_, to_create_ = filter_orders(actual_orders[symbol], ideal_orders[symbol], keys)
            for pside in ["long", "short"]:
                if self.PB_modes[pside][symbol] == "manual":
                    # neither create nor cancel orders
                    to_cancel_ = [x for x in to_cancel_ if x["position_side"] != pside]
                    to_create_ = [x for x in to_create_ if x["position_side"] != pside]
                elif self.PB_modes[pside][symbol] == "tp_only":
                    # if take profit only mode, neither cancel nor create entries
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
        to_create_with_pprice_diff = []
        for x in to_create:
            try:
                to_create_with_pprice_diff.append(
                    (calc_diff(x["price"], self.get_last_price(x["symbol"])), x)
                )
            except Exception as e:
                logging.info(f"debug: price missing sort to_create by pprice_diff {x} {e}")
                to_create_with_pprice_diff.append((0.0, x))
        to_create_with_pprice_diff.sort(key=lambda x: x[0])
        to_cancel_with_pprice_diff = []
        for x in to_cancel:
            try:
                to_cancel_with_pprice_diff.append(
                    (calc_diff(x["price"], self.get_last_price(x["symbol"])), x)
                )
            except Exception as e:
                logging.info(f"debug: price missing sort to_cancel by pprice_diff {x} {e}")
                to_cancel_with_pprice_diff.append((0.0, x))
        to_cancel_with_pprice_diff.sort(key=lambda x: x[0])
        return [x[1] for x in to_cancel_with_pprice_diff], [x[1] for x in to_create_with_pprice_diff]

    async def restart_bot_on_too_many_errors(self):
        if not hasattr(self, "error_counts"):
            self.error_counts = []
        now = utc_ms()
        self.error_counts = [x for x in self.error_counts if x > now - 1000 * 60 * 60] + [now]
        max_n_errors_per_hour = 10
        logging.info(
            f"error count: {len(self.error_counts)} of {max_n_errors_per_hour} errors per hour"
        )
        if len(self.error_counts) >= max_n_errors_per_hour:
            await self.restart_bot()
            raise Exception("too many errors... restarting bot.")

    def format_custom_ids(self, orders: [dict]) -> [dict]:
        new_orders = []
        for order in orders:
            order["custom_id"] = (
                shorten_custom_id(order["custom_id"] if "custom_id" in order else "") + uuid4().hex
            )[: self.custom_id_max_length]
            new_orders.append(order)
        return new_orders

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

    def fill_gaps_ohlcvs_1m(self):
        for symbol in self.ohlcvs_1m:
            self.fill_gaps_ohlcvs_1m_single(symbol)

    def fill_gaps_ohlcvs_1m_single(self, symbol):
        if symbol not in self.ohlcvs_1m or not self.ohlcvs_1m[symbol]:
            return
        now_minute = int(self.get_exchange_time() // 60000 * 60000)
        last_ts, last_ohlcv_1m = self.ohlcvs_1m[symbol].peekitem(-1)
        if now_minute > last_ts:
            self.ohlcvs_1m[symbol][now_minute] = [float(now_minute)] + [last_ohlcv_1m[4]] * 4 + [0.0]
        n_ohlcvs_1m = len(self.ohlcvs_1m[symbol])
        range_ms = self.ohlcvs_1m[symbol].peekitem(-1)[0] - self.ohlcvs_1m[symbol].peekitem(0)[0]
        ideal_n_ohlcvs_1m = int((range_ms) / 60000) + 1
        if ideal_n_ohlcvs_1m > n_ohlcvs_1m:
            ts = self.ohlcvs_1m[symbol].peekitem(0)[0]
            last_ts = self.ohlcvs_1m[symbol].peekitem(-1)[0]
            while ts < last_ts:
                ts += 60000
                if ts not in self.ohlcvs_1m[symbol]:
                    self.ohlcvs_1m[symbol][ts] = (
                        [float(ts)] + [self.ohlcvs_1m[symbol][ts - 60000][4]] * 4 + [0.0]
                    )

    def init_EMAs_single(self, symbol):
        first_ts, first_ohlcv = self.ohlcvs_1m[symbol].peekitem(0)
        for pside in ["long", "short"]:
            self.emas[pside][symbol] = np.repeat(first_ohlcv[4], 3)
            lc = self.live_configs[symbol][pside]
            es = [lc["ema_span_0"], lc["ema_span_1"], (lc["ema_span_0"] * lc["ema_span_1"]) ** 0.5]
            ema_spans = numpyize(sorted(es))
            self.ema_alphas[pside][symbol] = (a := (2.0 / (ema_spans + 1)), 1.0 - a)
        self.upd_minute_emas[symbol] = first_ts

    async def update_EMAs(self):
        for symbol in self.get_symbols_approved_or_has_pos():
            if symbol not in self.ohlcvs_1m or not self.ohlcvs_1m[symbol]:
                await self.update_ohlcvs_1m_single(symbol)
                sts = utc_ms()
                while symbol not in self.ohlcvs_1m:
                    await asyncio.sleep(0.2)
                    if utc_ms() - sts > 1000 * 5:
                        logging.error(f"timeout 5 secs waiting for ohlcvs_1m update for {symbol}")
                        break
            self.update_EMAs_single(symbol)

    def update_EMAs_single(self, symbol):
        try:
            if symbol not in self.ohlcvs_1m or not self.ohlcvs_1m[symbol]:
                return
            self.fill_gaps_ohlcvs_1m_single(symbol)
            if symbol not in self.emas["long"]:
                self.init_EMAs_single(symbol)
            last_ts, last_ohlcv_1m = self.ohlcvs_1m[symbol].peekitem(-1)
            mn = 60000
            for ts in range(self.upd_minute_emas[symbol] + mn, last_ts + mn, mn):
                for pside in ["long", "short"]:
                    self.emas[pside][symbol] = calc_ema(
                        self.ema_alphas[pside][symbol][0],
                        self.ema_alphas[pside][symbol][1],
                        self.emas[pside][symbol],
                        self.ohlcvs_1m[symbol][ts][4],
                    )
            self.upd_minute_emas[symbol] = last_ts
            return True
        except Exception as e:
            logging.error(f"error with {get_function_name()} for {symbol}: {e}")
            traceback.print_exc()
            return False

    def get_symbols_with_pos(self, pside=None):
        # returns symbols that have position
        if pside is None:
            return self.get_symbols_with_pos("long") | self.get_symbols_with_pos("short")
        return set([s for s in self.positions if self.positions[s][pside]["size"] != 0.0])

    def get_symbols_approved_or_has_pos(self, pside=None) -> set:
        if pside is None:
            return self.get_symbols_approved_or_has_pos(
                "long"
            ) | self.get_symbols_approved_or_has_pos("short")
        return (
            self.approved_coins_minus_ignored_coins[pside]
            | self.get_symbols_with_pos(pside)
            | {s for s in self.flagged_modes[pside] if self.flagged_modes[pside][s] == "normal"}
        )

    def get_ohlcvs_1m_file_mods(self, symbols=None):
        if symbols is None:
            symbols = self.get_symbols_approved_or_has_pos()
        last_update_tss = []
        for symbol in symbols:
            try:
                filepath = self.get_ohlcvs_1m_filepath(symbol)
                if os.path.exists(filepath):
                    last_update_tss.append((get_file_mod_utc(filepath), symbol))
                else:
                    last_update_tss.append((0.0, symbol))
            except Exception as e:
                logging.info(f"debug error with get_file_mod_utc for {symbol} {e}")
                last_update_tss.append((0.0, symbol))
        return last_update_tss

    async def restart_bot(self):
        logging.info("Initiating bot restart...")
        self.stop_signal_received = True
        self.stop_data_maintainers()
        await self.cca.close()
        await self.ccp.close()
        raise Exception("Bot will restart.")

    async def update_ohlcvs_1m_for_actives(self):
        try:
            utc_now = utc_ms()
            symbols_to_update = [
                s
                for s in self.active_symbols
                if s not in self.ohlcvs_1m_update_timestamps_WS
                or utc_now - self.ohlcvs_1m_update_timestamps_WS[s] > 1000 * 60
            ]
            if symbols_to_update:
                await asyncio.gather(*[self.update_ohlcvs_1m_single(s) for s in symbols_to_update])
        except Exception as e:
            logging.error(f"error with {get_function_name()} {e}")
            traceback.print_exc()

    async def maintain_hourly_cycle(self):
        logging.info(f"Starting hourly_cycle...")
        while not self.stop_signal_received:
            try:
                # update markets dict once every hour
                if utc_ms() - self.init_markets_last_update_ms > 1000 * 60 * 60:
                    await self.init_markets(verbose=False)
                await asyncio.sleep(1)
            except Exception as e:
                logging.error(f"error with {get_function_name()} {e}")
                traceback.print_exc()
                await asyncio.sleep(5)

    async def start_data_maintainers(self):
        # maintains REST hourly_cycle and ohlcv_1m
        if hasattr(self, "maintainers"):
            self.stop_data_maintainers()
        self.maintainers = {
            k: asyncio.create_task(getattr(self, k)())
            for k in [
                "maintain_hourly_cycle",
                "maintain_ohlcvs_1m_REST",
                "watch_ohlcvs_1m",
                "watch_orders",
            ]
        }

    async def watch_ohlcvs_1m(self):
        if not hasattr(self, "ohlcvs_1m"):
            self.ohlcvs_1m = {}
        self.WS_ohlcvs_1m_tasks = {}
        while not self.stop_websocket:
            current_symbols = set(self.active_symbols)
            started_symbols = set(self.WS_ohlcvs_1m_tasks.keys())
            for key in self.WS_ohlcvs_1m_tasks:
                if self.WS_ohlcvs_1m_tasks[key].cancelled():
                    logging.info(
                        f"debug ohlcv_1m watcher task is cancelled {key} {self.WS_ohlcvs_1m_tasks[key]}"
                    )
                if self.WS_ohlcvs_1m_tasks[key].done():
                    logging.info(
                        f"debug ohlcv_1m watcher task is done {key} {self.WS_ohlcvs_1m_tasks[key]}"
                    )
                try:
                    ex = self.WS_ohlcvs_1m_tasks[key].exception()
                    logging.info(
                        f"debug ohlcv_1m watcher task exception {key} {self.WS_ohlcvs_1m_tasks[key]} {ex}"
                    )
                except:
                    pass
            to_print = []
            # Start watch_ohlcv_1m_single tasks for new symbols
            for symbol in current_symbols - started_symbols:
                task = asyncio.create_task(self.watch_ohlcv_1m_single(symbol))
                self.WS_ohlcvs_1m_tasks[symbol] = task
                to_print.append(symbol)
            if to_print:
                coins = [symbol_to_coin(s) for s in to_print]
                logging.info(f"Started watching ohlcv_1m for {','.join(coins)}")
            to_print = []
            # Cancel tasks for symbols that are no longer active
            for symbol in started_symbols - current_symbols:
                self.WS_ohlcvs_1m_tasks[symbol].cancel()
                del self.WS_ohlcvs_1m_tasks[symbol]
                to_print.append(symbol)
            if to_print:
                coins = [symbol_to_coin(s) for s in to_print]
                logging.info(f"Stopped watching ohlcv_1m for: {','.join(coins)}")
            # Wait a bit before checking again
            await asyncio.sleep(1)  # Adjust sleep time as needed

    async def watch_ohlcv_1m_single(self, symbol):
        while not self.stop_websocket:
            try:
                res = await self.ccp.watch_ohlcv(symbol)
                self.handle_ohlcv_1m_update(symbol, res)
            except Exception as e:
                logging.error(f"exception watch_ohlcv_1m_single {symbol} {e}")
                traceback.print_exc()
                await asyncio.sleep(1)
            await asyncio.sleep(0.1)

    def calc_noisiness(self, pside, eligible_symbols=None):
        if eligible_symbols is None:
            eligible_symbols = self.eligible_symbols
        noisiness = {}
        n = int(round(self.config["bot"][pside]["filter_rolling_window"]))
        for symbol in eligible_symbols:
            if symbol in self.ohlcvs_1m and self.ohlcvs_1m[symbol]:
                ohlcvs_1m = [v for v in self.ohlcvs_1m[symbol].values()[-n:]]
                noisiness[symbol] = np.mean([(x[2] - x[3]) / x[4] for x in ohlcvs_1m])
            else:
                noisiness[symbol] = 0.0
        return noisiness

    def calc_volumes(self, pside, symbols=None):
        n = int(round(self.config["bot"][pside]["filter_rolling_window"]))
        volumes = {}
        if symbols is None:
            symbols = self.get_symbols_approved_or_has_pos(pside)
        for symbol in symbols:
            if (
                symbol in self.ohlcvs_1m
                and self.ohlcvs_1m[symbol]
                and len(self.ohlcvs_1m[symbol]) > 0
            ):
                ohlcvs_1m = [v for v in self.ohlcvs_1m[symbol].values()[-n:]]
                volumes[symbol] = sum([x[4] * x[5] for x in ohlcvs_1m])
            else:
                volumes[symbol] = 0.0
        return volumes

    async def execute_multiple(self, orders: [dict], type_: str, max_n_executions: int):
        if not orders:
            return []
        executions = []
        any_exceptions = False
        for order in orders[:max_n_executions]:  # sorted by PA dist
            execution = None
            try:
                execution = asyncio.create_task(getattr(self, type_)(order))
                executions.append((order, execution))
            except Exception as e:
                logging.error(f"error executing {type_} {order} {e}")
                print_async_exception(execution)
                traceback.print_exc()
                any_exceptions = True
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
                any_exceptions = True
        if any_exceptions:
            await self.restart_bot_on_too_many_errors()
        return results

    async def maintain_ohlcvs_1m_REST(self):
        if not hasattr(self, "ohlcvs_1m"):
            self.ohlcvs_1m = {}
        error_count = 0
        self.ohlcvs_1m_update_timestamps = {}
        symbol_approved_or_has_pos = self.get_symbols_approved_or_has_pos()
        self.n_symbols_missing_ohlcvs_1m = len(symbol_approved_or_has_pos)
        init_ohlcvs_sleep_time = 4.0 / self.n_symbols_missing_ohlcvs_1m
        for symbol in symbol_approved_or_has_pos:
            # print("debug update_ohlcvs_1m_single_from_disk", symbol)
            asyncio.create_task(self.update_ohlcvs_1m_single_from_disk(symbol))
            await asyncio.sleep(min(0.1, init_ohlcvs_sleep_time))
        self.ohlcvs_1m_max_age_ms = 1000 * 60 * self.ohlcvs_1m_update_after_minutes
        loop_sleep_time_ms = 1000 * 1
        logging.info(f"starting {get_function_name()}")
        while not self.stop_signal_received:
            try:
                symbols_too_old = []
                max_age_ms = self.ohlcvs_1m_max_age_ms
                now_utc = utc_ms()
                for symbol in self.get_symbols_approved_or_has_pos():
                    if symbol not in self.ohlcvs_1m_update_timestamps:
                        symbols_too_old.append((0, symbol))
                    else:
                        if (
                            now_utc - self.ohlcvs_1m_update_timestamps[symbol]
                            > self.ohlcvs_1m_max_age_ms
                        ):
                            symbols_too_old.append((self.ohlcvs_1m_update_timestamps[symbol], symbol))
                symbols_to_update = sorted(symbols_too_old)[: self.max_n_concurrent_ohlcvs_1m_updates]
                self.n_symbols_missing_ohlcvs_1m = len(symbols_too_old)
                if not symbols_to_update:
                    max_age_ms = self.ohlcvs_1m_max_age_ms / 2.0
                    if self.ohlcvs_1m_update_timestamps:
                        symbol, ts = sorted(
                            self.ohlcvs_1m_update_timestamps.items(), key=lambda x: x[1]
                        )[0]
                        symbols_to_update = [(ts, symbol)]
                if symbols_to_update:
                    await asyncio.gather(
                        *[
                            self.update_ohlcvs_1m_single(x[1], max_age_ms=max_age_ms)
                            for x in symbols_to_update
                        ]
                    )
                sleep_time_ms = loop_sleep_time_ms - (utc_ms() - now_utc)
                await asyncio.sleep(max(0.0, sleep_time_ms / 1000.0))
            except Exception as e:
                logging.error(f"error with {get_function_name()} {e}")
                traceback.print_exc()
                await asyncio.sleep(5)
                await self.restart_bot_on_too_many_errors()

    async def update_ohlcvs_1m_single_from_exchange(self, symbol):
        filepath = self.get_ohlcvs_1m_filepath(symbol)
        if self.lock_exists(filepath):
            return
        try:
            self.create_lock_file(filepath)
            ms_to_min = 1000 * 60
            if symbol in self.ohlcvs_1m and self.ohlcvs_1m[symbol]:
                last_ts = self.ohlcvs_1m[symbol].peekitem(-1)[0]
                now_minute = self.get_exchange_time() // ms_to_min * ms_to_min
                limit = min(999, max(3, int(round((now_minute - last_ts) / ms_to_min)) + 5))
                if limit >= 999:
                    limit = None
            else:
                self.ohlcvs_1m[symbol] = SortedDict()
                limit = None
            candles = await self.fetch_ohlcvs_1m(symbol, limit=limit)
            for x in candles:
                self.ohlcvs_1m[symbol][int(x[0])] = x
            self.dump_ohlcvs_1m_to_cache(symbol)
            self.ohlcvs_1m_update_timestamps[symbol] = or_default(
                get_file_mod_utc, filepath, default=0.0
            )
        finally:
            self.remove_lock_file(filepath)

    async def update_ohlcvs_1m_single_from_disk(self, symbol):
        filepath = self.get_ohlcvs_1m_filepath(symbol)
        if not os.path.exists(filepath):
            return
        if self.lock_exists(filepath):
            return
        try:
            self.create_lock_file(filepath)
            if symbol not in self.ohlcvs_1m:
                self.ohlcvs_1m[symbol] = SortedDict()
            data = np.load(filepath)
            for x in data:
                self.ohlcvs_1m[symbol][int(x[0])] = x
            self.ohlcvs_1m_update_timestamps[symbol] = or_default(
                get_file_mod_utc, filepath, default=0.0
            )
        except Exception as e:
            logging.error(f"error with update_ohlcvs_1m_single_from_disk {symbol} {e}")
            traceback.print_exc()
            try:
                os.remove(filepath)
            except Exception as e0:
                logging.error(f"failed to remove corrupted ohlcvs_1m file for {symbol} {e0}")
        finally:
            self.remove_lock_file(filepath)

    async def update_ohlcvs_1m_single(self, symbol, max_age_ms=None):
        if max_age_ms is None:
            max_age_ms = self.ohlcvs_1m_max_age_ms
        self.lock_timeout_ms = 5000.0
        try:
            if not (symbol in self.active_symbols or symbol in self.eligible_symbols):
                return
            filepath = self.get_ohlcvs_1m_filepath(symbol)
            if self.lock_exists(filepath):
                # is being updated by other instance
                if self.get_lock_age_ms(filepath) > self.lock_timeout_ms:
                    # other instance took too long to finish; assume it crashed
                    self.remove_lock_file(filepath)
                    await self.update_ohlcvs_1m_single_from_exchange(symbol)
            elif os.path.exists(filepath):
                mod_ts = or_default(get_file_mod_utc, filepath, default=0.0)
                if utc_ms() - mod_ts > max_age_ms:
                    await self.update_ohlcvs_1m_single_from_exchange(symbol)
                else:
                    if (
                        symbol not in self.ohlcvs_1m_update_timestamps
                        or self.ohlcvs_1m_update_timestamps[symbol] != mod_ts
                    ):
                        # was updated by other instance
                        await self.update_ohlcvs_1m_single_from_disk(symbol)
            else:
                await self.update_ohlcvs_1m_single_from_exchange(symbol)
        except Exception as e:
            logging.error(f"error with {get_function_name()} {e}")
            traceback.print_exc()
            await self.restart_bot_on_too_many_errors()

    def create_lock_file(self, filepath):
        try:
            open(f"{filepath}.lock", "w").close()
            return True
        except Exception as e:
            logging.error(f"error with {get_function_name()} {e}")
            traceback.print_exc()
            return False

    def lock_exists(self, filepath):
        try:
            return os.path.exists(f"{filepath}.lock")
        except Exception as e:
            logging.error(f"error with {get_function_name()} {e}")
            traceback.print_exc()
            return False

    def get_lock_age_ms(self, filepath):
        try:
            if self.lock_exists(filepath):
                return utc_ms() - get_file_mod_utc(f"{filepath}.lock")
        except Exception as e:
            logging.error(f"error with {get_function_name()} {e}")
            traceback.print_exc()
        return utc_ms()

    def remove_lock_file(self, filepath):
        try:
            if self.lock_exists(filepath):
                os.remove(f"{filepath}.lock")
                return True
        except Exception as e:
            logging.error(f"error with {get_function_name()} {e}")
            traceback.print_exc()
        return False

    async def close(self):
        logging.info(f"Stopped data maintainers: {self.stop_data_maintainers()}")
        await self.cca.close()
        await self.ccp.close()

    def add_to_coins_lists(self, content, k_coins):
        symbols = None
        psides_equal = content["long"] == content["short"]
        for pside in content:
            if not psides_equal or symbols is None:
                symbols = [self.coin_to_symbol(coin) for coin in content[pside]]
                symbols = set([s for s in symbols if s])
            symbols_already = getattr(self, k_coins)[pside]
            if symbols and symbols_already != symbols:
                added = symbols - symbols_already
                if added:
                    cstr = ",".join([symbol_to_coin(x) for x in sorted(added)])
                    logging.info(f"added {cstr} to {k_coins} {pside}")
                removed = symbols_already - symbols
                if removed:
                    cstr = ",".join([symbol_to_coin(x) for x in sorted(removed)])
                    logging.info(f"removed {cstr} from {k_coins} {pside}")
                getattr(self, k_coins)[pside] = symbols

    def refresh_approved_ignored_coins_lists(self):
        # if config.live.approved_coins or config.live.approved_coins are external files,
        # use content of files as approved/ignored coins
        # approved/ignored coins may be list of coins or {'long': list, 'short': list}
        for k_coins in ["approved_coins", "ignored_coins"]:
            if not hasattr(self, k_coins):
                setattr(self, k_coins, {"long": set(), "short": set()})
            path = self.config["live"][k_coins]
            if isinstance(path, list) and len(path) == 1 and isinstance(path[0], str):
                path = path[0]
            if isinstance(path, str):
                if os.path.exists(path):
                    try:
                        content = read_external_coins_lists(path)
                        if content:
                            self.add_to_coins_lists(content, k_coins)
                    except Exception as e:
                        logging.error(f"Failed to read contents of {path} {e}")
                elif self.coin_to_symbol(path):
                    self.add_to_coins_lists({"long": [path], "short": [path]}, k_coins)
                else:
                    logging.error(
                        f"error with refresh_approved_ignored_coins_lists: failed to load {path} {k_coins}"
                    )
            else:
                try:
                    if isinstance(path, (list, tuple)):
                        self.add_to_coins_lists({"long": path, "short": path}, k_coins)
                    elif isinstance(path, dict) and sorted(path) == ["long", "short"]:
                        self.add_to_coins_lists(path, k_coins)
                except Exception as e:
                    logging.error(f"Failed to read {k_coins} from config: {path}")
        self.approved_coins_minus_ignored_coins = {}
        for pside in self.approved_coins:
            if self.config["live"]["empty_means_all_approved"] and not self.approved_coins[pside]:
                # if approved_coins is empty, all coins are approved
                self.approved_coins[pside] = self.eligible_symbols
            self.approved_coins_minus_ignored_coins[pside] = (
                self.approved_coins[pside] - self.ignored_coins[pside]
            )


def setup_bot(config):
    # returns bot instance
    user_info = load_user_info(config["live"]["user"])
    if user_info["exchange"] == "bybit":
        from exchanges.bybit import BybitBot

        bot = BybitBot(config)
    elif user_info["exchange"] == "bitget":
        from exchanges.bitget import BitgetBot

        bot = BitgetBot(config)
    elif user_info["exchange"] == "binance":
        from exchanges.binance import BinanceBot

        bot = BinanceBot(config)
    elif user_info["exchange"] == "okx":
        from exchanges.okx import OKXBot

        bot = OKXBot(config)
    elif user_info["exchange"] == "hyperliquid":
        from exchanges.hyperliquid import HyperliquidBot

        bot = HyperliquidBot(config)
    elif user_info["exchange"] == "gateio":
        from exchanges.gateio import GateIOBot

        bot = GateIOBot(config)
    else:
        raise Exception(f"unknown exchange {user_info['exchange']}")
    return bot


async def shutdown_bot(bot):
    print("Shutting down bot...")
    bot.stop_data_maintainers()
    try:
        await asyncio.wait_for(bot.close(), timeout=3.0)
    except asyncio.TimeoutError:
        print("Shutdown timed out after 3 seconds. Forcing exit.")
    except Exception as e:
        print(f"Error during shutdown: {e}")


async def main():
    parser = argparse.ArgumentParser(prog="passivbot", description="run passivbot")
    parser.add_argument(
        "config_path", type=str, nargs="?", default=None, help="path to hjson passivbot config"
    )

    template_config = get_template_live_config("v7")
    del template_config["optimize"]
    del template_config["backtest"]
    add_arguments_recursively(parser, template_config)
    args = parser.parse_args()
    config = load_config(
        "configs/template.json" if args.config_path is None else args.config_path, live_only=True
    )
    update_config_with_args(config, args)
    config = format_config(config, live_only=True)
    cooldown_secs = 60
    restarts = []
    while True:

        bot = setup_bot(config)
        try:
            await bot.start_bot()
        except Exception as e:
            logging.error(f"passivbot error {e}")
            traceback.print_exc()
        finally:
            try:
                bot.stop_data_maintainers()
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
        if len(restarts) > bot.config["live"]["max_n_restarts_per_day"]:
            logging.info(
                f"n restarts exceeded {bot.config['live']['max_n_restarts_per_day']} last 24h"
            )
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot shutdown complete.")
