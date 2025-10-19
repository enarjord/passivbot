from __future__ import annotations
import os

# fix Crashes on Windows
from tools.event_loop_policy import set_windows_event_loop_policy

set_windows_event_loop_policy()

from ccxt.base.errors import NetworkError, RateLimitExceeded
import random
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
import math
from candlestick_manager import CandlestickManager
from typing import Dict, Iterable, Tuple, List, Optional
from utils import (
    load_markets,
    coin_to_symbol,
    symbol_to_coin,
    utc_ms,
    ts_to_date,
    make_get_filepath,
    format_approved_ignored_coins,
    filter_markets,
    normalize_exchange_name,
)
from prettytable import PrettyTable
from uuid import uuid4
from copy import deepcopy
from collections import defaultdict
from sortedcontainers import SortedDict

try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    import resource  # type: ignore
except Exception:
    resource = None
from config_utils import (
    load_config,
    add_arguments_recursively,
    update_config_with_args,
    format_config,
    normalize_coins_source,
    expand_PB_mode,
    get_template_config,
    parse_overrides,
    require_config_value,
    require_live_value,
    merge_negative_cli_values,
)
from procedures import (
    load_broker_code,
    load_user_info,
    get_first_timestamps_unified,
    print_async_exception,
)
from utils import get_file_mod_ms
from downloader import compute_per_coin_warmup_minutes
import re

from custom_endpoint_overrides import (
    apply_rest_overrides_to_ccxt,
    configure_custom_endpoint_loader,
    get_custom_endpoint_source,
    load_custom_endpoint_config,
    resolve_custom_endpoint_override,
)


calc_diff = pbr.calc_diff
calc_min_entry_qty = pbr.calc_min_entry_qty_py
round_ = pbr.round_
round_up = pbr.round_up
round_dn = pbr.round_dn
round_dynamic = pbr.round_dynamic

DEFAULT_MAX_MEMORY_CANDLES_PER_SYMBOL = 20_000

# Match "...0xABCD..." anywhere (case-insensitive)
_TYPE_MARKER_RE = re.compile(r"0x([0-9a-fA-F]{4})", re.IGNORECASE)
# Leading pure-hex fallback: optional 0x then 4 hex at the very start
_LEADING_HEX4_RE = re.compile(r"^(?:0x)?([0-9a-fA-F]{4})", re.IGNORECASE)


def _get_process_rss_bytes() -> Optional[int]:
    """Return current process RSS in bytes or None if unavailable."""
    try:
        if psutil is not None:
            return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:
        pass
    if resource is not None:
        try:
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform.startswith("linux"):
                usage = int(usage) * 1024
            else:
                usage = int(usage)
            return int(usage)
        except Exception:
            pass
    return None


def custom_id_to_snake(custom_id) -> str:
    """Translate a broker custom id into the snake_case order type name."""
    try:
        return snake_of(try_decode_type_id_from_custom_id(custom_id))
    except Exception as e:
        logging.error(f"failed to convert custom_id {custom_id} to str order_type")
        return "unknown"


def try_decode_type_id_from_custom_id(custom_id: str) -> int | None:
    """Extract the 16-bit order type id encoded in a custom order id string."""
    # 1) Preferred: look for "...0x<4-hex>..." anywhere
    m = _TYPE_MARKER_RE.search(custom_id)
    if m:
        return int(m.group(1), 16)

    # 2) Fallback: if string is pure-hex style (no broker code), parse the leading 4
    m = _LEADING_HEX4_RE.match(custom_id)
    if m:
        return int(m.group(1), 16)

    return None


def order_type_id_to_hex4(type_id: int) -> str:
    """Return the four-hex-digit representation of an order type id."""
    return f"{type_id:04x}"


def type_token(type_id: int, with_marker: bool = True) -> str:
    """Return the printable order type marker, optionally prefixed with `0x`."""
    h4 = order_type_id_to_hex4(type_id)
    return ("0x" + h4) if with_marker else h4


def snake_of(type_id: int) -> str:
    """Map an order type id to its snake_case string representation."""
    try:
        return pbr.order_type_id_to_snake(type_id)
    except Exception:
        return "unknown"


# Legacy EMA helper removed; CandlestickManager provides EMA utilities


def calc_pnl(position_side, entry_price, close_price, qty, inverse, c_mult):
    """Calculate trade PnL by delegating to the appropriate Rust helper."""
    try:
        if isinstance(position_side, str):
            if position_side == "long":
                return pbr.calc_pnl_long(entry_price, close_price, qty, c_mult)
            else:
                return pbr.calc_pnl_short(entry_price, close_price, qty, c_mult)
        else:
            # fallback: assume long
            return pbr.calc_pnl_long(entry_price, close_price, qty, c_mult)
    except Exception:
        # rethrow to preserve behavior
        raise


from pure_funcs import (
    numpyize,
    denumpyize,
    filter_orders,
    multi_replace,
    shorten_custom_id,
    determine_side_from_order_tuple,
    str2bool,
    add_missing_params_to_hjson_live_multi_config,
    flatten,
    log_dict_changes,
    ensure_millis,
)


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S",
)

ONE_MIN_MS = 60_000


def signal_handler(sig, frame):
    """Handle SIGINT by signalling the running bot to stop gracefully."""
    print("\nReceived shutdown signal. Stopping bot...")
    bot = globals().get("bot")
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None

    if bot is not None:
        bot.stop_signal_received = True
        if loop is not None:
            shutdown_task = getattr(bot, "_shutdown_task", None)
            if shutdown_task is None or shutdown_task.done():
                bot._shutdown_task = loop.create_task(bot.shutdown_gracefully())
            loop.call_soon_threadsafe(lambda: None)
    elif loop is not None:
        loop.call_soon_threadsafe(loop.stop)


signal.signal(signal.SIGINT, signal_handler)


def get_function_name():
    """Return the caller function name one frame above the current scope."""
    return inspect.currentframe().f_back.f_code.co_name


def get_caller_name():
    """Return the caller name two frames above the current scope."""
    return inspect.currentframe().f_back.f_back.f_code.co_name


def or_default(f, *args, default=None, **kwargs):
    """Execute `f` safely, returning `default` if an exception is raised."""
    try:
        return f(*args, **kwargs)
    except:
        return default


def orders_matching(o0, o1, tolerance_qty=0.01, tolerance_price=0.002):
    """Return True if two orders are equivalent within the supplied tolerances."""
    for k in ["symbol", "side", "position_side"]:
        if o0[k] != o1[k]:
            return False
    if tolerance_price:
        if abs(o0["price"] - o1["price"]) / o0["price"] > tolerance_price:
            return False
    else:
        if o0["price"] != o1["price"]:
            return False
    if tolerance_qty:
        if abs(o0["qty"] - o1["qty"]) / o0["qty"] > tolerance_qty:
            return False
    else:
        if o0["qty"] != o1["qty"]:
            return False
    return True


def order_has_match(order, orders, tolerance_qty=0.01, tolerance_price=0.002):
    """Return the first matching order in `orders` or False if none match."""
    for elm in orders:
        if orders_matching(order, elm, tolerance_qty, tolerance_price):
            return elm
    return False


class Passivbot:
    def __init__(self, config: dict):
        """Initialise the bot with configuration, user context, and runtime caches."""
        self.config = config
        self.user = require_live_value(config, "user")
        self.user_info = load_user_info(self.user)
        self.exchange = self.user_info["exchange"]
        self.broker_code = load_broker_code(self.user_info["exchange"])
        self.exchange_ccxt_id = normalize_exchange_name(self.exchange)
        self.endpoint_override = resolve_custom_endpoint_override(self.exchange_ccxt_id)
        self.ws_enabled = True
        if self.endpoint_override:
            self.ws_enabled = not self.endpoint_override.disable_ws
            source_path = get_custom_endpoint_source()
            logging.info(
                "Custom endpoint override active for %s (disable_ws=%s, source=%s)",
                self.exchange_ccxt_id,
                self.endpoint_override.disable_ws,
                source_path if source_path else "auto-discovered",
            )
        self.custom_id_max_length = 36
        self.sym_padding = 17
        self.action_str_max_len = max(
            len(a)
            for a in [
                "posting order",
                "cancelling order",
                "removed order",
                "added order",
            ]
        )
        self.stop_websocket = False
        self.balance = 1e-12
        self.hedge_mode = True
        self.inverse = False
        self.active_symbols = []
        self.fetched_positions = []
        self.fetched_open_orders = []
        self.open_orders = {}
        self.positions = {}
        self.symbol_ids = {}
        self.min_costs = {}
        self.min_qtys = {}
        self.qty_steps = {}
        self.price_steps = {}
        self.c_mults = {}
        self.max_leverage = {}
        self.pside_int_map = {"long": 0, "short": 1}
        self.PB_modes = {"long": {}, "short": {}}
        self.pnls_cache_filepath = make_get_filepath(f"caches/{self.exchange}/{self.user}_pnls.json")
        self.quote = "USDT"

        self.minimum_market_age_millis = (
            float(require_live_value(config, "minimum_coin_age_days")) * 24 * 60 * 60 * 1000
        )
        # Legacy EMA caches removed; use CandlestickManager EMA helpers
        # Legacy ohlcvs_1m fields removed in favor of CandlestickManager
        self.stop_signal_received = False
        self.cca = None
        self.ccp = None
        self.create_ccxt_sessions()
        self.debug_mode = False
        self.balance_threshold = 1.0  # don't create orders if balance is less than threshold
        self.hyst_rounding_balance_pct = 0.05
        self.hyst_rounding_balance_h = 0.75
        self.state_change_detected_by_symbol = set()
        self.recent_order_executions = []
        self.recent_order_cancellations = []
        self._disabled_psides_logged = set()
        # CandlestickManager settings from config.live
        cm_kwargs = {"exchange": self.cca, "debug": 0}
        mem_cap_raw = require_live_value(config, "max_memory_candles_per_symbol")
        mem_cap_effective = DEFAULT_MAX_MEMORY_CANDLES_PER_SYMBOL
        try:
            if mem_cap_raw is not None:
                mem_cap_effective = int(float(mem_cap_raw))
        except Exception:
            logging.warning(
                "Unable to parse live.max_memory_candles_per_symbol=%r, using default %d",
                mem_cap_raw,
                DEFAULT_MAX_MEMORY_CANDLES_PER_SYMBOL,
            )
            mem_cap_effective = DEFAULT_MAX_MEMORY_CANDLES_PER_SYMBOL
        if mem_cap_effective <= 0:
            logging.warning(
                "live.max_memory_candles_per_symbol=%r is non-positive; using default %d",
                mem_cap_raw,
                DEFAULT_MAX_MEMORY_CANDLES_PER_SYMBOL,
            )
            mem_cap_effective = DEFAULT_MAX_MEMORY_CANDLES_PER_SYMBOL
        cm_kwargs["max_memory_candles_per_symbol"] = mem_cap_effective
        disk_cap = require_live_value(config, "max_disk_candles_per_symbol_per_tf")
        if disk_cap is not None:
            cm_kwargs["max_disk_candles_per_symbol_per_tf"] = int(disk_cap)
        self.cm = CandlestickManager(**cm_kwargs)
        # TTL (minutes) for EMA candles on non-traded symbols
        ttl_min = require_live_value(config, "inactive_coin_candle_ttl_minutes")
        self.inactive_coin_candle_ttl_ms = int(float(ttl_min) * 60_000)
        auto_gs = bool(self.live_value("auto_gs"))
        self.PB_mode_stop = {
            "long": "graceful_stop" if auto_gs else "manual",
            "short": "graceful_stop" if auto_gs else "manual",
        }

    def live_value(self, key: str):
        return require_live_value(self.config, key)

    def bot_value(self, pside: str, key: str):
        return require_config_value(self.config, f"bot.{pside}.{key}")

    async def start_bot(self):
        """Initialise state, warm cached data, and launch background loops."""
        logging.info(f"Starting bot {self.exchange}...")
        await format_approved_ignored_coins(self.config, self.user_info["exchange"])
        await self.init_markets()
        # Staggered warmup of candles for approved symbols (large sets handled gracefully)
        try:
            await self.warmup_candles_staggered()
        except Exception as e:
            logging.info(f"warmup skipped due to: {e}")
        await asyncio.sleep(1)
        logging.info(f"Starting data maintainers...")
        await self.start_data_maintainers()

        logging.info(f"starting execution loop...")
        if not self.debug_mode:
            await self.run_execution_loop()

    async def init_markets(self, verbose=True):
        """Load exchange market metadata and refresh approval lists."""
        # called at bot startup and once an hour thereafter
        self.init_markets_last_update_ms = utc_ms()
        await self.update_exchange_config()  # set hedge mode
        self.markets_dict = await load_markets(self.exchange, 0, verbose=False)
        await self.determine_utc_offset(verbose)
        # ineligible symbols cannot open new positions
        eligible, _, reasons = filter_markets(self.markets_dict, self.exchange, verbose)
        self.eligible_symbols = set(eligible)
        self.ineligible_symbols = reasons
        self.set_market_specific_settings()
        # for prettier printing
        self.max_len_symbol = max([len(s) for s in self.markets_dict])
        self.sym_padding = max(self.sym_padding, self.max_len_symbol + 1)
        # await self.init_flags()
        self.init_coin_overrides()
        # await self.update_tickers()
        self.refresh_approved_ignored_coins_lists()
        # self.set_live_configs()
        self.set_wallet_exposure_limits()
        await self.update_positions()
        await self.update_open_orders()
        await self.update_effective_min_cost()
        # Legacy: no 1m OHLCV REST maintenance; CandlestickManager handles caching
        if self.is_forager_mode():
            await self.update_first_timestamps()

    def debug_print(self, *args):
        """Emit debug output only when the instance is in debug mode."""
        if hasattr(self, "debug_mode") and self.debug_mode:
            print(*args)

    def _log_memory_snapshot(self, *, now_ms: Optional[int] = None) -> None:
        """Log process RSS and key cache metrics for observability."""
        if now_ms is None:
            now_ms = utc_ms()
        rss = _get_process_rss_bytes()
        if rss is None:
            return
        cache_bytes = None
        cache_candles = None
        cache_symbols = None
        try:
            cache = getattr(self.cm, "_cache", {}) if hasattr(self, "cm") else {}
            cache_symbols = len(cache)
            cache_bytes = sum(
                int(getattr(arr, "nbytes", 0)) for arr in cache.values() if arr is not None
            )
            cache_candles = sum(int(arr.shape[0]) for arr in cache.values() if hasattr(arr, "shape"))
        except Exception:
            cache_bytes = None
        prev = getattr(self, "_mem_log_prev", None)
        pct_change = None
        if prev and prev.get("rss"):
            prev_rss = prev["rss"]
            if prev_rss:
                pct_change = 100.0 * (rss - prev_rss) / prev_rss
        parts = [f"Memory usage rss={rss / (1024 * 1024):.2f} MiB"]
        if pct_change is not None:
            parts.append(f"Δ={pct_change:+.2f}% vs previous snapshot")
        if cache_bytes is not None:
            cache_mib = cache_bytes / (1024 * 1024)
            cache_desc = f"cm_cache={cache_mib:.2f} MiB"
            if cache_candles is not None:
                detail = f"{cache_candles} candles"
                if cache_symbols is not None:
                    detail += f" across {cache_symbols} symbols"
                cache_desc += f" ({detail})"
            parts.append(cache_desc)
        logging.info("; ".join(parts))
        self._mem_log_prev = {"timestamp": now_ms, "rss": rss}
        if cache_bytes is not None:
            self._mem_log_prev["cm_cache_bytes"] = cache_bytes

    def init_coin_overrides(self):
        """Populate coin override map keyed by symbols for quick lookup."""
        self.coin_overrides = {
            s: v
            for k, v in self.config.get("coin_overrides", {}).items()
            if (s := self.coin_to_symbol(k))
        }

    def config_get(self, path: [str], symbol=None):
        """
        Retrieve a configuration value, preferring per-symbol overrides when provided.
        """
        if symbol and symbol in self.coin_overrides:
            d = self.coin_overrides[symbol]
            for p in path:
                if isinstance(d, dict) and p in d:
                    d = d[p]
                else:
                    d = None
                    break
            if d is not None:
                return d

        # fallback to global config
        d = self.config
        for p in path:
            if isinstance(d, dict) and p in d:
                d = d[p]
            else:
                raise KeyError(f"Key path {'.'.join(path)} not found in config or coin overrides.")
        return d

    def bp(self, pside, key, symbol=None):
        """
        condensed helper function (bp = bot param) for config_get(['bot', pside, key], symbol)
        """
        return self.config_get(["bot", pside, key], symbol)

    def maybe_log_ema_debug(
        self,
        ema_bounds_long: Dict[str, Tuple[float, float]],
        ema_bounds_short: Dict[str, Tuple[float, float]],
        entry_grid_log_ranges: Dict[str, Dict[str, float]],
    ) -> None:

        ema_debug_logging_enabled = False

        """Emit a throttled log of EMA inputs so toggling visibility stays simple."""
        if not ema_debug_logging_enabled:
            return
        self._ema_debug_log_interval_ms = 30_000
        self._last_ema_debug_log_ms = 0
        now = utc_ms()
        if now - getattr(self, "_last_ema_debug_log_ms", 0) < self._ema_debug_log_interval_ms:
            return
        self._last_ema_debug_log_ms = now

        def _safe_span(pside: str, key: str, symbol: str) -> Optional[int]:
            try:
                val = self.bp(pside, key, symbol)
                return int(val) if val is not None else None
            except Exception:
                return None

        logs: List[str] = []
        for pside, bounds in ("long", ema_bounds_long), ("short", ema_bounds_short):
            if not bounds:
                continue
            side_entries: List[str] = []
            for symbol, (lower, upper) in sorted(bounds.items()):
                span0 = _safe_span(pside, "ema_span_0", symbol)
                span1 = _safe_span(pside, "ema_span_1", symbol)
                grid_lr = (entry_grid_log_ranges or {}).get(pside, {}).get(symbol)
                parts = [f"{symbol}"]
                if span0 is not None or span1 is not None:
                    parts.append(
                        f"spans=({span0 if span0 is not None else '?'}"
                        f", {span1 if span1 is not None else '?'})"
                    )
                parts.append(f"lower={lower:.6g}")
                parts.append(f"upper={upper:.6g}")
                if grid_lr is not None:
                    parts.append(f"log_range_ema={grid_lr:.6g}")
                side_entries.append(" ".join(parts))
            if side_entries:
                logs.append(f"{pside} -> " + "; ".join(side_entries))

        if logs:
            logging.info("EMA debug | " + " | ".join(logs))

    async def warmup_candles_staggered(
        self, *, concurrency: int = 8, window_candles: int | None = None, ttl_ms: int = 300_000
    ) -> None:
        """Warm up recent candles for all approved symbols in a staggered way.

        - concurrency: max in-flight symbols
        - window_candles: number of 1m candles to warm; defaults to CandlestickManager.default_window_candles
        - ttl_ms: skip refresh if data newer than this TTL exists

        Logs a minimal countdown when warming >20 symbols.
        """
        # Build symbol set: union of approved (minus ignored) across both sides
        if not hasattr(self, "approved_coins_minus_ignored_coins"):
            return
        symbols = sorted(set().union(*self.approved_coins_minus_ignored_coins.values()))
        if not symbols:
            return
        n = len(symbols)
        now = utc_ms()
        end_final = (now // ONE_MIN_MS) * ONE_MIN_MS - ONE_MIN_MS
        # Determine window per symbol. For forager mode, use max EMA spans required for
        # volume/log-range across both sides; otherwise use provided/default window.
        default_win = int(getattr(self.cm, "default_window_candles", 120))
        warmup_map = {}
        try:
            warmup_map = compute_per_coin_warmup_minutes(self.config)
        except Exception:
            warmup_map = {}
        default_warm_minutes = warmup_map.get("__default__", default_win)
        if default_warm_minutes is None:
            default_warm_minutes = default_win
        is_forager = self.is_forager_mode()
        per_symbol_win: Dict[str, int] = {}
        for sym in symbols:
            if window_candles is not None:
                per_symbol_win[sym] = int(max(1, int(window_candles)))
                continue
            warm_minutes_val = warmup_map.get(sym, default_warm_minutes)
            warm_minutes = int(math.ceil(float(warm_minutes_val)))
            if warm_minutes <= 0:
                warm_minutes = 1
            if is_forager:
                # Filtering uses 1m log-range EMA spans; keep notation distinct from grid log ranges.
                try:
                    lv = int(round(self.bp("long", "filter_volume_ema_span", sym)))
                except Exception:
                    lv = default_win
                try:
                    ln = int(round(self.bp("long", "filter_log_range_ema_span", sym)))
                except Exception:
                    ln = default_win
                try:
                    sv = int(round(self.bp("short", "filter_volume_ema_span", sym)))
                except Exception:
                    sv = default_win
                try:
                    sn = int(round(self.bp("short", "filter_log_range_ema_span", sym)))
                except Exception:
                    sn = default_win
                per_symbol_win[sym] = max(1, lv, ln, sv, sn, warm_minutes)
            else:
                per_symbol_win[sym] = max(1, warm_minutes)

        sem = asyncio.Semaphore(max(1, int(concurrency)))
        completed = 0
        started_ms = utc_ms()
        last_log_ms = started_ms

        # Informative kickoff log
        if n > 0:
            wmins = [per_symbol_win[s] for s in symbols]
            wmin, wmax = (min(wmins), max(wmins)) if wmins else (default_win, default_win)
            logging.info(
                f"warmup starting: {n} symbols, concurrency={concurrency}, ttl={int(ttl_ms/1000)}s, window=[{wmin},{wmax}]m"
            )

        async def one(sym: str):
            nonlocal completed, last_log_ms
            async with sem:
                try:
                    win = int(per_symbol_win.get(sym, default_win))
                    start_ts = int(end_final - ONE_MIN_MS * max(1, win))
                    await self.cm.get_candles(
                        sym, start_ts=start_ts, end_ts=None, max_age_ms=ttl_ms, strict=False
                    )
                except Exception:
                    pass
                finally:
                    completed += 1
                    # Time-based throttle: log every ~2s or on completion
                    if n > 20:
                        now_ms = utc_ms()
                        if (completed == n) or (now_ms - last_log_ms >= 2000) or completed == 1:
                            elapsed_s = max(0.001, (now_ms - started_ms) / 1000.0)
                            rate = completed / elapsed_s
                            remaining = max(0, n - completed)
                            eta_s = int(remaining / max(1e-6, rate))
                            pct = int(100 * completed / n)
                            logging.info(
                                f"warmup candles: {completed}/{n} {pct}% elapsed={int(elapsed_s)}s eta~{eta_s}s"
                            )
                            last_log_ms = now_ms

        await asyncio.gather(*(one(s) for s in symbols))

        # Warm 1h candles for grid log-range EMAs
        hour_sem = asyncio.Semaphore(max(1, int(concurrency)))
        end_final_hour = (now // (60 * ONE_MIN_MS)) * (60 * ONE_MIN_MS) - 60 * ONE_MIN_MS

        async def warm_hour(sym: str):
            async with hour_sem:
                warm_minutes_val = warmup_map.get(sym, default_warm_minutes)
                warm_minutes = int(math.ceil(float(warm_minutes_val)))
                if warm_minutes <= 0:
                    return
                warm_hours = max(1, int(math.ceil(warm_minutes / 60.0)))
                start_ts = int(end_final_hour - warm_hours * 60 * ONE_MIN_MS)
                try:
                    await self.cm.get_candles(
                        sym,
                        start_ts=start_ts,
                        end_ts=None,
                        max_age_ms=ttl_ms,
                        timeframe="1h",
                        strict=False,
                    )
                except Exception:
                    pass

        await asyncio.gather(*(warm_hour(s) for s in symbols))

    async def update_first_timestamps(self, symbols=[]):
        """Fetch and cache first trade timestamps for the provided symbols."""
        if not hasattr(self, "first_timestamps"):
            self.first_timestamps = {}
        symbols = sorted(set(symbols + flatten(self.approved_coins_minus_ignored_coins.values())))
        if all([s in self.first_timestamps for s in symbols]):
            return
        first_timestamps = await get_first_timestamps_unified(symbols)
        self.first_timestamps.update(first_timestamps)
        for symbol in sorted(self.first_timestamps):
            symbolf = self.coin_to_symbol(symbol, verbose=False)
            if symbolf not in self.markets_dict:
                continue
            if symbolf not in self.first_timestamps:
                self.first_timestamps[symbolf] = self.first_timestamps[symbol]
        for symbol in symbols:
            if symbol not in self.first_timestamps:
                logging.info(f"warning: unable to get first timestamp for {symbol}. Setting to zero.")
                self.first_timestamps[symbol] = 0.0

    def get_first_timestamp(self, symbol):
        """Return the cached first tradable timestamp for `symbol`, populating defaults."""
        if symbol not in self.first_timestamps:
            logging.info(f"warning: {symbol} missing from first_timestamps. Setting to zero.")
            self.first_timestamps[symbol] = 0.0
        return self.first_timestamps[symbol]

    def coin_to_symbol(self, coin, verbose=True):
        """Map a coin identifier to the exchange-specific trading symbol."""
        if coin == "":
            return ""
        if not hasattr(self, "coin_to_symbol_map"):
            self.coin_to_symbol_map = {}
        if coin in self.coin_to_symbol_map:
            return self.coin_to_symbol_map[coin]
        coinf = symbol_to_coin(coin)
        if coinf in self.coin_to_symbol_map:
            self.coin_to_symbol_map[coin] = self.coin_to_symbol_map[coinf]
            return self.coin_to_symbol_map[coinf]
        result = coin_to_symbol(coin, self.exchange)
        self.coin_to_symbol_map[coin] = result
        return result

    def order_to_order_tuple(self, order):
        """Convert an order dictionary into a normalized tuple for comparisons."""
        return (
            order["symbol"],
            order["side"],
            order["position_side"],
            round(float(order["qty"]), 12),
            round(float(order["price"]), 12),
        )

    def has_open_unstuck_order(self) -> bool:
        """Return True if an unstuck order is currently live on the exchange."""
        for orders in getattr(self, "open_orders", {}).values():
            for order in orders or []:
                custom_id = order.get("custom_id") if isinstance(order, dict) else None
                if not custom_id:
                    continue
                type_id = try_decode_type_id_from_custom_id(custom_id)
                if type_id is None:
                    continue
                try:
                    order_type = snake_of(type_id)
                except Exception:
                    continue
                if order_type in {"close_unstuck_long", "close_unstuck_short"}:
                    return True
        return False

    async def run_execution_loop(self):
        """Main execution loop coordinating order generation and exchange interaction."""
        while not self.stop_signal_received:
            try:
                self.execution_scheduled = False
                self.state_change_detected_by_symbol = set()
                if not await self.update_pos_oos_pnls_ohlcvs():
                    await asyncio.sleep(0.5)
                    continue
                res = await self.execute_to_exchange()
                if self.debug_mode:
                    return res
                await asyncio.sleep(float(self.live_value("execution_delay_seconds")))
                sleep_duration = 30
                for i in range(sleep_duration * 10):
                    if self.execution_scheduled:
                        break
                    await asyncio.sleep(0.1)
            except Exception as e:
                logging.error(f"error with {get_function_name()} {e}")
                traceback.print_exc()
                await asyncio.sleep(1.0)

    async def shutdown_gracefully(self):
        if getattr(self, "_shutdown_in_progress", False):
            return
        self._shutdown_in_progress = True
        self.stop_signal_received = True
        logging.info("Shutdown requested; closing background tasks and sessions.")
        try:
            self.stop_data_maintainers(verbose=False)
        except Exception as e:
            logging.error(f"error stopping maintainers during shutdown {e}")
        await asyncio.sleep(0)
        try:
            if getattr(self, "ccp", None) is not None:
                await self.ccp.close()
        except Exception as e:
            logging.error(f"error closing private ccxt session {e}")
        try:
            if getattr(self, "cca", None) is not None:
                await self.cca.close()
        except Exception as e:
            logging.error(f"error closing public ccxt session {e}")
        logging.info("Shutdown cleanup complete.")

    async def update_pos_oos_pnls_ohlcvs(self) -> bool:
        """Refresh positions, open orders, realised PnL, and 1m candles."""
        if self.stop_signal_received:
            return False
        positions_ok = await self.update_positions()
        if not positions_ok:
            return False
        open_orders_ok, pnls_ok = await asyncio.gather(
            self.update_open_orders(),
            self.update_pnls(),
        )
        if not open_orders_ok or not pnls_ok:
            return False
        if self.stop_signal_received:
            return False
        await self.update_ohlcvs_1m_for_actives()
        return True

    def add_to_recent_order_cancellations(self, order):
        """Record a recently cancelled order to throttle repeated cancellations."""
        self.recent_order_cancellations.append({**order, **{"execution_timestamp": utc_ms()}})

    def order_was_recently_cancelled(self, order, max_age_ms=15_000) -> float:
        """Return remaining throttle delay if the order was cancelled within `max_age_ms`."""
        age_limit = utc_ms() - max_age_ms
        self.recent_order_cancellations = [
            x for x in self.recent_order_cancellations if x["execution_timestamp"] > age_limit
        ]
        if matching := order_has_match(
            order, self.recent_order_cancellations, tolerance_price=0.0, tolerance_qty=0.0
        ):
            return max(0.0, (matching["execution_timestamp"] + max_age_ms) - utc_ms())
        return 0.0

    def add_to_recent_order_executions(self, order):
        """Track newly created orders to limit duplicate submissions."""
        self.recent_order_executions.append({**order, **{"execution_timestamp": utc_ms()}})

    def order_was_recently_updated(self, order, max_age_ms=15_000) -> float:
        """Return throttle delay if the order was placed within `max_age_ms`."""
        age_limit = utc_ms() - max_age_ms
        self.recent_order_executions = [
            x for x in self.recent_order_executions if x["execution_timestamp"] > age_limit
        ]
        if matching := order_has_match(order, self.recent_order_executions):
            return max(0.0, (matching["execution_timestamp"] + max_age_ms) - utc_ms())
        return 0.0

    async def execute_to_exchange(self):
        """Run one execution cycle including config sync and order placement/cancellation."""
        await self.execution_cycle()
        # await self.update_EMAs()
        await self.update_exchange_configs()
        to_cancel, to_create = await self.calc_orders_to_cancel_and_create()

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
        if self.debug_mode:
            if to_cancel:
                print(f"would cancel {len(to_cancel)} order{'s' if len(to_cancel) > 1 else ''}")
        else:
            res = await self.execute_cancellations_parent(to_cancel)
        if self.debug_mode:
            if to_create:
                print(f"would create {len(to_create)} order{'s' if len(to_create) > 1 else ''}")
        elif self.balance < self.balance_threshold:
            logging.info(f"Balance too low: {self.balance} {self.quote}. Not creating any orders.")
        else:
            # to_create_mod = [x for x in to_create if not order_has_match(x, to_cancel)]
            to_create_mod = []
            for x in to_create:
                xf = f"{x['symbol']} {x['side']} {x['position_side']} {x['qty']} @ {x['price']}"
                if order_has_match(x, to_cancel):
                    logging.info(
                        f"matching order cancellation found; will be delayed until next cycle: {xf}"
                    )
                elif delay_time_ms := self.order_was_recently_updated(x):
                    logging.info(
                        f"matching recent order execution found; will be delayed for up to {delay_time_ms/1000:.1f} secs: {xf}"
                    )
                else:
                    to_create_mod.append(x)
            if self.state_change_detected_by_symbol:
                logging.info(
                    "state change during execution; skipping order creation"
                    f" for {self.state_change_detected_by_symbol} until next cycle"
                )
                to_create_mod = [
                    x
                    for x in to_create_mod
                    if x["symbol"] not in self.state_change_detected_by_symbol
                ]
            res = None
            try:
                res = await self.execute_orders_parent(to_create_mod)
            except Exception as e:
                logging.error(f"error executing orders {to_create_mod} {e}")
                print_async_exception(res)
                traceback.print_exc()
                await self.restart_bot_on_too_many_errors()
        if to_cancel or to_create:
            self.execution_scheduled = True
        if self.debug_mode:
            return to_cancel, to_create

    async def execute_orders_parent(self, orders: [dict]) -> [dict]:
        """Submit a batch of orders after throttling and bookkeeping."""
        orders = orders[: int(self.live_value("max_n_creations_per_batch"))]
        for order in orders:
            self.add_to_recent_order_executions(order)
            self.log_order_action(order, "posting order")
        res = await self.execute_orders(orders)
        if not res:
            return
        if len(orders) != len(res):
            print(
                f"debug unequal lengths execute_orders_parent: "
                f"{len(orders)} orders, {len(res)} executions",
                res,
            )
            return []
        to_return = []
        for ex, order in zip(res, orders):
            if not self.did_create_order(ex):
                print(f"debug did_create_order false {ex}")
                continue
            debug_prints = {}
            for key in order:
                if key not in ex:
                    debug_prints.setdefault("missing", []).append((key, order[key]))
                    ex[key] = order[key]
                elif ex[key] is None:
                    debug_prints.setdefault("is_none", []).append((key, order[key]))
                    ex[key] = order[key]
            if debug_prints and self.debug_mode:
                print("debug create_orders", debug_prints)
            to_return.append(ex)
        if to_return:
            for elm in to_return:
                self.add_new_order(elm, source="POST")
        return to_return

    async def execute_cancellations_parent(self, orders: [dict]) -> [dict]:
        """Submit a batch of cancellations, prioritising reduce-only orders."""
        max_cancellations = int(self.live_value("max_n_cancellations_per_batch"))
        if len(orders) > max_cancellations:
            # prioritize cancelling reduce-only orders
            try:
                reduce_only_orders = [
                    x for x in orders if x.get("reduce_only") or x.get("reduceOnly")
                ]
                rest = [x for x in orders if not x["reduce_only"]]
                orders = (reduce_only_orders + rest)[:max_cancellations]
            except Exception as e:
                logging.error(f"debug filter cancellations {e}")
                orders = orders[:max_cancellations]
        for order in orders:
            self.add_to_recent_order_cancellations(order)
            self.log_order_action(order, "cancelling order")
        res = await self.execute_cancellations(orders)
        to_return = []
        if len(orders) != len(res):
            self.execution_scheduled = True
            for od in orders:
                self.state_change_detected_by_symbol.add(od["symbol"])
            print(
                f"debug unequal lengths execute_cancellations_parent: "
                f"{len(orders)} orders, {len(res)} executions",
                res,
            )
            return []
        for ex, od in zip(res, orders):
            if not self.did_cancel_order(ex, od):
                self.state_change_detected_by_symbol.add(od["symbol"])
                print(f"debug did_cancel_order false {ex} {od}")
                continue
            debug_prints = {}
            for key in od:
                if key not in ex:
                    debug_prints.setdefault("missing", []).append((key, od[key]))
                    ex[key] = od[key]
                elif ex[key] is None:
                    debug_prints.setdefault("is_none", []).append((key, od[key]))
                    ex[key] = od[key]
            if debug_prints and self.debug_mode:
                print("debug cancel_orders", debug_prints)
            to_return.append(ex)
        if to_return:
            for elm in to_return:
                self.remove_order(elm, source="POST")
        return to_return

    def log_order_action(self, order, action, source="passivbot"):
        """Log a structured message describing an order action."""
        logging.info(
            f"{action: >{self.action_str_max_len}} {self.pad_sym(order['symbol'])} {order['side']} "
            f"{order['qty']} {order['position_side']} @ {order['price']} source: {source}"
        )

    def did_create_order(self, executed) -> bool:
        """Return True if the exchange acknowledged order creation."""
        try:
            return "id" in executed and executed["id"] is not None
        except:
            return False
        # further tests defined in child class

    def did_cancel_order(self, executed, order=None) -> bool:
        """Return True when the exchange response confirms cancellation."""
        if isinstance(executed, list) and len(executed) == 1:
            return self.did_cancel_order(executed[0], order)
        try:
            return "id" in executed and executed["id"] is not None
        except:
            return False
        # further tests defined in child class

    def is_forager_mode(self, pside=None):
        """Return True when the configuration allows forager grid deployment for the side."""
        if pside is None:
            return self.is_forager_mode("long") or self.is_forager_mode("short")
        if self.bot_value(pside, "total_wallet_exposure_limit") <= 0.0:
            return False
        if self.live_value(f"forced_mode_{pside}"):
            return False
        n_positions = self.get_max_n_positions(pside)
        if n_positions == 0:
            return False
        if n_positions >= len(self.approved_coins_minus_ignored_coins[pside]):
            return False
        return True

    def pad_sym(self, symbol):
        """Return the symbol left-aligned to the configured log width."""
        return f"{symbol: <{self.sym_padding}}"

    def _apply_endpoint_override(self, client) -> None:
        """Apply configured REST endpoint overrides to a ccxt client."""
        if client is None:
            return
        apply_rest_overrides_to_ccxt(client, self.endpoint_override)

    def stop_data_maintainers(self, verbose=True):
        """Cancel background candle/orderbook tasks and log the outcome."""
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
        """Return True if the bot currently holds a position for the given side and symbol."""
        if pside is None:
            return self.has_position("long", symbol) or self.has_position("short", symbol)
        if symbol is None:
            return any([self.has_position(pside, s) for s in self.positions])
        return symbol in self.positions and self.positions[symbol][pside]["size"] != 0.0

    def is_trailing(self, symbol, pside=None):
        """Return True when trailing logic is active for the given symbol and side."""
        if pside is None:
            return self.is_trailing(symbol, "long") or self.is_trailing(symbol, "short")
        return (
            self.bp(pside, "entry_trailing_grid_ratio", symbol) != 0.0
            or self.bp(pside, "close_trailing_grid_ratio", symbol) != 0.0
        )

    def get_last_position_changes(self, symbol=None):
        """Return the most recent fill timestamp per symbol/side for trailing logic."""
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

    # Legacy: wait_for_ohlcvs_1m_to_update removed (CandlestickManager handles freshness)

    # Legacy: get_ohlcvs_1m_filepath removed

    # Legacy: trim_ohlcvs_1m removed

    # Legacy: dump_ohlcvs_1m_to_cache removed

    async def update_trailing_data(self) -> None:
        """Update trailing price metrics using CandlestickManager candles.

        For each symbol and side with a trailing position, iterate candles since the
        last position change and compute:
        - max_since_open: highest high since open
        - min_since_max: lowest low after the most recent new high
        - min_since_open: lowest low since open
        - max_since_min: highest high (or close per legacy) after the most recent new low
        Fetches per-symbol candles concurrently to reduce latency.
        """
        if not hasattr(self, "trailing_prices"):
            self.trailing_prices = {}
        last_position_changes = self.get_last_position_changes()
        symbols = set(self.trailing_prices) | set(last_position_changes) | set(self.active_symbols)

        # Initialize containers for all symbols first
        for symbol in symbols:
            # Initialize containers
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

        # Build concurrent fetches per symbol that has position changes
        fetch_plan = {}
        for symbol in symbols:
            if symbol not in last_position_changes:
                continue
            # Determine earliest start among sides to avoid duplicate fetches
            starts = [last_position_changes[symbol][ps] for ps in last_position_changes[symbol]]
            if not starts:
                continue
            start_ts = int(min(starts))
            fetch_plan[symbol] = start_ts

        tasks = {
            sym: asyncio.create_task(self.cm.get_candles(sym, start_ts=st, end_ts=None, strict=False))
            for sym, st in fetch_plan.items()
        }

        results = {}
        for sym, task in tasks.items():
            try:
                results[sym] = await task
            except Exception as e:
                logging.info(f"debug: failed to fetch candles for trailing {sym}: {e}")
                results[sym] = None

        # Compute trailing metrics per symbol/side
        for symbol, arr in results.items():
            if arr is None or arr.size == 0:
                continue
            arr = np.sort(arr, order="ts")
            for pside, changed_ts in last_position_changes[symbol].items():
                for row in arr:
                    ts = int(row["ts"])  # only after change
                    if ts <= int(changed_ts):
                        continue
                    high = float(row["h"]) if "h" in row.dtype.names else float("nan")
                    low = float(row["l"]) if "l" in row.dtype.names else float("nan")
                    close = float(row["c"]) if "c" in row.dtype.names else float("nan")
                    if high > self.trailing_prices[symbol][pside]["max_since_open"]:
                        self.trailing_prices[symbol][pside]["max_since_open"] = high
                        self.trailing_prices[symbol][pside]["min_since_max"] = close
                    else:
                        self.trailing_prices[symbol][pside]["min_since_max"] = min(
                            self.trailing_prices[symbol][pside]["min_since_max"], low
                        )
                    if low < self.trailing_prices[symbol][pside]["min_since_open"]:
                        self.trailing_prices[symbol][pside]["min_since_open"] = low
                        self.trailing_prices[symbol][pside]["max_since_min"] = close
                    else:
                        self.trailing_prices[symbol][pside]["max_since_min"] = max(
                            self.trailing_prices[symbol][pside]["max_since_min"], high
                        )

    def symbol_is_eligible(self, symbol):
        """Return True when the symbol passes exchange-specific eligibility rules."""
        return True

    def set_market_specific_settings(self):
        """Initialise per-symbol market metadata (steps, ids, multipliers)."""
        self.symbol_ids = {symbol: self.markets_dict[symbol]["id"] for symbol in self.markets_dict}
        self.symbol_ids_inv = {v: k for k, v in self.symbol_ids.items()}

    def get_symbol_id(self, symbol):
        """Return the exchange-native identifier for `symbol`, caching defaults."""
        try:
            return self.symbol_ids[symbol]
        except:
            logging.info(f"debug: symbol {symbol} missing from self.symbol_ids. Using {symbol}")
            self.symbol_ids[symbol] = symbol
            return symbol

    def get_symbol_id_inv(self, symbol):
        """Return the human-friendly symbol for an exchange-native identifier."""
        try:
            return self.symbol_ids_inv[symbol]
        except:
            logging.info(f"debug: symbol {symbol} missing from self.symbol_ids_inv. Using {symbol}")
            self.symbol_ids_inv[symbol] = symbol
            return symbol

    def is_approved(self, pside, symbol) -> bool:
        """Return True when a symbol is approved, not ignored, and old enough for trading."""
        if symbol not in self.approved_coins_minus_ignored_coins[pside]:
            return False
        if symbol in self.ignored_coins[pside]:
            return False
        if not self.is_old_enough(pside, symbol):
            return False
        return True

    async def update_exchange_configs(self):
        """Ensure exchange-specific settings are initialised for all active symbols."""
        if not hasattr(self, "already_updated_exchange_config_symbols"):
            self.already_updated_exchange_config_symbols = set()
        symbols_not_done = [
            x for x in self.active_symbols if x not in self.already_updated_exchange_config_symbols
        ]
        if symbols_not_done:
            await self.update_exchange_config_by_symbols(symbols_not_done)
            self.already_updated_exchange_config_symbols.update(symbols_not_done)

    async def update_exchange_config_by_symbols(self, symbols):
        """Exchange-specific hook to refresh config for the given symbols."""
        # defined by each exchange child class
        pass

    async def update_exchange_config(self):
        """Exchange-specific hook to refresh global config state."""
        # defined by each exchange child class
        pass

    def is_old_enough(self, pside, symbol):
        """Return True if the market age exceeds the configured minimum for forager mode."""
        if self.is_forager_mode(pside) and self.minimum_market_age_millis > 0:
            first_timestamp = self.get_first_timestamp(symbol)
            if first_timestamp:
                return utc_ms() - first_timestamp > self.minimum_market_age_millis
            else:
                return False
        else:
            return True

    async def update_tickers(self):
        """Fetch latest ticker data and fill in missing bid/ask/last values."""
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
        """Prepare bot state before talking to the exchange in an execution loop."""
        # called before every execution to exchange
        # assumes positions, open orders are up to date
        # determine coins with position and open orders
        # determine eligible/ineligible coins
        # determine approved/ignored coins
        #   from external ignored/approved coins files
        #   from coin age
        #   from effective min cost (only if has updated price info)
        # determine and set special t,p,m modes and forced modes
        # determine ideal coins from log range and volume
        # determine coins with pos for normal or gs modes
        # determine coins from ideal coins for normal modes

        await self.update_effective_min_cost()
        self.refresh_approved_ignored_coins_lists()
        self.set_wallet_exposure_limits()
        previous_PB_modes = deepcopy(self.PB_modes) if hasattr(self, "PB_modes") else None
        self.PB_modes = {"long": {}, "short": {}}
        for pside, other_pside in [("long", "short"), ("short", "long")]:
            if self.is_forager_mode(pside):
                await self.update_first_timestamps()
            for symbol in self.coin_overrides:
                if flag := self.get_forced_PB_mode(pside, symbol):
                    self.PB_modes[pside][symbol] = flag
            ideal_coins = await self.get_filtered_coins(pside)
            slots_filled = {
                k for k, v in self.PB_modes[pside].items() if v in ["normal", "graceful_stop"]
            }
            max_n_positions = self.get_max_n_positions(pside)
            symbols_with_pos = self.get_symbols_with_pos(pside)
            for symbol in symbols_with_pos:
                if symbol in self.PB_modes[pside]:
                    continue
                elif forced_mode := self.get_forced_PB_mode(pside, symbol):
                    self.PB_modes[pside][symbol] = forced_mode
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
        await self.update_trailing_data()
        res = log_dict_changes(previous_PB_modes, self.PB_modes)
        for k, v in res.items():
            for elm in v:
                logging.info(f"{k} {elm}")

    async def get_filtered_coins(self, pside: str) -> List[str]:
        """Select ideal coins for a side using EMA-based volume and log-range filters.

        Steps (for forager mode):
        - Filter by age and effective min cost
        - Rank by 1m EMA quote volume
        - Drop the lowest filter_volume_drop_pct fraction
        - Rank remaining by 1m EMA log range
        - Return up to n_positions most volatile symbols
        For non-forager mode, returns all approved candidates.
        """
        # filter coins by age
        # filter coins by min effective cost
        # filter coins by relative volume
        # filter coins by log range
        if self.get_forced_PB_mode(pside):
            return []
        candidates = self.approved_coins_minus_ignored_coins[pside]
        candidates = [s for s in candidates if self.is_old_enough(pside, s)]
        candidates = [s for s in candidates if self.effective_min_cost_is_low_enough(pside, s)]
        if candidates == []:
            self.warn_on_high_effective_min_cost(pside)
        if self.is_forager_mode(pside):
            # filter coins by relative volume and log range
            clip_pct = self.bot_value(pside, "filter_volume_drop_pct")
            max_n_positions = self.get_max_n_positions(pside)
            if clip_pct > 0.0:
                volumes = await self.calc_volumes(pside, symbols=candidates)
                # filter by relative volume
                n_eligible = round(len(volumes) * (1 - clip_pct))
                candidates = sorted(volumes, key=lambda x: volumes[x], reverse=True)
                candidates = candidates[: int(max(n_eligible, max_n_positions))]
            # ideal symbols are high log-range symbols
            log_ranges = await self.calc_log_range(pside, eligible_symbols=candidates)
            log_ranges = {
                k: v for k, v in sorted(log_ranges.items(), key=lambda x: x[1], reverse=True)
            }
            ideal_coins = [k for k in log_ranges.keys()][:max_n_positions]
        else:
            # all approved coins are selected, no filtering by volume and log range
            ideal_coins = sorted(candidates)
        return ideal_coins

    def warn_on_high_effective_min_cost(self, pside):
        """Log a warning if min effective cost filtering removes every candidate."""
        if not self.live_value("filter_by_min_effective_cost"):
            return
        if not self.is_pside_enabled(pside):
            return
        approved_coins_filtered = [
            x
            for x in self.approved_coins_minus_ignored_coins[pside]
            if self.effective_min_cost_is_low_enough(pside, x)
        ]
        if len(approved_coins_filtered) == 0:
            logging.info(
                f"Warning: No {pside} symbols are approved due to min effective cost too high. "
                + f"Suggestions: 1) increase account balance, 2) "
                + f"set 'filter_by_min_effective_cost' to false, 3) reduce n_{pside}s"
            )

    def get_max_n_positions(self, pside):
        """Return the configured maximum number of concurrent positions for a side."""
        max_n_positions = min(
            self.bot_value(pside, "n_positions"),
            len(self.approved_coins_minus_ignored_coins[pside]),
        )
        return max(0, int(round(max_n_positions)))

    def get_current_n_positions(self, pside):
        """Count open positions for the side, excluding inactive forced modes."""
        n_positions = 0
        for symbol in self.positions:
            if self.positions[symbol][pside]["size"] != 0.0:
                forced_mode = self.get_forced_PB_mode(pside, symbol)
                if forced_mode in ["normal", "graceful_stop"]:
                    n_positions += 1
                else:
                    n_positions += 1
        return n_positions

    def get_forced_PB_mode(self, pside, symbol=None):
        """Return an explicitly forced mode for the side or symbol, if configured."""
        mode = self.config_get(["live", f"forced_mode_{pside}"], symbol)
        if mode:
            return mode
        elif symbol and not self.markets_dict[symbol]["active"]:
            return "tp_only"
        return None

    def set_wallet_exposure_limits(self):
        """Recalculate wallet exposure limits for both sides and per-symbol overrides."""
        for pside in ["long", "short"]:
            self.config["bot"][pside]["wallet_exposure_limit"] = self.get_wallet_exposure_limit(pside)
            for symbol in self.coin_overrides:
                ov_conf = self.coin_overrides[symbol].get("bot", {}).get(pside, {})
                if "wallet_exposure_limit" in ov_conf:
                    self.coin_overrides[symbol]["bot"][pside]["wallet_exposure_limit"] = (
                        self.get_wallet_exposure_limit(pside, symbol)
                    )

    def get_wallet_exposure_limit(self, pside, symbol=None):
        """Return the wallet exposure limit for a side, honoring per-symbol overrides."""
        if symbol:
            fwel = (
                self.coin_overrides.get(symbol, {})
                .get("bot", {})
                .get(pside, {})
                .get("wallet_exposure_limit")
            )
            if fwel is not None:
                return fwel
        twel = self.bot_value(pside, "total_wallet_exposure_limit")
        if twel <= 0.0:
            return 0.0
        n_positions = max(self.get_max_n_positions(pside), self.get_current_n_positions(pside))
        if n_positions == 0:
            return 0.0
        return round(twel / n_positions, 8)

    def is_pside_enabled(self, pside):
        """Return True if trading is enabled for the given side in the current config."""
        return (
            self.bot_value(pside, "total_wallet_exposure_limit") > 0.0
            and self.bot_value(pside, "n_positions") > 0.0
        )

    def effective_min_cost_is_low_enough(self, pside, symbol):
        """Check whether the symbol meets the effective minimum cost requirement."""
        if not self.live_value("filter_by_min_effective_cost"):
            return True
        return (
            self.balance
            * self.get_wallet_exposure_limit(pside, symbol)
            * self.bp(pside, "entry_initial_qty_pct", symbol)
            >= self.effective_min_cost[symbol]
        )

    def add_new_order(self, order, source="WS"):
        """No-op placeholder; subclasses update open orders through REST synchronisation."""
        return  # only add new orders via REST in self.update_open_orders()

    def remove_order(self, order: dict, source="WS", reason="cancelled"):
        """No-op placeholder; subclasses remove open orders through REST synchronisation."""
        return  # only remove orders via REST in self.update_open_orders()

    def handle_order_update(self, upd_list):
        """Mark the execution loop dirty when websocket order updates arrive."""
        if upd_list:
            self.execution_scheduled = True
        return

    async def handle_balance_update(self, upd, source="WS"):
        """Process websocket balance updates and trigger execution if equity changes."""
        try:
            upd[self.quote]["total"] = round_dynamic(upd[self.quote]["total"], 10)
            equity = upd[self.quote]["total"] + (await self.calc_upnl_sum())
            if self.balance != upd[self.quote]["total"]:
                logging.info(
                    f"balance changed: {self.balance} -> {upd[self.quote]['total']} equity: {equity:.4f} source: {source}"
                )
                self.execution_scheduled = True
            self.balance = max(upd[self.quote]["total"], 1e-12)
        except Exception as e:
            logging.error(f"error updating balance from websocket {upd} {e}")
            traceback.print_exc()

    # Legacy: handle_ohlcv_1m_update removed

    async def calc_upnl_sum(self):
        """Compute unrealised PnL across fetched positions using latest prices."""
        upnl_sum = 0.0
        last_prices = await self.cm.get_last_prices(
            set([x["symbol"] for x in self.fetched_positions]), max_age_ms=60_000
        )
        for elm in self.fetched_positions:
            try:
                upnl = calc_pnl(
                    elm["position_side"],
                    elm["price"],
                    last_prices[elm["symbol"]],
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
        """Initialise historical PnL cache, loading from disk when available."""
        if not hasattr(self, "pnls"):
            self.pnls = []
        else:
            return  # pnls already initiated; abort
        logging.info(f"initiating pnls...")
        age_limit = self.get_exchange_time() - 1000 * 60 * 60 * 24 * float(
            self.live_value("pnls_max_lookback_days")
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
                    f"fetching missing pnls from before {ts_to_date(pnls_cache[0]['timestamp'])}"
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
            if pnls_cache:
                try:
                    json.dump(pnls_cache, open(self.pnls_cache_filepath, "w"))
                except Exception as e:
                    logging.error(f"error dumping pnls to {self.pnls_cache_filepath} {e}")
        self.pnls = pnls_cache

    async def update_pnls(self):
        """Fetch latest fills, update the PnL cache, and persist it when changed."""
        age_limit = self.get_exchange_time() - 1000 * 60 * 60 * 24 * float(
            self.live_value("pnls_max_lookback_days")
        )
        if self.stop_signal_received:
            return False
        await self.init_pnls()  # will do nothing if already initiated
        old_ids = {elm["id"] for elm in self.pnls}
        start_time = self.pnls[-1]["timestamp"] - 1000 if self.pnls else age_limit
        try:
            res = await self.fetch_pnls(start_time=start_time, limit=100)
        except RateLimitExceeded:
            logging.warning("rate limit while fetching pnls; retrying next cycle")
            return False
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
        return True

    def log_pnls_change(self, old_pnls, new_pnls):
        """Log differences between previous and new PnL entries for debugging."""
        keys = ["id", "timestamp", "symbol", "side", "position_side", "price", "qty"]
        old_pnls_compressed = {(x[k] for k in keys) for x in old_pnls}
        new_pnls_compressed = [(x[k] for k in keys) for x in new_pnls]
        added_pnls = [x for x in new_pnls_compressed if x not in old_pnls_compressed]

    async def update_open_orders(self):
        """Refresh open orders from the exchange and reconcile the local cache."""
        if not hasattr(self, "open_orders"):
            self.open_orders = {}
        if self.stop_signal_received:
            return False
        res = None
        try:
            res = await self.fetch_open_orders()
            if res in [None, False]:
                return False
            self.fetched_open_orders = res
            open_orders = res
            oo_ids_old = {elm["id"] for sublist in self.open_orders.values() for elm in sublist}
            oo_ids_new = {elm["id"] for elm in open_orders}
            added_orders = [oo for oo in open_orders if oo["id"] not in oo_ids_old]
            removed_orders = [
                oo
                for oo in [elm for sublist in self.open_orders.values() for elm in sublist]
                if oo["id"] not in oo_ids_new
            ]
            schedule_update_positions = False
            if len(removed_orders) > 20:
                logging.info(f"removed {len(removed_orders)} orders")
            else:
                for order in removed_orders:
                    if not self.order_was_recently_cancelled(order):
                        # means order is no longer in open orders, but wasn't cancelled by bot
                        # possible fill
                        # force another update_positions
                        schedule_update_positions = True
                        self.log_order_action(order, "missing order", "fetch_open_orders")
                    else:
                        self.log_order_action(order, "removed order", "fetch_open_orders")
            if len(added_orders) > 20:
                logging.info(f"added {len(added_orders)} new orders")
            else:
                for order in added_orders:
                    self.log_order_action(order, "added order", "fetch_open_orders")
            self.open_orders = {}
            for elm in open_orders:
                if elm["symbol"] not in self.open_orders:
                    self.open_orders[elm["symbol"]] = []
                self.open_orders[elm["symbol"]].append(elm)
            if schedule_update_positions:
                await asyncio.sleep(1.5)
                await self.update_positions()
            return True
        except RateLimitExceeded:
            logging.warning("rate limit while fetching open orders; retrying next cycle")
            return False
        except Exception as e:
            logging.error(f"error with {get_function_name()} {e}")
            print_async_exception(res)
            traceback.print_exc()
            return False

    async def determine_utc_offset(self, verbose=True):
        """Derive the exchange server time offset in milliseconds."""
        result = await self.cca.fetch_balance()
        self.utc_offset = round((result["timestamp"] - utc_ms()) / (1000 * 60 * 60)) * (
            1000 * 60 * 60
        )
        if verbose:
            logging.info(f"Exchange time offset is {self.utc_offset}ms compared to UTC")

    def get_exchange_time(self):
        """Return current exchange time in milliseconds."""
        return utc_ms() + self.utc_offset

    async def log_position_changes(self, positions_old, positions_new, rd=6):
        """Log position transitions for debugging when differences are detected."""
        psold = {
            (x["symbol"], x["position_side"]): {k: x[k] for k in ["size", "price"]}
            for x in positions_old
        }
        psnew = {
            (x["symbol"], x["position_side"]): {k: x[k] for k in ["size", "price"]}
            for x in positions_new
        }

        if psold == psnew:
            return  # No changes

        # Ensure both dicts have all keys
        for k in psnew:
            if k not in psold:
                psold[k] = {"size": 0.0, "price": 0.0}
        for k in psold:
            if k not in psnew:
                psnew[k] = {"size": 0.0, "price": 0.0}

        changed = []
        for k in psnew:
            if psold[k] != psnew[k]:
                changed.append(k)

        if not changed:
            return

        # Create PrettyTable for aligned output
        table = PrettyTable()
        table.border = False
        table.header = False
        table.padding_width = 0

        for symbol, pside in changed:
            old = psold[(symbol, pside)]
            new = psnew[(symbol, pside)]

            # classify action ------------------------------------------------
            if old["size"] == 0.0 and new["size"] != 0.0:
                action = "     new pos"
            elif new["size"] == 0.0:
                action = "  closed pos"
            elif new["size"] > old["size"]:
                action = "added to pos"
            elif new["size"] < old["size"]:
                action = " reduced pos"
            else:
                action = "     unknown"

            # Compute metrics for new pos
            wallet_exposure = (
                pbr.qty_to_cost(new["size"], new["price"], self.c_mults[symbol]) / self.balance
                if new["size"] != 0
                else 0.0
            )
            try:
                wel = self.bp(pside, "wallet_exposure_limit", symbol)
                WE_ratio = wallet_exposure / wel if wel > 0.0 else 0.0
            except:
                WE_ratio = 0.0

            last_price = await self.cm.get_current_close(symbol, max_age_ms=60_000)
            try:
                pprice_diff = (
                    pbr.calc_pprice_diff_int(self.pside_int_map[pside], new["price"], last_price)
                    if last_price
                    else 0.0
                )
            except:
                pprice_diff = 0.0

            try:
                upnl = (
                    calc_pnl(
                        pside,
                        new["price"],
                        last_price,
                        new["size"],
                        self.inverse,
                        self.c_mults[symbol],
                    )
                    if last_price
                    else 0.0
                )
            except:
                upnl = 0.0

            table.add_row(
                [
                    action + " ",
                    symbol + " ",
                    pside + " ",
                    round_dynamic(old["size"], rd),
                    " @ ",
                    round_dynamic(old["price"], rd),
                    " -> ",
                    round_dynamic(new["size"], rd),
                    " @ ",
                    round_dynamic(new["price"], rd),
                    " WE: ",
                    pbr.round_dynamic(wallet_exposure, 3),
                    " WE ratio: ",
                    round(WE_ratio, 3),
                    " PA dist: ",
                    round(pprice_diff, 4),
                    " upnl: ",
                    pbr.round_dynamic(upnl, 3),
                ]
            )

        # Print aligned table
        for line in table.get_string().splitlines():
            logging.info(line)

    async def update_positions(self):
        """Fetch positions, update balance, and reconcile local position state."""
        if not hasattr(self, "positions"):
            self.positions = {}
        try:
            res = await self.fetch_positions()
        except RateLimitExceeded:
            logging.warning("rate limit while fetching positions; retrying next cycle")
            return False
        except NetworkError as e:
            logging.error(f"network error fetching positions: {e}")
            return False
        if not res or all(x in [None, False] for x in res):
            return False
        positions_list_new, balance_new = res
        fetched_positions_old = deepcopy(self.fetched_positions)
        self.fetched_positions = positions_list_new
        await self.handle_balance_update({self.quote: {"total": balance_new}}, source="REST")
        try:
            await self.log_position_changes(fetched_positions_old, self.fetched_positions)
        except Exception as e:
            logging.error(f"error logging position changes {e}")
        positions_new = {
            sym: {
                "long": {"size": 0.0, "price": 0.0},
                "short": {"size": 0.0, "price": 0.0},
            }
            for sym in set(list(self.positions) + list(self.active_symbols))
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
        self.positions = positions_new
        return True

    async def update_effective_min_cost(self, symbol=None):
        """Update the effective minimum order cost for one or all symbols."""
        if not hasattr(self, "effective_min_cost"):
            self.effective_min_cost = {}
        if symbol is None:
            symbols = sorted(self.get_symbols_approved_or_has_pos())
        else:
            symbols = [symbol]
        last_prices = await self.cm.get_last_prices(symbols, max_age_ms=600_000)
        for symbol in symbols:
            try:
                self.effective_min_cost[symbol] = max(
                    pbr.qty_to_cost(
                        self.min_qtys[symbol],
                        last_prices[symbol],
                        self.c_mults[symbol],
                    ),
                    self.min_costs[symbol],
                )
            except Exception as e:
                logging.error(f"error with {get_function_name()} for {symbol}: {e}")
                traceback.print_exc()

    async def calc_ideal_orders(self, allow_unstuck: bool = True):
        """Compute desired entry and exit orders for every active symbol."""
        # find out which symbols need fresh data
        to_update_last_prices: set[str] = set()
        to_update_emas: dict[str, set[str]] = {"long": set(), "short": set()}
        # 1h grid-spacing log range requirements per side (symbol -> span in hours)
        to_update_grid_log_ranges: dict[str, dict[str, float]] = {"long": {}, "short": {}}

        for pside in self.PB_modes:
            for symbol in self.PB_modes[pside]:
                mode = self.PB_modes[pside][symbol]
                if mode == "panic":
                    if self.has_position(pside, symbol):
                        to_update_last_prices.add(symbol)
                elif mode in {"graceful_stop", "tp_only"} and not self.has_position(pside, symbol):
                    continue
                elif mode == "manual":
                    continue
                else:
                    to_update_emas[pside].add(symbol)
                    to_update_last_prices.add(symbol)
                    if self.bp(pside, "entry_grid_spacing_log_weight", symbol) != 0.0:
                        grid_log_span_hours = float(
                            self.bp(pside, "entry_grid_spacing_log_span_hours", symbol)
                        )
                        if grid_log_span_hours > 0.0:
                            to_update_grid_log_ranges[pside][symbol] = max(1e-6, grid_log_span_hours)

        def build_ema_items(symbols_set: set[str], pside: str) -> list[tuple[str, float, float]]:
            return [
                (
                    sym,
                    self.bp(pside, "ema_span_0", sym),
                    self.bp(pside, "ema_span_1", sym),
                )
                for sym in symbols_set
            ]

        def build_grid_log_range_items(pside: str) -> list[tuple[str, float]]:
            return list(to_update_grid_log_ranges[pside].items())

        entry_grid_log_ranges: dict[str, dict[str, float]] = {"long": {}, "short": {}}
        (
            last_prices,
            ema_bounds_long,
            ema_bounds_short,
            entry_grid_log_ranges["long"],
            entry_grid_log_ranges["short"],
        ) = await asyncio.gather(
            self.cm.get_last_prices(list(to_update_last_prices), max_age_ms=10_000),
            self.cm.get_ema_bounds_many(
                build_ema_items(to_update_emas["long"], "long"), max_age_ms=30_000
            ),
            self.cm.get_ema_bounds_many(
                build_ema_items(to_update_emas["short"], "short"), max_age_ms=30_000
            ),
            # Grid spacing reacts to 1h log-range EMAs
            self.cm.get_latest_ema_log_range_many(
                build_grid_log_range_items("long"), tf="1h", max_age_ms=600_000
            ),
            self.cm.get_latest_ema_log_range_many(
                build_grid_log_range_items("short"), tf="1h", max_age_ms=600_000
            ),
        )
        self.maybe_log_ema_debug(
            ema_bounds_long,
            ema_bounds_short,
            entry_grid_log_ranges,
        )
        # long entries take lower bound; short entries take upper bound
        ema_anchor = {
            "long": {s: ema_bounds_long[s][0] for s in ema_bounds_long},
            "short": {s: ema_bounds_short[s][1] for s in ema_bounds_short},
        }

        ideal_orders = defaultdict(list)
        for pside in self.PB_modes:
            for symbol in self.PB_modes[pside]:
                if self.PB_modes[pside][symbol] == "panic":
                    if self.has_position(pside, symbol):
                        # if in panic mode, only one close order at current market price
                        qmul = -1 if pside == "long" else 1
                        panic_order_type = f"close_panic_{pside}"
                        ideal_orders[symbol].append(
                            (
                                abs(self.positions[symbol][pside]["size"]) * qmul,
                                last_prices[symbol],
                                panic_order_type,
                                pbr.order_type_snake_to_id(panic_order_type),
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
                        self.bp(pside, "entry_grid_double_down_factor", symbol),
                        self.bp(pside, "entry_grid_spacing_log_weight", symbol),
                        self.bp(pside, "entry_grid_spacing_we_weight", symbol),
                        self.bp(pside, "entry_grid_spacing_pct", symbol),
                        self.bp(pside, "entry_initial_ema_dist", symbol),
                        self.bp(pside, "entry_initial_qty_pct", symbol),
                        self.bp(pside, "entry_trailing_double_down_factor", symbol),
                        self.bp(pside, "entry_trailing_grid_ratio", symbol),
                        self.bp(pside, "entry_trailing_retracement_pct", symbol),
                        self.bp(pside, "entry_trailing_threshold_pct", symbol),
                        self.bp(pside, "wallet_exposure_limit", symbol),
                        self.balance,
                        self.positions[symbol][pside]["size"],
                        self.positions[symbol][pside]["price"],
                        self.trailing_prices[symbol][pside]["min_since_open"],
                        self.trailing_prices[symbol][pside]["max_since_min"],
                        self.trailing_prices[symbol][pside]["max_since_open"],
                        self.trailing_prices[symbol][pside]["min_since_max"],
                        ema_anchor[pside].get(symbol, last_prices.get(symbol, float("nan"))),
                        entry_grid_log_ranges.get(pside, {}).get(symbol, 0.0),
                        last_prices[symbol],
                    )
                    closes = getattr(pbr, f"calc_closes_{pside}_py")(
                        self.qty_steps[symbol],
                        self.price_steps[symbol],
                        self.min_qtys[symbol],
                        self.min_costs[symbol],
                        self.c_mults[symbol],
                        self.bp(pside, "close_grid_markup_end", symbol),
                        self.bp(pside, "close_grid_markup_start", symbol),
                        self.bp(pside, "close_grid_qty_pct", symbol),
                        self.bp(pside, "close_trailing_grid_ratio", symbol),
                        self.bp(pside, "close_trailing_qty_pct", symbol),
                        self.bp(pside, "close_trailing_retracement_pct", symbol),
                        self.bp(pside, "close_trailing_threshold_pct", symbol),
                        bool(self.bp(pside, "enforce_exposure_limit", symbol)),
                        self.bp(pside, "wallet_exposure_limit", symbol),
                        self.balance,
                        self.positions[symbol][pside]["size"],
                        self.positions[symbol][pside]["price"],
                        self.trailing_prices[symbol][pside]["min_since_open"],
                        self.trailing_prices[symbol][pside]["max_since_min"],
                        self.trailing_prices[symbol][pside]["max_since_open"],
                        self.trailing_prices[symbol][pside]["min_since_max"],
                        last_prices[symbol],
                    )
                    ideal_orders[symbol] += [
                        (x[0], x[1], snake_of(x[2]), x[2]) for x in entries + closes
                    ]

        unstucking_symbol, unstucking_close = await self.calc_unstucking_close(
            allow_new_unstuck=allow_unstuck
        )
        if unstucking_close[0] != 0.0:
            ideal_orders[unstucking_symbol] = [
                x for x in ideal_orders[unstucking_symbol] if not "close" in x[2]
            ]
            ideal_orders[unstucking_symbol].append(unstucking_close)

        ideal_orders_f = {}
        for symbol in ideal_orders:
            ideal_orders_f[symbol] = []
            last_mprice = last_prices[symbol]
            with_mprice_diff = [(calc_diff(x[1], last_mprice), x) for x in ideal_orders[symbol]]
            seen = set()
            any_partial = any(["partial" in order[2] for _, order in with_mprice_diff])
            for mprice_diff, order in sorted(with_mprice_diff):
                position_side = "long" if "long" in order[2] else "short"
                if order[0] == 0.0:
                    continue
                if mprice_diff > float(self.live_value("price_distance_threshold")):
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
                order_side = determine_side_from_order_tuple(order)
                order_type = "limit"
                if self.live_value("market_orders_allowed") and (
                    ("grid" in order[2] and mprice_diff < 0.0001)
                    or ("trailing" in order[2] and mprice_diff < 0.001)
                    or ("auto_reduce" in order[2] and mprice_diff < 0.001)
                    or (order_side == "buy" and order[1] >= last_mprice)
                    or (order_side == "sell" and order[1] <= last_mprice)
                ):
                    order_type = "market"
                ideal_orders_f[symbol].append(
                    {
                        "symbol": symbol,
                        "side": order_side,
                        "position_side": position_side,
                        "qty": abs(order[0]),
                        "price": order[1],
                        "reduce_only": "close" in order[2],
                        "custom_id": self.format_custom_id_single(order[3]),
                        "type": order_type,
                    }
                )
                seen.add(seen_key)
        # ensure close qtys don't exceed pos sizes
        for symbol in ideal_orders_f:
            for i in range(len(ideal_orders_f[symbol])):
                order = ideal_orders_f[symbol][i]
                if order["reduce_only"]:
                    pos_size_abs = abs(
                        self.positions[order["symbol"]][order["position_side"]]["size"]
                    )
                    if abs(order["qty"]) > pos_size_abs:
                        logging.info(
                            f"debug: reduce only order size greater than pos size. Order: {order} Position: {self.positions[order['symbol']]}"
                        )
                        order["qty"] = pos_size_abs
        return ideal_orders_f

    # Legacy calc_ema_bound removed; pricing uses CandlestickManager EMA bounds

    async def calc_unstucking_close(self, allow_new_unstuck: bool = True) -> (float, float, str, int):
        """Optionally return an emergency close order for stuck positions."""
        if not allow_new_unstuck or len(self.pnls) == 0:
            return "", (
                0.0,
                0.0,
                "",
                pbr.get_order_id_type_from_string("empty"),
            )  # needs trade history to calc unstucking order
        stuck_positions = []
        pnls_cumsum = np.array([x["pnl"] for x in self.pnls]).cumsum()
        pnls_cumsum_max, pnls_cumsum_last = (pnls_cumsum.max(), pnls_cumsum[-1])
        unstuck_allowances = {}
        last_prices = await self.cm.get_last_prices(set(self.positions), max_age_ms=10_000)
        for pside in ["long", "short"]:
            unstuck_allowances[pside] = (
                pbr.calc_auto_unstuck_allowance(
                    self.balance,
                    self.bot_value(pside, "unstuck_loss_allowance_pct")
                    * self.bot_value(pside, "total_wallet_exposure_limit"),
                    pnls_cumsum_max,
                    pnls_cumsum_last,
                )
                if self.bot_value(pside, "unstuck_loss_allowance_pct") > 0.0
                else 0.0
            )
            if unstuck_allowances[pside] <= 0.0:
                continue
            for symbol in self.positions:
                if self.has_position(pside, symbol):
                    wallet_exposure = pbr.calc_wallet_exposure(
                        self.c_mults[symbol],
                        self.balance,
                        self.positions[symbol][pside]["size"],
                        self.positions[symbol][pside]["price"],
                    )
                    we_limit = self.bp(pside, "wallet_exposure_limit", symbol)
                    if we_limit == 0.0 or wallet_exposure / we_limit > self.bp(
                        pside, "unstuck_threshold", symbol
                    ):
                        # is stuck. Use CandlestickManager EMA bounds to calc target price
                        try:
                            span_0 = self.bp(pside, "ema_span_0", symbol)
                            span_1 = self.bp(pside, "ema_span_1", symbol)
                            lower, upper = await self.cm.get_ema_bounds(
                                symbol,
                                span_0,
                                span_1,
                                max_age_ms=30_000,
                            )
                            # Apply unstuck_ema_dist and rounding per side
                            ema_dist = float(self.bp(pside, "unstuck_ema_dist", symbol))
                            # For closes: long uses upper bound (round up), short uses lower bound (round down)
                            if pside == "long":
                                base = float(upper)
                                ema_price = pbr.round_up(
                                    base * (1.0 + ema_dist), self.price_steps[symbol]
                                )
                            else:
                                base = float(lower)
                                ema_price = pbr.round_dn(
                                    base * (1.0 - ema_dist), self.price_steps[symbol]
                                )
                        except Exception as e:
                            logging.info(
                                f"debug: failed ema bounds for unstucking {symbol} {pside}: {e}; skipping unstuck check for this symbol"
                            )
                            # Skip unstuck evaluation for this symbol if EMA bounds are unavailable
                            continue
                        if (pside == "long" and last_prices[symbol] >= ema_price) or (
                            pside == "short" and last_prices[symbol] <= ema_price
                        ):
                            # eligible for unstucking
                            pprice_diff = pbr.calc_pprice_diff_int(
                                self.pside_int_map[pside],
                                self.positions[symbol][pside]["price"],
                                (await self.cm.get_current_close(symbol, max_age_ms=10_000)),
                            )
                            stuck_positions.append((symbol, pside, pprice_diff, ema_price))
        if not stuck_positions:
            return "", (0.0, 0.0, "", pbr.get_order_id_type_from_string("empty"))
        stuck_positions.sort(key=lambda x: x[2])
        for symbol, pside, pprice_diff, ema_price in stuck_positions:
            close_price = last_prices[symbol]
            if pside == "long":
                min_entry_qty = calc_min_entry_qty(
                    close_price,
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
                                * self.bp(pside, "wallet_exposure_limit", symbol)
                                * self.bp(pside, "unstuck_close_pct", symbol),
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
                    return symbol, (
                        close_qty,
                        close_price,
                        "close_unstuck_long",
                        pbr.get_order_id_type_from_string("close_unstuck_long"),
                    )
            elif pside == "short":
                min_entry_qty = calc_min_entry_qty(
                    close_price,
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
                                * self.bp(pside, "wallet_exposure_limit", symbol)
                                * self.bp(pside, "unstuck_close_pct", symbol),
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
                    return symbol, (
                        close_qty,
                        close_price,
                        "close_unstuck_short",
                        pbr.get_order_id_type_from_string("close_unstuck_short"),
                    )
        return "", (0.0, 0.0, "", pbr.get_order_id_type_from_string("empty"))

    async def calc_orders_to_cancel_and_create(self):
        """Determine which existing orders to cancel and which new ones to place."""
        allow_new_unstuck = not self.has_open_unstuck_order()
        ideal_orders = await self.calc_ideal_orders(allow_unstuck=allow_new_unstuck)

        # Sanity check: ideal orders should contain at most one unstuck order
        unstuck_names = {"close_unstuck_long", "close_unstuck_short"}
        unstuck_ideal_count = 0
        for orders in ideal_orders.values():
            for order in orders:
                custom_id = order.get("custom_id", "")
                order_type_id = try_decode_type_id_from_custom_id(custom_id)
                if order_type_id is None:
                    continue
                if snake_of(order_type_id) in unstuck_names:
                    unstuck_ideal_count += 1
        if unstuck_ideal_count > 1:
            logging.warning(
                "ideal_orders contains %s unstuck orders; trimming to one", unstuck_ideal_count
            )
            # keep the first encountered order, drop the rest
            keep_one = True
            for symbol, orders in ideal_orders.items():
                new_orders = []
                for order in orders:
                    custom_id = order.get("custom_id", "")
                    order_type_id = try_decode_type_id_from_custom_id(custom_id)
                    order_type = snake_of(order_type_id) if order_type_id is not None else ""
                    if order_type in unstuck_names:
                        if keep_one:
                            new_orders.append(order)
                            keep_one = False
                        else:
                            continue
                    else:
                        new_orders.append(order)
                ideal_orders[symbol] = new_orders
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
                            "id": x.get("id"),
                            "custom_id": x.get("custom_id"),
                        }
                    )
                except Exception as e:
                    logging.error(f"error in calc_orders_to_cancel_and_create {e}")
                    traceback.print_exc()
                    print(x)
        keys = ("symbol", "side", "position_side", "qty", "price")
        to_cancel, to_create = [], []
        for symbol in actual_orders:
            # Some symbols may have no ideal orders for this cycle; treat as empty list
            ideal_list = ideal_orders.get(symbol, []) if isinstance(ideal_orders, dict) else []
            to_cancel_, to_create_ = filter_orders(actual_orders[symbol], ideal_list, keys)
            seen_unstuck = False
            filtered_to_create = []
            for order in to_create_:
                custom_id = order.get("custom_id", "")
                order_type_id = try_decode_type_id_from_custom_id(custom_id)
                order_type = snake_of(order_type_id) if order_type_id is not None else ""
                if order_type in unstuck_names:
                    if seen_unstuck:
                        continue
                    seen_unstuck = True
                filtered_to_create.append(order)
            to_create_ = filtered_to_create
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
        to_create_with_mprice_diff = []
        for x in to_create:
            try:
                to_create_with_mprice_diff.append(
                    (
                        calc_diff(
                            x["price"],
                            (await self.cm.get_current_close(x["symbol"], max_age_ms=10_000)),
                        ),
                        x,
                    )
                )
            except Exception as e:
                logging.info(f"debug: price missing sort to_create by mprice_diff {x} {e}")
                to_create_with_mprice_diff.append((0.0, x))
        to_create_with_mprice_diff.sort(key=lambda x: x[0])
        to_cancel_with_mprice_diff = []
        for x in to_cancel:
            try:
                to_cancel_with_mprice_diff.append(
                    (
                        calc_diff(
                            x["price"],
                            (await self.cm.get_current_close(x["symbol"], max_age_ms=10_000)),
                        ),
                        x,
                    )
                )
            except Exception as e:
                logging.info(f"debug: price missing sort to_cancel by mprice_diff {x} {e}")
                to_cancel_with_mprice_diff.append((0.0, x))
        to_cancel_with_mprice_diff.sort(key=lambda x: x[0])
        to_cancel = [x[1] for x in to_cancel_with_mprice_diff]
        to_create = [x[1] for x in to_create_with_mprice_diff]
        return to_cancel, to_create

    async def restart_bot_on_too_many_errors(self):
        """Restart the bot if the hourly execution error budget is exhausted."""
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

    def format_custom_id_single(self, order_type_id: int) -> str:
        """Build a custom id embedding the order type marker and a UUID suffix."""
        token = type_token(order_type_id, with_marker=True)  # "0xABCD"
        return (token + uuid4().hex)[: self.custom_id_max_length]

    def debug_dump_bot_state_to_disk(self):
        """Persist internal state snapshots to disk for debugging purposes."""
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

    # Legacy EMA maintenance (init_EMAs_single/update_EMAs) removed in favor of CandlestickManager

    def get_symbols_with_pos(self, pside=None):
        """Return the set of symbols with open positions for the given side."""
        if pside is None:
            return self.get_symbols_with_pos("long") | self.get_symbols_with_pos("short")
        return set([s for s in self.positions if self.positions[s][pside]["size"] != 0.0])

    def get_symbols_approved_or_has_pos(self, pside=None) -> set:
        """Return symbols that are approved for trading or currently have a position."""
        if pside is None:
            return self.get_symbols_approved_or_has_pos(
                "long"
            ) | self.get_symbols_approved_or_has_pos("short")
        return (
            self.approved_coins_minus_ignored_coins[pside]
            | self.get_symbols_with_pos(pside)
            | {s for s in self.coin_overrides if self.get_forced_PB_mode(pside, s) == "normal"}
        )

    # Legacy get_ohlcvs_1m_file_mods removed

    async def restart_bot(self):
        """Stop all tasks and raise to trigger an external bot restart."""
        logging.info("Initiating bot restart...")
        self.stop_signal_received = True
        self.stop_data_maintainers()
        await self.cca.close()
        if self.ccp is not None:
            await self.ccp.close()
        raise Exception("Bot will restart.")

    async def update_ohlcvs_1m_for_actives(self):
        """Ensure active symbols have fresh 1m candles in CandlestickManager (<=10s old).

        Uses CandlestickManager.get_candles with max_age_ms=10_000 so it refreshes
        only when its internal last refresh is older than the TTL. Fetches a small
        recent window ending at the latest finalized minute.
        """
        max_age_ms = 10_000
        try:
            now = utc_ms()
            end_ts = (now // ONE_MIN_MS) * ONE_MIN_MS - ONE_MIN_MS
            # Use manager default window if available, otherwise a reasonable fallback
            try:
                window = int(getattr(self.cm, "default_window_candles", 120))
            except Exception:
                window = 120
            start_ts = end_ts - ONE_MIN_MS * window

            tasks = [
                asyncio.create_task(
                    self.cm.get_candles(
                        s, start_ts=start_ts, end_ts=end_ts, max_age_ms=max_age_ms, strict=False
                    )
                )
                for s in self.active_symbols
            ]
            if tasks:
                await asyncio.gather(*tasks)
        except Exception as e:
            logging.error(f"error with {get_function_name()} {e}")
            traceback.print_exc()

    async def maintain_hourly_cycle(self):
        """Periodically refresh market metadata while the bot is running."""
        logging.info(f"Starting hourly_cycle...")
        while not self.stop_signal_received:
            try:
                now = utc_ms()
                mem_prev = getattr(self, "_mem_log_prev", None)
                last_mem_log_ts = None
                if isinstance(mem_prev, dict):
                    last_mem_log_ts = mem_prev.get("timestamp")
                if last_mem_log_ts is None or now - last_mem_log_ts >= 1000 * 60 * 60:
                    self._log_memory_snapshot(now_ms=now)
                # update markets dict once every hour
                if now - self.init_markets_last_update_ms > 1000 * 60 * 60:
                    await self.init_markets(verbose=False)
                await asyncio.sleep(1)
            except Exception as e:
                logging.error(f"error with {get_function_name()} {e}")
                traceback.print_exc()
                await asyncio.sleep(5)

    async def start_data_maintainers(self):
        """Spawn background tasks responsible for market metadata and order watching."""
        if hasattr(self, "maintainers"):
            self.stop_data_maintainers()
        maintainer_names = ["maintain_hourly_cycle"]
        if self.ws_enabled:
            maintainer_names.append("watch_orders")
        else:
            logging.info("Websocket maintainers skipped (ws disabled via custom endpoints).")
        self.maintainers = {
            name: asyncio.create_task(getattr(self, name)()) for name in maintainer_names
        }

    # Legacy websocket 1m ohlcv watchers removed; CandlestickManager is authoritative

    async def calc_log_range(
        self,
        pside: str,
        eligible_symbols: Optional[Iterable[str]] = None,
        *,
        max_age_ms: Optional[int] = 60_000,
    ) -> Dict[str, float]:
        """Compute 1m EMA of log range per symbol: EMA(ln(high/low)).

        Returns mapping symbol -> ema_log_range; non-finite/failed computations yield 0.0.
        """
        if eligible_symbols is None:
            eligible_symbols = self.eligible_symbols
        span = int(round(self.bot_value(pside, "filter_log_range_ema_span")))

        # Compute EMA of log range on 1m candles: ln(high/low)
        async def one(symbol: str):
            try:
                # If caller passes a TTL, use it; otherwise select per-symbol TTL
                if max_age_ms is not None:
                    ttl = int(max_age_ms)
                else:
                    # More generous TTL for non-traded symbols
                    has_pos = self.has_position(symbol)
                    has_oo = (
                        bool(self.open_orders.get(symbol)) if hasattr(self, "open_orders") else False
                    )
                    ttl = (
                        60_000
                        if (has_pos or has_oo)
                        else int(getattr(self, "inactive_coin_candle_ttl_ms", 600_000))
                    )
                val = await self.cm.get_latest_ema_log_range(
                    symbol, span=span, timeframe=None, max_age_ms=ttl
                )
                return float(val) if np.isfinite(val) else 0.0
            except Exception:
                return 0.0

        syms = list(eligible_symbols)
        tasks = {s: asyncio.create_task(one(s)) for s in syms}
        out = {}
        # Progress logging for large batches
        n = len(syms)
        completed = 0
        started_ms = utc_ms()
        last_log_ms = started_ms
        for sym in tasks:
            try:
                val = await tasks[sym]
            except Exception:
                val = 0.0
            if sym is not None:
                out[sym] = val
            completed += 1
            if n > 20:
                now_ms = utc_ms()
                elapsed_ms = now_ms - started_ms
                should_log = False
                if completed == n and elapsed_ms >= 5000:
                    should_log = True
                elif elapsed_ms >= 5000 and (now_ms - last_log_ms) >= 5000:
                    should_log = True
                if should_log:
                    elapsed_s = max(0.001, elapsed_ms / 1000.0)
                    rate = completed / elapsed_s
                    pct = int(100 * completed / n)
                    eta_s = int((n - completed) / max(1e-6, rate))
                    logging.info(
                        f"log_range EMA: {completed}/{n} {pct}% elapsed={int(elapsed_s)}s eta~{eta_s}s"
                    )
                    last_log_ms = now_ms
        return out

    async def calc_volumes(
        self,
        pside: str,
        symbols: Optional[Iterable[str]] = None,
        *,
        max_age_ms: Optional[int] = 60_000,
    ) -> Dict[str, float]:
        """Compute 1m EMA of quote volume per symbol.

        Returns mapping symbol -> ema_quote_volume; non-finite/failed computations yield 0.0.
        """
        span = int(round(self.bot_value(pside, "filter_volume_ema_span")))
        if symbols is None:
            symbols = self.get_symbols_approved_or_has_pos(pside)

        # Compute EMA of quote volume on 1m candles
        async def one(symbol: str):
            try:
                if max_age_ms is not None:
                    ttl = int(max_age_ms)
                else:
                    has_pos = self.has_position(symbol)
                    has_oo = (
                        bool(self.open_orders.get(symbol)) if hasattr(self, "open_orders") else False
                    )
                    ttl = (
                        60_000
                        if (has_pos or has_oo)
                        else int(getattr(self, "inactive_coin_candle_ttl_ms", 600_000))
                    )
                val = await self.cm.get_latest_ema_quote_volume(
                    symbol, span=span, timeframe=None, max_age_ms=ttl
                )
                return float(val) if np.isfinite(val) else 0.0
            except Exception:
                return 0.0

        syms = list(symbols)
        tasks = {s: asyncio.create_task(one(s)) for s in syms}
        out = {}
        # Progress logging for large batches
        n = len(syms)
        completed = 0
        started_ms = utc_ms()
        last_log_ms = started_ms
        for sym in tasks:
            try:
                val = await tasks[sym]
            except Exception:
                val = 0.0
            if sym is not None:
                out[sym] = val
            completed += 1
            if n > 20:
                now_ms = utc_ms()
                elapsed_ms = now_ms - started_ms
                should_log = False
                if completed == n and elapsed_ms >= 5000:
                    should_log = True
                elif elapsed_ms >= 5000 and (now_ms - last_log_ms) >= 5000:
                    should_log = True
                if should_log:
                    elapsed_s = max(0.001, elapsed_ms / 1000.0)
                    rate = completed / elapsed_s
                    pct = int(100 * completed / n)
                    eta_s = int((n - completed) / max(1e-6, rate))
                    logging.info(
                        f"volume EMA: {completed}/{n} {pct}% elapsed={int(elapsed_s)}s eta~{eta_s}s"
                    )
                    last_log_ms = now_ms
        return out

    async def execute_multiple(self, orders: [dict], type_: str):
        """Execute a list of order operations sequentially while tracking failures."""
        if not orders:
            return []
        executions = []
        any_exceptions = False
        for order in orders:  # sorted by PA dist
            task = None
            try:
                task = asyncio.create_task(getattr(self, type_)(order))
                executions.append((order, task))
            except Exception as e:
                logging.error(f"error executing {type_} {order} {e}")
                print_async_exception(task)
                traceback.print_exc()
                executions.append((order, e))
                any_exceptions = True
        results = []
        for order, execution in executions:
            if isinstance(execution, Exception):
                # Already failed at task creation time
                results.append(execution)
                continue
            result = None
            try:
                result = await execution
                results.append(result)
            except Exception as e:
                logging.error(f"error executing {type_} {execution} {e}")
                print_async_exception(result)
                results.append(e)
                traceback.print_exc()
                any_exceptions = True
        if any_exceptions:
            await self.restart_bot_on_too_many_errors()
        return results

    # Legacy maintain_ohlcvs_1m_REST removed; CandlestickManager handles caching and TTL

    # Legacy update_ohlcvs_1m_single_from_exchange removed

    # Legacy update_ohlcvs_1m_single_from_disk removed

    # Legacy update_ohlcvs_1m_single removed

    # Legacy file lock helpers removed

    async def close(self):
        """Stop background tasks and close exchange clients."""
        logging.info(f"Stopped data maintainers: {self.stop_data_maintainers()}")
        await self.cca.close()
        if self.ccp is not None:
            await self.ccp.close()

    def add_to_coins_lists(self, content, k_coins, log_psides=None):
        """Update approved/ignored coin sets from configuration content."""
        if log_psides is None:
            log_psides = set(content.keys())
        symbols = None
        psides_equal = content["long"] == content["short"]
        for pside in content:
            if not psides_equal or symbols is None:
                coins = content[pside]
                # Check if coins is a single string that needs to be split
                if isinstance(coins, str):
                    coins = coins.split(",")
                # Handle case where list contains comma-separated values in its elements
                elif isinstance(coins, (list, tuple)):
                    expanded_coins = []
                    for item in coins:
                        if isinstance(item, str) and "," in item:
                            expanded_coins.extend(item.split(","))
                        else:
                            expanded_coins.append(item)
                    coins = expanded_coins

                symbols = [self.coin_to_symbol(coin) for coin in coins if coin]
                symbols = set([s for s in symbols if s])
            symbols_already = getattr(self, k_coins)[pside]
            if symbols and symbols_already != symbols:
                added = symbols - symbols_already
                if added:
                    if pside in log_psides:
                        cstr = ",".join([symbol_to_coin(x) for x in sorted(added)])
                        logging.info(f"added {cstr} to {k_coins} {pside}")
                removed = symbols_already - symbols
                if removed:
                    if pside in log_psides:
                        cstr = ",".join([symbol_to_coin(x) for x in sorted(removed)])
                        logging.info(f"removed {cstr} from {k_coins} {pside}")
                getattr(self, k_coins)[pside] = symbols

    def refresh_approved_ignored_coins_lists(self):
        """Reload approved and ignored coin lists from config sources."""
        try:
            for k in ("approved_coins", "ignored_coins"):
                if not hasattr(self, k):
                    setattr(self, k, {"long": set(), "short": set()})
                parsed = normalize_coins_source(self.live_value(k))
                if k == "approved_coins":
                    log_psides = {ps for ps in parsed if self.is_pside_enabled(ps)}
                else:
                    log_psides = set(parsed.keys())
                self.add_to_coins_lists(parsed, k, log_psides=log_psides)
            self.approved_coins_minus_ignored_coins = {}
            for pside in self.approved_coins:
                if not self.is_pside_enabled(pside):
                    if pside not in self._disabled_psides_logged:
                        if self.approved_coins[pside]:
                            logging.info(
                                f"{pside} side disabled (zero exposure or positions); clearing approved list."
                            )
                        else:
                            logging.info(
                                f"{pside} side disabled (zero exposure or positions); approved list already empty."
                            )
                        self._disabled_psides_logged.add(pside)
                    self.approved_coins[pside] = set()
                    self.approved_coins_minus_ignored_coins[pside] = set()
                    continue
                else:
                    if pside in self._disabled_psides_logged:
                        logging.info(f"{pside} side re-enabled; restoring approved coin handling.")
                        self._disabled_psides_logged.discard(pside)
                if self.live_value("empty_means_all_approved") and not self.approved_coins[pside]:
                    # if approved_coins is empty, all coins are approved
                    self.approved_coins[pside] = self.eligible_symbols
                self.approved_coins_minus_ignored_coins[pside] = (
                    self.approved_coins[pside] - self.ignored_coins[pside]
                )
        except Exception as e:
            logging.error(f"error with refresh_approved_ignored_coins_lists {e}")
            traceback.print_exc()

    def get_order_execution_params(self, order: dict) -> dict:
        """Return exchange-specific parameters for order placement."""
        # defined for each exchange
        return {}

    async def execute_order(self, order: dict) -> dict:
        """Place a single order via the exchange client."""
        params = {
            "symbol": order["symbol"],
            "type": order.get("type", "limit"),
            "side": order["side"],
            "amount": abs(order["qty"]),
            "price": order["price"],
            "params": self.get_order_execution_params(order),
        }
        executed = await self.cca.create_order(**params)
        return executed

    async def execute_orders(self, orders: [dict]) -> [dict]:
        """Execute a batch of order creations using the helper pipeline."""
        return await self.execute_multiple(orders, "execute_order")

    async def execute_cancellation(self, order: dict) -> dict:
        """Cancel a single order via the exchange client."""
        executed = None
        try:
            executed = await self.cca.cancel_order(order["id"], symbol=order["symbol"])
            return executed
        except Exception as e:
            logging.error(f"error cancelling order {order} {e}")
            print_async_exception(executed)
            traceback.print_exc()
            return {}

    async def execute_cancellations(self, orders: [dict]) -> [dict]:
        """Execute a batch of cancellations using the helper pipeline."""
        return await self.execute_multiple(orders, "execute_cancellation")


def setup_bot(config):
    """Instantiate the correct exchange bot implementation based on configuration."""
    user_info = load_user_info(require_live_value(config, "user"))
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
    elif user_info["exchange"] == "defx":
        from exchanges.defx import DefxBot

        bot = DefxBot(config)
    elif user_info["exchange"] == "kucoin":
        from exchanges.kucoin import KucoinBot

        bot = KucoinBot(config)
    else:
        raise Exception(f"unknown exchange {user_info['exchange']}")
    return bot


async def shutdown_bot(bot):
    """Stop background tasks and close the exchange clients gracefully."""
    print("Shutting down bot...")
    bot.stop_data_maintainers()
    try:
        await asyncio.wait_for(bot.close(), timeout=3.0)
    except asyncio.TimeoutError:
        print("Shutdown timed out after 3 seconds. Forcing exit.")
    except Exception as e:
        print(f"Error during shutdown: {e}")


async def main():
    """Entry point: parse CLI args, load config, and launch the bot lifecycle."""
    parser = argparse.ArgumentParser(prog="passivbot", description="run passivbot")
    parser.add_argument(
        "config_path",
        type=str,
        nargs="?",
        default="configs/template.json",
        help="path to hjson passivbot config",
    )
    parser.add_argument(
        "--custom-endpoints",
        dest="custom_endpoints",
        default=None,
        help=(
            "Path to custom endpoints JSON for this run. "
            "Use 'none' to disable overrides even if a default file exists."
        ),
    )

    template_config = get_template_config("v7")
    del template_config["optimize"]
    del template_config["backtest"]
    add_arguments_recursively(parser, template_config)
    raw_args = merge_negative_cli_values(sys.argv[1:])
    args = parser.parse_args(raw_args)
    config = load_config(args.config_path, live_only=True)
    update_config_with_args(config, args)
    config = format_config(config, live_only=True)

    custom_endpoints_cli = args.custom_endpoints
    live_section = config.get("live") if isinstance(config.get("live"), dict) else {}
    custom_endpoints_cfg = live_section.get("custom_endpoints_path") if live_section else None

    override_path = None
    autodiscover = True
    preloaded_override = None

    def _sanitize(value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return "none"
            return stripped
        return str(value)

    cli_value = _sanitize(custom_endpoints_cli) if custom_endpoints_cli is not None else None
    cfg_value = _sanitize(custom_endpoints_cfg) if custom_endpoints_cfg is not None else None

    if cli_value is not None:
        if cli_value.lower() in {"none", "off", "disable"}:
            override_path = None
            autodiscover = False
            logging.info("Custom endpoints disabled via CLI argument.")
        else:
            override_path = cli_value
            autodiscover = False
            preloaded_override = load_custom_endpoint_config(override_path)
            logging.info("Using custom endpoints from CLI path: %s", override_path)
    elif cfg_value:
        if cfg_value.lower() in {"none", "off", "disable"}:
            override_path = None
            autodiscover = False
            logging.info("Custom endpoints disabled via config live.custom_endpoints_path.")
        else:
            override_path = cfg_value
            autodiscover = False
            preloaded_override = load_custom_endpoint_config(override_path)
            logging.info(
                "Using custom endpoints from config live.custom_endpoints_path: %s", override_path
            )
    else:
        logging.debug("Custom endpoints not specified; falling back to auto-discovery.")

    configure_custom_endpoint_loader(
        override_path,
        autodiscover=autodiscover,
        preloaded=preloaded_override,
    )

    user_info = load_user_info(require_live_value(config, "user"))
    await load_markets(user_info["exchange"], verbose=True)

    config = parse_overrides(config, verbose=True)
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
                if bot.ccp is not None:
                    await bot.ccp.close()
                if bot.cca is not None:
                    await bot.cca.close()
            except:
                pass
        if bot.stop_signal_received:
            logging.info("Bot stopped via signal; exiting main loop.")
            break

        logging.info(f"restarting bot...")
        print()
        for z in range(cooldown_secs, -1, -1):
            print(f"\rcountdown {z}...  ")
            await asyncio.sleep(1)
        print()

        restarts.append(utc_ms())
        restarts = [x for x in restarts if x > utc_ms() - 1000 * 60 * 60 * 24]
        max_restarts = int(require_live_value(bot.config, "max_n_restarts_per_day"))
        if len(restarts) > max_restarts:
            logging.info(f"n restarts exceeded {max_restarts} last 24h")
            break


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nBot shutdown complete.")
