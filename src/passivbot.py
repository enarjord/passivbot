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
import bisect
import pprint
import numpy as np
import inspect
import passivbot_rust as pbr
import logging
import math
from pathlib import Path
from candlestick_manager import CandlestickManager, CANDLE_DTYPE
from fill_events_manager import (
    FillEventsManager,
    _build_fetcher_for_bot,
    _extract_symbol_pool,
)
from typing import Dict, Iterable, Tuple, List, Optional, Any
from logging_setup import configure_logging, resolve_log_level
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
    coin_symbol_warning_counts,
)
from prettytable import PrettyTable
from uuid import uuid4
from copy import deepcopy
from dataclasses import dataclass
from collections import defaultdict, Counter
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
    get_optional_config_value,
    get_optional_live_value,
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

# Orchestrator-only: ideal orders are computed via Rust orchestrator (JSON API).
# Legacy Python order calculation paths are removed in this branch.

from custom_endpoint_overrides import (
    apply_rest_overrides_to_ccxt,
    configure_custom_endpoint_loader,
    get_custom_endpoint_source,
    load_custom_endpoint_config,
    resolve_custom_endpoint_override,
)


calc_min_entry_qty = pbr.calc_min_entry_qty_py
round_ = pbr.round_
round_up = pbr.round_up
round_dn = pbr.round_dn
round_dynamic = pbr.round_dynamic
calc_order_price_diff = pbr.calc_order_price_diff

DEFAULT_MAX_MEMORY_CANDLES_PER_SYMBOL = 20_000
PARTIAL_FILL_MERGE_MAX_DELAY_MS = 60_000
FILL_EVENT_FETCH_OVERLAP_COUNT = 20
FILL_EVENT_FETCH_OVERLAP_MAX_MS = 86_400_000  # 24 hours
FILL_EVENT_FETCH_LIMIT_DEFAULT = 20

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


def clip_by_timestamp(xs, start_ts, end_ts):
    # assumes xs is already sorted by timestamp
    timestamps = [x["timestamp"] for x in xs]
    i0 = bisect.bisect_left(timestamps, start_ts) if start_ts else 0
    i1 = bisect.bisect_right(timestamps, end_ts) if end_ts else len(xs)
    return xs[i0:i1]


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


def _trailing_bundle_tuple_to_dict(bundle_tuple: tuple[float, float, float, float]) -> dict:
    min_since_open, max_since_min, max_since_open, min_since_max = bundle_tuple
    return {
        "min_since_open": float(min_since_open),
        "max_since_min": float(max_since_min),
        "max_since_open": float(max_since_open),
        "min_since_max": float(min_since_max),
    }


def _trailing_bundle_default_dict() -> dict:
    return _trailing_bundle_tuple_to_dict(pbr.trailing_bundle_default_py())


def _trailing_bundle_from_arrays(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> dict:
    if highs.size == 0:
        return _trailing_bundle_default_dict()
    bundle_tuple = pbr.update_trailing_bundle_py(
        np.asarray(highs, dtype=np.float64),
        np.asarray(lows, dtype=np.float64),
        np.asarray(closes, dtype=np.float64),
        bundle=None,
    )
    return _trailing_bundle_tuple_to_dict(bundle_tuple)


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


def order_market_diff(side: str, order_price: float, market_price: float) -> float:
    """Return side-aware relative price diff between order and market."""
    return float(calc_order_price_diff(side, float(order_price), float(market_price)))


from pure_funcs import (
    numpyize,
    denumpyize,
    filter_orders,
    multi_replace,
    shorten_custom_id,
    determine_side_from_order_tuple,
    str2bool,
    flatten,
    log_dict_changes,
    ensure_millis,
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
        try:
            lvl_raw = get_optional_config_value(config, "logging.level", 1)
            lvl = int(float(lvl_raw)) if lvl_raw is not None else 1
        except Exception:
            lvl = 1
        self.logging_level = max(0, min(int(lvl), 3))
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
        self.order_details_str_len = 34
        self.order_type_str_len = 32
        self.stop_websocket = False
        raw_balance_override = get_optional_live_value(self.config, "balance_override", None)
        self.balance_override = (
            None if raw_balance_override in (None, "") else float(raw_balance_override)
        )
        self._balance_override_logged = False
        self.balance = 1e-12
        self.previous_hysteresis_balance = None
        self.balance_hysteresis_snap_pct = float(
            get_optional_live_value(self.config, "balance_hysteresis_snap_pct", 0.02)
        )
        # hedge_mode controls whether simultaneous long/short on same coin is allowed.
        # This is the config-level setting; exchange-specific bots may override
        # self.hedge_mode to False if the exchange doesn't support two-way mode.
        # Effective hedge_mode = config setting AND exchange capability.
        self._config_hedge_mode = bool(
            get_optional_live_value(self.config, "hedge_mode", True)
        )
        self.hedge_mode = True  # Exchange capability, may be overridden by subclass
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
        self.hyst_pct = 0.02
        self.state_change_detected_by_symbol = set()
        self.recent_order_executions = []
        self.recent_order_cancellations = []
        self._disabled_psides_logged = set()
        self._last_coin_symbol_warning_counts = {
            "symbol_to_coin_fallbacks": 0,
            "coin_to_symbol_fallbacks": 0,
        }
        self._last_plan_detail: dict[str, tuple[int, int, int]] = {}
        self._last_action_summary: dict[tuple[str, str], str] = {}
        # CandlestickManager settings from config.live
        cm_kwargs = {"exchange": self.cca, "debug": self.logging_level}
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
        lock_timeout = get_optional_live_value(config, "candle_lock_timeout_seconds", None)
        if lock_timeout not in (None, ""):
            try:
                cm_kwargs["lock_timeout_seconds"] = float(lock_timeout)
            except Exception:
                logging.warning(
                    "Unable to parse live.candle_lock_timeout_seconds=%r; using default",
                    lock_timeout,
                )
        max_concurrent = get_optional_live_value(config, "max_concurrent_api_requests", None)
        if max_concurrent not in (None, "", 0):
            try:
                cm_kwargs["max_concurrent_requests"] = int(max_concurrent)
            except Exception:
                logging.warning(
                    "Unable to parse live.max_concurrent_api_requests=%r; ignoring",
                    max_concurrent,
                )
        self.cm = CandlestickManager(**cm_kwargs)
        # TTL (minutes) for EMA candles on non-traded symbols
        ttl_min = require_live_value(config, "inactive_coin_candle_ttl_minutes")
        self.inactive_coin_candle_ttl_ms = int(float(ttl_min) * 60_000)
        raw_mem_interval = get_optional_config_value(
            config, "logging.memory_snapshot_interval_minutes", 30.0
        )
        try:
            interval_minutes = float(raw_mem_interval)
        except Exception:
            logging.warning(
                "Unable to parse logging.memory_snapshot_interval_minutes=%r; using fallback 30",
                raw_mem_interval,
            )
            interval_minutes = 30.0
        if interval_minutes <= 0.0:
            logging.warning(
                "logging.memory_snapshot_interval_minutes=%r is non-positive; using fallback 30",
                raw_mem_interval,
            )
            interval_minutes = 30.0
        self.memory_snapshot_interval_ms = max(60_000, int(interval_minutes * 60_000))
        raw_volume_threshold = get_optional_config_value(
            config, "logging.volume_refresh_info_threshold_seconds", 30.0
        )
        try:
            volume_threshold = float(raw_volume_threshold)
        except Exception:
            logging.warning(
                "Unable to parse logging.volume_refresh_info_threshold_seconds=%r; using fallback 30",
                raw_volume_threshold,
            )
            volume_threshold = 30.0
        if volume_threshold < 0:
            logging.warning(
                "logging.volume_refresh_info_threshold_seconds=%r is negative; using 0",
                raw_volume_threshold,
            )
            volume_threshold = 0.0
        self.volume_refresh_info_threshold_seconds = float(volume_threshold)
        auto_gs = bool(self.live_value("auto_gs"))
        self.PB_mode_stop = {
            "long": "graceful_stop" if auto_gs else "manual",
            "short": "graceful_stop" if auto_gs else "manual",
        }

        # FillEventsManager shadow mode: runs in parallel with legacy pnls, logs comparison
        self._pnls_shadow_mode = bool(
            get_optional_live_value(self.config, "pnls_manager_shadow_mode", False)
        )
        self._pnls_manager: Optional[FillEventsManager] = None
        self._pnls_shadow_initialized = False
        self._pnls_shadow_last_comparison_ts = 0
        self._pnls_shadow_comparison_interval_ms = 60_000  # compare every 60 seconds

    def live_value(self, key: str):
        return require_live_value(self.config, key)

    def bot_value(self, pside: str, key: str):
        return require_config_value(self.config, f"bot.{pside}.{key}")

    def _build_ccxt_options(self, overrides: Optional[dict] = None) -> dict:
        options = {"adjustForTimeDifference": True}
        recv_window = get_optional_live_value(self.config, "recv_window_ms", None)
        if recv_window not in (None, ""):
            try:
                recv_int = int(float(recv_window))
                if recv_int > 0:
                    options["recvWindow"] = recv_int
            except (TypeError, ValueError):
                logging.warning("Unable to parse live.recv_window_ms=%r; ignoring", recv_window)
        if overrides:
            options.update(overrides)
        return options

    async def start_bot(self):
        """Initialise state, warm cached data, and launch background loops."""
        logging.info(f"Starting bot {self.exchange}...")
        await format_approved_ignored_coins(self.config, self.user_info["exchange"], quote=self.quote)
        await self.init_markets()
        # Staggered warmup of candles for approved symbols (large sets handled gracefully)
        try:
            await self.warmup_candles_staggered()
        except Exception as e:
            logging.info(f"warmup skipped due to: {e}")
        await asyncio.sleep(1)
        self._log_memory_snapshot()
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
        # Reuse existing ccxt session when available (ensures shared options such as fetchMarkets types).
        cc_instance = getattr(self, "cca", None)
        self.markets_dict = await load_markets(self.exchange, 0, verbose=False, cc=cc_instance, quote=self.quote)
        await self.determine_utc_offset(verbose)
        # ineligible symbols cannot open new positions
        eligible, _, reasons = filter_markets(self.markets_dict, self.exchange, quote=self.quote, verbose=verbose)
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
        await self.update_positions_and_balance()
        await self.update_open_orders()
        await self.update_effective_min_cost()
        # Legacy: no 1m OHLCV REST maintenance; CandlestickManager handles caching
        if self.is_forager_mode():
            await self.update_first_timestamps()

    def log_once(self, msg: str):
        if not hasattr(self, "log_once_set"):
            self.log_once_set = set()
        if msg in self.log_once_set:
            return
        logging.info(msg)
        self.log_once_set.add(msg)

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
        cache_top = None
        tf_cache_bytes = None
        tf_cache_ranges = None
        tf_cache_top = None
        try:
            cache = getattr(self.cm, "_cache", {}) if hasattr(self, "cm") else {}
            cache_symbols = len(cache)
            stats = []
            for sym, arr in cache.items():
                if arr is None:
                    continue
                arr_bytes = int(getattr(arr, "nbytes", 0))
                arr_rows = int(arr.shape[0]) if hasattr(arr, "shape") else 0
                stats.append((sym, arr_bytes, arr_rows))
            cache_bytes = sum(val for _, val, _ in stats)
            cache_candles = sum(rows for _, _, rows in stats)
            if stats:
                top = sorted(stats, key=lambda item: item[1], reverse=True)[:3]
                cache_top = ", ".join(
                    f"{sym}:{bytes_ / (1024 * 1024):.1f}MiB/{rows}" for sym, bytes_, rows in top
                )
            tf_cache = getattr(self.cm, "_tf_range_cache", {}) if hasattr(self, "cm") else {}
            tf_stats = []
            for sym, entries in tf_cache.items():
                if not isinstance(entries, dict):
                    continue
                for key, val in entries.items():
                    try:
                        tf_label = key[0] if isinstance(key, tuple) and key else str(key)
                    except Exception:
                        tf_label = "unknown"
                    arr = val[0] if isinstance(val, tuple) and val else val
                    if not hasattr(arr, "nbytes"):
                        continue
                    arr_bytes = int(getattr(arr, "nbytes", 0))
                    arr_rows = int(arr.shape[0]) if hasattr(arr, "shape") else 0
                    tf_stats.append(((sym, tf_label), arr_bytes, arr_rows))
            if tf_stats:
                tf_cache_bytes = sum(size for _, size, _ in tf_stats)
                tf_cache_ranges = len(tf_stats)
                top_tf = sorted(tf_stats, key=lambda item: item[1], reverse=True)[:3]
                tf_cache_top = ", ".join(
                    f"{sym}:{tf}:{bytes_ / (1024 * 1024):.1f}MiB/{rows}"
                    for (sym, tf), bytes_, rows in top_tf
                )
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
            if cache_top:
                parts.append(f"cm_top={cache_top}")
        if tf_cache_bytes is not None:
            tf_desc = f"cm_tf_cache={tf_cache_bytes / (1024 * 1024):.2f} MiB"
            if tf_cache_ranges is not None:
                tf_desc += f" ({tf_cache_ranges} ranges)"
            parts.append(tf_desc)
            if tf_cache_top:
                parts.append(f"cm_tf_top={tf_cache_top}")
        try:
            loop = asyncio.get_running_loop()
            tasks = asyncio.all_tasks(loop)
            total_tasks = len(tasks)
            pending = sum(1 for t in tasks if not t.done())
            task_counts: Dict[str, int] = {}
            for t in tasks:
                coro = getattr(t, "get_coro", None)
                name = None
                if callable(coro):
                    try:
                        coro_obj = coro()
                        name = getattr(coro_obj, "__qualname__", None)
                    except Exception:
                        name = None
                if not name:
                    name = getattr(t, "get_name", lambda: None)()
                if not name:
                    name = type(t).__name__
                task_counts[name] = task_counts.get(name, 0) + 1
            top_tasks = ", ".join(
                f"{name}:{count}"
                for name, count in sorted(task_counts.items(), key=lambda kv: kv[1], reverse=True)[:4]
            )
            parts.append(f"tasks={total_tasks} pending={pending}")
            if top_tasks:
                parts.append(f"task_top={top_tasks}")
        except Exception:
            pass
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
        if self.coin_overrides:
            logging.debug(
                "Initialized coin overrides for %s",
                ", ".join(sorted(self.coin_overrides.keys())),
            )

    def config_get(self, path: [str], symbol=None):
        """
        Retrieve a configuration value, preferring per-symbol overrides when provided.
        """
        log_key = None
        if symbol and symbol in self.coin_overrides:
            d = self.coin_overrides[symbol]
            for p in path:
                if isinstance(d, dict) and p in d:
                    d = d[p]
                else:
                    d = None
                    break
            if d is not None:
                log_key = (symbol, ".".join(path))
                if not hasattr(self, "_override_hits_logged"):
                    self._override_hits_logged = set()
                if log_key not in self._override_hits_logged:
                    logging.debug("Using override for %s: %s", symbol, ".".join(path))
                    self._override_hits_logged.add(log_key)
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
        entry_volatility_logrange_ema_1h: Dict[str, Dict[str, float]],
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
                grid_lr = (entry_volatility_logrange_ema_1h or {}).get(pside, {}).get(symbol)
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

        # Random jitter delay to prevent API rate limit storms when multiple bots start simultaneously
        max_jitter = get_optional_live_value(self.config, "warmup_jitter_seconds", 30.0)
        try:
            max_jitter = float(max_jitter)
        except Exception:
            max_jitter = 30.0
        if max_jitter > 0:
            jitter = random.uniform(0, max_jitter)
            logging.info(f"warmup jitter: sleeping {jitter:.1f}s (max={max_jitter}s)")
            await asyncio.sleep(jitter)

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
                    ln = int(round(self.bp("long", "filter_volatility_ema_span", sym)))
                except Exception:
                    ln = default_win
                try:
                    sv = int(round(self.bp("short", "filter_volume_ema_span", sym)))
                except Exception:
                    sv = default_win
                try:
                    sn = int(round(self.bp("short", "filter_volatility_ema_span", sym)))
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
        result = coin_to_symbol(coin, self.exchange, quote=self.quote)
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
        failed_update_pos_oos_pnls_ohlcvs_count = 0
        max_n_fails = 10
        while not self.stop_signal_received:
            try:
                self.execution_scheduled = False
                self.state_change_detected_by_symbol = set()
                if not await self.update_pos_oos_pnls_ohlcvs():
                    await asyncio.sleep(0.5)
                    failed_update_pos_oos_pnls_ohlcvs_count += 1
                    if failed_update_pos_oos_pnls_ohlcvs_count > max_n_fails:
                        await self.restart_bot_on_too_many_errors()
                    continue
                failed_update_pos_oos_pnls_ohlcvs_count = 0
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
        balance_ok, positions_ok = await self.update_positions_and_balance()
        if not positions_ok:
            return False
        if not balance_ok:
            return False

        # Build task list: always include open_orders and pnls
        tasks = [
            self.update_open_orders(),
            self.update_pnls(),
        ]
        # If shadow mode is enabled, also run the FillEventsManager update in parallel
        if self._pnls_shadow_mode:
            tasks.append(self._update_pnls_shadow())

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Check results (first two are always open_orders and pnls)
        open_orders_ok = results[0] is True
        pnls_ok = results[1] is True

        # Handle shadow mode result (just log errors, don't fail the bot)
        if self._pnls_shadow_mode and len(results) > 2:
            shadow_result = results[2]
            if isinstance(shadow_result, Exception):
                logging.warning("[shadow] Shadow update raised exception: %s", shadow_result)
            # Run comparison after both updates complete
            self._compare_pnls_shadow()

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
                logging.debug("duplicate cancel candidate: %s", elm)
            seen.add(key)

        seen = set()
        for elm in to_create:
            key = str(elm["price"]) + str(elm["qty"])
            if key in seen:
                logging.debug("duplicate create candidate: %s", elm)
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
                    logging.debug(
                        "matching order cancellation found; will be delayed until next cycle: %s",
                        xf,
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
        grouped_orders: dict[str, list[dict]] = defaultdict(list)
        for order in orders:
            self.add_to_recent_order_executions(order)
            self.log_order_action(
                order,
                "posting order",
                context=order.get("_context", "plan_sync"),
                level=logging.DEBUG,
                delta=order.get("_delta"),
            )
            grouped_orders[order["symbol"]].append(order)
        self._log_order_action_summary(grouped_orders, "post")
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
        grouped_orders: dict[str, list[dict]] = defaultdict(list)
        for order in orders:
            self.add_to_recent_order_cancellations(order)
            self.log_order_action(
                order,
                "cancelling order",
                context=order.get("_context", "plan_sync"),
                level=logging.DEBUG,
                delta=order.get("_delta"),
            )
            grouped_orders[order["symbol"]].append(order)
        self._log_order_action_summary(grouped_orders, "cancel")
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

    def log_order_action(
        self,
        order,
        action,
        source="passivbot",
        *,
        level=logging.DEBUG,
        context: str | None = None,
        delta: dict | None = None,
    ):
        """Log a structured message describing an order action."""
        pb_order_type = self._resolve_pb_order_type(order)

        def _fmt(val):
            try:
                return f"{float(val):g}"
            except (TypeError, ValueError):
                return str(val)

        side = order.get("side", "?")
        qty = _fmt(order.get("qty", "?"))
        position_side = order.get("position_side", "?")
        price = _fmt(order.get("price", "?"))
        symbol = order.get("symbol", "?")
        coin = symbol_to_coin(symbol, verbose=False) or symbol
        details = f"{side} {qty} {position_side}@{price}"
        extra_parts = []
        if context:
            extra_parts.append(f"context={context}")
        elif order.get("_context"):
            extra_parts.append(f"context={order.get('_context')}")
        if delta:
            parts = []
            po, pn = delta.get("price_old"), delta.get("price_new")
            qo, qn = delta.get("qty_old"), delta.get("qty_new")
            if po is not None and pn is not None:
                parts.append(f"price {po} -> {pn} ({delta.get('price_pct_diff','?')}%)")
            if qo is not None and qn is not None:
                parts.append(f"qty {qo} -> {qn} ({delta.get('qty_pct_diff','?')}%)")
            if parts:
                extra_parts.append("delta=" + "; ".join(parts))
        msg = f"[order] {action: >{self.action_str_max_len}} {coin} | {details} | type={pb_order_type} | src={source}"
        if extra_parts:
            msg += " | " + " ".join(extra_parts)
        logging.log(level, msg)

    def _log_order_action_summary(self, grouped_orders: dict[str, list[dict]], action: str) -> None:
        """Emit condensed INFO summaries for batched order actions, skipping repeats."""
        max_entries = 4
        for symbol, orders in grouped_orders.items():
            if not orders:
                continue
            descriptors = []
            for order in orders:
                pb_order_type = self._resolve_pb_order_type(order)
                qty = order.get("qty")
                price = order.get("price")
                qty_str = f"{float(qty):g}" if isinstance(qty, (int, float)) else str(qty)
                price_str = f"{float(price):g}" if isinstance(price, (int, float)) else str(price)
                desc = (
                    f"{order.get('side','?')} {order.get('position_side','?')} "
                    f"{qty_str}@{price_str} {pb_order_type}"
                )
                extras = []
                context = order.get("_context")
                reason = order.get("_reason")
                if context:
                    extras.append(context)
                if reason and reason != context:
                    extras.append(f"reason={reason}")
                delta = order.get("_delta") or {}
                price_diff = delta.get("price_pct_diff")
                qty_diff = delta.get("qty_pct_diff")
                delta_parts = []
                if isinstance(price_diff, (int, float)) and price_diff:
                    delta_parts.append(f"Δp={price_diff:.3g}%")
                if isinstance(qty_diff, (int, float)) and qty_diff:
                    delta_parts.append(f"Δq={qty_diff:.3g}%")
                extras.extend(delta_parts)
                if extras:
                    desc += f" [{' '.join(extras)}]"
                descriptors.append(desc)
            if not descriptors:
                continue
            display = "; ".join(descriptors[:max_entries])
            if len(descriptors) > max_entries:
                display += f"; ... +{len(descriptors) - max_entries} more"
            key = (symbol, action)
            if self._last_action_summary.get(key) == display:
                continue
            self._last_action_summary[key] = display
            reason_counts = Counter(order.get("_reason") for order in orders if order.get("_reason"))
            reason_str = ""
            if reason_counts:
                reason_str = " | reasons=" + ", ".join(
                    f"{reason}:{count}" for reason, count in sorted(reason_counts.items())
                )
            coin = symbol_to_coin(symbol, verbose=False) or symbol
            logging.info("[order] %6s %s | %s%s", action, coin, display, reason_str)

    def _resolve_pb_order_type(self, order) -> str:
        """Best-effort decoding of Passivbot order type for logging."""
        if not isinstance(order, dict):
            return "unknown"
        pb_type = order.get("pb_order_type")
        if pb_type:
            return str(pb_type)
        symbol = order.get("symbol")
        if symbol and symbol in self.open_orders:
            for existing in self.open_orders[symbol]:
                if order_has_match(order, [existing], tolerance_price=0.0, tolerance_qty=0.0):
                    existing_type = existing.get("pb_order_type")
                    if existing_type:
                        return str(existing_type)
                    candidate = self._decode_pb_type_from_ids(existing)
                    if candidate:
                        return candidate
        candidate_ids = [
            order.get("custom_id"),
            order.get("customId"),
            order.get("client_order_id"),
            order.get("clientOrderId"),
            order.get("client_oid"),
            order.get("clientOid"),
            order.get("order_link_id"),
            order.get("orderLinkId"),
        ]
        candidate = self._decode_pb_type_from_ids(order, candidate_ids)
        if candidate:
            return candidate
        return "unknown"

    def _decode_pb_type_from_ids(
        self, order: dict, candidate_ids: Optional[list] = None
    ) -> Optional[str]:
        ids = candidate_ids
        if ids is None:
            ids = [
                order.get("custom_id"),
                order.get("customId"),
                order.get("client_order_id"),
                order.get("clientOrderId"),
                order.get("client_oid"),
                order.get("clientOid"),
                order.get("order_link_id"),
                order.get("orderLinkId"),
            ]
        for cid in ids:
            if not cid:
                continue
            snake = custom_id_to_snake(str(cid))
            if snake and snake != "unknown":
                return snake
        return None

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
            self.trailing_prices[symbol] = {
                "long": _trailing_bundle_default_dict(),
                "short": _trailing_bundle_default_dict(),
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
                logging.debug("failed to fetch candles for trailing %s: %s", sym, e)
                results[sym] = None

        # Compute trailing metrics per symbol/side
        for symbol, arr in results.items():
            if arr is None or arr.size == 0:
                continue
            if symbol not in last_position_changes:
                continue
            arr = np.sort(arr, order="ts")
            for pside, changed_ts in last_position_changes[symbol].items():
                mask = arr["ts"] > int(changed_ts)
                if not np.any(mask):
                    continue
                subset = arr[mask]
                try:
                    bundle = _trailing_bundle_from_arrays(subset["h"], subset["l"], subset["c"])
                    self.trailing_prices[symbol][pside] = bundle
                except Exception as e:
                    logging.debug("failed to compute trailing bundle for %s %s: %s", symbol, pside, e)

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
            logging.debug("symbol %s missing from self.symbol_ids. Using raw symbol.", symbol)
            self.symbol_ids[symbol] = symbol
            return symbol

    def to_ccxt_symbol(self, symbol: str) -> str:
        """Convert to ccxt standardized symbol"""
        candidates = []
        try:
            candidates.append(self.get_symbol_id_inv(symbol))
        except:
            pass
        try:
            candidates.append(self.coin_to_symbol(symbol))
        except:
            pass
        if candidates:
            return candidates[0]
        else:
            logging.info(f"failed to convert {symbol} to ccxt symbol. Using {symbol} as is.")

    def get_symbol_id_inv(self, symbol):
        """Return the human-friendly symbol for an exchange-native identifier."""
        try:
            if symbol in self.symbol_ids_inv:
                return self.symbol_ids_inv[symbol]
            else:
                return self.coin_to_symbol(symbol)
        except:
            logging.info(f"failed to convert {symbol} to ccxt symbol. Using {symbol} as is.")
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
            try:
                await self.update_exchange_config_by_symbols(symbols_not_done)
            except Exception as e:
                logging.info(f"error with update_exchange_config_by_symbols {e} {symbols_not_done}")
                traceback.print_exc()
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
                logging.info(f"[mode] {k:7s} {elm}")

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
        min_cost_flags = {s: self.effective_min_cost_is_low_enough(pside, s) for s in candidates}
        if not any(min_cost_flags.values()):
            if self.live_value("filter_by_min_effective_cost"):
                self.warn_on_high_effective_min_cost(pside)
            return []
        if self.is_forager_mode(pside):
            # filter coins by relative volume and log range
            clip_pct = self.bot_value(pside, "filter_volume_drop_pct")
            volatility_drop = self.bot_value(pside, "filter_volatility_drop_pct")
            max_n_positions = self.get_max_n_positions(pside)
            if clip_pct > 0.0:
                volumes, log_ranges = await self.calc_volumes_and_log_ranges(
                    pside, symbols=candidates
                )
            else:
                volumes = {
                    symbol: float(len(candidates) - idx) for idx, symbol in enumerate(candidates)
                }
                log_ranges = await self.calc_log_range(pside, eligible_symbols=candidates)
            features = [
                {
                    "index": idx,
                    "enabled": min_cost_flags.get(symbol, True),
                    "volume_score": volumes.get(symbol, 0.0),
                    "volatility_score": log_ranges.get(symbol, 0.0),
                }
                for idx, symbol in enumerate(candidates)
            ]
            selected = pbr.select_coin_indices_py(
                features,
                max_n_positions,
                clip_pct,
                volatility_drop,
                True,
            )
            ideal_coins = [candidates[i] for i in selected]
            if not ideal_coins and self.live_value("filter_by_min_effective_cost"):
                if any(not flag for flag in min_cost_flags.values()):
                    self.warn_on_high_effective_min_cost(pside)
        else:
            eligible = [s for s in candidates if min_cost_flags.get(s, True)]
            if not eligible:
                if self.live_value("filter_by_min_effective_cost"):
                    self.warn_on_high_effective_min_cost(pside)
                return []
            # all approved coins are selected, no filtering by volume and log range
            ideal_coins = sorted(eligible)
        return ideal_coins

    async def calc_volumes_and_log_ranges(
        self,
        pside: str,
        symbols: Optional[Iterable[str]] = None,
        *,
        max_age_ms: Optional[int] = 60_000,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Compute 1m EMA quote volume and 1m EMA log range per symbol with one candles fetch.

        This uses CandlestickManager.get_latest_ema_metrics() to avoid calling get_candles() twice
        per symbol (once for volume and once for log range).
        """
        span_volume = int(round(self.bot_value(pside, "filter_volume_ema_span")))
        span_volatility = int(round(self.bot_value(pside, "filter_volatility_ema_span")))
        if symbols is None:
            symbols = self.get_symbols_approved_or_has_pos(pside)

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
                res = await self.cm.get_latest_ema_metrics(
                    symbol,
                    {"qv": span_volume, "log_range": span_volatility},
                    max_age_ms=ttl,
                    timeframe=None,
                )
                vol = float(res.get("qv", float("nan")))
                lr = float(res.get("log_range", float("nan")))
                return (0.0 if not np.isfinite(vol) else vol, 0.0 if not np.isfinite(lr) else lr)
            except Exception:
                return (0.0, 0.0)

        syms = list(symbols)
        tasks = {s: asyncio.create_task(one(s)) for s in syms}
        volumes: Dict[str, float] = {}
        log_ranges: Dict[str, float] = {}
        started_ms = utc_ms()
        for sym, task in tasks.items():
            try:
                vol, lr = await task
            except Exception:
                vol, lr = 0.0, 0.0
            volumes[sym] = float(vol)
            log_ranges[sym] = float(lr)

        # Preserve the low-noise "top ranking changed" logging from calc_volumes/calc_log_range.
        elapsed_s = max(0.001, (utc_ms() - started_ms) / 1000.0)
        if volumes:
            top_n = min(8, len(volumes))
            top = sorted(volumes.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
            top_syms = tuple(sym for sym, _ in top)
            if not hasattr(self, "_volume_top_cache"):
                self._volume_top_cache = {}
            cache_key = (pside, span_volume)
            last_top = self._volume_top_cache.get(cache_key)
            if last_top != top_syms:
                self._volume_top_cache[cache_key] = top_syms
                summary = ", ".join(f"{symbol_to_coin(sym)}={val:.2f}" for sym, val in top)
                logging.info(
                    f"volume EMA span {span_volume}: {len(syms)} coins elapsed={int(elapsed_s)}s, top{top_n}: {summary}"
                )
        if log_ranges:
            top_n = min(8, len(log_ranges))
            top = sorted(log_ranges.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
            top_syms = tuple(sym for sym, _ in top)
            if not hasattr(self, "_log_range_top_cache"):
                self._log_range_top_cache = {}
            cache_key = (pside, span_volatility)
            last_top = self._log_range_top_cache.get(cache_key)
            if last_top != top_syms:
                self._log_range_top_cache[cache_key] = top_syms
                summary = ", ".join(f"{symbol_to_coin(sym)}={val:.6f}" for sym, val in top)
                logging.info(
                    f"log_range EMA span {span_volatility}: {len(syms)} coins elapsed={int(elapsed_s)}s, top{top_n}: {summary}"
                )

        return volumes, log_ranges

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
            return expand_PB_mode(mode)
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
        base_limit = self.get_wallet_exposure_limit(pside, symbol)
        allowance_pct = float(self.bp(pside, "risk_we_excess_allowance_pct", symbol))
        allowance_multiplier = 1.0 + max(0.0, allowance_pct)
        effective_limit = base_limit * allowance_multiplier
        return (
            self.balance * effective_limit * self.bp(pside, "entry_initial_qty_pct", symbol)
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

    async def handle_balance_update(self, source="REST"):
        if not hasattr(self, "_previous_balance"):
            self._previous_balance = 0.0
        if self.balance != self._previous_balance:
            try:
                equity = self.balance + (await self.calc_upnl_sum())
                logging.info(
                    f"[balance] {self._previous_balance} -> {self.balance} equity: {equity:.4f} source: {source}"
                )
            except Exception as e:
                logging.error(f"error with handle_balance_update {e}")
                traceback.print_exc()
            finally:
                self._previous_balance = self.balance
                self.execution_scheduled = True

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
        if not hasattr(self, "_pnls_cursor_ts"):
            self._pnls_cursor_ts = age_limit
        if self.pnls:
            start_time = max(self.pnls[-1]["timestamp"] - 1000, age_limit)
        else:
            start_time = max(self._pnls_cursor_ts, age_limit)
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
        if self.pnls:
            self._pnls_cursor_ts = max(self.pnls[-1]["timestamp"] - 1000, age_limit)
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
        elif not self.pnls:
            # no fills yet; avoid re-scanning entire lookback on the next cycle
            self._pnls_cursor_ts = max(self.get_exchange_time() - 1000, age_limit)
        return True

    # -------------------------------------------------------------------------
    # FillEventsManager Shadow Mode (dry run comparison with legacy pnls)
    # -------------------------------------------------------------------------

    async def _init_pnls_shadow_manager(self) -> bool:
        """Initialize the FillEventsManager for shadow mode comparison.

        Returns True if initialization succeeded, False otherwise.
        Shadow mode runs the new FillEventsManager in parallel with legacy pnls,
        caching data and logging comparisons, but not using it for bot decisions.
        """
        if not self._pnls_shadow_mode:
            return False

        if self._pnls_shadow_initialized:
            return True

        try:
            logging.info(
                "[shadow] Initializing FillEventsManager shadow mode for %s:%s",
                self.exchange,
                self.user,
            )

            # Extract symbol pool from config (same as legacy pnls uses)
            symbol_pool = _extract_symbol_pool(self.config, None)

            # Build the fetcher for this bot
            fetcher = _build_fetcher_for_bot(self, symbol_pool)

            # Create the FillEventsManager with its own cache path
            cache_path = Path(f"caches/fill_events/{self.exchange}/{self.user}")

            self._pnls_manager = FillEventsManager(
                exchange=self.exchange,
                user=self.user,
                fetcher=fetcher,
                cache_path=cache_path,
            )

            # Load cached events
            await self._pnls_manager.ensure_loaded()

            cached_count = len(self._pnls_manager._events)
            logging.info(
                "[shadow] FillEventsManager initialized: %d cached events loaded",
                cached_count,
            )

            self._pnls_shadow_initialized = True
            return True

        except Exception as e:
            logging.error("[shadow] Failed to initialize FillEventsManager: %s", e)
            traceback.print_exc()
            self._pnls_shadow_mode = False  # Disable shadow mode on init failure
            return False

    def _log_fill_event(self, event) -> str:
        """Format a FillEvent for logging.

        Format: [fill] BTC long entry +0.001 @ 100000.00
        For closes: [fill] BTC long close -0.001 @ 100500.00, pnl=+5.50 USDT
        For unknown orders: [fill] BTC long unknown -0.2 @ 2.05, pnl=-0.005 USDT (coid=abc123)
        """
        coin = symbol_to_coin(event.symbol, verbose=False) or event.symbol
        pside = event.position_side.lower()
        order_type = event.pb_order_type.lower() if event.pb_order_type else "fill"

        # Format qty with sign (+ for buys, - for sells)
        qty_sign = "+" if event.side.lower() == "buy" else "-"
        qty_str = f"{qty_sign}{abs(event.qty):.6g}"

        msg = f"[fill] {coin} {pside} {order_type} {qty_str} @ {event.price:.2f}"

        # Add pnl for closes (use 3 significant digits)
        if event.pnl != 0.0:
            pnl_sign = "+" if event.pnl >= 0 else ""
            msg += f", pnl={pnl_sign}{round_dynamic(event.pnl, 3)} USDT"

        # Add client_order_id for unknown orders
        if order_type == "unknown" and event.client_order_id:
            msg += f" (coid={event.client_order_id})"

        return msg

    def _log_new_fill_events(self, new_events: list) -> None:
        """Log new fill events. Truncates to summary if > 20 events."""
        if not new_events:
            return

        if len(new_events) > 20:
            # Truncate to summary
            total_pnl = sum(ev.pnl for ev in new_events)
            pnl_sign = "+" if total_pnl >= 0 else ""
            logging.info("[fill] %d fills, pnl=%s%s USDT", len(new_events), pnl_sign, round_dynamic(total_pnl, 3))
        else:
            # Log each event
            for event in sorted(new_events, key=lambda e: e.timestamp):
                logging.info(self._log_fill_event(event))

    def _calc_unstuck_allowances_from_fill_events(self, allow_new_unstuck: bool) -> dict[str, float]:
        """Calculate unstuck allowances using FillEventsManager data (shadow mode equivalent)."""
        if not allow_new_unstuck or self._pnls_manager is None:
            return {"long": 0.0, "short": 0.0}

        events = self._pnls_manager.get_events()
        if not events:
            return {"long": 0.0, "short": 0.0}

        pnls_cumsum = np.array([ev.pnl for ev in events]).cumsum()
        pnls_cumsum_max, pnls_cumsum_last = pnls_cumsum.max(), pnls_cumsum[-1]

        out = {}
        for pside in ["long", "short"]:
            pct = float(self.bot_value(pside, "unstuck_loss_allowance_pct") or 0.0)
            if pct > 0.0:
                out[pside] = float(
                    pbr.calc_auto_unstuck_allowance(
                        float(self.balance),
                        pct * float(self.bot_value(pside, "total_wallet_exposure_limit") or 0.0),
                        float(pnls_cumsum_max),
                        float(pnls_cumsum_last),
                    )
                )
            else:
                out[pside] = 0.0
        return out

    def _get_last_position_changes_from_fill_events(self) -> dict:
        """Get last position changes using FillEventsManager data (shadow mode equivalent)."""
        last_position_changes = defaultdict(dict)
        if self._pnls_manager is None:
            return last_position_changes

        events = self._pnls_manager.get_events()
        for symbol in self.positions:
            for pside in ["long", "short"]:
                if self.has_position(pside, symbol) and self.is_trailing(symbol, pside):
                    last_position_changes[symbol][pside] = utc_ms() - 1000 * 60 * 60 * 24 * 7
                    for ev in reversed(events):
                        try:
                            if ev.symbol == symbol and ev.position_side == pside:
                                last_position_changes[symbol][pside] = ev.timestamp
                                break
                        except Exception as e:
                            logging.error(f"Error in _get_last_position_changes_from_fill_events: {e}")
        return last_position_changes

    async def _update_pnls_shadow(self) -> bool:
        """Run the FillEventsManager refresh in shadow mode.

        Returns True if update succeeded, False otherwise.
        This runs in parallel with update_pnls() but results are only logged,
        not used for bot decisions.
        """
        if not self._pnls_shadow_mode:
            return False

        if not self._pnls_shadow_initialized:
            if not await self._init_pnls_shadow_manager():
                return False

        if self._pnls_manager is None:
            return False

        try:
            # Use the same lookback window as legacy pnls
            age_limit = self.get_exchange_time() - 1000 * 60 * 60 * 24 * float(
                self.live_value("pnls_max_lookback_days")
            )

            # Get existing event IDs before refresh
            existing_ids = set(ev.id for ev in self._pnls_manager.get_events())

            # Check if we need a full refresh (cache empty or too old)
            events = self._pnls_manager.get_events()
            needs_full_refresh = not events
            if events:
                oldest_event_ts = events[0].timestamp
                if oldest_event_ts > age_limit + 1000 * 60 * 60 * 24:  # > 1 day newer than limit
                    needs_full_refresh = True
                    logging.info(
                        "[shadow] Cache oldest event (%s) is newer than lookback (%s), doing full refresh",
                        ts_to_date(oldest_event_ts)[:19],
                        ts_to_date(age_limit)[:19],
                    )

            if needs_full_refresh:
                # Full refresh with proper lookback window
                logging.info(
                    "[shadow] Performing full refresh from %s",
                    ts_to_date(age_limit)[:19],
                )
                await self._pnls_manager.refresh(start_ms=int(age_limit), end_ms=None)
            else:
                # Incremental refresh like legacy
                await self._pnls_manager.refresh_latest(overlap=20)

            # Find and log new events (those not in cache before refresh)
            all_events = self._pnls_manager.get_events()
            new_events = [ev for ev in all_events if ev.id not in existing_ids]
            if new_events:
                self._log_new_fill_events(new_events)

            return True

        except RateLimitExceeded:
            logging.warning("[shadow] Rate limit while fetching fill events; retrying next cycle")
            return False
        except Exception as e:
            logging.error("[shadow] Failed to update FillEventsManager: %s", e)
            if self.logging_level >= 2:
                traceback.print_exc()
            return False

    def _compare_pnls_shadow(self) -> None:
        """Compare legacy pnls with FillEventsManager data and log differences.

        This comparison is logged periodically to avoid spamming logs.
        Differences are logged at DEBUG level normally, INFO level for significant discrepancies.
        """
        if not self._pnls_shadow_mode or self._pnls_manager is None:
            return

        now_ms = utc_ms()
        if now_ms - self._pnls_shadow_last_comparison_ts < self._pnls_shadow_comparison_interval_ms:
            return

        self._pnls_shadow_last_comparison_ts = now_ms

        try:
            # Get lookback window
            age_limit = self.get_exchange_time() - 1000 * 60 * 60 * 24 * float(
                self.live_value("pnls_max_lookback_days")
            )

            # Legacy pnls data
            legacy_events = [p for p in getattr(self, "pnls", []) if p.get("timestamp", 0) >= age_limit]
            legacy_count = len(legacy_events)
            legacy_pnl_sum = sum(p.get("pnl", 0.0) for p in legacy_events)
            legacy_ids = set(p.get("id", "") for p in legacy_events)

            # New FillEventsManager data
            manager_events = self._pnls_manager.get_events(start_ms=int(age_limit))
            manager_count = len(manager_events)
            manager_pnl_sum = sum(ev.pnl for ev in manager_events)
            manager_ids = set(ev.id for ev in manager_events)

            # Calculate differences
            count_diff = manager_count - legacy_count
            pnl_diff = manager_pnl_sum - legacy_pnl_sum

            # Find IDs only in one system
            only_in_legacy = legacy_ids - manager_ids
            only_in_manager = manager_ids - legacy_ids

            # Latest timestamps
            legacy_latest = max((p.get("timestamp", 0) for p in legacy_events), default=0)
            manager_latest = max((ev.timestamp for ev in manager_events), default=0)

            # Log comparison
            log_level = logging.DEBUG
            if abs(count_diff) > 10 or abs(pnl_diff) > 1.0:
                log_level = logging.INFO  # Log more prominently if significant difference

            logging.log(
                log_level,
                "[shadow] Comparison: legacy=%d events (pnl=%.4f), manager=%d events (pnl=%.4f), "
                "diff=%+d events (pnl=%+.4f)",
                legacy_count,
                legacy_pnl_sum,
                manager_count,
                manager_pnl_sum,
                count_diff,
                pnl_diff,
            )

            if only_in_legacy:
                logging.debug(
                    "[shadow] IDs only in legacy (%d): %s",
                    len(only_in_legacy),
                    list(only_in_legacy)[:5],  # Show first 5
                )

            if only_in_manager:
                logging.debug(
                    "[shadow] IDs only in manager (%d): %s",
                    len(only_in_manager),
                    list(only_in_manager)[:5],  # Show first 5
                )

            # Log timestamp comparison
            if legacy_latest > 0 and manager_latest > 0:
                ts_diff_ms = manager_latest - legacy_latest
                if abs(ts_diff_ms) > 60_000:  # More than 1 minute difference
                    logging.info(
                        "[shadow] Latest timestamp diff: legacy=%s, manager=%s (diff=%+.1fs)",
                        ts_to_date(legacy_latest),
                        ts_to_date(manager_latest),
                        ts_diff_ms / 1000.0,
                    )

            # Compare unstuck allowances (always log at INFO for validation)
            legacy_unstuck = self._calc_unstuck_allowances_live(allow_new_unstuck=True)
            manager_unstuck = self._calc_unstuck_allowances_from_fill_events(allow_new_unstuck=True)
            unstuck_diff_long = manager_unstuck["long"] - legacy_unstuck["long"]
            unstuck_diff_short = manager_unstuck["short"] - legacy_unstuck["short"]
            match_str = "MATCH" if abs(unstuck_diff_long) < 0.01 and abs(unstuck_diff_short) < 0.01 else "DIFF"
            logging.info(
                "[shadow] Unstuck allowances %s: legacy=(long=%.4f, short=%.4f), "
                "manager=(long=%.4f, short=%.4f), diff=(long=%+.4f, short=%+.4f)",
                match_str,
                legacy_unstuck["long"],
                legacy_unstuck["short"],
                manager_unstuck["long"],
                manager_unstuck["short"],
                unstuck_diff_long,
                unstuck_diff_short,
            )

            # Compare last position changes (timestamps for trailing logic)
            legacy_lpc = self.get_last_position_changes()
            manager_lpc = self._get_last_position_changes_from_fill_events()
            all_symbols = set(legacy_lpc.keys()) | set(manager_lpc.keys())
            for symbol in sorted(all_symbols):
                coin = symbol_to_coin(symbol, verbose=False) or symbol
                for pside in ["long", "short"]:
                    legacy_ts = legacy_lpc.get(symbol, {}).get(pside)
                    manager_ts = manager_lpc.get(symbol, {}).get(pside)
                    if legacy_ts is None and manager_ts is None:
                        continue
                    legacy_dt = ts_to_date(legacy_ts) if legacy_ts else "N/A"
                    manager_dt = ts_to_date(manager_ts) if manager_ts else "N/A"
                    if legacy_ts != manager_ts:
                        diff_s = ((manager_ts or 0) - (legacy_ts or 0)) / 1000.0 if legacy_ts and manager_ts else 0
                        logging.info(
                            "[shadow] Last position change %s %s DIFF: legacy=%s, manager=%s (diff=%+.1fs)",
                            coin,
                            pside,
                            legacy_dt,
                            manager_dt,
                            diff_s,
                        )
                    else:
                        logging.info(
                            "[shadow] Last position change %s %s MATCH: %s",
                            coin,
                            pside,
                            legacy_dt,
                        )

        except Exception as e:
            logging.error("[shadow] Error during pnls comparison: %s", e)
            if self.logging_level >= 2:
                traceback.print_exc()

    async def init_fill_events(self):
        """Initialise in-memory fill events cache."""
        if not hasattr(self, "fill_events"):
            self.fill_events = []
        if not hasattr(self, "fill_events_cache_path"):
            self.fill_events_cache_path = make_get_filepath(
                f"caches/{self.exchange}/{self.user}_fill_events.json"
            )
        if not hasattr(self, "fill_events_loaded"):
            self.fill_events_loaded = False

        if self.fill_events_loaded:
            return

        age_limit = self.get_exchange_time() - 1000 * 60 * 60 * 24 * float(
            self.live_value("pnls_max_lookback_days")
        )
        loaded_events: List[dict] = []
        cache_needs_dump = False
        if os.path.exists(self.fill_events_cache_path):
            try:
                loaded_events = json.load(open(self.fill_events_cache_path))
            except Exception as exc:
                logging.error(f"error loading {self.fill_events_cache_path}: {exc}")

        merged: Dict[str, dict] = {}
        if loaded_events:
            normalized: List[dict] = []
            for raw in loaded_events:
                try:
                    evt = self._canonicalize_fill_event(raw)
                except Exception as exc:
                    logging.error(f"discarding malformed cached fill event: {exc}")
                    continue
                if evt["timestamp"] >= age_limit:
                    normalized.append(evt)
            merged.update({evt["id"]: evt for evt in normalized})
            if normalized != loaded_events:
                cache_needs_dump = True

            oldest = min((evt["timestamp"] for evt in normalized), default=None)
            newest = max((evt["timestamp"] for evt in normalized), default=None)
            gap_fills: List[dict] = []
            if newest is not None:
                try:
                    gap_fills.extend(await self.fetch_fill_events(start_time=newest - 1000) or [])
                except Exception as exc:
                    logging.error(f"failed to fetch recent fill events: {exc}")
            if oldest is not None and oldest > age_limit + 1000 * 60 * 60 * 4:
                logging.info(
                    "fetching missing fill events from before %s",
                    ts_to_date(oldest),
                )
                try:
                    gap_fills.extend(
                        await self.fetch_fill_events(start_time=age_limit, end_time=oldest) or []
                    )
                except Exception as exc:
                    logging.error(f"failed to fetch historical fill events: {exc}")

            for raw in gap_fills:
                try:
                    evt = self._canonicalize_fill_event(raw)
                except Exception as exc:
                    logging.error(f"discarding malformed fill event: {exc}")
                    continue
                if evt["timestamp"] >= age_limit:
                    merged[evt["id"]] = evt
            if gap_fills:
                cache_needs_dump = True
        else:
            try:
                fresh = await self.fetch_fill_events(start_time=age_limit)
            except RateLimitExceeded:
                logging.warning("rate limit while fetching initial fill events; retrying later")
                fresh = []
            except Exception as exc:
                logging.error(f"failed to fetch initial fill events: {exc}")
                fresh = []

            for raw in fresh or []:
                try:
                    evt = self._canonicalize_fill_event(raw)
                except Exception as exc:
                    logging.error(f"discarding malformed fill event: {exc}")
                    continue
                if evt["timestamp"] >= age_limit:
                    merged[evt["id"]] = evt
            if fresh:
                cache_needs_dump = True

        self.fill_events = sorted(merged.values(), key=lambda x: x["timestamp"])
        if not hasattr(self, "_fill_event_fingerprints"):
            self._fill_event_fingerprints = {}
        for evt in self.fill_events:
            fp = self._fingerprint_event(evt)
            self._fill_event_fingerprints.setdefault(evt["id"], set()).add(fp)
        self.fill_events_loaded = True
        if cache_needs_dump and self.fill_events:
            payload = [dict(evt) for evt in self.fill_events]
            try:
                json.dump(payload, open(self.fill_events_cache_path, "w"))
            except Exception as exc:
                logging.error(f"error dumping fill events to {self.fill_events_cache_path}: {exc}")

    async def fetch_fill_events(self, start_time=None, end_time=None, limit=None):
        """Exchange-specific fill event fetcher (to be implemented by subclasses)."""
        raise NotImplementedError("fetch_fill_events must be implemented by exchange subclasses")

    def _canonicalize_fill_event(self, raw: dict) -> dict:
        """Validate and normalise a raw fill event into canonical FillEvent shape."""
        required_keys = (
            "id",
            "timestamp",
            "symbol",
            "side",
            "qty",
            "price",
            "pnl",
            "position_side",
        )
        missing = [key for key in required_keys if key not in raw]
        if missing:
            raise ValueError(f"fill event missing required keys: {missing}")

        try:
            event_id = str(raw["id"])
        except Exception as exc:
            raise ValueError(f"invalid fill id {raw.get('id')}") from exc
        if not event_id:
            raise ValueError("fill event id cannot be empty")

        try:
            ts = int(ensure_millis(raw["timestamp"]))
        except Exception as exc:
            raise ValueError(f"invalid fill timestamp {raw.get('timestamp')}") from exc

        symbol = str(raw["symbol"])
        if not symbol:
            raise ValueError("fill event symbol cannot be empty")

        side = str(raw["side"]).lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"unsupported fill side {raw.get('side')}")

        pside = str(raw["position_side"]).lower()
        if pside not in ("long", "short"):
            raise ValueError(f"unsupported position_side {raw.get('position_side')}")

        try:
            qty = float(raw["qty"])
        except Exception as exc:
            raise ValueError(f"invalid fill qty {raw.get('qty')}") from exc

        try:
            price = float(raw["price"])
        except Exception as exc:
            raise ValueError(f"invalid fill price {raw.get('price')}") from exc

        try:
            pnl = float(raw["pnl"])
        except Exception as exc:
            raise ValueError(f"invalid fill pnl {raw.get('pnl')}") from exc

        result = {
            "id": event_id,
            "timestamp": ts,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price,
            "pnl": pnl,
            "position_side": pside,
        }

        if "fees" in raw and raw["fees"] is not None:
            try:
                if "fees" in result:
                    result["fees"] = eval(raw["fees"])
            except Exception:
                logging.warning(f"failed to parse fees for fill {event_id}; dropping field. {result}")
        if "pb_order_type" in raw and raw["pb_order_type"] is not None:
            result["pb_order_type"] = str(raw["pb_order_type"])

        return result

    def _fingerprint_event(self, event: dict) -> tuple:
        return (
            event.get("id"),
            int(event.get("timestamp", 0)),
            round(float(event.get("qty", 0.0)), 12),
            round(float(event.get("price", 0.0)), 12),
            round(float(event.get("pnl", 0.0)), 12),
            str(event.get("pb_order_type")),
        )

    def _merge_fill_event_group(self, event_id: str, events: List[dict]) -> List[dict]:
        events = [dict(evt) for evt in sorted(events, key=lambda x: x["timestamp"])]
        if len(events) == 1:
            return [events[0]]

        symbols = {evt["symbol"] for evt in events}
        sides = {evt["side"] for evt in events}
        psides = {evt["position_side"] for evt in events}

        time_span = events[-1]["timestamp"] - events[0]["timestamp"]
        if (
            len(symbols) > 1
            or len(sides) > 1
            or len(psides) > 1
            or time_span > PARTIAL_FILL_MERGE_MAX_DELAY_MS
        ):
            reason = []
            if len(symbols) > 1:
                reason.append("symbol mismatch")
            if len(sides) > 1:
                reason.append("side mismatch")
            if len(psides) > 1:
                reason.append("position_side mismatch")
            if time_span > PARTIAL_FILL_MERGE_MAX_DELAY_MS:
                reason.append(f"time span {time_span/1000:.1f}s")
            logging.warning(
                "fill id %s emitted as multiple events (%s)",
                event_id,
                ", ".join(reason) or "unknown reason",
            )
            merged_events = []
            for idx, evt in enumerate(events, start=1):
                new_evt = dict(evt)
                if idx > 1:
                    new_evt["id"] = f"{evt['id']}#{idx}"
                merged_events.append(new_evt)
            return merged_events

        total_qty = sum(abs(evt["qty"]) for evt in events)
        if total_qty <= 0.0:
            total_qty = sum(evt["qty"] for evt in events)
        if total_qty == 0.0:
            total_qty = events[-1]["qty"]

        weighted_price = 0.0
        for evt in events:
            qty = abs(evt["qty"]) if total_qty != 0 else evt["qty"]
            weighted_price += evt["price"] * qty

        merged = dict(events[-1])
        merged["timestamp"] = events[-1]["timestamp"]
        merged["qty"] = sum(evt["qty"] for evt in events)
        merged["pnl"] = sum(evt["pnl"] for evt in events)
        if total_qty != 0:
            merged["price"] = weighted_price / total_qty
        merged_fees = [evt.get("fees") for evt in events if evt.get("fees") is not None]
        if merged_fees:
            merged["fees"] = sum(merged_fees)
        pb_types = {evt.get("pb_order_type") for evt in events if evt.get("pb_order_type")}
        if len(pb_types) > 1:
            logging.warning(
                "fill id %s had multiple pb_order_type values: %s; keeping the latest",
                event_id,
                pb_types,
            )
        return [merged]

    def _merge_fill_events_collection(self, events: Iterable[dict], age_limit: int) -> List[dict]:
        grouped: Dict[str, List[dict]] = defaultdict(list)
        for evt in events:
            if evt["timestamp"] >= age_limit:
                grouped[evt["id"]].append(evt)

        merged: List[dict] = []
        used_ids: set[str] = set()
        for event_id, group in grouped.items():
            merged_group = self._merge_fill_event_group(event_id, group)
            for evt in merged_group:
                new_evt = dict(evt)
                if new_evt["id"] in used_ids:
                    base_id = evt["id"]
                    suffix = 2
                    candidate = f"{base_id}#{suffix}"
                    while candidate in used_ids:
                        suffix += 1
                        candidate = f"{base_id}#{suffix}"
                    new_evt["id"] = candidate
                used_ids.add(new_evt["id"])
                merged.append(new_evt)

        merged.sort(key=lambda x: x["timestamp"])
        return merged

    def _events_close(
        self,
        a: dict,
        b: dict,
        *,
        qty_tol: float = 1e-9,
        price_tol: float = 1e-8,
        pnl_tol: float = 1e-9,
    ) -> bool:
        if a is None or b is None:
            return False
        if a.get("symbol") != b.get("symbol"):
            return False
        if a.get("side") != b.get("side"):
            return False
        if a.get("position_side") != b.get("position_side"):
            return False
        if abs(a.get("qty", 0.0) - b.get("qty", 0.0)) > qty_tol:
            return False
        price_ref = max(1.0, abs(a.get("price", 0.0)), abs(b.get("price", 0.0)))
        if abs(a.get("price", 0.0) - b.get("price", 0.0)) > price_tol * price_ref:
            return False
        if abs(a.get("pnl", 0.0) - b.get("pnl", 0.0)) > pnl_tol:
            return False
        fees_a = a.get("fees")
        fees_b = b.get("fees")
        if fees_a is None and fees_b is None:
            return True
        if fees_a is None or fees_b is None:
            return False
        if abs(fees_a - fees_b) > pnl_tol:
            return False
        if str(a.get("pb_order_type")) != str(b.get("pb_order_type")):
            return False
        if str(a.get("client_order_id")) != str(b.get("client_order_id")):
            return False
        return True

    async def update_fill_events(
        self, start_time: Optional[int] = None, end_time: Optional[int] = None
    ):
        """
        Fetch canonical fill events and maintain an up-to-date, deduplicated cache.

        The cache is stored in self.fill_events (sorted by timestamp ascending) and may be
        reused by downstream consumers (e.g., equity reconstruction).
        """

        if self.stop_signal_received:
            return False
        await self.init_fill_events()

        age_limit = self.get_exchange_time() - 1000 * 60 * 60 * 24 * float(
            self.live_value("pnls_max_lookback_days")
        )

        previous_map = {evt["id"]: evt for evt in self.fill_events}
        known_event_ids: set[str] = set(previous_map.keys())
        for key in list(previous_map.keys()):
            if "#" in key:
                known_event_ids.add(key.split("#", 1)[0])
        latest_ts = self.fill_events[-1]["timestamp"] if self.fill_events else None

        effective_start = start_time
        fetch_limit = None if start_time is not None else FILL_EVENT_FETCH_LIMIT_DEFAULT
        if effective_start is None:
            effective_start = age_limit
            if self.fill_events:
                overlap_count = min(len(self.fill_events), FILL_EVENT_FETCH_OVERLAP_COUNT)
                overlap_idx = max(0, len(self.fill_events) - overlap_count)
                overlap_ts = self.fill_events[overlap_idx]["timestamp"]
                now_ts = self.get_exchange_time()
                min_start = max(age_limit, now_ts - FILL_EVENT_FETCH_OVERLAP_MAX_MS)
                effective_start = max(age_limit, min_start, overlap_ts)
            elif latest_ts is not None:
                effective_start = max(age_limit, latest_ts - 1000)

        try:
            fetched = await self.fetch_fill_events(
                start_time=effective_start,
                end_time=end_time,
                limit=fetch_limit,
            )
        except RateLimitExceeded:
            logging.warning("rate limit while fetching fill events; retrying next cycle")
            return False
        except NotImplementedError:
            logging.error("fetch_fill_events not implemented for this exchange")
            return False
        except Exception as exc:
            logging.error(f"failed to fetch fill events: {exc}")
            traceback.print_exc()
            return False

        grouped_updates: Dict[str, List[dict]] = defaultdict(list)
        for raw in fetched or []:
            try:
                event = self._canonicalize_fill_event(raw)
            except Exception as exc:
                logging.error(f"discarding malformed fill event: {exc}")
                continue
            if event["timestamp"] < age_limit and event["id"] not in known_event_ids:
                continue
            fp = self._fingerprint_event(event)
            fp_set = self._fill_event_fingerprints.setdefault(event["id"], set())
            if fp in fp_set:
                continue
            fp_set.add(fp)
            grouped_updates[event["id"]].append(event)
        if not grouped_updates:
            return True

        result_map: Dict[str, dict] = {
            evt["id"]: evt for evt in self.fill_events if evt["timestamp"] >= age_limit
        }

        delta_count = 0
        delta_pnl = 0.0

        def _lists_close(new_events: List[dict], old_events: List[dict]) -> bool:
            if len(new_events) != len(old_events):
                return False
            for a, b in zip(new_events, old_events):
                if not self._events_close(a, b):
                    return False
            return True

        for event_id, updates in grouped_updates.items():
            if not updates:
                continue

            related_keys = [
                key for key in result_map.keys() if key == event_id or key.startswith(f"{event_id}#")
            ]
            prev_entries = [
                previous_map[key]
                for key in previous_map
                if key == event_id or key.startswith(f"{event_id}#")
            ]
            prev_pnl = sum(evt.get("pnl", 0.0) for evt in prev_entries)

            for key in related_keys:
                result_map.pop(key, None)
            for key in list(self._fill_event_fingerprints.keys()):
                if key == event_id or key.startswith(f"{event_id}#"):
                    del self._fill_event_fingerprints[key]

            merged_group = self._merge_fill_event_group(event_id, updates)
            if not merged_group:
                continue

            changed = not _lists_close(merged_group, prev_entries)
            if changed:
                delta_count += len(merged_group)
                new_pnl = sum(evt.get("pnl", 0.0) for evt in merged_group)
                delta_pnl += new_pnl - prev_pnl

            for evt in merged_group:
                result_map[evt["id"]] = evt
                self._fill_event_fingerprints.setdefault(evt["id"], set()).add(
                    self._fingerprint_event(evt)
                )

        self.fill_events = sorted(result_map.values(), key=lambda x: x["timestamp"])

        if delta_count > 0:
            logging.info(
                f"{delta_count} updated fill event"
                f"{'s' if delta_count != 1 else ''}"
                f" (delta pnl {delta_pnl} {self.quote})"
            )

        if delta_count > 0 or not os.path.exists(self.fill_events_cache_path):
            cache_payload = [evt for evt in self.fill_events if evt["timestamp"] >= age_limit]
            try:
                json.dump(cache_payload, open(self.fill_events_cache_path, "w"))
            except Exception as exc:
                logging.error(f"error dumping fill events to {self.fill_events_cache_path}: {exc}")
        return True

    def log_pnls_change(self, old_pnls, new_pnls):
        """Log differences between previous and new PnL entries for debugging."""
        keys = ["id", "timestamp", "symbol", "side", "position_side", "price", "qty"]
        old_pnls_compressed = {(x[k] for k in keys) for x in old_pnls}
        new_pnls_compressed = [(x[k] for k in keys) for x in new_pnls]
        added_pnls = [x for x in new_pnls_compressed if x not in old_pnls_compressed]

    async def get_balance_equity_history(
        self, fill_events: Optional[List[dict]] = None, current_balance: Optional[float] = None
    ) -> Dict[str, Any]:
        """Replay canonical fill events to produce historical balance/equity curves."""

        await self.init_pnls()

        def _safe_float(val: Any, default: float = 0.0) -> float:
            try:
                if val is None:
                    return default
                return float(val)
            except Exception:
                return default

        def _normalize_symbol(symbol: Any) -> str:
            sym = str(symbol) if symbol else ""
            if not sym:
                return ""
            if sym in self.c_mults:
                return sym
            try:
                converted = self.get_symbol_id_inv(sym)
                if converted:
                    return converted
            except Exception:
                pass
            return sym

        def _ensure_slot(container: Dict[str, Dict[str, Dict[str, float]]], symbol: str):
            if symbol not in container:
                container[symbol] = {
                    "long": {"size": 0.0, "price": 0.0},
                    "short": {"size": 0.0, "price": 0.0},
                }
            return container[symbol]

        def _determine_action(
            pside: str, side: str, qty_signed: Optional[float], explicit: Optional[str]
        ):
            if explicit in ("increase", "decrease"):
                return explicit
            if qty_signed is not None and qty_signed != 0.0:
                return "increase" if qty_signed > 0 else "decrease"
            side = side.lower()
            if pside == "long":
                return "increase" if side == "buy" else "decrease"
            return "increase" if side == "sell" else "decrease"

        def _extract_events(source: List[dict]) -> List[dict]:
            out = []
            for fill in source:
                ts_raw = fill.get("timestamp")
                if ts_raw is None:
                    continue
                try:
                    ts = int(ensure_millis(ts_raw))
                except Exception:
                    continue
                symbol = _normalize_symbol(fill.get("symbol"))
                if not symbol:
                    continue
                pside = str(fill.get("position_side", fill.get("pside", "long"))).lower()
                if pside not in ("long", "short"):
                    pside = "long"
                qty_signed = fill.get("qty_signed")
                qty_fallback_keys = ("qty", "amount", "size", "contracts")
                qty_val = _safe_float(
                    (
                        qty_signed
                        if qty_signed is not None
                        else next(
                            (fill.get(k) for k in qty_fallback_keys if fill.get(k) is not None), 0.0
                        )
                    ),
                    0.0,
                )
                qty = abs(qty_val)
                if qty <= 0.0:
                    continue
                price_keys = ("price", "avgPrice", "average", "avg_price", "execPrice")
                price = next((fill.get(k) for k in price_keys if fill.get(k) is not None), None)
                if price is None:
                    info = fill.get("info", {})
                    price = (
                        info.get("avgPrice") or info.get("execPrice") or info.get("avg_exec_price")
                    )
                price = _safe_float(price, 0.0)
                if price <= 0.0:
                    continue
                pnl_val = _safe_float(fill.get("pnl", 0.0), 0.0)
                fee_cost = 0.0
                fee_obj = fill.get("fee")
                if isinstance(fee_obj, dict):
                    fee_cost = _safe_float(fee_obj.get("cost", 0.0), 0.0)
                elif isinstance(fee_obj, (int, float, str)):
                    fee_cost = _safe_float(fee_obj, 0.0)
                elif isinstance(fill.get("fees"), (list, tuple)):
                    fee_cost = sum(
                        _safe_float(x.get("cost", 0.0), 0.0)
                        for x in fill["fees"]
                        if isinstance(x, dict)
                    )
                side = str(fill.get("side", "")).lower()
                action = _determine_action(pside, side, qty_signed, fill.get("action"))
                out.append(
                    {
                        "timestamp": ts,
                        "symbol": symbol,
                        "pside": pside,
                        "qty": qty,
                        "price": price,
                        "action": action,
                        "pnl": pnl_val,
                        "fee": fee_cost,
                        "c_mult": float(self.c_mults.get(symbol, 1.0)),
                    }
                )
            return sorted(out, key=lambda x: x["timestamp"])

        if fill_events is None:
            fill_events = getattr(self, "pnls", [])

        events = _extract_events(fill_events)
        if not events:
            ts_now = self.get_exchange_time()
            balance_now = (
                float(current_balance) if current_balance is not None else float(self.balance)
            )
            point = {
                "timestamp": ts_now,
                "balance": balance_now,
                "equity": balance_now,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
            }
            return {
                "timeline": [point],
                "balances": [{"timestamp": point["timestamp"], "balance": balance_now}],
                "equities": [
                    {"timestamp": point["timestamp"], "equity": balance_now, "unrealized_pnl": 0.0}
                ],
                "metadata": {
                    "lookback_days": float(self.live_value("pnls_max_lookback_days")),
                    "resolution_ms": ONE_MIN_MS,
                    "events_used": 0,
                    "symbols_covered": [],
                    "missing_price_symbols": [],
                },
            }

        lookback_days = float(self.live_value("pnls_max_lookback_days"))
        ts_now = self.get_exchange_time()
        lookback_ms = max(lookback_days, 0.0) * 24 * 60 * 60 * 1000
        lookback_start = ts_now - lookback_ms

        balance_now = float(current_balance) if current_balance is not None else float(self.balance)
        balance_now = max(balance_now, 0.0)
        total_realised = sum(
            evt["pnl"] + evt.get("fee", 0.0) for evt in events if evt["timestamp"] <= ts_now
        )
        baseline_balance = balance_now - total_realised

        start_ts = min(ensure_millis(events[0]["timestamp"]), lookback_start)
        start_minute = int(math.floor(start_ts / ONE_MIN_MS) * ONE_MIN_MS)
        record_start_minute = int(math.floor(lookback_start / ONE_MIN_MS) * ONE_MIN_MS)
        end_minute = int(math.floor(ts_now / ONE_MIN_MS) * ONE_MIN_MS)
        if end_minute < record_start_minute:
            end_minute = record_start_minute

        symbols = {evt["symbol"] for evt in events if evt["symbol"]}
        price_lookup: Dict[str, Dict[int, float]] = {}
        if symbols and getattr(self, "cm", None) is not None:
            tasks = {
                sym: asyncio.create_task(
                    self.cm.get_candles(sym, start_ts=start_minute, end_ts=end_minute, strict=False)
                )
                for sym in symbols
            }
            for sym, task in tasks.items():
                try:
                    arr = await task
                except Exception as exc:
                    logging.error(f"error fetching candles for {sym} {exc}")
                    arr = np.empty((0,), dtype=CANDLE_DTYPE)
                price_lookup[sym] = {
                    int(row["ts"]): float(row["c"]) for row in arr if float(row["c"]) > 0.0
                }
        else:
            price_lookup = {sym: {} for sym in symbols}

        positions: Dict[str, Dict[str, Dict[str, float]]] = {}
        active_symbols: set[str] = set()
        timeline: List[Dict[str, float]] = []
        missing_price_symbols: set[str] = set()

        def _apply_event(evt: dict):
            slot = _ensure_slot(positions, evt["symbol"])[evt["pside"]]
            qty = evt["qty"]
            price = evt["price"]
            if evt["action"] == "increase":
                old_size = slot["size"]
                new_size = old_size + qty
                if new_size <= 0.0:
                    slot["size"], slot["price"] = 0.0, 0.0
                elif old_size <= 0.0:
                    slot["size"], slot["price"] = new_size, price
                else:
                    slot["price"] = max((old_size * slot["price"] + qty * price) / new_size, 0.0)
                    slot["size"] = new_size
            else:
                slot["size"] = max(slot["size"] - qty, 0.0)
                if slot["size"] <= 0.0:
                    slot["price"] = 0.0
            has_pos = slot["size"] > 1e-12
            if has_pos:
                active_symbols.add(evt["symbol"])
            elif not any(positions[evt["symbol"]][ps]["size"] > 1e-12 for ps in ("long", "short")):
                active_symbols.discard(evt["symbol"])

        balance = baseline_balance
        event_idx = 0
        last_price: Dict[str, float] = {}

        minute = start_minute
        while minute <= end_minute:
            boundary = minute + ONE_MIN_MS
            while event_idx < len(events) and events[event_idx]["timestamp"] < boundary:
                evt = events[event_idx]
                _apply_event(evt)
                balance += evt["pnl"] + evt.get("fee", 0.0)
                event_idx += 1
            upnl = 0.0
            for symbol in list(active_symbols):
                price = price_lookup.get(symbol, {}).get(minute)
                if price is None:
                    price = last_price.get(symbol)
                else:
                    last_price[symbol] = price
                if price is None or price <= 0.0:
                    missing_price_symbols.add(symbol)
                    continue
                slot = positions.get(symbol)
                if not slot:
                    continue
                for pside in ("long", "short"):
                    size = slot[pside]["size"]
                    if size <= 0.0:
                        continue
                    avg_price = slot[pside]["price"]
                    if avg_price <= 0.0:
                        continue
                    c_mult = self.c_mults.get(symbol, 1.0)
                    upnl += calc_pnl(pside, avg_price, price, size, self.inverse, c_mult)
            if minute >= record_start_minute:
                timeline.append(
                    {
                        "timestamp": minute,
                        "balance": balance,
                        "equity": balance + upnl,
                        "unrealized_pnl": upnl,
                        "realized_pnl": balance - baseline_balance,
                    }
                )
            minute += ONE_MIN_MS

        if not timeline:
            point = {
                "timestamp": ts_now,
                "balance": balance_now,
                "equity": balance_now,
                "unrealized_pnl": 0.0,
                "realized_pnl": 0.0,
            }
            timeline = [point]

        balances = [{"timestamp": row["timestamp"], "balance": row["balance"]} for row in timeline]
        equities = [
            {
                "timestamp": row["timestamp"],
                "equity": row["equity"],
                "unrealized_pnl": row["unrealized_pnl"],
            }
            for row in timeline
        ]
        metadata = {
            "lookback_days": lookback_days,
            "resolution_ms": ONE_MIN_MS,
            "events_used": len(events),
            "symbols_covered": sorted(symbols),
            "missing_price_symbols": sorted(missing_price_symbols),
        }
        return {
            "timeline": timeline,
            "balances": balances,
            "equities": equities,
            "metadata": metadata,
        }

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
                        self.log_order_action(
                            order, "missing order", "fetch_open_orders", level=logging.INFO
                        )
                    else:
                        self.log_order_action(
                            order, "removed order", "fetch_open_orders", level=logging.DEBUG
                        )
            if len(added_orders) > 20:
                logging.info(f"[order] added {len(added_orders)} new orders")
            else:
                for order in added_orders:
                    self.log_order_action(
                        order, "added order", "fetch_open_orders", level=logging.DEBUG
                    )
            self.open_orders = {}
            for elm in open_orders:
                if elm["symbol"] not in self.open_orders:
                    self.open_orders[elm["symbol"]] = []
                self.open_orders[elm["symbol"]].append(elm)
            if schedule_update_positions:
                await asyncio.sleep(1.5)
                await self.update_positions_and_balance()
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
                action = "    new"
            elif new["size"] == 0.0:
                action = " closed"
            elif new["size"] > old["size"]:
                action = "  added"
            elif new["size"] < old["size"]:
                action = "reduced"
            else:
                action = "unknown"

            # Compute metrics for new pos
            wallet_exposure = (
                pbr.qty_to_cost(new["size"], new["price"], self.c_mults[symbol]) / self.balance
                if new["size"] != 0 and self.balance > 0
                else 0.0
            )
            wel = float(self.bp(pside, "wallet_exposure_limit", symbol))
            allowance_pct = float(self.bp(pside, "risk_we_excess_allowance_pct", symbol))
            effective_wel = wel * (1.0 + max(0.0, allowance_pct))
            WE_ratio = wallet_exposure / effective_wel if effective_wel > 0.0 else 0.0

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

            coin = symbol_to_coin(symbol, verbose=False) or symbol
            table.add_row(
                [
                    action + " ",
                    coin + " ",
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

        # Print aligned table with [pos] prefix
        for line in table.get_string().splitlines():
            logging.info("[pos] %s", line)

    async def _fetch_and_apply_positions(self):
        """Fetch raw positions, apply them to local state and return snapshots.

        Returns:
            Tuple of (success: bool, old_positions, new_positions).

        Raises:
            Exception: On API errors (caller handles via restart_bot_on_too_many_errors).
        """
        if not hasattr(self, "positions"):
            self.positions = {}
        res = await self.fetch_positions()
        if res is None:
            return False, None, None
        positions_list_new = res
        fetched_positions_old = deepcopy(self.fetched_positions)
        self.fetched_positions = positions_list_new
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
        return True, fetched_positions_old, self.fetched_positions

    async def update_positions(self, *, log_changes: bool = True):
        """Fetch positions, update local caches, and optionally log any changes."""
        ok, fetched_positions_old, fetched_positions_new = await self._fetch_and_apply_positions()
        if not ok:
            return False
        if log_changes and fetched_positions_old is not None:
            try:
                await self.log_position_changes(fetched_positions_old, fetched_positions_new)
            except Exception as e:
                logging.error(f"error logging position changes {e}")
        return True

    async def update_balance(self):
        """Fetch and apply the latest wallet balance.

        Returns:
            bool: True on success, False if balance_override is used but invalid.

        Raises:
            Exception: On API errors (caller handles via restart_bot_on_too_many_errors).
        """
        if not hasattr(self, "balance_override"):
            self.balance_override = None
        if not hasattr(self, "_balance_override_logged"):
            self._balance_override_logged = False
        if not hasattr(self, "previous_hysteresis_balance"):
            self.previous_hysteresis_balance = None
        if not hasattr(self, "balance_hysteresis_snap_pct"):
            self.balance_hysteresis_snap_pct = 0.02

        if self.balance_override is not None:
            balance = float(self.balance_override)
            if not self._balance_override_logged:
                logging.info("Using balance override: %.6f", balance)
                self._balance_override_logged = True
        else:
            if not hasattr(self, "fetch_balance"):
                logging.debug("update_balance: no fetch_balance implemented")
                return False
            balance = await self.fetch_balance()

        # Only accept numeric balances; keep previous value on failure
        if balance is None:
            logging.warning("balance fetch returned None; keeping previous balance")
            return False
        try:
            balance = float(balance)
        except (TypeError, ValueError):
            logging.warning("non-numeric balance fetch result; keeping previous balance")
            return False

        if self.balance_override is None:
            if self.previous_hysteresis_balance is None:
                self.previous_hysteresis_balance = balance
            balance = pbr.hysteresis(
                balance, self.previous_hysteresis_balance, self.balance_hysteresis_snap_pct
            )
            self.previous_hysteresis_balance = balance
        self.balance = balance
        return True

    async def update_positions_and_balance(self):
        """Convenience helper to refresh both positions and balance concurrently."""
        balance_task = asyncio.create_task(self.update_balance())
        positions_ok, fetched_positions_old, fetched_positions_new = (
            await self._fetch_and_apply_positions()
        )
        balance_ok = await balance_task
        if positions_ok and fetched_positions_old is not None:
            try:
                await self.log_position_changes(fetched_positions_old, fetched_positions_new)
            except Exception as e:
                logging.error(f"error logging position changes {e}")
        if balance_ok and positions_ok:
            await self.handle_balance_update(source="REST")
        return balance_ok, positions_ok

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

    async def calc_ideal_orders(self):
        """Compute desired entry and exit orders for every active symbol."""
        return await self.calc_ideal_orders_orchestrator()

    def _bot_params_to_rust_dict(self, pside: str, symbol: str | None) -> dict:
        """Build a dict matching Rust `BotParams` for JSON orchestrator input."""
        # Values which are configured globally (not per symbol) live under bot_value.
        global_keys = {
            "n_positions",
            "total_wallet_exposure_limit",
            "risk_twel_enforcer_threshold",
            "unstuck_loss_allowance_pct",
        }
        # Maintain 1:1 field coverage with `passivbot-rust/src/types.rs BotParams`.
        fields = [
            "close_grid_markup_end",
            "close_grid_markup_start",
            "close_grid_qty_pct",
            "close_trailing_retracement_pct",
            "close_trailing_grid_ratio",
            "close_trailing_qty_pct",
            "close_trailing_threshold_pct",
            "entry_grid_double_down_factor",
            "entry_grid_spacing_volatility_weight",
            "entry_grid_spacing_we_weight",
            "entry_grid_spacing_pct",
            "entry_volatility_ema_span_hours",
            "entry_initial_ema_dist",
            "entry_initial_qty_pct",
            "entry_trailing_double_down_factor",
            "entry_trailing_retracement_pct",
            "entry_trailing_retracement_we_weight",
            "entry_trailing_retracement_volatility_weight",
            "entry_trailing_grid_ratio",
            "entry_trailing_threshold_pct",
            "entry_trailing_threshold_we_weight",
            "entry_trailing_threshold_volatility_weight",
            "filter_volatility_ema_span",
            "filter_volatility_drop_pct",
            "filter_volume_ema_span",
            "filter_volume_drop_pct",
            "ema_span_0",
            "ema_span_1",
            "n_positions",
            "total_wallet_exposure_limit",
            "wallet_exposure_limit",
            "risk_wel_enforcer_threshold",
            "risk_twel_enforcer_threshold",
            "risk_we_excess_allowance_pct",
            "unstuck_close_pct",
            "unstuck_ema_dist",
            "unstuck_loss_allowance_pct",
            "unstuck_threshold",
        ]
        out: dict[str, float | int] = {}
        for key in fields:
            if key in global_keys:
                val = self.bot_value(pside, key)
            else:
                val = self.bp(pside, key, symbol) if symbol is not None else self.bp(pside, key)
            if key == "n_positions":
                out[key] = int(round(val or 0.0))
            else:
                out[key] = float(val or 0.0)
        return out

    def _pb_mode_to_orchestrator_mode(self, mode: str) -> str:
        m = (mode or "").strip().lower()
        if m in {"normal", "panic", "graceful_stop", "tp_only", "manual"}:
            return m
        return "manual"

    def _calc_unstuck_allowances_live(self, allow_new_unstuck: bool) -> dict[str, float]:
        if not allow_new_unstuck or len(getattr(self, "pnls", [])) == 0:
            return {"long": 0.0, "short": 0.0}
        pnls_cumsum = np.array([x["pnl"] for x in self.pnls]).cumsum()
        pnls_cumsum_max, pnls_cumsum_last = pnls_cumsum.max(), pnls_cumsum[-1]
        out = {}
        for pside in ["long", "short"]:
            pct = float(self.bot_value(pside, "unstuck_loss_allowance_pct") or 0.0)
            if pct > 0.0:
                out[pside] = float(
                    pbr.calc_auto_unstuck_allowance(
                        float(self.balance),
                        pct * float(self.bot_value(pside, "total_wallet_exposure_limit") or 0.0),
                        float(pnls_cumsum_max),
                        float(pnls_cumsum_last),
                    )
                )
            else:
                out[pside] = 0.0
        return out

    async def calc_ideal_orders_orchestrator_from_snapshot(
        self, snapshot: dict, *, return_snapshot: bool
    ):
        symbols = snapshot["symbols"]
        last_prices = snapshot["last_prices"]
        m1_close_emas = snapshot["m1_close_emas"]
        m1_volume_emas = snapshot["m1_volume_emas"]
        m1_log_range_emas = snapshot["m1_log_range_emas"]
        h1_log_range_emas = snapshot["h1_log_range_emas"]

        unstuck_allowances = snapshot.get("unstuck_allowances", {"long": 0.0, "short": 0.0})

        global_bp = {
            "long": self._bot_params_to_rust_dict("long", None),
            "short": self._bot_params_to_rust_dict("short", None),
        }
        # Effective hedge_mode = config setting AND exchange capability.
        # If either is False, we block same-coin hedging in the orchestrator.
        effective_hedge_mode = self._config_hedge_mode and self.hedge_mode
        input_dict = {
            "balance": float(self.balance),
            "global": {
                "filter_by_min_effective_cost": bool(self.live_value("filter_by_min_effective_cost")),
                "unstuck_allowance_long": float(unstuck_allowances.get("long", 0.0)),
                "unstuck_allowance_short": float(unstuck_allowances.get("short", 0.0)),
                "sort_global": True,
                "global_bot_params": global_bp,
                "hedge_mode": effective_hedge_mode,
            },
            "symbols": [],
            "peek_hints": None,
        }

        symbol_to_idx: dict[str, int] = {s: i for i, s in enumerate(symbols)}
        idx_to_symbol: dict[int, str] = {i: s for s, i in symbol_to_idx.items()}

        for symbol in symbols:
            idx = symbol_to_idx[symbol]
            mprice = float(last_prices.get(symbol, 0.0))
            if not math.isfinite(mprice) or mprice <= 0.0:
                raise Exception(f"invalid market price for {symbol}: {mprice}")

            active = bool(self.markets_dict.get(symbol, {}).get("active", True))
            effective_min_cost = float(
                getattr(self, "effective_min_cost", {}).get(symbol, 0.0) or 0.0
            )
            if effective_min_cost <= 0.0:
                effective_min_cost = float(
                    max(
                        pbr.qty_to_cost(self.min_qtys[symbol], mprice, self.c_mults[symbol]),
                        self.min_costs[symbol],
                    )
                )

            def side_input(pside: str) -> dict:
                mode = self._pb_mode_to_orchestrator_mode(
                    self.PB_modes.get(pside, {}).get(symbol, "manual")
                )
                pos = self.positions.get(symbol, {}).get(pside, {"size": 0.0, "price": 0.0})
                trailing = self.trailing_prices.get(symbol, {}).get(pside)
                if not trailing:
                    trailing = _trailing_bundle_default_dict()
                else:
                    trailing = dict(trailing)
                return {
                    "mode": mode,
                    "position": {"size": float(pos["size"]), "price": float(pos["price"])},
                    "trailing": {
                        "min_since_open": float(trailing.get("min_since_open", 0.0)),
                        "max_since_min": float(trailing.get("max_since_min", 0.0)),
                        "max_since_open": float(trailing.get("max_since_open", 0.0)),
                        "min_since_max": float(trailing.get("min_since_max", 0.0)),
                    },
                    "bot_params": self._bot_params_to_rust_dict(pside, symbol),
                }

            m1_close_pairs = [[float(k), float(v)] for k, v in sorted(m1_close_emas[symbol].items())]
            m1_volume_pairs = [
                [float(k), float(v)] for k, v in sorted(m1_volume_emas[symbol].items())
            ]
            m1_lr_pairs = [[float(k), float(v)] for k, v in sorted(m1_log_range_emas[symbol].items())]
            h1_lr_pairs = [[float(k), float(v)] for k, v in sorted(h1_log_range_emas[symbol].items())]

            input_dict["symbols"].append(
                {
                    "symbol_idx": int(idx),
                    "order_book": {"bid": mprice, "ask": mprice},
                    "exchange": {
                        "qty_step": float(self.qty_steps[symbol]),
                        "price_step": float(self.price_steps[symbol]),
                        "min_qty": float(self.min_qtys[symbol]),
                        "min_cost": float(self.min_costs[symbol]),
                        "c_mult": float(self.c_mults[symbol]),
                    },
                    "tradable": bool(active),
                    "next_candle": None,
                    "effective_min_cost": float(effective_min_cost),
                    "emas": {
                        "m1": {
                            "close": m1_close_pairs,
                            "log_range": m1_lr_pairs,
                            "volume": m1_volume_pairs,
                        },
                        "h1": {"close": [], "log_range": h1_lr_pairs, "volume": []},
                    },
                    "long": side_input("long"),
                    "short": side_input("short"),
                }
            )

        out_json = pbr.compute_ideal_orders_json(json.dumps(input_dict))
        out = json.loads(out_json)
        orders = out.get("orders", [])

        ideal_orders: dict[str, list] = {}
        for o in orders:
            symbol = idx_to_symbol.get(int(o["symbol_idx"]))
            if symbol is None:
                continue
            order_type = str(o["order_type"])
            order_type_id = int(pbr.order_type_snake_to_id(order_type))
            tup = (float(o["qty"]), float(o["price"]), order_type, order_type_id)
            ideal_orders.setdefault(symbol, []).append(tup)

        ideal_orders_f, _wel_blocked = self._to_executable_orders(ideal_orders, last_prices)
        ideal_orders_f = self._finalize_reduce_only_orders(ideal_orders_f, last_prices)

        if return_snapshot:
            snapshot_out = {
                "ts_ms": int(utc_ms()),
                "exchange": str(getattr(self, "exchange", "")),
                "user": str(self.config_get(["live", "user"]) or ""),
                "active_symbols": list(symbols),
                "orchestrator_input": input_dict,
                "orchestrator_output": out,
            }
            return ideal_orders_f, snapshot_out
        return ideal_orders_f, None

    async def _load_orchestrator_ema_bundle(
        self, symbols: list[str], modes: dict[str, dict[str, str]]
    ) -> tuple[
        dict[str, dict[float, float]],
        dict[str, dict[float, float]],
        dict[str, dict[float, float]],
        dict[str, dict[float, float]],
        dict[str, float],
        dict[str, float],
    ]:
        """Fetch the EMA values required by the Rust orchestrator for the given symbols.

        Returns:
        - m1_close_emas[symbol][span] = ema_close
        - m1_volume_emas[symbol][span] = ema_quote_volume
        - m1_log_range_emas[symbol][span] = ema_log_range (1m)
        - h1_log_range_emas[symbol][span] = ema_log_range (1h)
        - volumes_long[symbol], log_ranges_long[symbol] (for convenience)
        """
        # Determine which symbols/psides need full EMA context.
        need_close_spans: dict[str, set[float]] = {s: set() for s in symbols}
        need_h1_lr_spans: dict[str, set[float]] = {s: set() for s in symbols}

        for pside in ["long", "short"]:
            for symbol in symbols:
                mode = self._pb_mode_to_orchestrator_mode(modes.get(pside, {}).get(symbol, "manual"))
                has_pos = self.has_position(pside, symbol)
                if mode == "panic":
                    continue
                if mode == "manual" and not has_pos:
                    continue
                span0 = float(self.bp(pside, "ema_span_0", symbol))
                span1 = float(self.bp(pside, "ema_span_1", symbol))
                span2 = float((span0 * span1) ** 0.5) if span0 > 0.0 and span1 > 0.0 else 0.0
                for sp in (span0, span1, span2):
                    if sp > 0.0 and math.isfinite(sp):
                        need_close_spans[symbol].add(sp)
                h1_span = float(self.bp(pside, "entry_volatility_ema_span_hours", symbol) or 0.0)
                if h1_span > 0.0 and math.isfinite(h1_span):
                    need_h1_lr_spans[symbol].add(h1_span)

        # Forager metrics use global spans (per side); include them for all symbols.
        vol_span_long = float(self.bot_value("long", "filter_volume_ema_span") or 0.0)
        lr_span_long = float(self.bot_value("long", "filter_volatility_ema_span") or 0.0)
        vol_span_short = float(self.bot_value("short", "filter_volume_ema_span") or 0.0)
        lr_span_short = float(self.bot_value("short", "filter_volatility_ema_span") or 0.0)
        m1_volume_spans = sorted(
            {s for s in (vol_span_long, vol_span_short) if s > 0.0 and math.isfinite(s)}
        )
        m1_lr_spans = sorted(
            {s for s in (lr_span_long, lr_span_short) if s > 0.0 and math.isfinite(s)}
        )

        async def fetch_map(symbol: str, spans: list[float], fn):
            out: dict[float, float] = {}
            if not spans:
                return out
            tasks = [asyncio.create_task(fn(symbol, sp)) for sp in spans]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for sp, res in zip(spans, results):
                if isinstance(res, Exception):
                    continue
                val = float(res)
                if math.isfinite(val):
                    out[float(sp)] = val
            return out

        async def ema_close(symbol: str, span: float) -> float:
            return float(await self.cm.get_latest_ema_close(symbol, span=span, max_age_ms=30_000))

        async def ema_qv(symbol: str, span: float) -> float:
            return float(
                await self.cm.get_latest_ema_quote_volume(symbol, span=span, max_age_ms=60_000)
            )

        async def ema_lr_1m(symbol: str, span: float) -> float:
            return float(await self.cm.get_latest_ema_log_range(symbol, span=span, max_age_ms=60_000))

        async def ema_lr_1h(symbol: str, span: float) -> float:
            return float(
                await self.cm.get_latest_ema_log_range(symbol, span=span, tf="1h", max_age_ms=600_000)
            )

        close_tasks = {
            sym: asyncio.create_task(fetch_map(sym, sorted(need_close_spans[sym]), ema_close))
            for sym in symbols
        }
        h1_lr_tasks = {
            sym: asyncio.create_task(fetch_map(sym, sorted(need_h1_lr_spans[sym]), ema_lr_1h))
            for sym in symbols
        }
        vol_tasks = {
            sym: asyncio.create_task(fetch_map(sym, m1_volume_spans, ema_qv)) for sym in symbols
        }
        lr1m_tasks = {
            sym: asyncio.create_task(fetch_map(sym, m1_lr_spans, ema_lr_1m)) for sym in symbols
        }

        m1_close_emas: dict[str, dict[float, float]] = {}
        m1_volume_emas: dict[str, dict[float, float]] = {}
        m1_log_range_emas: dict[str, dict[float, float]] = {}
        h1_log_range_emas: dict[str, dict[float, float]] = {}
        for sym in symbols:
            m1_close_emas[sym] = await close_tasks[sym]
            h1_log_range_emas[sym] = await h1_lr_tasks[sym]
            m1_volume_emas[sym] = await vol_tasks[sym]
            m1_log_range_emas[sym] = await lr1m_tasks[sym]

        # Convenience: compute the single-span values used by legacy forager logging.
        volumes_long = {s: m1_volume_emas[s].get(vol_span_long, 0.0) for s in symbols}
        log_ranges_long = {s: m1_log_range_emas[s].get(lr_span_long, 0.0) for s in symbols}

        return (
            m1_close_emas,
            m1_volume_emas,
            m1_log_range_emas,
            h1_log_range_emas,
            volumes_long,
            log_ranges_long,
        )

    async def calc_ideal_orders_orchestrator(self, *, return_snapshot: bool = False):
        """Compute desired orders using Rust orchestrator (JSON API)."""
        # Use the same symbol universe as legacy live path (pre-selected in execution_cycle).
        symbols = sorted(set(getattr(self, "active_symbols", []) or []))
        if not symbols:
            return ({}, None) if return_snapshot else {}

        last_prices = await self.cm.get_last_prices(symbols, max_age_ms=10_000)

        # Ensure effective min cost is up to date.
        if not hasattr(self, "effective_min_cost") or not self.effective_min_cost:
            await self.update_effective_min_cost()

        (
            m1_close_emas,
            m1_volume_emas,
            m1_log_range_emas,
            h1_log_range_emas,
            _volumes_long,
            _log_ranges_long,
        ) = await self._load_orchestrator_ema_bundle(symbols, self.PB_modes)

        unstuck_allowances = self._calc_unstuck_allowances_live(
            allow_new_unstuck=not self.has_open_unstuck_order()
        )

        global_bp = {
            "long": self._bot_params_to_rust_dict("long", None),
            "short": self._bot_params_to_rust_dict("short", None),
        }
        # Effective hedge_mode = config setting AND exchange capability.
        # If either is False, we block same-coin hedging in the orchestrator.
        effective_hedge_mode = self._config_hedge_mode and self.hedge_mode
        input_dict = {
            "balance": float(self.balance),
            "global": {
                "filter_by_min_effective_cost": bool(self.live_value("filter_by_min_effective_cost")),
                "unstuck_allowance_long": float(unstuck_allowances.get("long", 0.0)),
                "unstuck_allowance_short": float(unstuck_allowances.get("short", 0.0)),
                "sort_global": True,
                "global_bot_params": global_bp,
                "hedge_mode": effective_hedge_mode,
            },
            "symbols": [],
            "peek_hints": None,
        }

        symbol_to_idx: dict[str, int] = {s: i for i, s in enumerate(symbols)}
        idx_to_symbol: dict[int, str] = {i: s for s, i in symbol_to_idx.items()}

        for symbol in symbols:
            idx = symbol_to_idx[symbol]
            mprice = float(last_prices.get(symbol, 0.0))
            if not math.isfinite(mprice) or mprice <= 0.0:
                raise Exception(f"invalid market price for {symbol}: {mprice}")

            active = bool(self.markets_dict.get(symbol, {}).get("active", True))
            effective_min_cost = float(self.effective_min_cost.get(symbol, 0.0) or 0.0)
            if effective_min_cost <= 0.0:
                effective_min_cost = float(
                    max(
                        pbr.qty_to_cost(self.min_qtys[symbol], mprice, self.c_mults[symbol]),
                        self.min_costs[symbol],
                    )
                )

            def side_input(pside: str) -> dict:
                mode = self._pb_mode_to_orchestrator_mode(
                    self.PB_modes.get(pside, {}).get(symbol, "manual")
                )
                pos = self.positions.get(symbol, {}).get(pside, {"size": 0.0, "price": 0.0})
                trailing = self.trailing_prices.get(symbol, {}).get(pside)
                if not trailing:
                    trailing = _trailing_bundle_default_dict()
                else:
                    trailing = dict(trailing)
                return {
                    "mode": mode,
                    "position": {"size": float(pos["size"]), "price": float(pos["price"])},
                    "trailing": {
                        "min_since_open": float(trailing.get("min_since_open", 0.0)),
                        "max_since_min": float(trailing.get("max_since_min", 0.0)),
                        "max_since_open": float(trailing.get("max_since_open", 0.0)),
                        "min_since_max": float(trailing.get("min_since_max", 0.0)),
                    },
                    "bot_params": self._bot_params_to_rust_dict(pside, symbol),
                }

            # Build EMA bundle for this symbol.
            m1_close_pairs = [[float(k), float(v)] for k, v in sorted(m1_close_emas[symbol].items())]
            m1_volume_pairs = [
                [float(k), float(v)] for k, v in sorted(m1_volume_emas[symbol].items())
            ]
            m1_lr_pairs = [[float(k), float(v)] for k, v in sorted(m1_log_range_emas[symbol].items())]
            h1_lr_pairs = [[float(k), float(v)] for k, v in sorted(h1_log_range_emas[symbol].items())]

            input_dict["symbols"].append(
                {
                    "symbol_idx": int(idx),
                    "order_book": {"bid": mprice, "ask": mprice},
                    "exchange": {
                        "qty_step": float(self.qty_steps[symbol]),
                        "price_step": float(self.price_steps[symbol]),
                        "min_qty": float(self.min_qtys[symbol]),
                        "min_cost": float(self.min_costs[symbol]),
                        "c_mult": float(self.c_mults[symbol]),
                    },
                    "tradable": bool(active),
                    "next_candle": None,
                    "effective_min_cost": float(effective_min_cost),
                    "emas": {
                        "m1": {
                            "close": m1_close_pairs,
                            "log_range": m1_lr_pairs,
                            "volume": m1_volume_pairs,
                        },
                        "h1": {"close": [], "log_range": h1_lr_pairs, "volume": []},
                    },
                    "long": side_input("long"),
                    "short": side_input("short"),
                }
            )

        out_json = pbr.compute_ideal_orders_json(json.dumps(input_dict))
        out = json.loads(out_json)
        orders = out.get("orders", [])

        ideal_orders: dict[str, list] = {}
        for o in orders:
            symbol = idx_to_symbol.get(int(o["symbol_idx"]))
            if symbol is None:
                continue
            order_type = str(o["order_type"])
            order_type_id = int(pbr.order_type_snake_to_id(order_type))
            tup = (float(o["qty"]), float(o["price"]), order_type, order_type_id)
            ideal_orders.setdefault(symbol, []).append(tup)

        ideal_orders_f, _wel_blocked = self._to_executable_orders(ideal_orders, last_prices)
        ideal_orders_f = self._finalize_reduce_only_orders(ideal_orders_f, last_prices)

        if return_snapshot:
            snapshot = {
                "ts_ms": int(utc_ms()),
                "exchange": str(getattr(self, "exchange", "")),
                "user": str(self.config_get(["live", "user"]) or ""),
                "active_symbols": list(symbols),
                "orchestrator_input": input_dict,
                "orchestrator_output": out,
            }
            return ideal_orders_f, snapshot
        return ideal_orders_f

    def _to_executable_orders(
        self, ideal_orders: dict, last_prices: Dict[str, float]
    ) -> tuple[Dict[str, list], set[str]]:
        """Convert raw order tuples into api-ready dicts and find WEL-restricted symbols."""
        ideal_orders_f: Dict[str, list] = {}
        wel_blocked_symbols: set[str] = set()

        for symbol, orders in ideal_orders.items():
            ideal_orders_f[symbol] = []
            last_mprice = last_prices[symbol]
            seen = set()
            with_mprice_diff = []
            for order in orders:
                side = determine_side_from_order_tuple(order)
                diff = order_market_diff(side, order[1], last_mprice)
                with_mprice_diff.append((diff, order, side))
                if (
                    isinstance(order, tuple)
                    and isinstance(order[2], str)
                    and "close_auto_reduce_wel" in order[2]
                ):
                    wel_blocked_symbols.add(symbol)
            any_partial = any("partial" in order[2] for _, order, _ in with_mprice_diff)
            for mprice_diff, order, order_side in sorted(with_mprice_diff, key=lambda item: item[0]):
                position_side = "long" if "long" in order[2] else "short"
                if order[0] == 0.0:
                    continue
                if mprice_diff > float(self.live_value("price_distance_threshold")):
                    if any_partial and "entry" in order[2]:
                        logging.debug(
                            "gated by price_distance_threshold (partial) | %s %s %s diff=%.5f",
                            symbol,
                            position_side,
                            order[2],
                            mprice_diff,
                        )
                        continue
                    if any(token in order[2] for token in ("initial", "unstuck")):
                        logging.debug(
                            "gated by price_distance_threshold (initial/unstuck) | %s %s %s diff=%.5f",
                            symbol,
                            position_side,
                            order[2],
                            mprice_diff,
                        )
                        continue
                    if not self.has_position(position_side, symbol):
                        logging.debug(
                            "gated by price_distance_threshold (no position) | %s %s %s diff=%.5f",
                            symbol,
                            position_side,
                            order[2],
                            mprice_diff,
                        )
                        continue
                seen_key = str(abs(order[0])) + str(order[1]) + order[2]
                if seen_key in seen:
                    logging.debug("duplicate ideal order for %s skipped: %s", symbol, order)
                    continue
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
        return self._finalize_reduce_only_orders(ideal_orders_f, last_prices), wel_blocked_symbols

    def _finalize_reduce_only_orders(
        self, orders_by_symbol: Dict[str, list], last_prices: Dict[str, float]
    ) -> Dict[str, list]:
        """Bound reduce-only quantities so they never exceed the current position size (per order and in sum)."""
        for symbol, orders in orders_by_symbol.items():
            market_price = float(last_prices.get(symbol, 0.0))

            # 1) clamp each reduce-only order to position size
            for order in orders:
                if not order.get("reduce_only"):
                    continue
                pos = self.positions.get(order["symbol"], {}).get(order["position_side"], {})
                pos_size_abs = abs(float(pos.get("size", 0.0)))
                if abs(order["qty"]) > pos_size_abs:
                    logging.warning(
                        "trimmed reduce-only qty to position size | order=%s | position=%s",
                        order,
                        pos,
                    )
                    order["qty"] = pos_size_abs

            # 2) cap sum(reduce_only qty) <= pos size by reducing furthest-from-market closes first
            for pside in ("long", "short"):
                pos_size_abs = abs(
                    float(self.positions.get(symbol, {}).get(pside, {}).get("size", 0.0))
                )
                if pos_size_abs <= 0.0:
                    continue
                ro = [o for o in orders if o.get("reduce_only") and o.get("position_side") == pside]
                if not ro:
                    continue
                total = sum(float(o.get("qty", 0.0)) for o in ro)
                if total <= pos_size_abs + 1e-12:
                    continue
                excess = total - pos_size_abs
                # furthest first: larger order_market_diff
                ro_sorted = sorted(
                    ro,
                    key=lambda o: order_market_diff(
                        o.get("side", ""), float(o.get("price", 0.0)), market_price
                    ),
                    reverse=True,
                )
                for o in ro_sorted:
                    if excess <= 0.0:
                        break
                    q = float(o.get("qty", 0.0))
                    if q <= 0.0:
                        continue
                    reduce_by = min(q, excess)
                    new_q = q - reduce_by
                    o["qty"] = float(round(new_q, 12))
                    excess -= reduce_by
                # drop any zeroed reduce-only orders
                orders_by_symbol[symbol] = [
                    o
                    for o in orders_by_symbol[symbol]
                    if not (o.get("reduce_only") and float(o.get("qty", 0.0)) <= 0.0)
                ]

        return orders_by_symbol

    async def calc_orders_to_cancel_and_create(self):
        """Determine which existing orders to cancel and which new ones to place."""
        if not hasattr(self, "_last_plan_detail"):
            self._last_plan_detail = {}
        ideal_orders = await self.calc_ideal_orders()

        actual_orders = self._snapshot_actual_orders()
        keys = ("symbol", "side", "position_side", "qty", "price")
        to_cancel, to_create = [], []
        plan_summaries = []
        for symbol, symbol_orders in actual_orders.items():
            ideal_list = ideal_orders.get(symbol, []) if isinstance(ideal_orders, dict) else []
            cancel_, create_ = self._reconcile_symbol_orders(symbol, symbol_orders, ideal_list, keys)
            cancel_, create_ = self._annotate_order_deltas(cancel_, create_)
            pre_cancel = len(cancel_)
            pre_create = len(create_)
            cancel_, create_, skipped = self._apply_order_match_tolerance(cancel_, create_)
            plan_summaries.append(
                (symbol, pre_cancel, len(cancel_), pre_create, len(create_), skipped)
            )
            to_cancel += cancel_
            to_create += create_

        to_cancel = await self._sort_orders_by_market_diff(to_cancel, "to_cancel")
        to_create = await self._sort_orders_by_market_diff(to_create, "to_create")
        if plan_summaries:
            total_pre_cancel = sum(p[1] for p in plan_summaries)
            total_cancel = sum(p[2] for p in plan_summaries)
            total_pre_create = sum(p[3] for p in plan_summaries)
            total_create = sum(p[4] for p in plan_summaries)
            total_skipped = sum(p[5] for p in plan_summaries)
            detail_parts = []
            untouched_cancel = total_pre_cancel - total_cancel
            untouched_create = total_pre_create - total_create
            for symbol, pre_c, c, pre_cr, cr, skipped in plan_summaries:
                prev = self._last_plan_detail.get(symbol)
                current = (c, cr, skipped)
                self._last_plan_detail[symbol] = current
                if c or cr or skipped:
                    if prev != current:
                        detail_parts.append(f"{symbol}:c{pre_c}->{c} cr{pre_cr}->{cr} skip{skipped}")
            detail = " | ".join(detail_parts[:6])
            summary_key = (
                total_pre_cancel,
                total_cancel,
                total_pre_create,
                total_create,
                total_skipped,
                untouched_cancel,
                untouched_create,
                detail,
            )
            if summary_key != getattr(self, "_last_order_plan_summary", None):
                self._last_order_plan_summary = summary_key
                if total_cancel or total_create or total_skipped:
                    extra = []
                    if untouched_cancel:
                        extra.append(f"unchanged_cancel={untouched_cancel}")
                    if untouched_create:
                        extra.append(f"unchanged_create={untouched_create}")
                    logging.info(
                        "order plan summary | cancel %d->%d | create %d->%d | skipped=%d%s%s",
                        total_pre_cancel,
                        total_cancel,
                        total_pre_create,
                        total_create,
                        total_skipped,
                        f" | {' '.join(extra)}" if extra else "",
                        f" | details: {detail}" if detail else "",
                    )
        return to_cancel, to_create

    def _snapshot_actual_orders(self) -> dict[str, list[dict]]:
        """Return a normalized snapshot of currently open orders keyed by symbol."""
        actual_orders: dict[str, list[dict]] = {}
        for symbol in self.active_symbols:
            symbol_orders = []
            for order in self.open_orders.get(symbol, []):
                try:
                    symbol_orders.append(
                        {
                            "symbol": order["symbol"],
                            "side": order["side"],
                            "position_side": order["position_side"],
                            "qty": abs(order["qty"]),
                            "price": order["price"],
                            "reduce_only": (
                                order["position_side"] == "long" and order["side"] == "sell"
                            )
                            or (order["position_side"] == "short" and order["side"] == "buy"),
                            "id": order.get("id"),
                            "custom_id": order.get("custom_id"),
                        }
                    )
                except Exception as exc:
                    logging.error(f"error in calc_orders_to_cancel_and_create {exc}")
                    traceback.print_exc()
                    print(order)
            actual_orders[symbol] = symbol_orders
        return actual_orders

    def _reconcile_symbol_orders(
        self,
        symbol: str,
        actual_orders: list[dict],
        ideal_orders: list,
        keys: tuple[str, ...],
    ) -> tuple[list[dict], list[dict]]:
        """Return cancel/create lists for a single symbol after mode filtering."""
        to_cancel, to_create = filter_orders(actual_orders, ideal_orders, keys)
        to_cancel, to_create = self._apply_mode_filters(symbol, to_cancel, to_create)
        return to_cancel, to_create

    def _annotate_order_deltas(
        self, to_cancel: list[dict], to_create: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """
        Attach best-effort delta info between existing and desired orders to aid logging.

        Matches orders by symbol/side/position_side and closest price distance.
        """
        remaining_create = list(to_create)
        for order in to_create:
            order.setdefault("_context", "new")
            order.setdefault("_reason", "new")
        for cancel_order in to_cancel:
            cancel_order.setdefault("_context", "retire")
            cancel_order.setdefault("_reason", "retire")

        def pct(a: float, b: float) -> float:
            if a == 0 and b == 0:
                return 0.0
            if a == 0:
                return float("inf")
            return abs(b - a) / abs(a) * 100.0

        # annotate cancellations
        for cancel_order in to_cancel:
            candidates = [
                (idx, co)
                for idx, co in enumerate(remaining_create)
                if co.get("symbol") == cancel_order.get("symbol")
                and co.get("side") == cancel_order.get("side")
                and co.get("position_side") == cancel_order.get("position_side")
            ]
            if not candidates:
                continue
            # choose closest by price difference
            best_idx, best_order = min(
                candidates,
                key=lambda c: abs(
                    float(c[1].get("price", 0.0)) - float(cancel_order.get("price", 0.0))
                ),
            )
            raw_price_diff = pct(
                float(cancel_order.get("price", 0.0)), float(best_order.get("price", 0.0))
            )
            raw_qty_diff = pct(float(cancel_order.get("qty", 0.0)), float(best_order.get("qty", 0.0)))
            price_diff = round(raw_price_diff, 4) if math.isfinite(raw_price_diff) else raw_price_diff
            qty_diff = round(raw_qty_diff, 4) if math.isfinite(raw_qty_diff) else raw_qty_diff
            reason_parts = []
            if price_diff > 0:
                reason_parts.append("price")
            if qty_diff > 0:
                reason_parts.append("qty")
            reason = "+".join(reason_parts) if reason_parts else "adjustment"
            cancel_order["_delta"] = {
                "price_old": cancel_order.get("price"),
                "price_new": best_order.get("price"),
                "price_pct_diff": price_diff,
                "qty_old": cancel_order.get("qty"),
                "qty_new": best_order.get("qty"),
                "qty_pct_diff": qty_diff,
            }
            cancel_order["_context"] = "replace"
            cancel_order["_reason"] = reason
            # also annotate the matched create order
            best_order["_delta"] = {
                "price_old": cancel_order.get("price"),
                "price_new": best_order.get("price"),
                "price_pct_diff": price_diff,
                "qty_old": cancel_order.get("qty"),
                "qty_new": best_order.get("qty"),
                "qty_pct_diff": qty_diff,
            }
            best_order["_context"] = "replace"
            best_order["_reason"] = reason
            remaining_create.pop(best_idx)

        for ord in remaining_create:
            ord.setdefault("_context", "new")
            ord.setdefault("_reason", "fresh")
        return to_cancel, to_create

    def _apply_order_match_tolerance(
        self, to_cancel: list[dict], to_create: list[dict]
    ) -> tuple[list[dict], list[dict], int]:
        """Drop cancel/create pairs that are within tolerance to avoid churn.

        Returns (remaining_cancel, remaining_create, skipped_pairs)
        """
        tolerance = float(self.live_value("order_match_tolerance_pct"))
        if tolerance <= 0.0:
            return to_cancel, to_create, 0

        used_cancel: set[int] = set()
        kept_create: list[dict] = []
        skipped = 0

        def pct_diff(a: float, b: float) -> float:
            if b == 0:
                return 0.0 if a == 0 else float("inf")
            return abs(a - b) / abs(b) * 100.0

        for order in to_create:
            match_idx = None
            for idx, existing in enumerate(to_cancel):
                if idx in used_cancel:
                    continue
                try:
                    if orders_matching(
                        order,
                        existing,
                        tolerance_qty=tolerance,
                        tolerance_price=tolerance,
                    ):
                        match_idx = idx
                        break
                except Exception:
                    continue
            if match_idx is None:
                kept_create.append(order)
            else:
                used_cancel.add(match_idx)
                skipped += 1
                try:
                    price_diff = pct_diff(float(order["price"]), float(to_cancel[match_idx]["price"]))
                    qty_diff = pct_diff(float(order["qty"]), float(to_cancel[match_idx]["qty"]))
                    logging.debug(
                        "skipped_recreate | %s | tolerance=%.4f%% price_diff=%.4f%% qty_diff=%.4f%%",
                        order.get("symbol", "?"),
                        tolerance * 100.0,
                        price_diff,
                        qty_diff,
                    )
                except Exception:
                    logging.debug(
                        "skipped_recreate | %s | tolerance=%.4f%%",
                        order.get("symbol", "?"),
                        tolerance * 100.0,
                    )

        remaining_cancel = [o for i, o in enumerate(to_cancel) if i not in used_cancel]
        return remaining_cancel, kept_create, skipped

    def _apply_mode_filters(
        self,
        symbol: str,
        to_cancel: list[dict],
        to_create: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        """Apply manual/tp_only mode rules to cancel/create order lists."""
        for pside in ["long", "short"]:
            mode = self.PB_modes[pside].get(symbol)
            if mode == "manual":
                to_cancel = [x for x in to_cancel if x["position_side"] != pside]
                to_create = [x for x in to_create if x["position_side"] != pside]
            elif mode == "tp_only":
                to_cancel = [
                    x
                    for x in to_cancel
                    if (
                        x["position_side"] != pside
                        or (x["position_side"] == pside and x["reduce_only"])
                    )
                ]
                to_create = [
                    x
                    for x in to_create
                    if (
                        x["position_side"] != pside
                        or (x["position_side"] == pside and x["reduce_only"])
                    )
                ]
        return to_cancel, to_create

    async def _sort_orders_by_market_diff(self, orders: list[dict], log_label: str) -> list[dict]:
        """Return orders sorted by market diff, fetching prices concurrently."""
        if not orders:
            return []
        market_prices = await self._fetch_market_prices({order["symbol"] for order in orders})
        entries = []
        for order in orders:
            market_price = market_prices.get(order["symbol"])
            if market_price is None:
                logging.debug("price missing sort %s by mprice_diff %s", log_label, order)
                diff = 0.0
            else:
                diff = order_market_diff(order["side"], order["price"], market_price)
            entries.append((diff, order))
        entries.sort(key=lambda item: item[0])
        return [order for _, order in entries]

    async def _fetch_market_prices(self, symbols: set[str]) -> dict[str, float | None]:
        """Fetch current close prices for the supplied symbols."""
        results: dict[str, float | None] = {}
        tasks: dict[str, asyncio.Task] = {}
        for symbol in symbols:
            try:
                fetch_result = self.cm.get_current_close(symbol, max_age_ms=10_000)
                if inspect.isawaitable(fetch_result):
                    tasks[symbol] = asyncio.create_task(fetch_result)
                else:
                    results[symbol] = fetch_result
            except Exception as exc:
                logging.debug("failed fetching mprice for %s: %s", symbol, exc)
                results[symbol] = None
        for symbol, task in tasks.items():
            try:
                results[symbol] = await task
            except Exception as exc:
                logging.debug("failed fetching mprice for %s: %s", symbol, exc)
                results[symbol] = None
        return results

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

            symbols = sorted(set(self.active_symbols))
            for sym in symbols:
                try:
                    await self.cm.get_candles(
                        sym, start_ts=start_ts, end_ts=end_ts, max_age_ms=max_age_ms, strict=False
                    )
                except TimeoutError as exc:
                    logging.warning(
                        "Timed out acquiring candle lock for %s; will retry next cycle (%s)",
                        sym,
                        exc,
                    )
                except Exception as exc:
                    logging.error("error refreshing candles for %s: %s", sym, exc, exc_info=True)
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
                interval = getattr(self, "memory_snapshot_interval_ms", 3_600_000)
                if last_mem_log_ts is None or now - last_mem_log_ts >= interval:
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
        span = int(round(self.bot_value(pside, "filter_volatility_ema_span")))

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
        n = len(syms)
        started_ms = utc_ms()
        for sym, task in tasks.items():
            try:
                out[sym] = await task
            except Exception:
                out[sym] = 0.0
        elapsed_s = max(0.001, (utc_ms() - started_ms) / 1000.0)
        if out:
            top_n = min(8, len(out))
            top = sorted(out.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
            top_syms = tuple(sym for sym, _ in top)
            # Only log when the ranking changes (membership/order) to reduce noise.
            if not hasattr(self, "_log_range_top_cache"):
                self._log_range_top_cache = {}
            cache_key = (pside, span)
            last_top = self._log_range_top_cache.get(cache_key)
            if last_top != top_syms:
                self._log_range_top_cache[cache_key] = top_syms
                summary = ", ".join(f"{symbol_to_coin(sym)}={val:.6f}" for sym, val in top)
                logging.info(
                    f"log_range EMA span {span}: {n} coins elapsed={int(elapsed_s)}s, top{top_n}: {summary}"
                )
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
        n = len(syms)
        started_ms = utc_ms()
        for sym, task in tasks.items():
            try:
                out[sym] = await task
            except Exception:
                out[sym] = 0.0
        elapsed_s = max(0.001, (utc_ms() - started_ms) / 1000.0)
        if out:
            top_n = min(8, len(out))
            top = sorted(out.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
            top_syms = tuple(sym for sym, _ in top)
            if not hasattr(self, "_volume_top_cache"):
                self._volume_top_cache = {}
            cache_key = (pside, span)
            last_top = self._volume_top_cache.get(cache_key)
            if last_top != top_syms:
                self._volume_top_cache[cache_key] = top_syms
                summary = ", ".join(f"{symbol_to_coin(sym)}={val:.2f}" for sym, val in top)
                logging.info(
                    f"volume EMA span {span}: {n} coins elapsed={int(elapsed_s)}s, top{top_n}: {summary}"
                )
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
        result = {"added": {}, "removed": {}}
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
                symbols = {s for s in symbols if s}
                eligible = getattr(self, "eligible_symbols", None)
                if eligible:
                    skipped = [sym for sym in symbols if sym not in eligible]
                    if skipped:
                        coin_list = ", ".join(sorted(symbol_to_coin(sym) or sym for sym in skipped))
                        symbol_list = ", ".join(sorted(skipped))
                        warned = getattr(self, "_unsupported_coin_warnings", None)
                        if warned is None:
                            warned = set()
                            setattr(self, "_unsupported_coin_warnings", warned)
                        warn_key = (self.exchange, coin_list, symbol_list, k_coins)
                        if warn_key not in warned:
                            logging.warning(
                                "Skipping unsupported markets for %s: coins=%s symbols=%s exchange=%s",
                                k_coins,
                                coin_list,
                                symbol_list,
                                getattr(self, "exchange", "?"),
                            )
                            warned.add(warn_key)
                        symbols = symbols - set(skipped)
            symbols_already = getattr(self, k_coins)[pside]
            if symbols_already != symbols:
                added = symbols - symbols_already
                removed = symbols_already - symbols
                if added and pside in log_psides:
                    result["added"][pside] = added
                if removed and pside in log_psides:
                    result["removed"][pside] = removed
                getattr(self, k_coins)[pside] = symbols
        return result

    def refresh_approved_ignored_coins_lists(self):
        """Reload approved and ignored coin lists from config sources."""
        try:
            added_summary = {}
            removed_summary = {}
            for k in ("approved_coins", "ignored_coins"):
                if not hasattr(self, k):
                    setattr(self, k, {"long": set(), "short": set()})
                config_sources = self.config.get("_coins_sources", {})
                raw_source = config_sources.get(k, self.live_value(k))
                parsed = normalize_coins_source(raw_source)
                if k == "approved_coins":
                    log_psides = {ps for ps in parsed if self.is_pside_enabled(ps)}
                else:
                    log_psides = set(parsed.keys())
                add_res = self.add_to_coins_lists(parsed, k, log_psides=log_psides)
                if add_res:
                    added_summary.setdefault(k, {}).update(add_res.get("added", {}))
                    removed_summary.setdefault(k, {}).update(add_res.get("removed", {}))
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
            # aggregate add/remove logs for readability
            for k, summary in (("added", added_summary.get("approved_coins", {})),):
                if summary:
                    parts = []
                    for pside, coins in summary.items():
                        if coins:
                            parts.append(
                                f"{pside}: {','.join(sorted(symbol_to_coin(x) for x in coins))}"
                            )
                    if parts:
                        logging.info("added to approved_coins | %s", " | ".join(parts))
            for k, summary in (("removed", removed_summary.get("approved_coins", {})),):
                if summary:
                    parts = []
                    for pside, coins in summary.items():
                        if coins:
                            parts.append(
                                f"{pside}: {','.join(sorted(symbol_to_coin(x) for x in coins))}"
                            )
                    if parts:
                        logging.info("removed from approved_coins | %s", " | ".join(parts))
            for k, summary in (("added", added_summary.get("ignored_coins", {})),):
                if summary:
                    parts = []
                    for pside, coins in summary.items():
                        if coins:
                            parts.append(
                                f"{pside}: {','.join(sorted(symbol_to_coin(x) for x in coins))}"
                            )
                    if parts:
                        logging.info("added to ignored_coins | %s", " | ".join(parts))
            for k, summary in (("removed", removed_summary.get("ignored_coins", {})),):
                if summary:
                    parts = []
                    for pside, coins in summary.items():
                        if coins:
                            parts.append(
                                f"{pside}: {','.join(sorted(symbol_to_coin(x) for x in coins))}"
                            )
                    if parts:
                        logging.info("removed from ignored_coins | %s", " | ".join(parts))
            self._log_coin_symbol_fallback_summary()
        except Exception as e:
            logging.error(f"error with refresh_approved_ignored_coins_lists {e}")
            traceback.print_exc()

    def _log_coin_symbol_fallback_summary(self):
        """Emit a brief summary of symbol/coin mapping fallbacks (once per change)."""
        counts = coin_symbol_warning_counts()
        if counts != self._last_coin_symbol_warning_counts:
            if counts["symbol_to_coin_fallbacks"] or counts["coin_to_symbol_fallbacks"]:
                logging.info(
                    "Symbol/coin mapping fallbacks: symbol->coin=%d | coin->symbol=%d (unique)",
                    counts["symbol_to_coin_fallbacks"],
                    counts["coin_to_symbol_fallbacks"],
                )
            self._last_coin_symbol_warning_counts = dict(counts)

    def _build_order_params(self, order: dict) -> dict:
        """Hook: Build execution parameters for order placement.

        Override in subclass with exchange-specific logic.
        """
        return {}

    async def execute_order(self, order: dict) -> dict:
        """Place a single order via the exchange client."""
        params = {
            "symbol": order["symbol"],
            "type": order.get("type", "limit"),
            "side": order["side"],
            "amount": abs(order["qty"]),
            "price": order["price"],
            "params": self._build_order_params(order),
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
    elif user_info["exchange"] == "paradex":
        from exchanges.paradex import ParadexBot

        bot = ParadexBot(config)
    else:
        # Generic CCXTBot for any CCXT-supported exchange
        from exchanges.ccxt_bot import CCXTBot

        bot = CCXTBot(config)
        logging.info(
            f"Using generic CCXTBot for '{user_info['exchange']}' (no custom implementation)"
        )
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
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        help="Logging verbosity (warning, info, debug, trace or 0-3).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose (debug) logging. Equivalent to --log-level debug.",
    )
    parser.add_argument(
        "--shadow-mode",
        dest="shadow_mode",
        action="store_true",
        default=False,
        help="Enable FillEventsManager shadow mode for PnL comparison logging.",
    )

    template_config = get_template_config()
    del template_config["optimize"]
    del template_config["backtest"]
    if "logging" in template_config and isinstance(template_config["logging"], dict):
        template_config["logging"].pop("level", None)
    add_arguments_recursively(parser, template_config)
    raw_args = merge_negative_cli_values(sys.argv[1:])
    args = parser.parse_args(raw_args)
    # --verbose flag overrides --log-level to debug (level 2)
    cli_log_level = "debug" if args.verbose else args.log_level
    initial_log_level = resolve_log_level(cli_log_level, None, fallback=1)
    configure_logging(debug=initial_log_level)
    config = load_config(args.config_path, live_only=True)
    update_config_with_args(config, args, verbose=True)
    config = format_config(config, live_only=True)
    config_logging_value = get_optional_config_value(config, "logging.level", None)
    effective_log_level = resolve_log_level(cli_log_level, config_logging_value, fallback=1)
    if effective_log_level != initial_log_level:
        configure_logging(debug=effective_log_level)
    logging_section = config.get("logging")
    if not isinstance(logging_section, dict):
        logging_section = {}
    config["logging"] = logging_section
    logging_section["level"] = effective_log_level

    # --shadow-mode flag enables FillEventsManager shadow mode
    if args.shadow_mode:
        if "live" not in config or not isinstance(config["live"], dict):
            config["live"] = {}
        config["live"]["pnls_manager_shadow_mode"] = True
        logging.info("[shadow] Shadow mode enabled via CLI flag")

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
