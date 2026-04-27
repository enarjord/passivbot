import os
import sys
import argparse
from cli_utils import help_requested
from rust_utils import check_and_maybe_compile, verify_loaded_runtime_extension

# Perform Rust compilation check before importing any modules that may load the extension
_rust_parser = argparse.ArgumentParser(add_help=False)
_rust_parser.add_argument("--skip-rust-compile", action="store_true", help="Skip Rust build check.")
_rust_parser.add_argument(
    "--force-rust-compile", action="store_true", help="Force rebuild of Rust extension."
)
_rust_parser.add_argument(
    "--fail-on-stale-rust",
    action="store_true",
    help="Abort if Rust extension appears stale instead of attempting rebuild.",
)
_rust_known, _rust_remaining = _rust_parser.parse_known_args()
_help_only = help_requested(_rust_remaining)
try:
    check_and_maybe_compile(
        skip=_help_only
        or _rust_known.skip_rust_compile
        or os.environ.get("SKIP_RUST_COMPILE", "").lower() in ("1", "true", "yes"),
        force=_rust_known.force_rust_compile,
        fail_on_stale=_rust_known.fail_on_stale_rust,
    )
except Exception as exc:
    print(f"Rust extension check failed: {exc}")
    sys.exit(1)
# rebuild argv for the real parser without rust flags
sys.argv = [sys.argv[0]] + _rust_remaining

import numpy as np
import math
import pandas as pd
import json
import asyncio
import numbers
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
from cli_utils import (
    add_help_all_argument,
    build_command_parser,
    expand_help_all_argv,
    get_cli_prog,
    help_all_requested,
)
from config import (
    compile_runtime_config,
    get_template_config,
    load_input_config,
    load_prepared_config,
    prepare_config,
)
from config.access import (
    get_optional_config_value,
    get_optional_live_value,
    require_config_value,
    require_live_value,
)
from config.pnl_lookback import parse_pnls_max_lookback_days
from config.metrics import ANALYSIS_SHARED_KEYS
from config.coerce import normalize_hsl_signal_mode
from config.overrides import parse_overrides
from backtest_dataset import dump_backtest_dataset_metadata
from config_utils import (
    dump_config,
    add_config_arguments,
    project_template_config_for_cli,
    update_config_with_args,
    recursive_config_update,
    format_config,
    strip_config_metadata,
    sanitize_prepared_config_for_dump,
    HSL_PSIDE_KEYS,
)
from analysis_visibility import filter_analysis_for_visibility
from utils import (
    utc_ms,
    make_get_filepath,
    load_markets,
    format_end_date,
    format_approved_ignored_coins,
    date_to_ts,
)
from pure_funcs import (
    ts_to_date,
    sort_dict_keys,
    calc_hash,
    str2bool,
)
import pprint
from copy import deepcopy
from hlcv_preparation import prepare_hlcvs, prepare_hlcvs_combined, try_prepare_hlcvs_v2_local
from ohlcv_utils import aggregate_hlcvs, align_and_aggregate_hlcvs
from warmup_utils import compute_backtest_warmup_minutes, compute_per_coin_warmup_minutes
from pathlib import Path
import re
from plotting import (
    create_forager_balance_figures,
    create_forager_coin_figures,
    create_forager_hard_stop_drawdown_figure,
    create_forager_pnl_figure,
    create_forager_twe_figure,
    save_figures,
)
from collections import defaultdict
import logging
import gzip
import traceback
from logging_setup import configure_logging, resolve_log_level
from suite_runner import extract_suite_config, filter_scenarios_by_label, run_backtest_suite_async
import passivbot_rust as pbr  # noqa: E402
verify_loaded_runtime_extension()
from tools.event_loop_policy import set_windows_event_loop_policy

PLOT_GROUP_SUMMARY = {"balance", "twe", "pnl", "hard_stop"}
PLOT_GROUP_ALL = PLOT_GROUP_SUMMARY | {"coin_fills"}
HLCVS_CACHE_ROOT = Path("caches") / "hlcvs_data"
HLCVS_CACHE_HASH_LEN = 16
HLCVS_CACHE_DIR_SEP = "__"


def parse_disabled_plot_groups(value) -> set[str]:
    if value in (None, False, "", [], ()):
        return set()
    if value is True:
        return set(PLOT_GROUP_ALL)
    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw.lower() in {"false", "none", "n", "no", "0"}:
            return set()
        tokens = [token.strip().lower() for token in raw.split(",") if token.strip()]
    elif isinstance(value, (list, tuple, set)):
        tokens = [str(token).strip().lower() for token in value if str(token).strip()]
    else:
        raise ValueError(f"invalid disable_plotting value type: {type(value).__name__}")

    disabled = set()
    for token in tokens:
        if token in {"true", "all", "y", "yes", "1"}:
            disabled.update(PLOT_GROUP_ALL)
        elif token == "summary":
            disabled.update(PLOT_GROUP_SUMMARY)
        elif token in PLOT_GROUP_ALL:
            disabled.add(token)
        else:
            raise ValueError(
                "disable_plotting must be one of all, summary, balance, twe, pnl, hard_stop, coin_fills"
            )
    return disabled

ANALYSIS_SHARED_PREFIXES = ("hard_stop_",)

# Fallback stubs for test environments without full extension symbols
if not hasattr(pbr, "HlcvsBundle"):  # pragma: no cover

    class HlcvsBundle:
        pass

    pbr.HlcvsBundle = HlcvsBundle  # type: ignore

# on Windows this will pick the SelectorEventLoopPolicy
set_windows_event_loop_policy()


def aggregate_candles(candles_1m: np.ndarray, interval: int) -> np.ndarray:
    return aggregate_hlcvs(candles_1m, interval)


def _looks_like_bool_token(value: str) -> bool:
    if value is None:
        return False
    lowered = value.lower()
    return lowered in {"1", "0", "true", "false", "t", "f", "yes", "no", "y", "n"}


def _resolve_backtest_hsl_configs(config: dict) -> tuple[dict, dict]:
    long_cfg = config.get("bot", {}).get("long", {})
    short_cfg = config.get("bot", {}).get("short", {})
    if not (
        all(key in long_cfg for key in HSL_PSIDE_KEYS)
        and all(key in short_cfg for key in HSL_PSIDE_KEYS)
    ):
        raise KeyError("missing required per-side HSL config under bot.long/bot.short")

    def _convert(pside_cfg: dict) -> dict:
        return {
            "enabled": bool(pside_cfg["hsl_enabled"]),
            "red_threshold": float(pside_cfg["hsl_red_threshold"]),
            "ema_span_minutes": float(pside_cfg["hsl_ema_span_minutes"]),
            "cooldown_minutes_after_red": float(pside_cfg["hsl_cooldown_minutes_after_red"]),
            "no_restart_drawdown_threshold": float(
                pside_cfg["hsl_no_restart_drawdown_threshold"]
            ),
            "tier_ratios": {
                "yellow": float(pside_cfg["hsl_tier_ratios"]["yellow"]),
                "orange": float(pside_cfg["hsl_tier_ratios"]["orange"]),
            },
            "orange_tier_mode": str(pside_cfg["hsl_orange_tier_mode"]),
            "panic_close_order_type": str(pside_cfg["hsl_panic_close_order_type"]),
        }

    return _convert(long_cfg), _convert(short_cfg)


def _resolve_backtest_hsl_signal_mode(config: dict) -> str:
    return normalize_hsl_signal_mode(
        get_optional_config_value(config, "live.hsl_signal_mode", "pside")
    )


def _normalize_optional_bool_flag(argv: list[str], flag: str) -> list[str]:
    result: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token == flag:
            next_token = argv[i + 1] if i + 1 < len(argv) else None
            if (
                next_token
                and not next_token.startswith("-")
                and not _looks_like_bool_token(next_token)
            ):
                result.append(f"{flag}=true")
                i += 1
                continue
        result.append(token)
        i += 1
    return result


def oj(*x):
    return os.path.join(*x)


def _split_symbol_parts(symbol: str):
    symbol = str(symbol or "")
    if not symbol:
        return "", "USD"
    if "/" in symbol:
        base, rest = symbol.split("/", 1)
    else:
        base, rest = symbol, ""
    if ":" in rest:
        quote = rest.split(":", 1)[0]
    elif rest:
        quote = rest
    elif ":" in symbol:
        base, quote = symbol.split(":", 1)
    else:
        quote = "USD"
    return base or symbol, quote or "USD"


def _float_or(value, default=0.0):
    try:
        if value is None:
            raise TypeError
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int_or(value, default=0):
    try:
        if value is None:
            raise TypeError
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _build_coin_metadata_entries(
    coins_order,
    exchange,
    mss,
    first_valid_indices,
    last_valid_indices,
    warmup_minutes,
    trade_start_indices,
):
    entries = []
    for idx, coin in enumerate(coins_order):
        entry = mss.get(coin, {}) if isinstance(mss, dict) else {}
        symbol = str(entry.get("symbol", coin))
        base_from_symbol, quote_from_symbol = _split_symbol_parts(symbol)
        coin_shorthand = entry.get("coin") or entry.get("base") or base_from_symbol
        entry_exchange = entry.get("exchange") or exchange
        maker_fee = entry.get("maker_fee")
        if maker_fee is None:
            maker_fee = entry.get("maker")
        taker_fee = entry.get("taker_fee")
        if taker_fee is None:
            taker_fee = entry.get("taker")
        entries.append(
            {
                "index": idx,
                "symbol": symbol,
                "coin": coin_shorthand,
                "exchange": entry_exchange,
                "quote": entry.get("quote") or quote_from_symbol,
                "base": entry.get("base") or base_from_symbol,
                "qty_step": _float_or(entry.get("qty_step")),
                "price_step": _float_or(entry.get("price_step")),
                "min_qty": _float_or(entry.get("min_qty")),
                "min_cost": _float_or(entry.get("min_cost")),
                "c_mult": _float_or(entry.get("c_mult"), 1.0),
                "maker_fee": _float_or(maker_fee),
                "taker_fee": _float_or(taker_fee),
                "first_valid_index": _int_or(first_valid_indices[idx]),
                "last_valid_index": _int_or(last_valid_indices[idx]),
                "warmup_minutes": _int_or(warmup_minutes[idx]),
                "trade_start_index": _int_or(trade_start_indices[idx]),
            }
        )
    return entries


def _as_c_contiguous_native_array(arr, dtype):
    dtype = np.dtype(dtype)
    out = np.asarray(arr)
    if out.dtype == dtype and out.flags.c_contiguous:
        return out
    return np.ascontiguousarray(out, dtype=dtype)


def _build_hlcvs_bundle(
    hlcvs,
    btc_usd_prices,
    timestamps,
    exchange,
    mss,
    coins_order,
    first_valid_indices,
    last_valid_indices,
    warmup_minutes,
    trade_start_indices,
    bundle_meta_base: dict,
    *,
    coin_indices: list[int] | None = None,
) -> pbr.HlcvsBundle:
    subset_positions = None
    if coin_indices is not None:
        if len(coin_indices) != len(coins_order):
            raise ValueError(
                f"coin_indices length ({len(coin_indices)}) does not match coins ({len(coins_order)})"
            )
        subset_positions = [int(idx) for idx in coin_indices]
        n_coins = int(hlcvs.shape[1])
        if len(subset_positions) == n_coins and subset_positions == list(range(n_coins)):
            subset_positions = None

    def _rss_mb() -> float | None:
        try:
            import resource as _resource
        except Exception:
            return None
        try:
            rss_kb = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
            # Linux reports in KB, macOS in bytes.
            if sys.platform == "darwin":
                return rss_kb / (1024**2)
            return rss_kb / 1024
        except Exception:
            return None

    rss_before = _rss_mb()
    logger = logging.getLogger(__name__)
    if hasattr(logger, "trace"):
        logger.trace(
            "Build bundle | pid=%s hlcvs_shape=%s coins=%d subset=%s rss_mb=%s",
            os.getpid(),
            getattr(hlcvs, "shape", None),
            len(coins_order),
            subset_positions is not None,
            f"{rss_before:.1f}" if rss_before is not None else "na",
        )
    if subset_positions is not None:
        hlcvs_view = hlcvs[:, subset_positions, :]
        hlcvs_arr = np.ascontiguousarray(hlcvs_view, dtype=np.float64)
    else:
        hlcvs_arr = _as_c_contiguous_native_array(hlcvs, np.float64)
    btc_arr = _as_c_contiguous_native_array(btc_usd_prices, np.float64)
    if timestamps is None:
        timestamps_arr = np.arange(hlcvs_arr.shape[0], dtype=np.int64)
    else:
        timestamps_arr = _as_c_contiguous_native_array(timestamps, np.int64)
    coin_meta_entries = _build_coin_metadata_entries(
        coins_order,
        exchange,
        mss,
        first_valid_indices,
        last_valid_indices,
        warmup_minutes,
        trade_start_indices,
    )
    bundle_meta = {**bundle_meta_base, "coins": coin_meta_entries}
    rss_after = _rss_mb()
    if hasattr(logger, "trace"):
        logger.trace(
            "Build bundle done | pid=%s rss_mb=%s",
            os.getpid(),
            f"{rss_after:.1f}" if rss_after is not None else "na",
        )
    return pbr.HlcvsBundle(hlcvs_arr, btc_arr, timestamps_arr, bundle_meta)


@dataclass
class BacktestExecutionSettings:
    market_orders_allowed: bool
    market_order_near_touch_threshold: float
    market_order_slippage_pct: float
    pnls_max_lookback_days: str | float
    pnls_max_lookback_days_backtest_value: float


@dataclass
class BacktestPayload:
    """Container for everything needed to run a backtest via the Rust engine."""

    bundle: Any
    bot_params_list: list
    exchange_params: list
    backtest_params: dict
    execution_settings: BacktestExecutionSettings | None = None
    hard_stop_plot_data: dict | None = None


def build_backtest_payload(
    hlcvs,
    mss,
    config: dict,
    exchange: str,
    btc_usd_prices,
    timestamps=None,
    *,
    coin_indices: list[int] | None = None,
    metrics_only: bool = False,
) -> BacktestPayload:
    """
    Assemble the bundle, bot params, and metadata needed to execute a backtest.
    """

    runtime_config = compile_runtime_config(config, runtime="backtest", record_step=False)
    execution_settings = get_backtest_execution_settings(runtime_config, is_runtime_compiled=True)
    bot_params_list, exchange_params, backtest_params = prep_backtest_args(
        runtime_config,
        mss,
        exchange,
        execution_settings=execution_settings,
        is_runtime_compiled=True,
        metrics_only=metrics_only,
    )
    backtest_params = dict(backtest_params)
    coins_order = backtest_params.get("coins", [])

    # Read candle interval from config (default to 1m)
    candle_interval = config.get("backtest", {}).get("candle_interval_minutes", 1)
    if isinstance(candle_interval, numbers.Real):
        candle_interval_float = float(candle_interval)
        if not candle_interval_float.is_integer():
            raise ValueError(
                "candle_interval_minutes must be an integer number of minutes, "
                f"got {candle_interval!r}"
            )
        candle_interval = int(candle_interval_float)
    else:
        raise TypeError(
            "candle_interval_minutes must be an integer number of minutes, "
            f"got {type(candle_interval).__name__}"
        )
    if candle_interval < 1:
        raise ValueError(f"candle_interval_minutes must be >= 1, got {candle_interval}")
    backtest_params["candle_interval_minutes"] = candle_interval
    meta = mss.get("__meta__", {}) if isinstance(mss, dict) else {}
    data_interval = int(meta.get("data_interval_minutes", 1) or 1)
    offset_bars = int(meta.get("candle_interval_offset_bars", 0) or 0)
    if data_interval < 1:
        data_interval = 1
    if candle_interval == 1 and data_interval != 1:
        raise ValueError(
            f"Input data is already aggregated at {data_interval}m but candle_interval_minutes=1"
        )
    if data_interval != 1 and data_interval != candle_interval:
        raise ValueError(
            f"Input data interval {data_interval} does not match candle_interval_minutes={candle_interval}"
        )

    # Aggregate candles if using coarser interval and data is still 1m
    if candle_interval > 1 and data_interval == 1:
        n_before = hlcvs.shape[0]
        hlcvs, timestamps, btc_usd_prices, offset_bars = align_and_aggregate_hlcvs(
            hlcvs,
            timestamps,
            btc_usd_prices,
            candle_interval,
        )
        logging.debug(
            "[backtest] aggregated %dm candles: %d bars -> %d bars (trimmed %d for alignment)",
            candle_interval,
            n_before,
            hlcvs.shape[0],
            offset_bars,
        )

    # Inject first timestamp (ms) into backtest params; default to 0 if unknown
    try:
        first_ts_ms = int(timestamps[0]) if (timestamps is not None and len(timestamps) > 0) else 0
    except Exception:
        first_ts_ms = 0

    # Ensure timestamp alignment for aggregated data
    if candle_interval > 1 and first_ts_ms > 0:
        interval_ms = candle_interval * 60_000
        remainder = first_ts_ms % interval_ms
        if remainder != 0:
            raise ValueError(
                f"First timestamp {first_ts_ms} is not aligned to {candle_interval}m interval"
            )
    backtest_params["first_timestamp_ms"] = first_ts_ms

    warmup_map = compute_per_coin_warmup_minutes(config)
    default_warm = int(warmup_map.get("__default__", 0))
    first_valid_indices = []
    last_valid_indices = []
    warmup_minutes = []
    trade_start_indices = []
    total_steps = hlcvs.shape[0]
    source_steps_1m = total_steps * (candle_interval if candle_interval > 1 else 1)
    for idx, coin in enumerate(coins_order):
        meta = mss.get(coin, {}) if isinstance(mss, dict) else {}
        # Metadata indices are based on 1m candles; adjust for aggregated interval
        first_idx_1m = int(meta.get("first_valid_index", 0)) - offset_bars
        last_idx_1m = int(meta.get("last_valid_index", source_steps_1m - 1)) - offset_bars
        if first_idx_1m < 0:
            first_idx_1m = 0
        if last_idx_1m < 0:
            last_idx_1m = 0
        if first_idx_1m >= source_steps_1m:
            first_idx_1m = source_steps_1m
        if last_idx_1m >= source_steps_1m:
            last_idx_1m = source_steps_1m - 1
        if candle_interval > 1:
            first_idx = int(math.ceil(first_idx_1m / candle_interval))
            last_idx = int(last_idx_1m // candle_interval)
        else:
            first_idx = int(first_idx_1m)
            last_idx = int(last_idx_1m)
        if first_idx >= total_steps:
            first_idx = total_steps
        if last_idx >= total_steps:
            last_idx = total_steps - 1
        first_valid_indices.append(first_idx)
        last_valid_indices.append(last_idx)
        # warmup_minutes stay in minutes (Rust adjusts based on interval)
        warm = int(meta.get("warmup_minutes", warmup_map.get(coin, default_warm)))
        warmup_minutes.append(warm)
        # trade_start_idx is in candle units, adjust warm from minutes to candle periods
        warm_bars = int(math.ceil(warm / candle_interval)) if candle_interval > 1 else int(warm)
        if first_idx > last_idx:
            trade_idx = first_idx
        else:
            trade_idx = min(last_idx, first_idx + warm_bars)
        trade_start_indices.append(trade_idx)
    backtest_params["first_valid_indices"] = first_valid_indices
    backtest_params["last_valid_indices"] = last_valid_indices
    backtest_params["warmup_minutes"] = warmup_minutes
    backtest_params["trade_start_indices"] = trade_start_indices
    global_warmup_minutes = compute_backtest_warmup_minutes(config)
    if candle_interval > 1:
        global_warmup_bars = int(math.ceil(global_warmup_minutes / candle_interval))
    else:
        global_warmup_bars = int(global_warmup_minutes)
    backtest_params["global_warmup_bars"] = global_warmup_bars

    meta = mss.get("__meta__", {}) if isinstance(mss, dict) else {}
    candidate_start = meta.get(
        "requested_start_ts", require_config_value(config, "backtest.start_date")
    )
    try:
        if isinstance(candidate_start, str):
            requested_start_ts = int(date_to_ts(candidate_start))
        else:
            requested_start_ts = int(candidate_start)
    except Exception:
        requested_start_ts = int(date_to_ts(require_config_value(config, "backtest.start_date")))
    backtest_params["requested_start_timestamp_ms"] = requested_start_ts

    warmup_requested = int(
        meta.get("warmup_minutes_requested", compute_backtest_warmup_minutes(config))
    )
    if "warmup_minutes_provided" in meta:
        warmup_provided = int(meta["warmup_minutes_provided"])
    elif timestamps is not None and len(timestamps) > 0:
        warmup_provided = max(
            0, (requested_start_ts - int(timestamps[0])) // 60_000
        )
    else:
        warmup_provided = warmup_requested

    if timestamps is not None and len(timestamps) > 0:
        effective_start_ts = int(timestamps[0])
    else:
        effective_start_ts = 0

    bundle_meta_base = {
        "requested_start_timestamp_ms": requested_start_ts,
        "effective_start_timestamp_ms": effective_start_ts,
        "warmup_minutes_requested": warmup_requested,
        "warmup_minutes_provided": warmup_provided,
    }

    bundle = _build_hlcvs_bundle(
        hlcvs,
        btc_usd_prices,
        timestamps,
        exchange,
        mss,
        coins_order,
        first_valid_indices,
        last_valid_indices,
        warmup_minutes,
        trade_start_indices,
        bundle_meta_base,
        coin_indices=coin_indices,
    )

    if coin_indices is not None:
        backtest_params["active_coin_indices"] = list(range(len(coins_order)))

    return BacktestPayload(
        bundle=bundle,
        bot_params_list=bot_params_list,
        exchange_params=exchange_params,
        backtest_params=backtest_params,
        execution_settings=execution_settings,
    )


def execute_backtest(payload: BacktestPayload, config: dict):
    """
    Execute a prepared backtest payload and expand the resulting analysis.
    """

    backtest_result = pbr.run_backtest_bundle(
        payload.bundle,
        payload.bot_params_list,
        payload.exchange_params,
        payload.backtest_params,
    )
    if len(backtest_result) == 5:
        (
            fills,
            equities_array,
            analysis_usd,
            analysis_btc,
            hard_stop_plot_data,
        ) = backtest_result
    elif len(backtest_result) == 4:
        fills, equities_array, analysis_usd, analysis_btc = backtest_result
        hard_stop_plot_data = {}
    else:
        raise ValueError(
            f"run_backtest_bundle returned {len(backtest_result)} values; expected 4 or 5"
        )

    if payload.backtest_params.get("metrics_only", False):
        payload.hard_stop_plot_data = {}
        analysis = expand_analysis(analysis_usd, analysis_btc, None, None, config)
        return None, None, analysis

    equities_array = np.asarray(equities_array)
    payload.hard_stop_plot_data = dict(hard_stop_plot_data or {})
    analysis = expand_analysis(analysis_usd, analysis_btc, fills, equities_array, config)
    if bool(analysis.get("liquidated", False)):
        final_equity_usd = float(equities_array[-1, 1]) if equities_array.size else float("nan")
        logging.debug(
            "Backtest liquidated early | final_equity_usd=%.6f | liquidation_threshold=%.6f",
            final_equity_usd,
            float(get_optional_config_value(config, "backtest.liquidation_threshold", 0.05) or 0.0),
        )
    return fills, equities_array, analysis


def subset_backtest_payload(
    payload: BacktestPayload,
    *,
    coin_indices: Sequence[int] | None = None,
    coin_symbols: Iterable[str] | None = None,
) -> BacktestPayload:
    """
    Return a new payload sliced down to a subset of coins.

    Args:
        payload: Source payload produced by build_backtest_payload.
        coin_indices: Ordered iterable of integer positions to keep.
        coin_symbols: Optional iterable of symbol strings (either full symbol or shorthand coin).
    """

    if coin_indices is None and coin_symbols is None:
        raise ValueError("Provide either coin_indices or coin_symbols when subsetting.")

    bundle_meta = payload.bundle.meta
    coins_meta = bundle_meta["coins"]

    if coin_symbols is not None:
        requested = {str(sym) for sym in coin_symbols}
        lookup = {}
        for pos, coin in enumerate(coins_meta):
            lookup.setdefault(coin["symbol"], pos)
            lookup.setdefault(coin.get("coin"), pos)
        selected_positions = []
        for sym in requested:
            if sym not in lookup:
                raise ValueError(f"Coin symbol '{sym}' not present in payload.")
            selected_positions.append(lookup[sym])
        selected_positions.sort()
    else:
        selected_positions = [int(idx) for idx in coin_indices]
        for idx in selected_positions:
            if idx < 0 or idx >= len(coins_meta):
                raise ValueError(f"Coin index {idx} outside valid range.")

    hlcvs_np = np.asarray(payload.bundle.hlcvs)
    subset_hlcvs = np.ascontiguousarray(hlcvs_np[:, selected_positions, :], dtype=np.float64)
    btc_np = _as_c_contiguous_native_array(payload.bundle.btc_usd, np.float64)
    ts_np = _as_c_contiguous_native_array(payload.bundle.timestamps, np.int64)

    new_meta = deepcopy(bundle_meta)
    new_meta["coins"] = []
    for new_idx, pos in enumerate(selected_positions):
        coin_entry = deepcopy(coins_meta[pos])
        coin_entry["index"] = new_idx
        new_meta["coins"].append(coin_entry)

    new_bundle = pbr.HlcvsBundle(subset_hlcvs, btc_np, ts_np, new_meta)

    def _select(seq):
        return [seq[pos] for pos in selected_positions]

    new_bot = _select(payload.bot_params_list)
    new_exchange_params = _select(payload.exchange_params)
    new_backtest_params = deepcopy(payload.backtest_params)
    for key in [
        "coins",
        "first_valid_indices",
        "last_valid_indices",
        "warmup_minutes",
        "trade_start_indices",
    ]:
        if key in new_backtest_params and isinstance(new_backtest_params[key], list):
            new_backtest_params[key] = _select(new_backtest_params[key])

    return BacktestPayload(
        bundle=new_bundle,
        bot_params_list=new_bot,
        exchange_params=new_exchange_params,
        backtest_params=new_backtest_params,
        execution_settings=payload.execution_settings,
    )


def process_forager_fills(
    fills,
    coins,
    hlcvs,
    equities_array,
    balance_sample_divider: int = 60,
):
    fdf = pd.DataFrame(
        fills,
        columns=[
            "index",
            "timestamp",
            "coin",
            "pnl",
            "fee_paid",
            "usd_total_balance",
            "btc_cash_wallet",
            "usd_cash_wallet",
            "btc_price",
            "qty",
            "price",
            "psize",
            "pprice",
            "type",
            "liquidity",
            "wallet_exposure",
            "twe_long",
            "twe_short",
            "twe_net",
        ],
    )
    if not fdf.empty:
        fdf["timestamp"] = pd.to_datetime(fdf["timestamp"].astype(np.int64), unit="ms")
        fdf["index"] = fdf["index"].astype(int)
        fdf["minute"] = fdf["index"].astype(int)
        numeric_cols = [
            "pnl",
            "fee_paid",
            "usd_total_balance",
            "btc_cash_wallet",
            "usd_cash_wallet",
            "btc_price",
            "qty",
            "price",
            "psize",
            "pprice",
            "wallet_exposure",
            "twe_long",
            "twe_short",
            "twe_net",
        ]
        fdf[numeric_cols] = fdf[numeric_cols].apply(pd.to_numeric, errors="coerce")
        fdf["btc_total_balance"] = np.where(
            fdf["btc_price"] > 0.0,
            fdf["usd_total_balance"] / fdf["btc_price"],
            np.nan,
        )
    else:
        fdf["minute"] = pd.Series(dtype=int)
        fdf["btc_total_balance"] = pd.Series(dtype=float)
    analysis_appendix = {}

    pnls = {}
    for pside in ["long", "short"]:
        fdfc = fdf[fdf.type.str.contains(pside)]
        profit = fdfc[fdfc.pnl > 0.0].pnl.sum()
        loss = fdfc[fdfc.pnl < 0.0].pnl.sum()
        if len(fdfc) == 0:
            pnls[pside] = 0.0
            analysis_appendix[f"loss_profit_ratio_{pside}"] = 1.0
            continue
        pnls[pside] = profit + loss
        analysis_appendix[f"loss_profit_ratio_{pside}"] = abs(loss / profit) if profit != 0.0 else 1.0
    pnl_sum = pnls["long"] + pnls["short"]
    analysis_appendix["pnl_ratio_long_short"] = pnls["long"] / pnl_sum if pnl_sum != 0.0 else 0.5
    sample_divider = max(1, int(balance_sample_divider))
    if not fdf.empty:
        timestamps_ns = fdf["timestamp"].astype("int64")
        bucket = (timestamps_ns // (sample_divider * 60_000 * 1_000_000)) * (
            sample_divider * 60_000 * 1_000_000
        )
        usd_cash_series = fdf.groupby(bucket)["usd_cash_wallet"].last().rename("usd_cash_wallet")
        usd_total_balance_series = (
            fdf.groupby(bucket)["usd_total_balance"].last().rename("usd_total_balance")
        )
        btc_cash_series = fdf.groupby(bucket)["btc_cash_wallet"].last().rename("btc_cash_wallet")
        btc_total_balance_series = (
            fdf.groupby(bucket)["btc_total_balance"].last().rename("btc_total_balance")
        )
        # convert to datetime index for easier alignment
        usd_cash_series.index = pd.to_datetime(usd_cash_series.index, unit="ns")
        usd_total_balance_series.index = pd.to_datetime(usd_total_balance_series.index, unit="ns")
        btc_cash_series.index = pd.to_datetime(btc_cash_series.index, unit="ns")
        btc_total_balance_series.index = pd.to_datetime(btc_total_balance_series.index, unit="ns")
    else:
        empty_dtidx = pd.DatetimeIndex([])
        usd_cash_series = pd.Series(dtype=float, name="usd_cash_wallet", index=empty_dtidx)
        usd_total_balance_series = pd.Series(dtype=float, name="usd_total_balance", index=empty_dtidx)
        btc_cash_series = pd.Series(dtype=float, name="btc_cash_wallet", index=empty_dtidx)
        btc_total_balance_series = pd.Series(dtype=float, name="btc_total_balance", index=empty_dtidx)
    equities_array = np.asarray(equities_array)
    equities_index = pd.to_datetime(equities_array[:, 0].astype(np.int64), unit="ms")
    edf = pd.Series(
        equities_array[:, 1],
        index=equities_index,
        name="usd_total_equity",
    )
    ebdf = pd.Series(
        equities_array[:, 2],
        index=equities_index,
        name="btc_total_equity",
    )
    if equities_array.shape[1] > 3:
        strategy_eq_series = pd.Series(
            equities_array[:, 3],
            index=equities_index,
            name="strategy_equity",
        )
    else:
        strategy_eq_series = pd.Series(dtype=float, name="strategy_equity", index=equities_index)
    bal_eq = pd.concat(
        [
            usd_cash_series,
            usd_total_balance_series,
            edf,
            strategy_eq_series,
            btc_cash_series,
            btc_total_balance_series,
            ebdf,
        ],
        axis=1,
        join="outer",
    )
    if bal_eq.empty:
        bal_eq = pd.DataFrame(
            columns=[
                "usd_cash_wallet",
                "usd_total_balance",
                "usd_total_equity",
                "strategy_equity",
                "btc_cash_wallet",
                "btc_total_balance",
                "btc_total_equity",
            ]
        )
    else:
        bal_eq = bal_eq.sort_index()
        bal_eq = bal_eq[~bal_eq.index.duplicated(keep="first")]
        bal_eq = (
            bal_eq.reindex(
                columns=[
                    "usd_cash_wallet",
                    "usd_total_balance",
                    "usd_total_equity",
                    "strategy_equity",
                    "btc_cash_wallet",
                    "btc_total_balance",
                    "btc_total_equity",
                ]
            )
            .ffill()
            .bfill()
        )
        if sample_divider > 1 and not bal_eq.empty:
            if isinstance(bal_eq.index, pd.DatetimeIndex):
                try:
                    bal_eq = bal_eq.resample(f"{sample_divider}min").last()
                except ValueError:
                    bal_eq = bal_eq.iloc[::sample_divider]
            else:
                bal_eq = bal_eq.iloc[::sample_divider]
            bal_eq = bal_eq.dropna(how="all").ffill().bfill()
    bal_eq = bal_eq.round(4).astype(np.float32)
    return fdf, sort_dict_keys(analysis_appendix), bal_eq


def compare_dicts(dict1, dict2, path=""):
    for key in sorted(set(dict1.keys()) | set(dict2.keys())):
        if key not in dict1:
            print(f"{path}{key}: Missing in first dict. Value in second dict: {dict2[key]}")
        elif key not in dict2:
            print(f"{path}{key}: Missing in second dict. Value in first dict: {dict1[key]}")
        elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            compare_dicts(dict1[key], dict2[key], f"{path}{key}.")
        elif dict1[key] != dict2[key]:
            print(f"{path}{key}: Values differ. First dict:  {dict1[key]} Second dict: {dict2[key]}")


def compare_dict_keys(dict1, dict2):
    def get_all_keys(d):
        keys = set(d.keys())
        for value in d.values():
            if isinstance(value, dict):
                keys.update(get_all_keys(value))
        return keys

    return get_all_keys(dict1) == get_all_keys(dict2)


def check_keys(dict0, dict1):
    def check_nested(d0, d1):
        for key, value in d0.items():
            if key not in d1:
                return False
            if isinstance(value, dict):
                if not isinstance(d1[key], dict):
                    return False
                if not check_nested(value, d1[key]):
                    return False
        return True

    return check_nested(dict0, dict1)


def get_cache_hash(config, exchange):
    exchanges_cfg = require_config_value(config, "backtest.exchanges")
    approved_coins_raw = require_live_value(config, "approved_coins")
    approved_coins = {
        "long": sorted(approved_coins_raw.get("long", [])),
        "short": sorted(approved_coins_raw.get("short", [])),
    }
    minimum_coin_age = require_live_value(config, "minimum_coin_age_days")
    backtest_cfg = config.get("backtest", {}) or {}
    coin_sources = backtest_cfg.get("coin_sources") or {}
    coin_sources_sorted = sorted((str(k), str(v)) for k, v in coin_sources.items())
    market_settings_sources = backtest_cfg.get("market_settings_sources") or {}
    market_settings_sources_sorted = sorted(
        (str(k), str(v)) for k, v in market_settings_sources.items()
    )
    to_hash = {
        "coins": approved_coins,
        "end_date": format_end_date(require_config_value(config, "backtest.end_date")),
        "start_date": require_config_value(config, "backtest.start_date"),
        "exchange": exchanges_cfg if exchange == "combined" else exchange,
        "minimum_coin_age_days": minimum_coin_age,
        "gap_tolerance_ohlcvs_minutes": require_config_value(
            config, "backtest.gap_tolerance_ohlcvs_minutes"
        ),
        "coin_sources": coin_sources_sorted,
        "market_settings_sources": market_settings_sources_sorted,
        "ohlcv_source_dir": backtest_cfg.get("ohlcv_source_dir"),
    }
    return calc_hash(to_hash)


def _extract_hlcvs_cache_hash_prefix(cache_hash) -> str:
    return str(cache_hash)[:HLCVS_CACHE_HASH_LEN]


def _sanitize_hlcvs_cache_label(value) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip())
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    return sanitized or "na"


def _normalize_hlcvs_cache_coin_label(coin) -> str:
    coin_str = str(coin or "").strip()
    if "/" in coin_str:
        coin_str = coin_str.split("/", 1)[0]
    return _sanitize_hlcvs_cache_label(coin_str).upper()


def _format_hlcvs_cache_coins_label(coins) -> str:
    normalized = []
    seen = set()
    for coin in coins or []:
        label = _normalize_hlcvs_cache_coin_label(coin)
        if label in seen:
            continue
        seen.add(label)
        normalized.append(label)
    if not normalized:
        return "0_coins"
    if len(normalized) > 5:
        return f"{len(normalized)}_coins"
    joined = "_".join(normalized)
    if len(joined) > 80:
        return f"{len(normalized)}_coins"
    return joined


def _format_hlcvs_cache_date_label(config, timestamps=None) -> str:
    try:
        if timestamps is not None and len(timestamps) > 0:
            start_ts = int(timestamps[0])
            end_ts = int(timestamps[-1])
            return f"{ts_to_date(start_ts)[:10]}_to_{ts_to_date(end_ts)[:10]}"
    except Exception:
        pass
    start_date = _sanitize_hlcvs_cache_label(require_config_value(config, "backtest.start_date"))
    end_date = _sanitize_hlcvs_cache_label(
        format_end_date(require_config_value(config, "backtest.end_date"))
    )
    return f"{start_date}_to_{end_date}"


def _build_hlcvs_cache_dir_name(config, exchange, coins, cache_hash, timestamps=None) -> str:
    exchange_label = _sanitize_hlcvs_cache_label(str(exchange).lower())
    coins_label = _format_hlcvs_cache_coins_label(coins)
    date_label = _format_hlcvs_cache_date_label(config, timestamps=timestamps)
    hash_prefix = _extract_hlcvs_cache_hash_prefix(cache_hash)
    return (
        f"{exchange_label}{HLCVS_CACHE_DIR_SEP}"
        f"{coins_label}{HLCVS_CACHE_DIR_SEP}"
        f"{date_label}{HLCVS_CACHE_DIR_SEP}"
        f"{hash_prefix}"
    )


def _resolve_hlcvs_cache_dir(cache_hash):
    hash_prefix = _extract_hlcvs_cache_hash_prefix(cache_hash)
    legacy_dir = HLCVS_CACHE_ROOT / hash_prefix
    if legacy_dir.is_dir():
        return legacy_dir
    if not HLCVS_CACHE_ROOT.exists():
        return None
    suffix = f"{HLCVS_CACHE_DIR_SEP}{hash_prefix}"
    candidates = sorted(
        path
        for path in HLCVS_CACHE_ROOT.iterdir()
        if path.is_dir() and path.name.endswith(suffix)
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        raise RuntimeError(
            f"multiple HLCV cache directories found for hash {hash_prefix}: "
            + ", ".join(str(path) for path in candidates)
        )
    return candidates[0]


def _get_hlcvs_cache_dir_for_save(config, exchange, coins, cache_hash, timestamps=None):
    existing_dir = _resolve_hlcvs_cache_dir(cache_hash)
    if existing_dir is not None:
        return existing_dir
    dir_name = _build_hlcvs_cache_dir_name(
        config,
        exchange,
        coins,
        cache_hash,
        timestamps=timestamps,
    )
    return HLCVS_CACHE_ROOT / dir_name


def load_coins_hlcvs_from_cache(config, exchange, warmup_minutes=0):
    cache_hash = get_cache_hash(config, exchange)
    cache_dir = _resolve_hlcvs_cache_dir(cache_hash)
    compress_cache = bool(require_config_value(config, "backtest.compress_cache"))
    if cache_dir and os.path.exists(cache_dir):
        # Check warmup sufficiency: cached data must cover at least the needed warmup
        meta_path = cache_dir / "cache_meta.json"
        if meta_path.exists():
            try:
                cache_meta = json.load(open(meta_path))
                cached_warmup = int(cache_meta.get("warmup_minutes", 0))
            except Exception:
                cached_warmup = 0
        else:
            cached_warmup = 0
        if cached_warmup < warmup_minutes:
            logging.info(
                f"{exchange} cache warmup insufficient: cached={cached_warmup} min, "
                f"needed={warmup_minutes} min. Will re-fetch."
            )
            return None
        coins = json.load(open(cache_dir / "coins.json"))
        mss = json.load(open(cache_dir / "market_specific_settings.json"))
        if compress_cache:
            fname = cache_dir / "hlcvs.npy.gz"
            logging.info(f"{exchange} Attempting to load hlcvs data from cache {fname}...")
            with gzip.open(fname, "rb") as f:
                hlcvs = np.load(f)
            # Load optional timestamps if present
            ts_fname = cache_dir / "timestamps.npy.gz"
            timestamps = None
            if os.path.exists(ts_fname):
                try:
                    with gzip.open(ts_fname, "rb") as f:
                        timestamps = np.load(f)
                except Exception:
                    timestamps = None
            btc_fname = cache_dir / "btc_usd_prices.npy.gz"
            if os.path.exists(btc_fname):
                logging.info(
                    f"{exchange} Attempting to load BTC/USD prices from cache {btc_fname}..."
                )
                with gzip.open(btc_fname, "rb") as f:
                    btc_usd_prices = np.load(f)
            else:
                logging.info(
                    f"{exchange} No BTC/USD prices in cache; cache invalid for fractional collateral"
                )
                return None
        else:
            fname = cache_dir / "hlcvs.npy"
            logging.info(f"{exchange} Attempting to load hlcvs data from cache {fname}...")
            hlcvs = np.load(fname)
            ts_fname = cache_dir / "timestamps.npy"
            timestamps = None
            if os.path.exists(ts_fname):
                try:
                    timestamps = np.load(ts_fname)
                except Exception:
                    timestamps = None
            btc_fname = cache_dir / "btc_usd_prices.npy"
            if os.path.exists(btc_fname):
                logging.info(
                    f"{exchange} Attempting to load BTC/USD prices from cache {btc_fname}..."
                )
                btc_usd_prices = np.load(btc_fname)
            else:
                logging.info(
                    f"{exchange} No BTC/USD prices in cache; cache invalid for fractional collateral"
                )
                return None
        results_path = oj(require_config_value(config, "backtest.base_dir"), exchange, "")
        return cache_dir, coins, hlcvs, mss, results_path, btc_usd_prices, timestamps
    return None


def save_coins_hlcvs_to_cache(
    config,
    coins,
    hlcvs,
    exchange,
    mss,
    btc_usd_prices,
    timestamps=None,
    warmup_minutes=0,
):
    cache_hash = get_cache_hash(config, exchange)
    cache_dir = _get_hlcvs_cache_dir_for_save(
        config,
        exchange,
        coins,
        cache_hash,
        timestamps=timestamps,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    is_compressed = bool(require_config_value(config, "backtest.compress_cache"))
    warmup_minutes = int(warmup_minutes)
    meta_path = cache_dir / "cache_meta.json"
    if meta_path.exists():
        try:
            existing_meta = json.load(open(meta_path))
            existing_warmup = int(existing_meta.get("warmup_minutes", 0))
            if existing_warmup >= warmup_minutes:
                return cache_dir
        except Exception:
            pass
    logging.info(f"Dumping cache...")
    json.dump(coins, open(cache_dir / "coins.json", "w"))
    json.dump(mss, open(cache_dir / "market_specific_settings.json", "w"))
    uncompressed_size = hlcvs.nbytes
    sts = utc_ms()
    if is_compressed:
        fpath = cache_dir / "hlcvs.npy.gz"
        logging.info(f"Attempting to save hlcvs data to cache {fpath}...")
        with gzip.open(fpath, "wb", compresslevel=1) as f:
            np.save(f, hlcvs)
        if timestamps is not None:
            ts_fpath = cache_dir / "timestamps.npy.gz"
            logging.info(f"Attempting to save timestamps to cache {ts_fpath}...")
            with gzip.open(ts_fpath, "wb", compresslevel=1) as f:
                np.save(f, timestamps)
        btc_fpath = cache_dir / "btc_usd_prices.npy.gz"
        logging.info(f"Attempting to save BTC/USD prices to cache {btc_fpath}...")
        with gzip.open(btc_fpath, "wb", compresslevel=1) as f:
            np.save(f, btc_usd_prices)
        compressed_size = (cache_dir / "hlcvs.npy.gz").stat().st_size
        btc_compressed_size = (cache_dir / "btc_usd_prices.npy.gz").stat().st_size
        line = (
            f"{compressed_size/(1024**3):.2f} GB compressed HLCVs "
            f"({compressed_size/uncompressed_size*100:.1f}%), "
            f"{btc_compressed_size/(1024**3):.2f} GB compressed BTC/USD prices"
        )
    else:
        fpath = cache_dir / "hlcvs.npy"
        logging.info(f"Attempting to save hlcvs data to cache {fpath}...")
        np.save(fpath, hlcvs)
        if timestamps is not None:
            ts_fpath = cache_dir / "timestamps.npy"
            logging.info(f"Attempting to save timestamps to cache {ts_fpath}...")
            np.save(ts_fpath, timestamps)
        btc_fpath = cache_dir / "btc_usd_prices.npy"
        logging.info(f"Attempting to save BTC/USD prices to cache {btc_fpath}...")
        np.save(btc_fpath, btc_usd_prices)
        line = ""
    logging.info(
        f"Successfully dumped hlcvs cache {fpath}: "
        f"{uncompressed_size/(1024**3):.2f} GB uncompressed, "
        f"{line}"
    )
    logging.info(f"Seconds to dump cache: {(utc_ms() - sts) / 1000:.4f}")
    json.dump({"warmup_minutes": warmup_minutes}, open(meta_path, "w"))
    return cache_dir


def ensure_valid_index_metadata(mss, hlcvs, coins, warmup_map=None):
    total_steps = hlcvs.shape[0]
    warmup_map = warmup_map or {}
    default_warm = int(warmup_map.get("__default__", 0))
    for idx, coin in enumerate(coins):
        meta = mss.setdefault(coin, {})
        if "first_valid_index" in meta and "last_valid_index" in meta:
            first_idx = int(meta["first_valid_index"])
            last_idx = int(meta["last_valid_index"])
        else:
            close_series = hlcvs[:, idx, 2]
            finite = np.isfinite(close_series)
            if finite.any():
                valid_indices = np.where(finite)[0]
                first_idx = int(valid_indices[0])
                last_idx = int(valid_indices[-1])
            else:
                first_idx = int(total_steps)
                last_idx = int(total_steps)
        meta["first_valid_index"] = first_idx
        meta["last_valid_index"] = last_idx
        if warmup_map:
            # Warmup from current config must override any historical cached metadata.
            warm_minutes = int(warmup_map.get(coin, default_warm))
        else:
            warm_minutes = int(meta.get("warmup_minutes", 0))
        meta["warmup_minutes"] = warm_minutes
        if first_idx > last_idx:
            trade_start_idx = first_idx
        else:
            trade_start_idx = min(last_idx, first_idx + warm_minutes)
        meta["trade_start_index"] = trade_start_idx


def warn_hlcv_valid_range_coverage(config, coins, mss, timestamps):
    if timestamps is None or len(timestamps) == 0:
        return
    requested_start_ts = int(date_to_ts(require_config_value(config, "backtest.start_date")))
    requested_end_ts = int(
        date_to_ts(format_end_date(require_config_value(config, "backtest.end_date")))
    )
    if requested_end_ts < requested_start_ts:
        return
    last_ts_index = len(timestamps) - 1
    for coin in coins:
        meta = mss.get(coin, {})
        first_idx = int(meta.get("first_valid_index", len(timestamps)))
        last_idx = int(meta.get("last_valid_index", -1))
        if first_idx > last_idx or first_idx > last_ts_index or last_idx < 0:
            logging.warning(
                "[hlcvs] %s has no valid local data for requested range %s -> %s",
                coin,
                ts_to_date(requested_start_ts),
                ts_to_date(requested_end_ts),
            )
            continue
        first_idx = max(0, min(first_idx, last_ts_index))
        last_idx = max(0, min(last_idx, last_ts_index))
        valid_start_ts = int(timestamps[first_idx])
        valid_end_ts = int(timestamps[last_idx])
        if valid_end_ts < requested_start_ts or valid_start_ts > requested_end_ts:
            logging.warning(
                "[hlcvs] %s valid data range %s -> %s is entirely outside requested range %s -> %s",
                coin,
                ts_to_date(valid_start_ts),
                ts_to_date(valid_end_ts),
                ts_to_date(requested_start_ts),
                ts_to_date(requested_end_ts),
            )
        elif valid_end_ts < requested_end_ts:
            missing_minutes = int((requested_end_ts - valid_end_ts) // 60_000)
            logging.warning(
                "[hlcvs] %s valid data ends before requested end: valid_end=%s requested_end=%s missing_minutes=%d",
                coin,
                ts_to_date(valid_end_ts),
                ts_to_date(requested_end_ts),
                missing_minutes,
            )


async def prepare_hlcvs_mss(config, exchange, *, force_refetch_gaps: bool = False):
    base_dir = require_config_value(config, "backtest.base_dir")
    results_path = oj(base_dir, exchange, "")
    warmup_map = compute_per_coin_warmup_minutes(config)
    default_warm = int(warmup_map.get("__default__", 0))
    backtest_warmup_minutes = compute_backtest_warmup_minutes(config)
    try:
        sts = utc_ms()
        result = load_coins_hlcvs_from_cache(config, exchange, backtest_warmup_minutes)
        if result:
            logging.info(f"Seconds to load cache: {(utc_ms() - sts) / 1000:.4f}")
            cache_dir, coins, hlcvs, mss, results_path, btc_usd_prices, timestamps = result
            logging.info(f"Successfully loaded hlcvs data from cache")
            ensure_valid_index_metadata(mss, hlcvs, coins, warmup_map)
            warn_hlcv_valid_range_coverage(config, coins, mss, timestamps)
            # Pass through cached timestamps if they were stored; fall back to None otherwise
            return coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, timestamps
    except Exception as e:
        logging.info(f"Unable to load hlcvs data from cache: {e}. Fetching...")
    local_v2 = None
    if exchange != "combined":
        try:
            local_v2 = await try_prepare_hlcvs_v2_local(
                config, exchange, force_refetch_gaps=force_refetch_gaps
            )
        except Exception as e:
            logging.info(f"Unable to prepare hlcvs from local v2 store: {e}. Falling back.")
    if local_v2 is not None:
        mss, timestamps, hlcvs, btc_usd_prices = local_v2
    if exchange == "combined":
        forced_sources = config.get("backtest", {}).get("coin_sources")
        market_settings_sources = config.get("backtest", {}).get("market_settings_sources")
        mss, timestamps, hlcvs, btc_usd_prices = await prepare_hlcvs_combined(
            config,
            forced_sources=forced_sources,
            market_settings_sources=market_settings_sources,
            force_refetch_gaps=force_refetch_gaps,
        )
    elif local_v2 is None:
        mss, timestamps, hlcvs, btc_usd_prices = await prepare_hlcvs(
            config,
            exchange,
            force_refetch_gaps=force_refetch_gaps,
            skip_v2_local=True,
        )
    coins = sorted([coin for coin in mss.keys() if not coin.startswith("__")])
    ensure_valid_index_metadata(mss, hlcvs, coins, warmup_map)
    warn_hlcv_valid_range_coverage(config, coins, mss, timestamps)
    logging.info(f"Finished preparing hlcvs data for {exchange}. Shape: {hlcvs.shape}")
    try:
        cache_dir = save_coins_hlcvs_to_cache(
            config,
            coins,
            hlcvs,
            exchange,
            mss,
            btc_usd_prices,
            timestamps,
            warmup_minutes=backtest_warmup_minutes,
        )
    except Exception as e:
        logging.error(f"Failed to save hlcvs to cache: {e}")
        traceback.print_exc()
        cache_dir = ""
    return coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, timestamps


def _coerce_config_bool(value, *, field_name):
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
        raise ValueError(f"{field_name} must be a boolean; got {value!r}")
    return bool(value)


def get_backtest_execution_settings(config, *, is_runtime_compiled: bool = False) -> BacktestExecutionSettings:
    if not is_runtime_compiled:
        config = compile_runtime_config(config, runtime="backtest", record_step=False)
    market_order_slippage_pct = float(
        get_optional_config_value(config, "backtest.market_order_slippage_pct", 0.0005) or 0.0
    )
    if market_order_slippage_pct < 0.0:
        raise ValueError("backtest.market_order_slippage_pct must be >= 0.0")
    market_orders_allowed_raw = get_optional_live_value(config, "market_orders_allowed", False)
    market_orders_allowed = _coerce_config_bool(
        market_orders_allowed_raw,
        field_name="live.market_orders_allowed",
    )
    market_order_near_touch_threshold_raw = get_optional_live_value(
        config, "market_order_near_touch_threshold", 0.001
    )
    market_order_near_touch_threshold = float(market_order_near_touch_threshold_raw or 0.0)
    if market_order_near_touch_threshold < 0.0:
        raise ValueError(
            "live.market_order_near_touch_threshold must be >= 0.0"
        )
    pnls_max_lookback = parse_pnls_max_lookback_days(
        get_optional_live_value(config, "pnls_max_lookback_days", 30.0),
        field_name="live.pnls_max_lookback_days",
    )
    return BacktestExecutionSettings(
        market_orders_allowed=market_orders_allowed,
        market_order_near_touch_threshold=market_order_near_touch_threshold,
        market_order_slippage_pct=market_order_slippage_pct,
        pnls_max_lookback_days=pnls_max_lookback.display_value,
        pnls_max_lookback_days_backtest_value=pnls_max_lookback.to_backtest_days_value(),
    )


def log_backtest_execution_settings(execution_settings: BacktestExecutionSettings | None) -> None:
    if execution_settings is None:
        return
    logging.info("[backtest] effective execution settings:")
    logging.info(
        "[backtest]   market_orders_allowed = %s (live)",
        execution_settings.market_orders_allowed,
    )
    logging.info(
        "[backtest]   market_order_near_touch_threshold = %s (live)",
        execution_settings.market_order_near_touch_threshold,
    )
    logging.info(
        "[backtest]   market_order_slippage_pct = %s (backtest)",
        execution_settings.market_order_slippage_pct,
    )
    logging.info(
        "[backtest]   pnls_max_lookback_days = %s (live)",
        execution_settings.pnls_max_lookback_days,
    )


def prep_backtest_args(
    config,
    mss,
    exchange,
    exchange_params=None,
    backtest_params=None,
    execution_settings: BacktestExecutionSettings | None = None,
    *,
    is_runtime_compiled: bool = False,
    metrics_only: bool = False,
):
    if not is_runtime_compiled:
        config = compile_runtime_config(config, runtime="backtest", record_step=False)
    if execution_settings is None:
        execution_settings = get_backtest_execution_settings(config, is_runtime_compiled=True)
    coins = sorted(set(require_config_value(config, f"backtest.coins.{exchange}")))
    candle_interval = int(config.get("backtest", {}).get("candle_interval_minutes", 1) or 1)
    bot_params_list = []
    bot_params_template = deepcopy(require_config_value(config, "bot"))
    for coin in coins:
        coin_specific_bot_params = deepcopy(bot_params_template)
        if coin in config.get("coin_overrides", {}):
            for pside in ["long", "short"]:
                for key in config["coin_overrides"][coin].get("bot", {}).get(pside, {}):
                    coin_specific_bot_params[pside][key] = config["coin_overrides"][coin]["bot"][
                        pside
                    ][key]
                coin_specific_bot_params[pside]["is_forced_active"] = (
                    config["coin_overrides"].get("live", {}).get(f"forced_mode_{pside}", "")
                    == "normal"
                )
        for pside in ["long", "short"]:
            if "wallet_exposure_limit" not in config["coin_overrides"].get(coin, {}).get(
                "bot", {}
            ).get(pside, {}):
                coin_specific_bot_params[pside]["wallet_exposure_limit"] = -1.0
        bot_params_list.append(coin_specific_bot_params)
    if exchange_params is None:
        exchange_params = [
            {
                "qty_step": mss[coin]["qty_step"],
                "price_step": mss[coin]["price_step"],
                "min_qty": mss[coin]["min_qty"],
                "min_cost": mss[coin]["min_cost"],
                "c_mult": mss[coin]["c_mult"],
                "maker_fee": float(mss[coin].get("maker_fee", mss[coin].get("maker", 0.0002))),
                "taker_fee": float(mss[coin].get("taker_fee", mss[coin].get("taker", 0.00055))),
            }
            for coin in coins
        ]
    if backtest_params is None:
        hard_stop_cfg_long, hard_stop_cfg_short = _resolve_backtest_hsl_configs(config)
        hsl_signal_mode = _resolve_backtest_hsl_signal_mode(config)
        if not isinstance(hard_stop_cfg_long, dict) or not isinstance(hard_stop_cfg_short, dict):
            raise TypeError("HSL pside configs must be dicts")

        def _normalize_hsl_cfg(cfg: dict, path_prefix: str) -> dict:
            tier_ratios = cfg.get("tier_ratios")
            if not isinstance(tier_ratios, dict):
                raise TypeError(
                    f"{path_prefix}.tier_ratios must be a dict, got {type(tier_ratios).__name__}"
                )
            enabled = bool(cfg["enabled"])
            red_threshold = float(cfg["red_threshold"])
            ema_span_minutes = float(cfg["ema_span_minutes"])
            cooldown_minutes_after_red = float(cfg["cooldown_minutes_after_red"])
            no_restart_drawdown_threshold = float(cfg["no_restart_drawdown_threshold"])
            tier_ratio_yellow = float(tier_ratios["yellow"])
            tier_ratio_orange = float(tier_ratios["orange"])
            orange_tier_mode = str(cfg["orange_tier_mode"])
            panic_close_order_type = str(cfg["panic_close_order_type"])
            if enabled and red_threshold <= 0.0:
                raise ValueError(f"{path_prefix}.red_threshold must be > 0.0 when enabled")
            if enabled and ema_span_minutes <= 0.0:
                raise ValueError(f"{path_prefix}.ema_span_minutes must be > 0.0 when enabled")
            if cooldown_minutes_after_red < 0.0:
                raise ValueError(f"{path_prefix}.cooldown_minutes_after_red must be >= 0.0")
            if no_restart_drawdown_threshold < red_threshold:
                logging.info(
                    "[config] clamped %s.no_restart_drawdown_threshold %.6f -> %.6f to match red_threshold",
                    path_prefix,
                    no_restart_drawdown_threshold,
                    red_threshold,
                )
                no_restart_drawdown_threshold = red_threshold
            if not (red_threshold <= no_restart_drawdown_threshold <= 1.0):
                raise ValueError(
                    f"{path_prefix}.no_restart_drawdown_threshold must satisfy red_threshold <= no_restart_drawdown_threshold <= 1.0"
                )
            if not (0.0 < tier_ratio_yellow < tier_ratio_orange < 1.0):
                raise ValueError(f"{path_prefix}.tier_ratios must satisfy 0 < yellow < orange < 1")
            if orange_tier_mode not in {"graceful_stop", "tp_only_with_active_entry_cancellation"}:
                raise ValueError(
                    f"{path_prefix}.orange_tier_mode must be one of {{graceful_stop, tp_only_with_active_entry_cancellation}}"
                )
            if panic_close_order_type not in {"market", "limit"}:
                raise ValueError(f"{path_prefix}.panic_close_order_type must be one of {{market, limit}}")
            return {
                "enabled": enabled,
                "signal_mode": hsl_signal_mode,
                "red_threshold": red_threshold,
                "ema_span_minutes": ema_span_minutes,
                "cooldown_minutes_after_red": cooldown_minutes_after_red,
                "no_restart_drawdown_threshold": no_restart_drawdown_threshold,
                "tier_ratios": {
                    "yellow": tier_ratio_yellow,
                    "orange": tier_ratio_orange,
                },
                "orange_tier_mode": orange_tier_mode,
                "panic_close_order_type": panic_close_order_type,
            }

        hard_stop_cfg_long = _normalize_hsl_cfg(hard_stop_cfg_long, "bot.long.hsl")
        hard_stop_cfg_short = _normalize_hsl_cfg(hard_stop_cfg_short, "bot.short.hsl")
        liquidation_threshold = float(
            get_optional_config_value(config, "backtest.liquidation_threshold", 0.05) or 0.0
        )
        if not (0.0 <= liquidation_threshold < 1.0):
            raise ValueError("backtest.liquidation_threshold must satisfy 0.0 <= x < 1.0")
        btc_collateral_cap = float(require_config_value(config, "backtest.btc_collateral_cap"))
        btc_collateral_ltv_cap = require_config_value(config, "backtest.btc_collateral_ltv_cap")
        if btc_collateral_ltv_cap is not None:
            btc_collateral_ltv_cap = float(btc_collateral_ltv_cap)
        maker_fee_override = get_optional_config_value(config, "backtest.maker_fee_override", None)
        taker_fee_override = get_optional_config_value(config, "backtest.taker_fee_override", None)
        if maker_fee_override is None:
            maker_fee = mss[coins[0]]["maker"]
        else:
            maker_fee = float(maker_fee_override)
        if taker_fee_override is None:
            taker_fee = mss[coins[0]].get("taker_fee", mss[coins[0]].get("taker", 0.00055))
        else:
            taker_fee = float(taker_fee_override)
        backtest_params = {
            "starting_balance": require_config_value(config, "backtest.starting_balance"),
            "maker_fee": maker_fee,
            "taker_fee": taker_fee,
            "coins": coins,
            "btc_collateral_cap": btc_collateral_cap,
            "btc_collateral_ltv_cap": btc_collateral_ltv_cap,
            "requested_start_timestamp_ms": 0,
            "first_valid_indices": [],
            "last_valid_indices": [],
            "warmup_minutes": [],
            "trade_start_indices": [],
            "global_warmup_bars": 0,
            "metrics_only": bool(metrics_only),
            "filter_by_min_effective_cost": bool(
                require_config_value(config, "backtest.filter_by_min_effective_cost")
            ),
            "dynamic_wel_by_tradability": bool(
                require_config_value(config, "backtest.dynamic_wel_by_tradability")
            ),
            "hedge_mode": bool(require_config_value(config, "live.hedge_mode")),
            "max_realized_loss_pct": float(require_config_value(config, "live.max_realized_loss_pct")),
            "pnls_max_lookback_days": execution_settings.pnls_max_lookback_days_backtest_value,
            "equity_hard_stop_loss": hard_stop_cfg_long,
            "market_order_slippage_pct": execution_settings.market_order_slippage_pct,
            "market_orders_allowed": execution_settings.market_orders_allowed,
            "market_order_near_touch_threshold": execution_settings.market_order_near_touch_threshold,
            "liquidation_threshold": liquidation_threshold,
        }
    return bot_params_list, exchange_params, backtest_params


def expand_analysis(analysis_usd, analysis_btc, fills, equities_array, config):
    analysis_usd = dict(analysis_usd)
    analysis_btc = dict(analysis_btc)
    keys = ["adg", "adg_w", "mdg", "mdg_w", "gain"]
    for pside in ["long", "short"]:
        twel = float(require_config_value(config, f"bot.{pside}.total_wallet_exposure_limit"))
        for key in keys:
            analysis_usd[f"{key}_per_exposure_{pside}"] = (
                (analysis_usd[key] / twel if twel > 0.0 else 0.0)
                if analysis_usd[key] is not None
                else None
            )
            analysis_btc[f"{key}_per_exposure_{pside}"] = (
                (analysis_btc[key] / twel if twel > 0.0 else 0.0)
                if analysis_btc[key] is not None
                else None
            )

    result = {}

    def _scalar_values_match(usd_val, btc_val) -> bool:
        if usd_val is None or btc_val is None:
            return False
        if isinstance(usd_val, numbers.Integral) and isinstance(btc_val, numbers.Integral):
            return usd_val == btc_val
        if isinstance(usd_val, bool) and isinstance(btc_val, bool):
            return usd_val == btc_val
        try:
            return bool(np.isclose(usd_val, btc_val, equal_nan=True))
        except Exception:
            return usd_val == btc_val

    def _is_shared_key(key: str, usd_val, btc_val) -> bool:
        if key in ANALYSIS_SHARED_KEYS:
            return True
        if key.startswith(ANALYSIS_SHARED_PREFIXES):
            return True
        if isinstance(usd_val, (bool, numbers.Integral)) and isinstance(
            btc_val, (bool, numbers.Integral)
        ):
            return usd_val == btc_val
        return False

    for key in sorted(set(analysis_usd) | set(analysis_btc)):
        usd_val = analysis_usd.get(key)
        btc_val = analysis_btc.get(key)
        if not _is_shared_key(key, usd_val, btc_val):
            continue
        if usd_val is not None:
            result[key] = usd_val
            if btc_val is not None and not _scalar_values_match(usd_val, btc_val):
                logging.debug(
                    "shared metric %s differs across denominations: usd=%s btc=%s",
                    key,
                    usd_val,
                    btc_val,
                )
        elif btc_val is not None:
            result[key] = btc_val
        analysis_usd.pop(key, None)
        analysis_btc.pop(key, None)

    def _add_metrics(metrics: dict, suffix: str):
        for key, value in metrics.items():
            suffix_lower = suffix.lower()
            key_lower = key.lower()
            if f"_{suffix_lower}" in key_lower:
                normalized_key = key
            else:
                normalized_key = f"{key}_{suffix}"
            result[normalized_key] = value

    _add_metrics(analysis_usd, "usd")
    _add_metrics(analysis_btc, "btc")

    return result


def run_backtest(
    hlcvs,
    mss,
    config: dict,
    exchange: str,
    btc_usd_prices,
    timestamps=None,
    *,
    return_payload: bool = False,
):
    """
    Backwards-compatible entry point that builds a payload and executes it immediately.
    """

    logging.info(f"Backtesting {exchange}...")
    sts = utc_ms()
    payload = build_backtest_payload(
        hlcvs,
        mss,
        config,
        exchange,
        btc_usd_prices,
        timestamps,
    )
    log_backtest_execution_settings(payload.execution_settings)
    fills, equities_array, analysis = execute_backtest(payload, config)
    logging.info(f"seconds elapsed for backtest: {(utc_ms() - sts) / 1000:.4f}")
    if return_payload:
        return fills, equities_array, analysis, payload
    return fills, equities_array, analysis


def post_process(
    config,
    hlcvs,
    fills,
    equities_array,
    btc_usd_prices,
    analysis,
    results_path,
    exchange,
    label=None,
    plot_hlcvs=None,
    hard_stop_plot_data=None,
):
    sts = utc_ms()
    disabled_plot_groups = parse_disabled_plot_groups(config.get("disable_plotting"))
    equities_array = np.asarray(equities_array)
    balance_sample_divider = get_optional_config_value(config, "backtest.balance_sample_divider", 60)
    try:
        balance_sample_divider = int(round(float(balance_sample_divider)))
    except (TypeError, ValueError):
        balance_sample_divider = 60
    balance_sample_divider = max(1, balance_sample_divider)
    fdf, analysis_py, bal_eq = process_forager_fills(
        fills,
        require_config_value(config, f"backtest.coins.{exchange}"),
        hlcvs,
        equities_array,
        balance_sample_divider=balance_sample_divider,
    )
    for k in analysis_py:
        if k not in analysis:
            analysis[k] = analysis_py[k]
    logging.info(f"seconds elapsed for analysis: {(utc_ms() - sts) / 1000:.4f}")
    label_prefix = f"[{label}] " if label else ""
    visible_analysis = filter_analysis_for_visibility(analysis, config)
    if visible_analysis.shown_count < visible_analysis.total_count:
        print(
            f"{label_prefix}Showing {visible_analysis.shown_count} of "
            f"{visible_analysis.total_count} metrics "
            "(set backtest.visible_metrics=[] to show all)."
        )
    print(f"{label_prefix}{pprint.pformat(visible_analysis.analysis)}")
    results_path = make_get_filepath(
        oj(results_path, f"{ts_to_date(utc_ms())[:19].replace(':', '_')}", "")
    )
    json.dump(analysis, open(f"{results_path}analysis.json", "w"), indent=4, sort_keys=True)
    sanitized_config = sanitize_prepared_config_for_dump(config)
    dump_config(sanitized_config, f"{results_path}config.json")
    dump_backtest_dataset_metadata(config, exchange, results_path)
    fdf.to_csv(f"{results_path}fills.csv")
    bal_eq.to_csv(oj(results_path, "balance_and_equity.csv.gz"), compression="gzip")
    if "balance" not in disabled_plot_groups:
        balance_figs = create_forager_balance_figures(
            bal_eq,
            include_logy=True,
            autoplot=False,
            return_figures=True,
        )
        save_figures(balance_figs, results_path)
    if "twe" not in disabled_plot_groups:
        twe_figs = create_forager_twe_figure(fdf, autoplot=False, return_figures=True)
        save_figures(twe_figs, results_path)
    if "pnl" not in disabled_plot_groups:
        pnl_figs = create_forager_pnl_figure(
            fdf,
            bal_eq,
            balance_sample_divider=balance_sample_divider,
            autoplot=False,
            return_figures=True,
        )
        save_figures(pnl_figs, results_path)
    if "hard_stop" not in disabled_plot_groups:
        hard_stop_figs = create_forager_hard_stop_drawdown_figure(
            bal_eq,
            config,
            hard_stop_plot_data=hard_stop_plot_data,
            autoplot=False,
            return_figures=True,
        )
        if hard_stop_figs:
            save_figures(hard_stop_figs, results_path)
    if "coin_fills" not in disabled_plot_groups:
        try:
            coins = require_config_value(config, f"backtest.coins.{exchange}")
            fills_plot_dir = oj(results_path, "fills_plots")
            hlcvs_for_plot = plot_hlcvs if plot_hlcvs is not None else hlcvs

            def _save_coin_figure(name, fig):
                save_figures({name: fig}, fills_plot_dir, close=True)

            create_forager_coin_figures(
                coins,
                fdf,
                hlcvs_for_plot,
                on_figure=_save_coin_figure,
                close_after_callback=False,
            )
        except Exception as e:
            logging.info(f"Error creating fill plots: {e}")


async def main():
    raw_argv = sys.argv[1:]
    help_all = help_all_requested(raw_argv)
    parser = build_command_parser(
        prog=get_cli_prog("backtest"),
        description="run backtest",
        usage="%(prog)s [config_path] [options]",
        epilog=(
            "Examples:\n"
            "  passivbot backtest configs/examples/default_trailing_grid_long_npos7.json -s XMR -sd 2025 --suite n\n"
            "  passivbot backtest -e bybit -s BTC,ETH -sd 2024-01 -ed 2024-06\n"
            "\n"
            "Use --help-all to show every config override flag."
        ),
    )
    parser.add_argument(
        "config_path",
        type=str,
        default=None,
        nargs="?",
        help="path to json/hjson passivbot config (defaults to in-code schema defaults if omitted)",
    )
    add_help_all_argument(
        parser,
        help_all=help_all,
        help_text="Show all backtest override flags, including advanced config overrides.",
    )

    logging_group = parser.add_argument_group("Logging")
    logging_group.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        help="Logging verbosity (warning, info, debug, trace or 0-3).",
    )
    suite_group = parser.add_argument_group("Suite")
    suite_group.add_argument(
        "--suite",
        nargs="?",
        const="true",
        default=None,
        type=str2bool,
        metavar="y/n",
        help="Enable or disable suite mode (omit to use config's suite_enabled setting).",
    )
    suite_group.add_argument(
        "--scenarios",
        "-sc",
        type=str,
        default=None,
        metavar="LABELS",
        help="Comma-separated list of scenario labels to run (implies --suite y). "
        "Example: --scenarios base,binance_only",
    )
    suite_group.add_argument(
        "--suite-config",
        type=str,
        default=None,
        help="Optional config file providing backtest.suite overrides.",
    )

    runtime_group = parser.add_argument_group("Backtest Runtime")
    runtime_group.add_argument(
        "--disable_plotting",
        "-dp",
        dest="disable_plotting",
        nargs="?",
        const="all",
        default=None,
        help=(
            "Disable selected plot groups. Use without a value to disable all plotting. "
            "Allowed values: all, summary, balance, twe, pnl, coin_fills, or a comma-separated combination."
        ),
    )
    runtime_group.add_argument(
        "--cm-debug",
        nargs="?",
        const=1,
        type=int,
        default=None,
        help=(
            "CandlestickManager debug level (0=off, 1=network only, 2=all). "
            "If passed without value, defaults to 1. Overrides config "
            "backtest.cm_debug_level."
            if help_all
            else argparse.SUPPRESS
        ),
    )
    runtime_group.add_argument(
        "--cm-progress",
        nargs="?",
        const=30.0,
        type=float,
        default=None,
        help=(
            "CandlestickManager INFO progress log interval in seconds (0 disables). "
            "If passed without value, defaults to 30. Overrides config "
            "backtest.cm_progress_log_interval_seconds."
            if help_all
            else argparse.SUPPRESS
        ),
    )
    runtime_group.add_argument(
        "--cm-remote",
        action="store_true",
        help=(
            "Show a per-exchange per-coin remote request progress bar "
            "(updates on every CCXT/archive request)."
            if help_all
            else argparse.SUPPRESS
        ),
    )
    runtime_group.add_argument(
        "--force-refetch-gaps",
        action="store_true",
        help=(
            "Force refetch all known gaps (clears gap cache before downloading). "
            "Useful when you suspect gaps may have been incorrectly marked as persistent."
            if help_all
            else argparse.SUPPRESS
        ),
    )

    group_map = {
        "Coin Selection": parser.add_argument_group("Coin Selection"),
        "Date Range": parser.add_argument_group("Date Range"),
        "Backtest Runtime": runtime_group,
        "Suite": suite_group,
        "Output / Analysis": parser.add_argument_group("Output / Analysis"),
        "Logging": logging_group,
        "Advanced Overrides": parser.add_argument_group("Advanced Overrides"),
    }

    template_config = project_template_config_for_cli(get_template_config(), "backtest")
    if "logging" in template_config and isinstance(template_config["logging"], dict):
        template_config["logging"].pop("level", None)
    allowed_config_keys = add_config_arguments(
        parser,
        template_config,
        command="backtest",
        help_all=help_all,
        group_map=group_map,
    )
    raw_args = _normalize_optional_bool_flag(expand_help_all_argv(raw_argv), "--suite")
    args = parser.parse_args(raw_args)
    cli_log_level = args.log_level
    initial_log_level = resolve_log_level(cli_log_level, None, fallback=1)
    configure_logging(debug=initial_log_level)
    source_config, base_config_path, raw_snapshot = load_input_config(args.config_path)
    update_config_with_args(source_config, args, verbose=True, allowed_keys=allowed_config_keys)
    config = prepare_config(
        source_config,
        base_config_path=base_config_path,
        verbose=False,
        log_config_transforms=True,
        raw_snapshot=raw_snapshot,
    )
    config_logging_value = get_optional_config_value(config, "logging.level", None)
    effective_log_level = resolve_log_level(cli_log_level, config_logging_value, fallback=1)
    if effective_log_level != initial_log_level:
        configure_logging(debug=effective_log_level)
    logging_section = config.get("logging")
    if not isinstance(logging_section, dict):
        logging_section = {}
    config["logging"] = logging_section
    logging_section["level"] = effective_log_level

    if args.cm_debug is not None:
        config.setdefault("backtest", {})
        config["backtest"]["cm_debug_level"] = int(args.cm_debug)
    if args.cm_progress is not None:
        config.setdefault("backtest", {})
        config["backtest"]["cm_progress_log_interval_seconds"] = float(args.cm_progress)
    if bool(getattr(args, "cm_remote", False)):
        config.setdefault("backtest", {})
        config["backtest"]["cm_remote_fetch_bar"] = True
    backtest_exchanges = require_config_value(config, "backtest.exchanges")
    config = parse_overrides(config, verbose=True)

    suite_override = None
    if args.suite_config:
        logging.info("loading suite config %s", args.suite_config)
        override_cfg = load_prepared_config(args.suite_config, verbose=False)
        override_backtest = override_cfg.get("backtest", {})
        # Support both new (scenarios at top level) and legacy (suite wrapper) formats
        if "scenarios" in override_backtest:
            suite_override = {
                "scenarios": override_backtest.get("scenarios", []),
                "aggregate": override_backtest.get("aggregate", {"default": "mean"}),
            }
        elif "suite" in override_backtest:
            # Legacy format - extract from suite wrapper
            suite_override = override_backtest["suite"]
        else:
            raise ValueError(f"Suite config {args.suite_config} does not define backtest.scenarios.")

    suite_cfg = extract_suite_config(config, suite_override)

    # Handle --scenarios filter (implies --suite y)
    scenario_filter = getattr(args, "scenarios", None)
    if scenario_filter:
        labels = [label.strip() for label in scenario_filter.split(",") if label.strip()]
        suite_cfg["scenarios"] = filter_scenarios_by_label(suite_cfg.get("scenarios", []), labels)
        suite_cfg["enabled"] = True  # --scenarios implies suite mode
        logging.info("Filtered to %d scenario(s): %s", len(labels), ", ".join(labels))

    # --suite CLI arg overrides config (applied after --scenarios so explicit --suite n wins)
    if args.suite is not None:
        recursive_config_update(config, "backtest.suite_enabled", bool(args.suite), verbose=True)
        suite_cfg["enabled"] = bool(args.suite)

    # Log disable_plotting if set (not a config key, just a runtime flag)
    if args.disable_plotting is not None:
        logging.info("changed disable_plotting False -> %s", args.disable_plotting)

    if suite_cfg.get("enabled"):
        logging.info("Running backtest suite (%d scenarios)...", len(suite_cfg.get("scenarios", [])))
        summary = await run_backtest_suite_async(
            config,
            suite_cfg,
            disable_plotting=args.disable_plotting,
        )
        logging.info(
            "Suite %s completed | scenarios=%d | output=%s",
            summary.suite_id,
            len(summary.scenarios),
            summary.output_dir,
        )
        return

    for ex in backtest_exchanges:
        await load_markets(ex)
    await format_approved_ignored_coins(config, backtest_exchanges)
    config["disable_plotting"] = args.disable_plotting if args.disable_plotting is not None else False
    config["backtest"]["cache_dir"] = {}
    config["backtest"]["coins"] = {}
    force_refetch_gaps = getattr(args, "force_refetch_gaps", False)

    # New behavior: derive data strategy from exchange count
    # - Single exchange = use that exchange's data only
    # - Multiple exchanges = best-per-coin combination (combined)
    use_combined = len(backtest_exchanges) > 1

    if use_combined:
        exchange = "combined"
        coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, timestamps = (
            await prepare_hlcvs_mss(config, exchange, force_refetch_gaps=force_refetch_gaps)
        )
        exchange_preference = defaultdict(list)
        for coin in coins:
            exchange_preference[mss[coin]["exchange"]].append(coin)
        for ex in exchange_preference:
            logging.info(f"chose {ex} for {','.join(exchange_preference[ex])}")
        config["backtest"]["coins"][exchange] = coins
        config["backtest"]["cache_dir"][exchange] = str(cache_dir)

        fills, equities_array, analysis, payload = run_backtest(
            hlcvs, mss, config, exchange, btc_usd_prices, timestamps, return_payload=True
        )
        post_process(
            config,
            hlcvs,
            fills,
            equities_array,
            btc_usd_prices,
            analysis,
            results_path,
            exchange,
            plot_hlcvs=np.asarray(payload.bundle.hlcvs),
            hard_stop_plot_data=payload.hard_stop_plot_data,
        )
    else:
        # Single exchange mode
        configs = {exchange: deepcopy(config) for exchange in backtest_exchanges}
        tasks = {}
        for exchange in backtest_exchanges:
            tasks[exchange] = asyncio.create_task(
                prepare_hlcvs_mss(configs[exchange], exchange, force_refetch_gaps=force_refetch_gaps)
            )
        for exchange in tasks:
            coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, timestamps = await tasks[
                exchange
            ]
            configs[exchange]["backtest"]["coins"][exchange] = coins
            configs[exchange]["backtest"]["cache_dir"][exchange] = str(cache_dir)
            fills, equities_array, analysis, payload = run_backtest(
                hlcvs,
                mss,
                configs[exchange],
                exchange,
                btc_usd_prices,
                timestamps,
                return_payload=True,
            )
            post_process(
                configs[exchange],
                hlcvs,
                fills,
                equities_array,
                btc_usd_prices,
                analysis,
                results_path,
                exchange,
                plot_hlcvs=np.asarray(payload.bundle.hlcvs),
                hard_stop_plot_data=payload.hard_stop_plot_data,
            )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ValueError as exc:
        logging.error("Backtest failed: %s", exc, exc_info=True)
        sys.exit(1)
