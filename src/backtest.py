import numpy as np
import pandas as pd
import os
import json
import sys
import passivbot_rust as pbr
from tools.event_loop_policy import set_windows_event_loop_policy

# on Windows this will pick the SelectorEventLoopPolicy
set_windows_event_loop_policy()
import asyncio
import argparse
from dataclasses import dataclass
from typing import Any, Iterable, Sequence
from config_utils import (
    load_config,
    dump_config,
    add_arguments_recursively,
    update_config_with_args,
    format_config,
    get_template_config,
    parse_overrides,
    require_config_value,
    require_live_value,
    get_optional_config_value,
    strip_config_metadata,
)
from utils import (
    utc_ms,
    make_get_filepath,
    load_markets,
    format_end_date,
    format_approved_ignored_coins,
    date_to_ts,
    trim_analysis_aliases,
)
from pure_funcs import (
    ts_to_date,
    sort_dict_keys,
    calc_hash,
    str2bool,
)
import pprint
from copy import deepcopy
from downloader import (
    prepare_hlcvs,
    prepare_hlcvs_combined,
    compute_backtest_warmup_minutes,
    compute_per_coin_warmup_minutes,
)
from pathlib import Path
from plotting import (
    create_forager_balance_figures,
    create_forager_coin_figures,
    save_figures,
)
from collections import defaultdict
import logging
from main import manage_rust_compilation
import gzip
import traceback

from logging_setup import configure_logging, resolve_log_level
from suite_runner import extract_suite_config, run_backtest_suite_async


def _looks_like_bool_token(value: str) -> bool:
    if value is None:
        return False
    lowered = value.lower()
    return lowered in {"1", "0", "true", "false", "t", "f", "yes", "no", "y", "n"}


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


def _build_hlcvs_bundle(
    hlcvs,
    btc_usd_prices,
    timestamps,
    config,
    exchange,
    mss,
    coins_order,
    first_valid_indices,
    last_valid_indices,
    warmup_minutes,
    trade_start_indices,
    requested_start_ts,
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
    hlcvs_arr = np.ascontiguousarray(hlcvs, dtype=np.float64)
    if subset_positions is not None:
        hlcvs_arr = np.ascontiguousarray(hlcvs_arr[:, subset_positions, :], dtype=np.float64)
    btc_arr = np.ascontiguousarray(btc_usd_prices, dtype=np.float64)
    if timestamps is None:
        timestamps_arr = np.arange(hlcvs_arr.shape[0], dtype=np.int64)
    else:
        timestamps_arr = np.ascontiguousarray(timestamps, dtype=np.int64)
    meta_overrides = mss.get("__meta__", {}) if isinstance(mss, dict) else {}
    warmup_requested = int(
        meta_overrides.get("warmup_minutes_requested", compute_backtest_warmup_minutes(config))
    )
    warmup_provided = int(meta_overrides.get("warmup_minutes_provided", warmup_requested))
    requested_ts = int(meta_overrides.get("requested_start_ts", requested_start_ts))
    effective_start_ts = int(
        meta_overrides.get("effective_start_ts", int(timestamps_arr[0]) if len(timestamps_arr) else 0)
    )
    coin_meta_entries = _build_coin_metadata_entries(
        coins_order,
        exchange,
        mss,
        first_valid_indices,
        last_valid_indices,
        warmup_minutes,
        trade_start_indices,
    )
    bundle_meta = {
        "requested_start_timestamp_ms": requested_ts,
        "effective_start_timestamp_ms": effective_start_ts,
        "warmup_minutes_requested": warmup_requested,
        "warmup_minutes_provided": warmup_provided,
        "coins": coin_meta_entries,
    }
    return pbr.HlcvsBundle(hlcvs_arr, btc_arr, timestamps_arr, bundle_meta)


@dataclass
class BacktestPayload:
    """Container for everything needed to run a backtest via the Rust engine."""

    bundle: Any
    bot_params_list: list
    exchange_params: list
    backtest_params: dict


def build_backtest_payload(
    hlcvs,
    mss,
    config: dict,
    exchange: str,
    btc_usd_prices,
    timestamps=None,
    *,
    coin_indices: list[int] | None = None,
) -> BacktestPayload:
    """
    Assemble the bundle, bot params, and metadata needed to execute a backtest.
    """

    bot_params_list, exchange_params, backtest_params = prep_backtest_args(config, mss, exchange)
    backtest_params = dict(backtest_params)
    coins_order = backtest_params.get("coins", [])

    # Inject first timestamp (ms) into backtest params; default to 0 if unknown
    try:
        first_ts_ms = int(timestamps[0]) if (timestamps is not None and len(timestamps) > 0) else 0
    except Exception:
        first_ts_ms = 0
    backtest_params["first_timestamp_ms"] = first_ts_ms

    warmup_map = compute_per_coin_warmup_minutes(config)
    default_warm = int(warmup_map.get("__default__", 0))
    first_valid_indices = []
    last_valid_indices = []
    warmup_minutes = []
    trade_start_indices = []
    total_steps = hlcvs.shape[0]
    for idx, coin in enumerate(coins_order):
        meta = mss.get(coin, {}) if isinstance(mss, dict) else {}
        first_idx = int(meta.get("first_valid_index", 0))
        last_idx = int(meta.get("last_valid_index", total_steps - 1))
        if first_idx >= total_steps:
            first_idx = total_steps
        if last_idx >= total_steps:
            last_idx = total_steps - 1
        first_valid_indices.append(first_idx)
        last_valid_indices.append(last_idx)
        warm = int(meta.get("warmup_minutes", warmup_map.get(coin, default_warm)))
        warmup_minutes.append(warm)
        if first_idx > last_idx:
            trade_idx = first_idx
        else:
            trade_idx = min(last_idx, first_idx + warm)
        trade_start_indices.append(trade_idx)
    backtest_params["first_valid_indices"] = first_valid_indices
    backtest_params["last_valid_indices"] = last_valid_indices
    backtest_params["warmup_minutes"] = warmup_minutes
    backtest_params["trade_start_indices"] = trade_start_indices
    backtest_params["global_warmup_bars"] = compute_backtest_warmup_minutes(config)

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

    bundle = _build_hlcvs_bundle(
        hlcvs,
        btc_usd_prices,
        timestamps,
        config,
        exchange,
        mss,
        coins_order,
        first_valid_indices,
        last_valid_indices,
        warmup_minutes,
        trade_start_indices,
        requested_start_ts,
        coin_indices=coin_indices,
    )

    if coin_indices is not None:
        backtest_params["active_coin_indices"] = list(range(len(coins_order)))

    return BacktestPayload(
        bundle=bundle,
        bot_params_list=bot_params_list,
        exchange_params=exchange_params,
        backtest_params=backtest_params,
    )


def execute_backtest(payload: BacktestPayload, config: dict):
    """
    Execute a prepared backtest payload and expand the resulting analysis.
    """

    (
        fills,
        equities_array,
        analysis_usd,
        analysis_btc,
    ) = pbr.run_backtest_bundle(
        payload.bundle,
        payload.bot_params_list,
        payload.exchange_params,
        payload.backtest_params,
    )

    equities_array = np.asarray(equities_array)
    analysis = expand_analysis(analysis_usd, analysis_btc, fills, equities_array, config)
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
    btc_np = np.ascontiguousarray(np.asarray(payload.bundle.btc_usd), dtype=np.float64)
    ts_np = np.ascontiguousarray(np.asarray(payload.bundle.timestamps), dtype=np.int64)

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
            "wallet_exposure",
            "total_wallet_exposure",
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
        analysis_appendix[f"loss_profit_ratio_{pside}"] = abs(loss / profit)
    analysis_appendix["pnl_ratio_long_short"] = pnls["long"] / (pnls["long"] + pnls["short"])
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
        usd_cash_series = pd.Series(dtype=float, name="usd_cash_wallet")
        usd_total_balance_series = pd.Series(dtype=float, name="usd_total_balance")
        btc_cash_series = pd.Series(dtype=float, name="btc_cash_wallet")
        btc_total_balance_series = pd.Series(dtype=float, name="btc_total_balance")
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
    bal_eq = pd.concat(
        [
            usd_cash_series,
            usd_total_balance_series,
            edf,
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
                    "btc_cash_wallet",
                    "btc_total_balance",
                    "btc_total_equity",
                ]
            )
            .ffill()
            .bfill()
        )
        if sample_divider > 1 and not bal_eq.empty:
            try:
                bal_eq = bal_eq.resample(f"{sample_divider}min").last()
            except ValueError:
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
    approved_coins = require_live_value(config, "approved_coins")
    minimum_coin_age = require_live_value(config, "minimum_coin_age_days")
    coin_sources = config.get("backtest", {}).get("coin_sources") or {}
    coin_sources_sorted = sorted((str(k), str(v)) for k, v in coin_sources.items())
    to_hash = {
        "coins": approved_coins,
        "end_date": format_end_date(require_config_value(config, "backtest.end_date")),
        "start_date": require_config_value(config, "backtest.start_date"),
        "exchange": exchanges_cfg if exchange == "combined" else exchange,
        "minimum_coin_age_days": minimum_coin_age,
        "gap_tolerance_ohlcvs_minutes": require_config_value(
            config, "backtest.gap_tolerance_ohlcvs_minutes"
        ),
        "warmup_minutes": compute_backtest_warmup_minutes(config),
        "coin_sources": coin_sources_sorted,
    }
    return calc_hash(to_hash)


def load_coins_hlcvs_from_cache(config, exchange):
    cache_hash = get_cache_hash(config, exchange)
    cache_dir = Path("caches") / "hlcvs_data" / cache_hash[:16]
    compress_cache = bool(require_config_value(config, "backtest.compress_cache"))
    if os.path.exists(cache_dir):
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
):
    cache_hash = get_cache_hash(config, exchange)
    cache_dir = Path("caches") / "hlcvs_data" / cache_hash[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    is_compressed = bool(require_config_value(config, "backtest.compress_cache"))
    expected_files = [
        "coins.json",
        "hlcvs.npy.gz" if is_compressed else "hlcvs.npy",
        "btc_usd_prices.npy.gz" if is_compressed else "btc_usd_prices.npy",
    ]
    if timestamps is not None:
        expected_files.append("timestamps.npy.gz" if is_compressed else "timestamps.npy")
    if all((cache_dir / fname).exists() for fname in expected_files):
        return
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
        warm_minutes = int(meta.get("warmup_minutes", warmup_map.get(coin, default_warm)))
        meta["warmup_minutes"] = warm_minutes
        if first_idx > last_idx:
            trade_start_idx = first_idx
        else:
            trade_start_idx = min(last_idx, first_idx + warm_minutes)
        meta["trade_start_index"] = trade_start_idx


async def prepare_hlcvs_mss(config, exchange):
    base_dir = require_config_value(config, "backtest.base_dir")
    results_path = oj(base_dir, exchange, "")
    warmup_map = compute_per_coin_warmup_minutes(config)
    default_warm = int(warmup_map.get("__default__", 0))
    try:
        sts = utc_ms()
        result = load_coins_hlcvs_from_cache(config, exchange)
        if result:
            logging.info(f"Seconds to load cache: {(utc_ms() - sts) / 1000:.4f}")
            cache_dir, coins, hlcvs, mss, results_path, btc_usd_prices, timestamps = result
            logging.info(f"Successfully loaded hlcvs data from cache")
            ensure_valid_index_metadata(mss, hlcvs, coins, warmup_map)
            # Pass through cached timestamps if they were stored; fall back to None otherwise
            return coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, timestamps
    except Exception as e:
        logging.info(f"Unable to load hlcvs data from cache: {e}. Fetching...")
    if exchange == "combined":
        forced_sources = config.get("backtest", {}).get("coin_sources")
        mss, timestamps, hlcvs, btc_usd_prices = await prepare_hlcvs_combined(
            config, forced_sources=forced_sources
        )
    else:
        mss, timestamps, hlcvs, btc_usd_prices = await prepare_hlcvs(config, exchange)
    coins = sorted([coin for coin in mss.keys() if not coin.startswith("__")])
    ensure_valid_index_metadata(mss, hlcvs, coins, warmup_map)
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
        )
    except Exception as e:
        logging.error(f"Failed to save hlcvs to cache: {e}")
        traceback.print_exc()
        cache_dir = ""
    return coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, timestamps


def prep_backtest_args(config, mss, exchange, exchange_params=None, backtest_params=None):
    coins = sorted(set(require_config_value(config, f"backtest.coins.{exchange}")))
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
            {k: mss[coin][k] for k in ["qty_step", "price_step", "min_qty", "min_cost", "c_mult"]}
            for coin in coins
        ]
    if backtest_params is None:
        btc_collateral_cap = float(require_config_value(config, "backtest.btc_collateral_cap"))
        btc_collateral_ltv_cap = require_config_value(config, "backtest.btc_collateral_ltv_cap")
        if btc_collateral_ltv_cap is not None:
            btc_collateral_ltv_cap = float(btc_collateral_ltv_cap)
        backtest_params = {
            "starting_balance": require_config_value(config, "backtest.starting_balance"),
            "maker_fee": mss[coins[0]]["maker"],
            "coins": coins,
            "btc_collateral_cap": btc_collateral_cap,
            "btc_collateral_ltv_cap": btc_collateral_ltv_cap,
            "requested_start_timestamp_ms": 0,
            "first_valid_indices": [],
            "last_valid_indices": [],
            "warmup_minutes": [],
            "trade_start_indices": [],
            "global_warmup_bars": 0,
            "metrics_only": False,
            "filter_by_min_effective_cost": bool(
                require_config_value(config, "backtest.filter_by_min_effective_cost")
            ),
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

    shared_keys = {
        "positions_held_per_day",
        "position_held_hours_mean",
        "position_held_hours_max",
        "position_held_hours_median",
        "position_unchanged_hours_max",
        "loss_profit_ratio",
        "loss_profit_ratio_w",
        "volume_pct_per_day_avg",
        "volume_pct_per_day_avg_w",
        "peak_recovery_hours_pnl",
        "total_wallet_exposure_max",
        "total_wallet_exposure_mean",
        "total_wallet_exposure_median",
        "entry_initial_balance_pct_long",
        "entry_initial_balance_pct_short",
    }

    result = {}

    for key in shared_keys:
        usd_val = analysis_usd.pop(key, None)
        btc_val = analysis_btc.pop(key, None)
        if usd_val is not None:
            result[key] = usd_val
            if btc_val is not None and not np.isclose(usd_val, btc_val, equal_nan=True):
                logging.debug(
                    "shared metric %s differs across denominations: usd=%s btc=%s",
                    key,
                    usd_val,
                    btc_val,
                )
        elif btc_val is not None:
            result[key] = btc_val

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


def run_backtest(hlcvs, mss, config: dict, exchange: str, btc_usd_prices, timestamps=None):
    """
    Backwards-compatible entry point that builds a payload and executes it immediately.
    """

    logging.info(f"Backtesting {exchange}...")
    sts = utc_ms()
    payload = build_backtest_payload(hlcvs, mss, config, exchange, btc_usd_prices, timestamps)
    fills, equities_array, analysis = execute_backtest(payload, config)
    logging.info(f"seconds elapsed for backtest: {(utc_ms() - sts) / 1000:.4f}")
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
):
    sts = utc_ms()
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
    pprint.pprint(trim_analysis_aliases(analysis))
    results_path = make_get_filepath(
        oj(results_path, f"{ts_to_date(utc_ms())[:19].replace(':', '_')}", "")
    )
    json.dump(analysis, open(f"{results_path}analysis.json", "w"), indent=4, sort_keys=True)
    config["analysis"] = analysis
    formatted_config = format_config(config)
    sanitized_config = strip_config_metadata(formatted_config)
    dump_config(sanitized_config, f"{results_path}config.json")
    fdf.to_csv(f"{results_path}fills.csv")
    bal_eq.to_csv(oj(results_path, "balance_and_equity.csv.gz"), compression="gzip")
    balance_figs = create_forager_balance_figures(
        bal_eq,
        include_logy=True,
        autoplot=False,
        return_figures=True,
    )
    save_figures(balance_figs, results_path)

    if not config["disable_plotting"]:
        try:
            coins = require_config_value(config, f"backtest.coins.{exchange}")
            fills_plot_dir = oj(results_path, "fills_plots")

            def _save_coin_figure(name, fig):
                save_figures({name: fig}, fills_plot_dir, close=True)

            create_forager_coin_figures(
                coins,
                fdf,
                hlcvs,
                on_figure=_save_coin_figure,
                close_after_callback=False,
            )
        except Exception as e:
            logging.info(f"Error creating fill plots: {e}")


async def main():
    manage_rust_compilation()
    parser = argparse.ArgumentParser(prog="backtest", description="run forager backtest")
    parser.add_argument(
        "config_path", type=str, default=None, nargs="?", help="path to json passivbot config"
    )
    parser.add_argument(
        "--disable_plotting",
        "-dp",
        dest="disable_plotting",
        action="store_true",
        help="disable plotting",
    )
    parser.add_argument(
        "--log-level",
        dest="log_level",
        default=None,
        help="Logging verbosity (warning, info, debug, trace or 0-3).",
    )
    parser.add_argument(
        "--suite",
        nargs="?",
        const="true",
        default=None,
        type=str2bool,
        metavar="y/n",
        help="Enable or disable suite mode (omit to use config).",
    )
    parser.add_argument(
        "--suite-config",
        type=str,
        default=None,
        help="Optional config file providing backtest.suite overrides.",
    )
    template_config = get_template_config()
    del template_config["optimize"]
    keep_live_keys = {
        "approved_coins",
        "ignored_coins",
        "minimum_coin_age_days",
    }
    for key in sorted(template_config["live"]):
        if key not in keep_live_keys:
            del template_config["live"][key]
    if "logging" in template_config and isinstance(template_config["logging"], dict):
        template_config["logging"].pop("level", None)
    add_arguments_recursively(parser, template_config)
    raw_args = _normalize_optional_bool_flag(sys.argv[1:], "--suite")
    args = parser.parse_args(raw_args)
    cli_log_level = args.log_level
    initial_log_level = resolve_log_level(cli_log_level, None, fallback=1)
    configure_logging(debug=initial_log_level)
    if args.config_path is None:
        logging.info(f"loading default template config configs/template.json")
        config = load_config("configs/template.json", verbose=False)
    else:
        logging.info(f"loading config {args.config_path}")
        config = load_config(args.config_path)
    update_config_with_args(config, args, verbose=True)
    config = format_config(config, verbose=False)
    config_logging_value = get_optional_config_value(config, "logging.level", None)
    effective_log_level = resolve_log_level(cli_log_level, config_logging_value, fallback=1)
    if effective_log_level != initial_log_level:
        configure_logging(debug=effective_log_level)
    logging_section = config.get("logging")
    if not isinstance(logging_section, dict):
        logging_section = {}
    config["logging"] = logging_section
    logging_section["level"] = effective_log_level
    backtest_exchanges = require_config_value(config, "backtest.exchanges")
    config = parse_overrides(config, verbose=True)

    suite_override = None
    if args.suite_config:
        logging.info("loading suite config %s", args.suite_config)
        override_cfg = load_config(args.suite_config, verbose=False)
        suite_override = override_cfg.get("backtest", {}).get("suite")
        if suite_override is None:
            raise ValueError(f"Suite config {args.suite_config} does not define backtest.suite.")

    suite_cfg = extract_suite_config(config, suite_override)
    if args.suite is not None:
        suite_cfg["enabled"] = bool(args.suite)

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
    config["disable_plotting"] = args.disable_plotting
    config["backtest"]["cache_dir"] = {}
    config["backtest"]["coins"] = {}
    if bool(require_config_value(config, "backtest.combine_ohlcvs")):
        exchange = "combined"
        coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, timestamps = (
            await prepare_hlcvs_mss(config, exchange)
        )
        exchange_preference = defaultdict(list)
        for coin in coins:
            exchange_preference[mss[coin]["exchange"]].append(coin)
        for ex in exchange_preference:
            logging.info(f"chose {ex} for {','.join(exchange_preference[ex])}")
        config["backtest"]["coins"][exchange] = coins
        config["backtest"]["cache_dir"][exchange] = str(cache_dir)

        fills, equities_array, analysis = run_backtest(
            hlcvs, mss, config, exchange, btc_usd_prices, timestamps
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
        )
    else:
        print("combined false")
        configs = {exchange: deepcopy(config) for exchange in backtest_exchanges}
        tasks = {}
        for exchange in backtest_exchanges:
            tasks[exchange] = asyncio.create_task(prepare_hlcvs_mss(configs[exchange], exchange))
        for exchange in tasks:
            coins, hlcvs, mss, results_path, cache_dir, btc_usd_prices, timestamps = await tasks[
                exchange
            ]
            configs[exchange]["backtest"]["coins"][exchange] = coins
            configs[exchange]["backtest"]["cache_dir"][exchange] = str(cache_dir)
            fills, equities_array, analysis = run_backtest(
                hlcvs, mss, configs[exchange], exchange, btc_usd_prices, timestamps
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
            )


if __name__ == "__main__":
    asyncio.run(main())
