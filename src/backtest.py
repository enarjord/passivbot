import numpy as np
import pandas as pd
import os
import json
import passivbot_rust as pbr
from tools.event_loop_policy import set_windows_event_loop_policy

# on Windows this will pick the SelectorEventLoopPolicy
set_windows_event_loop_policy()
import asyncio
import argparse
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
)
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
from plotting import plot_fills_forager
from collections import defaultdict
import matplotlib.pyplot as plt
import logging
from main import manage_rust_compilation
import gzip
import traceback

import tempfile
from contextlib import contextmanager
from logging_setup import configure_logging


@contextmanager
def create_shared_memory_file(array: np.ndarray):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        filepath = f.name
        array.tofile(f)

    try:
        yield filepath
    finally:
        # Ensure file is closed before deleting
        try:
            os.unlink(filepath)
        except PermissionError:
            import time

            time.sleep(0.1)  # Wait briefly and try again
            try:
                os.unlink(filepath)
            except Exception as e:
                print(f"Failed to delete temporary file {filepath}: {e}")


plt.rcParams["figure.figsize"] = [29, 18]


def oj(*x):
    return os.path.join(*x)


def calculate_flat_btc_balance_minutes(fills):
    if fills is None or len(fills) == 0:
        return 0.0

    change_minutes = []
    last_balance_btc = None
    for fill in fills:
        try:
            minute = float(fill[0])
            balance_btc = float(fill[5])
        except (IndexError, TypeError, ValueError):
            return 0.0

        if last_balance_btc is None or balance_btc != last_balance_btc:
            change_minutes.append(minute)
            last_balance_btc = balance_btc

    if len(change_minutes) >= 2:
        longest = max(
            change_minutes[i] - change_minutes[i - 1] for i in range(1, len(change_minutes))
        )
        return float(longest)

    return 0.0


def process_forager_fills(fills, coins, hlcvs, equities, equities_btc):
    fdf = pd.DataFrame(
        fills,
        columns=[
            "minute",
            "coin",
            "pnl",
            "fee_paid",
            "balance",
            "balance_btc",
            "balance_usd",
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
    div_by = 60  # save some disk space. Set to 1 to dump uncropped
    analysis_appendix["pnl_ratio_long_short"] = pnls["long"] / (pnls["long"] + pnls["short"])
    bdf = fdf.groupby((fdf.minute // div_by) * div_by).balance.last()
    bbdf = fdf.groupby((fdf.minute // div_by) * div_by).balance_btc.last()
    edf = pd.Series(equities).iloc[::div_by]
    ebdf = pd.Series(equities_btc).iloc[::div_by]
    nidx = np.arange(min(bdf.index[0], edf.index[0]), max(bdf.index[-1], edf.index[-1]), div_by)
    bal_eq = (
        pd.DataFrame(
            {"balance": bdf, "equity": edf, "balance_btc": bbdf, "equity_btc": ebdf}, index=nidx
        )
        .astype(float)
        .ffill()
        .bfill()
    )
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
                # Backward compatibility: default to 1.0s if not cached
                logging.info(f"{exchange} No BTC/USD prices in cache, using default array of 1.0s")
                btc_usd_prices = np.ones(hlcvs.shape[0], dtype=np.float64)
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
                # Backward compatibility: default to 1.0s if not cached
                logging.info(f"{exchange} No BTC/USD prices in cache, using default array of 1.0s")
                btc_usd_prices = np.ones(hlcvs.shape[0], dtype=np.float64)
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
        mss, timestamps, hlcvs, btc_usd_prices = await prepare_hlcvs_combined(config)
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
        backtest_params = {
            "starting_balance": require_config_value(config, "backtest.starting_balance"),
            "maker_fee": mss[coins[0]]["maker"],
            "coins": coins,
            "use_btc_collateral": bool(require_config_value(config, "backtest.use_btc_collateral")),
            "requested_start_timestamp_ms": 0,
            "first_valid_indices": [],
            "last_valid_indices": [],
            "warmup_minutes": [],
            "trade_start_indices": [],
            "global_warmup_bars": 0,
        }
    return bot_params_list, exchange_params, backtest_params


def expand_analysis(analysis_usd, analysis_btc, fills, config):
    analysis_usd["flat_btc_balance_hours"] = calculate_flat_btc_balance_minutes(fills) / 60.0
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
    if not bool(require_config_value(config, "backtest.use_btc_collateral")):
        return analysis_usd
    return {
        **{
            f"btc_{k}": v
            for k, v in analysis_btc.items()
            if "position" not in k and "volume_pct_per_day" not in k
        },
        **analysis_usd,
    }


def run_backtest(hlcvs, mss, config: dict, exchange: str, btc_usd_prices, timestamps=None):
    bot_params_list, exchange_params, backtest_params = prep_backtest_args(config, mss, exchange)
    if not bool(require_config_value(config, "backtest.use_btc_collateral")):
        btc_usd_prices = np.ones(len(btc_usd_prices))
    logging.info(f"Backtesting {exchange}...")
    sts = utc_ms()

    # Inject first timestamp (ms) into backtest params; default to 0 if unknown
    try:
        first_ts_ms = int(timestamps[0]) if (timestamps is not None and len(timestamps) > 0) else 0
    except Exception:
        first_ts_ms = 0
    backtest_params = dict(backtest_params)
    backtest_params["first_timestamp_ms"] = first_ts_ms
    coins_order = backtest_params.get("coins", [])
    warmup_map = compute_per_coin_warmup_minutes(config)
    default_warm = int(warmup_map.get("__default__", 0))
    first_valid_indices = []
    last_valid_indices = []
    warmup_minutes = []
    trade_start_indices = []
    total_steps = hlcvs.shape[0]
    for idx, coin in enumerate(coins_order):
        meta = mss.get(coin, {})
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

    # Use context managers for both HLCV and BTC/USD shared memory files
    with create_shared_memory_file(hlcvs) as shared_memory_file, create_shared_memory_file(
        btc_usd_prices
    ) as btc_usd_shared_memory_file:
        fills, equities_usd, equities_btc, analysis_usd, analysis_btc = pbr.run_backtest(
            shared_memory_file,
            hlcvs.shape,
            hlcvs.dtype.str,
            btc_usd_shared_memory_file,
            btc_usd_prices.dtype.str,
            bot_params_list,
            exchange_params,
            backtest_params,
        )

    logging.info(f"seconds elapsed for backtest: {(utc_ms() - sts) / 1000:.4f}")
    return (
        fills,
        equities_usd,
        equities_btc,
        expand_analysis(analysis_usd, analysis_btc, fills, config),
    )


def post_process(
    config,
    hlcvs,
    fills,
    equities,
    equities_btc,
    btc_usd_prices,
    analysis,
    results_path,
    exchange,
):
    sts = utc_ms()
    equities = pd.Series(equities)
    equities_btc = pd.Series(equities_btc)
    fdf, analysis_py, bal_eq = process_forager_fills(
        fills,
        require_config_value(config, f"backtest.coins.{exchange}"),
        hlcvs,
        equities,
        equities_btc,
    )
    for k in analysis_py:
        if k not in analysis:
            analysis[k] = analysis_py[k]
    logging.info(f"seconds elapsed for analysis: {(utc_ms() - sts) / 1000:.4f}")
    pprint.pprint(analysis)
    results_path = make_get_filepath(
        oj(results_path, f"{ts_to_date(utc_ms())[:19].replace(':', '_')}", "")
    )
    json.dump(analysis, open(f"{results_path}analysis.json", "w"), indent=4, sort_keys=True)
    config["analysis"] = analysis
    dump_config(config, f"{results_path}config.json")
    fdf.to_csv(f"{results_path}fills.csv")
    bal_eq.to_csv(oj(results_path, "balance_and_equity.csv"))
    plot_forager(
        results_path,
        config,
        exchange,
        fdf,
        bal_eq,
        hlcvs,
    )


def plot_forager(
    results_path,
    config: dict,
    exchange: str,
    fdf: pd.DataFrame,
    bal_eq,
    hlcvs,
):
    plots_dir = make_get_filepath(oj(results_path, "fills_plots", ""))
    plt.clf()
    bal_eq[["balance", "equity"]].plot(logy=False)
    plt.savefig(oj(results_path, "balance_and_equity.png"))
    plt.clf()
    bal_eq[["balance", "equity"]].plot(logy=True)
    plt.savefig(oj(results_path, "balance_and_equity_logy.png"))
    plt.clf()
    if bool(require_config_value(config, "backtest.use_btc_collateral")):
        plt.clf()
        bal_eq[["balance_btc", "equity_btc"]].plot(logy=False)
        plt.savefig(oj(results_path, "balance_and_equity_btc.png"))
        plt.clf()
        bal_eq[["balance_btc", "equity_btc"]].plot(logy=True)
        plt.savefig(oj(results_path, "balance_and_equity_btc_logy.png"))

    if not config["disable_plotting"]:
        for i, coin in enumerate(require_config_value(config, f"backtest.coins.{exchange}")):
            try:
                logging.info(f"Plotting fills for {coin}")
                hlcvs_df = pd.DataFrame(hlcvs[:, i, :3], columns=["high", "low", "close"])
                fdfc = fdf[fdf.coin == coin]
                plt.clf()
                plot_fills_forager(fdfc, hlcvs_df)
                plt.title(f"Fills {coin}")
                plt.xlabel = "time"
                plt.ylabel = "price"
                plt.savefig(oj(plots_dir, f"{coin}.png"))
            except Exception as e:
                logging.info(f"Error plotting {coin} {e}")


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
        "--debug-level",
        "--log-level",
        dest="debug_level",
        type=int,
        choices=[0, 1, 2, 3],
        default=None,
        help="Logging verbosity: 0=warnings, 1=info, 2=debug, 3=trace.",
    )
    template_config = get_template_config("v7")
    del template_config["optimize"]
    keep_live_keys = {
        "approved_coins",
        "ignored_coins",
        "minimum_coin_age_days",
    }
    for key in sorted(template_config["live"]):
        if key not in keep_live_keys:
            del template_config["live"][key]
    add_arguments_recursively(parser, template_config)
    args = parser.parse_args()
    initial_debug = args.debug_level if args.debug_level is not None else 1
    configure_logging(debug=initial_debug)
    if args.config_path is None:
        logging.info(f"loading default template config configs/template.json")
        config = load_config("configs/template.json", verbose=False)
    else:
        logging.info(f"loading config {args.config_path}")
        config = load_config(args.config_path)
    update_config_with_args(config, args)
    config = format_config(config, verbose=False)
    config_logging_level = get_optional_config_value(config, "logging.level", 1)
    try:
        config_logging_level = int(float(config_logging_level))
    except Exception:
        config_logging_level = 1
    if args.debug_level is None:
        effective_debug = max(0, min(config_logging_level, 3))
        configure_logging(debug=effective_debug)
    else:
        effective_debug = max(0, min(int(args.debug_level), 3))
    logging_section = config.get("logging")
    if not isinstance(logging_section, dict):
        logging_section = {}
    config["logging"] = logging_section
    logging_section["level"] = effective_debug
    backtest_exchanges = require_config_value(config, "backtest.exchanges")
    for ex in backtest_exchanges:
        await load_markets(ex)
    config = parse_overrides(config, verbose=True)
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

        fills, equities, equities_btc, analysis = run_backtest(
            hlcvs, mss, config, exchange, btc_usd_prices, timestamps
        )
        post_process(
            config,
            hlcvs,
            fills,
            equities,
            equities_btc,
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
            fills, equities, equities_btc, analysis = run_backtest(
                hlcvs, mss, configs[exchange], exchange, btc_usd_prices, timestamps
            )
            post_process(
                configs[exchange],
                hlcvs,
                fills,
                equities,
                equities_btc,
                btc_usd_prices,
                analysis,
                results_path,
                exchange,
            )


if __name__ == "__main__":
    asyncio.run(main())
