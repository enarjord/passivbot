import json
import pprint
import os
import hjson
import pandas as pd
import numpy as np
import glob
from time import time
from pure_funcs import (
    numpyize,
    denumpyize,
    candidate_to_live_config,
    ts_to_date,
    get_dummy_settings,
    calc_spans,
    config_pretty_str,
    date_to_ts,
    get_template_live_config,
)
from njit_funcs import calc_samples
from datetime import datetime
import traceback


def load_live_config(live_config_path: str) -> dict:
    try:
        live_config = json.load(open(live_config_path))
        for src, dst in [
            ("secondary_grid_spacing", "secondary_pprice_diff"),
            ("shrt", "short"),
            ("secondary_pbr_allocation", "secondary_allocation"),
            ("pbr_limit", "wallet_exposure_limit"),
        ]:
            live_config = json.loads(json.dumps(live_config).replace(src, dst))
        for side in ["long", "short"]:
            if "initial_eprice_ema_dist" not in live_config[side]:
                live_config[side]["initial_eprice_ema_dist"] = -1000.0
            if "ema_span_min" not in live_config[side]:
                live_config[side]["ema_span_min"] = 1
            if "ema_span_max" not in live_config[side]:
                live_config[side]["ema_span_max"] = 1
            if "auto_unstuck_wallet_exposure_threshold" not in live_config[side]:
                live_config[side]["auto_unstuck_wallet_exposure_threshold"] = 0.0
            if "auto_unstuck_ema_dist" not in live_config[side]:
                live_config[side]["auto_unstuck_ema_dist"] = 0.0
        assert all(k in live_config["long"] for k in get_template_live_config()["long"])
        return numpyize(live_config)
    except Exception as e:
        raise Exception(f"failed to load live config {live_config_path} {e}")


def dump_live_config(config: dict, path: str):
    pretty_str = config_pretty_str(candidate_to_live_config(config))
    with open(path, "w") as f:
        f.write(pretty_str)


def load_config_files(config_paths: []) -> dict:
    config = {}
    for config_path in config_paths:
        try:
            loaded_config = hjson.load(open(config_path, encoding="utf-8"))
            config = {**config, **loaded_config}
        except Exception as e:
            raise Exception("failed to load config file", config_path, e)
    return config


def load_hjson_config(config_path: str) -> dict:
    try:
        return hjson.load(open(config_path, encoding="utf-8"))
    except Exception as e:
        raise Exception(f"failed to load config file {config_path} {e}")


async def prepare_backtest_config(args) -> dict:
    """
    takes argparse args, returns dict with backtest and optimize config
    """
    config = load_hjson_config(args.backtest_config_path)
    for key in [
        "symbol",
        "user",
        "start_date",
        "end_date",
        "starting_balance",
        "market_type",
        "base_dir",
    ]:
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)
        elif key not in config:
            config[key] = None
    if args.market_type is None:
        config["spot"] = False
    else:
        config["spot"] = args.market_type == "spot"
    config["exchange"], _, _ = load_exchange_key_secret(config["user"])
    config["session_name"] = (
        f"{config['start_date'].replace(' ', '').replace(':', '').replace('.', '')}_"
        f"{config['end_date'].replace(' ', '').replace(':', '').replace('.', '')}"
    )

    if config["base_dir"].startswith("~"):
        raise Exception(
            "error: using the ~ to indicate the user's home directory is not supported"
        )

    base_dirpath = os.path.join(
        config["base_dir"],
        f"{config['exchange']}{'_spot' if 'spot' in config['market_type'] else ''}",
        config["symbol"],
    )
    config["caches_dirpath"] = make_get_filepath(
        os.path.join(base_dirpath, "caches", "")
    )
    config["optimize_dirpath"] = make_get_filepath(
        os.path.join(base_dirpath, "optimize", "")
    )
    config["plots_dirpath"] = make_get_filepath(os.path.join(base_dirpath, "plots", ""))

    await add_market_specific_settings(config)

    return config


async def prepare_optimize_config(args) -> dict:
    config = await prepare_backtest_config(args)
    config.update(load_hjson_config(args.optimize_config_path))
    for key in ["starting_configs", "iters"]:
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)
        elif key not in config:
            config[key] = None
    return config


async def add_market_specific_settings(config):
    mss = config["caches_dirpath"] + "market_specific_settings.json"
    try:
        print("fetching market_specific_settings...")
        market_specific_settings = await fetch_market_specific_settings(config)
        json.dump(market_specific_settings, open(mss, "w"), indent=4)
    except Exception as e:
        traceback.print_exc()
        print("\nfailed to fetch market_specific_settings", e, "\n")
        try:
            if os.path.exists(mss):
                market_specific_settings = json.load(open(mss))
            print("using cached market_specific_settings")
        except Exception:
            raise Exception("failed to load cached market_specific_settings")
    config.update(market_specific_settings)


def make_get_filepath(filepath: str) -> str:
    """
    if not is path, creates dir and subdirs for path, returns path
    """
    dirpath = os.path.dirname(filepath) if filepath[-1] != "/" else filepath
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    return filepath


def load_exchange_key_secret(user: str) -> (str, str, str):
    try:
        keyfile = json.load(open("api-keys.json"))
        if user in keyfile:
            return (
                keyfile[user]["exchange"],
                keyfile[user]["key"],
                keyfile[user]["secret"],
            )
        else:
            print(
                "Looks like the keys aren't configured yet, or you entered the wrong username!"
            )
        raise Exception("API KeyFile Missing!")
    except FileNotFoundError:
        print("File Not Found!")
        raise Exception("API KeyFile Missing!")


def print_(args, r=False, n=False):
    line = ts_to_date(utc_ms())[:19] + "  "
    # line = ts_to_date(local_time())[:19] + '  '
    str_args = "{} " * len(args)
    line += str_args.format(*args)
    if n:
        print("\n" + line, end=" ")
    elif r:
        print("\r" + line, end=" ")
    else:
        print(line)
    return line


async def fetch_market_specific_settings(config: dict):
    user = config["user"]
    exchange = config["exchange"]
    symbol = config["symbol"]
    tmp_live_settings = get_dummy_settings(config)
    settings_from_exchange = {}
    if exchange == "binance":
        if "spot" in config["market_type"]:
            bot = await create_binance_bot_spot(tmp_live_settings)
            settings_from_exchange["maker_fee"] = 0.001
            settings_from_exchange["taker_fee"] = 0.001
            settings_from_exchange["spot"] = True
            settings_from_exchange["hedge_mode"] = False
        else:
            bot = await create_binance_bot(tmp_live_settings)
            settings_from_exchange["maker_fee"] = 0.0002
            settings_from_exchange["taker_fee"] = 0.0004
            settings_from_exchange["spot"] = False
        settings_from_exchange["exchange"] = "binance"
    elif exchange == "bybit":
        if "spot" in config["market_type"]:
            raise Exception("spot not implemented on bybit")
        bot = await create_bybit_bot(tmp_live_settings)
        settings_from_exchange["maker_fee"] = -0.00025
        settings_from_exchange["taker_fee"] = 0.00075
        settings_from_exchange["exchange"] = "bybit"
    else:
        raise Exception(f"unknown exchange {exchange}")
    await bot.session.close()
    if "inverse" in bot.market_type:
        settings_from_exchange["inverse"] = True
    elif any(x in bot.market_type for x in ["linear", "spot"]):
        settings_from_exchange["inverse"] = False
    else:
        raise Exception("unknown market type")
    for key in [
        "max_leverage",
        "min_qty",
        "min_cost",
        "qty_step",
        "price_step",
        "max_leverage",
        "c_mult",
        "hedge_mode",
    ]:
        settings_from_exchange[key] = getattr(bot, key)
    return settings_from_exchange


async def create_binance_bot(config: dict):
    from binance import BinanceBot

    bot = BinanceBot(config)
    await bot._init()
    return bot


async def create_binance_bot_spot(config: dict):
    from binance_spot import BinanceBotSpot

    bot = BinanceBotSpot(config)
    await bot._init()
    return bot


async def create_bybit_bot(config: dict):
    from bybit import Bybit

    bot = Bybit(config)
    await bot._init()
    return bot


def add_argparse_args(parser):
    parser.add_argument("--nojit", help="disable numba", action="store_true")
    parser.add_argument(
        "-b",
        "--backtest_config",
        type=str,
        required=False,
        dest="backtest_config_path",
        default="configs/backtest/default.hjson",
        help="backtest config hjson file",
    )
    parser.add_argument(
        "-s",
        "--symbol",
        type=str,
        required=False,
        dest="symbol",
        default=None,
        help="specify symbol, overriding symbol from backtest config",
    )
    parser.add_argument(
        "-u",
        "--user",
        type=str,
        required=False,
        dest="user",
        default=None,
        help="specify user, a.k.a. account_name, overriding user from backtest config",
    )
    parser.add_argument(
        "-sd",
        "--start_date",
        type=str,
        required=False,
        dest="start_date",
        default=None,
        help="specify start date, overriding value from backtest config",
    )
    parser.add_argument(
        "-ed",
        "--end_date",
        type=str,
        required=False,
        dest="end_date",
        default=None,
        help="specify end date, overriding value from backtest config",
    )
    parser.add_argument(
        "--starting_balance",
        type=float,
        required=False,
        dest="starting_balance",
        default=None,
        help="specify starting_balance, overriding value from backtest config",
    )
    parser.add_argument(
        "-m",
        "--market_type",
        type=str,
        required=False,
        dest="market_type",
        default=None,
        help="specify whether spot or futures (default), overriding value from backtest config",
    )
    parser.add_argument(
        "-bd",
        "--base_dir",
        type=str,
        required=False,
        dest="base_dir",
        default="backtests",
        help="specify the base output directory for the results",
    )

    return parser


def make_tick_samples(config: dict, sec_span: int = 1):

    """
    makes tick samples from agg_trades
    tick samples are [(qty, price, timestamp)]
    config must include parameters
    - exchange: str
    - symbol: str
    - spot: bool
    - start_date: str
    - end_date: str
    """
    for key in ["exchange", "symbol", "spot", "start_date", "end_date"]:
        assert key in config
    start_ts = date_to_ts(config["start_date"])
    end_ts = date_to_ts(config["end_date"])
    ticks_filepath = os.path.join(
        "historical_data",
        config["exchange"],
        f"agg_trades_{'spot' if config['spot'] else 'futures'}",
        config["symbol"],
        "",
    )
    if not os.path.exists(ticks_filepath):
        return
    ticks_filenames = sorted(
        [f for f in os.listdir(ticks_filepath) if f.endswith(".csv")]
    )
    ticks = np.empty((0, 3))
    sts = time()
    for f in ticks_filenames:
        _, _, first_ts, last_ts = map(int, f.replace(".csv", "").split("_"))
        if first_ts > end_ts or last_ts < start_ts:
            continue
        print(f"\rloading chunk {ts_to_date(first_ts / 1000)}", end="  ")
        tdf = pd.read_csv(ticks_filepath + f)
        tdf = tdf[(tdf.timestamp >= start_ts) & (tdf.timestamp <= end_ts)]
        ticks = np.concatenate((ticks, tdf[["timestamp", "qty", "price"]].values))
        del tdf
    samples = calc_samples(ticks[ticks[:, 0].argsort()], sec_span * 1000)
    print(
        f"took {time() - sts:.2f} seconds to load {len(ticks)} ticks, creating {len(samples)} samples"
    )
    del ticks
    return samples


def get_starting_configs(config) -> [dict]:
    starting_configs = []
    if config["starting_configs"] is not None:
        try:
            if os.path.isdir(config["starting_configs"]):
                starting_configs = [
                    json.load(open(f))
                    for f in glob.glob(
                        os.path.join(config["starting_configs"], "*.json")
                    )
                ]
                print("Starting with all configurations in directory.")
            else:
                starting_configs = [json.load(open(config["starting_configs"]))]
                print("Starting with specified configuration.")
        except Exception as e:
            print("Could not find specified configuration.", e)
    return starting_configs


def utc_ms() -> float:
    return datetime.utcnow().timestamp() * 1000


def local_time() -> float:
    return datetime.now().astimezone().timestamp() * 1000


def print_async_exception(coro):
    try:
        print(f"returned: {coro}")
    except:
        pass
    try:
        print(f"exception: {coro.exception()}")
    except:
        pass
    try:
        print(f"result: {coro.result()}")
    except:
        pass
