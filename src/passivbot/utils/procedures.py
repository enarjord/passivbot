import argparse
import json
import pathlib
import time
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List

import hjson
import numpy as np
import pandas as pd

from passivbot.utils.funcs.njit import calc_samples
from passivbot.utils.funcs.pure import candidate_to_live_config
from passivbot.utils.funcs.pure import config_pretty_str
from passivbot.utils.funcs.pure import date_to_ts
from passivbot.utils.funcs.pure import get_dummy_settings
from passivbot.utils.funcs.pure import get_template_live_config
from passivbot.utils.funcs.pure import numpyize
from passivbot.utils.funcs.pure import ts_to_date


def load_live_config(live_config_path: str) -> Dict[str, Any]:
    try:
        live_config = json.load(open(live_config_path))
        live_config = json.loads(
            json.dumps(live_config).replace("secondary_grid_spacing", "secondary_pprice_diff")
        )
        assert all(k in live_config["long"] for k in get_template_live_config()["long"])
        return numpyize(live_config)
    except Exception as e:
        raise Exception(f"failed to load live config {live_config_path} {e}")


def dump_live_config(config: Dict[str, Any], path: pathlib.Path) -> None:
    pretty_str = config_pretty_str(candidate_to_live_config(config))
    path.write_text(pretty_str)


def load_config_files(config_paths: List) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    for config_path in config_paths:
        try:
            loaded_config = hjson.load(open(config_path, encoding="utf-8"))
            config = {**config, **loaded_config}
        except Exception as e:
            raise Exception("failed to load config file", config_path, e)
    return config


def load_hjson_config(config_path: str) -> Dict[str, Any]:
    try:
        return hjson.load(open(config_path, encoding="utf-8"))
    except Exception as e:
        raise Exception(f"failed to load config file {config_path} {e}")


async def prepare_backtest_config(args) -> Dict[str, Any]:
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

    if args.backtests_dir is None:
        args.backtests_dir = args.basedir / "backtests"
    args.backtests_dir = args.backtests_dir.resolve()
    args.backtests_dir.mkdir(parents=True, exist_ok=True)

    backtests_session_dir: pathlib.Path = args.backtests_dir.joinpath(
        f"{config['exchange']}{'_spot' if 'spot' in config['market_type'] else ''}",
        config["symbol"],
    )
    backtests_session_dir.mkdir(parents=True, exist_ok=True)
    for key in ("caches", "optimize", "plots"):
        path = backtests_session_dir / key
        path.mkdir(parents=True, exist_ok=True)
        config[f"{path}_dirpath"] = path

    await add_market_specific_settings(config)

    return config


async def prepare_optimize_config(args) -> Dict[str, Any]:
    config = await prepare_backtest_config(args)
    config.update(load_hjson_config(args.optimize_config_path))
    for key in ["starting_configs", "iters"]:
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)
        elif key not in config:
            config[key] = None
    return config


async def add_market_specific_settings(config):
    mss = config["caches_dirpath"] / "market_specific_settings.json"
    try:
        print("fetching market_specific_settings...")
        market_specific_settings = await fetch_market_specific_settings(config)
        json.dump(market_specific_settings, open(mss, "w"), indent=4)
    except Exception as e:
        print("\nfailed to fetch market_specific_settings", e, "\n")
        try:
            if mss.exists():
                market_specific_settings = json.load(open(mss))
            print("using cached market_specific_settings")
        except Exception:
            raise Exception("failed to load cached market_specific_settings")
    config.update(market_specific_settings)


def load_exchange_key_secret(user: str) -> (str, str, str):
    try:
        keyfile = json.load(open("api-keys.json"))
        if user in keyfile:
            return keyfile[user]["exchange"], keyfile[user]["key"], keyfile[user]["secret"]
        else:
            print("Looks like the keys aren't configured yet, or you entered the wrong username!")
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
    exchange = config["exchange"]
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
    # Deferred import due to circular import issues
    from passivbot.exchanges.binance import BinanceBot

    bot = BinanceBot(config)
    await bot._init()
    return bot


async def create_binance_bot_spot(config: dict):
    # Deferred import due to circular import issues
    from passivbot.exchanges.binance_spot import BinanceBotSpot

    bot = BinanceBotSpot(config)
    await bot._init()
    return bot


async def create_bybit_bot(config: dict):
    # Deferred import due to circular import issues
    from passivbot.exchanges.bybit import Bybit

    bot = Bybit(config)
    await bot._init()
    return bot


def add_backtesting_argparse_args(parser):
    parser.add_argument(
        "-b",
        "--backtest_config",
        "--backtest-config",
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
        "--start-date",
        type=str,
        required=False,
        dest="start_date",
        default=None,
        help="specify start date, overriding value from backtest config",
    )
    parser.add_argument(
        "-ed",
        "--end_date",
        "--end-date",
        type=str,
        required=False,
        dest="end_date",
        default=None,
        help="specify end date, overriding value from backtest config",
    )
    parser.add_argument(
        "--starting_balance",
        "--starting-balance",
        type=float,
        required=False,
        dest="starting_balance",
        default=None,
        help="specify starting_balance, overriding value from backtest config",
    )
    parser.add_argument(
        "-m",
        "--market_type",
        "--market-type",
        type=str,
        choices=["futures", "spot"],
        required=False,
        dest="market_type",
        default=None,
        help="specify whether spot or futures (default), overriding value from backtest config",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        required=False,
        dest="base_dir",
        default=None,
        help="[DEPRECATED] specify the base output directory for the results",
    )
    parser.add_argument(
        "-bd",
        "--backtests-dir",
        type=pathlib.Path,
        required=False,
        dest="backtests_dir",
        default=None,
        help="Base output directory for the backtest results",
    )

    return parser


def validate_backtesting_argparse_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """
    Validates the argparse parsed arguments.
    """
    if args.base_dir and args.backtests_dir:
        parser.exit(
            status=1,
            message="'--base_dir' and '--backtests-dir' are mutually exclusive. Please use just '--backtests-dir'.",
        )
    elif args.base_dir:
        parser.exit(
            status=1, message="'--base_dir' has been deprecated. Please use '--backtests-dir'"
        )


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
    # XXX: Does not seem to be used anywhere. Remove?
    for key in ["exchange", "symbol", "spot", "start_date", "end_date"]:
        assert key in config
    start_ts = date_to_ts(config["start_date"])
    end_ts = date_to_ts(config["end_date"])
    ticks_filepath = config["basedir"].joinpath(
        "historical_data",
        config["exchange"],
        f"agg_trades_{'spot' if config['spot'] else 'futures'}",
        config["symbol"],
    )
    if not ticks_filepath.exists():
        return
    ticks_filenames = sorted(f for f in ticks_filepath.iterdir() if f.endswith(".csv"))
    ticks = np.empty((0, 3))
    sts = time.time()
    for f in ticks_filenames:
        _, _, first_ts, last_ts = map(int, str(f).replace(".csv", "").split("_"))
        if first_ts > end_ts or last_ts < start_ts:
            continue
        print(f"\rloading chunk {ts_to_date(first_ts / 1000)}", end="  ")
        tdf = pd.read_csv(ticks_filepath / f)
        tdf = tdf[(tdf.timestamp >= start_ts) & (tdf.timestamp <= end_ts)]
        ticks = np.concatenate((ticks, tdf[["timestamp", "qty", "price"]].values))
        del tdf
    samples = calc_samples(ticks[ticks[:, 0].argsort()], sec_span * 1000)
    print(
        f"took {time.time() - sts:.2f} seconds to load {len(ticks)} ticks, creating {len(samples)} samples"
    )
    del ticks
    return samples


def get_starting_configs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    starting_configs = []
    if config["starting_configs"]:
        for path in config["starting_configs"]:
            try:
                if path.isdir():
                    for fpath in path.glob("*.json"):
                        starting_configs.append(json.loads(fpath.read_text()))
                    print("Starting with all configurations in directory.")
                else:
                    starting_configs.append(json.loads(path.read_text()))
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
    except Exception:
        pass
    try:
        print(f"exception: {coro.exception()}")
    except Exception:
        pass
    try:
        print(f"result: {coro.result()}")
    except Exception:
        pass
