import glob
import json
import os
import traceback
from datetime import datetime
from time import time
import numpy as np

try:
    import hjson
except:
    print("hjson not found, trying without...")
    pass
try:
    import pandas as pd
except:
    print("pandas not found, trying without...")
    pass

from njit_funcs import calc_samples
from pure_funcs import (
    numpyize,
    candidate_to_live_config,
    ts_to_date,
    ts_to_date_utc,
    get_dummy_settings,
    config_pretty_str,
    date_to_ts,
    get_template_live_config,
    sort_dict_keys,
    make_compatible,
    determine_passivbot_mode,
)


def load_live_config(live_config_path: str) -> dict:
    try:
        live_config = json.load(open(live_config_path))
        return sort_dict_keys(numpyize(make_compatible(live_config)))
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
        "ohlcv",
    ]:
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)
        elif key not in config:
            config[key] = None
    if args.market_type is None:
        config["spot"] = False
    else:
        config["spot"] = args.market_type == "spot"
    config["start_date"] = ts_to_date_utc(date_to_ts(config["start_date"]))[:10]
    config["end_date"] = ts_to_date_utc(date_to_ts(config["end_date"]))[:10]
    config["exchange"] = load_exchange_key_secret_passphrase(config["user"])[0]
    config["session_name"] = (
        f"{config['start_date'].replace(' ', '').replace(':', '').replace('.', '')}_"
        f"{config['end_date'].replace(' ', '').replace(':', '').replace('.', '')}"
    )

    if config["base_dir"].startswith("~"):
        raise Exception("error: using the ~ to indicate the user's home directory is not supported")

    base_dirpath = os.path.join(
        config["base_dir"],
        f"{config['exchange']}{'_spot' if 'spot' in config['market_type'] else ''}",
        config["symbol"],
    )
    config["caches_dirpath"] = make_get_filepath(os.path.join(base_dirpath, "caches", ""))
    config["optimize_dirpath"] = make_get_filepath(os.path.join(base_dirpath, "optimize", ""))
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


def load_exchange_key_secret_passphrase(
    user: str, api_keys_path="api-keys.json"
) -> (str, str, str, str):
    if api_keys_path is None:
        api_keys_path = "api-keys.json"
    try:
        keyfile = json.load(open(api_keys_path))
        if user in keyfile:
            return (
                keyfile[user]["exchange"],
                keyfile[user]["key"],
                keyfile[user]["secret"],
                keyfile[user]["passphrase"] if "passphrase" in keyfile[user] else "",
            )
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
    elif exchange == "binance_us":
        bot = await create_binance_bot_spot(tmp_live_settings)
        settings_from_exchange["maker_fee"] = 0.001
        settings_from_exchange["taker_fee"] = 0.001
        settings_from_exchange["spot"] = True
        settings_from_exchange["hedge_mode"] = False
        settings_from_exchange["exchange"] = "binance"

    elif exchange == "bybit":
        if "spot" in config["market_type"]:
            raise Exception("spot not implemented on bybit")
        bot = await create_bybit_bot(tmp_live_settings)
        settings_from_exchange["maker_fee"] = 0.0001
        settings_from_exchange["taker_fee"] = 0.0006
        settings_from_exchange["exchange"] = "bybit"
    elif exchange == "bitget":
        if "spot" in config["market_type"]:
            raise Exception("spot not implemented on bitget")
        bot = await create_bitget_bot(tmp_live_settings)
        settings_from_exchange["maker_fee"] = 0.0002
        settings_from_exchange["taker_fee"] = 0.0006
        settings_from_exchange["exchange"] = "bitget"
    elif exchange == "okx":
        if "spot" in config["market_type"]:
            raise Exception("spot not implemented on okx")
        bot = await create_okx_bot(tmp_live_settings)
        settings_from_exchange["maker_fee"] = 0.0002
        settings_from_exchange["taker_fee"] = 0.0005
        settings_from_exchange["exchange"] = "okx"
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
    from bybit import BybitBot

    bot = BybitBot(config)
    await bot._init()
    return bot


async def create_bitget_bot(config: dict):
    from bitget import BitgetBot

    bot = BitgetBot(config)
    await bot._init()
    return bot


async def create_okx_bot(config: dict):
    from okx import OKXBot

    bot = OKXBot(config)
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
        help="specify symbol(s), overriding symbol from backtest config.  "
        + "multiple symbols separated with comma",
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
        "-sb",
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
        default=None,
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
    ticks_filenames = sorted([f for f in os.listdir(ticks_filepath) if f.endswith(".csv")])
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
                    for f in glob.glob(os.path.join(config["starting_configs"], "*.json"))
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


async def init_optimizer(logging):
    import argparse
    from downloader import Downloader, load_hlc_cache

    parser = argparse.ArgumentParser(
        prog="Optimize multi symbol", description="Optimize passivbot config multi symbol"
    )
    parser.add_argument(
        "-o",
        "--optimize_config",
        type=str,
        required=False,
        dest="optimize_config_path",
        default="configs/optimize/particle_swarm_optimization.hjson",
        help="optimize config hjson file",
    )
    parser.add_argument(
        "-t",
        "--start",
        type=str,
        required=False,
        dest="starting_configs",
        default=None,
        help="start with given live configs.  single json file or dir with multiple json files",
    )
    parser.add_argument(
        "-i", "--iters", type=int, required=False, dest="iters", default=None, help="n optimize iters"
    )
    parser.add_argument(
        "-c", "--n_cpus", type=int, required=False, dest="n_cpus", default=None, help="n cpus"
    )
    parser.add_argument(
        "-le",
        "--long",
        type=str,
        required=False,
        dest="long_enabled",
        default=None,
        help="long enabled: [y/n]",
    )
    parser.add_argument(
        "-se",
        "--short",
        type=str,
        required=False,
        dest="short_enabled",
        default=None,
        help="short enabled: [y/n]",
    )
    parser.add_argument(
        "-pm",
        "--passivbot_mode",
        "--passivbot-mode",
        type=str,
        required=False,
        dest="passivbot_mode",
        default=None,
        help="passivbot mode options: [s/static_grid, r/recursive_grid, n/neat_grid, e/emas]",
    )
    parser.add_argument(
        "-oh",
        "--ohlcv",
        help="use 1m ohlcv instead of 1s ticks",
        action="store_true",
    )
    parser = add_argparse_args(parser)
    args = parser.parse_args()
    args.symbol = "BTCUSDT"  # dummy symbol
    config = await prepare_optimize_config(args)
    if args.passivbot_mode is not None:
        if args.passivbot_mode in ["s", "static_grid", "static"]:
            config["passivbot_mode"] = "static_grid"
        elif args.passivbot_mode in ["r", "recursive_grid", "recursive"]:
            config["passivbot_mode"] = "recursive_grid"
        elif args.passivbot_mode in ["n", "neat_grid", "neat"]:
            config["passivbot_mode"] = "neat_grid"
        elif args.passivbot_mode in ["e", "emas"]:
            config["passivbot_mode"] = "emas"
        else:
            raise Exception(f"unknown passivbot mode {args.passivbot_mode}")
    passivbot_mode = config["passivbot_mode"]
    assert passivbot_mode in [
        "recursive_grid",
        "static_grid",
        "neat_grid",
        "emas",
    ], f"unknown passivbot mode {passivbot_mode}"
    config["exchange"] = load_exchange_key_secret_passphrase(config["user"])[0]
    args = parser.parse_args()
    if args.long_enabled is None:
        do_long = config["do_long"]
    else:
        if "y" in args.long_enabled.lower():
            do_long = config["do_long"] = True
        elif "n" in args.long_enabled.lower():
            do_long = config["do_long"] = False
        else:
            raise Exception("please specify y/n with kwarg -le/--long")
    if args.short_enabled is None:
        do_short = config["do_short"]
    else:
        if "y" in args.short_enabled.lower():
            do_short = config["do_short"] = True
        elif "n" in args.short_enabled.lower():
            do_short = config["do_short"] = False
        else:
            raise Exception("please specify y/n with kwarg -le/--short")
    template_config = get_template_live_config(passivbot_mode)
    if passivbot_mode == "emas":
        template_config["do_long"] = do_long
        template_config["do_short"] = do_short
        config["long"] = template_config.copy()
        config["short"] = template_config.copy()
        bounds = config["bounds_emas"].copy()
        config["bounds_emas"] = {"long": bounds, "short": bounds}
    config.update(template_config)
    config["long"]["enabled"], config["short"]["enabled"] = do_long, do_short
    config["long"]["backwards_tp"] = config["backwards_tp_long"]
    config["short"]["backwards_tp"] = config["backwards_tp_short"]
    config["do_long"], config["do_short"] = do_long, do_short
    if args.symbol is not None:
        config["symbols"] = args.symbol.split(",")
    if args.n_cpus is not None:
        config["n_cpus"] = args.n_cpus
    if args.base_dir is not None:
        config["base_dir"] = args.base_dir
    config["ohlcv"] = args.ohlcv if config["passivbot_mode"] != "emas" else True
    print()
    lines = [(k, getattr(args, k)) for k in args.__dict__ if args.__dict__[k] is not None]
    lines += [
        (k, config[k])
        for k in [
            "starting_balance",
            "start_date",
            "end_date",
            "w",
            "c0",
            "c1",
            "maximum_pa_distance_std_long",
            "maximum_pa_distance_std_short",
            "maximum_pa_distance_mean_long",
            "maximum_pa_distance_mean_short",
            "maximum_loss_profit_ratio_long",
            "maximum_loss_profit_ratio_short",
            "minimum_eqbal_ratio_min_long",
            "minimum_eqbal_ratio_min_short",
            "maximum_hrs_stuck_max_long",
            "maximum_hrs_stuck_max_short",
            "clip_threshold",
        ]
        if k in config and k not in [z[0] for z in lines]
    ]
    for line in lines:
        logging.info(f"{line[0]: <{max([len(x[0]) for x in lines]) + 2}} {line[1]}")
    print()

    # download ticks .npy file if missing
    if config["ohlcv"]:
        cache_fname = f"{config['start_date']}_{config['end_date']}_ohlcv_cache.npy"
    else:
        cache_fname = f"{config['start_date']}_{config['end_date']}_ticks_cache.npy"
    exchange_name = config["exchange"] + ("_spot" if config["market_type"] == "spot" else "")
    config["symbols"] = sorted(config["symbols"])
    for symbol in config["symbols"]:
        cache_dirpath = os.path.join(config["base_dir"], exchange_name, symbol, "caches", "")
        if not os.path.exists(cache_dirpath + cache_fname) or not os.path.exists(
            cache_dirpath + "market_specific_settings.json"
        ):
            logging.info(f"fetching data {symbol}")
            args.symbol = symbol
            tmp_cfg = await prepare_backtest_config(args)
            if config["ohlcv"]:
                data = load_hlc_cache(
                    symbol,
                    config["start_date"],
                    config["end_date"],
                    base_dir=config["base_dir"],
                    spot=config["spot"],
                    exchange=config["exchange"],
                )
            else:
                downloader = Downloader({**config, **tmp_cfg})
                await downloader.get_sampled_ticks()

    # prepare starting configs
    cfgs = []
    if args.starting_configs is not None:
        logging.info("preparing starting configs...")
        if os.path.isdir(args.starting_configs):
            for fname in os.listdir(args.starting_configs):
                try:
                    cfg = load_live_config(os.path.join(args.starting_configs, fname))
                    assert determine_passivbot_mode(cfg) == passivbot_mode, "wrong passivbot mode"
                    cfgs.append(cfg)
                    logging.info(f"successfully loaded config {fname}")

                except Exception as e:
                    logging.error(f"error loading config {fname}: {e}")
        elif os.path.exists(args.starting_configs):
            try:
                cfg = load_live_config(args.starting_configs)
                assert determine_passivbot_mode(cfg) == passivbot_mode, "wrong passivbot mode"
                cfgs.append(cfg)
                logging.info(f"successfully loaded config {args.starting_configs}")
            except Exception as e:
                logging.error(f"error loading config {args.starting_configs}: {e}")
    if passivbot_mode == "emas":
        cfgs = [{"long": cfg.copy(), "short": cfg.copy()} for cfg in cfgs]
    config["starting_configs"] = cfgs
    return config
