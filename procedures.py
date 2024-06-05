import glob
import json
import os
import traceback
import asyncio
from datetime import datetime, timezone
from time import time
import numpy as np
import pprint


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

from njit_funcs import calc_samples, round_
from pure_funcs import (
    numpyize,
    candidate_to_live_config,
    ts_to_date,
    ts_to_date_utc,
    get_dummy_settings,
    config_pretty_str,
    date_to_ts2,
    get_template_live_config,
    sort_dict_keys,
    make_compatible,
    determine_passivbot_mode,
    date2ts_utc,
    remove_OD,
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
        return remove_OD(hjson.load(open(config_path, encoding="utf-8")))
    except Exception as e:
        raise Exception(f"failed to load config file {config_path} {e}")


def prepare_backtest_config(args) -> dict:
    """
    takes argparse args, returns dict with backtest config
    """
    config = load_hjson_config(args.backtest_config_path)

    if args.symbols is not None:
        config["symbols"] = args.symbols.split(",")
    for key in [
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
    config["start_date"] = ts_to_date_utc(date_to_ts2(config["start_date"]))[:10]
    if config["end_date"] in ["today", "now", ""]:
        config["end_date"] = ts_to_date_utc(utc_ms())[:10]
    else:
        config["end_date"] = ts_to_date_utc(date_to_ts2(config["end_date"]))[:10]
    config["exchange"] = load_exchange_key_secret_passphrase(config["user"])[0]
    config["session_name"] = (
        f"{config['start_date'].replace(' ', '').replace(':', '').replace('.', '')}_"
        f"{config['end_date'].replace(' ', '').replace(':', '').replace('.', '')}"
    )
    if config["exchange"] in ["okx", "kucoin"]:
        config["ohlcv"] = True
    elif hasattr(args, "ohlcv"):
        if args.ohlcv is None:
            if "ohlcv" not in config:
                config["ohlcv"] = True
        else:
            if args.ohlcv.lower() in ["y", "t", "yes", "true"]:
                config["ohlcv"] = True
            else:
                config["ohlcv"] = False
    elif "ohlcv" not in config:
        config["ohlcv"] = True

    if config["base_dir"].startswith("~"):
        raise Exception("error: using the ~ to indicate the user's home directory is not supported")
    if len(config["symbols"]) == 1:
        config["symbol"] = config["symbols"][0]
        base_dirpath = os.path.join(
            config["base_dir"],
            f"{config['exchange']}{'_spot' if 'spot' in config['market_type'] else ''}",
            config["symbol"],
        )
        config["caches_dirpath"] = make_get_filepath(os.path.join(base_dirpath, "caches", ""))
        config["plots_dirpath"] = make_get_filepath(os.path.join(base_dirpath, "plots", ""))
        add_market_specific_settings(config)
    return config


def prepare_optimize_config(args) -> dict:
    config = prepare_backtest_config(args)
    config.update(load_hjson_config(args.optimize_config_path))

    for key in ["starting_configs", "iters", "algorithm", "clip_threshold", "passivbot_mode"]:
        if hasattr(args, key) and getattr(args, key) is not None:
            config[key] = getattr(args, key)
        elif key not in config:
            config[key] = None

    algo_map = {
        "h": "harmony_search",
        "hs": "harmony_search",
        "harmony_search": "harmony_search",
        "harmony-search": "harmony_search",
        "p": "particle_swarm_optimization",
        "pso": "particle_swarm_optimization",
        "PSO": "particle_swarm_optimization",
        "particle_swarm_optimization": "particle_swarm_optimization",
        "particle-swarm-optimization": "particle_swarm_optimization",
    }
    assert config["algorithm"] in algo_map, f"unknown algorithm {config['algorithm']}"
    config["algorithm"] = algo_map[config["algorithm"]]

    pm_map = {
        "r": "recursive_grid",
        "recursive": "recursive_grid",
        "recursive_grid": "recursive_grid",
        "recursive-grid": "recursive_grid",
        "n": "neat_grid",
        "neat": "neat_grid",
        "neat_grid": "neat_grid",
        "neat-grid": "neat_grid",
        "c": "clock",
        "clock": "clock",
    }
    assert config["passivbot_mode"] in pm_map, f"unknown passivbot mode {config['passivbot_mode']}"
    config["passivbot_mode"] = pm_map[config["passivbot_mode"]]

    if args.optimize_output_path is None:
        output_base_dir = f"results_{config['algorithm']}_{config['passivbot_mode']}/"
    else:
        output_base_dir = args.optimize_output_path
    identifying_name = (
        f"{len(config['symbols'])}_symbols" if len(config["symbols"]) > 1 else config["symbols"][0]
    )
    now_date = ts_to_date(time())[:19].replace(":", "-")
    config["results_fpath"] = os.path.join(output_base_dir, f"{now_date}_{identifying_name}", "")

    for key in [
        "skip_multicoin",
        "skip_singlecoin",
        "skip_non_matching_single_coin",
        "skip_matching_single_coin",
    ]:
        if getattr(args, key) is not None:
            if getattr(args, key).lower() in ["y", "t", "yes", "true"]:
                config["starting_configs_filtering_conditions"].append(key)
            elif key in config["starting_configs_filtering_conditions"]:
                config["starting_configs_filtering_conditions"] = [
                    x for x in config["starting_configs_filtering_conditions"] if x != key
                ]

    return config


def add_market_specific_settings(config):
    mss = config["caches_dirpath"] + "market_specific_settings.json"
    symbol = config["symbol"]
    try:
        print(f"fetching market_specific_settings for {symbol}...")
        market_specific_settings = fetch_market_specific_settings(config)
        json.dump(market_specific_settings, open(mss, "w"), indent=4)
    except Exception as e:
        traceback.print_exc()
        print(f"\nfailed to fetch market_specific_settings for symbol {symbol}", e, "\n")
        try:
            if os.path.exists(mss):
                market_specific_settings = json.load(open(mss))
                print("using cached market_specific_settings")
            else:
                raise Exception(f"no cached market_specific_settings for symbol {symbol}")
        except:
            raise Exception(f"failed to load cached market_specific_settings for symbol {symbol}")
    config.update(market_specific_settings)


def make_get_filepath(filepath: str) -> str:
    """
    if not is path, creates dir and subdirs for path, returns path
    """
    dirpath = os.path.dirname(filepath) if filepath[-1] != "/" else filepath
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    return filepath


def load_user_info(user: str, api_keys_path="api-keys.json") -> dict:
    if api_keys_path is None:
        api_keys_path = "api-keys.json"
    try:
        api_keys = json.load(open(api_keys_path))
    except Exception as e:
        raise Exception(f"error loading api keys file {api_keys_path} {e}")
    if user not in api_keys:
        raise Exception(f"user {user} not found in {api_keys_path}")
    return {
        k: api_keys[user][k] if k in api_keys[user] else ""
        for k in [
            "exchange",
            "key",
            "secret",
            "passphrase",
            "wallet_address",
            "private_key",
            "is_vault",
        ]
    }


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


def load_broker_code(exchange: str) -> str:
    try:
        return hjson.load(open("broker_codes.hjson"))[exchange]
    except Exception as e:
        print(f"failed to load broker code", e)
        traceback.print_exc()
        return ""


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


async def fetch_market_specific_settings_old(config: dict):
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
    elif exchange == "bingx":
        if "spot" in config["market_type"]:
            raise Exception("spot not implemented on bingx")
        bot = await create_bingx_bot(tmp_live_settings)
        settings_from_exchange["maker_fee"] = 0.0002
        settings_from_exchange["taker_fee"] = 0.0005
        settings_from_exchange["exchange"] = "bingx"
    else:
        raise Exception(f"unknown exchange {exchange}")
    if hasattr(bot, "session"):
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
    from exchanges.binance import BinanceBot

    bot = BinanceBot(config)
    await bot._init()
    return bot


async def create_binance_bot_spot(config: dict):
    from exchanges.binance_spot import BinanceBotSpot

    bot = BinanceBotSpot(config)
    await bot._init()
    return bot


async def create_bybit_bot(config: dict):
    from exchanges.bybit import BybitBot

    bot = BybitBot(config)
    await bot._init()
    return bot


async def create_bybit_bot_spot(config: dict):
    from exchanges.bybit_spot import BybitBotSpot

    bot = BybitBotSpot(config)
    await bot._init()
    return bot


async def create_bitget_bot(config: dict):
    from exchanges.bitget import BitgetBot

    bot = BitgetBot(config)
    await bot._init()
    return bot


async def create_okx_bot(config: dict):
    from exchanges.okx import OKXBot

    bot = OKXBot(config)
    await bot._init()
    return bot


async def create_kucoin_bot(config: dict):
    from exchanges.kucoin import KuCoinBot

    bot = KuCoinBot(config)
    await bot._init()
    return bot


async def create_bingx_bot(config: dict):
    from exchanges.bingx import BingXBot

    bot = BingXBot(config)
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
        "--symbols",
        type=str,
        required=False,
        dest="symbols",
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
    parser.add_argument(
        "-oh",
        "--ohlcv",
        type=str,
        required=False,
        dest="ohlcv",
        default=None,
        nargs="?",
        const="y",
        help="if no arg or [y/yes], use 1m ohlcv instead of 1s ticks, overriding param ohlcv from config/backtest/default.hjson",
    )
    return parser


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


def get_file_mod_utc(filepath):
    """
    Get the UTC timestamp of the last modification of a file.

    Args:
        filepath (str): The path to the file.

    Returns:
        float: The UTC timestamp in milliseconds of the last modification of the file.
    """
    # Get the last modification time of the file in seconds since the epoch
    mod_time_epoch = os.path.getmtime(filepath)

    # Convert the timestamp to a UTC datetime object
    mod_time_utc = datetime.utcfromtimestamp(mod_time_epoch)

    # Return the UTC timestamp
    return mod_time_utc.timestamp() * 1000


def print_async_exception(coro):
    if isinstance(coro, list):
        for elm in coro:
            print_async_exception(elm)
    try:
        print(f"result: {coro.result()}")
    except:
        pass
    try:
        print(f"exception: {coro.exception()}")
    except:
        pass
    try:
        print(f"returned: {coro}")
    except:
        pass


async def get_first_ohlcv_timestamps(cc=None, symbols=None, cache=True):
    if cc is None:
        import ccxt.async_support as ccxt

        cc = ccxt.binanceusdm()
    else:
        supported_exchanges = ["binanceusdm", "bybit", "bitget", "okx", "bingx", "hyperliquid"]
        if cc.id not in supported_exchanges:
            print(f"get_first_ohlcv_timestamps() currently only supports {supported_exchanges}")
            return {}
    try:
        if symbols is None:
            markets = await cc.load_markets()
            symbols = sorted([x for x in markets if markets[x]["swap"] and markets[x]["active"]])
        n = 30
        first_timestamps = {}
        cache_fname = f"caches/first_ohlcv_timestamps_{cc.id}.json"
        if cache:
            if os.path.exists(cache_fname):
                try:
                    first_timestamps = json.load(open(cache_fname))
                    symbols = [s for s in symbols if s not in first_timestamps]
                except Exception as e:
                    print(f"error loading ohlcv first ts cache", e)
        fetched = []
        for i, symbol in enumerate(symbols):
            if cc.id in ["bybit", "binanceusdm"]:
                fetched.append(
                    (
                        symbol,
                        asyncio.ensure_future(
                            cc.fetch_ohlcv(
                                symbol, timeframe="1d", since=int(date2ts_utc("2015-01-01"))
                            )
                        ),
                    )
                )
            else:
                if cc.id in ["hyperliquid"]:
                    timeframe_ = "1w"
                else:
                    timeframe_ = "1M"
                fetched.append(
                    (symbol, asyncio.ensure_future(cc.fetch_ohlcv(symbol, timeframe=timeframe_)))
                )
            if i + 1 == len(symbols) or (i + 1) % n == 0:
                for sym, task in fetched:
                    try:
                        res = await task
                        first_timestamps[sym] = res[0][0]
                    except Exception as e:
                        print(f"error fetching ohlcvs for {sym} {e}")
                        if "The symbol has been removed" in str(e):
                            first_timestamps[sym] = 0
                if cache:
                    try:
                        make_get_filepath(cache_fname)
                        json.dump(first_timestamps, open(cache_fname, "w"), indent=4, sort_keys=True)
                        print(
                            f"dumped first ohlcv timestamp cache for {cc.id} {[x[0] for x in fetched]}"
                        )
                    except Exception as e:
                        print(f"error dumping ohlcv first timestamps cache", e)
                fetched = []
                await asyncio.sleep(1)
    finally:
        await cc.close()
    return first_timestamps


def assert_correct_ccxt_version(version=None, ccxt=None):
    if version is None:
        version = load_ccxt_version()
    if ccxt is None:
        import ccxt

    assert (
        ccxt.__version__ == version
    ), f"Currently ccxt {ccxt.__version__} is installed. Please pip reinstall requirements.txt or install ccxt v{version} manually"


def load_ccxt_version():
    try:
        with open("requirements_liveonly.txt") as f:
            lines = f.readlines()
        ccxt_line = [line for line in lines if "ccxt" in line][0].strip()
        return ccxt_line[ccxt_line.find("==") + 2 :]
    except Exception as e:
        print(f"failed to load ccxt version {e}")
        return None


def fetch_market_specific_settings_multi(symbols=None, exchange="binance"):
    import ccxt

    assert_correct_ccxt_version(ccxt=ccxt)

    exchange_map = {
        # "kucoin": "kucoinfutures",
        # "okx": "okx",
        "bybit": "bybit",
        "binance": "binanceusdm",
        # "bitget": "bitget",
        # "bingx": "bingx",
    }
    cc = getattr(ccxt, exchange_map[exchange])()
    cc.options["defaultType"] = "swap"
    info = cc.load_markets()
    for symbol in info:
        if exchange == "binance":
            for felm in info[symbol]["info"]["filters"]:
                if felm["filterType"] == "PRICE_FILTER":
                    info[symbol]["price_step"] = float(felm["tickSize"])
                elif felm["filterType"] == "MARKET_LOT_SIZE":
                    info[symbol]["qty_step"] = float(felm["stepSize"])
            info[symbol]["c_mult"] = info[symbol]["contractSize"]
            info[symbol]["min_cost"] = info[symbol]["limits"]["cost"]["min"]
            info[symbol]["min_qty"] = info[symbol]["limits"]["amount"]["min"]
        elif exchange == "bybit":
            info[symbol]["price_step"] = info[symbol]["precision"]["price"]
            info[symbol]["qty_step"] = info[symbol]["precision"]["amount"]
            info[symbol]["c_mult"] = info[symbol]["contractSize"]
            info[symbol]["min_cost"] = 0.0
            info[symbol]["min_qty"] = info[symbol]["limits"]["amount"]["min"]
            # ccxt reports incorrect fees for bybit perps
            info[symbol]["maker"] = info[symbol]["maker_fee"] = 0.0002
            info[symbol]["taker"] = info[symbol]["taker_fee"] = 0.00055
    for symbol in sorted(info):
        info[info[symbol]["id"]] = info[symbol]
    return info if symbols is None else {symbol: info[symbol] for symbol in symbols}


def fetch_market_specific_settings(config: dict):
    import ccxt

    assert_correct_ccxt_version(ccxt=ccxt)
    exchange = config["exchange"]
    symbol = config["symbol"]
    market_type = config["market_type"]

    settings_from_exchange = {"exchange": exchange}
    if exchange == "binance":
        if "futures" in market_type:
            if symbol.endswith("USDT") or symbol.endswith("BUSD"):
                cc = ccxt.binanceusdm()
                settings_from_exchange["inverse"] = False

            elif symbol.endswith("PERP"):
                cc = ccxt.binancecoinm()
                settings_from_exchange["inverse"] = True
            else:
                raise Exception(f"unknown symbol {symbol}")
            settings_from_exchange["hedge_mode"] = True
            settings_from_exchange["spot"] = False

        elif "spot" in market_type:
            cc = ccxt.binance()
            settings_from_exchange["spot"] = True
            settings_from_exchange["inverse"] = False
            settings_from_exchange["hedge_mode"] = False
        else:
            raise Exception(f"unknown market type {market_type}")
        markets = cc.fetch_markets()
        for elm in markets:
            if elm["id"] == symbol:
                break
        else:
            raise Exception(f"unknown symbol {symbol}")
        settings_from_exchange["maker_fee"] = elm["maker"]
        settings_from_exchange["taker_fee"] = elm["taker"]
        settings_from_exchange["c_mult"] = 1.0 if elm["contractSize"] is None else elm["contractSize"]
        settings_from_exchange["min_qty"] = elm["limits"]["amount"]["min"]
        for elm1 in elm["info"]["filters"]:
            if elm1["filterType"] == "LOT_SIZE":
                settings_from_exchange["qty_step"] = float(elm1["stepSize"])
            if elm1["filterType"] == "PRICE_FILTER":
                settings_from_exchange["price_step"] = float(elm1["tickSize"])
    elif exchange == "bitget":
        cc = ccxt.bitget()
        cc.options["defaultType"] = "swap"
        markets = cc.fetch_markets()
        for elm in markets:
            if elm["id"] == symbol and elm["swap"]:
                break
        else:
            raise Exception(f"unknown symbol {symbol}")
        settings_from_exchange["hedge_mode"] = True
        settings_from_exchange["maker_fee"] = elm["maker"]
        settings_from_exchange["taker_fee"] = elm["taker"]
        settings_from_exchange["c_mult"] = 1.0
        settings_from_exchange["price_step"] = elm["precision"]["price"]
        settings_from_exchange["qty_step"] = elm["precision"]["amount"]
        settings_from_exchange["min_qty"] = max(
            elm["limits"]["amount"]["min"], elm["precision"]["amount"]
        )
        settings_from_exchange["min_cost"] = elm["limits"]["cost"]["min"]
        settings_from_exchange["spot"] = elm["spot"]
        settings_from_exchange["inverse"] = elm["linear"] is not None and not elm["linear"]
    elif exchange == "okx":
        cc = ccxt.okx()
        markets = cc.fetch_markets()
        for elm in markets:
            if elm["type"] == "swap" and symbol in elm["id"].replace("-", ""):
                break
        else:
            raise Exception(f"unknown symbol {symbol}")
        settings_from_exchange["hedge_mode"] = True
        settings_from_exchange["maker_fee"] = elm["maker"]
        settings_from_exchange["taker_fee"] = elm["taker"]
        settings_from_exchange["c_mult"] = elm["contractSize"]
        settings_from_exchange["qty_step"] = elm["precision"]["amount"]
        settings_from_exchange["price_step"] = elm["precision"]["price"]
        settings_from_exchange["spot"] = elm["spot"]
        settings_from_exchange["inverse"] = elm["linear"] is not None and not elm["linear"]
        settings_from_exchange["min_qty"] = elm["limits"]["amount"]["min"]
    elif exchange == "bybit":
        cc = ccxt.bybit()
        markets = cc.fetch_markets()
        spot = market_type == "spot"
        for elm in markets:
            if elm["id"] == symbol and elm["spot"] == spot:
                break
        else:
            raise Exception(f"unknown symbol {symbol}")
        settings_from_exchange["hedge_mode"] = not spot
        # ccxt reports incorrect fees for bybit perps
        settings_from_exchange["maker_fee"] = 0.0002 if not spot else elm["maker"]
        settings_from_exchange["taker_fee"] = 0.00055 if not spot else elm["taker"]
        settings_from_exchange["c_mult"] = 1.0 if elm["contractSize"] is None else elm["contractSize"]
        settings_from_exchange["qty_step"] = elm["precision"]["amount"]
        settings_from_exchange["price_step"] = elm["precision"]["price"]
        settings_from_exchange["spot"] = spot
        settings_from_exchange["inverse"] = elm["linear"] is not None and not elm["linear"]
        settings_from_exchange["min_qty"] = elm["limits"]["amount"]["min"]
    elif exchange == "kucoin":
        cc = ccxt.kucoinfutures()
        markets = cc.fetch_markets()
        for elm in markets:
            if elm["id"] == symbol + "M":
                break
        else:
            raise Exception(f"unknown symbol {symbol}")
        settings_from_exchange["hedge_mode"] = True
        settings_from_exchange["maker_fee"] = elm["maker"]
        settings_from_exchange["taker_fee"] = elm["taker"]
        settings_from_exchange["c_mult"] = elm["contractSize"]
        settings_from_exchange["qty_step"] = elm["precision"]["amount"]
        settings_from_exchange["price_step"] = elm["precision"]["price"]
        settings_from_exchange["spot"] = False
        settings_from_exchange["inverse"] = elm["linear"] is not None and not elm["linear"]
        settings_from_exchange["min_qty"] = (
            0.0 if elm["limits"]["amount"]["min"] is None else elm["limits"]["amount"]["min"]
        )
        settings_from_exchange["min_qty"] = float(elm["info"]["lotSize"])
    else:
        raise Exception(f"unknown exchange {exchange}")
    if "min_cost" not in settings_from_exchange:
        settings_from_exchange["min_cost"] = (
            0.0 if elm["limits"]["cost"]["min"] is None else elm["limits"]["cost"]["min"]
        )
    for key in [
        "c_mult",
        "exchange",
        "hedge_mode",
        "inverse",
        "maker_fee",
        "min_cost",
        "min_qty",
        "price_step",
        "qty_step",
        "spot",
        "taker_fee",
    ]:
        assert key in settings_from_exchange, f"missing {key}"
    # import pprint
    # pprint.pprint(elm)
    return sort_dict_keys(settings_from_exchange)


def main():
    mssm = fetch_market_specific_settings_multi(exchange="bybit")
    # pprint.pprint(mssm)
    return
    """
    cfg = {"exchange": "bybit", "symbol": "DOGEUSDT", "market_type": "spot"}
    mss = fetch_market_specific_settings(cfg)
    pprint.pprint(mss)
    return
    """
    # for exchange in ["bitget"]:
    for exchange in ["kucoin", "bitget", "binance", "bybit", "okx", "bingx"]:
        cfg = {"exchange": exchange, "symbol": "ETHUSDT", "market_type": "futures"}
        try:
            mss = fetch_market_specific_settings(cfg)
            print(mss)
        except:
            traceback.print_exc()


if __name__ == "__main__":
    main()
