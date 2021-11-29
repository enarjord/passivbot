from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import logging
import pathlib
import time
from typing import Any
from typing import TYPE_CHECKING

import dateutil.parser
import hjson
import numpy as np
import pandas as pd

from passivbot.datastructures.config import BaseBacktestConfig
from passivbot.datastructures.config import NamedConfig
from passivbot.utils.funcs.njit import calc_samples
from passivbot.utils.funcs.pure import candidate_to_live_config
from passivbot.utils.funcs.pure import date_to_ts
from passivbot.utils.funcs.pure import get_dummy_settings
from passivbot.utils.funcs.pure import get_template_live_config
from passivbot.utils.funcs.pure import numpyize
from passivbot.utils.funcs.pure import ts_to_date

if TYPE_CHECKING:
    from passivbot.exchanges.bybit import Bybit
    from passivbot.exchanges.binance import BinanceBot
    from passivbot.exchanges.binance_spot import BinanceBotSpot

log = logging.getLogger(__name__)


def load_live_config(live_config_path: str) -> dict[str, Any]:
    try:
        live_config = json.load(open(live_config_path))
        for orig, rpl in [
            ("secondary_grid_spacing", "secondary_pprice_diff"),
            ("secondary_pbr_allocation", "secondary_allocation"),
            ("pbr_limit", "wallet_exposure_limit"),
            ("shrt", "short"),
        ]:
            live_config = json.loads(json.dumps(live_config).replace(orig, rpl))
        assert all(k in live_config["long"] for k in get_template_live_config()["long"])
        return numpyize(live_config)
    except Exception as e:
        raise Exception(f"failed to load live config {live_config_path} {e}")


def dump_live_config(config: dict[str, Any], path: pathlib.Path) -> None:
    candidate_config = candidate_to_live_config(config)
    name = candidate_config.pop("config_name")
    live_config = NamedConfig.parse_obj(candidate_config)
    write_config = {
        "configs": {
            name: live_config.dict(
                exclude={
                    "stop_mode": ...,
                    "max_leverage": ...,
                    "assigned_balance": ...,
                }
            )
        }
    }
    path.write_text(json.dumps(write_config, indent=2))


def load_config_files(config_paths: list[str] | None = None) -> dict[str, Any] | None:
    config: dict[str, Any] = {}
    if config_paths is None:
        return config
    for config_path in config_paths:
        try:
            loaded_config = hjson.load(open(config_path, encoding="utf-8"))
            config = {**config, **loaded_config}
        except Exception as e:
            raise Exception("failed to load config file", config_path, e)
    return config


def load_hjson_config(config_path: str) -> dict[str, Any]:
    try:
        return hjson.load(open(config_path, encoding="utf-8"))
    except Exception as e:
        raise Exception(f"failed to load config file {config_path} {e}")


async def prepare_backtest_config(args: argparse.Namespace) -> dict[str, Any]:
    """
    takes argparse args, returns dict with backtest and optimize config
    """
    config = load_hjson_config(args.backtest_config_path)
    config["api_keys_path"] = args.api_keys
    for key in [
        "symbol",
        "user",
        "start_date",
        "end_date",
        "starting_balance",
        "market_type",
        "backtests_dir",
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


async def prepare_optimize_config(args) -> dict[str, Any]:
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
        log.info("fetching market_specific_settings...")
        market_specific_settings = await fetch_market_specific_settings(config)
        json.dump(market_specific_settings, open(mss, "w"), indent=4)
    except Exception as e:
        log.error("failed to fetch market_specific_settings: %s", e)
        try:
            if mss.exists():
                market_specific_settings = json.load(open(mss))
            log.info("using cached market_specific_settings")
        except Exception:
            raise Exception("failed to load cached market_specific_settings")
    config.update(market_specific_settings)


def load_exchange_key_secret(user: str) -> tuple[str, str, str]:
    try:
        keyfile = json.load(open("api-keys.json"))
        if user in keyfile:
            return keyfile[user]["exchange"], keyfile[user]["key"], keyfile[user]["secret"]
        else:
            log.info(
                "Looks like the keys aren't configured yet, or you entered the wrong username!"
            )
        raise Exception("API KeyFile Missing!")
    except FileNotFoundError:
        log.info("File Not Found!")
        raise Exception("API KeyFile Missing!")


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


async def create_binance_bot(config: NamedConfig) -> "BinanceBot":
    # Deferred import due to circular import issues
    from passivbot.exchanges.binance import BinanceBot

    bot = BinanceBot(config)
    await bot._init()
    return bot


async def create_binance_bot_spot(config: NamedConfig) -> "BinanceBotSpot":
    # Deferred import due to circular import issues
    from passivbot.exchanges.binance_spot import BinanceBotSpot

    bot = BinanceBotSpot(config)
    await bot._init()
    return bot


async def create_bybit_bot(config: NamedConfig) -> "Bybit":
    # Deferred import due to circular import issues
    from passivbot.exchanges.bybit import Bybit

    bot = Bybit(config)
    await bot._init()
    return bot


def add_backtesting_argparse_args(parser):
    parser.add_argument(
        "--dt",
        "--data-dir",
        type=pathlib.Path,
        default=None,
        dest="data_dir",
        help="Path to where the downloaded historical data should be stored",
    )
    parser.add_argument(
        "--bd",
        "--backtests-dir",
        type=pathlib.Path,
        default=None,
        dest="backtests_dir",
        help="Path to where the backtests data should be stored",
    )
    parser.add_argument(
        "-d",
        "--download-only",
        default=False,
        help="download only, do not dump ticks caches",
        action="store_true",
    )
    parser.add_argument(
        "--sd",
        "--start_date",
        "--start-date",
        type=str,
        required=True,
        dest="start_date",
        default=None,
        help="Specify start date",
    )
    parser.add_argument(
        "--ed",
        "--end_date",
        "--end-date",
        type=str,
        required=True,
        dest="end_date",
        default=None,
        help="Specify end date",
    )
    return parser


def post_process_backtesting_argparse_parsed_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace, config: BaseBacktestConfig
) -> None:
    if not args.data_dir:
        args.data_dir = args.basedir / "historical_data"

    if not args.backtests_dir:
        args.backtests_dir = args.basedir / "backtests"

    config.parent.data_dir = args.data_dir
    config.parent.backtests_dir = args.backtests_dir

    if args.start_date:
        config.parent.start_date = dateutil.parser.parse(args.start_date).replace(
            tzinfo=datetime.timezone.utc
        )
    if args.end_date:
        config.parent.end_date = dateutil.parser.parse(args.end_date).replace(
            tzinfo=datetime.timezone.utc
        )

    if config.parent.start_date is None:
        parser.exit(status=1, message="No start date was passed")
    if config.parent.end_date is None:
        parser.exit(status=1, message="No end date was passed")


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
        log.info(f"loading chunk {ts_to_date(first_ts / 1000)}", wipe_line=True)
        tdf = pd.read_csv(ticks_filepath / f)
        tdf = tdf[(tdf.timestamp >= start_ts) & (tdf.timestamp <= end_ts)]
        ticks = np.concatenate((ticks, tdf[["timestamp", "qty", "price"]].values))
        del tdf
    samples = calc_samples(ticks[ticks[:, 0].argsort()], sec_span * 1000)
    log.info(
        f"took {time.time() - sts:.2f} seconds to load {len(ticks)} ticks, creating {len(samples)} samples"
    )
    del ticks
    return samples


def get_starting_configs(config: dict[str, Any]) -> list[dict[str, Any]]:
    starting_configs = []
    if config["starting_configs"]:
        for path in config["starting_configs"]:
            try:
                if path.isdir():
                    log.info("Starting with all configurations in directory.")
                    for fpath in path.glob("*.json"):
                        starting_configs.append(json.loads(fpath.read_text()))
                else:
                    log.info("Starting with specified configuration.")
                    starting_configs.append(json.loads(path.read_text()))
            except Exception as e:
                log.error("Could not find specified configuration: %s", e)
    return starting_configs


def utc_ms() -> float:
    return datetime.datetime.utcnow().timestamp() * 1000


def local_time() -> float:
    return datetime.datetime.now().astimezone().timestamp() * 1000


def log_async_exception(coro):
    if coro is None:
        return
    exception = result = None
    with contextlib.suppress(Exception):
        exception = coro.exception()
    with contextlib.suppress(Exception):
        result = coro.result()
    log.info("Coro: %s; Result: %s: Exception: %s", coro, result, exception)
