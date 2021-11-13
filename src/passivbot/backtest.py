import argparse
import asyncio
import os
import pprint
import time

import numpy as np
import pandas as pd

from passivbot.downloader import Downloader
from passivbot.utils.funcs.njit import njit_backtest
from passivbot.utils.funcs.njit import round_
from passivbot.utils.funcs.pure import analyze_fills
from passivbot.utils.funcs.pure import create_xk
from passivbot.utils.funcs.pure import denumpyize
from passivbot.utils.funcs.pure import spotify_config
from passivbot.utils.funcs.pure import ts_to_date
from passivbot.utils.plotting import dump_plots
from passivbot.utils.procedures import add_backtesting_argparse_args
from passivbot.utils.procedures import load_live_config
from passivbot.utils.procedures import make_get_filepath
from passivbot.utils.procedures import prepare_backtest_config


def backtest(config: dict, data: np.ndarray, do_print=False) -> (list, bool):
    xk = create_xk(config)
    return njit_backtest(
        data, config["starting_balance"], config["latency_simulation_ms"], config["maker_fee"], **xk
    )


def plot_wrap(config, data):
    print("n_days", round_(config["n_days"], 0.1))
    print("starting_balance", config["starting_balance"])
    print("backtesting...")
    sts = time.time()
    fills, stats = backtest(config, data, do_print=True)
    print(f"{time.time() - sts:.2f} seconds elapsed")
    if not fills:
        print("no fills")
        return
    fdf, sdf, result = analyze_fills(fills, stats, config)
    config["result"] = result
    config["plots_dirpath"] = make_get_filepath(
        os.path.join(
            config["plots_dirpath"], f"{ts_to_date(time.time())[:19].replace(':', '')}", ""
        )
    )
    fdf.to_csv(config["plots_dirpath"] + "fills.csv")
    sdf.to_csv(config["plots_dirpath"] + "stats.csv")
    df = pd.DataFrame({**{"timestamp": data[:, 0], "qty": data[:, 1], "price": data[:, 2]}, **{}})
    print("dumping plots...")
    dump_plots(config, fdf, sdf, df)


async def _main(args: argparse.Namespace) -> None:
    config = await prepare_backtest_config(args)
    live_config = load_live_config(args.live_config_path)
    config.update(live_config)
    if "spot" in config["market_type"]:
        live_config = spotify_config(live_config)
    downloader = Downloader(config)
    print()
    for k in (
        keys := [
            "exchange",
            "spot",
            "symbol",
            "market_type",
            "config_type",
            "starting_balance",
            "start_date",
            "end_date",
            "latency_simulation_ms",
        ]
    ):
        if k in config:
            print(f"{k: <{max(map(len, keys)) + 2}} {config[k]}")
    print()
    data = await downloader.get_sampled_ticks()
    config["n_days"] = round_((data[-1][0] - data[0][0]) / (1000 * 60 * 60 * 24), 0.1)
    pprint.pprint(denumpyize(live_config))
    plot_wrap(config, data)


def main(args: argparse.Namespace) -> None:
    asyncio.run(_main(args))


def setup_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("backtest", help="Backtest given passivbot config.")
    parser.add_argument("live_config_path", type=str, help="path to live config to test")
    add_backtesting_argparse_args(parser)
    parser.set_defaults(func=main)
