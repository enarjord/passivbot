import argparse
import asyncio
import os
import pprint
import time

import numpy as np
import pandas as pd

from passivbot.downloader import Downloader
from passivbot.njit_funcs import njit_backtest
from passivbot.njit_funcs import round_
from passivbot.plotting import dump_plots
from passivbot.procedures import add_argparse_args
from passivbot.procedures import load_live_config
from passivbot.procedures import make_get_filepath
from passivbot.procedures import prepare_backtest_config
from passivbot.pure_funcs import analyze_fills
from passivbot.pure_funcs import create_xk
from passivbot.pure_funcs import denumpyize
from passivbot.pure_funcs import spotify_config
from passivbot.pure_funcs import ts_to_date


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


async def main():
    parser = argparse.ArgumentParser(
        prog="Backtest", description="Backtest given passivbot config."
    )
    parser.add_argument("live_config_path", type=str, help="path to live config to test")
    parser = add_argparse_args(parser)
    args = parser.parse_args()

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


if __name__ == "__main__":
    asyncio.run(main())
