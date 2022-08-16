import os

# os.environ["NOJIT"] = "false"

import argparse
import asyncio
import pprint
from time import time

import numpy as np
import pandas as pd

from downloader import Downloader, load_hlc_cache
from njit_funcs import backtest_static_grid, round_
from njit_funcs_recursive_grid import backtest_recursive_grid
from njit_funcs_neat_grid import backtest_neat_grid
from plotting import dump_plots
from procedures import (
    prepare_backtest_config,
    make_get_filepath,
    load_live_config,
    load_hjson_config,
    add_argparse_args,
)
from pure_funcs import (
    create_xk,
    denumpyize,
    ts_to_date,
    analyze_fills,
    spotify_config,
    determine_passivbot_mode,
)


def backtest(config: dict, data: np.ndarray, do_print=False) -> (list, bool):
    passivbot_mode = determine_passivbot_mode(config)
    xk = create_xk(config)
    if passivbot_mode == "recursive_grid":
        return backtest_recursive_grid(
            data,
            config["starting_balance"],
            config["latency_simulation_ms"],
            config["maker_fee"],
            **xk,
        )
    elif passivbot_mode == "neat_grid":
        return backtest_neat_grid(
            data,
            config["starting_balance"],
            config["latency_simulation_ms"],
            config["maker_fee"],
            **xk,
        )
    return backtest_static_grid(
        data,
        config["starting_balance"],
        config["latency_simulation_ms"],
        config["maker_fee"],
        **xk,
    )


def plot_wrap(config, data):
    print("n_days", round_(config["n_days"], 0.1))
    print("starting_balance", config["starting_balance"])
    print("backtesting...")
    sts = time()
    fills_long, fills_short, stats = backtest(config, data, do_print=True)
    print(f"{time() - sts:.2f} seconds elapsed")
    if not fills_long and not fills_short:
        print("no fills")
        return
    longs, shorts, sdf, result = analyze_fills(fills_long, fills_short, stats, config)
    config["result"] = result
    config["plots_dirpath"] = make_get_filepath(
        os.path.join(config["plots_dirpath"], f"{ts_to_date(time())[:19].replace(':', '')}", "")
    )
    longs.to_csv(config["plots_dirpath"] + "fills_long.csv")
    shorts.to_csv(config["plots_dirpath"] + "fills_short.csv")
    sdf.to_csv(config["plots_dirpath"] + "stats.csv")
    df = pd.DataFrame({**{"timestamp": data[:, 0], "qty": data[:, 1], "price": data[:, 2]}, **{}})
    print("dumping plots...")
    dump_plots(config, longs, shorts, sdf, df, n_parts=config["n_parts"])
    if config["enable_interactive_plot"]:
        import interactive_plot

        print("dumping interactive plot...")
        sts = time()
        interactive_plot.dump_interactive_plot(config, data, longs, shorts)
        print(f"{time() - sts:.2f} seconds spent on dumping interactive plot")


async def main():
    parser = argparse.ArgumentParser(prog="Backtest", description="Backtest given passivbot config.")
    parser.add_argument("live_config_path", type=str, help="path to live config to test")
    parser = add_argparse_args(parser)
    parser.add_argument(
        "-lw",
        "--long_wallet_exposure_limit",
        "--long-wallet-exposure-limit",
        type=float,
        required=False,
        dest="long_wallet_exposure_limit",
        default=None,
        help="specify long wallet exposure limit, overriding value from live config",
    )
    parser.add_argument(
        "-sw",
        "--short_wallet_exposure_limit",
        "--short-wallet-exposure-limit",
        type=float,
        required=False,
        dest="short_wallet_exposure_limit",
        default=None,
        help="specify short wallet exposure limit, overriding value from live config",
    )
    parser.add_argument(
        "-le",
        "--long_enabled",
        "--long-enabled",
        type=str,
        required=False,
        dest="long_enabled",
        default=None,
        help="specify long enabled [y/n], overriding value from live config",
    )
    parser.add_argument(
        "-se",
        "--short_enabled",
        "--short-enabled",
        type=str,
        required=False,
        dest="short_enabled",
        default=None,
        help="specify short enabled [y/n], overriding value from live config",
    )
    parser.add_argument(
        "-np",
        "--n_parts",
        "--n-parts",
        type=int,
        required=False,
        dest="n_parts",
        default=None,
        help="set n backtest slices to plot",
    )
    parser.add_argument(
        "-oh",
        "--ohlcv",
        help="use 1m ohlcv instead of 1s ticks",
        action="store_true",
    )
    args = parser.parse_args()
    if args.symbol is None:
        tmp_cfg = load_hjson_config(args.backtest_config_path)
        symbols = (
            tmp_cfg["symbol"] if type(tmp_cfg["symbol"]) == list else tmp_cfg["symbol"].split(",")
        )
    else:
        symbols = args.symbol.split(",")
    for symbol in symbols:
        args = parser.parse_args()
        args.symbol = symbol
        config = await prepare_backtest_config(args)
        config["n_parts"] = args.n_parts
        live_config = load_live_config(args.live_config_path)
        config.update(live_config)

        if args.long_wallet_exposure_limit is not None:
            print(
                f"overriding long wallet exposure limit ({config['long']['wallet_exposure_limit']}) "
                f"with new value: {args.long_wallet_exposure_limit}"
            )
            config["long"]["wallet_exposure_limit"] = args.long_wallet_exposure_limit
        if args.short_wallet_exposure_limit is not None:
            print(
                f"overriding short wallet exposure limit ({config['short']['wallet_exposure_limit']}) "
                f"with new value: {args.short_wallet_exposure_limit}"
            )
            config["short"]["wallet_exposure_limit"] = args.short_wallet_exposure_limit
        if args.long_enabled is not None:
            config["long"]["enabled"] = "y" in args.long_enabled.lower()
        if args.short_enabled is not None:
            config["short"]["enabled"] = "y" in args.short_enabled.lower()
        if "spot" in config["market_type"]:
            live_config = spotify_config(live_config)
        config["ohlcv"] = args.ohlcv

        print()
        for k in (
            keys := [
                "exchange",
                "spot",
                "symbol",
                "market_type",
                "passivbot_mode",
                "config_type",
                "starting_balance",
                "start_date",
                "end_date",
                "latency_simulation_ms",
                "base_dir",
            ]
        ):
            if k in config:
                print(f"{k: <{max(map(len, keys)) + 2}} {config[k]}")
        print()
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
            downloader = Downloader(config)
            data = await downloader.get_sampled_ticks()
        config["n_days"] = round_((data[-1][0] - data[0][0]) / (1000 * 60 * 60 * 24), 0.1)
        pprint.pprint(denumpyize(live_config))
        plot_wrap(config, data)


if __name__ == "__main__":
    asyncio.run(main())
