import asyncio
import hjson
import json
import pprint
import os
import argparse
import logging
import traceback
import numpy as np
import pandas as pd
from downloader import prepare_multsymbol_data
from procedures import (
    load_live_config,
    utc_ms,
    load_user_info,
    make_get_filepath,
    fetch_market_specific_settings_multi,
)
from pure_funcs import (
    ts_to_date_utc,
    tuplify,
    numpyize,
    live_config_dict_to_list_recursive_grid,
    fills_multi_to_df,
    stats_multi_to_df,
    analyze_fills_multi,
    calc_drawdowns,
    str2bool,
)
from plotting import plot_pnls_stuck, plot_pnls_separate, plot_pnls_long_short, plot_fills_multi
from collections import OrderedDict
from njit_multisymbol import backtest_multisymbol_recursive_grid
from njit_funcs import round_dynamic
import matplotlib.pyplot as plt


def oj(*x):
    return os.path.join(*x)


def backtest_multi(hlcs, config):
    res = backtest_multisymbol_recursive_grid(
        hlcs,
        config["starting_balance"],
        config["maker_fee"],
        config["do_longs"],
        config["do_shorts"],
        config["c_mults"],
        config["symbols"],
        config["qty_steps"],
        config["price_steps"],
        config["min_costs"],
        config["min_qtys"],
        config["live_configs"],
        config["loss_allowance_pct"],
        config["stuck_threshold"],
        config["unstuck_close_pct"],
    )
    return res


def prep_config_multi(parser):
    parser.add_argument(
        "-s",
        "--symbols",
        type=str,
        required=False,
        dest="symbols",
        default=None,
        help="specify symbols, comma separated, overriding symbols from backtest config.  ",
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
        help="specify start date, overriding value from hjson config",
    )
    parser.add_argument(
        "-ed",
        "--end_date",
        type=str,
        required=False,
        dest="end_date",
        default=None,
        help="specify end date, overriding value from hjson config",
    )
    parser.add_argument(
        "-sb",
        "--starting_balance",
        "--starting-balance",
        type=float,
        required=False,
        dest="starting_balance",
        default=None,
        help="specify starting_balance, overriding value from hjson config",
    )
    parser.add_argument(
        "-le",
        "--long_enabled",
        "--long-enabled",
        type=str2bool,
        required=False,
        dest="long_enabled",
        default=None,
        help="specify long_enabled (y/n or t/f), overriding value from hjson config",
    )
    parser.add_argument(
        "-se",
        "--short_enabled",
        "--short-enabled",
        type=str2bool,
        required=False,
        dest="short_enabled",
        default=None,
        help="specify short_enabled (y/n or t/f), overriding value from hjson config",
    )
    args = parser.parse_args()
    config = OrderedDict()

    for key, value in vars(args).items():
        print("debug", key, value)
        if "config_path" in key:
            logging.info(f"loading {value}")
            config = hjson.load(open(value))
        elif getattr(args, key) is not None:
            if key == "symbols":
                new_symbols = {s: "" for s in getattr(args, key).split(",")}
                if new_symbols != config["symbols"]:
                    logging.info(f"new symbols: {new_symbols}")
                    config["symbols"] = new_symbols
            else:
                if key in config and config[key] != getattr(args, key):
                    logging.info(f"changing {key}: {config[key]} -> {getattr(args, key)}")
                    config[key] = getattr(args, key)
    return config


async def main():

    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    parser = argparse.ArgumentParser(prog="backtest_multi", description="run multisym backtest")
    parser.add_argument(
        "-bc",
        "--backtest_config",
        type=str,
        required=False,
        dest="backtest_config_path",
        default="configs/backtest/multi.hjson",
        help="backtest config hjson file",
    )
    parser.add_argument(
        "-tl",
        "--total_wallet_exposure_long",
        "--total-wallet-exposure-long",
        type=float,
        required=False,
        dest="TWE_long",
        default=None,
        help="specify total_wallet_exposure_long, overriding value from hjson config",
    )
    parser.add_argument(
        "-ts",
        "--total_wallet_exposure_short",
        "--total-wallet-exposure-short",
        type=float,
        required=False,
        dest="TWE_short",
        default=None,
        help="specify total_wallet_exposure_short, overriding value from hjson config",
    )
    config = prep_config_multi(parser)

    # this parser is used to parse flags from backtest config
    parser = argparse.ArgumentParser(prog="flags_parser", description="used internally")
    parser.add_argument("-sm", type=str, required=False, dest="short_mode", default=None)
    parser.add_argument("-lm", type=str, required=False, dest="long_mode", default=None)
    parser.add_argument("-lw", type=float, required=False, dest="WE_limit_long", default=None)
    parser.add_argument("-sw", type=float, required=False, dest="WE_limit_short", default=None)
    parser.add_argument("-lc", type=str, required=False, dest="live_config_path", default=None)

    config["symbols"] = OrderedDict(sorted(config["symbols"].items()))
    config["exchange"] = load_user_info(config["user"])["exchange"]
    pprint.pprint(config)

    if os.path.isdir(config["live_configs_dir"]):
        live_configs_fnames = sorted(
            [f for f in os.listdir(config["live_configs_dir"]) if f.endswith(".json")]
        )
    else:
        live_configs_fnames = []
    live_configs = {}
    all_args = {}
    max_len_symbol = max([len(s) for s in config["symbols"]])

    for symbol in config["symbols"]:
        args = parser.parse_args(config["symbols"][symbol].split())
        all_args[symbol] = args
        live_config_fname_l = [x for x in live_configs_fnames if symbol in x]
        live_configs_dir_fname = (
            None
            if live_config_fname_l == []
            else oj(config["live_configs_dir"], live_config_fname_l[0])
        )
        for path in [
            args.live_config_path,
            live_configs_dir_fname,
            config["default_config_path"],
        ]:
            if path is not None and os.path.exists(path):
                try:
                    live_configs[symbol] = load_live_config(path)
                    logging.info(f"{symbol: <{max_len_symbol}} loaded live config: {path}")
                    break
                except Exception as e:
                    logging.error(f"failed to load live config {symbol} {path} {e}")
        else:
            raise Exception(f"no usable live config found for {symbol}")
        for pside in ["long", "short"]:
            if getattr(args, f"{pside}_mode") == "n":
                live_configs[symbol][pside]["enabled"] = True
            elif getattr(args, f"{pside}_mode") == "gs":
                live_configs[symbol][pside]["enabled"] = False
            else:
                live_configs[symbol][pside]["enabled"] = config[f"{pside}_enabled"]

    n_active_longs = len([s for s in config["symbols"] if live_configs[s]["long"]["enabled"]])
    n_active_shorts = len([s for s in config["symbols"] if live_configs[s]["short"]["enabled"]])

    WE_limits = {
        "long": config["TWE_long"] / n_active_longs if n_active_longs > 0 else 0.0,
        "short": config["TWE_short"] / n_active_shorts if n_active_shorts > 0 else 0.0,
    }

    for symbol in config["symbols"]:
        for pside in ["long", "short"]:
            for symbol in config["symbols"]:
                if getattr(all_args[symbol], f"WE_limit_{pside}") is None:
                    live_configs[symbol][pside]["wallet_exposure_limit"] = WE_limits[pside]
                else:
                    live_configs[symbol][pside]["wallet_exposure_limit"] = getattr(
                        all_args[symbol], f"WE_limit_{pside}"
                    )
                live_configs[symbol][pside]["wallet_exposure_limit"] = max(
                    live_configs[symbol][pside]["wallet_exposure_limit"], 0.01
                )

    if config["end_date"] in ["now", "", "today"]:
        config["end_date"] = ts_to_date_utc(utc_ms())[:10]
    coins = [s.replace("USDT", "") for s in config["symbols"]]
    config["cache_fpath"] = make_get_filepath(
        oj(
            f"{config['base_dir']}",
            "multisymbol",
            config["exchange"],
            f"{'_'.join(coins)}_{config['start_date']}_{config['end_date']}_hlc_cache.npy",
        )
    )
    # prepare_multsymbol_data() is computationally expensive, so use a cache
    try:
        hlcs = np.load(config["cache_fpath"])
        first_ts = 0
    except:
        first_ts, hlcs = await prepare_multsymbol_data(
            config["symbols"], config["start_date"], config["end_date"]
        )
        np.save(config["cache_fpath"], hlcs)

    pprint.pprint(config)
    config["live_configs"] = numpyize(
        [
            live_config_dict_to_list_recursive_grid(live_configs[symbol])
            for symbol in config["symbols"]
        ]
    )

    config["do_longs"] = tuplify([live_configs[s]["long"]["enabled"] for s in config["symbols"]])
    config["do_shorts"] = tuplify([live_configs[s]["short"]["enabled"] for s in config["symbols"]])

    mss_path = oj(
        f"{config['base_dir']}",
        "multisymbol",
        config["exchange"],
        "market_specific_settings",
        f"{symbol}_mss.json",
    )
    try:
        mss = fetch_market_specific_settings_multi(config["symbols"])
        json.dump(mss, open(make_get_filepath(mss_path), "w"))
    except Exception as e:
        print("failed to fetch market specific settings", e)
        try:
            mss = json.load(open(mss_path))
        except:
            raise Exception("failed to load market specific settings from cache")

    config["qty_steps"] = tuplify([mss[symbol]["qty_step"] for symbol in config["symbols"]])
    config["price_steps"] = tuplify([mss[symbol]["price_step"] for symbol in config["symbols"]])
    config["min_costs"] = tuplify([mss[symbol]["min_cost"] for symbol in config["symbols"]])
    config["min_qtys"] = tuplify([mss[symbol]["min_qty"] for symbol in config["symbols"]])
    config["c_mults"] = tuplify([mss[symbol]["c_mult"] for symbol in config["symbols"]])
    config["maker_fee"] = next(iter(mss.values()))["maker"]
    config["symbols"] = tuple(sorted(config["symbols"]))

    try:
        pd.set_option("display.precision", 10)
    except Exception as e:
        print("error setting pandas precision", e)

    print("backtesting...")

    sts = utc_ms()
    res = backtest_multi(hlcs, config)
    print(f"time elapsed for backtest {(utc_ms() - sts) / 1000:.6f}s")
    sts = utc_ms()
    fills, stats = res
    fdf = fills_multi_to_df(fills, config["symbols"], config["c_mults"])
    sdf = stats_multi_to_df(stats, config["symbols"], config["c_mults"])

    # fdf = pd.read_csv('backtests/multisymbol/binance/2024-01-06T01_38_47/fills.csv').set_index('minute')
    # sdf = pd.read_csv('backtests/multisymbol/binance/2024-01-06T01_38_47/stats.csv').set_index('minute')

    now_fname = ts_to_date_utc(utc_ms())[:19].replace(":", "_")
    backtest_metrics_path = make_get_filepath(
        oj(f"{config['base_dir']}", "multisymbol", config["exchange"], now_fname, "")
    )
    fdf.to_csv(oj(backtest_metrics_path, "fills.csv"))
    sdf.to_csv(oj(backtest_metrics_path, "stats.csv"))

    params = {"TWE_long": config["TWE_long"], "TWE_short": config["TWE_short"]}
    analysis = analyze_fills_multi(sdf, fdf, params)
    print(f"time elapsed for analysis {(utc_ms() - sts) / 1000:.6f}s")
    json.dump(
        analysis,
        open(oj(backtest_metrics_path, "analysis.json"), "w"),
        indent=4,
        sort_keys=True,
    )

    mkl = max([len(k) for k in analysis])
    for k, v in analysis.items():
        if isinstance(v, dict):
            continue
            mkls = max([len(s) for s in v])
            for symbol in v:
                mkl1 = max([len(k) for k in v[symbol]])
                for k1, v1 in v[symbol].items():
                    print(f"    {symbol: <{mkls}} {k1: <{mkl1}} {round_dynamic(v1, 6)}")
                print()
        else:
            print(f"{k: <{mkl}} {round_dynamic(v, 6)}")
    adf = pd.DataFrame({k: v for k, v in analysis["individual_analyses"].items()})
    adf.to_csv(oj(backtest_metrics_path, "analysis_summary.csv"))
    print(adf)

    # print

    if not adf.T.upnl_pct_min_long.isna().all():
        print("upnl pct min long")
        print(adf.T.upnl_pct_min_long.sort_values())
        print()
    if not adf.T.upnl_pct_min_short.isna().all():
        print("upnl pct min short")
        print(adf.T.upnl_pct_min_short.sort_values())

    if not (adf.T.loss_profit_ratio_long == 1.0).all():
        print("loss_profit_ratio_long")
        print(adf.T.loss_profit_ratio_long.sort_values(ascending=False))
        print()
    if not (adf.T.loss_profit_ratio_short == 1.0).all():
        print("loss_profit_ratio_short")
        print(adf.T.loss_profit_ratio_short.sort_values(ascending=False))

    print("pnl ratios")
    print(adf.T.pnl_ratio.sort_values())

    # plotting

    plt.rcParams["figure.figsize"] = [29, 18]

    print("plotting drawdowns...")
    min_multiplier = 60 * 24
    drawdowns = calc_drawdowns(sdf.equity)
    drawdowns_daily = drawdowns.groupby(drawdowns.index // min_multiplier * min_multiplier).min()
    drawdowns_ten_worst = drawdowns_daily.sort_values().iloc[:10]
    print("drawdowns ten worst")
    print(drawdowns_ten_worst)
    plt.clf()
    drawdowns_ten_worst.plot(style="ro")
    drawdowns.plot(xlabel="time", ylabel="drawdown", title="Drawdowns")
    plt.savefig(oj(backtest_metrics_path, "drawdowns.png"))

    print("plotting equity curve with stuckness...")
    plt.clf()
    plot_pnls_stuck(sdf, fdf)
    plt.title("Balance and equity")
    plt.xlabel = "time"
    plt.ylabel = "USDT"
    plt.savefig(oj(backtest_metrics_path, "balance_and_equity.png"))

    print("plotting pnl cumsums separate...")
    plt.clf()
    plot_pnls_separate(sdf, fdf)
    plt.title("Cumulative pnls")
    plt.xlabel = "time"
    plt.ylabel = "USDT"
    plt.savefig(oj(backtest_metrics_path, "cumulative_pnls.png"))

    print("plotting long and short pnl cumsums...")
    plt.clf()
    plot_pnls_long_short(sdf, fdf)
    plt.title("Long and short cumulative pnls")
    plt.xlabel = "time"
    plt.ylabel = "USDT"
    plt.savefig(oj(backtest_metrics_path, "cumulative_pnls_long_short.png"))

    # inspect two months before and two months after location of worst drawdown
    print("plotting around worst drawdown...")
    drawdowns_inspect_path = make_get_filepath(oj(backtest_metrics_path, "drawdown_inspections", ""))
    worst_drawdown_loc = drawdowns.sort_values().iloc[:1].index[0]
    wdls = worst_drawdown_loc - 60 * 24 * 30 * 2
    wdle = worst_drawdown_loc + 60 * 24 * 30 * 2
    sdfc = sdf.loc[wdls:wdle]
    plt.clf()
    sdfc.balance.plot()
    sdfc.equity.plot()
    plt.title("Worst Drawdown")
    plt.savefig(oj(backtest_metrics_path, "worst_drawdown.png"))

    # inspect for each symbol
    for symbol in config["symbols"]:
        print(f"plotting for {symbol}")
        plt.clf()
        plot_fills_multi(symbol, sdf.loc[wdls:wdle], fdf.loc[wdls:wdle])
        plt.title(f"{symbol} Fills two months before and after worst drawdown")
        plt.savefig(oj(drawdowns_inspect_path, f"{symbol}.png"))


if __name__ == "__main__":
    asyncio.run(main())
