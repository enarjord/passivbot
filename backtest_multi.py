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
    denumpyize,
    calc_hash,
    add_missing_params_to_hjson_live_multi_config,
    get_template_live_config,
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
    parser_items = [
        ("s", "symbols", "symbols", str, ", comma separated (SYM1USDT,SYM2USDT,...)"),
        ("e", "exchange", "exchange", str, ""),
        ("sd", "start_date", "start_date", str, ""),
        (
            "ed",
            "end_date",
            "end_date",
            str,
            ", if end date is 'now', will use current date as end date",
        ),
        ("sb", "starting_balance", "starting_balance", float, ""),
        ("lap", "loss_allowance_pct", "loss_allowance_pct", float, ""),
        ("st", "stuck_threshold", "stuck_threshold", float, ""),
        ("ucp", "unstuck_close_pct", "unstuck_close_pct", float, ""),
        ("le", "long_enabled", "long_enabled", str2bool, " (y/n or t/f)"),
        ("se", "short_enabled", "short_enabled", str2bool, " (y/n or t/f)"),
        ("bd", "base_dir", "base_dir", str, ""),
    ]
    for k0, k1, d, t, h in parser_items:
        parser.add_argument(
            *[f"-{k0}", f"--{k1}"] + ([f"--{k1.replace('_', '-')}"] if "_" in k1 else []),
            type=t,
            required=False,
            dest=d,
            default=None,
            help=f"specify {k1}{h}, overriding value from hjson config.",
        )
    args = parser.parse_args()
    return args2config(args)


def load_and_parse_config(path: str):
    loaded = hjson.load(open(path))
    if all([x in loaded for x in ["live_config", "args"]]):
        # single live config type
        formatted = {}
        for k in [
            "exchange",
            "long_enabled",
            "short_enabled",
            "start_date",
            "end_date",
            "starting_balance",
            "symbols",
        ]:
            formatted[k] = loaded["args"][k]
        for k in [
            "TWE_long",
            "TWE_short",
            "loss_allowance_pct",
            "stuck_threshold",
            "unstuck_close_pct",
        ]:
            formatted[k] = loaded["live_config"]["global"][k]

        formatted["live_configs_dir"] = "configs/live/multisymbol/no_AU/"
        formatted["default_config_path"] = "tmp/test.json"
        formatted["base_dir"] = "backtests"
        formatted["live_configs"] = {
            symbol: {pside: loaded["live_config"][pside] for pside in ["long", "short"]}
            for symbol in formatted["symbols"]
        }
        return formatted

    if all(
        [
            x in loaded
            for x in [
                "exchange",
                "loss_allowance_pct",
                "stuck_threshold",
                "unstuck_close_pct",
                "TWE_long",
                "TWE_short",
                "long_enabled",
                "short_enabled",
                "start_date",
                "end_date",
                "starting_balance",
                "symbols",
                "live_configs_dir",
                "default_config_path",
                "base_dir",
            ]
        ]
    ):
        # hjson backtest config type
        formatted = loaded
        formatted["live_configs"] = {}
        return formatted
    if all(
        [
            x in loaded
            for x in [
                "exchange",
                "start_date",
                "end_date",
                "symbols",
                "base_dir",
                "n_cpus",
                "iters",
                "starting_balance",
                "market_type",
                "worst_drawdown_lower_bound",
                "long_enabled",
                "short_enabled",
                "bounds",
            ]
        ]
    ):
        # hjson optimize config type
        formatted = loaded
        formatted["live_configs"] = {}
        return formatted
    try:
        loaded, _ = add_missing_params_to_hjson_live_multi_config(loaded)
        if all([x in loaded for x in get_template_live_config("multi_hjson")]):
            # hjson live multi config
            formatted = loaded
            formatted["exchange"] = "binance"
            formatted["start_date"] = "2021-05-01"
            formatted["end_date"] = "now"
            formatted["starting_balance"] = 100000.0
            formatted["base_dir"] = "backtests"
            formatted["symbols"] = loaded["approved_symbols"]
            if loaded["universal_live_config"]:
                formatted["live_configs"] = {
                    symbol: {
                        pside: loaded["universal_live_config"][pside] for pside in ["long", "short"]
                    }
                    for symbol in formatted["approved_symbols"]
                }
                for s in formatted["live_configs"]:
                    for pside in ["long", "short"]:
                        formatted["live_configs"][s][pside]["enabled"] = formatted[f"{pside}_enabled"]
            else:
                formatted["live_configs"] = {}
            return formatted
    except:
        pass
    raise Exception("unknown config type")


def args2config(args):
    config = OrderedDict()
    for key, value in vars(args).items():
        if "config_path" in key:
            logging.info(f"loading {value}")
            config = load_and_parse_config(value)
        elif key not in config:
            logging.info(f"setting {key}: {value}")
            config[key] = value
        elif getattr(args, key) is not None:
            if key == "symbols":
                new_symbols = {s: "" for s in getattr(args, key).split(",") if s}
                if new_symbols != config["symbols"]:
                    logging.info(f"new symbols: {new_symbols}")
                    config["symbols"] = new_symbols
            else:
                if key in config and config[key] != getattr(args, key):
                    logging.info(f"changing {key}: {config[key]} -> {getattr(args, key)}")
                    config[key] = getattr(args, key)
    if isinstance(config["symbols"], list):
        config["symbols"] = {s: "" for s in config["symbols"]}
    config["symbols"] = OrderedDict(sorted(config["symbols"].items()))
    for key, default_val in [
        ("base_dir", "backtests"),
        ("starting_balance", 10000),
        ("start_date", "2021-05-01"),
        ("end_date", "now"),
    ]:
        if key not in config:
            logging.info(
                f'key "{key}"" missing from config; setting to default value "{default_val}"'
            )
            config[key] = default_val
    return config


async def prep_hlcs_mss_config(config):
    if config["end_date"] in ["now", "", "today"]:
        config["end_date"] = ts_to_date_utc(utc_ms())[:10]
    coins = [s.replace("USDT", "") for s in sorted(set(config["symbols"]))]
    config["cache_fpath"] = make_get_filepath(
        oj(
            f"{config['base_dir']}",
            "multisymbol",
            config["exchange"],
            f"{calc_hash(coins)}_{config['start_date']}_{config['end_date']}_hlc_cache.npy",
        )
    )

    mss_path = oj(
        f"{config['base_dir']}",
        "multisymbol",
        config["exchange"],
        "market_specific_settings.json",
    )
    try:
        mss = fetch_market_specific_settings_multi(exchange=config["exchange"])
        json.dump(mss, open(make_get_filepath(mss_path), "w"))
    except Exception as e:
        print("failed to fetch market specific settings", e)
        try:
            mss = json.load(open(mss_path))
            print(f"loaded market specific settings from cache {mss_path}")
        except:
            raise Exception("failed to load market specific settings from cache")

    # prepare_multsymbol_data() is computationally expensive, so use a cache
    try:
        hlcs = np.load(config["cache_fpath"])
        first_ts = 0
    except:
        first_ts, hlcs = await prepare_multsymbol_data(
            config["symbols"],
            config["start_date"],
            config["end_date"],
            config["base_dir"],
            config["exchange"],
        )
        np.save(config["cache_fpath"], hlcs)
    return hlcs, mss, config


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
        help="backtest config hjson file (default: configs/backtest.multi.hjson) or json config from results_multi_analysis/",
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

    if os.path.isdir(config["live_configs_dir"]):
        live_configs_fnames = sorted(
            [f for f in os.listdir(config["live_configs_dir"]) if f.endswith(".json")]
        )
    else:
        live_configs_fnames = []
    if "live_configs" not in config or not config["live_configs"]:
        config["live_configs"] = {}
    all_args = {}
    max_len_symbol = max([len(s) for s in config["symbols"]])

    for symbol in config["symbols"]:
        args = parser.parse_args(config["symbols"][symbol].split())
        all_args[symbol] = args
        if symbol in config["live_configs"]:
            continue
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
                    config["live_configs"][symbol] = load_live_config(path)
                    logging.info(f"{symbol: <{max_len_symbol}} loaded live config: {path}")
                    break
                except Exception as e:
                    logging.error(f"failed to load live config {symbol} {path} {e}")
        else:
            raise Exception(f"no usable live config found for {symbol}")
        for pside in ["long", "short"]:
            if getattr(args, f"{pside}_mode") == "n":
                config["live_configs"][symbol][pside]["enabled"] = True
            elif getattr(args, f"{pside}_mode") == "gs":
                config["live_configs"][symbol][pside]["enabled"] = False
            else:
                config["live_configs"][symbol][pside]["enabled"] = config[f"{pside}_enabled"]

    n_active_longs = len(
        [s for s in config["symbols"] if config["live_configs"][s]["long"]["enabled"]]
    )
    n_active_shorts = len(
        [s for s in config["symbols"] if config["live_configs"][s]["short"]["enabled"]]
    )

    WE_limits = {
        "long": config["TWE_long"] / n_active_longs if n_active_longs > 0 else 0.0,
        "short": config["TWE_short"] / n_active_shorts if n_active_shorts > 0 else 0.0,
    }

    for symbol in config["symbols"]:
        for pside in ["long", "short"]:
            for symbol in config["symbols"]:
                if getattr(all_args[symbol], f"WE_limit_{pside}") is None:
                    config["live_configs"][symbol][pside]["wallet_exposure_limit"] = WE_limits[pside]
                else:
                    config["live_configs"][symbol][pside]["wallet_exposure_limit"] = getattr(
                        all_args[symbol], f"WE_limit_{pside}"
                    )
                config["live_configs"][symbol][pside]["wallet_exposure_limit"] = max(
                    config["live_configs"][symbol][pside]["wallet_exposure_limit"], 0.001
                )

    hlcs, mss, config = await prep_hlcs_mss_config(config)

    now_fname = ts_to_date_utc(utc_ms())[:19].replace(":", "_")
    backtest_metrics_path = make_get_filepath(
        oj(f"{config['base_dir']}", "multisymbol", config["exchange"], now_fname, "")
    )
    hjson.dump(denumpyize(config), open(oj(backtest_metrics_path, "backtest_config.hjson"), "w"))

    config["do_longs"] = tuplify(
        [config["live_configs"][s]["long"]["enabled"] for s in config["symbols"]]
    )
    config["do_shorts"] = tuplify(
        [config["live_configs"][s]["short"]["enabled"] for s in config["symbols"]]
    )
    config["live_configs"] = numpyize(
        [
            live_config_dict_to_list_recursive_grid(config["live_configs"][symbol])
            for symbol in config["symbols"]
        ]
    )

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

    # fdf = pd.read_csv("backtests/multisymbol/binance/2024-01-07T02_16_06/fills.csv").set_index("minute")
    # sdf = pd.read_csv("backtests/multisymbol/binance/2024-01-07T02_16_06/stats.csv").set_index("minute")

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
