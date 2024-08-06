import numpy as np
import pandas as pd
import os
import json
import passivbot_rust as pbr
import asyncio
import argparse
from procedures import (
    load_hjson_config,
    utc_ms,
    make_get_filepath,
    fetch_market_specific_settings_multi,
    load_config,
    dump_config,
)
from pure_funcs import (
    get_template_live_config,
    ts_to_date,
    calc_drawdowns,
    sort_dict_keys,
)
import pprint
from downloader import prepare_hlcs_forager
from njit_multisymbol import calc_noisiness_argsort_indices
from plotting import plot_fills_forager
import matplotlib.pyplot as plt
import logging

plt.rcParams["figure.figsize"] = [29, 18]


def oj(*x):
    return os.path.join(*x)


def process_forager_fills(fills):
    fdf = pd.DataFrame(
        fills,
        columns=[
            "minute",
            "symbol",
            "pnl",
            "fee_paid",
            "balance",
            "qty",
            "price",
            "psize",
            "pprice",
            "type",
        ],
    )
    return fdf


def analyze_fills_forager(symbols, hlcs, fdf, equities):
    analysis = {}
    pnls = {}
    for pside in ["long", "short"]:
        fdfc = fdf[fdf.type.str.contains(pside)]
        profit = fdfc[fdfc.pnl > 0.0].pnl.sum()
        loss = fdfc[fdfc.pnl < 0.0].pnl.sum()
        if len(fdfc) == 0:
            pnls[pside] = 0.0
            analysis[f"loss_profit_ratio_{pside}"] = 1.0
            continue
        pnls[pside] = profit + loss
        analysis[f"loss_profit_ratio_{pside}"] = abs(loss / profit)

    analysis["pnl_ratio_long_short"] = pnls["long"] / (pnls["long"] + pnls["short"])
    bdf = fdf.groupby((fdf.minute // 60) * 60).balance.last()
    edf = equities.iloc[::60]
    nidx = np.arange(min(bdf.index[0], edf.index[0]), max(bdf.index[-1], edf.index[-1]), 60)
    bal_eq = pd.DataFrame({"balance": bdf, "equity": edf}, index=nidx).ffill().bfill()
    return sort_dict_keys(analysis), bal_eq


def compare_dicts(dict1, dict2, path=""):
    for key in sorted(set(dict1.keys()) | set(dict2.keys())):
        if key not in dict1:
            print(f"{path}{key}: Missing in first dict. Value in second dict: {dict2[key]}")
        elif key not in dict2:
            print(f"{path}{key}: Missing in second dict. Value in first dict: {dict1[key]}")
        elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
            compare_dicts(dict1[key], dict2[key], f"{path}{key}.")
        elif dict1[key] != dict2[key]:
            print(f"{path}{key}: Values differ. First dict:  {dict1[key]} Second dict: {dict2[key]}")


def compare_dict_keys(dict1, dict2):
    def get_all_keys(d):
        keys = set(d.keys())
        for value in d.values():
            if isinstance(value, dict):
                keys.update(get_all_keys(value))
        return keys

    return get_all_keys(dict1) == get_all_keys(dict2)


def check_keys(dict0, dict1):
    def check_nested(d0, d1):
        for key, value in d0.items():
            if key not in d1:
                return False
            if isinstance(value, dict):
                if not isinstance(d1[key], dict):
                    return False
                if not check_nested(value, d1[key]):
                    return False
        return True

    return check_nested(dict0, dict1)


def convert_to_v7(cfg: dict):
    formatted = get_template_live_config("v7")
    if check_keys(formatted, cfg):
        return cfg
    cmap = {
        "ddown_factor": "entry_grid_double_down_factor",
        "initial_eprice_ema_dist": "entry_initial_ema_dist",
        "initial_qty_pct": "entry_initial_qty_pct",
        "markup_range": "close_grid_markup_range",
        "min_markup": "close_grid_min_markup",
        "rentry_pprice_dist": "entry_grid_spacing_pct",
        "rentry_pprice_dist_wallet_exposure_weighting": "entry_grid_spacing_weight",
        "ema_span_0": "ema_span_0",
        "ema_span_1": "ema_span_1",
    }
    if "args" in cfg:
        for key in ["start_date", "end_date", "starting_balance", "exchange"]:
            if key in formatted["backtest"]:
                formatted["backtest"][key] = cfg["args"][key]
        formatted["approved_symbols"] = cfg["args"]["symbols"]
        for pside in ["long", "short"]:
            if not cfg["args"][f"{pside}_enabled"]:
                cfg["live_config"]["global"][f"TWE_{pside}"] = 0.0
    if "live_config" in cfg and all([x in cfg["live_config"] for x in ["global", "long", "short"]]):
        for pside in ["long", "short"]:
            for k0 in cfg["live_config"][pside]:
                if k0 in cmap and cmap[k0] in formatted[pside]:
                    formatted[pside][cmap[k0]] = cfg["live_config"][pside][k0]
            formatted[pside]["close_grid_qty_pct"] = 1.0 / round(
                cfg["live_config"][pside]["n_close_orders"]
            )
            formatted[pside]["unstuck_loss_allowance_pct"] = cfg["live_config"]["global"][
                "loss_allowance_pct"
            ]
            formatted[pside]["unstuck_threshold"] = cfg["live_config"]["global"]["stuck_threshold"]
            formatted[pside]["total_wallet_exposure_limit"] = cfg["live_config"]["global"][
                f"TWE_{pside}"
            ]
            if formatted[pside]["total_wallet_exposure_limit"] > 0.00001:
                formatted[pside]["n_positions"] = len(formatted["approved_symbols"])
            else:
                formatted[pside]["n_positions"] = 0
            formatted[pside]["close_trailing_grid_ratio"] = 0.0
            formatted[pside]["entry_trailing_grid_ratio"] = 0.0
    return formatted


def add_argparse_args_to_config(config, args):
    for key, value in vars(args).items():
        try:
            if value is None:
                continue
            if key == "symbols":
                symbols = sorted(set(value.split(",")))
                if symbols != sorted(set(config["backtest"]["symbols"])):
                    logging.info(f"new symbols: {symbols}")
                    config["backtest"]["symbols"] = symbols
            elif key in config["backtest"]:
                if not isinstance(config["backtest"][key], dict):
                    if config["backtest"][key] != value:
                        logging.info(f"changing backtest {key} {config['backtest'][key]} -> {value}")
                        config["backtest"][key] = value
            elif key in config["optimize"]:
                if not isinstance(config["optimize"][key], dict):
                    if config["optimize"][key] != value:
                        logging.info(f"changing optimize {key} {config['optimize'][key]} -> {value}")
                        config["optimize"][key] = value
            elif key in config["optimize"]["bounds"]:
                new_value = [value, value]
                if config["optimize"]["bounds"][key] != new_value:
                    logging.info(f"fixing optimizing bound {key} to {value}")
                    config["optimize"]["bounds"][key] = new_value
            elif key in config["optimize"]["limits"]:
                old_value = config["optimize"]["limits"][key]
                if old_value != value:
                    logging.info(f"changing optimizing limit {key} from {old_value} to {value}")
                    config["optimize"]["limits"][key] = value
        except Exception as e:
            raise Exception(f"failed to add argparse arg to config {key}: {e}")
    return config


def load_and_process_config(path: str):
    """
    loads and processes configs of various types; returns standardized v7 config
    """
    loaded = load_hjson_config(path)
    formatted = convert_to_v7(loaded)
    return formatted


async def prepare_hlcs_mss(config):
    results_path = oj(
        config["backtest"]["base_dir"],
        "forager",
        config["backtest"]["exchange"],
        "",
    )
    mss_path = oj(
        results_path,
        "market_specific_settings.json",
    )
    try:
        mss = fetch_market_specific_settings_multi(exchange=config["backtest"]["exchange"])
        json.dump(mss, open(make_get_filepath(mss_path), "w"))
    except Exception as e:
        print("failed to fetch market specific settings", e)
        try:
            mss = json.load(open(mss_path))
            print(f"loaded market specific settings from cache {mss_path}")
        except:
            raise Exception("failed to load market specific settings from cache")

    timestamps, hlcs = await prepare_hlcs_forager(
        config["backtest"]["symbols"],
        config["backtest"]["start_date"],
        config["backtest"]["end_date"],
        base_dir=config["backtest"]["base_dir"],
        exchange=config["backtest"]["exchange"],
    )

    return hlcs, mss, results_path


def prep_backtest_args(config, mss, exchange_params=None, backtest_params=None):
    symbols = sorted(set(config["backtest"]["symbols"]))  # sort for consistency
    bot_params = {k: config["bot"][k].copy() for k in ["long", "short"]}
    for pside in bot_params:
        bot_params[pside]["wallet_exposure_limit"] = (
            bot_params[pside]["total_wallet_exposure_limit"] / bot_params[pside]["n_positions"]
            if bot_params[pside]["n_positions"] > 0
            else 0.0
        )
    if exchange_params is None:
        exchange_params = [
            {k: mss[symbol][k] for k in ["qty_step", "price_step", "min_qty", "min_cost", "c_mult"]}
            for symbol in symbols
        ]
    if backtest_params is None:
        backtest_params = {
            "starting_balance": config["backtest"]["starting_balance"],
            "maker_fee": mss[symbols[0]]["maker"],
            "symbols": symbols,
        }
    return bot_params, exchange_params, backtest_params


def run_backtest(hlcs, preferred_coins, mss, config: dict):
    bot_params, exchange_params, backtest_params = prep_backtest_args(config, mss)
    print(f"Starting backtest...")
    sts = utc_ms()
    fills, equities, analysis = pbr.run_backtest(
        hlcs, preferred_coins, bot_params, exchange_params, backtest_params
    )
    print(f"seconds elapsed for backtest: {(utc_ms() - sts) / 1000:.4f}")
    return fills, equities, analysis


def post_process(config, hlcs, fills, equities, analysis, results_path):
    sts = utc_ms()
    fdf = process_forager_fills(fills)
    equities = pd.Series(equities)
    analysis_py, bal_eq = analyze_fills_forager(config["backtest"]["symbols"], hlcs, fdf, equities)
    for k in analysis_py:
        if k not in analysis:
            analysis[k] = analysis_py[k]
    print(f"seconds elapsed for analysis: {(utc_ms() - sts) / 1000:.4f}")
    pprint.pprint(analysis)
    results_path = make_get_filepath(
        oj(results_path, f"{ts_to_date(utc_ms())[:19].replace(':', '_')}", "")
    )
    json.dump(analysis, open(f"{results_path}analysis.json", "w"), indent=4, sort_keys=True)
    dump_config(config, f"{results_path}config.json")
    fdf.to_csv(f"{results_path}fills.csv")
    plot_forager(results_path, config["backtest"]["symbols"], fdf, bal_eq, hlcs)


def plot_forager(results_path, symbols: [str], fdf: pd.DataFrame, bal_eq, hlcs):
    plots_dir = make_get_filepath(oj(results_path, "fills_plots", ""))
    plt.clf()
    bal_eq.plot()
    plt.savefig(oj(results_path, "balance_and_equity.png"))

    for i, symbol in enumerate(symbols):
        print(f"Plotting fills for {symbol}")
        hlcs_df = pd.DataFrame(hlcs[:, i, :], columns=["high", "low", "close"])
        fdfc = fdf[fdf.symbol == symbol]
        plt.clf()
        plot_fills_forager(fdfc, hlcs_df)
        plt.title(f"Fills {symbol}")
        plt.xlabel = "time"
        plt.ylabel = "price"
        plt.savefig(oj(plots_dir, f"{symbol}.png"))


def add_argparse_args_backtest_forager(parser):
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
    return args


def calc_preferred_coins(hlcs, config):
    return np.argsort(
        -pbr.calc_noisiness_py(hlcs, config["live"]["noisiness_rolling_mean_window_size"])
    )


async def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    parser = argparse.ArgumentParser(prog="backtest_forager", description="run forager backtest")
    parser.add_argument("config_path", type=str, default=None, help="path to hjson passivbot config")
    args = add_argparse_args_backtest_forager(parser)
    config = load_config("configs/template.hjson" if args.config_path is None else args.config_path)
    config = add_argparse_args_to_config(config, args)
    hlcs, mss, results_path = await prepare_hlcs_mss(config)
    preferred_coins = calc_preferred_coins(hlcs, config)
    fills, equities, analysis = run_backtest(hlcs, preferred_coins, mss, config)
    post_process(config, hlcs, fills, equities, analysis, results_path)


if __name__ == "__main__":
    asyncio.run(main())
