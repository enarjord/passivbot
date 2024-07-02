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
)
from pure_funcs import (
    get_template_live_config,
    process_forager_fills,
    analyze_fills_forager,
    ts_to_date,
)
import pprint
from downloader import prepare_hlcs_forager
from njit_multisymbol import calc_noisiness_argsort_indices
from plotting import plot_fills_forager
import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = [29, 18]


def oj(*x):
    return os.path.join(*x)


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


def convert_to_v7(cfg: dict):
    formatted = get_template_live_config("v7")
    if compare_dict_keys(cfg, formatted):
        return cfg
    cmap = {
        "ddown_factor": "entry_grid_double_down_factor",
        "initial_eprice_ema_dist": "entry_initial_ema_dist",
        "initial_qty_pct": "entry_initial_qty_pct",
        "markup_range": "close_grid_markup_range",
        "min_markup": "close_grid_min_markup",
        "rentry_pprice_dist": "entry_grid_spacing_pct",
        "rentry_pprice_dist_wallet_exposure_weighting": "entry_grid_spacing_weight",
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
            formatted[pside]["close_grid_qty_pct"] = 1.0 / cfg["live_config"][pside]["n_close_orders"]
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
        config["approved_symbols"], config["backtest"]["start_date"], config["backtest"]["end_date"]
    )
    return hlcs, mss, results_path


def run_backtest(hlcs, noisiness_indices, mss, config: dict):
    symbols = sorted(set(config["approved_symbols"]))
    bot_params = {k: config[k].copy() for k in ["long", "short"]}
    for pside in bot_params:
        bot_params[pside]["wallet_exposure_limit"] = (
            bot_params[pside]["total_wallet_exposure_limit"] / bot_params[pside]["n_positions"]
            if bot_params[pside]["n_positions"] > 0
            else 0.0
        )
    exchange_params = [
        {k: mss[symbol][k] for k in ["qty_step", "price_step", "min_qty", "min_cost", "c_mult"]}
        for symbol in symbols
    ]
    backtest_params = {
        "starting_balance": config["backtest"]["starting_balance"],
        "maker_fee": mss[symbols[0]]["maker"],
        "symbols": symbols,
    }
    print(f"Starting backtest...")
    sts = utc_ms()
    fills = pbr.run_backtest(hlcs, noisiness_indices, bot_params, exchange_params, backtest_params)
    print(f"seconds elapsed for backtest: {(utc_ms() - sts) / 1000:.4f}")
    return fills


def post_process(config, hlcs, fills, results_path):
    sts = utc_ms()
    fdf = process_forager_fills(fills)
    analysis, balance_and_equity = analyze_fills_forager(config["approved_symbols"], hlcs, fdf)
    print(f"seconds elapsed for analysis: {(utc_ms() - sts) / 1000:.4f}")
    pprint.pprint(analysis)
    print(balance_and_equity)
    results_path = make_get_filepath(
        oj(results_path, f"{ts_to_date(utc_ms())[:19].replace(':', '_')}", "")
    )
    json.dump(analysis, open(f"{results_path}analysis.json", "w"), indent=4, sort_keys=True)
    json.dump(config, open(f"{results_path}config.json", "w"), indent=4, sort_keys=True)
    plot_forager(results_path, config["approved_symbols"], fdf, hlcs, balance_and_equity)


def plot_forager(results_path, symbols: [str], fdf: pd.DataFrame, hlcs, balance_and_equity):
    plots_dir = make_get_filepath(oj(results_path, "fills_plots", ""))
    plt.clf()
    balance_and_equity.plot()
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


async def main():
    parser = argparse.ArgumentParser(prog="backtest_forager", description="run forager backtest")
    parser.add_argument("config_path", type=str, help="path to hjson passivbot config")
    args = parser.parse_args()
    config = load_and_process_config(args.config_path)
    hlcs, mss, results_path = await prepare_hlcs_mss(config)
    noisiness_indices = calc_noisiness_argsort_indices(hlcs).astype(np.int32)
    fills = run_backtest(hlcs, noisiness_indices, mss, config)
    post_process(config, hlcs, fills, results_path)


if __name__ == "__main__":
    asyncio.run(main())
