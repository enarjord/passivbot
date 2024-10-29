import numpy as np
import pandas as pd
import os
import json
import passivbot_rust as pbr
import asyncio
import argparse
from procedures import (
    utc_ms,
    make_get_filepath,
    fetch_market_specific_settings_multi,
    load_config,
    dump_config,
    coin_to_symbol,
    add_arguments_recursively,
    update_config_with_args,
    format_config,
)
from pure_funcs import (
    get_template_live_config,
    ts_to_date,
    sort_dict_keys,
)
import pprint
from downloader import prepare_hlcvs
from plotting import plot_fills_forager
import matplotlib.pyplot as plt
import logging
from main import manage_rust_compilation

import tempfile
from contextlib import contextmanager


@contextmanager
def create_shared_memory_file(hlcvs):
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    shared_memory_file = temp_file.name
    try:
        with open(shared_memory_file, "wb") as f:
            f.write(hlcvs.tobytes())
        yield shared_memory_file
    finally:
        os.unlink(shared_memory_file)


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


def analyze_fills_forager(symbols, hlcvs, fdf, equities):
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

    div_by = 60  # save some disk space. Set to 1 to dump uncropped
    analysis["pnl_ratio_long_short"] = pnls["long"] / (pnls["long"] + pnls["short"])
    bdf = fdf.groupby((fdf.minute // div_by) * div_by).balance.last()
    edf = equities.iloc[::div_by]
    nidx = np.arange(min(bdf.index[0], edf.index[0]), max(bdf.index[-1], edf.index[-1]), div_by)
    bal_eq = pd.DataFrame({"balance": bdf, "equity": edf}, index=nidx).astype(float).ffill().bfill()
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


async def prepare_hlcvs_mss(config):
    results_path = oj(
        config["backtest"]["base_dir"],
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

    symbols, timestamps, hlcvs = await prepare_hlcvs(config)
    logging.info(f"Finished preparing hlcvs data. Shape: {hlcvs.shape}")

    return symbols, hlcvs, mss, results_path


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


def run_backtest(hlcvs, mss, config: dict):
    bot_params, exchange_params, backtest_params = prep_backtest_args(config, mss)
    print(f"Starting backtest...")
    sts = utc_ms()

    with create_shared_memory_file(hlcvs) as shared_memory_file:
        fills, equities, analysis = pbr.run_backtest(
            shared_memory_file,
            hlcvs.shape,
            hlcvs.dtype.str,
            bot_params,
            exchange_params,
            backtest_params,
        )

    print(f"seconds elapsed for backtest: {(utc_ms() - sts) / 1000:.4f}")
    return fills, equities, analysis


def post_process(config, hlcvs, fills, equities, analysis, results_path):
    sts = utc_ms()
    fdf = process_forager_fills(fills)
    equities = pd.Series(equities)
    analysis_py, bal_eq = analyze_fills_forager(config["backtest"]["symbols"], hlcvs, fdf, equities)
    for k in analysis_py:
        if k not in analysis:
            analysis[k] = analysis_py[k]
    print(f"seconds elapsed for analysis: {(utc_ms() - sts) / 1000:.4f}")
    pprint.pprint(analysis)
    results_path = make_get_filepath(
        oj(results_path, f"{ts_to_date(utc_ms())[:19].replace(':', '_')}", "")
    )
    json.dump(analysis, open(f"{results_path}analysis.json", "w"), indent=4, sort_keys=True)
    config["analysis"] = analysis
    dump_config(config, f"{results_path}config.json")
    fdf.to_csv(f"{results_path}fills.csv")
    plot_forager(results_path, config["backtest"]["symbols"], fdf, bal_eq, hlcvs)


def plot_forager(results_path, symbols: [str], fdf: pd.DataFrame, bal_eq, hlcvs):
    plots_dir = make_get_filepath(oj(results_path, "fills_plots", ""))
    bal_eq.to_csv(oj(results_path, "balance_and_equity.csv"))
    plt.clf()
    bal_eq.plot()
    plt.savefig(oj(results_path, "balance_and_equity.png"))

    for i, symbol in enumerate(symbols):
        try:
            print(f"Plotting fills for {symbol}")
            hlcvs_df = pd.DataFrame(hlcvs[:, i, :3], columns=["high", "low", "close"])
            fdfc = fdf[fdf.symbol == symbol]
            plt.clf()
            plot_fills_forager(fdfc, hlcvs_df)
            plt.title(f"Fills {symbol}")
            plt.xlabel = "time"
            plt.ylabel = "price"
            plt.savefig(oj(plots_dir, f"{symbol}.png"))
        except Exception as e:
            print(f"Error plotting {symbol} {e}")


async def main():
    manage_rust_compilation()
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    parser = argparse.ArgumentParser(prog="backtest", description="run forager backtest")
    parser.add_argument(
        "config_path", type=str, default=None, nargs="?", help="path to json passivbot config"
    )
    template_config = get_template_live_config("v7")
    del template_config["optimize"]
    keep_live_keys = {
        "approved_coins",
        "ignored_coins",
        "minimum_coin_age_days",
    }
    for key in sorted(template_config["live"]):
        if key not in keep_live_keys:
            del template_config["live"][key]
    add_arguments_recursively(parser, template_config)
    args = parser.parse_args()
    if args.config_path is None:
        logging.info(f"loading default template config configs/template.json")
        config = load_config("configs/template.json")
    else:
        logging.info(f"loading config {args.config_path}")
        config = load_config(args.config_path)
    update_config_with_args(config, args)
    config = format_config(config)
    symbols, hlcvs, mss, results_path = await prepare_hlcvs_mss(config)
    config["backtest"]["symbols"] = symbols
    fills, equities, analysis = run_backtest(hlcvs, mss, config)
    post_process(config, hlcvs, fills, equities, analysis, results_path)


if __name__ == "__main__":
    asyncio.run(main())
