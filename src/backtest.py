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
    config["analysis"] = analysis
    dump_config(config, f"{results_path}config.json")
    fdf.to_csv(f"{results_path}fills.csv")
    plot_forager(results_path, config["backtest"]["symbols"], fdf, bal_eq, hlcs)


def plot_forager(results_path, symbols: [str], fdf: pd.DataFrame, bal_eq, hlcs):
    plots_dir = make_get_filepath(oj(results_path, "fills_plots", ""))
    plt.clf()
    bal_eq.plot()
    plt.savefig(oj(results_path, "balance_and_equity.png"))

    for i, symbol in enumerate(symbols):
        try:
            print(f"Plotting fills for {symbol}")
            hlcs_df = pd.DataFrame(hlcs[:, i, :], columns=["high", "low", "close"])
            fdfc = fdf[fdf.symbol == symbol]
            plt.clf()
            plot_fills_forager(fdfc, hlcs_df)
            plt.title(f"Fills {symbol}")
            plt.xlabel = "time"
            plt.ylabel = "price"
            plt.savefig(oj(plots_dir, f"{symbol}.png"))
        except Exception as e:
            print(f"Error plotting {symbol} {e}")


def calc_preferred_coins(hlcvs, config):
    w_size = config["live"]["ohlcv_rolling_window"]
    n_coins = hlcvs.shape[1]

    # Calculate noisiness indices
    noisiness_indices = np.argsort(-pbr.calc_noisiness_py(hlcvs[:, :, :3], w_size))

    # Calculate volume-based eligibility
    if config["live"]["relative_volume_filter_clip_pct"] > 0.0:
        n_eligibles = int(round(n_coins * (1 - config["live"]["relative_volume_filter_clip_pct"])))

        for pside in ["long", "short"]:
            if (
                config["bot"][pside]["n_positions"] > 0.0
                and config["bot"][pside]["total_wallet_exposure_limit"] > 0.0
            ):
                n_eligibles = max(n_eligibles, int(round(config["bot"][pside]["n_positions"])))

        if n_eligibles < n_coins:
            # Calculate rolling volumes and get volume-based ranking
            rolling_volumes = pbr.calc_volumes_py(hlcvs, w_size)
            volume_ranking = np.argsort(-rolling_volumes, axis=1)

            # Create a mask for eligible coins based on volume (vectorized)
            rows = np.arange(hlcvs.shape[0])[:, None]
            cols = volume_ranking[:, :n_eligibles]
            eligibility_mask = np.zeros((hlcvs.shape[0], n_coins), dtype=bool)
            eligibility_mask[rows, cols] = True

            # Filter noisiness_indices based on eligibility
            filtered_noisiness_indices = np.array(
                [
                    indices[mask]
                    for indices, mask in zip(
                        noisiness_indices, eligibility_mask[rows, noisiness_indices]
                    )
                ]
            )

            return filtered_noisiness_indices

    return noisiness_indices


async def main():
    manage_rust_compilation()
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    parser = argparse.ArgumentParser(prog="backtest", description="run forager backtest")
    parser.add_argument("config_path", type=str, default=None, help="path to hjson passivbot config")
    template_config = get_template_live_config("v7")
    del template_config["optimize"]
    keep_live_keys = {
        "approved_coins",
        "ignored_coins",
        "minimum_coin_age_days",
        "ohlcv_rolling_window",
        "relative_volume_filter_clip_pct",
    }
    for key in sorted(template_config["live"]):
        if key not in keep_live_keys:
            del template_config["live"][key]
    add_arguments_recursively(parser, template_config)
    args = parser.parse_args()
    config = load_config("configs/template.hjson" if args.config_path is None else args.config_path)
    update_config_with_args(config, args)
    config = format_config(config)
    symbols, hlcvs, mss, results_path = await prepare_hlcvs_mss(config)
    config["backtest"]["symbols"] = symbols
    preferred_coins = calc_preferred_coins(hlcvs, config)
    hlcs = hlcvs[:, :, :3]
    fills, equities, analysis = run_backtest(hlcs, preferred_coins, mss, config)
    post_process(config, hlcs, fills, equities, analysis, results_path)


if __name__ == "__main__":
    asyncio.run(main())
