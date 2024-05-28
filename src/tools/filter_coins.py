import pandas as pd
import pprint
import argparse
import ccxt.async_support as ccxt
import numpy as np
import argparse
import hjson
import json
import asyncio

import os

os.environ["NOJIT"] = "true"

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from forager import get_min_costs_and_contract_multipliers
from procedures import load_live_config, utc_ms, make_get_filepath, get_first_ohlcv_timestamps
from pure_funcs import floatify, ts_to_date_utc, date2ts_utc, str2bool, symbol2coin


def parse_backtest_metrics():
    path = "configs/live/multisymbol/metrics_no_AU.txt"

    with open(path) as f:
        all_lines = f.readlines()
    longs = []
    shorts = []
    k = 0
    for line in all_lines:
        xs = line.replace("|", "").split()
        if xs and xs[0] == "symbol":
            column_names = xs
        if xs == ["long"]:
            k = 1
        elif xs == ["short"]:
            k = 2
        if k == 1:
            if xs and "USDT" in xs[0]:
                longs.append(floatify(xs))
        if k == 2:
            if xs and "USDT" in xs[0]:
                shorts.append(floatify(xs))
    longs = pd.DataFrame(longs, columns=column_names)
    shorts = pd.DataFrame(shorts, columns=column_names)
    return longs, shorts


async def load_min_costs_single(exchange):
    results = []
    print(f"fetching info for {exchange}")
    cc = getattr(ccxt, exchange)()
    cc.options["defaultType"] = "swap"
    min_costs = {"bitget": 5.0, "bingx": 2.0, "hyperliquid": 10.0}
    quotes = {x: "USDT" for x in ["binanceusdm", "bybit", "okx", "bitget", "bingx"]}
    quotes["hyperliquid"] = "USDC"
    markets = await cc.load_markets()
    if exchange == "hyperliquid":
        coin2symbol_map = {markets[symbol]["info"]["name"]: symbol for symbol in markets}
        fetched = await cc.fetch(
            "https://api.hyperliquid.xyz/info",
            method="POST",
            headers={"Content-Type": "application/json"},
            body=json.dumps({"type": "allMids"}),
        )
        tickers = {
            coin2symbol_map[coin]: {
                "bid": float(fetched[coin]),
                "ask": float(fetched[coin]),
                "last": float(fetched[coin]),
            }
            for coin in coin2symbol_map
        }
    else:
        tickers = await cc.fetch_tickers()
    first_timestamps = await get_first_ohlcv_timestamps(cc)
    for symbol in markets:
        if not symbol.endswith(f"/{quotes[exchange]}:{quotes[exchange]}"):
            continue
        if not markets[symbol]["swap"] or not markets[symbol]["active"]:
            continue
        if symbol not in first_timestamps:
            continue
        if markets[symbol]["limits"]["cost"]["min"] is None:
            if exchange in min_costs:
                min_cost = min_costs[exchange]
            else:
                min_cost = 0.0
        else:
            min_cost = markets[symbol]["limits"]["cost"]["min"]
        if exchange == "hyperliquid":
            min_qty = markets[symbol]["precision"]["amount"]
        else:
            min_qty = markets[symbol]["limits"]["amount"]["min"]
        c_mult = markets[symbol]["contractSize"]
        results.append(
            {
                "exchange": cc.id,
                "symbol": symbol,
                "min_qty": min_qty,
                "min_cost": min_cost,
                "last_price": (last_price := tickers[symbol]["last"]),
                "c_mult": c_mult,
                "effective_min_cost": max(min_cost, min_qty * last_price * c_mult),
                "first_timestamp": first_timestamps[symbol],
            }
        )
    return results


async def load_min_costs(exchanges=["binanceusdm", "bybit", "okx", "bitget", "bingx", "hyperliquid"]):
    today = ts_to_date_utc(utc_ms())[:10]
    filepath = make_get_filepath(f"caches/min_costs_{today}.csv")
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    results = []
    for exchange in exchanges:
        results_single = await load_min_costs_single(exchange)
        results += results_single
    print(f"writing to {filepath}")
    mdf = pd.DataFrame(results)
    mdf.to_csv(filepath)
    return mdf


def load_live_configs():
    path = "configs/live/multisymbol/no_AU/"
    configs = {}
    for f in os.listdir(path):
        try:
            symbol = f[:-5]
            configs[symbol] = load_live_config(os.path.join(path, f))
        except Exception as e:
            print("error", f, e)
    return configs


def generate_hjson_config(user, TWE_limit_long, long_enabled, symbols):
    template = hjson.load(open("configs/live/example_config.hjson"))
    template["user"] = user
    template["TWE_long"] = TWE_limit_long
    template["long_enabled"] = long_enabled
    template["symbols"] = sorted(set(symbols))
    today = ts_to_date_utc(utc_ms())[:19].replace(":", "_")
    filepath = make_get_filepath(f"configs/live/live_config_{user}_{today}.hjson")
    hjson.dump(template, open(filepath, "w"))
    print("dumped", filepath)


async def main():
    parser = argparse.ArgumentParser(
        prog="filter_coins",
        description="filter coins based on exchange, balance, WE_limit and n_syms",
    )
    parser.add_argument(
        "-e",
        "--exchange",
        type=str,
        required=False,
        dest="exchange",
        default="bybit",
        help="exchange (default bybit; choices [binance, bybit, okx, bitget, bingx, hyperliquid])",
    )
    parser.add_argument(
        "-bs",
        "--backtested-since",
        "--backtested_since",
        type=str,
        required=False,
        dest="backtested_since",
        default="2021-05-10",
        help="backtested since (default '2021-05-10')",
    )
    parser.add_argument(
        "-nc",
        type=int,
        required=False,
        dest="n_coins",
        default=8,
        help="n coins (default 8)",
    )
    parser.add_argument(
        "-b",
        "--balance",
        type=float,
        required=False,
        dest="balance",
        default=1000.0,
        help="specify balance (default 1000.0)",
    )
    parser.add_argument(
        "-tl",
        "--total-wallet-exposure-long",
        "--total_wallet_exposure_long",
        type=float,
        required=False,
        dest="total_wallet_exposure_long",
        default=2.0,
        help="specify total wallet exposure long (default 2.0)",
    )
    parser.add_argument(
        "-ts",
        "--total-wallet-exposure-short",
        "--total_wallet_exposure_short",
        type=float,
        required=False,
        dest="total_wallet_exposure_short",
        default=2.0,
        help="specify total wallet exposure short (default 2.0)",
    )
    parser.add_argument(
        "-le",
        "--long_enabled",
        "--long-enabled",
        type=str2bool,
        required=False,
        dest="long_enabled",
        default=True,
        help="specify long_enabled (y/n or t/f)",
    )
    parser.add_argument(
        "-se",
        "--short_enabled",
        "--short-enabled",
        type=str2bool,
        required=False,
        dest="short_enabled",
        default=True,
        help="specify short_enabled (y/n or t/f)",
    )
    parser.add_argument(
        "-mi",
        "--min_initial_qty_pct",
        "--min-initial-qty-pct",
        type=float,
        required=False,
        dest="min_initial_qty_pct",
        default=0.005,
        help="specify min initial entry qty pct (default 0.005)",
    )
    parser.add_argument(
        "-u",
        "--user",
        type=str,
        required=False,
        dest="user",
        default=None,
        help="If given, will generate and dump live hjson config for user.",
    )
    args = parser.parse_args()

    exchanges = ["binanceusdm", "bybit", "okx", "bitget", "bingx", "hyperliquid"]
    exchange = args.exchange
    if "binance" in exchange:
        exchange = "binanceusdm"
    assert exchange in exchanges, f"unknown exchange {exchange}"
    backtested_since = args.backtested_since
    n_coins = args.n_coins
    balance = args.balance
    TWE_limit_long = args.total_wallet_exposure_long
    TWE_limit_short = args.total_wallet_exposure_short
    long_enabled = args.long_enabled
    short_enabled = args.short_enabled
    initial_qty_pct_lower_bound = args.min_initial_qty_pct

    minimum_coin_backtest_age_days = (utc_ms() - date2ts_utc(backtested_since)) / (
        1000 * 60 * 60 * 24
    )

    longs, shorts = parse_backtest_metrics()
    min_costs = await load_min_costs()
    configs = load_live_configs()

    longs.loc[:, "symbol"] = longs.symbol.apply(symbol2coin)
    shorts.loc[:, "symbol"] = shorts.symbol.apply(symbol2coin)
    min_costs.loc[:, "symbol"] = min_costs.symbol.apply(symbol2coin)
    configs = {symbol2coin(k): v for k, v in configs.items()}

    min_cost_lower_bound = min_costs[min_costs.exchange == exchange].effective_min_cost.median()
    min_cost_upper_bound = balance * TWE_limit_long * initial_qty_pct_lower_bound / n_coins

    if min_cost_upper_bound < min_cost_lower_bound:
        print("\n" + "#" * 40)
        print(f"Median effective_min_cost for {exchange} is {min_cost_lower_bound}.")
        print(
            f"On {exchange} with balance {balance}, TWE limit long {TWE_limit_long} and initial qty pct {initial_qty_pct_lower_bound},"
        )
        print(
            f"minimum cost per long position is balance * TWE_limit_long * initial_qty_pct_lower_bound / n_coins == {min_cost_upper_bound},"
        )
        print(
            f"which is too low.",
        )
        while n_coins > 1 and min_cost_upper_bound < min_cost_lower_bound:
            n_coins -= 1
            min_cost_upper_bound = balance * TWE_limit_long * initial_qty_pct_lower_bound / n_coins
        if min_cost_upper_bound < min_cost_lower_bound:
            print(f"Balance is too low, even with only one coin.")
            print(f"Try again with higher balance and/or higher TWE and/or on another exchange.")
            print("#" * 40 + "\n")
            return
        print(
            f"Reducing n_coins to {n_coins}. Now balance * TWE_limit_long * initial_qty_pct_lower_bound / n_coins == {min_cost_upper_bound}."
        )
        print()
        print("#" * 40 + "\n")

    first_timestamps = min_costs.groupby("symbol").first_timestamp.min()

    longs = longs.sort_values("symbol").reset_index()
    shorts = shorts.sort_values("symbol").reset_index()

    scores = (
        longs.adg_w_per_exp + longs.adg_per_exp + shorts.adg_w_per_exp + shorts.adg_per_exp
    ) / 4.0
    longs.loc[:, "score"] = scores
    shorts.loc[:, "score"] = scores

    print(f"{'balance': <30} {balance}")
    print(f"{'exchange': <30} {exchange}")
    print(f"{'TWE_limit_long': <30} {TWE_limit_long}")
    print(f"{'TWE_limit_short': <30} {TWE_limit_short}")
    print(f"{'long_enabled': <30} {long_enabled}")
    print(f"{'short_enabled': <30} {short_enabled}")
    print(f"{'n_coins': <30} {n_coins}")
    print(f"{'backtested_since': <30} {backtested_since}")
    print(f"{'minimum_coin_backtest_age_days': <30} {minimum_coin_backtest_age_days}")
    print(f"{'initial_qty_pct_lower_bound': <30} {initial_qty_pct_lower_bound}")
    print(f"{'min_cost_upper_bound': <30} {min_cost_upper_bound}")

    eligible_coins = set(
        min_costs[
            (min_costs.exchange == exchange) & (min_costs.effective_min_cost <= min_cost_upper_bound)
        ].symbol
    )
    longs_filtered = longs[
        (longs.symbol.isin(eligible_coins)) & (longs.n_days >= minimum_coin_backtest_age_days)
    ].sort_values("score", ascending=False)
    shorts_filtered = shorts[
        (shorts.symbol.isin(eligible_coins)) & (shorts.n_days >= minimum_coin_backtest_age_days)
    ].sort_values("score", ascending=False)

    longs_filtered.loc[:, "first_timestamp"] = [
        first_timestamps[symbol] if symbol in first_timestamps else -1
        for symbol in longs_filtered.symbol.values
    ]
    longs_filtered.loc[:, "first_date"] = longs_filtered.first_timestamp.apply(ts_to_date_utc)
    longs_filtered = longs_filtered.iloc[:n_coins]
    columns_to_print = [
        "symbol",
        "adg_w_per_exp",
        "adg_per_exp",
        "hrs_stuck_max",
        "pa_dist_1pct_worst_mean",
        "loss_profit_rt",
        "drawdown_1pct_worst_mean",
        "drawdown_max",
        "n_days",
        "first_timestamp",
        "first_date",
    ]
    symbols = longs_filtered.symbol.values
    longs_filtered = longs_filtered[columns_to_print]
    print(longs_filtered)
    if args.user is not None:
        generate_hjson_config(args.user, TWE_limit_long, long_enabled, symbols)


if __name__ == "__main__":
    asyncio.run(main())
