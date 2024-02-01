import pandas as pd
import pprint
import argparse
import ccxt
import numpy as np
import argparse

import os

os.environ["NOJIT"] = "true"

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from forager import get_min_costs_and_contract_multipliers
from procedures import load_live_config, utc_ms, make_get_filepath
from pure_funcs import floatify, ts_to_date_utc, date2ts_utc, str2bool


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


def load_min_costs():
    today = ts_to_date_utc(utc_ms())[:10]
    filepath = make_get_filepath(f"caches/min_costs_{today}.csv")
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    exchanges = ["binanceusdm", "bybit", "okx", "bitget"]
    min_costs = {"bitget": 5.0}
    results = []
    for exchange in exchanges:
        print(f"fetching info for {exchange}")
        cc = getattr(ccxt, exchange)()
        cc.options["defaultType"] = "swap"
        markets = cc.load_markets()
        tickers = cc.fetch_tickers()
        for symbol in markets:
            if not symbol.endswith("/USDT:USDT"):
                continue
            if not markets[symbol]["swap"] or not markets[symbol]["active"]:
                continue
            if markets[symbol]["limits"]["cost"]["min"] is None:
                if exchange in min_costs:
                    min_cost = min_costs[exchange]
                else:
                    min_cost = 0.0
            else:
                min_cost = markets[symbol]["limits"]["cost"]["min"]
            results.append(
                {
                    "exchange": cc.id,
                    "symbol": symbol,
                    "min_qty": (min_qty := markets[symbol]["limits"]["amount"]["min"]),
                    "min_cost": min_cost,
                    "last_price": (last_price := tickers[symbol]["last"]),
                    "c_mult": (c_mult := markets[symbol]["contractSize"]),
                    "effective_min_cost": max(min_cost, min_qty * last_price * c_mult),
                    "first_timestamp": markets[symbol]["created"],
                }
            )

    print(f"writing to {filepath}")
    mdf = pd.DataFrame(results)
    mdf.to_csv(filepath)
    return mdf


def load_live_configs():
    path = 'configs/live/multisymbol/no_AU/'
    configs = {}
    for f in os.listdir(path):
        try:
            symbol = f[:-5]
            configs[symbol] = load_live_config(os.path.join(path, f))
        except Exception as e:
            print('error', f, e)
    return configs


def sname(symbol):
    coin = symbol.replace("/USDT:USDT", "")
    if coin.endswith("USDT"):
        coin = coin[:-4]
    coin = coin.replace('1000', '')
    return coin + "USDT"



def main():
    parser = argparse.ArgumentParser(prog="filter_coins", description="filter coins based on exchange, balance, WE_limit and n_syms")
    parser.add_argument(
        "-e",
        "--exchange",
        type=str,
        required=False,
        dest="exchange",
        default='bybit',
        help="exchange (default bybit; choices [binance, bybit, okx, bitget])",
    )
    parser.add_argument(
        "-bs",
        "--backtested-since",
        "--backtested_since",
        type=str,
        required=False,
        dest="backtested_since",
        default='2021-05-10',
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
    args = parser.parse_args()


    exchange = args.exchange
    backtested_since = args.backtested_since
    n_coins = args.n_coins
    balance = args.balance
    TWE_limit_long = args.total_wallet_exposure_long
    TWE_limit_short = args.total_wallet_exposure_short
    long_enabled = args.long_enabled
    short_enabled = args.short_enabled
    initial_qty_pct_lower_bound = args.min_initial_qty_pct

    minimum_coin_backtest_age_days = (utc_ms() - date2ts_utc(backtested_since)) / (1000 * 60 * 60 * 24)

    if 'binance' in exchange:
        exchange = 'binanceusdm'


    min_cost_upper_bound = balance * TWE_limit_long * initial_qty_pct_lower_bound / n_coins

    longs, shorts = parse_backtest_metrics()
    min_costs = load_min_costs()
    configs = load_live_configs()


    min_costs.loc[:,'symbol'] = min_costs.symbol.apply(sname)
    longs.loc[:,'symbol'] = longs.symbol.apply(sname)
    shorts.loc[:,'symbol'] = shorts.symbol.apply(sname)
    configs = {sname(k): v for k, v in configs.items()}

    longs = longs.sort_values('symbol').reset_index()
    shorts = shorts.sort_values('symbol').reset_index()

    scores = longs.adg_w_per_exp + longs.adg_per_exp + shorts.adg_w_per_exp + shorts.adg_per_exp
    scores /= 4.0
    longs.loc[:,'score'] = scores
    shorts.loc[:,'score'] = scores

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


    eligible_coins = set(min_costs[(min_costs.exchange == exchange) & (min_costs.effective_min_cost <= min_cost_upper_bound)].symbol)
    longs_filtered = longs[(longs.symbol.isin(eligible_coins)) & (longs.n_days >= minimum_coin_backtest_age_days)].sort_values('score', ascending=False)
    shorts_filtered = shorts[(shorts.symbol.isin(eligible_coins)) & (shorts.n_days >= minimum_coin_backtest_age_days)].sort_values('score', ascending=False)
    print(longs_filtered.iloc[:n_coins])




if __name__ == "__main__":
    main()
