import requests
import json
import argparse
import sys
import os
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pure_funcs import calc_hash, symbol_to_coin, ts_to_date_utc
from procedures import utc_ms


def is_stablecoin(elm):
    try:
        if elm["symbol"] in ["tether", "usdb", "usdy", "tusd", "usd0", "usde"]:
            return True
        if (
            all([abs(elm[k] - 1.0) < 0.01 for k in ["high_24h", "low_24h", "current_price"]])
            and abs(elm["price_change_24h"]) < 0.01
        ):
            return True
        return False
    except Exception as e:
        print(f"error with is_stablecoin {elm} {e}")
        traceback.print_exc()
        return False


def get_top_market_caps(n_coins, minimum_market_cap_millions, exchange=None):
    # Fetch the top N coins by market cap
    markets_url = "https://api.coingecko.com/api/v3/coins/markets"
    per_page = 150
    page = 1
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": per_page,
        "page": 1,
        "sparkline": "false",
    }
    minimum_market_cap = minimum_market_cap_millions * 1e6
    approved_coins = {}
    prev_hash = None
    exchange_approved_coins = None
    if exchange is not None:
        exchanges = exchange.split(",")
        import ccxt

        exchange_map = {
            "bybit": ("bybit", "USDT"),
            "binance": ("binanceusdm", "USDT"),
            "bitget": ("bitget", "USDT"),
            "hyperliquid": ("hyperliquid", "USDC"),
            "gateio": ("gateio", "USDT"),
            "okx": ("okx", "USDT"),
        }
        exchange_approved_coins = set()
        for exchange in exchanges:
            try:
                cc = getattr(ccxt, exchange_map[exchange][0])()
                cc.options["defaultType"] = "swap"
                markets = cc.fetch_markets()
                for elm in markets:
                    if (
                        elm["swap"]
                        and elm["active"]
                        and elm["symbol"].endswith(f":{exchange_map[exchange][1]}")
                    ):
                        exchange_approved_coins.add(symbol_to_coin(elm["symbol"]))
                print(f"Added coin filter for {exchange}")
            except Exception as e:
                print(f"error loading ccxt for {exchange} {e}")
    while len(approved_coins) < n_coins:
        response = requests.get(markets_url, params=params)
        if response.status_code != 200:
            print(f"Error fetching market data: {response.status_code} - {response.text}")
            break
        market_data = response.json()
        new_hash = calc_hash(market_data)
        if new_hash == prev_hash:
            break
        prev_hash = new_hash
        added = []
        disapproved = {}
        for elm in market_data:
            circulating = elm.get("circulating_supply") or 0.0
            total = elm.get("total_supply") or elm.get("max_supply") or 1.0  # Avoid divide-by-zero
            price = elm.get("current_price") or 0.0
            mcap = circulating * price
            supply_ratio = circulating / total if total > 0 else 0.0
            penalized_mcap = mcap * supply_ratio  # downweight based on concentration
            elm["supply_ratio"] = supply_ratio
            elm["penalized_mcap"] = penalized_mcap
            elm["liquidity_ratio"] = elm["total_volume"] / elm["market_cap"]

            coin = elm["symbol"].upper()
            if len(approved_coins) >= n_coins:
                print(f"N coins == {n_coins}")
                if added:
                    print(f"Added approved coins {','.join(added)}")
                return approved_coins
            if elm["market_cap"] < minimum_market_cap:
                print("Lowest market cap", coin)
                if added:
                    print(f"Added approved coins {','.join(added)}")
                return approved_coins
            if is_stablecoin(elm):
                disapproved[coin] = "stablecoin"
                continue
            if exchange_approved_coins is not None and coin not in exchange_approved_coins:
                disapproved[coin] = "not_active"
                continue
            if coin not in approved_coins:
                approved_coins[coin] = elm
                added.append(coin)
        print(f"added approved coins {','.join(added)}")
        if disapproved:
            for key in set(disapproved.values()):
                to_print = [c for c in disapproved if disapproved[c] == key]
                print(f"disapproved {key} {','.join(to_print)}")
        disapproved = {}
        if len(approved_coins) >= n_coins:
            break
        params["page"] += 1
    return approved_coins


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="mcap generator", description="generate_mcap_list")
    parser.add_argument(
        f"--n_coins",
        f"-n",
        type=int,
        dest="n_coins",
        required=False,
        default=100,
        help=f"Maxiumum number of top market cap coins. Default=100",
    )
    parser.add_argument(
        f"--minimum_market_cap_dollars",
        f"-m",
        type=float,
        dest="minimum_market_cap_millions",
        required=False,
        default=300.0,
        help=f"Minimum market cap in millions of USD. Default=300.0",
    )
    parser.add_argument(
        f"--exchange",
        f"-e",
        type=str,
        dest="exchange",
        required=False,
        default=None,
        help=f"Optional: filter by coins available on exchange. Comma separated values. Default=None",
    )
    parser.add_argument(
        f"--output",
        f"-o",
        type=str,
        dest="output",
        required=False,
        default=None,
        help="Optional: Output path. Default=configs/approved_coins_{n_coins}_{min_mcap}.json",
    )
    args = parser.parse_args()

    market_caps = get_top_market_caps(args.n_coins, args.minimum_market_cap_millions, args.exchange)
    if args.output is None:
        fname = f"configs/approved_coins_{ts_to_date_utc(utc_ms())[:10]}"
        fname += f"_{args.n_coins}_coins_{int(args.minimum_market_cap_millions)}_min_mcap"
        if args.exchange is not None:
            fname += "_" + "_".join(args.exchange.split(","))
        fname += ".json"
    else:
        fname = args.output
    print(f"Dumping output to {fname}")
    json.dump(market_caps, open(fname.replace(".json", "_full.json"), "w"))
    json.dump(list(market_caps), open(fname, "w"))
