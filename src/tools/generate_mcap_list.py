import requests
import json
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pure_funcs import calc_hash


def is_stablecoin(elm):
    if elm["symbol"] in ["tether", "usdb", "usdy", "tusd"]:
        return True
    if (
        all([abs(elm[k] - 1.0) < 0.01 for k in ["high_24h", "low_24h", "current_price"]])
        and abs(elm["price_change_24h"]) < 0.01
    ):
        return True
    return False


def get_top_market_caps(n_coins, minimum_market_cap_millions):
    # Fetch the top N coins by market cap
    markets_url = "https://api.coingecko.com/api/v3/coins/markets"
    per_page = 100
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
        for elm in market_data:
            if len(approved_coins) >= n_coins:
                print(f"N coins == {n_coins}")
                if added:
                    print(f"added approved coins {','.join(added)}")
                return approved_coins
            if elm["market_cap"] < minimum_market_cap:
                print("Lowest market cap", elm)
                break
            if is_stablecoin(elm):
                print("is_stablecoin", elm["symbol"].upper())
                continue
            if (coin := elm["symbol"].upper()) not in approved_coins:
                approved_coins[coin] = elm
                added.append(coin)
        print(f"added approved coins {','.join(added)}")
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
        help=f"Maxiumum number of top market cap coins",
    )
    parser.add_argument(
        f"--minimum_market_cap_dollars",
        f"-m",
        type=float,
        dest="minimum_market_cap_millions",
        required=False,
        default=300.0,
        help=f"Minimum market cap in millions of USD",
    )
    args = parser.parse_args()

    market_caps = get_top_market_caps(args.n_coins, args.minimum_market_cap_millions)
    json.dump(list(market_caps), open("configs/approved_coins.json", "w"))
