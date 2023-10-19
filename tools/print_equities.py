import os

os.environ["NOJIT"] = "true"

import ccxt.async_support as ccxt
import asyncio
import pprint
import time
import json
import argparse
import pandas as pd

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from procedures import (
    load_exchange_key_secret_passphrase,
    create_binance_bot,
    load_live_config,
    utc_ms,
)
from pure_funcs import determine_passivbot_mode, date_to_ts2, ts_to_date_utc, floatify
from njit_funcs import round_dynamic, calc_diff

pd.set_option("display.float_format", lambda x: "%.5f" % x)
MS2DAY = 1000 * 60 * 60 * 24


def multi_replace(input_data, replacements: [(str, str)]):
    if isinstance(input_data, str):
        new_data = input_data
        for old, new in replacements:
            new_data = new_data.replace(old, new)
    elif isinstance(input_data, list):
        new_data = []
        for string in input_data:
            for old, new in replacements:
                string = string.replace(old, new)
            new_data.append(string)
    elif isinstance(input_data, dict):
        new_data = {}
        for key, string in input_data.items():
            for old, new in replacements:
                string = string.replace(old, new)
            new_data[key] = string
    return new_data


def flatten_dict(d: dict, separator="_", parent_key=""):
    items = []
    for key, value in d.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, separator, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


def calc_pprice_diff(elm):
    if elm["position_side"] == "long":
        return elm["pprice"] / elm["mark_price"] - 1
    else:
        return elm["mark_price"] / elm["pprice"] - 1


async def main():
    exchange_map = {
        "kucoin": "kucoinfutures",
        "okx": "okx",
        "bybit": "bybit",
        "binance": "binanceusdm",
        "bitget": "bitget",
    }

    parser = argparse.ArgumentParser(prog="print bals", description="print bals")
    parser.add_argument("users", type=str, help="comma separated")
    parser.add_argument(
        "-nd",
        "--n_days",
        "--n-days",
        type=float,
        required=False,
        dest="n_days",
        default=None,
        help="n days lookback",
    )
    args = parser.parse_args()
    users = args.users.split(",")
    now = utc_ms()
    all_data = {"timestamp": now, "date": ts_to_date_utc(now), "users": {}}
    tickers = {}
    log_path = "logs/equities_log.txt"
    for user in users:
        data = {}
        exchange, key, secret, passphrase = load_exchange_key_secret_passphrase(user)
        data["exchange"] = exchange
        cc = getattr(ccxt, exchange_map[exchange])(
            {"apiKey": key, "secret": secret, "password": passphrase}
        )
        if exchange not in tickers:
            tickers[exchange] = await cc.fetch_tickers()
            for key in sorted(tickers[exchange]):
                nkey = key + ":USDT"
                if nkey not in tickers[exchange]:
                    tickers[exchange][nkey] = tickers[exchange][key]
        if exchange == "bitget":
            bal = await cc.private_mix_get_account_accounts({"productType": "umcbl"})
            for elm in bal["data"]:
                if elm["marginCoin"] == "USDT":
                    bal = elm
                    break
        else:
            bal = await cc.fetch_balance()

        if exchange == "binance":
            data["balance"] = float(bal["info"]["totalWalletBalance"])
            data["upnl"] = float(bal["info"]["totalCrossUnPnl"])
        elif exchange == "bybit":
            data["balance"] = bal["total"]["USDT"]
            for elm in bal["info"]["result"]["list"][0]["coin"]:
                if elm["coin"] == "USDT":
                    break
            data["upnl"] = float(elm["unrealisedPnl"])
        elif exchange == "bitget":
            data["balance"] = float(bal["available"])
            data["upnl"] = float(bal["unrealizedPL"])
        else:
            data["balance"] = bal["total"]["USDT"]
            data["upnl"] = 0.0
        pos = await cc.fetch_positions()
        data["positions"] = [
            {
                "symbol": x["symbol"],
                "psize": x["contracts"],
                "pprice": x["entryPrice"],
                "upnl": 0.0 if x["unrealizedPnl"] is None else x["unrealizedPnl"],
                "position_side": x["side"],
                "mark_price": (mark_price := tickers[exchange][x["symbol"]]["last"]),
                "pcost": (pcost := x["contracts"] * x["entryPrice"]),
                "wallet_exposure": pcost / data["balance"],
            }
            for x in pos
            if x["contracts"] != 0.0
        ]
        all_data["users"][user] = data.copy()
        try:
            await cc.close()
        except:
            pass
        print(f"fetched data for {user}")
    try:
        with open(log_path, "a") as f:
            f.write(f"{json.dumps(all_data)}\n")
    except Exception as e:
        print("failed writing log", e)

    try:
        with open(log_path) as f:
            lines = f.readlines()
    except Exception as e:
        print("failed reading log", e)
    jsons = [json.loads(line) for line in lines]
    if args.n_days is not None:
        jsons = [x for x in jsons if x['timestamp'] > now - 1000 * 60 * 60 * 24 * args.n_days]

    dfts = []
    first = None
    for x in jsons:
        user_data = {}
        for user in users:
            dat = x["users"][user]
            df = pd.DataFrame(dat["positions"])
            dat["upnl_pct"] = dat["upnl"] / dat["balance"]
            dat["WE_long"] = df[df.position_side == "long"].wallet_exposure.sum()
            dat["WE_short"] = df[df.position_side == "short"].wallet_exposure.sum()
            dat["pcost_long"] = df[df.position_side == "long"].pcost.sum()
            dat["pcost_short"] = df[df.position_side == "short"].pcost.sum()
            dat["upnl_long"] = df[df.position_side == "long"].upnl.sum()
            dat["upnl_short"] = df[df.position_side == "short"].upnl.sum()
            dat["upnl_pct_long"] = dat["upnl_long"] / dat["balance"]
            dat["upnl_pct_short"] = dat["upnl_short"] / dat["balance"]
            # dat = {k: v for k, v in dat.items() if k not in ["positions"]}
            user_data[user] = dat
        df = pd.DataFrame(user_data)
        if first is None:
            first = df.copy()
        dft = df.sum(axis=1)
        dft["WE_long"] = dft["pcost_long"] / dft["balance"]
        dft["WE_short"] = dft["pcost_short"] / dft["balance"]
        dft["upnl_pct_long"] = dft["upnl_long"] / dft["balance"]
        dft["upnl_pct_short"] = dft["upnl_short"] / dft["balance"]
        dft["upnl_pct"] = dft["upnl"] / dft["balance"]
        dft = dft.drop(["exchange"], axis=0)
        dft["date"] = x["date"]
        dft["timestamp"] = x["timestamp"]
        dfts.append(dft)
    for user in user_data:
        print(user)
        df = pd.DataFrame(user_data[user]["positions"])
        df.loc[:, "pprice_diff"] = df.apply(calc_pprice_diff, axis=1)
        print(df.sort_values("pprice_diff", ascending=False))
        print()
    df = pd.DataFrame(user_data).drop("positions")
    df.loc["n_days"] = (jsons[-1]["timestamp"] - jsons[0]["timestamp"]) / MS2DAY
    df.loc["abs_gain"] = (df.T.balance - first.T.balance).T
    df.loc["gain_pct"] = (((df.T.balance / first.T.balance) - 1) * 100).T
    df.loc["adg_pct"] = (((df.T.balance / first.T.balance) ** (1 / df.T.n_days) - 1) * 100).T
    print(df)
    print()
    df = pd.DataFrame(dfts).drop("positions", axis=1)
    df.date = pd.to_datetime(df.date).round("h")
    df.loc[:, "n_days"] = (df.timestamp - df.iloc[0].timestamp) / MS2DAY
    df.loc[:, "abs_gain"] = df.balance - df.balance.iloc[0]
    df.loc[:, "gain_pct"] = (df.balance / df.balance.iloc[0] - 1) * 100
    df.loc[:, "adg_pct"] = ((df.balance / df.balance.iloc[0]) ** (1 / df.n_days) - 1) * 100
    print(
        df.groupby(df.timestamp // MS2DAY)
        .last()
        .drop(["timestamp", "pcost_long", "pcost_short", "upnl_long", "upnl_short"], axis=1)
        .set_index("date")
    )


if __name__ == "__main__":
    asyncio.run(main())
