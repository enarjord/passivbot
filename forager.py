import os

os.environ["NOJIT"] = "true"

import ccxt.async_support as ccxt
import json
import hjson
import pprint
import numpy as np
import asyncio
import time
import subprocess
import argparse
import traceback
from procedures import load_exchange_key_secret_passphrase, utc_ms


def volatility(ohlcv):
    closes = np.array(ohlcv)[:, 4]
    return closes.std() / closes.mean()


def generate_yaml(
    vols,
    approved,
    config,
    current_positions_long,
    current_positions_short,
    current_open_orders_long,
    current_open_orders_short,
):
    yaml = f"session_name: {config['user']}\nwindows:\n"
    user = config["user"]
    twe_long = config["twe_long"]
    twe_short = config["twe_short"]

    n_longs = config["n_longs"]
    n_shorts = config["n_shorts"]
    lw = round(twe_long / n_longs, 4) if n_longs > 0 else 0.1
    sw = round(twe_short / n_shorts, 4) if n_shorts > 0 else 0.1
    lm, sm = "gs", "gs"
    lev = 10 if config["leverage"] is None else config["leverage"]
    price_distance_threshold = (
        config["price_distance_threshold"]
        if "price_distance_threshold" in config and config["price_distance_threshold"] is not None
        else 0.5
    )
    syms_sorted_by_volatility = [
        x[0] for x in sorted(vols.items(), key=lambda x: x[1], reverse=True) if x[0] in approved
    ]
    ideal_longs = syms_sorted_by_volatility[:n_longs]
    print("ideal_longs", ideal_longs)
    ideal_shorts = syms_sorted_by_volatility[:n_shorts]
    print("ideal_shorts", ideal_shorts)
    current_positions_long = sorted(set(current_positions_long + current_open_orders_long))
    current_positions_short = sorted(set(current_positions_short + current_open_orders_short))
    longs_on_gs = [x for x in current_positions_long if x not in ideal_longs]
    print("longs_on_gs", longs_on_gs)
    shorts_on_gs = [x for x in current_positions_short if x not in ideal_shorts]
    print("shorts_on_gs", shorts_on_gs)
    active_longs = ideal_longs[: max(0, n_longs - len(longs_on_gs))]
    print("active_longs", active_longs)
    active_shorts = ideal_shorts[: max(0, n_shorts - len(shorts_on_gs))]
    print("active_shorts", active_shorts)
    active_bots, bots_on_gs = [], []
    for sym in sorted(set(active_longs + active_shorts + current_positions_long + current_positions_short)):
        elm = (sym, sym in active_longs, sym in active_shorts)
        if elm[1] or elm[2]:
            active_bots.append(elm)
        else:
            bots_on_gs.append(elm)
    max_n_panes = config["max_n_panes"] if "max_n_panes" in config else 8
    for z in range(0, len(active_bots), max_n_panes):
        active_bots_slice = active_bots[z : z + max_n_panes]
        yaml += f"- window_name: {config['user']}_normal_{z}\n  layout: "
        yaml += f"even-vertical\n  shell_command_before:\n    - cd ~/passivbot\n  panes:\n"
        for sym, long_enabled, short_enabled in active_bots_slice:
            lm = "n" if long_enabled and lw > 0.0 else "gs"
            sm = "n" if short_enabled and sw > 0.0 else "gs"
            conf_path = (
                config["live_configs_map"][sym] if sym in config["live_configs_map"] else config["default_config_path"]
            )
            pane = f"    - shell_command:\n      - python3 passivbot.py {user} {sym} {conf_path} "
            pane += f"-lw {lw} -sw {sw} -lm {lm} -sm {sm} -lev {lev} -cd -pt {price_distance_threshold}"
            yaml += pane + "\n"
    if bots_on_gs:
        for z in range(0, len(bots_on_gs), max_n_panes):
            bots_on_gs_slice = bots_on_gs[z : z + max_n_panes]
            yaml += (
                f"- window_name: {config['user']}_gs_{z}\n  layout: even-vertical\n  shell_command_before:\n    - cd ~/passivbot\n  panes:"
                + "\n"
            )
            gs_lw = lw if config["gs_lw"] is None else config["gs_lw"]
            gs_sw = lw if config["gs_sw"] is None else config["gs_sw"]
            for sym, _, _ in bots_on_gs_slice:
                conf_path = (
                    config["live_configs_map"][sym]
                    if sym in config["live_configs_map"]
                    else config["default_config_path"]
                )
                pane = f"    - shell_command:\n      - python3 passivbot.py {user} {sym} {conf_path} -lw {gs_lw} "
                pane += f"-sw {gs_sw} -lm gs -sm gs -lev {lev} -cd -pt {price_distance_threshold}"
                for k0, k1 in [("lmm", "gs_mm"), ("lmr", "gs_mr")]:
                    if config[k1] is not None:
                        pane += f" -{k0} {config[k1]}"
                yaml += pane + "\n"
            bots_on_gs_slice = bots_on_gs
    return yaml


async def get_ohlcvs(cc, symbols, config):
    ohs = {}
    n = 5
    if cc.id == "bybit":
        extra_args = {"since": int(utc_ms() - 1000 * 60 * 60 * 25)}
    else:
        extra_args = {}
    print("n syms", len(symbols))
    for i in range(0, len(symbols), n):
        js = list(range(i, min(len(symbols), i + n)))
        fetched = await asyncio.gather(
            *[cc.fetch_ohlcv(symbols[j], timeframe=config["ohlcv_interval"], **extra_args) for j in js]
        )
        print("fetching ohlcvs", [symbols[j] for j in js], f"{i}/{len(symbols)}")
        for k, j in enumerate(js):
            ohs[symbols[j]] = fetched[k]
    return ohs


async def get_current_symbols(cc):
    current_positions_long, current_positions_short = [], []
    current_open_orders = []
    poss = await cc.fetch_positions()
    for elm in poss:
        if elm["contracts"] != 0.0:
            if elm["side"] == "long":
                current_positions_long.append(elm["symbol"])
            elif elm["side"] == "short":
                current_positions_short.append(elm["symbol"])
    if cc.id == "bybit":
        oos = []
        delay_s = 0.5
        for symbol in cc.markets:
            if symbol.endswith("USDT"):
                sts = time.time()
                oosf = await cc.fetch_open_orders(symbol=symbol)
                spent = time.time() - sts
                print(f"\r fetching open orders for {symbol}     ", end=" ")
                oos += oosf
                time.sleep(max(0.0, delay_s - spent))
        print()
    elif cc.id == "binanceusdm":
        cc.options["warnOnFetchOpenOrdersWithoutSymbol"] = False
        oos = await cc.fetch_open_orders()
    else:
        oos = await cc.fetch_open_orders()
    current_open_orders_long, current_open_orders_short = [], []
    for elm in oos:
        if elm["side"] == "short":
            current_open_orders_short.append(elm["symbol"])
        else:
            current_open_orders_long.append(elm["symbol"])
    current_open_orders_long = sorted(set(current_open_orders_long))
    current_open_orders_short = sorted(set(current_open_orders_short))
    return current_positions_long, current_positions_short, current_open_orders_long, current_open_orders_short


async def get_min_costs(cc):
    exchange = cc.id
    info = await cc.fetch_markets()
    tickers = await cc.fetch_tickers()
    min_costs = {}
    for x in info:
        symbol = x["symbol"]
        if symbol.endswith("USDT"):
            if x["type"] != "spot":
                if exchange in ["okx", "bitget", "kucoinfutures"]:
                    ticker_symbol = symbol.replace(":USDT", "")
                else:
                    ticker_symbol = symbol
                if ticker_symbol in tickers:
                    if exchange == "bitget":
                        min_cost = 5.0
                        c_mult = 1.0
                        min_qty = float(x["info"]["minTradeNum"])
                    else:
                        min_cost = 0.0 if x["limits"]["cost"]["min"] is None else x["limits"]["cost"]["min"]
                        c_mult = 1.0 if x["contractSize"] is None else x["contractSize"]
                        min_qty = 0.0 if x["limits"]["amount"]["min"] is None else x["limits"]["amount"]["min"]
                    if exchange == "kucoinfutures":
                        last_price = float(tickers[ticker_symbol]["info"]["last"])
                        min_qty = float(x["info"]["multiplier"])
                        c_mult = 1.0
                    else:
                        last_price = tickers[ticker_symbol]["last"]
                    min_costs[symbol] = max(min_cost, min_qty * c_mult * last_price)
    return min_costs


async def dump_yaml(cc, config):
    max_min_cost = config["max_min_cost"]
    ohlcv_interval = config["ohlcv_interval"]
    n_ohlcvs = config["n_ohlcvs"]
    print("getting min costs...")
    min_costs = await get_min_costs(cc)
    symbols_map = {sym: sym.replace(":USDT", "").replace("/", "") for sym in min_costs}
    if config["approved_symbols_only"]:
        # only use approved symbols
        symbols_map = {k: v for k, v in symbols_map.items() if v in config["live_configs_map"]}
    symbols_map_inv = {v: k for k, v in symbols_map.items()}
    approved = [symbols_map[k] for k, v in min_costs.items() if v <= max_min_cost and k in symbols_map]
    print("getting current bots...")
    (
        current_positions_long,
        current_positions_short,
        current_open_orders_long,
        current_open_orders_short,
    ) = await get_current_symbols(cc)
    current_positions_long = [symbols_map[s] if s in symbols_map else s for s in current_positions_long]
    current_positions_short = [symbols_map[s] if s in symbols_map else s for s in current_positions_short]
    current_open_orders_long = [symbols_map[s] if s in symbols_map else s for s in current_open_orders_long]
    current_open_orders_short = [symbols_map[s] if s in symbols_map else s for s in current_open_orders_short]
    print("current_positions_long", current_positions_long)
    print("current_positions_short", current_positions_short)
    print("current_open_orders long", current_open_orders_long)
    print("current_open_orders short", current_open_orders_short)
    print("getting ohlcvs...")
    ohs = await get_ohlcvs(cc, [symbols_map_inv[sym] for sym in approved], config)
    vols = {symbols_map[sym]: volatility(ohs[sym][-n_ohlcvs:]) for sym in ohs}
    for elm in sorted(vols.items(), key=lambda x: x[1]):
        print(elm)
    print(f"generating yaml {config['yaml_filepath']}...")
    yaml = generate_yaml(
        vols,
        approved,
        config,
        current_positions_long,
        current_positions_short,
        current_open_orders_long,
        current_open_orders_short,
    )
    with open(config["yaml_filepath"], "w") as f:
        f.write(yaml)


async def main():
    exchange_map = {
        "kucoin": "kucoinfutures",
        "okx": "okx",
        "bybit": "bybit",
        "binance": "binanceusdm",
    }
    parser = argparse.ArgumentParser(prog="forager", description="start forager")
    parser.add_argument("forager_config_path", type=str, help="path to forager config")
    args = parser.parse_args()
    config = hjson.load(open(args.forager_config_path))
    config["yaml_filepath"] = f"{config['user']}.yaml"
    user = config["user"]
    exchange, key, secret, passphrase = load_exchange_key_secret_passphrase(config["user"])
    cc = getattr(ccxt, exchange_map[exchange])({"apiKey": key, "secret": secret, "password": passphrase})
    max_n_tries_per_hour = 5
    error_timestamps = []
    while True:
        try:
            await dump_yaml(cc, config)
            print("waiting one minute to avoid API rate limiting...")
            for i in range(60, -1, -1):
                time.sleep(1)
                print(f"\rcountdown: {i}    ", end=" ")
            print()
            subprocess.run(["tmux", "kill-session", "-t", config["user"]])
            subprocess.run(["tmuxp", "load", "-d", config["yaml_filepath"]])
            for i in range(3600, -1, -1):
                time.sleep(1)
                print(f"\rcountdown: {i}    ", end=" ")
            print()
        except Exception as e:
            traceback.print_exc()
            sleep_mins = 2
            print(f"error with forager {e} waiting {sleep_mins} minutes and trying again")
            for i in range(60 * sleep_mins, -1, -1):
                time.sleep(1)
                print(f"\rcountdown: {i}    ", end=" ")
            now = time.time()
            error_timestamps.append(now)
            error_timestamps = [x for x in error_timestamps if x > now - 60 * 60]
            if len(error_timestamps) > max_n_tries_per_hour:
                print(f"failed {max_n_tries_per_hour} times last hour; exiting")
                return
        finally:
            await cc.close()


if __name__ == "__main__":
    asyncio.run(main())
