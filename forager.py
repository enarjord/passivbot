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
from procedures import load_exchange_key_secret_passphrase, utc_ms


def volatility(ohlcv):
    closes = np.array(ohlcv)[:, 4]
    return closes.std() / closes.mean()


def generate_yaml(vols, approved, current, config):
    yaml = f"session_name: {config['user']}\nwindows:"
    k = 0
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
    active_syms = [
        x[0] for x in sorted(vols.items(), key=lambda x: x[1], reverse=True) if x[0] in approved
    ][: max(n_longs, n_shorts)]
    max_n_panes = config["max_n_panes"] if "max_n_panes" in config else 8
    for z in range(0, len(active_syms), max_n_panes):
        active_syms_slice = active_syms[z : z + max_n_panes]
        if active_syms_slice:
            yaml += f"\n- window_name: {config['user']}_normal_{z}\n  layout: "
            yaml += f"even-vertical\n  shell_command_before:\n    - cd ~/passivbot\n  panes:\n"
        for sym in active_syms_slice:
            k += 1
            lm = "n" if k <= n_longs else "gs"
            sm = "n" if k <= n_shorts else "gs"
            if lm == "gs" and sm == "gs":
                break
            conf_path = (
                config["live_configs_map"][sym]
                if sym in config["live_configs_map"]
                else config["default_config_path"]
            )
            pane = f"    - shell_command:\n      - python3 passivbot.py {user} {sym} {conf_path} "
            pane += (
                f"-lw {lw} -sw {sw} -lm {lm} -sm {sm} -lev {lev} -cd -pt {price_distance_threshold}"
            )
            yaml += pane + "\n"
    bots_on_gs = sorted(set([sym for sym in current if sym not in active_syms]))
    if bots_on_gs:
        for z in range(0, len(bots_on_gs), max_n_panes):
            bots_on_gs_slice = bots_on_gs[z : z + max_n_panes]
            yaml += (
                f"- window_name: {config['user']}_gs_{z}\n  layout: even-vertical\n  shell_command_before:\n    - cd ~/passivbot\n  panes:"
                + "\n"
            )
            gs_lw = lw if config["gs_lw"] is None else config["gs_lw"]
            gs_sw = lw if config["gs_sw"] is None else config["gs_sw"]
            for sym in bots_on_gs_slice:
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
    print("active bots:", active_syms)
    print("bots on -gs:", bots_on_gs)
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
            *[
                cc.fetch_ohlcv(symbols[j], timeframe=config["ohlcv_interval"], **extra_args)
                for j in js
            ]
        )
        print("fetching ohlcvs", [symbols[j] for j in js], f"{i}/{len(symbols)}")
        for k, j in enumerate(js):
            ohs[symbols[j]] = fetched[k]
    return ohs


async def get_current_symbols(cc):
    poss = await cc.fetch_positions()
    if cc.id == "bybit":
        pos_syms = [elm["info"]["symbol"] for elm in poss if float(elm["info"]["size"]) != 0.0]
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
        oos_syms = sorted(set([elm["symbol"] for elm in oos]))
        return sorted(set(pos_syms + oos_syms))
    elif cc.id == "binanceusdm":
        cc.options["warnOnFetchOpenOrdersWithoutSymbol"] = False
        oos = await cc.fetch_open_orders()
        pos_syms = [elm["info"]["symbol"] for elm in poss if float(elm["info"]["positionAmt"]) != 0.0]
        oos_syms = [elm["info"]["symbol"] for elm in oos]
        return sorted(set(pos_syms + oos_syms))
    oos = await cc.fetch_open_orders()
    current = sorted(set([x["symbol"] for x in poss + oos]))
    current = [x.replace("/", "")[:-5] for x in current]
    return current


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
                        min_cost = (
                            0.0 if x["limits"]["cost"]["min"] is None else x["limits"]["cost"]["min"]
                        )
                        c_mult = 1.0 if x["contractSize"] is None else x["contractSize"]
                        min_qty = (
                            0.0
                            if x["limits"]["amount"]["min"] is None
                            else x["limits"]["amount"]["min"]
                        )
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
    approved = [
        symbols_map[k] for k, v in min_costs.items() if v <= max_min_cost and k in symbols_map
    ]
    print("getting current bots...")
    current_ = await get_current_symbols(cc)
    current = []
    for elm in current_:
        if elm in symbols_map:
            current.append(symbols_map[elm])
        elif elm in symbols_map_inv:
            current.append(elm)
    print("getting ohlcvs...")
    ohs = await get_ohlcvs(cc, [symbols_map_inv[sym] for sym in approved], config)
    vols = {symbols_map[sym]: volatility(ohs[sym][-n_ohlcvs:]) for sym in ohs}
    for elm in sorted(vols.items(), key=lambda x: x[1]):
        print(elm)
    print(f"generating yaml {config['yaml_filepath']}...")
    yaml = generate_yaml(vols, approved, current, config)
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
    cc = getattr(ccxt, exchange_map[exchange])(
        {"apiKey": key, "secret": secret, "password": passphrase}
    )
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
        finally:
            await cc.close()


if __name__ == "__main__":
    asyncio.run(main())
