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
    yaml = f"session_name: {config['user']}\nwindows:\n- window_name: {config['user']}_normal\n  layout: "
    yaml += f"even-vertical\n  shell_command_before:\n    - cd ~/passivbot\n  panes:\n"
    conf_path = config["conf_path"]
    k = 0
    new = []
    user = config["user"]
    twe_long = config["twe_long"]
    twe_short = config["twe_short"]

    n_longs = config["n_longs"]
    n_shorts = config["n_shorts"]
    lw = round(twe_long / n_longs, 4) if n_longs > 0 else 0.1
    sw = round(twe_short / n_shorts, 4) if n_shorts > 0 else 0.1
    lm, sm = "gs", "gs"
    lev = 10 if config['lev'] is None else config['lev']
    for sym, vol in sorted(vols.items(), key=lambda x: x[1], reverse=True):
        if sym in approved:
            k += 1
            lm = "n" if k <= n_longs else "gs"
            sm = "n" if k <= n_shorts else "gs"
            if lm == "gs" and sm == "gs":
                break
            pane = f"    - shell_command:\n      - python3 passivbot.py {user} {sym} {conf_path} "
            pane += f"-lw {lw} -sw {sw} -lm {lm} -sm {sm} -lev {lev} -cd -pt 0.06"
            yaml += pane + "\n"
            new.append(sym)
    bots_on_gs = [sym for sym in current if sym not in new]
    if bots_on_gs:
        yaml += (
            f"- window_name: {config['user']}_gs\n  layout: even-vertical\n  shell_command_before:\n    - cd ~/passivbot\n  panes:"
            + "\n"
        )
        gs_lw = lw if config["gs_lw"] is None else config["gs_lw"]
        gs_sw = lw if config["gs_sw"] is None else config["gs_sw"]
        for sym in bots_on_gs:
            pane = f"    - shell_command:\n      - python3 passivbot.py {user} {sym} {conf_path} -lw {gs_lw} "
            pane += f"-sw {gs_sw} -lm gs -sm gs -lev {lev} -cd -pt 0.06"
            for k0, k1 in [("lmm", "gs_mm"), ("lmr", "gs_mr")]:
                if config[k1] is not None:
                    pane += f" -{k0} {config[k1]}"
            yaml += pane + "\n"
    print("active bots:", new)
    print("bots on -gs:", bots_on_gs)
    return yaml


async def get_ohlcvs(cc, symbols, config):
    ohs = {}
    n = 5
    if cc.id == 'bybit':
        extra_args = {'since': int(utc_ms() - 1000 * 60 * 60 * 25)}
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
    poss = await cc.fetch_positions()
    if cc.id == 'bybit':
        return sorted(set([elm['info']['symbol'] for elm in poss if float(elm['info']['size']) != 0.0]))
    elif cc.id == 'binanceusdm':
        cc.options["warnOnFetchOpenOrdersWithoutSymbol"] = False
        oos = await cc.fetch_open_orders()
        posss = [elm['info']['symbol'] for elm in poss if float(elm['info']['positionAmt']) != 0.0]
        ooss = [elm['info']['symbol'] for elm in oos]
        return sorted(set(posss + ooss))
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
                        min_qty = float(x['info']['multiplier'])
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
    symbols_map_inv = {v: k for k, v in symbols_map.items()}
    approved = [symbols_map[k] for k, v in min_costs.items() if v <= max_min_cost]
    print("getting current bots...")
    current = await get_current_symbols(cc)
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
    exchange_map = {'kucoin': 'kucoinfutures', 'okx': 'okx', 'bybit': 'bybit', 'binance': 'binanceusdm'}
    parser = argparse.ArgumentParser(prog="forager", description="start forager")
    parser.add_argument("forager_config_path", type=str, help="path to forager config")
    args = parser.parse_args()
    config = hjson.load(open(args.forager_config_path))
    config["yaml_filepath"] = f"{config['user']}.yaml"
    # choices: okx, binanceusdm, bitget, bybit, kucoinfutures
    user = config["user"]
    exchange, key, secret, passphrase = load_exchange_key_secret_passphrase(config["user"])
    cc = getattr(ccxt, exchange_map[exchange])({"apiKey": key, "secret": secret, "password": passphrase})
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
