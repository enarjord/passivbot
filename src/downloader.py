import argparse
import asyncio
import datetime
import gzip
import os
import sys
import requests
import json
from io import BytesIO
from time import time
from typing import Tuple
from urllib.request import urlopen
from functools import reduce
import zipfile
import traceback
import aiohttp
import ccxt.async_support as ccxt


from functools import partial


import numpy as np
import pandas as pd
from dateutil import parser
from tqdm import tqdm

from njit_funcs import calc_samples
from procedures import (
    prepare_backtest_config,
    make_get_filepath,
    create_binance_bot,
    create_bybit_bot,
    create_binance_bot_spot,
    print_,
    add_argparse_args,
    utc_ms,
    get_first_ohlcv_timestamps,
)
from pure_funcs import ts_to_date, ts_to_date_utc, date_to_ts2, get_dummy_settings, get_day, numpyize


async def fetch_zips(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                response.raise_for_status()
                zip_content = await response.read()
        zips = []
        with zipfile.ZipFile(BytesIO(zip_content), "r") as zip_ref:
            for contained_file in zip_ref.namelist():
                zips.append(zip_ref.open(contained_file))
        return zips

    except aiohttp.ClientError as e:
        print("Error during HTTP request:", e)
    except zipfile.BadZipFile:
        print("Error extracting the zip file. Make sure it contains a valid CSV file.")
    except pd.errors.EmptyDataError:
        print("The CSV file is empty or could not be loaded as a DataFrame.")


async def get_zip_binance(url):
    col_names = ["timestamp", "open", "high", "low", "close", "volume"]
    zips = await fetch_zips(url)
    dfs = []
    for zip in zips:
        df = pd.read_csv(zip, header=None)
        df.columns = col_names + [str(i) for i in range(len(df.columns) - len(col_names))]
        dfs.append(df[col_names])
    dfc = pd.concat(dfs).sort_values("timestamp").reset_index()
    return dfc[dfc.timestamp != "open_time"]


def get_first_ohlcv_ts(symbol: str, spot=False) -> int:
    try:
        if spot:
            url = "https://api.binance.com/api/v3/klines"
        else:
            url = "https://fapi.binance.com/fapi/v1/klines"
        res = requests.get(
            url, params={"symbol": symbol, "startTime": 0, "limit": 100, "interval": "1m"}
        )
        first_ohlcvs = json.loads(res.text)
        first_ts = first_ohlcvs[0][0]
        return first_ts
    except Exception as e:
        print(f"error getting first ohlcv ts {e}, returning 0")
        return 0


def get_days_in_between(start_day, end_day):
    date_format = "%Y-%m-%d"
    start_date = datetime.datetime.strptime(start_day, date_format)
    end_date = datetime.datetime.strptime(end_day, date_format)

    days_in_between = []
    current_date = start_date
    while current_date <= end_date:
        days_in_between.append(current_date.strftime(date_format))
        current_date += datetime.timedelta(days=1)

    return days_in_between


async def download_ohlcvs_bybit(symbol, start_date, end_date, spot=False, download_only=False):
    ns = [30, 10, 1]
    for i, n in enumerate(ns):
        try:
            return await download_ohlcvs_bybit_sub(
                symbol, start_date, end_date, spot=False, download_only=False, n_concurrent_fetches=n
            )
        except Exception as e:
            print(f"Error fetching trades from bybit for {symbol} {e}. ")
            if i < len(ns):
                f"Retrying with concurrent fetches changed {n} -> {ns[i+1]}."


async def download_ohlcvs_bybit_sub(
    symbol, start_date, end_date, spot=False, download_only=False, n_concurrent_fetches=10
):
    start_date, end_date = get_day(start_date), get_day(end_date)
    assert date_to_ts2(end_date) >= date_to_ts2(start_date), "end_date is older than start_date"
    dirpath = make_get_filepath(f"historical_data/ohlcvs_bybit{'_spot' if spot else ''}/{symbol}/")
    convert_csv_to_npy(dirpath)
    ideal_days = get_days_in_between(start_date, end_date)
    days_done = [filename[:-4] for filename in os.listdir(dirpath) if ".npy" in filename]
    days_to_get = [day for day in ideal_days if day not in days_done]
    dfs = {}
    if len(days_to_get) > 0:
        base_url = f"https://public.bybit.com/{'spot' if spot else 'trading'}/"
        webpage = await get_bybit_webpage(base_url, symbol)
        filenames = [
            cand
            for day in days_to_get
            if (cand := f"{symbol}{'_' if spot else ''}{day}.csv.gz") in webpage
        ]
        if len(filenames) > 0:
            for i in range(0, len(filenames), n_concurrent_fetches):
                filenames_sublist = filenames[i : i + n_concurrent_fetches]
                print(
                    f"fetching {len(filenames_sublist)} files with {symbol} trades from {filenames_sublist[0][-17:-7]} to {filenames_sublist[-1][-17:-7]}"
                )
                dfs_ = await get_bybit_trades(base_url, symbol, filenames_sublist)
                dfs_ = {k[-17:-7]: convert_to_ohlcv(v, spot) for k, v in dfs_.items()}
                dumped = []
                for day, df in sorted(dfs_.items()):
                    if day in days_done:
                        continue
                    filepath = f"{dirpath}{day}.npy"
                    dump_ohlcv_data(df, filepath)
                    dumped.append(day)
                if not download_only:
                    dfs.update(dfs_)
    if not download_only:
        for day in ideal_days:
            if os.path.exists(f"{dirpath}{day}.npy"):
                dfs[day] = load_ohlcv_data(f"{dirpath}{day}.npy")
        if len(dfs) == 0:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = pd.concat(dfs.values()).sort_values("timestamp").reset_index()
        return df[["timestamp", "open", "high", "low", "close", "volume"]]


async def get_bybit_webpage(base_url: str, symbol: str):
    return urlopen(f"{base_url}{symbol}/").read().decode()


async def get_bybit_trades(base_url: str, symbol: str, filenames: [str]):
    if len(filenames) == 0:
        return None
    async with aiohttp.ClientSession() as session:
        tasks = {}
        for url in [f"{base_url}{symbol}/{filename}" for filename in filenames]:
            tasks[url] = asyncio.ensure_future(get_csv_gz(session, url))
        responses = {}
        for url in tasks:
            responses[url] = await tasks[url]
    return {k: v.sort_values("timestamp") for k, v in responses.items()}


async def fetch_url(session, url):
    async with session.get(url) as response:
        content = await response.read()
        return content


async def get_csv_gz(session, url: str):
    # from bybit
    try:
        resp = await fetch_url(session, url)
        with gzip.open(BytesIO(resp)) as f:
            tdf = pd.read_csv(f)
        return tdf
    except Exception as e:
        print("error fetching bybit trades", e)
        traceback.print_exc()
        return pd.DataFrame()


def convert_to_ohlcv(df, spot, interval=60000):
    # bybit data
    # timestamps are in seconds for futures, millis for spot
    groups = df.groupby((df.timestamp * (1 if spot else 1000)) // interval * interval)
    ohlcvs = pd.DataFrame(
        {
            "open": groups.price.first(),
            "high": groups.price.max(),
            "low": groups.price.min(),
            "close": groups.price.last(),
            "volume": groups["volume" if spot else "size"].sum(),
        }
    )
    new_index = np.arange(ohlcvs.index[0], ohlcvs.index[-1] + interval, interval)
    ohlcvs = ohlcvs.reindex(new_index)
    closes = ohlcvs.close.ffill()
    for x in ["open", "high", "low", "close"]:
        ohlcvs[x] = ohlcvs[x].fillna(closes)
    ohlcvs["volume"] = ohlcvs["volume"].fillna(0.0)
    ohlcvs.loc[:, "timestamp"] = ohlcvs.index.values
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    return ohlcvs[columns]


async def download_single_ohlcvs_binance(url: str, fpath: str):
    try:
        print(f"fetching {url}")
        csv = await get_zip_binance(url)
        dump_ohlcv_data(csv, fpath)
    except Exception as e:
        print(f"failed to download {url} {e}")


async def download_ohlcvs_binance(
    symbol,
    inverse,
    start_date,
    end_date,
    spot=False,
    download_only=False,
    start_tss=None,
) -> pd.DataFrame:
    dirpath = make_get_filepath(f"historical_data/ohlcvs_{'spot' if spot else 'futures'}/{symbol}/")
    convert_csv_to_npy(dirpath)
    base_url = "https://data.binance.vision/data/"
    base_url += "spot/" if spot else f"futures/{'cm' if inverse else 'um'}/"
    col_names = ["timestamp", "open", "high", "low", "close", "volume"]
    if start_tss is not None and symbol in start_tss:
        start_ts = start_tss[symbol]
    elif spot:
        start_ts = get_first_ohlcv_ts(symbol, spot=spot)
    else:
        start_ts = (await get_first_ohlcv_timestamps(symbols=[symbol]))[symbol]
    start_ts = int(max(start_ts, date_to_ts2(start_date)))
    end_ts = int(date_to_ts2(end_date))
    days = [ts_to_date_utc(x)[:10] for x in list(range(start_ts, end_ts, 1000 * 60 * 60 * 24))]
    months = sorted({x[:7] for x in days})
    month_now = ts_to_date(utc_ms())[:7]
    months = [m for m in months if m != month_now]

    # do months async
    months_filepaths = {month: os.path.join(dirpath, month + ".npy") for month in months}
    missing_months = {k: v for k, v in months_filepaths.items() if not os.path.exists(v)}
    await asyncio.gather(
        *[
            download_single_ohlcvs_binance(
                base_url + f"monthly/klines/{symbol}/1m/{symbol}-1m-{k}.zip", v
            )
            for k, v in missing_months.items()
        ]
    )
    months_done = sorted([x for x in os.listdir(dirpath) if x[:-4] in months_filepaths])

    # do days async
    days_filepaths = {day: os.path.join(dirpath, day + ".npy") for day in days}
    missing_days = {
        k: v
        for k, v in days_filepaths.items()
        if not os.path.exists(v) and k[:7] + ".npy" not in months_done
    }
    await asyncio.gather(
        *[
            download_single_ohlcvs_binance(
                base_url + f"daily/klines/{symbol}/1m/{symbol}-1m-{k}.zip", v
            )
            for k, v in missing_days.items()
        ]
    )
    days_done = sorted([x for x in os.listdir(dirpath) if x[:-4] in days_filepaths])

    # delete days contained in months
    fnames = os.listdir(dirpath)
    for fname in fnames:
        if fname.endswith(".npy") and len(fname) == 14:
            if fname[:7] + ".npy" in fnames:
                print("deleting", os.path.join(dirpath, fname))
                os.remove(os.path.join(dirpath, fname))

    if not download_only:
        fnames = os.listdir(dirpath)
        dfs = [
            load_ohlcv_data(os.path.join(dirpath, fpath))
            for fpath in months_done + days_done
            if fpath in fnames and fpath.endswith(".npy")
        ]
        try:
            df = pd.concat(dfs)[col_names].sort_values("timestamp")
        except ValueError as e:
            print(
                f"error with download_ohlcvs_binance {symbol} {start_date} {end_date}: {e}. Returning empty"
            )
            return pd.DataFrame()
        df = df.drop_duplicates(subset=["timestamp"]).reset_index()
        nindex = np.arange(df.timestamp.iloc[0], df.timestamp.iloc[-1] + 60000, 60000)
        return df[col_names].set_index("timestamp").reindex(nindex).ffill().reset_index()


def count_longest_identical_data(hlc, symbol, verbose=True):
    line = f"checking ohlcv integrity of {symbol}"
    diffs = (np.diff(hlc[:, 1:], axis=0) == [0.0, 0.0, 0.0]).all(axis=1)
    longest_consecutive = 0
    counter = 0
    i_ = 0
    for i, x in enumerate(diffs):
        if x:
            counter += 1
        else:
            if counter > longest_consecutive:
                longest_consecutive = counter
                i_ = i
            counter = 0
    if verbose:
        print(
            f"{symbol} most n days of consecutive identical ohlcvs: {longest_consecutive / 60 / 24:.3f}, index last: {i_}"
        )
    return longest_consecutive


def attempt_gap_fix_hlcvs(df, symbol=None):
    interval = 60 * 1000
    max_hours = 12
    max_gap = interval * 60 * max_hours
    greatest_gap = df.timestamp.diff().max()
    if greatest_gap == interval:
        return df
    if greatest_gap > max_gap:
        raise Exception(
            f"ohlcvs gap greater than {max_hours} hours: {greatest_gap / (1000 * 60 * 60)} hours"
        )
    print(
        f"ohlcvs for {symbol} has greatest gap {greatest_gap / (1000 * 60 * 60):.3f} hours. Filling gaps..."
    )
    new_timestamps = np.arange(df["timestamp"].iloc[0], df["timestamp"].iloc[-1] + interval, interval)
    new_df = df.set_index("timestamp").reindex(new_timestamps)
    new_df.close = new_df.close.ffill()
    new_df.open = new_df.open.fillna(new_df.close)
    new_df.high = new_df.high.fillna(new_df.close)
    new_df.low = new_df.low.fillna(new_df.close)
    new_df.volume = new_df.volume.fillna(0.0)
    new_df = new_df.reset_index()
    return new_df[["timestamp", "open", "high", "low", "close", "volume"]]


async def load_hlcvs(symbol, start_date, end_date, base_dir="backtests", exchange="binance"):
    # returns matrix [[timestamp, high, low, close, volume]]
    if exchange == "binance":
        df = await download_ohlcvs_binance(symbol, False, start_date, end_date, False)
    elif exchange == "bybit":
        df = await download_ohlcvs_bybit(symbol, start_date, end_date)
        df = attempt_gap_fix_hlcvs(df, symbol=symbol)
    else:
        raise Exception(f"downloading ohlcvs from exchange {exchange} not supported")
    if len(df) == 0:
        return pd.DataFrame()
    df = df[df.timestamp >= date_to_ts2(start_date)]
    df = df[df.timestamp <= date_to_ts2(end_date)]
    return df[["timestamp", "high", "low", "close", "volume"]].values


async def prepare_hlcvs_old(config: dict):
    symbols = sorted(set(config["backtest"]["symbols"]))
    start_date = config["backtest"]["start_date"]
    end_date = config["backtest"]["end_date"]
    base_dir = config["backtest"]["base_dir"]
    exchange = config["backtest"]["exchange"]
    minimum_coin_age_days = config["live"]["minimum_coin_age_days"]

    ms2day = 1000 * 60 * 60 * 24
    if end_date in ["today", "now", "", None]:
        end_ts = (utc_ms() - ms2day) // ms2day * ms2day
        end_date = ts_to_date_utc(end_ts)[:10]
    else:
        end_ts = date_to_ts2(end_date) // ms2day * ms2day

    interval_ms = 60000
    start_tss = None
    if exchange == "binance":
        start_tss = await get_first_ohlcv_timestamps(cc=ccxt.binanceusdm(), symbols=symbols)
    elif exchange == "bybit":
        start_tss = await get_first_ohlcv_timestamps(cc=ccxt.bybit(), symbols=symbols)
    else:
        raise Exception("failed to load start timestamps")

    # Calculate global start and end times
    global_start_ts = date_to_ts2(start_date)
    global_end_ts = end_ts
    n_timesteps = int((global_end_ts - global_start_ts) / interval_ms) + 1

    # Pre-allocate the unified array
    n_coins = len(symbols)
    unified_array = np.zeros((n_timesteps, n_coins, 4), dtype=np.float64)

    for i, symbol in enumerate(symbols):
        if symbol not in start_tss:
            print(f"coin {symbol} missing from first timestamps, skipping")
            continue
        adjusted_start_ts = global_start_ts
        if minimum_coin_age_days > 0.0:
            min_coin_age_ms = 1000 * 60 * 60 * 24 * minimum_coin_age_days
            new_start_ts = start_tss[symbol] + min_coin_age_ms
            if new_start_ts >= end_ts:
                print(
                    f"Coin {symbol} too young, start date {ts_to_date_utc(start_tss[symbol])}, skipping"
                )
                continue
            if new_start_ts > adjusted_start_ts:
                print(
                    f"First date for {symbol} was {ts_to_date_utc(start_tss[symbol])}. Adjusting start date to {ts_to_date_utc(new_start_ts)}"
                )
            adjusted_start_ts = max(adjusted_start_ts, new_start_ts)

        data = await load_hlcvs(
            symbol,
            ts_to_date_utc(adjusted_start_ts)[:10],
            end_date,
            base_dir,
            exchange,
        )

        if len(data) == 0:
            continue

        assert (np.diff(data[:, 0]) == interval_ms).all(), f"gaps in hlcv data {symbol}"

        # Calculate indices for this coin's data
        start_idx = int((data[0, 0] - global_start_ts) / interval_ms)
        end_idx = start_idx + len(data)

        # Extract and process the required data (high, low, close, volume)
        coin_data = data[:, 1:]
        coin_data[:, 3] = coin_data[:, 2] * coin_data[:, 3]  # Use quote volume as volume

        # Place the data in the unified array
        unified_array[start_idx:end_idx, i, :] = coin_data

        # Front-fill
        if start_idx > 0:
            unified_array[:start_idx, i, :3] = coin_data[0, 2]

        # Back-fill
        if end_idx < n_timesteps:
            unified_array[end_idx:, i, :3] = coin_data[-1, 2]
    print(f"Finished fetching all data. Returning unified array.")

    timestamps = np.arange(global_start_ts, global_end_ts + interval_ms, interval_ms)

    return symbols, timestamps, unified_array


async def prepare_hlcvs(config: dict):
    symbols = sorted(set(config["backtest"]["symbols"]))
    start_date = config["backtest"]["start_date"]
    end_date = config["backtest"]["end_date"]
    base_dir = config["backtest"]["base_dir"]
    exchange = config["backtest"]["exchange"]
    minimum_coin_age_days = config["live"]["minimum_coin_age_days"]

    ms2day = 1000 * 60 * 60 * 24
    if end_date in ["today", "now", "", None]:
        end_ts = (utc_ms() - ms2day) // ms2day * ms2day
        end_date = ts_to_date_utc(end_ts)[:10]
    else:
        end_ts = date_to_ts2(end_date) // ms2day * ms2day

    interval_ms = 60000
    start_tss = None
    if exchange == "binance":
        start_tss = await get_first_ohlcv_timestamps(cc=ccxt.binanceusdm(), symbols=symbols)
    elif exchange == "bybit":
        start_tss = await get_first_ohlcv_timestamps(cc=ccxt.bybit(), symbols=symbols)
    else:
        raise Exception("failed to load start timestamps")

    # Calculate global start and end times
    global_start_ts = date_to_ts2(start_date)
    global_end_ts = end_ts
    n_timesteps = int((global_end_ts - global_start_ts) / interval_ms) + 1

    # Pre-allocate the unified array
    n_coins = len(symbols)
    unified_array = np.zeros((n_timesteps, n_coins, 4), dtype=np.float64)

    # Create a partial function with fixed arguments
    process_symbol_partial = partial(
        process_symbol,
        start_date=start_date,
        end_date=end_date,
        base_dir=base_dir,
        exchange=exchange,
        minimum_coin_age_days=minimum_coin_age_days,
        global_start_ts=global_start_ts,
        global_end_ts=global_end_ts,
        interval_ms=interval_ms,
        start_tss=start_tss,
    )

    # Use asyncio.gather for concurrent processing
    tasks = [process_symbol_partial(symbol) for symbol in symbols]
    results = await asyncio.gather(*tasks)

    # Process results and update unified_array
    for i, (symbol, data) in enumerate(zip(symbols, results)):
        if data is not None:
            start_idx, end_idx, coin_data = data
            unified_array[start_idx:end_idx, i, :] = coin_data

            # Front-fill
            if start_idx > 0:
                unified_array[:start_idx, i, :3] = coin_data[0, 2]

            # Back-fill
            if end_idx < n_timesteps:
                unified_array[end_idx:, i, :3] = coin_data[-1, 2]

    print(f"Finished fetching all data. Returning unified array.")

    timestamps = np.arange(global_start_ts, global_end_ts + interval_ms, interval_ms)

    return symbols, timestamps, unified_array


async def process_symbol(
    symbol,
    start_date,
    end_date,
    base_dir,
    exchange,
    minimum_coin_age_days,
    global_start_ts,
    global_end_ts,
    interval_ms,
    start_tss,
):
    if symbol not in start_tss:
        print(f"coin {symbol} missing from first timestamps, skipping")
        return None

    adjusted_start_ts = global_start_ts
    if minimum_coin_age_days > 0.0:
        min_coin_age_ms = 1000 * 60 * 60 * 24 * minimum_coin_age_days
        new_start_ts = start_tss[symbol] + min_coin_age_ms
        if new_start_ts >= global_end_ts:
            print(
                f"Coin {symbol} too young, start date {ts_to_date_utc(start_tss[symbol])}, skipping"
            )
            return None
        if new_start_ts > adjusted_start_ts:
            print(
                f"First date for {symbol} was {ts_to_date_utc(start_tss[symbol])}. Adjusting start date to {ts_to_date_utc(new_start_ts)}"
            )
        adjusted_start_ts = max(adjusted_start_ts, new_start_ts)

    data = await load_hlcvs(
        symbol,
        ts_to_date_utc(adjusted_start_ts)[:10],
        end_date,
        base_dir,
        exchange,
    )

    if len(data) == 0:
        return None

    assert (np.diff(data[:, 0]) == interval_ms).all(), f"gaps in hlcv data {symbol}"

    # Calculate indices for this coin's data
    start_idx = int((data[0, 0] - global_start_ts) / interval_ms)
    end_idx = start_idx + len(data)

    # Extract and process the required data (high, low, close, volume)
    coin_data = data[:, 1:]
    coin_data[:, 3] = coin_data[:, 2] * coin_data[:, 3]  # Use quote volume as volume

    return start_idx, end_idx, coin_data


def convert_csv_to_npy(filepath):
    if not os.path.exists(filepath):
        return False
    if os.path.isdir(filepath):
        for fp in os.listdir(filepath):
            convert_csv_to_npy(os.path.join(filepath, fp))
        return False
    if filepath.endswith(".csv"):
        columns = ["timestamp", "open", "high", "low", "close", "volume"]
        npy_filepath = filepath.replace(".csv", ".npy")
        csv_data = pd.read_csv(filepath)[columns]
        dump_ohlcv_data(csv_data, npy_filepath)
        os.remove(filepath)
        print(f"successfully converted {filepath} to {npy_filepath}")
        return True


def dump_ohlcv_data(data, filepath):
    npy_filepath = filepath.replace(".csv", ".npy")
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    if isinstance(data, pd.DataFrame):
        to_dump = data[columns].astype(float).values
    elif isinstance(data, np.ndarray):
        to_dump = data
    else:
        raise Exception(f"unknown file type {filepath} dump_ohlcv_data")
    np.save(npy_filepath, to_dump)


def load_ohlcv_data(filepath):
    npy_filepath = filepath.replace(".csv", ".npy")
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    if os.path.exists(npy_filepath):
        loaded_data = np.load(npy_filepath, allow_pickle=True)
    else:
        print(f"loading {filepath}")
        csv_data = pd.read_csv(filepath)[columns]
        print(f"dumping {npy_filepath}")
        dump_ohlcv_data(csv_data, npy_filepath)
        print(f"removing {filepath}")
        os.remove(filepath)
        loaded_data = csv_data.values
    return pd.DataFrame(loaded_data, columns=columns)


async def main():
    parser = argparse.ArgumentParser(
        prog="Downloader", description="Download ticks from exchange API."
    )
    parser.add_argument(
        "-d",
        "--download-only",
        help="download only, do not dump ticks caches",
        action="store_true",
    )
    parser = add_argparse_args(parser)

    args = parser.parse_args()
    config = prepare_backtest_config(args)
    data = await load_hlc_cache(
        config["symbol"],
        config["inverse"],
        config["start_date"],
        config["end_date"],
        spot=config["spot"],
        exchange=config["exchange"],
    )


if __name__ == "__main__":
    asyncio.run(main())
