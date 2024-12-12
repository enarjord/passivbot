import argparse
import asyncio
import datetime
import gzip
import json
import logging
import os
import sys
import traceback
import zipfile
from collections import deque
from functools import wraps
from io import BytesIO
from pathlib import Path
from time import time
from typing import List, Dict, Any
from uuid import uuid4
from urllib.request import urlopen

import aiohttp
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from dateutil import parser
from tqdm import tqdm
from pure_funcs import date_to_ts2, ts_to_date_utc, safe_filename
from procedures import make_get_filepath, format_end_date, coin_to_symbol

# ========================= CONFIGURABLES & GLOBALS =========================

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S",
)

MAX_REQUESTS_PER_MINUTE = 120
REQUEST_TIMESTAMPS = deque(maxlen=1000)  # for rate-limiting checks

# ========================= HELPER FUNCTIONS =========================


async def check_rate_limit():
    current_time = time()
    while REQUEST_TIMESTAMPS and current_time - REQUEST_TIMESTAMPS[0] > 60:
        REQUEST_TIMESTAMPS.popleft()

    if len(REQUEST_TIMESTAMPS) >= MAX_REQUESTS_PER_MINUTE:
        sleep_time = 60 - (current_time - REQUEST_TIMESTAMPS[0])
        if sleep_time > 0:
            logging.info(f"Rate limit reached, sleeping for {sleep_time:.2f} seconds")
            await asyncio.sleep(sleep_time)

    REQUEST_TIMESTAMPS.append(current_time)


def dump_ohlcv_data(data, filepath):
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    if isinstance(data, pd.DataFrame):
        data = data[columns].astype(float).values
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise Exception(f"Unknown data format for {filepath}")
    np.save(filepath, data)


def load_ohlcv_data(filepath: str) -> pd.DataFrame:
    arr = np.load(filepath, allow_pickle=True)
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    return ensure_millis(pd.DataFrame(arr, columns=columns))


def get_days_in_between(start_day, end_day):
    date_format = "%Y-%m-%d"
    start_date = datetime.datetime.strptime(start_day, date_format)
    end_date = datetime.datetime.strptime(end_day, date_format)
    days = []
    current_date = start_date
    while current_date <= end_date:
        days.append(current_date.strftime(date_format))
        current_date += datetime.timedelta(days=1)
    return days


def attempt_gap_fix_hlcvs(df, symbol=None):
    interval = 60_000
    max_hours = 12
    max_gap = interval * 60 * max_hours
    greatest_gap = df.timestamp.diff().max()
    if pd.isna(greatest_gap) or greatest_gap == interval:
        return df
    if greatest_gap > max_gap:
        raise Exception(f"Huge gap in data for {symbol}: {greatest_gap/(1000*60*60)} hours.")
    logging.info(
        f"Filling small gaps in {symbol}. Largest gap: {greatest_gap/(1000*60*60):.3f} hours."
    )
    new_timestamps = np.arange(df["timestamp"].iloc[0], df["timestamp"].iloc[-1] + interval, interval)
    new_df = df.set_index("timestamp").reindex(new_timestamps)
    new_df.close = new_df.close.ffill()
    for col in ["open", "high", "low"]:
        new_df[col] = new_df[col].fillna(new_df.close)
    new_df["volume"] = new_df["volume"].fillna(0.0)
    return new_df.reset_index().rename(columns={"index": "timestamp"})


async def fetch_url(session, url):
    async with session.get(url) as response:
        response.raise_for_status()
        return await response.read()


async def fetch_zips(url):
    try:
        async with aiohttp.ClientSession() as session:
            content = await fetch_url(session, url)
        zips = []
        with zipfile.ZipFile(BytesIO(content), "r") as z:
            for f in z.namelist():
                zips.append(z.open(f))
        return zips
    except Exception as e:
        logging.error(f"Error fetching zips {url}: {e}")


async def get_zip_binance(url):
    col_names = ["timestamp", "open", "high", "low", "close", "volume"]
    zips = await fetch_zips(url)
    if not zips:
        return pd.DataFrame(columns=col_names)
    dfs = []
    for z in zips:
        df = pd.read_csv(z, header=None)
        df.columns = col_names + [f"extra_{i}" for i in range(len(df.columns) - len(col_names))]
        dfs.append(df[col_names])
    dfc = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
    return dfc[dfc.timestamp != "open_time"].astype(float)

async def get_zip_bitget(url):
    col_names = ["timestamp", "open", "high", "low", "close", "volume"]
    zips = await fetch_zips(url)
    if not zips:
        return pd.DataFrame(columns=col_names)
    dfs = []
    for z in zips:
        df = ensure_millis(pd.read_excel(z))
        df.columns = col_names + [f"extra_{i}" for i in range(len(df.columns) - len(col_names))]
        dfs.append(df[col_names])
    dfc = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
    return dfc[dfc.timestamp != "open_time"]


def ensure_millis(df):
    if 'timestamp' not in df.columns:
        return df
    if df.timestamp.iloc[0] > 1e14: # is microseconds
        df.timestamp /= 1000
    elif df.timestamp.iloc[0] > 1e11: # is milliseconds
        pass
    else: # is seconds
        df.timestamp *= 1000
    return df

# ========================= BASE CLASS & EXCHANGE CLASSES =========================


class BaseDownloader:
    def __init__(self, ccxt_exchange=None):
        self.exchange = ccxt_exchange

    def filter_date_range(self, df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
        """Filter dataframe to include only data within start_date and end_date (inclusive)"""
        if df.empty:
            return df
        start_ts = date_to_ts2(start_date)
        end_ts = date_to_ts2(format_end_date(end_date))
        return df[(df.timestamp >= start_ts) & (df.timestamp <= end_ts)].reset_index(drop=True)

    async def download_ohlcv(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        raise NotImplementedError


class BinanceDownloader(BaseDownloader):
    async def download_ohlcv(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        # Uses Binance's data archives via binance.vision
        symbol = symbol.replace("/USDT:", "")
        dirpath = make_get_filepath(f"historical_data/ohlcvs_binance/{safe_filename(symbol)}/")
        end_date = format_end_date(end_date)
        start_ts = date_to_ts2(start_date)
        end_ts = date_to_ts2(end_date)
        days = [
            ts_to_date_utc(x)[:10] for x in range(int(start_ts), int(end_ts), 1000 * 60 * 60 * 24)
        ]
        months = sorted({x[:7] for x in days})
        base_url = "https://data.binance.vision/data/futures/um/"
        # Download monthly
        for month in months:
            fpath = os.path.join(dirpath, month + ".npy")
            if not os.path.exists(fpath):
                url = base_url + f"monthly/klines/{symbol}/1m/{symbol}-1m-{month}.zip"
                logging.info(f"Downloading Binance monthly data for {symbol}: {month}")
                await self.download_single(url, fpath)

        # Download daily for any gap months
        fnames = os.listdir(dirpath)
        months_done = [x[:-4] for x in fnames if x.endswith(".npy")]
        missing_days = {
            day: os.path.join(dirpath, day + ".npy") for day in days if (day[:7] not in months_done)
        }

        for day, fpath in missing_days.items():
            if not os.path.exists(fpath):
                url = base_url + f"daily/klines/{symbol}/1m/{symbol}-1m-{day}.zip"
                logging.info(f"Downloading Binance daily data for {symbol}: {day}")

                await self.download_single(url, fpath)

        # Cleanup days contained in months
        fnames = os.listdir(dirpath)
        for fname in fnames:
            if fname.endswith(".npy") and len(fname) == 14:
                if fname[:7] + ".npy" in fnames:
                    os.remove(os.path.join(dirpath, fname))

        # Load all data
        all_files = [f for f in os.listdir(dirpath) if f.endswith(".npy")]
        dfs = [load_ohlcv_data(os.path.join(dirpath, f)) for f in all_files]
        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs).drop_duplicates("timestamp").sort_values("timestamp")
        # fill gaps
        interval = 60000
        nindex = np.arange(df.timestamp.iloc[0], df.timestamp.iloc[-1] + interval, interval)
        df = df.set_index("timestamp").reindex(nindex).ffill().reset_index()
        df = df.rename(columns={"index": "timestamp"})
        return self.filter_date_range(df, start_date, end_date)

    async def download_single(self, url: str, fpath: str):
        try:
            csv = await get_zip_binance(url)
            dump_ohlcv_data(ensure_millis(csv), fpath)
        except Exception as e:
            logging.error(f"Failed to download {url}: {e}")
            traceback.print_exc()


class BitgetDownloader(BaseDownloader):
    async def download_ohlcv(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        # Uses Binance's data archives via binance.vision
        symbol = symbol.replace("/USDT:", "")
        dirpath = make_get_filepath(f"historical_data/ohlcvs_bitget/{safe_filename(symbol)}/")
        end_date = format_end_date(end_date)
        days = get_days_in_between(start_date, end_date)
        base_url = "https://img.bitgetimg.com/online/kline/"
        # Download daily
        fnames = os.listdir(dirpath)
        days_done = {f[:10] for f in fnames}
        missing_days = {d for d in days if d not in days_done}
        tasks = []
        for day in sorted(missing_days):
            fpath = day + '.npy'
            url = f"{base_url}{symbol}/{symbol}_UMCBL_1min_{day.replace('-', '')}.zip"
            logging.info(f"Downloading Bitget daily data for {symbol}: {day}")
            await check_rate_limit()
            tasks.append((symbol, day, asyncio.create_task(get_zip_bitget(url))))
        for symbol, day, task in tasks:
            res = await task
            dump_ohlcv_data(ensure_millis(res), os.path.join(dirpath, day + '.npy'))

        # Load all data
        all_files = [f for f in os.listdir(dirpath) if f.endswith(".npy")]
        dfs = [load_ohlcv_data(os.path.join(dirpath, f)) for f in all_files]
        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs).drop_duplicates("timestamp").sort_values("timestamp")
        # fill gaps
        interval = 60000
        nindex = np.arange(df.timestamp.iloc[0], df.timestamp.iloc[-1] + interval, interval)
        df = df.set_index("timestamp").reindex(nindex).ffill().reset_index()
        df = df.rename(columns={"index": "timestamp"})
        return self.filter_date_range(df, start_date, end_date)


class BybitDownloader(BaseDownloader):
    # Bybit has public data archives
    async def download_ohlcv(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        symbol_u = symbol.replace('/USDT:', '')
        start_date = start_date[:10]
        end_date = end_date[:10]
        dirpath = make_get_filepath(f"historical_data/ohlcvs_bybit/{safe_filename(symbol_u)}/")

        # Bybit public data: "https://public.bybit.com/trading/"
        base_url = "https://public.bybit.com/trading/"
        webpage = urlopen(f"{base_url}{symbol_u}/").read().decode()

        days = get_days_in_between(start_date, end_date)
        days_done = [f[:-4] for f in os.listdir(dirpath) if f.endswith(".npy")]
        days_to_get = [d for d in days if d not in days_done]

        if not days_to_get:
            # load existing
            dfs = []
            for d in days:
                fpath = os.path.join(dirpath, d + ".npy")
                if os.path.exists(fpath):
                    dfs.append(load_ohlcv_data(fpath))
            if not dfs:
                return pd.DataFrame()
            df = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
            df = attempt_gap_fix_hlcvs(df, symbol=symbol_u)
            return self.filter_date_range(df, start_date, end_date)

        filenames = [
            f"{symbol_u}{day}.csv.gz" for day in days_to_get if f"{symbol_u}{day}.csv.gz" in webpage
        ]
        # Download concurrently
        df_map = {}
        async with aiohttp.ClientSession() as session:
            tasks = []
            for fn in filenames:
                url = f"{base_url}{symbol_u}/{fn}"
                tasks.append(asyncio.create_task(self.get_and_convert_bybit(session, url)))
            results = await asyncio.gather(*tasks, return_exceptions=True)

        for day, res in zip(filenames, results):
            if isinstance(res, pd.DataFrame) and not res.empty:
                d = day[-17:-7]
                dump_ohlcv_data(ensure_millis(res), os.path.join(dirpath, d + ".npy"))
                df_map[d] = res

        # Combine all
        dfs = []
        for d in days:
            fpath = os.path.join(dirpath, d + ".npy")
            if os.path.exists(fpath):
                dfs.append(load_ohlcv_data(fpath))
        if not dfs:
            return pd.DataFrame()

        df = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
        df = attempt_gap_fix_hlcvs(df, symbol=symbol_u)
        return self.filter_date_range(df, start_date, end_date)

    async def get_and_convert_bybit(self, session, url: str) -> pd.DataFrame:
        await check_rate_limit()
        try:
            resp = await fetch_url(session, url)
            with gzip.open(BytesIO(resp)) as f:
                raw = pd.read_csv(f)
            # Convert trades to OHLCV
            interval = 60000
            groups = raw.groupby((raw.timestamp * 1000) // interval * interval)
            ohlcvs = pd.DataFrame(
                {
                    "open": groups.price.first(),
                    "high": groups.price.max(),
                    "low": groups.price.min(),
                    "close": groups.price.last(),
                    "volume": groups["size"].sum(),
                }
            )
            ohlcvs["timestamp"] = ohlcvs.index
            return ohlcvs[["timestamp", "open", "high", "low", "close", "volume"]]
        except Exception as e:
            logging.error(f"Bybit error {url}: {e}")
            traceback.print_exc()
            return pd.DataFrame()


class GateioDownloader(BaseDownloader):
    async def download_ohlcv(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            return await self.download_ohlcv_(symbol, start_date, end_date)
        finally:
            await self.exchange.close()

    async def download_ohlcv_(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        dirpath = make_get_filepath(f"historical_data/ohlcvs_gateio/{safe_filename(symbol)}/")
        days = get_days_in_between(start_date[:10], end_date[:10])
        days_done = [f[:-4] for f in os.listdir(dirpath) if f.endswith(".npy")]
        days_to_get = [d for d in days if d not in days_done]

        if not days_to_get:
            dfs = [load_ohlcv_data(os.path.join(dirpath, d + ".npy")) for d in days]
            return self._combine_and_fix(dfs, symbol, start_date, end_date)

        tasks = [self.fetch_and_save_day(symbol, d, dirpath) for d in days_to_get]

        await asyncio.gather(*tasks)

        dfs = [
            load_ohlcv_data(os.path.join(dirpath, d + ".npy"))
            for d in days
            if os.path.exists(os.path.join(dirpath, d + ".npy"))
        ]
        return self._combine_and_fix(dfs, symbol, start_date, end_date)

    async def fetch_and_save_day(self, symbol: str, day: str, dirpath: str):
        fpath = os.path.join(dirpath, f"{day}.npy")
        start_ts_day = date_to_ts2(day)
        end_ts_day = start_ts_day + 24 * 60 * 60 * 1000
        interval = "1m"
        limit = 1000

        dfs_day = []
        since = start_ts_day
        while True:
            await check_rate_limit()
            ohlcvs = await self.exchange.fetch_ohlcv(
                symbol, timeframe=interval, since=since, limit=limit
            )
            if not ohlcvs:
                break
            df_part = pd.DataFrame(
                ohlcvs, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            dfs_day.append(df_part)
            last_ts = df_part["timestamp"].iloc[-1]
            if last_ts >= end_ts_day or len(df_part) < limit:
                break
            logging.info(f"fetched GateIO ohlcv for {symbol} {ts_to_date_utc(since)}")
            since = last_ts + 60000

        if dfs_day:
            df_day = (
                pd.concat(dfs_day)
                .drop_duplicates("timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            df_day = df_day[(df_day.timestamp >= start_ts_day) & (df_day.timestamp < end_ts_day)]
            # convert volume to base volume before dumping
            df_day.volume = df_day.volume / df_day.close
            dump_ohlcv_data(ensure_millis(df_day), fpath)

    def _combine_and_fix(
        self, dfs: List[pd.DataFrame], symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        if not dfs:
            return pd.DataFrame()
        df = (
            pd.concat(dfs)
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        df = attempt_gap_fix_hlcvs(df, symbol=symbol)
        return self.filter_date_range(df, start_date, end_date)


# ========================= ENTRY POINTS =========================


async def main():
    parser = argparse.ArgumentParser(description="Download OHLCV data")
    parser.add_argument("--exchange", type=str, default="binance")
    parser.add_argument("--symbol", type=str, default="BTC/USDT:USDT")
    parser.add_argument("--start_date", type=str, default="2021-01-01")
    parser.add_argument("--end_date", type=str, default="2021-03-01")
    args = parser.parse_args()

    # tests
    for exchange, downloader in [
        ("binance", BinanceDownloader()),
        ("bybit", BybitDownloader()),
        ("gateio", GateioDownloader(ccxt.gateio({"enableRateLimit": True}))),
        ("bitget", BitgetDownloader()),
    ]:
        logging.info(f"Testing {exchange} downloader...")
        try:
            df = await downloader.download_ohlcv(args.symbol, args.start_date, args.end_date)
            logging.info(f"data fetched: {len(df)} rows")
            print(df)
        except Exception as e:
            logging.error(f"error with {exchange} {e}")
            traceback.print_exc()
    return


if __name__ == "__main__":
    asyncio.run(main())
