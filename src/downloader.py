import argparse
import asyncio
import datetime
import gzip
import json
import logging
import inspect
import os
import shutil
import sys
import traceback
import zipfile
from collections import deque
from functools import wraps
from io import BytesIO
from pathlib import Path
from time import time
from typing import List, Dict, Any, Tuple
from uuid import uuid4
from urllib.request import urlopen
from collections import defaultdict

import aiohttp
import pprint
import ccxt.async_support as ccxt
import numpy as np
import pandas as pd
from dateutil import parser
from tqdm import tqdm
from pure_funcs import (
    date_to_ts,
    ts_to_date_utc,
    safe_filename,
    symbol_to_coin,
    get_template_live_config,
)
from procedures import (
    make_get_filepath,
    format_end_date,
    coin_to_symbol,
    utc_ms,
    get_file_mod_utc,
    get_first_timestamps_unified,
    add_arguments_recursively,
    load_config,
)

# ========================= CONFIGURABLES & GLOBALS =========================

logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S",
)

MAX_REQUESTS_PER_MINUTE = 120
REQUEST_TIMESTAMPS = deque(maxlen=1000)  # for rate-limiting checks

# ========================= HELPER FUNCTIONS =========================


def is_valid_date(date):
    try:
        ts = date_to_ts(date)
        return True
    except:
        return False


def get_function_name():
    return inspect.currentframe().f_back.f_code.co_name


def dump_ohlcv_data(data, filepath):
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    if isinstance(data, pd.DataFrame):
        data = ensure_millis(data[columns]).astype(float).values
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise Exception(f"Unknown data format for {filepath}")
    np.save(filepath, deduplicate_rows(data))


def deduplicate_rows(arr):
    """
    Remove duplicate rows from a 2D NumPy array while preserving order.

    Parameters:
    arr (numpy.ndarray): Input 2D array of shape (x, y)

    Returns:
    numpy.ndarray: Array with duplicate rows removed, maintaining original order
    """
    # Convert rows to tuples for hashing
    rows_as_tuples = map(tuple, arr)

    # Keep track of seen rows while preserving order
    seen = set()
    unique_indices = [
        i
        for i, row_tuple in enumerate(rows_as_tuples)
        if not (row_tuple in seen or seen.add(row_tuple))
    ]

    # Return array with only unique rows
    return arr[unique_indices]


def load_ohlcv_data(filepath: str) -> pd.DataFrame:
    arr = np.load(filepath, allow_pickle=True)
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    arr_deduplicated = deduplicate_rows(arr)
    if len(arr) != len(arr_deduplicated):
        dump_ohlcv_data(arr_deduplicated, filepath)
        print(
            f"Caught .npy file with duplicate rows: {filepath} Overwrote with deduplicated version."
        )
    return ensure_millis(pd.DataFrame(arr_deduplicated, columns=columns))


def get_days_in_between(start_day, end_day):
    date_format = "%Y-%m-%d"
    start_date = datetime.datetime.strptime(start_day[:10], date_format)
    end_date = datetime.datetime.strptime(end_day[:10], date_format)
    days = []
    current_date = start_date
    while current_date <= end_date:
        days.append(current_date.strftime(date_format))
        current_date += datetime.timedelta(days=1)
    return days


def fill_gaps_in_ohlcvs(df):
    interval = 60000
    new_timestamps = np.arange(df["timestamp"].iloc[0], df["timestamp"].iloc[-1] + interval, interval)
    new_df = df.set_index("timestamp").reindex(new_timestamps)
    new_df.close = new_df.close.ffill()
    for col in ["open", "high", "low"]:
        new_df[col] = new_df[col].fillna(new_df.close)
    new_df["volume"] = new_df["volume"].fillna(0.0)
    return new_df.reset_index().rename(columns={"index": "timestamp"})


def attempt_gap_fix_ohlcvs(df, symbol=None):
    interval = 60_000
    max_hours = 12
    max_gap = interval * 60 * max_hours
    greatest_gap = df.timestamp.diff().max()
    if pd.isna(greatest_gap) or greatest_gap == interval:
        return df
    if greatest_gap > max_gap:
        raise Exception(f"Huge gap in data for {symbol}: {greatest_gap/(1000*60*60)} hours.")
    if self.verbose:
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
    if "timestamp" not in df.columns:
        return df
    if df.timestamp.iloc[0] > 1e14:  # is microseconds
        df.timestamp /= 1000
    elif df.timestamp.iloc[0] > 1e11:  # is milliseconds
        pass
    else:  # is seconds
        df.timestamp *= 1000
    return df


class OHLCVManager:
    """
    Manages OHLCVs for multiple exchanges.
    """

    def __init__(
        self,
        exchange,
        start_date=None,
        end_date=None,
        cc=None,
        gap_tolerance_ohlcvs_minutes=120.0,
        verbose=True,
    ):
        self.exchange = "binanceusdm" if exchange == "binance" else exchange
        self.quote = "USDC" if exchange == "hyperliquid" else "USDT"
        self.start_date = "2020-01-01" if start_date is None else start_date
        self.end_date = format_end_date("now" if end_date is None else end_date)
        self.start_ts = date_to_ts(self.start_date)
        self.end_ts = date_to_ts(self.end_date)
        self.cc = cc
        self.cache_filepaths = {
            "markets": os.path.join("caches", self.exchange, "markets.json"),
            "ohlcvs": os.path.join("historical_data", f"ohlcvs_{self.exchange}"),
            "first_timestamps": os.path.join("caches", self.exchange, "first_timestamps.json"),
        }
        self.markets = None
        self.verbose = verbose
        self.max_requests_per_minute = {"": 120, "gateio": 60}
        self.request_timestamps = deque(maxlen=1000)  # for rate-limiting checks
        self.gap_tolerance_ohlcvs_minutes = gap_tolerance_ohlcvs_minutes

    def update_date_range(self, new_start_date=None, new_end_date=None):
        if new_start_date:
            if isinstance(new_start_date, (float, int)):
                self.start_date = ts_to_date_utc(new_start_date)
            elif isinstance(new_start_date, str):
                self.start_date = new_start_date
            else:
                raise Exception(f"invalid start date {new_start_date}")
            self.start_ts = date_to_ts(self.start_date)
        if new_end_date:
            if isinstance(new_end_date, (float, int)):
                self.end_date = ts_to_date_utc(new_end_date)
            elif isinstance(new_end_date, str):
                self.end_date = new_end_date
            else:
                raise Exception(f"invalid end date {new_end_date}")
            self.end_date = format_end_date(self.end_date)
            self.end_ts = date_to_ts(self.end_date)

    def get_symbol(self, coin):
        assert self.markets, "needs to call self.load_markets() first"
        return coin_to_symbol(
            coin,
            eligible_symbols={
                k for k in self.markets if self.markets[k]["swap"] and k.endswith(f":{self.quote}")
            },
            verbose=self.verbose,
        )

    def get_market_specific_settings(self, coin):
        mss = self.markets[self.get_symbol(coin)]
        mss["hedge_mode"] = True
        mss["maker_fee"] = mss["maker"]
        mss["taker_fee"] = mss["taker"]
        mss["c_mult"] = mss["contractSize"]
        mss["min_cost"] = mc if (mc := mss["limits"]["cost"]["min"]) is not None else 0.01
        mss["price_step"] = mss["precision"]["price"]
        mss["min_qty"] = max(
            lm if (lm := mss["limits"]["amount"]["min"]) is not None else 0.0,
            pm if (pm := mss["precision"]["amount"]) is not None else 0.0,
        )
        mss["qty_step"] = mss["precision"]["amount"]
        if self.exchange == "binanceusdm":
            pass
        elif self.exchange == "bybit":
            # ccxt reports incorrect fees for bybit perps
            mss["maker"] = mss["maker_fee"] = 0.0002
            mss["taker"] = mss["taker_fee"] = 0.00055
        elif self.exchange == "bitget":
            pass
        elif self.exchange == "gateio":
            # ccxt reports incorrect fees for gateio perps. Assume VIP0
            mss["maker"] = mss["maker_fee"] = 0.0002
            mss["taker"] = mss["taker_fee"] = 0.0005
        return mss

    def filter_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter dataframe to include only data within start_date and end_date (inclusive)"""
        if df.empty:
            return df
        return df[(df.timestamp >= self.start_ts) & (df.timestamp <= self.end_ts)].reset_index(
            drop=True
        )

    def has_coin(self, coin):
        symbol = self.get_symbol(coin)
        if not symbol:
            return False
        return True

    async def check_rate_limit(self):
        current_time = time()
        while self.request_timestamps and current_time - self.request_timestamps[0] > 60:
            self.request_timestamps.popleft()
        mrpm = (
            self.max_requests_per_minute[self.exchange]
            if self.exchange in self.max_requests_per_minute
            else self.max_requests_per_minute[""]
        )
        if len(self.request_timestamps) >= mrpm:
            sleep_time = 60 - (current_time - self.request_timestamps[0])
            if sleep_time > 0:
                if self.verbose:
                    logging.info(
                        f"{self.exchange} Rate limit reached, sleeping for {sleep_time:.2f} seconds"
                    )
                await asyncio.sleep(sleep_time)

        self.request_timestamps.append(current_time)

    async def get_ohlcvs(self, coin, start_date=None, end_date=None):
        """
        - Attempts to get ohlcvs for coin from cache.
        - If any data is missing, checks if it exists to download
        - If so, download.
        - Return ohlcvs.
        - If exchange unsupported,
        coin unsupported on exchange,
        or date range for coin not existing on exchange,
        return empty dataframe
        """
        if not self.markets:
            await self.load_markets()
        if not self.has_coin(coin):
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        if start_date or end_date:
            self.update_date_range(new_start_date=start_date, new_end_date=end_date)
        missing_days = await self.get_missing_days_ohlcvs(coin)
        if missing_days:
            await self.download_ohlcvs(coin)
        ohlcvs = await self.load_ohlcvs_from_cache(coin)
        ohlcvs.volume = ohlcvs.volume * ohlcvs.close  # use quote volume
        return ohlcvs

    async def get_start_date_modified(self, coin):
        fts = await self.get_first_timestamp(coin)
        return ts_to_date_utc(max(self.start_ts, fts))[:10]

    async def get_missing_days_ohlcvs(self, coin):
        start_date = await self.get_start_date_modified(coin)
        days = get_days_in_between(start_date, self.end_date)
        dirpath = os.path.join(self.cache_filepaths["ohlcvs"], coin)
        if not os.path.exists(dirpath):
            return days
        all_files = os.listdir(dirpath)
        return sorted([x for x in days if x + ".npy" not in all_files])

    async def download_ohlcvs(self, coin):
        if not self.markets:
            await self.load_markets()
        if not self.has_coin(coin):
            return
        if self.exchange == "binanceusdm":
            await self.download_ohlcvs_binance(coin)
        elif self.exchange == "bybit":
            await self.download_ohlcvs_bybit(coin)
        elif self.exchange == "bitget":
            await self.download_ohlcvs_bitget(coin)
        elif self.exchange == "gateio":
            if self.cc is None:
                self.load_cc()
            await self.download_ohlcvs_gateio(coin)

    def dump_ohlcvs_to_cache(self, coin):
        """
        Dumps new ohlcv data to cache if not already existing. Only whole days are dumped.
        """
        pass

    async def get_first_timestamp(self, coin):
        """
        Get first timestamp of available ohlcv data for given exchange & coin
        """
        if (fts := self.load_first_timestamp(coin)) not in [None, 0.0]:
            return fts
        if not self.markets:
            self.load_cc()
            await self.load_markets()
        if not self.has_coin(coin):
            self.dump_first_timestamp(coin, 0.0)
            return 0.0
        if self.exchange == "binanceusdm":
            # Fetches first by default
            ohlcvs = await self.cc.fetch_ohlcv(self.get_symbol(coin), since=1, timeframe="1d")
        elif self.exchange == "bybit":
            fts = await self.find_first_day_bybit(coin)
            return fts
        elif self.exchange == "gateio":
            # Data since 2018
            ohlcvs = await self.cc.fetch_ohlcv(
                self.get_symbol(coin), since=int(date_to_ts("2018-01-01")), timeframe="1d"
            )
            if not ohlcvs:
                ohlcvs = await self.cc.fetch_ohlcv(
                    self.get_symbol(coin), since=int(date_to_ts("2020-01-01")), timeframe="1d"
                )
        elif self.exchange == "bitget":
            fts = await self.find_first_day_bitget(coin)
            return fts
        if ohlcvs:
            fts = ohlcvs[0][0]
        else:
            fts = 0.0
        self.dump_first_timestamp(coin, fts)
        return fts

    def load_cc(self):
        if self.cc is None:
            self.cc = getattr(ccxt, self.exchange)({"enableRateLimit": True})
            self.cc.options["defaultType"] = "swap"

    async def load_markets(self):
        self.load_cc()
        self.markets = self.load_markets_from_cache()
        if self.markets:
            return
        self.markets = await self.cc.load_markets()
        self.dump_markets_to_cache()

    def load_markets_from_cache(self, max_age_ms=1000 * 60 * 60 * 24):
        try:
            if os.path.exists(self.cache_filepaths["markets"]):
                if utc_ms() - get_file_mod_utc(self.cache_filepaths["markets"]) < max_age_ms:
                    markets = json.load(open(self.cache_filepaths["markets"]))
                    if self.verbose:
                        logging.info(f"{self.exchange} Loaded markets from cache")
                    return markets
            return {}
        except Exception as e:
            logging.error(f"Error with {get_function_name()} {e}")
            return {}

    def dump_markets_to_cache(self):
        if self.markets:
            try:
                json.dump(self.markets, open(make_get_filepath(self.cache_filepaths["markets"]), "w"))
                if self.verbose:
                    logging.info(f"{self.exchange} Dumped markets to cache")
            except Exception as e:
                logging.error(f"Error with {get_function_name()} {e}")

    async def load_ohlcvs_from_cache(self, coin):
        """
        Loads any cached ohlcv data for exchange, coin and date range from cache
        and *strictly* enforces no gaps. If any gap is found, return empty.
        """
        dirpath = os.path.join(self.cache_filepaths["ohlcvs"], coin, "")
        if not os.path.exists(dirpath):
            return pd.DataFrame()

        all_files = sorted([f for f in os.listdir(dirpath) if f.endswith(".npy")])
        all_days = get_days_in_between(self.start_date, self.end_date)
        all_months = sorted(set([x[:7] for x in all_days]))

        # Load month files first
        files_to_load = [x for x in all_files if x.replace(".npy", "") in all_months]
        # Add day files (exclude if they were loaded already as a month)
        files_to_load += [
            x for x in all_files if x.replace(".npy", "") in all_days and x not in files_to_load
        ]

        dfs = []
        for f in files_to_load:
            try:
                filepath = os.path.join(dirpath, f)
                df_part = load_ohlcv_data(filepath)
                dfs.append(df_part)
            except Exception as e:
                logging.error(f"Error loading file {f}: {e}")

        if not dfs:
            return pd.DataFrame()

        # Concatenate, drop duplicates, sort by timestamp
        df = (
            pd.concat(dfs)
            .drop_duplicates("timestamp")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        # ----------------------------------------------------------------------
        # 1) Clip to [start_ts, end_ts] and return
        # ----------------------------------------------------------------------
        df = self.filter_date_range(df)

        # ----------------------------------------------------------------------
        # 2) Gap check with tolerance: if intervals != 60000 for any bar, return empty.
        # ----------------------------------------------------------------------
        intervals = np.diff(df["timestamp"].values)
        # If any interval is not exactly 60000, we have a gap.
        if (intervals != 60000).any():
            greatest_gap = int(intervals.max() / 60000.0)
            if greatest_gap > self.gap_tolerance_ohlcvs_minutes:
                logging.warning(
                    f"[{self.exchange}] Gaps detected in {coin} OHLCV data. Greatest gap: {greatest_gap} minutes. Returning empty DataFrame."
                )
                return pd.DataFrame(columns=df.columns)
            else:
                df = fill_gaps_in_ohlcvs(df)
        return df

    def copy_ohlcvs_from_old_dir(self, new_dirpath, old_dirpath, missing_days, coin):
        symbolf = self.get_symbol(coin).replace("/USDT:", "")
        files_copied = 0
        if os.path.exists(old_dirpath):
            for d0 in os.listdir(old_dirpath):
                if d0.endswith(".npy") and d0[:10] in missing_days:
                    src = os.path.join(old_dirpath, d0)
                    dst = os.path.join(new_dirpath, d0)
                    if os.path.exists(dst):
                        continue
                    try:
                        shutil.copy(src, dst)
                        files_copied += 1
                    except Exception as e:
                        logging.error(f"{self.exchange} error copying {src} -> {dst} {e}")
        if files_copied:
            logging.info(
                f"{self.exchange} copied {files_copied} files from {old_dirpath} to {new_dirpath}"
            )
            return True
        else:
            return False

    async def download_ohlcvs_binance(self, coin: str):
        # Uses Binance's data archives via binance.vision
        symbolf = self.get_symbol(coin).replace("/USDT:", "")
        dirpath = make_get_filepath(os.path.join(self.cache_filepaths["ohlcvs"], coin, ""))
        base_url = "https://data.binance.vision/data/futures/um/"
        missing_days = await self.get_missing_days_ohlcvs(coin)

        # Copy from old directory first
        old_dirpath = f"historical_data/ohlcvs_futures/{symbolf}/"
        if self.copy_ohlcvs_from_old_dir(dirpath, old_dirpath, missing_days, coin):
            missing_days = await self.get_missing_days_ohlcvs(coin)
            if not missing_days:
                return

        # Download monthy first (there may be gaps)
        month_now = ts_to_date_utc(utc_ms())[:7]
        missing_months = sorted({x[:7] for x in missing_days if x[:7] != month_now})
        tasks = []
        for month in missing_months:
            fpath = os.path.join(dirpath, month + ".npy")
            if not os.path.exists(fpath):
                url = f"{base_url}monthly/klines/{symbolf}/1m/{symbolf}-1m-{month}.zip"
                await self.check_rate_limit()
                tasks.append(asyncio.create_task(self.download_single_binance(url, fpath)))
        for task in tasks:
            await task

        # Convert any monthly data to daily data
        for f in os.listdir(dirpath):
            if len(f) == 11:
                df = load_ohlcv_data(os.path.join(dirpath, f))

                df.loc[:, "datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df.set_index("datetime", inplace=True)

                daily_groups = df.groupby(df.index.date)
                n_days_dumped = 0
                for date, daily_data in daily_groups:
                    if len(daily_data) == 1440:
                        fpath = str(date) + ".npy"
                        d_fpath = os.path.join(dirpath, fpath)
                        if not os.path.exists(d_fpath):
                            n_days_dumped += 1
                            dump_ohlcv_data(daily_data, d_fpath)
                    else:
                        logging.info(
                            f"binanceusdm incomplete daily data for {coin} {date} {len(daily_data)}"
                        )
                if n_days_dumped:
                    logging.info(f"binanceusdm dumped {n_days_dumped} daily files for {coin} {f}")
                m_fpath = os.path.join(dirpath, f)
                logging.info(f"binanceusdm removing {m_fpath}")
                os.remove(m_fpath)

        # Download missing daily
        missing_days = await self.get_missing_days_ohlcvs(coin)
        tasks = []
        for day in missing_days:
            fpath = os.path.join(dirpath, day + ".npy")
            if not os.path.exists(fpath):
                url = base_url + f"daily/klines/{symbolf}/1m/{symbolf}-1m-{day}.zip"
                await self.check_rate_limit()
                tasks.append(asyncio.create_task(self.download_single_binance(url, fpath)))
        for task in tasks:
            await task

    async def download_single_binance(self, url: str, fpath: str):
        try:
            csv = await get_zip_binance(url)
            if not csv.empty:
                dump_ohlcv_data(ensure_millis(csv), fpath)
                if self.verbose:
                    logging.info(f"binanceusdm Dumped data {fpath}")
        except Exception as e:
            logging.error(f"binanceusdm Failed to download {url}: {e}")
            traceback.print_exc()

    async def download_ohlcvs_bybit(self, coin: str):
        # Bybit has public data archives
        missing_days = await self.get_missing_days_ohlcvs(coin)
        if not missing_days:
            return
        symbolf = self.get_symbol(coin).replace("/USDT:", "")
        dirpath = make_get_filepath(os.path.join(self.cache_filepaths["ohlcvs"], coin, ""))

        # Copy from old directory first
        old_dirpath = f"historical_data/ohlcvs_bybit/{symbolf}/"
        if self.copy_ohlcvs_from_old_dir(dirpath, old_dirpath, missing_days, coin):
            missing_days = await self.get_missing_days_ohlcvs(coin)
            if not missing_days:
                return

        # Bybit public data: "https://public.bybit.com/trading/"
        base_url = "https://public.bybit.com/trading/"
        webpage = urlopen(f"{base_url}{symbolf}/").read().decode()

        filenames = [
            f"{symbolf}{day}.csv.gz" for day in missing_days if f"{symbolf}{day}.csv.gz" in webpage
        ]
        # Download concurrently
        async with aiohttp.ClientSession() as session:
            tasks = []
            for fn in filenames:
                url = f"{base_url}{symbolf}/{fn}"
                day = fn[-17:-7]
                await self.check_rate_limit()
                tasks.append(
                    asyncio.create_task(self.download_single_bybit(session, url, dirpath, day))
                )
            results = await asyncio.gather(*tasks, return_exceptions=True)

    async def find_first_day_bybit(self, coin: str, webpage=None) -> float:
        symbolf = self.get_symbol(coin).replace("/USDT:", "")
        # Bybit public data: "https://public.bybit.com/trading/"
        base_url = "https://public.bybit.com/trading/"
        if webpage is None:
            webpage = urlopen(f"{base_url}{symbolf}/").read().decode()
        dates = [date for x in webpage.split(".csv.gz") if is_valid_date((date := x[-10:]))]
        first_ts = date_to_ts(sorted(dates)[0])
        self.dump_first_timestamp(coin, first_ts)
        return first_ts

    async def download_single_bybit(self, session, url: str, dirpath: str, day: str) -> pd.DataFrame:
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
            fpath = os.path.join(dirpath, day + ".npy")
            dump_ohlcv_data(
                ensure_millis(ohlcvs[["timestamp", "open", "high", "low", "close", "volume"]]),
                fpath,
            )
            if self.verbose:
                logging.info(f"bybit Dumped {fpath}")
        except Exception as e:
            logging.error(f"bybit error {url}: {e}")
            traceback.print_exc()
            return pd.DataFrame()

    async def download_ohlcvs_bitget(self, coin: str):
        # Bitget has public data archives
        fts = await self.find_first_day_bitget(coin)
        if fts == 0.0:
            return
        first_day = ts_to_date_utc(fts)
        missing_days = await self.get_missing_days_ohlcvs(coin)
        if not missing_days:
            return
        symbolf = self.get_symbol(coin).replace("/USDT:", "")
        if not symbolf:
            return
        dirpath = make_get_filepath(os.path.join(self.cache_filepaths["ohlcvs"], coin, ""))
        base_url = "https://img.bitgetimg.com/online/kline/"
        # Download daily
        tasks = []
        for day in sorted(missing_days):
            fpath = day + ".npy"
            await self.check_rate_limit()
            tasks.append(
                asyncio.create_task(
                    self.download_single_bitget(
                        base_url, symbolf, day, os.path.join(dirpath, day + ".npy")
                    )
                )
            )
        for task in tasks:
            try:
                await task
            except Exception as e:
                logging.error(f"bitget Error with downloader for {coin} {e}")
                # traceback.print_exc()

    def get_url_bitget(self, base_url, symbolf, day):
        if day <= "2024-04-18":
            return f"{base_url}{symbolf}/{symbolf}_UMCBL_1min_{day.replace('-', '')}.zip"
        else:
            return f"{base_url}{symbolf}/UMCBL/{day.replace('-', '')}.zip"

    async def download_single_bitget(self, base_url, symbolf, day, fpath):
        url = self.get_url_bitget(base_url, symbolf, day)
        res = await get_zip_bitget(url)
        dump_ohlcv_data(ensure_millis(res), fpath)
        if self.verbose:
            logging.info(f"bitget Dumped daily data {fpath}")

    async def find_first_day_bitget(self, coin: str, start_year=2020) -> float:
        """Find first day where data is available for a given symbol"""
        if fts := self.load_first_timestamp(coin):
            return fts
        if not self.markets:
            await self.load_markets()
        symbol = self.get_symbol(coin).replace("/USDT:", "")
        if not symbol:
            fts = 0.0
            self.dump_first_timestamp(coin, fts)
            return fts
        base_url = "https://img.bitgetimg.com/online/kline/"
        start = datetime.datetime(start_year, 1, 1)
        end = datetime.datetime.now()
        earliest = None

        while start <= end:
            mid = start + (end - start) // 2
            date_str = mid.strftime("%Y%m%d")
            url = self.get_url_bitget(base_url, symbol, date_str)

            try:
                await self.check_rate_limit()
                async with aiohttp.ClientSession() as session:
                    async with session.head(url) as response:
                        if self.verbose:
                            logging.info(
                                f"bitget, searching for first day of data for {symbol} {str(mid)[:10]}"
                            )
                        if response.status == 200:
                            earliest = mid
                            end = mid - datetime.timedelta(days=1)
                        else:
                            start = mid + datetime.timedelta(days=1)
            except Exception as e:
                start = mid + datetime.timedelta(days=1)

        if earliest:
            # Verify by checking the previous day
            prev_day = earliest - datetime.timedelta(days=1)
            prev_url = self.get_url_bitget(base_url, symbol, prev_day.strftime("%Y%m%d"))
            try:
                await check_rate_limit()
                async with aiohttp.ClientSession() as session:
                    async with session.head(prev_url) as response:
                        if response.status == 200:
                            earliest = prev_day
            except Exception:
                pass
            if self.verbose:
                logging.info(f"Bitget, found first day for {symbol}: {earliest.strftime('%Y-%m-%d')}")
            # dump cache
            fts = date_to_ts(earliest.strftime("%Y-%m-%d"))
            self.dump_first_timestamp(coin, fts)
            return fts
        return None

    async def download_ohlcvs_gateio(self, coin: str):
        # GateIO doesn't have public data archives, but has ohlcvs via REST API
        missing_days = await self.get_missing_days_ohlcvs(coin)
        if not missing_days:
            return
        if self.cc is None:
            self.load_cc()
        dirpath = make_get_filepath(os.path.join(self.cache_filepaths["ohlcvs"], coin, ""))
        symbol = self.get_symbol(coin)

        # Instead of downloading in small chunks, do a single fetch for each day
        # This avoids multiple .fetch_ohlcv() calls that might exceed rate limits.
        tasks = []
        for day in missing_days:
            await self.check_rate_limit()
            tasks.append(asyncio.create_task(self.fetch_and_save_day_gateio(symbol, day, dirpath)))
        for task in tasks:
            await task

    async def fetch_and_save_day_gateio(self, symbol: str, day: str, dirpath: str):
        """
        Fetches one full day of OHLCV data from GateIO with a single call,
        then dumps it to disk. Uses self.check_rate_limit() to avoid exceeding
        the per-minute request cap.
        """
        fpath = os.path.join(dirpath, f"{day}.npy")
        start_ts_day = date_to_ts(day)  # 00:00:00 UTC of 'day'
        end_ts_day = start_ts_day + 24 * 60 * 60 * 1000  # next 24 hours
        interval = "1m"

        # GateIO typically allows up to 1440+ limit for 1m timeframe in one call
        limit = 1500
        ohlcvs = await self.cc.fetch_ohlcv(
            symbol, timeframe=interval, since=start_ts_day, limit=limit
        )
        if not ohlcvs:
            # No data returned; skip
            if self.verbose:
                logging.info(f"No data returned for GateIO {symbol} {day}")
            return

        # Convert to DataFrame
        df_day = pd.DataFrame(ohlcvs, columns=["timestamp", "open", "high", "low", "close", "volume"])
        # Filter exactly for the given day (start_ts_day <= ts < end_ts_day)
        df_day = df_day[
            (df_day.timestamp >= start_ts_day) & (df_day.timestamp < end_ts_day)
        ].reset_index(drop=True)

        # Convert volume from quote to base volume if needed
        # (Gate.io's swap markets typically return quote-volume in "volume")
        # Adjust if your usage needs base volume. E.g.:
        df_day["volume"] = df_day["volume"] / df_day["close"]

        # Dump final day data only if is a full day
        if len(df_day) == 1440:
            dump_ohlcv_data(ensure_millis(df_day), fpath)
            if self.verbose:
                logging.info(f"gateio Dumped daily OHLCV data for {symbol} to {fpath}")

    def load_first_timestamp(self, coin):
        if os.path.exists(self.cache_filepaths["first_timestamps"]):
            try:
                ftss = json.load(open(self.cache_filepaths["first_timestamps"]))
                if coin in ftss:
                    return ftss[coin]
            except Exception as e:
                logging.error(f"Error loading {self.cache_filepaths['first_timestamps']} {e}")

    def dump_first_timestamp(self, coin, fts):
        try:
            fpath = self.cache_filepaths["first_timestamps"]
            if os.path.exists(fpath):
                try:
                    ftss = json.load(open(fpath))
                except Exception as e0:
                    logging.error(f"Error loading {fpath} {e0}")
                    ftss = {}
            else:
                make_get_filepath(fpath)
                ftss = {}
            ftss[coin] = fts
            json.dump(ftss, open(fpath, "w"), indent=True, sort_keys=True)
            if self.verbose:
                logging.info(f"{self.exchange} Dumped {fpath}")
        except Exception as e:
            logging.error(f"Error with {get_function_name()} {e}")


async def prepare_hlcvs(config: dict, exchange: str):
    coins = sorted(
        set([symbol_to_coin(c) for c in config["live"]["approved_coins"]["long"]])
        | set([symbol_to_coin(c) for c in config["live"]["approved_coins"]["short"]])
    )
    if exchange == "binance":
        exchange = "binanceusdm"
    start_date = config["backtest"]["start_date"]
    end_date = format_end_date(config["backtest"]["end_date"])
    om = OHLCVManager(
        exchange,
        start_date,
        end_date,
        gap_tolerance_ohlcvs_minutes=config["backtest"]["gap_tolerance_ohlcvs_minutes"],
    )
    try:
        return await prepare_hlcvs_internal(config, coins, exchange, start_date, end_date, om)
    finally:
        if om.cc:
            await om.cc.close()


async def prepare_hlcvs_internal(config, coins, exchange, start_date, end_date, om):
    end_ts = date_to_ts(end_date)
    minimum_coin_age_days = config["live"]["minimum_coin_age_days"]
    interval_ms = 60000

    first_timestamps_unified = await get_first_timestamps_unified(coins)

    # Create cache directory if it doesn't exist
    cache_dir = Path(f"./caches/hlcvs_data/{uuid4().hex[:16]}")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # First pass: Download data and store metadata
    coin_metadata = {}

    valid_coins = {}
    global_start_time = float("inf")
    global_end_time = float("-inf")
    await om.load_markets()
    min_coin_age_ms = 1000 * 60 * 60 * 24 * minimum_coin_age_days

    # First pass: Download and save data, collect metadata
    for coin in coins:
        adjusted_start_ts = date_to_ts(start_date)
        if not om.has_coin(coin):
            logging.info(f"{exchange} coin {coin} missing, skipping")
            continue
        if coin not in first_timestamps_unified:
            logging.info(f"coin {coin} missing from first_timestamps_unified, skipping")
            continue
        if minimum_coin_age_days > 0.0:
            first_ts = await om.get_first_timestamp(coin)
            if first_ts >= end_ts:
                logging.info(
                    f"{exchange} Coin {coin} too young, start date {ts_to_date_utc(first_ts)}. Skipping"
                )
                continue
            first_ts_plus_min_coin_age = first_timestamps_unified[coin] + min_coin_age_ms
            if first_ts_plus_min_coin_age >= end_ts:
                logging.info(
                    f"{exchange} Coin {coin}: Not traded due to min_coin_age {int(minimum_coin_age_days)} days"
                    f"{ts_to_date_utc(first_ts_plus_min_coin_age)}. Skipping"
                )
                continue
            new_adjusted_start_ts = max(first_timestamps_unified[coin] + min_coin_age_ms, first_ts)
            if new_adjusted_start_ts > adjusted_start_ts:
                logging.info(
                    f"{exchange} Coin {coin}: Adjusting start date from {start_date} "
                    f"to {ts_to_date_utc(new_adjusted_start_ts)}"
                )
                adjusted_start_ts = new_adjusted_start_ts
        try:
            om.update_date_range(adjusted_start_ts)
            df = await om.get_ohlcvs(coin)
            data = df[["timestamp", "high", "low", "close", "volume"]].values
        except Exception as e:
            logging.error(f"error with get_ohlcvs for {coin} {e}. Skipping")
            traceback.print_exc()
            continue
        if len(data) == 0:
            continue

        assert (np.diff(data[:, 0]) == interval_ms).all(), f"gaps in hlcv data {coin}"

        # Save data to disk
        file_path = cache_dir / f"{coin}.npy"
        dump_ohlcv_data(data, file_path)

        # Update metadata
        coin_metadata[coin] = {
            "start_time": int(data[0, 0]),
            "end_time": int(data[-1, 0]),
            "length": len(data),
        }

        valid_coins[coin] = file_path
        global_start_time = min(global_start_time, data[0, 0])
        global_end_time = max(global_end_time, data[-1, 0])

    if not valid_coins:
        raise ValueError("No valid coins found with data")

    # Calculate dimensions for the unified array
    n_timesteps = int((global_end_time - global_start_time) / interval_ms) + 1
    n_coins = len(valid_coins)

    # Create the timestamp array
    timestamps = np.arange(global_start_time, global_end_time + interval_ms, interval_ms)

    # Pre-allocate the unified array
    unified_array = np.zeros((n_timesteps, n_coins, 4))

    # Second pass: Load data from disk and populate the unified array
    logging.info(f"{exchange} Unifying data for {len(valid_coins)} coins into single numpy array...")
    for i, coin in enumerate(tqdm(valid_coins, desc="Processing coins", unit="coin")):
        file_path = valid_coins[coin]
        ohlcv = np.load(file_path)

        # Calculate indices
        start_idx = int((ohlcv[0, 0] - global_start_time) / interval_ms)
        end_idx = start_idx + len(ohlcv)

        # Extract and process data
        coin_data = ohlcv[:, 1:]

        # Place the data in the unified array
        unified_array[start_idx:end_idx, i, :] = coin_data

        # Front-fill
        if start_idx > 0:
            unified_array[:start_idx, i, :3] = coin_data[0, 2]

        # Back-fill
        if end_idx < n_timesteps:
            unified_array[end_idx:, i, :3] = coin_data[-1, 2]

        # Clean up temporary file
        os.remove(file_path)

    # Clean up cache directory if empty
    try:
        os.rmdir(cache_dir)
    except OSError:
        pass
    mss = {coin: om.get_market_specific_settings(coin) for coin in sorted(valid_coins)}
    return mss, timestamps, unified_array


async def prepare_hlcvs_combined(config):
    """
    Public function that sets up any needed resources,
    calls the internal implementation, and ensures
    ccxt connections are closed in a finally block.
    """
    # Create or load the OHLCVManager dict
    exchanges_to_consider = [
        "binanceusdm" if e == "binance" else e for e in config["backtest"]["exchanges"]
    ]
    om_dict = {}
    for ex in exchanges_to_consider:
        om = OHLCVManager(
            ex,
            config["backtest"]["start_date"],
            config["backtest"]["end_date"],
            gap_tolerance_ohlcvs_minutes=config["backtest"]["gap_tolerance_ohlcvs_minutes"],
        )
        # await om.load_markets()  # if you want to do this up front
        om_dict[ex] = om

    try:
        return await _prepare_hlcvs_combined_impl(config, om_dict)
    finally:
        # Cleanly close all ccxt sessions
        for om in om_dict.values():
            if om.cc:
                await om.cc.close()


async def _prepare_hlcvs_combined_impl(config, om_dict):
    """
    Amalgamates data from different exchanges for each coin in config, then unifies them into a single
    numpy array with shape (n_timestamps, n_coins, 4). The final data per coin is chosen using:

        1) Filter out exchanges that don't fully cover [start_date, end_date]
        2) Among the remaining, pick the exchange with the fewest data gaps
        3) If still tied, pick the exchange with the highest total volume

    Returns:
        mss: dict of coin -> market_specific_settings from the chosen exchange
        timestamps: 1D numpy array of all timestamps (1min granularity) covering the entire combined range
        unified_array: 3D numpy array with shape (len(timestamps), n_coins, 4),
                       where the last dimension is [high, low, close, volume].
                       Price fields are forward-filled; volume is 0-filled for missing data.
    """
    # ---------------------------------------------------------------
    # 0) Define or load relevant info from config
    # ---------------------------------------------------------------
    start_date = config["backtest"]["start_date"]
    end_date = format_end_date(config["backtest"]["end_date"])
    start_ts = date_to_ts(start_date)
    end_ts = date_to_ts(end_date)

    # Pull out all coins from config:
    coins = sorted(
        set([symbol_to_coin(c) for c in config["live"]["approved_coins"]["long"]])
        | set([symbol_to_coin(c) for c in config["live"]["approved_coins"]["short"]])
    )

    # If your config includes a list of exchanges, grab it; else pick a default set:
    exchanges_to_consider = [
        "binanceusdm" if e == "binance" else e for e in config["backtest"]["exchanges"]
    ]

    # Minimum coin age handling (same approach as prepare_hlcvs)
    min_coin_age_days = config["live"].get("minimum_coin_age_days", 0.0)
    min_coin_age_ms = int(min_coin_age_days * 24 * 60 * 60 * 1000)

    # First timestamps from your pre-cached or dynamically fetched data
    # (some procedures rely on e.g. get_first_timestamps_unified())
    first_timestamps_unified = await get_first_timestamps_unified(coins)

    for ex in exchanges_to_consider:
        await om_dict[ex].load_markets()

    # ---------------------------------------------------------------
    # 2) For each coin, gather 1m data from all exchanges, filter/choose best
    # ---------------------------------------------------------------
    chosen_data_per_coin = {}  # coin -> pd.DataFrame of final chosen data
    chosen_mss_per_coin = {}  # coin -> market_specific_settings from chosen exchange

    for coin in coins:
        # If the global "first_timestamps_unified" says we have no data for coin, skip immediately
        coin_fts = first_timestamps_unified.get(coin, 0.0)
        if coin_fts == 0.0:
            logging.info(f"Skipping coin {coin}, no first timestamp recorded.")
            continue

        # Check if coin is "too young": first_ts + min_coin_age >= end_ts
        # meaning there's effectively no eligible window to trade/backtest
        if coin_fts + min_coin_age_ms >= end_ts:
            logging.info(
                f"Skipping coin {coin}: it does not satisfy the minimum_coin_age_days = {min_coin_age_days}"
            )
            continue

        # The earliest time we can start from, given coin's first trade time plus coin age
        effective_start_ts = max(start_ts, coin_fts + min_coin_age_ms)
        if effective_start_ts >= end_ts:
            # No coverage needed or possible
            continue

        # >>> Instead of a normal for-loop over exchanges, do concurrent tasks:
        tasks = []
        for ex in exchanges_to_consider:
            tasks.append(
                asyncio.create_task(
                    fetch_data_for_coin_and_exchange(
                        coin, ex, om_dict[ex], effective_start_ts, end_ts
                    )
                )
            )
        # Gather results concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None/Exceptions, build exchange_candidates
        exchange_candidates = []
        for r in results:
            if r is None or isinstance(r, Exception):
                continue
            ex, df, coverage_count, gap_count, total_volume = r
            exchange_candidates.append((ex, df, coverage_count, gap_count, total_volume))

        if not exchange_candidates:
            logging.info(f"No exchange data found at all for coin {coin}. Skipping.")
            continue

        # Now pick the "best" exchange (per your partial-coverage logic):
        if len(exchange_candidates) == 1:
            best_exchange, best_df, best_cov, best_gaps, best_vol = exchange_candidates[0]
        else:
            # Sort by coverage desc, gap_count asc, volume desc
            exchange_candidates.sort(key=lambda x: (x[2], -x[3], x[4]), reverse=True)
            best_exchange, best_df, best_cov, best_gaps, best_vol = exchange_candidates[0]
        logging.info(f"{coin} exchange preference: {[x[0] for x in exchange_candidates]}")

        chosen_data_per_coin[coin] = best_df
        chosen_mss_per_coin[coin] = om_dict[best_exchange].get_market_specific_settings(coin)
        chosen_mss_per_coin[coin]["exchange"] = best_exchange
    # ---------------------------------------------------------------
    # If no coins survived, raise error
    # ---------------------------------------------------------------
    if not chosen_data_per_coin:
        raise ValueError("No coin data found on any exchange for the requested date range.")

    # ---------------------------------------------------------------
    # 6) Unify across coins into a single (n_timestamps, n_coins, 4) array
    #    We'll unify on 1m timestamps from the earliest to latest across all chosen coins
    # ---------------------------------------------------------------
    global_start_time = min(df.timestamp.iloc[0] for df in chosen_data_per_coin.values())
    global_end_time = max(df.timestamp.iloc[-1] for df in chosen_data_per_coin.values())

    timestamps = np.arange(global_start_time, global_end_time + 60000, 60000)
    n_timesteps = len(timestamps)
    valid_coins = sorted(chosen_data_per_coin.keys())
    n_coins = len(valid_coins)
    # use at most last 60 days of date range to compute volume ratios
    start_date_for_volume_ratios = ts_to_date_utc(
        max(global_start_time, global_end_time - 1000 * 60 * 60 * 24 * 60)
    )
    end_date_for_volume_ratios = ts_to_date_utc(global_end_time)

    exchanges_with_data = sorted(set([chosen_mss_per_coin[coin]["exchange"] for coin in valid_coins]))
    exchange_volume_ratios = await compute_exchange_volume_ratios(
        exchanges_with_data,
        valid_coins,
        start_date_for_volume_ratios,
        end_date_for_volume_ratios,
        {ex: om_dict[ex] for ex in exchanges_with_data},
    )
    exchanges_counts = defaultdict(int)
    for coin in chosen_mss_per_coin:
        exchanges_counts[chosen_mss_per_coin[coin]["exchange"]] += 1
    reference_exchange = sorted(exchanges_counts.items(), key=lambda x: x[1])[-1][0]
    exchange_volume_ratios_mapped = defaultdict(dict)
    if len(exchanges_counts) == 1:
        exchange_volume_ratios_mapped[reference_exchange][reference_exchange] = 1.0
    else:
        for ex0, ex1 in exchange_volume_ratios:
            exchange_volume_ratios_mapped[ex0][ex1] = 1 / exchange_volume_ratios[(ex0, ex1)]
            exchange_volume_ratios_mapped[ex1][ex0] = exchange_volume_ratios[(ex0, ex1)]
            exchange_volume_ratios_mapped[ex1][ex1] = 1.0
            exchange_volume_ratios_mapped[ex0][ex0] = 1.0

    pprint.pprint(dict(exchange_volume_ratios_mapped))

    # We'll store [high, low, close, volume] in the last dimension
    unified_array = np.zeros((n_timesteps, n_coins, 4), dtype=np.float64)

    # For each coin i, reindex its DataFrame onto the full timestamps
    for i, coin in enumerate(valid_coins):
        df = chosen_data_per_coin[coin].copy()

        # Reindex on the global minute timestamps
        df = df.set_index("timestamp").reindex(timestamps)

        # Forward fill 'close' for all missing rows, then backward fill any leading edge
        df["close"] = df["close"].ffill().bfill()

        # For O/H/L, fill with whatever the 'close' ended up being
        df["open"] = df["open"].fillna(df["close"])
        df["high"] = df["high"].fillna(df["close"])
        df["low"] = df["low"].fillna(df["close"])

        # Fill volume with 0.0 for missing bars, then apply scaling factor
        df["volume"] = df["volume"].fillna(0.0)
        exchange_for_this_coin = chosen_mss_per_coin[coin]["exchange"]
        scaling_factor = exchange_volume_ratios_mapped[exchange_for_this_coin][reference_exchange]
        df["volume"] *= scaling_factor

        # Now extract columns in correct order
        coin_data = df[["high", "low", "close", "volume"]].values
        unified_array[:, i, :] = coin_data

    # ---------------------------------------------------------------
    # 7) Cleanup: close all ccxt clients if needed
    # ---------------------------------------------------------------
    for om in om_dict.values():
        if om.cc:
            await om.cc.close()

    # ---------------------------------------------------------------
    # Return final:
    #   - chosen_mss_per_coin: dict coin-> market settings from the chosen exchange
    #   - timestamps: 1D array of all unified timestamps
    #   - unified_array: shape (n_timestamps, n_coins, 4) => [H, L, C, V]
    # ---------------------------------------------------------------
    return chosen_mss_per_coin, timestamps, unified_array


async def fetch_data_for_coin_and_exchange(
    coin: str, ex: str, om: OHLCVManager, effective_start_ts: int, end_ts: int
):
    """
    Fetch data for (coin, ex) between [effective_start_ts, end_ts].
    Returns (ex, df, coverage_count, gap_count, total_volume), where:
        - ex:                the exchange name
        - df:                the OHLCV dataframe
        - coverage_count:    total number of rows in df
        - gap_count:         sum of missing minutes across all gaps
        - total_volume:      sum of 'volume' column (within the timeframe)
    """

    # Check if coin is listed on this exchange
    if not om.has_coin(coin):
        return None

    # Adjust the manager's date range to [effective_start_ts, end_ts]
    om.update_date_range(effective_start_ts, end_ts)

    try:
        # Get the DataFrame of 1m OHLCVs
        df = await om.get_ohlcvs(coin)
    except Exception as e:
        logging.warning(f"Error retrieving {coin} from {ex}: {e}")
        return None

    if df.empty:
        return None

    # Filter strictly to [effective_start_ts, end_ts]
    df = df[(df.timestamp >= effective_start_ts) & (df.timestamp <= end_ts)].reset_index(drop=True)
    if df.empty:
        return None

    # coverage_count = total number of 1m bars in df
    coverage_count = len(df)

    # ------------------------------------------------------------------
    # 1) Compute sum of all missing minutes (gap_count)
    # ------------------------------------------------------------------
    # For each consecutive pair, the difference in timestamps should be 60000 ms.
    # If it's bigger, we measure how many 1-minute bars are missing.
    intervals = np.diff(df["timestamp"].values)

    gap_count = sum(
        (gap // 60000) - 1  # e.g. if gap is 5 minutes => 5 - 1 = 4 missing bars
        for gap in intervals
        if gap > 60000
    )

    # total_volume = sum of volume column
    total_volume = df["volume"].sum()

    return (ex, df, coverage_count, gap_count, total_volume)


async def compute_exchange_volume_ratios(
    exchanges: List[str],
    coins: List[str],
    start_date: str,
    end_date: str,
    om_dict: Dict[str, "OHLCVManager"] = None,
) -> Dict[Tuple[str, str], float]:
    """
    Gathers daily volume for each coin on each exchange,
    filters out incomplete days (days missing from any exchange),
    and then computes pairwise volume ratios (ex0, ex1) = sumVol(ex0) / sumVol(ex1).
    Finally, it averages those ratios across all coins.

    :param exchanges: list of exchange names (e.g. ["binanceusdm", "bybit"]).
    :param coins:     list of coins (e.g. ["BTC", "ETH"]).
    :param start_date: "YYYY-MM-DD" inclusive
    :param end_date:   "YYYY-MM-DD" inclusive
    :param om_dict:   dict of {exchange_name -> OHLCVManager}, already initialized
    :return: dict {(ex0, ex1): average_ratio}, where ex0 < ex1 in alphabetical order, for example
    """
    # -------------------------------------------------------
    # 1) Build all pairs of exchanges
    # -------------------------------------------------------
    if om_dict is None:
        om_dict = {ex: OHLCVManager(ex, start_date, end_date) for ex in exchanges}
        await asyncio.gather(*[om_dict[ex].load_markets() for ex in om_dict])
    assert all([ex in om_dict for ex in exchanges])
    exchange_pairs = []
    for i, ex0 in enumerate(sorted(exchanges)):
        for ex1 in exchanges[i + 1 :]:
            # (Optional) sort them or keep them as-is
            # We'll just keep them in the (ex0, ex1) order for clarity
            exchange_pairs.append((ex0, ex1))

    # -------------------------------------------------------
    # 2) For each coin, gather data from all exchanges
    # -------------------------------------------------------
    # We'll store: all_data[coin][(ex0, ex1)] = ratio_of_volumes_for_that_coin
    all_data = {}

    for coin in coins:
        # If coin does not exist on ALL exchanges, skip
        if not all(om_dict[ex].has_coin(coin) for ex in exchanges):
            continue

        # Gather concurrent tasks => each exchange's DF for that coin
        tasks = []
        for ex in exchanges:
            om = om_dict[ex]
            om.update_date_range(start_date, end_date)
            tasks.append(
                om.get_ohlcvs(coin)
            )  # returns a DataFrame: [timestamp, open, high, low, close, volume]

        dfs = await asyncio.gather(*tasks, return_exceptions=True)
        # Filter out any exceptions or empty data
        # We'll keep them in the same order as `exchanges`
        for i, df in enumerate(dfs):
            if isinstance(df, Exception) or df is None or df.empty:
                dfs[i] = pd.DataFrame()  # mark as empty

        # If any are empty, skip coin
        if any(df.empty for df in dfs):
            continue

        # -------------------------------------------------------
        # 3) Convert each DF to daily volume.
        #    We'll produce: daily_df[day_str or day_int] = volume
        # -------------------------------------------------------
        # Approach: group by day (UTC). E.g. day_key = df.timestamp // 86400000
        # Then sum up df["volume"] for each day.

        daily_volumes = []  # daily_volumes[i] will be a dict day->volume for exchange i
        for df in dfs:
            df["day"] = df["timestamp"] // 86400000  # integer day
            grouped = df.groupby("day", as_index=False)["volume"].sum()
            # build dict {day: volume}
            daily_dict = dict(zip(grouped["day"], grouped["volume"]))
            daily_volumes.append(daily_dict)

        # Now we want to find the set of "common days" that appear in all daily_volumes
        # E.g. intersection of day keys across all exchanges
        sets_of_days = [set(dv.keys()) for dv in daily_volumes]
        common_days = set.intersection(*sets_of_days)
        if not common_days:
            continue

        # Filter out days that have no volume on some exchange
        # (Already done by intersection, but you might want to check if the volume is zero and exclude, etc.)

        # -------------------------------------------------------
        # 4) For each pair of exchanges, compute ratio over the *full* range of common days
        # -------------------------------------------------------
        # i.e. ratio = (sum of daily volumes on ex0) / (sum of daily volumes on ex1)
        coin_data = {}  # coin_data[(ex0, ex1)] = ratio for this coin
        for ex0, ex1 in exchange_pairs:
            i0 = exchanges.index(ex0)
            i1 = exchanges.index(ex1)
            sum0 = sum(daily_volumes[i0][day] for day in common_days)
            sum1 = sum(daily_volumes[i1][day] for day in common_days)
            ratio = (sum0 / sum1) if sum1 > 0 else 0.0
            coin_data[(ex0, ex1)] = ratio

        if coin_data:
            all_data[coin] = coin_data

    # -------------------------------------------------------
    # 5) Compute average ratio per (ex0, ex1) across all coins
    # -------------------------------------------------------
    # all_data is: { coin: {(ex0, ex1): ratio, (exA, exB): ratio, ...}, ... }
    # We'll gather lists of ratios per exchange pair, then compute the mean.
    averages = {}
    if not all_data:
        return averages  # empty if no coin data

    # Build a list of all pairs we actually used:
    used_pairs = set()
    for coin in all_data:
        for pair in all_data[coin]:
            used_pairs.add(pair)

    for pair in used_pairs:
        # collect all coin-specific ratios for that pair
        ratios_for_pair = []
        for coin in all_data:
            if pair in all_data[coin]:
                ratios_for_pair.append(all_data[coin][pair])
        if ratios_for_pair:
            averages[pair] = float(np.mean(ratios_for_pair))
        else:
            averages[pair] = 0.0

    return averages


async def add_all_eligible_coins_to_config(config):
    path = config["live"]["approved_coins"]
    if config["live"]["empty_means_all_approved"] and path in [
        [""],
        [],
        None,
        "",
        0,
        0.0,
        {"long": [], "short": []},
        {"long": [""], "short": [""]},
    ]:
        approved_coins = await get_all_eligible_coins(config["backtest"]["exchanges"])
        config["live"]["approved_coins"] = {"long": approved_coins, "short": approved_coins}


async def get_all_eligible_coins(exchanges):
    oms = {}
    for ex in exchanges:
        oms[ex] = OHLCVManager(ex, verbose=False)
    await asyncio.gather(*[oms[ex].load_markets() for ex in oms])
    approved_coins = set()
    for ex in oms:
        for s in oms[ex].markets:
            if oms[ex].has_coin(s):
                coin = symbol_to_coin(s)
                if coin:
                    approved_coins.add(symbol_to_coin(s))
    return sorted(approved_coins)


async def main():
    parser = argparse.ArgumentParser(prog="downloader", description="download ohlcv data")
    parser.add_argument(
        "config_path", type=str, default=None, nargs="?", help="path to json passivbot config"
    )
    template_config = get_template_live_config("v7")
    del template_config["optimize"]
    del template_config["bot"]
    template_config["live"] = {
        k: v
        for k, v in template_config["live"].items()
        if k
        in {
            "approved_coins",
            "ignored_coins",
        }
    }
    template_config["backtest"] = {
        k: v
        for k, v in template_config["backtest"].items()
        if k
        in {
            "combine_ohlcvs",
            "end_date",
            "start_date",
            "exchanges",
        }
    }
    add_arguments_recursively(parser, template_config)
    args = parser.parse_args()
    if args.config_path is None:
        logging.info(f"loading default template config configs/template.json")
        config = load_config("configs/template.json", verbose=False)
    else:
        logging.info(f"loading config {args.config_path}")
        config = load_config(args.config_path)
    await add_all_eligible_coins_to_config(config)
    oms = {}
    try:
        for ex in config["backtest"]["exchanges"]:
            oms[ex] = OHLCVManager(
                ex, config["backtest"]["start_date"], config["backtest"]["end_date"]
            )
        logging.info("loading markets for {config['backtest']['exchanges']}")
        await asyncio.gather(*[oms[ex].load_markets() for ex in oms])
        coins = [x for y in config["live"]["approved_coins"].values() for x in y]
        for coin in sorted(set(coins)):
            tasks = {}
            for ex in oms:
                try:
                    tasks[ex] = asyncio.create_task(oms[ex].get_ohlcvs(coin))
                except Exception as e:
                    logging.error(f"{ex} {coin} error a with get_ohlcvs() {e}")
            for ex in tasks:
                try:
                    await tasks[ex]
                except Exception as e:
                    logging.error(f"{ex} {coin} error b with get_ohlcvs() {e}")
    finally:
        for om in oms.values():
            if om.cc:
                await om.cc.close()


if __name__ == "__main__":
    asyncio.run(main())
