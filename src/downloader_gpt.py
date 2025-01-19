import argparse
import asyncio
import datetime
import gzip
import json
import logging
import inspect
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
from pure_funcs import date_to_ts, ts_to_date_utc, safe_filename
from procedures import (
    make_get_filepath,
    format_end_date,
    coin_to_symbol,
    utc_ms,
    get_file_mod_utc,
    get_first_timestamps_unified,
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


def get_function_name():
    return inspect.currentframe().f_back.f_code.co_name


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
    start_date = datetime.datetime.strptime(start_day[:10], date_format)
    end_date = datetime.datetime.strptime(end_day[:10], date_format)
    days = []
    current_date = start_date
    while current_date <= end_date:
        days.append(current_date.strftime(date_format))
        current_date += datetime.timedelta(days=1)
    return days


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

    def __init__(self, exchange, start_date, end_date, cc=None):
        self.exchange = "binanceusdm" if exchange == "binance" else exchange
        self.quote = "USDC" if exchange == "hyperliquid" else "USDT"
        self.start_date = start_date
        self.end_date = format_end_date(end_date)
        self.start_ts = date_to_ts(self.start_date)
        self.end_ts = date_to_ts(self.end_date)
        self.cc = cc
        self.cache_filepaths = {
            "markets": os.path.join("caches", self.exchange, "markets.json"),
            "ohlcvs": os.path.join("historical_data", f"ohlcvs_{self.exchange}"),
            "first_timestamps": os.path.join("caches", self.exchange, "first_timestamps.json"),
        }
        self.markets = None
        self.verbose = True
        self.max_requests_per_minute = {"": 120, "gateio": 60}
        self.request_timestamps = deque(maxlen=1000)  # for rate-limiting checks

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
            self.end_date = format_end_date(new_end_date)
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
                        f"Rate limit reached {self.exchange}, sleeping for {sleep_time:.2f} seconds"
                    )
                await asyncio.sleep(sleep_time)

        self.request_timestamps.append(current_time)

    async def get_ohlcvs(self, coin):
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
        missing_days = await self.get_missing_days_ohlcvs(coin)
        if missing_days:
            if not self.markets:
                await self.load_markets()
            await self.download_ohlcvs(coin)
        return self.load_ohlcvs_from_cache(coin)

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
        return sorted(
            [x for x in days if (x + ".npy" not in all_files and x[:7] + ".npy" not in all_files)]
        )

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
        if (fts := self.load_first_timestamp(coin)) is not None:
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
        elif self.exchange in ["bybit", "gateio"]:
            # Data since 2018
            ohlcvs = await self.cc.fetch_ohlcv(
                self.get_symbol(coin), since=int(date_to_ts("2018-01-01")), timeframe="1d"
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
                        logging.info(f"Loaded markets from cache {self.exchange}")
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
                    logging.info(f"Dumped markets to cache {self.exchange}")
            except Exception as e:
                logging.error(f"Error with {get_function_name()} {e}")

    def load_ohlcvs_from_cache(self, coin):
        """
        Loads any cached ohlcv data for exchange, coin and date range from cache
        """
        dirpath = os.path.join(self.cache_filepaths["ohlcvs"], coin, "")
        if not os.path.exists(dirpath):
            return pd.DataFrame()
        all_files = sorted([f for f in os.listdir(dirpath) if f.endswith(".npy")])
        all_days = get_days_in_between(self.start_date, self.end_date)
        all_months = sorted(set([x[:7] for x in all_days]))
        files_to_load = [x for x in all_files if x.replace(".npy", "") in all_months]
        files_to_load += [
            x for x in all_files if x.replace(".npy", "") in all_days and x not in files_to_load
        ]
        dfs = [load_ohlcv_data(os.path.join(dirpath, f)) for f in files_to_load]
        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs).drop_duplicates("timestamp").sort_values("timestamp")
        # fill gaps
        interval = 60000
        nindex = np.arange(df.timestamp.iloc[0], df.timestamp.iloc[-1] + interval, interval)
        df = df.set_index("timestamp").reindex(nindex).ffill().reset_index()
        df = df.rename(columns={"index": "timestamp"})
        df = self.filter_date_range(df)
        return attempt_gap_fix_ohlcvs(df, symbol=coin)

    async def download_ohlcvs_binance(self, coin: str):
        # Uses Binance's data archives via binance.vision
        symbolf = self.get_symbol(coin).replace("/USDT:", "")
        dirpath = make_get_filepath(os.path.join(self.cache_filepaths["ohlcvs"], coin, ""))
        base_url = "https://data.binance.vision/data/futures/um/"

        missing_days = await self.get_missing_days_ohlcvs(coin)
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
        # Download daily for any gap months
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

        # Cleanup days contained in months
        fnames = os.listdir(dirpath)
        for fname in fnames:
            if fname.endswith(".npy") and len(fname) == 14:
                if fname[:7] + ".npy" in fnames:
                    os.remove(os.path.join(dirpath, fname))

    async def download_single_binance(self, url: str, fpath: str):
        try:
            csv = await get_zip_binance(url)
            if not csv.empty:
                if self.verbose:
                    logging.info(f"Dumping Binance data {fpath}")
                dump_ohlcv_data(ensure_millis(csv), fpath)
        except Exception as e:
            logging.error(f"Failed to download {url}: {e}")
            traceback.print_exc()

    async def download_ohlcvs_bybit(self, coin: str):
        # Bybit has public data archives
        missing_days = await self.get_missing_days_ohlcvs(coin)
        if not missing_days:
            return
        symbolf = self.get_symbol(coin).replace("/USDT:", "")
        dirpath = make_get_filepath(os.path.join(self.cache_filepaths["ohlcvs"], coin, ""))

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
            if self.verbose:
                logging.info(f"Dumping Bybit {fpath}")
            dump_ohlcv_data(
                ensure_millis(ohlcvs[["timestamp", "open", "high", "low", "close", "volume"]]),
                fpath,
            )
        except Exception as e:
            logging.error(f"Bybit error {url}: {e}")
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
                logging.error(f"Error with Bitget downloader for {coin} {e}")
                # traceback.print_exc()

    def get_url_bitget(self, base_url, symbolf, day):
        if day <= "2024-04-18":
            return f"{base_url}{symbolf}/{symbolf}_UMCBL_1min_{day.replace('-', '')}.zip"
        else:
            return f"{base_url}{symbolf}/UMCBL/{day.replace('-', '')}.zip"

    async def download_single_bitget(self, base_url, symbolf, day, fpath):
        url = self.get_url_bitget(base_url, symbolf, day)
        res = await get_zip_bitget(url)
        if self.verbose:
            logging.info(f"Dumping Bitget {fpath}")
        dump_ohlcv_data(ensure_millis(res), fpath)
        if self.verbose:
            logging.info(f"Dumped Bitget daily data {fpath}")

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
                                f"Bitget, searching for first day of data for {symbol} {str(mid)[:10]}"
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
            if self.verbose:
                logging.info(f"Dumping GateIO daily OHLCV data for {symbol} to {fpath}")
            dump_ohlcv_data(ensure_millis(df_day), fpath)

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
                logging.info(f"Dumped {fpath}")
        except Exception as e:
            logging.error(f"Error with {get_function_name()} {e}")


async def prepare_hlcvs(config: dict, exchange: str):
    coins = sorted(
        set(config["live"]["approved_coins"]["long"]) | set(config["live"]["approved_coins"]["short"])
    )
    if exchange == "binance":
        exchange = "binanceusdm"
    start_date = config["backtest"]["start_date"]
    end_date = format_end_date(config["backtest"]["end_date"])
    om = OHLCVManager(exchange, start_date, end_date)
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
            logging.info(f"coin {coin} missing from {exchange}, skipping")
            continue
        if coin not in first_timestamps_unified:
            logging.info(f"coin {coin} missing from first_timestamps_unified, skipping")
            continue
        if minimum_coin_age_days > 0.0:
            first_ts = await om.get_first_timestamp(coin)
            if first_ts >= end_ts:
                logging.info(
                    f"Coin {coin} too young for {exchange}, start date {ts_to_date_utc(first_ts)}. Skipping"
                )
                continue
            first_ts_plus_min_coin_age = first_timestamps_unified[coin] + min_coin_age_ms
            if first_ts_plus_min_coin_age >= end_ts:
                logging.info(
                    f"Coin {coin}: Not traded due to min_coin_age {int(minimum_coin_age_days)} days"
                    f"{ts_to_date_utc(first_ts_plus_min_coin_age)}. Skipping"
                )
                continue
            new_adjusted_start_ts = max(first_timestamps_unified[coin] + min_coin_age_ms, first_ts)
            if new_adjusted_start_ts > adjusted_start_ts:
                logging.info(
                    f"Coin {coin}: Adjusting start date from {start_date} "
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
        np.save(file_path, data)

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
    logging.info(f"Unifying data for {len(valid_coins)} coins into single numpy array...")
    for i, coin in enumerate(tqdm(valid_coins, desc="Processing coins", unit="coin")):
        file_path = valid_coins[coin]
        ohlcv = np.load(file_path)

        # Calculate indices
        start_idx = int((ohlcv[0, 0] - global_start_time) / interval_ms)
        end_idx = start_idx + len(ohlcv)

        # Extract and process data
        coin_data = ohlcv[:, 1:]
        coin_data[:, 3] = coin_data[:, 2] * coin_data[:, 3]  # Use quote volume

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


async def main():
    pass


if __name__ == "__main__":
    asyncio.run(main())
