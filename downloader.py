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

import numpy as np
import pandas as pd
from dateutil import parser
from tqdm import tqdm

from njit_funcs import calc_samples, round_up, round_dn, round_
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
from pure_funcs import ts_to_date, ts_to_date_utc, date_to_ts2, get_dummy_settings, get_day


class Downloader:
    """
    Downloader class for tick data. Fetches data from specified time until now or specified time.
    """

    def __init__(self, config: dict):
        self.fetch_delay_seconds = 0.75
        self.config = config
        # use binance data for bybit
        self.config["exchange"] = (
            "binance"
            if self.config["exchange"] in ["bybit", "bitget", "okx", "kucoin"]
            else self.config["exchange"]
        )
        self.spot = "spot" in config and config["spot"]
        self.tick_filepath = os.path.join(
            config["caches_dirpath"], f"{config['session_name']}_ticks_cache.npy"
        )
        try:
            self.start_time = int(
                parser.parse(self.config["start_date"])
                .replace(tzinfo=datetime.timezone.utc)
                .timestamp()
                * 1000
            )
        except Exception:
            raise Exception(f"Unrecognized date format for start time {config['start_date']}")
        try:
            self.end_time = int(
                parser.parse(self.config["end_date"])
                .replace(tzinfo=datetime.timezone.utc)
                .timestamp()
                * 1000
            )
            if self.end_time > utc_ms():
                raise Exception(f"End date later than current time {config['end_date']}")
        except Exception:
            raise Exception(f"Unrecognized date format for end time {config['end_date']}")
        if self.config["exchange"] == "binance":
            if self.spot:
                self.daily_base_url = "https://data.binance.vision/data/spot/daily/aggTrades/"
                self.monthly_base_url = "https://data.binance.vision/data/spot/monthly/aggTrades/"
            else:
                market_type = "cm" if config["inverse"] else "um"
                self.daily_base_url = (
                    f"https://data.binance.vision/data/futures/{market_type}/daily/aggTrades/"
                )
                self.monthly_base_url = (
                    f"https://data.binance.vision/data/futures/{market_type}/monthly/aggTrades/"
                )
        elif self.config["exchange"] == "bybit":
            self.daily_base_url = "https://public.bybit.com/trading/"
        else:
            raise Exception(f"unknown exchange {config['exchange']}")
        if "historical_data_path" in self.config and self.config["historical_data_path"]:
            self.filepath = make_get_filepath(
                os.path.join(
                    self.config["historical_data_path"],
                    "historical_data",
                    self.config["exchange"],
                    f"agg_trades_{'spot' if self.spot else 'futures'}",
                    self.config["symbol"],
                    "",
                )
            )
        else:
            self.filepath = make_get_filepath(
                os.path.join(
                    "historical_data",
                    self.config["exchange"],
                    f"agg_trades_{'spot' if self.spot else 'futures'}",
                    self.config["symbol"],
                    "",
                )
            )

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, pd.DataFrame, pd.DataFrame]:
        """
        Validates a dataframe and detects gaps in it. Also detects missing trades in the beginning and end.
        @param df: Dataframe to check for gaps.
        @return: A tuple with following result: if missing values present, the cleaned dataframe, a dataframe with start and end of gaps.
        """
        df.sort_values("trade_id", inplace=True)
        df.drop_duplicates("trade_id", inplace=True)
        df.reset_index(drop=True, inplace=True)
        missing_end_frame = df["trade_id"][df["trade_id"].diff() != 1]
        gaps = pd.DataFrame()
        gaps["start"] = df.iloc[missing_end_frame[1:].index - 1]["trade_id"].tolist()
        gaps["end"] = missing_end_frame[1:].tolist()
        missing_ids = df["trade_id"].iloc[0] % 100000
        if missing_ids != 0:
            gaps = gaps.append(
                {
                    "start": df["trade_id"].iloc[0] - missing_ids,
                    "end": df["trade_id"].iloc[0] - 1,
                },
                ignore_index=True,
            )
        missing_ids = df["trade_id"].iloc[-1] % 100000
        if missing_ids != 99999:
            gaps = gaps.append(
                {
                    "start": df["trade_id"].iloc[-1],
                    "end": df["trade_id"].iloc[-1] + (100000 - missing_ids - 1),
                },
                ignore_index=True,
            )
        if gaps.empty:
            return False, df, gaps
        else:
            gaps["start"] = gaps["start"].astype(np.int64)
            gaps["end"] = gaps["end"].astype(np.int64)
            gaps.sort_values("start", inplace=True)
            gaps.reset_index(drop=True, inplace=True)
            gaps["start"] = gaps["start"].replace(0, 1)
            return True, df, gaps

    def read_dataframe(self, path: str) -> pd.DataFrame:
        """
        Reads a dataframe with correct data types.
        @param path: The path to the dataframe.
        @return: The read dataframe.
        """
        try:
            df = pd.read_csv(
                path,
                dtype={
                    "trade_id": np.int64,
                    "price": np.float64,
                    "qty": np.float64,
                    "timestamp": np.int64,
                    "is_buyer_maker": np.int8,
                },
            )
        except ValueError as e:
            df = pd.DataFrame()
            print_(["Error in reading dataframe", e])
        return df

    def save_dataframe(self, df: pd.DataFrame, filename: str, missing: bool, verified: bool) -> str:
        """
        Saves a processed dataframe. Creates the name based on first and last trade id and first and last timestamp.
        Deletes dataframes that are obsolete. For example, when gaps were filled.
        @param df: The dataframe to save.
        @param filename: The current name of the dataframe.
        @param missing: If the dataframe had gaps.
        @return:
        """
        if verified:
            new_name = f'{df["trade_id"].iloc[0]}_{df["trade_id"].iloc[-1]}_{df["timestamp"].iloc[0]}_{df["timestamp"].iloc[-1]}_verified.csv'
        else:
            new_name = f'{df["trade_id"].iloc[0]}_{df["trade_id"].iloc[-1]}_{df["timestamp"].iloc[0]}_{df["timestamp"].iloc[-1]}.csv'
        if new_name != filename:
            print_(
                [
                    "Saving file",
                    new_name,
                    ts_to_date(int(new_name.split("_")[2]) / 1000),
                ]
            )
            df.to_csv(os.path.join(self.filepath, new_name), index=False)
            new_name = ""
            try:
                os.remove(os.path.join(self.filepath, filename))
                print_(["Removed file", filename])
            except:
                pass
        elif missing:
            print_(["Replacing file", filename])
            df.to_csv(os.path.join(self.filepath, filename), index=False)
        else:
            new_name = ""
        return new_name

    def transform_ticks(self, ticks: list) -> pd.DataFrame:
        """
        Transforms tick data into a cleaned dataframe with correct data types.
        @param ticks: List of tick dictionaries.
        @return: Clean dataframe with correct data types.
        """
        df = pd.DataFrame(ticks)
        if not df.empty:
            df["trade_id"] = df["trade_id"].astype(np.int64)
            df["price"] = df["price"].astype(np.float64)
            df["qty"] = df["qty"].astype(np.float64)
            df["timestamp"] = df["timestamp"].astype(np.int64)
            df["is_buyer_maker"] = df["is_buyer_maker"].astype(np.int8)
            df.sort_values("trade_id", inplace=True)
            df.drop_duplicates("trade_id", inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df

    def get_filenames(self) -> list:
        """
        Returns a sorted list of all file names in the directory.
        @return: Sorted list of file names.
        """
        return sorted(
            [f for f in os.listdir(self.filepath) if f.endswith(".csv")],
            key=lambda x: int(eval(x[: x.find("_")].replace(".cs", "").replace("v", ""))),
        )

    def new_id(
        self,
        first_timestamp,
        last_timestamp,
        first_trade_id,
        length,
        start_time,
        prev_div,
    ):
        """
        Calculates a new id based on several parameters. Uses a weighted approach for more stability.
        @param first_timestamp: First timestamp in current result.
        @param last_timestamp: Last timestamp in current result.
        @param first_trade_id: First trade id in current result.
        @param length: The amount of trades in the current result.
        @param start_time: The time to look for.
        @param prev_div: Previous results of this function.
        @return: Estimated trade id.
        """
        div = int((last_timestamp - first_timestamp) / length)
        prev_div.append(div)
        forward = int((first_timestamp - start_time) / np.mean(prev_div))
        return max(1, int(first_trade_id - forward)), prev_div, forward

    async def find_time(self, start_time) -> pd.DataFrame:
        """
        Finds the trades according to the time.
        Uses different approaches for exchanges depending if time based fetching is supported.
        If time based searching is supported, directly fetch the data.
        If time based searching is not supported, start with current trades and move closer to start time based on estimation.
        @param start_time: Time to look for.
        @return: Dataframe with first trade later or equal to start time.
        """
        try:
            ticks = await self.bot.fetch_ticks_time(start_time)
            return self.transform_ticks(ticks)
        except:
            print_(["Finding id for start time..."])
            ticks = await self.bot.fetch_ticks()
            df = self.transform_ticks(ticks)
            highest_id = df["trade_id"].iloc[-1]
            prev_div = []
            first_ts = df["timestamp"].iloc[0]
            last_ts = df["timestamp"].iloc[-1]
            first_id = df["trade_id"].iloc[0]
            length = len(df)
            while not start_time >= first_ts or not start_time <= last_ts:
                loop_start = time()
                nw_id, prev_div, forward = self.new_id(
                    first_ts, last_ts, first_id, length, start_time, prev_div
                )
                print_(
                    [
                        "Current time span from",
                        df["timestamp"].iloc[0],
                        "to",
                        df["timestamp"].iloc[-1],
                        "with earliest trade id",
                        df["trade_id"].iloc[0],
                        "estimating distance of",
                        forward,
                        "trades",
                    ]
                )
                if nw_id > highest_id:
                    nw_id = highest_id
                try:
                    ticks = await self.bot.fetch_ticks(from_id=int(nw_id), do_print=False)
                    df = self.transform_ticks(ticks)
                    if not df.empty:
                        first_ts = df["timestamp"].iloc[0]
                        last_ts = df["timestamp"].iloc[-1]
                        first_id = df["trade_id"].iloc[0]
                        length = len(df)
                        if nw_id == 1 and first_ts >= start_time:
                            break
                except Exception:
                    print("Failed to fetch or transform...")
                await asyncio.sleep(max(0.0, self.fetch_delay_seconds - time() + loop_start))
            print_(["Found id for start time!"])
            return df[df["timestamp"] >= start_time]

    def get_zip(self, base_url, symbol, date):
        """
        Fetches a full day of trades from the Binance repository.
        @param symbol: Symbol to fetch.
        @param date: Day to download.
        @return: Dataframe with full day.
        """
        print_(["Fetching", symbol, date])
        url = f"{base_url}{symbol.upper()}/{symbol.upper()}-aggTrades-{date}.zip"
        print(url)
        df = pd.DataFrame(columns=["trade_id", "price", "qty", "timestamp", "is_buyer_maker"])
        column_names = [
            "trade_id",
            "price",
            "qty",
            "first",
            "last",
            "timestamp",
            "is_buyer_maker",
        ]
        if self.spot:
            column_names.append("best_match")
        try:
            resp = urlopen(url)
            file_tmp = BytesIO()
            with tqdm.wrapattr(
                open(os.devnull, "wb"), "write", miniters=1, total=getattr(resp, "length", None)
            ) as fout:
                for chunk in resp:
                    fout.write(chunk)
                    file_tmp.write(chunk)

            with zipfile.ZipFile(file_tmp) as my_zip_file:
                for contained_file in my_zip_file.namelist():
                    tf = pd.read_csv(my_zip_file.open(contained_file), names=column_names)
                    if tf.trade_id.iloc[0] == "agg_trade_id":
                        # catch cases where header is included as first row
                        print("header in first row: attempting fix...")
                        tf = tf.iloc[1:].reset_index()
                        tf.is_buyer_maker = tf.is_buyer_maker == "true"
                        tf.drop(
                            errors="ignore",
                            columns=["index"],
                            inplace=True,
                        )
                    tf.drop(
                        errors="ignore",
                        columns=["first", "last", "best_match"],
                        inplace=True,
                    )
                    tf["trade_id"] = tf["trade_id"].astype(np.int64)
                    tf["price"] = tf["price"].astype(np.float64)
                    tf["qty"] = tf["qty"].astype(np.float64)
                    tf["timestamp"] = tf["timestamp"].astype(np.int64)
                    tf["is_buyer_maker"] = tf["is_buyer_maker"].astype(np.int8)
                    tf.sort_values("trade_id", inplace=True)
                    tf.drop_duplicates("trade_id", inplace=True)
                    tf.reset_index(drop=True, inplace=True)
                    if df.empty:
                        df = tf
                    else:
                        df = pd.concat([df, tf])
        except Exception as e:
            print("Failed to fetch", date, e)
        return df

    async def find_df_enclosing_timestamp(self, timestamp, guessed_chunk=None):
        if guessed_chunk is not None:
            if guessed_chunk[0]["timestamp"] < timestamp < guessed_chunk[-1]["timestamp"]:
                print_(["found id"])
                return self.transform_ticks(guessed_chunk)
        else:
            guessed_chunk = sorted(
                await self.bot.fetch_ticks(do_print=False), key=lambda x: x["trade_id"]
            )
            return await self.find_df_enclosing_timestamp(timestamp, guessed_chunk)

        if timestamp < guessed_chunk[0]["timestamp"]:
            guessed_id = guessed_chunk[0]["trade_id"] - len(guessed_chunk) * (
                guessed_chunk[0]["timestamp"] - timestamp
            ) / (guessed_chunk[-1]["timestamp"] - guessed_chunk[0]["timestamp"])
        else:
            guessed_id = guessed_chunk[-1]["trade_id"] + len(guessed_chunk) * (
                timestamp - guessed_chunk[-1]["timestamp"]
            ) / (guessed_chunk[-1]["timestamp"] - guessed_chunk[0]["timestamp"])
        guessed_id = int(guessed_id - len(guessed_chunk) / 2)
        guessed_chunk = sorted(
            await self.bot.fetch_ticks(guessed_id, do_print=False),
            key=lambda x: x["trade_id"],
        )
        print_(
            [
                f"guessed_id {guessed_id} earliest ts {ts_to_date(guessed_chunk[0]['timestamp'] / 1000)[:19]} last ts {ts_to_date(guessed_chunk[-1]['timestamp'] / 1000)[:19]} target ts {ts_to_date(timestamp / 1000)[:19]}"
            ]
        )
        return await self.find_df_enclosing_timestamp(timestamp, guessed_chunk)

    def deduce_trade_ids(self, daily_ticks, df_for_id_matching):
        for idx in [0, -1]:
            match = daily_ticks[
                (daily_ticks.timestamp == df_for_id_matching.timestamp.iloc[idx])
                & (daily_ticks.price == df_for_id_matching.price.iloc[idx])
                & (daily_ticks.qty == df_for_id_matching.qty.iloc[idx])
            ]
            if len(match) == 1:
                id_at_match = df_for_id_matching.trade_id.iloc[idx]
                return np.arange(
                    id_at_match - match.index[0],
                    id_at_match - match.index[0] + len(daily_ticks),
                )
                # trade_ids = np.arange(id_at_match, id_at_match + len(daily_ticks.loc[match.index:]))
                return match, id_at_match
        raise Exception("unable to make trade ids")

    async def get_csv_gz_old(self, base_url, symbol, date, df_for_id_matching):
        """
        Fetches a full day of trades from the Bybit repository.
        @param symbol: Symbol to fetch.
        @param date: Day to download.
        @return: Dataframe with full day.
        """
        print_(["Fetching", symbol, date])
        url = f"{base_url}{symbol.upper()}/{symbol.upper()}{date}.csv.gz"
        df = pd.DataFrame(columns=["trade_id", "price", "qty", "timestamp", "is_buyer_maker"])
        try:
            resp = urlopen(url)
            with gzip.open(BytesIO(resp.read())) as f:
                ff = pd.read_csv(f)
                trade_ids = np.zeros(len(ff)).astype(np.int64)
                tf = pd.DataFrame(
                    {
                        "trade_id": trade_ids,
                        "price": ff.price.astype(np.float64),
                        "qty": ff["size"].astype(np.float64),
                        "timestamp": (ff.timestamp * 1000).astype(np.int64),
                        "is_buyer_maker": (ff.side == "Sell").astype(np.int8),
                    }
                )
                tf["trade_id"] = self.deduce_trade_ids(tf, df_for_id_matching)
                tf.sort_values("timestamp", inplace=True)
                tf.reset_index(drop=True, inplace=True)
                del ff
                df = tf
        except Exception as e:
            print("Failed to fetch", date, e)
        return df

    async def download_ticks(self):
        """
        Searches for previously downloaded files and fills gaps in them if necessary.
        Downloads any missing data based on the specified time frame.
        @return:
        """
        if self.config["exchange"] == "binance":
            if self.spot:
                self.bot = await create_binance_bot_spot(get_dummy_settings(self.config))
            else:
                self.bot = await create_binance_bot(get_dummy_settings(self.config))
        elif self.config["exchange"] == "bybit":
            self.bot = await create_bybit_bot(get_dummy_settings(self.config))
        else:
            print(self.config["exchange"], "not found")
            return

        filenames = self.get_filenames()
        mod_files = []
        highest_id = 0
        for f in filenames:
            verified = False
            try:
                first_time = int(f.split("_")[2])
                last_time = int(f.split("_")[3].split(".")[0])
                if len(f.split("_")) > 4:
                    verified = True
            except:
                first_time = sys.maxsize
                last_time = sys.maxsize
            if (
                not verified
                and last_time >= self.start_time
                and (self.end_time == -1 or (first_time <= self.end_time))
                or last_time == sys.maxsize
            ):
                print_(["Validating file", f, ts_to_date(first_time / 1000)])
                df = self.read_dataframe(os.path.join(self.filepath, f))
                missing, df, gaps = self.validate_dataframe(df)
                exists = False
                if gaps.empty:
                    first_id = df["trade_id"].iloc[0]
                    self.save_dataframe(df, f, missing, True)
                else:
                    first_id = (
                        df["trade_id"].iloc[0]
                        if df["trade_id"].iloc[0] < gaps["start"].iloc[0]
                        else gaps["start"].iloc[0]
                    )
                if not gaps.empty and (
                    f != filenames[-1] or str(first_id - first_id % 100000) not in f
                ):
                    last_id = df["trade_id"].iloc[-1]
                    for i in filenames:
                        tmp_first_id = int(i.split("_")[0])
                        tmp_last_id = int(i.split("_")[1].replace(".csv", ""))
                        if (
                            (first_id - first_id % 100000) == tmp_first_id
                            and (
                                (first_id - first_id % 100000 + 99999) == tmp_last_id
                                or (highest_id == tmp_first_id or highest_id == tmp_last_id)
                                or highest_id > last_id
                            )
                            and first_id != 1
                            and i != f
                        ):
                            exists = True
                            break
                if missing and df["timestamp"].iloc[-1] > self.start_time and not exists:
                    current_time = df["timestamp"].iloc[-1]
                    for i in gaps.index:
                        print_(
                            [
                                "Filling gaps from id",
                                gaps["start"].iloc[i],
                                "to id",
                                gaps["end"].iloc[i],
                            ]
                        )
                        current_id = gaps["start"].iloc[i]
                        while current_id < gaps["end"].iloc[i] and utc_ms() - current_time > 10000:
                            loop_start = time()
                            try:
                                fetched_new_trades = await self.bot.fetch_ticks(int(current_id))
                                tf = self.transform_ticks(fetched_new_trades)
                                if tf.empty:
                                    print_(["Response empty. No new trades, exiting..."])
                                    await asyncio.sleep(
                                        max(
                                            0.0,
                                            self.fetch_delay_seconds - time() + loop_start,
                                        )
                                    )
                                    break
                                if current_id == tf["trade_id"].iloc[-1]:
                                    print_(["Same trade ID again. No new trades, exiting..."])
                                    await asyncio.sleep(
                                        max(
                                            0.0,
                                            self.fetch_delay_seconds - time() + loop_start,
                                        )
                                    )
                                    break
                                current_id = tf["trade_id"].iloc[-1]
                                df = pd.concat([df, tf])
                                df.sort_values("trade_id", inplace=True)
                                df.drop_duplicates("trade_id", inplace=True)
                                df = df[
                                    df["trade_id"]
                                    <= gaps["end"].iloc[i] - gaps["end"].iloc[i] % 100000 + 99999
                                ]
                                df.reset_index(drop=True, inplace=True)
                                current_time = df["timestamp"].iloc[-1]
                            except Exception:
                                print_(["Failed to fetch or transform..."])
                            await asyncio.sleep(
                                max(0.0, self.fetch_delay_seconds - time() + loop_start)
                            )
                if not df.empty:
                    if df["trade_id"].iloc[-1] > highest_id:
                        highest_id = df["trade_id"].iloc[-1]
                if not exists:
                    tf = df[df["trade_id"].mod(100000) == 0]
                    if len(tf) > 1:
                        df = df[: tf.index[-1]]
                    nf = self.save_dataframe(df, f, missing, verified)
                    mod_files.append(nf)
                elif df["trade_id"].iloc[0] != 1:
                    os.remove(os.path.join(self.filepath, f))
                    print_(["Removed file fragment", f])

        chunk_gaps = []
        filenames = self.get_filenames()
        prev_last_id = 0
        prev_last_time = self.start_time
        for f in filenames:
            first_id = int(f.split("_")[0])
            last_id = int(f.split("_")[1])
            first_time = int(f.split("_")[2])
            last_time = int(f.split("_")[3].split(".")[0])
            if (
                first_id - 1 != prev_last_id
                and f not in mod_files
                and first_time >= prev_last_time
                and first_time >= self.start_time
                and not prev_last_time > self.end_time
            ):
                chunk_gaps.append((prev_last_time, first_time, prev_last_id, first_id - 1))
            if first_time >= self.start_time or last_time >= self.start_time:
                prev_last_id = last_id
                prev_last_time = last_time

        if len(filenames) < 1:
            chunk_gaps.append((self.start_time, self.end_time, 0, 0))
        if prev_last_time < self.end_time:
            chunk_gaps.append((prev_last_time, self.end_time, prev_last_id, 0))

        seen = set()
        chunk_gaps_dedup = []
        for elm in chunk_gaps:
            if elm not in seen:
                chunk_gaps_dedup.append(elm)
                seen.add(elm)
        chunk_gaps = chunk_gaps_dedup

        for gaps in chunk_gaps:
            start_time, end_time, start_id, end_id = gaps
            df = pd.DataFrame()

            current_id = start_id + 1
            current_time = start_time

            if self.config["exchange"] == "binance":
                fetched_new_trades = await self.bot.fetch_ticks(1)
                tf = self.transform_ticks(fetched_new_trades)
                earliest = tf["timestamp"].iloc[0]

                if earliest > start_time:
                    start_time = earliest
                    current_time = start_time

                tmp = pd.date_range(
                    start=datetime.datetime.fromtimestamp(
                        start_time / 1000, datetime.timezone.utc
                    ).date(),
                    end=datetime.datetime.fromtimestamp(
                        end_time / 1000, datetime.timezone.utc
                    ).date(),
                    freq="D",
                ).to_pydatetime()
                days = [date.strftime("%Y-%m-%d") for date in tmp]
                df = pd.DataFrame(columns=["trade_id", "price", "qty", "timestamp", "is_buyer_maker"])

                months_done = set()
                months_failed = set()
                for day in days:
                    month = day[:7]
                    if month in months_done:
                        continue
                    if month in months_failed:
                        tf = self.get_zip(self.daily_base_url, self.config["symbol"], day)
                        if tf.empty:
                            print_(["failed to fetch daily", day])
                            continue
                    else:
                        tf = self.get_zip(self.monthly_base_url, self.config["symbol"], month)
                        if tf.empty:
                            print_(["failed to fetch monthly", month])
                            months_failed.add(month)
                            tf = self.get_zip(self.daily_base_url, self.config["symbol"], day)
                        else:
                            months_done.add(month)
                    tf = tf[tf["timestamp"] >= start_time]
                    tf = tf[tf["timestamp"] <= end_time]
                    if start_id != 0:
                        tf = tf[tf["trade_id"] > start_id]
                    if end_id != 0:
                        tf = tf[tf["trade_id"] <= end_id]
                    if df.empty:
                        df = tf
                    else:
                        df = pd.concat([df, tf])
                    df.sort_values("trade_id", inplace=True)
                    df.drop_duplicates("trade_id", inplace=True)
                    df.reset_index(drop=True, inplace=True)

                    if not df.empty and (
                        (df["trade_id"].iloc[0] % 100000 == 0 and len(df) >= 100000)
                        or df["trade_id"].iloc[0] % 100000 != 0
                    ):
                        for index, row in df[df["trade_id"] % 100000 == 0].iterrows():
                            if index != 0:
                                self.save_dataframe(
                                    df[
                                        (df["trade_id"] >= row["trade_id"] - 1000000)
                                        & (df["trade_id"] < row["trade_id"])
                                    ],
                                    "",
                                    True,
                                    False,
                                )
                                df = df[df["trade_id"] >= row["trade_id"]]
                    if not df.empty:
                        start_id = df["trade_id"].iloc[0] - 1
                        start_time = df["timestamp"].iloc[0]
                        current_time = df["timestamp"].iloc[-1]
                        current_id = df["trade_id"].iloc[-1] + 1
            if start_id == 0:
                df = await self.find_time(start_time)
                current_id = df["trade_id"].iloc[-1] + 1
                current_time = df["timestamp"].iloc[-1]

            end_id = sys.maxsize if end_id == 0 else end_id - 1

            if current_id <= end_id and current_time <= end_time and utc_ms() - current_time > 10000:
                print_(
                    [
                        "Downloading from",
                        ts_to_date(float(current_time) / 1000),
                        "to",
                        ts_to_date(float(end_time) / 1000),
                    ]
                )

            while (
                current_id <= end_id and current_time <= end_time and utc_ms() - current_time > 10000
            ):
                loop_start = time()
                fetched_new_trades = await self.bot.fetch_ticks(int(current_id))
                tf = self.transform_ticks(fetched_new_trades)
                if tf.empty:
                    print_(["Response empty. No new trades, exiting..."])
                    await asyncio.sleep(max(0.0, self.fetch_delay_seconds - time() + loop_start))
                    break
                if current_id == tf["trade_id"].iloc[-1]:
                    print_(["Same trade ID again. No new trades, exiting..."])
                    await asyncio.sleep(max(0.0, self.fetch_delay_seconds - time() + loop_start))
                    break
                df = pd.concat([df, tf])
                df.sort_values("trade_id", inplace=True)
                df.drop_duplicates("trade_id", inplace=True)
                df.reset_index(drop=True, inplace=True)
                current_time = tf["timestamp"].iloc[-1]
                current_id = tf["trade_id"].iloc[-1] + 1
                tf = df[df["trade_id"].mod(100000) == 0]
                if not tf.empty and len(df) > 1:
                    if df["trade_id"].iloc[0] % 100000 == 0 and len(tf) > 1:
                        self.save_dataframe(df[: tf.index[-1]], "", True, False)
                        df = df[tf.index[-1] :]
                    elif df["trade_id"].iloc[0] % 100000 != 0 and len(tf) == 1:
                        self.save_dataframe(df[: tf.index[-1]], "", True, False)
                        df = df[tf.index[-1] :]
                await asyncio.sleep(max(0.0, self.fetch_delay_seconds - time() + loop_start))
            if not df.empty:
                df = df[df["timestamp"] >= start_time]
                if start_id != 0 and not df.empty:
                    df = df[df["trade_id"] > start_id]
                elif end_id != sys.maxsize and not df.empty:
                    df = df[df["trade_id"] <= end_id]
                elif end_time != sys.maxsize and not df.empty:
                    df = df[df["timestamp"] <= end_time]
                if not df.empty:
                    self.save_dataframe(df, "", True, False)

        try:
            await self.bot.session.close()
        except:
            pass

    async def prepare_files(self):
        """
        Takes downloaded data and prepares a numpy array for use in backtesting.
        @return:
        """
        filenames = [
            f
            for f in self.get_filenames()
            if int(f.split("_")[3].split(".")[0]) >= self.start_time
            and int(f.split("_")[2]) <= self.end_time
        ]
        left_overs = pd.DataFrame()
        sample_size_ms = 1000
        current_index = 0

        try:
            first_frame = pd.read_csv(
                os.path.join(self.filepath, filenames[0]),
                dtype={
                    "price": np.float64,
                    "is_buyer_maker": np.float64,
                    "timestamp": np.float64,
                    "qty": np.float64,
                },
                usecols=["price", "is_buyer_maker", "timestamp", "qty"],
            )
            first_frame = first_frame[
                (first_frame["timestamp"] >= self.start_time)
                & (first_frame["timestamp"] <= self.end_time)
            ]
            earliest_time = first_frame.timestamp.iloc[0] // sample_size_ms * sample_size_ms
        except Exception as e:
            print_(["Error in determining earliest time", e])
            earliest_time = self.start_time

        try:
            last_frame = pd.read_csv(
                os.path.join(self.filepath, filenames[-1]),
                dtype={
                    "price": np.float64,
                    "is_buyer_maker": np.float64,
                    "timestamp": np.float64,
                    "qty": np.float64,
                },
                usecols=["price", "is_buyer_maker", "timestamp", "qty"],
            )
            last_frame = last_frame[
                (last_frame["timestamp"] >= self.start_time)
                & (last_frame["timestamp"] <= self.end_time)
            ]
            latest_time = last_frame.timestamp.iloc[-1] // sample_size_ms * sample_size_ms
        except Exception as e:
            print_(["Error in determining latest time", e])
            latest_time = self.end_time

        array = np.zeros(
            (int((latest_time - earliest_time) / sample_size_ms + 1), 3),
            dtype=np.float64,
        )

        for f in filenames:
            chunk = pd.read_csv(
                os.path.join(self.filepath, f),
                dtype={
                    "price": np.float64,
                    "is_buyer_maker": np.float64,
                    "timestamp": np.float64,
                    "qty": np.float64,
                },
                usecols=["price", "is_buyer_maker", "timestamp", "qty"],
            )

            chunk = pd.concat([left_overs, chunk])
            chunk.sort_values("timestamp", inplace=True)
            chunk = chunk[
                (chunk["timestamp"] >= self.start_time) & (chunk["timestamp"] <= self.end_time)
            ]

            cut_off = (
                chunk.timestamp.iloc[-1] // sample_size_ms * sample_size_ms - 1 - (1 * sample_size_ms)
            )

            left_overs = chunk[chunk["timestamp"] > cut_off]
            chunk = chunk[chunk["timestamp"] <= cut_off]

            sampled_ticks = calc_samples(chunk[["timestamp", "qty", "price"]].values)
            if current_index != 0 and array[current_index - 1, 0] + 1000 != sampled_ticks[0, 0]:
                size = int((sampled_ticks[0, 0] - array[current_index - 1, 0]) / sample_size_ms) - 1
                tmp = np.zeros((size, 3), dtype=np.float64)
                tmp[:, 0] = np.arange(
                    array[current_index - 1, 0] + sample_size_ms,
                    sampled_ticks[0, 0],
                    sample_size_ms,
                    dtype=np.float64,
                )
                tmp[:, 2] = array[current_index - 1, 2]
                array[current_index : current_index + len(tmp)] = tmp
                current_index += len(tmp)
            array[current_index : current_index + len(sampled_ticks)] = sampled_ticks
            current_index += len(sampled_ticks)

            print(
                "\rloaded chunk of data",
                f,
                ts_to_date(float(f.split("_")[2]) / 1000),
                end="     ",
            )
        print("\n")

        # Fill in anything left over
        if not left_overs.empty:
            sampled_ticks = calc_samples(left_overs[["timestamp", "qty", "price"]].values)
            if current_index != 0 and array[current_index - 1, 0] + 1000 != sampled_ticks[0, 0]:
                size = int((sampled_ticks[0, 0] - array[current_index - 1, 0]) / sample_size_ms) - 1
                tmp = np.zeros((size, 3), dtype=np.float64)
                tmp[:, 0] = np.arange(
                    array[current_index - 1, 0] + sample_size_ms,
                    sampled_ticks[0, 0],
                    sample_size_ms,
                    dtype=np.float64,
                )
                tmp[:, 2] = array[current_index - 1, 2]
                array[current_index : current_index + len(tmp)] = tmp
                current_index += len(tmp)
            array[current_index : current_index + len(sampled_ticks)] = sampled_ticks
            current_index += len(sampled_ticks)

        # Fill the gap at the end with the latest price
        # Should not be necessary anymore
        if current_index + 1 < len(array):
            size = len(array) - current_index
            tmp = np.zeros((size, 3), dtype=np.float64)
            tmp[:, 0] = np.arange(
                array[current_index - 1, 0] + sample_size_ms,
                array[current_index - 1, 0] + ((size + 1) * sample_size_ms),
                sample_size_ms,
                dtype=np.float64,
            )
            tmp[:, 2] = array[current_index - 1, 2]
            array[current_index : current_index + len(tmp)] = tmp
            current_index += len(tmp)

        print_(
            [
                "Saving single file with",
                len(array),
                " ticks to",
                self.tick_filepath,
                "...",
            ]
        )
        np.save(self.tick_filepath, array)
        print_(["Saved single file!"])

    async def get_sampled_ticks(self) -> np.ndarray:
        """
        Function for direct use in the backtester. Checks if the numpy arrays exist and if so loads them.
        If they do not exist or if their length doesn't match, download the missing data and create them.
        @return: numpy array.
        """
        if os.path.exists(self.tick_filepath):
            print_(["Loading cached tick data from", self.tick_filepath])
            tick_data = np.load(self.tick_filepath)
            return tick_data
        await self.download_ticks()
        await self.prepare_files()
        tick_data = np.load(self.tick_filepath)
        return tick_data


def get_zip(url: str):
    col_names = ["timestamp", "open", "high", "low", "close", "volume"]
    try:
        resp = urlopen(url)
        file_tmp = BytesIO()
        with tqdm.wrapattr(
            open(os.devnull, "wb"), "write", miniters=1, total=getattr(resp, "length", None)
        ) as fout:
            for chunk in resp:
                fout.write(chunk)
                file_tmp.write(chunk)
        dfs = []
        with zipfile.ZipFile(file_tmp) as my_zip_file:
            for contained_file in my_zip_file.namelist():
                df = pd.read_csv(my_zip_file.open(contained_file))
                df.columns = col_names + [str(i) for i in range(len(df.columns) - len(col_names))]
                dfs.append(df[col_names])
        return pd.concat(dfs).sort_values("timestamp").reset_index()
    except Exception as e:
        print(e)


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


def findall(string, pattern):
    """Yields all the positions of
    the pattern in the string"""
    i = string.find(pattern)
    while i != -1:
        yield i
        i = string.find(pattern, i + 1)


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
    start_date, end_date = get_day(start_date), get_day(end_date)
    assert date_to_ts2(end_date) >= date_to_ts2(start_date), "end_date is older than start_date"
    dirpath = make_get_filepath(f"historical_data/ohlcvs_bybit{'_spot' if spot else ''}/{symbol}/")
    ideal_days = get_days_in_between(start_date, end_date)
    days_done = [filename[:-4] for filename in os.listdir(dirpath) if ".csv" in filename]
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
            n_concurrent_fetches = 2
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
                    filepath = f"{dirpath}{day}.csv"
                    df.to_csv(filepath)
                    dumped.append(day)
                if not download_only:
                    dfs.update(dfs_)
    if not download_only:
        for day in ideal_days:
            if os.path.exists(f"{dirpath}{day}.csv"):
                dfs[day] = pd.read_csv(f"{dirpath}{day}.csv")
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
    return ohlcvs


async def download_single_ohlcvs_binance(url: str, fpath: str):
    try:
        print(f"fetching {url}")
        csv = await get_zip_binance(url)
        csv.to_csv(fpath)
    except Exception as e:
        print(f"failed to download {url} {e}")


async def download_ohlcvs_binance(
    symbol, inverse, start_date, end_date, spot=False, download_only=False
) -> pd.DataFrame:
    dirpath = make_get_filepath(f"historical_data/ohlcvs_{'spot' if spot else 'futures'}/{symbol}/")
    base_url = "https://data.binance.vision/data/"
    base_url += "spot/" if spot else f"futures/{'cm' if inverse else 'um'}/"
    col_names = ["timestamp", "open", "high", "low", "close", "volume"]
    if spot:
        start_ts = get_first_ohlcv_ts(symbol, spot=spot)
    else:
        start_ts = (await get_first_ohlcv_timestamps(symbols=[symbol]))[symbol]
    if start_ts != 0:
        print(f"first ohlcv at {ts_to_date(start_ts)}")
    start_ts = int(max(start_ts, date_to_ts2(start_date)))
    end_ts = int(date_to_ts2(end_date))
    days = [ts_to_date_utc(x)[:10] for x in list(range(start_ts, end_ts, 1000 * 60 * 60 * 24))]
    months = sorted({x[:7] for x in days})
    month_now = ts_to_date(utc_ms())[:7]
    months = [m for m in months if m != month_now]

    # do months async
    months_filepaths = {month: os.path.join(dirpath, month + ".csv") for month in months}
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
    days_filepaths = {day: os.path.join(dirpath, day + ".csv") for day in days}
    missing_days = {
        k: v
        for k, v in days_filepaths.items()
        if not os.path.exists(v) and k[:7] + ".csv" not in months_done
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
        if fname.endswith(".csv") and len(fname) == 14:
            if fname[:7] + ".csv" in fnames:
                print("deleting", os.path.join(dirpath, fname))
                os.remove(os.path.join(dirpath, fname))

    if not download_only:
        fnames = os.listdir(dirpath)
        dfs = [
            pd.read_csv(os.path.join(dirpath, fpath))
            for fpath in months_done + days_done
            if fpath in fnames
        ]
        df = pd.concat(dfs)[col_names].sort_values("timestamp")
        df = df.drop_duplicates(subset=["timestamp"]).reset_index()
        nindex = np.arange(df.timestamp.iloc[0], df.timestamp.iloc[-1] + 60000, 60000)
        return df[col_names].set_index("timestamp").reindex(nindex).ffill().reset_index()


def count_longest_identical_data(hlc, symbol):
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
    print(
        f"{symbol} most n days of consecutive identical ohlcvs: {longest_consecutive / 60 / 24:.3f}, index last: {i_}"
    )
    return longest_consecutive


def attempt_gap_fix_hlcs(df):
    interval = 60 * 1000
    max_gap = interval * 60 * 12  # 12 hours
    greatest_gap = df.timestamp.diff().max()
    if greatest_gap == interval:
        return df
    if greatest_gap > max_gap:
        raise Exception(
            f"ohlcvs gap greater than {max_gap / (1000 * 60 * 60)} hours: {greatest_gap / (1000 * 60 * 60)} hours"
        )
    print("gap(s) in ohlcvs... attempting fix")
    new_timestamps = np.arange(df["timestamp"].iloc[0], df["timestamp"].iloc[-1] + interval, interval)
    new_df = df.set_index("timestamp").reindex(new_timestamps)
    new_df.close = new_df.close.ffill()
    new_df.open = new_df.open.fillna(new_df.close)
    new_df.high = new_df.high.fillna(new_df.close)
    new_df.low = new_df.low.fillna(new_df.close)
    new_df.volume = new_df.volume.fillna(0.0)
    new_df = new_df.reset_index()
    return new_df[["timestamp", "open", "high", "low", "close", "volume"]]


async def load_hlc_cache(
    symbol,
    inverse,
    start_date,
    end_date,
    base_dir="backtests",
    spot=False,
    exchange="binance",
):
    cache_fname = (
        f"{ts_to_date_utc(date_to_ts2(start_date))[:10]}_"
        + f"{ts_to_date_utc(date_to_ts2(end_date))[:10]}_ohlcv_cache.npy"
    )

    filepath = make_get_filepath(
        os.path.join(base_dir, exchange + ("_spot" if spot else ""), symbol, "caches", cache_fname)
    )
    if os.path.exists(filepath):
        data = np.load(filepath)
    else:
        if exchange == "bybit":
            df = await download_ohlcvs_bybit(symbol, start_date, end_date, spot, download_only=False)
            df = attempt_gap_fix_hlcs(df)
        else:
            df = await download_ohlcvs_binance(symbol, inverse, start_date, end_date, spot)
        df = df[df.timestamp >= date_to_ts2(start_date)]
        df = df[df.timestamp <= date_to_ts2(end_date)]
        data = df[["timestamp", "high", "low", "close"]].values
        np.save(filepath, data)
    try:
        count_longest_identical_data(data, symbol)
    except Exception as e:
        print("error checking integrity", e)
    return data


async def prepare_multsymbol_data(
    symbols, start_date, end_date, base_dir, exchange
) -> (float, np.ndarray):
    """
    returns first timestamp and hlc data in the form
    [
        [
            [sym0_high0, sym0_low0, sym0_close0],
            [sym0_high1, sym0_low1, sym0_close1],
            ...
        ],
        [
            [sym1_high0, sym1_low0, sym1_close0],
            [sym1_high1, sym1_low1, sym1_close1],
            ...
        ],
        ...
    ]
    """
    if end_date in ["today", "now", ""]:
        end_date = ts_to_date_utc(utc_ms())[:10]
    hlcs = []
    interval = 60000.0
    for symbol in symbols:
        data = await load_hlc_cache(symbol, False, start_date, end_date, base_dir, False, exchange)
        assert (
            np.diff(data[:, 0]) == interval
        ).all(), f"gaps in hlc data {symbol}"  # verify integrous 1m hlcs
        dft = pd.DataFrame(
            data, columns=["timestamp"] + [f"{symbol}_{key}" for key in ["high", "low", "close"]]
        )
        hlcs.append(dft)

    tss = np.arange(
        min([x.timestamp.iloc[0] for x in hlcs]),
        max([x.timestamp.iloc[-1] for x in hlcs]) + interval,
        interval,
    )
    df = pd.concat([x.set_index("timestamp").reindex(tss) for x in hlcs], axis=1, join="outer")
    df = df.fillna(0.0)
    return df.index[0], np.array([df.values[:, i : i + 3] for i in range(0, len(symbols) * 3, 3)])


async def prepare_hlcs_forager(symbols, start_date, end_date, base_dir, exchange):
    """ """
    if end_date in ["today", "now", ""]:
        end_date = ts_to_date_utc(utc_ms())[:10]
    hlcsd = {}
    interval_ms = 60000
    for symbol in symbols:
        data = await load_hlc_cache(symbol, False, start_date, end_date, base_dir, False, exchange)
        assert (
            np.diff(data[:, 0]) == interval_ms
        ).all(), f"gaps in hlc data {symbol}"  # verify integrous 1m hlcs
        hlcsd[symbol] = data

    # hlcsd is {symbol: array([[timestamp, high, low, close]])}
    first_timestamp = min([x[0][0] for x in hlcsd.values()])
    last_timestamp = max([x[-1][0] for x in hlcsd.values()])
    timestamps = np.arange(first_timestamp, last_timestamp + interval_ms, interval_ms)

    unified_data = []
    for symbol, data in hlcsd.items():
        unified_data.append(np.zeros((len(timestamps), 3)))
        offset = int((data[0][0] - timestamps[0]) // interval_ms)
        unified_data[-1][offset : offset + len(data)] = data[:, 1:]

    return np.array(unified_data).transpose(1, 0, 2)


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
    if config["ohlcv"]:
        data = await load_hlc_cache(
            config["symbol"],
            config["inverse"],
            config["start_date"],
            config["end_date"],
            spot=config["spot"],
            exchange=config["exchange"],
        )
    else:
        downloader = Downloader(config)
        await downloader.download_ticks()
        if not args.download_only:
            await downloader.prepare_files()


if __name__ == "__main__":
    asyncio.run(main())
