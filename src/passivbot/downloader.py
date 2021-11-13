from __future__ import annotations

import argparse
import asyncio
import datetime
import gzip
import io
import logging
import os
import sys
import time
import urllib.request
import zipfile

import dateutil.parser
import numpy as np
import pandas as pd

from passivbot.utils.funcs.njit import calc_samples
from passivbot.utils.funcs.pure import get_dummy_settings
from passivbot.utils.funcs.pure import ts_to_date
from passivbot.utils.procedures import add_backtesting_argparse_args
from passivbot.utils.procedures import create_binance_bot
from passivbot.utils.procedures import create_binance_bot_spot
from passivbot.utils.procedures import create_bybit_bot
from passivbot.utils.procedures import make_get_filepath
from passivbot.utils.procedures import prepare_backtest_config
from passivbot.utils.procedures import utc_ms

log = logging.getLogger(__name__)


class Downloader:
    """
    Downloader class for tick data. Fetches data from specified time until now or specified time.
    """

    def __init__(self, config: dict):
        self.fetch_delay_seconds = 0.75
        self.config = config
        self.spot = "spot" in config and config["spot"]
        self.tick_filepath = os.path.join(
            config["caches_dirpath"], f"{config['session_name']}_ticks_cache.npy"
        )
        try:
            self.start_time = int(
                dateutil.parser.parse(self.config["start_date"])
                .replace(tzinfo=datetime.timezone.utc)
                .timestamp()
                * 1000
            )
        except Exception:
            raise Exception(f"Unrecognized date format for start time {config['start_date']}")
        try:
            self.end_time = int(
                dateutil.parser.parse(self.config["end_date"])
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

    def validate_dataframe(self, df: pd.DataFrame) -> tuple[bool, pd.DataFrame, pd.DataFrame]:
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
        if missing_ids := df["trade_id"].iloc[0] % 100000 != 0:
            gaps.append(
                {"start": df["trade_id"].iloc[0] - missing_ids, "end": df["trade_id"].iloc[0] - 1},
                ignore_index=True,
            )
        if missing_ids := df["trade_id"].iloc[-1] % 100000 != 99999:
            gaps.append(
                {
                    "start": df["trade_id"].iloc[-1],
                    "end": df["trade_id"].iloc[-1] + (100000 - missing_ids - 1),
                },
                ignore_index=True,
            )
        missing_ids = df["trade_id"].iloc[0] % 100000
        if missing_ids != 0:
            gaps = gaps.append(
                {"start": df["trade_id"].iloc[0] - missing_ids, "end": df["trade_id"].iloc[0] - 1},
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
            log.error("Error in reading dataframe: %s", e)
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
            log.info(
                "Saving file %s date=%s", new_name, ts_to_date(int(new_name.split("_")[2]) / 1000)
            )
            df.to_csv(os.path.join(self.filepath, new_name), index=False)
            new_name = ""
            try:
                os.remove(os.path.join(self.filepath, filename))
                log.info("Removed file", filename)
            except Exception:
                pass
        elif missing:
            log.info("Replacing file: %s", filename)
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
            (f for f in os.listdir(self.filepath) if f.endswith(".csv")),
            key=lambda x: int(eval(x[: x.find("_")].replace(".cs", "").replace("v", ""))),
        )

    def new_id(self, first_timestamp, last_timestamp, first_trade_id, length, start_time, prev_div):
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
        except Exception:
            log.info("Finding id for start time...")
            ticks = await self.bot.fetch_ticks()
            df = self.transform_ticks(ticks)
            highest_id = df["trade_id"].iloc[-1]
            prev_div = []
            first_ts = df["timestamp"].iloc[0]
            last_ts = df["timestamp"].iloc[-1]
            first_id = df["trade_id"].iloc[0]
            length = len(df)
            while not start_time >= first_ts or not start_time <= last_ts:
                loop_start = time.time()
                nw_id, prev_div, forward = self.new_id(
                    first_ts, last_ts, first_id, length, start_time, prev_div
                )
                log.info(
                    "Current time span from %s to %s with earliest trade id %s estimating distance of %s trades",
                    df["timestamp"].iloc[0],
                    df["timestamp"].iloc[-1],
                    df["trade_id"].iloc[0],
                    forward,
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
                    log.info("Failed to fetch or transform trades according to time...")
                await asyncio.sleep(max(0.0, self.fetch_delay_seconds - time.time() + loop_start))
            log.info("Found id for start time!")
            return df[df["timestamp"] >= start_time]

    def get_zip(self, base_url, symbol, date):
        """
        Fetches a full day of trades from the Binance repository.
        @param symbol: Symbol to fetch.
        @param date: Day to download.
        @return: Dataframe with full day.
        """
        log.info("Fetching symbol=%s date=%s", symbol, date)
        url = f"{base_url}{symbol.upper()}/{symbol.upper()}-aggTrades-{date}.zip"
        df = pd.DataFrame(columns=["trade_id", "price", "qty", "timestamp", "is_buyer_maker"])
        column_names = ["trade_id", "price", "qty", "first", "last", "timestamp", "is_buyer_maker"]
        if self.spot:
            column_names.append("best_match")
        try:
            resp = urllib.request.urlopen(url)
            with zipfile.ZipFile(io.BytesIO(resp.read())) as my_zip_file:
                for contained_file in my_zip_file.namelist():
                    tf = pd.read_csv(my_zip_file.open(contained_file), names=column_names)
                    tf.drop(errors="ignore", columns=["first", "last", "best_match"], inplace=True)
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
            log.info("Failed to fetch %s: %s", date, e)
        return df

    async def find_df_enclosing_timestamp(self, timestamp, guessed_chunk=None):
        if guessed_chunk is not None:
            if guessed_chunk[0]["timestamp"] < timestamp < guessed_chunk[-1]["timestamp"]:
                log.info("found id")
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
            await self.bot.fetch_ticks(guessed_id, do_print=False), key=lambda x: x["trade_id"]
        )
        log.info(
            f"guessed_id {guessed_id} earliest ts {ts_to_date(guessed_chunk[0]['timestamp'] / 1000)[:19]} "
            f"last ts {ts_to_date(guessed_chunk[-1]['timestamp'] / 1000)[:19]} target ts "
            f"{ts_to_date(timestamp / 1000)[:19]}"
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
                    id_at_match - match.index[0], id_at_match - match.index[0] + len(daily_ticks)
                )
                # trade_ids = np.arange(id_at_match, id_at_match + len(daily_ticks.loc[match.index:]))
                return match, id_at_match
        raise Exception("unable to make trade ids")

    async def get_csv_gz(self, base_url, symbol, date, df_for_id_matching):
        """
        Fetches a full day of trades from the Bybit repository.
        @param symbol: Symbol to fetch.
        @param date: Day to download.
        @return: Dataframe with full day.
        """
        log.info("Fetching symbol=%s, date=%s", symbol, date)
        url = f"{base_url}{symbol.upper()}/{symbol.upper()}{date}.csv.gz"
        df = pd.DataFrame(columns=["trade_id", "price", "qty", "timestamp", "is_buyer_maker"])
        try:
            resp = urllib.request.urlopen(url)
            with gzip.open(io.BytesIO(resp.read())) as f:
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
            log.info("Failed to fetch %s: %s", date, e)
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
            log.info("%s not found", self.config["exchange"])
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
            except Exception:
                first_time = sys.maxsize
                last_time = sys.maxsize
            if (
                not verified
                and last_time >= self.start_time
                and (self.end_time == -1 or (first_time <= self.end_time))
                or last_time == sys.maxsize
            ):
                log.info("Validating file: %s; date=%s", f, ts_to_date(first_time / 1000))
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
                        log.info(
                            "Filling gaps from id %s to id %s",
                            gaps["start"].iloc[i],
                            gaps["end"].iloc[i],
                        )
                        current_id = gaps["start"].iloc[i]
                        while current_id < gaps["end"].iloc[i] and utc_ms() - current_time > 10000:
                            loop_start = time.time()
                            try:
                                fetched_new_trades = await self.bot.fetch_ticks(int(current_id))
                                tf = self.transform_ticks(fetched_new_trades)
                                if tf.empty:
                                    log.info("Response empty. No new trades, exiting...")
                                    await asyncio.sleep(
                                        max(
                                            0.0, self.fetch_delay_seconds - time.time() + loop_start
                                        )
                                    )
                                    break
                                if current_id == tf["trade_id"].iloc[-1]:
                                    log.info("Same trade ID again. No new trades, exiting...")
                                    await asyncio.sleep(
                                        max(
                                            0.0, self.fetch_delay_seconds - time.time() + loop_start
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
                                log.info("Failed to fetch or transform...")
                            await asyncio.sleep(
                                max(0.0, self.fetch_delay_seconds - time.time() + loop_start)
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
                    log.info("Removed file fragment: %s", f)

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
                df = pd.DataFrame(
                    columns=["trade_id", "price", "qty", "timestamp", "is_buyer_maker"]
                )

                months_done = set()
                months_failed = set()
                for day in days:
                    month = day[:7]
                    if month in months_done:
                        continue
                    if month in months_failed:
                        tf = self.get_zip(self.daily_base_url, self.config["symbol"], day)
                        if tf.empty:
                            log.error("failed to fetch daily: %s", day)
                            continue
                    else:
                        tf = self.get_zip(self.monthly_base_url, self.config["symbol"], month)
                        if tf.empty:
                            log.error("failed to fetch monthly", month)
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

            if (
                current_id <= end_id
                and current_time <= end_time
                and utc_ms() - current_time > 10000
            ):
                log.info(
                    "Downloading from %s to %s",
                    ts_to_date(float(current_time) / 1000),
                    ts_to_date(float(end_time) / 1000),
                )

            while (
                current_id <= end_id
                and current_time <= end_time
                and utc_ms() - current_time > 10000
            ):
                loop_start = time.time()
                fetched_new_trades = await self.bot.fetch_ticks(int(current_id))
                tf = self.transform_ticks(fetched_new_trades)
                if tf.empty:
                    log.info("Response empty. No new trades, exiting...")
                    await asyncio.sleep(
                        max(0.0, self.fetch_delay_seconds - time.time() + loop_start)
                    )
                    break
                if current_id == tf["trade_id"].iloc[-1]:
                    log.info("Same trade ID again. No new trades, exiting...")
                    await asyncio.sleep(
                        max(0.0, self.fetch_delay_seconds - time.time() + loop_start)
                    )
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
                await asyncio.sleep(max(0.0, self.fetch_delay_seconds - time.time() + loop_start))
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
        except Exception:
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
            log.error("Error in determining earliest time: %s", e)
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
            log.error("Error in determining latest time: %s", e)
            latest_time = self.end_time

        array = np.zeros(
            (int((latest_time - earliest_time) / sample_size_ms + 1), 3), dtype=np.float64
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
                chunk.timestamp.iloc[-1] // sample_size_ms * sample_size_ms
                - 1
                - (1 * sample_size_ms)
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

            log.info("\rloaded chunk of data", f, ts_to_date(float(f.split("_")[2]) / 1000))

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

        log.info("Saving single file with %d ticks to %s ...", len(array), self.tick_filepath)
        np.save(self.tick_filepath, array)
        log.info("Saved single file!")

    async def get_sampled_ticks(self) -> np.ndarray:
        """
        Function for direct use in the backtester. Checks if the numpy arrays exist and if so loads them.
        If they do not exist or if their length doesn't match, download the missing data and create them.
        @return: numpy array.
        """
        if os.path.exists(self.tick_filepath):
            log.info("Loading cached tick data from %s", self.tick_filepath)
            tick_data = np.load(self.tick_filepath)
            return tick_data
        await self.download_ticks()
        await self.prepare_files()
        tick_data = np.load(self.tick_filepath)
        return tick_data


async def _main(args: argparse.Namespace) -> None:
    config = await prepare_backtest_config(args)
    downloader = Downloader(config)
    await downloader.download_ticks()
    if not args.download_only:
        await downloader.prepare_files()


def main(args: argparse.Namespace) -> None:
    asyncio.run(_main(args))


def setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-d", "--download-only", help="download only, do not dump ticks caches", action="store_true"
    )
    add_backtesting_argparse_args(parser)
    parser.set_defaults(func=main)


def validate_argparse_parsed_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    pass
