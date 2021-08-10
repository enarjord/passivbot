import asyncio
import datetime
import os
import sys
from time import time
from typing import Tuple, List, Union

import numpy as np
import pandas as pd
from dateutil import parser

from bots.base_live_bot import LiveConfig
from definitions.candle import empty_candle
from definitions.tick import Tick, empty_tick_list
from helpers.misc import make_get_filepath, ts_to_date, get_filenames, get_utc_now_timestamp
from helpers.optimized import convert_array_to_tick_list, prepare_candles, candles_to_array
from helpers.print_functions import print_


class Downloader:
    """
    Downloader class for tick data. Fetches data from specified time until now or specified time.
    """

    def __init__(self, config: dict):
        self.fetch_delay_seconds = 0.75
        self.config = config
        self.tick_interval = 0.25
        self.spot = 'spot' in config and config['spot']
        self.tick_filepath = os.path.join(config["caches_dirpath"], f"{config['session_name']}_ticks_cache")
        os.makedirs(config["caches_dirpath"], exist_ok=True)
        try:
            self.start_time = int(
                parser.parse(self.config["start_date"]).replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
        except Exception:
            raise Exception(f"Unrecognized date format for start time {config['start_date']}")
        try:
            self.end_time = int(
                parser.parse(self.config["end_date"]).replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
            if self.end_time > get_utc_now_timestamp():
                raise Exception(f"End date later than current time {config['end_date']}")
        except Exception:
            raise Exception(f"Unrecognized date format for end time {config['end_date']}")
        if "historical_data_path" in self.config and self.config["historical_data_path"]:
            self.filepath = make_get_filepath(
                os.path.join(self.config["historical_data_path"], "historical_data", self.config["exchange"],
                             f"agg_trades_{'spot' if self.spot else 'futures'}", self.config["symbol"], ""))
        else:
            self.filepath = make_get_filepath(os.path.join("historical_data", self.config["exchange"],
                                                           f"agg_trades_{'spot' if self.spot else 'futures'}",
                                                           self.config["symbol"], ""))

    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[bool, pd.DataFrame, pd.DataFrame]:
        """
        Validates a dataframe and detects gaps in it. Also detects missing trades in the beginning and end.
        :param df: Dataframe to check for gaps.
        :return: A tuple with following result: if missing values present, the cleaned dataframe, a dataframe with start and end of gaps.
        """
        df.sort_values("trade_id", inplace=True)
        df.drop_duplicates("trade_id", inplace=True)
        df.reset_index(drop=True, inplace=True)
        missing_end_frame = df["trade_id"][df["trade_id"].diff() != 1]
        gaps = pd.DataFrame()
        gaps["start"] = df.iloc[missing_end_frame[1:].index - 1]["trade_id"].tolist()
        gaps["end"] = missing_end_frame[1:].tolist()
        if missing_ids := df["trade_id"].iloc[0] % 100000 != 0:
            gaps.append({"start": df["trade_id"].iloc[0] - missing_ids, "end": df["trade_id"].iloc[0] - 1},
                        ignore_index=True)
        if missing_ids := df["trade_id"].iloc[-1] % 100000 != 99999:
            gaps.append({"start": df["trade_id"].iloc[-1], "end": df["trade_id"].iloc[-1] + (100000 - missing_ids - 1)},
                        ignore_index=True)
        missing_ids = df["trade_id"].iloc[0] % 100000
        if missing_ids != 0:
            gaps = gaps.append({"start": df["trade_id"].iloc[0] - missing_ids, "end": df["trade_id"].iloc[0] - 1},
                               ignore_index=True)
        missing_ids = df["trade_id"].iloc[-1] % 100000
        if missing_ids != 99999:
            gaps = gaps.append(
                {"start": df["trade_id"].iloc[-1], "end": df["trade_id"].iloc[-1] + (100000 - missing_ids - 1)},
                ignore_index=True)
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
        :param path: The path to the dataframe.
        :return: The read dataframe.
        """
        try:
            df = pd.read_csv(path,
                             dtype={"trade_id": np.int64, "price": np.float64, "qty": np.float64, "timestamp": np.int64,
                                    "is_buyer_maker": np.int8})
        except ValueError as e:
            df = pd.DataFrame()
            print_(['Error in reading dataframe', e], n=True)
        return df

    def save_dataframe(self, df: pd.DataFrame, filename: str, missing: bool, verified: bool) -> str:
        """
        Saves a processed dataframe. Creates the name based on first and last trade id and first and last timestamp.
        Deletes dataframes that are obsolete. For example, when gaps were filled.
        :param df: The dataframe to save.
        :param filename: The current name of the dataframe.
        :param missing: If the dataframe had gaps.
        :return:
        """
        if verified:
            new_name = f'{df["trade_id"].iloc[0]}_{df["trade_id"].iloc[-1]}_{df["timestamp"].iloc[0]}_{df["timestamp"].iloc[-1]}_verified.csv'
        else:
            new_name = f'{df["trade_id"].iloc[0]}_{df["trade_id"].iloc[-1]}_{df["timestamp"].iloc[0]}_{df["timestamp"].iloc[-1]}.csv'
        if new_name != filename:
            print_(['Saving file', new_name])
            df.to_csv(os.path.join(self.filepath, new_name), index=False)
            new_name = ""
            try:
                os.remove(os.path.join(self.filepath, filename))
                print_(['Removed file', filename])
            except:
                pass
        elif missing:
            print_(['Replacing file', filename])
            df.to_csv(os.path.join(self.filepath, filename), index=False)
        else:
            new_name = ""
        return new_name

    def transform_ticks(self, ticks: List[Tick]) -> pd.DataFrame:
        """
        Transforms tick data into a cleaned dataframe with correct data types.
        :param ticks: List of tick dictionaries.
        :return: Clean dataframe with correct data types.
        """
        ticks = [{"trade_id": tick.trade_id, "price": tick.price, "qty": tick.quantity, "timestamp": tick.timestamp,
                  "is_buyer_maker": tick.is_buyer_maker} for tick in ticks]
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

    async def download_ticks(self):
        """
        Searches for previously downloaded files and fills gaps in them if necessary.
        Downloads any missing data based on the specified time frame.
        :return:
        """
        config = LiveConfig(self.config["symbol"], self.config["user"], self.config["exchange"],
                            self.config["market_type"], 1, 1.0, 0.0, 0.0)
        if self.config["exchange"] == "binance":
            from bots.binance import BinanceBot
            self.bot = BinanceBot(config, None)
        elif self.config["exchange"] == "bybit":
            pass
            # from bots.bybit import BybitBot
            # self.bot = BybitBot(config, None)
        else:
            print(self.config["exchange"], 'not found')
            return
        await self.bot.fetch_exchange_info()

        filenames = get_filenames(self.filepath)
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
            if not verified and last_time >= self.start_time and (
                    self.end_time == -1 or (first_time <= self.end_time)) or last_time == sys.maxsize:
                print_(['Validating file', f])
                df = self.read_dataframe(os.path.join(self.filepath, f))
                missing, df, gaps = self.validate_dataframe(df)
                exists = False
                if gaps.empty:
                    first_id = df["trade_id"].iloc[0]
                    self.save_dataframe(df, f, missing, True)
                else:
                    first_id = df["trade_id"].iloc[0] if df["trade_id"].iloc[0] < gaps["start"].iloc[0] else \
                        gaps["start"].iloc[0]
                if not gaps.empty and (f != filenames[-1] or str(first_id - first_id % 100000) not in f):
                    last_id = df["trade_id"].iloc[-1]
                    for i in filenames:
                        tmp_first_id = int(i.split("_")[0])
                        tmp_last_id = int(i.split("_")[1].replace('.csv', ''))
                        if (first_id - first_id % 100000) == tmp_first_id and (
                                (first_id - first_id % 100000 + 99999) == tmp_last_id or (
                                highest_id == tmp_first_id or highest_id == tmp_last_id) or highest_id > last_id) and first_id != 1 and i != f:
                            exists = True
                            break
                if missing and df["timestamp"].iloc[-1] > self.start_time and not exists:
                    current_time = df["timestamp"].iloc[-1]
                    for i in gaps.index:
                        print_(['Filling gaps from id', gaps["start"].iloc[i], 'to id', gaps["end"].iloc[i]])
                        current_id = gaps["start"].iloc[i]
                        while current_id < gaps["end"].iloc[i] and get_utc_now_timestamp() - current_time > 10000:
                            loop_start = time()
                            try:
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
                                current_id = tf["trade_id"].iloc[-1]
                                df = pd.concat([df, tf])
                                df.sort_values("trade_id", inplace=True)
                                df.drop_duplicates("trade_id", inplace=True)
                                df = df[df["trade_id"] <= gaps["end"].iloc[i] - gaps["end"].iloc[i] % 100000 + 99999]
                                df.reset_index(drop=True, inplace=True)
                                current_time = df["timestamp"].iloc[-1]
                            except Exception:
                                print("Failed to fetch or transform...")
                            await asyncio.sleep(max(0.0, self.fetch_delay_seconds - time() + loop_start))
                if not df.empty:
                    if df["trade_id"].iloc[-1] > highest_id:
                        highest_id = df["trade_id"].iloc[-1]
                if not exists:
                    tf = df[df["trade_id"].mod(100000) == 0]
                    if len(tf) > 1:
                        df = df[:tf.index[-1]]
                    nf = self.save_dataframe(df, f, missing, verified)
                    mod_files.append(nf)
                elif df["trade_id"].iloc[0] != 1:
                    os.remove(os.path.join(self.filepath, f))
                    print_(['Removed file fragment', f])

        chunk_gaps = []
        filenames = get_filenames(self.filepath)
        prev_last_id = 0
        prev_last_time = self.start_time
        for f in filenames:
            first_id = int(f.split("_")[0])
            last_id = int(f.split("_")[1])
            first_time = int(f.split("_")[2])
            last_time = int(f.split("_")[3].split(".")[0])
            if first_id - 1 != prev_last_id and f not in mod_files and first_time >= prev_last_time and \
                    first_time >= self.start_time and self.end_time < first_time and not prev_last_time > self.end_time:
                chunk_gaps.append((prev_last_time, self.end_time, prev_last_id, 0))
            if first_time >= self.start_time or last_time >= self.start_time:
                prev_last_id = last_id
                prev_last_time = last_time

        if len(filenames) < 1:
            chunk_gaps.append((self.start_time, self.end_time, 0, 0))
        elif prev_last_time < self.end_time:
            chunk_gaps.append((prev_last_time, self.end_time, prev_last_id, 0))

        for gaps in chunk_gaps:
            start_time, end_time, start_id, end_id = gaps

            current_id = start_id + 1
            current_time = start_time

            fetched_new_trades = await self.bot.fetch_ticks(from_id=1)
            tf = self.transform_ticks(fetched_new_trades)
            earliest = tf['timestamp'].iloc[0]

            if earliest > start_time:
                start_time = earliest
                current_time = start_time

            tmp = pd.date_range(start=datetime.datetime.fromtimestamp(start_time / 1000, datetime.timezone.utc).date(),
                                end=datetime.datetime.fromtimestamp(end_time / 1000, datetime.timezone.utc).date(),
                                freq='D').to_pydatetime()

            days = [date.strftime("%Y-%m-%d") for date in tmp]
            current_month = ts_to_date(time() - 60 * 60 * 3)[:7]
            months = sorted([e for e in set([d[:7] for d in days]) if e != current_month])
            dates = sorted(months + [d for d in days if d[:7] not in months])
            dates = [(i.split('-')[0], i.split('-')[1], i.split('-')[2]) if len(i.split('-')) == 3 else (
                i.split('-')[0], i.split('-')[1]) for i in dates]

            df = pd.DataFrame(columns=['trade_id', 'price', 'qty', 'timestamp', 'is_buyer_maker'])

            for date in dates:
                tf = self.bot.fetch_from_repo(date)
                if tf.empty:
                    break
                tf = tf[tf['timestamp'] >= start_time]
                if end_time != -1:
                    tf = tf[tf['timestamp'] <= end_time]
                if start_id != 0:
                    tf = tf[tf['trade_id'] > start_id]
                if end_id != 0:
                    tf = tf[tf['trade_id'] <= end_id]
                if df.empty:
                    df = tf
                else:
                    df = pd.concat([df, tf])
                df.sort_values("trade_id", inplace=True)
                df.drop_duplicates("trade_id", inplace=True)
                df.reset_index(drop=True, inplace=True)

                if not df.empty and (
                        (df['trade_id'].iloc[0] % 100000 == 0 and len(df) >= 100000) or df['trade_id'].iloc[
                    0] % 100000 != 0):
                    for index, row in df[df['trade_id'] % 100000 == 0].iterrows():
                        if index != 0:
                            self.save_dataframe(df[(df['trade_id'] >= row['trade_id'] - 1000000) & (
                                    df['trade_id'] < row['trade_id'])], "", True, False)
                            df = df[df['trade_id'] >= row['trade_id']]
                if not df.empty:
                    start_id = df["trade_id"].iloc[0] - 1
                    start_time = df["timestamp"].iloc[0]
                    current_time = df["timestamp"].iloc[-1]
                    current_id = df["trade_id"].iloc[-1] + 1

            if start_id == 0:
                fetched_new_trades = await self.bot.fetch_ticks(start_time=start_time)
                df = self.transform_ticks(fetched_new_trades)
                current_id = df["trade_id"].iloc[-1] + 1
                current_time = df["timestamp"].iloc[-1]

            end_id = sys.maxsize if end_id == 0 else end_id - 1

            if current_id <= end_id and current_time <= end_time and get_utc_now_timestamp() - current_time > 10000:
                print_(['Downloading from', ts_to_date(float(current_time) / 1000), 'to',
                        ts_to_date(float(end_time) / 1000)])

            while current_id <= end_id and current_time <= end_time and get_utc_now_timestamp() - current_time > 10000:
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
                        self.save_dataframe(df[:tf.index[-1]], "", True, False)
                        df = df[tf.index[-1]:]
                    elif df["trade_id"].iloc[0] % 100000 != 0 and len(tf) == 1:
                        self.save_dataframe(df[:tf.index[-1]], "", True, False)
                        df = df[tf.index[-1]:]
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

    def get_unabridged_df(self):
        filenames = get_filenames(self.filepath)
        start_index = 0
        for i in range(len(filenames)):
            if int(filenames[i].split("_")[2]) <= self.start_time <= int(filenames[i].split("_")[3].split(".")[0]):
                start_index = i
                break
        end_index = -1
        if self.end_time != -1:
            for i in range(len(filenames)):
                if int(filenames[i].split("_")[2]) <= self.end_time <= int(filenames[i].split("_")[3].split(".")[0]):
                    end_index = i
                    break
        filenames = filenames[start_index:] if end_index == -1 else filenames[start_index:end_index + 1]
        df = pd.DataFrame()
        chunks = []
        for f in filenames:
            chunk = pd.read_csv(os.path.join(self.filepath, f)).set_index('trade_id')
            if self.end_time != -1:
                chunk = chunk[(chunk['timestamp'] >= self.start_time) & (chunk['timestamp'] <= self.end_time)]
            else:
                chunk = chunk[(chunk['timestamp'] >= self.start_time)]
            chunks.append(chunk)
            if len(chunks) >= 100:
                if df.empty:
                    df = pd.concat(chunks, axis=0)
                else:
                    chunks.insert(0, df)
                    df = pd.concat(chunks, axis=0)
                chunks = []
            print('\rloaded chunk of data', f, ts_to_date(float(f.split("_")[2]) / 1000), end='     ')
        print()
        if chunks:
            if df.empty:
                df = pd.concat(chunks, axis=0)
            else:
                chunks.insert(0, df)
                df = pd.concat(chunks, axis=0)
            del chunks
        return df

    async def prepare_files(self, single_file: bool = True):
        """
        Takes downloaded data and prepares numpy arrays consisting of candles for use in backtesting.
        :param single_file: If a single array should be created or multiple ones.
        :return:
        """
        filenames = get_filenames(self.filepath)
        start_index = 0
        for i in range(len(filenames)):
            if int(filenames[i].split("_")[2]) <= self.start_time <= int(filenames[i].split("_")[3].split(".")[0]):
                start_index = i
                break
        end_index = -1
        for i in range(len(filenames)):
            if int(filenames[i].split("_")[2]) <= self.end_time <= int(filenames[i].split("_")[3].split(".")[0]):
                end_index = i
                break
        filenames = filenames[start_index:] if end_index == -1 else filenames[start_index:end_index + 1]

        tick_list = empty_tick_list()
        last_candle = empty_candle()
        last_tick_update = int(self.start_time - (self.start_time % (self.tick_interval * 1000)))
        last_update = int(self.end_time - (self.end_time % (self.tick_interval * 1000))) + int(
            self.tick_interval * 1000)

        if single_file:
            array = np.zeros((int((self.end_time - self.start_time) / (self.tick_interval * 1000)), 6),
                             dtype=np.float64)
        else:
            array = {"timestamp": np.zeros((int((self.end_time - self.start_time) / (self.tick_interval * 1000)), 1),
                                           dtype=np.int64),
                     "open": np.zeros((int((self.end_time - self.start_time) / (self.tick_interval * 1000)), 1),
                                      dtype=np.float32),
                     "high": np.zeros((int((self.end_time - self.start_time) / (self.tick_interval * 1000)), 1),
                                      dtype=np.float32),
                     "low": np.zeros((int((self.end_time - self.start_time) / (self.tick_interval * 1000)), 1),
                                     dtype=np.float32),
                     "close": np.zeros((int((self.end_time - self.start_time) / (self.tick_interval * 1000)), 1),
                                       dtype=np.float32),
                     "volume": np.zeros((int((self.end_time - self.start_time) / (self.tick_interval * 1000)), 1),
                                        dtype=np.float32)}
        current_index = 0
        for f in filenames:
            chunk = pd.read_csv(os.path.join(self.filepath, f),
                                dtype={"trade_id": np.float64, "timestamp": np.float64, "price": np.float64,
                                       "qty": np.float64, "is_buyer_maker": np.float64},
                                usecols=["trade_id", "timestamp", "price", "qty", "is_buyer_maker"])[
                ["trade_id", "timestamp", "price", "qty", "is_buyer_maker"]]
            tick_list = convert_array_to_tick_list(tick_list, chunk.values)
            candle_list, tick_list, last_tick_update = prepare_candles(tick_list, last_tick_update, last_update,
                                                                       last_candle, self.tick_interval)
            if len(candle_list) > 0:
                last_candle = candle_list[-1]
            tmp_array = candles_to_array(candle_list)
            if single_file:
                array[current_index:current_index + len(tmp_array)] = tmp_array
            else:
                array["timestamp"][current_index:current_index + len(tmp_array)] = np.reshape(
                    np.asarray(tmp_array[:, 0], dtype=np.int64), (len(tmp_array[:, 0]), 1))
                array["open"][current_index:current_index + len(tmp_array)] = np.reshape(
                    np.asarray(tmp_array[:, 1], dtype=np.float32), (len(tmp_array[:, 1]), 1))
                array["high"][current_index:current_index + len(tmp_array)] = np.reshape(
                    np.asarray(tmp_array[:, 2], dtype=np.float32), (len(tmp_array[:, 2]), 1))
                array["low"][current_index:current_index + len(tmp_array)] = np.reshape(
                    np.asarray(tmp_array[:, 3], dtype=np.float32), (len(tmp_array[:, 3]), 1))
                array["close"][current_index:current_index + len(tmp_array)] = np.reshape(
                    np.asarray(tmp_array[:, 4], dtype=np.float32), (len(tmp_array[:, 4]), 1))
                array["volume"][current_index:current_index + len(tmp_array)] = np.reshape(
                    np.asarray(tmp_array[:, 5], dtype=np.float32), (len(tmp_array[:, 5]), 1))
            current_index += len(tmp_array)
            print('\rloaded chunk of data', f, ts_to_date(float(f.split("_")[2]) / 1000), end='     ')
        print('\n')

        if single_file:
            print_(["Saving single file with", len(array), " ticks to", self.tick_filepath + ".npy", "..."])
            np.save(self.tick_filepath + ".npy", array)
            print_(["Saved single file!"])
        else:
            for key, value in array.items():
                print_([f"Saving {key} file with", len(value), " ticks to", self.tick_filepath + "_" + key + ".npy",
                        "..."])
                np.save(self.tick_filepath + "_" + key + ".npy", value)
                print_([f"Saved {key} file!"])

    async def get_candles(self, single_file: bool = True) -> Union[np.ndarray, dict]:
        """
        Function for direct use in the backtester. Checks if the numpy arrays exist and if so loads it.
        If they do not exist or if their length doesn't match, download the missing data and create them.
        :param single_file: If a single array should be created or multiple ones.
        :return: Candles in a numpy array with same data type or in a dictionary in different data types.
        """
        if single_file:
            if os.path.exists(self.tick_filepath + ".npy"):
                print_(['Loading cached tick data from', self.tick_filepath])
                candle_data = np.load(self.tick_filepath + ".npy")
                return candle_data
            await self.download_ticks()
            await self.prepare_files(single_file)
            candle_data = np.load(self.tick_filepath)
            return candle_data
        else:
            exists = True
            array = {
                "timestamp": None,
                "open": None,
                "high": None,
                "low": None,
                "close": None,
                "volume": None}
            for file in array.keys():
                if not os.path.exists(self.tick_filepath + "_" + file + ".npy"):
                    exists = False
                    break
            if exists:
                for file in array.keys():
                    print_(['Loading cached tick data from', self.tick_filepath + "_" + file + ".npy"])
                    candle_data = np.load(self.tick_filepath + "_" + file + ".npy")
                    array[file] = candle_data
                return array
            else:
                await self.download_ticks()
                await self.prepare_files(single_file)
                for file in array.keys():
                    candle_data = np.load(self.tick_filepath + "_" + file + ".npy")
                    array[file] = candle_data
                return array
