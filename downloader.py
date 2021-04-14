import gc

import hjson
import pandas as pd
from dateutil import parser, tz

from binance import create_bot as create_bot_binance
from bybit import create_bot as create_bot_bybit
from passivbot import *


class Downloader:
    """
    Downloader class for tick data. Fetches data from specified time until now or specified time.
    """

    def __init__(self, backtest_config: dict):
        self.backtest_config = backtest_config
        try:
            self.start_time = int(parser.parse(self.backtest_config["start_date"]).replace(
                tzinfo=datetime.timezone.utc).timestamp() * 1000)
        except Exception:
            print("Not recognized date format for start time.")
        self.end_time = self.backtest_config["end_date"]
        if self.end_time != -1:
            try:
                self.end_time = int(
                    parser.parse(self.end_time).replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)
            except Exception:
                print("Not recognized date format for end time.")

    def validate_dataframe(self, df: pd.DataFrame) -> tuple:
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

    def read_dataframe(self, path) -> pd.DataFrame:
        """
        Reads a dataframe with correct data types.
        @param path: The path to the dataframe.
        @return: The read dataframe.
        """
        try:
            df = pd.read_csv(path,
                             dtype={"trade_id": np.int64, "price": np.float64, "qty": np.float64, "timestamp": np.int64,
                                    "is_buyer_maker": np.int8})
        except ValueError as e:
            df = pd.read_csv(path)
            df = df.drop("side", axis=1).join(pd.Series(df.side == "Sell", name="is_buyer_maker", index=df.index))
            df = df.astype({"trade_id": np.int64, "price": np.float64, "qty": np.float64, "timestamp": np.int64,
                            "is_buyer_maker": np.int8})
        return df

    def save_dataframe(self, df, filename, missing):
        """
        Saves a processed dataframe. Creates the name based on first and last trade id and first and last timestamp.
        Deletes dataframes that are obsolete. For example, when gaps were filled.
        @param df: The dataframe to save.
        @param filename: The current name of the dataframe.
        @param missing: If the dataframe had gaps.
        @return:
        """
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
        return sorted([f for f in os.listdir(self.filepath) if f.endswith(".csv")],
                      key=lambda x: int(eval(x[:x.find("_")].replace(".cs", "").replace("v", ""))))

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
            df = self.transform_ticks(ticks)
            return df
        except:
            print_(['Finding id for start time...'])
            ticks = await self.bot.fetch_ticks()
            df = self.transform_ticks(ticks)
            highest_id = df["trade_id"].iloc[-1]
            prev_div = []
            first_ts = df["timestamp"].iloc[0]
            last_ts = df["timestamp"].iloc[-1]
            first_id = df["trade_id"].iloc[0]
            length = len(df)
            while not start_time >= first_ts or not start_time <= last_ts:
                await asyncio.sleep(0.75)
                nw_id, prev_div, forward = self.new_id(first_ts, last_ts, first_id, length, start_time, prev_div)
                print_(['Current time span from', df["timestamp"].iloc[0], 'to', df["timestamp"].iloc[-1],
                        'with earliest trade id', df["trade_id"].iloc[0], 'estimating distance of', forward, 'trades'])
                if nw_id > highest_id:
                    nw_id = highest_id
                try:
                    ticks = await self.bot.fetch_ticks(from_id=nw_id)
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
            print_(['Found id for start time!'])
            return df[df["timestamp"] >= start_time]

    async def download_ticks(self):
        """
        Searches for previously downloaded files and fills gaps in them if necessary.
        Downloads any missing data based on the specified time frame.
        @return:
        """
        if "historical_data_path" in self.backtest_config and self.backtest_config["historical_data_path"]:
            self.filepath = make_get_filepath(
                os.path.join(self.backtest_config["historical_data_path"], "historical_data",
                             self.backtest_config["exchange"], "agg_trades_futures",
                             self.backtest_config["symbol"], ""))
        else:
            self.filepath = make_get_filepath(
                os.path.join("historical_data", self.backtest_config["exchange"], "agg_trades_futures",
                             self.backtest_config["symbol"], ""))

        if self.backtest_config["exchange"] == "binance":
            self.bot = await create_bot_binance(self.backtest_config["user"],
                                                {**load_live_settings("binance", do_print=False),
                                                 **{"symbol": self.backtest_config["symbol"]}})
        elif self.backtest_config["exchange"] == "bybit":
            self.bot = await create_bot_bybit(self.backtest_config["user"],
                                              {**load_live_settings("bybit", do_print=False),
                                               **{"symbol": self.backtest_config["symbol"]}})
        else:
            print(self.backtest_config["exchange"], 'not found')
            return

        filenames = self.get_filenames()
        mod_files = []
        highest_id = 0
        for f in filenames:
            try:
                first_time = int(f.split("_")[2])
                last_time = int(f.split("_")[3].split(".")[0])
            except:
                first_time = sys.maxsize
                last_time = sys.maxsize
            if last_time >= self.start_time and (
                    self.end_time == -1 or (first_time <= self.end_time)) or last_time == sys.maxsize:
                df = self.read_dataframe(os.path.join(self.filepath, f))
                print_(['Validating file', f])
                missing, df, gaps = self.validate_dataframe(df)
                exists = False
                if gaps.empty:
                    first_id = df["trade_id"].iloc[0]
                else:
                    first_id = df["trade_id"].iloc[0] if df["trade_id"].iloc[0] < gaps["start"].iloc[0] else \
                        gaps["start"].iloc[0]
                if not gaps.empty and (f != filenames[-1] or str(first_id - first_id % 100000) not in f):
                    last_id = df["trade_id"].iloc[-1]
                    for i in filenames:
                        if str(first_id - first_id % 100000) in i and (str(
                                first_id - first_id % 100000 + 99999) in i or str(
                            highest_id) in i or highest_id > last_id) and first_id != 1 and i != f:
                            exists = True
                            break
                if missing and df["timestamp"].iloc[-1] > self.start_time and not exists:
                    current_time = df["timestamp"].iloc[-1]
                    for i in gaps.index:
                        print_(['Filling gaps from id', gaps["start"].iloc[i], 'to id', gaps["end"].iloc[i]])
                        current_id = gaps["start"].iloc[i]
                        while current_id < gaps["end"].iloc[i] and int(
                                datetime.datetime.now(tz.UTC).timestamp() * 1000) - current_time > 10000:
                            try:
                                fetched_new_trades = await self.bot.fetch_ticks(int(current_id))
                                tf = self.transform_ticks(fetched_new_trades)
                                if tf.empty:
                                    print_(["Response empty. No new trades, exiting..."])
                                    break
                                if current_id == tf["trade_id"].iloc[-1]:
                                    print_(["Same trade ID again. No new trades, exiting..."])
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
                            await asyncio.sleep(0.75)
                if not df.empty:
                    if df["trade_id"].iloc[-1] > highest_id:
                        highest_id = df["trade_id"].iloc[-1]
                if not exists:
                    tf = df[df["trade_id"].mod(100000) == 0]
                    if len(tf) > 1:
                        df = df[:tf.index[-1]]
                    nf = self.save_dataframe(df, f, missing)
                    mod_files.append(nf)
                elif df["trade_id"].iloc[0] != 1:
                    os.remove(os.path.join(self.filepath, f))
                    print_(['Removed file fragment', f])

        chunk_gaps = []
        filenames = self.get_filenames()
        prev_last_id = 0
        prev_last_time = self.start_time
        for f in filenames:
            first_id = int(f.split("_")[0])
            last_id = int(f.split("_")[1])
            first_time = int(f.split("_")[2])
            last_time = int(f.split("_")[3].split(".")[0])
            if first_id - 1 != prev_last_id and f not in mod_files:
                if first_time >= prev_last_time and first_time >= self.start_time:
                    if self.end_time != -1 and self.end_time < first_time and not prev_last_time > self.end_time:
                        chunk_gaps.append((prev_last_time, self.end_time, prev_last_id, 0))
                    elif self.end_time == -1 or self.end_time > first_time:
                        chunk_gaps.append((prev_last_time, first_time, prev_last_id, first_id))
            if first_time >= self.start_time or last_time >= self.start_time:
                prev_last_id = last_id
                prev_last_time = last_time

        if len(filenames) < 1:
            chunk_gaps.append((self.start_time, self.end_time, 0, 0))
        else:
            if self.end_time == -1:
                chunk_gaps.append((prev_last_time, self.end_time, prev_last_id, 0))
            elif prev_last_time < self.end_time:
                chunk_gaps.append((prev_last_time, self.end_time, prev_last_id, 0))

        for gaps in chunk_gaps:
            start_time, end_time, start_id, end_id = gaps

            if start_id == 0:
                df = await self.find_time(start_time)
                current_id = df["trade_id"].iloc[-1] + 1
                current_time = df["timestamp"].iloc[-1]
            else:
                df = pd.DataFrame()
                current_id = start_id + 1
                current_time = start_time

            if end_id == 0:
                end_id = sys.maxsize
            else:
                end_id = end_id - 1

            if end_time == -1:
                end_time = sys.maxsize

            if current_id <= end_id and current_time <= end_time and int(
                    datetime.datetime.now(tz.UTC).timestamp() * 1000) - current_time > 10000:
                if end_time == sys.maxsize:
                    print_(['Downloading from', ts_to_date(float(current_time) / 1000), 'to current time...'])
                else:
                    print_(['Downloading from', ts_to_date(float(current_time) / 1000), 'to',
                            ts_to_date(float(end_time) / 1000)])

            while current_id <= end_id and current_time <= end_time and int(
                    datetime.datetime.now(tz.UTC).timestamp() * 1000) - current_time > 10000:
                fetched_new_trades = await self.bot.fetch_ticks(int(current_id))
                tf = self.transform_ticks(fetched_new_trades)
                if tf.empty:
                    print_(["Response empty. No new trades, exiting..."])
                    break
                if current_id == tf["trade_id"].iloc[-1]:
                    print_(["Same trade ID again. No new trades, exiting..."])
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
                        self.save_dataframe(df[:tf.index[-1]], "", True)
                        df = df[tf.index[-1]:]
                    elif df["trade_id"].iloc[0] % 100000 != 0 and len(tf) == 1:
                        self.save_dataframe(df[:tf.index[-1]], "", True)
                        df = df[tf.index[-1]:]
                await asyncio.sleep(0.75)
            if not df.empty:
                df = df[df["timestamp"] >= start_time]
                if start_id != 0 and not df.empty:
                    df = df[df["trade_id"] > start_id]
                elif end_id != sys.maxsize and not df.empty:
                    df = df[df["trade_id"] <= end_id]
                elif end_time != sys.maxsize and not df.empty:
                    df = df[df["timestamp"] <= end_time]
                if not df.empty:
                    self.save_dataframe(df, "", True)

        try:
            await self.bot.session.close()
        except:
            pass

    async def prepare_files(self, filepaths: dict, single_file: bool = False):
        """
        Takes downloaded data and prepares numpy arrays for use in backtesting.
        @param filepaths: Dictionary of filepaths.
        @param single_file: If a single array should be created ot multiple ones.
        @return:
        """
        filenames = self.get_filenames()
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
        if end_index == -1:
            filenames = filenames[start_index:]
        else:
            filenames = filenames[start_index:end_index + 1]

        chunks = []
        df = pd.DataFrame()

        if single_file:
            for f in filenames:
                chunk = pd.read_csv(os.path.join(self.filepath, f),
                                    dtype={"price": np.float64, "is_buyer_maker": np.float64, "timestamp": np.float64},
                                    usecols=["price", "is_buyer_maker", "timestamp"])
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
            print('\n')
            if chunks:
                if df.empty:
                    df = pd.concat(chunks, axis=0)
                else:
                    chunks.insert(0, df)
                    df = pd.concat(chunks, axis=0)
                del chunks
            print_(["Saving single file with", len(df), " ticks to", filepaths["tick_filepath"], "..."])
            np.save(filepaths["tick_filepath"], df[["price", "is_buyer_maker", "timestamp"]])
            print_(["Saved single file!"])
        else:
            for f in filenames:
                start_time = int(f.split("_")[2])
                end_time = int(f.split("_")[3].split(".")[0])
                if (start_time <= self.start_time <= end_time) or (
                        self.end_time != -1 and start_time <= self.end_time <= end_time):
                    chunk = pd.read_csv(os.path.join(self.filepath, f),
                                        dtype={"timestamp": np.int64, "price": np.float64},
                                        usecols=["timestamp", "price"])
                    if self.end_time != -1:
                        chunk = chunk[(chunk['timestamp'] >= self.start_time) & (chunk['timestamp'] <= self.end_time)][
                            ["price"]]
                    else:
                        chunk = chunk[(chunk['timestamp'] >= self.start_time)][["price"]]
                else:
                    chunk = pd.read_csv(os.path.join(self.filepath, f), dtype={"price": np.float64}, usecols=["price"])
                chunks.append(chunk)
                if len(chunks) >= 100:
                    if df.empty:
                        df = pd.concat(chunks, axis=0)
                    else:
                        chunks.insert(0, df)
                        df = pd.concat(chunks, axis=0)
                    chunks = []
                print('\rloaded chunk of price data', f, ts_to_date(float(f.split("_")[2]) / 1000), end='     ')
            print('\n')
            if chunks:
                if df.empty:
                    df = pd.concat(chunks, axis=0)
                else:
                    chunks.insert(0, df)
                    df = pd.concat(chunks, axis=0)
                del chunks
            print_(["Saving price file with", len(df), " ticks to", filepaths["price_filepath"], "..."])
            np.save(filepaths["price_filepath"], df["price"])
            print_(["Saved price file!"])

            chunks = []
            df = pd.DataFrame()
            for f in filenames:
                start_time = int(f.split("_")[2])
                end_time = int(f.split("_")[3].split(".")[0])
                if (start_time <= self.start_time <= end_time) or (
                        self.end_time != -1 and start_time <= self.end_time <= end_time):
                    chunk = pd.read_csv(os.path.join(self.filepath, f),
                                        dtype={"timestamp": np.int64, "is_buyer_maker": np.int8},
                                        usecols=["timestamp", "is_buyer_maker"])
                    if self.end_time != -1:
                        chunk = chunk[(chunk['timestamp'] >= self.start_time) & (chunk['timestamp'] <= self.end_time)][
                            ["is_buyer_maker"]]
                    else:
                        chunk = chunk[(chunk['timestamp'] >= self.start_time)][["is_buyer_maker"]]
                else:
                    chunk = pd.read_csv(os.path.join(self.filepath, f), dtype={"is_buyer_maker": np.int8},
                                        usecols=["is_buyer_maker"])
                chunks.append(chunk)
                if len(chunks) >= 100:
                    if df.empty:
                        df = pd.concat(chunks, axis=0)
                    else:
                        chunks.insert(0, df)
                        df = pd.concat(chunks, axis=0)
                    chunks = []
                print('\rloaded chunk of buyer maker data', f, ts_to_date(float(f.split("_")[2]) / 1000), end='     ')
            print('\n')
            if chunks:
                if df.empty:
                    df = pd.concat(chunks, axis=0)
                else:
                    chunks.insert(0, df)
                    df = pd.concat(chunks, axis=0)
                del chunks
            print_(["Saving buyer_maker file with", len(df), " ticks to", filepaths["buyer_maker_filepath"], "..."])
            np.save(filepaths["buyer_maker_filepath"], df["is_buyer_maker"])
            print_(["Saved buyer_maker file!"])

            chunks = []
            df = pd.DataFrame()
            for f in filenames:
                chunk = pd.read_csv(os.path.join(self.filepath, f), dtype=np.int64, usecols=["timestamp"])
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
                print('\rloaded chunk of time data', f, ts_to_date(float(f.split("_")[2]) / 1000), end='     ')
            print('\n')
            if chunks:
                if df.empty:
                    df = pd.concat(chunks, axis=0)
                else:
                    chunks.insert(0, df)
                    df = pd.concat(chunks, axis=0)
                del chunks
            print_(["Saving timestamp file with", len(df), " ticks to", filepaths["time_filepath"], "..."])
            np.save(filepaths["time_filepath"], df["timestamp"])
            print_(["Saved timestamp file!"])

    async def get_ticks(self, single_file: bool = False) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Function for direct use in the backtester. Checks if the numpy arrays exist and if so loads them.
        If they do not exist or if their length doesn't match, download the missing data and create them.
        @return: A tuple of three numpy arrays.
        """
        price_filepath = os.path.join(self.backtest_config["session_dirpath"], f"price_cache.npy")
        buyer_maker_filepath = os.path.join(self.backtest_config["session_dirpath"], f"buyer_maker_cache.npy")
        time_filepath = os.path.join(self.backtest_config["session_dirpath"], f"time_cache.npy")
        tick_filepath = os.path.join(self.backtest_config["session_dirpath"], f"ticks_cache.npy")
        filepaths = {"price_filepath": price_filepath, "buyer_maker_filepath": buyer_maker_filepath,
                     "time_filepath": time_filepath, "tick_filepath": tick_filepath}
        if single_file:
            if os.path.exists(tick_filepath):
                print_(['Loading cached tick data'])
                tick_data = np.load(tick_filepath)
                return tick_data
            await self.download_ticks()
            await self.prepare_files(filepaths, single_file)
            tick_data = np.load(tick_filepath)
            return tick_data
        else:
            if os.path.exists(price_filepath) and os.path.exists(buyer_maker_filepath) and os.path.exists(
                    time_filepath):
                print_(['Loading cached tick data'])
                price_data = np.load(price_filepath)
                buyer_maker_data = np.load(buyer_maker_filepath)
                time_data = np.load(time_filepath)
                if len(price_data) == len(buyer_maker_data) == len(time_data):
                    return price_data, buyer_maker_data, time_data
                else:
                    print_(['Tick data does not match, starting over...'])
                    del price_data
                    del buyer_maker_data
                    del time_data
                    gc.collect()

            await self.download_ticks()
            await self.prepare_files(filepaths, single_file)
            price_data = np.load(price_filepath)
            buyer_maker_data = np.load(buyer_maker_filepath)
            time_data = np.load(time_filepath)
            return price_data, buyer_maker_data, time_data


async def fetch_market_specific_settings(exchange: str, user: str, symbol: str):
    tmp_live_settings = load_live_settings(exchange, do_print=False)
    tmp_live_settings['symbol'] = symbol
    settings_from_exchange = {}
    if exchange == 'binance':
        bot = await create_bot_binance(user, tmp_live_settings)
        settings_from_exchange['maker_fee'] = 0.00018
        settings_from_exchange['taker_fee'] = 0.00036
        settings_from_exchange['exchange'] = 'binance'
    elif exchange == 'bybit':
        bot = await create_bot_bybit(user, tmp_live_settings)
        settings_from_exchange['maker_fee'] = -0.00025
        settings_from_exchange['taker_fee'] = 0.00075
        settings_from_exchange['exchange'] = 'bybit'
    else:
        raise Exception(f'unknown exchange {exchange}')
    if 'inverse' in bot.market_type:
        settings_from_exchange['inverse'] = True
    elif 'linear' in bot.market_type:
        settings_from_exchange['inverse'] = False
    else:
        raise Exception('unknown market type')
    await bot.session.close()
    settings_from_exchange['max_leverage'] = bot.max_leverage
    settings_from_exchange['min_qty'] = bot.min_qty
    settings_from_exchange['min_cost'] = bot.min_cost
    settings_from_exchange['qty_step'] = bot.qty_step
    settings_from_exchange['price_step'] = bot.price_step
    settings_from_exchange['max_leverage'] = bot.max_leverage
    settings_from_exchange['contract_multiplier'] = bot.contract_size
    return settings_from_exchange


async def prep_backtest_config(config_name: str):
    backtest_config = hjson.load(open(f'backtest_configs/{config_name}.hjson'))

    exchange = backtest_config['exchange']
    user = backtest_config['user']
    symbol = backtest_config['symbol']
    session_name = backtest_config['session_name']

    start_date = backtest_config['start_date'].replace(' ', '_').replace(':', '_').replace('.', '_')
    if backtest_config['end_date'] and backtest_config['end_date'] != -1:
        end_date = backtest_config['end_date'].replace(' ', '_').replace(':', '_').replace('.', '_')
    else:
        end_date = 'now'
        backtest_config['end_date'] = -1

    session_dirpath = make_get_filepath(
        os.path.join('backtest_results', exchange, symbol, f"{session_name}_{start_date}_{end_date}", ''))

    if os.path.exists((mss := session_dirpath + 'market_specific_settings.json')):
        market_specific_settings = json.load(open(mss))
    else:
        market_specific_settings = await fetch_market_specific_settings(exchange, user, symbol)
        json.dump(market_specific_settings, open(mss, 'w'))
    backtest_config.update(market_specific_settings)

    # setting absolute min/max ranges
    for key in ['qty_pct', 'ddown_factor', 'ema_span', 'ema_spread',
                'grid_coefficient', 'grid_spacing']:
        if key in backtest_config['ranges']:
            backtest_config['ranges'][key][0] = max(0.0, backtest_config['ranges'][key][0])
    for key in ['qty_pct']:
        if key in backtest_config['ranges']:
            backtest_config['ranges'][key][1] = min(1.0, backtest_config['ranges'][key][1])

    if 'leverage' in backtest_config['ranges']:
        backtest_config['ranges']['leverage'][1] = \
            min(backtest_config['ranges']['leverage'][1],
                backtest_config['max_leverage'])
        backtest_config['ranges']['leverage'][0] = \
            min(backtest_config['ranges']['leverage'][0],
                backtest_config['ranges']['leverage'][1])

    backtest_config['session_dirpath'] = session_dirpath

    return backtest_config


async def main(args: list):
    config_name = args[1]
    backtest_config = await prep_backtest_config(config_name)
    downloader = Downloader(backtest_config)

    price_filepath = os.path.join(backtest_config["session_dirpath"], f"price_cache.npy")
    buyer_maker_filepath = os.path.join(backtest_config["session_dirpath"], f"buyer_maker_cache.npy")
    time_filepath = os.path.join(backtest_config["session_dirpath"], f"time_cache.npy")
    tick_filepath = os.path.join(backtest_config["session_dirpath"], f"ticks_cache.npy")
    filepaths = {"price_filepath": price_filepath, "buyer_maker_filepath": buyer_maker_filepath,
                 "time_filepath": time_filepath, "tick_filepath": tick_filepath}
    single = False
    if '--single' in args:
        single = True

    await downloader.download_ticks()
    await downloader.prepare_files(filepaths, single)


if __name__ == "__main__":
    asyncio.run(main(sys.argv))
