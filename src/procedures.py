import glob
import json
import os
import traceback
import asyncio
from datetime import datetime, timezone
from time import time
import numpy as np
import pprint
from copy import deepcopy
import argparse
import re
from collections import defaultdict
from collections.abc import Sized
from utils import (
    coin_to_symbol,
    symbol_to_coin,
    make_get_filepath,
    load_markets,
    get_file_mod_ms,
    date_to_ts,
    get_first_ohlcv_iteratively,
)
import sys
import passivbot_rust as pbr
from typing import Union, Optional, Set, Any, List
from pathlib import Path
import ccxt.async_support as ccxta

try:
    import hjson
except:
    print("hjson not found, trying without...")
    pass
try:
    import pandas as pd
except:
    print("pandas not found, trying without...")
    pass

from pure_funcs import (
    numpyize,
    ts_to_date,
    config_pretty_str,
    sort_dict_keys,
    flatten,
)


def get_all_eligible_symbols(exchange="binance"):
    exchange_map = {
        "bybit": "bybit",
        "binance": "binanceusdm",
        # "bitget": "bitget", TODO
        # "hyperliquid": "hyperliquid", TODO
        # "gateio": "gateio", TODO
    }
    quote_map = {k: "USDT" for k in exchange_map}
    quote_map["hyperliquid"] = "USDC"
    if exchange not in exchange_map:
        raise Exception(f"only exchanges {list(exchange_map.values())} are supported for backtesting")
    filepath = make_get_filepath(f"caches/{exchange}/eligible_symbols.json")
    loaded_json = None
    try:
        loaded_json = json.load(open(filepath))
        if utc_ms() - get_file_mod_ms(filepath) > 1000 * 60 * 60 * 24:
            print(f"Eligible_symbols cache more than 24h old. Fetching new.")
        else:
            return loaded_json
    except Exception as e:
        print(f"failed to load {filepath}. Fetching from {exchange}")
        pass
    try:
        quote = quote_map[exchange]
        import ccxt

        cc = getattr(ccxt, exchange_map[exchange])()
        markets = cc.fetch_markets()
        symbols = [
            x["symbol"] for x in markets if "symbol" in x and x["symbol"].endswith(f":{quote}")
        ]
        eligible_symbols = sorted(set([x.replace(f"/{quote}:", "") for x in symbols]))
        eligible_symbols = [x for x in eligible_symbols if x]
        json.dump(eligible_symbols, open(filepath, "w"))
        return eligible_symbols
    except Exception as e:
        print(f"error fetching eligible symbols {e}")
        if loaded_json:
            print(f"using cached data")
            return loaded_json
        raise Exception("unable to fetch or load from cache")


def dump_pretty_json(data: dict, filepath: str):
    try:
        with open(filepath, "w") as f:
            f.write(config_pretty_str(sort_dict_keys(data)) + "\n")
    except Exception as e:
        raise Exception(f"failed to dump data {filepath}: {e}")


def ensure_parent_directory(
    filepath: Union[str, Path], mode: int = 0o755, exist_ok: bool = True
) -> Path:
    """
    Creates directory and subdirectories for a given filepath if they don't exist,
    then returns the path as a Path object.

    Args:
        filepath: String or Path object representing the file or directory path
        mode: Directory permissions (default: 0o755)
        exist_ok: If False, raise FileExistsError if directory exists (default: True)

    Returns:
        Path object representing the input filepath

    Raises:
        TypeError: If filepath is neither str nor Path
        PermissionError: If user lacks permission to create directory
        FileExistsError: If directory exists and exist_ok is False
    """
    try:
        # Convert to Path object
        path = Path(filepath)

        # Determine if the path points to a directory
        # (either ends with separator or is explicitly a directory)
        if str(path).endswith(os.path.sep) or (path.exists() and path.is_dir()):
            dirpath = path
        else:
            dirpath = path.parent

        # Create directory if it doesn't exist
        if not dirpath.exists():
            dirpath.mkdir(parents=True, mode=mode, exist_ok=exist_ok)
        elif not exist_ok:
            raise FileExistsError(f"Directory already exists: {dirpath}")

        return path

    except TypeError as e:
        raise TypeError(f"filepath must be str or Path, not {type(filepath)}") from e
    except PermissionError as e:
        raise PermissionError(f"Permission denied creating directory: {dirpath}") from e
    except Exception as e:
        raise RuntimeError(f"Error processing filepath: {str(e)}") from e


def load_user_info(user: str, api_keys_path="api-keys.json") -> dict:
    if api_keys_path is None:
        api_keys_path = "api-keys.json"
    try:
        api_keys = json.load(open(api_keys_path))
    except Exception as e:
        raise Exception(f"error loading api keys file {api_keys_path} {e}")
    if user not in api_keys:
        raise Exception(f"user {user} not found in {api_keys_path}")
    return {
        k: api_keys[user][k] if k in api_keys[user] else ""
        for k in [
            "exchange",
            "key",
            "secret",
            "passphrase",
            "wallet_address",
            "private_key",
            "is_vault",
        ]
    }


def load_exchange_key_secret_passphrase(
    user: str, api_keys_path="api-keys.json"
) -> (str, str, str, str):
    if api_keys_path is None:
        api_keys_path = "api-keys.json"
    try:
        keyfile = json.load(open(api_keys_path))
        if user in keyfile:
            return (
                keyfile[user]["exchange"],
                keyfile[user]["key"],
                keyfile[user]["secret"],
                keyfile[user]["passphrase"] if "passphrase" in keyfile[user] else "",
            )
        else:
            print("Looks like the keys aren't configured yet, or you entered the wrong username!")
        raise Exception("API KeyFile Missing!")
    except FileNotFoundError:
        print("File Not Found!")
        raise Exception("API KeyFile Missing!")


def load_broker_code(exchange: str) -> str:
    try:
        return hjson.load(open("broker_codes.hjson"))[exchange]
    except Exception as e:
        print(f"failed to load broker code", e)
        traceback.print_exc()
        return ""


def print_(args, r=False, n=False):
    line = ts_to_date(utc_ms())[:19] + "  "
    # line = ts_to_date(local_time())[:19] + '  '
    str_args = "{} " * len(args)
    line += str_args.format(*args)
    if n:
        print("\n" + line, end=" ")
    elif r:
        print("\r" + line, end=" ")
    else:
        print(line)
    return line


def local_time() -> float:
    return datetime.now().astimezone().timestamp() * 1000


def print_async_exception(coro):
    if isinstance(coro, list):
        for elm in coro:
            print_async_exception(elm)
    try:
        print(f"result: {coro.result()}")
    except:
        pass
    try:
        print(f"exception: {coro.exception()}")
    except:
        pass
    try:
        print(f"returned: {coro}")
    except:
        pass


async def get_first_timestamps_unified(coins: List[str], exchange: str = None):
    """
    Returns earliest timestamp each coin was found on any exchange by default.
    If 'exchange' is specified, returns earliest timestamps specifically for that exchange.

    Batches requests in groups of 10 coins at a time, and dumps results to disk
    immediately after each batch is processed.

    :param coins: List of coin symbols to retrieve first-timestamp data for.
    :param exchange: Optional string specifying a single exchange (e.g., 'binanceusdm').
                     If set, tries to return first timestamps for only that exchange.
    :return: Dictionary of coin -> earliest timestamp (ms). If `exchange` is provided,
             only entries for the specified exchange are returned.
    """

    # cheap_exchanges = {"binanceusdm", "bybit", "okx", "gateio", "hyperliquid"}
    cheap_exchanges = {"binanceusdm", "bybit", "okx"}

    async def fetch_ohlcv_with_start(exchange_name, symbol, cc):
        """
        Fetch OHLCV data for `symbol` on `exchange_name`, starting from a
        specific date range based on the exchange’s known data availability.
        Returns a list of candle data.
        """
        if exchange_name == "binanceusdm":
            # Data starts practically 'forever' in this example
            return await cc.fetch_ohlcv(symbol, since=1, timeframe="1d")

        elif exchange_name in ["bybit", "gateio"]:
            # Data since 2018
            return await cc.fetch_ohlcv(symbol, since=int(date_to_ts("2018-01-01")), timeframe="1d")

        elif exchange_name == "okx":
            # Monthly timeframe; data since 2018
            return await cc.fetch_ohlcv(symbol, since=int(date_to_ts("2018-01-01")), timeframe="1M")

        elif exchange_name == "bitget":
            first_candle = await get_first_ohlcv_iteratively(cc, symbol)
            return [first_candle] if first_candle else []

        else:  # e.g., 'hyperliquid'
            # Weekly timeframe; data since 2021
            return await cc.fetch_ohlcv(symbol, since=int(date_to_ts("2021-01-01")), timeframe="1w")

    # Remove duplicates and sort the input coins for consistency
    coins = sorted(set(symbol_to_coin(coin) for coin in coins))

    # Paths to the cache files
    cache_fpath = make_get_filepath("caches/first_ohlcv_timestamps_unified.json")
    cache_fpath_exchange_specific = "caches/first_ohlcv_timestamps_unified_exchange_specific.json"

    # In-memory dictionaries for storing timestamps
    ftss = {}  # coin -> earliest timestamp across all exchanges
    ftss_exchange_specific = {}  # coin -> {exchange -> earliest timestamp}

    # Load main cache if it exists
    if os.path.exists(cache_fpath):
        try:
            with open(cache_fpath, "r") as f:
                ftss = json.load(f)
            print(f"Loaded from main cache: {cache_fpath}")
        except Exception as e:
            print(f"Error reading {cache_fpath}: {e}")

    # Load exchange-specific cache if it exists
    if os.path.exists(cache_fpath_exchange_specific):
        try:
            with open(cache_fpath_exchange_specific, "r") as f:
                ftss_exchange_specific = json.load(f)
            print(f"Loaded from exchange-specific cache: {cache_fpath_exchange_specific}")
        except Exception as e:
            print(f"Error reading {cache_fpath_exchange_specific}: {e}")

    # If an exchange is specified, handle "binance" alias
    if exchange == "binance":
        exchange = "binanceusdm"

    # 1) If no exchange is specified and all coins are in ftss, just return ftss
    if exchange is None:
        if all(coin in ftss for coin in coins):
            return ftss

    # 2) If a specific exchange is requested:
    else:
        # If all coins exist in the exchange-specific cache for that exchange, return them
        if all(coin in ftss_exchange_specific for coin in coins):
            if all(exchange in ftss_exchange_specific[coin] for coin in coins):
                # Return a simplified dict coin->timestamp
                return {c: ftss_exchange_specific[c][exchange] for c in coins}

    # Figure out which coins are missing from the main dictionary
    missing_coins = {c for c in coins if c not in ftss}
    if not missing_coins:
        # No missing coins => all already in ftss
        return ftss

    print("Missing coins:", sorted(missing_coins))

    # Map of exchange -> quote currency
    exchange_map = {
        "okx": "USDT",
        "binanceusdm": "USDT",
        "bybit": "USDT",
        "gateio": "USDT",
        "bitget": "USDT",
        "hyperliquid": "USDC",
    }

    # Initialize ccxt clients for each exchange
    ccxt_clients = {}
    for ex_name in sorted(exchange_map):
        try:
            ccxt_clients[ex_name] = getattr(ccxta, ex_name)()
            ccxt_clients[ex_name].options["defaultType"] = "swap"
        except Exception as e:
            print(f"Error loading {ex_name} from ccxt. Skipping. {e}")
            del exchange_map[ex_name]
            if ex_name in ccxt_clients:
                del ccxt_clients[ex_name]
    try:
        print("Loading markets for each exchange...")
        load_tasks = {}
        for ex_name in sorted(ccxt_clients):
            try:
                load_tasks[ex_name] = load_markets(ex_name)
            except Exception as e:
                print(f"Error creating task for {ex_name}: {e}")
                del ccxt_clients[ex_name]
                if ex_name in exchange_map:
                    del exchange_map[ex_name]
        all_markets = {}
        for ex_name, task in load_tasks.items():
            try:
                res = await task
                all_markets[ex_name] = res
            except Exception as e:
                print(f"Warning: failed to load markets for {ex_name}: {e}")
                del ccxt_clients[ex_name]
                if ex_name in exchange_map:
                    del exchange_map[ex_name]
        # We'll fetch missing coins in batches of 10 to avoid overloading
        BATCH_SIZE = 10
        missing_coins = sorted(missing_coins)

        for i in range(0, len(missing_coins), BATCH_SIZE):
            batch = missing_coins[i : i + BATCH_SIZE]
            print(f"\nProcessing batch: {batch}")

            # Create tasks for every coin/exchange pair in this batch
            tasks = {}
            bitget_symbols = {}
            for coin in batch:
                tasks[coin] = {}
                for ex_name, quote in exchange_map.items():
                    # Convert coin to a symbol recognized by the exchange, e.g. "BTC/USDT:USDT"
                    symbol = coin_to_symbol(coin, ex_name)
                    if not symbol:
                        continue
                    if ex_name == "bitget":
                        bitget_symbols[coin] = symbol
                        continue
                    tasks[coin][ex_name] = asyncio.create_task(
                        fetch_ohlcv_with_start(ex_name, symbol, ccxt_clients[ex_name])
                    )

            # Gather all results for this batch
            batch_results = {}
            fast_exchanges = [ex for ex in exchange_map if ex != "bitget"]
            for coin in batch:
                batch_results[coin] = {}
                for ex_name in fast_exchanges:
                    if ex_name in tasks[coin]:
                        try:
                            data = await tasks[coin][ex_name]
                            if data:
                                batch_results[coin][ex_name] = data
                                print(
                                    f"Fetched {ex_name} {coin} => first candle: {data[0] if data else 'no data'}"
                                )
                        except Exception as e:
                            print(f"Warning: failed to fetch OHLCV for {coin} on {ex_name}: {e}")

            # Second pass: issue expensive Bitget fetch only for unresolved coins.
            for coin in batch:
                symbol = bitget_symbols.get(coin)
                if not symbol:
                    continue
                has_valid = False
                for ex_name, arr in batch_results[coin].items():
                    if ex_name not in cheap_exchanges:
                        continue
                    if arr and arr[0][0] > 1262304000000.0:
                        has_valid = True
                        break
                if has_valid:
                    continue
                try:
                    data = await fetch_ohlcv_with_start("bitget", symbol, ccxt_clients["bitget"])
                    if data:
                        batch_results[coin]["bitget"] = data
                        print(
                            f"Fetched bitget {coin} => first candle: {data[0] if data else 'no data'}"
                        )
                except Exception as e:
                    print(f"Warning: failed to fetch OHLCV for {coin} on bitget: {e}")

            # Process results for each coin in this batch
            for coin in batch:
                exchange_data = batch_results.get(coin, {})
                fts_for_this_coin = {ex: 0.0 for ex in exchange_map}  # default 0.0 for all
                earliest_candidates = []

                for ex_name, arr in exchange_data.items():
                    if arr and len(arr) > 0:
                        # arr[0][0] is the timestamp in ms
                        # Only consider "reasonable" timestamps after 2010
                        if arr[0][0] > 1262304000000.0:
                            earliest_candidates.append(arr[0][0])
                            fts_for_this_coin[ex_name] = arr[0][0]

                # If any valid timestamps found, keep the earliest
                if earliest_candidates:
                    ftss[coin] = min(earliest_candidates)
                else:
                    print(f"No valid first timestamp for coin {coin}")
                    ftss[coin] = 0.0

                # Update the exchange-specific dictionary
                ftss_exchange_specific[coin] = fts_for_this_coin

            # Immediately dump updated dictionaries to disk after each batch
            with open(cache_fpath, "w") as f:
                json.dump(ftss, f, indent=4, sort_keys=True)

            with open(cache_fpath_exchange_specific, "w") as f:
                json.dump(ftss_exchange_specific, f, indent=4, sort_keys=True)

            print(f"Finished batch {batch}. Caches updated.")

        # Close all ccxt client sessions

        # If a single exchange was requested, return only those exchange-specific timestamps.
        if exchange is not None:
            return {coin: ftss_exchange_specific.get(coin, {}).get(exchange, 0.0) for coin in coins}

        # Otherwise, return earliest cross-exchange timestamps
        return ftss
    finally:
        await asyncio.gather(
            *(ccxt_clients[e].close() for e in ccxt_clients if hasattr(ccxt_clients[e], "close"))
        )


def assert_correct_ccxt_version(version=None, ccxt=None):
    if version is None:
        version = load_ccxt_version()
    if ccxt is None:
        import ccxt

    assert (
        ccxt.__version__ == version
    ), f"Currently ccxt {ccxt.__version__} is installed. Please pip reinstall requirements.txt or install ccxt v{version} manually"


def load_ccxt_version():
    try:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to the requirements.txt file
        requirements_path = os.path.join(script_dir, "..", "requirements-live.txt")

        # Open and read the requirements.txt file
        with open(requirements_path, "r") as f:
            lines = f.readlines()

        # Find the line with 'ccxt' and extract the version number
        ccxt_line = [line for line in lines if "ccxt" in line][0].strip()
        return ccxt_line[ccxt_line.find("==") + 2 :]
    except Exception as e:
        print(f"failed to load ccxt version {e}")
        return None


def get_size(obj: Any, seen: Set = None) -> int:
    """
    Recursively calculate size of object and its contents in bytes.

    Args:
        obj: The object to calculate size for
        seen: Set of object ids already seen (for handling circular references)

    Returns:
        Total size in bytes
    """
    # Initialize the set of seen objects if this is the top-level call
    if seen is None:
        seen = set()

    # Get object id to handle circular references
    obj_id = id(obj)

    # If object has been seen, don't count it again
    if obj_id in seen:
        return 0

    # Add this object to seen
    seen.add(obj_id)

    # Get basic size of object
    size = sys.getsizeof(obj)

    # Handle different types of containers
    if isinstance(obj, (str, bytes, bytearray)):
        pass  # Basic size already includes contents

    elif isinstance(obj, (tuple, list, set, frozenset)):
        size += sum(get_size(item, seen) for item in obj)

    elif isinstance(obj, dict):
        size += sum(get_size(k, seen) + get_size(v, seen) for k, v in obj.items())

    elif hasattr(obj, "__dict__"):
        # Add size of all attributes for custom objects
        size += get_size(obj.__dict__, seen)

    elif hasattr(obj, "__slots__"):
        # Handle objects using __slots__
        size += sum(
            get_size(getattr(obj, attr), seen) for attr in obj.__slots__ if hasattr(obj, attr)
        )

    return size


def format_size(size_bytes: int) -> str:
    """
    Format byte size into human readable string.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted string like '1.23 MB'
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} PB"


def compare_dicts_table(dict1, dict2, dict1_name="Dict 1", dict2_name="Dict 2"):
    """
    Compare two dictionaries with identical keys in a neat table format.

    Args:
        dict1: First dictionary
        dict2: Second dictionary
        dict1_name: Name for first dictionary column
        dict2_name: Name for second dictionary column
    """
    # Get all keys (assuming identical keys)
    keys = list(dict1.keys())

    # Calculate column widths
    key_width = max(len("Key"), max(len(str(k)) for k in keys))
    val1_width = max(len(dict1_name), max(len(str(dict1[k])) for k in keys))
    val2_width = max(len(dict2_name), max(len(str(dict2[k])) for k in keys))

    # Create separator line
    separator = (
        "+"
        + "-" * (key_width + 2)
        + "+"
        + "-" * (val1_width + 2)
        + "+"
        + "-" * (val2_width + 2)
        + "+"
    )

    # Print table
    print(separator)
    print(f"| {'Key':<{key_width}} | {dict1_name:<{val1_width}} | {dict2_name:<{val2_width}} |")
    print(separator)

    for key in sorted(keys):
        val1 = str(pbr.round_dynamic(dict1[key], 4))
        val2 = str(pbr.round_dynamic(dict2[key], 4))
        print(f"| {str(key):<{key_width}} | {val1:<{val1_width}} | {val2:<{val2_width}} |")

    print(separator)


def main():
    pass


if __name__ == "__main__":
    main()
