import re
import json
import ccxt.async_support as ccxt
import os
import datetime
import logging
import dateutil.parser
from collections import defaultdict


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# In-memory caches for symbol/coin maps with on-disk change detection
_COIN_TO_SYMBOL_CACHE = {}  # {exchange: {"map": dict, "mtime_ns": int, "size": int}}
_SYMBOL_TO_COIN_CACHE = {"map": None, "mtime_ns": None, "size": None}


def get_file_mod_utc(filepath):
    """
    Get the UTC timestamp of the last modification of a file.

    Args:
        filepath (str): The path to the file.

    Returns:
        float: The UTC timestamp in milliseconds of the last modification of the file.
    """
    # Get the last modification time of the file in seconds since the epoch
    mod_time_epoch = os.path.getmtime(filepath)

    # Convert the timestamp to a UTC datetime object
    mod_time_utc = datetime.datetime.utcfromtimestamp(mod_time_epoch)

    # Return the UTC timestamp
    return mod_time_utc.timestamp() * 1000


def ts_to_date_utc(timestamp: float) -> str:
    if timestamp > 253402297199:
        return str(datetime.datetime.utcfromtimestamp(timestamp / 1000)).replace(" ", "T")
    return str(datetime.datetime.utcfromtimestamp(timestamp)).replace(" ", "T")


def date_to_ts(d):
    return int(dateutil.parser.parse(d).replace(tzinfo=datetime.timezone.utc).timestamp() * 1000)


def format_end_date(end_date) -> str:
    if end_date in ["today", "now", "", None]:
        ms2day = 1000 * 60 * 60 * 24
        end_date = ts_to_date_utc((utc_ms() - ms2day * 2) // ms2day * ms2day)
    else:
        end_date = ts_to_date_utc(date_to_ts(end_date))
    return end_date[:10]


def make_get_filepath(filepath: str) -> str:
    """
    if not is path, creates dir and subdirs for path, returns path
    """
    dirpath = os.path.dirname(filepath) if filepath[-1] != "/" else filepath
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    return filepath


def utc_ms() -> float:
    return datetime.datetime.utcnow().timestamp() * 1000


async def load_markets(exchange: str, max_age_ms: int = 1000 * 60 * 60 * 24, verbose=True) -> dict:
    """
    Standalone helper to load and cache CCXT markets for a given exchange.

    - Normalizes 'binance' -> 'binanceusdm'
    - Reads from caches/{exchange}/markets.json if fresh
    - Otherwise fetches via ccxt, writes cache, and returns the markets dict

    Returns a markets dictionary as provided by ccxt.
    """
    ex = normalize_exchange_name(exchange)
    markets_path = os.path.join("caches", ex, "markets.json")

    # Try cache first
    try:
        if os.path.exists(markets_path):
            if utc_ms() - get_file_mod_utc(markets_path) < max_age_ms:
                markets = json.load(open(markets_path))
                if verbose:
                    logging.info(f"{ex} Loaded markets from cache")
                create_coin_symbol_map_cache(ex, markets, verbose=verbose)
                return markets
    except Exception as e:
        logging.error(f"Error loading {markets_path} {e}")

    # Fetch from exchange via ccxt
    cc = getattr(ccxt, ex)({"enableRateLimit": True})
    try:
        cc.options["defaultType"] = "swap"
    except Exception:
        pass

    try:
        markets = await cc.load_markets(True)
    except Exception as e:
        logging.error(f"Error loading markets from {ex}: {e}")
        raise
    finally:
        try:
            await cc.close()
        except Exception:
            pass

    # Dump to cache
    try:
        json.dump(markets, open(make_get_filepath(markets_path), "w"))
        if verbose:
            logging.info(f"{ex} Dumped markets to cache")
    except Exception as e:
        logging.error(f"Error dumping markets to cache at {markets_path} {e}")
    create_coin_symbol_map_cache(ex, markets, verbose=verbose)
    return markets


def normalize_exchange_name(exchange: str) -> str:
    """
    Normalize an exchange id to its USD-margined perpetual futures id when available.

    Examples:
    - "binance" -> "binanceusdm"
    - "kucoin"  -> "kucoinfutures"
    - "kraken"  -> "krakenfutures"

    If no specific futures id exists (e.g. "okx", "bybit", "mexc"), the input is returned unchanged.
    The function uses ccxt.exchanges to detect available ids, so it will automatically catch
    new exchanges that follow common suffix patterns like 'usdm' or 'futures'.
    """
    ex = (exchange or "").lower()
    valid = set(getattr(ccxt, "exchanges", []))

    # Explicit mapping for known special case
    if ex == "binance":
        return "binanceusdm"

    # If already a futures/perp id, keep as-is
    if ex.endswith("usdm") or ex.endswith("futures"):
        return ex

    # Heuristic: prefer '{exchange}usdm' then '{exchange}futures' if available in ccxt
    for suffix in ("usdm", "futures"):
        cand = f"{ex}{suffix}"
        if cand in valid:
            return cand

    return ex


def get_quote(exchange):
    exchange = normalize_exchange_name(exchange)
    return "USDC" if exchange in ["hyperliquid", "defx"] else "USDT"


def remove_powers_of_ten(text):
    """
    Remove any variant of "10", "100", "1000", "10000", etc. from a string.
    Handles cases like "1000SHIB" by using lookahead/lookbehind assertions.
    """
    # Match 1 followed by one or more zeros, with word boundaries or start/end of string
    pattern = r"(?<!\d)1(?:0+)(?!\d)"
    return re.sub(pattern, "", text)


def _load_coin_to_symbol_map(exchange: str) -> dict:
    """
    Lazily load and cache caches/{exchange}/coin_to_symbol_map.json in memory.
    Reloads if the file changes on disk (mtime or size).
    """
    path = os.path.join("caches", exchange, "coin_to_symbol_map.json")
    try:
        st = os.stat(path)
        mtime_ns, size = st.st_mtime_ns, st.st_size
    except Exception:
        return {}
    entry = _COIN_TO_SYMBOL_CACHE.get(exchange)
    if entry and entry.get("mtime_ns") == mtime_ns and entry.get("size") == size:
        return entry.get("map", {})
    try:
        with open(path) as f:
            data = json.load(f)
        _COIN_TO_SYMBOL_CACHE[exchange] = {"map": data, "mtime_ns": mtime_ns, "size": size}
        return data
    except Exception as e:
        logging.error(f"failed to load coin_to_symbol_map for {exchange}: {e}")
        return {}


def _load_symbol_to_coin_map() -> dict:
    """
    Lazily load and cache caches/symbol_to_coin_map.json in memory.
    Reloads if the file changes on disk (mtime or size).
    """
    path = os.path.join("caches", "symbol_to_coin_map.json")
    try:
        st = os.stat(path)
        mtime_ns, size = st.st_mtime_ns, st.st_size
    except Exception:
        return {}
    entry = _SYMBOL_TO_COIN_CACHE
    if (
        entry.get("map") is not None
        and entry.get("mtime_ns") == mtime_ns
        and entry.get("size") == size
    ):
        return entry.get("map", {})
    try:
        with open(path) as f:
            data = json.load(f)
        _SYMBOL_TO_COIN_CACHE["map"] = data
        _SYMBOL_TO_COIN_CACHE["mtime_ns"] = mtime_ns
        _SYMBOL_TO_COIN_CACHE["size"] = size
        return data
    except Exception as e:
        logging.error(f"failed to load symbol_to_coin_map: {e}")
        return {}


def create_coin_symbol_map_cache(exchange: str, markets, verbose=True):
    try:
        exchange = normalize_exchange_name(exchange)
        quote = get_quote(exchange)
        coin_to_symbol_map = defaultdict(set)
        symbol_to_coin_map = {}
        symbol_to_coin_map_path = make_get_filepath(os.path.join("caches", "symbol_to_coin_map.json"))
        try:
            if os.path.exists(symbol_to_coin_map_path):
                symbol_to_coin_map = json.load(open(symbol_to_coin_map_path))
        except Exception as e:
            logging.error(f"failed to load symbol_to_coin_map {e}")
        for k, v in markets.items():
            if not v.get("swap"):
                continue
            if not v.get("linear"):
                continue
            if not k.endswith(f":{quote}"):
                continue
            base_name, base = v.get("baseName", ""), v.get("base", "")
            if base_name:
                coin = remove_powers_of_ten(base_name.replace("k", ""))
                coin_to_symbol_map[coin].add(k)
                coin_to_symbol_map[k].add(k)
                coin_to_symbol_map[base_name].add(k)
                symbol_to_coin_map[k] = coin
                symbol_to_coin_map[coin] = coin
                symbol_to_coin_map[base_name] = coin
            if base:
                coin = remove_powers_of_ten(base.replace("k", ""))
                coin_to_symbol_map[coin].add(k)
                coin_to_symbol_map[k].add(k)
                coin_to_symbol_map[base].add(k)
                symbol_to_coin_map[coin] = coin
                symbol_to_coin_map[base] = coin
                if not base_name:
                    symbol_to_coin_map[k] = coin
        coin_to_symbol_map_path = make_get_filepath(
            os.path.join("caches", exchange, "coin_to_symbol_map.json")
        )
        coin_to_symbol_map = {k: list(v) for k, v in coin_to_symbol_map.items()}
        if verbose:
            logging.info(f"dumping coin_to_symbol_map {coin_to_symbol_map_path}")
        json.dump(coin_to_symbol_map, open(coin_to_symbol_map_path, "w"), indent=4, sort_keys=True)
        if verbose:
            logging.info(f"dumping symbol_to_coin_map {symbol_to_coin_map_path}")
        json.dump(symbol_to_coin_map, open(symbol_to_coin_map_path, "w"))
        # update in-memory caches to avoid stale reads
        try:
            st = os.stat(coin_to_symbol_map_path)
            _COIN_TO_SYMBOL_CACHE[exchange] = {
                "map": coin_to_symbol_map,
                "mtime_ns": st.st_mtime_ns,
                "size": st.st_size,
            }
        except Exception:
            pass
        try:
            st2 = os.stat(symbol_to_coin_map_path)
            _SYMBOL_TO_COIN_CACHE["map"] = symbol_to_coin_map
            _SYMBOL_TO_COIN_CACHE["mtime_ns"] = st2.st_mtime_ns
            _SYMBOL_TO_COIN_CACHE["size"] = st2.st_size
        except Exception:
            pass
        return True
    except Exception as e:
        print(f"error with create_coin_symbol_map_cache {exchange}, {e}")
        return False


def coin_to_symbol(coin, exchange):
    # caches coin_to_symbol_map in memory and reloads if file changes
    try:
        ex = normalize_exchange_name(exchange)
        loaded = _load_coin_to_symbol_map(ex)
        if not loaded or coin not in loaded:
            return ""
        candidates = loaded.get(coin, [])
        if len(candidates) == 1:
            return candidates[0]
        elif len(candidates) == 0:
            logging.info(f"No candidates for {coin}")
            return ""
        else:
            logging.info(f"Multiple candidates for {coin}: {candidates}")
            return ""
    except Exception as e:
        logging.error(f"error with coin_to_symbol {coin} {exchange} {e}")
    quote = get_quote(normalize_exchange_name(exchange))
    return f"{coin}/{quote}:{quote}"


def symbol_to_coin(symbol):
    # caches symbol_to_coin_map in memory and reloads if file changes
    try:
        loaded = _load_symbol_to_coin_map()
        if symbol in loaded:
            return loaded[symbol]
        msg = f"failed to convert {symbol} to its coin with symbol_to_coin_map"
    except Exception:
        msg = f"failed to convert {symbol} to its coin with symbol_to_coin_map"

    if symbol == "":
        return ""
    if "/" in symbol:
        coin = symbol[: symbol.find("/")]
    else:
        coin = symbol
    for x in ["USDT", "USDC", "BUSD", "USD", "/:"]:
        coin = coin.replace(x, "")
    if "1000" in coin:
        istart = coin.find("1000")
        iend = istart + 1
        while True:
            if iend >= len(coin):
                break
            if coin[iend] != "0":
                break
            iend += 1
        coin = coin[:istart] + coin[iend:]
    if coin.startswith("k") and coin[1:].isupper():
        # hyperliquid uses e.g. kSHIB instead of 1000SHIB
        coin = coin[1:]
    if coin:
        msg += f". Using heuristics to guess coin: {coin}"
    logging.warning(msg)
    return coin


async def format_approved_ignored_coins(config, exchanges: [str]):
    path = config["live"]["approved_coins"]
    if path in [
        [""],
        [],
        None,
        "",
        0,
        0.0,
        {"long": [], "short": []},
        {"long": "", "short": ""},
        {"long": [""], "short": [""]},
    ]:
        if config["live"]["empty_means_all_approved"]:
            marketss = await asyncio.gather(*[load_markets(ex, verbose=False) for ex in exchanges])
            approved_coins = set()
            for markets in marketss:
                for symbol in markets:
                    approved_coins.add(symbol_to_coin(symbol))
            approved_coins_sorted = sorted([x for x in approved_coins if x])
            config["live"]["approved_coins"] = {
                "long": approved_coins_sorted,
                "short": approved_coins_sorted,
            }
        else:
            # leave empty
            config["live"]["approved_coins"] = {"long": [], "short": []}
    else:
        ac = normalize_coins_source(config["live"]["approved_coins"])
        config["live"]["approved_coins"] = {
            pside: [cf for x in ac[pside] if (cf := symbol_to_coin(x))] for pside in ac
        }

    ic = normalize_coins_source(config["live"]["ignored_coins"])
    config["live"]["ignored_coins"] = {
        pside: [cf for x in ic[pside] if (cf := symbol_to_coin(x))] for pside in ic
    }


def normalize_coins_source(src):
    """
    Always return: {'long': [symbols…], 'short': [symbols…]}
    – Handles:
        • direct coin lists or comma-separated strings
        • lists/tuples containing paths or strings
        • dicts with 'long' / 'short' keys whose values may themselves
          be strings, lists, or paths to external lists
    """

    # --------------------------------------------------------------------- #
    #  Helpers                                                              #
    # --------------------------------------------------------------------- #
    def _expand(seq):
        """Flatten seq and split any comma-delimited strings it contains."""
        out = []
        for item in seq:
            if isinstance(item, (list, tuple, set)):
                out.extend(_expand(item))  # recurse
            elif isinstance(item, str):
                out.extend(x.strip() for x in item.split(",") if x.strip())
            elif item is not None:
                out.append(str(item).strip())
        return out

    def _load_if_file(x):
        """
        If *x* (or *x[0]* when x is a single-item list/tuple) is a
        readable file path, load it with `read_external_coins_lists`.
        Otherwise just return *x* unchanged.
        """
        if isinstance(x, str) and os.path.exists(x):
            return read_external_coins_lists(x)

        if (
            isinstance(x, (list, tuple))
            and len(x) == 1
            and isinstance(x[0], str)
            and os.path.exists(x[0])
        ):
            return read_external_coins_lists(x[0])

        return x

    def _normalize_side(value, side):
        """
        Resolve one *long*/*short* entry:
        1. Load from file if necessary.
        2. If the loader returned a dict, pluck the correct side.
        3. Flatten & split with _expand so we end up with a clean list.
        """
        value = _load_if_file(value)

        if isinstance(value, dict) and sorted(value.keys()) == ["long", "short"]:
            value = value.get(side, [])

        # guarantee a sensible sequence for _expand
        if not isinstance(value, (list, tuple)):
            value = [value]

        return _expand(value)

    # --------------------------------------------------------------------- #
    #  Main logic                                                           #
    # --------------------------------------------------------------------- #
    src = _load_if_file(src)  # try to load *src* itself

    # Case 1 – already a dict with 'long' & 'short' keys
    if isinstance(src, dict) and sorted(src.keys()) == ["long", "short"]:
        return {
            "long": _normalize_side(src.get("long", []), "long"),
            "short": _normalize_side(src.get("short", []), "short"),
        }

    # Case 2 – anything else is treated the same for both sides
    return {
        "long": _normalize_side(src, "long"),
        "short": _normalize_side(src, "short"),
    }


def read_external_coins_lists(filepath) -> dict:
    """
    reads filepath and returns dict {'long': [str], 'short': [str]}
    """
    try:
        content = hjson.load(open(filepath))
        if isinstance(content, list) and all([isinstance(x, str) for x in content]):
            return {"long": content, "short": content}
        elif isinstance(content, dict):
            if all(
                [
                    pside in content
                    and isinstance(content[pside], list)
                    and all([isinstance(x, str) for x in content[pside]])
                    for pside in ["long", "short"]
                ]
            ):
                return content
    except:
        pass
    with open(filepath, "r") as file:
        content = file.read().strip()
    # Check if the content is in list format
    if content.startswith("[") and content.endswith("]"):
        # Remove brackets and split by comma
        items = content[1:-1].split(",")
        # Remove quotes and whitespace
        items = [item.strip().strip("\"'") for item in items if item.strip()]
    elif all(
        line.strip().startswith('"') and line.strip().endswith('"')
        for line in content.split("\n")
        if line.strip()
    ):
        # Split by newline, remove quotes and whitespace
        items = [line.strip().strip("\"'") for line in content.split("\n") if line.strip()]
    else:
        # Split by newline, comma, and/or space, and filter out empty strings
        items = [item.strip() for item in content.replace(",", " ").split() if item.strip()]
    return {"long": items, "short": items}
