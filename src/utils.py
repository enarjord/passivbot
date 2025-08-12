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


async def load_markets(exchange: str, max_age_ms: int = 1000 * 60 * 60 * 24) -> dict:
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
                logging.info(f"{ex} Loaded markets from cache")
                await create_coin_symbol_map_cache(ex, markets)
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
        logging.info(f"{ex} Dumped markets to cache")
    except Exception as e:
        logging.error(f"Error dumping markets to cache at {markets_path} {e}")
    await create_coin_symbol_map_cache(ex, markets)
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


async def create_coin_symbol_map_cache(exchange: str, markets=None):
    try:
        exchange = normalize_exchange_name(exchange)
        quote = get_quote(exchange)
        if markets is None:
            markets = await load_markets(exchange)
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
        logging.info(f"dumping coin_to_symbol_map {coin_to_symbol_map_path}")
        json.dump(coin_to_symbol_map, open(coin_to_symbol_map_path, "w"), indent=4, sort_keys=True)
        logging.info(f"dumping symbol_to_coin_map {symbol_to_coin_map_path}")
        json.dump(symbol_to_coin_map, open(symbol_to_coin_map_path, "w"))
        return True
    except Exception as e:
        print(f"error with create_coin_symbol_map_cache {exchange}, {e}")
        return False


def coin_to_symbol(coin, exchange):
    # assumes coin_to_symbol_map is cached
    try:
        loaded = json.load(
            open(os.path.join("caches", normalize_exchange_name(exchange), "coin_to_symbol_map.json"))
        )
        if coin not in loaded:
            return ""
        candidates = loaded[coin]
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
    # assumes symbol_to_coin_map is cached
    try:
        return json.load(open(os.path.join("caches", "symbol_to_coin_map.json")))[symbol]
    except Exception as e:
        msg = f"failed to convert {symbol} to its coin with symbol_to_coin_map"
        # logging.error(f"error with symbol_to_coin {symbol} {e}")

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
