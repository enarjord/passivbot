import re
import json
import ccxt.async_support as ccxt
import os
import datetime
import dateutil.parser
import asyncio
import hjson
import inspect
import time
from collections import defaultdict
from typing import Dict, Any, List, Union, Optional
import re
import logging
from copy import deepcopy
from pathlib import Path
import portalocker  # type: ignore
from custom_endpoint_overrides import (
    apply_rest_overrides_to_ccxt,
    resolve_custom_endpoint_override,
)
from config_transform import record_transform


logging.basicConfig(
    format="%(asctime)s %(levelname)-8s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# In-memory caches for symbol/coin maps with on-disk change detection
_COIN_TO_SYMBOL_CACHE = {}  # {exchange: {"map": dict, "mtime_ns": int, "size": int}}
_SYMBOL_TO_COIN_CACHE = {"map": None, "mtime_ns": None, "size": None}
_SYMBOL_TO_COIN_WARNINGS: set[str] = set()
_COIN_TO_SYMBOL_FALLBACKS: set[tuple[str, str]] = set()

# File locking constants for symbol/coin map files
_SYMBOL_MAP_LOCK_STALE_SECONDS = 180  # Remove locks older than 3 minutes
_SYMBOL_MAP_LOCK_TIMEOUT = 5  # Seconds to wait for lock acquisition
_SYMBOL_MAP_STALE_CLEANUP_DONE = False  # Track if cleanup has run this session
PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_COINS_FILE_ALIASES = {
    "approved_coins_topmcap.json": Path("configs/approved_coins.json"),
    "approved_coins_topmcap.txt": Path("configs/approved_coins.json"),
}


def _atomic_write_json(path: str, data: dict, indent=None, sort_keys=False) -> None:
    """Write JSON atomically: write to .tmp then os.replace() for crash safety."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=indent, sort_keys=sort_keys)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _cleanup_stale_symbol_map_locks() -> None:
    """
    Remove leftover .lock files for symbol/coin maps that are clearly stale.
    Runs once per session on first access to prevent accumulation.
    """
    global _SYMBOL_MAP_STALE_CLEANUP_DONE
    if _SYMBOL_MAP_STALE_CLEANUP_DONE:
        return
    _SYMBOL_MAP_STALE_CLEANUP_DONE = True

    cache_dir = Path("caches")
    if not cache_dir.exists():
        return

    now = time.time()
    threshold = _SYMBOL_MAP_LOCK_STALE_SECONDS

    # Clean up lock files in caches/ and caches/{exchange}/
    lock_patterns = [
        "*.lock",  # Top-level locks (symbol_to_coin_map.json.lock)
        "*/*.lock",  # Per-exchange locks (caches/{exchange}/coin_to_symbol_map.json.lock)
    ]

    for pattern in lock_patterns:
        for lock_path in cache_dir.glob(pattern):
            # Only clean up symbol/coin map related locks
            if "symbol" not in lock_path.name and "coin" not in lock_path.name:
                continue
            try:
                stat = lock_path.stat()
                age = now - stat.st_mtime
                if age > threshold:
                    lock_path.unlink()
                    logging.warning("removed stale symbol map lock %s (age %.1fs)", lock_path, age)
            except FileNotFoundError:
                continue
            except Exception as exc:
                logging.debug("failed to remove stale lock %s: %s", lock_path, exc)


def _resolve_coins_file_path(value: str) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        return None
    raw_path = Path(value.strip())
    candidates: List[Path] = []

    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.extend(
            [
                PROJECT_ROOT / raw_path,
                Path.cwd() / raw_path,
            ]
        )

    alias = LEGACY_COINS_FILE_ALIASES.get(raw_path.name)
    if alias is not None:
        if not alias.is_absolute():
            candidates.append(PROJECT_ROOT / alias)
        else:
            candidates.append(alias)

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            if candidate.name != raw_path.name and raw_path.name in LEGACY_COINS_FILE_ALIASES:
                try:
                    rel = candidate.relative_to(PROJECT_ROOT)
                except ValueError:
                    rel = candidate
                logging.warning(
                    "Resolved legacy coins file '%s' to '%s'. Update your config to the new path.",
                    raw_path,
                    rel,
                )
            return candidate
    return None


def _require_live_value(config: Dict[str, Any], key: str):
    if "live" not in config or not isinstance(config["live"], dict):
        raise KeyError("config missing required key 'live'")
    live = config["live"]
    if key not in live:
        raise KeyError(f"config missing required key 'live.{key}'")
    return live[key]


def ts_to_date(timestamp: Union[float, str, int]) -> str:
    """
    Convert a timestamp to UTC date string in ISO format.

    Args:
        timestamp: Timestamp as float, str, or int - may be seconds, milliseconds, or nanoseconds

    Returns:
        UTC date string in ISO format (e.g., "2025-03-12T12:43:22.123")
    """
    # Convert to float if string or int
    if isinstance(timestamp, (str, int)):
        timestamp = float(timestamp)

    # Detect timestamp precision and convert to seconds
    if timestamp > 1e15:  # Likely nanoseconds (> ~2033 in milliseconds)
        # Nanoseconds
        timestamp_seconds = timestamp / 1_000_000_000
    elif timestamp > 1e10:  # Likely milliseconds (> ~2001 in seconds)
        # Milliseconds
        timestamp_seconds = timestamp / 1000
    else:
        # Seconds
        timestamp_seconds = timestamp

    # Convert to UTC datetime
    dt = datetime.datetime.fromtimestamp(timestamp_seconds, tz=datetime.timezone.utc)

    # Return ISO format without timezone suffix
    return dt.isoformat().replace("+00:00", "")


def date_to_ts(date_str: str) -> float:
    """
    Convert a flexible date string to UTC timestamp in milliseconds.

    Args:
        date_str: Date string in various formats:
                 - "2020" -> "2020-01-01T00:00:00"
                 - "2024-04" -> "2024-04-01T00:00:00"
                 - "2022-04-23" -> "2022-04-23T00:00:00"
                 - "2021-11-13T03:23:12" (full ISO format)
                 - And other common variants

    Returns:
        UTC timestamp in milliseconds as float
    """
    date_str = date_str.strip()

    # Use dateutil.parser with default date of Jan 1, 2000 for missing components
    default_date = datetime.datetime(2000, 1, 1)

    try:
        dt = dateutil.parser.parse(date_str, default=default_date)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Unable to parse date string '{date_str}': {e}")

    # If the datetime is naive (no timezone info), treat it as UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)

    # Convert to UTC timestamp in milliseconds
    return dt.timestamp() * 1000


def get_file_mod_ms(filepath):
    """
    Get the UTC timestamp of the last modification of a file.
    Args:
        filepath (str): The path to the file.
    Returns:
        float: The UTC timestamp in milliseconds of the last modification of the file.
    """
    # Get the last modification time in seconds since epoch (already UTC-based)
    mod_time_epoch = os.path.getmtime(filepath)
    # Convert to milliseconds
    return mod_time_epoch * 1000


def format_end_date(end_date) -> str:
    if end_date in ["today", "now", "", None]:
        ms2day = 1000 * 60 * 60 * 24
        end_date = ts_to_date((utc_ms() - ms2day * 2) // ms2day * ms2day)
    else:
        end_date = ts_to_date(date_to_ts(end_date))
    return end_date[:10]


def make_get_filepath(filepath: str) -> str:
    """
    Ensure directory for filepath exists and return the filepath.
    """
    dirpath = os.path.dirname(filepath) if not filepath.endswith("/") else filepath
    if dirpath and not os.path.isdir(dirpath):
        os.makedirs(dirpath, exist_ok=True)
    return filepath


def utc_ms() -> float:
    return time.time() * 1000


def _inline_simple_containers(text: str, max_inline: int) -> str:
    """Collapse flat list/dict blocks that fit within ``max_inline`` characters."""

    result: list[str] = []
    i = 0
    length = len(text)

    while i < length:
        char = text[i]
        if char in "[{":
            closing = "]" if char == "[" else "}"
            j = i + 1
            depth = 1
            nested = False
            while j < length and depth > 0:
                if text[j] == char:
                    depth += 1
                    nested = True
                elif text[j] == closing:
                    depth -= 1
                j += 1
            segment = text[i:j]
            if (
                depth == 0
                and not nested
                and "\n" in segment
                and len("".join(segment.split())) <= max_inline
            ):
                inner = "".join(line.strip() for line in segment.splitlines()[1:-1])
                result.append(f"{char}{inner}{closing}")
            else:
                result.append(segment)
            i = j
        else:
            result.append(char)
            i += 1
    return "".join(result)


def dump_json_streamlined(
    data: Any,
    fp,
    *,
    indent: int = 4,
    max_inline: int = 60,
    separators: tuple[str, str] = (",", ":"),
    sort_keys: bool = False,
) -> None:
    """
    Write JSON where short lists/dicts stay on one line while larger blocks keep
    normal indentation.

    Args:
        data: Object to serialize.
        fp: File-like object with ``write``.
        indent: Base indentation level (like ``json.dump``).
        max_inline: Maximum character count (including brackets/braces) allowed
            for an inline container.
        separators: Passed through to ``json.dumps`` for spacing control.
        sort_keys: Whether to sort dictionary keys.
    """

    fp.write(
        json_dumps_streamlined(
            data,
            indent=indent,
            max_inline=max_inline,
            separators=separators,
            sort_keys=sort_keys,
        )
    )


def json_dumps_streamlined(
    data: Any,
    *,
    indent: int = 4,
    max_inline: int = 60,
    separators: tuple[str, str] = (",", ":"),
    sort_keys: bool = False,
) -> str:
    """Return the streamlined JSON string (like ``dump_json_streamlined`` but in-memory)."""

    compact_separators = separators

    def _inline_repr(value: Any) -> Optional[str]:
        try:
            return json.dumps(value, separators=compact_separators, sort_keys=sort_keys)
        except TypeError:
            return None

    def _render(value: Any, level: int) -> str:
        inline = _inline_repr(value)
        if inline is not None and len(inline) <= max_inline:
            return inline

        indent_str = " " * (indent * level)
        child_indent = " " * (indent * (level + 1))

        if isinstance(value, dict):
            items = list(value.items())
            if sort_keys:
                items = sorted(items)
            parts = ["{"]
            total = len(items)
            for idx, (key, val) in enumerate(items):
                rendered = _render(val, level + 1)
                comma = "," if idx < total - 1 else ""
                parts.append(f"{child_indent}{json.dumps(key)}: {rendered}{comma}")
            parts.append(f"{indent_str}}}")
            return "\n".join(parts)

        if isinstance(value, (list, tuple)):
            total = len(value)
            parts = ["["]
            for idx, item in enumerate(value):
                rendered = _render(item, level + 1)
                comma = "," if idx < total - 1 else ""
                parts.append(f"{child_indent}{rendered}{comma}")
            parts.append(f"{indent_str}]")
            return "\n".join(parts)

        return json.dumps(value, separators=compact_separators)

    return _render(data, 0)


def trim_analysis_aliases(analysis: dict) -> dict:
    """Return a copy of ``analysis`` with redundant alias metrics removed.

    Two clean-up rules are applied:

    1. If a key ends with ``"_usd"`` and its value matches the base metric
       (the same key with the suffix removed), the base entry is dropped while
       the explicit ``*_usd`` key is retained.
    2. Within the remaining items, if multiple keys are permutations of the same
       underscore-separated tokens and share the exact value (e.g.
       ``"drawdown_btc_worst"`` vs ``"drawdown_worst_btc"``), only a single key
       is kept. Preference is given to keys whose trailing token is a currency
       tag (``usd``/``btc``); ties fall back to key length and lexical order.

    The original ``analysis`` mapping is left untouched.
    """

    trimmed = dict(analysis)

    # Step 1: remove base keys when *_usd carries the same value.
    for key, value in list(trimmed.items()):
        if key.endswith("_usd"):
            base_key = key[:-4]
            if base_key in trimmed and trimmed[base_key] == value:
                trimmed.pop(base_key)

    # Step 2: remove duplicate permutations sharing identical values.
    groups = {}
    for key in trimmed:
        canon = tuple(sorted(key.split("_")))
        groups.setdefault(canon, []).append(key)

    def _score(alias: str) -> tuple:
        tokens = alias.split("_")
        tail_currency = 1 if tokens and tokens[-1] in {"usd", "btc"} else 0
        return (tail_currency, -len(alias), alias)

    for keys in groups.values():
        if len(keys) < 2:
            continue
        values = {}
        for key in keys:
            values.setdefault(trimmed[key], []).append(key)
        for aliases in values.values():
            if len(aliases) < 2:
                continue
            keep = max(aliases, key=_score)
            for alias in aliases:
                if alias != keep:
                    trimmed.pop(alias, None)

    return trimmed


def filter_markets(markets: dict, exchange: str, quote=None, verbose=False) -> (dict, dict, dict):
    """
    returns (eligible, ineligible, reasons)
    """
    eligible = {}
    ineligible = {}
    reasons = {}
    quote = get_quote(normalize_exchange_name(exchange), quote)
    for k, v in markets.items():
        if not v["active"]:
            ineligible[k] = v
            reasons[k] = "not active"
        elif not v["swap"]:
            ineligible[k] = v
            reasons[k] = "not swap"
        elif not v["linear"]:
            ineligible[k] = v
            reasons[k] = "not linear"
        elif not k.endswith(f"/{quote}:{quote}"):
            ineligible[k] = v
            reasons[k] = "wrong quote"
        elif exchange == "hyperliquid" and (
            v.get("info", {}).get("onlyIsolated")
            or float(v.get("info", {}).get("openInterest")) == 0.0
        ):
            ineligible[k] = v
            reasons[k] = f"ineligible on {exchange}"
        else:
            eligible[k] = v

    if verbose:
        for line in sorted(set(reasons.values())):
            syms = [k for k in reasons if reasons[k] == line]
            if len(syms) > 12:
                logging.info(f"{line}: {len(syms)} symbols")
            elif len(syms) > 0:
                logging.info(f"{line}: {','.join(sorted(set([s for s in syms])))}")

    return eligible, ineligible, reasons


async def load_markets(
    exchange: str,
    max_age_ms: int = 1000 * 60 * 60 * 24,
    verbose=True,
    cc=None,
    quote=None,
) -> dict:
    """
    Standalone helper to load and cache CCXT markets for a given exchange.

    - Reads from caches/{exchange}/markets.json if fresh
    - Otherwise fetches via ccxt, writes cache, and returns the markets dict

    Returns a markets dictionary as provided by ccxt.

    Note: Uses the exchange name as-is (e.g., "binance" not "binanceusdm") for
    consistency with other cache paths (pnls, ohlcv, fill_events).
    """
    # Prefer cc.id when a ccxt instance is supplied, otherwise use the provided exchange string.
    ex = (getattr(cc, "id", None) or exchange or "").lower()
    markets_path = os.path.join("caches", ex, "markets.json")

    # Try cache first
    try:
        if os.path.exists(markets_path):
            if utc_ms() - get_file_mod_ms(markets_path) < max_age_ms:
                with open(markets_path, "r") as f:
                    markets = json.load(f)
                if verbose:
                    logging.info(f"{ex} Loaded markets from cache")
                create_coin_symbol_map_cache(ex, markets, quote=quote, verbose=verbose)
                return markets
    except Exception as e:
        logging.error("Error loading %s: %s", markets_path, e)

    # Fetch from exchange via ccxt
    owned_cc = cc is None
    if owned_cc:
        cc = load_ccxt_instance(ex, enable_rate_limit=True)
    try:
        markets = await cc.load_markets(True)
    except Exception as e:
        logging.error(f"Error loading markets from {ex}: {e}")
        raise
    finally:
        # Only close the ccxt client if we created it here.
        if owned_cc:
            try:
                await cc.close()
            except Exception:
                pass

    # Dump to cache
    try:
        path = make_get_filepath(markets_path)
        with open(path, "w") as f:
            json.dump(markets, f)
        if verbose:
            logging.info(f"{ex} Dumped markets to cache")
    except Exception as e:
        logging.error("Error dumping markets to cache at %s: %s", markets_path, e)
    create_coin_symbol_map_cache(ex, markets, quote=quote, verbose=verbose)
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


def load_ccxt_instance(exchange_id: str, enable_rate_limit: bool = True, timeout_ms: int = 60_000):
    """
    Return a ccxt async-support exchange instance for the given exchange id.

    The returned instance should be closed by the caller with: await cc.close()
    """
    ex = normalize_exchange_name(exchange_id)
    try:
        cc = getattr(ccxt, ex)(
            {
                "enableRateLimit": bool(enable_rate_limit),
                # Default ccxt timeout can be too low for long lookbacks; raise to be tolerant.
                "timeout": int(timeout_ms),
            }
        )
    except Exception:
        raise RuntimeError(f"ccxt exchange '{ex}' not available")
    try:
        cc.options["defaultType"] = "swap"
        if ex == "hyperliquid":
            cc.options["fetchMarkets"]["types"] = ["swap"]
    except Exception:
        pass
    try:
        override = resolve_custom_endpoint_override(ex)
        apply_rest_overrides_to_ccxt(cc, override)
    except Exception as exc:
        logging.warning("Failed to apply custom endpoint override for %s: %s", ex, exc)
    return cc


def get_quote(exchange, quote=None):
    """Return quote currency for an exchange.

    Args:
        exchange: Exchange name
        quote: Explicit quote override (from api-keys.json).
               If provided, returns this value directly.

    Returns:
        Quote currency string (e.g., "USDT", "USDC")
    """
    if quote is not None:
        return quote
    # Legacy hardcoded defaults for backward compatibility
    exchange = normalize_exchange_name(exchange)
    return "USDC" if exchange in ["hyperliquid", "defx", "paradex"] else "USDT"


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
    Uses shared locking to prevent reading during concurrent writes.
    """
    # Run stale lock cleanup on first access
    _cleanup_stale_symbol_map_locks()

    path = os.path.join("caches", exchange, "coin_to_symbol_map.json")
    try:
        st = os.stat(path)
        mtime_ns, size = st.st_mtime_ns, st.st_size
    except Exception:
        return {}
    entry = _COIN_TO_SYMBOL_CACHE.get(exchange)
    if entry and entry.get("mtime_ns") == mtime_ns and entry.get("size") == size:
        return entry.get("map", {})
    lock_path = path + ".lock"
    try:
        with portalocker.Lock(lock_path, timeout=_SYMBOL_MAP_LOCK_TIMEOUT, flags=portalocker.LOCK_SH):
            with open(path) as f:
                data = json.load(f)
        _COIN_TO_SYMBOL_CACHE[exchange] = {"map": data, "mtime_ns": mtime_ns, "size": size}
        return data
    except portalocker.LockException:
        logging.warning("Could not acquire shared lock for %s, returning cached data", path)
        return entry.get("map", {}) if entry else {}
    except Exception as e:
        logging.error(f"failed to load coin_to_symbol_map for {exchange}: {e}")
        return {}


def _load_symbol_to_coin_map() -> dict:
    """
    Lazily load and cache caches/symbol_to_coin_map.json in memory.
    Reloads if the file changes on disk (mtime or size).
    Uses shared locking to prevent reading during concurrent writes.
    """
    # Run stale lock cleanup on first access
    _cleanup_stale_symbol_map_locks()

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
    lock_path = path + ".lock"
    try:
        with portalocker.Lock(lock_path, timeout=_SYMBOL_MAP_LOCK_TIMEOUT, flags=portalocker.LOCK_SH):
            with open(path) as f:
                data = json.load(f)
        _SYMBOL_TO_COIN_CACHE["map"] = data
        _SYMBOL_TO_COIN_CACHE["mtime_ns"] = mtime_ns
        _SYMBOL_TO_COIN_CACHE["size"] = size
        return data
    except portalocker.LockException:
        logging.warning("Could not acquire shared lock for %s, returning cached data", path)
        return entry.get("map") if entry.get("map") is not None else {}
    except Exception as e:
        logging.error(f"failed to load symbol_to_coin_map: {e}")
        return {}


def _build_coin_symbol_maps(markets, quote):
    """
    Build coin_to_symbol_map (as dict of lists) and symbol_to_coin_map from markets data.
    This function is pure and performs no disk I/O.
    """
    coin_to_symbol_map = defaultdict(set)
    symbol_to_coin_map = {}
    for k, v in markets.items():
        try:
            # Only include swap markets with the right quote.
            if not v.get("swap"):
                continue
            # If "linear" is explicitly False, skip; otherwise treat missing as acceptable.
            if v.get("linear") is False:
                continue
            if not k.endswith(f":{quote}"):
                continue
            coin = ""
            variants = set()
            for k0 in ["baseName", "base"]:
                if base := v.get(k0):
                    variants.add(base)
                    variants.add(base.replace("k", ""))
                    variants.add(remove_powers_of_ten(base))
                    cleaned = remove_powers_of_ten(base.replace("k", ""))
                    variants.add(cleaned)
                    if not coin:
                        coin = cleaned
            for variant in variants:
                symbol_to_coin_map[variant] = coin
                symbol_to_coin_map[k] = coin
                coin_to_symbol_map[variant].add(k)
            if symbol_id := v.get("id"):
                symbol_to_coin_map[symbol_id] = coin
        except Exception:
            # Skip malformed market entries but continue processing others
            continue

    # Convert sets to lists for JSON serialisation / on-disk storage
    coin_to_symbol_map = {k: list(v) for k, v in coin_to_symbol_map.items()}
    return coin_to_symbol_map, symbol_to_coin_map


def _write_coin_symbol_maps(
    exchange: str, coin_to_symbol_map: dict, symbol_to_coin_map: dict, verbose=True
):
    """
    Write coin/symbol maps to disk with file locking and atomic writes.
    Uses portalocker to prevent race conditions when multiple bots start simultaneously.
    """
    # Run stale lock cleanup on first access
    _cleanup_stale_symbol_map_locks()

    coin_to_symbol_map_path = make_get_filepath(
        os.path.join("caches", exchange, "coin_to_symbol_map.json")
    )
    symbol_to_coin_map_path = make_get_filepath(os.path.join("caches", "symbol_to_coin_map.json"))

    # Write coin_to_symbol_map (per-exchange) with locking
    c2s_lock_path = coin_to_symbol_map_path + ".lock"
    try:
        with portalocker.Lock(c2s_lock_path, timeout=_SYMBOL_MAP_LOCK_TIMEOUT):
            if verbose:
                logging.info("dumping coin_to_symbol_map %s", coin_to_symbol_map_path)
            _atomic_write_json(coin_to_symbol_map_path, coin_to_symbol_map, indent=4, sort_keys=True)
    except portalocker.LockException:
        logging.warning("Could not acquire lock for %s, skipping write", coin_to_symbol_map_path)

    # Write symbol_to_coin_map (global) with locking
    s2c_lock_path = symbol_to_coin_map_path + ".lock"
    try:
        with portalocker.Lock(s2c_lock_path, timeout=_SYMBOL_MAP_LOCK_TIMEOUT):
            if verbose:
                logging.info("dumping symbol_to_coin_map %s", symbol_to_coin_map_path)
            _atomic_write_json(symbol_to_coin_map_path, symbol_to_coin_map)
    except portalocker.LockException:
        logging.warning("Could not acquire lock for %s, skipping write", symbol_to_coin_map_path)

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


def create_coin_symbol_map_cache(exchange: str, markets, quote=None, verbose=True):
    """
    High-level function that coordinates loading any existing symbol_to_coin_map,
    building fresh maps from markets, merging them (new data overrides), and
    writing results to disk. IO is performed here; conversion logic lives in
    _build_coin_symbol_maps().

    Uses file locking to make the read-modify-write cycle atomic, preventing
    race conditions when multiple bots start simultaneously.

    Note: Uses the exchange name as-is (e.g., "binance" not "binanceusdm") for
    consistency with other cache paths.
    """
    # Run stale lock cleanup on first access
    _cleanup_stale_symbol_map_locks()

    try:
        exchange = (exchange or "").lower()
        quote = get_quote(exchange, quote)

        symbol_to_coin_map_path = make_get_filepath(os.path.join("caches", "symbol_to_coin_map.json"))
        s2c_lock_path = symbol_to_coin_map_path + ".lock"

        # Lock the symbol_to_coin_map for the entire read-modify-write cycle
        try:
            with portalocker.Lock(s2c_lock_path, timeout=_SYMBOL_MAP_LOCK_TIMEOUT):
                # Read existing symbol->coin mappings while holding lock
                symbol_to_coin_map = {}
                try:
                    if os.path.exists(symbol_to_coin_map_path):
                        with open(symbol_to_coin_map_path, "r") as f:
                            symbol_to_coin_map = json.load(f)
                except Exception as e:
                    logging.error("failed to load symbol_to_coin_map %s", e)

                # Build fresh maps from provided markets (pure logic)
                coin_to_symbol_map, new_symbol_to_coin_map = _build_coin_symbol_maps(markets, quote)

                # Merge: prefer new discovered mappings while retaining others
                symbol_to_coin_map.update(new_symbol_to_coin_map)

                # Write symbol_to_coin_map atomically while still holding lock
                if verbose:
                    logging.info("dumping symbol_to_coin_map %s", symbol_to_coin_map_path)
                _atomic_write_json(symbol_to_coin_map_path, symbol_to_coin_map)

                # Update in-memory cache
                try:
                    st2 = os.stat(symbol_to_coin_map_path)
                    _SYMBOL_TO_COIN_CACHE["map"] = symbol_to_coin_map
                    _SYMBOL_TO_COIN_CACHE["mtime_ns"] = st2.st_mtime_ns
                    _SYMBOL_TO_COIN_CACHE["size"] = st2.st_size
                except Exception:
                    pass

            # Write coin_to_symbol_map separately (per-exchange, uses its own lock)
            coin_to_symbol_map_path = make_get_filepath(
                os.path.join("caches", exchange, "coin_to_symbol_map.json")
            )
            c2s_lock_path = coin_to_symbol_map_path + ".lock"
            try:
                with portalocker.Lock(c2s_lock_path, timeout=_SYMBOL_MAP_LOCK_TIMEOUT):
                    if verbose:
                        logging.info("dumping coin_to_symbol_map %s", coin_to_symbol_map_path)
                    _atomic_write_json(coin_to_symbol_map_path, coin_to_symbol_map, indent=4, sort_keys=True)
                    # Update in-memory cache
                    try:
                        st = os.stat(coin_to_symbol_map_path)
                        _COIN_TO_SYMBOL_CACHE[exchange] = {
                            "map": coin_to_symbol_map,
                            "mtime_ns": st.st_mtime_ns,
                            "size": st.st_size,
                        }
                    except Exception:
                        pass
            except portalocker.LockException:
                logging.warning("Could not acquire lock for %s, skipping write", coin_to_symbol_map_path)

        except portalocker.LockException:
            logging.warning("Could not acquire lock for symbol map cache update, skipping")
            return False

        return True
    except Exception as e:
        logging.error("error with create_coin_symbol_map_cache %s: %s", exchange, e)
        return False


def coin_to_symbol(coin, exchange, quote=None):
    # caches coin_to_symbol_map in memory and reloads if file changes
    if coin == "":
        return ""
    ex = (exchange or "").lower()
    quote = get_quote(ex, quote)
    coin_sanitized = symbol_to_coin(coin)
    fallback = f"{coin_sanitized}/{quote}:{quote}"
    try:
        loaded = _load_coin_to_symbol_map(ex)
        candidates = loaded.get(coin_sanitized, []) if loaded else []
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            logging.info(
                "Multiple candidates for %s (raw=%s): %s",
                coin_sanitized,
                coin,
                candidates,
            )
            return candidates[0]
        if loaded:
            # map present but coin missing
            warn_key = (ex, coin_sanitized)
            if warn_key not in _COIN_TO_SYMBOL_FALLBACKS:
                logging.warning(
                    "No mapping for %s (raw=%s) on %s; using fallback %s",
                    coin_sanitized,
                    coin,
                    ex,
                    fallback,
                )
                _COIN_TO_SYMBOL_FALLBACKS.add(warn_key)
        else:
            warn_key = (ex, coin_sanitized)
            if warn_key not in _COIN_TO_SYMBOL_FALLBACKS:
                logging.warning(
                    "coin_to_symbol map for %s missing; using fallback for %s (raw=%s) -> %s",
                    ex,
                    coin_sanitized,
                    coin,
                    fallback,
                )
                _COIN_TO_SYMBOL_FALLBACKS.add(warn_key)
    except Exception as e:
        logging.error(
            "error with coin_to_symbol %s (raw=%s) %s: %s", coin_sanitized, coin, exchange, e
        )
    return fallback


def get_caller_name():
    return inspect.currentframe().f_back.f_back.f_code.co_name


def symbol_to_coin(symbol, verbose=True):
    # caches symbol_to_coin_map in memory and reloads if file changes
    try:
        loaded = _load_symbol_to_coin_map()
        if symbol in loaded:
            return loaded[symbol]
        msg = f"failed to convert {symbol} to its coin with symbol_to_coin_map. Caller: {get_caller_name()}"
    except Exception:
        msg = f"failed to convert {symbol} to its coin with symbol_to_coin_map. Caller: {get_caller_name()}"

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
    if verbose:
        warn_key = str(symbol)
        if warn_key not in _SYMBOL_TO_COIN_WARNINGS:
            logging.warning(msg)
            _SYMBOL_TO_COIN_WARNINGS.add(warn_key)
    return coin


def coin_symbol_warning_counts() -> dict[str, int]:
    """Return counts of fallback conversions for summary logging."""
    return {
        "coin_to_symbol_fallbacks": len(_COIN_TO_SYMBOL_FALLBACKS),
        "symbol_to_coin_fallbacks": len(_SYMBOL_TO_COIN_WARNINGS),
    }


def _snapshot(value):
    return deepcopy(value) if isinstance(value, (dict, list)) else value


def _diff_snapshot(before, after):
    if before == after:
        return None
    return {"old": _snapshot(before), "new": _snapshot(after)}


async def format_approved_ignored_coins(config, exchanges: [str], quote=None, verbose=True):
    if isinstance(exchanges, str):
        exchanges = [exchanges]
    before_approved = deepcopy(config.get("live", {}).get("approved_coins"))
    before_ignored = deepcopy(config.get("live", {}).get("ignored_coins"))
    before_sources = deepcopy(config.get("_coins_sources", {}))
    coin_sources = config.setdefault("_coins_sources", {})
    approved_source = coin_sources.get("approved_coins", config.get("live", {}).get("approved_coins"))
    if approved_source is None:
        approved_source = _require_live_value(config, "approved_coins")
    coin_sources["approved_coins"] = deepcopy(approved_source)
    if approved_source in [
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
        if bool(_require_live_value(config, "empty_means_all_approved")):
            marketss = await asyncio.gather(*[load_markets(ex, verbose=False, quote=quote) for ex in exchanges])
            marketss = [filter_markets(m, ex, quote=quote)[0] for m, ex in zip(marketss, exchanges)]
            approved_coins = set()
            for markets in marketss:
                for symbol in markets:
                    approved_coins.add(symbol_to_coin(symbol, verbose=verbose))
            approved_coins_sorted = sorted([x for x in approved_coins if x])
            config["live"]["approved_coins"] = {
                "long": approved_coins_sorted,
                "short": approved_coins_sorted,
            }
        else:
            # leave empty
            config["live"]["approved_coins"] = {"long": [], "short": []}
    else:
        ac = normalize_coins_source(approved_source)
        config["live"]["approved_coins"] = {
            pside: [cf for x in ac[pside] if (cf := symbol_to_coin(x))] for pside in ac
        }

    ignored_source = coin_sources.get("ignored_coins", config.get("live", {}).get("ignored_coins"))
    if ignored_source is None:
        ignored_source = _require_live_value(config, "ignored_coins")
    coin_sources["ignored_coins"] = deepcopy(ignored_source)
    ic = normalize_coins_source(ignored_source)
    config["live"]["ignored_coins"] = {
        pside: [cf for x in ic[pside] if (cf := symbol_to_coin(x))] for pside in ic
    }

    approved_diff = _diff_snapshot(before_approved, config["live"]["approved_coins"])
    ignored_diff = _diff_snapshot(before_ignored, config["live"]["ignored_coins"])
    sources_diff = _diff_snapshot(before_sources, config.get("_coins_sources", {}))
    if approved_diff or ignored_diff or sources_diff:
        details = {"exchanges": list(exchanges)}
        if approved_diff:
            details["approved_coins"] = approved_diff
        if ignored_diff:
            details["ignored_coins"] = ignored_diff
        if sources_diff:
            details["coin_sources"] = sources_diff
        record_transform(config, "format_approved_ignored_coins", details)


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

        def _maybe_read(path_candidate):
            resolved = _resolve_coins_file_path(path_candidate)
            if resolved is not None:
                return read_external_coins_lists(str(resolved))
            return None

        if isinstance(x, str):
            loaded = _maybe_read(x)
            if loaded is not None:
                return loaded
        if isinstance(x, (list, tuple)) and len(x) == 1 and isinstance(x[0], str):
            loaded = _maybe_read(x[0])
            if loaded is not None:
                return loaded
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
        with open(filepath, "r") as f:
            content = hjson.load(f)
        if isinstance(content, list) and all(isinstance(x, str) for x in content):
            return {"long": content, "short": content}
        if isinstance(content, dict) and all(
            pside in content
            and isinstance(content[pside], list)
            and all(isinstance(x, str) for x in content[pside])
            for pside in ["long", "short"]
        ):
            return content
    except Exception:
        # fallback to plain-text reading below
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


async def get_first_ohlcv_iteratively(cc, symbol):
    """Return the earliest OHLCV candle for a Bitget market.

    Bitget does not accept a conventional ``since`` parameter for swap OHLCV
    queries. Instead we page backwards using ``params={"until": ms}``, where an
    empty response indicates that ``until`` predates the instrument listing.  We
    leverage that behaviour to binary-search over monthly candles and then
    refine the result with a daily fetch.  The returned value is the first full
    candle ``[timestamp, open, high, low, close, volume]`` if available, else
    ``None``."""

    DAY_MS = 86_400_000
    MONTH_MS = 30 * DAY_MS

    async def fetch_month(until: Optional[int] = None):
        params = {"limit": 200}
        if until is not None:
            params["until"] = int(until)
        return await cc.fetch_ohlcv(symbol, timeframe="1M", params=params)

    async def fetch_day(until: int):
        return await cc.fetch_ohlcv(
            symbol, timeframe="1d", params={"until": int(until), "limit": 200}
        )

    month_chunk = await fetch_month()
    if not month_chunk:
        return None

    best_candle = month_chunk[0]
    first_month_ts = int(best_candle[0])

    # Initial bounds for binary search: start near zero, clamp upper bound to now.
    now_ms = int(getattr(cc, "milliseconds")())
    lo = 0
    hi = max(now_ms, int(month_chunk[-1][0]) + MONTH_MS)

    while hi - lo > MONTH_MS:
        mid = (lo + hi) // 2
        candles = await fetch_month(mid)
        if candles:
            new_first = int(candles[0][0])
            if new_first >= hi:
                break
            best_candle = candles[0]
            hi = new_first
            first_month_ts = new_first
        else:
            lo = mid

    # Sequentially step back in case the monthly page was capped by the limit.
    while True:
        prev_until = max(0, first_month_ts - 1)
        if prev_until <= 0:
            break
        prev_chunk = await fetch_month(prev_until)
        if not prev_chunk:
            break
        prev_first = int(prev_chunk[0][0])
        if prev_first >= first_month_ts:
            break
        first_month_ts = prev_first
        best_candle = prev_chunk[0]

    # Refine with daily candles near the discovered month boundary.
    daily_chunk = await fetch_day(first_month_ts + 32 * DAY_MS)
    if daily_chunk:
        return daily_chunk[0]

    return best_candle
