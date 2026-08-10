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
import warnings
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
    resolve_custom_endpoint_override_with_aliases,
)

warnings.filterwarnings(
    "ignore", message="timeout has no effect in blocking mode", module="portalocker"
)

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
_KNOWN_EXCHANGE_QUALIFIERS: Optional[frozenset[str]] = None

# File locking constants for symbol/coin map files
_SYMBOL_MAP_LOCK_STALE_SECONDS = 180  # Remove locks older than 3 minutes
_SYMBOL_MAP_LOCK_TIMEOUT = 5  # Seconds to wait for lock acquisition
_SYMBOL_MAP_STALE_CLEANUP_DONE = False  # Track if cleanup has run this session
PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIRST_OHLCV_TIMESTAMPS_CACHE_VERSION = 2
LEGACY_COINS_FILE_ALIASES = {
    "approved_coins_topmcap.json": Path("configs/approved_coins.json"),
    "approved_coins_topmcap.txt": Path("configs/approved_coins.json"),
}


class MarketIdentifierResolutionError(ValueError):
    """Base class for fail-closed exchange market identifier errors."""


class AmbiguousMarketIdentifier(MarketIdentifierResolutionError):
    """Raised when a convenience alias matches multiple exchange markets."""


class MarketIdentifierExchangeMismatch(MarketIdentifierResolutionError):
    """Raised when an exchange-qualified identifier targets another exchange."""


class UnknownMarketIdentifier(MarketIdentifierResolutionError):
    """Raised when an exact market identifier is absent from a loaded exchange map."""


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
                    logging.debug("removed stale symbol map lock %s (age %.1fs)", lock_path, age)
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
    separators: tuple[str, str] = (", ", ": "),
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
    separators: tuple[str, str] = (", ", ": "),
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
    quote = get_quote(to_ccxt_exchange_id(exchange), quote)
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
        elif exchange == "hyperliquid" and float(v.get("info", {}).get("openInterest", 0)) == 0.0:
            # Zero open interest means market is inactive
            # Note: onlyIsolated=True is allowed for HIP-3 stock perps
            ineligible[k] = v
            reasons[k] = f"ineligible on {exchange}"
        else:
            eligible[k] = v

    if verbose:
        for line in sorted(set(reasons.values())):
            syms = [k for k in reasons if reasons[k] == line]
            log = (
                logging.debug
                if line in {"not active", "wrong quote", "not swap", "not linear"}
                else logging.info
            )
            if len(syms) > 12:
                log(f"{line}: {len(syms)} symbols")
            elif len(syms) > 0:
                log(f"{line}: {','.join(sorted(set([s for s in syms])))}")

    return eligible, ineligible, reasons


async def load_markets(
    exchange: str,
    max_age_ms: int = 1000 * 60 * 60 * 24,
    verbose=True,
    cc=None,
    quote=None,
) -> dict:
    """
    Standalone helper to load and cache markets for a given exchange.

    - Reads from caches/{exchange}/markets.json if fresh
    - Otherwise fetches through the exchange client, writes cache, and returns the markets dict

    Returns a CCXT-compatible markets dictionary.

    Note: Uses the exchange name as-is (e.g., "binance" not "binanceusdm") for
    consistency with other cache paths (pnls, ohlcv, fill_events).
    """
    # The explicit exchange name is the canonical cache identity. A supplied
    # client's library id may differ (for example Gate.io is ``gate`` in CCXT).
    ex = to_standard_exchange_name(exchange or getattr(cc, "id", None) or "")
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

    # Fetch from the exchange client.
    owned_cc = cc is None
    if owned_cc:
        if ex == "bitunix":
            # Bitunix is intentionally native because it is absent from CCXT.
            # Standalone market preloads must use the same public REST client as
            # the live bot so a cold cache cannot fall through to CCXT.
            from exchanges.bitunix import (
                BitunixClient,
                apply_bitunix_endpoint_override,
            )

            client_config = apply_bitunix_endpoint_override(
                {
                    "enableRateLimit": True,
                    "timeout": 60_000,
                    "wsEnabled": False,
                },
                resolve_custom_endpoint_override(ex),
            )
            cc = BitunixClient(client_config)
        else:
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


def to_ccxt_exchange_id(exchange: str) -> str:
    """
    Convert a short exchange name to its ccxt USD-margined perpetual futures id.

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


def to_ccxt_client_id(exchange: str) -> str:
    """Return the CCXT class id used only when constructing a client session."""
    exchange_id = to_ccxt_exchange_id(exchange)
    return "gate" if exchange_id == "gateio" else exchange_id


def exchange_name_aliases(exchange: str) -> tuple[str, ...]:
    """Return accepted canonical, connector, and CCXT names for one venue."""
    raw = str(exchange or "").lower()
    canonical = to_standard_exchange_name(raw)
    aliases = {
        raw,
        canonical,
        to_ccxt_exchange_id(canonical),
        to_ccxt_client_id(canonical),
    }
    return tuple(sorted(alias for alias in aliases if alias))


def to_standard_exchange_name(exchange: str) -> str:
    """
    Convert a ccxt exchange id to the canonical short form used in configs, caches, and logs.

    Examples:
    - "binanceusdm" -> "binance"
    - "kucoinfutures" -> "kucoin"
    - "krakenfutures" -> "kraken"

    If the exchange doesn't have a known suffix, returns it unchanged.
    """
    ex = (exchange or "").lower()

    # CCXT 4.5.66 renamed its Gate.io client from ``gateio`` to ``gate``.
    # Passivbot retains ``gateio`` as the canonical identity for connector
    # routing, caches, broker attribution, events, and persisted state.
    if ex == "gate":
        return "gateio"

    # Remove known futures suffixes
    for suffix in ("usdm", "futures"):
        if ex.endswith(suffix):
            return ex[: -len(suffix)]

    return ex


# Deprecated aliases for backward compatibility - will be removed in a future release
def normalize_exchange_name(exchange: str) -> str:
    """Deprecated: Use to_ccxt_exchange_id() instead."""
    import warnings

    warnings.warn(
        "normalize_exchange_name() is deprecated, use to_ccxt_exchange_id() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return to_ccxt_exchange_id(exchange)


def denormalize_exchange_name(exchange: str) -> str:
    """Deprecated: Use to_standard_exchange_name() instead."""
    import warnings

    warnings.warn(
        "denormalize_exchange_name() is deprecated, use to_standard_exchange_name() instead",
        DeprecationWarning,
        stacklevel=2,
    )
    return to_standard_exchange_name(exchange)


def load_ccxt_instance(exchange_id: str, enable_rate_limit: bool = True, timeout_ms: int = 60_000):
    """
    Return a ccxt async-support exchange instance for the given exchange id.

    The returned instance should be closed by the caller with: await cc.close()
    """
    ex = to_ccxt_exchange_id(exchange_id)
    client_id = to_ccxt_client_id(ex)
    try:
        cc = getattr(ccxt, client_id)(
            {
                "enableRateLimit": bool(enable_rate_limit),
                # Default ccxt timeout can be too low for long lookbacks; raise to be tolerant.
                "timeout": int(timeout_ms),
            }
        )
    except Exception as exc:
        raise RuntimeError(
            f"ccxt exchange client {client_id!r} not available for canonical exchange {ex!r}"
        ) from exc
    try:
        cc.options["defaultType"] = "swap"
        if client_id == "okx":
            cc.options["fetchMarkets"] = {"types": ["swap"]}
        if client_id == "hyperliquid":
            # Include HIP-3 stock perps from TradeXYZ
            cc.options["fetchMarkets"] = {
                "types": ["swap", "hip3"],
                "hip3": {
                    "dex": ["xyz"],  # TradeXYZ DEX for stock perps
                },
            }
    except Exception:
        pass
    override = (
        resolve_custom_endpoint_override_with_aliases(ex, (client_id,))
        if client_id != ex
        else resolve_custom_endpoint_override(ex)
    )
    apply_rest_overrides_to_ccxt(cc, override)
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
    exchange = to_ccxt_exchange_id(exchange)
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


def _build_coin_symbol_maps(markets, quote, exchange=None):
    """
    Build coin_to_symbol_map (as dict of lists) and symbol_to_coin_map from markets data.
    This function is pure and performs no disk I/O.
    """

    def _namespaced_aliases(base: str, market: dict) -> set[str]:
        aliases = set()
        if not isinstance(base, str) or not base:
            return aliases
        is_namespaced_hip3 = bool((market.get("info") or {}).get("hip3")) or base.startswith(
            ("XYZ-", "xyz:")
        )
        if not is_namespaced_hip3:
            return aliases
        if ":" in base:
            prefix, ticker = base.split(":", 1)
            if prefix and ticker:
                aliases.add(ticker)
                aliases.add(f"{prefix.upper()}-{ticker}")
        elif "-" in base:
            prefix, ticker = base.split("-", 1)
            if prefix and ticker:
                aliases.add(ticker)
                aliases.add(f"{prefix.lower()}:{ticker}")
        return aliases

    alias_to_symbols = defaultdict(set)
    exact_alias_to_symbols = defaultdict(set)
    alias_to_coins = defaultdict(set)
    exact_alias_to_coins = defaultdict(set)
    exchange_name = to_standard_exchange_name(exchange) if exchange else None

    def add_exact_alias(alias, symbol, coin):
        alias = str(alias)
        exact_alias_to_symbols[alias].add(symbol)
        exact_alias_to_coins[alias].add(coin)

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
            active = v.get("active") is not False
            coin = ""
            variants = set()
            namespaced_full_aliases = set()
            for k0 in ["baseName", "base"]:
                if base := v.get(k0):
                    variants.add(base)
                    underlying, denomination = market_denomination_identity(
                        base, exchange=exchange, market=v
                    )
                    cleaned = underlying if denomination != 1 else base
                    if denomination != 1:
                        variants.add(underlying)
                    if not coin:
                        coin = cleaned
                    namespaced_aliases = _namespaced_aliases(base, v)
                    variants.update(namespaced_aliases)
                    if "/" in k:
                        suffix = k[k.find("/") :]
                        for alias in {base, *namespaced_aliases}:
                            if ":" in alias or "-" in alias:
                                namespaced_full_aliases.add(f"{alias}{suffix}")
            # Exact exchange identifiers are lossless routing aliases.  They
            # must be available before any convenience normalization such as
            # removing a 1000/k contract multiplier.
            add_exact_alias(k, k, coin)
            for alias in namespaced_full_aliases:
                add_exact_alias(alias, k, coin)
            if active:
                for variant in variants:
                    alias_to_coins[variant].add(coin)
                    alias_to_symbols[variant].add(k)
            if symbol_id := v.get("id"):
                add_exact_alias(symbol_id, k, coin)
        except Exception:
            # Skip malformed market entries but continue processing others
            continue

    # A canonical-looking unqualified alias keeps all convenience candidates
    # when its text also equals a native ID. Explicitly qualified aliases retain
    # the lossless native-ID route, while other exact aliases remain unchanged.
    for alias, symbols in list(exact_alias_to_symbols.items()):
        if alias in alias_to_symbols and not looks_like_exact_market_identifier(alias):
            alias_to_symbols[alias].update(symbols)
            if exchange_name:
                qualified_alias = f"{exchange_name}::{alias}"
                alias_to_symbols[qualified_alias] = set(symbols)
                exact_alias_to_coins[qualified_alias].update(exact_alias_to_coins[alias])
        else:
            alias_to_symbols[alias] = set(symbols)
    coin_to_symbol_map = {
        alias: sorted(symbols) for alias, symbols in sorted(alias_to_symbols.items())
    }

    # The global symbol-to-coin map is used only for canonical labels.  Omit
    # ambiguous convenience aliases instead of choosing whichever market CCXT
    # happened to return first, then overlay unique exact identifiers.
    symbol_to_coin_map = {
        alias: next(iter(alias_to_coins[alias]))
        for alias, symbols in sorted(alias_to_symbols.items())
        if len(symbols) == 1 and len(alias_to_coins[alias]) == 1
    }
    symbol_to_coin_map.update(
        {
            alias: next(iter(coins))
            for alias, coins in sorted(exact_alias_to_coins.items())
            if len(coins) == 1
        }
    )
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
                logging.debug("dumping coin_to_symbol_map %s", coin_to_symbol_map_path)
            _atomic_write_json(coin_to_symbol_map_path, coin_to_symbol_map, indent=4, sort_keys=True)
    except portalocker.LockException:
        logging.warning("Could not acquire lock for %s, skipping write", coin_to_symbol_map_path)

    # Write symbol_to_coin_map (global) with locking
    s2c_lock_path = symbol_to_coin_map_path + ".lock"
    try:
        with portalocker.Lock(s2c_lock_path, timeout=_SYMBOL_MAP_LOCK_TIMEOUT):
            if verbose:
                logging.debug("dumping symbol_to_coin_map %s", symbol_to_coin_map_path)
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
        ambiguity_map_path = make_get_filepath(
            os.path.join("caches", "symbol_to_coin_ambiguities.json")
        )
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
                coin_to_symbol_map, new_symbol_to_coin_map = _build_coin_symbol_maps(
                    markets, quote, exchange=exchange
                )

                # Persist ambiguity ownership per exchange.  The global label
                # map cannot represent an alias which is ambiguous on any
                # loaded exchange, so tombstone the union regardless of cache
                # refresh order.  Replacing this exchange's entry lets a
                # delisted collision become usable again after refresh.
                ambiguity_map = {}
                try:
                    if os.path.exists(ambiguity_map_path):
                        with open(ambiguity_map_path, "r") as f:
                            loaded_ambiguities = json.load(f)
                        if isinstance(loaded_ambiguities, dict):
                            ambiguity_map = loaded_ambiguities
                except Exception as e:
                    logging.error("failed to load symbol ambiguity map %s", e)
                ambiguity_map[exchange] = sorted(
                    alias
                    for alias, candidates in coin_to_symbol_map.items()
                    if len(candidates) > 1
                )
                ambiguous_aliases = {
                    alias
                    for aliases in ambiguity_map.values()
                    if isinstance(aliases, list)
                    for alias in aliases
                }

                for alias in ambiguous_aliases:
                    symbol_to_coin_map.pop(alias, None)
                symbol_to_coin_map.update(
                    {
                        alias: coin
                        for alias, coin in new_symbol_to_coin_map.items()
                        if alias not in ambiguous_aliases
                    }
                )

                # Write symbol_to_coin_map atomically while still holding lock
                if verbose:
                    logging.debug("dumping symbol_to_coin_map %s", symbol_to_coin_map_path)
                _atomic_write_json(symbol_to_coin_map_path, symbol_to_coin_map)
                _atomic_write_json(ambiguity_map_path, ambiguity_map, sort_keys=True)

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
                        logging.debug("dumping coin_to_symbol_map %s", coin_to_symbol_map_path)
                    _atomic_write_json(
                        coin_to_symbol_map_path, coin_to_symbol_map, indent=4, sort_keys=True
                    )
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
                logging.info(
                    "[mapping] could not acquire lock for %s, skipping write", coin_to_symbol_map_path
                )

        except portalocker.LockException:
            logging.info("[mapping] could not acquire lock for symbol map cache update, skipping")
            return False

        return True
    except Exception as e:
        logging.error("error with create_coin_symbol_map_cache %s: %s", exchange, e)
        return False


def split_exchange_qualified_market_identifier(identifier: str):
    """Return ``(exchange, value)`` for a recognized ``exchange::value`` identifier."""
    if not isinstance(identifier, str) or "::" not in identifier:
        return None, identifier
    prefix, value = identifier.split("::", 1)
    if not prefix or not value or "/" in prefix:
        return None, identifier
    global _KNOWN_EXCHANGE_QUALIFIERS
    if _KNOWN_EXCHANGE_QUALIFIERS is None:
        known = set(getattr(ccxt, "exchanges", []))
        known.update(to_standard_exchange_name(exchange) for exchange in tuple(known))
        known.update(
            {
                "binance",
                "binanceusdm",
                "bitget",
                "bitunix",
                "bybit",
                "defx",
                "fake",
                "gate",
                "gateio",
                "hyperliquid",
                "kucoin",
                "kucoinfutures",
                "okx",
                "paradex",
                "weex",
            }
        )
        _KNOWN_EXCHANGE_QUALIFIERS = frozenset(known)
    if prefix.lower() not in _KNOWN_EXCHANGE_QUALIFIERS:
        return None, identifier
    return to_standard_exchange_name(prefix), value


def looks_like_exact_market_identifier(identifier: str) -> bool:
    """Return whether an identifier should reach an exchange map losslessly."""
    raw = str(identifier).strip()
    qualified_exchange, _ = split_exchange_qualified_market_identifier(raw)
    if (
        qualified_exchange is not None
        or "/" in raw
        or raw.isdigit()
        or (":" in raw and all(raw.split(":", 1)))
    ):
        return True
    if "-" in raw:
        return True
    if raw != raw.upper():
        return False
    for quote in ("USDT", "USDC", "BUSD", "USD"):
        match = re.search(rf"{quote}[A-Z0-9_-]*$", raw)
        if match is not None and match.start() >= 1:
            return True
    return False


_MARKET_DENOMINATION_CONVENTIONS = {
    "binance": frozenset({"prefix"}),
    "bitget": frozenset({"prefix"}),
    "bybit": frozenset({"prefix", "suffix"}),
    "hyperliquid": frozenset({"k_prefix"}),
    "kucoin": frozenset({"prefix"}),
}


def market_denomination_identity(
    symbol: str, *, underlying=None, exchange=None, market=None
) -> tuple[str, int]:
    """Return a denomination only when an alias or venue convention establishes it."""
    base = str(symbol).split("/", 1)[0].strip()
    expected = str(underlying).strip() if underlying is not None else None
    conventions = set()
    if exchange:
        conventions = set(
            _MARKET_DENOMINATION_CONVENTIONS.get(
                to_standard_exchange_name(str(exchange)), frozenset()
            )
        )
    if isinstance(market, dict) and bool((market.get("info") or {}).get("hip3")):
        # HIP-3 contains legitimate numeric ticker suffixes such as XYZ100.
        conventions.discard("prefix")
        conventions.discard("suffix")
    elif isinstance(market, dict):
        metadata_underlying = market.get("baseName")
        if (
            isinstance(metadata_underlying, str)
            and metadata_underlying
            and ":" not in metadata_underlying
            and "-" not in metadata_underlying
            and not (
                metadata_underlying.startswith("k")
                and metadata_underlying[1:].isupper()
            )
        ):
            metadata_identity = market_denomination_identity(
                base, underlying=metadata_underlying
            )
            if metadata_identity[1] != 1:
                return metadata_identity

    if expected:
        expected_upper = expected.upper()
        if base.upper() == expected_upper:
            return expected_upper, 1
        prefix_match = re.fullmatch(rf"(1(?:0+)){re.escape(expected)}", base, re.IGNORECASE)
        suffix_match = re.fullmatch(rf"{re.escape(expected)}(1(?:0+))", base, re.IGNORECASE)
        k_prefix_match = re.fullmatch(rf"[kK]{re.escape(expected)}", base)
        if prefix_match:
            return expected_upper, int(prefix_match.group(1))
        if suffix_match:
            return expected_upper, int(suffix_match.group(1))
        if k_prefix_match:
            return expected_upper, 1000
        return base.upper(), 1

    if "k_prefix" in conventions and isinstance(market, dict):
        metadata_base = market.get("baseName")
        if (
            isinstance(metadata_base, str)
            and metadata_base.startswith("k")
            and metadata_base[1:].isupper()
            and base.upper() == metadata_base.upper()
        ):
            return metadata_base[1:].upper(), 1000
    if "prefix" in conventions:
        prefix_match = re.fullmatch(r"(1(?:0+))(.+)", base)
        if prefix_match:
            multiplier, parsed_underlying = prefix_match.groups()
            return parsed_underlying.upper(), int(multiplier)
    if "suffix" in conventions:
        suffix_match = re.fullmatch(r"(.+?)(1(?:0+))", base)
        if suffix_match:
            parsed_underlying, multiplier = suffix_match.groups()
            return parsed_underlying.upper(), int(multiplier)
    return base.upper(), 1


def _preferred_convenience_market_candidate(raw, candidates):
    """Choose one denomination variant for a plain underlying, if selection is safe."""
    if looks_like_exact_market_identifier(raw):
        return None
    requested_underlying = str(raw).strip().upper()

    identities = {
        candidate: market_denomination_identity(candidate, underlying=requested_underlying)
        for candidate in candidates
    }
    if (
        {underlying for underlying, _denomination in identities.values()}
        != {requested_underlying}
        or len(set(identities.values())) != len(identities)
    ):
        return None
    return min(
        candidates,
        key=lambda candidate: (
            identities[candidate][1] != 1,
            identities[candidate][1],
            candidate,
        ),
    )


def _resolve_market_candidates(raw, candidates, exchange):
    candidates = sorted(set(candidates or []))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        if preferred := _preferred_convenience_market_candidate(raw, candidates):
            return preferred
        raise AmbiguousMarketIdentifier(
            f"ambiguous market identifier {raw!r} on {exchange}: matches {candidates}; "
            "use an exact CCXT symbol, native market ID, or "
            f"{exchange}::<native-id>"
        )
    return None


def coin_to_symbol(coin, exchange, quote=None, verbose=True):
    # caches coin_to_symbol_map in memory and reloads if file changes
    if coin == "":
        return ""
    # Denormalize to use canonical form for cache paths (e.g., "binance" not "binanceusdm")
    ex = to_standard_exchange_name(exchange or "")
    quote = get_quote(ex, quote)
    raw = str(coin).strip()
    qualified_exchange, unqualified = split_exchange_qualified_market_identifier(raw)
    if qualified_exchange is not None and qualified_exchange != ex:
        raise MarketIdentifierExchangeMismatch(
            f"market identifier {raw!r} targets {qualified_exchange}, not {ex}"
        )
    lookup_raw = unqualified if qualified_exchange is not None else raw
    try:
        loaded = _load_coin_to_symbol_map(ex)
        if loaded:
            # Exact identifiers and base aliases take precedence over lossy
            # canonicalization.  Ambiguous aliases fail closed.
            lookup_keys = (
                [raw, lookup_raw] if qualified_exchange is not None else [lookup_raw]
            )
            for lookup_key in lookup_keys:
                if resolved := _resolve_market_candidates(
                    raw, loaded.get(lookup_key, []), ex
                ):
                    return resolved

        coin_sanitized = symbol_to_coin(lookup_raw, verbose=verbose)
        if looks_like_exact_market_identifier(raw) or (
            loaded and lookup_raw != coin_sanitized
        ):
            raise UnknownMarketIdentifier(
                f"exact market identifier {raw!r} is unavailable on {ex}; "
                "refresh exchange market metadata or use a valid exact identifier"
            )
        fallback = f"{coin_sanitized}/{quote}:{quote}"
        if loaded:
            if resolved := _resolve_market_candidates(
                raw, loaded.get(coin_sanitized, []), ex
            ):
                return resolved
        if loaded:
            # map present but coin missing
            warn_key = (ex, coin_sanitized)
            if warn_key not in _COIN_TO_SYMBOL_FALLBACKS:
                if verbose:
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
                if verbose:
                    logging.warning(
                        "coin_to_symbol map for %s missing; using fallback for %s (raw=%s) -> %s",
                        ex,
                        coin_sanitized,
                        coin,
                        fallback,
                    )
                _COIN_TO_SYMBOL_FALLBACKS.add(warn_key)
    except MarketIdentifierResolutionError:
        raise
    except Exception as e:
        coin_sanitized = symbol_to_coin(lookup_raw, verbose=verbose)
        fallback = f"{coin_sanitized}/{quote}:{quote}"
        if verbose:
            logging.error(
                "error with coin_to_symbol %s (raw=%s) %s: %s", coin_sanitized, coin, exchange, e
            )
    return fallback


def get_caller_name():
    return inspect.currentframe().f_back.f_back.f_code.co_name


def heuristic_symbol_to_coin(symbol):
    """Return the legacy deterministic coin heuristic without consulting caches."""
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
    return coin


def symbol_to_coin(symbol, verbose=True):
    # caches symbol_to_coin_map in memory and reloads if file changes
    raw = str(symbol).strip()
    qualified_exchange, _ = split_exchange_qualified_market_identifier(raw)
    if qualified_exchange is not None:
        return raw
    try:
        loaded = _load_symbol_to_coin_map()
        if raw in loaded:
            return loaded[raw]
        msg = f"failed to convert {symbol} to its coin with symbol_to_coin_map. Caller: {get_caller_name()}"
    except Exception:
        msg = f"failed to convert {symbol} to its coin with symbol_to_coin_map. Caller: {get_caller_name()}"

    coin = heuristic_symbol_to_coin(raw)
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


def _resolve_fake_scenario_path(config) -> Optional[str]:
    live = config.get("live", {}) if isinstance(config, dict) else {}
    scenario_path = live.get("fake_scenario_path")
    if scenario_path:
        return scenario_path
    user = live.get("user")
    if not user:
        return None
    from procedures import load_user_info

    user_info = load_user_info(user)
    return user_info.get("fake_scenario_path")


def _load_fake_approved_coins(config, *, quote=None):
    live = config.get("live", {}) if isinstance(config, dict) else {}
    scenario_path = _resolve_fake_scenario_path(config)
    if not scenario_path:
        raise ValueError(
            "fake exchange approved_coins='all' requires live.fake_scenario_path "
            "or api-keys fake_scenario_path during startup"
        )
    from exchanges.fake import load_fake_scenario

    scenario = load_fake_scenario(scenario_path)
    symbols_config = scenario.get("symbols")
    if not isinstance(symbols_config, dict) or not symbols_config:
        raise ValueError("Fake scenario must define symbols to expand approved_coins='all'")
    approved_coins = []
    for symbol in symbols_config:
        if quote is not None:
            symbol_quote = str(symbol).split("/", 1)[1].split(":", 1)[0]
            if symbol_quote != str(quote):
                continue
        coin = symbol_to_coin(symbol)
        if coin:
            approved_coins.append(coin)
    return sorted(set(approved_coins))


def _approved_all_market_identifiers(exchange_markets_quotes) -> set[str]:
    """Return identifiers whose collision scope is safe across all exchanges."""
    records = []
    collision_canonicals = set()
    canonical_underlyings = defaultdict(set)
    for exchange, markets, quote in exchange_markets_quotes:
        coin_to_symbol_map, symbol_to_coin_map = _build_coin_symbol_maps(
            markets, quote, exchange=exchange
        )
        canonical_groups = defaultdict(list)
        for symbol, market in markets.items():
            if (market or {}).get("active") is False:
                continue
            canonical = symbol_to_coin_map.get(symbol) or heuristic_symbol_to_coin(symbol)
            if canonical:
                canonical_groups[canonical].append(symbol)

        exchange_name = to_standard_exchange_name(exchange)
        for canonical, symbols in canonical_groups.items():
            identities = {
                symbol: market_denomination_identity(symbol, underlying=canonical)
                for symbol in symbols
            }
            canonical_underlyings[canonical].update(
                underlying for underlying, _denomination in identities.values()
            )
            candidates = coin_to_symbol_map.get(canonical, [])
            preferred_candidate = _preferred_convenience_market_candidate(
                canonical, candidates
            )
            if (
                len(set(identities.values())) != len(identities)
                or (len(candidates) > 1 and preferred_candidate is None)
            ):
                collision_canonicals.add(canonical)
            for symbol in symbols:
                market = markets.get(symbol) or {}
                exact_identifier = str(market.get("id") or symbol)
                records.append((exchange_name, canonical, exact_identifier))

    collision_canonicals.update(
        canonical
        for canonical, underlyings in canonical_underlyings.items()
        if len(underlyings) > 1
    )

    return {
        f"{exchange}::{exact_identifier}" if canonical in collision_canonicals else canonical
        for exchange, canonical, exact_identifier in records
    }


def _preserve_market_identifiers(values) -> list[str]:
    """Strip configured identifiers without collapsing exchange market identity."""
    return [raw for value in values if (raw := str(value).strip())]


async def reject_cross_exchange_market_identifier_collisions(
    identifiers, exchanges, *, quote=None, verbose=True
) -> None:
    """Reject cross-venue collisions for explicit exact market identifiers."""
    standard_exchanges = list(
        dict.fromkeys(
            to_standard_exchange_name(str(exchange))
            for exchange in exchanges
            if str(exchange).lower() != "fake"
        )
    )
    if len(standard_exchanges) < 2:
        return

    identifiers = {
        str(identifier).strip()
        for identifier in identifiers
        if str(identifier).strip()
    }
    candidates = sorted(
        identifier
        for identifier in identifiers
        if split_exchange_qualified_market_identifier(identifier)[0] is None
    )
    if not candidates:
        return

    loaded_markets = await asyncio.gather(
        *[load_markets(exchange, verbose=False, quote=quote) for exchange in standard_exchanges]
    )
    markets_by_exchange = dict(zip(standard_exchanges, loaded_markets))
    for identifier in candidates:
        resolved = {}
        for exchange in standard_exchanges:
            try:
                symbol = coin_to_symbol(
                    identifier, exchange, quote=quote, verbose=verbose
                )
            except (MarketIdentifierExchangeMismatch, UnknownMarketIdentifier):
                continue
            market = markets_by_exchange[exchange].get(symbol) or {}
            underlying, denomination = market_denomination_identity(
                symbol, exchange=exchange, market=market
            )
            resolved[exchange] = {
                "coin": heuristic_symbol_to_coin(symbol),
                "symbol": symbol,
                "underlying": underlying,
                "denomination": denomination,
                "identity": f"{underlying}@{denomination}",
            }
        requested_upper = str(identifier).strip().upper()
        # A canonical-looking unqualified name denotes the economic underlying, even
        # when it happens to equal one venue's native market ID. Exact contract intent
        # must be expressed with exact syntax (for example exchange::<native-id>).
        compare_denomination = looks_like_exact_market_identifier(identifier) or any(
            item["underlying"] != requested_upper for item in resolved.values()
        )
        comparison_identities = {
            (
                item["underlying"],
                item["denomination"] if compare_denomination else None,
            )
            for item in resolved.values()
        }
        if len(comparison_identities) > 1:
            raise AmbiguousMarketIdentifier(
                f"market identifier {identifier!r} resolves to different contracts across "
                f"configured exchanges: {resolved}; use exchange::<native-id>"
            )


async def _remove_resolved_ignored_markets(
    config,
    exchanges,
    *,
    quote=None,
    verbose=True,
    include_backtest_coin_source_venues=False,
) -> None:
    """Subtract ignored aliases resolving to the same venue market."""
    approved_by_side = config["live"]["approved_coins"]
    ignored_by_side = config["live"]["ignored_coins"]
    potentially_overlapping_sides = [
        pside
        for pside in ("long", "short")
        if any(
            approved != ignored
            and (
                looks_like_exact_market_identifier(approved)
                or looks_like_exact_market_identifier(ignored)
                or heuristic_symbol_to_coin(approved) == heuristic_symbol_to_coin(ignored)
            )
            for approved in approved_by_side[pside]
            for ignored in ignored_by_side[pside]
        )
    ]
    if not potentially_overlapping_sides:
        return

    source_exchanges = (
        (config.get("backtest", {}).get("coin_sources") or {}).values()
        if include_backtest_coin_source_venues
        else []
    )
    standard_exchanges = list(
        dict.fromkeys(
            to_standard_exchange_name(str(exchange))
            for exchange in [*exchanges, *source_exchanges]
            if str(exchange).lower() != "fake"
        )
    )
    if not standard_exchanges:
        return
    loaded_markets = await asyncio.gather(
        *[load_markets(ex, verbose=False, quote=quote) for ex in standard_exchanges]
    )
    markets_by_exchange = dict(zip(standard_exchanges, loaded_markets))

    def resolve_if_available(identifier, exchange):
        try:
            return coin_to_symbol(identifier, exchange, quote=quote, verbose=verbose)
        except (MarketIdentifierExchangeMismatch, UnknownMarketIdentifier):
            return None

    for pside in potentially_overlapping_sides:
        ignored_symbols_by_exchange = {
            exchange: {
                symbol
                for identifier in ignored_by_side[pside]
                if (symbol := resolve_if_available(identifier, exchange)) is not None
            }
            for exchange in standard_exchanges
        }
        surviving_approved = []
        for identifier in approved_by_side[pside]:
            resolved_symbols = {
                exchange: symbol
                for exchange in standard_exchanges
                if (symbol := resolve_if_available(identifier, exchange)) is not None
            }
            ignored_exchanges = {
                exchange
                for exchange, symbol in resolved_symbols.items()
                if symbol in ignored_symbols_by_exchange[exchange]
            }
            if not ignored_exchanges:
                surviving_approved.append(identifier)
                continue
            surviving_exchanges = [
                exchange
                for exchange in standard_exchanges
                if exchange in resolved_symbols and exchange not in ignored_exchanges
            ]
            for exchange in surviving_exchanges:
                symbol = resolved_symbols[exchange]
                market = markets_by_exchange[exchange].get(symbol) or {}
                exact_identifier = str(market.get("id") or symbol)
                surviving_approved.append(f"{exchange}::{exact_identifier}")
        approved_by_side[pside] = list(dict.fromkeys(surviving_approved))


async def _coalesce_resolved_approved_markets(
    config,
    exchanges,
    *,
    quote=None,
    verbose=True,
    prefer_backtest_coin_source_keys=False,
) -> None:
    """Keep one approved dataset key for each resolved venue-market identity."""
    coin_sources = (
        config.get("backtest", {}).get("coin_sources") or {}
        if prefer_backtest_coin_source_keys
        else {}
    )
    standard_exchanges = list(
        dict.fromkeys(
            to_standard_exchange_name(str(exchange))
            for exchange in [*exchanges, *coin_sources.values()]
            if str(exchange).lower() != "fake"
        )
    )
    approved_identifiers = [
        identifier
        for pside in ("long", "short")
        for identifier in config["live"]["approved_coins"][pside]
    ]
    identifiers_by_canonical = defaultdict(set)
    for identifier in approved_identifiers:
        identifiers_by_canonical[heuristic_symbol_to_coin(identifier)].add(identifier)
    needs_resolved_coalescing = bool(coin_sources) or any(
        len(identifiers) > 1 for identifiers in identifiers_by_canonical.values()
    )
    if needs_resolved_coalescing:
        missing_map_exchanges = [
            exchange
            for exchange in standard_exchanges
            if not _load_coin_to_symbol_map(exchange)
        ]
        await asyncio.gather(
            *[
                load_markets(exchange, verbose=False, quote=quote)
                for exchange in missing_map_exchanges
            ]
        )

    source_identities = []
    for source_identifier, exchange in sorted(coin_sources.items()):
        exchange = to_standard_exchange_name(str(exchange))
        try:
            source_symbol = coin_to_symbol(
                source_identifier, exchange, quote=quote, verbose=verbose
            )
        except UnknownMarketIdentifier:
            continue
        source_identities.append(
            (source_identifier, exchange, source_symbol)
        )

    def identity(identifier):
        for _source_identifier, exchange, source_symbol in source_identities:
            try:
                identifier_symbol = coin_to_symbol(
                    identifier, exchange, quote=quote, verbose=verbose
                )
            except (MarketIdentifierExchangeMismatch, UnknownMarketIdentifier):
                continue
            if identifier_symbol == source_symbol:
                return ("source", exchange, source_symbol)
        resolved = []
        for exchange in standard_exchanges:
            try:
                symbol = coin_to_symbol(
                    identifier, exchange, quote=quote, verbose=verbose
                )
            except (MarketIdentifierExchangeMismatch, UnknownMarketIdentifier):
                continue
            resolved.append((exchange, symbol))
        return ("resolved", tuple(resolved)) if resolved else ("raw", identifier)

    representatives = {
        ("source", exchange, source_symbol): source_identifier
        for source_identifier, exchange, source_symbol in source_identities
    }
    for pside in ("long", "short"):
        for identifier in config["live"]["approved_coins"][pside]:
            representatives.setdefault(identity(identifier), identifier)

    for pside in ("long", "short"):
        seen = set()
        coalesced = []
        for identifier in config["live"]["approved_coins"][pside]:
            market_identity = identity(identifier)
            if market_identity in seen:
                continue
            seen.add(market_identity)
            coalesced.append(representatives[market_identity])
        config["live"]["approved_coins"][pside] = coalesced


async def format_approved_ignored_coins(
    config,
    exchanges: [str],
    quote=None,
    verbose=True,
    *,
    prefer_backtest_coin_source_keys=False,
):
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
    ac = normalize_coins_source(approved_source, allow_all=True)
    needs_market_expansion = any(
        _coins_source_side_is_all(ac[pside]) for pside in ("long", "short")
    )

    approved_coins_sorted = None
    if needs_market_expansion:
        approved_coins = set()
        standard_exchanges = []
        for ex in exchanges:
            if str(ex).lower() == "fake":
                approved_coins.update(_load_fake_approved_coins(config, quote=quote))
            else:
                standard_exchanges.append(ex)
        if standard_exchanges:
            marketss = await asyncio.gather(
                *[load_markets(ex, verbose=False, quote=quote) for ex in standard_exchanges]
            )
            marketss = [
                filter_markets(m, ex, quote=quote)[0] for m, ex in zip(marketss, standard_exchanges)
            ]
            approved_coins.update(
                _approved_all_market_identifiers(
                    [
                        (exchange, markets, get_quote(exchange, quote))
                        for exchange, markets in zip(standard_exchanges, marketss)
                    ]
                )
            )
        approved_coins_sorted = sorted([x for x in approved_coins if x])

    config["live"]["approved_coins"] = {}
    for pside in ("long", "short"):
        if _coins_source_side_is_all(ac[pside]):
            config["live"]["approved_coins"][pside] = list(approved_coins_sorted or [])
        else:
            config["live"]["approved_coins"][pside] = _preserve_market_identifiers(ac[pside])

    ignored_source = coin_sources.get("ignored_coins", config.get("live", {}).get("ignored_coins"))
    if ignored_source is None:
        ignored_source = _require_live_value(config, "ignored_coins")
    coin_sources["ignored_coins"] = deepcopy(ignored_source)
    ic = normalize_coins_source(ignored_source, allow_all=False)
    config["live"]["ignored_coins"] = {
        pside: _preserve_market_identifiers(ic[pside]) for pside in ic
    }
    await reject_cross_exchange_market_identifier_collisions(
        [
            identifier
            for field in ("approved_coins", "ignored_coins")
            for side in ("long", "short")
            for identifier in config["live"][field][side]
        ],
        exchanges,
        quote=quote,
        verbose=verbose,
    )
    await _coalesce_resolved_approved_markets(
        config,
        exchanges,
        quote=quote,
        verbose=verbose,
        prefer_backtest_coin_source_keys=prefer_backtest_coin_source_keys,
    )
    await _remove_resolved_ignored_markets(
        config,
        exchanges,
        quote=quote,
        verbose=verbose,
        include_backtest_coin_source_venues=prefer_backtest_coin_source_keys,
    )

    approved_diff = _diff_snapshot(before_approved, config["live"]["approved_coins"])
    ignored_diff = _diff_snapshot(before_ignored, config["live"]["ignored_coins"])
    sources_diff = _diff_snapshot(before_sources, config.get("_coins_sources", {}))
    if approved_diff or ignored_diff or sources_diff:
        from config_transform import record_transform

        details = {"exchanges": list(exchanges)}
        if approved_diff:
            details["approved_coins"] = approved_diff
        if ignored_diff:
            details["ignored_coins"] = ignored_diff
        if sources_diff:
            details["coin_sources"] = sources_diff
        record_transform(config, "format_approved_ignored_coins", details)


def _coins_source_side_is_all(value) -> bool:
    return isinstance(value, list) and len(value) == 1 and str(value[0]).strip().lower() == "all"


def normalize_coins_source(src, *, allow_all: bool = True):
    """
    Always return: {'long': [symbols…], 'short': [symbols…]}
    – Handles:
        • direct coin lists or comma-separated strings
        • lists/tuples containing paths or strings
        • dicts with 'long' / 'short' keys whose values may themselves
          be strings, lists, or paths to external lists
        • explicit 'all' sentinel for approved coins
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

    def _parse_jsonish(raw: str):
        raw = raw.strip()
        if not raw:
            return None
        if raw[0] not in "[{" or raw[-1] not in "]}":
            return None
        parsed = None
        try:
            import hjson

            parsed = hjson.loads(raw)
        except Exception:
            parsed = None
        if parsed is None:
            try:
                import json

                parsed = json.loads(raw)
            except Exception:
                parsed = None
        return parsed

    def _maybe_parse_jsonish(val):
        if isinstance(val, str):
            parsed = _parse_jsonish(val)
            return parsed if parsed is not None else val
        if isinstance(val, (list, tuple)) and val and all(isinstance(x, str) for x in val):
            joined = ",".join(x.strip() for x in val if x.strip())
            parsed = _parse_jsonish(joined)
            return parsed if parsed is not None else val
        return val

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
        value = _maybe_parse_jsonish(value)

        if isinstance(value, dict) and set(value).issubset({"long", "short"}):
            value = value.get(side, [])

        if value in (None, "", [], (), {}, {"long": [], "short": []}):
            return []

        # guarantee a sensible sequence for _expand
        if not isinstance(value, (list, tuple)):
            value = [value]

        expanded = _expand(value)
        if not expanded:
            return []
        if allow_all and len(expanded) == 1 and expanded[0].strip().lower() == "all":
            return ["all"]
        return expanded

    # --------------------------------------------------------------------- #
    #  Main logic                                                           #
    # --------------------------------------------------------------------- #
    src = _load_if_file(src)  # try to load *src* itself
    src = _maybe_parse_jsonish(src)

    # Case 1 – already a dict with 'long' & 'short' keys
    if isinstance(src, dict):
        if not src:
            return {"long": [], "short": []}
        if set(src).issubset({"long", "short"}):
            return {
                "long": _normalize_side(src.get("long", []), "long"),
                "short": _normalize_side(src.get("short", []), "short"),
            }

    if src in (None, "", [], (), {}):
        return {"long": [], "short": []}

    if allow_all:
        global_tokens = _normalize_side(src, "long")
        if _coins_source_side_is_all(global_tokens):
            return {"long": ["all"], "short": ["all"]}

    # Case 1 – already a dict with 'long' / 'short' keys (including partial)
    if isinstance(src, dict) and set(src).issubset({"long", "short"}):
        return {
            "long": _normalize_side(src.get("long", []), "long"),
            "short": _normalize_side(src.get("short", []), "short"),
        }

    # Case 2 – anything else is treated the same for both sides
    return {
        "long": global_tokens if allow_all else _normalize_side(src, "long"),
        "short": global_tokens if allow_all else _normalize_side(src, "short"),
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
    """Return the earliest OHLCV candle for a backward-paginated native market.

    Native Bitget and Bitunix clients page backwards using
    ``params={"until": ms}``, where an empty response indicates that ``until``
    predates the instrument listing. We leverage that behaviour to binary-search
    over monthly candles and then refine the result with a daily fetch. The
    returned value is the first full candle
    ``[timestamp, open, high, low, close, volume]`` if available, else ``None``.
    """

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


def deep_get(d, key_path, *args):
    """
    Retrieves a value from a nested dict using dot notation.
    Handles keys that may contain dots via greedy matching.
    """
    # Check if a default was provided via *args
    has_default = len(args) > 0
    default = args[0] if has_default else None

    segments = key_path.split(".")
    current = d

    i = 0
    while i < len(segments):
        found = False

        # Greedy look-ahead: try to find the longest matching key
        for j in range(i + 1, len(segments) + 1):
            candidate_key = ".".join(segments[i:j])

            if isinstance(current, dict) and candidate_key in current:
                current = current[candidate_key]
                i = j  # Jump the pointer forward
                found = True
                break

        if not found:
            if has_default:
                return default
            raise KeyError(f"Path segment '{segments[i]}' not found in '{key_path}'")

    return current
