"""
CandlestickManager: lightweight 1m OHLCV manager with gap standardization.

This module provides a minimal, self-contained implementation tailored to the
unit tests in tests/test_candlestick_manager.py while following the requested
API and data format. It focuses on:

- UTC millisecond timestamps and structured NumPy dtype for candles
- Gap standardization with synthesized zero-candles (not persisted)
- Inclusive range selection with minute alignment
- Latest EMA for close/volume/log range computed lazily from cached candles
- Shard saving with atomic write and index.json maintenance

Example
-------
>>> from candlestick_manager import CandlestickManager, ONE_MIN_MS
>>> cm = CandlestickManager(exchange=None, exchange_name="demo")
>>> # Preload some candles directly into cache (ts, o, h, l, c, bv)
>>> import time, numpy as np
>>> now = int(time.time() * 1000)
>>> base = _floor_minute(now) - 5 * ONE_MIN_MS
>>> arr = np.array([
...     (base + i * ONE_MIN_MS, 1+i, 1+i, 1+i, 1+i, float(i)) for i in range(5)
... ], dtype=CANDLE_DTYPE)
>>> cm._cache["FOO/USDT"] = arr
>>> import asyncio
>>> asyncio.run(cm.get_latest_ema_close("FOO/USDT", span=5))
1.0
"""

from __future__ import annotations

import asyncio
import calendar
import hashlib
import heapq
import inspect
import json
import logging
import math
import os
import re
import shutil
import sys

import time
import zlib
import atexit
from contextlib import asynccontextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TypedDict,
    TYPE_CHECKING,
)
import threading
from collections import OrderedDict

if TYPE_CHECKING:
    import aiohttp

import warnings
import time
from datetime import datetime, timezone

import numpy as np
import portalocker  # type: ignore

try:
    import passivbot_rust as pbr
except ImportError:  # pragma: no cover - editable source-only tooling fallback
    pbr = None

_RUST_EMA_LAST = getattr(pbr, "ema_last", None) if pbr is not None else None

from legacy_data_migrator import (
    standardize_cache_directories,
    migrate_legacy_data_all_on_init,
    merge_duplicate_symbol_directories,
    normalize_ccxt_volume_to_base,
)
from live.diagnostic_safety import bounded_exception_type
from utils import (
    FIRST_OHLCV_TIMESTAMPS_CACHE_VERSION,
    exchange_name_aliases,
    symbol_to_coin,
)

# Suppress portalocker's "timeout has no effect in blocking mode" warning
warnings.filterwarnings(
    "ignore", message="timeout has no effect in blocking mode", module="portalocker"
)

# ----- Constants and dtypes -----

ONE_MIN_MS = 60_000

_LOCK_TIMEOUT_SECONDS = 10.0
_REMOTE_FETCH_ERROR_TYPE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,79}")
_REMOTE_FETCH_SENSITIVE_ERROR_TYPE_RE = re.compile(
    r"(?i)(?:api_?key|apikey|authorization|cookie|passphrase|password|private_?key|"
    r"privatekey|secret|signature|token|wallet_?address|walletaddress)"
)
_REMOTE_FETCH_PARAM_KEYS_MAX = 32


def _bounded_remote_fetch_param_keys(value: Any) -> list[str]:
    if isinstance(value, dict):
        values = value.keys()
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        return []
    keys = []
    for key in values:
        if isinstance(key, str) and 0 < len(key) <= 80 and key.isascii():
            keys.append(key)
    return sorted(set(keys))[:_REMOTE_FETCH_PARAM_KEYS_MAX]


def _bounded_remote_fetch_error_type(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not _REMOTE_FETCH_ERROR_TYPE_RE.fullmatch(value)
        or _REMOTE_FETCH_SENSITIVE_ERROR_TYPE_RE.search(value)
    ):
        return "Error"
    return value


def _remote_fetch_url_hash(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sanitize_remote_fetch_diagnostic(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return the callback-safe projection of a remote-fetch diagnostic payload."""
    data = dict(payload)
    data.pop("error", None)
    data.pop("error_repr", None)

    raw_url = data.pop("url", None)
    url_hash = _remote_fetch_url_hash(raw_url) if raw_url is not None else None
    if url_hash is None:
        candidate_hash = data.get("url_hash")
        if isinstance(candidate_hash, str) and re.fullmatch(r"[0-9a-f]{64}", candidate_hash):
            url_hash = candidate_hash
    if url_hash is not None:
        data["url_hash"] = url_hash
    else:
        data.pop("url_hash", None)

    raw_params = data.pop("params", None)
    param_keys = _bounded_remote_fetch_param_keys(
        raw_params if isinstance(raw_params, dict) else data.pop("param_keys", None)
    )
    if param_keys:
        data["param_keys"] = param_keys
    else:
        data.pop("param_keys", None)

    if "error_type" in data:
        data["error_type"] = _bounded_remote_fetch_error_type(data["error_type"])
    return data


class OhlcvFetchError(RuntimeError):
    """Raised when a remote OHLCV fetch exhausts retries without a successful response."""


class OhlcvTerminalEmptyPage(OhlcvFetchError):
    """Raised when pagination reaches a terminal empty page after fetching rows."""

    def __init__(
        self,
        message: str,
        *,
        partial_rows: np.ndarray,
        terminal_start_ts: int,
        requested_end_ts: int,
        pages: int,
    ) -> None:
        super().__init__(message)
        self.partial_rows = partial_rows
        self.terminal_start_ts = int(terminal_start_ts)
        self.requested_end_ts = int(requested_end_ts)
        self.pages = int(pages)


_LOCK_STALE_SECONDS = 180.0


def _log_symbol(symbol: Any) -> str:
    """Return compact symbol labels for operator-facing candle logs."""
    if symbol is None:
        return "unknown"
    sym = str(symbol)
    return symbol_to_coin(sym, verbose=False) or sym


_LOCK_BACKOFF_INITIAL = 0.1
_LOCK_BACKOFF_MAX = 2.0
_GATEIO_RECENT_1M_LIMIT_CANDLES = 9_990
_WEEX_RECENT_OHLCV_LIMIT = 1_000
_WEEX_HISTORICAL_OHLCV_LIMIT = 100
DEFAULT_NATIVE_SPARSE_GAP_TOLERANCE_MINUTES = 120.0


def _ensure_legacy_ohlcv_cache_base(cache_dir: str) -> str:
    """Create the legacy daily-shard cache root, repairing stale symlinks left by v2 migration."""
    cache_base = Path(cache_dir) / "ohlcv"
    try:
        cache_base.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        if cache_base.is_symlink() and not cache_base.exists():
            target = os.readlink(cache_base)
            cache_base.unlink()
            cache_base.mkdir(parents=True, exist_ok=True)
            logging.info(
                "removed dangling legacy OHLCV cache symlink %s -> %s; created local directory",
                cache_base,
                target,
            )
        else:
            raise
    return str(cache_base)

# See: https://github.com/enarjord/passivbot/issues/547
# True if running on Windows (used for file/path compatible names)
windows_compatibility = (
    sys.platform.startswith("win") or os.environ.get("WINDOWS_COMPATIBILITY") == "1"
)


@dataclass
class _LockRecord:
    lock: portalocker.Lock
    count: int
    acquired_at: float
    path: str
    owner_task: Optional[asyncio.Task[Any]]


class GapEntry(TypedDict, total=False):
    """Enhanced gap metadata stored in index.json known_gaps."""

    start_ts: int  # Gap start timestamp (ms)
    end_ts: int  # Gap end timestamp (ms)
    retry_count: int  # Number of fetch attempts (max 3 before marking persistent)
    reason: str  # "auto_detected", "exchange_downtime", "no_archive", "fetch_failed", "manual", "no_trades"
    added_at: int  # Timestamp when gap was first detected (ms)
    last_retry_at: int  # Timestamp of the latest remote attempt for this gap (ms)
    last_contextual_retry_at: int  # Latest KuCoin boundary-proof attempt (ms)


# Maximum fetch attempts before marking gap as persistent
_GAP_MAX_RETRIES = 3
_GAP_PERSISTENT_RETRY_MS = 7 * 24 * 60 * 60 * 1000
_HYPERLIQUID_RECENT_GAP_HORIZON_MS = 2 * 60 * 60 * 1000
_HYPERLIQUID_RECENT_GAP_MAX_SPAN_MS = 2 * 60 * 60 * 1000
_HYPERLIQUID_RECENT_GAP_RETRY_MS = 5 * 60 * 1000
_HYPERLIQUID_RECENT_PERSISTENT_GAP_RETRY_MS = 15 * 60 * 1000

# Valid gap reasons
GAP_REASON_AUTO = "auto_detected"
GAP_REASON_EXCHANGE_DOWNTIME = "exchange_downtime"
GAP_REASON_NO_ARCHIVE = "no_archive"
GAP_REASON_FETCH_FAILED = "fetch_failed"
GAP_REASON_MANUAL = "manual"
GAP_REASON_NO_TRADES = "no_trades"
_GAP_NON_EXPIRING_REASONS = frozenset(
    {"pre_inception", GAP_REASON_NO_ARCHIVE, GAP_REASON_MANUAL, GAP_REASON_NO_TRADES}
)


_FIRST_OHLCV_EXCHANGE_CACHE_ALIASES = {
    "binance": "binanceusdm",
}


CANDLE_DTYPE = np.dtype(
    [
        ("ts", "int64"),
        ("o", "float32"),
        ("h", "float32"),
        ("l", "float32"),
        ("c", "float32"),
        ("bv", "float32"),
    ]
)

CANDLE_RESOLUTION_LADDER: tuple[tuple[str, int], ...] = (
    ("1m", 1),
    ("5m", 5),
    ("15m", 15),
    ("1h", 60),
)


@dataclass(frozen=True)
class CandleResolutionResult:
    """Candles assembled from the finest available historical resolutions."""

    candles: np.ndarray
    source_counts: Dict[str, int]
    failures: Dict[str, Exception]


_LIVE_WS_OBSERVATION_RETENTION_MS = 2 * 60 * ONE_MIN_MS

EMA_SERIES_DTYPE = np.dtype(
    [
        ("ts", "int64"),
        ("ema", "float32"),
    ]
)


# ----- Utilities -----


def _linear_interpolate(value0: float, value1: float, ratio: float) -> float:
    return float(value0 + (value1 - value0) * ratio)


def ohlcv_xm_to_1m(candle: np.void, minutes: int) -> np.ndarray:
    """Expand one higher-timeframe OHLCV candle into deterministic synthetic 1m candles."""
    if minutes <= 0:
        raise ValueError(f"minutes must be > 0, got {minutes}")

    ts = int(candle["ts"])
    o = float(candle["o"])
    h = float(candle["h"])
    l = float(candle["l"])
    c = float(candle["c"])
    bv = float(candle["bv"])

    if not all(math.isfinite(x) for x in (o, h, l, c, bv)):
        raise ValueError("all OHLCV values must be finite")
    if h < l:
        h, l = l, h
    o = min(max(o, l), h)
    c = min(max(c, l), h)

    out = np.zeros(minutes, dtype=CANDLE_DTYPE)
    out["ts"] = np.arange(ts, ts + minutes * ONE_MIN_MS, ONE_MIN_MS, dtype=np.int64)
    out["bv"] = float(bv / minutes)

    last_idx = minutes - 1
    if last_idx == 0:
        out[0]["o"] = o
        out[0]["h"] = h
        out[0]["l"] = l
        out[0]["c"] = c
        return out

    pivot_a = min(last_idx, max(1, minutes // 3))
    pivot_b = min(last_idx, max(pivot_a + 1, (2 * minutes) // 3))

    if c >= o:
        waypoints = [(0, o), (pivot_a, l), (pivot_b, h), (last_idx, c)]
        low_idx = pivot_a
        high_idx = pivot_b
    else:
        waypoints = [(0, o), (pivot_a, h), (pivot_b, l), (last_idx, c)]
        high_idx = pivot_a
        low_idx = pivot_b

    deduped = [waypoints[0]]
    for idx, value in waypoints[1:]:
        if idx > deduped[-1][0]:
            deduped.append((idx, value))
        else:
            deduped[-1] = (idx, value)

    close_path = np.empty(minutes, dtype=np.float64)
    close_path[0] = o
    for (i0, v0), (i1, v1) in zip(deduped, deduped[1:]):
        span = max(1, i1 - i0)
        for minute_idx in range(i0, i1 + 1):
            ratio = 0.0 if i1 == i0 else (minute_idx - i0) / span
            close_path[minute_idx] = min(max(_linear_interpolate(v0, v1, ratio), l), h)

    prev_close = o
    for minute_idx in range(minutes):
        minute_open = prev_close
        minute_close = float(close_path[minute_idx])
        minute_high = max(minute_open, minute_close)
        minute_low = min(minute_open, minute_close)
        if minute_idx == high_idx:
            minute_high = max(minute_high, h)
        if minute_idx == low_idx:
            minute_low = min(minute_low, l)
        out[minute_idx]["o"] = minute_open
        out[minute_idx]["h"] = minute_high
        out[minute_idx]["l"] = minute_low
        out[minute_idx]["c"] = minute_close
        prev_close = minute_close

    return out


def ohlcv_5m_to_1m(candle: np.void) -> np.ndarray:
    return ohlcv_xm_to_1m(candle, 5)


def ohlcv_15m_to_1m(candle: np.void) -> np.ndarray:
    return ohlcv_xm_to_1m(candle, 15)


def synthesize_1m_from_higher_tf(candles: np.ndarray, tf_minutes: int) -> np.ndarray:
    """Expand a higher-timeframe candle array into synthetic 1m OHLCV candles."""
    arr = _ensure_dtype(candles)
    if arr.size == 0:
        return np.empty((0,), dtype=CANDLE_DTYPE)
    if tf_minutes <= 1:
        raise ValueError(f"tf_minutes must be > 1, got {tf_minutes}")
    expanded = [ohlcv_xm_to_1m(row, tf_minutes) for row in arr]
    if not expanded:
        return np.empty((0,), dtype=CANDLE_DTYPE)
    return np.sort(np.concatenate(expanded), order="ts")


async def fetch_candles_with_resolution_ladder(
    fetch_candles: Callable[..., Awaitable[np.ndarray]],
    *,
    start_ts: int,
    end_ts: int,
    supported_timeframes: Optional[Iterable[str]] = None,
) -> CandleResolutionResult:
    """Fetch exact 1m candles, then cover only older leading history coarsely.

    The first available 1m candle is the precision boundary. Higher-timeframe
    candles may supply minutes before that boundary only when their full bucket
    ends there or earlier; they never patch gaps at or after it. Within the
    older prefix, the finest successful source wins.
    """
    start_minute = _floor_minute(start_ts)
    end_minute = _floor_minute(end_ts)
    if end_minute < start_minute:
        return CandleResolutionResult(
            candles=np.empty((0,), dtype=CANDLE_DTYPE),
            source_counts={},
            failures={},
        )

    supported = (
        {str(timeframe).lower() for timeframe in supported_timeframes}
        if supported_timeframes is not None
        else None
    )
    rows_by_ts: Dict[int, np.void] = {}
    sources_by_ts: Dict[int, str] = {}
    failures: Dict[str, Exception] = {}
    precision_boundary = end_minute + ONE_MIN_MS

    for index, (timeframe, tf_minutes) in enumerate(CANDLE_RESOLUTION_LADDER):
        timeframe = str(timeframe).lower()
        if supported is not None and timeframe not in supported:
            continue
        fetch_end = end_minute if index == 0 else precision_boundary - ONE_MIN_MS
        if fetch_end < start_minute:
            break
        try:
            fetched = _ensure_dtype(
                await fetch_candles(
                    timeframe=timeframe,
                    start_ts=start_minute,
                    end_ts=fetch_end,
                )
            )
        except Exception as exc:
            failures[timeframe] = exc
            continue

        if fetched.size == 0:
            continue
        if tf_minutes == 1:
            candidates = fetched
        else:
            period_ms = tf_minutes * ONE_MIN_MS
            complete_before_boundary = (
                fetched["ts"].astype(np.int64) + period_ms <= precision_boundary
            )
            candidates = synthesize_1m_from_higher_tf(
                fetched[complete_before_boundary], tf_minutes
            )
        if index == 0:
            exact_timestamps = [
                int(row["ts"])
                for row in candidates
                if start_minute <= int(row["ts"]) <= end_minute
            ]
            if exact_timestamps:
                precision_boundary = min(exact_timestamps)

        for row in candidates:
            ts = int(row["ts"])
            if ts < start_minute or ts > end_minute:
                continue
            if index > 0 and ts >= precision_boundary:
                continue
            if ts in rows_by_ts:
                continue
            rows_by_ts[ts] = row
            sources_by_ts[ts] = timeframe

        if precision_boundary <= start_minute:
            break
        expected_prefix = range(start_minute, precision_boundary, ONE_MIN_MS)
        if all(ts in rows_by_ts for ts in expected_prefix):
            break

    if not rows_by_ts:
        candles = np.empty((0,), dtype=CANDLE_DTYPE)
    else:
        candles = np.empty((len(rows_by_ts),), dtype=CANDLE_DTYPE)
        for index, ts in enumerate(sorted(rows_by_ts)):
            candles[index] = rows_by_ts[ts]
    source_counts: Dict[str, int] = {}
    for source in sources_by_ts.values():
        source_counts[source] = source_counts.get(source, 0) + 1
    return CandleResolutionResult(
        candles=candles,
        source_counts=source_counts,
        failures=failures,
    )


def get_caller_name(depth: int = 2, logger: Optional[logging.Logger] = None) -> str:
    """Return a more useful origin for debug logs.

    Heuristics:
    - Skip CandlestickManager frames and common wrappers ("one", "<listcomp>", asyncio internals)
    - Prefer frames from a Passivbot instance method if present (module contains "passivbot")
    - Otherwise return the first non-wrapper frame as module.Class.func or module.func
    """

    def frame_to_name(fr) -> str:
        try:
            func = getattr(fr.f_code, "co_name", "unknown")
            mod = fr.f_globals.get("__name__", None)
            cls = None
            if "self" in fr.f_locals and fr.f_locals["self"] is not None:
                cls = type(fr.f_locals["self"]).__name__
            elif "cls" in fr.f_locals and fr.f_locals["cls"] is not None:
                cls = getattr(fr.f_locals["cls"], "__name__", None)
            parts = []
            if isinstance(mod, str) and mod:
                parts.append(mod)
            if isinstance(cls, str) and cls:
                parts.append(cls)
            if isinstance(func, str) and func:
                parts.append(func)
            return ".".join(parts) if parts else "unknown"
        except Exception:
            return "unknown"

    frame = inspect.currentframe()
    target = frame
    fallback_name = "unknown"
    try:
        # Initial hop
        for _ in range(max(0, int(depth))):
            if target is None:
                break
            target = target.f_back  # type: ignore[attr-defined]
        if target is not None:
            fallback_name = frame_to_name(target)

        # Walk up to find a meaningful caller
        cur = target
        preferred: Optional[str] = None
        for _ in range(20):  # safety cap
            if cur is None:
                break
            try:
                slf = cur.f_locals.get("self") if hasattr(cur, "f_locals") else None
                is_cm = slf is not None and type(slf).__name__ == "CandlestickManager"
            except Exception:
                is_cm = False
            func = getattr(getattr(cur, "f_code", None), "co_name", "")
            mod = None
            try:
                mod = cur.f_globals.get("__name__")
            except Exception:
                mod = None

            # Skip common wrappers and asyncio internals
            skip_names = {
                "one",
                "<listcomp>",
                "<dictcomp>",
                "<lambda>",
                "_run",
                "gather",
                "create_task",
            }
            is_asyncio = isinstance(mod, str) and (
                mod.startswith("asyncio.") or mod == "asyncio.events"
            )
            if not is_cm and func not in skip_names and not is_asyncio:
                name = frame_to_name(cur)
                if isinstance(mod, str) and "passivbot" in mod and name and name != "unknown":
                    # Prefer first passivbot frame
                    preferred = name
                    break
                if name and name != "unknown" and preferred is None:
                    preferred = name
            cur = cur.f_back  # type: ignore[attr-defined]
    finally:
        try:
            del frame
        except Exception:
            pass
        try:
            del target  # type: ignore[name-defined]
        except Exception:
            pass
    return preferred or fallback_name


def _utc_now_ms() -> int:
    return int(time.time() * 1000)


def _floor_minute(ms: int) -> int:
    return (int(ms) // ONE_MIN_MS) * ONE_MIN_MS


def _ensure_dtype(a: np.ndarray) -> np.ndarray:
    if a.dtype != CANDLE_DTYPE:
        return a.astype(CANDLE_DTYPE, copy=False)
    return a


def _ts_index(a: np.ndarray) -> np.ndarray:
    """Return sorted ts column as plain int64 array."""
    if a.size == 0:
        return np.empty((0,), dtype=np.int64)
    return np.asarray(a["ts"], dtype=np.int64)


def _sanitize_symbol(symbol: str) -> str:
    sanitized = symbol.replace("/", "_")
    # See: https://github.com/enarjord/passivbot/issues/547
    # If running under "Windows Compatibility" mode,
    # also replace ':' with '_' to ensure compatibility with Windows file naming restrictions.
    if windows_compatibility:
        sanitized = sanitized.replace(":", "_")
    return sanitized


def _quarantine_gateio_cache_if_stale(cache_base: str, cutoff_date: str) -> None:
    """
    Move gateio cache to a timestamped backup if any shard predates cutoff_date.
    """
    try:
        cutoff = datetime.strptime(cutoff_date, "%Y-%m-%d").date()
    except Exception:
        logging.warning(
            "Invalid GATEIO_CACHE_CUTOFF_DATE=%r; skipping gateio cache check", cutoff_date
        )
        return

    gateio_root = os.path.join(cache_base, "gateio")
    if not os.path.isdir(gateio_root):
        return

    tf_root = os.path.join(gateio_root, "1m")
    if not os.path.isdir(tf_root):
        return

    for sym in os.listdir(tf_root):
        sym_dir = os.path.join(tf_root, sym)
        if not os.path.isdir(sym_dir):
            continue
        for fname in os.listdir(sym_dir):
            if not fname.endswith(".npy"):
                continue
            try:
                day = datetime.strptime(fname[:10], "%Y-%m-%d").date()
            except Exception:
                continue
            if day < cutoff:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                backup = f"{gateio_root}_backup_{stamp}"
                logging.warning(
                    "GateIO cache has shards before %s; moving %s -> %s. "
                    "Delete backup after confirming volumes are correct.",
                    cutoff_date,
                    gateio_root,
                    backup,
                )
                try:
                    os.rename(gateio_root, backup)
                except Exception as exc:
                    logging.error(
                        "Failed to move GateIO cache to backup error_type=%s cache_base=%s backup=%s",
                        bounded_exception_type(exc),
                        gateio_root,
                        backup,
                    )
                return


def _looks_like_daily_shard_filename(name: str) -> bool:
    if not isinstance(name, str) or not name.endswith(".npy"):
        return False
    stem = name[:-4]
    if len(stem) != 10 or stem[4] != "-" or stem[7] != "-":
        return False
    try:
        datetime.strptime(stem, "%Y-%m-%d")
    except Exception:
        return False
    return True


def _quarantine_root_level_timeframe_debris(cache_base: str) -> int:
    """
    Quarantine invalid files found directly under exchange/timeframe roots.

    Valid OHLCV layout is:
    `{cache_base}/{exchange}/{timeframe}/{symbol}/YYYY-MM-DD.npy`

    Any daily shard files or index.json files found directly under
    `{cache_base}/{exchange}/{timeframe}` are debris from older/corrupt layouts and
    should not remain in place.
    """
    root = Path(cache_base)
    if not root.is_dir():
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    moved = 0

    for exchange_dir in root.iterdir():
        if not exchange_dir.is_dir() or exchange_dir.name.startswith("."):
            continue
        if exchange_dir.name.startswith("_"):
            continue

        for tf_dir in exchange_dir.iterdir():
            if not tf_dir.is_dir():
                continue

            debris: List[Path] = []
            for child in tf_dir.iterdir():
                if not child.is_file():
                    continue
                if child.name == "index.json" or _looks_like_daily_shard_filename(child.name):
                    debris.append(child)

            if not debris:
                continue

            quarantine_dir = (
                root / "_quarantine_root_level" / stamp / exchange_dir.name / tf_dir.name
            )
            quarantine_dir.mkdir(parents=True, exist_ok=True)

            for child in debris:
                shutil.move(str(child), str(quarantine_dir / child.name))
                moved += 1

            logging.warning(
                "Quarantined %d invalid root-level OHLCV cache artifact(s) from %s -> %s",
                len(debris),
                tf_dir,
                quarantine_dir,
            )

    return moved


# Parse timeframe string like '1m','5m','1h','1d' to milliseconds.
# Falls back to ONE_MIN_MS on invalid input. Seconds are rounded down to minutes.
def _tf_to_ms(s: Optional[str]) -> int:
    if not s:
        return ONE_MIN_MS
    try:
        st = s.strip().lower()
    except Exception:
        return ONE_MIN_MS
    import re

    m = re.fullmatch(r"(\d+)([smhd])", st)
    if not m:
        return ONE_MIN_MS
    n, unit = int(m.group(1)), m.group(2)
    if unit == "s":
        return max(ONE_MIN_MS, (n // 60) * ONE_MIN_MS)
    if unit == "m":
        return n * ONE_MIN_MS
    if unit == "h":
        return n * 60 * ONE_MIN_MS
    if unit == "d":
        return n * 1440 * ONE_MIN_MS
    return ONE_MIN_MS


def candle_range_has_full_coverage(
    arr: np.ndarray,
    start_ts: int,
    end_ts: int,
    *,
    timeframe: Optional[str] = None,
    tf: Optional[str] = None,
) -> bool:
    """Return whether every aligned candle in the requested range exists."""
    period_ms = _tf_to_ms(timeframe if timeframe is not None else tf)
    start = (int(start_ts) // period_ms) * period_ms
    end = (int(end_ts) // period_ms) * period_ms
    if end < start:
        return True
    if not isinstance(arr, np.ndarray) or arr.size == 0:
        return False
    try:
        timestamps = np.asarray(arr["ts"], dtype=np.int64)
    except (KeyError, TypeError, ValueError):
        return False
    timestamps = np.sort(timestamps[(timestamps >= start) & (timestamps <= end)])
    expected_count = int((end - start) // period_ms) + 1
    if timestamps.size != expected_count:
        return False
    if int(timestamps[0]) != start or int(timestamps[-1]) != end:
        return False
    return expected_count <= 1 or bool(np.all(np.diff(timestamps) == period_ms))


# ----- CandlestickManager -----


class CandlestickManager:
    """Manage 1m OHLCV candles with simple cache and gap standardization.

    Parameters
    ----------
    exchange : Any
        CCXT exchange instance or None. Tests pass None, so network fetch is skipped.
    exchange_name : str
        Name of the exchange used for cache directory layout.
    cache_dir : str
        Root directory for on-disk cache. Default "caches".
    default_window_candles : int
        Default window used when start_ts is not provided.
    overlap_candles : int
        Overlap applied when refreshing from network (not exercised in tests).
    max_memory_candles_per_symbol : int
        Max number of 1m candles in RAM per symbol (rolling window).
    max_disk_candles_per_symbol_per_tf : int
        Max total candles per symbol+timeframe on disk (oldest shards pruned).
    debug : int | bool
        Logging verbosity (0=warnings, 1=network info, 2=debug, 3=trace).
    """

    # Many helpers accept both `timeframe=` and the concise `tf=` alias.  The alias keeps
    # existing call sites terse while still advertising the more descriptive name.

    def __init__(
        self,
        exchange=None,
        exchange_name: str = "unknown",
        *,
        cache_dir: str = "caches",
        default_window_candles: int = 100,
        overlap_candles: int = 30,
        # Retention knobs (candle-count based):
        max_memory_candles_per_symbol: int = 200_000,
        max_disk_candles_per_symbol_per_tf: int = 2_000_000,
        debug: int | bool = False,
        # Optional progress logging (INFO, throttled). 0 disables, 30.0 recommended.
        progress_log_interval_seconds: float = 10.0,
        # Optional callback invoked for every external (network) fetch attempt.
        remote_fetch_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        # Optional global concurrency limiter for remote ccxt calls
        max_concurrent_requests: int | None = None,
        # Optional minimum spacing between ccxt OHLCV calls from this manager.
        remote_fetch_min_interval_ms: float | None = None,
        lock_timeout_seconds: float | None = None,
        gap_tolerance_ohlcvs_minutes: float = DEFAULT_NATIVE_SPARSE_GAP_TOLERANCE_MINUTES,
        provisional_internal_gap_tolerance_minutes: float = 10.0,
        # Archive fetching: if False, only use ccxt REST API even if archives are available.
        # Useful for live bots where archives may timeout; backtester enables by default.
        archive_enabled: bool = True,
        # Optional list of symbols to log per-page OHLCV ranges (debugging pagination).
        page_debug_symbols: Optional[Iterable[str]] = None,
        # Optional live shutdown hook. If it returns True, in-flight warmup/fetch loops abort.
        stop_requested_callback: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.exchange = exchange
        # If no explicit exchange_name provided, infer from ccxt instance id
        if (not exchange_name or exchange_name == "unknown") and getattr(exchange, "id", None):
            self.exchange_name = str(getattr(exchange, "id"))
        else:
            self.exchange_name = exchange_name
        # Normalize ccxt IDs to standard cache names (e.g. "binanceusdm" -> "binance")
        _en = self.exchange_name.lower()
        for _suffix in ("usdm", "futures"):
            if _en.endswith(_suffix):
                self.exchange_name = _en[: -len(_suffix)]
                break
        self.cache_dir = cache_dir
        self.default_window_candles = int(default_window_candles)
        self.overlap_candles = int(overlap_candles)
        self.max_memory_candles_per_symbol = int(max_memory_candles_per_symbol)
        self.max_disk_candles_per_symbol_per_tf = int(max_disk_candles_per_symbol_per_tf)
        self.gap_tolerance_ohlcvs_minutes = max(
            0.0, float(gap_tolerance_ohlcvs_minutes)
        )
        self.provisional_internal_gap_tolerance_minutes = max(
            0.0, float(provisional_internal_gap_tolerance_minutes)
        )
        # Archive fetching: if False, only use ccxt REST API
        self.archive_enabled = bool(archive_enabled)
        # Debug levels: 0=warnings, 1=network summaries, 2=debug, 3=trace/firehose
        try:
            dbg = int(float(debug))
        except Exception:
            dbg = 2 if bool(debug) else 0
        self.debug_level = max(0, min(int(dbg), 3))
        try:
            self._progress_log_interval_seconds = max(0.0, float(progress_log_interval_seconds))
        except Exception:
            self._progress_log_interval_seconds = 0.0
        self._progress_last_log: Dict[Tuple[str, str, str], float] = {}
        self._skipped_trailing_gap_summary: Dict[Tuple[str, str], Dict[str, int]] = {}
        self._skipped_trailing_gap_summary_last_log: float = time.monotonic()
        self._warning_last_log: Dict[str, float] = {}  # throttle repeated warnings
        self._warning_throttle_seconds: float = 300.0  # 5 minutes between repeated warnings
        self._persist_batch_observer: Optional[
            Callable[[str, str, np.ndarray], None]
        ] = None
        self._disk_load_observer: Optional[Callable[[Dict[str, Any]], None]] = None
        # Summary tracking for strict gap warnings (logged once per 15 min instead of per-event)
        self._strict_gaps_summary: Dict[str, int] = {}  # symbol -> missing count
        self._strict_gaps_summary_last_log: float = 0.0
        self._strict_gaps_summary_interval: float = 900.0  # 15 minutes
        self._remote_fetch_callback = remote_fetch_callback
        self._stop_requested_callback = stop_requested_callback
        self._now_ms_callback: Optional[Callable[[], int]] = None
        # Cache of legacy shard paths per (exchange, symbol, tf)
        self._legacy_shard_paths_cache: Dict[Tuple[str, str, str], Dict[str, str]] = {}
        # Cache for legacy day quality decisions: (symbol, tf, date_key) -> legacy_is_complete
        self._legacy_day_quality_cache: Dict[Tuple[str, str, str], bool] = {}
        # Cache of primary shard paths per (symbol, tf) - avoids redundant glob scans
        self._shard_paths_cache: Dict[Tuple[str, str], Dict[str, str]] = {}

        self._cache: Dict[str, np.ndarray] = {}
        # Recent public websocket observations are RAM-only provenance used to
        # distinguish a fresh/changed CCXT update from rows repeated out of
        # CCXT Pro's sliding cache. Proven-final rows themselves are persisted
        # through the canonical 1m candle path.
        self._live_ws_ohlcv_observations: Dict[
            str, Dict[int, Tuple[Tuple[float, ...], bool]]
        ] = {}
        self._index: Dict[str, dict] = {}
        self._index_mtime: Dict[
            str, Optional[Tuple[int, int, int]]
        ] = {}
        # Cache for EMA computations: per symbol -> {(metric, span, tf): (value, end_ts, computed_at_ms)}
        self._ema_cache: Dict[str, Dict[Tuple[str, int, str], Tuple[float, int, int]]] = {}
        # Non-persistent provenance for EMA values computed with later-bounded
        # internal zero-volume continuity rows. Keys mirror the EMA cache.
        self._ema_provisional_internal_gap_context: Dict[
            str, Dict[Tuple[str, float, str], Dict[str, int]]
        ] = {}
        # Compatibility attribute retained for older monitor/tool code.  The
        # candlestick manager no longer owns current in-progress prices.
        self._current_close_cache: Dict[str, Tuple[float, int]] = {}
        # Cache for fetched higher-timeframe windows to avoid duplicate remote calls (LRU per symbol)
        # Keyed per symbol -> OrderedDict[(tf_str, start_ts, end_ts) -> (array, fetched_at_ms)]
        self._tf_range_cache: Dict[str, OrderedDict[Tuple[str, int, int], Tuple[np.ndarray, int]]] = (
            {}
        )
        self._tf_range_cache_cap = 8
        self._projected_open_tail_ema_cache: Dict[
            str,
            OrderedDict[
                Tuple[Any, ...],
                Dict[str, Dict[float, float]],
            ],
        ] = {}
        self._projected_open_tail_ema_cache_cap = 4
        self._step_warning_keys: set[Tuple[str, str, str]] = set()
        # Deduplication for zero-candle synthesis warnings - only warn once per unique gap
        # Key: (symbol, first_ts) to identify a gap by its starting point
        # The end timestamp changes as time passes, but the start identifies the gap origin
        self._synth_gap_warned: set[Tuple[str, int]] = set()
        # Batch mode for startup: when enabled, collect warnings and log summary later
        self._synth_candle_batch_mode: bool = False
        # symbol -> {"count": int, "min_ts": int, "max_ts": int} during batch
        self._synth_candle_batch: Dict[str, Dict[str, int]] = {}
        # Batch mode for candle replacement logs: collect replacements and log summary at INFO
        self._candle_replace_batch_mode: bool = False
        self._candle_replace_batch: Dict[str, int] = {}  # symbol -> count replaced during batch
        # Track which timestamps were synthesized (per symbol) for EMA recomputation detection
        # When real data arrives for a previously synthetic timestamp, EMAs should be recomputed
        self._synthetic_timestamps: Dict[str, set[int]] = {}  # symbol -> set of synthetic ts (ms)
        # Timeout parameters for cross-process fetch locks
        self._lock_timeout_seconds = float(_LOCK_TIMEOUT_SECONDS)
        if lock_timeout_seconds is not None:
            try:
                candidate = float(lock_timeout_seconds)
                if candidate > 0.0 and math.isfinite(candidate):
                    self._lock_timeout_seconds = candidate
            except Exception:
                pass
        self._lock_stale_seconds = float(_LOCK_STALE_SECONDS)
        self._lock_backoff_initial = float(_LOCK_BACKOFF_INITIAL)
        self._lock_backoff_max = float(_LOCK_BACKOFF_MAX)
        self._lock_hold_timeout_seconds = max(60.0, self._lock_timeout_seconds * 6.0)
        # Reentrant bookkeeping for portalocker fetch locks: key -> _LockRecord.
        # Reentrancy is valid only within the owning asyncio task; a different
        # coroutine using this manager must wait for the same symbol/timeframe.
        self._held_fetch_locks: Dict[Tuple[str, str], _LockRecord] = {}
        self._fetch_lock_watchdogs: Dict[Tuple[str, str], asyncio.Task] = {}
        self._shutdown_guard = threading.Lock()
        self._closed = False
        atexit.register(self._cleanup_on_exit)

        # Standardize cache directory names (e.g., binanceusdm -> binance),
        # migrate any legacy data from historical_data/ to caches/ohlcv/,
        # and merge any duplicate symbol directories from inconsistent sanitization
        ohlcv_cache_base = _ensure_legacy_ohlcv_cache_base(self.cache_dir)
        historical_data_path = os.path.join(
            os.path.dirname(os.path.abspath(self.cache_dir)),
            "historical_data",
        )
        GATEIO_CACHE_CUTOFF_DATE = "2026-02-07"
        if self.exchange_name == "gateio" and GATEIO_CACHE_CUTOFF_DATE:
            _quarantine_gateio_cache_if_stale(
                ohlcv_cache_base,
                GATEIO_CACHE_CUTOFF_DATE,
            )
        migration_lock = os.path.join(ohlcv_cache_base, ".migration.lock")
        migration_done = os.path.join(ohlcv_cache_base, ".migration_done")
        try:
            with portalocker.Lock(migration_lock, timeout=0.1, fail_when_locked=True):
                try:
                    _quarantine_root_level_timeframe_debris(ohlcv_cache_base)
                except Exception as exc:
                    logging.error(
                        "Root-level OHLCV cache cleanup failed (non-fatal). Continuing "
                        "error_type=%s cache_base=%s",
                        bounded_exception_type(exc),
                        ohlcv_cache_base,
                    )
                if not os.path.exists(migration_done):
                    try:
                        standardize_cache_directories(ohlcv_cache_base)
                        migrate_legacy_data_all_on_init(
                            cache_base=ohlcv_cache_base,
                            historical_data_path=historical_data_path,
                        )
                        merge_duplicate_symbol_directories(ohlcv_cache_base)
                        try:
                            with open(migration_done, "w", encoding="utf-8") as handle:
                                handle.write(str(int(time.time())))
                        except Exception:
                            pass
                    except Exception as exc:
                        logging.error(
                            "Cache migration failed (non-fatal). Continuing without migration "
                            "error_type=%s cache_base=%s historical_data_path=%s",
                            bounded_exception_type(exc),
                            ohlcv_cache_base,
                            historical_data_path,
                        )
        except portalocker.exceptions.LockException:
            # Another process is handling migrations; skip.
            pass

        self._setup_logging()
        self._cleanup_stale_locks()

        # Initialize optional global semaphore for remote calls
        try:
            mcr = None if max_concurrent_requests in (None, 0) else int(max_concurrent_requests)
            self._net_sem = asyncio.Semaphore(mcr) if (mcr and mcr > 0) else None
        except Exception:
            self._net_sem = None

        # Global rate limit coordination: when a rate limit is hit, all concurrent
        # requests pause until this timestamp (prevents thundering herd retries)
        self._rate_limit_until: float = 0.0
        self._rate_limit_lock = asyncio.Lock()
        self._rate_limit_count: int = 0
        self._remote_fetch_spacing_lock = asyncio.Lock()
        try:
            self._remote_fetch_min_interval_ms = max(
                0.0,
                float(remote_fetch_min_interval_ms)
                if remote_fetch_min_interval_ms is not None
                else 0.0,
            )
        except Exception:
            self._remote_fetch_min_interval_ms = 0.0
        self._remote_fetch_last_started_ms: int = 0

        # Persistent HTTP session for archive fetches (created lazily)
        self._http_session: Optional["aiohttp.ClientSession"] = None
        self._http_session_lock = asyncio.Lock()

        # fetch controls
        # Base timeframe for storage/fetching is always 1m; higher TFs are per-call
        self._ccxt_timeframe = "1m"
        # Determine exchange id and adjust defaults per exchange quirks
        client_id = getattr(self.exchange, "id", self.exchange_name) or self.exchange_name
        self._ex_id = (
            "gateio"
            if str(client_id).lower() == "gate" and self.exchange_name == "gateio"
            else client_id
        )
        self._ccxt_limit_default = 1000
        self._ccxt_page_overlap_candles = 0
        self._record_payload_gaps_as_known = False
        self._ccxt_since_exclusive = False
        self._ccxt_limit_probe_done = False
        self._gateio_recent_window_clip_warned: set[str] = set()
        if isinstance(self._ex_id, str) and "bitget" in self._ex_id.lower():
            # Bitget often serves 1m klines with 200 limit per page
            self._ccxt_limit_default = 200
            # Overlap page boundaries to avoid missing the boundary candle
            self._ccxt_page_overlap_candles = 1
            # Bitget since parameter behaves as exclusive for 1m OHLCV
            self._ccxt_since_exclusive = True
            # Probe at runtime to see if Bitget now accepts >200 rows per page
            self._ccxt_limit_probe_done = False
        if isinstance(self._ex_id, str) and "kucoin" in self._ex_id.lower():
            # KuCoin futures returns max 200 rows per OHLCV call and can be sparse (trade-only minutes).
            self._ccxt_limit_default = 200
            # Overlap page boundaries to validate gaps between fetches.
            self._ccxt_page_overlap_candles = 1
            # Gaps inside a single payload are considered verified no-trade gaps.
            self._record_payload_gaps_as_known = True
            # KuCoin since behaves as exclusive for 1m OHLCV.
            self._ccxt_since_exclusive = True
        if isinstance(self._ex_id, str) and "bitunix" in self._ex_id.lower():
            # Bitunix futures caps every kline response at 200 rows.
            self._ccxt_limit_default = 200

        # Optional per-page range logging for selected symbols (debug pagination)
        self._page_debug_all = False
        self._page_debug_symbols: set[str] = set()
        if page_debug_symbols:
            try:
                for sym in page_debug_symbols:
                    if sym is None:
                        continue
                    sym_str = str(sym).strip()
                    if not sym_str:
                        continue
                    if sym_str == "*":
                        self._page_debug_all = True
                    else:
                        self._page_debug_symbols.add(sym_str)
            except Exception:
                self._page_debug_symbols = set()

    def set_stop_requested_callback(self, callback: Optional[Callable[[], bool]]) -> None:
        """Install a shutdown hook used by live bots to abort non-critical candle work."""
        self._stop_requested_callback = callback

    def _shutdown_requested(self) -> bool:
        callback = getattr(self, "_stop_requested_callback", None)
        if not callable(callback):
            return False
        try:
            return bool(callback())
        except Exception:
            return False

    def _raise_if_shutdown_requested(self, stage: str) -> None:
        if self._shutdown_requested():
            self._log("debug", "shutdown_abort", stage=stage)
            raise asyncio.CancelledError(f"candlestick manager shutdown during {stage}")

    async def _sleep_interruptible(self, seconds: float, *, stage: str) -> None:
        remaining = max(0.0, float(seconds))
        while remaining > 0.0:
            self._raise_if_shutdown_requested(stage)
            chunk = min(remaining, 0.25)
            await asyncio.sleep(chunk)
            remaining -= chunk
        self._raise_if_shutdown_requested(stage)

    # ----- Logging -----

    def _setup_logging(self) -> None:
        trace_level = getattr(logging, "TRACE", None)
        if not isinstance(trace_level, int):
            trace_level = 5
            logging.addLevelName(trace_level, "TRACE")
            setattr(logging, "TRACE", trace_level)
        level_map = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG,
            3: trace_level,
        }
        desired_level = level_map.get(self.debug_level, logging.INFO)
        self.log = logging.getLogger("passivbot.candlestick_manager")
        self.log.setLevel(desired_level)

    def start_synth_candle_batch(self) -> None:
        """Start batching zero-candle synthesis warnings for later aggregated logging."""
        self._synth_candle_batch_mode = True
        self._synth_candle_batch.clear()

    def _sparse_ohlcv_gaps_are_expected(self) -> bool:
        exid = (self._ex_id or "").lower() if isinstance(self._ex_id, str) else ""
        return "kucoin" in exid

    def flush_synth_candle_batch(self) -> None:
        """Log aggregated zero-candle synthesis summary and exit batch mode."""
        self._synth_candle_batch_mode = False
        if not self._synth_candle_batch:
            return
        total_symbols = len(self._synth_candle_batch)
        total_candles = sum(v.get("count", 0) for v in self._synth_candle_batch.values())
        # KuCoin futures commonly omits no-trade minutes. Large zero-volume synthesis
        # batches are expected there and should be visible without looking like data corruption.
        sparse_expected = self._sparse_ohlcv_gaps_are_expected()
        log_fn = self.log.info if sparse_expected or total_candles <= 1000 else self.log.warning

        def _fmt_range(min_ts: Optional[int], max_ts: Optional[int]) -> str:
            try:
                from datetime import datetime, timezone

                if min_ts is None or max_ts is None:
                    return "-"
                start = datetime.fromtimestamp(int(min_ts) / 1000, tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M"
                )
                end = datetime.fromtimestamp(int(max_ts) / 1000, tz=timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M"
                )
                return f"{start} to {end}" if start != end else start
            except Exception:
                return "-"

        if total_symbols == 1:
            symbol, meta = next(iter(self._synth_candle_batch.items()))
            count = int(meta.get("count", 0))
            rng = _fmt_range(meta.get("min_ts"), meta.get("max_ts"))
            suffix = (
                " | expected on sparse KuCoin no-trade minutes"
                if sparse_expected
                else ""
            )
            log_fn(
                "[candle] synthesized %d zero-candle%s for %s at %s "
                "(no data for requested minutes)%s",
                count,
                "s" if count > 1 else "",
                _log_symbol(symbol),
                rng,
                suffix,
            )
        else:
            # Log top symbols by synthesized count (limit to keep logs concise)
            top_n = 5
            sorted_syms = sorted(
                self._synth_candle_batch.items(),
                key=lambda kv: int(kv[1].get("count", 0)),
                reverse=True,
            )
            top_parts = []
            for sym, meta in sorted_syms[:top_n]:
                count = int(meta.get("count", 0))
                rng = _fmt_range(meta.get("min_ts"), meta.get("max_ts"))
                top_parts.append(f"{_log_symbol(sym)}:{count}@{rng}")
            extra = total_symbols - min(top_n, total_symbols)
            top_str = ", ".join(top_parts)
            if extra > 0:
                top_str = f"{top_str} (+{extra} more)"
            suffix = (
                " | expected on sparse KuCoin no-trade minutes"
                if sparse_expected
                else ""
            )
            log_fn(
                "[candle] synthesized %d zero-candle%s across %d symbols "
                "(no data for requested minutes) top=%s%s",
                total_candles,
                "s" if total_candles > 1 else "",
                total_symbols,
                top_str,
                suffix,
            )
        self._synth_candle_batch.clear()

    def start_candle_replace_batch(self) -> None:
        """Start batching candle replacement logs for later aggregated logging."""
        self._candle_replace_batch_mode = True
        self._candle_replace_batch.clear()

    def flush_candle_replace_batch(self) -> None:
        """Log aggregated candle replacement summary at INFO and exit batch mode."""
        self._candle_replace_batch_mode = False
        if not self._candle_replace_batch:
            return
        total_symbols = len(self._candle_replace_batch)
        total_candles = sum(self._candle_replace_batch.values())
        if total_symbols == 1:
            symbol, count = next(iter(self._candle_replace_batch.items()))
            self.log.info(
                "[candle] %s: real data replaced %d synthetic candle%s, EMA cache invalidated",
                _log_symbol(symbol),
                count,
                "s" if count > 1 else "",
            )
        else:
            self.log.info(
                "[candle] real data replaced %d synthetic candle%s across %d symbols, EMA caches invalidated",
                total_candles,
                "s" if total_candles > 1 else "",
                total_symbols,
            )
        self._candle_replace_batch.clear()

    # ----- Retention helpers -----

    def _cleanup_stale_locks(self) -> None:
        """Remove leftover lock files that are clearly stale."""
        try:
            base = Path(self.cache_dir) / self.exchange_name
        except Exception:
            return
        if not base.exists():
            return
        now = time.time()
        threshold = self._lock_stale_seconds
        for lock_path in base.glob("*/locks/*.lock"):
            try:
                stat = lock_path.stat()
            except FileNotFoundError:
                continue
            except Exception as exc:
                self.log.warning(
                    "failed to stat lock %s during cleanup error_type=%s",
                    lock_path,
                    bounded_exception_type(exc),
                )
                continue
            age = now - stat.st_mtime
            if age > threshold:
                lock = portalocker.Lock(str(lock_path), timeout=0, fail_when_locked=True)
                try:
                    lock.acquire()
                except portalocker.exceptions.LockException:
                    continue
                try:
                    lock_path.unlink()
                    self.log.info("removed stale candle lock %s (age %.1fs)", lock_path, age)
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    self.log.error(
                        "failed to remove stale lock %s error_type=%s",
                        lock_path,
                        bounded_exception_type(exc),
                    )
                finally:
                    try:
                        lock.release()
                    except Exception:
                        pass

    def _cleanup_on_exit(self) -> None:
        with self._shutdown_guard:
            if self._closed:
                return
            self._closed = True
        records = list(self._held_fetch_locks.values())
        self._held_fetch_locks.clear()
        watchdogs = list(self._fetch_lock_watchdogs.values())
        self._fetch_lock_watchdogs.clear()
        for task in watchdogs:
            task.cancel()
        for record in records:
            self._release_lock_sync(record)

    def _now_ms(self) -> int:
        """Return this manager's time source.

        Live bots use UTC wall time. Fake-live replay installs a scenario-time
        callback so completed-candle/EMA windows advance with the fake exchange.
        """
        callback = getattr(self, "_now_ms_callback", None)
        if callback is not None:
            try:
                now = int(callback())
                if now > 0:
                    return now
            except Exception:
                pass
        return _utc_now_ms()

    def _release_lock_sync(self, record: _LockRecord) -> None:
        try:
            record.lock.release()
        except Exception:
            pass
        self._remove_lockfile(record.path)

    def _remove_lockfile(self, path: str) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            return
        except Exception:
            return

    async def _release_lock(
        self, lock: portalocker.Lock, path: str, symbol: str, timeframe: str
    ) -> None:
        """Release a portalocker lock safely and refresh its metadata."""
        try:
            await asyncio.to_thread(lock.release)
        except portalocker.exceptions.LockException as exc:
            self._log(
                "warning",
                "fetch_lock_release_failed",
                symbol=symbol,
                timeframe=timeframe,
                error_type=bounded_exception_type(exc),
            )
        except Exception as exc:
            self._log(
                "warning",
                "fetch_lock_release_error",
                symbol=symbol,
                timeframe=timeframe,
                error_type=bounded_exception_type(exc),
            )
        finally:
            self._remove_lockfile(path)

    def _touch_lockfile(
        self,
        path: str,
        *,
        symbol: str | None = None,
        timeframe: str | None = None,
        acquired_at: float | None = None,
        attempt: int | None = None,
    ) -> None:
        payload = {
            "pid": os.getpid(),
            "exchange": str(self.exchange_name),
            "symbol": symbol,
            "timeframe": timeframe,
            "acquired_at": float(acquired_at if acquired_at is not None else time.time()),
            "attempt": attempt,
        }
        try:
            task = asyncio.current_task()
            if task is not None:
                payload["task"] = task.get_name()
        except RuntimeError:
            pass
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, sort_keys=True)
        except FileNotFoundError:
            return
        except Exception:
            with suppress(Exception):
                os.utime(path, None)

    def _read_lockfile_owner(self, path: str, *, compact: bool = False) -> str:
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            return "-"
        if not isinstance(payload, dict):
            return "-"
        bits = []
        keys = (
            ("pid", "task", "attempt")
            if compact
            else ("pid", "exchange", "symbol", "timeframe", "task", "attempt")
        )
        for key in keys:
            value = payload.get(key)
            if value is not None:
                bits.append(f"{key}={value}")
        acquired_at = payload.get("acquired_at")
        if acquired_at is not None:
            try:
                bits.append(f"held_s={max(0.0, time.time() - float(acquired_at)):.1f}")
            except Exception:
                pass
        return " ".join(bits) if bits else "-"

    def _cancel_fetch_lock_watchdog(self, key: Tuple[str, str]) -> None:
        task = self._fetch_lock_watchdogs.pop(key, None)
        if task is not None:
            task.cancel()

    def _start_fetch_lock_watchdog(
        self,
        key: Tuple[str, str],
        *,
        symbol: str,
        timeframe: str,
        acquired_at: float,
    ) -> None:
        timeout_s = float(getattr(self, "_lock_hold_timeout_seconds", 0.0) or 0.0)
        if timeout_s <= 0.0 or not math.isfinite(timeout_s):
            return

        async def _watchdog() -> None:
            try:
                await asyncio.sleep(timeout_s)
            except asyncio.CancelledError:
                return
            record = self._held_fetch_locks.get(key)
            if record is None or float(record.acquired_at) != float(acquired_at):
                return
            owner = self._read_lockfile_owner(record.path, compact=True)
            self._log(
                "warning",
                "fetch_lock_hold_timeout",
                symbol=symbol,
                timeframe=timeframe,
                owner=owner,
            )

        self._cancel_fetch_lock_watchdog(key)
        self._fetch_lock_watchdogs[key] = asyncio.create_task(_watchdog())

    def _lockfile_age(self, path: str) -> Optional[float]:
        try:
            mtime = os.path.getmtime(path)
        except FileNotFoundError:
            return None
        except Exception:
            return None
        return time.time() - mtime

    def _enforce_memory_retention(self, symbol: str) -> None:
        try:
            arr = self._cache.get(symbol)
            if arr is None or arr.size == 0:
                return
            nmax = self.max_memory_candles_per_symbol
            if nmax > 0 and arr.shape[0] > nmax:
                # keep last nmax by ts
                arr = np.sort(arr, order="ts")
                self._cache[symbol] = arr[-nmax:]
        except Exception:
            return

    def _enforce_disk_retention(
        self, symbol: str, timeframe: Optional[str] = None, *, tf: Optional[str] = None
    ) -> None:
        try:
            tf_norm = self._normalize_timeframe_arg(timeframe, tf)
            idx = self._ensure_symbol_index(symbol, tf=tf_norm)
            shards = idx.get("shards", {})
            if not shards:
                return
            # Sum counts; if over limit, delete oldest shard files until within limit
            total = 0
            items = []
            for k, v in shards.items():
                try:
                    count = int(v.get("count", 0))
                except Exception:
                    count = 0
                total += count
                items.append((k, v))
            limit = self.max_disk_candles_per_symbol_per_tf
            if limit <= 0 or total <= limit:
                return
            # Sort shards by date_key ascending (oldest first)
            items.sort(key=lambda x: x[0])
            # Remove oldest until under limit
            for date_key, meta in items:
                path = meta.get("path")
                try:
                    if path and os.path.exists(path):
                        os.remove(path)
                except Exception:
                    pass
                # update index
                try:
                    cnt = int(meta.get("count", 0))
                except Exception:
                    cnt = 0
                total -= cnt
                shards.pop(date_key, None)
                if total <= limit:
                    break
            # persist updated index
            idx["shards"] = shards
            key = f"{symbol}::{tf_norm}"
            self._index[key] = idx
            self._save_index(symbol, tf=tf_norm)
        except Exception:
            return

    # ----- Logging helpers -----

    @staticmethod
    def _fmt_ts(ms: Optional[int]) -> str:
        try:
            return (
                time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(int(ms) / 1000.0))
                if ms is not None
                else "-"
            )
        except Exception:
            return str(ms)

    def _log(self, level: str, event: str, **fields) -> None:
        try:
            ex = getattr(self, "_ex_id", self.exchange_name)
        except Exception:
            ex = self.exchange_name
        base = [f"[candle] event={event}"]
        # In debug modes, include caller info for traceability. The bounded OHLCV
        # failure warning retains its incident signature without this low-value detail.
        if self.debug_level >= 1 and event != "ccxt_fetch_ohlcv_failed":
            try:
                caller = get_caller_name()
                base.append(f"called_by={caller}")
            except Exception:
                pass
        base.append(f"exchange={ex}")
        parts = []
        for k, v in fields.items():
            if k.endswith("_ts") and isinstance(v, (int, np.integer)):
                parts.append(f"{k}={self._fmt_ts(int(v))}")
            elif k == "symbol":
                parts.append(f"{k}={_log_symbol(v)}")
            else:
                parts.append(f"{k}={v}")
        msg = " ".join(base + parts)
        if level == "debug":
            # Apply filtering: level 0 -> drop; level 1 -> network summaries;
            # level 2 -> actionable debug; level 3 -> trace/firehose.
            if self.debug_level <= 0:
                return
            is_network = isinstance(event, str) and (
                event.startswith("ccxt_") or event.startswith("archive_")
            )
            if self.debug_level == 1 and not is_network:
                return
            # Storage/cache hot paths are useful only when explicitly running TRACE-level
            # candle diagnostics. At normal DEBUG they can dominate live logs.
            high_volume_events = {
                "disk_load_done",
                "disk_load_plan",
                "disk_load_progress",
                "get_candles_check_refresh",
                "get_candles_present_decision",
                "ccxt_fetch_ohlcv",
                "ccxt_fetch_ohlcv_ok",
                "ccxt_fetch_paginated_done",
                "ccxt_fetch_progress",
                "fetch_lock_acquired",
                "fetch_lock_reentrant",
                "get_candles_present_inner",
                "historical_missing_spans",
                "historical_missing_spans_coalesced",
                "index_cached",
                "index_rebuild_range",
                "index_reload",
                "large_span_check",
                "large_span_needs_gap_fill",
                "legacy_index_built",
                "load_from_disk",
                "refresh_fetch",
                "refresh_skip_since",
                "runtime_synthetic_gap_materialized",
                "ttl_bypass_missing_coverage",
                "ttl_skip_trailing_present_gap",
            }
            if event in high_volume_events:
                if self.debug_level < 3:
                    return
                self.log.log(int(getattr(logging, "TRACE", 5)), msg)
                return
            self.log.debug(msg)
        elif level == "info":
            self.log.info(msg)
        elif level == "warning":
            self.log.warning(msg)
        else:
            self.log.error(msg)

    def _progress_log(self, key: Tuple[str, str, str], event: str, **fields) -> None:
        """Emit throttled DEBUG progress logs when enabled."""
        if self._progress_log_interval_seconds <= 0.0:
            return
        now = time.monotonic()
        last = self._progress_last_log.get(key, 0.0)
        if (now - last) < self._progress_log_interval_seconds:
            return
        self._progress_last_log[key] = now
        self._log("debug", event, **fields)

    def _record_skipped_trailing_gap(
        self,
        *,
        symbol: Optional[str],
        requested_end_ts: int,
        actual_end_ts: int,
        skipped_minutes: int,
    ) -> None:
        """Aggregate open-tail skip diagnostics instead of logging per EMA call."""
        if self.debug_level <= 0:
            return
        try:
            caller = get_caller_name()
        except Exception:
            caller = "-"
        key = (_log_symbol(symbol) if symbol else "-", caller)
        item = self._skipped_trailing_gap_summary.setdefault(
            key, {"count": 0, "max_minutes": 0, "latest_requested": 0, "latest_actual": 0}
        )
        item["count"] += 1
        item["max_minutes"] = max(int(item["max_minutes"]), int(skipped_minutes))
        item["latest_requested"] = max(int(item["latest_requested"]), int(requested_end_ts))
        item["latest_actual"] = max(int(item["latest_actual"]), int(actual_end_ts))
        now = time.monotonic()
        total = sum(int(v["count"]) for v in self._skipped_trailing_gap_summary.values())
        if total < 250 and (now - self._skipped_trailing_gap_summary_last_log) < 300.0:
            return
        self._skipped_trailing_gap_summary_last_log = now
        top_items = sorted(
            self._skipped_trailing_gap_summary.items(),
            key=lambda kv: (-int(kv[1]["count"]), -int(kv[1]["max_minutes"])),
        )[:8]
        details = "; ".join(
            f"{sym} caller={caller} count={data['count']} max_gap={data['max_minutes']}m"
            for (sym, caller), data in top_items
        )
        self.log.debug(
            "[candle] skipped trailing gap summary | total=%d groups=%d%s",
            total,
            len(self._skipped_trailing_gap_summary),
            f" | {details}" if details else "",
        )
        self._skipped_trailing_gap_summary.clear()

    def _log_persistent_gap_summary(self) -> None:
        """Log accumulated persistent gap summary if any, throttled to once per 30 min."""
        if not hasattr(self, "_persistent_gap_summary") or not self._persistent_gap_summary:
            return
        now = time.monotonic()
        last = getattr(self, "_persistent_gap_summary_last_log", 0.0)
        if (now - last) < 1800.0:  # Only log summary once per 30 minutes
            return
        self._persistent_gap_summary_last_log = now
        summary = self._persistent_gap_summary
        total = sum(summary.values())
        symbols = ", ".join(
            f"{_log_symbol(s)}:{c}" for s, c in sorted(summary.items())[:5]
        )
        if len(summary) > 5:
            symbols += f", +{len(summary) - 5} more"
        self.log.info(
            "[candle] persistent gaps: %d across %d symbols (%s). Use --force-refetch-gaps to retry.",
            total,
            len(summary),
            symbols,
        )
        self._persistent_gap_summary.clear()

    def _throttled_warning(self, throttle_key: str, event: str, **fields) -> None:
        """Emit a warning at most once per throttle window (default 5 min).

        Use this for warnings that may repeat frequently but only need to
        inform the user once. After the throttle window expires, the warning
        will be emitted again if the condition persists.
        """
        now = time.monotonic()
        last = self._warning_last_log.get(throttle_key)
        if last is not None and (now - last) < self._warning_throttle_seconds:
            return
        self._warning_last_log[throttle_key] = now
        self._log("warning", event, **fields)

    def _record_strict_gap(self, symbol: str, missing_count: int) -> None:
        """Accumulate strict gap counts for summary logging."""
        self._strict_gaps_summary[symbol] = self._strict_gaps_summary.get(symbol, 0) + missing_count

    def _log_strict_gaps_summary(self) -> None:
        """Log accumulated strict gap summary if any, throttled to once per 15 min."""
        if not self._strict_gaps_summary:
            return
        now = time.monotonic()
        if (now - self._strict_gaps_summary_last_log) < self._strict_gaps_summary_interval:
            return
        self._strict_gaps_summary_last_log = now
        summary = self._strict_gaps_summary
        total = sum(summary.values())
        symbols = ", ".join(f"{s}:{c}" for s, c in sorted(summary.items(), key=lambda x: -x[1])[:5])
        if len(summary) > 5:
            symbols += f", +{len(summary) - 5} more"
        self.log.debug(
            "[candle] strict mode gaps: %d missing candles across %d symbols (%s)",
            total,
            len(summary),
            symbols,
        )
        self._strict_gaps_summary.clear()

    def _emit_remote_fetch(self, payload: Dict[str, Any]) -> None:
        cb = getattr(self, "_remote_fetch_callback", None)
        if cb is None:
            return
        try:
            cb(sanitize_remote_fetch_diagnostic(payload))
        except Exception:
            # Must never break the fetch path due to logging/progress UI.
            return

    def set_persist_batch_observer(
        self,
        observer: Optional[Callable[[str, str, np.ndarray], None]],
    ) -> None:
        self._persist_batch_observer = observer

    def set_disk_load_observer(
        self,
        observer: Optional[Callable[[Dict[str, Any]], None]],
    ) -> None:
        self._disk_load_observer = observer

    def _emit_disk_load_observer(self, payload: Dict[str, Any]) -> None:
        observer = self._disk_load_observer
        if observer is None:
            return
        try:
            observer(payload)
        except Exception:
            # Observability hooks must never break cache loading or trading.
            return

    # ----- Paths and index -----

    def _symbol_dir(
        self, symbol: str, timeframe: Optional[str] = None, *, tf: Optional[str] = None
    ) -> str:
        sym = _sanitize_symbol(symbol)
        tf_dir = self._normalize_timeframe_arg(timeframe, tf)
        return str(Path(self.cache_dir) / "ohlcv" / self.exchange_name / tf_dir / sym)

    def _index_path(
        self, symbol: str, timeframe: Optional[str] = None, *, tf: Optional[str] = None
    ) -> str:
        return str(Path(self._symbol_dir(symbol, timeframe=timeframe, tf=tf)) / "index.json")

    def _shard_path(
        self,
        symbol: str,
        date_key: str,
        timeframe: Optional[str] = None,
        *,
        tf: Optional[str] = None,
    ) -> str:
        return str(Path(self._symbol_dir(symbol, timeframe=timeframe, tf=tf)) / f"{date_key}.npy")

    def _recompute_shard_derived_meta(self, idx: dict) -> None:
        """Recompute index bounds which are derived exclusively from shard metadata."""
        shards = idx.get("shards", {})
        if not isinstance(shards, dict):
            shards = {}
            idx["shards"] = shards
        meta = idx.setdefault("meta", {})
        try:
            last_ts = 0
            observed_start_ts: Optional[int] = None
            for shard_meta in shards.values():
                if not isinstance(shard_meta, dict):
                    continue
                mt = shard_meta.get("max_ts")
                if mt is not None:
                    last_ts = max(last_ts, int(mt))
                mi = shard_meta.get("min_ts")
                if mi is not None:
                    observed_start_ts = (
                        int(mi)
                        if observed_start_ts is None
                        else min(observed_start_ts, int(mi))
                    )
            meta["last_final_ts"] = int(last_ts)
            meta["observed_start_ts"] = observed_start_ts
            meta["inception_ts"] = observed_start_ts
        except Exception:
            meta["last_final_ts"] = 0
            meta["observed_start_ts"] = None
            meta["inception_ts"] = None

    def _prune_missing_shards_from_index(self, idx: dict) -> int:
        """Remove shard entries whose files are missing; refresh derived meta fields."""
        try:
            shards = idx.get("shards", {})
            if not isinstance(shards, dict) or not shards:
                return 0
            removed = 0
            for day_key, shard_meta in list(shards.items()):
                if not isinstance(shard_meta, dict):
                    continue
                path = shard_meta.get("path")
                if not path:
                    continue
                if not os.path.exists(str(path)):
                    shards.pop(day_key, None)
                    removed += 1
            if not removed:
                return 0
            idx["shards"] = shards
            self._recompute_shard_derived_meta(idx)
            return int(removed)
        except Exception:
            return 0

    def _ensure_symbol_index(
        self, symbol: str, timeframe: Optional[str] = None, *, tf: Optional[str] = None
    ) -> dict:
        tf_norm = self._normalize_timeframe_arg(timeframe, tf)
        key = f"{symbol}::{tf_norm}"
        idx_path = self._index_path(symbol, timeframe=timeframe, tf=tf_norm)
        existing = self._index.get(key)
        cached_mtime = self._index_mtime.get(key)
        try:
            stat = os.stat(idx_path)
            current_mtime = (
                int(stat.st_mtime_ns),
                int(stat.st_size),
                int(stat.st_ino),
            )
        except FileNotFoundError:
            current_mtime = None
        except Exception:
            current_mtime = None

        if existing is None or cached_mtime != current_mtime:
            idx = {"shards": {}, "meta": {}}
            # Try load from disk
            if current_mtime is not None:
                try:
                    with open(idx_path, "r", encoding="utf-8") as f:
                        idx = json.load(f)
                except FileNotFoundError:
                    pass
                except Exception as e:  # pragma: no cover
                    self._log(
                        "warning",
                        "index_load_failed",
                        symbol=symbol,
                        timeframe=tf_norm,
                        error_type=bounded_exception_type(e),
                    )
            if not isinstance(idx, dict):
                idx = {"shards": {}, "meta": {}}
            idx.setdefault("shards", {})
            meta = idx.setdefault("meta", {})
            legacy_history_bounds = (
                "observed_start_ts" not in meta and "authoritative_start_ts" not in meta
            )
            meta.setdefault("known_gaps", [])  # list of [start_ts, end_ts]
            meta.setdefault("last_refresh_ms", 0)
            meta.setdefault("last_final_ts", 0)
            observed_start_ts = meta.get("observed_start_ts", meta.get("inception_ts"))
            meta["observed_start_ts"] = int(observed_start_ts) if observed_start_ts is not None else None
            meta["inception_ts"] = meta["observed_start_ts"]  # legacy alias for earliest observed candle
            meta.setdefault("authoritative_start_ts", None)
            meta.setdefault("authoritative_start_source", None)
            meta.setdefault("inception_ts_probe_ms", 0)
            meta.setdefault("inception_ts_probe_end_ts", 0)
            migrated_pre_inception = False
            if legacy_history_bounds and meta.get("authoritative_start_ts") is None:
                legacy_authoritative_start = self._infer_legacy_authoritative_start_ts(meta)
                if legacy_authoritative_start is not None:
                    meta["authoritative_start_ts"] = int(legacy_authoritative_start)
                    meta["authoritative_start_source"] = "legacy_pre_inception_gap"
                else:
                    original_gaps = list(meta.get("known_gaps", []))
                    retained_gaps = []
                    for gap in original_gaps:
                        if isinstance(gap, dict) and str(gap.get("reason", "")) == "pre_inception":
                            migrated_pre_inception = True
                            continue
                        retained_gaps.append(gap)
                    if migrated_pre_inception:
                        meta["known_gaps"] = retained_gaps

            # Keep index consistent if shard files were deleted.
            removed = self._prune_missing_shards_from_index(idx)
            if removed:
                self._log(
                    "warning",
                    "index_pruned_missing_shards",
                    symbol=symbol,
                    timeframe=tf_norm,
                    removed=removed,
                )
            self._index[key] = idx
            self._index_mtime[key] = current_mtime
            if migrated_pre_inception:
                self._save_index(symbol, tf=tf_norm)
            self._log(
                "debug",
                "index_reload",
                symbol=symbol,
                timeframe=tf_norm,
                mtime=current_mtime,
                cache_hit=existing is not None,
            )
            return idx

        idx = existing
        # Ensure meta keys even for cached entries (in case earlier versions lacked them)
        idx.setdefault("shards", {})
        meta = idx.setdefault("meta", {})
        legacy_history_bounds = (
            "observed_start_ts" not in meta and "authoritative_start_ts" not in meta
        )
        meta.setdefault("known_gaps", [])
        meta.setdefault("last_refresh_ms", 0)
        meta.setdefault("last_final_ts", 0)
        observed_start_ts = meta.get("observed_start_ts", meta.get("inception_ts"))
        migrated_pre_inception = False
        meta["observed_start_ts"] = int(observed_start_ts) if observed_start_ts is not None else None
        meta["inception_ts"] = meta["observed_start_ts"]
        meta.setdefault("authoritative_start_ts", None)
        meta.setdefault("authoritative_start_source", None)
        if legacy_history_bounds and meta.get("authoritative_start_ts") is None:
            legacy_authoritative_start = self._infer_legacy_authoritative_start_ts(meta)
            if legacy_authoritative_start is not None:
                meta["authoritative_start_ts"] = int(legacy_authoritative_start)
                meta["authoritative_start_source"] = "legacy_pre_inception_gap"
            else:
                original_gaps = list(meta.get("known_gaps", []))
                retained_gaps = []
                for gap in original_gaps:
                    if isinstance(gap, dict) and str(gap.get("reason", "")) == "pre_inception":
                        migrated_pre_inception = True
                        continue
                    retained_gaps.append(gap)
                if migrated_pre_inception:
                    meta["known_gaps"] = retained_gaps

        # Keep cached index consistent if shard files were deleted while running.
        removed = self._prune_missing_shards_from_index(idx)
        if removed:
            self._log(
                "warning",
                "index_pruned_missing_shards",
                symbol=symbol,
                timeframe=tf_norm,
                removed=removed,
            )
        self._index[key] = idx
        self._index_mtime[key] = current_mtime
        if migrated_pre_inception:
            self._save_index(symbol, tf=tf_norm)
        if current_mtime is not None:
            self._log("debug", "index_cached", symbol=symbol, timeframe=tf_norm, mtime=current_mtime)
        return idx

    def _atomic_write_bytes(self, path: str, data: bytes) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def _save_index(
        self, symbol: str, timeframe: Optional[str] = None, *, tf: Optional[str] = None
    ) -> None:
        tf_norm = self._normalize_timeframe_arg(timeframe, tf)
        key = f"{symbol}::{tf_norm}"
        idx_path = self._index_path(symbol, timeframe=timeframe, tf=tf_norm)
        payload = json.dumps(self._index[key], sort_keys=True).encode("utf-8")
        # Lock the final target index.json to serialize writers
        os.makedirs(os.path.dirname(idx_path), exist_ok=True)
        # Use portalocker with a filename, not a file handle, so it creates the file if missing
        lock_path = idx_path + ".lock"
        with portalocker.Lock(lock_path, timeout=5):
            self._atomic_write_bytes(idx_path, payload)
        try:
            stat = os.stat(idx_path)
            self._index_mtime[key] = (
                int(stat.st_mtime_ns),
                int(stat.st_size),
                int(stat.st_ino),
            )
        except Exception:
            self._index_mtime[key] = None

    def _fetch_lock_path(self, symbol: str, timeframe: str) -> str:
        safe_symbol = _sanitize_symbol(symbol)
        lock_dir = os.path.join(
            self.cache_dir,
            self.exchange_name,
            safe_symbol,
            "locks",
        )
        os.makedirs(lock_dir, exist_ok=True)
        return os.path.join(lock_dir, f"{timeframe}.lock")

    @asynccontextmanager
    async def _acquire_fetch_lock(self, symbol: str, timeframe: Optional[str]) -> AsyncIterator[None]:
        tf_norm = self._normalize_timeframe_arg(timeframe, None)

        lock_path = self._fetch_lock_path(symbol, tf_norm)
        key = (symbol, tf_norm)
        current_task = asyncio.current_task()
        held = self._held_fetch_locks.get(key)
        if held is not None and held.owner_task is current_task:
            self._held_fetch_locks[key] = _LockRecord(
                lock=held.lock,
                path=held.path,
                count=held.count + 1,
                acquired_at=held.acquired_at,
                owner_task=held.owner_task,
            )
            self._log(
                "debug",
                "fetch_lock_reentrant",
                symbol=symbol,
                timeframe=tf_norm,
                depth=held.count + 1,
            )
            try:
                yield
            finally:
                record = self._held_fetch_locks.get(key)
                if record is not None and record.count <= 1:
                    self._held_fetch_locks.pop(key, None)
                    self._cancel_fetch_lock_watchdog(key)
                    await self._release_lock(record.lock, record.path, symbol, tf_norm)
                elif record is not None:
                    self._held_fetch_locks[key] = _LockRecord(
                        lock=record.lock,
                        path=record.path,
                        count=record.count - 1,
                        acquired_at=record.acquired_at,
                        owner_task=record.owner_task,
                    )
            return

        backoff = self._lock_backoff_initial
        deadline = time.monotonic() + self._lock_timeout_seconds
        attempt = 0

        while True:
            attempt += 1
            self._raise_if_shutdown_requested("fetch_lock_wait")
            lock_obj = portalocker.Lock(lock_path, timeout=0, fail_when_locked=True)
            try:
                await asyncio.to_thread(lock_obj.acquire)
                acquired_at = time.time()
                self._touch_lockfile(
                    lock_path,
                    symbol=symbol,
                    timeframe=tf_norm,
                    acquired_at=acquired_at,
                    attempt=attempt,
                )
                self._held_fetch_locks[key] = _LockRecord(
                    lock=lock_obj,
                    path=lock_path,
                    count=1,
                    acquired_at=acquired_at,
                    owner_task=current_task,
                )
                self._start_fetch_lock_watchdog(
                    key,
                    symbol=symbol,
                    timeframe=tf_norm,
                    acquired_at=acquired_at,
                )
                self._log(
                    "debug",
                    "fetch_lock_acquired",
                    symbol=symbol,
                    timeframe=tf_norm,
                    attempt=attempt,
                )
                try:
                    yield
                finally:
                    record = self._held_fetch_locks.pop(key, None)
                    self._cancel_fetch_lock_watchdog(key)
                    if record is not None:
                        await self._release_lock(record.lock, record.path, symbol, tf_norm)
                return
            except portalocker.exceptions.LockException as exc:
                age = self._lockfile_age(lock_path)
                if age is not None and age > self._lock_stale_seconds:
                    self._log(
                        "warning",
                        "fetch_lock_stale_waiting",
                        symbol=symbol,
                        timeframe=tf_norm,
                        age=f"{age:.2f}",
                        lock_path=lock_path,
                        owner=self._read_lockfile_owner(lock_path),
                    )

                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out acquiring candle lock for {symbol} ({tf_norm}) after "
                        f"{self._lock_timeout_seconds:.1f}s"
                    ) from exc

                self._log(
                    "debug",
                    "fetch_lock_wait",
                    symbol=symbol,
                    timeframe=tf_norm,
                    attempt=attempt,
                    error_type=bounded_exception_type(exc),
                )
                await self._sleep_interruptible(backoff, stage="fetch_lock_wait")
                backoff = min(backoff * 2.0, self._lock_backoff_max)

    @staticmethod
    def _normalize_timeframe_arg(
        timeframe: Optional[str], tf: Optional[str], default: str = "1m"
    ) -> str:
        """Resolve alias combination to a canonical, lowercase timeframe string."""
        value = tf if tf is not None else timeframe
        if not value:
            return default
        try:
            return str(value).strip().lower() or default
        except Exception:
            return default

    def _ensure_symbol_cache(self, symbol: str) -> np.ndarray:
        arr = self._cache.get(symbol)
        if arr is None:
            arr = np.empty((0,), dtype=CANDLE_DTYPE)
            self._cache[symbol] = arr
        return arr

    # ----- Shard loading helpers -----

    def _iter_shard_paths(
        self, symbol: str, timeframe: Optional[str] = None, *, tf: Optional[str] = None
    ) -> Dict[str, str]:
        """Return mapping date_key -> path for available shard files on disk.

        Results are cached per (symbol, tf) to avoid redundant glob scans.
        Call _invalidate_shard_paths_cache(symbol, tf) after saving new shards.
        """
        tf_norm = self._normalize_timeframe_arg(timeframe, tf)
        cache_key = (symbol, tf_norm)
        if cache_key in self._shard_paths_cache:
            return self._shard_paths_cache[cache_key]

        sd = Path(self._symbol_dir(symbol, timeframe=timeframe, tf=tf))
        if not sd.exists():
            # Cache empty result to avoid repeated directory checks
            self._shard_paths_cache[cache_key] = {}
            return {}
        out: Dict[str, str] = {}
        for p in sd.glob("*.npy"):
            name = p.stem  # YYYY-MM-DD
            if len(name) == 10 and name[4] == "-" and name[7] == "-":
                out[name] = str(p)
        self._shard_paths_cache[cache_key] = out
        return out

    def _invalidate_shard_paths_cache(
        self, symbol: str, timeframe: Optional[str] = None, *, tf: Optional[str] = None
    ) -> None:
        """Invalidate the cached shard paths for a symbol/tf after saving new shards."""
        tf_norm = self._normalize_timeframe_arg(timeframe, tf)
        cache_key = (symbol, tf_norm)
        self._shard_paths_cache.pop(cache_key, None)

    def _date_range_of_key(self, date_key: str) -> Tuple[int, int]:
        """Return [start_ms, end_ms] inclusive for a date key YYYY-MM-DD in UTC."""
        # Parse simple date without importing datetime to keep deps minimal
        y, m, d = map(int, date_key.split("-"))
        # Use time.gmtime to compute midnight UTC of that date
        tm = time.struct_time((y, m, d, 0, 0, 0, 0, 0, 0))
        start = int(calendar.timegm(tm)) * 1000
        end = start + 24 * 60 * 60 * 1000 - ONE_MIN_MS
        return start, end

    def _date_key(self, ts_ms: int) -> str:
        """Return YYYY-MM-DD for a UTC ms timestamp."""
        return time.strftime("%Y-%m-%d", time.gmtime(int(ts_ms) / 1000.0))

    def _date_keys_between(self, start_ts: int, end_ts: int) -> Dict[str, Tuple[int, int]]:
        """Return mapping of date_key -> (day_start_ms, day_end_ms) covering [start,end]."""
        # Align to 00:00 UTC of the start day
        first_key = self._date_key(start_ts)
        y, m, d = map(int, first_key.split("-"))
        tm = time.struct_time((y, m, d, 0, 0, 0, 0, 0, 0))
        day_start = int(calendar.timegm(tm)) * 1000
        res: Dict[str, Tuple[int, int]] = {}
        t = day_start
        while t <= end_ts:
            key = self._date_key(t)
            ds, de = self._date_range_of_key(key)
            res[key] = (ds, de)
            t = de + ONE_MIN_MS
        return res

    def _legacy_coin_from_symbol(self, symbol: str) -> str:
        """Return the coin key used by legacy downloader caches."""
        symbol = str(symbol or "")
        if not symbol:
            return ""
        if "/" in symbol:
            base = symbol.split("/", 1)[0]
        elif ":" in symbol:
            base = symbol.split(":", 1)[0]
        else:
            base = symbol
        base = base.strip()
        # Some exchanges encode symbols like "HYPE_USDT:USDT".
        # Legacy downloader caches typically use the base coin only ("HYPE").
        if "_" in base:
            left, right = base.rsplit("_", 1)
            if right in {"USDT", "USDC", "USD", "BUSD"}:
                base = left
        return base

    def _legacy_symbol_code_from_symbol(self, symbol: str) -> str:
        """Return legacy symbol codes used in some historical_data subtrees."""
        try:
            return self._archive_symbol_code(symbol)
        except Exception:
            return ""

    def _legacy_shard_dirs(self, symbol: str, tf: str) -> List[str]:
        if tf != "1m":
            return []
        ex = str(self.exchange_name or "").lower()
        coin = self._legacy_coin_from_symbol(symbol)
        sym_code = self._legacy_symbol_code_from_symbol(symbol)
        out: List[str] = []
        if coin:
            out.append(os.path.join("historical_data", f"ohlcvs_{ex}", coin))
        if ex == "binanceusdm" and sym_code:
            out.append(os.path.join("historical_data", "ohlcvs_futures", sym_code))
        if ex == "bybit" and sym_code:
            out.append(os.path.join("historical_data", "ohlcvs_bybit", sym_code))
        return out

    def _get_legacy_shard_paths(self, symbol: str, tf: str) -> Dict[str, str]:
        """Return mapping date_key -> legacy shard path for a symbol+tf (cached)."""
        ex = str(self.exchange_name or "").lower()
        key = (ex, str(symbol), str(tf))
        cached = self._legacy_shard_paths_cache.get(key)
        if cached is not None:
            return cached
        mapping: Dict[str, str] = {}
        scanned_dirs: List[str] = []
        for d in self._legacy_shard_dirs(symbol, tf):
            try:
                dp = Path(d)
                if not dp.exists():
                    continue
                scanned_dirs.append(str(dp))
                for p in dp.glob("*.npy"):
                    name = p.stem
                    if len(name) == 10 and name[4] == "-" and name[7] == "-":
                        # Prefer earlier directories in the list if duplicates exist.
                        mapping.setdefault(name, str(p))
            except Exception:
                continue
        self._legacy_shard_paths_cache[key] = mapping
        if mapping:
            self._log(
                "debug",
                "legacy_index_built",
                symbol=symbol,
                timeframe=tf,
                legacy_days=len(mapping),
                legacy_dirs=";".join(scanned_dirs[:3]) + (";..." if len(scanned_dirs) > 3 else ""),
            )
        return mapping

    def _load_shard(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            # Missing file is expected for pre-inception dates - log at debug level
            self.log.debug(f"Shard not found (expected for pre-inception): {path}")
            return np.empty((0,), dtype=CANDLE_DTYPE)
        try:
            with open(path, "rb") as f:
                arr = np.load(f, allow_pickle=False)
            if isinstance(arr, np.ndarray) and arr.dtype == CANDLE_DTYPE:
                return arr
            # Legacy downloader shards are often stored as 2D float arrays:
            # [timestamp, open, high, low, close, volume]
            if isinstance(arr, np.ndarray) and arr.ndim == 2 and arr.shape[1] >= 6:
                raw = np.asarray(arr[:, :6], dtype=np.float64)
                out = np.empty((raw.shape[0],), dtype=CANDLE_DTYPE)
                out["ts"] = raw[:, 0].astype(np.int64)
                out["o"] = raw[:, 1].astype(np.float32)
                out["h"] = raw[:, 2].astype(np.float32)
                out["l"] = raw[:, 3].astype(np.float32)
                out["c"] = raw[:, 4].astype(np.float32)
                out["bv"] = raw[:, 5].astype(np.float32)
                return out
            return _ensure_dtype(arr)
        except Exception as e:  # pragma: no cover - best effort
            self.log.warning(
                "Failed loading shard %s error_type=%s",
                path,
                bounded_exception_type(e),
            )
            return np.empty((0,), dtype=CANDLE_DTYPE)

    def _legacy_day_is_complete(self, symbol: str, tf: str, date_key: str) -> bool:
        """Return True if legacy has a continuous shard for this day.

        "Complete" is defined as a full UTC-day of 1m candles:
        - exactly 1440 minutes
        - spanning [00:00, 23:59] UTC for the given date_key
        - strictly 1m-continuous with no duplicates

        This is intentionally strict because this flag gates whether we skip writing a
        primary shard overlay. If we mistakenly treat a partial legacy shard as complete,
        we will keep re-downloading the missing minutes every run but never persist them.
        """
        cache_key = (str(symbol), str(tf), str(date_key))
        cached = self._legacy_day_quality_cache.get(cache_key)
        if cached is not None:
            return bool(cached)
        ok = False
        try:
            legacy_paths = self._get_legacy_shard_paths(symbol, tf)
            legacy_path = legacy_paths.get(date_key)
            if not legacy_path or not os.path.exists(str(legacy_path)):
                ok = False
            else:
                arr = self._load_shard(str(legacy_path))
                if arr.size == 0:
                    ok = False
                else:
                    day_start, day_end = self._date_range_of_key(str(date_key))
                    expected_len = int((day_end - day_start) // ONE_MIN_MS) + 1  # 1440
                    if int(arr.shape[0]) != int(expected_len):
                        ok = False
                    else:
                        ts = np.sort(arr["ts"].astype(np.int64, copy=False))
                        if int(ts[0]) != int(day_start) or int(ts[-1]) != int(day_end):
                            ok = False
                        else:
                            diffs = np.diff(ts)
                            ok = bool(
                                diffs.size
                                and int(diffs.min()) == ONE_MIN_MS
                                and int(diffs.max()) == ONE_MIN_MS
                            )
        except Exception:
            ok = False
        self._legacy_day_quality_cache[cache_key] = bool(ok)
        return bool(ok)

    def _load_from_disk(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
        merge_memory_cache: bool = True,
    ) -> Optional[np.ndarray]:
        """Load any shards intersecting [start_ts, end_ts] and merge into cache.

        Primary cache: `{cache_dir}/ohlcv/{exchange}/{tf}/{symbol}/YYYY-MM-DD.npy`

        Note: Legacy data from `historical_data/` is automatically migrated to the
        primary cache on CandlestickManager initialization. The legacy fallback below
        remains as a safety net for any data that wasn't migrated.

        Set ``merge_memory_cache=False`` to inspect canonical disk contents
        without merging their rows into the in-memory candle cache.
        """
        try:
            tf_norm = self._normalize_timeframe_arg(timeframe, tf)
            shard_paths = self._iter_shard_paths(symbol, tf=tf_norm)
            legacy_paths = self._get_legacy_shard_paths(symbol, tf_norm)
            days = self._date_keys_between(start_ts, end_ts)
            load_keys: List[Tuple[str, str]] = []
            day_ctx: Dict[str, Dict[str, Any]] = {}
            legacy_hits = 0
            primary_hits = 0
            merged_hits = 0
            for key, (day_start, day_end) in days.items():
                if day_end < start_ts or day_start > end_ts:
                    continue
                primary_path = shard_paths.get(key)
                legacy_path = legacy_paths.get(key)
                if primary_path is None and legacy_path is None:
                    continue

                chosen_path: Optional[str] = None
                chosen_source: str = ""

                # For 1m, treat legacy downloader shards as canonical and use primary as an
                # overlay only when legacy is missing/incomplete.
                if tf_norm == "1m" and legacy_path is not None:
                    legacy_complete = False
                    try:
                        legacy_complete = self._legacy_day_is_complete(symbol, tf_norm, key)
                    except Exception:
                        legacy_complete = False

                    if legacy_complete:
                        chosen_path = legacy_path
                        chosen_source = "legacy"
                        legacy_hits += 1
                    else:
                        if primary_path is not None:
                            # Load both and merge to maximize coverage (reduces slow refetch paths).
                            chosen_path = legacy_path
                            chosen_source = "merge"
                            merged_hits += 1
                        else:
                            chosen_path = legacy_path
                            chosen_source = "legacy"
                            legacy_hits += 1
                else:
                    if primary_path is not None:
                        chosen_path = primary_path
                        chosen_source = "primary"
                        primary_hits += 1
                    else:
                        chosen_path = legacy_path
                        chosen_source = "legacy"
                        legacy_hits += 1

                if chosen_path is not None:
                    load_keys.append((key, chosen_path))
                    day_ctx[key] = {
                        "day_start": int(day_start),
                        "day_end": int(day_end),
                        "source": chosen_source,
                        "primary_path": primary_path,
                        "legacy_path": legacy_path,
                    }
            if not load_keys:
                return
            self._log(
                "debug",
                "disk_load_plan",
                symbol=symbol,
                timeframe=tf_norm,
                days_total=len(days),
                primary_days=primary_hits,
                legacy_days=legacy_hits,
                merged_days=merged_hits,
            )
            # Load and merge with coarse progress updates to show activity for large ranges.
            arrays: List[np.ndarray] = []
            t0 = time.monotonic()
            last_progress_log = t0
            for i, (day_key, path) in enumerate(sorted(load_keys), start=1):
                ctx = day_ctx.get(day_key, {})
                src = str(ctx.get("source") or "")
                if tf_norm == "1m" and src == "merge":
                    legacy_arr = self._load_shard(path)
                    primary_arr = np.empty((0,), dtype=CANDLE_DTYPE)
                    try:
                        pp = ctx.get("primary_path")
                        if pp:
                            primary_arr = self._load_shard(str(pp))
                    except Exception:
                        primary_arr = np.empty((0,), dtype=CANDLE_DTYPE)
                    # Keep legacy canonical: primary should only fill legacy gaps.
                    a = self._merge_overwrite(primary_arr, legacy_arr)
                else:
                    a = self._load_shard(path)

                # NOTE: We intentionally do NOT write legacy data into primary shards.
                # Primary is only used to fill gaps where legacy is missing/incomplete.
                arrays.append(a)
                now = time.monotonic()
                if now - last_progress_log >= 5.0 or i == len(load_keys):
                    last_progress_log = now
                    self._log(
                        "debug",
                        "disk_load_progress",
                        symbol=symbol,
                        timeframe=tf_norm,
                        loaded=i,
                        total=len(load_keys),
                        current_day=day_key,
                        elapsed_s=f"{(now - t0):.1f}",
                    )
            arrays = [a for a in arrays if a.size]
            if not arrays:
                return
            merged_disk = np.sort(np.concatenate(arrays), order="ts")

            # If legacy data revealed earlier candles than our stored inception_ts,
            # update inception_ts now so archive prefetch logic doesn't skip.
            if tf_norm == "1m":
                try:
                    self._maybe_update_inception_ts(symbol, merged_disk, save=True)
                except Exception as exc:
                    self._log(
                        "warning",
                        "maybe_update_inception_ts_failed",
                        symbol=symbol,
                        error_type=bounded_exception_type(exc),
                    )
            self._log(
                "debug",
                "disk_load_done",
                symbol=symbol,
                timeframe=tf_norm,
                rows=int(merged_disk.shape[0]),
                elapsed_s=f"{(time.monotonic() - t0):.1f}",
            )
            self._log(
                "debug",
                "load_from_disk",
                symbol=symbol,
                timeframe=tf_norm,
                days=len(load_keys),
                primary_days=primary_hits,
                legacy_days=legacy_hits,
                rows=int(merged_disk.shape[0]),
                start_ts=start_ts,
                end_ts=end_ts,
            )
            loaded_start_ts = None
            loaded_end_ts = None
            if merged_disk.size:
                try:
                    loaded_start_ts = int(merged_disk["ts"][0])
                    loaded_end_ts = int(merged_disk["ts"][-1])
                except Exception:
                    loaded_start_ts = None
                    loaded_end_ts = None
            try:
                self._emit_disk_load_observer(
                    {
                        "symbol": symbol,
                        "timeframe": tf_norm,
                        "start_ts": int(start_ts),
                        "end_ts": int(end_ts),
                        "loaded_rows": int(merged_disk.shape[0]),
                        "loaded_start_ts": loaded_start_ts,
                        "loaded_end_ts": loaded_end_ts,
                        "days": int(len(load_keys)),
                        "primary_days": int(primary_hits),
                        "legacy_days": int(legacy_hits),
                        "merged_days": int(merged_hits),
                        "source_days": {
                            "primary": int(primary_hits),
                            "legacy": int(legacy_hits),
                            "merged": int(merged_hits),
                        },
                        "elapsed_ms": int(
                            max(0.0, (time.monotonic() - t0) * 1000.0)
                        ),
                    }
                )
            except Exception:
                # Disk-load telemetry must never break cache loading.
                pass
            if tf_norm == "1m":
                if not merge_memory_cache:
                    return merged_disk
                existing = self._ensure_symbol_cache(symbol)
                merged = self._merge_overwrite(existing, merged_disk)
                self._cache[symbol] = merged
                return merged
            else:
                # Do not touch 1m cache for higher TF; let caller handle
                return merged_disk
        except Exception as e:  # pragma: no cover - noncritical
            self._log(
                "warning",
                "disk_load_error",
                symbol=symbol,
                timeframe=tf_norm,
                error_type=bounded_exception_type(e),
            )
            return None

    def _save_range_incremental(
        self,
        symbol: str,
        arr: np.ndarray,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
        defer_index: bool = False,
    ) -> None:
        """Persist candles by merging with existing shards on disk.

        Args:
            defer_index: If True, defer index.json write until flush_deferred_index is called.
        """
        if arr.size == 0:
            return
        arr = np.sort(_ensure_dtype(arr), order="ts")
        tf_norm = self._normalize_timeframe_arg(timeframe, tf)
        shard_paths = self._iter_shard_paths(symbol, tf=tf_norm)
        shards_saved = []

        def flush_bucket(key: Optional[str], bucket: List[Tuple], is_last: bool = False) -> None:
            if key is None or not bucket:
                return
            chunk = np.array(bucket, dtype=CANDLE_DTYPE)
            existing = np.empty((0,), dtype=CANDLE_DTYPE)
            path = shard_paths.get(key)
            if path and os.path.exists(path):
                existing = self._load_shard(path)
            merged = self._merge_overwrite(existing, chunk)
            # Defer index write for all but the last shard (or all if defer_index=True)
            should_defer = defer_index or not is_last
            self._save_shard(symbol, key, merged, tf=tf_norm, defer_index=should_defer)
            shard_paths[key] = self._shard_path(symbol, key, tf=tf_norm)
            shards_saved.append(key)

        current_key: Optional[str] = None
        bucket: List[Tuple] = []
        keys_to_process = []

        # First pass: collect all keys
        for row in arr:
            key = self._date_key(int(row["ts"]))
            if current_key is None:
                current_key = key
            if key != current_key:
                keys_to_process.append((current_key, bucket))
                bucket = []
                current_key = key
            bucket.append(tuple(row.tolist()))
        if current_key is not None:
            keys_to_process.append((current_key, bucket))

        # Second pass: flush with is_last flag
        for i, (key, bucket_data) in enumerate(keys_to_process):
            is_last = i == len(keys_to_process) - 1
            flush_bucket(key, bucket_data, is_last=is_last)

        # Invalidate shard paths cache so subsequent lookups see newly saved files
        if shards_saved:
            self._invalidate_shard_paths_cache(symbol, tf=tf_norm)

    def _persist_batch(
        self,
        symbol: str,
        batch: np.ndarray,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
        merge_cache: bool = False,
        last_refresh_ms: Optional[int] = None,
        defer_index: bool = False,
        skip_memory_retention: bool = False,
        source: Optional[str] = None,
    ) -> None:
        """Merge `batch` into memory (optional) and persist incrementally to disk.

        Args:
            defer_index: If True, defer index.json write until flush_deferred_index is called.
            skip_memory_retention: If True, skip memory retention enforcement to preserve
                full historical data in cache (useful for backtest data preparation).
            source: Optional ingestion provenance. ``"ws"`` records the latest
                finalized WebSocket timestamp without advancing REST refresh
                metadata.
        """
        if batch.size == 0:
            return
        arr = np.sort(_ensure_dtype(batch), order="ts")
        tf_norm = self._normalize_timeframe_arg(timeframe, tf)
        source_norm = str(source or "").lower()
        persist_before_cache = tf_norm == "1m" and source_norm == "ws"
        candle_content_changed = True
        replaces_synthetic = False
        if tf_norm == "1m":
            cached_before = np.sort(
                _ensure_dtype(self._ensure_symbol_cache(symbol)),
                order="ts",
            )
            incoming_unique = self._merge_overwrite(
                np.empty((0,), dtype=CANDLE_DTYPE),
                arr,
            )
            if cached_before.size and incoming_unique.size:
                positions = np.searchsorted(
                    cached_before["ts"].astype(np.int64),
                    incoming_unique["ts"].astype(np.int64),
                )
                if np.all(positions < cached_before.size):
                    matched = cached_before[positions]
                    candle_content_changed = not (
                        np.array_equal(
                            matched["ts"].astype(np.int64),
                            incoming_unique["ts"].astype(np.int64),
                        )
                        and np.array_equal(matched, incoming_unique)
                    )
            synthetic = self._synthetic_timestamps.get(symbol, set())
            replaces_synthetic = bool(
                synthetic
                and any(int(ts) in synthetic for ts in incoming_unique["ts"])
            )
        # WebSocket rows become canonical trading inputs only after the shard
        # write succeeds. A failed write must not leave restart-sensitive
        # candles or EMA state visible exclusively in RAM.
        if persist_before_cache:
            self._save_range_incremental(
                symbol,
                arr,
                timeframe=tf_norm,
                defer_index=defer_index,
            )
            durable = self._load_from_disk(
                symbol,
                int(arr[0]["ts"]),
                int(arr[-1]["ts"]),
                timeframe=tf_norm,
                merge_memory_cache=False,
            )
            durable_by_ts = (
                {int(row["ts"]): row for row in durable}
                if isinstance(durable, np.ndarray) and durable.size
                else {}
            )
            if any(
                int(row["ts"]) not in durable_by_ts
                or not np.array_equal(durable_by_ts[int(row["ts"])], row)
                for row in arr
            ):
                # A legacy-backed day may intentionally reject a primary
                # overlay. Treat any other successful-but-invisible save the
                # same way: do not expose restart-sensitive content in RAM.
                raise OSError("WebSocket candle persistence verification failed")
        if tf_norm == "1m" and (candle_content_changed or replaces_synthetic):
            self._projected_open_tail_ema_cache.pop(symbol, None)
            self._invalidate_ema_cache(symbol, timeframe="1m")

        # Update inception_ts if this is new earliest data for 1m (defer save until end)
        if tf_norm == "1m":
            self._maybe_update_inception_ts(symbol, arr, save=not defer_index)

        if merge_cache or tf_norm == "1m":
            merged_cache = self._merge_overwrite(self._ensure_symbol_cache(symbol), arr)
            self._cache[symbol] = merged_cache
            if not skip_memory_retention:
                try:
                    self._enforce_memory_retention(symbol)
                except Exception:
                    pass
            if last_refresh_ms is not None and merged_cache.size:
                self._set_last_refresh_meta(
                    symbol,
                    last_refresh_ms=last_refresh_ms,
                    last_final_ts=int(merged_cache[-1]["ts"]),
                )
            # Check if real data has replaced any previously synthetic timestamps
            # If so, mark that EMAs should be recomputed for this symbol
            self._check_synthetic_replacement(symbol, arr)

        if not persist_before_cache:
            self._save_range_incremental(
                symbol,
                arr,
                timeframe=tf_norm,
                defer_index=defer_index,
            )
        if tf_norm == "1m":
            if source_norm == "ws":
                idx = self._ensure_symbol_index(symbol, tf="1m")
                meta = idx.setdefault("meta", {})
                meta["last_ws_final_ts"] = max(
                    int(meta.get("last_ws_final_ts", 0) or 0),
                    int(arr[-1]["ts"]),
                )
                meta["last_ws_persist_ms"] = int(self._now_ms())
                self._index[f"{symbol}::1m"] = idx
                if not defer_index:
                    self._save_index(symbol, tf="1m")
            gaps_changed = self._trim_known_gaps_covered_by_rows(
                symbol,
                arr,
                defer_index=defer_index,
            )
            if gaps_changed:
                self._projected_open_tail_ema_cache.pop(symbol, None)
        observer = self._persist_batch_observer
        if observer is not None:
            try:
                observer(symbol, tf_norm, arr)
            except Exception:
                # Observability hooks must never break persistence or trading.
                return

    def _check_synthetic_replacement(self, symbol: str, real_data: np.ndarray) -> None:
        """Check if real data replaces previously synthetic timestamps and invalidate EMA cache if so."""
        if symbol not in self._synthetic_timestamps or not self._synthetic_timestamps[symbol]:
            return
        if real_data.size == 0:
            return

        real_ts_set = set(real_data["ts"].astype(np.int64).tolist())
        replaced = self._synthetic_timestamps[symbol] & real_ts_set
        if replaced:
            # Real data arrived for previously synthetic timestamps - invalidate EMA cache
            self._synthetic_timestamps[symbol] -= replaced
            self._invalidate_ema_cache(symbol)
            count = len(replaced)
            if self._candle_replace_batch_mode:
                # Batch mode: collect for aggregated summary later
                self._candle_replace_batch[symbol] = self._candle_replace_batch.get(symbol, 0) + count
            else:
                # Normal operation: log at DEBUG (individual messages are noisy)
                self.log.debug(
                    "[candle] %s: real data replaced %d synthetic candle%s, EMA cache invalidated",
                    symbol,
                    count,
                    "s" if count > 1 else "",
                )

    def _track_synthetic_timestamps(self, symbol: str, timestamps: List[int]) -> None:
        """Track runtime synthetic timestamps for replacement detection."""
        if not symbol or not timestamps:
            return
        ts_set = {int(ts) for ts in timestamps if int(ts) > 0}
        if not ts_set:
            return
        if symbol not in self._synthetic_timestamps:
            self._synthetic_timestamps[symbol] = set()
        self._synthetic_timestamps[symbol].update(ts_set)
        # Keep only the most recent week to bound memory usage.
        cutoff = self._now_ms() - 7 * 24 * 60 * ONE_MIN_MS
        self._synthetic_timestamps[symbol] = {
            ts for ts in self._synthetic_timestamps[symbol] if ts > cutoff
        }

    def _materialize_runtime_synthetic_gap(self, symbol: str, through_ts: int) -> int:
        """Deprecated open-tail synthetic materialization path.

        Open-ended tail gaps are intentionally not synthesized.  Synthetic zero
        candles are allowed only for bounded gaps where a real candle exists both
        before and after the missing span; that is handled by `standardize_gaps()`
        on the returned runtime array.
        """
        self._log(
            "debug",
            "runtime_synthetic_open_tail_skipped",
            symbol=symbol,
            through_ts=_floor_minute(int(through_ts)),
        )
        return 0

    def _invalidate_ema_cache(
        self, symbol: str, *, timeframe: Optional[str] = None
    ) -> None:
        """Invalidate cached EMA values for a symbol, optionally for one timeframe."""
        if symbol in self._ema_provisional_internal_gap_context:
            if timeframe is None:
                self._ema_provisional_internal_gap_context.pop(symbol, None)
            else:
                tf_key = str(_tf_to_ms(timeframe))
                retained_context = {
                    key: value
                    for key, value in self._ema_provisional_internal_gap_context[
                        symbol
                    ].items()
                    if len(key) < 3 or str(key[2]) != tf_key
                }
                if retained_context:
                    self._ema_provisional_internal_gap_context[symbol] = (
                        retained_context
                    )
                else:
                    self._ema_provisional_internal_gap_context.pop(symbol, None)
        if symbol not in self._ema_cache:
            return
        if timeframe is None:
            del self._ema_cache[symbol]
            return
        tf_key = str(_tf_to_ms(timeframe))
        retained = {
            key: value
            for key, value in self._ema_cache[symbol].items()
            if len(key) < 3 or str(key[2]) != tf_key
        }
        if retained:
            self._ema_cache[symbol] = retained
        else:
            del self._ema_cache[symbol]

    def _invalidate_tf_range_cache(
        self,
        symbol: str,
        *,
        timeframe: str,
        start_ts: int,
        end_ts: int,
    ) -> None:
        """Invalidate cached ranges overlapping persisted candles for one timeframe."""
        sym_cache = self._tf_range_cache.get(symbol)
        if not sym_cache:
            return
        overlapping = [
            key
            for key in sym_cache
            if str(key[0]) == str(timeframe)
            and int(key[1]) <= int(end_ts)
            and int(key[2]) >= int(start_ts)
        ]
        for key in overlapping:
            sym_cache.pop(key, None)
        if not sym_cache:
            self._tf_range_cache.pop(symbol, None)

    def needs_ema_recompute(self, symbol: str) -> bool:
        """Check if EMAs for a symbol should be recomputed due to synthetic data replacement.

        The bot can call this method to check if real data has replaced synthetic data
        since the last EMA computation, indicating EMAs should be recomputed.

        Returns True if:
        - EMA cache was invalidated due to synthetic replacement
        - Symbol has no cached EMAs (will be computed fresh anyway)

        Returns False if:
        - Symbol has valid cached EMAs computed from real data
        """
        # If there's no EMA cache for this symbol, it will be computed fresh
        if symbol not in self._ema_cache or not self._ema_cache[symbol]:
            return True
        # If the cache exists, it's valid (invalidation clears it)
        return False

    def clear_synthetic_tracking(self, symbol: Optional[str] = None) -> None:
        """Clear synthetic timestamp tracking for a symbol or all symbols.

        Useful after warmup completes or when the bot knows all real data has been fetched.
        """
        if symbol is None:
            self._synthetic_timestamps.clear()
        elif symbol in self._synthetic_timestamps:
            del self._synthetic_timestamps[symbol]

    def _merge_overwrite(self, existing: np.ndarray, new: np.ndarray) -> np.ndarray:
        """Merge two candle arrays by ts, preferring values from `new` on conflict."""
        if existing.size == 0:
            return np.sort(_ensure_dtype(new), order="ts")
        if new.size == 0:
            return np.sort(_ensure_dtype(existing), order="ts")
        a = _ensure_dtype(existing)
        b = _ensure_dtype(new)
        # Put existing first, then new; then keep last seen per ts to prefer new.
        # Sort the scalar timestamp vector rather than structured rows: NumPy
        # may use unspecified structured fields as tie-breakers even when
        # ``order="ts"``, which can otherwise let an older row win.
        combo = np.concatenate([a, b])
        order = np.argsort(combo["ts"].astype(np.int64, copy=False), kind="stable")
        combo = combo[order]
        ts = combo["ts"].astype(np.int64, copy=False)
        if combo.size <= 1:
            return combo
        # Deduplicate keeping the last occurrence per timestamp (vectorized).
        keep = np.empty(combo.size, dtype=bool)
        keep[:-1] = ts[:-1] != ts[1:]
        keep[-1] = True
        merged = combo[keep]
        # Enforce in-memory retention: keep only the latest N candles per symbol (applied by caller after assign)
        return merged

    def _latest_cached_ts_before(
        self, symbol: str, before_ts: int, *, timeframe: str
    ) -> Optional[int]:
        """Return latest known cached candle timestamp before `before_ts` without remote fetch."""
        tf_norm = self._normalize_timeframe_arg(timeframe, None)
        threshold = int(before_ts)
        best: Optional[int] = None
        if tf_norm == "1m":
            try:
                cached = self._cache.get(symbol)
                if cached is not None and cached.size:
                    arr = _ensure_dtype(cached)
                    ts = arr["ts"].astype(np.int64, copy=False)
                    prior = ts[ts < threshold]
                    if prior.size:
                        best = int(np.max(prior))
            except Exception:
                pass
        try:
            idx = self._ensure_symbol_index(symbol, timeframe=tf_norm)
            meta_last = idx.get("meta", {}).get("last_final_ts")
            if meta_last is not None:
                meta_i = int(meta_last)
                if 0 < meta_i < threshold:
                    best = meta_i if best is None else max(best, meta_i)
            shards = idx.get("shards", {})
            if isinstance(shards, dict):
                for shard_meta in shards.values():
                    if not isinstance(shard_meta, dict):
                        continue
                    max_ts = shard_meta.get("max_ts")
                    if max_ts is None:
                        continue
                    max_i = int(max_ts)
                    if 0 < max_i < threshold:
                        best = max_i if best is None else max(best, max_i)
        except Exception:
            pass
        return best

    # ----- Known gap helpers -----

    def _get_known_gaps_enhanced(self, symbol: str) -> List[GapEntry]:
        """Return known gaps as enhanced GapEntry objects with full metadata."""
        idx = self._ensure_symbol_index(symbol)
        gaps = idx.get("meta", {}).get("known_gaps", [])
        out: List[GapEntry] = []
        now_ms = self._now_ms()
        for it in gaps:
            try:
                # Support both old format [[start, end], ...] and new format [GapEntry, ...]
                if isinstance(it, dict):
                    # New enhanced format
                    entry: GapEntry = {
                        "start_ts": int(it.get("start_ts", 0)),
                        "end_ts": int(it.get("end_ts", 0)),
                        "retry_count": int(it.get("retry_count", 0)),
                        "reason": str(it.get("reason", GAP_REASON_AUTO)),
                        "added_at": int(it.get("added_at", now_ms)),
                        "last_retry_at": int(
                            it.get("last_retry_at", it.get("added_at", now_ms))
                        ),
                        "last_contextual_retry_at": int(
                            it.get("last_contextual_retry_at", 0)
                        ),
                    }
                    if entry["start_ts"] <= entry["end_ts"]:
                        out.append(entry)
                elif isinstance(it, (list, tuple)) and len(it) >= 2:
                    # Legacy format: auto-upgrade to enhanced
                    a, b = int(it[0]), int(it[1])
                    if a <= b:
                        out.append(
                            {
                                "start_ts": a,
                                "end_ts": b,
                                "retry_count": _GAP_MAX_RETRIES,  # Assume old gaps are persistent
                                "reason": GAP_REASON_AUTO,
                                "added_at": now_ms,
                                "last_retry_at": now_ms,
                            }
                        )
            except Exception:
                continue
        return out

    def _get_known_gaps(self, symbol: str) -> List[Tuple[int, int]]:
        """Return known gaps as simple (start_ts, end_ts) tuples for backward compatibility."""
        enhanced = self._get_known_gaps_enhanced(symbol)
        return [(g["start_ts"], g["end_ts"]) for g in enhanced]

    def _save_known_gaps_enhanced(
        self,
        symbol: str,
        gaps: List[GapEntry],
        *,
        defer_index: bool = False,
    ) -> None:
        """Save normalized gaps without broadening proof across reason boundaries.

        A terminal reason such as ``no_trades`` is evidence about an exact
        interval, not about an adjacent unresolved minute.  Normalize overlaps
        into disjoint segments and merge only segments carrying the same
        reason and retry epoch.  Terminal evidence wins an actual overlap with
        retryable metadata, but neither proof nor cooldown state expands across
        adjacency.
        """
        prepared = [
            dict(gap)
            for gap in gaps
            if int(gap.get("start_ts", 0)) <= int(gap.get("end_ts", -1))
        ]
        boundaries = sorted(
            {
                boundary
                for gap in prepared
                for boundary in (
                    int(gap["start_ts"]),
                    int(gap["end_ts"]) + ONE_MIN_MS,
                )
            }
        )
        starts = sorted(
            (int(gap["start_ts"]), index)
            for index, gap in enumerate(prepared)
        )
        active: list[tuple[int, int]] = []
        start_index = 0
        normalized: List[GapEntry] = []
        for segment_start, next_start in zip(boundaries, boundaries[1:]):
            segment_end = int(next_start) - ONE_MIN_MS
            if segment_start > segment_end:
                continue
            while (
                start_index < len(starts)
                and starts[start_index][0] <= segment_start
            ):
                _start_ts, gap_index = starts[start_index]
                terminal = int(
                    str(prepared[gap_index].get("reason", GAP_REASON_AUTO))
                    in _GAP_NON_EXPIRING_REASONS
                )
                # Highest proof class wins; within one class the later input
                # retains the legacy last-write-wins behavior.
                heapq.heappush(active, (-terminal, -gap_index))
                start_index += 1
            while active:
                winner_index = -active[0][1]
                if int(prepared[winner_index]["end_ts"]) >= segment_end:
                    break
                heapq.heappop(active)
            if not active:
                continue
            winner = prepared[-active[0][1]]
            normalized.append(
                {
                    **winner,
                    "start_ts": int(segment_start),
                    "end_ts": int(segment_end),
                }
            )

        gaps = sorted(normalized, key=lambda g: g["start_ts"])
        merged: List[GapEntry] = []
        for gap in gaps:
            previous = merged[-1] if merged else None
            same_reason = bool(
                previous
                and str(gap.get("reason", GAP_REASON_AUTO))
                == str(previous.get("reason", GAP_REASON_AUTO))
            )
            same_retry_epoch = bool(
                previous
                and int(gap.get("retry_count", 0))
                == int(previous.get("retry_count", 0))
                and int(gap.get("added_at", 0))
                == int(previous.get("added_at", 0))
                and int(gap.get("last_retry_at", gap.get("added_at", 0)))
                == int(
                    previous.get(
                        "last_retry_at", previous.get("added_at", 0)
                    )
                )
                and int(gap.get("last_contextual_retry_at", 0))
                == int(previous.get("last_contextual_retry_at", 0))
            )
            if (
                not merged
                or gap["start_ts"] > merged[-1]["end_ts"] + ONE_MIN_MS
                or not same_reason
                or not same_retry_epoch
            ):
                merged.append(gap)
            else:
                # Merge overlapping gaps, keeping max retry count and earliest added_at
                prev = merged[-1]
                merged[-1] = {
                    "start_ts": prev["start_ts"],
                    "end_ts": max(prev["end_ts"], gap["end_ts"]),
                    "retry_count": max(prev.get("retry_count", 0), gap.get("retry_count", 0)),
                    "reason": prev.get("reason", GAP_REASON_AUTO),
                    "added_at": min(prev.get("added_at", 0), gap.get("added_at", 0)),
                    "last_retry_at": max(
                        prev.get("last_retry_at", prev.get("added_at", 0)),
                        gap.get("last_retry_at", gap.get("added_at", 0)),
                    ),
                    "last_contextual_retry_at": max(
                        prev.get("last_contextual_retry_at", 0),
                        gap.get("last_contextual_retry_at", 0),
                    ),
                }
        idx = self._ensure_symbol_index(symbol)
        idx["meta"]["known_gaps"] = [
            {
                "start_ts": int(g["start_ts"]),
                "end_ts": int(g["end_ts"]),
                "retry_count": int(g.get("retry_count", 0)),
                "reason": str(g.get("reason", GAP_REASON_AUTO)),
                "added_at": int(g.get("added_at", 0)),
                "last_retry_at": int(
                    g.get("last_retry_at", g.get("added_at", 0))
                ),
                "last_contextual_retry_at": int(
                    g.get("last_contextual_retry_at", 0)
                ),
            }
            for g in merged
        ]
        self._index[symbol] = idx
        if not defer_index:
            self._save_index(symbol)

    def _trim_known_gaps_covered_by_rows(
        self,
        symbol: str,
        rows: np.ndarray,
        *,
        defer_index: bool = False,
    ) -> bool:
        """Remove authoritative 1m timestamps from persisted known-gap ranges."""
        if rows.size == 0:
            return False
        gaps = self._get_known_gaps_enhanced(symbol)
        if not gaps:
            return False
        timestamps = np.unique(np.asarray(rows["ts"], dtype=np.int64))
        if timestamps.size == 0:
            return False
        retained: List[GapEntry] = []
        changed = False
        for gap in gaps:
            start_ts = int(gap["start_ts"])
            end_ts = int(gap["end_ts"])
            left = int(np.searchsorted(timestamps, start_ts, side="left"))
            right = int(np.searchsorted(timestamps, end_ts, side="right"))
            covered = timestamps[left:right]
            if covered.size == 0:
                retained.append(gap)
                continue
            changed = True
            next_start = start_ts
            for covered_ts in covered:
                ts = int(covered_ts)
                if next_start <= ts - ONE_MIN_MS:
                    retained.append(
                        {
                            **gap,
                            "start_ts": next_start,
                            "end_ts": ts - ONE_MIN_MS,
                        }
                    )
                next_start = ts + ONE_MIN_MS
            if next_start <= end_ts:
                retained.append(
                    {
                        **gap,
                        "start_ts": next_start,
                        "end_ts": end_ts,
                    }
                )
        if changed:
            self._save_known_gaps_enhanced(
                symbol,
                retained,
                defer_index=defer_index,
            )
        return changed

    def _add_known_gap(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        reason: str = GAP_REASON_AUTO,
        increment_retry: bool = True,
        retry_count: Optional[int] = None,
    ) -> None:
        """Add or update a known gap with enhanced metadata.

        Existing overlap receives the requested retry update, while previously
        unseen prefix/suffix minutes start a fresh retry epoch. Adjacent ranges
        remain separate unless normalization proves all metadata identical.

        If retry_count reaches _GAP_MAX_RETRIES, the gap is considered persistent
        and will not be re-fetched unless force_refetch_gaps is used.

        Args:
            retry_count: If specified, set retry_count directly instead of incrementing.
                         Useful for pre-inception gaps that should be immediately persistent.
        """
        now_ms = self._now_ms()
        gaps = self._get_known_gaps_enhanced(symbol)
        requested_start = int(start_ts)
        requested_end = int(end_ts)
        covered: List[Tuple[int, int]] = []
        updates: List[GapEntry] = []
        persistent_transitions = 0

        # Update only exact overlaps with compatible evidence. The original gap
        # remains in the input; later update segments win their overlap during
        # normalization while preserving the original metadata on either side.
        # Retryable reasons may evolve between auto/fetch classifications, but
        # terminal proof must retain its exact range. Adjacency alone is not an
        # update: a newly finalized minute starts a fresh retry epoch and must
        # not inherit a neighboring gap's cooldown.
        for gap in gaps:
            existing_reason = str(gap.get("reason", GAP_REASON_AUTO))
            compatible_reason = existing_reason == reason or (
                existing_reason not in _GAP_NON_EXPIRING_REASONS
                and reason not in _GAP_NON_EXPIRING_REASONS
            )
            overlap_start = max(int(gap["start_ts"]), requested_start)
            overlap_end = min(int(gap["end_ts"]), requested_end)
            if not compatible_reason or overlap_start > overlap_end:
                continue
            covered.append((overlap_start, overlap_end))

            updated_gap: GapEntry = {
                **gap,
                "start_ts": overlap_start,
                "end_ts": overlap_end,
            }
            previous_retry_count = int(gap.get("retry_count", 0))
            retry_due = self._should_retry_gap(gap, now_ms=now_ms)
            if retry_count is not None:
                updated_gap["retry_count"] = retry_count
                updated_gap["last_retry_at"] = now_ms
                if (
                    retry_count >= _GAP_MAX_RETRIES
                    and previous_retry_count < _GAP_MAX_RETRIES
                ):
                    updated_gap["added_at"] = now_ms
            elif increment_retry and retry_due:
                # Cap retry_count at _GAP_MAX_RETRIES to prevent unbounded growth
                # without reverting a recent Hyperliquid gap to the ordinary
                # retry cadence. Other expired persistent gaps start a fresh
                # retry cycle under the existing retention contract.
                if previous_retry_count >= _GAP_MAX_RETRIES:
                    new_retry_count = (
                        _GAP_MAX_RETRIES
                        if self._is_recent_hyperliquid_gap(
                            gap, now_ms=now_ms
                        )
                        else 1
                    )
                else:
                    new_retry_count = previous_retry_count + 1
                updated_gap["retry_count"] = min(
                    new_retry_count, _GAP_MAX_RETRIES
                )
                updated_gap["last_retry_at"] = now_ms
                updated_gap["added_at"] = now_ms
            if reason != GAP_REASON_AUTO:
                updated_gap["reason"] = reason
            if (
                int(updated_gap.get("retry_count", 0)) >= _GAP_MAX_RETRIES
                and previous_retry_count < _GAP_MAX_RETRIES
                and str(updated_gap.get("reason", GAP_REASON_AUTO))
                != "pre_inception"
            ):
                persistent_transitions += 1
            updates.append(updated_gap)

        uncovered: List[Tuple[int, int]] = []
        cursor = requested_start
        for covered_start, covered_end in sorted(covered):
            if covered_end < cursor:
                continue
            if covered_start > cursor:
                uncovered.append(
                    (cursor, min(requested_end, covered_start - ONE_MIN_MS))
                )
            cursor = max(cursor, covered_end + ONE_MIN_MS)
            if cursor > requested_end:
                break
        if cursor <= requested_end:
            uncovered.append((cursor, requested_end))

        initial_retry = (
            retry_count
            if retry_count is not None
            else (1 if increment_retry else 0)
        )
        fresh_gaps: List[GapEntry] = []
        for fresh_start, fresh_end in uncovered:
            new_gap: GapEntry = {
                "start_ts": fresh_start,
                "end_ts": fresh_end,
                "retry_count": initial_retry,
                "reason": reason,
                "added_at": now_ms,
                "last_retry_at": now_ms,
            }
            fresh_gaps.append(new_gap)
            self._log(
                "debug",
                "gap_added",
                symbol=symbol,
                start_ts=fresh_start,
                end_ts=fresh_end,
                reason=reason,
                retry_count=new_gap["retry_count"],
            )

        gaps.extend(updates)
        gaps.extend(fresh_gaps)
        if persistent_transitions:
            # Track transitions for the existing throttled summary logger.
            if not hasattr(self, "_persistent_gap_summary"):
                self._persistent_gap_summary: Dict[str, int] = {}
            self._persistent_gap_summary[symbol] = (
                self._persistent_gap_summary.get(symbol, 0)
                + persistent_transitions
            )

        self._save_known_gaps_enhanced(symbol, gaps)
        # A newly recorded or expanded 1m gap changes whether EMA inputs are
        # authoritative even when the candle tail/end timestamp is unchanged.
        # Do not let values computed before that evidence survive in RAM.
        self._invalidate_ema_cache(symbol, timeframe="1m")
        self._projected_open_tail_ema_cache.pop(symbol, None)

    def _record_verified_gap(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        reason: str = GAP_REASON_NO_TRADES,
    ) -> None:
        """Record a gap as verified (no data on exchange), so we don't retry it."""
        if start_ts > end_ts:
            return
        self._add_known_gap(
            symbol,
            int(start_ts),
            int(end_ts),
            reason=reason,
            increment_retry=False,
            retry_count=_GAP_MAX_RETRIES,
        )

    def _kucoin_contextual_retry_due(
        self,
        gap: GapEntry,
        *,
        now_ms: Optional[int] = None,
    ) -> bool:
        """Return whether a KuCoin sparse-gap boundary proof is due.

        Boundary proof is independent of the ordinary missing-range retry
        count.  A newly discovered sparse interval should be proved as soon as
        real rows bracket it, rather than spending three empty-range retries
        and then waiting for the persistent-gap cooldown.  Failed boundary
        proofs retain their own cooldown so routine candle reads cannot cause
        a REST treadmill.
        """
        if str(gap.get("reason", GAP_REASON_AUTO)) not in {
            GAP_REASON_AUTO,
            GAP_REASON_FETCH_FAILED,
        }:
            return False
        now = self._now_ms() if now_ms is None else int(now_ms)
        last_retry_at = int(gap.get("last_contextual_retry_at", 0))
        return (
            last_retry_at <= 0
            or now - last_retry_at >= _GAP_PERSISTENT_RETRY_MS
        )

    def _defer_kucoin_contextual_gap_retry(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        now_ms: Optional[int] = None,
    ) -> bool:
        """Restart only the KuCoin contextual-proof cooldown after failure."""
        now = self._now_ms() if now_ms is None else int(now_ms)
        changed = False
        gaps = self._get_known_gaps_enhanced(symbol)
        for gap in gaps:
            if (
                int(gap["start_ts"]) <= int(end_ts)
                and int(gap["end_ts"]) >= int(start_ts)
                and str(gap.get("reason", GAP_REASON_AUTO))
                in {GAP_REASON_AUTO, GAP_REASON_FETCH_FAILED}
            ):
                gap["last_contextual_retry_at"] = now
                changed = True
        if changed:
            self._save_known_gaps_enhanced(symbol, gaps)
            self._log(
                "debug",
                "kucoin_contextual_gap_verification_deferred",
                symbol=symbol,
                start_ts=int(start_ts),
                end_ts=int(end_ts),
                retry_after_ms=now + _GAP_PERSISTENT_RETRY_MS,
            )
        return changed

    def _is_recent_hyperliquid_gap(
        self,
        gap: GapEntry,
        *,
        now_ms: int,
    ) -> bool:
        exid = str(self._ex_id or "").lower()
        reason = str(gap.get("reason", GAP_REASON_AUTO))
        start_ts = int(gap.get("start_ts", 0))
        end_ts = int(gap.get("end_ts", 0))
        span_ms = end_ts - start_ts + ONE_MIN_MS
        return (
            "hyperliquid" in exid
            and reason in {GAP_REASON_AUTO, GAP_REASON_FETCH_FAILED}
            and 0 < span_ms <= _HYPERLIQUID_RECENT_GAP_MAX_SPAN_MS
            and end_ts >= int(now_ms) - _HYPERLIQUID_RECENT_GAP_HORIZON_MS
        )

    def _known_gap_retry_deferred_at(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        now_ms: Optional[int] = None,
    ) -> bool:
        """Whether a fetch is covered by a known gap whose retry is not due.

        A range extending beyond a deferred gap is not wholly deferred; callers
        must use ``_fetch_start_after_deferred_gap_prefix`` to skip the gap while
        still fetching the newly finalized suffix.
        """
        return (
            self._fetch_start_after_deferred_gap_prefix(
                symbol,
                start_ts,
                end_ts,
                now_ms=now_ms,
            )
            is None
        )

    def _fetch_start_after_deferred_gap_prefix(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        now_ms: Optional[int] = None,
    ) -> Optional[int]:
        """Skip only a non-due known-gap prefix and preserve any later suffix."""
        if now_ms is None:
            now_ms = self._now_ms()
        fetch_start = int(start_ts)
        fetch_end = int(end_ts)
        while fetch_start <= fetch_end:
            deferred_gap_end = None
            for gap in self._get_known_gaps_enhanced(symbol):
                gap_start = int(gap["start_ts"])
                gap_end = int(gap["end_ts"])
                if (
                    gap_start <= fetch_start <= gap_end
                    and not self._should_retry_gap(gap, now_ms=now_ms)
                ):
                    deferred_gap_end = (
                        gap_end
                        if deferred_gap_end is None
                        else max(deferred_gap_end, gap_end)
                    )
            if deferred_gap_end is None:
                return fetch_start
            fetch_start = int(deferred_gap_end) + ONE_MIN_MS
        return None

    def _unverified_gap_ranges(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
    ) -> List[Tuple[int, int]]:
        """Return unresolved unknown gaps that must remain unavailable."""
        ranges: List[Tuple[int, int]] = []
        for gap in self._get_known_gaps_enhanced(symbol):
            if str(gap.get("reason", GAP_REASON_AUTO)) not in {
                GAP_REASON_AUTO,
                GAP_REASON_FETCH_FAILED,
            }:
                continue
            overlap_start = max(int(start_ts), int(gap["start_ts"]))
            overlap_end = min(int(end_ts), int(gap["end_ts"]))
            if overlap_start <= overlap_end:
                ranges.append((overlap_start, overlap_end))
        return ranges

    def _unverified_uncovered_gap_ranges(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
    ) -> List[Tuple[int, int]]:
        """Return full still-missing spans from unknown gaps overlapping a range.

        Known-gap metadata may temporarily cover minutes which a later refresh
        has already populated. Width-sensitive provisional policy must measure
        the remaining contiguous outage, not either the caller-clipped overlap
        or stale outer metadata bounds.
        """
        cached = self._cache.get(symbol)
        if cached is None or cached.size == 0:
            cached = np.empty((0,), dtype=CANDLE_DTYPE)
        else:
            cached = np.sort(_ensure_dtype(cached), order="ts")
            provisional_ts = self._synthetic_timestamps.get(symbol, set())
            if provisional_ts:
                cached = cached[
                    ~np.isin(
                        cached["ts"].astype(np.int64),
                        np.asarray(tuple(provisional_ts), dtype=np.int64),
                    )
                ]

        missing: List[Tuple[int, int]] = []
        for gap in self._get_known_gaps_enhanced(symbol):
            if str(gap.get("reason", GAP_REASON_AUTO)) not in {
                GAP_REASON_AUTO,
                GAP_REASON_FETCH_FAILED,
            }:
                continue
            gap_start = int(gap["start_ts"])
            gap_end = int(gap["end_ts"])
            if gap_start > int(end_ts) or gap_end < int(start_ts):
                continue
            gap_rows = self._slice_ts_range(
                cached,
                gap_start,
                gap_end,
                assume_sorted=True,
            )
            missing.extend(self._missing_spans(gap_rows, gap_start, gap_end))

        merged: List[Tuple[int, int]] = []
        for gap_start, gap_end in sorted(missing):
            if merged and int(gap_start) <= int(merged[-1][1]) + ONE_MIN_MS:
                merged[-1] = (
                    int(merged[-1][0]),
                    max(int(merged[-1][1]), int(gap_end)),
                )
            else:
                merged.append((int(gap_start), int(gap_end)))
        return [
            (gap_start, gap_end)
            for gap_start, gap_end in merged
            if gap_start <= int(end_ts) and gap_end >= int(start_ts)
        ]

    def _due_unverified_gap_ranges(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        now_ms: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """Return retry-due unknown-gap intersections for one attempted range."""
        if now_ms is None:
            now_ms = self._now_ms()
        ranges: List[Tuple[int, int]] = []
        for gap in self._get_known_gaps_enhanced(symbol):
            if str(gap.get("reason", GAP_REASON_AUTO)) not in {
                GAP_REASON_AUTO,
                GAP_REASON_FETCH_FAILED,
            }:
                continue
            if not self._should_retry_gap(gap, now_ms=now_ms):
                continue
            overlap_start = max(int(start_ts), int(gap["start_ts"]))
            overlap_end = min(int(end_ts), int(gap["end_ts"]))
            if overlap_start <= overlap_end:
                ranges.append((overlap_start, overlap_end))
        return ranges

    def _stamp_unresolved_gap_attempts(
        self,
        symbol: str,
        attempted_ranges: List[Tuple[int, int]],
    ) -> None:
        """Record retry time for every still-missing minute in attempted known gaps."""
        if not attempted_ranges:
            return
        cached = _ensure_dtype(
            self._cache.get(symbol, np.empty((0,), dtype=CANDLE_DTYPE))
        )
        for start_ts, end_ts in attempted_ranges:
            present = (
                self._slice_ts_range(cached, int(start_ts), int(end_ts))
                if cached.size
                else cached
            )
            for missing_start, missing_end in self._missing_spans(
                present, int(start_ts), int(end_ts)
            ):
                self._add_known_gap(
                    symbol,
                    int(missing_start),
                    int(missing_end),
                    reason=GAP_REASON_FETCH_FAILED,
                    increment_retry=True,
                )

    def _fetch_ranges_excluding_deferred_gaps(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        now_ms: Optional[int] = None,
    ) -> List[Tuple[int, int]]:
        """Split a remote range around every known gap whose retry is not due."""
        if now_ms is None:
            now_ms = self._now_ms()
        ranges = [(int(start_ts), int(end_ts))]
        for gap in sorted(
            self._get_known_gaps_enhanced(symbol),
            key=lambda item: int(item["start_ts"]),
        ):
            if self._should_retry_gap(gap, now_ms=now_ms):
                continue
            gap_start = int(gap["start_ts"])
            gap_end = int(gap["end_ts"])
            next_ranges: List[Tuple[int, int]] = []
            for range_start, range_end in ranges:
                if gap_end < range_start or gap_start > range_end:
                    next_ranges.append((range_start, range_end))
                    continue
                if range_start < gap_start:
                    next_ranges.append((range_start, gap_start - ONE_MIN_MS))
                if gap_end < range_end:
                    next_ranges.append((gap_end + ONE_MIN_MS, range_end))
            ranges = next_ranges
            if not ranges:
                break
        return ranges

    def _should_retry_gap(
        self,
        gap: GapEntry,
        *,
        now_ms: Optional[int] = None,
    ) -> bool:
        """Check if a gap should be retried."""
        if now_ms is None:
            now_ms = self._now_ms()
        if int(gap.get("retry_count", 0)) < _GAP_MAX_RETRIES:
            if self._is_recent_hyperliquid_gap(gap, now_ms=now_ms):
                last_retry_at = int(
                    gap.get("last_retry_at", gap.get("added_at", now_ms))
                )
                return (
                    int(now_ms) - last_retry_at
                    >= _HYPERLIQUID_RECENT_GAP_RETRY_MS
                )
            return True
        return self._persistent_gap_retry_due(gap, now_ms=now_ms)

    def _persistent_gap_retry_due(self, gap: GapEntry, *, now_ms: Optional[int] = None) -> bool:
        reason = str(gap.get("reason", GAP_REASON_AUTO))
        if reason in _GAP_NON_EXPIRING_REASONS:
            return False
        if int(gap.get("retry_count", 0)) < _GAP_MAX_RETRIES:
            return False
        if now_ms is None:
            now_ms = self._now_ms()
        last_retry_at = int(
            gap.get("last_retry_at", gap.get("added_at", now_ms))
        )
        retry_ms = (
            _HYPERLIQUID_RECENT_PERSISTENT_GAP_RETRY_MS
            if self._is_recent_hyperliquid_gap(gap, now_ms=now_ms)
            else _GAP_PERSISTENT_RETRY_MS
        )
        return now_ms - last_retry_at >= retry_ms

    def clear_known_gaps(
        self,
        symbol: str,
        *,
        date_range: Optional[Tuple[int, int]] = None,
    ) -> int:
        """Clear known gaps for a symbol, optionally filtered by date range.

        Args:
            symbol: The symbol to clear gaps for
            date_range: Optional (start_ts, end_ts) to only clear gaps within this range

        Returns:
            Number of gaps cleared
        """
        gaps = self._get_known_gaps_enhanced(symbol)
        if not gaps:
            return 0

        if date_range is None:
            # Clear all gaps
            cleared = len(gaps)
            idx = self._ensure_symbol_index(symbol)
            idx["meta"]["known_gaps"] = []
            self._index[symbol] = idx
            self._save_index(symbol)
            self._log(
                "info",
                "gaps_cleared",
                symbol=symbol,
                cleared_count=cleared,
            )
            return cleared

        # Clear only gaps overlapping with date_range
        range_start, range_end = date_range
        remaining = []
        cleared = 0
        for gap in gaps:
            if gap["end_ts"] < range_start or gap["start_ts"] > range_end:
                # Outside range - keep
                remaining.append(gap)
            else:
                cleared += 1

        if cleared > 0:
            self._save_known_gaps_enhanced(symbol, remaining)
            self._log(
                "info",
                "gaps_cleared",
                symbol=symbol,
                cleared_count=cleared,
                date_range_start=range_start,
                date_range_end=range_end,
            )
        return cleared

    def get_gap_summary(self, symbol: str) -> Dict[str, Any]:
        """Get summary of known gaps for a symbol.

        Returns:
            Dict with keys:
            - total_gaps: Number of gap entries
            - total_minutes: Total minutes of gaps
            - persistent_gaps: Gaps with retry_count >= max
            - retryable_gaps: Gaps with retry_count < max
            - by_reason: Dict of reason -> count
            - gaps: List of gap details
        """
        gaps = self._get_known_gaps_enhanced(symbol)
        if not gaps:
            return {
                "total_gaps": 0,
                "total_minutes": 0,
                "persistent_gaps": 0,
                "retryable_gaps": 0,
                "by_reason": {},
                "gaps": [],
            }

        total_minutes = sum((g["end_ts"] - g["start_ts"]) // ONE_MIN_MS + 1 for g in gaps)
        persistent = sum(
            1
            for g in gaps
            if g.get("retry_count", 0) >= _GAP_MAX_RETRIES and not self._should_retry_gap(g)
        )
        retryable = len(gaps) - persistent

        by_reason: Dict[str, int] = {}
        for g in gaps:
            reason = g.get("reason", GAP_REASON_AUTO)
            by_reason[reason] = by_reason.get(reason, 0) + 1

        return {
            "total_gaps": len(gaps),
            "total_minutes": total_minutes,
            "persistent_gaps": persistent,
            "retryable_gaps": retryable,
            "by_reason": by_reason,
            "gaps": [
                {
                    "start_ts": g["start_ts"],
                    "end_ts": g["end_ts"],
                    "minutes": (g["end_ts"] - g["start_ts"]) // ONE_MIN_MS + 1,
                    "retry_count": g.get("retry_count", 0),
                    "reason": g.get("reason", GAP_REASON_AUTO),
                    "persistent": g.get("retry_count", 0) >= _GAP_MAX_RETRIES
                    and not self._should_retry_gap(g),
                }
                for g in gaps
            ],
        }

    def _missing_spans(self, arr: np.ndarray, start_ts: int, end_ts: int) -> List[Tuple[int, int]]:
        """Return list of inclusive [gap_start, gap_end] minute-aligned spans missing in arr."""
        spans: List[Tuple[int, int]] = []
        if start_ts > end_ts:
            return spans
        if arr.size == 0:
            return [(start_ts, end_ts)]
        ts = np.asarray(arr["ts"], dtype=np.int64)
        ts = ts[(ts >= start_ts) & (ts <= end_ts)]
        if ts.size == 0:
            return [(start_ts, end_ts)]
        # head gap
        if ts[0] > start_ts:
            spans.append((start_ts, int(ts[0] - ONE_MIN_MS)))
        # middle gaps
        for i in range(len(ts) - 1):
            if ts[i + 1] - ts[i] > ONE_MIN_MS:
                spans.append((int(ts[i] + ONE_MIN_MS), int(ts[i + 1] - ONE_MIN_MS)))
        # tail gap
        if ts[-1] < end_ts:
            spans.append((int(ts[-1] + ONE_MIN_MS), end_ts))
        return spans

    @staticmethod
    def _missing_spans_step(
        arr: np.ndarray, start_ts: int, end_ts: int, step_ms: int
    ) -> List[Tuple[int, int]]:
        """Return list of inclusive [gap_start, gap_end] spans missing in arr at step_ms."""
        spans: List[Tuple[int, int]] = []
        if start_ts > end_ts or step_ms <= 0:
            return spans
        if arr.size == 0:
            return [(start_ts, end_ts)]
        ts = np.asarray(arr["ts"], dtype=np.int64)
        ts = ts[(ts >= start_ts) & (ts <= end_ts)]
        if ts.size == 0:
            return [(start_ts, end_ts)]
        ts = np.sort(ts)
        # head gap
        if ts[0] > start_ts:
            spans.append((int(start_ts), int(ts[0] - step_ms)))
        # middle gaps
        for i in range(len(ts) - 1):
            if ts[i + 1] - ts[i] > step_ms:
                spans.append((int(ts[i] + step_ms), int(ts[i + 1] - step_ms)))
        # tail gap
        if ts[-1] < end_ts:
            spans.append((int(ts[-1] + step_ms), int(end_ts)))
        return spans

    def check_disk_coverage(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
        log_level: str = "info",
        max_span_log: int = 3,
    ) -> Dict[str, Any]:
        """Check whether disk cache fully covers [start_ts, end_ts] for a symbol.

        Returns a dict with:
            ok, missing_spans, missing_candles, loaded_rows, timeframe.
        """
        tf_norm = self._normalize_timeframe_arg(timeframe, tf)
        step_ms = _tf_to_ms(tf_norm)
        if step_ms <= 0:
            step_ms = ONE_MIN_MS
        s_ts = (int(start_ts) // step_ms) * step_ms
        e_ts = (int(end_ts) // step_ms) * step_ms
        if s_ts > e_ts:
            return {
                "ok": True,
                "missing_spans": [],
                "missing_candles": 0,
                "loaded_rows": 0,
                "timeframe": tf_norm,
            }

        arr = self._load_from_disk(symbol, s_ts, e_ts, timeframe=tf_norm)
        if arr is None or arr.size == 0:
            missing = [(s_ts, e_ts)]
            loaded_rows = 0
        else:
            sub = self._slice_ts_range(arr, s_ts, e_ts)
            missing = (
                self._missing_spans(sub, s_ts, e_ts)
                if step_ms == ONE_MIN_MS
                else self._missing_spans_step(sub, s_ts, e_ts, step_ms)
            )
            loaded_rows = int(sub.shape[0]) if sub is not None else 0

        missing_candles = 0
        if missing:
            missing_candles = int(sum((e - s) // step_ms + 1 for s, e in missing))
            top_parts = []
            for s, e in missing[: max(1, int(max_span_log))]:
                top_parts.append(f"{self._fmt_ts(int(s))} to {self._fmt_ts(int(e))}")
            top_str = ", ".join(top_parts)
            if len(missing) > max_span_log:
                top_str = f"{top_str} (+{len(missing) - max_span_log} more)"
            self._log(
                log_level,
                "disk_coverage_missing",
                symbol=symbol,
                timeframe=tf_norm,
                start_ts=s_ts,
                end_ts=e_ts,
                missing_spans=len(missing),
                missing_candles=missing_candles,
                top=top_str,
            )
        return {
            "ok": len(missing) == 0,
            "missing_spans": missing,
            "missing_candles": missing_candles,
            "loaded_rows": loaded_rows,
            "timeframe": tf_norm,
        }

    def get_completed_candle_health(
        self,
        symbol: str,
        windows: Optional[Dict[str, int]] = None,
        *,
        now_ms: Optional[int] = None,
        max_span_log: int = 3,
    ) -> Dict[str, Any]:
        """Return read-only diagnostics for completed-candle coverage.

        This method never fetches remote data and never includes the current
        in-progress timeframe bucket. It inspects disk cache plus runtime 1m
        cache and reports whether the requested completed candle windows are
        covered.
        """
        ts_now = _utc_now_ms() if now_ms is None else int(now_ms)
        requested = windows or {"1m": 1, "15m": 1, "1h": 1}
        normalized_windows: Dict[str, Dict[str, Any]] = {}
        for tf_raw, raw_spec in requested.items():
            tf_norm = self._normalize_timeframe_arg(str(tf_raw), None)
            required = True
            count_raw = raw_spec
            if isinstance(raw_spec, dict):
                count_raw = raw_spec.get("candles", raw_spec.get("required_candles", 1))
                required = bool(raw_spec.get("required", True))
            try:
                count = int(math.ceil(float(count_raw)))
            except Exception:
                count = 1
            normalized_windows[tf_norm] = {"candles": max(1, count), "required": required}

        tf_reports: Dict[str, Dict[str, Any]] = {}
        overall_ok = True
        for tf_norm, spec in sorted(
            normalized_windows.items(), key=lambda item: _tf_to_ms(item[0])
        ):
            required_candles = int(spec["candles"])
            required = bool(spec["required"])
            step_ms = _tf_to_ms(tf_norm)
            if step_ms <= 0:
                step_ms = ONE_MIN_MS
            latest_expected = (ts_now // step_ms) * step_ms - step_ms
            if latest_expected < 0:
                start_ts = 0
                end_ts = -1
            else:
                end_ts = int(latest_expected)
                start_ts = int(max(0, end_ts - (required_candles - 1) * step_ms))

            disk_arr = np.empty((0,), dtype=CANDLE_DTYPE)
            runtime_arr = np.empty((0,), dtype=CANDLE_DTYPE)
            combined = np.empty((0,), dtype=CANDLE_DTYPE)
            if end_ts >= start_ts:
                try:
                    loaded = self._load_from_disk(symbol, start_ts, end_ts, timeframe=tf_norm)
                    if loaded is not None and loaded.size:
                        disk_arr = self._slice_ts_range(_ensure_dtype(loaded), start_ts, end_ts)
                except Exception as exc:
                    self._log(
                        "debug",
                        "candle_health_disk_load_failed",
                        symbol=symbol,
                        timeframe=tf_norm,
                        error_type=bounded_exception_type(exc),
                    )
                    disk_arr = np.empty((0,), dtype=CANDLE_DTYPE)

                if step_ms == ONE_MIN_MS:
                    try:
                        cached = self._cache.get(symbol)
                        if cached is not None and cached.size:
                            runtime_arr = self._slice_ts_range(
                                _ensure_dtype(cached), start_ts, end_ts
                            )
                    except Exception as exc:
                        self._log(
                            "debug",
                            "candle_health_runtime_cache_failed",
                            symbol=symbol,
                            timeframe=tf_norm,
                            error_type=bounded_exception_type(exc),
                        )
                        runtime_arr = np.empty((0,), dtype=CANDLE_DTYPE)
                combined = self._merge_overwrite(disk_arr, runtime_arr)
                combined = self._slice_ts_range(combined, start_ts, end_ts, assume_sorted=True)

            missing = (
                []
                if end_ts < start_ts
                else (
                    self._missing_spans(combined, start_ts, end_ts)
                    if step_ms == ONE_MIN_MS
                    else self._missing_spans_step(combined, start_ts, end_ts, step_ms)
                )
            )
            missing_candles = int(sum((e - s) // step_ms + 1 for s, e in missing))
            verified_no_trade_spans: List[Tuple[int, int]] = []
            deferred_missing_spans: List[Tuple[int, int]] = []
            refreshable_missing_spans: List[Tuple[int, int]] = list(missing)
            if step_ms == ONE_MIN_MS and missing:
                known_gaps = self._get_known_gaps_enhanced(symbol)

                def merge_spans(
                    spans: Iterable[Tuple[int, int]],
                ) -> List[Tuple[int, int]]:
                    merged: List[Tuple[int, int]] = []
                    for span_start, span_end in sorted(
                        (int(a), int(b)) for a, b in spans if int(a) <= int(b)
                    ):
                        if merged and span_start <= merged[-1][1] + ONE_MIN_MS:
                            merged[-1] = (
                                merged[-1][0],
                                max(merged[-1][1], span_end),
                            )
                        else:
                            merged.append((span_start, span_end))
                    return merged

                def intersect_missing(
                    masks: Iterable[Tuple[int, int]],
                ) -> List[Tuple[int, int]]:
                    missing_sorted = merge_spans(missing)
                    masks_sorted = merge_spans(masks)
                    intersections: List[Tuple[int, int]] = []
                    missing_index = 0
                    mask_index = 0
                    while (
                        missing_index < len(missing_sorted)
                        and mask_index < len(masks_sorted)
                    ):
                        missing_start, missing_end = missing_sorted[missing_index]
                        mask_start, mask_end = masks_sorted[mask_index]
                        overlap_start = max(missing_start, mask_start)
                        overlap_end = min(missing_end, mask_end)
                        if overlap_start <= overlap_end:
                            intersections.append((overlap_start, overlap_end))
                        if missing_end < mask_end:
                            missing_index += 1
                        else:
                            mask_index += 1
                    return intersections

                def subtract_spans(
                    source: Iterable[Tuple[int, int]],
                    masks: Iterable[Tuple[int, int]],
                ) -> List[Tuple[int, int]]:
                    source_sorted = merge_spans(source)
                    masks_sorted = merge_spans(masks)
                    remaining: List[Tuple[int, int]] = []
                    mask_index = 0
                    for span_start, span_end in source_sorted:
                        while (
                            mask_index < len(masks_sorted)
                            and masks_sorted[mask_index][1] < span_start
                        ):
                            mask_index += 1
                        cursor = span_start
                        scan_index = mask_index
                        while (
                            scan_index < len(masks_sorted)
                            and masks_sorted[scan_index][0] <= span_end
                        ):
                            mask_start, mask_end = masks_sorted[scan_index]
                            if mask_end < cursor:
                                scan_index += 1
                                continue
                            if mask_start > cursor:
                                remaining.append(
                                    (cursor, min(span_end, mask_start - ONE_MIN_MS))
                                )
                            cursor = max(cursor, mask_end + ONE_MIN_MS)
                            if cursor > span_end:
                                break
                            scan_index += 1
                        if cursor <= span_end:
                            remaining.append((cursor, span_end))
                        mask_index = scan_index
                    return remaining

                verified_masks: List[Tuple[int, int]] = []
                deferred_masks: List[Tuple[int, int]] = []
                combined_ts = combined["ts"].astype(np.int64, copy=False)
                for gap in known_gaps:
                    gap_start = int(gap["start_ts"])
                    gap_end = int(gap["end_ts"])
                    if gap_end < start_ts or gap_start > end_ts:
                        continue
                    clipped_gap = (max(gap_start, start_ts), min(gap_end, end_ts))
                    reason = str(gap.get("reason", GAP_REASON_AUTO))
                    if reason == GAP_REASON_NO_TRADES:
                        # A continuity candle needs real price on both sides.
                        # Open tails remain unavailable so a delayed exchange
                        # candle can replace them authoritatively.
                        if gap_end >= end_ts:
                            continue
                        prior_index = int(
                            np.searchsorted(combined_ts, gap_start, side="left")
                        )
                        if prior_index == 0:
                            prior_ts = self._latest_cached_ts_before(
                                symbol, gap_start, timeframe=tf_norm
                            )
                            if prior_ts is None:
                                continue
                        successor_index = int(
                            np.searchsorted(combined_ts, gap_end, side="right")
                        )
                        if successor_index < combined_ts.size:
                            verified_masks.append(clipped_gap)
                        continue

                    retry_due = self._should_retry_gap(gap, now_ms=ts_now)
                    if (
                        not retry_due
                        and isinstance(self._ex_id, str)
                        and "kucoin" in self._ex_id.lower()
                        and reason in {GAP_REASON_AUTO, GAP_REASON_FETCH_FAILED}
                        and self._kucoin_contextual_retry_due(gap, now_ms=ts_now)
                    ):
                        retry_due = True
                    if not retry_due:
                        deferred_masks.append(clipped_gap)

                verified_no_trade_spans = intersect_missing(verified_masks)
                deferred_missing_spans = intersect_missing(deferred_masks)
                refreshable_missing_spans = subtract_spans(
                    missing,
                    [*verified_no_trade_spans, *deferred_missing_spans],
                )

            def span_candle_count(spans: Iterable[Tuple[int, int]]) -> int:
                return int(sum((e - s) // step_ms + 1 for s, e in spans))

            verified_no_trade_missing_candles = span_candle_count(
                verified_no_trade_spans
            )
            deferred_missing_candles = span_candle_count(deferred_missing_spans)
            refreshable_missing_candles = span_candle_count(
                refreshable_missing_spans
            )
            last_cached_ts: Optional[int] = None
            if combined.size:
                last_cached_ts = int(np.max(combined["ts"].astype(np.int64)))
            last_disk_ts: Optional[int] = None
            if disk_arr.size:
                last_disk_ts = int(np.max(disk_arr["ts"].astype(np.int64)))
            last_runtime_ts: Optional[int] = None
            if runtime_arr.size:
                last_runtime_ts = int(np.max(runtime_arr["ts"].astype(np.int64)))
            last_ws_final_ts = (
                self.get_last_live_ws_ohlcv_ts(symbol)
                if step_ms == ONE_MIN_MS
                else 0
            )
            last_ws_persist_ms = (
                self.get_last_live_ws_persist_ms(symbol)
                if step_ms == ONE_MIN_MS
                else 0
            )

            synthetic_count = 0
            if step_ms == ONE_MIN_MS:
                try:
                    synthetic_count = int(
                        sum(
                            1
                            for ts in self._synthetic_timestamps.get(symbol, set())
                            if int(start_ts) <= int(ts) <= int(end_ts)
                        )
                    )
                except Exception:
                    synthetic_count = 0

            top_spans = [
                {
                    "start_ts": int(s),
                    "end_ts": int(e),
                    "start": self._fmt_ts(int(s)),
                    "end": self._fmt_ts(int(e)),
                    "candles": int((e - s) // step_ms + 1),
                }
                for s, e in missing[: max(1, int(max_span_log))]
            ]
            last_refresh_ms = self.get_last_refresh_ms(symbol) if step_ms == ONE_MIN_MS else 0
            gap_summary = (
                self.get_gap_summary(symbol)
                if step_ms == ONE_MIN_MS
                else {
                    "total_gaps": 0,
                    "total_minutes": 0,
                    "persistent_gaps": 0,
                    "retryable_gaps": 0,
                    "by_reason": {},
                    "gaps": [],
                }
            )
            coverage_ok = len(missing) == 0
            open_tail_gap = bool(
                missing and end_ts >= start_ts and int(missing[-1][1]) >= int(end_ts)
            )
            if open_tail_gap and last_cached_ts is None and end_ts >= start_ts:
                prior_cached_ts = self._latest_cached_ts_before(
                    symbol, start_ts, timeframe=tf_norm
                )
                if prior_cached_ts is not None:
                    last_cached_ts = int(prior_cached_ts)
                    if last_disk_ts is None:
                        last_disk_ts = int(prior_cached_ts)
            tail_gap_candles = 0
            if open_tail_gap:
                if last_cached_ts is None:
                    tail_gap_candles = int((end_ts - start_ts) // step_ms) + 1
                else:
                    tail_gap_candles = int(max(0, (end_ts - int(last_cached_ts)) // step_ms))
            overall_ok = overall_ok and (coverage_ok or not required)
            tf_reports[tf_norm] = {
                "timeframe": tf_norm,
                "required": bool(required),
                "period_ms": int(step_ms),
                "required_candles": int(required_candles),
                "start_ts": int(start_ts),
                "end_ts": int(end_ts),
                "latest_expected_ts": int(latest_expected),
                "current_in_progress_excluded": True,
                "coverage_ok": bool(coverage_ok),
                "loaded_rows": int(combined.shape[0]),
                "disk_loaded_rows": int(disk_arr.shape[0]),
                "runtime_loaded_rows": int(runtime_arr.shape[0]),
                "missing_spans": missing,
                "missing_spans_preview": top_spans,
                "missing_candles": int(missing_candles),
                "verified_no_trade_missing_candles": int(
                    verified_no_trade_missing_candles
                ),
                "deferred_missing_candles": int(deferred_missing_candles),
                "refreshable_missing_candles": int(
                    refreshable_missing_candles
                ),
                "refresh_needed": bool(refreshable_missing_candles > 0),
                "open_tail_gap": bool(open_tail_gap),
                "tail_gap_candles": int(tail_gap_candles),
                "tail_gap_age_ms": (
                    int(max(0, latest_expected - int(last_cached_ts)))
                    if open_tail_gap and last_cached_ts is not None and latest_expected >= 0
                    else None
                ),
                "last_cached_ts": last_cached_ts,
                "last_cached_age_ms": (
                    int(max(0, latest_expected - last_cached_ts))
                    if last_cached_ts is not None and latest_expected >= 0
                    else None
                ),
                "last_disk_ts": last_disk_ts,
                "last_runtime_ts": last_runtime_ts,
                "last_ws_final_ts": (
                    int(last_ws_final_ts) if last_ws_final_ts > 0 else None
                ),
                "last_ws_persist_ms": int(last_ws_persist_ms),
                "ws_persisted_contributed_to_tail": bool(
                    last_ws_final_ts > 0
                    and last_cached_ts is not None
                    and int(last_ws_final_ts) == int(last_cached_ts)
                ),
                "last_refresh_ms": int(last_refresh_ms),
                "refresh_age_ms": (
                    int(max(0, ts_now - int(last_refresh_ms))) if int(last_refresh_ms) > 0 else None
                ),
                "runtime_synthetic_count": int(synthetic_count),
                "known_gaps_total": int(gap_summary.get("total_gaps", 0)),
                "known_gaps_minutes": int(gap_summary.get("total_minutes", 0)),
                "known_gaps_persistent": int(gap_summary.get("persistent_gaps", 0)),
                "known_gaps_retryable": int(gap_summary.get("retryable_gaps", 0)),
                "known_gaps_by_reason": dict(gap_summary.get("by_reason", {}) or {}),
            }

        return {
            "symbol": symbol,
            "generated_ms": int(ts_now),
            "ok": bool(overall_ok),
            "timeframes": tf_reports,
        }

    def rebuild_index_for_range(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
        log_level: str = "info",
    ) -> Dict[str, Any]:
        """Rebuild index.json metadata for shards intersecting [start_ts, end_ts]."""
        tf_norm = self._normalize_timeframe_arg(timeframe, tf)
        step_ms = _tf_to_ms(tf_norm)
        if step_ms <= 0:
            step_ms = ONE_MIN_MS
        s_ts = (int(start_ts) // step_ms) * step_ms
        e_ts = (int(end_ts) // step_ms) * step_ms
        if s_ts > e_ts:
            return {
                "updated": 0,
                "removed": 0,
                "scanned": 0,
                "timeframe": tf_norm,
                "start_ts": s_ts,
                "end_ts": e_ts,
            }

        # Ensure shard path cache is fresh for this symbol/tf.
        self._invalidate_shard_paths_cache(symbol, tf=tf_norm)
        shard_paths = self._iter_shard_paths(symbol, tf=tf_norm)

        idx = self._ensure_symbol_index(symbol, tf=tf_norm)
        shards = idx.setdefault("shards", {})

        updated = 0
        removed = 0
        scanned = 0

        for day_key, (day_start, day_end) in self._date_keys_between(s_ts, e_ts).items():
            if day_end < s_ts or day_start > e_ts:
                continue
            path = shard_paths.get(day_key)
            if path is None or not os.path.exists(path):
                if day_key in shards:
                    shards.pop(day_key, None)
                    removed += 1
                continue
            try:
                arr = _ensure_dtype(self._load_shard(path))
            except Exception:
                arr = np.empty((0,), dtype=CANDLE_DTYPE)
            if arr.size == 0:
                if day_key in shards:
                    shards.pop(day_key, None)
                    removed += 1
                continue
            arr = np.sort(arr, order="ts")
            crc = int(zlib.crc32(arr.tobytes()) & 0xFFFFFFFF)
            shards[day_key] = {
                "path": path,
                "min_ts": int(arr[0]["ts"]),
                "max_ts": int(arr[-1]["ts"]),
                "count": int(arr.shape[0]),
                "crc32": crc,
            }
            updated += 1
            scanned += 1

        idx["shards"] = shards
        pruned = 0
        try:
            pruned = int(self._prune_missing_shards_from_index(idx) or 0)
        except Exception:
            pruned = 0
        if pruned:
            removed += pruned

        # Guard against corrupted refresh timestamps that prevent updates.
        meta = idx.setdefault("meta", {})
        now = self._now_ms()
        try:
            last_refresh = int(meta.get("last_refresh_ms", 0) or 0)
        except Exception:
            last_refresh = 0
        meta_changed = False
        if last_refresh > (now + ONE_MIN_MS):
            meta["last_refresh_ms"] = 0
            meta_changed = True
            self._log(
                "warning",
                "index_last_refresh_in_future",
                symbol=symbol,
                timeframe=tf_norm,
                last_refresh_ms=last_refresh,
                now=now,
            )

        if updated or removed or meta_changed:
            self._save_index(symbol, tf=tf_norm)

        self._log(
            log_level,
            "index_rebuild_range",
            symbol=symbol,
            timeframe=tf_norm,
            start_ts=s_ts,
            end_ts=e_ts,
            scanned=scanned,
            updated=updated,
            removed=removed,
        )

        return {
            "updated": updated,
            "removed": removed,
            "scanned": scanned,
            "timeframe": tf_norm,
            "start_ts": s_ts,
            "end_ts": e_ts,
        }

    # ----- Refresh metadata helpers -----

    def _get_last_refresh_ms(self, symbol: str) -> int:
        idx = self._ensure_symbol_index(symbol)
        try:
            return int(idx.get("meta", {}).get("last_refresh_ms", 0))
        except Exception:
            return 0

    def get_last_refresh_ms(self, symbol: str) -> int:
        """Public helper to read last refresh timestamp (ms) from index metadata."""
        return self._get_last_refresh_ms(symbol)

    def get_last_final_ts(
        self,
        symbol: str,
        timeframe: Optional[str] = None,
        *,
        tf: Optional[str] = None,
    ) -> int:
        """Return the last locally cached finalized timestamp for one timeframe."""
        idx = self._ensure_symbol_index(symbol, timeframe=timeframe, tf=tf)
        try:
            last_final = int(idx.get("meta", {}).get("last_final_ts", 0) or 0)
        except Exception:
            last_final = 0
        if last_final > 0:
            return last_final
        try:
            return max(
                int(shard.get("max_ts", 0) or 0)
                for shard in (idx.get("shards", {}) or {}).values()
                if isinstance(shard, dict)
            )
        except (TypeError, ValueError):
            return 0

    def _set_last_refresh_meta(
        self, symbol: str, last_refresh_ms: int, last_final_ts: Optional[int] = None
    ) -> None:
        idx = self._ensure_symbol_index(symbol)
        meta = idx.setdefault("meta", {})
        meta["last_refresh_ms"] = int(last_refresh_ms)
        if last_final_ts is not None:
            meta["last_final_ts"] = int(last_final_ts)
        self._index[symbol] = idx
        self._save_index(symbol)

    # ----- Coverage / history-bound tracking -----

    def _infer_legacy_authoritative_start_ts(self, meta: Dict[str, Any]) -> Optional[int]:
        """Infer authoritative lower bound from legacy inception/pre_inception metadata.

        Old caches used `inception_ts` both as earliest observed candle and as an implicit
        lower bound when paired with persistent `pre_inception` gaps immediately preceding it.
        Preserve that learned boundary during migration instead of dropping it.
        """
        try:
            observed_start = meta.get("observed_start_ts", meta.get("inception_ts"))
            if observed_start is None:
                return None
            observed_start = int(observed_start)
            cutoff_end = observed_start - ONE_MIN_MS
            for gap in meta.get("known_gaps", []):
                if not isinstance(gap, dict):
                    continue
                if str(gap.get("reason", "")) != "pre_inception":
                    continue
                try:
                    gap_end = int(gap.get("end_ts"))
                    gap_start = int(gap.get("start_ts"))
                    retry_count = int(gap.get("retry_count", 0))
                except Exception:
                    continue
                if retry_count < _GAP_MAX_RETRIES:
                    continue
                if gap_start < observed_start and gap_end >= cutoff_end:
                    return observed_start
        except Exception:
            return None
        return None

    def _get_inception_ts(self, symbol: str) -> Optional[int]:
        """Return earliest observed candle timestamp for this symbol, or None.

        Historically this field was also used as an authoritative exchange-history lower bound.
        It now tracks only local observed coverage, while authoritative clipping uses
        ``authoritative_start_ts``.
        """
        idx = self._ensure_symbol_index(symbol)
        try:
            meta = idx.get("meta", {})
            val = meta.get("observed_start_ts", meta.get("inception_ts"))
            return int(val) if val is not None else None
        except Exception:
            return None

    def _set_inception_ts(self, symbol: str, ts: int, *, save: bool = True) -> None:
        """Set earliest observed candle timestamp for this symbol."""
        idx = self._ensure_symbol_index(symbol)
        meta = idx.setdefault("meta", {})
        current = meta.get("observed_start_ts", meta.get("inception_ts"))
        # Only update if unset or if new ts is earlier
        if current is None or int(ts) < int(current):
            observed_ts = int(ts)
            meta["observed_start_ts"] = observed_ts
            meta["inception_ts"] = observed_ts  # legacy alias
            auth_current = meta.get("authoritative_start_ts")
            auth_updated = False
            if auth_current is not None and observed_ts < int(auth_current):
                meta["authoritative_start_ts"] = observed_ts
                meta["authoritative_start_source"] = "observed_data"
                auth_updated = True
            self._index[f"{symbol}::1m"] = idx
            if save:
                self._save_index(symbol)
            if auth_updated:
                # If we previously marked ranges as pre-inception but later observed earlier
                # real data, that authoritative lower bound became stale.
                try:
                    self._prune_pre_inception_gaps(symbol, observed_ts, save=save)
                except Exception as exc:
                    self._log(
                        "warning",
                        "prune_pre_inception_gaps_failed",
                        symbol=symbol,
                        error_type=bounded_exception_type(exc),
                    )
            self._log(
                "debug",
                "inception_ts_updated",
                symbol=symbol,
                old_ts=current,
                new_ts=observed_ts,
            )

    def _first_ohlcv_cache_path(self) -> Path:
        return Path(self.cache_dir) / "first_ohlcv_timestamps_unified_exchange_specific.json"

    def _first_ohlcv_cache_version_path(self) -> Path:
        return Path(self.cache_dir) / "first_ohlcv_timestamps_unified.version"

    def _first_ohlcv_cache_symbols_path(self) -> Path:
        return (
            Path(self.cache_dir)
            / "first_ohlcv_timestamps_unified_exchange_specific_symbols.json"
        )

    def _first_ohlcv_cache_exchange_name(self) -> str:
        exchange_name = str(self.exchange_name or self._ex_id or "").lower()
        return _FIRST_OHLCV_EXCHANGE_CACHE_ALIASES.get(exchange_name, exchange_name)

    def _lookup_cached_authoritative_start_ts(self, symbol: str) -> Optional[int]:
        cache_path = self._first_ohlcv_cache_path()
        symbols_path = self._first_ohlcv_cache_symbols_path()
        if not cache_path.exists() or not symbols_path.exists():
            return None
        try:
            with open(self._first_ohlcv_cache_version_path(), "r", encoding="utf-8") as f:
                if int(f.read().strip()) != FIRST_OHLCV_TIMESTAMPS_CACHE_VERSION:
                    return None
            with open(cache_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            with open(symbols_path, "r", encoding="utf-8") as f:
                cached_symbols = json.load(f)
            if not isinstance(data, dict) or not isinstance(cached_symbols, dict):
                return None
            exchange_name = self._first_ohlcv_cache_exchange_name()
            cache_keys = [symbol]
            markets = getattr(self.exchange, "markets", {}) if self.exchange is not None else {}
            if isinstance(markets, dict):
                for market_symbol, market in markets.items():
                    if not isinstance(market, dict):
                        continue
                    resolved_symbol = str(market.get("symbol") or market_symbol)
                    if resolved_symbol != symbol:
                        continue
                    native_id = str(market.get("id") or "").strip()
                    if native_id and native_id not in cache_keys:
                        cache_keys.append(native_id)
                    if native_id:
                        qualifiers = set(exchange_name_aliases(exchange_name))
                        qualifiers.update(exchange_name_aliases(self.exchange_name))
                        qualifiers.update(exchange_name_aliases(self._ex_id))
                        for qualifier in sorted(qualifiers):
                            qualified_id = f"{qualifier}::{native_id}"
                            if qualified_id not in cache_keys:
                                cache_keys.append(qualified_id)
            base_coin = symbol.split("/")[0].strip()
            if base_coin != symbol:
                cache_keys.append(base_coin)
            for cache_key in cache_keys:
                exchange_values = data.get(cache_key, {})
                if not isinstance(exchange_values, dict):
                    continue
                value = exchange_values.get(exchange_name)
                provenance_values = cached_symbols.get(cache_key, {})
                cached_symbol = (
                    provenance_values.get(exchange_name)
                    if isinstance(provenance_values, dict)
                    else None
                )
                if (
                    value is not None
                    and float(value) > 0.0
                    and cached_symbol == symbol
                ):
                    return int(value)
            return None
        except Exception:
            return None

    def _get_authoritative_start_ts(self, symbol: str) -> Optional[int]:
        """Return authoritative exchange-history lower bound, if known."""
        idx = self._ensure_symbol_index(symbol)
        meta = idx.setdefault("meta", {})
        try:
            value = meta.get("authoritative_start_ts")
            if value is not None:
                return int(value)
        except Exception:
            pass

        cached = self._lookup_cached_authoritative_start_ts(symbol)
        if cached is not None:
            self._set_authoritative_start_ts(
                symbol,
                cached,
                source="exchange_specific_cache",
                save=True,
            )
            idx = self._ensure_symbol_index(symbol)
            value = idx.get("meta", {}).get("authoritative_start_ts")
            return int(value) if value is not None else None
        return None

    def _set_authoritative_start_ts(
        self, symbol: str, ts: int, *, source: str, save: bool = True
    ) -> None:
        """Persist authoritative exchange-history lower bound for this symbol."""
        idx = self._ensure_symbol_index(symbol)
        meta = idx.setdefault("meta", {})
        observed_start = self._get_inception_ts(symbol)
        authoritative_ts = int(ts)
        authoritative_source = str(source)
        if observed_start is not None and observed_start < authoritative_ts:
            authoritative_ts = int(observed_start)
            authoritative_source = "observed_data"
        current = meta.get("authoritative_start_ts")
        if current is None or authoritative_ts < int(current):
            meta["authoritative_start_ts"] = authoritative_ts
            meta["authoritative_start_source"] = authoritative_source
            self._index[f"{symbol}::1m"] = idx
            if save:
                self._save_index(symbol)
            try:
                self._prune_pre_inception_gaps(symbol, authoritative_ts, save=save)
            except Exception as exc:
                self._log(
                    "warning",
                    "prune_pre_inception_gaps_failed",
                    symbol=symbol,
                    error_type=bounded_exception_type(exc),
                )

    def _prune_pre_inception_gaps(self, symbol: str, inception_ts: int, *, save: bool = True) -> None:
        """Trim/remove known gaps with reason='pre_inception' now covered by real data."""
        gaps = self._get_known_gaps_enhanced(symbol)
        if not gaps:
            return
        cutoff_end = int(inception_ts) - ONE_MIN_MS
        changed = False
        new_gaps: List[GapEntry] = []

        for g in gaps:
            try:
                if str(g.get("reason", "")) != "pre_inception":
                    new_gaps.append(g)
                    continue
                s = int(g.get("start_ts", 0))
                e = int(g.get("end_ts", 0))
                if e <= cutoff_end:
                    new_gaps.append(g)
                    continue
                if s <= cutoff_end:
                    # Overlaps: trim to end before inception
                    trimmed: GapEntry = {
                        "start_ts": s,
                        "end_ts": cutoff_end,
                        "retry_count": int(g.get("retry_count", 0)),
                        "reason": "pre_inception",
                        "added_at": int(g.get("added_at", 0)),
                    }
                    if trimmed["start_ts"] <= trimmed["end_ts"]:
                        new_gaps.append(trimmed)
                    changed = True
                    continue
                # Entirely after inception: remove
                changed = True
            except Exception:
                new_gaps.append(g)

        if changed and save:
            self._save_known_gaps_enhanced(symbol, new_gaps)

    def _maybe_update_inception_ts(self, symbol: str, arr: np.ndarray, *, save: bool = True) -> None:
        """Update inception_ts if arr contains an earlier timestamp than known."""
        if arr.size == 0:
            return
        first_ts = int(arr[0]["ts"]) if arr.ndim else int(arr["ts"])
        current = self._get_inception_ts(symbol)
        if current is None or first_ts < current:
            self._set_inception_ts(symbol, first_ts, save=save)

    # ----- CCXT fetching -----

    async def _apply_rate_limit_backoff(self) -> None:
        """Wait if we're in a global rate limit backoff period.

        When a rate limit is hit, all concurrent requests should pause to avoid
        the thundering herd problem where they all retry simultaneously.
        """
        self._raise_if_shutdown_requested("rate_limit_backoff")
        now = time.time()
        if now < self._rate_limit_until:
            wait_time = self._rate_limit_until - now
            if wait_time > 0:
                self._log("debug", "rate_limit_global_wait", wait_seconds=round(wait_time, 2))
                await self._sleep_interruptible(wait_time, stage="rate_limit_backoff")

    async def _apply_remote_fetch_spacing(self, *, symbol: str, tf: str) -> None:
        """Pace ccxt OHLCV calls from this manager to avoid local request bursts."""
        self._raise_if_shutdown_requested("remote_fetch_spacing")
        interval_ms = float(getattr(self, "_remote_fetch_min_interval_ms", 0.0) or 0.0)
        if interval_ms <= 0.0:
            return
        async with self._remote_fetch_spacing_lock:
            self._raise_if_shutdown_requested("remote_fetch_spacing")
            now_ms = _utc_now_ms()
            last_started_ms = int(getattr(self, "_remote_fetch_last_started_ms", 0) or 0)
            wait_ms = int(last_started_ms + interval_ms - now_ms)
            if wait_ms > 0:
                self._log(
                    "debug",
                    "remote_fetch_spacing_wait",
                    symbol=symbol,
                    tf=tf,
                    wait_ms=wait_ms,
                    interval_ms=int(interval_ms),
                )
                await self._sleep_interruptible(wait_ms / 1000.0, stage="remote_fetch_spacing")
                now_ms = _utc_now_ms()
            self._remote_fetch_last_started_ms = int(now_ms)

    async def _set_global_rate_limit(self, backoff_seconds: float = 5.0) -> None:
        """Set a global rate limit backoff that affects all concurrent requests."""
        async with self._rate_limit_lock:
            new_until = time.time() + backoff_seconds
            # Only extend if the new backoff is longer than existing
            if new_until > self._rate_limit_until:
                self._rate_limit_until = new_until
                self._rate_limit_count += 1
                self._log(
                    "debug",
                    "rate_limit_global_set",
                    backoff_seconds=backoff_seconds,
                    total_count=self._rate_limit_count,
                )

    async def _ccxt_fetch_ohlcv_once(
        self,
        symbol: str,
        since_ms: int,
        limit: int,
        end_exclusive_ms: Optional[int] = None,
        timeframe: Optional[str] = None,
        *,
        tf: Optional[str] = None,
    ) -> list:
        """Fetch a single OHLCV page from ccxt, with basic retry/backoff."""
        self._raise_if_shutdown_requested("ccxt_fetch_ohlcv_once")
        if self.exchange is None:
            return []
        # Determine method to call (exchange instance or module)
        ex = self.exchange
        if not hasattr(ex, "fetch_ohlcv"):
            return []

        exid = (self._ex_id or "").lower() if isinstance(self._ex_id, str) else ""
        is_bybit = "bybit" in exid
        is_weex = "weex" in exid
        is_hyperliquid = "hyperliquid" in exid
        max_attempts = 9 if is_bybit else 5
        backoff = 1.0 if is_bybit else 0.5
        backoff_cap = 20.0 if is_bybit else 8.0
        last_exc: Optional[Exception] = None
        tf_norm = self._normalize_timeframe_arg(timeframe, tf, default=self._ccxt_timeframe)
        for attempt in range(max_attempts):
            self._raise_if_shutdown_requested("ccxt_fetch_ohlcv_once")
            # Wait for global rate limit backoff if one is active
            await self._apply_rate_limit_backoff()
            try:
                params: Dict[str, Any] = {}
                request_limit = int(limit)
                # Bitget account-mode detection enables UTA routing on the shared
                # authenticated CCXT client. Public candle history is account-
                # independent, and the UTA candle endpoint may expose a shorter
                # history than the classic futures history endpoint. Override only
                # this public request so live UTA accounts retain private v3 routing
                # without losing older strategy candles.
                if "bitget" in exid:
                    params["uta"] = False
                # Provide an end bound for exchanges that support it.
                # Note: Avoid passing 'until' to Bitget due to API validation errors on non-1m tfs.
                if is_weex and end_exclusive_ms is not None:
                    request_limit, params = self._weex_ohlcv_request_plan(
                        since_ms=int(since_ms),
                        end_exclusive_ms=int(end_exclusive_ms),
                        timeframe=tf_norm,
                    )
                elif end_exclusive_ms is not None:
                    exid = (self._ex_id or "").lower() if isinstance(self._ex_id, str) else ""
                    # Avoid 'until' for exchanges where it yields tail-anchored or inconsistent pages
                    # leading to incomplete forward pagination on first run.
                    if (
                        "bitget" not in exid
                        and "okx" not in exid
                        and "bybit" not in exid
                        and "kucoin" not in exid
                        and "gateio" not in exid
                    ):
                        params["until"] = int(end_exclusive_ms) - 1

                # Bybit v5 requires a category for some market data routes. CCXT usually infers
                # this from the market, but being explicit avoids intermittent misclassification.
                if "bybit" in exid:
                    params.setdefault("category", "linear")

                await self._apply_remote_fetch_spacing(symbol=symbol, tf=tf_norm)
                t0 = time.monotonic()
                self._emit_remote_fetch(
                    {
                        "kind": "ccxt_fetch_ohlcv",
                        "stage": "start",
                        "exchange": str(self._ex_id),
                        "symbol": symbol,
                        "tf": tf_norm,
                        "since_ts": int(since_ms),
                        "limit": int(request_limit),
                        "attempt": int(attempt + 1),
                        "params": dict(params),
                    }
                )
                self._log(
                    "debug",
                    "ccxt_fetch_ohlcv",
                    symbol=symbol,
                    tf=tf_norm,
                    since_ts=int(since_ms),
                    limit=request_limit,
                    attempt=attempt + 1,
                    param_keys=_bounded_remote_fetch_param_keys(params),
                )
                if getattr(self, "_net_sem", None) is not None:
                    async with self._net_sem:  # type: ignore[attr-defined]
                        # Re-check rate limit after acquiring semaphore.
                        # Tasks may have been queued before a 429 set the
                        # global backoff; honour it now instead of firing
                        # immediately after the semaphore unblocks.
                        await self._apply_rate_limit_backoff()
                        res = await ex.fetch_ohlcv(
                            symbol,
                            timeframe=tf_norm,
                            since=since_ms,
                            limit=request_limit,
                            params=params,
                        )
                else:
                    res = await ex.fetch_ohlcv(
                        symbol,
                        timeframe=tf_norm,
                        since=since_ms,
                        limit=request_limit,
                        params=params,
                    )
                first_ts = None
                last_ts = None
                if res:
                    try:
                        first_ts = int(res[0][0])
                    except Exception:
                        first_ts = None
                    try:
                        last_ts = int(res[-1][0])
                    except Exception:
                        last_ts = None
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                self._emit_remote_fetch(
                    {
                        "kind": "ccxt_fetch_ohlcv",
                        "stage": "ok",
                        "exchange": str(self._ex_id),
                        "symbol": symbol,
                        "tf": tf_norm,
                        "since_ts": int(since_ms),
                        "rows": int(len(res) if res else 0),
                        "first_ts": first_ts,
                        "last_ts": last_ts,
                        "elapsed_ms": elapsed_ms,
                    }
                )
                self._log(
                    "debug",
                    "ccxt_fetch_ohlcv_ok",
                    symbol=symbol,
                    tf=tf_norm,
                    since_ts=int(since_ms),
                    rows=(len(res) if res else 0),
                    first_ts=first_ts,
                    last_ts=last_ts,
                )
                return res or []
            except Exception as e:  # pragma: no cover - network not used in tests
                last_exc = e
                raw_err_type = type(e).__name__
                err_type = bounded_exception_type(e)
                elapsed_ms = int((time.monotonic() - t0) * 1000) if "t0" in locals() else None
                self._emit_remote_fetch(
                    {
                        "kind": "ccxt_fetch_ohlcv",
                        "stage": "error",
                        "exchange": str(self._ex_id),
                        "symbol": symbol,
                        "tf": tf_norm,
                        "since_ts": int(since_ms),
                        "attempt": int(attempt + 1),
                        "elapsed_ms": elapsed_ms,
                        "params": dict(params) if "params" in locals() else None,
                        "error_type": err_type,
                    }
                )
                action = "exhausted" if attempt == max_attempts - 1 else "retry"
                self._throttled_warning(
                    f"ccxt_fetch_ohlcv_failed:{symbol}:{tf_norm}:{err_type}:{action}",
                    "ccxt_fetch_ohlcv_failed",
                    symbol=symbol,
                    tf=tf_norm,
                    attempt=attempt + 1,
                    max_attempts=max_attempts,
                    elapsed_ms=elapsed_ms,
                    error_type=err_type,
                    action=action,
                )
                sleep_s = backoff
                msg = str(e) or ""
                msg_l = msg.lower()
                # Heuristic: slow down harder on rate-limit style responses.
                is_rate_limit = any(x in msg_l for x in ("rate limit", "too many", "429", "10006"))
                if is_rate_limit:
                    # Set global backoff to coordinate all concurrent requests
                    # Hyperliquid needs longer backoff due to stricter limits
                    global_backoff = 10.0 if is_hyperliquid else 5.0
                    await self._set_global_rate_limit(global_backoff)
                    sleep_s = max(sleep_s, global_backoff)
                # Bybit: be more persistent on transient network-ish errors.
                if is_bybit and (
                    raw_err_type
                    in {"RequestTimeout", "NetworkError", "ExchangeNotAvailable", "DDoSProtection"}
                    or any(
                        x in msg_l
                        for x in (
                            "timed out",
                            "timeout",
                            "etimedout",
                            "econnreset",
                            "502",
                            "503",
                            "504",
                        )
                    )
                ):
                    sleep_s = max(sleep_s, 2.0)
                if attempt == max_attempts - 1:
                    break
                await self._sleep_interruptible(sleep_s, stage="ccxt_fetch_ohlcv_retry")
                backoff = min(backoff * 2.0, backoff_cap)
        if last_exc is not None:
            raise OhlcvFetchError(
                "ccxt OHLCV fetch exhausted retries "
                f"exchange={self._ex_id} symbol={symbol} tf={tf_norm} "
                f"since_ts={int(since_ms)} limit={int(limit)} attempts={max_attempts}"
            ) from last_exc
        return []

    def _weex_ohlcv_request_plan(
        self,
        *,
        since_ms: int,
        end_exclusive_ms: int,
        timeframe: str,
    ) -> Tuple[int, Dict[str, Any]]:
        """Choose WEEX's recent or bounded historical kline endpoint.

        The recent endpoint ignores ``since`` and tail-anchors up to 1,000
        rows, including the currently forming candle. The historical endpoint
        returns at most 100 rows and tail-anchors over-wide ranges. Bound every
        historical request to one forward page, then use the cheaper recent
        endpoint only when the remaining range fits its finalized-candle tail.
        """
        period_ms = _tf_to_ms(timeframe)
        since = int(since_ms)
        end_exclusive = int(end_exclusive_ms)
        now = int(self._now_ms())
        current_bucket = (now // period_ms) * period_ms
        latest_finalized = current_bucket - period_ms
        recent_finalized_capacity = _WEEX_RECENT_OHLCV_LIMIT - 1
        recent_oldest = latest_finalized - period_ms * (
            recent_finalized_capacity - 1
        )

        if end_exclusive >= current_bucket and since >= recent_oldest:
            requested_rows = max(
                1, int(math.ceil((end_exclusive - since) / float(period_ms)))
            )
            # Include the current forming candle; completed-candle callers
            # discard it at end_exclusive.
            recent_limit = min(
                _WEEX_RECENT_OHLCV_LIMIT,
                max(1, requested_rows + 1),
            )
            return recent_limit, {}

        page_end = min(
            end_exclusive - 1,
            since + period_ms * (_WEEX_HISTORICAL_OHLCV_LIMIT - 1),
        )
        return _WEEX_HISTORICAL_OHLCV_LIMIT, {
            "historical": True,
            "until": int(page_end),
        }

    # ----- Array slicing helpers -----

    def _slice_ts_range(
        self, arr: np.ndarray, start_ts: int, end_ts: int, *, assume_sorted: bool = False
    ) -> np.ndarray:
        """Return arr sliced to [start_ts, end_ts] inclusive by 'ts'.

        Assumes arr is structured dtype CANDLE_DTYPE.

        Parameters
        ----------
        assume_sorted : bool
            If True, skip the sort (caller guarantees arr is already sorted by ts).
            Use this when arr comes from get_candles/standardize_gaps which already sorts.
        """
        if arr.size == 0:
            return arr
        arr = _ensure_dtype(arr)
        if not assume_sorted:
            # Only sort if needed - check if already sorted to skip O(n log n) sort
            ts_arr = arr["ts"]
            if ts_arr.size > 1 and not np.all(ts_arr[:-1] <= ts_arr[1:]):
                arr = np.sort(arr, order="ts")
        ts_arr = _ts_index(arr)
        i0 = int(np.searchsorted(ts_arr, start_ts, side="left"))
        i1 = int(np.searchsorted(ts_arr, end_ts, side="right"))
        return arr[i0:i1]

    def _normalize_ccxt_ohlcv(self, rows: list) -> np.ndarray:
        """Convert ccxt rows [ms,o,h,l,c,vol] to CANDLE_DTYPE and filter alignment."""
        if not rows:
            return np.empty((0,), dtype=CANDLE_DTYPE)
        out = []
        for r in rows:
            try:
                ts = int(r[0])
                # keep only fully minute-aligned candles
                if ts % ONE_MIN_MS != 0:
                    ts = _floor_minute(ts)
                o, h, l, c = map(float, (r[1], r[2], r[3], r[4]))
                bv = float(r[5]) if len(r) > 5 else 0.0
                if not all(math.isfinite(x) for x in (o, h, l, c, bv)):
                    continue
                if min(o, h, l, c) <= 0.0 or bv < 0.0:
                    continue
                bv = normalize_ccxt_volume_to_base(self._ex_id or "", c, bv)
                out.append((ts, o, h, l, c, bv))
            except Exception:
                continue
        if not out:
            return np.empty((0,), dtype=CANDLE_DTYPE)
        arr = np.array(out, dtype=CANDLE_DTYPE)
        arr = np.sort(arr, order="ts")
        # drop duplicate ts keeping last
        ts = arr["ts"].astype(np.int64)
        keep = np.ones(len(arr), dtype=bool)
        last = None
        for i in range(len(arr)):
            if last is not None and ts[i] == last:
                keep[i - 1] = False
            last = ts[i]
        return arr[keep]

    def get_last_live_ws_ohlcv_ts(self, symbol: str) -> int:
        """Return the newest finalized WS candle persisted for this symbol."""
        idx = self._ensure_symbol_index(symbol, tf="1m")
        try:
            return int(idx.get("meta", {}).get("last_ws_final_ts", 0) or 0)
        except Exception:
            return 0

    def get_last_live_ws_persist_ms(self, symbol: str) -> int:
        """Return when a finalized WS candle was most recently persisted."""
        idx = self._ensure_symbol_index(symbol, tf="1m")
        try:
            return int(idx.get("meta", {}).get("last_ws_persist_ms", 0) or 0)
        except Exception:
            return 0

    def clear_live_ws_ohlcv_state(self, symbol: str) -> None:
        """Discard volatile CCXT-cache provenance without deleting candles."""
        self._live_ws_ohlcv_observations.pop(symbol, None)

    async def ingest_live_ws_ohlcv(
        self,
        symbol: str,
        rows: list,
        *,
        now_ms: Optional[int] = None,
    ) -> int:
        """Persist proven-final public WS rows through the canonical 1m path.

        CCXT Pro may repeat its whole sliding OHLCV cache on each update. The
        first non-empty snapshot of each watcher session only primes provenance.
        After that, a changed row may correct an already canonical timestamp.
        A new canonical timestamp additionally requires a fresh successor
        proving a trusted preceding bucket closed. The current minute,
        malformed rows, silence, and reconnect gaps never synthesize candles.
        An existing disk/index basis is required before WS may extend history.
        """
        now = int(self._now_ms() if now_ms is None else now_ms)
        latest_finalized = _floor_minute(now) - ONE_MIN_MS
        if latest_finalized < 0:
            return 0
        arr = self._normalize_ccxt_ohlcv(rows)
        if arr.size == 0:
            return 0

        cutoff = int(
            latest_finalized - _LIVE_WS_OBSERVATION_RETENTION_MS + ONE_MIN_MS
        )
        arr = arr[arr["ts"].astype(np.int64) >= cutoff]
        if arr.size == 0:
            return 0

        session_primed = symbol in self._live_ws_ohlcv_observations
        previous = self._live_ws_ohlcv_observations.get(symbol, {})
        current: Dict[int, Tuple[Tuple[float, ...], bool]] = {}
        changed_or_new: set[int] = set()
        rows_by_ts: Dict[int, np.void] = {}
        for row in arr:
            ts = int(row["ts"])
            values = tuple(float(value) for value in row.tolist())
            rows_by_ts[ts] = row
            prior = previous.get(ts)
            if prior is None or prior[0] != values:
                changed_or_new.add(ts)
            # A row first seen while its bucket is still open may later be
            # sealed by a fresh successor. Processing time cannot prove a
            # boundary crossing because a pre-boundary payload may resume from
            # the event loop only after the bucket closes. Past rows in the
            # first snapshot of a watcher session are untrusted replay-cache
            # contents until that row itself changes.
            successor_eligible = (
                now < ts + ONE_MIN_MS
                if not session_primed
                else bool(
                    prior is None
                    or prior[0] != values
                    or prior[1]
                )
            )
            current[ts] = (values, successor_eligible)
        # The first non-empty snapshot of every watcher session establishes
        # provenance only. It may be a replay of a partial pre-disconnect row,
        # so reconnect recovery remains REST-owned.
        if not session_primed:
            self._live_ws_ohlcv_observations[symbol] = current
            return 0
        # In CCXT ``newUpdates`` mode the fresh successor may be returned
        # alone. Reuse the immediately preceding observation only within the
        # same uninterrupted watcher session so that successor provenance can
        # seal it without inventing or remotely refetching the row.
        successor_proven_predecessors: set[int] = set()
        for successor_ts in changed_or_new:
            predecessor_ts = int(successor_ts - ONE_MIN_MS)
            prior_predecessor = previous.get(predecessor_ts)
            if prior_predecessor is None or not bool(prior_predecessor[1]):
                continue
            successor_proven_predecessors.add(predecessor_ts)
            if predecessor_ts not in rows_by_ts:
                predecessor = np.array(
                    [prior_predecessor[0]],
                    dtype=CANDLE_DTYPE,
                )
                rows_by_ts[predecessor_ts] = predecessor[0]
        # A first-ever WS snapshot must not become a restart-sensitive trading
        # basis. REST or previously persisted canonical data must exist first.
        if int(self.get_last_final_ts(symbol) or 0) <= 0:
            self._live_ws_ohlcv_observations[symbol] = current
            return 0

        candidate_rows = []
        for ts, row in rows_by_ts.items():
            if ts > latest_finalized:
                continue
            if (
                ts in changed_or_new
                or ts in successor_proven_predecessors
            ):
                candidate_rows.append(tuple(row.tolist()))
        if not candidate_rows:
            self._live_ws_ohlcv_observations[symbol] = current
            return 0
        candidates = np.array(candidate_rows, dtype=CANDLE_DTYPE)

        async with self._acquire_fetch_lock(symbol, "1m"):
            disk = self._load_from_disk(
                symbol,
                int(candidates[0]["ts"]),
                int(candidates[-1]["ts"]),
                timeframe="1m",
            )
            cached = self._slice_ts_range(
                self._ensure_symbol_cache(symbol),
                int(candidates[0]["ts"]),
                int(candidates[-1]["ts"]),
            )
            canonical = self._merge_overwrite(disk, cached)
            canonical_by_ts = {int(row["ts"]): row for row in canonical}
            # Value changes alone can correct a timestamp already admitted by
            # REST or WS. Extending canonical history requires independent
            # post-boundary proof, so a pre-boundary update delayed in the
            # consumer queue cannot become final merely because wall time
            # advanced before it was processed.
            eligible_rows = [
                row
                for row in candidates
                if int(row["ts"]) in canonical_by_ts
                or int(row["ts"]) in successor_proven_predecessors
            ]
            changed_rows = [
                tuple(row.tolist())
                for row in eligible_rows
                if int(row["ts"]) not in canonical_by_ts
                or not np.array_equal(canonical_by_ts[int(row["ts"])], row)
            ]
            if not changed_rows:
                self._live_ws_ohlcv_observations[symbol] = current
                return 0
            persisted = np.array(changed_rows, dtype=CANDLE_DTYPE)
            self._persist_batch(
                symbol,
                persisted,
                timeframe="1m",
                merge_cache=True,
                source="ws",
            )
        # Commit the new provenance frame only after any required persistence
        # succeeds. A failed write remains retryable on the next WS update.
        self._live_ws_ohlcv_observations[symbol] = current
        return int(persisted.shape[0])

    def _rejected_ccxt_ohlcv_timestamps(
        self,
        rows: list,
        normalized: np.ndarray,
    ) -> set[int]:
        """Return parseable payload timestamps discarded during normalization.

        A timestamp present in any accepted duplicate is not rejected. Callers
        use the remaining timestamps as unavailable buckets rather than treating
        invalid exchange data as an omitted no-trade interval.
        """
        accepted = (
            {int(ts) for ts in normalized["ts"]}
            if isinstance(normalized, np.ndarray) and normalized.size
            else set()
        )
        present = set()
        for row in rows:
            try:
                ts = int(row[0])
                if ts % ONE_MIN_MS != 0:
                    ts = _floor_minute(ts)
                present.add(ts)
            except (IndexError, TypeError, ValueError, OverflowError):
                continue
        return present - accepted

    def _ccxt_ohlcv_has_unidentifiable_rejected_row(self, rows: list) -> bool:
        """Return whether a raw row cannot be attributed to a candle bucket."""
        for row in rows:
            try:
                int(row[0])
            except (IndexError, TypeError, ValueError, OverflowError):
                return True
        return False

    async def _fetch_kucoin_contextual_gap_page(
        self,
        symbol: str,
        *,
        left_boundary_ts: int,
        right_boundary_ts: int,
        gap_start_ts: int,
        gap_end_ts: int,
    ) -> Tuple[np.ndarray, bool]:
        """Fetch one raw KuCoin page and verify an omission between real bounds.

        The proof deliberately bypasses generic pagination's immediate
        intra-page gap recording. Only one successful raw response containing
        both selected boundary candles and no accepted or rejected row inside
        the unresolved interval may certify that interval as no-trade
        continuity.
        """
        left_ts = int(left_boundary_ts)
        right_ts = int(right_boundary_ts)
        gap_start = int(gap_start_ts)
        gap_end = int(gap_end_ts)
        request_since = left_ts
        if (
            self._ccxt_since_exclusive
            and self._ccxt_page_overlap_candles > 0
            and left_ts > 0
        ):
            request_since = max(
                0,
                left_ts
                - ONE_MIN_MS * int(self._ccxt_page_overlap_candles),
            )
        requested_buckets = max(
            2,
            (right_ts - request_since) // ONE_MIN_MS + 1,
        )
        page = await self._ccxt_fetch_ohlcv_once(
            symbol,
            request_since,
            min(int(self._ccxt_limit_default), int(requested_buckets)),
            end_exclusive_ms=right_ts + ONE_MIN_MS,
            tf="1m",
        )
        normalized = self._normalize_ccxt_ohlcv(page)
        if normalized.size:
            normalized = normalized[
                (normalized["ts"] >= left_ts)
                & (normalized["ts"] <= right_ts)
            ]
        accepted_timestamps = (
            {int(ts) for ts in normalized["ts"]}
            if normalized.size
            else set()
        )
        rejected_timestamps = self._rejected_ccxt_ohlcv_timestamps(
            page, normalized
        )
        proof = bool(
            left_ts in accepted_timestamps
            and right_ts in accepted_timestamps
            and not self._ccxt_ohlcv_has_unidentifiable_rejected_row(page)
            and not any(
                gap_start <= int(ts) <= gap_end
                for ts in rejected_timestamps
            )
            and not any(
                gap_start <= int(ts) <= gap_end
                for ts in accepted_timestamps
            )
        )
        return normalized, proof

    def _evict_rejected_native_sparse_synthetics(
        self,
        symbol: str,
        *,
        timeframe: str,
        rejected_timestamps: Optional[set[int]] = None,
        rejected_ranges: Optional[list[tuple[int, int]]] = None,
    ) -> set[int]:
        """Remove persisted KuCoin sparse placeholders contradicted by invalid real rows.

        KuCoin omits native kline buckets that have no ticks. Passivbot's
        internally bounded placeholders are therefore identifiable on disk as
        flat zero-volume rows. A later payload row at that timestamp proves the
        bucket was not an omitted no-trade interval; if that row is invalid, the
        cached placeholder must become unavailable rather than survive the
        merge.
        """
        rejected_timestamps = {
            int(ts) for ts in (rejected_timestamps or set())
        }
        rejected_ranges = [
            (int(start_ts), int(end_ts))
            for start_ts, end_ts in (rejected_ranges or [])
            if int(end_ts) >= int(start_ts)
        ]
        if not rejected_timestamps and not rejected_ranges:
            return set()
        tf_norm = self._normalize_timeframe_arg(timeframe, None)
        if tf_norm == "1m":
            return set()
        shard_paths = self._iter_shard_paths(symbol, tf=tf_norm)
        exact_ts_arr = (
            np.fromiter(rejected_timestamps, dtype=np.int64)
            if rejected_timestamps
            else np.empty((0,), dtype=np.int64)
        )
        removed: set[int] = set()
        for day_key, path in shard_paths.items():
            if not path or not os.path.exists(path):
                continue
            day_start, day_end = self._date_range_of_key(day_key)
            if (
                not any(day_start <= ts <= day_end for ts in rejected_timestamps)
                and not any(
                    start_ts <= day_end and end_ts >= day_start
                    for start_ts, end_ts in rejected_ranges
                )
            ):
                continue
            shard = self._load_shard(path)
            if shard.size == 0:
                continue
            shard_ts = shard["ts"].astype(np.int64, copy=False)
            candidate = np.isin(shard_ts, exact_ts_arr)
            for start_ts, end_ts in rejected_ranges:
                candidate |= (shard_ts >= start_ts) & (shard_ts <= end_ts)
            flat_zero = (
                (shard["bv"] == 0.0)
                & (shard["o"] == shard["h"])
                & (shard["o"] == shard["l"])
                & (shard["o"] == shard["c"])
            )
            remove_mask = candidate & flat_zero
            if not bool(np.any(remove_mask)):
                continue
            removed.update(int(ts) for ts in shard["ts"][remove_mask])
            kept = shard[~remove_mask]
            if kept.size:
                self._save_shard(symbol, day_key, kept, tf=tf_norm)
            else:
                os.remove(path)
                idx = self._ensure_symbol_index(symbol, tf=tf_norm)
                idx.setdefault("shards", {}).pop(day_key, None)
                self._index[f"{symbol}::{tf_norm}"] = idx
                self._save_index(symbol, tf=tf_norm)
                self._invalidate_shard_paths_cache(symbol, tf=tf_norm)
        if removed:
            idx = self._ensure_symbol_index(symbol, tf=tf_norm)
            self._recompute_shard_derived_meta(idx)
            self._index[f"{symbol}::{tf_norm}"] = idx
            self._save_index(symbol, tf=tf_norm)
            self._invalidate_ema_cache(symbol, timeframe=tf_norm)
            self._invalidate_tf_range_cache(
                symbol,
                timeframe=tf_norm,
                start_ts=min(removed),
                end_ts=max(removed),
            )
        return removed

    def _synthesize_verified_sparse_payload_gaps(
        self,
        arr: np.ndarray,
        *,
        period_ms: int,
        start_ts: int,
        end_exclusive_ts: int,
        rejected_timestamps: Optional[set[int]] = None,
        has_unidentifiable_rejected_row: bool = False,
    ) -> np.ndarray:
        """Fill KuCoin higher-timeframe no-tick buckets bounded in one payload.

        KuCoin documents that it omits kline buckets with no ticks.  Two real
        adjacent rows in the same successful response prove that an internal
        timestamp absent from that response is a no-trade interval. Timestamps
        present in rejected rows remain unavailable. Leading, trailing, and
        between-page gaps remain unproven and are deliberately not filled.

        The established 1m path records verified gaps and standardizes them
        later, so this helper is limited to native higher timeframes.
        """
        if (
            not self._record_payload_gaps_as_known
            or int(period_ms) <= ONE_MIN_MS
            or not isinstance(arr, np.ndarray)
            or arr.size < 2
            or has_unidentifiable_rejected_row
        ):
            return arr
        period_ms = int(period_ms)
        first_requested_bucket = (
            (int(start_ts) + period_ms - 1) // period_ms
        ) * period_ms
        requested_end_exclusive = int(end_exclusive_ts)
        arr = np.sort(_ensure_dtype(arr), order="ts")
        rows = []
        changed = False
        for idx in range(arr.shape[0] - 1):
            current = arr[idx]
            following = arr[idx + 1]
            rows.append(tuple(current.tolist()))
            current_ts = int(current["ts"])
            following_ts = int(following["ts"])
            if following_ts <= current_ts + int(period_ms):
                continue
            missing_span_ms = following_ts - current_ts - period_ms
            tolerance_ms = int(self.gap_tolerance_ohlcvs_minutes * ONE_MIN_MS)
            if tolerance_ms <= 0 or missing_span_ms > tolerance_ms:
                continue
            previous_close = float(current["c"])
            synthesis_end = min(following_ts, requested_end_exclusive)
            if rejected_timestamps:
                barriers = [
                    int(ts)
                    for ts in rejected_timestamps
                    if current_ts < int(ts) < following_ts
                ]
                if barriers:
                    synthesis_end = min(synthesis_end, min(barriers))
            for ts in range(
                max(current_ts + period_ms, first_requested_bucket),
                synthesis_end,
                period_ms,
            ):
                rows.append(
                    (
                        int(ts),
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    )
                )
                changed = True
        rows.append(tuple(arr[-1].tolist()))
        if not changed:
            return arr
        return np.array(rows, dtype=CANDLE_DTYPE)

    async def _fetch_ohlcv_paginated(
        self,
        symbol: str,
        since_ms: int,
        end_exclusive_ms: int,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
        on_batch: Optional[Callable[[np.ndarray], None]] = None,
        raise_on_partial_empty_page: bool = False,
    ) -> np.ndarray:
        """Fetch OHLCV from `since_ms` up to but excluding `end_exclusive_ms`.

        Uses ccxt pagination via since+limit. Returns CANDLE_DTYPE array.
        """
        if self.exchange is None:
            return np.empty((0,), dtype=CANDLE_DTYPE)
        self._raise_if_shutdown_requested("fetch_ohlcv_paginated")
        since_start = int(since_ms)
        since = int(since_ms)
        end_excl = int(end_exclusive_ms)
        limit = self._ccxt_limit_default
        tf_norm = self._normalize_timeframe_arg(timeframe, tf, default=self._ccxt_timeframe)
        # Derive pagination step from timeframe
        period_ms = _tf_to_ms(tf_norm)
        # Some exchanges treat `since` as exclusive. Back up by overlap to avoid missing the first candle.
        if self._ccxt_since_exclusive and self._ccxt_page_overlap_candles > 0 and since > 0:
            overlap_ms = period_ms * int(self._ccxt_page_overlap_candles)
            since = max(0, since - overlap_ms)
        all_rows = []
        pages = 0
        prev_last_ts: Optional[int] = None
        verified_sparse_synthetic_count = 0
        verified_sparse_synthetic_pages = 0
        total_span = max(1, end_excl - since_start)

        def _partial_rows() -> np.ndarray:
            if not all_rows:
                return np.empty((0,), dtype=CANDLE_DTYPE)
            return np.sort(np.concatenate(all_rows), order="ts")

        def _raise_terminal_empty_page(message: str) -> None:
            raise OhlcvTerminalEmptyPage(
                (
                    f"{message} | exchange={self._ex_id} symbol={symbol} "
                    f"tf={tf_norm} since={since} end_exclusive={end_excl} pages={pages}"
                ),
                partial_rows=_partial_rows(),
                terminal_start_ts=since,
                requested_end_ts=end_excl,
                pages=pages,
            )

        def _requested_end_is_covered() -> bool:
            if not all_rows:
                return False
            return max(int(rows[-1]["ts"]) for rows in all_rows if rows.size) >= (
                end_excl - period_ms
            )

        while since < end_excl:
            self._raise_if_shutdown_requested("fetch_ohlcv_paginated")
            # Bitget auto-probe: try a larger limit once to see if the API supports it.
            probe_limit = None
            if (
                not self._ccxt_limit_probe_done
                and isinstance(self._ex_id, str)
                and "bitget" in self._ex_id.lower()
                and tf_norm == "1m"
            ):
                probe_limit = 1000
            use_limit = probe_limit or limit
            page = await self._ccxt_fetch_ohlcv_once(
                symbol, since, use_limit, end_exclusive_ms=end_excl, tf=tf_norm
            )
            if not page:
                if (
                    raise_on_partial_empty_page
                    and pages > 0
                    and not _requested_end_is_covered()
                ):
                    _raise_terminal_empty_page(
                        "ccxt OHLCV pagination returned an empty page before reaching "
                        "the requested end"
                    )
                break
            arr = self._normalize_ccxt_ohlcv(page)
            rejected_payload_timestamps = (
                self._rejected_ccxt_ohlcv_timestamps(page, arr)
                if self._record_payload_gaps_as_known and period_ms > ONE_MIN_MS
                else set()
            )
            has_unidentifiable_rejected_row = bool(
                self._record_payload_gaps_as_known
                and period_ms > ONE_MIN_MS
                and self._ccxt_ohlcv_has_unidentifiable_rejected_row(page)
            )
            rejected_ranges = []
            if has_unidentifiable_rejected_row:
                if arr.size >= 2:
                    accepted_ts = np.sort(arr["ts"].astype(np.int64, copy=False))
                    bounded_start = max(
                        int(since_start), int(accepted_ts[0]) + period_ms
                    )
                    bounded_end = min(
                        int(end_excl) - period_ms,
                        int(accepted_ts[-1]) - period_ms,
                    )
                else:
                    # With fewer than two accepted timestamps, the malformed row
                    # cannot be bounded inside the payload. Treat the remaining
                    # requested range as unavailable rather than allowing a
                    # previously persisted placeholder to prove continuity.
                    bounded_start = max(int(since_start), int(since))
                    bounded_end = int(end_excl) - period_ms
                if bounded_end >= bounded_start:
                    rejected_ranges.append((bounded_start, bounded_end))
            if rejected_payload_timestamps or rejected_ranges:
                self._evict_rejected_native_sparse_synthetics(
                    symbol,
                    timeframe=tf_norm,
                    rejected_timestamps=rejected_payload_timestamps,
                    rejected_ranges=rejected_ranges,
                )
            if arr.size == 0:
                if (
                    raise_on_partial_empty_page
                    and pages > 0
                    and not _requested_end_is_covered()
                ):
                    _raise_terminal_empty_page(
                        "ccxt OHLCV pagination returned an empty normalized page before "
                        "reaching the requested end"
                    )
                break
            raw_arr_size = int(arr.size)
            arr = self._synthesize_verified_sparse_payload_gaps(
                arr,
                period_ms=period_ms,
                start_ts=since_start,
                end_exclusive_ts=end_excl,
                rejected_timestamps=rejected_payload_timestamps,
                has_unidentifiable_rejected_row=has_unidentifiable_rejected_row,
            )
            if int(arr.size) > raw_arr_size:
                verified_sparse_synthetic_count += int(arr.size) - raw_arr_size
                verified_sparse_synthetic_pages += 1
            if probe_limit is not None and not self._ccxt_limit_probe_done:
                # If Bitget returns >200 rows, we can safely use 1000 going forward.
                if arr.shape[0] > 200:
                    self._ccxt_limit_default = 1000
                    limit = 1000
                    self._log(
                        "debug",
                        "bitget_ohlcv_limit_probe",
                        symbol=symbol,
                        tf=tf_norm,
                        supported_limit=1000,
                        rows=int(arr.shape[0]),
                    )
                else:
                    self._ccxt_limit_default = 200
                    limit = 200
                    self._log(
                        "debug",
                        "bitget_ohlcv_limit_probe",
                        symbol=symbol,
                        tf=tf_norm,
                        supported_limit=200,
                        rows=int(arr.shape[0]),
                    )
                self._ccxt_limit_probe_done = True
            # Exclude any candles >= end_exclusive
            arr = arr[arr["ts"] < end_excl]
            if arr.size == 0:
                if (
                    raise_on_partial_empty_page
                    and pages > 0
                    and not _requested_end_is_covered()
                ):
                    _raise_terminal_empty_page(
                        "ccxt OHLCV pagination returned only out-of-range candles before "
                        "reaching the requested end"
                    )
                break
            # Diagnostics: page ts range and step
            try:
                first_ts = int(arr[0]["ts"])  # type: ignore[index]
                last_ts = int(arr[-1]["ts"])  # type: ignore[index]
                if arr.shape[0] > 1:
                    diffs = np.diff(arr["ts"].astype(np.int64))
                    max_step = int(diffs.max())
                    min_step = int(diffs.min())
                    # Expect step to match the requested timeframe's period
                    # Log at DEBUG - unexpected steps are common on illiquid exchanges and aren't actionable
                    if max_step != period_ms or min_step != period_ms:
                        warn_key = (self._ex_id, symbol, tf_norm)
                        if warn_key not in self._step_warning_keys:
                            self._step_warning_keys.add(warn_key)
                            self.log.debug(
                                f"[candle] unexpected step for tf exchange={self._ex_id} symbol={symbol} tf={tf_norm} expected={period_ms} min_step={min_step} max_step={max_step}"
                            )
                else:
                    max_step = ONE_MIN_MS
            except Exception:
                first_ts = last_ts = 0
            # Record intra-payload holes as verified no-trade gaps: the exchange
            # affirmatively returned the surrounding candles in one response.
            # Between-page holes are not exchange-verified — a pagination stall
            # or outage produces the same shape — so they get the expiring
            # auto-detected classification and are re-verified on later fetches
            # instead of being permanently masked as no_trades.
            if self._record_payload_gaps_as_known and tf_norm == "1m":
                try:
                    ts_arr = arr["ts"].astype(np.int64)
                    if ts_arr.size > 1:
                        diffs = np.diff(ts_arr)
                        gap_idxs = np.where(diffs > period_ms)[0]
                        for i in gap_idxs:
                            gap_start = int(ts_arr[i] + period_ms)
                            gap_end = int(ts_arr[i + 1] - period_ms)
                            self._record_verified_gap(symbol, gap_start, gap_end)
                    if prev_last_ts is not None and first_ts > prev_last_ts + period_ms:
                        gap_start = int(prev_last_ts + period_ms)
                        gap_end = int(first_ts - period_ms)
                        self._add_known_gap(
                            symbol,
                            gap_start,
                            gap_end,
                            reason=GAP_REASON_AUTO,
                            increment_retry=True,
                        )
                except Exception:
                    pass

            all_rows.append(arr)
            pages += 1
            if self._page_debug_all or symbol in self._page_debug_symbols:
                self._log(
                    "info",
                    "ccxt_page_range",
                    symbol=symbol,
                    tf=tf_norm,
                    page=pages,
                    rows=int(arr.shape[0]),
                    first_ts=first_ts,
                    last_ts=last_ts,
                    since_ts=int(since),
                    end_exclusive_ts=int(end_excl),
                )
            if on_batch is not None:
                try:
                    on_batch(arr)
                except Exception as on_batch_err:
                    error_type = bounded_exception_type(on_batch_err)
                    self.log.error(
                        "on_batch callback failed; stopping pagination | "
                        "symbol=%s timeframe=%s error_type=%s",
                        symbol,
                        tf_norm,
                        error_type,
                        extra={
                            "symbol": symbol,
                            "timeframe": tf_norm,
                            "error_type": error_type,
                        },
                    )
                    break
            last_ts = int(arr[-1]["ts"])  # inclusive last
            if last_ts >= end_excl - period_ms:
                break
            # Throttled progress logs (INFO) for long-running paginated fetches
            try:
                progressed = max(
                    0, min(100.0, 100.0 * float(last_ts - since_start) / float(total_span))
                )
            except Exception:
                progressed = 0.0
            self._progress_log(
                (symbol, tf_norm, "ccxt"),
                "ccxt_fetch_progress",
                symbol=symbol,
                tf=tf_norm,
                pages=pages,
                rows=sum(int(a.shape[0]) for a in all_rows) if all_rows else 0,
                since_ts=since_start,
                end_exclusive_ts=end_excl,
                last_ts=last_ts,
                progress_pct=f"{progressed:.1f}",
            )
            new_since = last_ts + period_ms
            if self._ccxt_page_overlap_candles > 0:
                overlap_ms = period_ms * int(self._ccxt_page_overlap_candles)
                new_since = max(last_ts - overlap_ms, since + period_ms)
            # Safety to avoid infinite loops if exchange returns overlapping data
            if new_since <= since:
                self.log.debug(
                    f"pagination stop (no progress) exchange={self._ex_id} symbol={symbol} since={since} last_ts={last_ts}"
                )
                break
            since = new_since
            prev_last_ts = last_ts
        if verified_sparse_synthetic_count > 0:
            self._log(
                "info",
                "kucoin_sparse_payload_gaps_synthesized",
                symbol=symbol,
                tf=tf_norm,
                count=verified_sparse_synthetic_count,
                pages=verified_sparse_synthetic_pages,
                source="same_successful_payload",
            )
        self._log(
            "debug",
            "ccxt_fetch_paginated_done",
            symbol=symbol,
            tf=tf_norm,
            rows=sum(a.shape[0] for a in all_rows) if all_rows else 0,
        )
        if not all_rows:
            return np.empty((0,), dtype=CANDLE_DTYPE)
        return np.sort(np.concatenate(all_rows), order="ts")

    # ----- Public helpers required by tests -----

    def standardize_gaps(
        self,
        candles: np.ndarray,
        *,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        strict: bool = False,
        fill_leading_gaps: bool = False,
        fill_trailing_gaps: bool = True,
        assume_sorted: bool = False,
        symbol: Optional[str] = None,
        excluded_synthetic_ranges: Optional[List[Tuple[int, int]]] = None,
    ) -> np.ndarray:
        """Return a new array with zero-candles synthesized for missing minutes.

        Parameters
        ----------
        candles : np.ndarray
            Structured array of dtype CANDLE_DTYPE. Must be sorted by `ts`.
        start_ts : int, optional
            Inclusive start timestamp in ms. If None, inferred from first candle.
        end_ts : int, optional
            Inclusive end timestamp in ms. If None, inferred from last candle.
        strict : bool
            If True, raises when a gap exists and no previous candle is available
            to seed the synthesized zero-candle.
        fill_leading_gaps : bool
            If False (default), do NOT synthesize candles before the first real data point.
            This prevents creating fake flat data when data doesn't exist at start_ts.
            If True, forward-fill from first available candle to fill leading gaps.
        fill_trailing_gaps : bool
            If False, do NOT synthesize an open-ended tail after the last real candle.
            Missing spans are synthesized only when bounded by real candles before
            and after the gap.
        assume_sorted : bool
            If True, skip sorting (caller guarantees array is already sorted by ts).
        excluded_synthetic_ranges : list[tuple[int, int]], optional
            Inclusive missing-minute ranges that must remain absent rather than
            being represented by synthetic zero-volume continuity candles.
        """
        a = _ensure_dtype(candles)
        if a.size == 0:
            # Nothing to standardize; caller decides how to handle empty ranges
            return a

        if not assume_sorted:
            # Check if already sorted to skip O(n log n) sort
            ts_check = a["ts"]
            if ts_check.size > 1 and not np.all(ts_check[:-1] <= ts_check[1:]):
                a = np.sort(a, order="ts")
        ts_arr = _ts_index(a)

        # Determine effective boundaries
        first_real_ts = int(ts_arr[0])
        last_real_ts = int(ts_arr[-1])

        lo = start_ts if start_ts is not None else first_real_ts
        hi = end_ts if end_ts is not None else last_real_ts
        lo = _floor_minute(lo)
        hi = _floor_minute(hi)

        # If not filling leading gaps, don't start before actual data
        effective_lo = lo
        if not fill_leading_gaps and first_real_ts > lo:
            leading_gap_minutes = (first_real_ts - lo) // ONE_MIN_MS
            if leading_gap_minutes > 0:
                self._log(
                    "debug",
                    "standardize_gaps_skipping_leading",
                    requested_start_ts=lo,
                    actual_start_ts=first_real_ts,
                    skipped_minutes=int(leading_gap_minutes),
                )
            effective_lo = _floor_minute(first_real_ts)

        effective_hi = hi
        if not fill_trailing_gaps and last_real_ts < hi:
            trailing_gap_minutes = (hi - last_real_ts) // ONE_MIN_MS
            if trailing_gap_minutes > 0:
                self._record_skipped_trailing_gap(
                    symbol=symbol,
                    requested_end_ts=hi,
                    actual_end_ts=last_real_ts,
                    skipped_minutes=int(trailing_gap_minutes),
                )
            effective_hi = _floor_minute(last_real_ts)
        if effective_hi < effective_lo:
            return np.empty((0,), dtype=CANDLE_DTYPE)

        if strict:
            # In strict mode: do not synthesize zero-candles.
            # If there are gaps, log a warning and return whatever real candles exist in range.
            i0 = int(np.searchsorted(ts_arr, effective_lo, side="left"))
            i1 = int(np.searchsorted(ts_arr, effective_hi, side="right"))
            missing_count = 0
            try:
                expected_len = int((effective_hi - effective_lo) // ONE_MIN_MS) + 1
                slice_ts = ts_arr[i0:i1].astype(np.int64, copy=False)
                if slice_ts.size:
                    # Missing at head + tail + internal gaps (but NOT leading gaps if not filling)
                    if fill_leading_gaps:
                        missing_count += int((int(slice_ts[0]) - effective_lo) // ONE_MIN_MS)
                    missing_count += int((effective_hi - int(slice_ts[-1])) // ONE_MIN_MS)
                    if slice_ts.size > 1:
                        diffs = np.diff(slice_ts)
                        gaps = diffs[diffs > ONE_MIN_MS]
                        if gaps.size:
                            missing_count += int(np.sum((gaps // ONE_MIN_MS) - 1))
                    # If duplicates exist, treat them as missing coverage too
                    missing_count += int(
                        max(0, expected_len - int(np.unique(slice_ts).size) - missing_count)
                    )
                else:
                    missing_count = expected_len
            except Exception:
                # fallback: keep behavior safe (no warning rather than exploding)
                missing_count = 0
            if missing_count:
                # Accumulate for summary logging instead of per-event warnings
                sym_key = symbol or "unknown"
                self._record_strict_gap(sym_key, int(missing_count))
                self._log_strict_gaps_summary()
            return a[i0:i1]

        # Complete ranges are overwhelmingly the common live path.  Avoid
        # rebuilding every real candle through Python tuples when there is
        # nothing to synthesize.  Keep returning a new array as documented.
        i0 = int(np.searchsorted(ts_arr, effective_lo, side="left"))
        i1 = int(np.searchsorted(ts_arr, effective_hi, side="right"))
        expected_len = int((effective_hi - effective_lo) // ONE_MIN_MS) + 1
        complete = a[i0:i1]
        if (
            complete.shape[0] == expected_len
            and int(complete[0]["ts"]) == effective_lo
            and int(complete[-1]["ts"]) == effective_hi
            and (
                expected_len == 1
                or np.all(np.diff(_ts_index(complete)) == ONE_MIN_MS)
            )
        ):
            return complete.copy()

        expected = np.arange(
            effective_lo,
            effective_hi + ONE_MIN_MS,
            ONE_MIN_MS,
            dtype=np.int64,
        )
        # Map from ts to row index in a
        pos = {int(t): i for i, t in enumerate(ts_arr)}
        excluded_ranges = sorted(
            (
                (max(effective_lo, int(start)), min(effective_hi, int(end)))
                for start, end in (excluded_synthetic_ranges or [])
                if int(start) <= effective_hi and int(end) >= effective_lo
            ),
            key=lambda item: item[0],
        )
        excluded_range_idx = 0

        def excluded_from_synthesis(timestamp: int) -> bool:
            nonlocal excluded_range_idx
            while (
                excluded_range_idx < len(excluded_ranges)
                and excluded_ranges[excluded_range_idx][1] < timestamp
            ):
                excluded_range_idx += 1
            return (
                excluded_range_idx < len(excluded_ranges)
                and excluded_ranges[excluded_range_idx][0]
                <= timestamp
                <= excluded_ranges[excluded_range_idx][1]
            )

        out_rows = []
        prev_close: Optional[float] = None

        # Seed prev_close from:
        # 1) the candle exactly at effective_lo, else
        # 2) the last candle before effective_lo (ffill from earlier data), else
        # 3) if fill_leading_gaps=True, use the first available candle (bfill for leading gaps)
        if effective_lo in pos:
            prev_close = float(a[pos[effective_lo]]["c"])
        else:
            idx = int(np.searchsorted(ts_arr, effective_lo))
            if idx > 0:
                # There's a candle before effective_lo - use it for ffill
                prev_close = float(a[idx - 1]["c"])
            elif fill_leading_gaps and a.size > 0:
                # No candle before effective_lo, but fill_leading_gaps=True
                # Use first candle's close to backward-fill leading gaps
                prev_close = float(a[0]["c"])
            # If no candle before, prev_close stays None until we hit real data

        synthesized_count = 0
        synthesized_timestamps: List[int] = []
        for t in expected:
            if t in pos:
                row = a[pos[t]]
                out_rows.append(tuple(row.tolist()))
                prev_close = float(row["c"])  # update seed
            else:
                if excluded_from_synthesis(int(t)):
                    continue
                if prev_close is None:
                    # No previous data to forward-fill from - skip this timestamp
                    continue
                # Synthesize a zero-candle using previous close (internal gaps only)
                out_rows.append((int(t), prev_close, prev_close, prev_close, prev_close, 0.0))
                synthesized_timestamps.append(int(t))
                synthesized_count += 1

        # Track synthetic timestamps for EMA recomputation detection
        if symbol and synthesized_timestamps:
            self._track_synthetic_timestamps(symbol, synthesized_timestamps)

        # Log when zero-candles were synthesized (rate-limited or batched)
        if synthesized_count > 0 and symbol:
            # In batch mode, collect for later aggregated logging
            if self._synth_candle_batch_mode:
                try:
                    first_ts = min(synthesized_timestamps)
                    last_ts = max(synthesized_timestamps)
                except Exception:
                    first_ts = None
                    last_ts = None
                meta = self._synth_candle_batch.get(symbol)
                if not isinstance(meta, dict):
                    meta = {"count": 0, "min_ts": None, "max_ts": None}
                meta["count"] = int(meta.get("count", 0)) + int(synthesized_count)
                if first_ts is not None:
                    try:
                        meta["min_ts"] = (
                            int(first_ts)
                            if meta.get("min_ts") is None
                            else min(int(meta["min_ts"]), int(first_ts))
                        )
                    except Exception:
                        meta["min_ts"] = int(first_ts)
                if last_ts is not None:
                    try:
                        meta["max_ts"] = (
                            int(last_ts)
                            if meta.get("max_ts") is None
                            else max(int(meta["max_ts"]), int(last_ts))
                        )
                    except Exception:
                        meta["max_ts"] = int(last_ts)
                self._synth_candle_batch[symbol] = meta
            else:
                # Normal mode: deduplicate by gap start (only warn once per unique gap origin)
                # Round first_ts to nearest hour to reduce duplicate warnings when the same
                # underlying gap is detected at slightly different boundaries in different fetch windows
                first_ts = min(synthesized_timestamps)
                last_ts = max(synthesized_timestamps)
                hour_ms = 3600_000
                first_ts_hour = (first_ts // hour_ms) * hour_ms  # Floor to hour boundary
                gap_key = (symbol, first_ts_hour)

                # Skip if we've already warned about a gap starting in this hour window
                if gap_key in self._synth_gap_warned:
                    pass  # Already warned, skip
                else:
                    self._synth_gap_warned.add(gap_key)
                    # Format timestamp range for human readability
                    from datetime import datetime, timezone

                    first_dt = datetime.fromtimestamp(first_ts / 1000, tz=timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M"
                    )
                    if synthesized_count == 1:
                        ts_info = first_dt
                    else:
                        last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).strftime(
                            "%Y-%m-%dT%H:%M"
                        )
                        ts_info = f"{first_dt} to {last_dt}"
                    # KuCoin futures returns sparse no-trade minutes, so even large individual
                    # synthetic runs are expected. Other exchanges keep WARNING for large gaps.
                    sparse_expected = self._sparse_ohlcv_gaps_are_expected()
                    log_fn = (
                        self.log.debug
                        if sparse_expected
                        else self.log.warning
                        if synthesized_count > 1000
                        else self.log.debug
                    )
                    suffix = (
                        " | expected on sparse KuCoin no-trade minutes"
                        if sparse_expected
                        else ""
                    )
                    log_fn(
                        "[candle] %s: synthesized %d zero-candle%s at %s "
                        "(no data for requested minutes) using prev_close=%.6f%s",
                        _log_symbol(symbol),
                        synthesized_count,
                        "s" if synthesized_count > 1 else "",
                        ts_info,
                        prev_close if prev_close is not None else 0.0,
                        suffix,
                    )

        if not out_rows:
            return np.empty((0,), dtype=CANDLE_DTYPE)
        return np.array(out_rows, dtype=CANDLE_DTYPE)

    # ----- External archives (historical) -----

    def _archive_supported(self) -> bool:
        """Check if archive fetching is supported and enabled for this exchange."""
        if not self.archive_enabled:
            return False
        try:
            exid = (self._ex_id or "").lower() if isinstance(self._ex_id, str) else ""
        except Exception:
            exid = ""
        # Note: Bybit excluded - CCXT is faster and uses far less bandwidth.
        # Bybit's archive endpoint provides raw trades (not bucketed OHLCVs), so fetching
        # and bucketing trades costs ~700x more data for BTC. Keep the archive fetch logic
        # below for reference/optional use, but avoid it by default.
        return exid in {"binanceusdm", "bitget", "kucoinfutures", "hyperliquid"}

    @staticmethod
    def _archive_symbol_code(symbol: str) -> str:
        """Return archive symbol code (typically BASEQUOTE) for ccxt-style symbols."""
        symbol = str(symbol or "")
        if not symbol:
            return ""
        base = symbol
        quote = ""
        if "/" in symbol:
            base, rest = symbol.split("/", 1)
            quote = rest.split(":", 1)[0] if ":" in rest else rest
        elif ":" in symbol:
            base, quote = symbol.split(":", 1)
        # best-effort fallback
        base = (base or "").replace("/", "").replace(":", "")
        quote = (quote or "").replace("/", "").replace(":", "")
        return f"{base}{quote}" if quote else base

    async def _archive_fetch_day(self, symbol: str, day_key: str) -> Optional[np.ndarray]:
        """Fetch a full-day (1440x1m) candle array from external archives.

        Returns CANDLE_DTYPE with inclusive timestamps spanning the UTC day, or None if not available.
        """
        try:
            exid = (self._ex_id or "").lower() if isinstance(self._ex_id, str) else ""
        except Exception:
            exid = ""
        if exid not in {"binanceusdm", "bybit", "bitget", "kucoinfutures", "hyperliquid"}:
            return None

        symbol_code = self._archive_symbol_code(symbol)
        if not symbol_code:
            return None

        if exid == "kucoinfutures":
            symbol_code = f"{symbol_code}M"

        if exid == "binanceusdm":
            url = (
                "https://data.binance.vision/data/futures/um/"
                f"daily/klines/{symbol_code}/1m/{symbol_code}-1m-{day_key}.zip"
            )
            return await self._archive_fetch_binance_zip(url, day_key)

        if exid == "bybit":
            # Note: Bybit archive provides raw trades, not bucketed OHLCVs.
            # It's intentionally disabled by _archive_supported() by default due to bandwidth.
            url = f"https://public.bybit.com/trading/{symbol_code}/{symbol_code}{day_key}.csv.gz"
            return await self._archive_fetch_bybit_trades(url, day_key)

        if exid == "bitget":
            # Bitget archive layout varies by date; mirror existing logic.
            day_comp = day_key
            day_yymmdd = day_key.replace("-", "")
            if day_comp <= "2024-04-18":
                url = (
                    "https://img.bitgetimg.com/online/kline/"
                    f"{symbol_code}/{symbol_code}_UMCBL_1min_{day_yymmdd}.zip"
                )
            else:
                url = f"https://img.bitgetimg.com/online/kline/{symbol_code}/UMCBL/{day_yymmdd}.zip"
            return await self._archive_fetch_bitget_zip(url, day_key)

        if exid == "kucoinfutures":
            url = (
                "https://historical-data.kucoin.com/data/futures/daily/klines/"
                f"{symbol_code}/1m/{symbol_code}-1m-{day_key}.zip"
            )
            return await self._archive_fetch_kucoin_zip(url, day_key)

        if exid == "hyperliquid":
            return await self._archive_fetch_hyperliquid(symbol, day_key)

        return None

    async def _get_http_session(self) -> "aiohttp.ClientSession":
        """Get or create a persistent HTTP session for archive fetches."""
        import aiohttp

        async with self._http_session_lock:
            if self._http_session is None or self._http_session.closed:
                # Archive hosts can be slow and archives can be large; use tolerant timeouts.
                # Keep connect timeout bounded, but allow more time for reads.
                timeout = aiohttp.ClientTimeout(total=120, connect=20, sock_read=60)
                connector = aiohttp.TCPConnector(
                    # Keep concurrency moderate to avoid timeouts under load.
                    limit=20,
                    limit_per_host=6,
                    ttl_dns_cache=300,  # DNS cache TTL in seconds
                    enable_cleanup_closed=True,
                )
                self._http_session = aiohttp.ClientSession(
                    timeout=timeout,
                    connector=connector,
                )
            return self._http_session

    async def _close_http_session(self) -> None:
        """Close the HTTP session if open."""
        async with self._http_session_lock:
            if self._http_session is not None and not self._http_session.closed:
                await self._http_session.close()
                self._http_session = None

    async def _archive_fetch_bytes(self, url: str) -> Optional[bytes]:
        t0 = time.monotonic()
        url_hash = _remote_fetch_url_hash(url)
        self._emit_remote_fetch(
            {
                "kind": "archive_http_get",
                "stage": "start",
                "exchange": str(self._ex_id),
                "url": str(url),
            }
        )
        self._log("debug", "archive_http_get", url_hash=url_hash)

        session = await self._get_http_session()
        try:
            async with session.get(url) as resp:
                if resp.status == 404:
                    self._emit_remote_fetch(
                        {
                            "kind": "archive_http_get",
                            "stage": "not_found",
                            "exchange": str(self._ex_id),
                            "url": str(url),
                            "status": 404,
                            "elapsed_ms": int((time.monotonic() - t0) * 1000),
                        }
                    )
                    self._log(
                        "debug",
                        "archive_http_404",
                        url_hash=url_hash,
                        elapsed_ms=int((time.monotonic() - t0) * 1000),
                    )
                    return None
                resp.raise_for_status()
                data = await resp.read()
        except Exception as e:
            err_type = bounded_exception_type(e)
            self._emit_remote_fetch(
                {
                    "kind": "archive_http_get",
                    "stage": "error",
                    "exchange": str(self._ex_id),
                    "url": str(url),
                    "error_type": err_type,
                    "elapsed_ms": int((time.monotonic() - t0) * 1000),
                }
            )
            self._log(
                "debug",
                "archive_http_error",
                url_hash=url_hash,
                error_type=err_type,
            )
            raise

        self._emit_remote_fetch(
            {
                "kind": "archive_http_get",
                "stage": "ok",
                "exchange": str(self._ex_id),
                "url": str(url),
                "bytes": int(len(data)),
                "elapsed_ms": int((time.monotonic() - t0) * 1000),
            }
        )
        self._log(
            "debug",
            "archive_http_ok",
            url_hash=url_hash,
            bytes=len(data),
            elapsed_ms=int((time.monotonic() - t0) * 1000),
        )
        return data

    async def _archive_fetch_binance_zip(self, url: str, day_key: str) -> Optional[np.ndarray]:
        raw = await self._archive_fetch_bytes(url)
        if raw is None:
            return None
        import zipfile
        from io import BytesIO
        import pandas as pd

        col_names = ["timestamp", "open", "high", "low", "close", "volume"]
        with zipfile.ZipFile(BytesIO(raw), "r") as z:
            dfs = []
            for name in z.namelist():
                with z.open(name) as f:
                    df = pd.read_csv(f, header=None)
                df.columns = col_names + [
                    f"extra_{i}" for i in range(len(df.columns) - len(col_names))
                ]
                dfs.append(df[col_names])
        if not dfs:
            return None
        dfc = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
        dfc = dfc[dfc.timestamp != "open_time"]
        for c in col_names:
            dfc[c] = pd.to_numeric(dfc[c], errors="coerce")
        dfc = dfc.dropna(subset=["timestamp"]).reset_index(drop=True)
        start_ts, end_ts = self._date_range_of_key(day_key)
        # Binance timestamps should already be ms.
        dfc = dfc[(dfc["timestamp"] >= start_ts) & (dfc["timestamp"] <= end_ts)]
        if dfc.empty:
            return None
        return self._ohlcv_df_to_day_arr(dfc, day_key)

    async def _archive_fetch_bitget_zip(self, url: str, day_key: str) -> Optional[np.ndarray]:
        raw = await self._archive_fetch_bytes(url)
        if raw is None:
            return None
        import zipfile
        from io import BytesIO
        import pandas as pd

        col_names = ["timestamp", "open", "high", "low", "close", "volume"]
        with zipfile.ZipFile(BytesIO(raw), "r") as z:
            dfs = []
            for name in z.namelist():
                with z.open(name) as f:
                    # Bitget provides xlsx-like sheets; pandas can read excel from bytes.
                    df = pd.read_excel(f)
                df.columns = col_names + [
                    f"extra_{i}" for i in range(len(df.columns) - len(col_names))
                ]
                dfs.append(df[col_names])
        if not dfs:
            return None
        dfc = pd.concat(dfs).sort_values("timestamp").reset_index(drop=True)
        for c in col_names:
            dfc[c] = pd.to_numeric(dfc[c], errors="coerce")
        dfc = dfc.dropna(subset=["timestamp"]).reset_index(drop=True)
        start_ts, end_ts = self._date_range_of_key(day_key)
        # Bitget timestamps sometimes come in seconds.
        ts = dfc["timestamp"].astype("float64").values
        if np.isfinite(ts).any() and float(np.nanmax(np.abs(ts))) < 1e11:
            dfc["timestamp"] = dfc["timestamp"] * 1000.0
        dfc = dfc[(dfc["timestamp"] >= start_ts) & (dfc["timestamp"] <= end_ts)]
        if dfc.empty:
            return None
        return self._ohlcv_df_to_day_arr(dfc, day_key)

    async def _archive_fetch_kucoin_zip(self, url: str, day_key: str) -> Optional[np.ndarray]:
        raw = await self._archive_fetch_bytes(url)
        if raw is None:
            return None
        import zipfile
        from io import BytesIO
        import pandas as pd

        required = ["timestamp", "open", "high", "low", "close", "volume"]
        with zipfile.ZipFile(BytesIO(raw), "r") as z:
            dfs = []
            for name in z.namelist():
                with z.open(name) as f:
                    df = pd.read_csv(f)
                df.columns = [str(c).strip().lower() for c in df.columns]
                if "time" in df.columns and "timestamp" not in df.columns:
                    df = df.rename(columns={"time": "timestamp"})
                missing = [c for c in required if c not in df.columns]
                if missing:
                    raise ValueError(f"kucoin archive missing columns {missing} in {url}")
                dfs.append(df[required])
        if not dfs:
            return None
        dfc = pd.concat(dfs, ignore_index=True)
        for c in required:
            dfc[c] = pd.to_numeric(dfc[c], errors="coerce")
        dfc = dfc.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
        # Kucoin timestamps are typically seconds.
        ts = dfc["timestamp"].astype("float64").values
        if np.isfinite(ts).any() and float(np.nanmax(np.abs(ts))) < 1e11:
            dfc["timestamp"] = dfc["timestamp"] * 1000.0
        start_ts, end_ts = self._date_range_of_key(day_key)
        dfc = dfc[(dfc["timestamp"] >= start_ts) & (dfc["timestamp"] <= end_ts)]
        if dfc.empty:
            return None
        return self._ohlcv_df_to_day_arr(dfc, day_key)

    async def _archive_fetch_bybit_trades(self, url: str, day_key: str) -> Optional[np.ndarray]:
        raw = await self._archive_fetch_bytes(url)
        if raw is None:
            return None
        import gzip
        from io import BytesIO
        import pandas as pd

        with gzip.open(BytesIO(raw)) as f:
            trades = pd.read_csv(f)
        if "timestamp" not in trades.columns or "price" not in trades.columns:
            return None
        # Bybit archive timestamps are in seconds (trade time).
        ts_sec = pd.to_numeric(trades["timestamp"], errors="coerce").astype("float64")
        price = pd.to_numeric(trades["price"], errors="coerce").astype("float64")
        size = pd.to_numeric(trades.get("size", 0.0), errors="coerce").astype("float64")
        trades = pd.DataFrame({"timestamp": ts_sec, "price": price, "size": size}).dropna(
            subset=["timestamp", "price"]
        )
        if trades.empty:
            return None
        interval = 60_000
        minute_ts = (trades["timestamp"] * 1000.0) // interval * interval
        groups = trades.groupby(minute_ts)
        ohlcvs = pd.DataFrame(
            {
                "open": groups["price"].first(),
                "high": groups["price"].max(),
                "low": groups["price"].min(),
                "close": groups["price"].last(),
                "volume": groups["size"].sum(),
            }
        )
        ohlcvs["timestamp"] = ohlcvs.index.astype("int64")
        ohlcvs = ohlcvs.reset_index(drop=True)
        start_ts, end_ts = self._date_range_of_key(day_key)
        ohlcvs = ohlcvs[(ohlcvs["timestamp"] >= start_ts) & (ohlcvs["timestamp"] <= end_ts)]
        if ohlcvs.empty:
            return None
        return self._ohlcv_df_to_day_arr(ohlcvs, day_key)

    async def _archive_fetch_hyperliquid(self, symbol: str, day_key: str) -> Optional[np.ndarray]:
        """Fetch Hyperliquid archive data for backtesting.

        Data sources tried in order:
        1. Local pre-processed cache (caches/ohlcv/hyperliquid/{coin}/{day_key}.parquet)
        2. For stock perps: TradFi API (if credentials available in api-keys.json)

        For Hyperliquid's S3 raw trade data, users can pre-process it using:
            python -m src.tools.hyperliquid_s3_fetcher --start YYYY-MM-DD --end YYYY-MM-DD
            python -m src.tools.trades_to_ohlcv --input caches/hyperliquid_trades --output caches/ohlcv/hyperliquid

        Args:
            symbol: CCXT-style symbol (e.g., "BTC/USDC:USDC" or "xyz:TSLA/USDC:USDC")
            day_key: Date string (YYYY-MM-DD)

        Returns:
            CANDLE_DTYPE array with 1440 candles, or None if not available
        """
        import pandas as pd
        from pathlib import Path

        # Derive coin name from symbol for cache path
        base = symbol.split("/")[0] if "/" in symbol else symbol
        # Handle xyz: prefix in path (replace : with _ for filesystem)
        safe_coin = base.replace(":", "_")

        # 1. Check local pre-processed cache first
        cache_path = Path("caches/ohlcv/hyperliquid") / safe_coin / f"{day_key}.parquet"

        if cache_path.exists():
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(cache_path)
                df = table.to_pandas()

                # Rename columns to match expected format
                col_map = {
                    "ts": "timestamp",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "bv": "volume",
                }
                df = df.rename(columns=col_map)

                self._log(
                    "debug",
                    "hyperliquid_archive_hit",
                    symbol=symbol,
                    day_key=day_key,
                    path=str(cache_path),
                )
                return self._ohlcv_df_to_day_arr(df, day_key)
            except Exception as e:
                self._log(
                    "debug",
                    "hyperliquid_archive_error",
                    symbol=symbol,
                    day_key=day_key,
                    error_type=bounded_exception_type(e),
                )

        # 2. For stock perps, try TradFi data fetch
        try:
            from tradfi_data import is_stock_ticker, hip3_to_tradfi_symbol
        except ImportError:
            self._log("debug", "hyperliquid_archive_miss", symbol=symbol, day_key=day_key)
            return None

        if is_stock_ticker(base):
            arr = await self._fetch_tradfi_day(base, day_key, cache_path)
            if arr is not None and arr.size > 0:
                return arr

        self._log("debug", "hyperliquid_archive_miss", symbol=symbol, day_key=day_key)
        return None

    async def _fetch_tradfi_day(
        self, coin: str, day_key: str, cache_path: "Path"
    ) -> Optional[np.ndarray]:
        """Fetch stock data from TradFi API and cache it.

        Args:
            coin: Stock ticker (e.g., "TSLA", "xyz:TSLA")
            day_key: Date string (YYYY-MM-DD)
            cache_path: Path to save cached data

        Returns:
            CANDLE_DTYPE array or None
        """
        from pathlib import Path

        try:
            from tradfi_data import (
                get_provider,
                hip3_to_tradfi_symbol,
                TradFiDataFetcher,
            )
        except ImportError:
            return None

        # Load TradFi credentials from api-keys.json
        # Default to yfinance (free, no API key required)
        tradfi_config = self._load_tradfi_config()
        if tradfi_config:
            provider_name = tradfi_config.get("provider", "yfinance")
            api_key = tradfi_config.get("api_key")
            api_secret = tradfi_config.get("api_secret")  # For Alpaca
        else:
            # Use yfinance as free default
            provider_name = "yfinance"
            api_key = None
            api_secret = None

        try:
            provider = get_provider(provider_name, api_key=api_key, api_secret=api_secret)
            ticker = hip3_to_tradfi_symbol(coin)

            self._log("info", "tradfi_fetch", ticker=ticker, day_key=day_key, provider=provider_name)

            async with TradFiDataFetcher(provider) as fetcher:
                # Construct HIP-3 symbol for the fetcher
                hip3_symbol = f"xyz:{ticker}/USDC:USDC" if not coin.startswith("xyz:") else coin
                arr = await fetcher.fetch_day(hip3_symbol, day_key)

            if arr is not None and arr.size > 0:
                # Cache the result for future use
                self._save_tradfi_cache(arr, cache_path)
                # Convert to day array format
                import pandas as pd

                df = pd.DataFrame(
                    {
                        "timestamp": arr["ts"],
                        "open": arr["o"],
                        "high": arr["h"],
                        "low": arr["l"],
                        "close": arr["c"],
                        "volume": arr["bv"],
                    }
                )
                return self._ohlcv_df_to_day_arr(df, day_key)

        except Exception as e:
            self._log(
                "debug",
                "tradfi_fetch_error",
                coin=coin,
                day_key=day_key,
                error_type=bounded_exception_type(e),
            )

        return None

    def _load_tradfi_config(self) -> Optional[dict]:
        """Load TradFi API configuration from api-keys.json."""
        from pathlib import Path
        import json

        api_keys_path = Path("api-keys.json")
        if not api_keys_path.exists():
            return None

        try:
            with open(api_keys_path) as f:
                api_keys = json.load(f)
            return api_keys.get("tradfi")
        except Exception:
            return None

    def _save_tradfi_cache(self, arr: np.ndarray, cache_path: "Path") -> None:
        """Save TradFi data to local cache."""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            cache_path.parent.mkdir(parents=True, exist_ok=True)

            table = pa.table(
                {
                    "ts": pa.array(arr["ts"].astype("int64")),
                    "o": pa.array(arr["o"].astype("float32")),
                    "h": pa.array(arr["h"].astype("float32")),
                    "l": pa.array(arr["l"].astype("float32")),
                    "c": pa.array(arr["c"].astype("float32")),
                    "bv": pa.array(arr["bv"].astype("float32")),
                }
            )
            pq.write_table(table, cache_path, compression="zstd")
            self._log("debug", "tradfi_cache_saved", path=str(cache_path))
        except Exception as e:
            self._log(
                "debug",
                "tradfi_cache_save_error",
                path=str(cache_path),
                error_type=bounded_exception_type(e),
            )

    def _ohlcv_df_to_day_arr(self, df, day_key: str) -> np.ndarray:
        """Convert a dataframe with timestamp/open/high/low/close/volume to 1m day array."""
        start_ts, end_ts = self._date_range_of_key(day_key)
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for c in cols:
            df[c] = df[c].astype("float64")
        df = (
            df.dropna(subset=["timestamp", "close"])
            .sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"], keep="last")
        )
        # Convert to CANDLE_DTYPE and then standardize to full-day grid.
        arr = np.empty((df.shape[0],), dtype=CANDLE_DTYPE)
        arr["ts"] = df["timestamp"].astype("int64").values
        arr["o"] = df["open"].values
        arr["h"] = df["high"].values
        arr["l"] = df["low"].values
        arr["c"] = df["close"].values
        arr["bv"] = df["volume"].values
        arr = arr[(arr["ts"] >= start_ts) & (arr["ts"] <= end_ts)]
        if arr.size == 0:
            return np.empty((0,), dtype=CANDLE_DTYPE)
        # Archive rows are persisted as real cache data. Never synthesize edge rows here:
        # on listing/delisting days, leading or trailing fills would turn future/old
        # prices into apparently tradable historical candles. Internal no-trade gaps
        # remain standardized between real candles.
        out = self.standardize_gaps(
            arr,
            start_ts=start_ts,
            end_ts=end_ts,
            strict=False,
            fill_leading_gaps=False,
            fill_trailing_gaps=False,
        )
        if out.size == 0 or int(out[0]["ts"]) < start_ts or int(out[-1]["ts"]) > end_ts:
            return np.empty((0,), dtype=CANDLE_DTYPE)
        return out

    async def _prefetch_archives_for_range(
        self, symbol: str, start_ts: int, end_ts: int, *, parallel_days: int = 5
    ) -> None:
        """Try to materialize missing full-day shards using external archives.

        Args:
            symbol: The symbol to fetch archives for
            start_ts: Start timestamp (ms)
            end_ts: End timestamp (ms)
            parallel_days: Number of days to fetch in parallel (default 5)
        """
        if not self._archive_supported():
            return

        # For stock-perps with TradFi configured, allow fetches before exchange inception
        # so historical data can be backfilled from TradFi providers (alpaca/polygon/etc.).
        allow_pre_inception_for_stock_perp = False
        try:
            from tradfi_data import is_stock_ticker

            base = symbol.split("/")[0].strip()
            tradfi_cfg = self._load_tradfi_config()
            allow_pre_inception_for_stock_perp = bool(tradfi_cfg) and is_stock_ticker(base)
        except Exception:
            allow_pre_inception_for_stock_perp = False

        # Clip/skip only when we know an authoritative lower bound for exchange-available
        # history. Earliest cached shard is observed local coverage, not proof of
        # exchange inception, so it must not suppress earlier backfills.
        authoritative_start_ts = self._get_authoritative_start_ts(symbol)
        if (
            authoritative_start_ts is not None
            and start_ts < authoritative_start_ts
            and not allow_pre_inception_for_stock_perp
        ):
            if authoritative_start_ts > end_ts:
                pre_inception_end = min(authoritative_start_ts - ONE_MIN_MS, end_ts)
                if start_ts <= pre_inception_end:
                    self._add_known_gap(
                        symbol,
                        start_ts,
                        pre_inception_end,
                        reason="pre_inception",
                        retry_count=_GAP_MAX_RETRIES,
                    )
                    self._log(
                        "warning",
                        "skip_pre_inception_fetch",
                        symbol=symbol,
                        original_start=start_ts,
                        original_end=end_ts,
                        authoritative_start_ts=authoritative_start_ts,
                        uncovered_start=start_ts,
                        uncovered_end=pre_inception_end,
                    )
                return

            pre_inception_end = min(authoritative_start_ts - ONE_MIN_MS, end_ts)
            if start_ts <= pre_inception_end:
                self._add_known_gap(
                    symbol,
                    start_ts,
                    pre_inception_end,
                    reason="pre_inception",
                    retry_count=_GAP_MAX_RETRIES,
                )
                self._log(
                    "warning",
                    "skip_pre_inception_fetch",
                    symbol=symbol,
                    original_start=start_ts,
                    original_end=end_ts,
                    authoritative_start_ts=authoritative_start_ts,
                    uncovered_start=start_ts,
                    uncovered_end=pre_inception_end,
                )
            start_ts = authoritative_start_ts
            if start_ts > end_ts:
                return  # Nothing left to fetch

        day_map = self._date_keys_between(start_ts, end_ts)
        shard_paths = self._iter_shard_paths(symbol, tf="1m")
        legacy_paths = self._get_legacy_shard_paths(symbol, "1m")

        # Determine primary shard completeness via index.json (cheap; avoids loading npy files).
        idx_shards: Dict[str, Dict[str, Any]] = {}
        try:
            idx = self._ensure_symbol_index(symbol, tf="1m")
            idx_shards = idx.get("shards") or {}
            if not isinstance(idx_shards, dict):
                idx_shards = {}
        except Exception:
            idx_shards = {}

        # Don't try to fetch archives for recent days - they don't exist yet
        # Exchanges typically need 48-72 hours to publish archive data
        archive_freshness_hours = 72
        archive_cutoff_ms = _utc_now_ms() - (archive_freshness_hours * 3600 * 1000)

        # First pass: count days to fetch
        days_to_fetch = []
        skipped_reasons = {
            "partial_day_request": 0,
            "too_recent": 0,
            "legacy_present": 0,
            "primary_complete": 0,
            "verified_from_disk": 0,  # Files verified by loading (no index metadata)
        }
        for day_key, (day_start, day_end) in day_map.items():
            if start_ts > day_start or end_ts < day_end:
                skipped_reasons["partial_day_request"] += 1
                continue  # not a full-day request for this day
            if day_end > archive_cutoff_ms:
                skipped_reasons["too_recent"] += 1
                continue  # too recent - archive not available yet, use CCXT
            if day_key in legacy_paths:
                skipped_reasons["legacy_present"] += 1
                continue  # legacy cache already covers this day

            # Only fetch archives for days missing or incomplete in primary.
            # NOTE: Previously we skipped any day with an existing primary shard path.
            # That can block archive healing if a prior CCXT run wrote a partial/incomplete day.
            if day_key in shard_paths:
                meta = idx_shards.get(day_key) if isinstance(idx_shards, dict) else None
                try:
                    if isinstance(meta, dict):
                        # full UTC day coverage (inclusive endpoints)
                        if (
                            int(meta.get("count") or -1) == 1440
                            and int(meta.get("min_ts") or 0) == int(day_start)
                            and int(meta.get("max_ts") or 0) == int(day_end)
                        ):
                            skipped_reasons["primary_complete"] += 1
                            continue
                    else:
                        # No index metadata but file exists - verify by loading the shard
                        # to avoid redundant re-downloads of already complete files.
                        try:
                            arr = self._load_shard(shard_paths[day_key])
                            if (
                                len(arr) == 1440
                                and len(arr) > 0
                                and int(arr["ts"][0]) == int(day_start)
                                and int(arr["ts"][-1]) == int(day_end)
                            ):
                                # File is complete - update index with metadata and skip
                                crc = int(zlib.crc32(arr.tobytes()) & 0xFFFFFFFF)
                                idx_shards[day_key] = {
                                    "path": shard_paths[day_key],
                                    "min_ts": int(arr["ts"][0]),
                                    "max_ts": int(arr["ts"][-1]),
                                    "count": int(len(arr)),
                                    "crc32": crc,
                                }
                                skipped_reasons["verified_from_disk"] += 1
                                continue
                        except Exception:
                            # Load failed - proceed to re-download
                            pass
                except Exception:
                    # If meta is missing/corrupt, treat as incomplete and allow archive fetch.
                    pass

            days_to_fetch.append((day_key, day_start, day_end))

        # If we verified any files from disk, persist the index updates
        if skipped_reasons["verified_from_disk"] > 0:
            try:
                idx["shards"] = idx_shards
                self._index[f"{symbol}::1m"] = idx
                self._save_index(symbol, tf="1m")
                self._log(
                    "debug",
                    "index_updated_from_disk_verification",
                    symbol=symbol,
                    shards_verified=skipped_reasons["verified_from_disk"],
                )
            except Exception:
                pass

        if not days_to_fetch:
            # Surface why archive prefetch didn't run (useful when large gaps exist but
            # they are not eligible for full-day archive materialization).
            try:
                self._emit_remote_fetch(
                    {
                        "kind": "archive_prefetch",
                        "stage": "skip",
                        "exchange": str(self._ex_id),
                        "symbol": symbol,
                        "reasons": dict(skipped_reasons),
                    }
                )
            except Exception:
                pass
            return

        total_days = len(days_to_fetch)
        completed = 0
        skipped = 0
        start_time = time.monotonic()

        # Log start of archive prefetch
        self._log(
            "info",
            "archive_prefetch_start",
            symbol=symbol,
            days_to_fetch=total_days,
            parallel=parallel_days,
            date_range=f"{days_to_fetch[0][0]}..{days_to_fetch[-1][0]}",
        )
        self._emit_remote_fetch(
            {
                "kind": "archive_prefetch",
                "stage": "start",
                "exchange": str(self._ex_id),
                "symbol": symbol,
                "days_to_fetch": int(total_days),
                "parallel": int(parallel_days),
                "date_range": f"{days_to_fetch[0][0]}..{days_to_fetch[-1][0]}",
            }
        )

        last_progress_emit = 0.0

        # Semaphore to limit concurrent fetches
        sem = asyncio.Semaphore(max(1, parallel_days))

        def _format_archive_exc(exc: BaseException) -> str:
            return bounded_exception_type(exc)

        async def fetch_single_day(
            day_info: Tuple[str, int, int],
        ) -> Tuple[str, Optional[np.ndarray], Optional[str]]:
            """Fetch a single day's archive data with a bounded failure type."""
            day_key, day_start, day_end = day_info
            async with sem:
                try:
                    self._log("debug", "archive_day_attempt", symbol=symbol, day=day_key)
                    arr = await self._archive_fetch_day(symbol, day_key)
                    return (day_key, arr, None)
                except Exception as e:
                    return (day_key, None, _format_archive_exc(e))

        try:
            # Process in batches matching semaphore limit to avoid task queuing
            batch_size = max(1, parallel_days)  # Match semaphore for optimal throughput

            for batch_start in range(0, total_days, batch_size):
                batch = days_to_fetch[batch_start : batch_start + batch_size]
                batch_start_time = time.monotonic()

                # Throttled progress log (every ~10 seconds)
                self._progress_log(
                    (symbol, "1m", "archive"),
                    "archive_prefetch_progress",
                    symbol=symbol,
                    progress=f"{completed}/{total_days}",
                    pct=int(100 * completed / total_days) if total_days > 0 else 0,
                    batch=f"{batch[0][0]}..{batch[-1][0]}",
                    elapsed_s=round(time.monotonic() - start_time, 1),
                )
                try:
                    now = time.monotonic()
                    if (now - last_progress_emit) >= float(
                        self._progress_log_interval_seconds or 0.0
                    ):
                        last_progress_emit = now
                        self._emit_remote_fetch(
                            {
                                "kind": "archive_prefetch",
                                "stage": "progress",
                                "exchange": str(self._ex_id),
                                "symbol": symbol,
                                "completed": int(completed),
                                "total": int(total_days),
                                "pct": int(100 * completed / total_days) if total_days > 0 else 0,
                                "batch": f"{batch[0][0]}..{batch[-1][0]}",
                                "elapsed_s": round(time.monotonic() - start_time, 1),
                            }
                        )
                except Exception:
                    pass

                # Fetch batch in parallel
                tasks = [fetch_single_day(d) for d in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Process results and persist (with deferred index writes)
                batch_had_saves = False
                for i, result in enumerate(results):
                    day_key = batch[i][0]
                    if isinstance(result, Exception):
                        err_type = _format_archive_exc(result)
                        self._log(
                            "warning",
                            "archive_day_failed",
                            symbol=symbol,
                            day=day_key,
                            error_type=err_type,
                        )
                        skipped += 1
                    elif result[2] is not None:
                        err_type = result[2]
                        self._log(
                            "warning",
                            "archive_day_failed",
                            symbol=symbol,
                            day=day_key,
                            error_type=err_type,
                        )
                        skipped += 1
                    elif result[1] is None or result[1].size == 0:
                        self._log("debug", "archive_day_unavailable", symbol=symbol, day=day_key)
                        skipped += 1
                    else:
                        arr = result[1]
                        # Defer index write - we'll flush once at the end of the batch
                        # Skip memory retention to preserve full historical data for backtesting
                        self._persist_batch(
                            symbol,
                            arr,
                            timeframe="1m",
                            merge_cache=True,
                            last_refresh_ms=_utc_now_ms(),
                            defer_index=True,
                            skip_memory_retention=True,
                        )
                        shard_paths[day_key] = self._shard_path(symbol, day_key, tf="1m")
                        self._log(
                            "debug",  # Changed from info to debug to reduce log noise
                            "archive_day_saved",
                            symbol=symbol,
                            day=day_key,
                            rows=int(arr.size),
                        )
                        completed += 1
                        batch_had_saves = True

                batch_elapsed = round(time.monotonic() - batch_start_time, 2)
                if len(batch) > 1:
                    self._log(
                        "debug",
                        "archive_batch_complete",
                        symbol=symbol,
                        batch_size=len(batch),
                        elapsed_s=batch_elapsed,
                    )
        except Exception:
            # Re-raise, but ensure we still log completion below
            raise

        # Flush deferred index writes once after all batches complete
        if completed > 0:
            self.flush_deferred_index(symbol, tf="1m")

        # Log completion summary
        total_elapsed = round(time.monotonic() - start_time, 1)
        self._log(
            "info",
            "archive_prefetch_complete",
            symbol=symbol,
            fetched=completed,
            skipped=skipped,
            total=total_days,
            elapsed_s=total_elapsed,
        )
        self._emit_remote_fetch(
            {
                "kind": "archive_prefetch",
                "stage": "done",
                "exchange": str(self._ex_id),
                "symbol": symbol,
                "fetched": int(completed),
                "skipped": int(skipped),
                "total": int(total_days),
                "elapsed_s": float(total_elapsed),
            }
        )

    async def get_candles_with_resolution_ladder(
        self,
        symbol: str,
        *,
        start_ts: int,
        end_ts: Optional[int] = None,
        strict: bool = False,
    ) -> CandleResolutionResult:
        """Return exact recent 1m candles with a coarser historical prefix.

        Higher-timeframe candles are read through the same manager and may only
        fill complete buckets ending inside the requested old-history prefix.
        Callers remain responsible for requiring an exact recent 1m suffix when
        their decision contract needs one.
        """
        self._raise_if_shutdown_requested("get_candles_with_resolution_ladder")
        now = self._now_ms()
        latest_finalized = _floor_minute(now) - ONE_MIN_MS
        effective_end = (
            latest_finalized
            if end_ts is None
            else min(_floor_minute(int(end_ts)), latest_finalized)
        )
        effective_start = _floor_minute(int(start_ts))
        if effective_end < effective_start:
            return CandleResolutionResult(
                candles=np.empty((0,), dtype=CANDLE_DTYPE),
                source_counts={},
                failures={},
            )

        exchange_timeframes = getattr(self.exchange, "timeframes", None)
        supported_timeframes = (
            set(exchange_timeframes)
            if isinstance(exchange_timeframes, dict) and exchange_timeframes
            else None
        )

        async def fetch_resolution(*, timeframe: str, start_ts: int, end_ts: int):
            return await self.get_candles(
                symbol,
                start_ts=start_ts,
                end_ts=end_ts,
                strict=strict,
                timeframe=None if timeframe == "1m" else timeframe,
            )

        return await fetch_candles_with_resolution_ladder(
            fetch_resolution,
            start_ts=effective_start,
            end_ts=effective_end,
            supported_timeframes=supported_timeframes,
        )

    async def get_candles(
        self,
        symbol: str,
        *,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        max_age_ms: Optional[int] = None,
        strict: bool = False,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
        force_refetch_gaps: bool = False,
        fill_leading_gaps: bool = False,
        fill_trailing_gaps: Optional[bool] = None,
        skip_historical_gap_fill: bool = False,
        max_lookback_candles: Optional[int] = None,
        allow_remote_fetch: bool = True,
        allow_provisional_internal_gaps: bool = False,
    ) -> np.ndarray:
        """Return candles in inclusive range [start_ts, end_ts].

        - If `end_ts` is None: floor(now/1m)*1m + 1m
        - If `start_ts` is None: last `default_window_candles` minutes
        - If `end_ts` provided but `start_ts` is None: end_ts - window
        - If `max_age_ms` == 0: force refresh (no-op when exchange is None)
        - Negative `max_age_ms` raises ValueError
        - Applies gap standardization (1m only)
        - If `force_refetch_gaps` is True: clears known gaps in the requested range
          before fetching, forcing a retry of all gaps regardless of retry count
        - If `fill_leading_gaps` is True: synthesize zero-candles even before the
          first real data point (useful for EMA calculation)
        - If `fill_trailing_gaps` is set: override the default open-tail policy.
          By default, open tails ending at the latest finalized candle are not
          synthesized.
        - If `skip_historical_gap_fill` is True: do not attempt to fetch/fill gaps
          in historical data older than 1 day. Useful for live bot warmup where
          recent data is sufficient and filling old gaps wastes time.
        - If `max_lookback_candles` is set: clamp start_ts so the request spans
          at most that many candles ending at end_ts (per timeframe).
        - If `allow_remote_fetch` is False: serve only local memory/disk data and
          do not call exchange/archive fetchers. This is used for non-critical
          live forager candidates where cache misses should not block trading.
        - If `allow_provisional_internal_gaps` is True: unresolved gaps already
          bounded by later candles may be represented in this returned array by
          non-persistent zero-volume continuity rows only when each gap is no
          wider than `provisional_internal_gap_tolerance_minutes`. Open-ended
          tails retain the separate `fill_trailing_gaps` policy.
        """
        self._raise_if_shutdown_requested("get_candles")
        if max_age_ms is not None and max_age_ms < 0:
            raise ValueError("max_age_ms cannot be negative")

        # Force refetch: clear known gaps in the requested range
        if force_refetch_gaps:
            # Compute actual range first
            now = self._now_ms()
            eff_end = end_ts if end_ts is not None else _floor_minute(now)
            eff_start = (
                start_ts
                if start_ts is not None
                else (int(eff_end) - self.default_window_candles * ONE_MIN_MS)
            )
            cleared = self.clear_known_gaps(symbol, date_range=(eff_start, eff_end))
            if cleared > 0:
                self._log(
                    "info",
                    "force_refetch_gaps",
                    symbol=symbol,
                    start_ts=eff_start,
                    end_ts=eff_end,
                    gaps_cleared=cleared,
                )

        # When a higher timeframe is requested, fetch it directly from the exchange
        # and bypass the 1m cache/standardization logic.
        out_tf = timeframe or tf
        if out_tf is not None:
            # parse timeframe to ms (bucket size)
            period_ms = _tf_to_ms(out_tf)
            if period_ms > ONE_MIN_MS and self.exchange is not None:
                now = self._now_ms()
                finalized_end = (int(now) // period_ms) * period_ms - period_ms
                if end_ts is None:
                    end_ts = finalized_end
                else:
                    end_ts = min((int(end_ts) // period_ms) * period_ms, finalized_end)

                if start_ts is None:
                    # default window expressed in number of requested-tf buckets
                    start_ts = int(end_ts) - self.default_window_candles * period_ms
                start_ts = (int(start_ts) // period_ms) * period_ms

                if max_lookback_candles is not None:
                    try:
                        lookback = max(1, int(max_lookback_candles))
                        lookback_start = int(end_ts) - period_ms * (lookback - 1)
                        if int(start_ts) < int(lookback_start):
                            start_ts = int(lookback_start)
                    except Exception:
                        pass

                if start_ts > end_ts:
                    return np.empty((0,), dtype=CANDLE_DTYPE)

                # Hyperliquid special case: max 5000 candles from current time for any tf
                try:
                    exid = (self._ex_id or "").lower() if isinstance(self._ex_id, str) else ""
                except Exception:
                    exid = ""
                if "hyperliquid" in exid:
                    earliest = int(finalized_end - period_ms * (5000 - 1))
                    if start_ts < earliest:
                        # Mark older part as known gap to avoid repeated fetch attempts
                        gap_end = min(end_ts, earliest - period_ms)
                        if start_ts <= gap_end:
                            self._add_known_gap(symbol, int(start_ts), int(gap_end))
                        start_ts = max(start_ts, earliest)

                # Load from disk shards for this TF (if present) before resorting to network
                try:
                    disk_arr = self._load_from_disk(symbol, start_ts, end_ts, timeframe=out_tf)
                except Exception:
                    disk_arr = None

                # Check in-memory TF range cache first (LRU)
                cache_key = (str(out_tf), int(start_ts), int(end_ts))
                sym_cache = self._tf_range_cache.setdefault(symbol, OrderedDict())
                if cache_key in sym_cache:
                    arr_cached, fetched_at = sym_cache[cache_key]
                    if self._candle_range_has_full_coverage(
                        arr_cached, int(start_ts), int(end_ts), int(period_ms)
                    ):
                        try:
                            sym_cache.move_to_end(cache_key)
                        except Exception:
                            pass
                        if (
                            max_age_ms is None
                            or (
                                max_age_ms > 0
                                and (now - int(fetched_at)) <= int(max_age_ms)
                            )
                        ):
                            return arr_cached
                    else:
                        sym_cache.pop(cache_key, None)

                # If disk has full coverage for this TF window, serve it without network
                if max_age_ms != 0 and isinstance(disk_arr, np.ndarray) and disk_arr.size:
                    out_disk = self._slice_ts_range(disk_arr, start_ts, end_ts)
                    if out_disk.size:
                        # verify full coverage with proper step
                        tsd = _ts_index(out_disk)
                        expected_len = int((end_ts - start_ts) // period_ms) + 1
                        if (
                            out_disk.shape[0] == expected_len
                            and int(tsd[0]) == int(start_ts)
                            and int(tsd[-1]) == int(end_ts)
                            and (
                                expected_len == 1
                                or (
                                    int(np.diff(tsd).min(initial=period_ms)) == period_ms
                                    and int(np.diff(tsd).max(initial=period_ms)) == period_ms
                                )
                            )
                        ):
                            sym_cache[cache_key] = (out_disk, int(now))
                            try:
                                sym_cache.move_to_end(cache_key)
                            except Exception:
                                pass
                            while len(sym_cache) > self._tf_range_cache_cap:
                                sym_cache.popitem(last=False)
                            self._tf_range_cache[symbol] = sym_cache
                            return out_disk

                if not allow_remote_fetch:
                    return (
                        self._slice_ts_range(disk_arr, start_ts, end_ts)
                        if isinstance(disk_arr, np.ndarray) and disk_arr.size
                        else np.empty((0,), dtype=CANDLE_DTYPE)
                    )

                end_excl = int(end_ts) + period_ms

                async with self._acquire_fetch_lock(symbol, out_tf):
                    try:
                        disk_arr = self._load_from_disk(symbol, start_ts, end_ts, timeframe=out_tf)
                    except Exception:
                        disk_arr = None

                    if max_age_ms != 0 and isinstance(disk_arr, np.ndarray) and disk_arr.size:
                        out_disk = self._slice_ts_range(disk_arr, start_ts, end_ts)
                        if out_disk.size:
                            tsd = _ts_index(out_disk)
                            expected_len = int((end_ts - start_ts) // period_ms) + 1
                            if (
                                out_disk.shape[0] == expected_len
                                and int(tsd[0]) == int(start_ts)
                                and int(tsd[-1]) == int(end_ts)
                                and (
                                    expected_len == 1
                                    or (
                                        int(np.diff(tsd).min(initial=period_ms)) == period_ms
                                        and int(np.diff(tsd).max(initial=period_ms)) == period_ms
                                    )
                                )
                            ):
                                sym_cache[cache_key] = (out_disk, int(now))
                                try:
                                    sym_cache.move_to_end(cache_key)
                                except Exception:
                                    pass
                                while len(sym_cache) > self._tf_range_cache_cap:
                                    sym_cache.popitem(last=False)
                                self._tf_range_cache[symbol] = sym_cache
                                return out_disk

                    persisted_batches = False

                    def _persist_tf_batch(batch: np.ndarray) -> None:
                        nonlocal persisted_batches
                        persisted_batches = True
                        self._persist_batch(symbol, batch, timeframe=out_tf)
                        self._invalidate_ema_cache(symbol, timeframe=out_tf)

                    try:
                        fetched = await self._fetch_ohlcv_paginated(
                            symbol,
                            int(start_ts),
                            int(end_excl),
                            timeframe=out_tf,
                            on_batch=_persist_tf_batch,
                            raise_on_partial_empty_page=max_age_ms == 0,
                        )
                    except TypeError:
                        fetched = await self._fetch_ohlcv_paginated(
                            symbol,
                            int(start_ts),
                            int(end_excl),
                            timeframe=out_tf,
                        )
                    # The fetch may have evicted a previously persisted KuCoin
                    # sparse placeholder after a rejected real payload row
                    # proved that bucket unavailable. Reload before merging so
                    # this call cannot reintroduce the stale local copy.
                    try:
                        disk_arr = self._load_from_disk(
                            symbol, start_ts, end_ts, timeframe=out_tf
                        )
                    except Exception:
                        disk_arr = None
                    if fetched.size == 0:
                        if max_age_ms == 0:
                            sym_cache.pop(cache_key, None)
                            self._tf_range_cache[symbol] = sym_cache
                            return fetched
                        if isinstance(disk_arr, np.ndarray) and disk_arr.size:
                            out = self._slice_ts_range(disk_arr, start_ts, end_ts)
                            sym_cache[cache_key] = (out, int(now))
                            try:
                                sym_cache.move_to_end(cache_key)
                            except Exception:
                                pass
                            while len(sym_cache) > self._tf_range_cache_cap:
                                sym_cache.popitem(last=False)
                            self._tf_range_cache[symbol] = sym_cache
                            return out
                        return fetched
                    fetched_out = self._slice_ts_range(fetched, start_ts, end_ts)
                    if fetched_out.size and not persisted_batches:
                        self._persist_batch(symbol, fetched_out, timeframe=out_tf)
                    if isinstance(disk_arr, np.ndarray) and disk_arr.size:
                        disk_out = self._slice_ts_range(disk_arr, start_ts, end_ts)
                        out = self._slice_ts_range(
                            self._merge_overwrite(disk_out, fetched_out),
                            start_ts,
                            end_ts,
                        )
                    else:
                        out = fetched_out
                    self._invalidate_ema_cache(symbol, timeframe=out_tf)
                    if fetched_out.size:
                        self._invalidate_tf_range_cache(
                            symbol,
                            timeframe=out_tf,
                            start_ts=int(_ts_index(fetched_out)[0]),
                            end_ts=int(_ts_index(fetched_out)[-1]),
                        )
                        sym_cache = self._tf_range_cache.setdefault(symbol, OrderedDict())
                    if self._candle_range_has_full_coverage(
                        out, int(start_ts), int(end_ts), int(period_ms)
                    ):
                        sym_cache[cache_key] = (out, int(now))
                        try:
                            sym_cache.move_to_end(cache_key)
                        except Exception:
                            pass
                        while len(sym_cache) > self._tf_range_cache_cap:
                            sym_cache.popitem(last=False)
                    else:
                        sym_cache.pop(cache_key, None)
                    self._tf_range_cache[symbol] = sym_cache
                    return out

        now = self._now_ms()
        if end_ts is None:
            # Use last completed minute as inclusive end (exclude current in-progress minute)
            end_ts = _floor_minute(now) - ONE_MIN_MS
        else:
            # Clamp to last completed minute
            end_ts = min(_floor_minute(int(end_ts)), _floor_minute(now) - ONE_MIN_MS)

        if start_ts is None:
            start_ts = int(end_ts) - ONE_MIN_MS * self.default_window_candles
        else:
            start_ts = _floor_minute(int(start_ts))

        if max_lookback_candles is not None:
            try:
                lookback = max(1, int(max_lookback_candles))
                lookback_start = int(end_ts) - ONE_MIN_MS * (lookback - 1)
                if int(start_ts) < int(lookback_start):
                    start_ts = int(lookback_start)
            except Exception:
                pass

        if start_ts > end_ts:
            return np.empty((0,), dtype=CANDLE_DTYPE)

        # Optionally refresh if range touches the latest finalized minute
        allow_fetch_present = True
        skip_present_fetch_due_to_ttl = False
        skip_initial_refresh_due_to_deferred_prefix = False
        latest_finalized = _floor_minute(now) - ONE_MIN_MS
        if not allow_remote_fetch:
            allow_fetch_present = False
        if allow_fetch_present and end_ts >= latest_finalized and self.exchange is not None:
            last_known_final = 0
            try:
                idx = self._ensure_symbol_index(symbol, tf="1m")
                last_known_final = int(
                    idx.get("meta", {}).get("last_final_ts", 0) or 0
                )
            except Exception:
                last_known_final = 0
            cached_now = self._cache.get(symbol)
            if isinstance(cached_now, np.ndarray) and cached_now.size:
                last_known_final = max(
                    last_known_final,
                    int(np.max(_ensure_dtype(cached_now)["ts"])),
                )
            present_fetch_start = (
                max(int(start_ts), last_known_final + ONE_MIN_MS)
                if last_known_final
                else int(start_ts)
            )
            if present_fetch_start <= int(end_ts):
                adjusted_present_fetch_start = (
                    self._fetch_start_after_deferred_gap_prefix(
                        symbol, present_fetch_start, int(end_ts), now_ms=now
                    )
                )
                if adjusted_present_fetch_start is None:
                    skip_initial_refresh_due_to_deferred_prefix = True
                    self._log(
                        "debug",
                        "known_gap_retry_deferred_present",
                        symbol=symbol,
                        fetch_start=present_fetch_start,
                        end_ts=end_ts,
                    )
                elif adjusted_present_fetch_start > present_fetch_start:
                    skip_initial_refresh_due_to_deferred_prefix = True
                    self._log(
                        "debug",
                        "known_gap_retry_deferred_prefix",
                        symbol=symbol,
                        fetch_start=present_fetch_start,
                        adjusted_fetch_start=adjusted_present_fetch_start,
                        end_ts=end_ts,
                    )
        if allow_fetch_present and end_ts >= latest_finalized and self.exchange is not None:
            if skip_initial_refresh_due_to_deferred_prefix:
                pass
            elif max_age_ms == 0:
                self._log(
                    "debug",
                    "get_candles_force_refresh",
                    symbol=symbol,
                    end_ts=end_ts,
                )
                await self.refresh(
                    symbol=symbol,
                    through_ts=end_ts,
                    raise_on_partial_empty_page=True,
                )
            elif max_age_ms is not None and max_age_ms > 0:
                last_ref = self._get_last_refresh_ms(symbol)
                last_final = 0
                try:
                    idx = self._ensure_symbol_index(symbol, tf="1m")
                    last_final = int(idx.get("meta", {}).get("last_final_ts", 0) or 0)
                except Exception:
                    last_final = 0
                self._log(
                    "debug",
                    "get_candles_check_refresh",
                    symbol=symbol,
                    end_ts=end_ts,
                    last_refresh_ms=last_ref,
                    last_final_ts=last_final,
                    max_age_ms=max_age_ms,
                    now=now,
                )
                need_refresh = last_ref == 0 or (now - last_ref) > int(max_age_ms)
                if not need_refresh:
                    # Only force refresh if cached data lags by MORE than 1 candle
                    # period.  Being exactly 1 minute behind is normal when a new
                    # minute boundary crosses (e.g. right after warmup).  The TTL
                    # alone governs refresh timing in that case — avoiding a
                    # thundering-herd where all symbols refresh simultaneously on
                    # minute transitions.
                    if last_final and (int(end_ts) - int(last_final)) > ONE_MIN_MS:
                        need_refresh = True
                if need_refresh:
                    await self.refresh(symbol, through_ts=end_ts)
                else:
                    allow_fetch_present = False
                    skip_present_fetch_due_to_ttl = True

        # Try to load from disk shards for this range before slicing memory
        try:
            self._load_from_disk(symbol, start_ts, end_ts, timeframe="1m")
        except Exception:  # pragma: no cover - best effort
            pass

        # Get in-memory cached candles for the symbol and slice to requested range
        arr = _ensure_dtype(self._cache.get(symbol, np.empty((0,), dtype=CANDLE_DTYPE)))
        sub = self._slice_ts_range(arr, start_ts, end_ts) if arr.size else arr

        # Determine if the requested historical window is fully covered in memory
        def _is_fully_covered(a: np.ndarray, s_ts: int, e_ts: int) -> bool:
            if a.size == 0:
                return False
            expected_len = int((e_ts - s_ts) // ONE_MIN_MS) + 1
            if a.shape[0] != expected_len:
                return False
            if int(a[0]["ts"]) != s_ts or int(a[-1]["ts"]) != e_ts:
                return False
            if expected_len > 1:
                diffs = np.diff(a["ts"].astype(np.int64))
                if int(diffs.max()) != ONE_MIN_MS or int(diffs.min()) != ONE_MIN_MS:
                    return False
            return True

        fully_covered = _is_fully_covered(sub, start_ts, end_ts)
        if skip_present_fetch_due_to_ttl and not fully_covered:
            # TTL says data is fresh, but coverage is incomplete for requested range.
            # Allow present fetch/gap fill to repair real older gaps, but do not
            # defeat TTL for the normal one-candle tail gap seen at minute boundaries.
            allow_fetch_present = True
            try:
                missing_now = self._missing_spans(sub, start_ts, end_ts)
                trailing_one_candle_gap = (
                    len(missing_now) == 1
                    and int(missing_now[0][0]) == int(end_ts)
                    and int(missing_now[0][1]) == int(end_ts)
                    and end_ts >= latest_finalized
                )
                if trailing_one_candle_gap:
                    allow_fetch_present = False
                    self._log(
                        "debug",
                        "ttl_skip_trailing_present_gap",
                        symbol=symbol,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        max_age_ms=max_age_ms,
                        last_refresh_ms=self._get_last_refresh_ms(symbol),
                    )
                else:
                    self._log(
                        "debug",
                        "ttl_bypass_missing_coverage",
                        symbol=symbol,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        max_age_ms=max_age_ms,
                        last_refresh_ms=self._get_last_refresh_ms(symbol),
                        missing_spans=len(missing_now),
                    )
            except Exception:
                allow_fetch_present = True

        # For historical ranges, if we don't have shards for all days yet, fetch
        # exactly the range and persist shards for future calls.
        end_finalized = latest_finalized

        # Large span prefetch: If the request spans more than 2 days and is not fully
        # covered, trigger archive prefetch for the historical portion even if end_ts
        # touches the present. This fixes warmup requests that span 31 days but were
        # previously skipping archive fetch because end_ts == latest_finalized.
        span_minutes = (end_ts - start_ts) // ONE_MIN_MS
        large_span_threshold = 2 * 24 * 60  # 2 days in minutes
        archive_supported = self._archive_supported()
        if not fully_covered and span_minutes > large_span_threshold:
            self._log(
                "debug",
                "large_span_check",
                symbol=symbol,
                exchange_present=self.exchange is not None,
                span_minutes=int(span_minutes),
                fully_covered=fully_covered,
                archive_supported=archive_supported,
                sub_size=sub.size if hasattr(sub, "size") else 0,
            )
        if (
            allow_remote_fetch
            and
            self.exchange is not None
            and span_minutes > large_span_threshold
            and not fully_covered
            and archive_supported
        ):
            # Prefetch archives for the historical portion (up to 2 days ago, since
            # archives typically lag by 1-2 days). Also respect the user's requested
            # end_ts to avoid fetching beyond the requested date range.
            archive_end_ts = min(end_finalized - 2 * 24 * 60 * ONE_MIN_MS, end_ts)
            if start_ts < archive_end_ts:
                self._log(
                    "info",
                    "large_span_archive_prefetch",
                    symbol=symbol,
                    span_minutes=int(span_minutes),
                    start_ts=start_ts,
                    archive_end_ts=archive_end_ts,
                )
                await self._prefetch_archives_for_range(symbol, start_ts, archive_end_ts)
                # Reload from disk after archive fetch
                try:
                    self._load_from_disk(symbol, start_ts, end_ts, timeframe="1m")
                except Exception:
                    pass
                arr = _ensure_dtype(self._cache.get(symbol, np.empty((0,), dtype=CANDLE_DTYPE)))
                sub = self._slice_ts_range(arr, start_ts, end_ts) if arr.size else arr
                fully_covered = _is_fully_covered(sub, start_ts, end_ts)

        # Treat ranges ending exactly at the latest finalized minute as present-touching
        # for trailing synthesis purposes.  Large warmup windows may still need historical
        # gap fetches, but that must not grant permission to synthesize an unbounded tail.
        historical = end_ts < end_finalized
        fetch_historical_gaps = historical
        if (
            not historical
            and span_minutes > large_span_threshold
            and not fully_covered
            and self.exchange is not None
            and allow_remote_fetch
        ):
            self._log(
                "debug",
                "large_span_needs_gap_fill",
                symbol=symbol,
                span_minutes=int(span_minutes),
                fully_covered=fully_covered,
            )
            fetch_historical_gaps = True
        if (
            allow_remote_fetch
            and self.exchange is not None
            and fetch_historical_gaps
            and not skip_historical_gap_fill
        ):
            # If the requested historical window is not fully covered in memory,
            # attempt to fetch unknown missing spans, regardless of shard presence.
            # Skip this if skip_historical_gap_fill is set (e.g., live warmup where
            # we only need recent data and old gaps don't matter).
            if not fully_covered:
                # Hyperliquid special case: cap lookback to last 5000 minutes
                try:
                    exid = (self._ex_id or "").lower() if isinstance(self._ex_id, str) else ""
                except Exception:
                    exid = ""
                adj_start_ts = start_ts
                if "hyperliquid" in exid:
                    earliest = int(end_finalized - ONE_MIN_MS * (5000 - 1))
                    if adj_start_ts < earliest:
                        gap_end = min(end_ts, earliest - ONE_MIN_MS)
                        if adj_start_ts <= gap_end:
                            self._add_known_gap(symbol, int(adj_start_ts), int(gap_end))
                        adj_start_ts = max(adj_start_ts, earliest)
                if "gateio" in exid:
                    earliest = int(
                        end_finalized - ONE_MIN_MS * (_GATEIO_RECENT_1M_LIMIT_CANDLES - 1)
                    )
                    if adj_start_ts < earliest:
                        gap_end = min(end_ts, earliest - ONE_MIN_MS)
                        if adj_start_ts <= gap_end:
                            self._record_verified_gap(
                                symbol,
                                int(adj_start_ts),
                                int(gap_end),
                                reason=GAP_REASON_NO_ARCHIVE,
                            )
                        if symbol not in self._gateio_recent_window_clip_warned:
                            self._log(
                                "warning",
                                "gateio_ohlcv_recent_window_clipped",
                                symbol=symbol,
                                requested_start_ts=int(start_ts),
                                requested_end_ts=int(end_ts),
                                earliest_fetchable_ts=int(earliest),
                                reason="gateio_public_1m_ohlcv_recent_window",
                            )
                            self._gateio_recent_window_clip_warned.add(symbol)
                        adj_start_ts = max(adj_start_ts, earliest)

                # Skip fetch if all missing spans are already known persistent gaps
                missing_before = self._missing_spans(sub, start_ts, end_ts)

                def span_in_persistent_gap(s: int, e: int) -> bool:
                    """Check if span is fully contained in a persistent (max retries) gap.

                    NOTE: We reload gaps fresh each call to avoid stale closures when
                    _add_known_gap() is called within the same function context.
                    """
                    known_enhanced = self._get_known_gaps_enhanced(symbol)
                    for gap in known_enhanced:
                        if s >= gap["start_ts"] and e <= gap["end_ts"]:
                            # Only consider it "known" if it's persistent (max retries reached)
                            if not self._should_retry_gap(gap):
                                return True
                    return False

                unknown_missing = [
                    (s, e) for (s, e) in missing_before if not span_in_persistent_gap(s, e)
                ]

                if unknown_missing:
                    end_excl = min(end_ts + ONE_MIN_MS, end_finalized + ONE_MIN_MS)
                    if adj_start_ts < end_excl:
                        async with self._acquire_fetch_lock(symbol, "1m"):
                            try:
                                self._load_from_disk(symbol, start_ts, end_ts, timeframe="1m")
                            except Exception:
                                pass
                            arr = _ensure_dtype(
                                self._cache.get(symbol, np.empty((0,), dtype=CANDLE_DTYPE))
                            )
                            sub = self._slice_ts_range(arr, start_ts, end_ts) if arr.size else arr
                            missing_after = self._missing_spans(sub, start_ts, end_ts)
                            unknown_after = [
                                (s, e) for (s, e) in missing_after if not span_in_persistent_gap(s, e)
                            ]
                            if unknown_after:
                                # Only attempt archive prefetch for genuinely missing full days.
                                await self._prefetch_archives_for_range(symbol, adj_start_ts, end_ts)
                                try:
                                    self._load_from_disk(symbol, start_ts, end_ts, timeframe="1m")
                                except Exception:
                                    pass
                                arr = _ensure_dtype(
                                    self._cache.get(symbol, np.empty((0,), dtype=CANDLE_DTYPE))
                                )
                                sub = self._slice_ts_range(arr, start_ts, end_ts) if arr.size else arr
                                missing_after = self._missing_spans(sub, start_ts, end_ts)
                                unknown_after = [
                                    (s, e)
                                    for (s, e) in missing_after
                                    if not span_in_persistent_gap(s, e)
                                ]
                            if unknown_after:
                                persisted_batches = False
                                deferred_index_any = False
                                flush_failed_once = False

                                def _persist_hist_batch(batch: np.ndarray) -> None:
                                    nonlocal persisted_batches, deferred_index_any
                                    persisted_batches = True
                                    deferred_index_any = True
                                    # Skip memory retention to preserve full historical data
                                    self._persist_batch(
                                        symbol,
                                        batch,
                                        timeframe="1m",
                                        merge_cache=True,
                                        last_refresh_ms=now,
                                        defer_index=True,
                                        skip_memory_retention=True,
                                    )

                                def _flush_hist_deferred_index() -> None:
                                    nonlocal deferred_index_any, flush_failed_once
                                    if not deferred_index_any:
                                        return
                                    try:
                                        self.flush_deferred_index(symbol, tf="1m")
                                    except (
                                        Exception
                                    ) as exc:  # best-effort; preserve fetch failure policy
                                        if not flush_failed_once:
                                            self._log(
                                                "warning",
                                                "flush_deferred_index_failed",
                                                symbol=symbol,
                                                timeframe="1m",
                                                error_type=bounded_exception_type(exc),
                                            )
                                            flush_failed_once = True
                                    finally:
                                        deferred_index_any = False

                                self._log(
                                    "debug",
                                    "historical_missing_spans",
                                    symbol=symbol,
                                    spans=len(unknown_after),
                                    first_start_ts=int(unknown_after[0][0]),
                                    last_end_ts=int(unknown_after[-1][1]),
                                )

                                # Coalesce many small missing spans into per-day fetch windows.
                                # This avoids thousands of tiny CCXT requests when gaps are fragmented.
                                spans_to_fetch: List[Tuple[int, int]] = list(unknown_after)
                                try:
                                    day_windows: Dict[str, Tuple[int, int]] = {}
                                    for s0, e0 in spans_to_fetch:
                                        s = int(s0)
                                        e = int(e0)
                                        if e < s:
                                            continue
                                        while s <= e:
                                            dk = self._date_key(s)
                                            ds, de = self._date_range_of_key(dk)
                                            w_start = max(int(ds), int(adj_start_ts))
                                            w_end = min(int(de), int(end_ts))
                                            if w_end >= w_start:
                                                prev = day_windows.get(dk)
                                                if prev is None:
                                                    day_windows[dk] = (w_start, w_end)
                                                else:
                                                    day_windows[dk] = (
                                                        min(int(prev[0]), w_start),
                                                        max(int(prev[1]), w_end),
                                                    )
                                            s = int(de) + ONE_MIN_MS
                                    spans_to_fetch = [
                                        day_windows[k] for k in sorted(day_windows.keys())
                                    ]
                                    if len(spans_to_fetch) != len(unknown_after):
                                        self._log(
                                            "debug",
                                            "historical_missing_spans_coalesced",
                                            symbol=symbol,
                                            spans_before=len(unknown_after),
                                            spans_after=len(spans_to_fetch),
                                        )
                                except Exception:
                                    spans_to_fetch = list(unknown_after)

                                # Fetch only the missing spans (not the whole historical range),
                                # split around any known gap whose retry is still deferred.
                                fetch_windows: List[Tuple[int, int]] = []
                                for s, e in spans_to_fetch:
                                    s2 = max(int(s), int(adj_start_ts))
                                    e2 = int(e)
                                    if e2 < s2:
                                        continue
                                    fetch_windows.extend(
                                        self._fetch_ranges_excluding_deferred_gaps(
                                            symbol,
                                            s2,
                                            e2,
                                            now_ms=now,
                                        )
                                    )
                                for s2, e2 in fetch_windows:
                                    span_end_excl = min(
                                        e2 + ONE_MIN_MS, end_excl
                                    )
                                    if s2 >= span_end_excl:
                                        continue
                                    try:
                                        try:
                                            fetched = await self._fetch_ohlcv_paginated(
                                                symbol,
                                                s2,
                                                span_end_excl,
                                                on_batch=_persist_hist_batch,
                                                raise_on_partial_empty_page=max_age_ms == 0,
                                            )
                                        except TypeError:
                                            fetched = await self._fetch_ohlcv_paginated(
                                                symbol,
                                                s2,
                                                span_end_excl,
                                            )
                                    finally:
                                        _flush_hist_deferred_index()
                                    if fetched.size and not persisted_batches:
                                        # Skip memory retention to preserve full historical data
                                        self._persist_batch(
                                            symbol,
                                            fetched,
                                            timeframe="1m",
                                            merge_cache=True,
                                            last_refresh_ms=now,
                                            defer_index=True,
                                            skip_memory_retention=True,
                                        )
                                        deferred_index_any = True
                                        _flush_hist_deferred_index()
                            arr = (
                                np.sort(self._cache[symbol], order="ts")
                                if symbol in self._cache
                                else np.empty((0,), dtype=CANDLE_DTYPE)
                            )
                            sub = self._slice_ts_range(arr, start_ts, end_ts) if arr.size else arr
                            still_missing = self._missing_spans(sub, start_ts, end_ts)
                            # Re-fetch authoritative lower bound after archive prefetch.
                            authoritative_start_ts = self._get_authoritative_start_ts(symbol)
                            for s, e in still_missing:
                                if not span_in_persistent_gap(s, e):
                                    # Only mark pre_inception when we know a real authoritative
                                    # lower bound for exchange-available history.
                                    if (
                                        authoritative_start_ts is not None
                                        and e < authoritative_start_ts
                                    ):
                                        self._add_known_gap(
                                            symbol,
                                            s,
                                            e,
                                            reason="pre_inception",
                                            retry_count=_GAP_MAX_RETRIES,  # Persistent immediately
                                        )
                                    else:
                                        # Normal gap - will retry and eventually warn
                                        self._add_known_gap(
                                            symbol,
                                            s,
                                            e,
                                            reason=GAP_REASON_FETCH_FAILED,
                                            increment_retry=True,
                                        )
        elif self.exchange is not None and allow_fetch_present:
            # Range touches present (end at or beyond current minute); fetch up to current minute inclusive
            end_current = _floor_minute(now)
            end_excl = min(end_ts + ONE_MIN_MS, end_current + ONE_MIN_MS)
            if start_ts < end_excl:
                need_fetch = False
                fetch_start = start_ts
                if sub.size == 0:
                    need_fetch = True
                else:
                    last_have = int(sub[-1]["ts"]) if sub.size else start_ts - ONE_MIN_MS
                    if last_have < end_excl - ONE_MIN_MS:
                        need_fetch = True
                        fetch_start = max(start_ts, last_have + ONE_MIN_MS)
                self._log(
                    "debug",
                    "get_candles_present_decision",
                    symbol=symbol,
                    need_fetch=need_fetch,
                    fetch_start=fetch_start,
                    last_have=int(sub[-1]["ts"]) if sub.size else None,
                    end_excl=end_excl,
                    sub_size=int(sub.shape[0]) if sub.size else 0,
                )
                if need_fetch:
                    adjusted_fetch_start = (
                        self._fetch_start_after_deferred_gap_prefix(
                            symbol,
                            fetch_start,
                            end_excl - ONE_MIN_MS,
                            now_ms=now,
                        )
                    )
                    if adjusted_fetch_start is None:
                        need_fetch = False
                        self._log(
                            "debug",
                            "known_gap_retry_deferred_present",
                            symbol=symbol,
                            fetch_start=fetch_start,
                            end_ts=end_ts,
                        )
                    else:
                        fetch_start = adjusted_fetch_start
                if need_fetch:
                    async with self._acquire_fetch_lock(symbol, "1m"):
                        try:
                            self._load_from_disk(symbol, start_ts, end_ts, timeframe="1m")
                        except Exception:
                            pass
                        arr = _ensure_dtype(
                            self._cache.get(symbol, np.empty((0,), dtype=CANDLE_DTYPE))
                        )
                        sub = self._slice_ts_range(arr, start_ts, end_ts) if arr.size else arr
                        last_have = int(sub[-1]["ts"]) if sub.size else start_ts - ONE_MIN_MS
                        need_fetch_inner = sub.size == 0 or last_have < end_excl - ONE_MIN_MS
                        fetch_start_inner = (
                            max(start_ts, last_have + ONE_MIN_MS)
                            if sub.size
                            else start_ts
                        )
                        if need_fetch_inner:
                            adjusted_fetch_start_inner = (
                                self._fetch_start_after_deferred_gap_prefix(
                                    symbol,
                                    fetch_start_inner,
                                    end_excl - ONE_MIN_MS,
                                    now_ms=now,
                                )
                            )
                            if adjusted_fetch_start_inner is None:
                                need_fetch_inner = False
                                self._log(
                                    "debug",
                                    "known_gap_retry_deferred_present",
                                    symbol=symbol,
                                    fetch_start=fetch_start_inner,
                                    end_ts=end_ts,
                                )
                            else:
                                fetch_start_inner = adjusted_fetch_start_inner
                        self._log(
                            "debug",
                            "get_candles_present_inner",
                            symbol=symbol,
                            need_fetch=need_fetch_inner,
                            fetch_start=fetch_start_inner,
                            last_have=last_have if sub.size else None,
                            end_excl=end_excl,
                            sub_size=int(sub.shape[0]) if sub.size else 0,
                        )
                        if need_fetch_inner:
                            persisted_batches = False

                            def _persist_present_batch(batch: np.ndarray) -> None:
                                nonlocal persisted_batches
                                batch = self._slice_ts_range(_ensure_dtype(batch), start_ts, end_ts)
                                if not batch.size:
                                    return
                                persisted_batches = True
                                self._persist_batch(
                                    symbol,
                                    batch,
                                    timeframe="1m",
                                    merge_cache=True,
                                    last_refresh_ms=now,
                                )

                            try:
                                fetched = await self._fetch_ohlcv_paginated(
                                    symbol,
                                    fetch_start_inner,
                                    end_excl,
                                    on_batch=_persist_present_batch,
                                    raise_on_partial_empty_page=max_age_ms == 0,
                                )
                            except TypeError:
                                fetched = await self._fetch_ohlcv_paginated(
                                    symbol,
                                    fetch_start_inner,
                                    end_excl,
                                )
                            fetched = self._slice_ts_range(_ensure_dtype(fetched), start_ts, end_ts)
                            if fetched.size and not persisted_batches:
                                self._persist_batch(
                                    symbol,
                                    fetched,
                                    timeframe="1m",
                                    merge_cache=True,
                                    last_refresh_ms=now,
                                )
                        arr = (
                            np.sort(self._cache[symbol], order="ts")
                            if symbol in self._cache
                            else np.empty((0,), dtype=CANDLE_DTYPE)
                        )
                        sub = self._slice_ts_range(arr, start_ts, end_ts) if arr.size else arr

        # Best-effort tail completion (present-only): if we still miss trailing
        # minutes within the requested window, attempt one more fetch from the
        # last available ts. Skip for historical ranges to avoid redundant calls
        # when exchanges have permanent holes.
        if self.exchange is not None and allow_fetch_present and not historical:
            end_current = _floor_minute(now)
            end_excl_range = (
                end_ts + ONE_MIN_MS
                if historical
                else min(end_ts + ONE_MIN_MS, end_current + ONE_MIN_MS)
            )
            for _ in range(2):
                if sub.size == 0:
                    break
                last_have = int(sub[-1]["ts"]) if sub.size else start_ts - ONE_MIN_MS
                if last_have >= end_excl_range - ONE_MIN_MS:
                    break
                fetch_start = last_have + ONE_MIN_MS
                if fetch_start >= end_excl_range:
                    break
                adjusted_fetch_start = self._fetch_start_after_deferred_gap_prefix(
                    symbol,
                    fetch_start,
                    end_excl_range - ONE_MIN_MS,
                    now_ms=now,
                )
                if adjusted_fetch_start is None:
                    self._log(
                        "debug",
                        "known_gap_retry_deferred_tail_completion",
                        symbol=symbol,
                        fetch_start=fetch_start,
                        end_ts=end_ts,
                    )
                    break
                fetch_start = adjusted_fetch_start
                async with self._acquire_fetch_lock(symbol, "1m"):
                    try:
                        self._load_from_disk(symbol, start_ts, end_ts, timeframe="1m")
                    except Exception:
                        pass
                    arr = _ensure_dtype(self._cache.get(symbol, np.empty((0,), dtype=CANDLE_DTYPE)))
                    sub = self._slice_ts_range(arr, start_ts, end_ts) if arr.size else arr
                    if sub.size == 0:
                        break
                    last_have = int(sub[-1]["ts"]) if sub.size else start_ts - ONE_MIN_MS
                    if last_have >= end_excl_range - ONE_MIN_MS:
                        break
                    fetch_start = last_have + ONE_MIN_MS
                    if fetch_start >= end_excl_range:
                        break
                    adjusted_fetch_start = (
                        self._fetch_start_after_deferred_gap_prefix(
                            symbol,
                            fetch_start,
                            end_excl_range - ONE_MIN_MS,
                            now_ms=now,
                        )
                    )
                    if adjusted_fetch_start is None:
                        self._log(
                            "debug",
                            "known_gap_retry_deferred_tail_completion",
                            symbol=symbol,
                            fetch_start=fetch_start,
                            end_ts=end_ts,
                        )
                        break
                    fetch_start = adjusted_fetch_start
                    persisted_batches = False

                    def _persist_tail_batch(batch: np.ndarray) -> None:
                        nonlocal persisted_batches
                        batch = self._slice_ts_range(_ensure_dtype(batch), start_ts, end_ts)
                        if not batch.size:
                            return
                        persisted_batches = True
                        self._persist_batch(
                            symbol,
                            batch,
                            timeframe="1m",
                            merge_cache=True,
                            last_refresh_ms=now,
                        )

                    try:
                        fetched = await self._fetch_ohlcv_paginated(
                            symbol,
                            fetch_start,
                            end_excl_range,
                            on_batch=_persist_tail_batch,
                            raise_on_partial_empty_page=max_age_ms == 0,
                        )
                    except TypeError:
                        fetched = await self._fetch_ohlcv_paginated(
                            symbol,
                            fetch_start,
                            end_excl_range,
                        )
                    fetched = self._slice_ts_range(_ensure_dtype(fetched), start_ts, end_ts)
                    if fetched.size == 0:
                        break
                    if not persisted_batches:
                        self._persist_batch(
                            symbol,
                            fetched,
                            timeframe="1m",
                            merge_cache=True,
                            last_refresh_ms=now,
                        )
                    arr = np.sort(self._cache[symbol], order="ts")
                    sub = self._slice_ts_range(arr, start_ts, end_ts)

        # Gap-oriented fetch and tagging (present-only): try filling internal
        # gaps once; mark remaining as known gaps. Skip for pure historical
        # windows; those are handled above with known-gap marking.
        if self.exchange is not None and allow_fetch_present and not historical:
            end_current = _floor_minute(now)
            inclusive_end = end_ts if historical else min(end_ts, end_current)
            missing = self._missing_spans(sub, start_ts, inclusive_end)
            if missing:
                # Helper to test if a span is fully inside any persistent known gap
                def span_in_persistent_gap_present(s: int, e: int) -> bool:
                    """Check if span is in persistent gap. Reloads gaps to avoid stale data."""
                    known_enhanced_present = self._get_known_gaps_enhanced(symbol)
                    for gap in known_enhanced_present:
                        if s >= gap["start_ts"] and e <= gap["end_ts"]:
                            if not self._should_retry_gap(gap):
                                return True
                    return False

                def span_has_unverified_gap_present(s: int, e: int) -> bool:
                    return any(
                        int(gap["start_ts"]) <= int(e)
                        and int(gap["end_ts"]) >= int(s)
                        and str(gap.get("reason", GAP_REASON_AUTO))
                        in {GAP_REASON_AUTO, GAP_REASON_FETCH_FAILED}
                        for gap in self._get_known_gaps_enhanced(symbol)
                    )

                def unverified_gap_covering(
                    s: int, e: int
                ) -> Optional[GapEntry]:
                    for gap in self._get_known_gaps_enhanced(symbol):
                        if (
                            int(gap["start_ts"]) <= int(s)
                            and int(gap["end_ts"]) >= int(e)
                            and str(gap.get("reason", GAP_REASON_AUTO))
                            in {GAP_REASON_AUTO, GAP_REASON_FETCH_FAILED}
                        ):
                            return gap
                    return None

                def kucoin_verification_bounds(
                    candles: np.ndarray, s: int, e: int
                ) -> Optional[Tuple[int, int]]:
                    """Return real rows bracketing a KuCoin sparse 1m gap.

                    A retry restricted to the absent timestamps cannot prove a
                    no-trade interval: KuCoin simply returns an empty payload.
                    Querying the nearest real row on each side lets one
                    successful raw payload prove the omission without treating
                    an outage or terminal empty page as market data.
                    """
                    if not self._record_payload_gaps_as_known or not (
                        isinstance(self._ex_id, str)
                        and "kucoin" in self._ex_id.lower()
                    ):
                        return None
                    unverified_gap = unverified_gap_covering(s, e)
                    if unverified_gap is None or not self._kucoin_contextual_retry_due(
                        unverified_gap, now_ms=now
                    ):
                        return None
                    real = np.sort(_ensure_dtype(candles), order="ts")
                    provisional = self._synthetic_timestamps.get(symbol, set())
                    if provisional:
                        real = real[
                            ~np.isin(
                                real["ts"].astype(np.int64),
                                np.asarray(tuple(provisional), dtype=np.int64),
                            )
                        ]
                    if real.size < 2:
                        return None
                    timestamps = real["ts"].astype(np.int64)
                    before = timestamps[timestamps < int(s)]
                    after = timestamps[timestamps > int(e)]
                    if before.size == 0 or after.size == 0:
                        return None
                    return int(before[-1]), int(after[0])

                # Attempt limited targeted fetches for unknown spans
                attempts = 0
                max_attempts = 10 if self._ccxt_since_exclusive else 3
                attempted: List[Tuple[int, int]] = []
                for s, e in missing:
                    if attempts >= max_attempts:
                        break
                    contextual_bounds = kucoin_verification_bounds(sub, s, e)
                    if (
                        span_in_persistent_gap_present(s, e)
                        and contextual_bounds is None
                    ):
                        continue
                    if contextual_bounds is None:
                        adjusted_gap_start = (
                            self._fetch_start_after_deferred_gap_prefix(
                                symbol,
                                s,
                                e,
                                now_ms=now,
                            )
                        )
                        if adjusted_gap_start is None:
                            continue
                        fetch_start = int(adjusted_gap_start)
                        end_excl_gap = int(e) + ONE_MIN_MS
                    else:
                        fetch_start = int(contextual_bounds[0])
                        end_excl_gap = int(contextual_bounds[1]) + ONE_MIN_MS
                    async with self._acquire_fetch_lock(symbol, "1m"):
                        try:
                            self._load_from_disk(symbol, start_ts, end_ts, timeframe="1m")
                        except Exception:
                            pass
                        arr = _ensure_dtype(
                            self._cache.get(symbol, np.empty((0,), dtype=CANDLE_DTYPE))
                        )
                        sub = self._slice_ts_range(arr, start_ts, end_ts) if arr.size else arr
                        missing_now = self._missing_spans(sub, start_ts, inclusive_end)
                        if not any(
                            ms <= int(s) <= me
                            for ms, me in missing_now
                        ):
                            continue
                        contextual_bounds = kucoin_verification_bounds(arr, s, e)
                        if contextual_bounds is None:
                            adjusted_gap_start = (
                                self._fetch_start_after_deferred_gap_prefix(
                                    symbol,
                                    s,
                                    e,
                                    now_ms=now,
                                )
                            )
                            if adjusted_gap_start is None:
                                continue
                            fetch_start = int(adjusted_gap_start)
                            end_excl_gap = int(e) + ONE_MIN_MS
                        else:
                            fetch_start = int(contextual_bounds[0])
                            end_excl_gap = int(contextual_bounds[1]) + ONE_MIN_MS
                        contextual_verified = False
                        persisted_batches = False
                        if contextual_bounds is not None:
                            fetched, contextual_verified = (
                                await self._fetch_kucoin_contextual_gap_page(
                                    symbol,
                                    left_boundary_ts=int(contextual_bounds[0]),
                                    right_boundary_ts=int(contextual_bounds[1]),
                                    gap_start_ts=int(s),
                                    gap_end_ts=int(e),
                                )
                            )
                        else:

                            def _persist_gap_batch(batch: np.ndarray) -> None:
                                nonlocal persisted_batches
                                batch = self._slice_ts_range(
                                    _ensure_dtype(batch), start_ts, end_ts
                                )
                                if not batch.size:
                                    return
                                persisted_batches = True
                                self._persist_batch(
                                    symbol,
                                    batch,
                                    timeframe="1m",
                                    merge_cache=True,
                                    last_refresh_ms=now,
                                )

                            try:
                                fetched = await self._fetch_ohlcv_paginated(
                                    symbol,
                                    fetch_start,
                                    end_excl_gap,
                                    on_batch=_persist_gap_batch,
                                    raise_on_partial_empty_page=max_age_ms == 0,
                                )
                            except TypeError:
                                fetched = await self._fetch_ohlcv_paginated(
                                    symbol,
                                    fetch_start,
                                    end_excl_gap,
                                )
                        attempts += 1
                        if contextual_bounds is None:
                            attempted.append((int(s), int(e)))
                        fetched = self._slice_ts_range(
                            _ensure_dtype(fetched), start_ts, end_ts
                        )
                        if fetched.size and not persisted_batches:
                            self._persist_batch(
                                symbol,
                                fetched,
                                timeframe="1m",
                                merge_cache=True,
                                last_refresh_ms=now,
                            )
                        if contextual_bounds is not None:
                            if contextual_verified:
                                self._record_verified_gap(
                                    symbol, int(s), int(e)
                                )
                            else:
                                self._defer_kucoin_contextual_gap_retry(
                                    symbol,
                                    int(s),
                                    int(e),
                                    now_ms=now,
                                )
                        if fetched.size:
                            arr = np.sort(self._cache[symbol], order="ts")
                            sub = self._slice_ts_range(
                                arr,
                                start_ts,
                                end_ts,
                                assume_sorted=True,
                            )
                # After attempts, recompute missing and tag remaining as known gaps
                still_missing = self._missing_spans(sub, start_ts, inclusive_end)
                # Stamp every attempted remainder, including a partially recovered
                # gap, so it is deferred and cannot be synthesized or retried on
                # each caller cycle.
                for s, e in attempted:
                    # find overlapping portion with any still missing
                    for ms, me in still_missing:
                        unresolved_start = max(s, ms)
                        unresolved_end = min(e, me)
                        if (
                            unresolved_start <= unresolved_end
                            and span_has_unverified_gap_present(
                                unresolved_start, unresolved_end
                            )
                        ):
                            self._add_known_gap(
                                symbol,
                                unresolved_start,
                                unresolved_end,
                                reason=GAP_REASON_FETCH_FAILED,
                            )

        # Standardize gaps: synthesize zero-candles where missing.
        # To help seed forward-fill, include one candle before start_ts if available.
        # This ensures standardize_gaps has a prev_close even if sub starts after start_ts.
        data_for_gaps = sub
        if sub.size == 0 or (sub.size > 0 and int(sub[0]["ts"]) > start_ts):
            full_arr = self._cache.get(symbol)
            if full_arr is not None and full_arr.size > 0:
                full_arr = _ensure_dtype(full_arr)
                ts_idx = full_arr["ts"].astype(np.int64)
                idx = int(np.searchsorted(ts_idx, start_ts, side="left"))
                if idx > 0:
                    seed_candle = full_arr[idx - 1 : idx]
                    if sub.size > 0:
                        data_for_gaps = np.concatenate([seed_candle, sub])
                    else:
                        data_for_gaps = seed_candle

        trailing_fill = bool(end_ts < latest_finalized)
        if fill_trailing_gaps is not None:
            trailing_fill = bool(fill_trailing_gaps)

        unverified_gap_ranges = self._unverified_gap_ranges(
            symbol,
            start_ts,
            end_ts,
        )
        if allow_provisional_internal_gaps:
            provisional_tolerance_ms = int(
                self.provisional_internal_gap_tolerance_minutes * ONE_MIN_MS
            )
            excluded_synthetic_ranges = []
            for full_gap_start, full_gap_end in self._unverified_uncovered_gap_ranges(
                symbol,
                start_ts,
                end_ts,
            ):
                overlap_start = max(int(start_ts), full_gap_start)
                overlap_end = min(int(end_ts), full_gap_end)
                if (
                    provisional_tolerance_ms <= 0
                    or full_gap_end - full_gap_start + ONE_MIN_MS
                    > provisional_tolerance_ms
                ):
                    excluded_synthetic_ranges.append(
                        (overlap_start, overlap_end)
                    )
        else:
            excluded_synthetic_ranges = unverified_gap_ranges
        result = self.standardize_gaps(
            data_for_gaps,
            start_ts=start_ts,
            end_ts=end_ts,
            strict=strict,
            fill_leading_gaps=fill_leading_gaps,
            fill_trailing_gaps=trailing_fill,
            assume_sorted=True,
            symbol=symbol,
            excluded_synthetic_ranges=excluded_synthetic_ranges,
        )

        # Log accumulated gap summaries (throttled)
        self._log_persistent_gap_summary()
        self._log_strict_gaps_summary()

        return result

    async def get_latest_completed_close(
        self, symbol: str, max_age_ms: Optional[int] = None
    ) -> float:
        """Return the close of the latest completed 1m candle for `symbol`.

        CandlestickManager is completed-candle-only.  It does not fetch, cache,
        merge, or persist current in-progress candles, and it does not use ticker
        endpoints as live price truth.  Current bid/ask/last belongs in
        MarketSnapshotProvider.
        """
        if max_age_ms is not None and max_age_ms < 0:
            raise ValueError("max_age_ms cannot be negative")
        now = self._now_ms()
        last_final = _floor_minute(now) - ONE_MIN_MS
        if last_final < 0:
            return float("nan")
        got = await self.get_candles(
            symbol,
            start_ts=last_final,
            end_ts=last_final,
            max_age_ms=max_age_ms,
            timeframe=None,
            strict=False,
        )
        if got is None or got.size == 0:
            return float("nan")
        got_sorted = np.sort(_ensure_dtype(got), order="ts")
        if int(got_sorted[-1]["ts"]) > int(last_final):
            raise RuntimeError(
                f"candlestick manager returned in-progress candle for {symbol}: "
                f"ts={int(got_sorted[-1]['ts'])} last_final={int(last_final)}"
            )
        price = float(got_sorted[-1]["c"])
        self._log(
            "debug",
            "get_latest_completed_close",
            symbol=symbol,
            ts=int(got_sorted[-1]["ts"]),
        )
        return price

    async def get_current_close(self, symbol: str, max_age_ms: Optional[int] = None) -> float:
        """Compatibility alias for get_latest_completed_close().

        Live current price truth must use ticker/market snapshots, not
        CandlestickManager.  This method intentionally returns completed-candle
        close only despite its legacy name.
        """
        return await self.get_latest_completed_close(symbol, max_age_ms=max_age_ms)

    def set_current_close(self, symbol: str, price: float, timestamp_ms: int) -> None:
        """Deprecated no-op.

        Current price cache injection belongs to MarketSnapshotProvider.  The
        method is retained to avoid breaking older helpers during the staged
        transition, but CandlestickManager ignores current/in-progress prices.
        """
        self._log(
            "debug",
            "set_current_close_ignored_completed_only",
            symbol=symbol,
            timestamp_ms=int(timestamp_ms),
        )

    def is_rate_limited(self) -> bool:
        """Return True if a global rate-limit backoff is active."""
        return self._rate_limit_until > time.time()

    # ----- EMA helpers -----

    def _ema(self, values: np.ndarray, span: float) -> float:
        """Return the final bias-corrected EMA without allocating a full series."""
        if _RUST_EMA_LAST is not None:
            contiguous = np.ascontiguousarray(values, dtype=np.float64)
            return float(_RUST_EMA_LAST(contiguous, float(span)))
        n = int(values.shape[0])
        if n == 0:
            return float("nan")
        span = float(span)
        alpha = 2.0 / (span + 1.0)
        one_minus = 1.0 - alpha
        first_finite_idx = None
        for i in range(n):
            if np.isfinite(float(values[i])):
                first_finite_idx = i
                break
        if first_finite_idx is None:
            return float("nan")
        num = float(values[first_finite_idx])
        den = 1.0
        for i in range(first_finite_idx + 1, n):
            value = float(values[i])
            if not np.isfinite(value):
                continue
            num = alpha * value + one_minus * num
            den = alpha + one_minus * den
            if den <= np.finfo(np.float64).tiny:
                num = alpha * value
                den = alpha
        return float(num / den)

    def _ema_series(self, values: np.ndarray, span: float) -> np.ndarray:
        """Return bias-corrected EMA (pandas ewm adjust=True) over `values`."""

        n = int(values.shape[0])
        if n == 0:
            return np.empty((0,), dtype=np.float64)
        span = float(span)
        alpha = 2.0 / (span + 1.0)
        one_minus = 1.0 - alpha
        out = np.empty((n,), dtype=np.float64)
        first_finite_idx = None
        for i in range(n):
            if np.isfinite(float(values[i])):
                first_finite_idx = i
                break
        if first_finite_idx is None:
            out.fill(float("nan"))
            return out
        if first_finite_idx > 0:
            out[:first_finite_idx] = float("nan")
        num = float(values[first_finite_idx])
        den = 1.0
        out[first_finite_idx] = num / den
        for i in range(first_finite_idx + 1, n):
            v = float(values[i])
            if not np.isfinite(v):
                out[i] = out[i - 1]
                continue
            num = alpha * v + one_minus * num
            den = alpha + one_minus * den
            if den <= np.finfo(np.float64).tiny:
                num = alpha * v
                den = alpha
            out[i] = num / den
        return out

    @staticmethod
    def _normalize_spans_by_metric(
        spans_by_metric: Dict[str, Any],
    ) -> Dict[str, List[float]]:
        normalized: Dict[str, List[float]] = {}
        for metric_key, raw_spans in (spans_by_metric or {}).items():
            if raw_spans is None:
                spans_iter = []
            elif isinstance(raw_spans, (int, float)):
                spans_iter = [raw_spans]
            else:
                spans_iter = list(raw_spans)
            spans: List[float] = []
            for raw_span in spans_iter:
                span = float(raw_span)
                if not math.isfinite(span) or span <= 0.0:
                    raise ValueError(
                        f"projected open-tail EMA span must be finite and > 0.0; "
                        f"metric={metric_key} span={raw_span!r}"
                    )
                spans.append(span)
            if spans:
                normalized[str(metric_key)] = sorted(set(spans))
        return normalized

    def _ema_metric_series(self, metric_key: str, arr: np.ndarray) -> np.ndarray:
        if metric_key == "close":
            return np.asarray(arr["c"], dtype=np.float64)
        if metric_key == "volume":
            return np.asarray(arr["bv"], dtype=np.float64)
        if metric_key == "qv":
            return (
                np.asarray(arr["bv"], dtype=np.float64)
                * (
                    np.asarray(arr["h"], dtype=np.float64)
                    + np.asarray(arr["l"], dtype=np.float64)
                    + np.asarray(arr["c"], dtype=np.float64)
                )
                / 3.0
            )
        if metric_key == "log_range":
            return np.log(
                np.maximum(np.asarray(arr["h"], dtype=np.float64), 1e-12)
                / np.maximum(np.asarray(arr["l"], dtype=np.float64), 1e-12)
            )
        raise KeyError(f"Unknown EMA metric_key {metric_key!r}")

    def _projection_shared_content_signature(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        timeframe: str,
    ) -> Tuple[Any, ...]:
        """Return content-bearing shared-cache state for a projection window.

        The index is reloaded when another process atomically replaces it.
        Shard checksums and gap coverage affect projection inputs; refresh and
        retry timestamps intentionally do not.
        """
        idx = self._ensure_symbol_index(symbol, timeframe=timeframe)
        shards = idx.get("shards", {})
        shard_state: List[Tuple[Any, ...]] = []
        if isinstance(shards, dict):
            for date_key in self._date_keys_between(start_ts, end_ts):
                shard = shards.get(date_key)
                if not isinstance(shard, dict):
                    continue
                shard_state.append(
                    (
                        str(date_key),
                        shard.get("min_ts"),
                        shard.get("max_ts"),
                        shard.get("count"),
                        shard.get("crc32"),
                    )
                )

        gap_state: List[Tuple[int, int, str]] = []
        known_gaps = idx.get("meta", {}).get("known_gaps", [])
        if isinstance(known_gaps, list):
            for gap in known_gaps:
                try:
                    if isinstance(gap, dict):
                        gap_start = int(gap.get("start_ts", 0))
                        gap_end = int(gap.get("end_ts", 0))
                        reason = str(gap.get("reason", GAP_REASON_AUTO))
                    elif isinstance(gap, (list, tuple)) and len(gap) >= 2:
                        gap_start = int(gap[0])
                        gap_end = int(gap[1])
                        reason = GAP_REASON_AUTO
                    else:
                        continue
                except Exception:
                    continue
                clipped_start = max(int(start_ts), gap_start)
                clipped_end = min(int(end_ts), gap_end)
                if clipped_start <= clipped_end:
                    gap_state.append((clipped_start, clipped_end, reason))

        return (tuple(shard_state), tuple(sorted(gap_state)))

    async def get_projected_open_tail_ema_metrics(
        self,
        symbol: str,
        spans_by_metric: Dict[str, Any],
        *,
        latest_expected_ts: int,
        last_cached_ts: int,
        max_tail_gap_ms: int,
        timeframe: str = "1m",
    ) -> Dict[str, Dict[float, float]]:
        """Return provisional open-tail EMA metrics without mutating candle/EMA state.

        Projection is intentionally stateless. Real candles and existing bounded
        internal gap synthesis are used first; flat zero-volume rows are appended
        only for the still-open tail of this single read.
        """
        period_ms = _tf_to_ms(timeframe)
        if period_ms != ONE_MIN_MS:
            raise ValueError("open-tail EMA projection currently supports only 1m candles")
        latest_expected = _floor_minute(int(latest_expected_ts))
        last_cached = _floor_minute(int(last_cached_ts))
        max_tail_gap = int(max_tail_gap_ms)
        if latest_expected < last_cached:
            raise ValueError(
                f"latest_expected_ts must be >= last_cached_ts for open-tail projection: "
                f"latest_expected_ts={latest_expected} last_cached_ts={last_cached}"
            )
        tail_gap_ms = int(max(0, latest_expected - last_cached))
        if max_tail_gap <= 0 or tail_gap_ms > max_tail_gap:
            raise ValueError(
                f"open-tail projection exceeds max_tail_gap_ms: "
                f"tail_gap_ms={tail_gap_ms} max_tail_gap_ms={max_tail_gap}"
            )

        normalized = self._normalize_spans_by_metric(spans_by_metric)
        if not normalized:
            return {}
        max_span = max(span for spans in normalized.values() for span in spans)
        window_candles = max(1, int(math.ceil(max_span)))
        start_ts = min(
            int(latest_expected - ONE_MIN_MS * (window_candles - 1)),
            last_cached,
        )
        shared_content_signature = self._projection_shared_content_signature(
            symbol,
            start_ts,
            latest_expected,
            timeframe=timeframe,
        )
        cache_key = (
            int(latest_expected),
            int(last_cached),
            int(max_tail_gap),
            shared_content_signature,
            tuple(
                (metric_key, tuple(float(span) for span in spans))
                for metric_key, spans in sorted(normalized.items())
            ),
        )
        projection_cache = self._projected_open_tail_ema_cache.setdefault(
            symbol, OrderedDict()
        )
        cached = projection_cache.get(cache_key)
        if cached is not None:
            projection_cache.move_to_end(cache_key)
            return {
                metric_key: dict(values)
                for metric_key, values in cached.items()
            }
        before_cache_keys = set(self._ema_cache.get(symbol, {}).keys())
        before_synthetic = set(self._synthetic_timestamps.get(symbol, set()))
        arr = await self.get_candles(
            symbol,
            start_ts=start_ts,
            end_ts=latest_expected,
            max_age_ms=None,
            timeframe="1m",
            allow_remote_fetch=False,
            allow_provisional_internal_gaps=True,
        )
        # get_candles may legitimately record bounded internal synthetic gaps.
        # Projection itself must not write EMA cache entries or open-tail synthetic timestamps.
        if symbol in self._ema_cache:
            after_cache_keys = set(self._ema_cache.get(symbol, {}).keys())
            if after_cache_keys != before_cache_keys:
                self._ema_cache[symbol] = {
                    key: self._ema_cache[symbol][key]
                    for key in before_cache_keys
                    if key in self._ema_cache[symbol]
                }
        projected_open_tail: List[int] = []
        if arr.size == 0:
            raise RuntimeError(
                f"open-tail projection unavailable for {symbol}: no local candles "
                f"start_ts={start_ts} latest_expected_ts={latest_expected}"
            )
        arr = np.sort(_ensure_dtype(arr), order="ts")
        newest_ts = int(arr[-1]["ts"])
        if newest_ts > latest_expected:
            arr = self._slice_ts_range(arr, start_ts, latest_expected, assume_sorted=True)
            if arr.size == 0:
                raise RuntimeError(
                    f"open-tail projection unavailable for {symbol}: no candles after range clamp"
                )
            newest_ts = int(arr[-1]["ts"])
        if newest_ts < last_cached:
            raise RuntimeError(
                f"open-tail projection unavailable for {symbol}: "
                f"authoritative tail anchor missing last_cached_ts={last_cached} "
                f"newest_ts={newest_ts}"
            )
        # The deterministic fake-live harness intentionally permits sparse
        # scenario timelines. Those rows are authoritative test inputs rather
        # than incomplete exchange history.
        #
        # Unknown-gap metadata wholly after last_cached describes the exact
        # bounded open tail this method is responsible for projecting. It must
        # not turn a permitted provisional tail into an "internal gap" failure.
        # Gaps at or before the authoritative anchor remain strict, except when
        # every named timestamp is already covered by a real row and the gap
        # metadata is merely stale.
        if str(self.exchange_name or "").lower() != "fake":
            for gap_start, gap_end in self._unverified_gap_ranges(
                symbol,
                start_ts,
                min(last_cached, latest_expected),
            ):
                mask = (arr["ts"] >= int(gap_start)) & (arr["ts"] <= int(gap_end))
                if not self._candle_range_has_full_coverage(
                    arr[mask],
                    int(gap_start),
                    int(gap_end),
                    ONE_MIN_MS,
                ):
                    raise RuntimeError(
                        f"open-tail projection unavailable for {symbol}: "
                        "unverified internal candle gap"
                    )
        if newest_ts < latest_expected:
            prev_close = float(arr[-1]["c"])
            rows = []
            for ts in range(newest_ts + ONE_MIN_MS, latest_expected + ONE_MIN_MS, ONE_MIN_MS):
                rows.append((int(ts), prev_close, prev_close, prev_close, prev_close, 0.0))
                projected_open_tail.append(int(ts))
            if rows:
                arr = np.concatenate([arr, np.array(rows, dtype=CANDLE_DTYPE)])

        if projected_open_tail:
            current_synthetic = set(self._synthetic_timestamps.get(symbol, set()))
            self._synthetic_timestamps[symbol] = current_synthetic - set(projected_open_tail)
            if not self._synthetic_timestamps[symbol]:
                self._synthetic_timestamps.pop(symbol, None)
        elif before_synthetic and symbol not in self._synthetic_timestamps:
            self._synthetic_timestamps[symbol] = before_synthetic

        out: Dict[str, Dict[float, float]] = {}
        for metric_key, spans in normalized.items():
            series = self._ema_metric_series(metric_key, arr)
            metric_out: Dict[float, float] = {}
            for span in spans:
                span_candles = max(1, int(math.ceil(span)))
                tail = series[-span_candles:] if series.shape[0] > span_candles else series
                if tail.shape[0] == 0:
                    metric_out[span] = float("nan")
                else:
                    metric_out[span] = float(self._ema(tail, span))
            out[metric_key] = metric_out
        projection_cache[cache_key] = {
            metric_key: dict(values) for metric_key, values in out.items()
        }
        projection_cache.move_to_end(cache_key)
        while len(projection_cache) > self._projected_open_tail_ema_cache_cap:
            projection_cache.popitem(last=False)
        return out

    async def get_latest_cached_ema_metrics(
        self,
        symbol: str,
        spans_by_metric: Dict[str, Any],
        *,
        max_staleness_ms: Optional[int],
        window_candles: Optional[int] = None,
        timeframe: str = "1m",
    ) -> Dict[str, float]:
        """Return EMA metrics ending at the newest locally cached completed candle.

        This is for non-critical live candidate ranking where callers may carry
        forward qv/log-range EMAs through a bounded unknown stale tail. It never
        fetches remote candles and never appends synthetic tail rows.
        """
        normalized = self._normalize_spans_by_metric(spans_by_metric)
        nested = await self.get_latest_cached_ema_metric_spans(
            symbol,
            normalized,
            max_staleness_ms=max_staleness_ms,
            window_candles=window_candles,
            timeframe=timeframe,
        )
        # Preserve the legacy one-value-per-metric API.  Callers historically
        # supplied one span for each metric; if several are supplied, the
        # largest normalized span retains the prior overwrite behavior.
        out: Dict[str, float] = {}
        for metric_key, spans in normalized.items():
            metric_values = nested.get(metric_key, {})
            for span in spans:
                if span in metric_values:
                    out[str(metric_key)] = float(metric_values[span])
        return out

    async def get_latest_cached_ema_metric_spans(
        self,
        symbol: str,
        spans_by_metric: Dict[str, Any],
        *,
        max_staleness_ms: Optional[int],
        window_candles: Optional[int] = None,
        timeframe: str = "1m",
    ) -> Dict[str, Dict[float, float]]:
        """Return cache-only EMA values for multiple metrics and spans in one load."""
        period_ms = _tf_to_ms(timeframe)
        normalized = self._normalize_spans_by_metric(spans_by_metric)
        if not normalized:
            return {}
        last_cached = int(
            self.get_last_final_ts(symbol, timeframe=timeframe) or 0
        )
        last_cached = (last_cached // period_ms) * period_ms
        if last_cached <= 0:
            return {}
        latest_expected = (int(self._now_ms()) // period_ms) * period_ms - period_ms
        stale_tail_ms = max(0, int(latest_expected) - int(last_cached))
        if (
            max_staleness_ms is not None
            and int(max_staleness_ms) >= 0
            and stale_tail_ms > int(max_staleness_ms)
        ):
            return {}
        max_span = max(span for spans in normalized.values() for span in spans)
        max_candles = max(1, int(math.ceil(max_span)))
        if window_candles is not None:
            try:
                max_candles = max(max_candles, int(window_candles))
            except Exception:
                pass
        start_ts = int(last_cached - period_ms * (max_candles - 1))
        raw = await self.get_candles(
            symbol,
            start_ts=start_ts,
            end_ts=int(last_cached),
            max_age_ms=None,
            strict=False,
            timeframe=timeframe,
            max_lookback_candles=max_candles,
            fill_trailing_gaps=False,
            allow_remote_fetch=False,
            allow_provisional_internal_gaps=False,
        )
        if raw.size == 0:
            return {}

        out: Dict[str, Dict[float, float]] = {
            metric_key: {} for metric_key in normalized
        }
        for metric_key, spans in normalized.items():
            for span in spans:
                span_candles = max(1, int(math.ceil(span)))
                metric_start_ts = int(last_cached - period_ms * (span_candles - 1))
                tail = self._slice_ts_range(
                    raw, metric_start_ts, int(last_cached), assume_sorted=True
                )
                if tail.size == 0 or not candle_range_has_full_coverage(
                    tail,
                    metric_start_ts,
                    int(last_cached),
                    timeframe=timeframe,
                ):
                    continue
                series = self._ema_metric_series(metric_key, tail)
                if series.shape[0] == 0:
                    continue
                val = float(self._ema(series, span))
                if math.isfinite(val):
                    out[str(metric_key)][float(span)] = val
        return out

    async def _latest_finalized_range(
        self, span: float, *, period_ms: int = ONE_MIN_MS
    ) -> Tuple[int, int]:
        span_candles = max(1, int(math.ceil(float(span))))
        now = self._now_ms()
        # Align to timeframe buckets and exclude current in-progress bucket
        end_floor = (int(now) // int(period_ms)) * int(period_ms)
        end_ts = int(end_floor - period_ms)
        start_ts = int(end_ts - period_ms * (span_candles - 1))
        return start_ts, end_ts

    def _ema_window_has_expected_tail(self, arr: np.ndarray, end_ts: int) -> bool:
        if not isinstance(arr, np.ndarray) or arr.size == 0:
            return False
        try:
            timestamps = np.asarray(arr["ts"], dtype=np.int64)
        except (KeyError, TypeError, ValueError):
            return False
        return bool(np.any(timestamps == int(end_ts)))

    def _ema_window_has_required_coverage(
        self,
        arr: np.ndarray,
        start_ts: int,
        end_ts: int,
        *,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
    ) -> bool:
        # WEEX's recent endpoint silently tail-anchors a 1,000-row response.
        # Requiring the whole window prevents a truncated warmup from looking
        # like a valid long-span EMA while preserving the established sparse
        # leading-history contract for other exchanges.
        period_ms = _tf_to_ms(timeframe if timeframe is not None else tf)
        if (
            period_ms == ONE_MIN_MS
            and symbol is not None
            and str(self.exchange_name or "").lower() != "fake"
        ):
            timestamps = np.asarray(arr["ts"], dtype=np.int64)
            for gap_start, gap_end in self._unverified_gap_ranges(
                symbol, start_ts, end_ts
            ):
                mask = (timestamps >= int(gap_start)) & (timestamps <= int(gap_end))
                if not self._candle_range_has_full_coverage(
                    arr[mask],
                    int(gap_start),
                    int(gap_end),
                    ONE_MIN_MS,
                ):
                    return False
        exid = str(self._ex_id or "").lower()
        if "weex" not in exid:
            return self._ema_window_has_expected_tail(arr, end_ts)
        return candle_range_has_full_coverage(
            arr, start_ts, end_ts, timeframe=timeframe, tf=tf
        )

    def _candle_range_has_full_coverage(
        self,
        arr: np.ndarray,
        start_ts: int,
        end_ts: int,
        period_ms: int,
    ) -> bool:
        """Return whether candles cover every requested bucket exactly once."""
        if not isinstance(arr, np.ndarray) or arr.size == 0 or period_ms <= 0:
            return False
        try:
            timestamps = np.unique(np.asarray(arr["ts"], dtype=np.int64))
        except (KeyError, TypeError, ValueError):
            return False
        expected_len = int((int(end_ts) - int(start_ts)) // int(period_ms)) + 1
        if (
            expected_len <= 0
            or timestamps.size != expected_len
            or int(timestamps[0]) != int(start_ts)
            or int(timestamps[-1]) != int(end_ts)
        ):
            return False
        return bool(
            expected_len == 1
            or np.all(np.diff(timestamps) == int(period_ms))
        )

    def candle_range_has_full_coverage(
        self,
        arr: np.ndarray,
        start_ts: int,
        end_ts: int,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
    ) -> bool:
        return candle_range_has_full_coverage(
            arr,
            start_ts,
            end_ts,
            timeframe=timeframe,
            tf=tf,
        )

    def _record_ema_provisional_internal_gap_context(
        self,
        symbol: str,
        metric_key: str,
        span: float,
        period_ms: int,
        start_ts: int,
        end_ts: int,
    ) -> None:
        key = (str(metric_key), float(span), str(period_ms))
        contexts = self._ema_provisional_internal_gap_context.setdefault(symbol, {})
        if period_ms != ONE_MIN_MS:
            contexts.pop(key, None)
            return
        tolerance_ms = int(
            self.provisional_internal_gap_tolerance_minutes * ONE_MIN_MS
        )
        if tolerance_ms <= 0:
            contexts.pop(key, None)
            return
        last_final_ts = int(self.get_last_final_ts(symbol) or 0)
        cached = self._cache.get(symbol)
        if isinstance(cached, np.ndarray) and cached.size > 0:
            last_final_ts = max(
                last_final_ts, int(np.max(np.asarray(cached["ts"], dtype=np.int64)))
            )
        used_ranges: List[Tuple[int, int]] = []
        for full_gap_start, full_gap_end in self._unverified_uncovered_gap_ranges(
            symbol, int(start_ts), int(end_ts)
        ):
            gap_width_ms = int(full_gap_end) - int(full_gap_start) + ONE_MIN_MS
            overlap_start = max(int(start_ts), int(full_gap_start))
            overlap_end = min(int(end_ts), int(full_gap_end))
            if (
                overlap_start <= overlap_end
                and gap_width_ms <= tolerance_ms
                and int(full_gap_end) < last_final_ts
            ):
                used_ranges.append((overlap_start, overlap_end))
        if not used_ranges:
            contexts.pop(key, None)
            if not contexts:
                self._ema_provisional_internal_gap_context.pop(symbol, None)
            return
        gap_candles = sum(
            (gap_end - gap_start) // ONE_MIN_MS + 1
            for gap_start, gap_end in used_ranges
        )
        max_gap_candles = max(
            (gap_end - gap_start) // ONE_MIN_MS + 1
            for gap_start, gap_end in used_ranges
        )
        oldest_gap_start = min(gap_start for gap_start, _gap_end in used_ranges)
        contexts[key] = {
            "gap_count": int(len(used_ranges)),
            "gap_candles": int(gap_candles),
            "max_gap_candles": int(max_gap_candles),
            "oldest_gap_age_ms": max(0, int(self._now_ms()) - oldest_gap_start),
            "window_start_ts": int(start_ts),
            "window_end_ts": int(end_ts),
        }

    def get_ema_provisional_internal_gap_context(
        self,
        symbol: str,
        metric_key: str,
        span: float,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
    ) -> Optional[Dict[str, int]]:
        period_ms = _tf_to_ms(timeframe if timeframe is not None else tf)
        context = self._ema_provisional_internal_gap_context.get(symbol, {}).get(
            (str(metric_key), float(span), str(period_ms))
        )
        return None if context is None else dict(context)

    async def get_latest_ema_close(
        self,
        symbol: str,
        span: float,
        max_age_ms: Optional[int] = None,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
        allow_remote_fetch: bool = True,
        allow_provisional_internal_gaps: bool = True,
    ) -> float:
        """Return latest EMA of close over last `span` finalized candles.

        Supports higher timeframe via `tf`/`timeframe`.
        """
        out_tf = timeframe if timeframe is not None else tf
        period_ms = _tf_to_ms(out_tf)
        start_ts, end_ts = await self._latest_finalized_range(span, period_ms=period_ms)
        # EMA result cache: reuse if end_ts unchanged and within TTL
        now = self._now_ms()
        tf_key = str(period_ms)
        cache_metric_key = (
            "close" if allow_provisional_internal_gaps else "close:strict"
        )
        key = (cache_metric_key, float(span), tf_key)
        cache = self._ema_cache.setdefault(symbol, {})
        if max_age_ms is not None and max_age_ms > 0 and key in cache:
            val, cached_end_ts, computed_at = cache[key]
            if int(cached_end_ts) == int(end_ts) and (now - int(computed_at)) <= int(max_age_ms):
                return float(val)
        arr = await self.get_candles(
            symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            max_age_ms=max_age_ms,
            timeframe=out_tf,
            allow_remote_fetch=allow_remote_fetch,
            fill_trailing_gaps=False,
            allow_provisional_internal_gaps=allow_provisional_internal_gaps,
        )
        if arr.size == 0:
            return float("nan")
        if not self._ema_window_has_required_coverage(
            arr, start_ts, end_ts, symbol=symbol, timeframe=out_tf
        ):
            return float("nan")
        closes = np.asarray(arr["c"], dtype=np.float64)
        res = float(self._ema(closes, span))
        if allow_provisional_internal_gaps:
            self._record_ema_provisional_internal_gap_context(
                symbol,
                "close",
                span,
                period_ms,
                start_ts,
                end_ts,
            )
        # Store in cache
        cache[key] = (res, int(end_ts), int(now))
        return res

    async def get_ema_bounds(
        self,
        symbol: str,
        span_0: float,
        span_1: float,
        max_age_ms: Optional[int] = None,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Return (lower, upper) bounds from EMAs at spans {span_0, span_1, span_2}.

        span_2 = sqrt(span_0 * span_1). Spans are treated as floats (no rounding),
        matching the canonical EMA-alpha formulation `2/(span+1)`.
        Forwards timeframe and TTL to get_latest_ema_close and computes the three EMAs concurrently.
        """
        from math import isfinite

        s2 = (float(span_0) * float(span_1)) ** 0.5
        e0, e1, e2 = await asyncio.gather(
            self.get_latest_ema_close(
                symbol, span_0, max_age_ms=max_age_ms, timeframe=timeframe, tf=tf
            ),
            self.get_latest_ema_close(
                symbol, span_1, max_age_ms=max_age_ms, timeframe=timeframe, tf=tf
            ),
            self.get_latest_ema_close(symbol, s2, max_age_ms=max_age_ms, timeframe=timeframe, tf=tf),
        )
        vals = [e for e in (e0, e1, e2) if isinstance(e, (int, float)) and isfinite(float(e))]
        if not vals:
            nan = float("nan")
            return nan, nan
        return float(min(vals)), float(max(vals))

    async def get_last_prices(self, symbols: List[str], max_age_ms: int = 10_000) -> Dict[str, float]:
        """Return latest completed-candle close per symbol.

        This is a completed-candle helper, not live price truth.  Callers that
        need current bid/ask/last should use MarketSnapshotProvider.
        """
        out: Dict[str, float] = {}
        if not symbols:
            return out

        ordered_symbols = list(dict.fromkeys(symbols))

        async def one(sym: str) -> float:
            try:
                val = await self.get_latest_completed_close(sym, max_age_ms=max_age_ms)
                return float(val) if isinstance(val, (int, float)) else 0.0
            except Exception as exc:
                self._log(
                    "debug",
                    "get_last_prices_completed_close_failed",
                    symbol=sym,
                    error_type=bounded_exception_type(exc),
                )
                return 0.0

        tasks = {s: asyncio.create_task(one(s)) for s in ordered_symbols}
        for s, t in tasks.items():
            out[s] = await t
        return out

    async def get_ema_bounds_many(
        self,
        items: List[Tuple[str, float, float]],
        *,
        max_age_ms: Optional[int] = 60_000,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """Return EMA bounds per symbol for a list of (symbol, span_0, span_1).

        Returns mapping symbol -> (lower, upper), using get_ema_bounds per symbol.
        """
        out: Dict[str, Tuple[float, float]] = {}
        if not items:
            return out

        async def one(sym: str, s0: float, s1: float) -> Tuple[float, float]:
            try:
                lo, hi = await self.get_ema_bounds(
                    sym, s0, s1, max_age_ms=max_age_ms, timeframe=timeframe, tf=tf
                )
                lo = float(lo) if isinstance(lo, (int, float)) else float("nan")
                hi = float(hi) if isinstance(hi, (int, float)) else float("nan")
                if not (np.isfinite(lo) and np.isfinite(hi)):
                    return (0.0, 0.0)
                return (lo, hi)
            except Exception:
                return (0.0, 0.0)

        tasks = {sym: asyncio.create_task(one(sym, s0, s1)) for (sym, s0, s1) in items}
        for sym, t in tasks.items():
            out[sym] = await t
        return out

    async def get_latest_ema_log_range_many(
        self,
        items: List[Tuple[str, float]],
        *,
        max_age_ms: Optional[int] = 600_000,
        timeframe: Optional[str] = None,
        tf: Optional[str] = "1h",
    ) -> Dict[str, float]:
        """Return latest log-range EMA for each (symbol, span) pair.

        Each span is interpreted in candle units of the provided timeframe (`tf` defaults to 1h).
        Returns 0.0 on failures or non-finite results.
        """
        out: Dict[str, float] = {}
        if not items:
            return out

        async def one(sym: str, span: float) -> float:
            try:
                val = await self.get_latest_ema_log_range(
                    sym,
                    span,
                    max_age_ms=max_age_ms,
                    timeframe=timeframe,
                    tf=tf,
                )
                return float(val) if np.isfinite(val) else 0.0
            except Exception:
                return 0.0

        tasks = {sym: asyncio.create_task(one(sym, span)) for (sym, span) in items}
        for sym, t in tasks.items():
            out[sym] = await t
        return out

    async def get_latest_ema_volume(
        self,
        symbol: str,
        span: float,
        max_age_ms: Optional[int] = None,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
        allow_remote_fetch: bool = True,
        allow_provisional_internal_gaps: Optional[bool] = None,
    ) -> float:
        return await self._get_latest_ema_generic(
            symbol,
            span,
            max_age_ms,
            timeframe,
            tf=tf,
            allow_remote_fetch=allow_remote_fetch,
            allow_provisional_internal_gaps=allow_provisional_internal_gaps,
            metric_key="volume",
            series_fn=lambda a: np.asarray(a["bv"], dtype=np.float64),
        )

    async def get_latest_ema_quote_volume(
        self,
        symbol: str,
        span: float,
        max_age_ms: Optional[int] = None,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
        allow_remote_fetch: bool = True,
        allow_provisional_internal_gaps: Optional[bool] = None,
    ) -> float:
        """Return latest EMA of quote volume over last `span` finalized candles.

        Quote volume per candle is approximated as base_volume * typical_price,
        where typical_price = (high + low + close) / 3. This is a common
        approximation when trade-level VWAP is not available.
        """
        return await self._get_latest_ema_generic(
            symbol,
            span,
            max_age_ms,
            timeframe,
            tf=tf,
            allow_remote_fetch=allow_remote_fetch,
            allow_provisional_internal_gaps=allow_provisional_internal_gaps,
            metric_key="qv",
            series_fn=lambda a: (
                np.asarray(a["bv"], dtype=np.float64)
                * (
                    np.asarray(a["h"], dtype=np.float64)
                    + np.asarray(a["l"], dtype=np.float64)
                    + np.asarray(a["c"], dtype=np.float64)
                )
                / 3.0
            ),
        )

    async def _get_latest_ema_generic(
        self,
        symbol: str,
        span: float,
        max_age_ms: Optional[int],
        timeframe: Optional[str],
        *,
        tf: Optional[str] = None,
        allow_remote_fetch: bool = True,
        allow_provisional_internal_gaps: Optional[bool] = None,
        metric_key: str,
        series_fn,
    ) -> float:
        """Shared implementation for EMA helpers over a derived series.

        series_fn: callable taking the candles ndarray and returning a 1-D float64 series.
        metric_key: short key used in EMA cache to distinguish metrics (e.g., 'volume', 'qv').
        """
        out_tf = timeframe if timeframe is not None else tf
        period_ms = _tf_to_ms(out_tf)
        start_ts, end_ts = await self._latest_finalized_range(span, period_ms=period_ms)
        now = self._now_ms()
        tf_key = str(period_ms)
        if allow_provisional_internal_gaps is None:
            allow_provisional_internal_gaps = bool(allow_remote_fetch)
        cache_metric_key = (
            metric_key
            if allow_provisional_internal_gaps
            else f"{metric_key}:strict"
        )
        key = (cache_metric_key, float(span), tf_key)
        cache = self._ema_cache.setdefault(symbol, {})
        if max_age_ms is not None and max_age_ms > 0 and key in cache:
            val, cached_end_ts, computed_at = cache[key]
            if int(cached_end_ts) == int(end_ts) and (now - int(computed_at)) <= int(max_age_ms):
                return float(val)
        arr = await self.get_candles(
            symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            max_age_ms=max_age_ms,
            timeframe=out_tf,
            allow_remote_fetch=allow_remote_fetch,
            fill_trailing_gaps=False,
            allow_provisional_internal_gaps=allow_provisional_internal_gaps,
        )
        if arr.size == 0:
            return float("nan")
        if not self._ema_window_has_required_coverage(
            arr, start_ts, end_ts, symbol=symbol, timeframe=out_tf
        ):
            return float("nan")
        if period_ms > ONE_MIN_MS and not self._candle_range_has_full_coverage(
            arr, start_ts, end_ts, period_ms
        ):
            return float("nan")
        series = series_fn(arr)
        res = float(self._ema(series, span))
        if allow_provisional_internal_gaps:
            if math.isfinite(res):
                self._record_ema_provisional_internal_gap_context(
                    symbol,
                    metric_key,
                    span,
                    period_ms,
                    start_ts,
                    end_ts,
                )
            else:
                self._ema_provisional_internal_gap_context.get(symbol, {}).pop(
                    (str(metric_key), float(span), str(period_ms)), None
                )
        cache[key] = (res, int(end_ts), int(now))
        return res

    async def get_latest_ema_metrics(
        self,
        symbol: str,
        spans_by_metric: Dict[str, float],
        max_age_ms: Optional[int] = None,
        *,
        window_candles: Optional[int] = None,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute multiple latest-EMA metrics with a single candles fetch.

        This is the strict completed-candle path used by forager ranking. It:
        - Uses a single `get_candles()` call for the largest requested span.
        - Requires authoritative or verified-zero coverage across every metric window.
        - Keeps its cache entries separate from provisional active-strategy values.
        """
        out: Dict[str, float] = {}
        if not spans_by_metric:
            return out

        out_tf = timeframe if timeframe is not None else tf
        period_ms = _tf_to_ms(out_tf)
        # Use the largest span to fetch a superset window.
        max_span = max(float(s) for s in spans_by_metric.values())
        max_candles = max(1, int(math.ceil(max_span)))
        start_ts, end_ts = await self._latest_finalized_range(max_span, period_ms=period_ms)
        if window_candles is not None:
            try:
                lookback = max(1, int(window_candles))
                start_ts = int(end_ts - period_ms * (lookback - 1))
            except Exception:
                pass
        now = self._now_ms()
        tf_key = str(period_ms)

        cache = self._ema_cache.setdefault(symbol, {})
        missing: List[str] = []
        for metric_key, span in spans_by_metric.items():
            key = (f"{metric_key}:strict", float(span), tf_key)
            if max_age_ms is not None and max_age_ms > 0 and key in cache:
                val, cached_end_ts, computed_at = cache[key]
                if int(cached_end_ts) == int(end_ts) and (now - int(computed_at)) <= int(max_age_ms):
                    out[str(metric_key)] = float(val)
                    continue
            missing.append(str(metric_key))

        if not missing:
            return out

        # Fetch raw candles for the superset range once.
        # For 1m, we re-apply standardize_gaps per metric window to match per-call behavior.
        raw = await self.get_candles(
            symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            max_age_ms=max_age_ms,
            strict=False,
            timeframe=out_tf,
            max_lookback_candles=window_candles,
            fill_trailing_gaps=False,
            allow_provisional_internal_gaps=False,
        )
        if raw.size == 0:
            for metric_key in missing:
                out[metric_key] = float("nan")
            return out
        if not self._ema_window_has_required_coverage(
            raw, start_ts, end_ts, symbol=symbol, timeframe=out_tf
        ):
            for metric_key in missing:
                out[metric_key] = float("nan")
            return out
        if period_ms > ONE_MIN_MS and not self._candle_range_has_full_coverage(
            raw, start_ts, end_ts, period_ms
        ):
            for metric_key in missing:
                out[metric_key] = float("nan")
            return out

        def series_for(metric_key: str, arr: np.ndarray) -> np.ndarray:
            if metric_key == "volume":
                return np.asarray(arr["bv"], dtype=np.float64)
            if metric_key == "qv":
                return (
                    np.asarray(arr["bv"], dtype=np.float64)
                    * (
                        np.asarray(arr["h"], dtype=np.float64)
                        + np.asarray(arr["l"], dtype=np.float64)
                        + np.asarray(arr["c"], dtype=np.float64)
                    )
                    / 3.0
                )
            if metric_key == "log_range":
                return np.log(
                    np.maximum(np.asarray(arr["h"], dtype=np.float64), 1e-12)
                    / np.maximum(np.asarray(arr["l"], dtype=np.float64), 1e-12)
                )
            if metric_key == "close":
                return np.asarray(arr["c"], dtype=np.float64)
            raise KeyError(f"Unknown EMA metric_key {metric_key!r}")

        for metric_key in missing:
            span = float(spans_by_metric[metric_key])
            span_candles = max(1, int(math.ceil(span)))
            metric_start_ts = int(end_ts - period_ms * (span_candles - 1))
            # Get window ending at end_ts. Prefer slicing by tail length; if data is short, use what we have.
            tail = raw[-span_candles:] if raw.size > span_candles else raw
            if period_ms == ONE_MIN_MS:
                # Re-apply gap standardization on the requested metric window.
                # This matches get_candles(strict=False) behavior for the same [start,end] window.
                # tail is a slice of sorted get_candles output, so assume_sorted=True
                tail = self.standardize_gaps(
                    tail,
                    start_ts=metric_start_ts,
                    end_ts=end_ts,
                    strict=False,
                    fill_trailing_gaps=False,
                    assume_sorted=True,
                    symbol=symbol,
                )
            if tail.size == 0 or not self._ema_window_has_required_coverage(
                tail,
                metric_start_ts,
                end_ts,
                symbol=symbol,
                timeframe=out_tf,
            ):
                out[metric_key] = float("nan")
                continue
            series = series_for(metric_key, tail)
            res = float(self._ema(series, span))
            out[metric_key] = res
            cache[(f"{metric_key}:strict", span, tf_key)] = (
                res,
                int(end_ts),
                int(now),
            )

        return out

    async def get_latest_ema_metric_spans(
        self,
        symbol: str,
        spans_by_metric: Dict[str, Any],
        max_age_ms: Optional[int] = None,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
        allow_remote_fetch: bool = True,
        allow_provisional_internal_gaps: Optional[bool] = None,
    ) -> Dict[str, Dict[float, float]]:
        """Compute multiple metrics and spans from one completed-candle load.

        Results and cache keys intentionally match the individual latest-EMA
        helpers.  This is the orchestration hot path: a symbol may need three
        close spans plus volatility and volume metrics, and loading the same
        candle window once per span is pure duplicate work.
        """
        normalized = self._normalize_spans_by_metric(spans_by_metric)
        if not normalized:
            return {}
        out_tf = timeframe if timeframe is not None else tf
        period_ms = _tf_to_ms(out_tf)
        if allow_provisional_internal_gaps is None:
            allow_provisional_internal_gaps = bool(allow_remote_fetch)
        now = self._now_ms()
        tf_key = str(period_ms)
        cache = self._ema_cache.setdefault(symbol, {})
        out: Dict[str, Dict[float, float]] = {
            metric_key: {} for metric_key in normalized
        }
        missing: Dict[str, List[float]] = {}
        for metric_key, spans in normalized.items():
            cache_metric_key = (
                metric_key
                if allow_provisional_internal_gaps
                else f"{metric_key}:strict"
            )
            for span in spans:
                _start_ts, end_ts = await self._latest_finalized_range(
                    span, period_ms=period_ms
                )
                key = (cache_metric_key, float(span), tf_key)
                if max_age_ms is not None and max_age_ms > 0 and key in cache:
                    val, cached_end_ts, computed_at = cache[key]
                    if (
                        int(cached_end_ts) == int(end_ts)
                        and now - int(computed_at) <= int(max_age_ms)
                    ):
                        out[metric_key][span] = float(val)
                        continue
                missing.setdefault(metric_key, []).append(span)
        if not missing:
            return out

        max_span = max(span for spans in missing.values() for span in spans)
        start_ts, end_ts = await self._latest_finalized_range(
            max_span, period_ms=period_ms
        )
        raw = await self.get_candles(
            symbol,
            start_ts=start_ts,
            end_ts=end_ts,
            max_age_ms=max_age_ms,
            timeframe=out_tf,
            allow_remote_fetch=allow_remote_fetch,
            fill_trailing_gaps=False,
            allow_provisional_internal_gaps=allow_provisional_internal_gaps,
        )
        if raw.size == 0:
            for metric_key, spans in missing.items():
                out[metric_key].update({span: float("nan") for span in spans})
            return out

        for metric_key, spans in missing.items():
            cache_metric_key = (
                metric_key
                if allow_provisional_internal_gaps
                else f"{metric_key}:strict"
            )
            for span in spans:
                span_candles = max(1, int(math.ceil(span)))
                metric_start_ts = int(
                    end_ts - period_ms * (span_candles - 1)
                )
                tail = self._slice_ts_range(
                    raw,
                    metric_start_ts,
                    end_ts,
                    assume_sorted=True,
                )
                if tail.size == 0 or not self._ema_window_has_required_coverage(
                    tail,
                    metric_start_ts,
                    end_ts,
                    symbol=symbol,
                    timeframe=out_tf,
                ):
                    out[metric_key][span] = float("nan")
                    continue
                if period_ms > ONE_MIN_MS and not self._candle_range_has_full_coverage(
                    tail,
                    metric_start_ts,
                    end_ts,
                    period_ms,
                ):
                    out[metric_key][span] = float("nan")
                    continue
                value = float(self._ema(self._ema_metric_series(metric_key, tail), span))
                out[metric_key][span] = value
                if not math.isfinite(value):
                    if allow_provisional_internal_gaps:
                        self._ema_provisional_internal_gap_context.get(symbol, {}).pop(
                            (str(metric_key), float(span), str(period_ms)), None
                        )
                    continue
                if allow_provisional_internal_gaps:
                    self._record_ema_provisional_internal_gap_context(
                        symbol,
                        metric_key,
                        span,
                        period_ms,
                        metric_start_ts,
                        end_ts,
                    )
                cache[(cache_metric_key, float(span), tf_key)] = (
                    value,
                    int(end_ts),
                    int(now),
                )
        return out

    async def get_latest_ema_log_range(
        self,
        symbol: str,
        span: float,
        max_age_ms: Optional[int] = None,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
        allow_remote_fetch: bool = True,
        allow_provisional_internal_gaps: Optional[bool] = None,
    ) -> float:
        return await self._get_latest_ema_generic(
            symbol,
            span,
            max_age_ms,
            timeframe,
            tf=tf,
            allow_remote_fetch=allow_remote_fetch,
            allow_provisional_internal_gaps=allow_provisional_internal_gaps,
            metric_key="log_range",
            series_fn=lambda a: np.log(
                np.maximum(np.asarray(a["h"], dtype=np.float64), 1e-12)
                / np.maximum(np.asarray(a["l"], dtype=np.float64), 1e-12)
            ),
        )

    # ----- EMA series helpers -----

    async def get_ema_close_series(
        self,
        symbol: str,
        span: float,
        max_age_ms: Optional[int] = None,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
    ) -> np.ndarray:
        out_tf = timeframe if timeframe is not None else tf
        period_ms = _tf_to_ms(out_tf)
        start_ts, end_ts = await self._latest_finalized_range(span, period_ms=period_ms)
        arr = await self.get_candles(
            symbol, start_ts=start_ts, end_ts=end_ts, max_age_ms=max_age_ms, timeframe=out_tf
        )
        if arr.size == 0:
            return np.empty((0,), dtype=EMA_SERIES_DTYPE)
        values = np.asarray(arr["c"], dtype=np.float64)
        ema_vals = self._ema_series(values, span)
        n = ema_vals.shape[0]
        out = np.empty((n,), dtype=EMA_SERIES_DTYPE)
        out["ts"] = np.asarray(arr["ts"], dtype=np.int64)
        out["ema"] = ema_vals.astype(np.float32, copy=False)
        return out

    async def get_ema_volume_series(
        self,
        symbol: str,
        span: float,
        max_age_ms: Optional[int] = None,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
    ) -> np.ndarray:
        out_tf = timeframe if timeframe is not None else tf
        period_ms = _tf_to_ms(out_tf)
        start_ts, end_ts = await self._latest_finalized_range(span, period_ms=period_ms)
        arr = await self.get_candles(
            symbol, start_ts=start_ts, end_ts=end_ts, max_age_ms=max_age_ms, timeframe=out_tf
        )
        if arr.size == 0:
            return np.empty((0,), dtype=EMA_SERIES_DTYPE)
        values = np.asarray(arr["bv"], dtype=np.float64)
        ema_vals = self._ema_series(values, span)
        n = ema_vals.shape[0]
        out = np.empty((n,), dtype=EMA_SERIES_DTYPE)
        out["ts"] = np.asarray(arr["ts"], dtype=np.int64)
        out["ema"] = ema_vals.astype(np.float32, copy=False)
        return out

    async def get_ema_log_range_series(
        self,
        symbol: str,
        span: float,
        max_age_ms: Optional[int] = None,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
    ) -> np.ndarray:
        out_tf = timeframe if timeframe is not None else tf
        period_ms = _tf_to_ms(out_tf)
        start_ts, end_ts = await self._latest_finalized_range(span, period_ms=period_ms)
        arr = await self.get_candles(
            symbol, start_ts=start_ts, end_ts=end_ts, max_age_ms=max_age_ms, timeframe=out_tf
        )
        if arr.size == 0:
            return np.empty((0,), dtype=EMA_SERIES_DTYPE)
        highs = np.asarray(arr["h"], dtype=np.float64)
        lows = np.asarray(arr["l"], dtype=np.float64)
        log_ranges = np.log(np.maximum(highs, 1e-12) / np.maximum(lows, 1e-12))
        ema_vals = self._ema_series(log_ranges, span)
        n = ema_vals.shape[0]
        out = np.empty((n,), dtype=EMA_SERIES_DTYPE)
        out["ts"] = np.asarray(arr["ts"], dtype=np.int64)
        out["ema"] = ema_vals.astype(np.float32, copy=False)
        return out

    # ----- Warmup and refresh -----

    async def warmup_since(self, symbols, since_ts: int) -> None:
        """Backfill/warmup for symbols since a timestamp (no-op network in tests)."""
        tasks = [self.refresh(sym, through_ts=None) for sym in symbols]
        # Do sequentially to match test monkeypatch expectations
        for t in tasks:
            await t

    async def refresh(
        self,
        symbol: str,
        through_ts: Optional[int] = None,
        *,
        raise_on_partial_empty_page: bool = False,
        force_overlap: bool = False,
    ) -> None:
        """Fetch new candles and merge into cache.

        - Overlaps by `overlap_candles`
        - Excludes current in-progress minute
        - No-op if `self.exchange` is None
        - ``force_overlap`` performs the overlap fetch even when the cached tail
          is already current; this is used for periodic REST integrity audits.
        """
        if self.exchange is None:
            return None

        now = self._now_ms()
        end_exclusive = _floor_minute(now)
        if through_ts is not None:
            end_exclusive = min(end_exclusive, _floor_minute(int(through_ts)) + ONE_MIN_MS)

        # Refresh only needs to reconcile recent on-disk candles to avoid unnecessary
        # full-history loads/sorts. Historical ranges are handled on-demand via get_candles().
        lookback_candles = max(int(self.default_window_candles), int(self.overlap_candles)) + 10
        disk_since = max(0, int(end_exclusive) - int(lookback_candles) * ONE_MIN_MS)

        try:
            self._load_from_disk(symbol, disk_since, end_exclusive, timeframe="1m")
        except Exception:
            pass

        existing = self._ensure_symbol_cache(symbol)
        existing_last_ts = (
            int(np.asarray(existing["ts"], dtype=np.int64).max()) if existing.size else None
        )
        if existing.size == 0:
            proposed_since = end_exclusive - self.default_window_candles * ONE_MIN_MS
        else:
            last_ts = existing_last_ts if existing_last_ts is not None else 0
            if last_ts >= end_exclusive - ONE_MIN_MS and not force_overlap:
                self._log(
                    "debug",
                    "refresh_skip_fresh",
                    symbol=symbol,
                    end_exclusive=end_exclusive,
                    last_ts=last_ts,
                )
                return None
            proposed_since = max(0, last_ts - self.overlap_candles * ONE_MIN_MS)

        if proposed_since >= end_exclusive:
            self._log(
                "debug",
                "refresh_skip_since",
                symbol=symbol,
                since=proposed_since,
                end_exclusive=end_exclusive,
            )
            return None

        async with self._acquire_fetch_lock(symbol, "1m"):
            # Re-evaluate with lock in case another process already fetched.
            try:
                self._load_from_disk(symbol, disk_since, end_exclusive, timeframe="1m")
            except Exception:
                pass

            existing = self._ensure_symbol_cache(symbol)
            existing_last_ts = (
                int(np.asarray(existing["ts"], dtype=np.int64).max()) if existing.size else None
            )
            if existing.size == 0:
                since = end_exclusive - self.default_window_candles * ONE_MIN_MS
            else:
                last_ts = existing_last_ts if existing_last_ts is not None else 0
                if last_ts >= end_exclusive - ONE_MIN_MS and not force_overlap:
                    self._log(
                        "debug",
                        "refresh_skip_fresh",
                        symbol=symbol,
                        end_exclusive=end_exclusive,
                        last_ts=last_ts,
                    )
                    return None
                since = max(0, last_ts - self.overlap_candles * ONE_MIN_MS)

            if since >= end_exclusive:
                self._log(
                    "debug",
                    "refresh_skip_since",
                    symbol=symbol,
                    since=since,
                    end_exclusive=end_exclusive,
                )
                return None

            persisted_batches = False
            now_fetch = _utc_now_ms()
            self._log(
                "debug",
                "refresh_fetch",
                symbol=symbol,
                since=since,
                end_exclusive=end_exclusive,
                existing_last_ts=existing_last_ts,
                force_overlap=bool(force_overlap),
            )

            def _persist_refresh_batch(batch: np.ndarray) -> None:
                nonlocal persisted_batches
                batch = self._slice_ts_range(_ensure_dtype(batch), since, end_exclusive - ONE_MIN_MS)
                if not batch.size:
                    return
                persisted_batches = True
                self._persist_batch(
                    symbol,
                    batch,
                    timeframe="1m",
                    merge_cache=True,
                    last_refresh_ms=now_fetch,
                )

            fetch_ranges = (
                [(int(since), int(end_exclusive - ONE_MIN_MS))]
                if force_overlap
                else self._fetch_ranges_excluding_deferred_gaps(
                    symbol,
                    since,
                    end_exclusive - ONE_MIN_MS,
                    now_ms=now,
                )
            )
            attempted_unknown_gaps: List[Tuple[int, int]] = []
            for fetch_start, fetch_end in fetch_ranges:
                attempted_unknown_gaps.extend(
                    self._due_unverified_gap_ranges(
                        symbol,
                        int(fetch_start),
                        int(fetch_end),
                        now_ms=now,
                    )
                )
            fetched_parts: List[np.ndarray] = []
            for fetch_start, fetch_end in fetch_ranges:
                fetch_end_exclusive = min(
                    end_exclusive, int(fetch_end) + ONE_MIN_MS
                )
                if int(fetch_start) >= fetch_end_exclusive:
                    continue
                try:
                    fetched = await self._fetch_ohlcv_paginated(
                        symbol,
                        int(fetch_start),
                        fetch_end_exclusive,
                        on_batch=_persist_refresh_batch,
                        raise_on_partial_empty_page=raise_on_partial_empty_page,
                    )
                except TypeError:
                    fetched = await self._fetch_ohlcv_paginated(
                        symbol, int(fetch_start), fetch_end_exclusive
                    )
                fetched = _ensure_dtype(fetched)
                if fetched.size:
                    fetched_parts.append(fetched)
            new_arr = (
                np.concatenate(fetched_parts)
                if fetched_parts
                else np.empty((0,), dtype=CANDLE_DTYPE)
            )
            new_arr = self._slice_ts_range(_ensure_dtype(new_arr), since, end_exclusive - ONE_MIN_MS)
            self._stamp_unresolved_gap_attempts(
                symbol,
                attempted_unknown_gaps,
            )
            if new_arr.size == 0:
                # A missing open-ended tail is not synthesized. Record the successful
                # empty poll to avoid repeated immediate refetches; future real candles
                # will bound the gap and normal runtime standardization can replay it.
                self._set_last_refresh_meta(symbol, now_fetch)
                return None
            if not persisted_batches:
                self._persist_batch(
                    symbol,
                    new_arr,
                    timeframe="1m",
                    merge_cache=True,
                    last_refresh_ms=now_fetch,
                )
            return None

    # ----- Persistence -----

    def _save_shard(
        self,
        symbol: str,
        date_key: str,
        array: np.ndarray,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
        defer_index: bool = False,
    ) -> None:
        """Save shard as .npy and update index.json atomically.

        Parameters
        ----------
        symbol : str
            Trading symbol.
        date_key : str
            YYYY-MM-DD string used as shard filename.
        array : np.ndarray
            Structured array of dtype CANDLE_DTYPE to write.
        defer_index : bool
            If True, skip writing index.json (caller must call flush_deferred_index later).
        """
        arr = _ensure_dtype(array)
        if arr.size == 0:
            return

        arr = np.sort(arr, order="ts")
        data_bytes = arr.tobytes()
        crc = int(zlib.crc32(data_bytes) & 0xFFFFFFFF)

        tf_norm = self._normalize_timeframe_arg(timeframe, tf)

        # If legacy already has a continuous 1m day shard, skip writing this primary shard.
        # Primary should only fill legacy gaps.
        if tf_norm == "1m":
            try:
                if self._legacy_day_is_complete(symbol, tf_norm, date_key):
                    return
            except (
                Exception
            ) as exc:  # best-effort; legacy cache may be unreadable, fall back to primary write
                self._log(
                    "warning",
                    "legacy_day_quality_check_failed",
                    symbol=symbol,
                    timeframe=tf_norm,
                    day=date_key,
                    error_type=bounded_exception_type(exc),
                )
        shard_path = self._shard_path(symbol, date_key, tf=tf_norm)
        os.makedirs(os.path.dirname(shard_path), exist_ok=True)
        # Write .npy content atomically
        # Use numpy.save to ensure .npy format, writing to a temp path then replace
        tmp_path = f"{shard_path}.tmp"
        with open(tmp_path, "wb") as f:
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, shard_path)

        # Update shard paths cache directly instead of invalidating (avoids re-scan)
        cache_key = (symbol, tf_norm)
        if cache_key in self._shard_paths_cache:
            self._shard_paths_cache[cache_key][date_key] = shard_path

        # Update in-memory index
        idx = self._ensure_symbol_index(symbol, tf=tf_norm)
        shards = idx.setdefault("shards", {})
        shards[date_key] = {
            "path": shard_path,
            "min_ts": int(arr[0]["ts"]),
            "max_ts": int(arr[-1]["ts"]),
            "count": int(arr.shape[0]),
            "crc32": crc,
        }
        meta = idx.setdefault("meta", {})
        meta["last_final_ts"] = max(
            int(meta.get("last_final_ts", 0) or 0),
            int(arr[-1]["ts"]),
        )
        observed_start = meta.get("observed_start_ts", meta.get("inception_ts"))
        if observed_start is None or int(arr[0]["ts"]) < int(observed_start):
            meta["observed_start_ts"] = int(arr[0]["ts"])
            meta["inception_ts"] = int(arr[0]["ts"])
        key = f"{symbol}::{tf_norm}"
        self._index[key] = idx

        # Write index to disk unless deferred
        if not defer_index:
            self._save_index(symbol, tf=tf_norm)
            # Enforce disk retention per timeframe after writing this shard
            try:
                self._enforce_disk_retention(symbol, tf=tf_norm)
            except Exception:
                pass

    def flush_deferred_index(
        self,
        symbol: str,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
    ) -> None:
        """Flush any deferred index updates for a symbol to disk."""
        tf_norm = self._normalize_timeframe_arg(timeframe, tf)
        self._save_index(symbol, tf=tf_norm)
        try:
            self._enforce_disk_retention(symbol, tf=tf_norm)
        except Exception:
            pass

    # ----- Context manager and shutdown -----

    async def aclose(self) -> None:
        """Async close: flush and close resources including HTTP session."""
        await self._close_http_session()

    def close(self) -> None:
        """Sync close: attempt to close HTTP session if event loop is running."""
        # Try to close HTTP session synchronously if possible
        if self._http_session is not None and not self._http_session.closed:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Schedule cleanup but don't wait
                    asyncio.create_task(self._close_http_session())
                else:
                    loop.run_until_complete(self._close_http_session())
            except Exception:
                pass  # Best effort cleanup

    def __enter__(self):  # pragma: no cover - not exercised by tests
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - not exercised by tests
        self.close()
        return False

    async def __aenter__(self):  # pragma: no cover
        return self

    async def __aexit__(self, exc_type, exc, tb):  # pragma: no cover
        await self.aclose()
        return False


__all__ = [
    "CandlestickManager",
    "CANDLE_DTYPE",
    "ONE_MIN_MS",
    "_floor_minute",
]
