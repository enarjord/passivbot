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
import inspect
import json
import logging
import math
import os
import sys

import time
import zlib
import atexit
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple, TypedDict, TYPE_CHECKING
import threading
from collections import OrderedDict

if TYPE_CHECKING:
    import aiohttp

import numpy as np
import portalocker  # type: ignore


# ----- Constants and dtypes -----

ONE_MIN_MS = 60_000

_LOCK_TIMEOUT_SECONDS = 10.0
_LOCK_STALE_SECONDS = 180.0
_LOCK_BACKOFF_INITIAL = 0.1
_LOCK_BACKOFF_MAX = 2.0

# See: https://github.com/enarjord/passivbot/issues/547
# True if running on Windows (used for file/path compatible names)
windows_compatibility = sys.platform.startswith("win") or os.environ.get("WINDOWS_COMPATIBILITY") == "1"

@dataclass
class _LockRecord:
    lock: portalocker.Lock
    count: int
    acquired_at: float
    path: str


class GapEntry(TypedDict, total=False):
    """Enhanced gap metadata stored in index.json known_gaps."""

    start_ts: int  # Gap start timestamp (ms)
    end_ts: int  # Gap end timestamp (ms)
    retry_count: int  # Number of fetch attempts (max 3 before marking persistent)
    reason: str  # "auto_detected", "exchange_downtime", "no_archive", "fetch_failed", "manual"
    added_at: int  # Timestamp when gap was first detected (ms)


# Maximum fetch attempts before marking gap as persistent
_GAP_MAX_RETRIES = 3

# Valid gap reasons
GAP_REASON_AUTO = "auto_detected"
GAP_REASON_EXCHANGE_DOWNTIME = "exchange_downtime"
GAP_REASON_NO_ARCHIVE = "no_archive"
GAP_REASON_FETCH_FAILED = "fetch_failed"
GAP_REASON_MANUAL = "manual"


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

EMA_SERIES_DTYPE = np.dtype(
    [
        ("ts", "int64"),
        ("ema", "float32"),
    ]
)


# ----- Utilities -----


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
        lock_timeout_seconds: float | None = None,
    ) -> None:
        self.exchange = exchange
        # If no explicit exchange_name provided, infer from ccxt instance id
        if (not exchange_name or exchange_name == "unknown") and getattr(exchange, "id", None):
            self.exchange_name = str(getattr(exchange, "id"))
        else:
            self.exchange_name = exchange_name
        self.cache_dir = cache_dir
        self.default_window_candles = int(default_window_candles)
        self.overlap_candles = int(overlap_candles)
        self.max_memory_candles_per_symbol = int(max_memory_candles_per_symbol)
        self.max_disk_candles_per_symbol_per_tf = int(max_disk_candles_per_symbol_per_tf)
        # Debug levels: 0=warnings, 1=network-only, 2=full debug, 3=trace
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
        self._warning_last_log: Dict[str, float] = {}  # throttle repeated warnings
        self._warning_throttle_seconds: float = 300.0  # 5 minutes between repeated warnings
        self._remote_fetch_callback = remote_fetch_callback
        # Cache of legacy shard paths per (exchange, symbol, tf)
        self._legacy_shard_paths_cache: Dict[Tuple[str, str, str], Dict[str, str]] = {}
        # Cache for legacy day quality decisions: (symbol, tf, date_key) -> legacy_is_complete
        self._legacy_day_quality_cache: Dict[Tuple[str, str, str], bool] = {}
        # Cache of primary shard paths per (symbol, tf) - avoids redundant glob scans
        self._shard_paths_cache: Dict[Tuple[str, str], Dict[str, str]] = {}

        self._cache: Dict[str, np.ndarray] = {}
        self._index: Dict[str, dict] = {}
        self._index_mtime: Dict[str, Optional[float]] = {}
        # Cache for EMA computations: per symbol -> {(metric, span, tf): (value, end_ts, computed_at_ms)}
        self._ema_cache: Dict[str, Dict[Tuple[str, int, str], Tuple[float, int, int]]] = {}
        # Cache for current (in-progress) minute close per symbol: symbol -> (price, updated_ms)
        self._current_close_cache: Dict[str, Tuple[float, int]] = {}
        # Cache for fetched higher-timeframe windows to avoid duplicate remote calls (LRU per symbol)
        # Keyed per symbol -> OrderedDict[(tf_str, start_ts, end_ts) -> (array, fetched_at_ms)]
        self._tf_range_cache: Dict[str, OrderedDict[Tuple[str, int, int], Tuple[np.ndarray, int]]] = (
            {}
        )
        self._tf_range_cache_cap = 8
        self._step_warning_keys: set[Tuple[str, str, str]] = set()
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
        # Reentrant bookkeeping for portalocker fetch locks: key -> _LockRecord
        self._held_fetch_locks: Dict[Tuple[str, str], _LockRecord] = {}
        self._shutdown_guard = threading.Lock()
        self._closed = False
        atexit.register(self._cleanup_on_exit)

        self._setup_logging()
        self._cleanup_stale_locks()

        # Initialize optional global semaphore for remote calls
        try:
            mcr = None if max_concurrent_requests in (None, 0) else int(max_concurrent_requests)
            self._net_sem = asyncio.Semaphore(mcr) if (mcr and mcr > 0) else None
        except Exception:
            self._net_sem = None

        # Persistent HTTP session for archive fetches (created lazily)
        self._http_session: Optional["aiohttp.ClientSession"] = None
        self._http_session_lock = asyncio.Lock()

        # fetch controls
        # Base timeframe for storage/fetching is always 1m; higher TFs are per-call
        self._ccxt_timeframe = "1m"
        # Determine exchange id and adjust defaults per exchange quirks
        self._ex_id = getattr(self.exchange, "id", self.exchange_name) or self.exchange_name
        self._ccxt_limit_default = 1000
        if isinstance(self._ex_id, str) and "bitget" in self._ex_id.lower():
            # Bitget often serves 1m klines with 200 limit per page
            self._ccxt_limit_default = 200

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
                self.log.warning("failed to stat lock %s during cleanup: %s", lock_path, exc)
                continue
            age = now - stat.st_mtime
            if age > threshold:
                try:
                    lock_path.unlink()
                    self.log.warning("removed stale candle lock %s (age %.1fs)", lock_path, age)
                except FileNotFoundError:
                    continue
                except Exception as exc:
                    self.log.error("failed to remove stale lock %s: %s", lock_path, exc)

    def _cleanup_on_exit(self) -> None:
        with self._shutdown_guard:
            if self._closed:
                return
            self._closed = True
        records = list(self._held_fetch_locks.values())
        self._held_fetch_locks.clear()
        for record in records:
            self._release_lock_sync(record)

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
                error=str(exc),
            )
        except Exception as exc:
            self._log(
                "warning",
                "fetch_lock_release_error",
                symbol=symbol,
                timeframe=timeframe,
                error=str(exc),
            )
        finally:
            self._remove_lockfile(path)

    def _touch_lockfile(self, path: str) -> None:
        try:
            os.utime(path, None)
        except FileNotFoundError:
            return
        except Exception:
            return

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
        base = [f"event={event}"]
        # In debug modes, include caller info for traceability
        if self.debug_level >= 1:
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
            else:
                parts.append(f"{k}={v}")
        msg = " ".join(base + parts)
        if level == "debug":
            # Apply debug filtering: level 0 -> drop; level 1 -> only ccxt_* events; level 2 -> all
            if self.debug_level <= 0:
                return
            is_network = isinstance(event, str) and (
                event.startswith("ccxt_") or event.startswith("archive_")
            )
            if self.debug_level == 1 and not is_network:
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

    def _log_persistent_gap_summary(self) -> None:
        """Log accumulated persistent gap summary if any, throttled to once per 60s."""
        if not hasattr(self, "_persistent_gap_summary") or not self._persistent_gap_summary:
            return
        now = time.monotonic()
        last = getattr(self, "_persistent_gap_summary_last_log", 0.0)
        if (now - last) < 60.0:  # Only log summary once per minute
            return
        self._persistent_gap_summary_last_log = now
        summary = self._persistent_gap_summary
        total = sum(summary.values())
        symbols = ", ".join(f"{s}:{c}" for s, c in sorted(summary.items())[:5])
        if len(summary) > 5:
            symbols += f", +{len(summary) - 5} more"
        self.log.warning(
            f"persistent gaps: {total} new ({symbols}). Use --force-refetch-gaps to retry."
        )
        self._persistent_gap_summary.clear()

    def _throttled_warning(self, throttle_key: str, event: str, **fields) -> None:
        """Emit a warning at most once per throttle window (default 5 min).

        Use this for warnings that may repeat frequently but only need to
        inform the user once. After the throttle window expires, the warning
        will be emitted again if the condition persists.
        """
        now = time.monotonic()
        last = self._warning_last_log.get(throttle_key, 0.0)
        if (now - last) < self._warning_throttle_seconds:
            return
        self._warning_last_log[throttle_key] = now
        self._log("warning", event, **fields)

    def _emit_remote_fetch(self, payload: Dict[str, Any]) -> None:
        cb = getattr(self, "_remote_fetch_callback", None)
        if cb is None:
            return
        try:
            cb(payload)
        except Exception:
            # Must never break the fetch path due to logging/progress UI.
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
            meta = idx.setdefault("meta", {})
            try:
                last_ts = 0
                inception_ts: Optional[int] = None
                for shard_meta in shards.values():
                    if not isinstance(shard_meta, dict):
                        continue
                    mt = shard_meta.get("max_ts")
                    if mt is not None:
                        last_ts = max(last_ts, int(mt))
                    mi = shard_meta.get("min_ts")
                    if mi is not None:
                        inception_ts = int(mi) if inception_ts is None else min(inception_ts, int(mi))
                meta["last_final_ts"] = int(last_ts)
                meta["inception_ts"] = inception_ts
            except Exception:
                meta["last_final_ts"] = 0
                meta["inception_ts"] = None
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
            current_mtime = os.path.getmtime(idx_path)
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
                        error=str(e),
                    )
            if not isinstance(idx, dict):
                idx = {"shards": {}, "meta": {}}
            idx.setdefault("shards", {})
            meta = idx.setdefault("meta", {})
            meta.setdefault("known_gaps", [])  # list of [start_ts, end_ts]
            meta.setdefault("last_refresh_ms", 0)
            meta.setdefault("last_final_ts", 0)
            meta.setdefault("inception_ts", None)  # first known candle timestamp

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
        meta.setdefault("known_gaps", [])
        meta.setdefault("last_refresh_ms", 0)
        meta.setdefault("last_final_ts", 0)
        meta.setdefault("inception_ts", None)  # first known candle timestamp

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
            self._index_mtime[key] = os.path.getmtime(idx_path)
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
        held = self._held_fetch_locks.get(key)
        if held is not None:
            self._held_fetch_locks[key] = _LockRecord(
                lock=held.lock,
                path=held.path,
                count=held.count + 1,
                acquired_at=held.acquired_at,
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
                if record is None:
                    return
                if record.count <= 1:
                    self._held_fetch_locks.pop(key, None)
                    await self._release_lock(record.lock, record.path, symbol, tf_norm)
                else:
                    self._held_fetch_locks[key] = _LockRecord(
                        lock=record.lock,
                        path=record.path,
                        count=record.count - 1,
                        acquired_at=record.acquired_at,
                    )
            return

        backoff = self._lock_backoff_initial
        deadline = time.monotonic() + self._lock_timeout_seconds
        attempt = 0

        while True:
            attempt += 1
            lock_obj = portalocker.Lock(lock_path, timeout=0, fail_when_locked=True)
            try:
                await asyncio.to_thread(lock_obj.acquire)
                acquired_at = time.time()
                self._touch_lockfile(lock_path)
                self._held_fetch_locks[key] = _LockRecord(
                    lock=lock_obj,
                    path=lock_path,
                    count=1,
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
                    if record is not None:
                        await self._release_lock(record.lock, record.path, symbol, tf_norm)
                return
            except portalocker.exceptions.LockException as exc:
                age = self._lockfile_age(lock_path)
                if age is not None and age > self._lock_stale_seconds:
                    self._log(
                        "warning",
                        "fetch_lock_stale",
                        symbol=symbol,
                        timeframe=tf_norm,
                        age=f"{age:.2f}",
                        lock_path=lock_path,
                    )
                    try:
                        os.remove(lock_path)
                    except FileNotFoundError:
                        pass
                    except Exception as rm_exc:
                        self._log(
                            "error",
                            "fetch_lock_stale_remove_failed",
                            symbol=symbol,
                            timeframe=tf_norm,
                            error=str(rm_exc),
                        )
                    continue

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
                    error=str(exc),
                )
                await asyncio.sleep(backoff)
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

    def _legacy_shard_candidates(self, symbol: str, date_key: str, tf: str) -> List[str]:
        if tf != "1m":
            return []
        ex = str(self.exchange_name or "").lower()
        coin = self._legacy_coin_from_symbol(symbol)
        sym_code = self._legacy_symbol_code_from_symbol(symbol)
        out: List[str] = []

        if coin:
            out.append(os.path.join("historical_data", f"ohlcvs_{ex}", coin, f"{date_key}.npy"))
        if ex == "binanceusdm" and sym_code:
            out.append(
                os.path.join("historical_data", "ohlcvs_futures", sym_code, f"{date_key}.npy")
            )
        if ex == "bybit" and sym_code:
            out.append(os.path.join("historical_data", "ohlcvs_bybit", sym_code, f"{date_key}.npy"))
        return out

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
                "info",
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
            self.log.warning(f"Failed loading shard {path}: {e}")
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
    ) -> Optional[np.ndarray]:
        """Load any shards intersecting [start_ts, end_ts] and merge into cache.

        Primary cache: `{cache_dir}/ohlcv/{exchange}/{tf}/{symbol}/YYYY-MM-DD.npy`
        Legacy fallback (read-only): `historical_data/` downloader caches.
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
                        error=str(exc),
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
            if tf_norm == "1m":
                existing = self._ensure_symbol_cache(symbol)
                merged = self._merge_overwrite(existing, merged_disk)
                self._cache[symbol] = merged
                return merged
            else:
                # Do not touch 1m cache for higher TF; let caller handle
                return merged_disk
        except Exception as e:  # pragma: no cover - noncritical
            self._log("warning", "disk_load_error", symbol=symbol, timeframe=tf_norm, error=str(e))
            return None

    def _save_range(
        self,
        symbol: str,
        arr: np.ndarray,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
    ) -> None:
        """Persist fetched candles to daily shards by date_key."""
        if arr.size == 0:
            return
        arr = np.sort(_ensure_dtype(arr), order="ts")
        tf_norm = self._normalize_timeframe_arg(timeframe, tf)
        current_key: Optional[str] = None
        bucket = []
        total = 0
        for row in arr:
            key = self._date_key(int(row["ts"]))
            if current_key is None:
                current_key = key
            if key != current_key:
                if bucket:
                    self._save_shard(
                        symbol,
                        current_key,
                        np.array(bucket, dtype=CANDLE_DTYPE),
                        tf=tf_norm,
                    )
                    total += len(bucket)
                bucket = []
                current_key = key
            bucket.append(tuple(row.tolist()))
        if bucket and current_key is not None:
            self._save_shard(symbol, current_key, np.array(bucket, dtype=CANDLE_DTYPE), tf=tf_norm)
            total += len(bucket)
            self._log("debug", "saved_range", symbol=symbol, rows=total)

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
            is_last = (i == len(keys_to_process) - 1)
            flush_bucket(key, bucket_data, is_last=is_last)

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
    ) -> None:
        """Merge `batch` into memory (optional) and persist incrementally to disk.

        Args:
            defer_index: If True, defer index.json write until flush_deferred_index is called.
            skip_memory_retention: If True, skip memory retention enforcement to preserve
                full historical data in cache (useful for backtest data preparation).
        """
        if batch.size == 0:
            return
        arr = np.sort(_ensure_dtype(batch), order="ts")
        tf_norm = self._normalize_timeframe_arg(timeframe, tf)

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

        self._save_range_incremental(symbol, arr, timeframe=tf_norm, defer_index=defer_index)

    def _merge_overwrite(self, existing: np.ndarray, new: np.ndarray) -> np.ndarray:
        """Merge two candle arrays by ts, preferring values from `new` on conflict."""
        if existing.size == 0:
            return np.sort(_ensure_dtype(new), order="ts")
        if new.size == 0:
            return np.sort(_ensure_dtype(existing), order="ts")
        a = _ensure_dtype(existing)
        b = _ensure_dtype(new)
        # Put existing first, then new; then keep last seen per ts to prefer new
        combo = np.concatenate([a, b])
        # Stable sort ensures that for equal timestamps, rows from `new` remain after `existing`.
        combo = np.sort(combo, order="ts", kind="stable")
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

    # ----- Known gap helpers -----

    def _get_known_gaps_enhanced(self, symbol: str) -> List[GapEntry]:
        """Return known gaps as enhanced GapEntry objects with full metadata."""
        idx = self._ensure_symbol_index(symbol)
        gaps = idx.get("meta", {}).get("known_gaps", [])
        out: List[GapEntry] = []
        now_ms = int(time.time() * 1000)
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
                    }
                    if entry["start_ts"] <= entry["end_ts"]:
                        out.append(entry)
                elif isinstance(it, (list, tuple)) and len(it) >= 2:
                    # Legacy format: auto-upgrade to enhanced
                    a, b = int(it[0]), int(it[1])
                    if a <= b:
                        out.append({
                            "start_ts": a,
                            "end_ts": b,
                            "retry_count": _GAP_MAX_RETRIES,  # Assume old gaps are persistent
                            "reason": GAP_REASON_AUTO,
                            "added_at": now_ms,
                        })
            except Exception:
                continue
        return out

    def _get_known_gaps(self, symbol: str) -> List[Tuple[int, int]]:
        """Return known gaps as simple (start_ts, end_ts) tuples for backward compatibility."""
        enhanced = self._get_known_gaps_enhanced(symbol)
        return [(g["start_ts"], g["end_ts"]) for g in enhanced]

    def _save_known_gaps_enhanced(self, symbol: str, gaps: List[GapEntry]) -> None:
        """Save gaps in enhanced format, merging overlapping ranges."""
        # Sort by start_ts
        gaps = sorted(gaps, key=lambda g: g["start_ts"])
        merged: List[GapEntry] = []
        for gap in gaps:
            if not merged or gap["start_ts"] > merged[-1]["end_ts"] + ONE_MIN_MS:
                merged.append(gap)
            else:
                # Merge overlapping gaps, keeping max retry count and earliest added_at
                prev = merged[-1]
                merged[-1] = {
                    "start_ts": prev["start_ts"],
                    "end_ts": max(prev["end_ts"], gap["end_ts"]),
                    "retry_count": max(prev.get("retry_count", 0), gap.get("retry_count", 0)),
                    "reason": prev.get("reason", GAP_REASON_AUTO),  # Keep original reason
                    "added_at": min(prev.get("added_at", 0), gap.get("added_at", 0)),
                }
        idx = self._ensure_symbol_index(symbol)
        idx["meta"]["known_gaps"] = [
            {
                "start_ts": int(g["start_ts"]),
                "end_ts": int(g["end_ts"]),
                "retry_count": int(g.get("retry_count", 0)),
                "reason": str(g.get("reason", GAP_REASON_AUTO)),
                "added_at": int(g.get("added_at", 0)),
            }
            for g in merged
        ]
        self._index[symbol] = idx
        self._save_index(symbol)

    def _save_known_gaps(self, symbol: str, gaps: List[Tuple[int, int]]) -> None:
        """Save gaps from simple tuples (backward compatibility wrapper)."""
        now_ms = int(time.time() * 1000)
        enhanced = [
            {
                "start_ts": int(s),
                "end_ts": int(e),
                "retry_count": _GAP_MAX_RETRIES,  # Assume caller-provided gaps are persistent
                "reason": GAP_REASON_AUTO,
                "added_at": now_ms,
            }
            for s, e in gaps
        ]
        self._save_known_gaps_enhanced(symbol, enhanced)

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

        If a gap overlapping with [start_ts, end_ts] already exists:
        - Extends the gap to cover the full range
        - Increments retry_count if increment_retry is True (unless retry_count is specified)
        - Updates reason if provided

        If retry_count reaches _GAP_MAX_RETRIES, the gap is considered persistent
        and will not be re-fetched unless force_refetch_gaps is used.

        Args:
            retry_count: If specified, set retry_count directly instead of incrementing.
                         Useful for pre-inception gaps that should be immediately persistent.
        """
        now_ms = int(time.time() * 1000)
        gaps = self._get_known_gaps_enhanced(symbol)

        # Check if we have an overlapping gap to update
        updated = False
        previous_retry_count = 0
        for gap in gaps:
            if gap["start_ts"] <= end_ts + ONE_MIN_MS and gap["end_ts"] >= start_ts - ONE_MIN_MS:
                # Overlapping - extend and optionally increment retry
                gap["start_ts"] = min(gap["start_ts"], int(start_ts))
                gap["end_ts"] = max(gap["end_ts"], int(end_ts))
                previous_retry_count = gap.get("retry_count", 0)
                if retry_count is not None:
                    gap["retry_count"] = retry_count
                elif increment_retry:
                    # Cap retry_count at _GAP_MAX_RETRIES to prevent unbounded growth
                    # and avoid redundant disk writes for persistent gaps
                    new_retry_count = previous_retry_count + 1
                    gap["retry_count"] = min(new_retry_count, _GAP_MAX_RETRIES)
                if reason != GAP_REASON_AUTO:
                    gap["reason"] = reason
                updated = True
                break

        if not updated:
            initial_retry = retry_count if retry_count is not None else (1 if increment_retry else 0)
            new_gap: GapEntry = {
                "start_ts": int(start_ts),
                "end_ts": int(end_ts),
                "retry_count": initial_retry,
                "reason": reason,
                "added_at": now_ms,
            }
            gaps.append(new_gap)
            self._log(
                "debug",
                "gap_added",
                symbol=symbol,
                start_ts=start_ts,
                end_ts=end_ts,
                reason=reason,
                retry_count=new_gap["retry_count"],
            )
        else:
            # Log warning only when gap transitions from retryable to persistent
            # (retry_count goes from <max to >=max). Use throttling to prevent spam
            # in edge cases where the same gap is processed multiple times.
            updated_gap = next(
                (g for g in gaps if g["start_ts"] <= end_ts + ONE_MIN_MS and g["end_ts"] >= start_ts - ONE_MIN_MS),
                None
            )
            if updated_gap:
                current_retry_count = updated_gap.get("retry_count", 0)
                gap_reason = updated_gap.get("reason", GAP_REASON_AUTO)
                # Only warn on transition to persistent status (skip pre_inception - expected behavior)
                if (
                    current_retry_count >= _GAP_MAX_RETRIES
                    and previous_retry_count < _GAP_MAX_RETRIES
                    and gap_reason != "pre_inception"
                ):
                    gap_minutes = (updated_gap["end_ts"] - updated_gap["start_ts"]) // ONE_MIN_MS + 1
                    # Track persistent gaps for summary logging
                    if not hasattr(self, "_persistent_gap_summary"):
                        self._persistent_gap_summary: Dict[str, int] = {}
                    self._persistent_gap_summary[symbol] = self._persistent_gap_summary.get(symbol, 0) + 1

        self._save_known_gaps_enhanced(symbol, gaps)

    def _should_retry_gap(self, gap: GapEntry) -> bool:
        """Check if a gap should be retried (retry_count < max)."""
        return gap.get("retry_count", 0) < _GAP_MAX_RETRIES

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

        total_minutes = sum(
            (g["end_ts"] - g["start_ts"]) // ONE_MIN_MS + 1
            for g in gaps
        )
        persistent = sum(1 for g in gaps if g.get("retry_count", 0) >= _GAP_MAX_RETRIES)
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
                    "persistent": g.get("retry_count", 0) >= _GAP_MAX_RETRIES,
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

    # ----- Refresh metadata helpers -----

    def _get_last_refresh_ms(self, symbol: str) -> int:
        idx = self._ensure_symbol_index(symbol)
        try:
            return int(idx.get("meta", {}).get("last_refresh_ms", 0))
        except Exception:
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

    # ----- Inception tracking -----

    def _get_inception_ts(self, symbol: str) -> Optional[int]:
        """Return the known inception timestamp (first candle) for this symbol, or None."""
        idx = self._ensure_symbol_index(symbol)
        try:
            val = idx.get("meta", {}).get("inception_ts")
            return int(val) if val is not None else None
        except Exception:
            return None

    def _set_inception_ts(self, symbol: str, ts: int, *, save: bool = True) -> None:
        """Set the inception timestamp for this symbol (only if earlier than current or unset)."""
        idx = self._ensure_symbol_index(symbol)
        meta = idx.setdefault("meta", {})
        current = meta.get("inception_ts")
        # Only update if unset or if new ts is earlier
        if current is None or int(ts) < int(current):
            meta["inception_ts"] = int(ts)
            self._index[f"{symbol}::1m"] = idx
            if save:
                self._save_index(symbol)
            # If we previously marked ranges as "pre_inception" (persistent), but we now
            # discovered earlier real data, that metadata becomes stale and can block
            # future repair/refetch. Trim/remove it best-effort.
            try:
                self._prune_pre_inception_gaps(symbol, int(ts), save=save)
            except Exception as exc:
                self._log(
                    "warning",
                    "prune_pre_inception_gaps_failed",
                    symbol=symbol,
                    error=str(exc),
                )
            self._log(
                "debug",
                "inception_ts_updated",
                symbol=symbol,
                old_ts=current,
                new_ts=int(ts),
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
        if self.exchange is None:
            return []
        # Determine method to call (exchange instance or module)
        ex = self.exchange
        if not hasattr(ex, "fetch_ohlcv"):
            return []

        exid = (self._ex_id or "").lower() if isinstance(self._ex_id, str) else ""
        is_bybit = "bybit" in exid
        max_attempts = 9 if is_bybit else 5
        backoff = 1.0 if is_bybit else 0.5
        backoff_cap = 20.0 if is_bybit else 8.0
        for attempt in range(max_attempts):
            try:
                params: Dict[str, Any] = {}
                # Provide an end bound for exchanges that support it.
                # Note: Avoid passing 'until' to Bitget due to API validation errors on non-1m tfs.
                if end_exclusive_ms is not None:
                    exid = (self._ex_id or "").lower() if isinstance(self._ex_id, str) else ""
                    # Avoid 'until' for exchanges where it yields tail-anchored or inconsistent pages
                    # leading to incomplete forward pagination on first run.
                    if (
                        "bitget" not in exid
                        and "okx" not in exid
                        and "bybit" not in exid
                        and "kucoin" not in exid
                    ):
                        params["until"] = int(end_exclusive_ms) - 1

                # Bybit v5 requires a category for some market data routes. CCXT usually infers
                # this from the market, but being explicit avoids intermittent misclassification.
                if "bybit" in exid:
                    params.setdefault("category", "linear")

                tf_norm = self._normalize_timeframe_arg(timeframe, tf, default=self._ccxt_timeframe)
                t0 = time.monotonic()
                self._emit_remote_fetch(
                    {
                        "kind": "ccxt_fetch_ohlcv",
                        "stage": "start",
                        "exchange": str(self._ex_id),
                        "symbol": symbol,
                        "tf": tf_norm,
                        "since_ts": int(since_ms),
                        "limit": int(limit),
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
                    limit=limit,
                    attempt=attempt + 1,
                    params=params,
                )
                if getattr(self, "_net_sem", None) is not None:
                    async with self._net_sem:  # type: ignore[attr-defined]
                        res = await ex.fetch_ohlcv(
                            symbol,
                            timeframe=tf_norm,
                            since=since_ms,
                            limit=limit,
                            params=params,
                        )
                else:
                    res = await ex.fetch_ohlcv(
                        symbol,
                        timeframe=tf_norm,
                        since=since_ms,
                        limit=limit,
                        params=params,
                    )
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
                )
                return res or []
            except Exception as e:  # pragma: no cover - network not used in tests
                err_type = type(e).__name__
                err_repr = repr(e)
                elapsed_ms = int((time.monotonic() - t0) * 1000) if "t0" in locals() else None
                self._emit_remote_fetch(
                    {
                        "kind": "ccxt_fetch_ohlcv",
                        "stage": "error",
                        "exchange": str(self._ex_id),
                        "symbol": symbol,
                        "tf": str(tf) if tf is not None else None,
                        "since_ts": int(since_ms),
                        "attempt": int(attempt + 1),
                        "elapsed_ms": elapsed_ms,
                        "params": dict(params) if "params" in locals() else None,
                        "error_type": err_type,
                        "error": str(e),
                        "error_repr": err_repr,
                    }
                )
                self._log(
                    "warning",
                    "ccxt_fetch_ohlcv_failed",
                    symbol=symbol,
                    tf=str(tf) if tf is not None else None,
                    attempt=attempt + 1,
                    params=params if "params" in locals() else None,
                    error_type=err_type,
                    error=str(e),
                    error_repr=err_repr,
                )
                sleep_s = backoff
                msg = (str(e) or "")
                msg_l = msg.lower()
                # Heuristic: slow down harder on rate-limit style responses.
                if any(x in msg_l for x in ("rate limit", "too many", "429", "10006")):
                    sleep_s = max(sleep_s, 5.0)
                # Bybit: be more persistent on transient network-ish errors.
                if is_bybit and (
                    err_type in {"RequestTimeout", "NetworkError", "ExchangeNotAvailable", "DDoSProtection"}
                    or any(x in msg_l for x in ("timed out", "timeout", "etimedout", "econnreset", "502", "503", "504"))
                ):
                    sleep_s = max(sleep_s, 2.0)
                await asyncio.sleep(sleep_s)
                backoff = min(backoff * 2.0, backoff_cap)
        return []

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

    async def _fetch_ohlcv_paginated(
        self,
        symbol: str,
        since_ms: int,
        end_exclusive_ms: int,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
        on_batch: Optional[Callable[[np.ndarray], None]] = None,
    ) -> np.ndarray:
        """Fetch OHLCV from `since_ms` up to but excluding `end_exclusive_ms`.

        Uses ccxt pagination via since+limit. Returns CANDLE_DTYPE array.
        """
        if self.exchange is None:
            return np.empty((0,), dtype=CANDLE_DTYPE)
        since_start = int(since_ms)
        since = int(since_ms)
        end_excl = int(end_exclusive_ms)
        limit = self._ccxt_limit_default
        tf_norm = self._normalize_timeframe_arg(timeframe, tf, default=self._ccxt_timeframe)
        # Derive pagination step from timeframe
        period_ms = _tf_to_ms(tf_norm)
        all_rows = []
        pages = 0
        total_span = max(1, end_excl - since_start)
        while since < end_excl:
            page = await self._ccxt_fetch_ohlcv_once(
                symbol, since, limit, end_exclusive_ms=end_excl, tf=tf_norm
            )
            if not page:
                break
            arr = self._normalize_ccxt_ohlcv(page)
            if arr.size == 0:
                break
            # Exclude any candles >= end_exclusive
            arr = arr[arr["ts"] < end_excl]
            if arr.size == 0:
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
                    if max_step != period_ms or min_step != period_ms:
                        warn_key = (self._ex_id, symbol, tf_norm)
                        if warn_key not in self._step_warning_keys:
                            self._step_warning_keys.add(warn_key)
                            self.log.warning(
                                f"unexpected step for tf exchange={self._ex_id} symbol={symbol} tf={tf_norm} expected={period_ms} min_step={min_step} max_step={max_step}"
                            )
                else:
                    max_step = ONE_MIN_MS
            except Exception:
                first_ts = last_ts = 0
            all_rows.append(arr)
            pages += 1
            if on_batch is not None:
                try:
                    on_batch(arr)
                except Exception as on_batch_err:
                    self.log.error(
                        "on_batch callback failed; stopping pagination",
                        extra={
                            "symbol": symbol,
                            "timeframe": tf_norm,
                            "error": str(on_batch_err),
                        },
                    )
                    break
            last_ts = int(arr[-1]["ts"])  # inclusive last
            # Throttled progress logs (INFO) for long-running paginated fetches
            try:
                progressed = max(0, min(100.0, 100.0 * float(last_ts - since_start) / float(total_span)))
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
            # Safety to avoid infinite loops if exchange returns overlapping data
            if new_since <= since:
                self.log.debug(
                    f"pagination stop (no progress) exchange={self._ex_id} symbol={symbol} since={since} last_ts={last_ts}"
                )
                break
            since = new_since
        self.log.debug(
            f"paginated fetch done exchange={self._ex_id} symbol={symbol} tf={tf_norm} rows={sum(a.shape[0] for a in all_rows) if all_rows else 0}"
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
        assume_sorted: bool = False,
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
        assume_sorted : bool
            If True, skip sorting (caller guarantees array is already sorted by ts).
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

        expected = np.arange(effective_lo, hi + ONE_MIN_MS, ONE_MIN_MS, dtype=np.int64)
        # Map from ts to row index in a
        pos = {int(t): i for i, t in enumerate(ts_arr)}

        if strict:
            # In strict mode: do not synthesize zero-candles.
            # If there are gaps, log a warning and return whatever real candles exist in range.
            i0 = int(np.searchsorted(ts_arr, effective_lo, side="left"))
            i1 = int(np.searchsorted(ts_arr, hi, side="right"))
            missing_count = 0
            try:
                expected_len = int((hi - effective_lo) // ONE_MIN_MS) + 1
                slice_ts = ts_arr[i0:i1].astype(np.int64, copy=False)
                if slice_ts.size:
                    # Missing at head + tail + internal gaps (but NOT leading gaps if not filling)
                    if fill_leading_gaps:
                        missing_count += int((int(slice_ts[0]) - effective_lo) // ONE_MIN_MS)
                    missing_count += int((hi - int(slice_ts[-1])) // ONE_MIN_MS)
                    if slice_ts.size > 1:
                        diffs = np.diff(slice_ts)
                        gaps = diffs[diffs > ONE_MIN_MS]
                        if gaps.size:
                            missing_count += int(np.sum((gaps // ONE_MIN_MS) - 1))
                    # If duplicates exist, treat them as missing coverage too
                    missing_count += int(max(0, expected_len - int(np.unique(slice_ts).size) - missing_count))
                else:
                    missing_count = expected_len
            except Exception:
                # fallback: keep behavior safe (no warning rather than exploding)
                missing_count = 0
            if missing_count:
                self._throttled_warning(
                    "standardize_gaps_strict_missing",  # throttle key
                    "standardize_gaps_strict_missing",
                    missing=int(missing_count),
                    start_ts=effective_lo,
                    end_ts=hi,
                )
            return a[i0:i1]

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

        for t in expected:
            if t in pos:
                row = a[pos[t]]
                out_rows.append(tuple(row.tolist()))
                prev_close = float(row["c"])  # update seed
            else:
                if prev_close is None:
                    # No previous data to forward-fill from - skip this timestamp
                    continue
                # Synthesize a zero-candle using previous close (internal gaps only)
                out_rows.append((int(t), prev_close, prev_close, prev_close, prev_close, 0.0))

        if not out_rows:
            return np.empty((0,), dtype=CANDLE_DTYPE)
        return np.array(out_rows, dtype=CANDLE_DTYPE)

    # ----- External archives (historical) -----

    def _archive_supported(self) -> bool:
        try:
            exid = (self._ex_id or "").lower() if isinstance(self._ex_id, str) else ""
        except Exception:
            exid = ""
        return exid in {"binanceusdm", "bybit", "bitget", "kucoinfutures"}

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
        if exid not in {"binanceusdm", "bybit", "bitget", "kucoinfutures"}:
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
        self._emit_remote_fetch(
            {
                "kind": "archive_http_get",
                "stage": "start",
                "exchange": str(self._ex_id),
                "url": str(url),
            }
        )
        self._log("debug", "archive_http_get", url=url)

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
                        url=url,
                        elapsed_ms=int((time.monotonic() - t0) * 1000),
                    )
                    return None
                resp.raise_for_status()
                data = await resp.read()
        except Exception as e:
            err_type = type(e).__name__
            err_repr = repr(e)
            self._emit_remote_fetch(
                {
                    "kind": "archive_http_get",
                    "stage": "error",
                    "exchange": str(self._ex_id),
                    "url": str(url),
                    "error_type": err_type,
                    "error": str(e),
                    "error_repr": err_repr,
                    "elapsed_ms": int((time.monotonic() - t0) * 1000),
                }
            )
            self._log(
                "debug",
                "archive_http_error",
                url=url,
                error_type=err_type,
                error=str(e),
                error_repr=err_repr,
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
            url=url,
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
                df.columns = col_names + [f"extra_{i}" for i in range(len(df.columns) - len(col_names))]
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
                df.columns = col_names + [f"extra_{i}" for i in range(len(df.columns) - len(col_names))]
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

    def _ohlcv_df_to_day_arr(self, df, day_key: str) -> np.ndarray:
        """Convert a dataframe with timestamp/open/high/low/close/volume to 1m day array."""
        start_ts, end_ts = self._date_range_of_key(day_key)
        cols = ["timestamp", "open", "high", "low", "close", "volume"]
        for c in cols:
            df[c] = df[c].astype("float64")
        df = df.dropna(subset=["timestamp", "close"]).sort_values("timestamp").drop_duplicates(
            subset=["timestamp"], keep="last"
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
        # For archive day data, we expect full day coverage - use fill_leading_gaps=True
        out = self.standardize_gaps(
            arr, start_ts=start_ts, end_ts=end_ts, strict=False, fill_leading_gaps=True
        )
        # Validate full-day coverage (best-effort; callers can fall back to ccxt).
        if out.size != 1440 or int(out[0]["ts"]) != start_ts or int(out[-1]["ts"]) != end_ts:
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

        # Skip fetches before known inception date
        inception_ts = self._get_inception_ts(symbol)
        if inception_ts is not None and start_ts < inception_ts:
            # Mark pre-inception range as persistent gap (no retries needed)
            pre_inception_end = min(inception_ts - ONE_MIN_MS, end_ts)
            if start_ts <= pre_inception_end:
                self._add_known_gap(
                    symbol,
                    start_ts,
                    pre_inception_end,
                    reason="pre_inception",
                    retry_count=_GAP_MAX_RETRIES,  # Mark as persistent immediately
                )
                self._log(
                    "debug",
                    "skip_pre_inception_fetch",
                    symbol=symbol,
                    original_start=start_ts,
                    inception_ts=inception_ts,
                    clipped_start=inception_ts,
                )
            # Clip to inception date
            start_ts = inception_ts
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
                except Exception:
                    # If meta is missing/corrupt, treat as incomplete and allow archive fetch.
                    pass

            days_to_fetch.append((day_key, day_start, day_end))

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

        def _format_archive_exc(exc: BaseException) -> Tuple[str, str]:
            """Return (error_type, error_repr) for logging."""
            try:
                return (type(exc).__name__, repr(exc))
            except Exception:
                return (type(exc).__name__, "<unrepresentable exception>")

        async def fetch_single_day(
            day_info: Tuple[str, int, int]
        ) -> Tuple[str, Optional[np.ndarray], Optional[Tuple[str, str]]]:
            """Fetch a single day's archive data. Returns (day_key, array or None, (err_type, err_repr) or None)."""
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
                batch = days_to_fetch[batch_start:batch_start + batch_size]
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
                    if (now - last_progress_emit) >= float(self._progress_log_interval_seconds or 0.0):
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
                        err_type, err_repr = _format_archive_exc(result)
                        self._log(
                            "warning",
                            "archive_day_failed",
                            symbol=symbol,
                            day=day_key,
                            error=err_repr,
                            error_type=err_type,
                        )
                        skipped += 1
                    elif result[2] is not None:  # (error_type, error_repr)
                        err_type, err_repr = result[2]
                        self._log(
                            "warning",
                            "archive_day_failed",
                            symbol=symbol,
                            day=day_key,
                            error=err_repr,
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
        """
        if max_age_ms is not None and max_age_ms < 0:
            raise ValueError("max_age_ms cannot be negative")

        # Force refetch: clear known gaps in the requested range
        if force_refetch_gaps:
            # Compute actual range first
            now = _utc_now_ms()
            eff_end = end_ts if end_ts is not None else _floor_minute(now)
            eff_start = start_ts if start_ts is not None else (
                int(eff_end) - self.default_window_candles * ONE_MIN_MS
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
                now = _utc_now_ms()
                finalized_end = (int(now) // period_ms) * period_ms - period_ms
                if end_ts is None:
                    end_ts = finalized_end
                else:
                    end_ts = min((int(end_ts) // period_ms) * period_ms, finalized_end)

                if start_ts is None:
                    # default window expressed in number of requested-tf buckets
                    start_ts = int(end_ts) - self.default_window_candles * period_ms
                start_ts = (int(start_ts) // period_ms) * period_ms

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
                    try:
                        sym_cache.move_to_end(cache_key)
                    except Exception:
                        pass
                    if (
                        max_age_ms is None
                        or max_age_ms == 0
                        or (now - int(fetched_at)) <= int(max_age_ms)
                    ):
                        return arr_cached

                # If disk has full coverage for this TF window, serve it without network
                if isinstance(disk_arr, np.ndarray) and disk_arr.size:
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

                end_excl = int(end_ts) + period_ms

                async with self._acquire_fetch_lock(symbol, out_tf):
                    try:
                        disk_arr = self._load_from_disk(symbol, start_ts, end_ts, timeframe=out_tf)
                    except Exception:
                        disk_arr = None

                    if isinstance(disk_arr, np.ndarray) and disk_arr.size:
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

                    try:
                        fetched = await self._fetch_ohlcv_paginated(
                            symbol,
                            int(start_ts),
                            int(end_excl),
                            timeframe=out_tf,
                            on_batch=_persist_tf_batch,
                        )
                    except TypeError:
                        fetched = await self._fetch_ohlcv_paginated(
                            symbol,
                            int(start_ts),
                            int(end_excl),
                            timeframe=out_tf,
                        )
                    if fetched.size == 0:
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
                    out = self._slice_ts_range(fetched, start_ts, end_ts)
                    if out.size and not persisted_batches:
                        self._persist_batch(symbol, out, timeframe=out_tf)
                    sym_cache[cache_key] = (out, int(now))
                    try:
                        sym_cache.move_to_end(cache_key)
                    except Exception:
                        pass
                    while len(sym_cache) > self._tf_range_cache_cap:
                        sym_cache.popitem(last=False)
                    self._tf_range_cache[symbol] = sym_cache
                    return out

        now = _utc_now_ms()
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

        if start_ts > end_ts:
            return np.empty((0,), dtype=CANDLE_DTYPE)

        # Optionally refresh if range touches the latest finalized minute
        allow_fetch_present = True
        latest_finalized = _floor_minute(now) - ONE_MIN_MS
        if end_ts >= latest_finalized and self.exchange is not None:
            if max_age_ms == 0:
                self._log(
                    "debug",
                    "get_candles_force_refresh",
                    symbol=symbol,
                    end_ts=end_ts,
                )
                await self.refresh(symbol, through_ts=end_ts)
            elif max_age_ms is not None and max_age_ms > 0:
                last_ref = self._get_last_refresh_ms(symbol)
                self._log(
                    "debug",
                    "get_candles_check_refresh",
                    symbol=symbol,
                    end_ts=end_ts,
                    last_refresh_ms=last_ref,
                    max_age_ms=max_age_ms,
                    now=now,
                )
                if last_ref == 0 or (now - last_ref) > int(max_age_ms):
                    await self.refresh(symbol, through_ts=end_ts)
                else:
                    allow_fetch_present = False

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

        # For historical ranges, if we don't have shards for all days yet, fetch
        # exactly the range and persist shards for future calls.
        end_finalized = latest_finalized
        # Treat ranges ending exactly at the latest finalized minute as present-touching
        historical = end_ts < end_finalized
        if self.exchange is not None and historical:
            # If the requested historical window is not fully covered in memory,
            # attempt to fetch unknown missing spans, regardless of shard presence.
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

                unknown_missing = [(s, e) for (s, e) in missing_before if not span_in_persistent_gap(s, e)]

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
                                    (s, e) for (s, e) in missing_after if not span_in_persistent_gap(s, e)
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
                                self._log(
                                    "info",
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
                                    spans_to_fetch = [day_windows[k] for k in sorted(day_windows.keys())]
                                    if len(spans_to_fetch) != len(unknown_after):
                                        self._log(
                                            "info",
                                            "historical_missing_spans_coalesced",
                                            symbol=symbol,
                                            spans_before=len(unknown_after),
                                            spans_after=len(spans_to_fetch),
                                        )
                                except Exception:
                                    spans_to_fetch = list(unknown_after)

                                # Fetch only the missing spans (not the whole historical range).
                                for s, e in spans_to_fetch:
                                    s2 = max(int(s), int(adj_start_ts))
                                    e2 = int(e)
                                    if e2 < s2:
                                        continue
                                    span_end_excl = min(e2 + ONE_MIN_MS, end_excl)
                                    if s2 >= span_end_excl:
                                        continue
                                    try:
                                        fetched = await self._fetch_ohlcv_paginated(
                                            symbol,
                                            s2,
                                            span_end_excl,
                                            on_batch=_persist_hist_batch,
                                        )
                                    except TypeError:
                                        fetched = await self._fetch_ohlcv_paginated(
                                            symbol,
                                            s2,
                                            span_end_excl,
                                        )
                                    if deferred_index_any:
                                        try:
                                            self.flush_deferred_index(symbol, tf="1m")
                                        except Exception as exc:  # best-effort; keep fetching even if index update fails
                                            if not flush_failed_once:
                                                try:
                                                    err_type = type(exc).__name__
                                                    err_repr = repr(exc)
                                                except Exception:
                                                    err_type = "Exception"
                                                    err_repr = "<unrepresentable exception>"
                                                self._log(
                                                    "warning",
                                                    "flush_deferred_index_failed",
                                                    symbol=symbol,
                                                    timeframe="1m",
                                                    error_type=err_type,
                                                    error=err_repr,
                                                )
                                                flush_failed_once = True
                                        deferred_index_any = False
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
                                        try:
                                            self.flush_deferred_index(symbol, tf="1m")
                                        except Exception as exc:  # best-effort; keep fetching even if index update fails
                                            if not flush_failed_once:
                                                try:
                                                    err_type = type(exc).__name__
                                                    err_repr = repr(exc)
                                                except Exception:
                                                    err_type = "Exception"
                                                    err_repr = "<unrepresentable exception>"
                                                self._log(
                                                    "warning",
                                                    "flush_deferred_index_failed",
                                                    symbol=symbol,
                                                    timeframe="1m",
                                                    error_type=err_type,
                                                    error=err_repr,
                                                )
                                                flush_failed_once = True
                            arr = (
                                np.sort(self._cache[symbol], order="ts")
                                if symbol in self._cache
                                else np.empty((0,), dtype=CANDLE_DTYPE)
                            )
                            sub = self._slice_ts_range(arr, start_ts, end_ts) if arr.size else arr
                            still_missing = self._missing_spans(sub, start_ts, end_ts)
                            # Re-fetch inception_ts after archive prefetch (may have been discovered)
                            inception_ts = self._get_inception_ts(symbol)
                            for s, e in still_missing:
                                if not span_in_persistent_gap(s, e):
                                    # If gap is before known inception, mark as pre_inception
                                    # immediately (no retries, no warning)
                                    if inception_ts is not None and e < inception_ts:
                                        self._add_known_gap(
                                            symbol, s, e,
                                            reason="pre_inception",
                                            retry_count=_GAP_MAX_RETRIES,  # Persistent immediately
                                        )
                                    else:
                                        # Normal gap - will retry and eventually warn
                                        self._add_known_gap(
                                            symbol, s, e,
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
                        self._log(
                            "debug",
                            "get_candles_present_inner",
                            symbol=symbol,
                            need_fetch=need_fetch_inner,
                            fetch_start=fetch_start,
                            last_have=last_have if sub.size else None,
                            end_excl=end_excl,
                            sub_size=int(sub.shape[0]) if sub.size else 0,
                        )
                        if need_fetch_inner:
                            persisted_batches = False

                            def _persist_present_batch(batch: np.ndarray) -> None:
                                nonlocal persisted_batches
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
                                    end_excl,
                                    on_batch=_persist_present_batch,
                                )
                            except TypeError:
                                fetched = await self._fetch_ohlcv_paginated(
                                    symbol,
                                    fetch_start,
                                    end_excl,
                                )
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
                    persisted_batches = False

                    def _persist_tail_batch(batch: np.ndarray) -> None:
                        nonlocal persisted_batches
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
                        )
                    except TypeError:
                        fetched = await self._fetch_ohlcv_paginated(
                            symbol,
                            fetch_start,
                            end_excl_range,
                        )
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

                # Attempt limited targeted fetches for unknown spans
                attempts = 0
                attempted: List[Tuple[int, int]] = []
                noresult: List[Tuple[int, int]] = []
                for s, e in missing:
                    if attempts >= 3:
                        break
                    if span_in_persistent_gap_present(s, e):
                        continue
                    end_excl_gap = e + ONE_MIN_MS
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
                        if not any(ms == s and me == e for ms, me in missing_now):
                            continue
                        persisted_batches = False

                        def _persist_gap_batch(batch: np.ndarray) -> None:
                            nonlocal persisted_batches
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
                                s,
                                end_excl_gap,
                                on_batch=_persist_gap_batch,
                            )
                        except TypeError:
                            fetched = await self._fetch_ohlcv_paginated(
                                symbol,
                                s,
                                end_excl_gap,
                            )
                        attempts += 1
                        attempted.append((s, e))
                        if fetched.size:
                            if not persisted_batches:
                                self._persist_batch(
                                    symbol,
                                    fetched,
                                    timeframe="1m",
                                    merge_cache=True,
                                    last_refresh_ms=now,
                                )
                            arr = np.sort(self._cache[symbol], order="ts")
                            sub = self._slice_ts_range(arr, start_ts, end_ts, assume_sorted=True)
                        else:
                            noresult.append((s, e))
                # After attempts, recompute missing and tag remaining as known gaps
                still_missing = self._missing_spans(sub, start_ts, inclusive_end)
                # Only mark as known those attempted spans that still remain missing
                for s, e in noresult:
                    # find overlapping portion with any still missing
                    for ms, me in still_missing:
                        if not (e < ms or s > me):
                            self._add_known_gap(symbol, max(s, ms), min(e, me))

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

        result = self.standardize_gaps(
            data_for_gaps,
            start_ts=start_ts,
            end_ts=end_ts,
            strict=strict,
            fill_leading_gaps=fill_leading_gaps,
            assume_sorted=True,
        )

        # Log accumulated persistent gap summary (throttled to once per minute)
        self._log_persistent_gap_summary()

        return result

    async def get_current_close(self, symbol: str, max_age_ms: Optional[int] = None) -> float:
        """Return latest close of the current in-progress minute for `symbol`.

        Prefers candles over tickers:
        - Cached current close within TTL
        - Fresh in-memory current-minute candle
        - get_candles for the current minute
        - As last resort: fetch_ticker
        - Fallback: last finalized cached close
        """
        if max_age_ms is not None and max_age_ms < 0:
            raise ValueError("max_age_ms cannot be negative")
        now = _utc_now_ms()
        end_current = _floor_minute(now)

        # 1) TTL cache
        if max_age_ms is not None and max_age_ms > 0:
            prev = self._current_close_cache.get(symbol)
            if prev is not None:
                price, updated = prev
                if (now - int(updated)) <= int(max_age_ms):
                    self._log("debug", "get_current_close_cache_hit", symbol=symbol)
                    return float(price)

        price: Optional[float] = None

        # 2) In-memory current-minute candle fresh enough
        try:
            arr = self._cache.get(symbol)
            if arr is not None and arr.size:
                arr_sorted = np.sort(_ensure_dtype(arr), order="ts")
                last_ts = int(arr_sorted[-1]["ts"])
                if last_ts == end_current:
                    fresh_enough = True
                    if max_age_ms is not None and max_age_ms > 0:
                        last_refresh = self._get_last_refresh_ms(symbol)
                        fresh_enough = (now - int(last_refresh)) <= int(max_age_ms)
                    if fresh_enough:
                        price = float(arr_sorted[-1]["c"])
                        self._current_close_cache[symbol] = (price, now)
                        self._log("debug", "get_current_close_mem_candle", symbol=symbol)
                        return price
        except Exception:
            pass

        # 3) Use candles API to get current minute
        got = None
        try:
            self._log(
                "debug",
                "get_current_close_via_candles",
                symbol=symbol,
                start_ts=end_current,
                end_ts=end_current,
            )
            got = await self.get_candles(
                symbol,
                start_ts=end_current,
                end_ts=end_current,
                max_age_ms=max_age_ms,
                timeframe=None,
                strict=False,
            )
            if got is not None and got.size:
                got_sorted = np.sort(_ensure_dtype(got), order="ts")
                price = float(got_sorted[-1]["c"])
                self._current_close_cache[symbol] = (price, now)
                self._log("debug", "get_current_close_from_candles", symbol=symbol)
                return price
        except Exception:
            pass

        if got is None or got.size == 0:
            try:
                last_ref = self._get_last_refresh_ms(symbol)
            except Exception:
                last_ref = 0
            # If we have a recent refresh (or TTL not enforced), fall back to last finalized candle
            # to avoid redundant tail fetches. Treat max_age_ms=None as no TTL barrier.
            ttl_ok = True
            if max_age_ms is not None and max_age_ms > 0:
                ttl_ok = (now - int(last_ref)) <= int(max_age_ms)
            if last_ref and ttl_ok:
                last_final = int(end_current - ONE_MIN_MS)
                if last_final >= 0:
                    try:
                        got_prev = await self.get_candles(
                            symbol,
                            start_ts=last_final,
                            end_ts=last_final,
                            max_age_ms=max_age_ms,
                            timeframe=None,
                            strict=False,
                        )
                        if got_prev is not None and got_prev.size:
                            got_prev_sorted = np.sort(_ensure_dtype(got_prev), order="ts")
                            price = float(got_prev_sorted[-1]["c"])
                            self._current_close_cache[symbol] = (price, now)
                            self._log(
                                "debug",
                                "get_current_close_from_candles_finalized",
                                symbol=symbol,
                                ts=int(got_prev_sorted[-1]["ts"]),
                            )
                            return price
                    except Exception:
                        pass

        # 3b) Directly fetch a small tail window via OHLCV (with cross-process lock) and merge to cache
        if self.exchange is not None:
            try:
                async with self._acquire_fetch_lock(symbol, "1m"):
                    now_locked = _utc_now_ms()
                    end_current_locked = _floor_minute(now_locked)
                    last_final_locked = end_current_locked - ONE_MIN_MS

                    # Refresh cache from disk before deciding to fetch
                    try:
                        self._load_from_disk(
                            symbol, last_final_locked, end_current_locked, timeframe="1m"
                        )
                    except Exception:
                        pass

                    arr_cache = self._cache.get(symbol)
                    if arr_cache is not None and arr_cache.size:
                        arr_sorted = np.sort(_ensure_dtype(arr_cache), order="ts")
                        last_ts = int(arr_sorted[-1]["ts"])
                        if last_ts >= end_current_locked:
                            price = float(arr_sorted[-1]["c"])
                            self._current_close_cache[symbol] = (price, now_locked)
                            self._set_last_refresh_meta(symbol, last_refresh_ms=now_locked)
                            self._log(
                                "debug",
                                "get_current_close_mem_candle_locked",
                                symbol=symbol,
                            )
                            return price
                        if last_ts >= last_final_locked:
                            price = float(arr_sorted[-1]["c"])
                            self._current_close_cache[symbol] = (price, now_locked)
                            self._log(
                                "debug",
                                "get_current_close_from_candles_finalized_locked",
                                symbol=symbol,
                                ts=last_ts,
                            )
                            return price

                    n = int(self.overlap_candles) if getattr(self, "overlap_candles", 0) else 1
                    if n <= 0:
                        n = 1
                    try:
                        n = int(min(max(1, n), int(self._ccxt_limit_default)))
                    except Exception:
                        n = max(1, n)
                    since_tail = max(0, int(end_current_locked) - ONE_MIN_MS * (n - 1))
                    self._log(
                        "debug",
                        "ccxt_fetch_ohlcv_tail_for_current_close",
                        symbol=symbol,
                        tf="1m",
                        since_ts=since_tail,
                        limit=n,
                    )
                    rows = await self._ccxt_fetch_ohlcv_once(
                        symbol,
                        since_ms=since_tail,
                        limit=n,
                        end_exclusive_ms=None,
                        timeframe="1m",
                    )
                    arr = self._normalize_ccxt_ohlcv(rows)
                    if arr.size:
                        price = float(arr[-1]["c"])
                        merged = self._merge_overwrite(self._ensure_symbol_cache(symbol), arr)
                        self._cache[symbol] = merged
                        try:
                            self._enforce_memory_retention(symbol)
                            self._save_range(symbol, arr, timeframe="1m")
                        except Exception:
                            pass
                        self._set_last_refresh_meta(symbol, last_refresh_ms=now_locked)
                        self._current_close_cache[symbol] = (price, now_locked)
                        self._log(
                            "debug",
                            "get_current_close_from_direct_ohlcv",
                            symbol=symbol,
                            rows=arr.shape[0],
                        )
                        return price
            except Exception:
                pass

        # 4) Last resort: ticker
        if self.exchange is not None:
            try:
                if hasattr(self.exchange, "fetch_ticker"):
                    self._log("debug", "ccxt_fetch_ticker", symbol=symbol)
                    if getattr(self, "_net_sem", None) is not None:
                        async with self._net_sem:  # type: ignore[attr-defined]
                            t = await self.exchange.fetch_ticker(symbol)
                    else:
                        t = await self.exchange.fetch_ticker(symbol)
                    self._log(
                        "debug",
                        "ccxt_fetch_ticker_ok",
                        symbol=symbol,
                        last=(t.get("last") if isinstance(t, dict) else None),
                        close=(t.get("close") if isinstance(t, dict) else None),
                    )
                    price = float(t.get("last") or t.get("bid") or t.get("ask")) if t else None
                    if price is not None:
                        self._current_close_cache[symbol] = (price, now)
                        return price
            except Exception:
                pass

        # 5) Fallback to last cached finalized candle
        if price is None:
            arr2 = self._cache.get(symbol)
            if arr2 is not None and arr2.size:
                arr2 = np.sort(_ensure_dtype(arr2), order="ts")
                price = float(arr2[-1]["c"])
                self._log("debug", "get_current_close_from_cache_finalized", symbol=symbol)

        if price is None:
            return float("nan")

        self._current_close_cache[symbol] = (float(price), int(now))
        return float(price)

    # ----- EMA helpers -----

    def _ema(self, values: np.ndarray, span: float) -> float:
        return float(self._ema_series(values, span)[-1])

    def _ema_series(self, values: np.ndarray, span: float) -> np.ndarray:
        """Return bias-corrected EMA (pandas ewm adjust=True) over `values`."""

        n = int(values.shape[0])
        if n == 0:
            return np.empty((0,), dtype=np.float64)
        span = float(span)
        alpha = 2.0 / (span + 1.0)
        one_minus = 1.0 - alpha
        out = np.empty((n,), dtype=np.float64)
        num = float(values[0])
        den = 1.0
        out[0] = num / den
        for i in range(1, n):
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

    async def _latest_finalized_range(
        self, span: float, *, period_ms: int = ONE_MIN_MS
    ) -> Tuple[int, int]:
        span_candles = max(1, int(math.ceil(float(span))))
        now = _utc_now_ms()
        # Align to timeframe buckets and exclude current in-progress bucket
        end_floor = (int(now) // int(period_ms)) * int(period_ms)
        end_ts = int(end_floor - period_ms)
        start_ts = int(end_ts - period_ms * (span_candles - 1))
        return start_ts, end_ts

    async def get_latest_ema_close(
        self,
        symbol: str,
        span: float,
        max_age_ms: Optional[int] = None,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
    ) -> float:
        """Return latest EMA of close over last `span` finalized candles.

        Supports higher timeframe via `tf`/`timeframe`.
        """
        out_tf = timeframe if timeframe is not None else tf
        period_ms = _tf_to_ms(out_tf)
        start_ts, end_ts = await self._latest_finalized_range(span, period_ms=period_ms)
        # EMA result cache: reuse if end_ts unchanged and within TTL
        now = _utc_now_ms()
        tf_key = str(period_ms)
        key = ("close", float(span), tf_key)
        cache = self._ema_cache.setdefault(symbol, {})
        if max_age_ms is not None and max_age_ms > 0 and key in cache:
            val, cached_end_ts, computed_at = cache[key]
            if int(cached_end_ts) == int(end_ts) and (now - int(computed_at)) <= int(max_age_ms):
                return float(val)
        arr = await self.get_candles(
            symbol, start_ts=start_ts, end_ts=end_ts, max_age_ms=max_age_ms, timeframe=out_tf
        )
        if arr.size == 0:
            return float("nan")
        closes = np.asarray(arr["c"], dtype=np.float64)
        res = float(self._ema(closes, span))
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
        """Return latest close for current minute per symbol.

        Uses get_current_close per symbol with TTL. Returns 0.0 on failure.
        """
        out: Dict[str, float] = {}
        if not symbols:
            return out

        async def one(sym: str) -> float:
            try:
                val = await self.get_current_close(sym, max_age_ms=max_age_ms)
                return float(val) if isinstance(val, (int, float)) else 0.0
            except Exception:
                return 0.0

        tasks = {s: asyncio.create_task(one(s)) for s in symbols}
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
    ) -> float:
        return await self._get_latest_ema_generic(
            symbol,
            span,
            max_age_ms,
            timeframe,
            tf=tf,
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
        now = _utc_now_ms()
        tf_key = str(period_ms)
        key = (metric_key, float(span), tf_key)
        cache = self._ema_cache.setdefault(symbol, {})
        if max_age_ms is not None and max_age_ms > 0 and key in cache:
            val, cached_end_ts, computed_at = cache[key]
            if int(cached_end_ts) == int(end_ts) and (now - int(computed_at)) <= int(max_age_ms):
                return float(val)
        arr = await self.get_candles(
            symbol, start_ts=start_ts, end_ts=end_ts, max_age_ms=max_age_ms, timeframe=out_tf
        )
        if arr.size == 0:
            return float("nan")
        series = series_fn(arr)
        res = float(self._ema(series, span))
        cache[key] = (res, int(end_ts), int(now))
        return res

    async def get_latest_ema_metrics(
        self,
        symbol: str,
        spans_by_metric: Dict[str, float],
        max_age_ms: Optional[int] = None,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute multiple latest-EMA metrics with a single candles fetch.

        This is an optimization wrapper around get_candles() + EMA calculations. It preserves the
        per-metric behavior of get_latest_ema_* helpers by:
        - Using a single `get_candles()` call for the largest requested span.
        - For 1m candles: applying gap standardization per metric window (same as get_candles()).
        - Caching results in `self._ema_cache` per (metric, span, timeframe).
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
        now = _utc_now_ms()
        tf_key = str(period_ms)

        cache = self._ema_cache.setdefault(symbol, {})
        missing: List[str] = []
        for metric_key, span in spans_by_metric.items():
            key = (str(metric_key), float(span), tf_key)
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
            strict=True if period_ms == ONE_MIN_MS else False,
            timeframe=out_tf,
        )
        if raw.size == 0:
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
            # Get window ending at end_ts. Prefer slicing by tail length; if data is short, use what we have.
            tail = raw[-span_candles:] if raw.size > span_candles else raw
            if period_ms == ONE_MIN_MS:
                # Re-apply gap standardization on the requested metric window.
                # This matches get_candles(strict=False) behavior for the same [start,end] window.
                # tail is a slice of sorted get_candles output, so assume_sorted=True
                metric_start_ts = int(end_ts - period_ms * (span_candles - 1))
                tail = self.standardize_gaps(
                    tail, start_ts=metric_start_ts, end_ts=end_ts, strict=False, assume_sorted=True
                )
            if tail.size == 0:
                out[metric_key] = float("nan")
                continue
            series = series_for(metric_key, tail)
            res = float(self._ema(series, span))
            out[metric_key] = res
            cache[(metric_key, span, tf_key)] = (res, int(end_ts), int(now))

        return out

    async def get_latest_ema_log_range(
        self,
        symbol: str,
        span: float,
        max_age_ms: Optional[int] = None,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
    ) -> float:
        return await self._get_latest_ema_generic(
            symbol,
            span,
            max_age_ms,
            timeframe,
            tf=tf,
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

    async def refresh(self, symbol: str, through_ts: Optional[int] = None) -> None:
        """Fetch new candles and merge into cache.

        - Overlaps by `overlap_candles`
        - Excludes current in-progress minute
        - No-op if `self.exchange` is None
        """
        if self.exchange is None:
            return None

        now = _utc_now_ms()
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
            if last_ts >= end_exclusive - ONE_MIN_MS:
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
                if last_ts >= end_exclusive - ONE_MIN_MS:
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
            )

            def _persist_refresh_batch(batch: np.ndarray) -> None:
                nonlocal persisted_batches
                persisted_batches = True
                self._persist_batch(
                    symbol,
                    batch,
                    timeframe="1m",
                    merge_cache=True,
                    last_refresh_ms=now_fetch,
                )

            try:
                new_arr = await self._fetch_ohlcv_paginated(
                    symbol,
                    since,
                    end_exclusive,
                    on_batch=_persist_refresh_batch,
                )
            except TypeError:
                new_arr = await self._fetch_ohlcv_paginated(symbol, since, end_exclusive)
            if new_arr.size == 0:
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
            except Exception as exc:  # best-effort; legacy cache may be unreadable, fall back to primary write
                try:
                    err_type = type(exc).__name__
                    err_repr = repr(exc)
                except Exception:
                    err_type = "Exception"
                    err_repr = "<unrepresentable exception>"
                self._log(
                    "warning",
                    "legacy_day_quality_check_failed",
                    symbol=symbol,
                    timeframe=tf_norm,
                    day=date_key,
                    error_type=err_type,
                    error=err_repr,
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
