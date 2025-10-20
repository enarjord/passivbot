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
import time
import zlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Callable, Dict, List, Optional, Tuple
from collections import OrderedDict

import numpy as np
import portalocker  # type: ignore


# ----- Constants and dtypes -----

ONE_MIN_MS = 60_000

_LOCK_TIMEOUT_SECONDS = 10.0
_LOCK_STALE_SECONDS = 180.0
_LOCK_BACKOFF_INITIAL = 0.1
_LOCK_BACKOFF_MAX = 2.0


@dataclass
class _LockRecord:
    lock: portalocker.Lock
    count: int
    acquired_at: float


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
    return symbol.replace("/", "_")


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
        # Optional global concurrency limiter for remote ccxt calls
        max_concurrent_requests: int | None = None,
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
        # Timeout parameters for cross-process fetch locks
        self._lock_timeout_seconds = float(_LOCK_TIMEOUT_SECONDS)
        self._lock_stale_seconds = float(_LOCK_STALE_SECONDS)
        self._lock_backoff_initial = float(_LOCK_BACKOFF_INITIAL)
        self._lock_backoff_max = float(_LOCK_BACKOFF_MAX)
        # Reentrant bookkeeping for portalocker fetch locks: key -> _LockRecord
        self._held_fetch_locks: Dict[Tuple[str, str], _LockRecord] = {}

        self._setup_logging()
        self._cleanup_stale_locks()

        # Initialize optional global semaphore for remote calls
        try:
            mcr = None if max_concurrent_requests in (None, 0) else int(max_concurrent_requests)
            self._net_sem = asyncio.Semaphore(mcr) if (mcr and mcr > 0) else None
        except Exception:
            self._net_sem = None

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
            self._touch_lockfile(path)

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
            is_network = isinstance(event, str) and event.startswith("ccxt_")
            if self.debug_level == 1 and not is_network:
                return
            self.log.debug(msg)
        elif level == "info":
            self.log.info(msg)
        elif level == "warning":
            self.log.warning(msg)
        else:
            self.log.error(msg)

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
                lock=held.lock, count=held.count + 1, acquired_at=held.acquired_at
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
                    await self._release_lock(record.lock, lock_path, symbol, tf_norm)
                else:
                    self._held_fetch_locks[key] = _LockRecord(
                        lock=record.lock, count=record.count - 1, acquired_at=record.acquired_at
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
                    lock=lock_obj, count=1, acquired_at=acquired_at
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
                        await self._release_lock(record.lock, lock_path, symbol, tf_norm)
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
        """Return mapping date_key -> path for available shard files on disk."""
        sd = Path(self._symbol_dir(symbol, timeframe=timeframe, tf=tf))
        if not sd.exists():
            return {}
        out: Dict[str, str] = {}
        for p in sd.glob("*.npy"):
            name = p.stem  # YYYY-MM-DD
            if len(name) == 10 and name[4] == "-" and name[7] == "-":
                out[name] = str(p)
        return out

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

    def _load_shard(self, path: str) -> np.ndarray:
        try:
            with open(path, "rb") as f:
                arr = np.load(f, allow_pickle=False)
            return _ensure_dtype(arr)
        except Exception as e:  # pragma: no cover - best effort
            self.log.warning(f"Failed loading shard {path}: {e}")
            return np.empty((0,), dtype=CANDLE_DTYPE)

    def _load_from_disk(
        self,
        symbol: str,
        start_ts: int,
        end_ts: int,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Load any shards intersecting [start_ts, end_ts] and merge into cache."""
        try:
            tf_norm = self._normalize_timeframe_arg(timeframe, tf)
            shards = self._iter_shard_paths(symbol, tf=tf_norm)
            if not shards:
                return
            load_keys = []
            # Select shards by filename key intersection to range
            for key, path in shards.items():
                try:
                    y, m, d = map(int, key.split("-"))
                except Exception:
                    continue
                # midnight UTC of date
                tm = time.struct_time((y, m, d, 0, 0, 0, 0, 0, 0))
                day_start = int(calendar.timegm(tm)) * 1000
                day_end = day_start + 24 * 60 * 60 * 1000 - ONE_MIN_MS
                if not (day_end < start_ts or day_start > end_ts):
                    load_keys.append((key, path))
            if not load_keys:
                return
            # Load and merge
            arrays = [self._load_shard(p) for _, p in sorted(load_keys)]
            arrays = [a for a in arrays if a.size]
            if not arrays:
                return
            merged_disk = np.sort(np.concatenate(arrays), order="ts")
            self._log(
                "debug",
                "load_from_disk",
                symbol=symbol,
                timeframe=tf_norm,
                days=len(load_keys),
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
    ) -> None:
        """Persist candles by merging with existing shards on disk."""
        if arr.size == 0:
            return
        arr = np.sort(_ensure_dtype(arr), order="ts")
        tf_norm = self._normalize_timeframe_arg(timeframe, tf)
        shard_paths = self._iter_shard_paths(symbol, tf=tf_norm)

        def flush_bucket(key: Optional[str], bucket: List[Tuple]) -> None:
            if key is None or not bucket:
                return
            chunk = np.array(bucket, dtype=CANDLE_DTYPE)
            existing = np.empty((0,), dtype=CANDLE_DTYPE)
            path = shard_paths.get(key)
            if path and os.path.exists(path):
                existing = self._load_shard(path)
            merged = self._merge_overwrite(existing, chunk)
            self._save_shard(symbol, key, merged, tf=tf_norm)
            shard_paths[key] = self._shard_path(symbol, key, tf=tf_norm)

        current_key: Optional[str] = None
        bucket: List[Tuple] = []
        for row in arr:
            key = self._date_key(int(row["ts"]))
            if current_key is None:
                current_key = key
            if key != current_key:
                flush_bucket(current_key, bucket)
                bucket = []
                current_key = key
            bucket.append(tuple(row.tolist()))
        flush_bucket(current_key, bucket)

    def _persist_batch(
        self,
        symbol: str,
        batch: np.ndarray,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
        merge_cache: bool = False,
        last_refresh_ms: Optional[int] = None,
    ) -> None:
        """Merge `batch` into memory (optional) and persist incrementally to disk."""
        if batch.size == 0:
            return
        arr = np.sort(_ensure_dtype(batch), order="ts")
        tf_norm = self._normalize_timeframe_arg(timeframe, tf)

        if merge_cache or tf_norm == "1m":
            merged_cache = self._merge_overwrite(self._ensure_symbol_cache(symbol), arr)
            self._cache[symbol] = merged_cache
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

        self._save_range_incremental(symbol, arr, timeframe=tf_norm)

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
        combo = np.sort(combo, order="ts")
        ts = combo["ts"].astype(np.int64)
        # deduplicate keeping last occurrence
        keep = np.ones(len(combo), dtype=bool)
        # scan from end, mark first-seen; then reverse mask
        seen = {}
        for i in range(len(combo) - 1, -1, -1):
            t = int(ts[i])
            if t in seen:
                keep[i] = False
            else:
                seen[t] = True
        merged = combo[keep]
        # Enforce in-memory retention: keep only the latest N candles per symbol (applied by caller after assign)
        return merged

    # ----- Known gap helpers -----

    def _get_known_gaps(self, symbol: str) -> List[Tuple[int, int]]:
        idx = self._ensure_symbol_index(symbol)
        gaps = idx.get("meta", {}).get("known_gaps", [])
        out: List[Tuple[int, int]] = []
        for it in gaps:
            try:
                a, b = int(it[0]), int(it[1])
                if a <= b:
                    out.append((a, b))
            except Exception:
                continue
        return out

    def _save_known_gaps(self, symbol: str, gaps: List[Tuple[int, int]]) -> None:
        # merge overlaps
        gaps = sorted(gaps)
        merged: List[Tuple[int, int]] = []
        for s, e in gaps:
            if not merged or s > merged[-1][1] + ONE_MIN_MS:
                merged.append((s, e))
            else:
                ps, pe = merged[-1]
                merged[-1] = (ps, max(pe, e))
        idx = self._ensure_symbol_index(symbol)
        idx["meta"]["known_gaps"] = [[int(s), int(e)] for s, e in merged]
        self._index[symbol] = idx
        self._save_index(symbol)

    def _add_known_gap(self, symbol: str, start_ts: int, end_ts: int) -> None:
        gaps = self._get_known_gaps(symbol)
        gaps.append((int(start_ts), int(end_ts)))
        self._save_known_gaps(symbol, gaps)

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

        backoff = 0.5
        for attempt in range(5):
            try:
                params = {}
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
                tf_norm = self._normalize_timeframe_arg(timeframe, tf, default=self._ccxt_timeframe)
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
                self._log(
                    "warning",
                    "ccxt_fetch_ohlcv_failed",
                    symbol=symbol,
                    attempt=attempt + 1,
                    error=str(e),
                )
                await asyncio.sleep(backoff)
                backoff *= 2
        return []

    # ----- Array slicing helpers -----

    def _slice_ts_range(self, arr: np.ndarray, start_ts: int, end_ts: int) -> np.ndarray:
        """Return arr sliced to [start_ts, end_ts] inclusive by 'ts'.

        Assumes arr is structured dtype CANDLE_DTYPE; sorts by ts before slicing.
        """
        if arr.size == 0:
            return arr
        arr = np.sort(_ensure_dtype(arr), order="ts")
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
        since = int(since_ms)
        end_excl = int(end_exclusive_ms)
        limit = self._ccxt_limit_default
        tf_norm = self._normalize_timeframe_arg(timeframe, tf, default=self._ccxt_timeframe)
        # Derive pagination step from timeframe
        period_ms = _tf_to_ms(tf_norm)
        all_rows = []
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
                        self.log.warning(
                            f"unexpected step for tf exchange={self._ex_id} symbol={symbol} tf={tf_norm} expected={period_ms} min_step={min_step} max_step={max_step}"
                        )
                else:
                    max_step = ONE_MIN_MS
            except Exception:
                first_ts = last_ts = 0
            all_rows.append(arr)
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
        """
        a = _ensure_dtype(candles)
        if a.size == 0:
            # Nothing to standardize; caller decides how to handle empty ranges
            return a

        a = np.sort(a, order="ts")
        ts_arr = _ts_index(a)
        lo = start_ts if start_ts is not None else int(ts_arr[0])
        hi = end_ts if end_ts is not None else int(ts_arr[-1])
        lo = _floor_minute(lo)
        hi = _floor_minute(hi)

        expected = np.arange(lo, hi + ONE_MIN_MS, ONE_MIN_MS, dtype=np.int64)
        # Map from ts to row index in a
        pos = {int(t): i for i, t in enumerate(ts_arr)}

        if strict:
            # In strict mode: do not synthesize zero-candles.
            # If there are gaps, log a warning and return whatever real candles exist in range.
            missing = [int(t) for t in expected if int(t) not in pos]
            i0 = int(np.searchsorted(ts_arr, lo, side="left"))
            i1 = int(np.searchsorted(ts_arr, hi, side="right"))
            if missing:
                self._log(
                    "warning",
                    "standardize_gaps_strict_missing",
                    missing=len(missing),
                    start_ts=lo,
                    end_ts=hi,
                )
            return a[i0:i1]

        out_rows = []
        prev_close: Optional[float] = None
        # Seed prev_close from earliest included real candle if available
        if lo in pos:
            prev_close = float(a[pos[lo]]["c"])  # after we append it, close is known
        else:
            # If first expected timestamp is missing, try to seed from the first
            # available candle earlier than lo.
            idx = np.searchsorted(ts_arr, lo)
            if idx > 0:
                prev_close = float(a[idx - 1]["c"])  # previous real close

        for t in expected:
            if t in pos:
                row = a[pos[t]]
                out_rows.append(tuple(row.tolist()))
                prev_close = float(row["c"])  # update seed
            else:
                if prev_close is None:
                    # Skip until we know a previous close; this keeps behavior predictable.
                    continue
                # Synthesize a zero-candle using previous close
                out_rows.append((int(t), prev_close, prev_close, prev_close, prev_close, 0.0))

        if not out_rows:
            return np.empty((0,), dtype=CANDLE_DTYPE)
        return np.array(out_rows, dtype=CANDLE_DTYPE)

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
    ) -> np.ndarray:
        """Return candles in inclusive range [start_ts, end_ts].

        - If `end_ts` is None: floor(now/1m)*1m + 1m
        - If `start_ts` is None: last `default_window_candles` minutes
        - If `end_ts` provided but `start_ts` is None: end_ts - window
        - If `max_age_ms` == 0: force refresh (no-op when exchange is None)
        - Negative `max_age_ms` raises ValueError
        - Applies gap standardization (1m only)
        """
        if max_age_ms is not None and max_age_ms < 0:
            raise ValueError("max_age_ms cannot be negative")

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

                # Skip fetch if all missing spans are already known gaps
                missing_before = self._missing_spans(sub, start_ts, end_ts)
                known = self._get_known_gaps(symbol)

                def span_in_known(s: int, e: int) -> bool:
                    for ks, ke in known:
                        if s >= ks and e <= ke:
                            return True
                    return False

                unknown_missing = [(s, e) for (s, e) in missing_before if not span_in_known(s, e)]

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
                                (s, e) for (s, e) in missing_after if not span_in_known(s, e)
                            ]
                            if unknown_after:
                                persisted_batches = False

                                def _persist_hist_batch(batch: np.ndarray) -> None:
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
                                        adj_start_ts,
                                        end_excl,
                                        on_batch=_persist_hist_batch,
                                    )
                                except TypeError:
                                    fetched = await self._fetch_ohlcv_paginated(
                                        symbol,
                                        adj_start_ts,
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
                            still_missing = self._missing_spans(sub, start_ts, end_ts)
                            for s, e in still_missing:
                                if not span_in_known(s, e):
                                    self._add_known_gap(symbol, s, e)
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
                known = self._get_known_gaps(symbol)

                # Helper to test if a span is fully inside any known gap
                def span_in_known(s: int, e: int) -> bool:
                    for ks, ke in known:
                        if s >= ks and e <= ke:
                            return True
                    return False

                # Attempt limited targeted fetches for unknown spans
                attempts = 0
                attempted: List[Tuple[int, int]] = []
                noresult: List[Tuple[int, int]] = []
                for s, e in missing:
                    if attempts >= 3:
                        break
                    if span_in_known(s, e):
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
                            sub = self._slice_ts_range(arr, start_ts, end_ts)
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
        result = self.standardize_gaps(sub, start_ts=start_ts, end_ts=end_ts, strict=strict)

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
        span_0: int,
        span_1: int,
        max_age_ms: Optional[int] = None,
        *,
        timeframe: Optional[str] = None,
        tf: Optional[str] = None,
    ) -> Tuple[float, float]:
        """Return (lower, upper) bounds from EMAs at spans {span_0, span_1, span_2}.

        span_2 = round(sqrt(span_0 * span_1)). Forwards timeframe and TTL to
        get_latest_ema_close. Computes the three EMAs concurrently.
        """
        from math import isfinite

        s2 = int(round((float(span_0) * float(span_1)) ** 0.5))
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
        items: List[Tuple[str, int, int]],
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

        async def one(sym: str, s0: int, s1: int) -> Tuple[float, float]:
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

        try:
            self._load_from_disk(symbol, 0, end_exclusive, timeframe="1m")
        except Exception:
            pass

        existing = self._ensure_symbol_cache(symbol)
        existing_sorted = np.sort(existing, order="ts") if existing.size else existing
        existing_last_ts = int(existing_sorted[-1]["ts"]) if existing_sorted.size else None
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
                self._load_from_disk(symbol, 0, end_exclusive, timeframe="1m")
            except Exception:
                pass

            existing = self._ensure_symbol_cache(symbol)
            existing_sorted = np.sort(existing, order="ts") if existing.size else existing
            existing_last_ts = int(existing_sorted[-1]["ts"]) if existing_sorted.size else None
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
        """
        arr = _ensure_dtype(array)
        if arr.size == 0:
            return

        arr = np.sort(arr, order="ts")
        data_bytes = arr.tobytes()
        crc = int(zlib.crc32(data_bytes) & 0xFFFFFFFF)

        tf_norm = self._normalize_timeframe_arg(timeframe, tf)
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

        # Update index
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
        self._save_index(symbol, tf=tf_norm)
        # Enforce disk retention per timeframe after writing this shard
        try:
            self._enforce_disk_retention(symbol, tf=tf_norm)
        except Exception:
            pass

    # ----- Context manager and shutdown -----

    def close(self) -> None:
        """Flush and close resources. Currently a no-op for tests."""
        return None

    def __enter__(self):  # pragma: no cover - not exercised by tests
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover - not exercised by tests
        self.close()
        return False


__all__ = [
    "CandlestickManager",
    "CANDLE_DTYPE",
    "ONE_MIN_MS",
    "_floor_minute",
]
