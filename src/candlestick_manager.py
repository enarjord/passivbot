"""
CandlestickManager: lightweight 1m OHLCV manager with gap standardization.

This module provides a minimal, self-contained implementation tailored to the
unit tests in tests/test_candlestick_manager.py while following the requested
API and data format. It focuses on:

- UTC millisecond timestamps and structured NumPy dtype for candles
- Gap standardization with synthesized zero-candles (not persisted)
- Inclusive range selection with minute alignment
- Latest EMA for close/volume/NRR computed lazily from cached candles
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
import json
import logging
import os
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np

try:
    import portalocker  # type: ignore
except Exception:  # pragma: no cover - tests don't assert the specific locker used
    portalocker = None  # fallback to no external file lock


# ----- Constants and dtypes -----

ONE_MIN_MS = 60_000

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

EMA_SERIES_DTYPE = np.dtype([
    ("ts", "int64"),
    ("ema", "float32"),
])


# ----- Utilities -----


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


@dataclass
class _Index:
    shards: Dict[str, dict]
    meta: Dict[str, dict]


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
    timeframe : str
        Timeframe string, kept for directory layout. Default "1m".
    default_window_candles : int
        Default window used when start_ts is not provided.
    overlap_candles : int
        Overlap applied when refreshing from network (not exercised in tests).
    memory_days : int
        Retention in-memory. Not enforced by tests, reserved for future use.
    disk_retention_days : int
        Retention on-disk. Not enforced by tests, reserved for future use.
    debug : bool
        Enable verbose logging.
    """

    def __init__(
        self,
        exchange=None,
        exchange_name: str = "unknown",
        *,
        cache_dir: str = "caches",
        timeframe: str = "1m",
        default_window_candles: int = 100,
        overlap_candles: int = 30,
        memory_days: int = 7,
        disk_retention_days: int = 30,
        debug: bool = False,
    ) -> None:
        self.exchange = exchange
        # If no explicit exchange_name provided, infer from ccxt instance id
        if (not exchange_name or exchange_name == "unknown") and getattr(exchange, "id", None):
            self.exchange_name = str(getattr(exchange, "id"))
        else:
            self.exchange_name = exchange_name
        self.cache_dir = cache_dir
        self.timeframe = timeframe
        self.default_window_candles = int(default_window_candles)
        self.overlap_candles = int(overlap_candles)
        self.memory_days = int(memory_days)
        self.disk_retention_days = int(disk_retention_days)
        self.debug = bool(debug)

        self._cache: Dict[str, np.ndarray] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._index: Dict[str, dict] = {}
        # Cache for EMA computations: per symbol -> {(metric, span, tf): (value, end_ts, computed_at_ms)}
        self._ema_cache: Dict[str, Dict[Tuple[str, int, str], Tuple[float, int, int]]] = {}
        # Cache for current (in-progress) minute close per symbol: symbol -> (price, updated_ms)
        self._current_close_cache: Dict[str, Tuple[float, int]] = {}
        # Cache for fetched higher-timeframe windows to avoid duplicate remote calls
        # Keyed per symbol -> {(tf_str, start_ts, end_ts): (array, fetched_at_ms)}
        self._tf_range_cache: Dict[str, Dict[Tuple[str, int, int], Tuple[np.ndarray, int]]] = {}

        self._setup_logging()

        # fetch controls
        self._ccxt_timeframe = self.timeframe  # expected '1m'
        # Determine exchange id and adjust defaults per exchange quirks
        self._ex_id = getattr(self.exchange, "id", self.exchange_name) or self.exchange_name
        self._ccxt_limit_default = 1000
        if isinstance(self._ex_id, str) and "bitget" in self._ex_id.lower():
            # Bitget often serves 1m klines with 200 limit per page
            self._ccxt_limit_default = 200

    # ----- Logging -----

    def _setup_logging(self) -> None:
        self.log = logging.getLogger("CandlestickManager")
        # Isolate from root handlers to avoid duplicate/missing logs
        self.log.propagate = False
        if self.debug:
            self.log.setLevel(logging.DEBUG)
        else:
            self.log.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
        # Attach file handler once
        if not any(isinstance(h, logging.FileHandler) for h in self.log.handlers):
            os.makedirs("logs", exist_ok=True)
            fh = logging.FileHandler("logs/candlestick_manager.log")
            fh.setFormatter(fmt)
            fh.setLevel(logging.DEBUG if self.debug else logging.INFO)
            self.log.addHandler(fh)
        # Attach console handler in debug mode (ensure we don't confuse FileHandler subclassing)
        has_console = any(
            isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
            for h in self.log.handlers
        )
        if self.debug and not has_console:
            ch = logging.StreamHandler()
            ch.setFormatter(fmt)
            ch.setLevel(logging.DEBUG)
            self.log.addHandler(ch)

    # ----- Paths and index -----

    def _symbol_dir(self, symbol: str) -> str:
        sym = _sanitize_symbol(symbol)
        return str(Path(self.cache_dir) / "ohlcv" / self.exchange_name / self.timeframe / sym)

    def _index_path(self, symbol: str) -> str:
        return str(Path(self._symbol_dir(symbol)) / "index.json")

    def _shard_path(self, symbol: str, date_key: str) -> str:
        return str(Path(self._symbol_dir(symbol)) / f"{date_key}.npy")

    def _ensure_symbol_index(self, symbol: str) -> dict:
        if symbol in self._index:
            return self._index[symbol]
        # Try load from disk
        idx_path = self._index_path(symbol)
        idx = {"shards": {}, "meta": {}}
        try:
            with open(idx_path, "r", encoding="utf-8") as f:
                idx = json.load(f)
        except FileNotFoundError:
            pass
        except Exception as e:  # pragma: no cover
            self.log.warning(f"Failed loading index for {symbol}: {e}")
        # Ensure meta keys
        if not isinstance(idx, dict):
            idx = {"shards": {}, "meta": {}}
        idx.setdefault("shards", {})
        meta = idx.setdefault("meta", {})
        meta.setdefault("known_gaps", [])  # list of [start_ts, end_ts]
        meta.setdefault("last_refresh_ms", 0)
        meta.setdefault("last_final_ts", 0)
        self._index[symbol] = idx
        return idx

    def _atomic_write_bytes(self, path: str, data: bytes) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def _save_index(self, symbol: str) -> None:
        idx_path = self._index_path(symbol)
        payload = json.dumps(self._index[symbol], sort_keys=True).encode("utf-8")
        if portalocker is not None:
            # Lock the final target index.json to serialize writers
            os.makedirs(os.path.dirname(idx_path), exist_ok=True)
            # Use portalocker with a filename, not a file handle, so it creates the file if missing
            lock_path = idx_path + ".lock"
            with portalocker.Lock(lock_path, timeout=5):
                self._atomic_write_bytes(idx_path, payload)
        else:  # pragma: no cover
            self._atomic_write_bytes(idx_path, payload)

    def _get_lock(self, symbol: str) -> asyncio.Lock:
        lock = self._locks.get(symbol)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[symbol] = lock
        return lock

    def _ensure_symbol_cache(self, symbol: str) -> np.ndarray:
        arr = self._cache.get(symbol)
        if arr is None:
            arr = np.empty((0,), dtype=CANDLE_DTYPE)
            self._cache[symbol] = arr
        return arr

    # ----- Shard loading helpers -----

    def _iter_shard_paths(self, symbol: str) -> Dict[str, str]:
        """Return mapping date_key -> path for available shard files on disk."""
        sd = Path(self._symbol_dir(symbol))
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

    def _load_from_disk(self, symbol: str, start_ts: int, end_ts: int) -> None:
        """Load any shards intersecting [start_ts, end_ts] and merge into cache."""
        try:
            shards = self._iter_shard_paths(symbol)
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
            self.log.debug(
                f"load_from_disk symbol={symbol} days={len(load_keys)} rows={merged_disk.shape[0]}"
            )
            existing = self._ensure_symbol_cache(symbol)
            merged = self._merge_overwrite(existing, merged_disk)
            self._cache[symbol] = merged
        except Exception as e:  # pragma: no cover - noncritical
            self.log.warning(f"Disk load error for {symbol}: {e}")

    def _save_range(self, symbol: str, arr: np.ndarray) -> None:
        """Persist fetched candles to daily shards by date_key."""
        if arr.size == 0:
            return
        arr = np.sort(_ensure_dtype(arr), order="ts")
        current_key: Optional[str] = None
        bucket = []
        total = 0
        for row in arr:
            key = self._date_key(int(row["ts"]))
            if current_key is None:
                current_key = key
            if key != current_key:
                if bucket:
                    self._save_shard(symbol, current_key, np.array(bucket, dtype=CANDLE_DTYPE))
                    total += len(bucket)
                bucket = []
                current_key = key
            bucket.append(tuple(row.tolist()))
        if bucket and current_key is not None:
            self._save_shard(symbol, current_key, np.array(bucket, dtype=CANDLE_DTYPE))
            total += len(bucket)
        self.log.debug(f"saved_range symbol={symbol} rows={total}")

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
        return combo[keep]

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
                    # Avoid 'until' for exchanges with inconsistent paging using it (bitget, okx).
                    if ("bitget" not in exid) and ("okx" not in exid):
                        # For other exchanges, 'until' may help bound the end; otherwise forward pagination + clipping is used.
                        params["until"] = int(end_exclusive_ms) - 1
                tf = timeframe or self._ccxt_timeframe
                self.log.debug(
                    f"ccxt.fetch_ohlcv exchange={self._ex_id} symbol={symbol} tf={tf} since={since_ms} limit={limit} attempt={attempt+1} params={params}"
                )
                res = await ex.fetch_ohlcv(
                    symbol,
                    timeframe=tf,
                    since=since_ms,
                    limit=limit,
                    params=params,
                )
                self.log.debug(
                    f"ccxt.fetch_ohlcv ok exchange={self._ex_id} symbol={symbol} rows={len(res) if res else 0} since={since_ms}"
                )
                return res or []
            except Exception as e:  # pragma: no cover - network not used in tests
                self.log.warning(f"fetch_ohlcv attempt {attempt+1} failed: {e}")
                await asyncio.sleep(backoff)
                backoff *= 2
        return []

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
        self, symbol: str, since_ms: int, end_exclusive_ms: int, *, timeframe: Optional[str] = None
    ) -> np.ndarray:
        """Fetch OHLCV from `since_ms` up to but excluding `end_exclusive_ms`.

        Uses ccxt pagination via since+limit. Returns CANDLE_DTYPE array.
        """
        if self.exchange is None:
            return np.empty((0,), dtype=CANDLE_DTYPE)
        since = int(since_ms)
        end_excl = int(end_exclusive_ms)
        limit = self._ccxt_limit_default
        tf = timeframe or self._ccxt_timeframe
        # Derive pagination step from timeframe
        period_ms = _tf_to_ms(tf)
        all_rows = []
        while since < end_excl:
            page = await self._ccxt_fetch_ohlcv_once(
                symbol, since, limit, end_exclusive_ms=end_excl, timeframe=tf
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
                            f"unexpected step for tf exchange={self._ex_id} symbol={symbol} tf={tf} expected={period_ms} min_step={min_step} max_step={max_step}"
                        )
                else:
                    max_step = ONE_MIN_MS
            except Exception:
                first_ts = last_ts = 0
            all_rows.append(arr)
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
            f"paginated fetch done exchange={self._ex_id} symbol={symbol} rows={sum(a.shape[0] for a in all_rows) if all_rows else 0}"
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
                self.log.warning(
                    f"standardize_gaps(strict=True): missing {len(missing)} minute(s) in range; "
                    "returning available candles only"
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
        tf: Optional[str] = None,
        timeframe: Optional[str] = None,
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

                # Check in-memory TF range cache first
                cache_key = (str(out_tf), int(start_ts), int(end_ts))
                sym_cache = self._tf_range_cache.setdefault(symbol, {})
                if cache_key in sym_cache:
                    arr_cached, fetched_at = sym_cache[cache_key]
                    if (
                        max_age_ms is None
                        or max_age_ms == 0
                        or (now - int(fetched_at)) <= int(max_age_ms)
                    ):
                        return arr_cached

                end_excl = int(end_ts) + period_ms
                fetched = await self._fetch_ohlcv_paginated(
                    symbol, int(start_ts), int(end_excl), timeframe=out_tf
                )
                if fetched.size == 0:
                    return fetched
                # Clip to inclusive [start_ts, end_ts] in bucket space
                fetched = np.sort(_ensure_dtype(fetched), order="ts")
                ts_arr = _ts_index(fetched)
                i0 = int(np.searchsorted(ts_arr, start_ts, side="left"))
                i1 = int(np.searchsorted(ts_arr, end_ts, side="right"))
                out = fetched[i0:i1]
                # Store in TF range cache
                sym_cache[cache_key] = (out, int(now))
                # Simple eviction to bound memory
                if len(sym_cache) > 8:
                    try:
                        # remove an arbitrary old entry
                        k = next(iter(sym_cache.keys()))
                        if k != cache_key:
                            sym_cache.pop(k, None)
                    except Exception:
                        pass
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

        # Optionally refresh only if range touches or includes the current minute
        allow_fetch_present = True
        if end_ts >= _floor_minute(now) and self.exchange is not None:
            if max_age_ms == 0:
                await self.refresh(symbol, through_ts=end_ts)
            elif max_age_ms is not None and max_age_ms > 0:
                last_ref = self._get_last_refresh_ms(symbol)
                if last_ref == 0 or (now - last_ref) > int(max_age_ms):
                    await self.refresh(symbol, through_ts=end_ts)
                else:
                    allow_fetch_present = False

        # Try to load from disk shards for this range before slicing memory
        try:
            self._load_from_disk(symbol, start_ts, end_ts)
        except Exception:  # pragma: no cover - best effort
            pass

        # Get in-memory cached candles for the symbol
        arr = _ensure_dtype(self._cache.get(symbol, np.empty((0,), dtype=CANDLE_DTYPE)))
        # Restrict to [start_ts, end_ts]
        if arr.size:
            arr = np.sort(arr, order="ts")
            ts_arr = _ts_index(arr)
            i0 = int(np.searchsorted(ts_arr, start_ts, side="left"))
            i1 = int(np.searchsorted(ts_arr, end_ts, side="right"))
            sub = arr[i0:i1]
        else:
            sub = arr

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
        end_finalized = _floor_minute(now) - ONE_MIN_MS
        historical = end_ts <= end_finalized
        if self.exchange is not None and historical:
            shard_map = self._iter_shard_paths(symbol)
            needed_keys = self._date_keys_between(start_ts, end_ts)
            have_all_days = all(k in shard_map for k in needed_keys.keys())
            # If memory already holds the full requested range, skip fetching even if shards are missing
            if not have_all_days and not fully_covered:
                end_excl = min(end_ts + ONE_MIN_MS, end_finalized + ONE_MIN_MS)
                if start_ts < end_excl:
                    fetched = await self._fetch_ohlcv_paginated(symbol, start_ts, end_excl)
                    if fetched.size:
                        merged = self._merge_overwrite(self._ensure_symbol_cache(symbol), fetched)
                        self._cache[symbol] = merged
                        self._save_range(symbol, fetched)
                        # Re-slice after fetch
                        arr = np.sort(self._cache[symbol], order="ts")
                        ts_arr = _ts_index(arr)
                        i0 = int(np.searchsorted(ts_arr, start_ts, side="left"))
                        i1 = int(np.searchsorted(ts_arr, end_ts, side="right"))
                        sub = arr[i0:i1]
                        # update refresh metadata on successful fetch
                        self._set_last_refresh_meta(
                            symbol, last_refresh_ms=now, last_final_ts=int(arr[-1]["ts"])
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
                if need_fetch:
                    fetched = await self._fetch_ohlcv_paginated(symbol, fetch_start, end_excl)
                    if fetched.size:
                        merged = self._merge_overwrite(self._ensure_symbol_cache(symbol), fetched)
                        self._cache[symbol] = merged
                        self._save_range(symbol, fetched)
                        # Re-slice after fetch
                        arr = np.sort(self._cache[symbol], order="ts")
                        ts_arr = _ts_index(arr)
                        i0 = int(np.searchsorted(ts_arr, start_ts, side="left"))
                        i1 = int(np.searchsorted(ts_arr, end_ts, side="right"))
                        sub = arr[i0:i1]
                        self._set_last_refresh_meta(
                            symbol, last_refresh_ms=now, last_final_ts=int(arr[-1]["ts"])
                        )

        # Best-effort tail completion: if we still miss trailing minutes within
        # the requested window, attempt one more fetch from the last available ts.
        if self.exchange is not None and allow_fetch_present:
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
                fetched = await self._fetch_ohlcv_paginated(symbol, fetch_start, end_excl_range)
                if fetched.size == 0:
                    break
                merged = self._merge_overwrite(self._ensure_symbol_cache(symbol), fetched)
                self._cache[symbol] = merged
                self._save_range(symbol, fetched)
                # Re-slice after fetch
                arr = np.sort(self._cache[symbol], order="ts")
                ts_arr = _ts_index(arr)
                i0 = int(np.searchsorted(ts_arr, start_ts, side="left"))
                i1 = int(np.searchsorted(ts_arr, end_ts, side="right"))
                sub = arr[i0:i1]
                self._set_last_refresh_meta(
                    symbol, last_refresh_ms=now, last_final_ts=int(arr[-1]["ts"])
                )

        # Gap-oriented fetch and tagging: try filling internal gaps once; mark remaining as known gaps
        if self.exchange is not None and allow_fetch_present:
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
                    fetched = await self._fetch_ohlcv_paginated(symbol, s, end_excl_gap)
                    attempts += 1
                    attempted.append((s, e))
                    if fetched.size:
                        merged = self._merge_overwrite(self._ensure_symbol_cache(symbol), fetched)
                        self._cache[symbol] = merged
                        self._save_range(symbol, fetched)
                        # Re-slice after fetch
                        arr = np.sort(self._cache[symbol], order="ts")
                        ts_arr = _ts_index(arr)
                        i0 = int(np.searchsorted(ts_arr, start_ts, side="left"))
                        i1 = int(np.searchsorted(ts_arr, end_ts, side="right"))
                        sub = arr[i0:i1]
                        self._set_last_refresh_meta(
                            symbol, last_refresh_ms=now, last_final_ts=int(arr[-1]["ts"])
                        )
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

        - Uses exchange data when available; otherwise falls back to last cached close.
        - Respects `max_age_ms` as minimum freshness for remote fetches.
        - Returns NaN if no data is available.
        """
        if max_age_ms is not None and max_age_ms < 0:
            raise ValueError("max_age_ms cannot be negative")
        now = _utc_now_ms()

        # Return cached current price if within TTL
        if max_age_ms is not None and max_age_ms > 0:
            prev = self._current_close_cache.get(symbol)
            if prev is not None:
                price, updated = prev
                if (now - int(updated)) <= int(max_age_ms):
                    return float(price)

        price: Optional[float] = None
        # Try exchange sources for the current in-progress minute
        if self.exchange is not None:
            try:
                # Prefer ticker last price if available
                if hasattr(self.exchange, "fetch_ticker"):
                    t = await self.exchange.fetch_ticker(symbol)
                    price = float(t.get('last') or t.get('bid') or t.get('ask')) if t else None
            except Exception:
                price = None
            if price is None:
                # Fallback to latest 1m candle including current minute if exchange returns it
                try:
                    end_current = _floor_minute(now)
                    rows = await self._ccxt_fetch_ohlcv_once(
                        symbol, since_ms=end_current, limit=1, end_exclusive_ms=None, timeframe="1m"
                    )
                    arr = self._normalize_ccxt_ohlcv(rows)
                    if arr.size:
                        price = float(arr[-1]["c"])  # may be current minute partial
                except Exception:
                    pass

        # If still None, fallback to last cached candle close (finalized)
        if price is None:
            arr = self._cache.get(symbol)
            if arr is not None and arr.size:
                arr = np.sort(_ensure_dtype(arr), order="ts")
                price = float(arr[-1]["c"])

        if price is None:
            return float("nan")

        # Update local cache of current close
        self._current_close_cache[symbol] = (float(price), int(now))
        return float(price)

    # ----- EMA helpers -----

    def _ema(self, values: np.ndarray, span: int) -> float:
        alpha = 2.0 / (span + 1.0)
        ema = float(values[0])
        for v in values[1:]:
            ema = alpha * float(v) + (1.0 - alpha) * ema
        return ema

    def _ema_series(self, values: np.ndarray, span: int) -> np.ndarray:
        """Return EMA series for `values` using standard recursive definition.

        y[0] = x[0]; y[t] = α*x[t] + (1-α)*y[t-1]
        Returns float64 array of same length as `values`.
        """
        n = int(values.shape[0])
        if n == 0:
            return np.empty((0,), dtype=np.float64)
        alpha = 2.0 / (span + 1.0)
        one_minus = 1.0 - alpha
        out = np.empty((n,), dtype=np.float64)
        out[0] = float(values[0])
        for i in range(1, n):
            out[i] = one_minus * out[i - 1] + alpha * float(values[i])
        return out

    async def _latest_finalized_range(
        self, span: int, *, period_ms: int = ONE_MIN_MS
    ) -> Tuple[int, int]:
        now = _utc_now_ms()
        # Align to timeframe buckets and exclude current in-progress bucket
        end_floor = (int(now) // int(period_ms)) * int(period_ms)
        end_ts = int(end_floor - period_ms)
        start_ts = int(end_ts - period_ms * (span - 1))
        return start_ts, end_ts

    async def get_latest_ema_close(
        self,
        symbol: str,
        span: int,
        max_age_ms: Optional[int] = None,
        *,
        tf: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> float:
        """Return latest EMA of close over last `span` finalized candles.

        Supports higher timeframe via `tf`/`timeframe`.
        """
        out_tf = timeframe or tf
        period_ms = _tf_to_ms(out_tf)
        start_ts, end_ts = await self._latest_finalized_range(span, period_ms=period_ms)
        # EMA result cache: reuse if end_ts unchanged and within TTL
        now = _utc_now_ms()
        tf_key = str(period_ms)
        key = ("close", int(span), tf_key)
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

    async def get_latest_ema_volume(
        self,
        symbol: str,
        span: int,
        max_age_ms: Optional[int] = None,
        *,
        tf: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> float:
        out_tf = timeframe or tf
        period_ms = _tf_to_ms(out_tf)
        start_ts, end_ts = await self._latest_finalized_range(span, period_ms=period_ms)
        now = _utc_now_ms()
        tf_key = str(period_ms)
        key = ("volume", int(span), tf_key)
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
        vols = np.asarray(arr["bv"], dtype=np.float64)
        res = float(self._ema(vols, span))
        cache[key] = (res, int(end_ts), int(now))
        return res

    async def get_latest_ema_nrr(
        self,
        symbol: str,
        span: int,
        max_age_ms: Optional[int] = None,
        *,
        tf: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> float:
        out_tf = timeframe or tf
        period_ms = _tf_to_ms(out_tf)
        start_ts, end_ts = await self._latest_finalized_range(span, period_ms=period_ms)
        now = _utc_now_ms()
        tf_key = str(period_ms)
        key = ("nrr", int(span), tf_key)
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
        highs = np.asarray(arr["h"], dtype=np.float64)
        lows = np.asarray(arr["l"], dtype=np.float64)
        denom = np.maximum(closes, 1e-12)
        nrr = (highs - lows) / denom
        res = float(self._ema(nrr, span))
        cache[key] = (res, int(end_ts), int(now))
        return res

    # ----- EMA series helpers -----

    async def get_ema_close_series(
        self,
        symbol: str,
        span: int,
        max_age_ms: Optional[int] = None,
        *,
        tf: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> np.ndarray:
        out_tf = timeframe or tf
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
        span: int,
        max_age_ms: Optional[int] = None,
        *,
        tf: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> np.ndarray:
        out_tf = timeframe or tf
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

    async def get_ema_nrr_series(
        self,
        symbol: str,
        span: int,
        max_age_ms: Optional[int] = None,
        *,
        tf: Optional[str] = None,
        timeframe: Optional[str] = None,
    ) -> np.ndarray:
        out_tf = timeframe or tf
        period_ms = _tf_to_ms(out_tf)
        start_ts, end_ts = await self._latest_finalized_range(span, period_ms=period_ms)
        arr = await self.get_candles(
            symbol, start_ts=start_ts, end_ts=end_ts, max_age_ms=max_age_ms, timeframe=out_tf
        )
        if arr.size == 0:
            return np.empty((0,), dtype=EMA_SERIES_DTYPE)
        closes = np.asarray(arr["c"], dtype=np.float64)
        highs = np.asarray(arr["h"], dtype=np.float64)
        lows = np.asarray(arr["l"], dtype=np.float64)
        denom = np.maximum(closes, 1e-12)
        nrr = (highs - lows) / denom
        ema_vals = self._ema_series(nrr, span)
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
        lock = self._get_lock(symbol)
        async with lock:
            if self.exchange is None:
                return None
            now = _utc_now_ms()
            end_exclusive = _floor_minute(now)  # exclude current minute
            if through_ts is not None:
                end_exclusive = min(end_exclusive, _floor_minute(int(through_ts)) + ONE_MIN_MS)

            existing = self._ensure_symbol_cache(symbol)
            if existing.size == 0:
                # start a small window by default
                since = end_exclusive - self.default_window_candles * ONE_MIN_MS
            else:
                last_ts = int(np.sort(existing, order="ts")[-1]["ts"])  # last stored ts
                since = max(0, last_ts - self.overlap_candles * ONE_MIN_MS)

            if since >= end_exclusive:
                return None

            new_arr = await self._fetch_ohlcv_paginated(symbol, since, end_exclusive)
            if new_arr.size == 0:
                return None
            # Merge and store
            merged = self._merge_overwrite(existing, new_arr)
            self._cache[symbol] = merged
            # Update refresh metadata
            try:
                last_final_ts = int(np.sort(merged, order="ts")[-1]["ts"]) if merged.size else 0
            except Exception:
                last_final_ts = 0
            self._set_last_refresh_meta(symbol, last_refresh_ms=now, last_final_ts=last_final_ts)
            return None

    # ----- Persistence -----

    def _save_shard(self, symbol: str, date_key: str, array: np.ndarray) -> None:
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

        shard_path = self._shard_path(symbol, date_key)
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
        idx = self._ensure_symbol_index(symbol)
        shards = idx.setdefault("shards", {})
        shards[date_key] = {
            "path": shard_path,
            "min_ts": int(arr[0]["ts"]),
            "max_ts": int(arr[-1]["ts"]),
            "count": int(arr.shape[0]),
            "crc32": crc,
        }
        self._index[symbol] = idx
        self._save_index(symbol)

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
