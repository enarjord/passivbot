"""
candlestick_manager.py

CandlestickManager: manage 1-minute OHLCV candlesticks for Passivbot.

Example usage:

    import ccxt.async_support as ccxt
    from candlestick_manager import CandlestickManager

    exchange = ccxt.binance({'enableRateLimit': True})
    manager = CandlestickManager(exchange, exchange_name='binance')
    await manager.refresh('BTC/USDT')
    arr = await manager.get_candles('BTC/USDT')

"""
from __future__ import annotations

import asyncio
import os
import io
import json
import time
import math
import logging
import datetime
import zlib
from typing import Optional, List, Dict, Tuple

import numpy as np
import portalocker
import random

# Optional ccxt import (only required if using exchange fetch)
try:
    import ccxt.async_support as ccxt
except Exception:  # pragma: no cover - optional dependency in tests
    ccxt = None

# Logging
LOG_PATH = os.path.join("logs", "candlestick_manager.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
logger = logging.getLogger("candlestick_manager")
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(handler)
file_handler = logging.FileHandler(LOG_PATH)
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# Core dtype for stored candles
CANDLE_DTYPE = np.dtype([
    ("ts", "int64"),  # timestamp in ms
    ("o", "float32"),
    ("h", "float32"),
    ("l", "float32"),
    ("c", "float32"),
    ("bv", "float32"),
])
ONE_MIN_MS = 60_000

_DEFAULTS = {
    "timeframe": "1m",
    "cache_dir": "caches/ohlcv",
    "memory_days": 7,
    "disk_retention_days": 30,
    "overlap_candles": 30,
    "default_window_candles": 100,
}


def _floor_minute(ts_ms: int) -> int:
    return (ts_ms // ONE_MIN_MS) * ONE_MIN_MS


def _ensure_ts_int64(ts) -> int:
    try:
        ts_i = int(ts)
    except Exception:
        raise ValueError("timestamp must be integer milliseconds")
    return int(ts_i)


def _date_str_from_ts(ts_ms: int) -> str:
    return datetime.datetime.utcfromtimestamp(ts_ms / 1000.0).strftime("%Y-%m-%d")


def _sanitize_symbol(symbol: str) -> str:
    return symbol.replace("/", "_")


class CandlestickManager:
    """Manage 1-minute OHLCV candles with caching, gap filling and EMAs.

    Parameters
    ----------
    exchange : ccxt.async_support.Exchange | None
        CCXT exchange instance or None for offline usage (tests).
    exchange_name : str
        Name used in cache paths.
    **kwargs : optional
        See _DEFAULTS for keys.
    """

    def __init__(self, exchange=None, exchange_name: str = "unknown", **kwargs):
        self.exchange = exchange
        self.exchange_name = exchange_name
        cfg = {**_DEFAULTS, **kwargs}
        self.timeframe = cfg["timeframe"]
        self.cache_dir = cfg["cache_dir"]
        self.memory_days = int(cfg["memory_days"])
        self.disk_retention_days = int(cfg["disk_retention_days"])
        self.overlap_candles = int(cfg["overlap_candles"])
        self.default_window_candles = int(cfg["default_window_candles"])
        self.debug = cfg.get("debug", False)

        # in-memory cache: symbol -> numpy structured array sorted by ts
        self._cache: Dict[str, np.ndarray] = {}
        # locks per symbol for asyncio concurrency
        self._locks: Dict[str, asyncio.Lock] = {}
        # index metadata per symbol (on-disk index.json)
        self._index: Dict[str, Dict] = {}
        # ema cache: metric -> symbol -> span -> (last_ts, last_ema)
        self._ema_state: Dict[str, Dict[str, Dict[int, Tuple[int, float]]]] = {
            "close": {},
            "volume": {},
            "nrr": {},
        }

        # ensure base cache dir
        os.makedirs(self.cache_dir, exist_ok=True)

    # ------------------ IO helpers ------------------
    def _symbol_dir(self, symbol: str) -> str:
        s = _sanitize_symbol(symbol)
        return os.path.join(self.cache_dir, self.exchange_name, self.timeframe, s)

    def _shard_path(self, symbol: str, date_str: str) -> str:
        return os.path.join(self._symbol_dir(symbol), f"{date_str}.npy")

    def _index_path(self, symbol: str) -> str:
        return os.path.join(self._symbol_dir(symbol), "index.json")

    def _ensure_symbol_dir(self, symbol: str):
        os.makedirs(self._symbol_dir(symbol), exist_ok=True)

    def _atomic_write_json(self, path: str, obj: dict):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _atomic_write_numpy(self, path: str, arr: np.ndarray):
        tmp = path + ".tmp"
        dirpath = os.path.dirname(path)
        os.makedirs(dirpath, exist_ok=True)
        # write to temp file and fsync
        with open(tmp, "wb") as f:
            np.save(f, arr)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _load_index(self, symbol: str) -> dict:
        path = self._index_path(symbol)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    idx = json.load(f)
                    return idx
            except Exception:
                logger.exception("failed to load index for %s", symbol)
        # default index
        return {
            "schema_version": "1.0",
            "cache_version": "1.0",
            "shards": {},
            "min_ts": None,
            "max_ts": None,
            "known_gaps": [],
            "last_final_ts": None,
            "ema_state": {},
        }

    def _save_index(self, symbol: str):
        path = self._index_path(symbol)
        idx = self._index.get(symbol) or self._load_index(symbol)
        self._ensure_symbol_dir(symbol)
        # use portalocker to lock index while writing
        with portalocker.Lock(path + ".lock", timeout=5):
            self._atomic_write_json(path, idx)

    def _load_shard(self, symbol: str, date_str: str) -> Optional[np.ndarray]:
        path = self._shard_path(symbol, date_str)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as f:
                # no locking for reads
                arr = np.load(f, allow_pickle=False)
            # ensure dtype
            if arr.dtype == CANDLE_DTYPE:
                return arr
            # coerce if needed
            coerced = np.asarray(arr, dtype=CANDLE_DTYPE)
            return coerced
        except Exception:
            logger.exception("failed loading shard %s", path)
            return None

    def _save_shard(self, symbol: str, date_str: str, arr: np.ndarray):
        path = self._shard_path(symbol, date_str)
        self._ensure_symbol_dir(symbol)
        lock_path = path + ".lock"
        with portalocker.Lock(lock_path, timeout=10):
            # write atomic
            self._atomic_write_numpy(path, arr)
            # update index entry
            idx = self._index.setdefault(symbol, self._load_index(symbol))
            idx["shards"][date_str] = {
                "path": os.path.relpath(path, self.cache_dir),
                "min_ts": int(arr[0]["ts"]),
                "max_ts": int(arr[-1]["ts"]),
                "crc32": zlib.crc32(arr.tobytes()) & 0xFFFFFFFF,
            }
            idx["min_ts"] = idx["min_ts"] or int(arr[0]["ts"]) if arr.size else idx["min_ts"]
            idx["max_ts"] = int(arr[-1]["ts"]) if arr.size else idx["max_ts"]
            self._save_index(symbol)

    # ------------------ internal helpers ------------------
    def _get_lock(self, symbol: str) -> asyncio.Lock:
        if symbol not in self._locks:
            self._locks[symbol] = asyncio.Lock()
        return self._locks[symbol]

    def _load_shards_into_memory(self, symbol: str):
        """Load shards from disk (index) into in-memory cache respecting memory_days."""
        idx = self._index.get(symbol) or self._load_index(symbol)
        self._index[symbol] = idx
        shards = idx.get("shards", {})
        if not shards:
            return
        arr_all = np.zeros(0, dtype=CANDLE_DTYPE)
        # load shards sorted by date
        for date_key in sorted(shards.keys()):
            shard_rel = shards[date_key].get("path")
            if shard_rel is None:
                continue
            shard_path = os.path.join(self.cache_dir, shard_rel)
            if not os.path.exists(shard_path):
                continue
            try:
                a = self._load_shard(symbol, date_key)
                if a is not None and a.size:
                    arr_all = self._merge_arrays(arr_all, a)
            except Exception:
                logger.exception("failed loading shard %s", shard_path)
        if arr_all.size:
            # keep only memory_days
            cutoff = int(time.time() * 1000) - int(self.memory_days) * 24 * 3600 * 1000
            mask = arr_all["ts"] >= cutoff
            arr_all = arr_all[mask]
            self._cache[symbol] = arr_all

    def _merge_arrays(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Merge two sorted arrays by ts and keep unique ts (b overwrites a on collisions)."""
        if a is None or a.size == 0:
            return b.copy()
        if b is None or b.size == 0:
            return a.copy()
        # concatenate and unique by ts, keeping last occurrence
        concat = np.concatenate([a, b])
        order = np.argsort(concat["ts"])
        concat = concat[order]
        # unique
        _, idx = np.unique(concat["ts"], return_index=True)
        # keep last occurrence: idx gives first index of each unique; we need last -> compute inverse
        # easiest: iterate from end
        seen = set()
        keep = []
        for i in range(concat.shape[0] - 1, -1, -1):
            ts = int(concat[i]["ts"]) 
            if ts not in seen:
                keep.append(i)
                seen.add(ts)
        keep = sorted(keep)
        return concat[keep]

    # ------------------ gap standardization ------------------
    def standardize_gaps(self, candles: np.ndarray, start_ts: int, end_ts: int, strict=False) -> np.ndarray:
        """Return a new array covering [start_ts, end_ts] with zero-candles inserted where missing.

        Zero candles are not persisted; they are synthesized based on previous close.
        """
        start_ts = _floor_minute(_ensure_ts_int64(start_ts))
        end_ts = _floor_minute(_ensure_ts_int64(end_ts))
        if end_ts < start_ts:
            raise ValueError("end_ts must be >= start_ts")
        # empty timeline
        desired_ts = np.arange(start_ts, end_ts + ONE_MIN_MS, ONE_MIN_MS, dtype=np.int64)
        res = np.zeros(desired_ts.shape[0], dtype=CANDLE_DTYPE)
        res["ts"] = desired_ts
        # if no candles provided, need seed close
        if candles is None or candles.size == 0:
            if strict:
                raise ValueError("no candles available to seed zero-gap filling")
            # leave as NaNs or zeros; set closes to 0
            res["o"] = res["h"] = res["l"] = res["c"] = 0.0
            res["bv"] = 0.0
            return res
        # map existing candles into timeline (avoid duplicates)
        ts_existing = candles["ts"].astype(np.int64)
        # ensure unique by ts in supplied candles taking last occurrence
        _, unique_indices = np.unique(ts_existing, return_index=True)
        # unique_indices is first occurrence; we want last -> compute reverse mapping
        # simplest: iterate and keep last
        last_map = {}
        for i, ts in enumerate(ts_existing):
            last_map[int(ts)] = i
        for ts_val, idx_c in last_map.items():
            pos = (ts_val - start_ts) // ONE_MIN_MS
            if 0 <= pos < desired_ts.shape[0]:
                res["o"][pos] = candles["o"][idx_c]
                res["h"][pos] = candles["h"][idx_c]
                res["l"][pos] = candles["l"][idx_c]
                res["c"][pos] = candles["c"][idx_c]
                res["bv"][pos] = candles["bv"][idx_c]
        # now fill gaps by forward-filling close from previous known candle
        last_close = None
        # we need to find previous close before start_ts if first entries are missing
        # search for previous candle in provided array
        prev_mask = ts_existing < start_ts
        if np.any(prev_mask):
            last_close = float(candles["c"][np.where(prev_mask)[0][-1]])
        else:
            # try to find in index or disk - caller should ensure warmup or strict
            if last_close is None and strict:
                raise ValueError("cannot seed gap filling (strict mode) - no prior close found")
            # else keep last_close None -> will fill zeros
        for i in range(res.shape[0]):
            if (res["c"][i] != 0.0) or (res["o"][i] != 0.0 or res["h"][i] != 0.0 or res["l"][i] != 0.0 or res["bv"][i] != 0.0):
                last_close = float(res["c"][i])
                continue
            # missing: generate zero candle
            if last_close is None:
                # leave zeros
                continue
            res["o"][i] = last_close
            res["h"][i] = last_close
            res["l"][i] = last_close
            res["c"][i] = last_close
            res["bv"][i] = 0.0
        return res

    # ------------------ public access ------------------
    async def get_candles(self, symbol: str, start_ts: Optional[int] = None, end_ts: Optional[int] = None,
                          max_age_ms: Optional[int] = None, strict: bool = False) -> np.ndarray:
        """Return inclusive candles standardized with zero-candle gaps.

        Parameters
        ----------
        symbol : str
        start_ts, end_ts : int ms timestamps
        max_age_ms : int | None
            If 0 -> force refresh; <0 -> ValueError; None -> default 60_000
        strict : bool
            If True, raise when unable to fill gaps.
        """
        now = int(time.time() * 1000)
        if end_ts is None:
            # floor to minute and add one minute to make last minute complete boundary
            end_ts = _floor_minute(now) + ONE_MIN_MS
        else:
            end_ts = _floor_minute(_ensure_ts_int64(end_ts))
        if start_ts is None:
            start_ts = end_ts - ONE_MIN_MS * self.default_window_candles
        else:
            start_ts = _floor_minute(_ensure_ts_int64(start_ts))
        if max_age_ms is None:
            max_age_ms = 60_000
        if max_age_ms < 0:
            raise ValueError("max_age_ms must be >= 0 or None")
        # refresh if requested
        if max_age_ms == 0:
            await self.refresh(symbol, through_ts=end_ts)
        else:
            # check last disk/cache timestamp
            idx = self._index.get(symbol) or self._load_index(symbol)
            last_final = idx.get("last_final_ts")
            if last_final is None or (now - last_final) > max_age_ms:
                await self.refresh(symbol, through_ts=end_ts)
                # reload index after refresh
                idx = self._index.get(symbol) or self._load_index(symbol)
            # If caller requested a start_ts earlier than our on-disk min, backfill from start_ts
            min_ts = idx.get("min_ts")
            if start_ts is not None:
                # request backfill when we don't have data that far back
                if min_ts is None or start_ts < int(min_ts):
                    # start slightly earlier to provide overlap
                    since = max(0, start_ts - self.overlap_candles * ONE_MIN_MS)
                    await self.refresh(symbol, through_ts=end_ts, since_ts=since)
                    # reload index after backfill
                    idx = self._index.get(symbol) or self._load_index(symbol)
        # ensure in-memory array exists
        async with self._get_lock(symbol):
            arr = self._cache.get(symbol)
            if arr is None:
                # attempt load shards into memory limited by memory_days
                self._load_shards_into_memory(symbol)
                arr = self._cache.get(symbol)
            if arr is None:
                # empty
                base = np.zeros(0, dtype=CANDLE_DTYPE)
                res = self.standardize_gaps(base, start_ts, end_ts, strict=strict)
                return res
            # slice from arr for requested range
            ts = arr["ts"].astype(np.int64)
            left = np.searchsorted(ts, start_ts, side="left")
            right = np.searchsorted(ts, end_ts, side="right")
            sub = arr[left:right].copy()
            # standardize gaps across requested range using sub and arr for seed
            res = self.standardize_gaps(sub, start_ts, end_ts, strict=strict)
            return res

    async def get_current_close(self, symbol: str, max_age_ms: Optional[int] = None) -> float:
        """Return latest finalized close (exclude current incomplete minute).

        Default max_age_ms = 10_000 ms.
        """
        if max_age_ms is None:
            max_age_ms = 10_000
        now = int(time.time() * 1000)
        end_ts = _floor_minute(now)  # current minute start is incomplete -> we want last final = floor(now/60000)*60000
        # fetch with overlap to ensure we have final candle
        await self.refresh(symbol, through_ts=end_ts)
        arr = await self.get_candles(symbol, start_ts=end_ts - ONE_MIN_MS * (self.overlap_candles + 1), end_ts=end_ts)
        if arr is None or arr.size == 0:
            raise ValueError("no candles available")
        # last element corresponds to end_ts
        last = arr[arr["ts"] <= end_ts]
        if last.size == 0:
            raise ValueError("no finalized candle available")
        close = float(last[-1]["c"])
        return close

    async def get_latest_ema(self, symbol: str, span: int, metric: str = "close", max_age_ms: Optional[int] = None) -> float:
        """Generic EMA retrieval for metric: 'close', 'volume' or 'nrr'."""
        if metric not in ("close", "volume", "nrr"):
            raise ValueError("invalid metric")
        span = int(span)
        # ensure we have recent data
        if max_age_ms is None:
            max_age_ms = 60_000
        now = int(time.time() * 1000)
        end_ts = _floor_minute(now) - ONE_MIN_MS  # last completed minute
        # need last `span` completed candles
        start_ts = end_ts - ONE_MIN_MS * (span - 1)
        arr = await self.get_candles(symbol, start_ts=start_ts, end_ts=end_ts, max_age_ms=max_age_ms)
        if arr is None or arr.size == 0:
            raise ValueError("no candles for EMA")
        # exclude incomplete (if any) - our get_candles returns standardized up to end_ts inclusive (end_ts is final)
        vals = None
        if metric == "close":
            vals = arr["c"].astype(np.float64)
        elif metric == "volume":
            vals = (arr["c"].astype(np.float64) * arr["bv"].astype(np.float64))
        else:  # nrr
            close = arr["c"].astype(np.float64)
            high = arr["h"].astype(np.float64)
            low = arr["l"].astype(np.float64)
            close_safe = np.maximum(close, 1e-12)
            vals = (high - low) / close_safe
        # compute EMA: seed = first value
        alpha = 2.0 / (span + 1.0)
        # vals may be longer/shorter than span depending on availability; use all available
        ema = float(vals[0])
        for x in vals[1:]:
            ema = alpha * float(x) + (1.0 - alpha) * ema
        # store state
        self._ema_state.setdefault(metric, {}).setdefault(symbol, {})[span] = (int(arr[-1]["ts"]), float(ema))
        return ema

    async def get_latest_ema_close(self, symbol: str, span: int, max_age_ms: Optional[int] = None) -> float:
        return await self.get_latest_ema(symbol, span, metric="close", max_age_ms=max_age_ms)

    async def get_latest_ema_volume(self, symbol: str, span: int, max_age_ms: Optional[int] = None) -> float:
        return await self.get_latest_ema(symbol, span, metric="volume", max_age_ms=max_age_ms)

    async def get_latest_ema_nrr(self, symbol: str, span: int, max_age_ms: Optional[int] = None) -> float:
        return await self.get_latest_ema(symbol, span, metric="nrr", max_age_ms=max_age_ms)

    # ------------------ fetch & refresh ------------------
    async def refresh(self, symbol: str, through_ts: Optional[int] = None, since_ts: Optional[int] = None):
        """Fetch data from exchange up to through_ts (inclusive final minute).

        Parameters
        ----------
        symbol: str
        through_ts: Optional[int]
            inclusive final minute to fetch up to (floored to minute). If None uses now.
        since_ts: Optional[int]
            If provided, start fetching from this timestamp (floored to minute). Useful for backfill.

        If exchange is None, this becomes a no-op (useful for tests).
        """
        if self.exchange is None:
            logger.debug("no exchange configured; refresh is a no-op")
            return
        # validate symbol
        if hasattr(self.exchange, "load_markets"):
            await self.exchange.load_markets()
        if symbol not in getattr(self.exchange, "symbols", [symbol]):
            # let exchange validate
            pass
        now = int(time.time() * 1000)
        if through_ts is None:
            through_ts = _floor_minute(now)
        through_ts = _floor_minute(_ensure_ts_int64(through_ts))
        # determine since - start from latest known ts minus overlap
        async with self._get_lock(symbol):
            idx = self._index.get(symbol) or self._load_index(symbol)
            self._index[symbol] = idx
            last_max = idx.get("max_ts")
            if since_ts is not None:
                # user requested explicit since for backfill; floor to minute
                since = _floor_minute(_ensure_ts_int64(since_ts))
            else:
                if last_max is None:
                    since = None
                else:
                    since = int(last_max) - self.overlap_candles * ONE_MIN_MS
            # paginate
            limit = 1000
            all_new = []
            backoff = 0.5
            while True:
                try:
                    params = {}
                    if since is None:
                        fetch_since = None
                    else:
                        fetch_since = max(0, since)
                    logger.info("fetching ohlcv for %s since=%s limit=%s", symbol, fetch_since, limit)
                    ohlcvs = await self.exchange.fetch_ohlcv(symbol, timeframe=self.timeframe, since=fetch_since, limit=limit, params=params)
                    if not ohlcvs:
                        break
                    # convert to our dtype and filter
                    batch = []
                    for row in ohlcvs:
                        ts = _floor_minute(int(row[0]))
                        o = float(row[1]); h = float(row[2]); l = float(row[3]); c = float(row[4]); bv = float(row[5]) if len(row) > 5 else 0.0
                        batch.append((ts, o, h, l, c, bv))
                    arr = np.array(batch, dtype=CANDLE_DTYPE)
                    # append
                    all_new.append(arr)
                    # set since for next page
                    last_ts = int(arr[-1]["ts"])
                    since = last_ts + ONE_MIN_MS
                    # stop if we've fetched through through_ts
                    if last_ts >= through_ts:
                        break
                    # if less than limit returned, we can stop
                    if arr.shape[0] < limit:
                        break
                    # small delay to respect rate limits
                    await asyncio.sleep(0.01)
                except Exception as e:
                    logger.exception("error fetching ohlcv for %s: %s", symbol, e)
                    await asyncio.sleep(backoff + (0.1 * (random.random() if 'random' in globals() else 0)))
                    backoff = min(backoff * 2, 30)
                    continue
            # merge all_new into single array
            if all_new:
                merged = all_new[0]
                for a in all_new[1:]:
                    merged = self._merge_arrays(merged, a)
                # merge into memory cache and persist shards
                existing = self._cache.get(symbol)
                merged_all = self._merge_arrays(existing if existing is not None else np.zeros(0, dtype=CANDLE_DTYPE), merged)
                self._cache[symbol] = merged_all
                # persist by shard (day)
                # group by date
                groups: Dict[str, List[np.ndarray]] = {}
                for rec in merged_all:
                    d = _date_str_from_ts(int(rec["ts"]))
                    groups.setdefault(d, []).append(rec)
                for d, items in groups.items():
                    arr = np.array(items, dtype=CANDLE_DTYPE)
                    self._save_shard(symbol, d, arr)
                # update index
                idx = self._index.setdefault(symbol, self._load_index(symbol))
                idx["min_ts"] = int(merged_all[0]["ts"]) if merged_all.size else idx.get("min_ts")
                idx["max_ts"] = int(merged_all[-1]["ts"]) if merged_all.size else idx.get("max_ts")
                idx["last_final_ts"] = int(_floor_minute(int(time.time() * 1000)))
                self._save_index(symbol)

    # ------------------ warmup ------------------
    async def warmup_since(self, symbols: List[str], since_ts: int):
        # warmup since -> trigger refresh with since_ts to backfill older history
        tasks = []
        for symbol in symbols:
            try:
                coro = self.refresh(symbol, through_ts=None, since_ts=since_ts)
            except TypeError:
                # fallback for older signature or monkeypatched functions that don't accept since_ts
                coro = self.refresh(symbol, through_ts=None)
            tasks.append(coro)
        await asyncio.gather(*tasks)

    # ------------------ utilities ------------------
    def close(self):
        # flush indexes
        for symbol in list(self._index.keys()):
            try:
                self._save_index(symbol)
            except Exception:
                logger.exception("error saving index for %s", symbol)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# end of module
