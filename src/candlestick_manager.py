import asyncio
import contextlib
import datetime as dt
import gzip
import io
import json
import logging
import os
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

# CCXT async support (REST-only)
import ccxt.async_support as ccxt  # type: ignore

try:
    # Lightweight cross-process file locking to avoid race conditions
    from filelock import FileLock
except Exception as e:  # pragma: no cover
    FileLock = None  # Will fallback to naive single-process async locks


TimestampLike = Union[int, float, str]
CandlesArray = np.ndarray  # shape (N, 6): [ts, open, high, low, close, quote_volume]


ONE_MIN_MS = 60_000
ONE_DAY_MS = 86_400_000


def _utc_now_ms() -> int:
    return int(dt.datetime.utcnow().timestamp() * 1000)


def _to_ms(ts: TimestampLike) -> int:
    if isinstance(ts, (int, float)):
        return int(ts)
    if isinstance(ts, str):
        # Accept "YYYY-MM-DD" or "YYYY-MM-DD HH:MM[:SS]"
        try:
            if len(ts) <= 10:
                d = dt.datetime.strptime(ts, "%Y-%m-%d")
            else:
                # Try ISO-like formats
                try:
                    d = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    d = dt.datetime.strptime(ts, "%Y-%m-%d %H:%M")
            # Interpret as UTC
            return int(d.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
        except Exception as e:
            raise ValueError(f"Unrecognized date string format: {ts}") from e
    raise TypeError(f"Unsupported timestamp type: {type(ts)}")


def _floor_minute(ms: int) -> int:
    return (ms // ONE_MIN_MS) * ONE_MIN_MS


def _ceil_minute(ms: int) -> int:
    return _floor_minute(ms + ONE_MIN_MS - 1)


def _day_start(ms: int) -> int:
    dt_utc = dt.datetime.utcfromtimestamp(ms / 1000.0)
    return int(
        dt.datetime(dt_utc.year, dt_utc.month, dt_utc.day, tzinfo=dt.timezone.utc).timestamp() * 1000
    )


def _day_end_exclusive(ms: int) -> int:
    return _day_start(ms) + ONE_DAY_MS


def _ym_str(ms: int) -> str:
    d = dt.datetime.utcfromtimestamp(ms / 1000.0)
    return f"{d.year:04d}-{d.month:02d}"


def _ymd_str(ms: int) -> str:
    d = dt.datetime.utcfromtimestamp(ms / 1000.0)
    return f"{d.year:04d}-{d.month:02d}-{d.day:02d}"


@dataclass(frozen=True)
class CacheLayout:
    """
    Data is stored under:
      {data_root}/{exchange}/{symbol}/{YYYY-MM}/{YYYY-MM-DD}.npy.gz

    NPY GZip-compressed to balance disk usage and speed.
    """

    data_root: pathlib.Path

    def day_dir(self, exchange: str, symbol: str, day_ms: int) -> pathlib.Path:
        return self.data_root / exchange / symbol / _ym_str(day_ms)

    def day_path(self, exchange: str, symbol: str, day_ms: int) -> pathlib.Path:
        return self.day_dir(exchange, symbol, day_ms) / f"{_ymd_str(day_ms)}.npy.gz"

    def metadata_path(self, exchange: str, symbol: str) -> pathlib.Path:
        return self.data_root / exchange / symbol / "metadata.json"

    def lock_path(self, target: pathlib.Path) -> pathlib.Path:
        return target.with_suffix(target.suffix + ".lock")


class CandlestickManager:
    """
    Async 1m candlestick manager for crypto perpetuals using CCXT (REST-only).

    Features:
    - Lazy fetch: only missing ranges are downloaded.
    - Disk cache per day, compressed: exchange/symbol/YYYY-MM/YYYY-MM-DD.npy.gz
    - Cross-process safe via file locks (requires 'filelock' package).
    - Candle integrity verification (dedup, sort, minute alignment, gap repair attempts).
    - Normalizes volume to quote volume (base_volume * close).
    - Returns numpy array [ts, open, high, low, close, quote_volume] for requested range.
    - Avoids incomplete current-minute candle.

    Typical usage:
      async with CandlestickManager("binanceusdm", "BTC/USDT:USDT") as cm:
          candles = await cm.get("2024-01-01", "2024-06-30")
    """

    def __init__(
        self,
        exchange: str,
        symbol: str,
        *,
        data_root: Union[str, os.PathLike] = "data/candles",
        ccxt_config: Optional[dict] = None,
        max_requests_in_flight: int = 5,
        request_limit: int = 1000,
        logger: Optional[logging.Logger] = None,
    ):
        self.exchange_name = exchange
        self.symbol = symbol
        self.layout = CacheLayout(pathlib.Path(data_root))
        self.ccxt_config = dict(ccxt_config or {})
        self.ccxt_config.setdefault("enableRateLimit", True)
        self._ex = None  # type: Optional[ccxt.Exchange]
        self._sem = asyncio.Semaphore(max_requests_in_flight)
        self._request_limit = max(100, min(10_000, int(request_limit)))
        self._logger = logger or logging.getLogger(__name__)
        self._local_lock = asyncio.Lock()  # for metadata operations when filelock missing

    async def __aenter__(self):
        self._ex = getattr(ccxt, self.exchange_name)({**self.ccxt_config})
        # Some exchanges need options set; user can pass via ccxt_config
        return self

    async def __aexit__(self, exc_type, exc, tb):
        with contextlib.suppress(Exception):
            if self._ex:
                await self._ex.close()
        self._ex = None

    # --------------- Public API ---------------

    async def get(
        self,
        start: Optional[TimestampLike],
        end: Optional[TimestampLike],
    ) -> CandlesArray:
        """
        Fetch candles in [start, end] (inclusive/exclusive semantics aligned to minutes).
        If start is None, it will attempt to discover earliest available candle from the exchange.
        If end is None, it uses "now - 1 minute" as the exclusive upper bound (avoids incomplete).

        Returns numpy array (N, 6) with columns:
          ts, open, high, low, close, quote_volume
        """
        if self._ex is None:
            raise RuntimeError("CandlestickManager must be used within an async context manager")

        now_ms = _utc_now_ms()
        safe_end_exclusive = _floor_minute(now_ms)  # exclude current in-progress minute

        start_ms = await self._resolve_start_ms(start)
        end_ms = _to_ms(end) if end is not None else safe_end_exclusive

        if end_ms <= start_ms:
            return np.empty((0, 6), dtype=np.float64)

        # Align bounds to minute
        start_ms = _floor_minute(start_ms)
        end_ms = _floor_minute(end_ms)

        # Gather day boundaries
        day_starts = list(self._iter_day_starts(start_ms, end_ms))

        # Load or fetch each day, merge, then slice to requested exact range
        day_arrays: List[np.ndarray] = []
        for ds in day_starts:
            de = min(_day_end_exclusive(ds), end_ms)
            arr = await self._get_day(ds, de, is_last_day=(de == end_ms))
            if arr.size:
                day_arrays.append(arr)

        if not day_arrays:
            return np.empty((0, 6), dtype=np.float64)

        all_arr = self._merge_and_verify(day_arrays)
        # Slice to requested range [start_ms, end_ms)
        mask = (all_arr[:, 0] >= start_ms) & (all_arr[:, 0] < end_ms)
        result = all_arr[mask]
        return result

    # --------------- Internal helpers ---------------

    async def _resolve_start_ms(self, start: Optional[TimestampLike]) -> int:
        """
        Resolve the effective start timestamp:
        - If start is None, use earliest available (discover and cache if missing).
        - If start is provided but earlier than earliest, clamp to earliest.
        - If no metadata and cache is empty, discover earliest and cache it.
        """
        requested = _to_ms(start) if start is not None else None

        # Try reading cached earliest
        meta = await self._load_metadata()
        earliest = int(meta["earliest_ts"]) if meta and "earliest_ts" in meta else None

        # If we don't know earliest yet and have no cache, discover and persist it
        if earliest is None and not self._has_any_cached_data():
            discovered = await self._discover_earliest_ts()
            earliest = int(discovered)
            await self._save_metadata({"earliest_ts": earliest})

        if requested is None:
            if earliest is not None:
                return earliest
            # As a final fallback, discover, persist, and return
            discovered = await self._discover_earliest_ts()
            earliest = int(discovered)
            await self._save_metadata({"earliest_ts": earliest})
            return earliest

        # Clamp requested to earliest if known
        if earliest is not None and requested < earliest:
            return earliest
        return requested

    def _iter_day_starts(self, start_ms: int, end_ms: int):
        cur = _day_start(start_ms)
        end_day = _day_start(end_ms - 1)
        while cur <= end_day:
            yield cur
            cur += ONE_DAY_MS

    async def _get_day(
        self, day_start_ms: int, day_end_exclusive_ms: int, is_last_day: bool
    ) -> CandlesArray:
        """
        Return candles within [day_start_ms, day_end_exclusive_ms), using cache if possible.
        For current/last day, cache may be partial; we will extend and update as needed.
        """
        path = self.layout.day_path(self.exchange_name, self.symbol, day_start_ms)
        # Attempt cache load
        cached = await self._load_day(path)
        need_fetch_ranges: List[Tuple[int, int]] = []

        expected_end = min(day_start_ms + ONE_DAY_MS, day_end_exclusive_ms)
        if cached is None or cached.size == 0:
            # Entire day missing: fetch all
            need_fetch_ranges.append((day_start_ms, expected_end))
            cached = np.empty((0, 6), dtype=np.float64)
        else:
            # Verify integrity and determine missing ranges
            cached = self._verify_and_normalize(cached, day_start_ms, expected_end)
            # Determine coverage
            covered = set(int(ts) for ts in cached[:, 0].tolist())
            # For non-last (full) days, expect 1440 minutes unless it's the very first day of data
            minute_ts = list(range(day_start_ms, expected_end, ONE_MIN_MS))
            missing_ts = [ts for ts in minute_ts if ts not in covered]
            if missing_ts:
                # Merge consecutive missing minutes into ranges
                rng_start = missing_ts[0]
                prev = missing_ts[0]
                for ts in missing_ts[1:]:
                    if ts == prev + ONE_MIN_MS:
                        prev = ts
                        continue
                    need_fetch_ranges.append((rng_start, prev + ONE_MIN_MS))
                    rng_start = ts
                    prev = ts
                need_fetch_ranges.append((rng_start, prev + ONE_MIN_MS))

        fetched_arrays: List[np.ndarray] = []
        for rs, re in need_fetch_ranges:
            chunk = await self._fetch_range(rs, re)
            if chunk.size:
                fetched_arrays.append(chunk)

        if fetched_arrays:
            merged = self._merge_and_verify([cached, *fetched_arrays])
            # After merge, persist for day (atomic write)
            await self._save_day(path, merged)
            return self._restrict_to_day(merged, day_start_ms, day_end_exclusive_ms)
        else:
            return self._restrict_to_day(cached, day_start_ms, day_end_exclusive_ms)

    def _restrict_to_day(self, arr: CandlesArray, start_ms: int, end_ms: int) -> CandlesArray:
        if arr.size == 0:
            return arr
        mask = (arr[:, 0] >= start_ms) & (arr[:, 0] < end_ms)
        return arr[mask]

    async def _fetch_range(self, start_ms: int, end_exclusive_ms: int) -> CandlesArray:
        """
        Fetch candles for [start_ms, end_exclusive_ms) using paginated CCXT REST calls.
        Ensures we never include the in-progress current minute.
        """
        assert self._ex is not None
        now_floor = _floor_minute(_utc_now_ms())
        hard_end = min(end_exclusive_ms, now_floor)
        if start_ms >= hard_end:
            return np.empty((0, 6), dtype=np.float64)

        out: List[List[float]] = []
        since = start_ms
        max_tries_per_call = 3

        while since < hard_end:
            # Limit covers up to ~limit minutes per request
            limit = self._request_limit
            # Prevent overshooting too far; some exchanges cap 1000-1500 anyway
            expected_last = since + limit * ONE_MIN_MS
            until = min(expected_last, hard_end)

            for attempt in range(max_tries_per_call):
                try:
                    async with self._sem:
                        ohlcv = await self._ex.fetch_ohlcv(
                            self.symbol, timeframe="1m", since=since, limit=limit
                        )
                    break
                except ccxt.RateLimitExceeded as e:
                    await asyncio.sleep(0.5 * (attempt + 1))
                except ccxt.NetworkError as e:
                    await asyncio.sleep(0.5 * (attempt + 1))
                except Exception as e:
                    # Log and retry a couple of times
                    self._logger.warning(
                        f"fetch_ohlcv error {type(e).__name__}: {e} (attempt {attempt+1}/{max_tries_per_call})"
                    )
                    await asyncio.sleep(0.5 * (attempt + 1))
            else:
                # All attempts failed, break to avoid tight loop
                self._logger.error(
                    f"Failed to fetch OHLCV after {max_tries_per_call} attempts; since={since}"
                )
                break

            if not ohlcv:
                # No data, advance cautiously to avoid infinite loops
                since = until
                continue

            # Transform to quote-volume and ensure bounds
            batch = []
            for row in ohlcv:
                # CCXT OHLCV: [timestamp, open, high, low, close, volume_base]
                ts, o, h, l, c, v_base = row
                ts = int(ts)
                if ts < start_ms or ts >= hard_end:
                    continue
                if ts >= until:
                    # Sometimes exchanges may include more than asked
                    continue
                # Ensure minute alignment
                if ts % ONE_MIN_MS != 0:
                    continue
                quote_vol = float(v_base) * float(c)
                batch.append([float(ts), float(o), float(h), float(l), float(c), float(quote_vol)])

            if batch:
                out.extend(batch)

            # Advance 'since' to next minute after the last received timestamp
            last_ts = max(int(r[0]) for r in batch) if batch else since
            # Prevent stuck loop if exchange returns the same earliest repeatedly
            next_since = max(since + ONE_MIN_MS, last_ts + ONE_MIN_MS)
            if next_since <= since:
                break
            since = next_since

        if not out:
            return np.empty((0, 6), dtype=np.float64)
        arr = np.array(out, dtype=np.float64)
        return self._verify_and_normalize(arr, start_ms, hard_end)

    def _verify_and_normalize(
        self, arr: CandlesArray, start_ms: int, end_exclusive_ms: int
    ) -> CandlesArray:
        """
        - Deduplicate by timestamp
        - Sort ascending
        - Drop any candle >= end_exclusive_ms
        - Drop any candle not aligned to minute
        - Convert to float64 with columns [ts, o, h, l, c, quote_vol]
        - Remove any candle whose OHLC are NaN or zero-length anomalies
        """
        if arr.size == 0:
            return arr

        # Ensure ndarray float64
        arr = np.array(arr, dtype=np.float64)

        # Filter alignment and range
        m_align = (
            (arr[:, 0] % ONE_MIN_MS == 0) & (arr[:, 0] >= start_ms) & (arr[:, 0] < end_exclusive_ms)
        )
        arr = arr[m_align]

        if arr.size == 0:
            return arr

        # Deduplicate by timestamp: keep the last occurrence
        # Use argsort then unique
        order = np.argsort(arr[:, 0], kind="mergesort")
        arr = arr[order]
        _, unique_idx = np.unique(arr[:, 0], return_index=True)
        arr = arr[unique_idx]

        # Sanity: remove rows with NaNs or invalid highs/lows
        valid = (
            np.isfinite(arr).all(axis=1)
            & (arr[:, 2] >= arr[:, 3])  # high >= low
            & (arr[:, 1] <= arr[:, 2])  # open <= high
            & (arr[:, 1] >= arr[:, 3])  # open >= low
            & (arr[:, 4] <= arr[:, 2])  # close <= high
            & (arr[:, 4] >= arr[:, 3])  # close >= low
        )
        arr = arr[valid]

        # Final sort by ts
        arr = arr[np.argsort(arr[:, 0], kind="mergesort")]
        return arr

    def _merge_and_verify(self, arrays: List[CandlesArray]) -> CandlesArray:
        arrays = [a for a in arrays if a is not None and a.size > 0]
        if not arrays:
            return np.empty((0, 6), dtype=np.float64)
        arr = np.concatenate(arrays, axis=0) if len(arrays) > 1 else arrays[0]
        start_ms = int(arr[:, 0].min())
        end_ms = int(arr[:, 0].max() + ONE_MIN_MS)
        return self._verify_and_normalize(arr, start_ms, end_ms)

    # --------------- Cache I/O (with cross-process safety) ---------------

    async def _load_day(self, path: pathlib.Path) -> Optional[CandlesArray]:
        if not path.exists():
            return None
        try:
            # Read without locking (readers don't need exclusive access if writers are atomic)
            with gzip.open(path, "rb") as f:
                buf = io.BytesIO(f.read())
            arr = np.load(buf, allow_pickle=False)
            # Backward compatibility: ensure shape (N,6)
            if arr.ndim != 2 or arr.shape[1] != 6:
                self._logger.warning(f"Unexpected array shape in cache {path}: {arr.shape}")
                return None
            return arr.astype(np.float64, copy=False)
        except Exception as e:
            self._logger.warning(f"Failed to load cache file {path}: {e}")
            return None

    async def _save_day(self, path: pathlib.Path, arr: CandlesArray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_ctx = self._acquire_lock(self.layout.lock_path(path))
        async with lock_ctx:
            tmp_path = path.with_suffix(path.suffix + ".tmp")
            try:
                # Atomic write: write tmp then replace
                with gzip.open(tmp_path, "wb") as gz:
                    np.save(gz, arr, allow_pickle=False)
                os.replace(tmp_path, path)
            finally:
                with contextlib.suppress(Exception):
                    if tmp_path.exists():
                        tmp_path.unlink()

    async def _load_metadata(self) -> Optional[dict]:
        meta_path = self.layout.metadata_path(self.exchange_name, self.symbol)
        if not meta_path.exists():
            return None
        lock_ctx = self._acquire_lock(self.layout.lock_path(meta_path))
        async with lock_ctx:
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return None

    async def _save_metadata(self, data: dict) -> None:
        meta_path = self.layout.metadata_path(self.exchange_name, self.symbol)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        lock_ctx = self._acquire_lock(self.layout.lock_path(meta_path))
        async with lock_ctx:
            tmp_path = meta_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, separators=(",", ":"), sort_keys=True)
            os.replace(tmp_path, meta_path)

    def _has_any_cached_data(self) -> bool:
        """
        Returns True if there is any cached candle file for this exchange+symbol.
        """
        root = self.layout.data_root / self.exchange_name / self.symbol
        if not root.exists():
            return False
        try:
            next(root.rglob("*.npy.gz"))
            return True
        except StopIteration:
            return False

    # --------------- Earliest timestamp discovery ---------------

    async def _discover_earliest_ts(self) -> int:
        """
        Heuristic binary search to discover earliest candle.
        Tries to be gentle with requests; capped iterations.
        """
        assert self._ex is not None
        now = _utc_now_ms()
        lo = 0
        hi = _floor_minute(now) - ONE_MIN_MS  # at least previous minute
        best = None

        for _ in range(24):  # up to 24 iterations (~log2 days range)
            mid = max(lo, min(hi, lo + (hi - lo) // 2))
            got = await self._fetch_first_at_or_after(mid)
            if got is None:
                # No data at/after mid; move right
                lo = mid + ONE_MIN_MS
            else:
                best = got
                # Try to find even earlier
                hi = max(0, got - ONE_MIN_MS)
            if hi < lo:
                break

        if best is None:
            # As a fallback, try asking at since=0 directly
            got = await self._fetch_first_at_or_after(0)
            if got is None:
                raise RuntimeError("Could not discover earliest candle (exchange returned no data)")
            best = got
        return int(best)

    async def _fetch_first_at_or_after(self, since_ms: int) -> Optional[int]:
        assert self._ex is not None
        try:
            async with self._sem:
                ohlcv = await self._ex.fetch_ohlcv(
                    self.symbol, timeframe="1m", since=since_ms, limit=1
                )
        except Exception:
            return None
        if not ohlcv:
            return None
        ts = int(ohlcv[0][0])
        if ts < since_ms:
            return None
        return _floor_minute(ts)

    # --------------- Lock adapter (async wrapper around filelock) ---------------

    class _AsyncFileLock:
        def __init__(self, path: pathlib.Path, local_lock: asyncio.Lock):
            self.path = path
            self.local_lock = local_lock
            self._flock = None

        async def __aenter__(self):
            if FileLock is None:
                # Fallback: single-process safety only
                await self.local_lock.acquire()
                return self
            # Use filelock for cross-process safety
            self._flock = FileLock(str(self.path))
            # Retry a few times to avoid deadlocks
            for _ in range(60):
                try:
                    self._flock.acquire(timeout=1.0)
                    return self
                except Exception:
                    await asyncio.sleep(0.1)
            # Final attempt blocking
            self._flock.acquire()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            if FileLock is None:
                with contextlib.suppress(Exception):
                    self.local_lock.release()
            else:
                with contextlib.suppress(Exception):
                    if self._flock:
                        self._flock.release()
            return False

    def _acquire_lock(self, path: pathlib.Path) -> "_AsyncFileLock":
        path.parent.mkdir(parents=True, exist_ok=True)
        return CandlestickManager._AsyncFileLock(path, self._local_lock)
