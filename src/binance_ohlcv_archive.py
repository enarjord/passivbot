from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
from io import BytesIO
import logging
from typing import Literal, Optional
import zipfile

import aiohttp
import numpy as np
import pandas as pd


BINANCE_ARCHIVE_BASE_URL = "https://data.binance.vision/data/futures/um"
BINANCE_MONTHLY_MIN_MISSING_CANDLES = 7 * 24 * 60
BINANCE_DAILY_MIN_MISSING_CANDLES = 1000
BINANCE_DAILY_LAG_DAYS = 2
BINANCE_MONTHLY_PUBLICATION_BUFFER_HOURS = 24
BINANCE_ARCHIVE_MAX_CONCURRENCY = 6


ArchiveKind = Literal["monthly", "daily"]


class BinanceArchiveError(RuntimeError):
    pass


class BinanceArchiveIntegrityError(BinanceArchiveError):
    pass


@dataclass(frozen=True)
class BinanceArchiveRequest:
    kind: ArchiveKind
    symbol_code: str
    period_key: str
    start_ts: int
    end_ts: int
    missing_candles: int

    @property
    def filename(self) -> str:
        return f"{self.symbol_code}-1m-{self.period_key}.zip"

    @property
    def url(self) -> str:
        return (
            f"{BINANCE_ARCHIVE_BASE_URL}/{self.kind}/klines/"
            f"{self.symbol_code}/1m/{self.filename}"
        )


@dataclass(frozen=True)
class BinanceArchiveResult:
    request: BinanceArchiveRequest
    status: Literal["ok", "not_found", "error"]
    frame: Optional[pd.DataFrame] = None
    error: Optional[str] = None


def _utc_datetime(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)


def _datetime_to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _month_bounds(year: int, month: int) -> tuple[int, int]:
    start = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        next_month = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        next_month = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    return _datetime_to_ms(start), _datetime_to_ms(next_month) - 60_000


def _day_bounds(day: datetime) -> tuple[int, int]:
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    return _datetime_to_ms(start), _datetime_to_ms(start + timedelta(days=1)) - 60_000


def first_monday_after_month(year: int, month: int) -> datetime:
    if month == 12:
        candidate = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        candidate = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    return candidate + timedelta(days=(7 - candidate.weekday()) % 7)


def monthly_archive_eligible(
    year: int,
    month: int,
    *,
    now_ms: int,
    publication_buffer_hours: int = BINANCE_MONTHLY_PUBLICATION_BUFFER_HOURS,
) -> bool:
    now = _utc_datetime(now_ms)
    if (year, month) >= (now.year, now.month):
        return False
    available_at = first_monday_after_month(year, month) + timedelta(
        hours=max(0, int(publication_buffer_hours))
    )
    return now >= available_at


def daily_archive_eligible(
    year: int,
    month: int,
    day: int,
    *,
    now_ms: int,
    lag_days: int = BINANCE_DAILY_LAG_DAYS,
) -> bool:
    day_start = datetime(year, month, day, tzinfo=timezone.utc)
    today_start = datetime.combine(
        _utc_datetime(now_ms).date(), datetime.min.time(), tzinfo=timezone.utc
    )
    return day_start + timedelta(days=1 + max(0, int(lag_days))) <= today_start


def plan_binance_archive_requests(
    timestamps: np.ndarray,
    valid: np.ndarray,
    *,
    symbol_code: str,
    now_ms: int,
    monthly_min_missing_candles: int = BINANCE_MONTHLY_MIN_MISSING_CANDLES,
    daily_min_missing_candles: int = BINANCE_DAILY_MIN_MISSING_CANDLES,
) -> tuple[list[BinanceArchiveRequest], list[BinanceArchiveRequest]]:
    ts = np.asarray(timestamps, dtype=np.int64)
    mask = np.asarray(valid, dtype=np.bool_)
    if ts.ndim != 1 or mask.ndim != 1 or len(ts) != len(mask):
        raise ValueError("timestamps and valid must be matching one-dimensional arrays")
    if len(ts) == 0 or mask.all():
        return [], []

    missing_ts = ts[~mask]
    missing_datetimes = missing_ts.astype("datetime64[ms]")
    month_keys = missing_datetimes.astype("datetime64[M]").astype(np.int64)
    unique_months, monthly_counts = np.unique(month_keys, return_counts=True)

    monthly: list[BinanceArchiveRequest] = []
    monthly_periods: set[int] = set()
    for month_key, count in zip(unique_months.tolist(), monthly_counts.tolist()):
        year_offset, month_offset = divmod(int(month_key), 12)
        year = 1970 + year_offset
        month = month_offset + 1
        if int(count) <= int(monthly_min_missing_candles):
            continue
        if not monthly_archive_eligible(year, month, now_ms=now_ms):
            continue
        start_ts, end_ts = _month_bounds(year, month)
        monthly.append(
            BinanceArchiveRequest(
                kind="monthly",
                symbol_code=symbol_code,
                period_key=f"{year:04d}-{month:02d}",
                start_ts=start_ts,
                end_ts=end_ts,
                missing_candles=int(count),
            )
        )
        monthly_periods.add(int(month_key))

    daily_candidate_mask = ~np.isin(month_keys, np.fromiter(monthly_periods, dtype=np.int64))
    daily_keys = missing_datetimes[daily_candidate_mask].astype("datetime64[D]").astype(np.int64)
    unique_days, daily_counts = np.unique(daily_keys, return_counts=True)

    daily: list[BinanceArchiveRequest] = []
    for day_key, count in zip(unique_days.tolist(), daily_counts.tolist()):
        if int(count) <= int(daily_min_missing_candles):
            continue
        day_dt = datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(days=int(day_key))
        year, month, day = day_dt.year, day_dt.month, day_dt.day
        if not daily_archive_eligible(year, month, day, now_ms=now_ms):
            continue
        start_ts, end_ts = _day_bounds(datetime(year, month, day, tzinfo=timezone.utc))
        daily.append(
            BinanceArchiveRequest(
                kind="daily",
                symbol_code=symbol_code,
                period_key=f"{year:04d}-{month:02d}-{day:02d}",
                start_ts=start_ts,
                end_ts=end_ts,
                missing_candles=int(count),
            )
        )
    return monthly, daily


def _parse_checksum(payload: bytes, filename: str) -> str:
    try:
        text = payload.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise BinanceArchiveIntegrityError("checksum sidecar is not UTF-8") from exc
    if not text:
        raise BinanceArchiveIntegrityError("checksum sidecar is empty")
    fields = text.split()
    digest = fields[0].lower() if fields else ""
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise BinanceArchiveIntegrityError("checksum sidecar has an invalid SHA-256 digest")
    if len(fields) > 1 and fields[-1].lstrip("*") != filename:
        raise BinanceArchiveIntegrityError(
            f"checksum sidecar names {fields[-1]!r}, expected {filename!r}"
        )
    return digest


def _parse_archive_zip(raw: bytes, request: BinanceArchiveRequest) -> pd.DataFrame:
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    try:
        with zipfile.ZipFile(BytesIO(raw), "r") as archive:
            members = [name for name in archive.namelist() if not name.endswith("/")]
            if not members:
                raise BinanceArchiveIntegrityError("archive contains no files")
            frames = []
            for name in members:
                with archive.open(name) as handle:
                    frame = pd.read_csv(handle, header=None)
                if frame.shape[1] < len(columns):
                    raise BinanceArchiveIntegrityError(
                        f"archive member {name!r} has only {frame.shape[1]} columns"
                    )
                frame = frame.iloc[:, : len(columns)].copy()
                frame.columns = columns
                frames.append(frame)
    except zipfile.BadZipFile as exc:
        raise BinanceArchiveIntegrityError("archive is not a valid ZIP file") from exc

    frame = pd.concat(frames, ignore_index=True)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=columns).reset_index(drop=True)
    if frame.empty:
        raise BinanceArchiveIntegrityError("archive contains no valid OHLCV rows")

    frame["timestamp"] = frame["timestamp"].astype(np.int64)
    frame = frame[
        (frame["timestamp"] >= int(request.start_ts))
        & (frame["timestamp"] <= int(request.end_ts))
    ]
    if frame.empty:
        raise BinanceArchiveIntegrityError("archive contains no rows in the requested period")
    if bool((frame["timestamp"] % 60_000 != 0).any()):
        raise BinanceArchiveIntegrityError("archive contains unaligned 1m timestamps")

    value_columns = ["open", "high", "low", "close", "volume"]
    values = frame[value_columns].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise BinanceArchiveIntegrityError("archive contains non-finite OHLCV values")
    if bool((frame["volume"] < 0).any()):
        raise BinanceArchiveIntegrityError("archive contains negative volume")
    if bool(
        (
            (frame["low"] > frame["high"])
            | (frame["open"] < frame["low"])
            | (frame["open"] > frame["high"])
            | (frame["close"] < frame["low"])
            | (frame["close"] > frame["high"])
        ).any()
    ):
        raise BinanceArchiveIntegrityError("archive contains invalid OHLC price bounds")

    duplicate_rows = frame[frame["timestamp"].duplicated(keep=False)]
    if not duplicate_rows.empty:
        for _, group in duplicate_rows.groupby("timestamp", sort=False):
            group_values = group[value_columns].to_numpy(dtype=np.float64, copy=False)
            if not np.all(group_values == group_values[0]):
                raise BinanceArchiveIntegrityError(
                    f"archive contains conflicting rows for timestamp {int(group.iloc[0]['timestamp'])}"
                )
    return (
        frame.sort_values("timestamp", kind="mergesort")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )


class BinanceOhlcvArchiveClient:
    def __init__(
        self,
        *,
        max_concurrency: int = BINANCE_ARCHIVE_MAX_CONCURRENCY,
        session: Optional[aiohttp.ClientSession] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._session = session
        self._owns_session = session is None
        self._sem = asyncio.Semaphore(max(1, int(max_concurrency)))
        self.log = logger or logging.getLogger(__name__)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=180, connect=20, sock_read=120)
            connector = aiohttp.TCPConnector(
                limit=BINANCE_ARCHIVE_MAX_CONCURRENCY,
                limit_per_host=BINANCE_ARCHIVE_MAX_CONCURRENCY,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
            )
            self._session = aiohttp.ClientSession(timeout=timeout, connector=connector)
            self._owns_session = True
        return self._session

    async def aclose(self) -> None:
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()

    async def _get(self, url: str) -> tuple[int, bytes]:
        session = await self._get_session()
        async with self._sem:
            async with session.get(url) as response:
                if response.status == 404:
                    return 404, b""
                response.raise_for_status()
                return response.status, await response.read()

    async def fetch(self, request: BinanceArchiveRequest) -> BinanceArchiveResult:
        try:
            checksum_status, checksum_payload = await self._get(request.url + ".CHECKSUM")
            if checksum_status == 404:
                return BinanceArchiveResult(request=request, status="not_found")
            expected_digest = _parse_checksum(checksum_payload, request.filename)
            archive_status, raw = await self._get(request.url)
            if archive_status == 404:
                return BinanceArchiveResult(request=request, status="not_found")
            actual_digest = hashlib.sha256(raw).hexdigest()
            if actual_digest != expected_digest:
                raise BinanceArchiveIntegrityError(
                    f"SHA-256 mismatch: expected {expected_digest}, got {actual_digest}"
                )
            frame = await asyncio.to_thread(_parse_archive_zip, raw, request)
            return BinanceArchiveResult(request=request, status="ok", frame=frame)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return BinanceArchiveResult(
                request=request,
                status="error",
                error=f"{type(exc).__name__}: {exc}",
            )

    async def fetch_many(
        self, requests: list[BinanceArchiveRequest]
    ) -> list[BinanceArchiveResult]:
        if not requests:
            return []
        tasks = [asyncio.create_task(self.fetch(request)) for request in requests]
        try:
            return list(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


def symbol_to_archive_code(symbol: str) -> str:
    value = str(symbol or "")
    if "/" in value:
        base, remainder = value.split("/", 1)
        quote = remainder.split(":", 1)[0]
        return f"{base}{quote}".replace("-", "").upper()
    return value.replace("/", "").replace(":", "").replace("-", "").upper()


__all__ = [
    "BINANCE_DAILY_LAG_DAYS",
    "BINANCE_DAILY_MIN_MISSING_CANDLES",
    "BINANCE_MONTHLY_MIN_MISSING_CANDLES",
    "BINANCE_MONTHLY_PUBLICATION_BUFFER_HOURS",
    "BinanceArchiveIntegrityError",
    "BinanceArchiveRequest",
    "BinanceArchiveResult",
    "BinanceOhlcvArchiveClient",
    "daily_archive_eligible",
    "first_monday_after_month",
    "monthly_archive_eligible",
    "plan_binance_archive_requests",
    "symbol_to_archive_code",
]
