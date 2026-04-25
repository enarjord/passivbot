from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from ohlcv_catalog import OhlcvCatalog, ChunkRecord


BACKTEST_OHLCV_DTYPE = np.float32
BACKTEST_OHLCV_FIELDS = ("high", "low", "close", "volume")


def timeframe_to_interval_ms(timeframe: str) -> int:
    normalized = str(timeframe).strip().lower()
    if normalized == "1m":
        return 60_000
    if normalized == "1h":
        return 60 * 60_000
    raise ValueError(f"unsupported timeframe {timeframe!r}")


def month_start_ts(year: int, month: int) -> int:
    dt = datetime(year, month, 1, tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def rows_in_month(year: int, month: int, timeframe: str) -> int:
    interval_ms = timeframe_to_interval_ms(timeframe)
    n_days = calendar.monthrange(year, month)[1]
    return (n_days * 24 * 60 * 60_000) // interval_ms


def month_key_for_ts(ts_ms: int) -> tuple[int, int]:
    dt = datetime.fromtimestamp(int(ts_ms) / 1000, tz=timezone.utc)
    return dt.year, dt.month


def month_end_ts(year: int, month: int, timeframe: str) -> int:
    return month_start_ts(year, month) + rows_in_month(year, month, timeframe) * timeframe_to_interval_ms(
        timeframe
    ) - timeframe_to_interval_ms(timeframe)


def month_offset(ts_ms: int, year: int, month: int, timeframe: str) -> int:
    start = month_start_ts(year, month)
    interval_ms = timeframe_to_interval_ms(timeframe)
    delta = int(ts_ms) - start
    if delta < 0 or delta % interval_ms != 0:
        raise ValueError(f"timestamp {ts_ms} is not aligned to {timeframe} in {year:04d}-{month:02d}")
    offset = delta // interval_ms
    max_rows = rows_in_month(year, month, timeframe)
    if not 0 <= offset < max_rows:
        raise ValueError(f"timestamp {ts_ms} out of month bounds for {year:04d}-{month:02d}")
    return int(offset)


def _sanitize_symbol(symbol: str) -> str:
    out = str(symbol).strip()
    for ch in ("/", ":", "\\"):
        out = out.replace(ch, "_")
    return out


@dataclass(frozen=True)
class MonthChunkPaths:
    body_path: Path
    valid_path: Path


@dataclass(frozen=True)
class OhlcvRange:
    timestamps: np.ndarray
    values: np.ndarray
    valid: np.ndarray


class OhlcvStore:
    def __init__(self, root: str | Path, catalog: OhlcvCatalog):
        self.root = Path(root)
        self.catalog = catalog
        self.root.mkdir(parents=True, exist_ok=True)

    def month_paths(self, exchange: str, timeframe: str, symbol: str, year: int, month: int) -> MonthChunkPaths:
        base = (
            self.root
            / "data"
            / str(exchange)
            / str(timeframe)
            / _sanitize_symbol(symbol)
            / f"{int(year):04d}"
        )
        base.mkdir(parents=True, exist_ok=True)
        stem = f"{int(month):02d}"
        return MonthChunkPaths(body_path=base / f"{stem}.npy", valid_path=base / f"{stem}.valid.npy")

    def ensure_month(
        self,
        exchange: str,
        timeframe: str,
        symbol: str,
        year: int,
        month: int,
        *,
        status: str = "open",
    ) -> MonthChunkPaths:
        paths = self.month_paths(exchange, timeframe, symbol, year, month)
        n_rows = rows_in_month(year, month, timeframe)
        if not paths.body_path.exists():
            body = np.lib.format.open_memmap(
                paths.body_path, mode="w+", dtype=BACKTEST_OHLCV_DTYPE, shape=(n_rows, 4)
            )
            body[:] = np.nan
            body.flush()
            del body
        if not paths.valid_path.exists():
            valid = np.lib.format.open_memmap(
                paths.valid_path, mode="w+", dtype=np.bool_, shape=(n_rows,)
            )
            valid[:] = False
            valid.flush()
            del valid
        self.catalog.register_chunk(
            exchange=exchange,
            timeframe=timeframe,
            symbol=symbol,
            year=year,
            month=month,
            body_path=str(paths.body_path.resolve()),
            valid_path=str(paths.valid_path.resolve()),
            start_ts=month_start_ts(year, month),
            end_ts=month_end_ts(year, month, timeframe),
            rows=n_rows,
            status=status,
        )
        return paths

    def write_rows(
        self,
        exchange: str,
        timeframe: str,
        symbol: str,
        timestamps_ms: np.ndarray,
        values: np.ndarray,
        *,
        status: str = "open",
    ) -> None:
        ts_arr = np.asarray(timestamps_ms, dtype=np.int64)
        val_arr = np.asarray(values, dtype=np.float32)
        if ts_arr.ndim != 1:
            raise ValueError("timestamps_ms must be a 1D array")
        if val_arr.ndim != 2 or val_arr.shape[1] != 4:
            raise ValueError("values must have shape [n, 4]")
        if len(ts_arr) != len(val_arr):
            raise ValueError("timestamps_ms and values must have the same length")
        if len(ts_arr) == 0:
            return
        if np.any(np.diff(ts_arr) < 0):
            raise ValueError("timestamps_ms must be sorted ascending")
        interval_ms = timeframe_to_interval_ms(timeframe)
        if np.any(ts_arr % interval_ms != 0):
            raise ValueError(f"timestamps must align to {timeframe}")

        grouped: dict[tuple[int, int], list[int]] = {}
        for idx, ts_ms in enumerate(ts_arr):
            grouped.setdefault(month_key_for_ts(int(ts_ms)), []).append(idx)

        for (year, month), indices in grouped.items():
            paths = self.ensure_month(exchange, timeframe, symbol, year, month, status=status)
            body = np.load(paths.body_path, mmap_mode="r+")
            valid = np.load(paths.valid_path, mmap_mode="r+")
            for src_idx in indices:
                offset = month_offset(int(ts_arr[src_idx]), year, month, timeframe)
                body[offset] = val_arr[src_idx]
                valid[offset] = True
            body.flush()
            valid.flush()
            del body
            del valid

        self.catalog.upsert_symbol_bounds(exchange, timeframe, symbol, int(ts_arr.min()), int(ts_arr.max()))

    def read_range(
        self, exchange: str, timeframe: str, symbol: str, start_ts: int, end_ts: int
    ) -> OhlcvRange:
        if end_ts < start_ts:
            raise ValueError("end_ts must be >= start_ts")
        interval_ms = timeframe_to_interval_ms(timeframe)
        if start_ts % interval_ms != 0 or end_ts % interval_ms != 0:
            raise ValueError(f"range must align to {timeframe}")

        timestamps = np.arange(start_ts, end_ts + interval_ms, interval_ms, dtype=np.int64)
        values = np.full((len(timestamps), 4), np.nan, dtype=np.float32)
        valid = np.zeros(len(timestamps), dtype=np.bool_)
        for chunk in self.catalog.list_chunks(exchange, timeframe, symbol, start_ts, end_ts):
            self._copy_chunk_into_range(chunk, timeframe, start_ts, end_ts, values, valid)
        return OhlcvRange(timestamps=timestamps, values=values, valid=valid)

    def copy_range_into(
        self,
        exchange: str,
        timeframe: str,
        symbol: str,
        start_ts: int,
        end_ts: int,
        out_values: np.ndarray,
        out_valid: np.ndarray,
    ) -> None:
        if out_values.ndim != 2 or out_values.shape[1] != 4:
            raise ValueError("out_values must have shape [n, 4]")
        if out_valid.ndim != 1 or out_valid.shape[0] != out_values.shape[0]:
            raise ValueError("out_valid must be 1D and match out_values length")
        for chunk in self.catalog.list_chunks(exchange, timeframe, symbol, start_ts, end_ts):
            self._copy_chunk_into_range(chunk, timeframe, start_ts, end_ts, out_values, out_valid)

    def iter_overlapping_chunks(
        self, exchange: str, timeframe: str, symbol: str, start_ts: int, end_ts: int
    ) -> Iterator[ChunkRecord]:
        yield from self.catalog.list_chunks(exchange, timeframe, symbol, start_ts, end_ts)

    def _copy_chunk_into_range(
        self,
        chunk: ChunkRecord,
        timeframe: str,
        start_ts: int,
        end_ts: int,
        out_values: np.ndarray,
        out_valid: np.ndarray,
    ) -> None:
        overlap_start = max(int(chunk.start_ts), int(start_ts))
        overlap_end = min(int(chunk.end_ts), int(end_ts))
        if overlap_end < overlap_start:
            return
        year = int(chunk.year)
        month = int(chunk.month)
        src_start = month_offset(overlap_start, year, month, timeframe)
        src_end = month_offset(overlap_end, year, month, timeframe) + 1
        interval_ms = timeframe_to_interval_ms(timeframe)
        dest_start = int((overlap_start - start_ts) // interval_ms)
        dest_end = dest_start + (src_end - src_start)

        body = np.load(chunk.body_path, mmap_mode="r")
        valid = np.load(chunk.valid_path, mmap_mode="r")
        out_values[dest_start:dest_end] = body[src_start:src_end]
        out_valid[dest_start:dest_end] = valid[src_start:src_end]
        del body
        del valid
