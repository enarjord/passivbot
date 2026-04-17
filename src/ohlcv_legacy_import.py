from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from legacy_data_migrator import _sanitize_symbol
from ohlcv_store import OhlcvStore


@dataclass(frozen=True)
class LegacyRangeInspection:
    exchange: str
    timeframe: str
    symbol: str
    start_ts: int
    end_ts: int
    present_days: tuple[str, ...]
    missing_days: tuple[str, ...]

    @property
    def all_days_present(self) -> bool:
        return len(self.missing_days) == 0


def _iter_utc_days(start_ts: int, end_ts: int):
    current = datetime.fromtimestamp(int(start_ts) / 1000, tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    end_day = datetime.fromtimestamp(int(end_ts) / 1000, tz=timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    while current <= end_day:
        yield current.strftime("%Y-%m-%d")
        current += timedelta(days=1)


def resolve_legacy_symbol_dir(
    legacy_root: str | Path, exchange: str, timeframe: str, symbol: str
) -> Path:
    return Path(legacy_root) / str(exchange) / str(timeframe) / _sanitize_symbol(symbol)


def inspect_legacy_range(
    *,
    legacy_root: str | Path,
    exchange: str,
    timeframe: str,
    symbol: str,
    start_ts: int,
    end_ts: int,
) -> LegacyRangeInspection:
    if end_ts < start_ts:
        raise ValueError("end_ts must be >= start_ts")
    symbol_dir = resolve_legacy_symbol_dir(legacy_root, exchange, timeframe, symbol)
    present_days: list[str] = []
    missing_days: list[str] = []
    for day in _iter_utc_days(start_ts, end_ts):
        if (symbol_dir / f"{day}.npy").exists():
            present_days.append(day)
        else:
            missing_days.append(day)
    return LegacyRangeInspection(
        exchange=str(exchange),
        timeframe=str(timeframe),
        symbol=str(symbol),
        start_ts=int(start_ts),
        end_ts=int(end_ts),
        present_days=tuple(present_days),
        missing_days=tuple(missing_days),
    )


def import_legacy_range_into_store(
    *,
    store: OhlcvStore,
    legacy_root: str | Path,
    exchange: str,
    timeframe: str,
    symbol: str,
    start_ts: int,
    end_ts: int,
) -> int:
    if end_ts < start_ts:
        raise ValueError("end_ts must be >= start_ts")
    symbol_dir = resolve_legacy_symbol_dir(legacy_root, exchange, timeframe, symbol)
    if not symbol_dir.exists():
        return 0

    imported_rows = 0
    for day in _iter_utc_days(start_ts, end_ts):
        fpath = symbol_dir / f"{day}.npy"
        if not fpath.exists():
            continue
        arr = np.load(fpath, allow_pickle=False)
        if not isinstance(arr, np.ndarray) or arr.dtype.names is None:
            raise ValueError(f"expected structured OHLCV array in {fpath}")
        required = ("ts", "h", "l", "c", "bv")
        missing = [name for name in required if name not in arr.dtype.names]
        if missing:
            raise ValueError(f"{fpath} missing required fields {missing}")
        ts = arr["ts"].astype(np.int64, copy=False)
        mask = (ts >= int(start_ts)) & (ts <= int(end_ts))
        if not mask.any():
            continue
        values = np.column_stack(
            [
                arr["h"][mask].astype(np.float32, copy=False),
                arr["l"][mask].astype(np.float32, copy=False),
                arr["c"][mask].astype(np.float32, copy=False),
                arr["bv"][mask].astype(np.float32, copy=False),
            ]
        )
        store.write_rows(exchange, timeframe, symbol, ts[mask], values)
        imported_rows += int(mask.sum())
    return imported_rows
