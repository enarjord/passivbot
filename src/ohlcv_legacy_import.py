from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

import numpy as np

from legacy_data_migrator import (
    STANDARD_TO_CCXT_ID,
    _sanitize_symbol,
    get_legacy_exchange_name,
)
from ohlcv_store import (
    MonthChunkPaths,
    OhlcvStore,
    _sanitize_symbol as _sanitize_store_symbol,
    month_end_ts,
    month_key_for_ts,
    month_start_ts,
    rows_in_month,
    timeframe_to_interval_ms,
)
from utils import heuristic_symbol_to_coin


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


def _resolve_legacy_day_path(symbol_dir: Path, day: str) -> Path | None:
    for suffix in (".npy", ".npz"):
        fpath = symbol_dir / f"{day}{suffix}"
        if fpath.exists():
            return fpath
    return None


def _load_legacy_day_file(fpath: Path) -> tuple[np.ndarray, np.ndarray]:
    if fpath.suffix.lower() == ".npz":
        with np.load(fpath, allow_pickle=False) as archive:
            if "candles" not in archive:
                raise ValueError(f"Missing 'candles' key in {fpath}")
            arr = archive["candles"]
        if arr.dtype.names is None:
            raise ValueError(f"Expected structured dtype in {fpath}")
    else:
        arr = np.load(fpath, allow_pickle=False)

    if arr.dtype.names is not None:
        required = ("ts", "h", "l", "c", "bv")
        if fpath.suffix.lower() == ".npz":
            required += ("o",)
        missing = [name for name in required if name not in arr.dtype.names]
        if missing:
            raise ValueError(f"{fpath} missing required fields {missing}")
        raw_ts = arr["ts"]
        values = np.column_stack([arr[name] for name in ("h", "l", "c", "bv")])
    elif arr.ndim == 2 and arr.shape[1] == 6:
        raw_ts = arr[:, 0]
        values = arr[:, [2, 3, 4, 5]]
    else:
        raise ValueError(f"Expected structured candles or six OHLCV columns in {fpath}")

    # Inspect the original millisecond values before conversion. Casting first
    # would turn fractional or overflowing timestamps into different candles.
    if raw_ts.ndim != 1 or raw_ts.dtype.kind not in "iuf":
        raise ValueError(f"Invalid legacy timestamps in {fpath}")
    invalid = not np.isfinite(raw_ts).all()
    if raw_ts.dtype.kind == "f":
        invalid |= (np.any(raw_ts < -(2**63)) or np.any(raw_ts >= 2**63)
                    or np.any(raw_ts != np.trunc(raw_ts)))
    elif raw_ts.dtype.kind == "u":
        invalid |= np.any(raw_ts > np.iinfo(np.int64).max)
    if invalid:
        raise ValueError(f"Invalid legacy timestamps in {fpath}")
    return raw_ts.astype(np.int64, copy=False), values.astype(np.float32)


def _preflight_destination(
    store: OhlcvStore,
    exchange: str,
    timeframe: str,
    symbol: str,
    start_ts: int,
    end_ts: int,
    batches: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]],
) -> int:
    """Read and validate the target without creating directories or lock files."""
    def paths_for(year: int, month: int) -> MonthChunkPaths:
        base = store.root / "data" / exchange / timeframe / _sanitize_store_symbol(symbol) / f"{year:04d}"
        return MonthChunkPaths(base / f"{month:02d}.npy", base / f"{month:02d}.valid.npy")

    # Include registered intervening months even when no source rows target them.
    # The write path rechecks under its lock; this is validation, not a transaction
    # against concurrent writers or later filesystem failures.
    existing = {}
    for chunk in store.catalog.list_chunks(exchange, timeframe, symbol, start_ts, end_ts):
        key = (chunk.year, chunk.month)
        paths = paths_for(*key)
        if (Path(chunk.body_path).resolve() != paths.body_path.resolve()
                or Path(chunk.valid_path).resolve() != paths.valid_path.resolve()
                or chunk.rows != rows_in_month(*key, timeframe)
                or chunk.start_ts != month_start_ts(*key)
                or chunk.end_ts != month_end_ts(*key, timeframe)):
            raise ValueError("current OHLCV chunk metadata differs from its month")
        store._validate_chunk_files(chunk, paths)
        store._invalidate_verified_checksum_cache(paths.body_path)
        store._verify_chunk_checksum_unlocked(chunk, paths)
        existing[key] = chunk

    interval = timeframe_to_interval_ms(timeframe)
    missing = 0
    for key, (timestamps, _) in batches.items():
        paths = paths_for(*key)
        if key not in existing:
            if paths.body_path.exists() or paths.valid_path.exists():
                raise ValueError("uncatalogued current OHLCV chunk; repair the catalog before import")
            missing += len(timestamps)
            continue
        valid = np.load(paths.valid_path, mmap_mode="r", allow_pickle=False)
        try:
            offsets = ((timestamps - month_start_ts(*key)) // interval).astype(np.int64)
            missing += int((~valid[offsets]).sum())
        finally:
            del valid
    return missing


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
        if _resolve_legacy_day_path(symbol_dir, day) is not None:
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
    source_layout: str = "daily_cache",
    execute: bool = True,
) -> int:
    if end_ts < start_ts:
        raise ValueError("end_ts must be >= start_ts")
    if source_layout == "daily_cache":
        symbol_dirs = [resolve_legacy_symbol_dir(legacy_root, exchange, timeframe, symbol)]
    elif source_layout == "historical":
        if timeframe != "1m":
            raise ValueError("historical source layout supports 1m only")
        source_exchange = get_legacy_exchange_name(f"ohlcvs_{str(exchange).lower()}")
        if source_exchange == "gateio":
            raise ValueError("historical GateIO volume units are ambiguous; use base-volume daily_cache shards")
        coin = heuristic_symbol_to_coin(str(symbol).strip())
        if not coin or coin in {".", ".."} or any(ch in coin for ch in ("/", "\\", ":")):
            raise ValueError("invalid historical source coin")
        names = [f"ohlcvs_{source_exchange}",
                 f"ohlcvs_{STANDARD_TO_CCXT_ID.get(source_exchange, source_exchange)}"]
        if source_exchange == "binance":
            names.append("ohlcvs_futures")
        symbol_dirs = [Path(legacy_root) / name / coin for name in dict.fromkeys(names)]
    else:
        raise ValueError("source_layout must be daily_cache or historical")
    interval = timeframe_to_interval_ms(timeframe)
    if start_ts % interval or end_ts % interval:
        raise ValueError(f"import range must align to {timeframe}")

    previous_source_end: int | None = None
    month_batches: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    for day in _iter_utc_days(start_ts, end_ts):
        fpath = next((path for directory in symbol_dirs
                      if (path := _resolve_legacy_day_path(directory, day)) is not None), None)
        if fpath is None:
            continue
        ts, values = _load_legacy_day_file(fpath)
        if (ts.ndim != 1 or values.shape != (len(ts), 4)
            or np.any(ts[1:] <= ts[:-1]) or np.any(ts % interval)
            or not np.isfinite(values).all() or np.any(values[:, :3] <= 0)
            or np.any(values[:, 0] < values[:, 1])
            or np.any(values[:, 2] > values[:, 0])
            or np.any(values[:, 2] < values[:, 1]) or np.any(values[:, 3] < 0)):
            raise ValueError(f"invalid or unordered legacy OHLCV rows in {fpath}")
        mask = (ts >= int(start_ts)) & (ts <= int(end_ts))
        if not mask.any():
            continue
        day_ts = ts[mask]
        day_values = values[mask]
        if previous_source_end is not None and int(day_ts[0]) <= previous_source_end:
            raise ValueError("legacy shards overlap or are not chronologically ordered")
        previous_source_end = int(day_ts[-1])
        cursor = 0
        while cursor < len(day_ts):
            key = month_key_for_ts(int(day_ts[cursor]))
            stop = int(np.searchsorted(day_ts, month_end_ts(*key, timeframe) + interval))
            month_batches[key].append((day_ts[cursor:stop], day_values[cursor:stop]))
            cursor = stop

    batches = {}
    for key in sorted(month_batches):
        ts_parts, value_parts = zip(*month_batches[key])
        timestamps = np.concatenate(ts_parts)
        values = np.concatenate(value_parts)
        if np.any(timestamps[1:] <= timestamps[:-1]):
            raise ValueError("legacy shards overlap or are not chronologically ordered")
        batches[key] = (timestamps, values)
    missing = _preflight_destination(store, exchange, timeframe, symbol, start_ts, end_ts, batches)
    if not execute:
        return missing
    imported_rows = 0
    for timestamps, values in batches.values():
        imported_rows += store.write_rows(
            exchange, timeframe, symbol, timestamps, values, preserve_existing=True,
        )
    return imported_rows
