from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ohlcv_store import OhlcvStore, timeframe_to_interval_ms


def _fill_sparse_hlcv_gaps(
    values: np.ndarray, valid_mask: np.ndarray, *, fill_edge_gaps: bool = False
) -> int:
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        return 0
    first_valid = int(valid_indices[0])
    last_valid = int(valid_indices[-1])
    filled = 0

    if fill_edge_gaps and first_valid > 0:
        anchor_close = float(values[first_valid, 2])
        values[:first_valid, :3] = anchor_close
        values[:first_valid, 3] = 0.0
        valid_mask[:first_valid] = True
        filled += first_valid

    interior_start = first_valid + 1
    interior_end = last_valid - 1
    if interior_start <= interior_end:
        missing = np.flatnonzero(~valid_mask[interior_start : interior_end + 1]) + interior_start
        if missing.size:
            span_breaks = np.flatnonzero(np.diff(missing) > 1) + 1
            span_starts = np.r_[missing[0], missing[span_breaks]]
            span_ends = np.r_[missing[span_breaks - 1], missing[-1]]
            for span_start, span_end in zip(span_starts, span_ends):
                start_idx = int(span_start)
                end_idx = int(span_end)
                anchor_close = float(values[start_idx - 1, 2])
                values[start_idx : end_idx + 1, :3] = anchor_close
                values[start_idx : end_idx + 1, 3] = 0.0
                valid_mask[start_idx : end_idx + 1] = True
                filled += end_idx - start_idx + 1

    if fill_edge_gaps and last_valid < len(valid_mask) - 1:
        anchor_close = float(values[last_valid, 2])
        trailing_start = last_valid + 1
        values[trailing_start:, :3] = anchor_close
        values[trailing_start:, 3] = 0.0
        valid_mask[trailing_start:] = True
        filled += len(valid_mask) - trailing_start

    return int(filled)


def _valid_span(valid_mask: np.ndarray) -> tuple[int, int, int]:
    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        return int(len(valid_mask)), int(len(valid_mask)), 0
    return int(valid_indices[0]), int(valid_indices[-1]), int(valid_indices.size)


def _longest_contiguous_valid_span(valid_mask: np.ndarray) -> tuple[int, int, int]:
    valid = np.asarray(valid_mask, dtype=bool)
    if not valid.any():
        return int(len(valid)), int(len(valid)), 0
    padded = np.concatenate(([False], valid, [False]))
    changes = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    if starts.size == 0:
        return int(len(valid)), int(len(valid)), 0
    lengths = ends - starts
    best = int(np.argmax(lengths))
    start_idx = int(starts[best])
    end_idx = int(ends[best]) - 1
    return start_idx, end_idx, int(lengths[best])


def _coverage_metadata(valid_mask: np.ndarray, *, start_ts: int, interval_ms: int) -> dict:
    first_idx, last_idx, valid_count = _valid_span(valid_mask)
    total = int(len(valid_mask))
    invalid_count = int(total - valid_count)
    meta = {
        "coverage_requested_start_ts": int(start_ts),
        "coverage_requested_end_ts": int(start_ts + (total - 1) * interval_ms) if total else int(start_ts),
        "coverage_valid_rows": int(valid_count),
        "coverage_invalid_rows": int(invalid_count),
        "coverage_internal_gap_count": 0,
        "coverage_internal_gap_minutes": 0,
        "coverage_trailing_missing_minutes": 0,
        "coverage_leading_missing_minutes": 0,
    }
    if valid_count <= 0:
        return meta
    meta["coverage_valid_start_ts"] = int(start_ts + first_idx * interval_ms)
    meta["coverage_valid_end_ts"] = int(start_ts + last_idx * interval_ms)
    meta["coverage_leading_missing_minutes"] = int(first_idx * interval_ms // 60_000)
    meta["coverage_trailing_missing_minutes"] = int((total - last_idx - 1) * interval_ms // 60_000)
    if first_idx + 1 <= last_idx - 1:
        interior_invalid = np.flatnonzero(~valid_mask[first_idx + 1 : last_idx]) + first_idx + 1
        if interior_invalid.size:
            breaks = np.flatnonzero(np.diff(interior_invalid) > 1) + 1
            starts = np.r_[interior_invalid[0], interior_invalid[breaks]]
            ends = np.r_[interior_invalid[breaks - 1], interior_invalid[-1]]
            windows = []
            minutes = 0
            for raw_start, raw_end in zip(starts, ends):
                gap_start = int(raw_start)
                gap_end = int(raw_end)
                gap_minutes = int((gap_end - gap_start + 1) * interval_ms // 60_000)
                minutes += gap_minutes
                if len(windows) < 20:
                    windows.append(
                        [
                            int(start_ts + gap_start * interval_ms),
                            int(start_ts + gap_end * interval_ms),
                        ]
                    )
            meta["coverage_internal_gap_count"] = int(len(starts))
            meta["coverage_internal_gap_minutes"] = int(minutes)
            meta["coverage_internal_gap_windows"] = windows
    return meta


@dataclass(frozen=True)
class SharedBacktestDatasetHandle:
    root: str
    hlcvs_path: str
    timestamps_path: str
    btc_usd_prices_path: str
    hlcvs_shape: tuple[int, int, int]
    timestamps_shape: tuple[int]
    btc_shape: tuple[int]
    hlcvs_dtype: str
    timestamps_dtype: str
    btc_dtype: str
    coins: list[str]
    exchange: str
    mss: dict
    meta: dict

    def open_hlcvs(self) -> np.memmap:
        return np.memmap(self.hlcvs_path, mode="r", dtype=np.dtype(self.hlcvs_dtype), shape=self.hlcvs_shape)

    def open_timestamps(self) -> np.memmap:
        return np.memmap(
            self.timestamps_path,
            mode="r",
            dtype=np.dtype(self.timestamps_dtype),
            shape=self.timestamps_shape,
        )

    def open_btc_usd_prices(self) -> np.memmap:
        return np.memmap(
            self.btc_usd_prices_path,
            mode="r",
            dtype=np.dtype(self.btc_dtype),
            shape=self.btc_shape,
        )


class BacktestDatasetMaterializer:
    def __init__(self, store: OhlcvStore, output_root: str | Path):
        self.store = store
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)

    def materialize(
        self,
        *,
        exchange: str,
        coins: list[str],
        symbols_by_coin: dict[str, str] | None = None,
        source_windows_by_coin: dict[str, tuple[int, int]] | None = None,
        start_ts: int,
        end_ts: int,
        btc_usd_prices: np.ndarray,
        mss: dict,
        run_id: str,
        fill_edge_gaps: bool = False,
    ) -> SharedBacktestDatasetHandle:
        interval_ms = timeframe_to_interval_ms("1m")
        if end_ts < start_ts:
            raise ValueError("end_ts must be >= start_ts")
        if start_ts % interval_ms != 0 or end_ts % interval_ms != 0:
            raise ValueError("materialized backtest range must align to 1m")

        n_steps = int((end_ts - start_ts) // interval_ms) + 1
        btc_arr = np.asarray(btc_usd_prices, dtype=np.float64)
        if btc_arr.shape != (n_steps,):
            raise ValueError(
                f"btc_usd_prices must have shape ({n_steps},), got {tuple(int(x) for x in btc_arr.shape)}"
            )

        run_root = self.output_root / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        hlcvs_path = run_root / "hlcvs.dat"
        timestamps_path = run_root / "timestamps.dat"
        btc_path = run_root / "btc_usd_prices.dat"

        hlcvs = np.memmap(hlcvs_path, mode="w+", dtype=np.float64, shape=(n_steps, len(coins), 4))
        hlcvs[:] = np.nan
        timestamps = np.memmap(timestamps_path, mode="w+", dtype=np.int64, shape=(n_steps,))
        timestamps[:] = np.arange(start_ts, end_ts + interval_ms, interval_ms, dtype=np.int64)
        btc_mm = np.memmap(btc_path, mode="w+", dtype=np.float64, shape=(n_steps,))
        btc_mm[:] = btc_arr

        enriched_mss = deepcopy(mss)
        valid_buffer = np.zeros(n_steps, dtype=np.bool_)
        symbols_by_coin = symbols_by_coin or {}
        source_windows_by_coin = source_windows_by_coin or {}
        for coin_idx, coin in enumerate(coins):
            coin_t0 = time.monotonic()
            logging.info(
                "[materializer] coin start %d/%d exchange=%s coin=%s",
                coin_idx + 1,
                len(coins),
                exchange,
                coin,
            )
            valid_buffer[:] = False
            coin_view = hlcvs[:, coin_idx, :]
            store_symbol = symbols_by_coin.get(coin, coin)
            copy_start_ts, copy_end_ts = source_windows_by_coin.get(coin, (start_ts, end_ts))
            copy_start_ts = int(copy_start_ts)
            copy_end_ts = int(copy_end_ts)
            if copy_start_ts % interval_ms != 0 or copy_end_ts % interval_ms != 0:
                raise ValueError(f"source window for {coin} must align to 1m")
            if copy_start_ts < start_ts or copy_end_ts > end_ts or copy_end_ts < copy_start_ts:
                raise ValueError(
                    f"source window for {coin} must be inside materialized range: "
                    f"{copy_start_ts}..{copy_end_ts} not within {start_ts}..{end_ts}"
                )
            dest_start = int((copy_start_ts - start_ts) // interval_ms)
            dest_end = int((copy_end_ts - start_ts) // interval_ms) + 1
            self.store.copy_range_into(
                exchange,
                "1m",
                store_symbol,
                copy_start_ts,
                copy_end_ts,
                coin_view[dest_start:dest_end],
                valid_buffer[dest_start:dest_end],
            )
            source_first_valid_index, source_last_valid_index, source_valid_count = _valid_span(
                valid_buffer
            )
            tradable_first_index, tradable_last_index, _tradable_count = (
                _longest_contiguous_valid_span(valid_buffer)
            )
            invalid_rows = int(n_steps - source_valid_count)
            coverage_meta = _coverage_metadata(
                valid_buffer.copy(), start_ts=int(start_ts), interval_ms=int(interval_ms)
            )
            synthetic_gap_fill_count = _fill_sparse_hlcv_gaps(
                coin_view, valid_buffer, fill_edge_gaps=fill_edge_gaps
            )
            meta = enriched_mss.setdefault(coin, {})
            meta["first_valid_index"] = tradable_first_index
            meta["last_valid_index"] = tradable_last_index
            meta["source_window_start_ts"] = int(copy_start_ts)
            meta["source_window_end_ts"] = int(copy_end_ts)
            meta["source_first_valid_index"] = source_first_valid_index
            meta["source_last_valid_index"] = source_last_valid_index
            meta.update(coverage_meta)
            if synthetic_gap_fill_count:
                meta["synthetic_gap_fill_count"] = synthetic_gap_fill_count
                meta["synthetic_gap_fill_source"] = "previous_or_edge_close"
            logging.info(
                "[materializer] coin done %d/%d exchange=%s coin=%s source_first_valid=%d "
                "source_last_valid=%d invalid_rows=%d synthetic_filled=%d elapsed_s=%.1f",
                coin_idx + 1,
                len(coins),
                exchange,
                coin,
                source_first_valid_index,
                source_last_valid_index,
                invalid_rows,
                synthetic_gap_fill_count,
                time.monotonic() - coin_t0,
            )

        hlcvs.flush()
        timestamps.flush()
        btc_mm.flush()
        del hlcvs
        del timestamps
        del btc_mm

        meta = {
            "exchange": exchange,
            "coins": list(coins),
            "start_ts": int(start_ts),
            "end_ts": int(end_ts),
            "interval_ms": int(interval_ms),
            "hlcvs_shape": [int(n_steps), int(len(coins)), 4],
            "hlcvs_dtype": "float64",
            "timestamps_shape": [int(n_steps)],
            "timestamps_dtype": "int64",
            "btc_shape": [int(n_steps)],
            "btc_dtype": "float64",
        }

        return SharedBacktestDatasetHandle(
            root=str(run_root.resolve()),
            hlcvs_path=str(hlcvs_path.resolve()),
            timestamps_path=str(timestamps_path.resolve()),
            btc_usd_prices_path=str(btc_path.resolve()),
            hlcvs_shape=(n_steps, len(coins), 4),
            timestamps_shape=(n_steps,),
            btc_shape=(n_steps,),
            hlcvs_dtype="float64",
            timestamps_dtype="int64",
            btc_dtype="float64",
            coins=list(coins),
            exchange=exchange,
            mss=enriched_mss,
            meta=meta,
        )


def materialize_frames(
    *,
    output_root: str | Path,
    exchange: str,
    coins: list[str],
    timestamps: np.ndarray,
    aligned_values_by_coin: dict[str, np.ndarray],
    btc_usd_prices: np.ndarray,
    mss: dict,
    run_id: str,
) -> SharedBacktestDatasetHandle:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    ts_arr = np.asarray(timestamps, dtype=np.int64)
    if ts_arr.ndim != 1:
        raise ValueError("timestamps must be a 1D array")
    n_steps = int(ts_arr.shape[0])
    btc_arr = np.asarray(btc_usd_prices, dtype=np.float64)
    if btc_arr.shape != (n_steps,):
        raise ValueError(
            f"btc_usd_prices must have shape ({n_steps},), got {tuple(int(x) for x in btc_arr.shape)}"
        )

    run_root = output_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    hlcvs_path = run_root / "hlcvs.dat"
    timestamps_path = run_root / "timestamps.dat"
    btc_path = run_root / "btc_usd_prices.dat"

    hlcvs = np.memmap(hlcvs_path, mode="w+", dtype=np.float64, shape=(n_steps, len(coins), 4))
    hlcvs[:] = np.nan
    timestamps_mm = np.memmap(timestamps_path, mode="w+", dtype=np.int64, shape=(n_steps,))
    timestamps_mm[:] = ts_arr
    btc_mm = np.memmap(btc_path, mode="w+", dtype=np.float64, shape=(n_steps,))
    btc_mm[:] = btc_arr

    enriched_mss = deepcopy(mss)
    for coin_idx, coin in enumerate(coins):
        coin_t0 = time.monotonic()
        logging.info(
            "[materializer] frame coin start %d/%d exchange=%s coin=%s",
            coin_idx + 1,
            len(coins),
            exchange,
            coin,
        )
        aligned = np.array(aligned_values_by_coin[coin], dtype=np.float64, copy=True)
        if aligned.shape != (n_steps, 4):
            raise ValueError(
                f"aligned_values_by_coin[{coin!r}] must have shape ({n_steps}, 4), got {aligned.shape}"
            )
        valid_mask = ~np.isnan(aligned[:, 0])
        source_first_valid_index, source_last_valid_index, source_valid_count = _valid_span(
            valid_mask
        )
        tradable_first_index, tradable_last_index, _tradable_count = (
            _longest_contiguous_valid_span(valid_mask)
        )
        invalid_rows = int(n_steps - source_valid_count)
        coverage_meta = _coverage_metadata(
            valid_mask.copy(),
            start_ts=int(ts_arr[0]) if n_steps else 0,
            interval_ms=60_000 if n_steps < 2 else int(ts_arr[1] - ts_arr[0]),
        )
        synthetic_gap_fill_count = _fill_sparse_hlcv_gaps(aligned, valid_mask)
        hlcvs[:, coin_idx, :] = aligned
        meta = enriched_mss.setdefault(coin, {})
        meta["first_valid_index"] = tradable_first_index
        meta["last_valid_index"] = tradable_last_index
        meta["source_first_valid_index"] = source_first_valid_index
        meta["source_last_valid_index"] = source_last_valid_index
        meta.update(coverage_meta)
        if synthetic_gap_fill_count:
            meta["synthetic_gap_fill_count"] = synthetic_gap_fill_count
            meta["synthetic_gap_fill_source"] = "previous_or_edge_close"
        logging.info(
            "[materializer] frame coin done %d/%d exchange=%s coin=%s source_first_valid=%d "
            "source_last_valid=%d invalid_rows=%d synthetic_filled=%d elapsed_s=%.1f",
            coin_idx + 1,
            len(coins),
            exchange,
            coin,
            source_first_valid_index,
            source_last_valid_index,
            invalid_rows,
            synthetic_gap_fill_count,
            time.monotonic() - coin_t0,
        )

    hlcvs.flush()
    timestamps_mm.flush()
    btc_mm.flush()
    del hlcvs
    del timestamps_mm
    del btc_mm

    meta = {
        "exchange": exchange,
        "coins": list(coins),
        "start_ts": int(ts_arr[0]) if n_steps else None,
        "end_ts": int(ts_arr[-1]) if n_steps else None,
        "interval_ms": 60_000 if n_steps < 2 else int(ts_arr[1] - ts_arr[0]),
        "hlcvs_shape": [int(n_steps), int(len(coins)), 4],
        "hlcvs_dtype": "float64",
        "timestamps_shape": [int(n_steps)],
        "timestamps_dtype": "int64",
        "btc_shape": [int(n_steps)],
        "btc_dtype": "float64",
    }

    return SharedBacktestDatasetHandle(
        root=str(run_root.resolve()),
        hlcvs_path=str(hlcvs_path.resolve()),
        timestamps_path=str(timestamps_path.resolve()),
        btc_usd_prices_path=str(btc_path.resolve()),
        hlcvs_shape=(n_steps, len(coins), 4),
        timestamps_shape=(n_steps,),
        btc_shape=(n_steps,),
        hlcvs_dtype="float64",
        timestamps_dtype="int64",
        btc_dtype="float64",
        coins=list(coins),
        exchange=exchange,
        mss=enriched_mss,
        meta=meta,
    )
