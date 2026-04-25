from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ohlcv_store import OhlcvStore, timeframe_to_interval_ms


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
        start_ts: int,
        end_ts: int,
        btc_usd_prices: np.ndarray,
        mss: dict,
        run_id: str,
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
        for coin_idx, coin in enumerate(coins):
            valid_buffer[:] = False
            coin_view = hlcvs[:, coin_idx, :]
            store_symbol = symbols_by_coin.get(coin, coin)
            self.store.copy_range_into(
                exchange, "1m", store_symbol, start_ts, end_ts, coin_view, valid_buffer
            )
            if valid_buffer.any():
                valid_indices = np.flatnonzero(valid_buffer)
                first_valid_index = int(valid_indices[0])
                last_valid_index = int(valid_indices[-1])
            else:
                first_valid_index = int(n_steps)
                last_valid_index = int(n_steps)
            meta = enriched_mss.setdefault(coin, {})
            meta["first_valid_index"] = first_valid_index
            meta["last_valid_index"] = last_valid_index

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
        aligned = np.asarray(aligned_values_by_coin[coin], dtype=np.float64)
        if aligned.shape != (n_steps, 4):
            raise ValueError(
                f"aligned_values_by_coin[{coin!r}] must have shape ({n_steps}, 4), got {aligned.shape}"
            )
        hlcvs[:, coin_idx, :] = aligned
        valid_mask = ~np.isnan(aligned[:, 0])
        if valid_mask.any():
            valid_indices = np.flatnonzero(valid_mask)
            first_valid_index = int(valid_indices[0])
            last_valid_index = int(valid_indices[-1])
        else:
            first_valid_index = int(n_steps)
            last_valid_index = int(n_steps)
        meta = enriched_mss.setdefault(coin, {})
        meta["first_valid_index"] = first_valid_index
        meta["last_valid_index"] = last_valid_index

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
