import datetime
import logging
from typing import Iterable, List

import numpy as np
import pandas as pd

from utils import date_to_ts, format_end_date


def ensure_millis_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a DataFrame's 'timestamp' column to milliseconds.

    Heuristic:
    - If no valid (non-zero, finite) timestamps exist, assume timestamps are already ms.
    - If there are multiple unique timestamps, use the median difference between unique timestamps:
      if the median difference is a multiple of 1000 (within a small tolerance), treat timestamps as ms.
      Otherwise treat them as seconds and multiply by 1000.
    - If only one non-zero timestamp exists, fall back to magnitude-based detection using epoch-scale
      thresholds:
        - >= 1e15 -> microseconds
        - >= 1e12 -> milliseconds
        - >= 1e9  -> seconds
        - <  1e9  -> assume milliseconds (likely small ms values)
    """
    if "timestamp" not in df.columns:
        return df

    try:
        ts = df["timestamp"].astype("float64").values
    except Exception:
        return df

    finite_mask = np.isfinite(ts) & (ts != 0)
    if not finite_mask.any():
        return df

    non_zero_ts = ts[finite_mask]
    uniq = np.unique(non_zero_ts)

    if uniq.size > 1:
        diffs = np.diff(uniq)
        median_diff = float(np.median(diffs))
        if abs(median_diff - round(median_diff / 1000.0) * 1000.0) < 1e-6:
            return df
        df["timestamp"] = df["timestamp"] * 1000.0
        return df

    rep = float(np.abs(non_zero_ts).max())
    if rep >= 1e15:
        df["timestamp"] = df["timestamp"] / 1000.0
    elif rep >= 1e12:
        pass
    elif rep >= 1e9:
        df["timestamp"] = df["timestamp"] * 1000.0
    else:
        pass
    return df


def canonicalize_daily_ohlcvs(data, start_ts: int, interval_ms: int = 60_000) -> pd.DataFrame:
    """
    Return a 1-minute canonical OHLCV DataFrame for the given day.

    Missing minutes are forward/back filled for price columns and zero-filled for volume.
    Duplicate timestamps keep the last observation.
    """
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    if isinstance(data, np.ndarray):
        df = pd.DataFrame(data, columns=columns)
    elif isinstance(data, pd.DataFrame):
        df = data[columns].copy()
    else:
        raise TypeError("data must be a pandas DataFrame or numpy array")

    df = df.reset_index(drop=True)
    df = ensure_millis_df(df)
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    end_ts = start_ts + 24 * 60 * 60 * 1000
    df = df.dropna(subset=["timestamp"])
    df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)]
    df = df.drop_duplicates(subset=["timestamp"], keep="last").sort_values("timestamp")

    if df.empty:
        raise ValueError("No data available for canonicalization")

    expected_ts = np.arange(start_ts, end_ts, interval_ms)
    reindexed = df.set_index("timestamp").reindex(expected_ts)
    missing_mask = reindexed[["open", "high", "low", "close", "volume"]].isna().all(axis=1)

    close = reindexed["close"].astype(float)
    close = close.ffill().bfill()
    if close.isna().any():
        raise ValueError("Unable to fill close prices while canonicalizing daily OHLCV data")
    reindexed["close"] = close

    for col in ["open", "high", "low"]:
        series = reindexed[col].astype(float)
        series = series.ffill().bfill()
        series = series.fillna(reindexed["close"])
        series = pd.Series(
            np.where(missing_mask, reindexed["close"], series),
            index=reindexed.index,
            dtype=float,
        )
        reindexed[col] = series

    volume = reindexed["volume"].astype(float)
    volume = volume.fillna(0.0)
    reindexed["volume"] = volume

    result = reindexed.reset_index().rename(columns={"index": "timestamp"})
    return result[columns]


def deduplicate_rows(arr: np.ndarray) -> np.ndarray:
    rows_as_tuples = map(tuple, arr)
    seen = set()
    unique_indices = [
        i
        for i, row_tuple in enumerate(rows_as_tuples)
        if not (row_tuple in seen or seen.add(row_tuple))
    ]
    return arr[unique_indices]


def dump_ohlcv_data(data, filepath: str) -> None:
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    if isinstance(data, pd.DataFrame):
        data = ensure_millis_df(data[columns]).astype(float).values
    elif isinstance(data, np.ndarray):
        pass
    else:
        raise TypeError(f"Unknown data format for {filepath}")
    np.save(filepath, deduplicate_rows(data))


def dump_daily_ohlcv_data(data, filepath: str, start_ts: int, interval_ms: int = 60_000) -> None:
    canonical = canonicalize_daily_ohlcvs(data, start_ts, interval_ms=interval_ms)
    dump_ohlcv_data(canonical, str(filepath))


def load_ohlcv_data(filepath: str) -> pd.DataFrame:
    path = str(filepath)
    if path.lower().endswith(".npz"):
        with np.load(path) as data:
            if "candles" not in data:
                raise ValueError(f"Missing 'candles' key in {filepath}")
            arr = data["candles"]
        if not isinstance(arr, np.ndarray) or arr.dtype.names is None:
            raise ValueError(f"Expected structured dtype in {filepath}")
        required = ("ts", "o", "h", "l", "c", "bv")
        missing = [name for name in required if name not in arr.dtype.names]
        if missing:
            raise ValueError(f"Missing fields {missing} in {filepath}")
        df = pd.DataFrame(
            {
                "timestamp": arr["ts"].astype(np.int64),
                "open": arr["o"].astype(float),
                "high": arr["h"].astype(float),
                "low": arr["l"].astype(float),
                "close": arr["c"].astype(float),
                "volume": arr["bv"].astype(float),
            }
        )
        df = df.drop_duplicates(subset=["timestamp"], keep="last")
        return ensure_millis_df(df)

    arr = np.load(filepath, allow_pickle=True)
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    arr_deduplicated = deduplicate_rows(arr)
    if len(arr) != len(arr_deduplicated):
        dump_ohlcv_data(arr_deduplicated, filepath)
        print(
            f"Caught .npy file with duplicate rows: {filepath} Overwrote with deduplicated version."
        )
    return ensure_millis_df(pd.DataFrame(arr_deduplicated, columns=columns))


def get_days_in_between(start_day: str, end_day: str) -> List[str]:
    date_format = "%Y-%m-%d"
    start_date = datetime.datetime.strptime(format_end_date(start_day), date_format)
    end_date = datetime.datetime.strptime(format_end_date(end_day), date_format)
    days = []
    current_date = start_date
    while current_date <= end_date:
        days.append(current_date.strftime(date_format))
        current_date += datetime.timedelta(days=1)
    return days


def fill_gaps_in_ohlcvs(df: pd.DataFrame) -> pd.DataFrame:
    interval = 60_000
    new_timestamps = np.arange(df["timestamp"].iloc[0], df["timestamp"].iloc[-1] + interval, interval)
    new_df = df.set_index("timestamp").reindex(new_timestamps)
    new_df.close = new_df.close.ffill()
    for col in ["open", "high", "low"]:
        new_df[col] = new_df[col].fillna(new_df.close)
    new_df["volume"] = new_df["volume"].fillna(0.0)
    return new_df.reset_index().rename(columns={"index": "timestamp"})


def attempt_gap_fix_ohlcvs(
    df: pd.DataFrame, symbol: str | None = None, verbose: bool = True
) -> pd.DataFrame:
    interval = 60_000
    max_hours = 12
    max_gap = interval * 60 * max_hours
    greatest_gap = df.timestamp.diff().max()
    if pd.isna(greatest_gap) or greatest_gap == interval:
        return df
    if greatest_gap > max_gap:
        raise Exception(f"Huge gap in data for {symbol}: {greatest_gap/(1000*60*60)} hours.")
    if verbose:
        logging.info(
            f"Filling small gaps in {symbol}. Largest gap: {greatest_gap/(1000*60*60):.3f} hours."
        )
    new_timestamps = np.arange(df["timestamp"].iloc[0], df["timestamp"].iloc[-1] + interval, interval)
    new_df = df.set_index("timestamp").reindex(new_timestamps)
    new_df.close = new_df.close.ffill()
    for col in ["open", "high", "low"]:
        new_df[col] = new_df[col].fillna(new_df.close)
    new_df["volume"] = new_df["volume"].fillna(0.0)
    return new_df.reset_index().rename(columns={"index": "timestamp"})


def aggregate_hlcvs(candles_1m: np.ndarray, interval: int) -> np.ndarray:
    """
    Aggregate 1m HLCV candles to coarser interval.

    Args:
        candles_1m: Array of shape (n_timesteps, n_coins, 4) for HLCV or
            (n_timesteps, n_coins, 5) for OHLCV (open, high, low, close, volume).
        interval: Number of 1m candles to combine (e.g., 5 for 5m candles)

    Returns:
        Aggregated array of shape (n_timesteps // interval, n_coins, 4) in HLCV order.
    """
    if interval <= 1:
        return candles_1m
    if candles_1m.ndim != 3:
        raise ValueError(f"Expected 3D candle array, got ndim={candles_1m.ndim}")
    if candles_1m.shape[-1] == 5:
        # Drop open; keep HLCV
        candles = candles_1m[:, :, 1:]
    elif candles_1m.shape[-1] == 4:
        candles = candles_1m
    else:
        raise ValueError(f"Expected HLCV/OHLCV last-dim 4 or 5, got {candles_1m.shape[-1]}")
    n_timesteps = candles.shape[0]
    n_out = n_timesteps // interval
    if n_out == 0:
        raise ValueError(f"Not enough candles ({n_timesteps}) for interval {interval}")
    truncated = candles[: n_out * interval]
    reshaped = truncated.reshape(n_out, interval, *candles.shape[1:])
    # HLCV indices: 0=high, 1=low, 2=close, 3=volume
    aggregated = np.stack(
        [
            reshaped[:, :, :, 0].max(axis=1),  # high: max across interval
            reshaped[:, :, :, 1].min(axis=1),  # low: min across interval
            reshaped[:, -1, :, 2],  # close: last candle's close
            reshaped[:, :, :, 3].sum(axis=1),  # volume: sum across interval
        ],
        axis=-1,
    )
    return aggregated


def align_and_aggregate_hlcvs(
    hlcvs: np.ndarray,
    timestamps: np.ndarray | None,
    btc_usd_prices: np.ndarray | None,
    interval: int,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, int]:
    """
    Align candles to interval boundaries (by trimming leading bars) and aggregate.

    Returns aggregated HLCVs, timestamps, BTC/USD prices, and number of 1m bars trimmed.
    """
    if interval <= 1:
        return hlcvs, timestamps, btc_usd_prices, 0
    offset_bars = 0
    if timestamps is not None and len(timestamps) > 0:
        interval_ms = interval * 60_000
        first_ts = int(timestamps[0])
        remainder = first_ts % interval_ms
        if remainder != 0:
            offset_ms = interval_ms - remainder
            offset_bars = int(offset_ms // 60_000)
            if offset_bars >= hlcvs.shape[0]:
                raise ValueError(
                    f"Not enough candles to align to interval {interval} (offset {offset_bars})"
                )
            hlcvs = hlcvs[offset_bars:]
            timestamps = timestamps[offset_bars:]
            if btc_usd_prices is not None and len(btc_usd_prices) >= offset_bars:
                btc_usd_prices = btc_usd_prices[offset_bars:]
    hlcvs_agg = aggregate_hlcvs(hlcvs, interval)
    n_out = hlcvs_agg.shape[0]
    ts_agg = None
    btc_agg = None
    if timestamps is not None:
        ts_agg = timestamps[::interval][:n_out]
    if btc_usd_prices is not None:
        btc_agg = btc_usd_prices[interval - 1 :: interval][:n_out]
    return hlcvs_agg, ts_agg, btc_agg, offset_bars


__all__ = [
    "attempt_gap_fix_ohlcvs",
    "canonicalize_daily_ohlcvs",
    "date_to_ts",
    "deduplicate_rows",
    "dump_daily_ohlcv_data",
    "dump_ohlcv_data",
    "ensure_millis_df",
    "fill_gaps_in_ohlcvs",
    "format_end_date",
    "get_days_in_between",
    "load_ohlcv_data",
]
