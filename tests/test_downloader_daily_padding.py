import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import downloader as dl


def _build_df(start_ts, rows):
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    data = [
        {
            "timestamp": start_ts + offset * 60_000,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
        for offset, open_, high, low, close, volume in rows
    ]
    return pd.DataFrame(data, columns=columns)


def test_canonicalize_daily_ohlcvs_inserts_missing_minutes():
    day = "2024-01-01"
    start_ts = dl.date_to_ts(day)
    df = _build_df(
        start_ts,
        [
            (0, 100.0, 101.0, 99.5, 100.5, 1.0),
            (2, 101.0, 102.0, 99.0, 100.0, 2.0),  # minute 1 missing
        ],
    )

    canonical = dl.canonicalize_daily_ohlcvs(dl.ensure_millis(df), start_ts)

    assert len(canonical) == 1440
    ts_missing = start_ts + 60_000
    row_missing = canonical[canonical["timestamp"] == ts_missing].iloc[0]
    assert row_missing["volume"] == pytest.approx(0.0)
    assert (
        row_missing["open"]
        == row_missing["high"]
        == row_missing["low"]
        == row_missing["close"]
        == pytest.approx(100.5)
    )

    last_ts = start_ts + 1439 * 60_000
    last_row = canonical[canonical["timestamp"] == last_ts].iloc[0]
    assert last_row["close"] == pytest.approx(100.0)
    assert last_row["volume"] == pytest.approx(0.0)


def test_dump_daily_ohlcv_data_writes_canonical(tmp_path):
    day = "2024-02-02"
    start_ts = dl.date_to_ts(day)
    df = _build_df(
        start_ts,
        [
            (0, 10.0, 11.0, 9.0, 10.5, 5.0),
            (1, 11.0, 12.0, 10.5, 11.5, 4.0),
            (3, 12.0, 12.5, 11.8, 12.1, 3.0),
            (3, 13.0, 13.0, 12.0, 12.5, 1.5),  # duplicate timestamp, expect last row kept
        ],
    )
    out_path = tmp_path / "2024-02-02.npy"

    dl.dump_daily_ohlcv_data(dl.ensure_millis(df), out_path, start_ts)

    arr = np.load(out_path)
    assert arr.shape == (1440, 6)

    ts_missing = start_ts + 2 * 60_000
    idx_missing = int((ts_missing - start_ts) // 60_000)
    missing_row = arr[idx_missing]
    assert missing_row[0] == ts_missing
    assert missing_row[5] == pytest.approx(0.0)
    assert missing_row[1] == missing_row[2] == missing_row[3] == missing_row[4] == pytest.approx(11.5)

    idx_duplicate = int((start_ts + 3 * 60_000 - start_ts) // 60_000)
    duplicate_row = arr[idx_duplicate]
    assert duplicate_row[4] == pytest.approx(12.5)
    assert duplicate_row[5] == pytest.approx(1.5)


def test_canonicalize_handles_timestamp_index():
    day = "2024-03-01"
    start_ts = dl.date_to_ts(day)
    df = _build_df(
        start_ts,
        [
            (0, 100.0, 101.0, 99.0, 100.5, 1.0),
            (2, 101.0, 102.0, 100.0, 101.5, 2.0),
        ],
    )
    indexed = df.set_index("timestamp")
    indexed["timestamp"] = indexed.index

    canonical = dl.canonicalize_daily_ohlcvs(indexed, start_ts)
    assert len(canonical) == 1440
