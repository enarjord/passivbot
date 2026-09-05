import os

import numpy as np
import pytest

from ohlcv_utils import load_ohlcv_data


class _PickleMarker:
    def __init__(self, path):
        self.path = str(path)

    def __reduce__(self):
        return os.mkdir, (self.path,)


def test_candle_object_array_is_rejected_without_executing_pickle(tmp_path):
    marker = tmp_path / "pickle-executed"
    path = tmp_path / "candles.npy"
    np.save(path, np.array([_PickleMarker(marker)], dtype=object))

    with pytest.raises(ValueError, match="allow_pickle=False"):
        load_ohlcv_data(str(path))

    assert not marker.exists()


def test_numeric_candle_array_still_loads_and_deduplicates(tmp_path):
    path = tmp_path / "candles.npy"
    rows = np.array([
        [1700000040000, 100, 102, 99, 101, 10],
        [1700000040000, 100, 102, 99, 101, 10],
        [1700000100000, 101, 103, 100, 102, 20],
    ], dtype=np.float64)
    np.save(path, rows)

    loaded = load_ohlcv_data(str(path))

    assert list(loaded.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
    np.testing.assert_array_equal(loaded.to_numpy(), rows[[0, 2]])
    np.testing.assert_array_equal(np.load(path, allow_pickle=False), rows[[0, 2]])
