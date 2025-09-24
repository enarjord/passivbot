import json
import numpy as np

from backtest import ensure_valid_index_metadata


def test_ensure_valid_index_metadata_infers_ranges_for_each_coin():
    hlcvs = np.full((6, 2, 4), np.nan, dtype=np.float64)

    # coin 0 valid at indices 1..3
    hlcvs[1:4, 0, 0:3] = [[2.0, 1.0, 1.5], [2.2, 1.1, 1.6], [2.1, 1.05, 1.55]]
    hlcvs[1:4, 0, 3] = [10.0, 11.0, 12.0]

    # coin 1 valid at indices 0..4 (last index remains NaN)
    for i in range(5):
        hlcvs[i, 1, 0:3] = [4.0 + i * 0.1, 3.0 + i * 0.1, 3.5 + i * 0.1]
        hlcvs[i, 1, 3] = 5.0 + i

    coins = ["coin0", "coin1"]
    mss = {coin: {} for coin in coins}

    ensure_valid_index_metadata(mss, hlcvs, coins)

    assert mss["coin0"]["first_valid_index"] == 1
    assert mss["coin0"]["last_valid_index"] == 3

    assert mss["coin1"]["first_valid_index"] == 0
    assert mss["coin1"]["last_valid_index"] == 4


def test_ensure_valid_index_metadata_preserves_existing_values():
    hlcvs = np.zeros((3, 1, 4), dtype=np.float64)
    coins = ["coinX"]
    mss = {"coinX": {"first_valid_index": 2, "last_valid_index": 2}}

    ensure_valid_index_metadata(mss, hlcvs, coins)

    # values should remain unchanged
    assert mss["coinX"]["first_valid_index"] == 2
    assert mss["coinX"]["last_valid_index"] == 2


def test_ensure_valid_index_metadata_handles_all_nan_coin():
    hlcvs = np.full((5, 1, 4), np.nan, dtype=np.float64)
    coins = ["coin_nan"]
    mss = {"coin_nan": {}}

    ensure_valid_index_metadata(mss, hlcvs, coins)

    # first index becomes length of series, last index equals first (no valid window)
    assert mss["coin_nan"]["first_valid_index"] == hlcvs.shape[0]
    assert mss["coin_nan"]["last_valid_index"] == hlcvs.shape[0]
