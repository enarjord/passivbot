import json
import sys
import types

import numpy as np

sys.modules.setdefault("passivbot_rust", types.SimpleNamespace())

from backtest import ensure_valid_index_metadata
from downloader import compute_per_coin_warmup_minutes


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
    assert mss["coin0"].get("warmup_minutes", 0) == 0
    assert mss["coin0"].get("trade_start_index", 1) == 1

    assert mss["coin1"]["first_valid_index"] == 0
    assert mss["coin1"]["last_valid_index"] == 4
    assert mss["coin1"].get("trade_start_index", 0) == 0


def test_ensure_valid_index_metadata_preserves_existing_values():
    hlcvs = np.zeros((3, 1, 4), dtype=np.float64)
    coins = ["coinX"]
    mss = {"coinX": {"first_valid_index": 2, "last_valid_index": 2}}

    ensure_valid_index_metadata(mss, hlcvs, coins)

    # values should remain unchanged
    assert mss["coinX"]["first_valid_index"] == 2
    assert mss["coinX"]["last_valid_index"] == 2
    assert mss["coinX"].get("trade_start_index") == 2


def test_ensure_valid_index_metadata_handles_all_nan_coin():
    hlcvs = np.full((5, 1, 4), np.nan, dtype=np.float64)
    coins = ["coin_nan"]
    mss = {"coin_nan": {}}

    ensure_valid_index_metadata(mss, hlcvs, coins)

    no_valid_first = mss["coin_nan"]["first_valid_index"]
    assert no_valid_first == hlcvs.shape[0]
    assert mss["coin_nan"]["last_valid_index"] == hlcvs.shape[0]
    assert mss["coin_nan"].get("trade_start_index", no_valid_first) == no_valid_first


def test_ensure_valid_index_metadata_applies_warmup_map():
    hlcvs = np.full((10, 1, 4), np.nan, dtype=np.float64)
    hlcvs[2:8, 0, 0:3] = 1.0
    coins = ["coinwarm"]
    mss = {"coinwarm": {}}
    ensure_valid_index_metadata(mss, hlcvs, coins, {"coinwarm": 3})
    assert mss["coinwarm"]["first_valid_index"] == 2
    assert mss["coinwarm"]["warmup_minutes"] == 3
    assert mss["coinwarm"]["trade_start_index"] == 5


def test_ensure_valid_index_metadata_default_warmup_fallback():
    hlcvs = np.full((8, 1, 4), np.nan, dtype=np.float64)
    hlcvs[1:5, 0, 0:3] = 2.0
    coins = ["fallback"]
    mss = {"fallback": {}}
    ensure_valid_index_metadata(mss, hlcvs, coins, {"__default__": 4})
    assert mss["fallback"]["first_valid_index"] == 1
    assert mss["fallback"]["warmup_minutes"] == 4
    assert mss["fallback"]["trade_start_index"] == 4


def test_compute_per_coin_warmup_minutes_handles_overrides():
    config = {
        "backtest": {"max_warmup_minutes": 0.0},
        "live": {"warmup_ratio": 0.1},
        "bot": {
            "long": {"ema_span_0": 50},
            "short": {},
        },
        "coin_overrides": {
            "ALTC": {
                "bot": {
                    "long": {"ema_span_0": 200},
                }
            }
        },
    }
    result = compute_per_coin_warmup_minutes(config)
    assert result["__default__"] == 5
    assert result["ALTC"] == 20
