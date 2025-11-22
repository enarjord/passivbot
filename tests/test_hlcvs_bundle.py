import numpy as np
import pytest

pbr = pytest.importorskip("passivbot_rust")

if not hasattr(pbr, "HlcvsBundle"):
    pytest.skip("passivbot_rust built without HlcvsBundle", allow_module_level=True)

from backtest import (
    BacktestPayload,
    subset_backtest_payload,
    _build_hlcvs_bundle,
)


def _base_meta():
    return {
        "requested_start_timestamp_ms": 0,
        "effective_start_timestamp_ms": 0,
        "warmup_minutes_requested": 0,
        "warmup_minutes_provided": 0,
        "coins": [
            {
                "index": 0,
                "symbol": "BTC/USDT:USDT",
                "coin": "BTC",
                "exchange": "binanceusdm",
                "quote": "USDT",
                "base": "BTC",
                "qty_step": 0.001,
                "price_step": 0.1,
                "min_qty": 0.001,
                "min_cost": 10.0,
                "c_mult": 1.0,
                "maker_fee": 0.0002,
                "taker_fee": 0.0004,
                "first_valid_index": 0,
                "last_valid_index": 1,
                "warmup_minutes": 0,
                "trade_start_index": 0,
            }
        ],
    }


def test_hlcvs_bundle_roundtrip_meta():
    hlcvs = np.zeros((2, 1, 4), dtype=np.float64)
    btc = np.zeros(2, dtype=np.float64)
    timestamps = np.array([0, 60_000], dtype=np.int64)
    bundle = pbr.HlcvsBundle(hlcvs, btc, timestamps, _base_meta())
    assert bundle.coins_len() == 1
    meta = bundle.meta
    assert meta["coins"][0]["symbol"] == "BTC/USDT:USDT"
    btc_meta = bundle.coin_meta("BTC")
    assert btc_meta is not None
    assert pytest.approx(btc_meta["qty_step"]) == 0.001


def test_hlcvs_bundle_validates_shapes():
    hlcvs = np.zeros((2, 1, 4), dtype=np.float64)
    btc = np.zeros(2, dtype=np.float64)
    timestamps = np.array([0], dtype=np.int64)
    with pytest.raises(ValueError):
        pbr.HlcvsBundle(hlcvs, btc, timestamps, _base_meta())


def _payload_two_coins():
    hlcvs = np.zeros((2, 2, 4), dtype=np.float64)
    btc = np.zeros(2, dtype=np.float64)
    timestamps = np.array([0, 60_000], dtype=np.int64)
    meta = _base_meta()
    coin2 = meta["coins"][0].copy()
    coin2.update(
        {
            "index": 1,
            "symbol": "ETH/USDT:USDT",
            "coin": "ETH",
            "base": "ETH",
        }
    )
    meta["coins"].append(coin2)
    bundle = pbr.HlcvsBundle(hlcvs, btc, timestamps, meta)
    backtest_params = {
        "coins": ["BTC/USDT:USDT", "ETH/USDT:USDT"],
        "first_valid_indices": [0, 0],
        "last_valid_indices": [1, 1],
        "warmup_minutes": [0, 0],
        "trade_start_indices": [0, 0],
        "requested_start_timestamp_ms": 0,
        "first_timestamp_ms": 0,
    }
    return BacktestPayload(
        bundle=bundle,
        bot_params_list=["bp0", "bp1"],
        exchange_params=["ex0", "ex1"],
        backtest_params=backtest_params,
    )


def test_subset_backtest_payload_by_index():
    payload = _payload_two_coins()
    subset = subset_backtest_payload(payload, coin_indices=[1])
    assert subset.bundle.meta["coins"][0]["symbol"] == "ETH/USDT:USDT"
    assert subset.bundle.hlcvs.shape[1] == 1
    assert subset.bot_params_list == ["bp1"]
    assert subset.exchange_params == ["ex1"]
    assert subset.backtest_params["coins"] == ["ETH/USDT:USDT"]


def test_build_hlcvs_bundle_applies_coin_indices():
    hlcvs = np.zeros((3, 5, 4), dtype=np.float64)
    btc = np.zeros(3, dtype=np.float64)
    timestamps = np.array([0, 60_000, 120_000], dtype=np.int64)
    config = {
        "backtest": {"start_date": "2021-01-01", "end_date": "2021-01-02"},
        "live": {"warmup_ratio": 0.0, "max_warmup_minutes": 0},
        "bot": {"long": {}, "short": {}},
    }
    coins_order = ["COIN_A", "COIN_B", "COIN_C"]
    mss = {coin: {"symbol": f"{coin}/USDT:USDT"} for coin in coins_order}
    first_valid = [0, 0, 0]
    last_valid = [2, 2, 2]
    warmup = [0, 0, 0]
    trade_start = [0, 0, 0]
    bundle = _build_hlcvs_bundle(
        hlcvs,
        btc,
        timestamps,
        config,
        "binanceusdm",
        mss,
        coins_order,
        first_valid,
        last_valid,
        warmup,
        trade_start,
        requested_start_ts=0,
        coin_indices=[0, 2, 4],
    )
    assert bundle.hlcvs.shape[1] == 3
    assert len(bundle.meta["coins"]) == 3
