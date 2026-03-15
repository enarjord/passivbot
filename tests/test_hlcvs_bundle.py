import numpy as np
import pytest

pbr = pytest.importorskip("passivbot_rust")

if not hasattr(pbr, "HlcvsBundle"):
    pytest.skip("passivbot_rust built without HlcvsBundle", allow_module_level=True)

from backtest import (
    BacktestPayload,
    build_backtest_payload,
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


def test_build_hlcvs_bundle_reuses_contiguous_arrays_without_copy():
    hlcvs = np.zeros((3, 2, 4), dtype=np.float64)
    btc = np.zeros(3, dtype=np.float64)
    timestamps = np.array([0, 60_000, 120_000], dtype=np.int64)
    config = {
        "backtest": {"start_date": "2021-01-01", "end_date": "2021-01-02"},
        "live": {"warmup_ratio": 0.0, "max_warmup_minutes": 0},
        "bot": {"long": {}, "short": {}},
    }
    coins_order = ["COIN_A", "COIN_B"]
    mss = {coin: {"symbol": f"{coin}/USDT:USDT"} for coin in coins_order}
    bundle = _build_hlcvs_bundle(
        hlcvs,
        btc,
        timestamps,
        config,
        "binanceusdm",
        mss,
        coins_order,
        [0, 0],
        [2, 2],
        [0, 0],
        [0, 0],
        requested_start_ts=0,
    )
    assert bundle.hlcvs is hlcvs
    assert bundle.btc_usd is btc
    assert bundle.timestamps is timestamps


def test_build_hlcvs_bundle_copies_non_contiguous_subset():
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
    bundle = _build_hlcvs_bundle(
        hlcvs,
        btc,
        timestamps,
        config,
        "binanceusdm",
        mss,
        coins_order,
        [0, 0, 0],
        [2, 2, 2],
        [0, 0, 0],
        [0, 0, 0],
        requested_start_ts=0,
        coin_indices=[0, 2, 4],
    )
    assert bundle.hlcvs is not hlcvs
    assert bundle.hlcvs.flags.c_contiguous


def test_build_hlcvs_bundle_reuses_full_coin_axis_with_active_coin_indices():
    hlcvs = np.zeros((3, 5, 4), dtype=np.float64)
    btc = np.zeros(3, dtype=np.float64)
    timestamps = np.array([0, 60_000, 120_000], dtype=np.int64)
    config = {
        "backtest": {"start_date": "2021-01-01", "end_date": "2021-01-02"},
        "live": {"warmup_ratio": 0.0, "max_warmup_minutes": 0},
        "bot": {"long": {}, "short": {}},
    }
    selected_coins = ["COIN_A", "COIN_C", "COIN_E"]
    bundle_coins = ["COIN_A", "COIN_B", "COIN_C", "COIN_D", "COIN_E"]
    mss = {
        coin: {
            "symbol": f"{coin}/USDT:USDT",
            "first_valid_index": 0,
            "last_valid_index": 2,
            "warmup_minutes": 0,
            "trade_start_index": 0,
        }
        for coin in bundle_coins
    }
    mss["__meta__"] = {"bundle_coins_order": bundle_coins}
    bundle = _build_hlcvs_bundle(
        hlcvs,
        btc,
        timestamps,
        config,
        "binanceusdm",
        mss,
        selected_coins,
        [0, 0, 0],
        [2, 2, 2],
        [0, 0, 0],
        [0, 0, 0],
        requested_start_ts=0,
        coin_indices=[0, 2, 4],
    )
    assert bundle.hlcvs is hlcvs
    assert bundle.btc_usd is btc
    assert bundle.timestamps is timestamps
    assert bundle.hlcvs.shape[1] == 5
    assert len(bundle.meta["coins"]) == 5
    assert bundle.meta["coins"][2]["symbol"] == "COIN_C/USDT:USDT"


def test_build_backtest_payload_sets_original_active_coin_indices_for_bundle_coin_order():
    hlcvs = np.zeros((3, 5, 4), dtype=np.float64)
    btc = np.zeros(3, dtype=np.float64)
    timestamps = np.array([0, 60_000, 120_000], dtype=np.int64)
    selected_coins = ["COIN_A", "COIN_C", "COIN_E"]
    bundle_coins = ["COIN_A", "COIN_B", "COIN_C", "COIN_D", "COIN_E"]
    config = {
        "backtest": {
            "start_date": "2021-01-01",
            "end_date": "2021-01-02",
            "coins": {"binanceusdm": selected_coins},
            "starting_balance": 1000.0,
            "btc_collateral_cap": 0.0,
            "btc_collateral_ltv_cap": None,
            "filter_by_min_effective_cost": False,
            "dynamic_wel_by_tradability": False,
        },
        "live": {
            "warmup_ratio": 0.0,
            "max_warmup_minutes": 0,
            "hedge_mode": True,
            "max_realized_loss_pct": 1.0,
            "pnls_max_lookback_days": 0.0,
            "market_orders_allowed": False,
            "market_order_near_touch_threshold": 0.001,
        },
        "bot": {
            "long": {"wallet_exposure_limit": 0.0, "total_wallet_exposure_limit": 1.0, "n_positions": 3},
            "short": {"wallet_exposure_limit": 0.0, "total_wallet_exposure_limit": 1.0, "n_positions": 3},
            "common": {
                "equity_hard_stop_loss": {
                    "enabled": False,
                    "red_threshold": 0.1,
                    "ema_span_minutes": 60.0,
                    "cooldown_minutes_after_red": 0.0,
                    "no_restart_drawdown_threshold": 1.0,
                    "tier_ratios": {"yellow": 0.5, "orange": 0.75},
                    "orange_tier_mode": "tp_only_with_active_entry_cancellation",
                    "panic_close_order_type": "limit",
                }
            },
        },
        "coin_overrides": {},
    }
    mss = {
        coin: {
            "symbol": f"{coin}/USDT:USDT",
            "qty_step": 0.001,
            "price_step": 0.1,
            "min_qty": 0.001,
            "min_cost": 10.0,
            "c_mult": 1.0,
            "maker": 0.0002,
            "taker": 0.00055,
            "first_valid_index": 0,
            "last_valid_index": 2,
            "warmup_minutes": 0,
            "trade_start_index": 0,
        }
        for coin in bundle_coins
    }
    mss["__meta__"] = {"bundle_coins_order": bundle_coins}
    payload = build_backtest_payload(
        hlcvs,
        mss,
        config,
        "binanceusdm",
        btc,
        timestamps,
        coin_indices=[0, 2, 4],
    )
    assert payload.backtest_params["active_coin_indices"] == [0, 2, 4]
