import logging

import numpy as np

import backtest as backtest_module
from config_utils import get_template_config
from backtest import (
    build_backtest_payload,
    get_backtest_execution_settings,
    log_backtest_execution_settings,
    prep_backtest_args,
)


def _base_config():
    cfg = get_template_config()
    cfg["backtest"]["coins"] = {"binance": ["BTC/USDT:USDT"]}
    cfg["backtest"]["maker_fee_override"] = None
    cfg["backtest"]["taker_fee_override"] = None
    return cfg


def _base_mss():
    return {
        "BTC/USDT:USDT": {
            "maker": 0.0001,
            "taker": 0.0005,
            "qty_step": 0.001,
            "price_step": 0.1,
            "min_qty": 0.001,
            "min_cost": 10.0,
            "c_mult": 1.0,
        }
    }


def test_prep_backtest_args_uses_exchange_maker_fee_when_no_override():
    config = _base_config()
    mss = _base_mss()
    _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["maker_fee"] == 0.0001
    assert backtest_params["taker_fee"] == 0.0005


def test_prep_backtest_args_uses_maker_fee_override_when_set():
    config = _base_config()
    config["backtest"]["maker_fee_override"] = 0.0002
    mss = _base_mss()
    _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["maker_fee"] == 0.0002

def test_prep_backtest_args_uses_taker_fee_override_when_set():
    config = _base_config()
    config["backtest"]["taker_fee_override"] = 0.0008
    mss = _base_mss()
    _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["taker_fee"] == 0.0008


def test_prep_backtest_args_passes_market_order_slippage_pct():
    config = _base_config()
    config["backtest"]["market_order_slippage_pct"] = 0.0015
    mss = _base_mss()
    _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["market_order_slippage_pct"] == 0.0015


def test_prep_backtest_args_passes_market_order_near_touch_threshold_from_live():
    config = _base_config()
    config["live"]["market_order_near_touch_threshold"] = 0.0017
    mss = _base_mss()
    _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["market_order_near_touch_threshold"] == 0.0017


def test_prep_backtest_args_uses_market_orders_allowed_from_live():
    config = _base_config()
    config["live"]["market_orders_allowed"] = True
    mss = _base_mss()
    _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["market_orders_allowed"] is True


def test_prep_backtest_args_uses_pnls_max_lookback_days_from_live():
    config = _base_config()
    config["live"]["pnls_max_lookback_days"] = 30.0
    mss = _base_mss()
    _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["pnls_max_lookback_days"] == 30.0


def test_prep_backtest_args_encodes_all_pnls_lookback_for_backtest():
    config = _base_config()
    config["live"]["pnls_max_lookback_days"] = "all"
    mss = _base_mss()
    _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["pnls_max_lookback_days"] == -1.0


def test_prep_backtest_args_passes_liquidation_threshold():
    config = _base_config()
    config["backtest"]["liquidation_threshold"] = 0.07
    mss = _base_mss()
    _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["liquidation_threshold"] == 0.07


def test_prep_backtest_args_passes_dynamic_wel_by_tradability_flag():
    config = _base_config()
    mss = _base_mss()

    _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["dynamic_wel_by_tradability"] is True

    config["backtest"]["dynamic_wel_by_tradability"] = False
    _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["dynamic_wel_by_tradability"] is False


def test_prep_backtest_args_does_not_log_execution_settings_by_default(caplog):
    config = _base_config()
    mss = _base_mss()

    with caplog.at_level(logging.INFO):
        prep_backtest_args(config, mss, "binance")

    assert "[backtest] effective execution settings:" not in caplog.text


def test_log_backtest_execution_settings_emits_summary(caplog):
    config = _base_config()
    execution_settings = get_backtest_execution_settings(config)

    with caplog.at_level(logging.INFO):
        log_backtest_execution_settings(execution_settings)

    assert "[backtest] effective execution settings:" in caplog.text
    assert "market_orders_allowed" in caplog.text
    assert "pnls_max_lookback_days" in caplog.text


def test_build_backtest_payload_compiles_runtime_config_once(monkeypatch):
    config = _base_config()
    mss = _base_mss()
    hlcvs = np.array([[[101.0, 99.0, 100.0, 1.0]]], dtype=np.float64)
    btc_usd_prices = np.array([20000.0], dtype=np.float64)
    timestamps = np.array([0], dtype=np.int64)
    call_count = {"count": 0}
    original = backtest_module.compile_runtime_config

    def counting_compile_runtime_config(*args, **kwargs):
        call_count["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(backtest_module, "compile_runtime_config", counting_compile_runtime_config)

    build_backtest_payload(hlcvs, mss, config, "binance", btc_usd_prices, timestamps)

    assert call_count["count"] == 1
