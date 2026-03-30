from config_utils import get_template_config
from backtest import prep_backtest_args


def _base_config():
    cfg = get_template_config()
    cfg["backtest"]["coins"] = {"binance": ["BTC/USDT:USDT"]}
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
