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


def test_prep_backtest_args_uses_maker_fee_override_when_set():
    config = _base_config()
    config["backtest"]["maker_fee_override"] = 0.0002
    mss = _base_mss()
    _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["maker_fee"] == 0.0002
