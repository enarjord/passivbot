from config_utils import get_template_config
from backtest import prep_backtest_args


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
    _, _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["maker_fee"] == 0.0001
    assert backtest_params["taker_fee"] == 0.0005


def test_prep_backtest_args_uses_maker_fee_override_when_set():
    config = _base_config()
    config["backtest"]["maker_fee_override"] = 0.0002
    mss = _base_mss()
    _, _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["maker_fee"] == 0.0002


def test_prep_backtest_args_uses_taker_fee_override_when_set():
    config = _base_config()
    config["backtest"]["taker_fee_override"] = 0.0008
    mss = _base_mss()
    _, _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["taker_fee"] == 0.0008


def test_prep_backtest_args_passes_market_order_slippage_pct():
    config = _base_config()
    config["backtest"]["market_order_slippage_pct"] = 0.0015
    mss = _base_mss()
    _, _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["market_order_slippage_pct"] == 0.0015


def test_prep_backtest_args_passes_market_order_near_touch_threshold_from_live():
    config = _base_config()
    config["live"]["market_order_near_touch_threshold"] = 0.0017
    mss = _base_mss()
    _, _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["market_order_near_touch_threshold"] == 0.0017


def test_prep_backtest_args_passes_liquidation_threshold():
    config = _base_config()
    config["backtest"]["liquidation_threshold"] = 0.07
    mss = _base_mss()
    _, _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["liquidation_threshold"] == 0.07


def test_prep_backtest_args_passes_dynamic_wel_by_tradability_flag():
    config = _base_config()
    mss = _base_mss()

    _, _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["dynamic_wel_by_tradability"] is True

    config["backtest"]["dynamic_wel_by_tradability"] = False
    _, _, _, backtest_params = prep_backtest_args(config, mss, "binance")
    assert backtest_params["dynamic_wel_by_tradability"] is False


def test_prep_backtest_args_injects_dynamic_wallet_exposure_sentinel_without_coin_override():
    config = _base_config()
    mss = _base_mss()

    bot_params_list, _, _, _ = prep_backtest_args(config, mss, "binance")

    assert len(bot_params_list) == 1
    assert bot_params_list[0]["long"]["wallet_exposure_limit"] == -1.0
    assert bot_params_list[0]["short"]["wallet_exposure_limit"] == -1.0


def test_prep_backtest_args_preserves_explicit_coin_wallet_exposure_override():
    config = _base_config()
    config["coin_overrides"] = {
        "BTC/USDT:USDT": {
            "bot": {
                "long": {"wallet_exposure_limit": 0.25},
                "short": {"wallet_exposure_limit": 0.5},
            }
        }
    }
    mss = _base_mss()

    bot_params_list, _, _, _ = prep_backtest_args(config, mss, "binance")

    assert len(bot_params_list) == 1
    assert bot_params_list[0]["long"]["wallet_exposure_limit"] == 0.25
    assert bot_params_list[0]["short"]["wallet_exposure_limit"] == 0.5


def test_prep_backtest_args_uses_canonical_strategy_params_for_runtime_payload():
    config = _base_config()
    config["bot"]["long"]["ema_span_0"] = -1.0
    config["bot"]["short"]["entry_grid_spacing_pct"] = -1.0
    config["bot"]["long"]["strategy"]["trailing_grid"]["ema_span_0"] = 321.0
    config["bot"]["short"]["strategy"]["trailing_grid"]["entry_grid_spacing_pct"] = 0.0123
    config["bot"]["long"]["strategy"]["trailing_grid"]["grid_close_price_anchor"] = "ema_band"
    config["bot"]["short"]["strategy"]["trailing_grid"]["grid_close_price_anchor"] = "pprice"
    mss = _base_mss()

    bot_params_list, strategy_params_list, _, _ = prep_backtest_args(config, mss, "binance")

    assert len(bot_params_list) == 1
    assert "ema_span_0" not in bot_params_list[0]["long"]
    assert "entry_grid_spacing_pct" not in bot_params_list[0]["short"]
    assert strategy_params_list[0]["long"]["ema_span_0"] == 321.0
    assert strategy_params_list[0]["short"]["entry_grid_spacing_pct"] == 0.0123
    assert strategy_params_list[0]["long"]["grid_close_price_anchor"] == "ema_band_upper"
    assert strategy_params_list[0]["short"]["grid_close_price_anchor"] == "position_price"


def test_prep_backtest_args_emits_separate_ema_anchor_strategy_payload():
    config = _base_config()
    config["live"]["strategy_kind"] = "ema_anchor"
    config["bot"]["long"]["strategy"]["ema_anchor"] = {
            "base_qty_pct": 0.02,
            "ema_span_0": 55.0,
            "ema_span_1": 144.0,
            "offset": 0.003,
            "offset_volatility_ema_span_minutes": 30.0,
            "offset_volatility_1m_weight": 2.5,
            "entry_volatility_ema_span_hours": 12.0,
            "offset_volatility_1h_weight": 1.75,
            "offset_psize_weight": 0.2,
    }
    config["bot"]["short"]["strategy"]["ema_anchor"] = {
            "base_qty_pct": 0.03,
            "ema_span_0": 34.0,
            "ema_span_1": 89.0,
            "offset": 0.004,
            "offset_volatility_ema_span_minutes": 45.0,
            "offset_volatility_1m_weight": 3.5,
            "entry_volatility_ema_span_hours": 18.0,
            "offset_volatility_1h_weight": 0.5,
            "offset_psize_weight": 0.1,
    }
    mss = _base_mss()

    bot_params_list, strategy_params_list, _, backtest_params = prep_backtest_args(
        config, mss, "binance"
    )

    assert backtest_params["strategy_kind"] == "ema_anchor"
    assert len(bot_params_list) == 1
    assert len(strategy_params_list) == 1
    assert "base_qty_pct" not in bot_params_list[0]["long"]
    assert "offset_volatility_ema_span_minutes" not in bot_params_list[0]["long"]
    assert strategy_params_list[0]["long"]["base_qty_pct"] == 0.02
    assert strategy_params_list[0]["long"]["offset_volatility_ema_span_minutes"] == 30.0
    assert strategy_params_list[0]["long"]["offset_volatility_1m_weight"] == 2.5
    assert strategy_params_list[0]["short"]["entry_volatility_ema_span_hours"] == 18.0
    assert strategy_params_list[0]["short"]["offset_volatility_1h_weight"] == 0.5
    assert strategy_params_list[0]["short"]["offset"] == 0.004
