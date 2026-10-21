"""Supported strategy metadata and retained diagnostic API after v7 retirement."""

import json

import pytest


@pytest.fixture(scope="module")
def require_real_passivbot_rust_module():
    import passivbot_rust as native

    if getattr(native, "__is_stub__", False):
        pytest.skip("requires the compiled Rust extension")
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    return native


def test_native_registry_and_metadata_expose_only_supported_strategies(
    require_real_passivbot_rust_module,
):
    native = require_real_passivbot_rust_module
    assert native.get_strategy_kinds() == ["trailing_martingale", "ema_anchor"]
    for name in native.get_strategy_kinds():
        assert native.get_strategy_spec(name)["strategy_kind"] == name
    with pytest.raises(ValueError, match="unknown strategy kind"):
        native.get_strategy_spec("trailing_grid_v7")
    assert not hasattr(native, "calc_trailing_grid_v7_diagnostic_py")


@pytest.mark.parametrize("side,sign", [("long", 1.0), ("short", -1.0)])
def test_martingale_close_diagnostics_remain_available(
    require_real_passivbot_rust_module, side, sign,
):
    payload = {"minimum_coin_age_days": 0.0,
        "pside": side,
        "exchange": {
            "qty_step": 0.01, "price_step": 0.01, "min_qty": 0.01,
            "min_cost": 1.0, "c_mult": 1.0, "maker_fee": 0.0, "taker_fee": 0.0,
        },
        "state": {
            "balance": 1000.0, "order_book": {"bid": 110.0, "ask": 110.1},
            "ema_bands": {"lower": 100.0, "upper": 100.0},
            "volatility_ema_1m": 0.0, "volatility_ema_1h": 0.0,
        },
        "bot_params": {
            "wallet_exposure_limit": 1.0, "total_wallet_exposure_limit": 1.0,
            "n_positions": 1,
        },
        "runtime": {"effective_wallet_exposure_limit": 1.0},
        "position": {"size": sign * 5.0, "price": 100.0},
        "trailing": {
            "min_since_open": 100.0, "max_since_min": 110.0,
            "max_since_open": 110.0, "min_since_max": 100.0,
        },
        "close_params": {"qty_pct": 0.25, "threshold_base_pct": 0.01},
    }
    result = json.loads(require_real_passivbot_rust_module.calc_trailing_martingale_close_diagnostic_py(
        json.dumps(payload)
    ))
    assert result["kind"] == "close"
    assert result["pside"] == side
    assert result["qty"] * sign <= 0.0
    assert result["threshold_terms"]["base"] == 0.01
