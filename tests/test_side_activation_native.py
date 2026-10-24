"""Exercise the explicit activation permission at the real native JSON boundary."""

from copy import deepcopy

import pytest

from test_orchestrator_json_api import (
    adaptive_strategy_params, bot_params_pair, compute, make_input, make_symbol,
)


def activation_snapshot(kind, side, *, held=False, mode="normal"):
    policy = {
        "n_positions": 1, "total_wallet_exposure_limit": 1.0,
        "wallet_exposure_limit": 1.0, "entry_initial_qty_pct": 0.1,
        "entry_grid_double_down_factor": 1.2, "entry_grid_spacing_pct": 0.01,
        "close_grid_qty_pct": 0.5, "close_trailing_threshold_pct": 0.01,
    }
    strategy = (
        adaptive_strategy_params(
            entry={"initial_qty_pct": 0.1, "double_down_factor": 1.2,
                   "threshold_base_pct": 0.01},
            close={"qty_pct": 0.5, "threshold_base_pct": 0.01},
        ) if kind == "trailing_martingale" else
        {"base_qty_pct": 0.1, "ema_span_0": 10.0, "ema_span_1": 20.0,
         "offset": 0.002, "offset_psize_weight": 0.0}
    )
    position = ({f"{side}_pos_size": 1.0 if side == "long" else -1.0,
                 f"{side}_pos_price": 100.0} if held else {})
    symbol = make_symbol(
        0, bid=100.0, ask=100.0, long_bp=policy, short_bp=policy,
        long_strategy=deepcopy(strategy), short_strategy=deepcopy(strategy), **position,
    )
    symbol[side]["mode"] = mode
    value = make_input(balance=1000.0, strategy_kind=kind,
                       global_bp=bot_params_pair(long_overrides=policy, short_overrides=policy),
                       symbols=[symbol])
    value["global"]["side_activation"] = {name: name == side for name in ("long", "short")}
    return value


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("held", [False, True])
def test_actual_native_activation_blocks_initial_and_dca_preserving_existing_closes(kind, side, held):
    import passivbot_rust as native
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    value = activation_snapshot(kind, side, held=held)
    original = deepcopy(value)
    baseline = compute(native, value)["orders"]
    assert any(order["order_type"].startswith("entry_") for order in baseline)
    value["global"]["side_activation"][side] = False
    orders = compute(native, value)["orders"]
    assert not any(order["order_type"].startswith("entry_") for order in orders)
    close_orders = lambda rows: [o for o in rows if o["order_type"].startswith("close_")]
    assert close_orders(orders) == close_orders(baseline)
    assert bool(close_orders(orders)) is held
    original["global"]["side_activation"][side] = False
    assert value == original


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("mode", ["normal", "graceful_stop", "tp_only", "manual", "panic"])
def test_actual_native_disabled_activation_keeps_explicit_operational_mode(kind, side, mode):
    import passivbot_rust as native

    value = activation_snapshot(kind, side, held=True, mode=mode)
    value["global"]["side_activation"][side] = False
    orders = compute(native, value)["orders"]
    assert not any(order["order_type"].startswith("entry_") for order in orders)
    assert bool(orders) is (mode != "manual")
    assert any(o["order_type"].startswith("close_panic_") for o in orders) is (mode == "panic")


@pytest.mark.parametrize("invalid", [None, True, {}, {"long": True},
    {"long": 1, "short": False}, {"long": True, "short": "false"},
    {"long": True, "short": False, "extra": False}])
def test_actual_native_activation_rejects_non_boolean_or_incomplete_packet(invalid):
    import passivbot_rust as native

    value = activation_snapshot("ema_anchor", "long")
    value["global"]["side_activation"] = invalid
    with pytest.raises(ValueError):
        compute(native, value)


def test_actual_native_activation_is_required_in_packet():
    import passivbot_rust as native

    value = activation_snapshot("ema_anchor", "long")
    del value["global"]["side_activation"]
    with pytest.raises(ValueError, match="side_activation"):
        compute(native, value)
