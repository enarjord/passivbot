import math

import pytest

import passivbot_rust as pbr


def _extension_available() -> bool:
    # C-extension functions do not expose __code__; stubbed Python fallbacks do.
    return not hasattr(pbr.calc_entries_long_py, "__code__")


requires_extension = pytest.mark.skipif(
    not _extension_available(), reason="passivbot_rust extension not available"
)


def _is_step_aligned(value: float, step: float, tol: float = 1e-12) -> bool:
    if step <= 0.0 or not math.isfinite(value):
        return True
    ratio = value / step
    return abs(ratio - round(ratio)) < tol


@requires_extension
def test_calc_entries_long_py_quantizes_results():
    step_qty = 0.001
    step_price = 0.0001
    result = pbr.calc_entries_long_py(
        step_qty,
        step_price,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        1000.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.6545000076293945,
    )
    assert result, "expected at least one entry order"
    qty, price, _ = result[0]
    assert _is_step_aligned(price, step_price)
    assert _is_step_aligned(abs(qty), step_qty)


@requires_extension
def test_calc_closes_long_py_quantizes_results():
    step_qty = 0.001
    step_price = 0.0001
    # Seed a simple position so that a close order is generated.
    result = pbr.calc_closes_long_py(
        step_qty,
        step_price,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1000.0,
        0.5,
        0.6545000076293945,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.6545000076293945,
    )
    assert result, "expected at least one close order"
    qty, price, _ = result[0]
    assert _is_step_aligned(abs(qty), step_qty)
    assert _is_step_aligned(price, step_price)


@requires_extension
def test_calc_twel_enforcer_orders_quantizes_results():
    order_type = pbr.order_type_snake_to_id("close_auto_reduce_twel_long")
    actions = pbr.calc_twel_enforcer_orders_py(
        "long",
        0.5,
        1.0,
        1,
        1000.0,
        [
            {
                "idx": 0,
                "position_size": 1958.0,
                "position_price": 0.1955000076293945,
                "market_price": 0.19550000131130219,
                "base_wallet_exposure_limit": 0.8,
                "c_mult": 1.0,
                "qty_step": 0.1,
                "price_step": 0.0005,
                "min_qty": 0.0,
            }
        ],
        None,
    )
    assert actions, "expected action from TWEL enforcer"
    idx, qty, price, ot = actions[0]
    assert idx == 0
    assert ot == order_type
    assert _is_step_aligned(abs(qty), 0.1)
    assert _is_step_aligned(price, 0.0005)
