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
    params = dict(
        qty_step=step_qty,
        price_step=step_price,
        min_qty=0.0,
        min_cost=0.0,
        c_mult=1.0,
        entry_grid_double_down_factor=1.0,
        entry_grid_spacing_volatility_weight=0.0,
        entry_grid_spacing_we_weight=0.0,
        entry_grid_spacing_pct=0.0,
        entry_initial_ema_dist=-0.01,
        entry_initial_qty_pct=0.05,
        entry_trailing_double_down_factor=1.0,
        entry_trailing_grid_ratio=1.0,
        entry_trailing_retracement_pct=0.0,
        entry_trailing_retracement_we_weight=0.0,
        entry_trailing_retracement_volatility_weight=0.0,
        entry_trailing_threshold_pct=0.0,
        entry_trailing_threshold_we_weight=0.0,
        entry_trailing_threshold_volatility_weight=0.0,
        wallet_exposure_limit=1.0,
        risk_we_excess_allowance_pct=0.0,
        balance=1000.0,
        position_size=0.0,
        position_price=0.0,
        min_since_open=0.0,
        max_since_min=0.0,
        max_since_open=0.0,
        min_since_max=0.0,
        ema_bands_lower=0.6545,
        grid_log_range=0.0,
        order_book_bid=0.66,
    )
    result = pbr.calc_entries_long_py(**params)
    assert result, "expected at least one entry order"
    qty, price, _ = result[0]
    assert _is_step_aligned(price, step_price)
    assert _is_step_aligned(abs(qty), step_qty)


@requires_extension
def test_calc_closes_long_py_quantizes_results():
    step_qty = 0.001
    step_price = 0.0001
    # Seed a simple position so that a close order is generated.
    params = dict(
        qty_step=step_qty,
        price_step=step_price,
        min_qty=0.0,
        min_cost=0.0,
        c_mult=1.0,
        close_grid_markup_end=0.0,
        close_grid_markup_start=0.0,
        close_grid_qty_pct=0.0,
        close_trailing_grid_ratio=0.0,
        close_trailing_qty_pct=0.0,
        close_trailing_retracement_pct=0.0,
        close_trailing_threshold_pct=0.0,
        wallet_exposure_limit=0.5,
        risk_we_excess_allowance_pct=0.0,
        risk_wel_enforcer_threshold=1.0,
        balance=1000.0,
        position_size=0.5,
        position_price=0.6545000076293945,
        min_since_open=0.0,
        max_since_min=0.0,
        max_since_open=0.0,
        min_since_max=0.0,
        order_book_ask=0.6545000076293945,
    )
    result = pbr.calc_closes_long_py(**params)
    assert result, "expected at least one close order"
    qty, price, _ = result[0]
    assert _is_step_aligned(abs(qty), step_qty)
    assert _is_step_aligned(price, step_price)


@requires_extension
def test_calc_twel_enforcer_orders_quantizes_results():
    order_type = pbr.order_type_snake_to_id("close_auto_reduce_twel_long")
    positions = [
        {
            "idx": 0,
            "position_size": 3000.0,
            "position_price": 0.1955,
            "market_price": 0.1956,
            "base_wallet_exposure_limit": 0.2,
            "c_mult": 1.0,
            "qty_step": 0.1,
            "price_step": 0.0005,
            "min_qty": 0.0,
        }
    ]
    actions = pbr.calc_twel_enforcer_orders_py(
        "long",
        0.5,
        1.0,
        1,
        1000.0,
        positions,
        None,
    )
    assert actions, "expected action from TWEL enforcer"
    idx, qty, price, ot = actions[0]
    assert idx == 0
    assert ot == order_type
    assert _is_step_aligned(abs(qty), 0.1)
    assert _is_step_aligned(price, 0.0005)
