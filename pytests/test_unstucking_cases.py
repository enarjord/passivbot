import math

import pytest
import passivbot_rust as pbr


def _build_position(
    *,
    idx=0,
    side="long",
    position_size=None,
    position_price=100.0,
    current_price=None,
    ema_band_upper=100.0,
    ema_band_lower=100.0,
    wallet_exposure_limit=1.0,
    unstuck_threshold=0.0,
    unstuck_close_pct=0.1,
    unstuck_ema_dist=0.0,
    qty_step=0.01,
    price_step=0.01,
    min_qty=0.0,
    min_cost=0.0,
    c_mult=1.0,
):
    if position_size is None:
        position_size = 2.0 if side == "long" else -2.0
    if current_price is None:
        if side == "long":
            current_price = ema_band_upper * (1.0 + 0.05)
        else:
            current_price = ema_band_lower * (1.0 - 0.05)

    return {
        "idx": idx,
        "side": side,
        "position_size": float(position_size),
        "position_price": float(position_price),
        "wallet_exposure_limit": float(wallet_exposure_limit),
        "unstuck_threshold": float(unstuck_threshold),
        "unstuck_close_pct": float(unstuck_close_pct),
        "unstuck_ema_dist": float(unstuck_ema_dist),
        "ema_band_upper": float(ema_band_upper),
        "ema_band_lower": float(ema_band_lower),
        "current_price": float(current_price),
        "price_step": float(price_step),
        "qty_step": float(qty_step),
        "min_qty": float(min_qty),
        "min_cost": float(min_cost),
        "c_mult": float(c_mult),
    }


@pytest.mark.skipif(pbr is None, reason="passivbot_rust extension not available")
def test_unstucking_returns_none_when_allowance_zero():
    positions = [_build_position()]
    result = pbr.calc_unstucking_close_py(1000.0, 0.0, 0.0, positions)
    assert result is None


@pytest.mark.skipif(pbr is None, reason="passivbot_rust extension not available")
def test_unstucking_emits_long_order_when_triggered():
    positions = [
        _build_position(
            side="long",
            position_size=2.0,
            position_price=100.0,
            current_price=105.0,
            ema_band_upper=100.0,
        )
    ]
    result = pbr.calc_unstucking_close_py(1000.0, 5.0, 0.0, positions)
    assert result is not None
    idx, side_code, qty, price, order_type_id = result
    assert idx == 0
    assert qty < 0.0
    assert math.isclose(price, 105.0)
    assert order_type_id == pbr.get_order_id_type_from_string("close_unstuck_long")
    expected_qty = -0.95
    assert math.isclose(qty, expected_qty, rel_tol=1e-9)


@pytest.mark.skipif(pbr is None, reason="passivbot_rust extension not available")
def test_unstucking_emits_short_order_when_triggered():
    positions = [
        _build_position(
            side="short",
            position_size=-5.0,
            position_price=50.0,
            current_price=48.0,
            ema_band_lower=50.0,
            unstuck_close_pct=0.2,
        )
    ]
    result = pbr.calc_unstucking_close_py(1000.0, 0.0, 10.0, positions)
    assert result is not None
    idx, side_code, qty, price, order_type_id = result
    assert idx == 0
    assert qty > 0.0
    assert math.isclose(price, 48.0)
    assert order_type_id == pbr.get_order_id_type_from_string("close_unstuck_short")
    expected_qty = 4.16
    assert math.isclose(qty, expected_qty, rel_tol=1e-9)


@pytest.mark.skipif(pbr is None, reason="passivbot_rust extension not available")
def test_unstucking_skips_when_price_not_beyond_ema():
    positions = [
        _build_position(
            side="long",
            current_price=95.0,
            ema_band_upper=100.0,
        )
    ]
    result = pbr.calc_unstucking_close_py(1000.0, 5.0, 0.0, positions)
    assert result is None
