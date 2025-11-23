import math

import pytest

try:
    import passivbot_rust as pbr
except Exception:  # pragma: no cover - exercised when rust extension missing
    pbr = None

pbr_is_stub = bool(getattr(pbr, "__is_stub__", False)) if pbr is not None else False


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
    risk_we_excess_allowance_pct=0.0,
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
        "risk_we_excess_allowance_pct": float(risk_we_excess_allowance_pct),
        "ema_band_upper": float(ema_band_upper),
        "ema_band_lower": float(ema_band_lower),
        "current_price": float(current_price),
        "price_step": float(price_step),
        "qty_step": float(qty_step),
        "min_qty": float(min_qty),
        "min_cost": float(min_cost),
        "c_mult": float(c_mult),
    }


@pytest.mark.skipif(pbr is None or pbr_is_stub, reason="passivbot_rust extension not available")
def test_unstucking_returns_none_when_allowance_zero():
    positions = [_build_position()]
    result = pbr.calc_unstucking_close_py(1000.0, 0.0, 0.0, positions)
    assert result is None


@pytest.mark.skipif(pbr is None or pbr_is_stub, reason="passivbot_rust extension not available")
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
    expected_qty = -0.95  # 1000 * 1 * 0.1 / 105 rounded down to 0.95
    assert math.isclose(qty, expected_qty, rel_tol=1e-9)


@pytest.mark.skipif(pbr is None or pbr_is_stub, reason="passivbot_rust extension not available")
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
    expected_qty = 4.16  # 1000 * 1 * 0.2 / 48 rounded down to 4.16
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


@pytest.mark.skipif(pbr is None or pbr_is_stub, reason="passivbot_rust extension not available")
def test_unstucking_respects_effective_wel_threshold():
    balance = 1000.0
    wel = 0.2
    allowance_pct = 1.0  # doubles the effective WEL
    unstuck_threshold = 0.8
    # Exposure threshold -> 0.2 * (1 + 1.0) * 0.8 = 0.32
    base_kwargs = dict(
        side="long",
        position_price=100.0,
        ema_band_upper=100.0,
        current_price=105.0,
        wallet_exposure_limit=wel,
        risk_we_excess_allowance_pct=allowance_pct,
        unstuck_threshold=unstuck_threshold,
        unstuck_close_pct=0.1,
        qty_step=0.001,
        price_step=0.01,
    )

    below_threshold = [
        _build_position(position_size=3.0, **base_kwargs),  # exposure ≈ 0.30
    ]
    assert (
        pbr.calc_unstucking_close_py(balance, 100.0, 0.0, below_threshold) is None
    ), "Exposure below effective threshold should not trigger unstucking"

    above_threshold = [
        _build_position(position_size=4.0, **base_kwargs),  # exposure ≈ 0.40
    ]
    result = pbr.calc_unstucking_close_py(balance, 100.0, 0.0, above_threshold)
    assert result is not None, "Exposure above effective threshold should trigger unstucking"
    idx, side_code, qty, price, order_type_id = result
    assert idx == 0
    assert qty < 0.0
    expected_qty = -0.38  # 1000 * (0.2 * 2) * 0.1 / 105 rounded down to 0.38
    assert math.isclose(qty, expected_qty, rel_tol=1e-9)
    assert math.isclose(price, 105.0)
    expected_order_type = pbr.order_type_snake_to_id("close_unstuck_long")
    assert order_type_id == expected_order_type


@pytest.mark.skipif(pbr is None or pbr_is_stub, reason="passivbot_rust extension not available")
def test_unstucking_respects_effective_wel_threshold_short():
    balance = 1000.0
    wel = 0.2
    allowance_pct = 1.0
    unstuck_threshold = 0.8
    base_kwargs = dict(
        side="short",
        position_price=100.0,
        ema_band_lower=100.0,
        current_price=95.0,
        wallet_exposure_limit=wel,
        risk_we_excess_allowance_pct=allowance_pct,
        unstuck_threshold=unstuck_threshold,
        unstuck_close_pct=0.1,
        qty_step=0.001,
        price_step=0.01,
    )

    below_threshold = [
        _build_position(position_size=-3.0, **base_kwargs),  # exposure ≈ 0.30
    ]
    assert (
        pbr.calc_unstucking_close_py(balance, 0.0, 100.0, below_threshold) is None
    ), "Exposure below effective threshold should not trigger short unstucking"

    above_threshold = [
        _build_position(position_size=-4.0, **base_kwargs),  # exposure ≈ 0.40
    ]
    result = pbr.calc_unstucking_close_py(balance, 0.0, 100.0, above_threshold)
    assert result is not None, "Exposure above effective threshold should trigger short unstucking"
    idx, side_code, qty, price, order_type_id = result
    assert idx == 0
    assert qty > 0.0
    expected_qty = 0.421  # 1000 * (0.2 * 2) * 0.1 / 95 rounded down to 0.421
    assert math.isclose(qty, expected_qty, rel_tol=1e-9)
    assert math.isclose(price, 95.0)
    expected_order_type = pbr.order_type_snake_to_id("close_unstuck_short")
    assert order_type_id == expected_order_type


@pytest.mark.skipif(pbr is None or pbr_is_stub, reason="passivbot_rust extension not available")
def test_unstucking_triggers_with_fractional_allowance():
    balance = 2000.0
    wel = 0.25
    allowance_pct = 0.3  # effective wel = 0.325
    unstuck_threshold = 0.5
    base_kwargs = dict(
        side="long",
        position_price=200.0,
        ema_band_upper=200.0,
        current_price=210.0,
        wallet_exposure_limit=wel,
        risk_we_excess_allowance_pct=allowance_pct,
        unstuck_threshold=unstuck_threshold,
        unstuck_close_pct=0.25,
        qty_step=0.001,
        price_step=0.01,
    )

    below_threshold = [
        _build_position(position_size=1.5, **base_kwargs),  # exposure = 0.15
    ]
    assert (
        pbr.calc_unstucking_close_py(balance, 100.0, 0.0, below_threshold) is None
    ), "Exposure below effective threshold should not trigger long unstucking"

    above_threshold = [
        _build_position(position_size=1.7, **base_kwargs),  # exposure ≈ 0.1785
    ]
    result = pbr.calc_unstucking_close_py(balance, 100.0, 0.0, above_threshold)
    assert result is not None
    idx, side_code, qty, price, order_type_id = result
    assert idx == 0
    assert qty < 0.0
    expected_qty = -0.773  # 2000 * 0.325 * 0.25 / 210 rounded down to 0.773
    assert math.isclose(qty, expected_qty, rel_tol=1e-9)
    assert math.isclose(price, 210.0)
    expected_order_type = pbr.order_type_snake_to_id("close_unstuck_long")
    assert order_type_id == expected_order_type


@pytest.mark.skipif(pbr is None or pbr_is_stub, reason="passivbot_rust extension not available")
def test_unstucking_requires_exposure_when_threshold_above_one():
    balance = 1000.0
    wel = 0.2
    allowance_pct = 0.0
    threshold = 1.2  # trigger at 0.24 exposure
    base_kwargs = dict(
        side="short",
        position_price=100.0,
        ema_band_lower=100.0,
        current_price=95.0,
        wallet_exposure_limit=wel,
        risk_we_excess_allowance_pct=allowance_pct,
        unstuck_threshold=threshold,
        unstuck_close_pct=0.1,
        qty_step=0.001,
        price_step=0.01,
    )

    below_threshold = [
        _build_position(position_size=-2.3, **base_kwargs),  # exposure ≈ 0.2185
    ]
    assert (
        pbr.calc_unstucking_close_py(balance, 0.0, 100.0, below_threshold) is None
    ), "Exposure below high threshold should not trigger short unstucking"

    above_threshold = [
        _build_position(position_size=-2.5, **base_kwargs),  # exposure ≈ 0.25
    ]
    result = pbr.calc_unstucking_close_py(balance, 0.0, 100.0, above_threshold)
    assert result is not None
    idx, side_code, qty, price, order_type_id = result
    assert idx == 0
    assert qty > 0.0
    expected_qty = 0.21  # 1000 * 0.2 * 0.1 / 95 rounded down to 0.21
    assert math.isclose(qty, expected_qty, rel_tol=1e-9)
    assert math.isclose(price, 95.0)
    expected_order_type = pbr.order_type_snake_to_id("close_unstuck_short")
    assert order_type_id == expected_order_type
