import pytest

try:
    import passivbot_rust as pbr
except Exception:  # pragma: no cover - exercised when rust extension is missing
    pbr = None

pbr_is_stub = bool(getattr(pbr, "__is_stub__", False)) if pbr is not None else False


def _find_order(orders, order_type_name):
    order_type_id = pbr.order_type_snake_to_id(order_type_name)
    for qty, price, oid in orders:
        if oid == order_type_id:
            return qty, price, oid
    return None


@pytest.mark.skipif(pbr is None or pbr_is_stub, reason="passivbot_rust extension not available")
def test_wel_enforcer_reduces_long_exposure_below_threshold():
    params = dict(
        qty_step=0.1,
        price_step=0.1,
        min_qty=0.0,
        min_cost=0.0,
        c_mult=1.0,
        close_grid_markup_end=0.05,
        close_grid_markup_start=0.05,
        close_grid_qty_pct=0.5,
        close_trailing_grid_ratio=0.0,
        close_trailing_qty_pct=0.0,
        close_trailing_retracement_pct=0.0,
        close_trailing_threshold_pct=0.0,
        wallet_exposure_limit=0.2,
        risk_we_excess_allowance_pct=0.5,
        risk_wel_enforcer_threshold=0.85,
        balance=1000.0,
        position_size=5.0,
        position_price=100.0,
        min_since_open=0.0,
        max_since_min=0.0,
        max_since_open=0.0,
        min_since_max=0.0,
        order_book_ask=101.0,
    )

    closes = pbr.calc_closes_long_py(**params)
    wel_order = _find_order(closes, "close_auto_reduce_wel_long")
    assert wel_order is not None, "Expected WEL auto-reducer order for long position"
    qty, price, _ = wel_order

    assert qty < 0.0
    assert price == pytest.approx(101.0)

    new_size = params["position_size"] + qty
    new_exposure = pbr.calc_wallet_exposure(
        params["c_mult"], params["balance"], abs(new_size), params["position_price"]
    )
    allowed = params["wallet_exposure_limit"] * (1.0 + params["risk_we_excess_allowance_pct"])
    target = allowed * params["risk_wel_enforcer_threshold"]
    assert new_exposure < target


@pytest.mark.skipif(pbr is None or pbr_is_stub, reason="passivbot_rust extension not available")
def test_wel_enforcer_reduces_short_exposure_below_threshold():
    params = dict(
        qty_step=0.1,
        price_step=0.1,
        min_qty=0.0,
        min_cost=0.0,
        c_mult=1.0,
        close_grid_markup_end=0.05,
        close_grid_markup_start=0.05,
        close_grid_qty_pct=0.5,
        close_trailing_grid_ratio=0.0,
        close_trailing_qty_pct=0.0,
        close_trailing_retracement_pct=0.0,
        close_trailing_threshold_pct=0.0,
        wallet_exposure_limit=0.2,
        risk_we_excess_allowance_pct=0.5,
        risk_wel_enforcer_threshold=0.85,
        balance=1000.0,
        position_size=-5.0,
        position_price=100.0,
        min_since_open=0.0,
        max_since_min=0.0,
        max_since_open=0.0,
        min_since_max=0.0,
        order_book_bid=99.0,
    )

    closes = pbr.calc_closes_short_py(**params)
    wel_order = _find_order(closes, "close_auto_reduce_wel_short")
    assert wel_order is not None, "Expected WEL auto-reducer order for short position"
    qty, price, _ = wel_order

    assert qty > 0.0
    assert price == pytest.approx(99.0)

    new_size = params["position_size"] + qty
    new_exposure = pbr.calc_wallet_exposure(
        params["c_mult"], params["balance"], abs(new_size), params["position_price"]
    )
    allowed = params["wallet_exposure_limit"] * (1.0 + params["risk_we_excess_allowance_pct"])
    target = allowed * params["risk_wel_enforcer_threshold"]
    assert new_exposure < target
