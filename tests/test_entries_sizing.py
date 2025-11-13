import pytest

try:
    import passivbot_rust as pbr
except Exception:  # pragma: no cover - exercised when the extension is unavailable
    pbr = None

pbr_is_stub = bool(getattr(pbr, "__is_stub__", False)) if pbr is not None else False


def _calc_next_entry_long(**kwargs):
    qty, price, order_type = pbr.calc_next_entry_long_py(**kwargs)
    return qty, price, order_type


def _calc_next_entry_short(**kwargs):
    qty, price, order_type = pbr.calc_next_entry_short_py(**kwargs)
    return qty, price, order_type


@pytest.mark.skipif(pbr is None or pbr_is_stub, reason="passivbot_rust extension not available")
def test_initial_entry_qty_long_respects_min_qty_and_allowance():
    params = dict(
        qty_step=0.01,
        price_step=0.1,
        min_qty=0.0,
        min_cost=0.0,
        c_mult=1.0,
        entry_grid_double_down_factor=1.0,
        entry_grid_spacing_volatility_weight=0.0,
        entry_grid_spacing_we_weight=0.0,
        entry_grid_spacing_pct=0.0,
        entry_initial_ema_dist=0.0,
        entry_initial_qty_pct=0.25,
        entry_trailing_double_down_factor=1.0,
        entry_trailing_grid_ratio=0.0,
        entry_trailing_retracement_pct=0.0,
        entry_trailing_retracement_we_weight=0.0,
        entry_trailing_retracement_volatility_weight=0.0,
        entry_trailing_threshold_pct=0.0,
        entry_trailing_threshold_we_weight=0.0,
        entry_trailing_threshold_volatility_weight=0.0,
        wallet_exposure_limit=0.1,
        risk_we_excess_allowance_pct=1.0,
        balance=1000.0,
        position_size=0.0,
        position_price=0.0,
        min_since_open=0.0,
        max_since_min=0.0,
        max_since_open=0.0,
        min_since_max=0.0,
        ema_bands_lower=95.0,
        grid_log_range=0.0,
        order_book_bid=94.0,
    )

    qty, price, order_type = _calc_next_entry_long(**params)
    assert order_type == "entry_initial_normal_long"

    allowed = params["wallet_exposure_limit"] * (1.0 + params["risk_we_excess_allowance_pct"])
    target_cost = params["balance"] * allowed * params["entry_initial_qty_pct"]
    expected_qty = pbr.round_(
        pbr.cost_to_qty(target_cost, price, params["c_mult"]), params["qty_step"]
    )

    assert qty == pytest.approx(expected_qty)


@pytest.mark.skipif(pbr is None or pbr_is_stub, reason="passivbot_rust extension not available")
def test_initial_entry_qty_short_respects_allowance():
    params = dict(
        qty_step=0.01,
        price_step=0.1,
        min_qty=0.0,
        min_cost=0.0,
        c_mult=1.0,
        entry_grid_double_down_factor=1.0,
        entry_grid_spacing_volatility_weight=0.0,
        entry_grid_spacing_we_weight=0.0,
        entry_grid_spacing_pct=0.0,
        entry_initial_ema_dist=0.0,
        entry_initial_qty_pct=0.25,
        entry_trailing_double_down_factor=1.0,
        entry_trailing_grid_ratio=0.0,
        entry_trailing_retracement_pct=0.0,
        entry_trailing_retracement_we_weight=0.0,
        entry_trailing_retracement_volatility_weight=0.0,
        entry_trailing_threshold_pct=0.0,
        entry_trailing_threshold_we_weight=0.0,
        entry_trailing_threshold_volatility_weight=0.0,
        wallet_exposure_limit=0.1,
        risk_we_excess_allowance_pct=1.0,
        balance=1000.0,
        position_size=0.0,
        position_price=0.0,
        min_since_open=0.0,
        max_since_min=0.0,
        max_since_open=0.0,
        min_since_max=0.0,
        ema_bands_upper=105.0,
        grid_log_range=0.0,
        order_book_ask=106.0,
    )

    qty, price, order_type = _calc_next_entry_short(**params)
    assert order_type == "entry_initial_normal_short"

    allowed = params["wallet_exposure_limit"] * (1.0 + params["risk_we_excess_allowance_pct"])
    target_cost = params["balance"] * allowed * params["entry_initial_qty_pct"]
    expected_qty = pbr.round_(
        pbr.cost_to_qty(target_cost, price, params["c_mult"]), params["qty_step"]
    )

    assert abs(qty) == pytest.approx(expected_qty)


@pytest.mark.skipif(pbr is None or pbr_is_stub, reason="passivbot_rust extension not available")
def test_reentry_blocked_when_cap_reached():
    params = dict(
        qty_step=0.1,
        price_step=0.1,
        min_qty=0.0,
        min_cost=0.0,
        c_mult=1.0,
        entry_grid_double_down_factor=1.0,
        entry_grid_spacing_volatility_weight=0.0,
        entry_grid_spacing_we_weight=0.0,
        entry_grid_spacing_pct=0.0,
        entry_initial_ema_dist=0.0,
        entry_initial_qty_pct=0.1,
        entry_trailing_double_down_factor=1.0,
        entry_trailing_grid_ratio=0.0,
        entry_trailing_retracement_pct=0.0,
        entry_trailing_retracement_we_weight=0.0,
        entry_trailing_retracement_volatility_weight=0.0,
        entry_trailing_threshold_pct=0.0,
        entry_trailing_threshold_we_weight=0.0,
        entry_trailing_threshold_volatility_weight=0.0,
        wallet_exposure_limit=0.05,
        risk_we_excess_allowance_pct=0.0,
        balance=1000.0,
        position_size=10.0,
        position_price=100.0,
        min_since_open=0.0,
        max_since_min=0.0,
        max_since_open=0.0,
        min_since_max=0.0,
        ema_bands_lower=95.0,
        grid_log_range=0.0,
        order_book_bid=94.0,
    )

    qty, price, order_type = _calc_next_entry_long(**params)
    assert qty == 0.0
    assert price == 0.0
    assert order_type == "empty"


@pytest.mark.skipif(pbr is None or pbr_is_stub, reason="passivbot_rust extension not available")
def test_reentry_long_is_cropped_to_cap():
    params = dict(
        qty_step=0.1,
        price_step=0.1,
        min_qty=0.0,
        min_cost=0.0,
        c_mult=1.0,
        entry_grid_double_down_factor=2.0,
        entry_grid_spacing_volatility_weight=0.0,
        entry_grid_spacing_we_weight=0.0,
        entry_grid_spacing_pct=0.05,
        entry_initial_ema_dist=0.0,
        entry_initial_qty_pct=0.1,
        entry_trailing_double_down_factor=1.0,
        entry_trailing_grid_ratio=0.0,
        entry_trailing_retracement_pct=0.0,
        entry_trailing_retracement_we_weight=0.0,
        entry_trailing_retracement_volatility_weight=0.0,
        entry_trailing_threshold_pct=0.0,
        entry_trailing_threshold_we_weight=0.0,
        entry_trailing_threshold_volatility_weight=0.0,
        wallet_exposure_limit=0.3,
        risk_we_excess_allowance_pct=0.0,
        balance=1000.0,
        position_size=2.0,
        position_price=100.0,
        min_since_open=0.0,
        max_since_min=0.0,
        max_since_open=0.0,
        min_since_max=0.0,
        ema_bands_lower=100.0,
        grid_log_range=0.0,
        order_book_bid=95.0,
    )

    qty, price, order_type = _calc_next_entry_long(**params)
    assert order_type == "entry_grid_cropped_long"
    assert qty > 0.0

    new_size, new_price = pbr.calc_new_psize_pprice(
        params["position_size"], params["position_price"], qty, price, params["qty_step"]
    )
    new_exposure = pbr.calc_wallet_exposure(params["c_mult"], params["balance"], new_size, new_price)
    allowed = params["wallet_exposure_limit"] * (1.0 + params["risk_we_excess_allowance_pct"])
    assert new_exposure <= allowed * 1.02 + 1e-9


@pytest.mark.skipif(pbr is None or pbr_is_stub, reason="passivbot_rust extension not available")
def test_reentry_short_is_cropped_to_cap():
    params = dict(
        qty_step=0.1,
        price_step=0.1,
        min_qty=0.0,
        min_cost=0.0,
        c_mult=1.0,
        entry_grid_double_down_factor=2.0,
        entry_grid_spacing_volatility_weight=0.0,
        entry_grid_spacing_we_weight=0.0,
        entry_grid_spacing_pct=0.05,
        entry_initial_ema_dist=0.0,
        entry_initial_qty_pct=0.1,
        entry_trailing_double_down_factor=1.0,
        entry_trailing_grid_ratio=0.0,
        entry_trailing_retracement_pct=0.0,
        entry_trailing_retracement_we_weight=0.0,
        entry_trailing_retracement_volatility_weight=0.0,
        entry_trailing_threshold_pct=0.0,
        entry_trailing_threshold_we_weight=0.0,
        entry_trailing_threshold_volatility_weight=0.0,
        wallet_exposure_limit=0.3,
        risk_we_excess_allowance_pct=0.0,
        balance=1000.0,
        position_size=-2.0,
        position_price=100.0,
        min_since_open=0.0,
        max_since_min=0.0,
        max_since_open=0.0,
        min_since_max=0.0,
        ema_bands_upper=100.0,
        grid_log_range=0.0,
        order_book_ask=105.0,
    )

    qty, price, order_type = _calc_next_entry_short(**params)
    assert order_type == "entry_grid_cropped_short"
    assert qty < 0.0

    new_size, new_price = pbr.calc_new_psize_pprice(
        params["position_size"], params["position_price"], qty, price, params["qty_step"]
    )
    new_exposure = pbr.calc_wallet_exposure(
        params["c_mult"], params["balance"], abs(new_size), new_price
    )
    allowed = params["wallet_exposure_limit"] * (1.0 + params["risk_we_excess_allowance_pct"])
    assert new_exposure <= allowed * 1.02 + 1e-9
