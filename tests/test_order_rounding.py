import math

import numpy as np
import pandas as pd
import pytest

import passivbot_rust as pbr


def _extension_available() -> bool:
    # C-extension functions do not expose __code__; stubbed Python fallbacks do.
    return not hasattr(pbr.calc_entries_long_py, "__code__")


requires_extension = pytest.mark.skipif(
    not _extension_available(), reason="passivbot_rust extension not available"
)


@requires_extension
@pytest.mark.parametrize("kind", ["entry", "close"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("threshold", [.1, .1 - 1e-12, .1 + 1e-12, .1001])
def test_compound_trailing_prices_follow_authored_decimal_alignment(kind, side, threshold):
    from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    increasing = (kind == "close") != (side == "short")
    distance = Decimal(str(threshold)) - Decimal("0.09375")
    raw = Decimal("43.2") * (1 + (distance if increasing else -distance))
    rounding = ROUND_CEILING if increasing else ROUND_FLOOR
    expected = float((raw / Decimal("0.01")).to_integral_value(rounding=rounding) * Decimal("0.01"))
    common = dict(qty_step=.001, price_step=.01, min_qty=.001, min_cost=0., c_mult=1.,
        wallet_exposure_limit=1., risk_we_excess_allowance_pct=0., balance=1000.,
        position_size=-1. if side == "short" else 1., position_price=43.2,
        min_since_open=30., max_since_min=40., max_since_open=60., min_since_max=40.,
        volatility_ema_1h=0., volatility_ema_1m=0.)
    if kind == "entry":
        common.update(entry_grid_double_down_factor=.2, entry_grid_spacing_pct=threshold,
            entry_initial_ema_dist=0., entry_initial_qty_pct=.0001,
            entry_trailing_double_down_factor=.2, entry_trailing_retracement_pct=.09375,
            entry_trailing_threshold_pct=threshold, entry_weight_volatility_1h=0.,
            entry_weight_volatility_1m=0., entry_we_weight=0.)
        common["ema_bands_upper" if side == "short" else "ema_bands_lower"] = 1. if side == "short" else 100.
        common["order_book_ask" if side == "short" else "order_book_bid"] = 30. if side == "short" else 60.
    else:
        common.update(close_grid_qty_pct=1., close_trailing_qty_pct=1.,
            close_trailing_retracement_pct=.09375, close_trailing_threshold_pct=threshold,
            risk_wel_enforcer_threshold=1.)
        common["order_book_bid" if side == "short" else "order_book_ask"] = 60. if side == "short" else 30.
    orders = getattr(pbr, f"calc_{'entries' if kind == 'entry' else 'closes'}_{side}_py")(**common)
    assert orders and abs(orders[0][0]) > 0.0
    assert orders[0][1] == expected


def _is_step_aligned(value: float, step: float, tol: float = 1e-12) -> bool:
    if step <= 0.0 or not math.isfinite(value):
        return True
    ratio = value / step
    return abs(ratio - round(ratio)) < tol


@requires_extension
def test_select_forager_candidates_py_is_exported():
    assert hasattr(pbr, "select_forager_candidates_py")


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
        entry_weight_volatility_1h=0.0,
        entry_weight_volatility_1m=0.0,
        entry_we_weight=0.0,
        entry_grid_spacing_pct=0.0,
        entry_initial_ema_dist=-0.01,
        entry_initial_qty_pct=0.05,
        entry_trailing_double_down_factor=1.0,
        entry_trailing_retracement_pct=0.0,
        entry_trailing_threshold_pct=0.0,
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
        volatility_ema_1h=0.0,
        volatility_ema_1m=0.0,
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
        close_grid_qty_pct=0.0,
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
def test_calc_closes_long_py_quantizes_threshold_price():
    result = pbr.calc_closes_long_py(
        qty_step=0.01,
        price_step=0.01,
        min_qty=0.0,
        min_cost=0.0,
        c_mult=1.0,
        close_grid_qty_pct=1.0,
        close_trailing_qty_pct=0.0,
        close_trailing_retracement_pct=0.0,
        close_trailing_threshold_pct=0.01,
        wallet_exposure_limit=1.0,
        risk_we_excess_allowance_pct=0.0,
        risk_wel_enforcer_threshold=1.0,
        balance=1000.0,
        position_size=1.0,
        position_price=100.0,
        min_since_open=0.0,
        max_since_min=0.0,
        max_since_open=0.0,
        min_since_max=0.0,
        order_book_ask=100.0,
        ema_bands_upper=110.0,
    )
    assert result
    _, price, _ = result[0]
    assert price == pytest.approx(101.0)


@requires_extension
def test_calc_closes_short_py_quantizes_threshold_price():
    result = pbr.calc_closes_short_py(
        qty_step=0.01,
        price_step=0.01,
        min_qty=0.0,
        min_cost=0.0,
        c_mult=1.0,
        close_grid_qty_pct=1.0,
        close_trailing_qty_pct=0.0,
        close_trailing_retracement_pct=0.0,
        close_trailing_threshold_pct=0.01,
        wallet_exposure_limit=1.0,
        risk_we_excess_allowance_pct=0.0,
        risk_wel_enforcer_threshold=1.0,
        balance=1000.0,
        position_size=-1.0,
        position_price=100.0,
        min_since_open=0.0,
        max_since_min=0.0,
        max_since_open=0.0,
        min_since_max=0.0,
        order_book_bid=100.0,
    )
    assert result
    _, price, _ = result[0]
    assert price == pytest.approx(99.0)


@requires_extension
def test_calc_ema_anchor_quote_series_py_supports_inventory_and_volatility_inputs():
    timestamps = np.array(
        [int(ts.value // 1_000_000) for ts in pd.date_range("2026-01-01", periods=4, freq="1min")],
        dtype=np.int64,
    )
    highs = np.array([101.0, 103.0, 104.0, 105.0], dtype=float)
    lows = np.array([99.0, 100.0, 101.0, 102.0], dtype=float)
    closes = np.array([100.0, 101.0, 102.0, 103.0], dtype=float)
    balances = np.array([1000.0, 1000.0, 1000.0, 1000.0], dtype=float)
    flat_psize = np.zeros(4, dtype=float)
    long_psize = np.array([0.0, 0.0, 1.0, 1.0], dtype=float)

    bid_flat, ask_flat = pbr.calc_ema_anchor_quote_series_py(
        "long",
        timestamps,
        highs,
        lows,
        closes,
        balances,
        flat_psize,
        price_step=0.1,
        ema_span_0=2.0,
        ema_span_1=6.0,
        offset=0.01,
        offset_volatility_ema_span_1m=2.0,
        offset_volatility_1m_weight=0.5,
        offset_volatility_ema_span_1h=2.0,
        offset_volatility_1h_weight=0.25,
        offset_psize_weight=0.2,
    )
    bid_long, ask_long = pbr.calc_ema_anchor_quote_series_py(
        "long",
        timestamps,
        highs,
        lows,
        closes,
        balances,
        long_psize,
        price_step=0.1,
        ema_span_0=2.0,
        ema_span_1=6.0,
        offset=0.01,
        offset_volatility_ema_span_1m=2.0,
        offset_volatility_1m_weight=0.5,
        offset_volatility_ema_span_1h=2.0,
        offset_volatility_1h_weight=0.25,
        offset_psize_weight=0.2,
    )

    assert len(bid_flat) == 4
    assert len(ask_flat) == 4
    assert bid_long[2] < bid_flat[2]
    assert ask_long[2] < ask_flat[2]


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
            "min_cost": 0.0,
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


@requires_extension
@pytest.mark.parametrize("side", ["long", "short"])
def test_single_trailing_close_matches_expanded_price_at_raw_touch(side):
    is_long = side == "long"
    params = dict(
        qty_step=0.01, price_step=0.01, min_qty=0.01, min_cost=0.0, c_mult=1.0,
        close_grid_qty_pct=1.0, close_trailing_qty_pct=1.0,
        close_trailing_retracement_pct=0.001, close_trailing_threshold_pct=0.001,
        wallet_exposure_limit=1.0, risk_we_excess_allowance_pct=0.0,
        risk_wel_enforcer_threshold=1.0, balance=1000.0,
        position_size=1.0 if is_long else -1.0, position_price=100.0,
        min_since_open=98.0, max_since_min=99.0,
        max_since_open=102.0, min_since_max=101.0,
    )
    params["order_book_ask" if is_long else "order_book_bid"] = (
        101.006 if is_long else 98.994
    )
    single = getattr(pbr, f"calc_next_close_{side}_py")(**params)
    expanded = getattr(pbr, f"calc_closes_{side}_py")(**params)
    assert single[1] == expanded[0][1] == (101.01 if is_long else 98.99)
    assert single[0] == expanded[0][0]
    # Strict fill tests must agree for a candle between the raw touch and tick.
    assert not (101.008 > single[1] if is_long else 98.992 < single[1])

@requires_extension
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("retracement", [0.0, 0.001])
@pytest.mark.parametrize("size,expected", [(0.066, 0.028), (0.066000001, 0.027)])
def test_partial_initial_entries_preserve_quantity_steps(side, retracement, size, expected):
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    params = dict(
        qty_step=0.001, price_step=0.01, min_qty=0.001, min_cost=1.0, c_mult=1.0,
        entry_grid_double_down_factor=0.2, entry_grid_spacing_pct=0.001,
        entry_initial_ema_dist=0.0, entry_initial_qty_pct=0.012,
        entry_trailing_double_down_factor=0.2, entry_trailing_retracement_pct=retracement,
        entry_trailing_threshold_pct=0.001, entry_weight_volatility_1h=0.0,
        entry_weight_volatility_1m=0.0, entry_we_weight=0.0, wallet_exposure_limit=0.5,
        risk_we_excess_allowance_pct=0.0, balance=1102.0,
        position_size=-size if side == "short" else size, position_price=99.85,
        min_since_open=0.0, max_since_min=0.0, max_since_open=0.0, min_since_max=0.0,
        volatility_ema_1h=0.0, volatility_ema_1m=0.0,
    )
    params["ema_bands_upper" if side == "short" else "ema_bands_lower"] = 70.62
    params["order_book_ask" if side == "short" else "order_book_bid"] = 70.62
    order = getattr(pbr, f"calc_entries_{side}_py")(**params)[0]
    assert order[0] == (-expected if side == "short" else expected)
    assert order[1] == 70.62
    assert order[2] == (12 if side == "short" else 1)

@requires_extension
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("factor,expected", [(1.15, 5.8), (1.15 - 1e-10, 5.7), (1.15 + 1e-10, 5.8)])
def test_reentry_quantity_recovers_only_half_step_representation_noise(side, factor, expected):
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    params = dict(
        qty_step=0.1, price_step=0.01, min_qty=0.1, min_cost=0.0, c_mult=1.0,
        entry_grid_double_down_factor=factor, entry_grid_spacing_pct=0.01,
        entry_initial_ema_dist=0.0, entry_initial_qty_pct=0.0001,
        entry_trailing_double_down_factor=factor, entry_trailing_retracement_pct=0.0,
        entry_trailing_threshold_pct=0.0, entry_weight_volatility_1h=0.0,
        entry_weight_volatility_1m=0.0, entry_we_weight=0.0, wallet_exposure_limit=1.0,
        risk_we_excess_allowance_pct=0.0, balance=10000.0,
        position_size=-5.0 if side == "short" else 5.0, position_price=100.0,
        min_since_open=90.0, max_since_min=100.0, max_since_open=110.0, min_since_max=100.0,
        volatility_ema_1h=0.0, volatility_ema_1m=0.0,
    )
    params["ema_bands_upper" if side == "short" else "ema_bands_lower"] = 100.0
    params["order_book_ask" if side == "short" else "order_book_bid"] = 100.0
    order = getattr(pbr, f"calc_entries_{side}_py")(**params)[0]
    assert abs(order[0]) == pytest.approx(expected, abs=1e-12)


@requires_extension
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("offset", [0.0, 1e-8])
def test_trailing_reentry_preserves_exact_ticks_without_absorbing_off_tick_targets(side, offset):
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    short = side == "short"
    params = dict(
        qty_step=0.001, price_step=0.01, min_qty=0.001, min_cost=0.0, c_mult=1.0,
        entry_grid_double_down_factor=0.2,
        entry_grid_spacing_pct=0.0021 if short else 0.0026,
        entry_initial_ema_dist=0.0, entry_initial_qty_pct=0.0001,
        entry_trailing_double_down_factor=0.2, entry_trailing_retracement_pct=0.001,
        entry_trailing_threshold_pct=0.0021 if short else 0.0026,
        entry_weight_volatility_1h=0.0, entry_weight_volatility_1m=0.0,
        entry_we_weight=0.0, wallet_exposure_limit=1.0, risk_we_excess_allowance_pct=0.0,
        balance=1000.0, position_size=-1.0 if short else 1.0,
        position_price=100.0 + offset if short else 131.25 - offset,
        min_since_open=90.0 if short else 120.0,
        max_since_min=100.0 if short else 130.0,
        max_since_open=110.0 if short else 140.0,
        min_since_max=100.0 if short else 120.0,
        volatility_ema_1h=0.0, volatility_ema_1m=0.0,
    )
    params["ema_bands_upper" if short else "ema_bands_lower"] = 100.0 if short else 140.0
    params["order_book_ask" if short else "order_book_bid"] = 90.0 if short else 140.0
    order = getattr(pbr, f"calc_entries_{side}_py")(**params)[0]
    expected = (100.11 if offset == 0.0 else 100.12) if short else (131.04 if offset == 0.0 else 131.03)
    assert abs(order[0]) > 0.0
    assert order[1] == pytest.approx(expected, abs=1e-12)

@requires_extension
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("retracement", [0.0, 0.001])
def test_partial_initial_qty_at_power_of_two_uses_inward_spacing(side, retracement):
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    short = side == "short"
    params = dict(
        qty_step=1e-12, price_step=0.01, min_qty=1e-12, min_cost=0.0, c_mult=1.0,
        entry_grid_double_down_factor=1.0, entry_grid_spacing_pct=0.5,
        entry_initial_ema_dist=0.0, entry_initial_qty_pct=1.0,
        entry_trailing_double_down_factor=1.0,
        entry_trailing_retracement_pct=retracement, entry_trailing_threshold_pct=0.5,
        entry_weight_volatility_1h=0.0, entry_weight_volatility_1m=0.0, entry_we_weight=0.0,
        wallet_exposure_limit=1.0, risk_we_excess_allowance_pct=0.0,
        balance=819200.0, position_size=(-1 if short else 1) * 4217.878608603804,
        position_price=100.0, min_since_open=100.0, max_since_min=100.0,
        max_since_open=100.0, min_since_max=100.0,
        volatility_ema_1h=0.0, volatility_ema_1m=0.0,
    )
    params["ema_bands_upper" if short else "ema_bands_lower"] = 100.0
    params["order_book_ask" if short else "order_book_bid"] = 100.0
    orders = (pbr.calc_entries_short_py if short else pbr.calc_entries_long_py)(**params)
    assert len(orders) == 1
    qty, price, _ = orders[0]
    assert qty == (-1 if short else 1) * 3974.121391396196
    assert price == 100.0
