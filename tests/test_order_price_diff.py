import pytest

import passivbot_rust as pbr


def test_calc_order_price_gap_fraction_buy_side():
    diff = pbr.calc_order_price_gap_fraction("buy", 98.0, 100.0)
    assert diff == pytest.approx(0.02)

    diff_inside = pbr.calc_order_price_gap_fraction("BUY", 100.0, 100.0)
    assert diff_inside == pytest.approx(0.0)

    diff_over = pbr.calc_order_price_gap_fraction("  buy  ", 101.0, 100.0)
    assert diff_over == pytest.approx(-0.01)


def test_calc_order_price_gap_fraction_sell_side():
    diff = pbr.calc_order_price_gap_fraction("sell", 102.0, 100.0)
    assert diff == pytest.approx(0.02)

    diff_inside = pbr.calc_order_price_gap_fraction("SELL", 100.0, 100.0)
    assert diff_inside == pytest.approx(0.0)

    diff_under = pbr.calc_order_price_gap_fraction("sell", 99.0, 100.0)
    assert diff_under == pytest.approx(-0.01)


def test_calc_order_price_gap_fraction_invalid_side():
    with pytest.raises(ValueError):
        pbr.calc_order_price_gap_fraction("hold", 100.0, 100.0)
