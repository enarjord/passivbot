import pytest

import passivbot_rust as pbr


def test_calc_order_price_diff_buy_side():
    diff = pbr.calc_order_price_diff("buy", 98.0, 100.0)
    assert diff == pytest.approx(0.02)

    diff_inside = pbr.calc_order_price_diff("BUY", 100.0, 100.0)
    assert diff_inside == pytest.approx(0.0)

    diff_over = pbr.calc_order_price_diff("  buy  ", 101.0, 100.0)
    assert diff_over == pytest.approx(-0.01)


def test_calc_order_price_diff_sell_side():
    diff = pbr.calc_order_price_diff("sell", 102.0, 100.0)
    assert diff == pytest.approx(0.02)

    diff_inside = pbr.calc_order_price_diff("SELL", 100.0, 100.0)
    assert diff_inside == pytest.approx(0.0)

    diff_under = pbr.calc_order_price_diff("sell", 99.0, 100.0)
    assert diff_under == pytest.approx(-0.01)


def test_calc_order_price_diff_invalid_side():
    with pytest.raises(ValueError):
        pbr.calc_order_price_diff("hold", 100.0, 100.0)
