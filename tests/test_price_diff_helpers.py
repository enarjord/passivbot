import pytest

import passivbot_rust as pbr

LONG = 0
SHORT = 1


@pytest.mark.parametrize(
    "pside,pprice,price,expected",
    [
        (LONG, 100.0, 100.0, 0.0),
        (LONG, 100.0, 99.0, 0.01),
        (LONG, 100.0, 101.0, -0.01),
        (SHORT, 100.0, 100.0, 0.0),
        (SHORT, 100.0, 101.0, 0.01),
        (SHORT, 100.0, 99.0, -0.01),
    ],
)
def test_calc_pprice_diff_int(pside, pprice, price, expected):
    assert pbr.calc_pprice_diff_int(pside, pprice, price) == pytest.approx(expected)


@pytest.mark.parametrize("pside", [LONG, SHORT])
def test_calc_pprice_diff_int_returns_zero_for_non_positive_pprice(pside):
    assert pbr.calc_pprice_diff_int(pside, 0.0, 100.0) == 0.0
    assert pbr.calc_pprice_diff_int(pside, float("nan"), 100.0) == 0.0


def test_pside_aliases_match_core_function():
    assert pbr.calc_pside_price_diff_int(LONG, 100.0, 95.0) == pbr.calc_pprice_diff_int(
        LONG, 100.0, 95.0
    )
    assert pbr.calc_price_diff_pside_int(SHORT, 100.0, 105.0) == pbr.calc_pprice_diff_int(
        SHORT, 100.0, 105.0
    )
