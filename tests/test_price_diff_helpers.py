import pytest

import passivbot_rust as pbr

pytestmark = pytest.mark.usefixtures("require_real_passivbot_rust_module")


@pytest.fixture(scope="module")
def require_real_passivbot_rust_module():
    from rust_utils import verify_loaded_runtime_extension

    assert verify_loaded_runtime_extension()["runtime_compiled_path"] is not None

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
def test_pside_adverse_price_change_fraction(pside, pprice, price, expected):
    assert pbr.calc_pside_adverse_price_change_fraction(pside, pprice, price) == pytest.approx(expected)


@pytest.mark.parametrize("pside", [LONG, SHORT])
def test_pside_adverse_price_change_fraction_returns_zero_for_non_positive_pprice(pside):
    assert pbr.calc_pside_adverse_price_change_fraction(pside, 0.0, 100.0) == 0.0
    assert pbr.calc_pside_adverse_price_change_fraction(pside, float("nan"), 100.0) == 0.0


def test_adverse_change_keywords_identify_the_reference_and_current_price():
    assert pbr.calc_pside_adverse_price_change_fraction(
        pside=LONG, reference_price=100.0, current_price=99.0,
    ) == pytest.approx(.01)


@pytest.mark.parametrize(
    "value,reference,expected",
    [(99.0, 100.0, .01), (101.0, 100.0, .01), (-2.0, -4.0, .5),
     (0.0, 0.0, 0.0), (1.0, 0.0, float("inf")), (-1.0, 0.0, float("inf"))],
)
def test_absolute_relative_difference_fraction(value, reference, expected):
    assert pbr.calc_abs_relative_difference_fraction(value, reference) == pytest.approx(expected)


@pytest.mark.parametrize("name", [
    "calc_diff", "calc_pprice_diff_int", "calc_pside_price_diff_int",
    "calc_price_diff_pside_int", "calc_order_price_diff",
])
def test_retired_native_price_helpers_are_not_exported(name):
    assert not hasattr(pbr, name)
