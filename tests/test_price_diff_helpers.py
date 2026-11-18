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
    "pside,entry,observed,expected",
    [
        (LONG, 100.0, 100.0, 0.0),
        (LONG, 100.0, 98.0, -0.02),
        (LONG, 100.0, 102.0, 0.02),
        (SHORT, 50.0, 50.0, 0.0),
        (SHORT, 50.0, 51.0, -0.02),
        (SHORT, 50.0, 49.0, 0.02),
    ],
)
def test_signed_entry_distance_fraction(pside, entry, observed, expected):
    assert pbr.calc_signed_entry_distance_fraction(pside, entry, observed) == pytest.approx(
        expected
    )


@pytest.mark.parametrize("pside", [LONG, SHORT])
@pytest.mark.parametrize("entry", [0.0, -1.0, float("nan")])
def test_signed_entry_distance_preserves_native_invalid_entry_value(pside, entry):
    assert pbr.calc_signed_entry_distance_fraction(pside, entry, 100.0) == 0.0


def test_entry_distance_keywords_identify_average_entry_and_observed_prices():
    assert pbr.calc_signed_entry_distance_fraction(
        pside=LONG, average_entry_price=100.0, observed_price=98.0,
    ) == pytest.approx(-0.02)


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
    "calc_pside_adverse_price_change_fraction",
])
def test_retired_native_price_helpers_are_not_exported(name):
    assert not hasattr(pbr, name)
