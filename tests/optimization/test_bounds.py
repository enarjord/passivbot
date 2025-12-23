import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from optimization.bounds import (
    Bound,
    enforce_bounds,
)
from optimization.deap_adapters import (
    to_index_space,
    prepare_bounds_for_deap,
    from_index_space,
    mutPolynomialBoundedWrapper,
    cxSimulatedBinaryBoundedWrapper,
)


class TestBound:
    """Test Bound dataclass and its methods."""

    def test_continuous_bound_properties(self):
        b = Bound(0.0, 10.0)
        assert b.low == 0.0
        assert b.high == 10.0
        assert b.step is None
        assert not b.is_stepped
        with pytest.raises(ValueError, match="max_index only valid for stepped parameters"):
            _ = b.max_index

    def test_stepped_bound_properties(self):
        b = Bound(0.0, 1.0, 0.1)
        assert b.low == 0.0
        assert b.high == 1.0
        assert b.step == 0.1
        assert b.is_stepped
        assert b.max_index == 10

    def test_quantize_continuous(self):
        b = Bound(0.0, 10.0)
        assert b.quantize(5.5) == 5.5
        assert b.quantize(-1.0) == 0.0
        assert b.quantize(11.0) == 10.0

    def test_quantize_stepped(self):
        b = Bound(0.0, 1.0, 0.1)
        assert b.quantize(0.14) == pytest.approx(0.1)
        assert b.quantize(0.16) == pytest.approx(0.2)
        assert b.quantize(-0.1) == 0.0
        assert b.quantize(1.1) == 1.0
        assert b.quantize(0.5) == pytest.approx(0.5)

    def test_random_on_grid_continuous(self):
        b = Bound(0.0, 10.0)
        val = b.random_on_grid()
        assert 0.0 <= val <= 10.0

    def test_random_on_grid_stepped(self):
        b = Bound(0.0, 1.0, 0.2)
        expected = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        for _ in range(20):
            val = b.random_on_grid()
            # Must be one of [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            assert any(val == pytest.approx(e) for e in expected)

    def test_value_to_index(self):
        b_cont = Bound(0.0, 10.0)
        assert b_cont.value_to_index(5.0) == 5.0

        b_step = Bound(1.0, 2.0, 0.1)
        assert b_step.value_to_index(1.3) == pytest.approx(3.0)

    def test_index_to_value(self):
        b_cont = Bound(0.0, 10.0)
        assert b_cont.index_to_value(5.0) == 5.0
        assert b_cont.index_to_value(-1.0) == 0.0
        assert b_cont.index_to_value(11.0) == 10.0

        b_step = Bound(1.0, 2.0, 0.1)
        assert b_step.index_to_value(3.0) == pytest.approx(1.3)
        assert b_step.index_to_value(3.4) == pytest.approx(1.3)
        assert b_step.index_to_value(3.6) == pytest.approx(1.4)

    def test_get_index_bounds(self):
        b_cont = Bound(1.0, 5.0)
        assert b_cont.get_index_bounds() == (1.0, 5.0)

        b_step = Bound(1.0, 2.0, 0.1)
        assert b_step.get_index_bounds() == (0.0, 10.0)

    def test_from_config_single_value(self):
        b = Bound.from_config("test", 5.0)
        assert b.low == 5.0
        assert b.high == 5.0
        assert b.step is None

    def test_from_config_single_element_list(self):
        b = Bound.from_config("test", [5.0])
        assert b.low == 5.0
        assert b.high == 5.0
        assert b.step is None

    def test_from_config_two_values(self):
        b = Bound.from_config("test", [10.0, 5.0])
        assert b.low == 5.0
        assert b.high == 10.0
        assert b.step is None

    def test_from_config_three_values(self):
        b = Bound.from_config("test", [0.0, 1.0, 0.1])
        assert b.low == 0.0
        assert b.high == 1.0
        assert b.step == 0.1

    def test_from_config_continuous_with_zero_or_none_step(self):
        b1 = Bound.from_config("test", [0.0, 1.0, 0.0])
        assert b1.step is None
        b2 = Bound.from_config("test", [0.0, 1.0, None])
        assert b2.step is None

    def test_from_config_errors(self):
        with pytest.raises(Exception, match="malformed bound"):
            Bound.from_config("test", [])
        with pytest.raises(Exception, match="malformed bound"):
            Bound.from_config("test", "not a bound")

    def test_from_config_warning_cases(self, caplog):
        # Case with 4 elements
        caplog.clear()
        b1 = Bound.from_config("test", [0.0, 1.0, 0.1, 0.2])
        assert b1.step is None
        assert any("expected 1, 2, or 3" in r.message for r in caplog.records)

        # Case with non-numeric step
        caplog.clear()
        b2 = Bound.from_config("test", [0.0, 1.0, "not a number"])
        assert b2.step is None
        assert any("step is not a number" in r.message for r in caplog.records)

        # Case with negative step
        caplog.clear()
        b3 = Bound.from_config("test", [0.0, 1.0, -0.1])
        assert b3.step is None
        assert any("step must be > 0" in r.message for r in caplog.records)

        # Case with step > range
        caplog.clear()
        b4 = Bound.from_config("test", [0.0, 1.0, 1.5])
        assert b4.step is None
        assert any("is larger than range" in r.message for r in caplog.records)


class TestBoundUtils:
    """Test utility functions in bounds.py."""

    def test_enforce_bounds(self):
        bounds = [Bound(0.0, 1.0), Bound(1.0, 2.0, 0.1)]
        values = [1.5, 1.34]
        result = enforce_bounds(values, bounds)
        assert result[0] == 1.0
        assert result[1] == pytest.approx(1.3)

    def test_enforce_bounds_with_sig_digits(self):
        bounds = [Bound(0.0, 1.0), Bound(1.0, 2.0)]
        values = [0.123456, 1.123456]
        result = enforce_bounds(values, bounds, sig_digits=3)
        assert result[0] == 0.123
        assert result[1] == 1.12

    def test_enforce_bounds_mismatch_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            enforce_bounds([1.0], [Bound(0.0, 1.0), Bound(0.0, 1.0)])
