import numpy as np
import pytest
from unittest.mock import MagicMock
from optimization.repair import SignificantDigitsRepair


class TestSignificantDigitsRepair:
    def _make_problem(self, xl, xu):
        p = MagicMock()
        p.xl = np.array(xl)
        p.xu = np.array(xu)
        return p

    def test_rounds_to_sig_digits(self):
        repair = SignificantDigitsRepair(3)
        problem = self._make_problem([0.0, 0.0], [1.0, 1.0])
        X = np.array([[0.123456, 0.987654]])
        result = repair._do(problem, X)
        np.testing.assert_array_almost_equal(result, [[0.123, 0.988]])

    def test_clamps_to_bounds(self):
        repair = SignificantDigitsRepair(3)
        problem = self._make_problem([0.0, 0.0], [1.0, 1.0])
        X = np.array([[1.5, -0.5]])
        result = repair._do(problem, X)
        assert result[0, 0] <= 1.0
        assert result[0, 1] >= 0.0

    def test_preserves_fixed_params(self):
        repair = SignificantDigitsRepair(3)
        problem = self._make_problem([1.25, 0.0], [1.25, 1.0])
        X = np.array([[1.3, 0.5]])
        result = repair._do(problem, X)
        assert result[0, 0] == 1.25

    def test_handles_zero(self):
        repair = SignificantDigitsRepair(3)
        problem = self._make_problem([-1.0], [1.0])
        X = np.array([[0.0]])
        result = repair._do(problem, X)
        assert result[0, 0] == 0.0

    def test_batch(self):
        repair = SignificantDigitsRepair(3)
        problem = self._make_problem([0.0], [10.0])
        X = np.array([[1.23456], [7.89012], [0.00123456]])
        result = repair._do(problem, X)
        assert result.shape == (3, 1)
        np.testing.assert_array_almost_equal(
            result, [[1.23], [7.89], [0.00123]]
        )
