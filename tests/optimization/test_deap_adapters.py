import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from optimization.bounds import Bound
from optimization.deap_adapters import (
    to_index_space,
    prepare_bounds_for_deap,
    from_index_space,
    mutPolynomialBoundedWrapper,
    cxSimulatedBinaryBoundedWrapper,
)


class TestDeapAdapters:
    """Test DEAP adapter functions."""

    def test_to_index_space(self):
        bounds = [Bound(0.0, 10.0), Bound(1.0, 2.0, 0.1)]
        values = [5.0, 1.3]
        idx_vals, idx_low, idx_up = to_index_space(values, bounds)
        assert idx_vals == [5.0, pytest.approx(3.0)]
        assert idx_low == [0.0, 0.0]
        assert idx_up == [10.0, 10.0]

    def test_prepare_bounds_for_deap(self):
        low = [0.0, 5.0]
        up = [10.0, 5.0]
        t_low, t_up, mask = prepare_bounds_for_deap(low, up)
        assert mask[0] == False
        assert mask[1] == True
        assert t_low[1] < 5.0
        assert t_up[1] > 5.0

    def test_from_index_space(self):
        bounds = [Bound(0.0, 10.0), Bound(1.0, 2.0, 0.1)]
        idx_vals = [5.0, 3.0]
        mask = np.array([False, False])
        vals = from_index_space(idx_vals, bounds, mask)
        assert vals == [5.0, pytest.approx(1.3)]

        mask_with_equal = np.array([False, True])
        vals_eq = from_index_space([5.0, 99.0], bounds, mask_with_equal)
        assert vals_eq[1] == 1.0  # Reset to low

    @patch("optimization.deap_adapters.deap_tools")
    def test_mutPolynomialBoundedWrapper(self, mock_deap_tools):
        mock_deap_tools.mutPolynomialBounded = MagicMock()
        individual = [0.5, 1.3]
        bounds = [Bound(0.0, 1.0), Bound(1.0, 2.0, 0.1)]

        mutPolynomialBoundedWrapper(individual, eta=20.0, indpb=0.5, bounds=bounds)

        assert mock_deap_tools.mutPolynomialBounded.called
        # Verify that it was called with index space values if needed
        # (our test individual is already on grid so indices are 0.5 and 3.0)

    @patch("optimization.deap_adapters.deap_tools")
    def test_cxSimulatedBinaryBoundedWrapper(self, mock_deap_tools):
        mock_deap_tools.cxSimulatedBinaryBounded = MagicMock()
        ind1 = [0.5, 1.3]
        ind2 = [0.6, 1.4]
        bounds = [Bound(0.0, 1.0), Bound(1.0, 2.0, 0.1)]

        cxSimulatedBinaryBoundedWrapper(ind1, ind2, eta=20.0, bounds=bounds)

        assert mock_deap_tools.cxSimulatedBinaryBounded.called
