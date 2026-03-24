import numpy as np
import pytest
from optimization.bounds import extract_bounds_arrays


class TestExtractBoundsArrays:
    def test_basic_extraction(self):
        config = {
            "bot": {
                "long": {"ema_span_0": 200.0, "n_positions": 5},
                "short": {"ema_span_0": 100.0, "n_positions": 3},
            },
            "optimize": {
                "bounds": {
                    "long_ema_span_0": [200, 1440],
                    "long_n_positions": [1, 20],
                    "short_ema_span_0": [100, 1440],
                    "short_n_positions": [1, 10],
                },
            },
        }
        xl, xu, keys = extract_bounds_arrays(config)
        assert isinstance(xl, np.ndarray)
        assert isinstance(xu, np.ndarray)
        assert keys == [
            "long_ema_span_0",
            "long_n_positions",
            "short_ema_span_0",
            "short_n_positions",
        ]
        np.testing.assert_array_equal(xl, [200, 1, 100, 1])
        np.testing.assert_array_equal(xu, [1440, 20, 1440, 10])

    def test_fixed_param_equal_bounds(self):
        config = {
            "bot": {"long": {"total_wel": 1.25}},
            "optimize": {"bounds": {"long_total_wel": [1.25, 1.25]}},
        }
        xl, xu, keys = extract_bounds_arrays(config)
        assert xl[0] == xu[0] == 1.25

    def test_single_value_bound(self):
        config = {
            "bot": {"long": {"param": 5.0}},
            "optimize": {"bounds": {"long_param": 5.0}},
        }
        xl, xu, keys = extract_bounds_arrays(config)
        assert xl[0] == xu[0] == 5.0

    def test_three_element_bound_ignores_step(self):
        config = {
            "bot": {"long": {"spacing": 0.02}},
            "optimize": {"bounds": {"long_spacing": [0.02, 0.04, 0.001]}},
        }
        xl, xu, keys = extract_bounds_arrays(config)
        assert xl[0] == 0.02
        assert xu[0] == 0.04

    def test_missing_bound_uses_bot_value(self):
        config = {
            "bot": {"long": {"a": 1.0, "b": 2.0}},
            "optimize": {"bounds": {"long_a": [0, 10]}},
        }
        xl, xu, keys = extract_bounds_arrays(config)
        assert keys == ["long_a", "long_b"]
        assert xl[1] == xu[1] == 2.0

    def test_bounds_sorted_low_high(self):
        config = {
            "bot": {"long": {"x": 5.0}},
            "optimize": {"bounds": {"long_x": [10, 1]}},
        }
        xl, xu, keys = extract_bounds_arrays(config)
        assert xl[0] == 1.0
        assert xu[0] == 10.0
