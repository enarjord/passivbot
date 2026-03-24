from math import comb

import pytest

from optimize import compute_n_partitions


class TestComputeNPartitions:
    def test_2_objectives_exact_match(self):
        # 2 obj: n_points = n_partitions + 1, so pop_size=128 -> n_partitions=127
        assert compute_n_partitions(2, 128) == 127

    def test_3_objectives_closest_match(self):
        # 3 obj, n_partitions=15 -> C(17,2) = 136 points
        # 3 obj, n_partitions=14 -> C(16,2) = 120 points
        # pop_size=128: |136-128|=8, |120-128|=8 -> tie, prefer lower
        result = compute_n_partitions(3, 128)
        assert result in (14, 15)

    def test_4_objectives(self):
        # 4 obj, n_partitions=8 -> C(11,3) = 165
        # 4 obj, n_partitions=7 -> C(10,3) = 120
        # pop_size=128: |165-128|=37, |120-128|=8 -> pick 7
        assert compute_n_partitions(4, 128) == 7

    def test_small_pop(self):
        # 3 obj, pop_size=10: n_partitions=3 -> C(5,2)=10 exact
        assert compute_n_partitions(3, 10) == 3

    def test_minimum_returns_1(self):
        # Any n_obj with pop_size=1 should return 1
        assert compute_n_partitions(2, 1) == 1
        assert compute_n_partitions(3, 1) == 1

    def test_ref_point_count_close_to_pop_size(self):
        """The chosen n_partitions should produce a ref point count
        that is the closest possible to pop_size."""
        for n_obj in (2, 3, 4):
            for pop_size in (10, 50, 100, 200):
                p = compute_n_partitions(n_obj, pop_size)
                n_points = comb(p + n_obj - 1, n_obj - 1)
                # Check neighbors are not closer
                if p > 1:
                    below = comb(p - 1 + n_obj - 1, n_obj - 1)
                    assert abs(n_points - pop_size) <= abs(below - pop_size)
                above = comb(p + 1 + n_obj - 1, n_obj - 1)
                assert abs(n_points - pop_size) <= abs(above - pop_size)

    def test_pop_size_adjustment_formula(self):
        """Verify the pop_size adjustment (smallest multiple of 4 >= n_ref) works."""
        for n_obj in (2, 3, 4):
            for configured_pop in (10, 50, 100, 200):
                p = compute_n_partitions(n_obj, configured_pop)
                n_ref = comb(p + n_obj - 1, n_obj - 1)
                adjusted = n_ref + (-n_ref % 4)
                assert adjusted >= n_ref
                assert adjusted % 4 == 0
                assert adjusted - n_ref < 4

    def test_invalid_inputs_raise(self):
        with pytest.raises(ValueError):
            compute_n_partitions(1, 100)
        with pytest.raises(ValueError):
            compute_n_partitions(3, 0)
