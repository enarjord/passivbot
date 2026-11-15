import numpy as np
import pytest

from optimization.bounds import Bound, enforce_bounds
from optimization.gpu.search_sampling import max_starting_vectors


def sample(starts, bounds, *, width, seed=12, sig_digits=5, preserve_indices=()):
    return max_starting_vectors(
        starts, bounds=bounds, sig_digits=sig_digits, width=width,
        random_state=np.random.default_rng(seed), preserve_indices=preserve_indices,
    )


def test_seeded_bootstrap_preserves_order_and_uses_only_repaired_mutants():
    seeds = [(0.0, 3.0, 0.125, 7.5), (1.0, 5.0, 0.625, 7.5)]
    bounds = [Bound(0.0, 2.0, 1.0), Bound(3.0, 5.0, 0.1), Bound(0.125, 0.625), Bound(7.5, 7.5)]
    starts = sample(seeds, bounds, width=1024, preserve_indices=(0,))

    assert len(starts) == len({tuple(vector) for vector in starts}) == 1024
    assert starts[:2] == seeds
    assert starts == sample(seeds, bounds, width=1024, preserve_indices=(0,))
    assert starts != sample(seeds, bounds, width=1024, seed=13, preserve_indices=(0,))
    prior = {tuple(vector) for vector in seeds}
    for vector in starts[2:]:
        assert vector[0] in {0.0, 1.0}
        assert vector[3] == 7.5
        assert vector == enforce_bounds(vector, bounds, 5)
        assert any(sum(a != b for a, b in zip(vector, parent, strict=True)) == 1 for parent in prior)
        prior.add(tuple(vector))


def test_original_seeds_above_target_are_kept_in_authored_order():
    seeds = [(0.5,), (0.3,), (0.2,)]
    assert sample(seeds, [Bound(0.0, 1.0)], width=2) == seeds


@pytest.mark.parametrize("supply_seeds", [True, False])
def test_finite_grid_returns_distinct_partial_width(supply_seeds):
    bounds = [Bound(0.0, 1.0, 1.0), Bound(0.0, 1.0, 1.0), Bound(3.0, 3.0, 1.0)]
    seeds = [(1.0, 1.0, 3.0)] if supply_seeds else []
    starts = sample(seeds, bounds, width=4096)
    assert len(starts) == 4
    assert {tuple(vector) for vector in starts} == {
        (0.0, 0.0, 3.0), (0.0, 1.0, 3.0), (1.0, 0.0, 3.0), (1.0, 1.0, 3.0),
    }


def test_unseeded_bootstrap_uses_the_caller_rng_and_grid():
    bounds = [Bound(0.0, 2.0, 1.0), Bound(0.125, 0.625), Bound(0.7, 1.3, 0.17)]
    starts = sample([], bounds, width=256, preserve_indices=(0,))
    assert starts == sample([], bounds, width=256, preserve_indices=(0,))
    assert starts != sample([], bounds, width=256, seed=13)
    assert len(starts) == len({tuple(vector) for vector in starts}) == 256
    assert {vector[0] for vector in starts} == {0.0, 1.0, 2.0}
    assert starts == [enforce_bounds(vector, bounds, 5) for vector in starts]


def test_fixed_or_anchor_only_space_returns_without_needing_mutation():
    assert sample([], [Bound(0.5, 0.5)], width=1024) == [[0.5]]
    assert sample([], [], width=1024) == [[]]
    seeds = [(1.0, 0.5), (0.0, 0.5)]
    bounds = [Bound(0.0, 2.0, 1.0), Bound(0.5, 0.5)]
    assert sample(seeds, bounds, width=1024, preserve_indices=(0,)) == seeds


@pytest.mark.parametrize("width", [0, -1, True, 2.5])
def test_invalid_bootstrap_width_is_rejected(width):
    with pytest.raises(ValueError, match="positive integer"):
        sample([], [Bound(0.0, 1.0)], width=width)
