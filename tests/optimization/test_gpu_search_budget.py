"""Max mode budgets wider work independently of the kept parent population."""

import pickle

import pytest

from optimization.gpu.search_budget import CudaMaxSearchBudget


@pytest.mark.parametrize("target_width,cohort_width", [
    (256, 256), (512, 512), (64, 128),
])
def test_wider_cohort_respects_target_width_and_parent_population(target_width, cohort_width):
    plan = CudaMaxSearchBudget(population_size=128, target_width=target_width)

    assert plan.population_size == 128
    assert plan.target_width == target_width
    assert plan.cohort_width == cohort_width


@pytest.mark.parametrize("iters,authored,expected", [
    (1, 0, 1), (7, 2, 7), (256, 0, 256), (1024, 120, 136), (1024, 256, 0), (1024, 300, 0),
])
def test_authored_seeds_do_not_spend_bootstrap_budget(iters, authored, expected):
    plan = CudaMaxSearchBudget(population_size=128, target_width=256)

    assert plan.bootstrap_size(iters=iters, authored_seed_count=authored) == expected


def test_persisted_plan_keeps_wide_cohorts_and_caps_short_final_tail():
    plan = pickle.loads(pickle.dumps(CudaMaxSearchBudget(128, 256)))

    assert plan.bootstrap_size(iters=500, authored_seed_count=64, allocated=480) == 20
    assert plan.offspring_size(iters=500, allocated=192) == 256
    assert plan.offspring_size(iters=500, allocated=448) == 52
    assert plan.offspring_size(iters=500, allocated=500) == 0
    assert plan.offspring_size(iters=700, allocated=500) == 200
    assert plan.offspring_size(iters=250, allocated=500) == 0


@pytest.mark.parametrize("population_size,observed_width", [(True, 64), (0, 64), (128, 0), (128, 64.5)])
def test_invalid_queue_sizes_are_rejected(population_size, observed_width):
    with pytest.raises(ValueError, match="positive integer"):
        CudaMaxSearchBudget(population_size, observed_width)


@pytest.mark.parametrize("iters,allocated", [(True, 0), (0, 0), (1.5, 0), (8, -1), (8, False)])
def test_invalid_search_slot_counts_are_rejected(iters, allocated):
    plan = CudaMaxSearchBudget(128, 256)
    with pytest.raises(ValueError, match="integer"):
        plan.remaining(iters=iters, allocated=allocated)
