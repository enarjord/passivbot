"""Window-level queued demand must survive finite cohort boundaries."""

import pytest

from optimization.gpu.autotune import WINDOW
from optimization.gpu.execution_tuning import ExecutionBatchTuner


def policy():
    result = ExecutionBatchTuner(initial=4, headroom=lambda: True)
    for name in ('a', 'b'):
        result.constrain(name, 16)
        result.width(name, 16)
    return result


def window(result, name='a', backlog=0):
    width = result.width(name, 16)
    result.observe(name, width, 2, backlog=backlog, closing=False)
    for _ in range(WINDOW):
        result.observe(name, width, 2, backlog=backlog, closing=False)


def test_growth_uses_work_available_during_window_not_only_last_cohort_tail():
    result = policy()
    result.observe('a', 4, 2, backlog=4, closing=False)  # Cold shape, successful work.
    for index in range(WINDOW):
        result.observe('a', 4, 2, backlog=4 if index % 2 == 0 else 0, closing=False)
    assert result.width('a', 16) == 8
    assert result.controllers['a'].baseline == (4, 2.0)


def test_another_dataset_cannot_supply_growth_demand():
    result = policy()
    result.observe('a', 4, 2, backlog=100, closing=False)
    window(result, 'b', backlog=0)
    assert result.width('b', 16) == 2
    assert result.width('a', 16) == 4


def test_invalid_work_cannot_supply_growth_demand():
    result = policy()
    for count, seconds in ((5, 2), (4, float('nan')), (4, 0)):
        result.observe('a', count, seconds, backlog=100, closing=False)
    window(result, backlog=0)
    assert result.width('a', 16) == 2


def test_consumed_window_demand_does_not_authorize_future_growth():
    result = policy()
    window(result, backlog=100)
    assert result.width('a', 16) == 8
    window(result, backlog=0)  # Faster accepted trial; one cooldown window follows.
    assert result.controllers['a'].baseline is None
    assert result.width('a', 16) == 8
    window(result, backlog=0)  # Consume cooldown without reviving earlier backlog.
    assert result.width('a', 16) == 8
    window(result, backlog=0)
    assert result.width('a', 16) == 4


def test_new_prepared_ceiling_discards_previous_window_demand():
    result = policy()
    result.observe('a', 4, 2, backlog=100, closing=False)
    result.constrain('a', 8)
    assert result.width('a', 16) == 4
    window(result, backlog=0)
    assert result.width('a', 16) == 2


@pytest.mark.parametrize('ceiling', [2, 32])
def test_ceiling_change_before_observation_discards_obsolete_window(ceiling):
    from tools.gpu_cohort_benchmark import _observe_batches
    result = policy()
    evidence = _observe_batches(result, [])
    result.observe('a', 4, 2, backlog=100, closing=False)
    for _ in range(WINDOW - 1):
        result.observe('a', 4, 2, backlog=100, closing=False)
    obsolete = result.controllers['a']
    result.constrain('a', ceiling)  # Production applies this inside successful replay.
    result.observe('a', 4, 2, backlog=0, closing=False)
    assert not evidence['completed_windows']
    assert obsolete.width == 4 and obsolete.baseline is None
    assert 'a' not in result.controllers and 'a' not in result._demand
    assert result.width('a', 16) == min(4, ceiling)
    assert not result.controllers['a'].samples
    assert not result.controllers['a'].seen
    assert result.controllers['b'].width == 4


def test_unchanged_ceiling_preserves_current_window_and_demand():
    result = policy()
    result.observe('a', 4, 2, backlog=100, closing=False)
    owner = result.controllers['a']
    result.constrain('a', 16)
    for _ in range(WINDOW):
        result.observe('a', 4, 2, backlog=0, closing=False)
    assert result.controllers['a'] is owner and owner.width == 8
