"""Window-level queued demand must survive finite cohort boundaries."""

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
