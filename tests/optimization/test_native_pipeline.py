from optimization.native_pipeline import ResultCadence


def test_result_grouping_grows_from_measured_cost_and_shrinks_immediately():
    cadence = ResultCadence()
    assert cadence.limit == 1
    for expected in (2, 4, 8, 16, 32, 50):
        cadence.observe(cadence.limit, cadence.limit * 0.001)
        assert cadence.limit == expected
    cadence.observe(20, 0.4)
    assert cadence.limit == 2
    cadence.observe(2, 0.2)
    assert cadence.limit == 1


def test_fast_work_is_bounded_and_idle_or_unmeasured_work_does_not_tune():
    cadence = ResultCadence()
    for _ in range(12):
        cadence.observe(1, 0.000001)
    assert cadence.limit == 256
    for count, seconds in ((0, 30), (1, 0), (1, float("nan")), (1, float("inf"))):
        cadence.observe(count, seconds)
        assert cadence.limit == 256


def test_loss_of_cadence_state_does_not_change_search_or_execution_settings():
    first, fresh = ResultCadence(), ResultCadence()
    first.observe(1, 0.001)
    assert fresh.limit == 1
    assert first.budget_seconds == fresh.budget_seconds == 0.05
