from live.hsl_budget import BudgetObservations, CURRENT_ORDER


def capture(history, *, start=0, now=60_000, balance=100, slots=1, balance_at=None):
    return history.capture(("coin", "long", "A"), start=start, now=now,
                           balance=balance, slots=slots, scale=False, exposure=None,
                           balance_at=balance_at)


def test_bootstrap_estimate_is_anchored_and_balance_change_does_not_rewrite_it():
    history = BudgetObservations()
    first = capture(history)
    next_ = capture(history, now=120_000, balance=1000)
    assert next_[:2] == first
    assert first[0] == dict(timestamp=0, order=-1, estimated=True,
                            balance=100, slots=1,
                            scale_budget_with_excess_allowance=False,
                            exposure_budget=None)
    assert next_[-1]["balance"] == 1000
    assert next_[-1]["order"] == CURRENT_ORDER


def test_repeated_observation_is_disposable_and_result_mutation_cannot_edit_history():
    history = BudgetObservations()
    first = capture(history)
    assert capture(history, now=120_000) == first
    first[0]["balance"] = -1
    assert capture(history, now=120_000)[0]["balance"] == 100


def test_budget_fact_before_clipped_window_is_retained_as_anchor():
    history = BudgetObservations()
    capture(history)
    capture(history, now=120_000, balance=101)
    retained = capture(history, start=100_000, now=180_000, balance=102)
    assert retained[0]["timestamp"] == 60_000
    assert not retained[0]["estimated"]


def test_new_current_read_in_same_millisecond_replaces_only_current_phase():
    history = BudgetObservations()
    initial = capture(history)
    revised = capture(history, balance=200)
    assert revised[0] == initial[0]
    assert len(revised) == len(initial)
    assert revised[-1]["balance"] == 200


def test_unchanged_policy_preserves_balance_source_time_before_later_execution():
    history = BudgetObservations()
    first = capture(history, now=1, balance=1000)
    revised = capture(history, now=60_000, balance_at=59_999, balance=10_000)
    assert revised[:len(first)] == first
    assert revised[-1]["timestamp"] == 59_999
    assert revised[-1]["observed_at"] == 60_000
    assert revised[-1]["order"] == CURRENT_ORDER
    assert capture(history, now=90_000, balance_at=89_999, balance=10_000) == revised


def test_changed_slots_do_not_inherit_an_older_balance_source_clock():
    history = BudgetObservations()
    capture(history, now=1, balance=1000)
    revised = capture(history, now=60_000, balance_at=59_999, balance=10_000, slots=2)
    assert revised[-2]["timestamp"] == 59_999
    assert revised[-2]["balance"] == 10_000
    assert revised[-2]["slots"] == 1
    assert revised[-1]["timestamp"] == 60_000
    assert revised[-1]["observed_at"] == 60_000
    assert revised[-1]["slots"] == 2


def test_first_policy_observation_does_not_claim_an_earlier_configuration():
    observed = capture(BudgetObservations(), now=60_000, balance_at=59_999)
    assert observed[0]["estimated"]
    assert observed[-1]["timestamp"] == observed[-1]["observed_at"] == 60_000
