import json

import pytest

from live import hsl_budget
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


def test_late_balance_correction_inserts_replaces_and_prunes_in_causal_order():
    history = BudgetObservations()
    capture(history, now=1, balance=1000)
    capture(history, now=30, balance_at=29, balance=1500)

    late = capture(history, now=60, balance_at=20, balance=1200, slots=2)
    assert [(fact["timestamp"], fact["balance"], fact["slots"]) for fact in late] == [
        (0, 1000, 1), (1, 1000, 1), (20, 1200, 1), (29, 1500, 1), (60, 1200, 2),
    ]

    corrected = capture(history, start=25, now=70, balance_at=20, balance=1300, slots=2)
    assert [(fact["timestamp"], fact["observed_at"], fact["balance"], fact["slots"])
            for fact in corrected] == [
        (20, 70, 1300, 1), (29, 30, 1500, 1), (60, 70, 1300, 2),
    ]
    assert late[2]["balance"] == 1200


def test_exposure_policy_in_captured_facts_is_detached_from_inputs_and_future_captures():
    history = BudgetObservations()
    exposure = dict(wallet_exposure_limit_ratio=0.2, total_wallet_exposure_limit_ratio=1.0,
                    we_excess_allowance_fraction=0.1)

    def capture_exposure(now, balance, policy):
        return history.capture(("coin", "long", "A"), start=0, now=now,
                               balance=balance, slots=1, scale=True, exposure=policy)

    first = capture_exposure(60_000, 100, exposure)
    exposure["wallet_exposure_limit_ratio"] = 0.4
    first[-1]["exposure_budget"]["wallet_exposure_limit_ratio"] = 0.9

    revised = capture_exposure(120_000, 200, exposure)
    assert [fact["exposure_budget"]["wallet_exposure_limit_ratio"] for fact in revised] == [
        0.2, 0.2, 0.4,
    ]
    assert first[-1]["exposure_budget"]["wallet_exposure_limit_ratio"] == 0.9


def encode(history, *, start=0, now=60_000, balance=100, slots=1, balance_at=None, exposure=None):
    return history._encode_rows(history._capture_rows(
        ("coin", "long", "A"), start=start, now=now, balance=balance, slots=slots,
        scale=exposure is not None, exposure=exposure, balance_at=balance_at,
    ))


def test_budget_json_encodes_only_new_rows_and_releases_retired_versions(monkeypatch):
    history = BudgetObservations()
    observed = []

    def dumps(value, **kwargs):
        observed.append(value)
        return json.dumps(value, **kwargs)

    monkeypatch.setattr(hsl_budget, "dumps", dumps)
    initial = encode(history)
    assert len(observed) == 2
    observed.clear()
    assert encode(history) == initial
    assert not observed
    retired = next(reversed(history._encoded_rows))
    revised = encode(history, balance=200)
    assert json.loads(revised)[-1]["balance"] == 200
    assert len(observed) == 1
    assert retired not in history._encoded_rows

    encode(history, start=60_000, now=70_000)
    live_ids = {id(fact) for facts in history._scopes.values() for fact in facts}
    assert set(history._encoded_rows) == live_ids
    assert history._encoded_bytes == sum(map(len, history._encoded_rows.values()))
    history.retain_scopes(set())
    assert not history._encoded_rows and history._encoded_bytes == 0


@pytest.mark.parametrize("entry_limit,byte_limit", [(2, 4_096), (100, 320)])
def test_budget_json_keeps_exact_rows_at_the_cache_limit(monkeypatch, entry_limit, byte_limit):
    monkeypatch.setattr(hsl_budget, "_MAX_ENCODED_ROWS", entry_limit)
    monkeypatch.setattr(hsl_budget, "_MAX_ENCODED_BYTES", byte_limit)
    history, reference = BudgetObservations(), BudgetObservations()
    for kwargs in (
        dict(now=10),
        dict(now=30, balance=200),
        dict(now=40, start=15, balance=300),
        dict(now=60, start=15, balance=250, balance_at=20, slots=2),
        dict(now=70, start=25, balance=270, balance_at=20, slots=2),
    ):
        assert encode(history, **kwargs) == json.dumps(capture(reference, **kwargs), allow_nan=False)
        assert len(history._encoded_rows) <= entry_limit
        assert history._encoded_bytes == sum(map(len, history._encoded_rows.values())) <= byte_limit
        live_ids = {id(fact) for facts in history._scopes.values() for fact in facts}
        assert set(history._encoded_rows) <= live_ids


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_budget_json_does_not_cache_nonfinite_rows(value):
    history, reference = BudgetObservations(), BudgetObservations()
    encode(history)
    capture(reference)
    with pytest.raises(ValueError) as baseline:
        json.dumps(capture(reference, balance=value), allow_nan=False)
    with pytest.raises(ValueError) as actual:
        encode(history, balance=value)
    assert str(actual.value) == str(baseline.value)
    assert len(history._encoded_rows) == 1
    assert encode(history, balance=200) == json.dumps(capture(reference, balance=200), allow_nan=False)
