import math

import pytest

from optimization.gpu.coalescing import BatchCoalescer


def burst_stream(policy, dataset="a"):
    now = 1.0
    policy.arrived(dataset, now, active=False)
    for _ in range(8):
        for gap in (0.0001, 0.0001, 0.0001, 0.05):
            now += gap
            policy.arrived(dataset, now, active=True)
    return now


def warm(policy, dataset="a", count=16, seconds=1.0):
    policy.observe(dataset, count, 20.0)
    policy.observe(dataset, count, seconds)


def test_burst_gaps_control_idle_tail_and_absolute_bound():
    policy = BatchCoalescer()
    now = burst_stream(policy)
    warm(policy)
    assert policy.deadline("a", now) == pytest.approx(now + 0.1)
    started = now
    for _ in range(20):
        now += 0.04
        policy.arrived("a", now, active=True)
    assert policy.deadline("a", started) == pytest.approx(started + 0.5)
    # Work buffered through another replay receives no new idle-tail allowance.
    assert policy.deadline("a", now + 5) < now + 5


def test_first_allocation_per_count_and_invalid_timing_are_not_warm_evidence():
    policy = BatchCoalescer()
    now = burst_stream(policy)
    for seconds in (0, -1, math.inf, math.nan):
        policy.observe("a", 16, seconds)
    assert not policy._streams["a"].seen_counts
    policy.observe("a", 16, 20.0)
    assert policy.deadline("a", now) == pytest.approx(now + 0.005)
    policy.observe("a", 16, 0.02)
    assert policy.deadline("a", now) == pytest.approx(now + 0.02)
    policy.observe("a", 32, 40.0)
    assert policy.deadline("a", now) == pytest.approx(now + 0.02)


def test_completed_cohort_gaps_and_other_datasets_do_not_pollute_cadence():
    policy = BatchCoalescer()
    burst_stream(policy)
    warm(policy)
    prior = tuple(policy._streams["a"].gaps)
    policy.arrived("a", 100.0, active=False)
    assert tuple(policy._streams["a"].gaps) == prior
    assert policy.deadline("a", 100.0) == pytest.approx(100.1)
    policy.arrived("b", 100.0, active=False)
    assert policy.deadline("b", 100.0) == pytest.approx(100.005)
    assert policy._streams["b"].replay_seconds is None


def test_sparse_initial_stream_and_faster_work_retain_bounded_latency():
    policy = BatchCoalescer()
    policy.arrived("a", 1.0, active=False)
    warm(policy, seconds=0.0001)
    assert policy.deadline("a", 1.0) == pytest.approx(1.005)
    now = burst_stream(policy)
    assert policy.deadline("a", now) == pytest.approx(now + 0.005)
    assert len(policy._streams["a"].gaps) == 32


def test_timing_smoothing_uses_successive_warm_observations():
    policy = BatchCoalescer()
    now = burst_stream(policy)
    warm(policy, seconds=0.1)
    policy.observe("a", 16, 0.2)
    assert policy._streams["a"].replay_seconds == pytest.approx(0.12)
    assert policy.deadline("a", now) == pytest.approx(now + 0.1)
