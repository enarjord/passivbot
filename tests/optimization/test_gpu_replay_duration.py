"""Duration policy uses completed work and preserves a fixed execution envelope."""
import math

import pytest

from optimization.gpu.autotune import ReplayDurationController


def test_slow_dispatch_reduces_next_linear_work_duration():
    controller = ReplayDurationController(128)
    controller.observe(128, 4.0)
    assert 1 <= controller.bars < 128
    assert 4.0 * controller.bars / 128 <= controller.target_seconds


def test_growth_requires_sustained_fast_work_and_stays_inside_envelope():
    controller = ReplayDurationController(128)
    controller.observe(128, 4.0)
    smaller = controller.bars
    for _ in range(2):
        controller.observe(controller.bars, 0.1)
        assert controller.bars == smaller
    controller.observe(controller.bars, 0.1)
    assert smaller < controller.bars <= smaller * 2
    for _ in range(30):
        controller.observe(controller.bars, 0.1)
        assert 1 <= controller.bars <= controller.ceiling
    assert controller.bars == controller.ceiling


def test_one_bar_cannot_promise_preemption_or_make_progress_zero():
    controller = ReplayDurationController(1)
    for seconds in (20.0, 1000.0, 0.001):
        controller.observe(1, seconds)
        assert controller.bars == 1


@pytest.mark.parametrize("seconds", [0.0, -1.0, math.nan, math.inf])
def test_invalid_timings_do_not_change_policy(seconds):
    controller = ReplayDurationController(128)
    controller.observe(128, seconds)
    assert controller.bars == 128
    assert controller.fast_chunks == 0


@pytest.mark.parametrize("bars", [0, 1, 127])
def test_incomplete_tail_does_not_train_duration_policy(bars):
    controller = ReplayDurationController(128)
    controller.observe(bars, 100.0)
    assert controller.bars == 128


@pytest.mark.parametrize("ceiling", [0, -1, 1.5, True])
def test_invalid_envelope_rejected(ceiling):
    with pytest.raises(ValueError, match="ceiling"):
        ReplayDurationController(ceiling)


@pytest.mark.parametrize("target", [0.0, -1.0, math.nan, math.inf])
def test_invalid_target_rejected(target):
    with pytest.raises(ValueError, match="target"):
        ReplayDurationController(128, target_seconds=target)
