"""Execution adaptation must preserve work, bounds, and search policy."""

from types import SimpleNamespace

import pytest

from optimization.gpu import autotune as tune


def window(controller, seconds=2.0):
    width = controller.width
    for _ in range(tune.WINDOW + (width not in controller.seen)):
        controller.observe(width, seconds)


def proxy():
    return SimpleNamespace(
        checkpoint_contract={
            "hlcvs": {"shape": [1000, 2, 4], "sha256": "data-a"},
            "timestamps": {"sha256": "time-a"},
            "backtest": {"first_timestamp_ms": 123, "global_warmup_bars": 100},
            "base_params": {"hsl_enabled": 0},
        },
        needed_metrics={"adg"},
        max_dispatch_candidate_bars=1000000,
    )


def test_waits_for_full_window_and_minimum_duration():
    controller = tune.BatchController(1024, 128)
    for _ in range(100):
        controller.observe(127, 5.0)  # partial tails never count
    assert not controller.samples
    window(controller, 0.1)
    assert controller.width == 128
    assert len(controller.samples) == tune.WINDOW
    controller.observe(128, 30.0)
    assert controller.width == 256


def test_noise_does_not_trigger_early_and_failed_trial_rolls_back():
    saved = []
    controller = tune.BatchController(1024, 128, save=lambda *args: saved.append(args))
    controller.observe(128, 1000)  # cold compile excluded
    for index in range(tune.WINDOW):
        controller.observe(128, 50 if index == 12 else 2)
    assert controller.width == 256
    assert saved[0][1] == 64  # rolling median rejects isolated stall
    window(controller, 5)  # larger batch is slower per candidate
    assert controller.width == 128
    assert controller.cooldown == 3
    for _ in range(3):
        window(controller)
        assert controller.width == 128
    window(controller)
    assert controller.width == 64


def test_successful_growth_and_smaller_plateau_are_accepted():
    controller = tune.BatchController(256, 128)
    window(controller)
    window(controller, 3)  # 85.3/s vs 64/s
    assert controller.width == 256
    assert controller.baseline is None
    controller.cooldown = 0
    controller.direction = -1
    window(controller, 4)
    assert controller.width == 128
    window(controller, 2.01)  # nearly equal throughput with half the allocation
    assert controller.width == 128
    assert controller.baseline is None


def test_ceiling_memory_pressure_and_invalid_timings():
    controller = tune.BatchController(150, 128, can_grow=lambda: False)
    for value in [0, -1, float("nan"), float("inf")]:
        controller.observe(128, value)
    assert not controller.samples
    window(controller)
    assert controller.width == 128
    controller.cooldown = 0
    controller.can_grow = lambda: True
    window(controller)
    assert controller.width == 150


def test_fixed_batches_do_not_read_clock():
    def forbidden():
        raise AssertionError("fixed sizing does not collect timings")

    assert list(tune.proxy_batches(SimpleNamespace(), list(range(5)), 3, clock=forbidden)) == [
        (0, [0, 1, 2]),
        (3, [3, 4]),
    ]


@pytest.mark.parametrize("topology", ["single", "fused", "directional"])
def test_outer_batches_honor_replay_and_reduction_history_budget(topology):
    runner = SimpleNamespace(
        _history_bytes_per_candidate=lambda: 120, hsl_scratch_budget_bytes=360,
    )
    item = SimpleNamespace(dispatch_batch_size=8)
    if topology == "single":
        item.runner = runner
    elif topology == "fused":
        item.fused_runner = runner
        # Inactive directional scratch must not constrain the active fused view.
        item.runners = {"unused": SimpleNamespace(
            _history_bytes_per_candidate=lambda: 120, hsl_scratch_budget_bytes=120,
        )}
    else:
        item.runners = {"long": runner, "short": SimpleNamespace(
            _history_bytes_per_candidate=lambda: 0, hsl_scratch_budget_bytes=0,
        )}
    from optimization.gpu.native import CudaBacktestService

    assert CudaBacktestService._dispatch_ceiling(item) == 3
    chunks = list(tune.proxy_batches(item, list(range(8)), 8))
    assert chunks == [(0, [0, 1, 2]), (3, [3, 4, 5]), (6, [6, 7])]


def test_outer_history_bound_preserves_empty_and_oversized_candidate_paths():
    runner = SimpleNamespace(
        _history_bytes_per_candidate=lambda: 0, hsl_scratch_budget_bytes=0,
    )
    item = SimpleNamespace(runner=runner)
    assert tune.history_dispatch_ceiling(item, 8) == 8
    runner._history_bytes_per_candidate = lambda: 120
    # The producer still diagnoses an individual unfit request; never skip it.
    assert list(tune.proxy_batches(item, [0, 1], 8)) == [(0, [0]), (1, [1])]
    def failed():
        raise MemoryError("history metadata failure")
    runner._history_bytes_per_candidate = failed
    with pytest.raises(MemoryError, match="history metadata failure"):
        list(tune.proxy_batches(item, [0], 8))
