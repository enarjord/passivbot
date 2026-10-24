"""Scope-local EMA-tail workspaces preserve the previous row-wise reduction."""

from types import SimpleNamespace
import weakref

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from optimization.gpu import ema_tail


def _previous_reduction(samples):
    batch_size, scope_count, capacity = samples.shape
    if not batch_size or not capacity:
        return torch.zeros((batch_size, scope_count), dtype=samples.dtype, device=samples.device)
    matrix = samples.reshape(batch_size * scope_count, capacity)
    finite = torch.isfinite(matrix)
    counts = finite.sum(dim=1)
    values = torch.where(finite, matrix.abs(), 0.0)
    top = torch.topk(values, max(capacity // 100, 1), dim=1, sorted=True).values
    wanted = (counts // 100).clamp(min=1)
    selected = torch.arange(top.shape[1], device=samples.device)[None, :] < wanted[:, None]
    means = torch.where(selected, top, 0.0).sum(dim=1) / wanted
    return torch.where(counts > 0, means, 0.0).reshape(batch_size, scope_count)


def _device_or_skip(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS required")
    return device


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("scope_count", [2, 3])
@pytest.mark.parametrize("capacity", [1, 99, 100, 1003, 4097])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_scope_reduction_is_exactly_equal_to_previous_rows(
    device, scope_count, capacity, noncontiguous,
):
    device = _device_or_skip(device)
    rng = np.random.default_rng(43)
    samples = np.full((9, scope_count, capacity), np.nan, dtype=np.float32)
    for row, count in enumerate((0, 1, 99, 100, 199, 200, 999, 1000, capacity)):
        indices = rng.permutation(capacity)[:min(count, capacity)]
        if row % 2:
            values = rng.choice(
                np.asarray([-.002, -.001, -0., 0., .001, .002], dtype=np.float32),
                (len(indices), scope_count),
            )
        else:
            values = rng.uniform(-.002, .002, (len(indices), scope_count)).astype(np.float32)
        samples[row, :, indices] = values
    samples[1, 0, 0], samples[1, 1, 0] = np.inf, -np.inf
    tensor = torch.tensor(samples, device=device)
    if noncontiguous:
        backing = torch.empty((9, scope_count, capacity * 2), device=device)
        backing[:, :, ::2] = tensor
        tensor = backing[:, :, ::2]
        assert not tensor.is_contiguous()
    original = tensor.clone()
    expected = _previous_reduction(tensor)
    actual = ema_tail.drawdown_ema_tail_from_samples(tensor)
    assert actual.device == tensor.device and actual.shape == (9, scope_count)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(tensor, original, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("scope_count", [2, 3])
def test_each_topk_uses_one_scope_and_releases_full_width_inputs(monkeypatch, scope_count):
    actual_topk = torch.topk
    previous_values = []
    seen_shapes = []

    def topk(values, *args, **kwargs):
        assert all(reference() is None for reference in previous_values)
        seen_shapes.append(tuple(values.shape))
        previous_values.append(weakref.ref(values))
        return actual_topk(values, *args, **kwargs)

    monkeypatch.setattr(torch, "topk", topk)
    samples = torch.arange(5 * scope_count * 1003, dtype=torch.float32).reshape(
        5, scope_count, 1003,
    )
    result = ema_tail.drawdown_ema_tail_from_samples(samples)
    assert seen_shapes == [(5, 1003)] * scope_count
    assert all(reference() is None for reference in previous_values)
    assert result.shape == (5, scope_count)


@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
@pytest.mark.parametrize("scope_count", [2, 3])
def test_metal_completion_precedes_the_next_scope_workspace(monkeypatch, device, scope_count):
    events = []

    class Samples:
        dtype = torch.float32
        ndim = 3
        shape = (2, scope_count, 1003)

        def __init__(self):
            self.device = SimpleNamespace(type=device)

        def __getitem__(self, key):
            return key[1]

    def reduce_scope(scope):
        events.append(scope)
        return torch.tensor([float(scope), float(scope + 1)])

    monkeypatch.setattr(ema_tail, "_drawdown_ema_tail_scope", reduce_scope)
    monkeypatch.setattr(torch.mps, "synchronize", lambda: events.append("complete"))
    actual = ema_tail.drawdown_ema_tail_from_samples(Samples())
    expected_events = []
    for scope in range(scope_count):
        if scope and device == "mps":
            expected_events.append("complete")
        expected_events.append(scope)
    assert events == expected_events
    assert actual.tolist() == [[float(scope) for scope in range(scope_count)],
                               [float(scope + 1) for scope in range(scope_count)]]


@pytest.mark.parametrize("entry_intervals", [False, True])
@pytest.mark.parametrize("scope_count", [2, 3])
def test_combined_metric_admission_retains_a_128_candidate_batch(entry_intervals, scope_count):
    from optimization.gpu import mps_kernel

    runner = object.__new__(mps_kernel.MpsTrailingMartingaleMulticoinRunner)
    runner.n, runner.n_days, runner.n_coins = 40_000, 28, 1
    runner.replay_sides = scope_count - 1
    runner.side = "long"
    runner.native_factual_hsl = True
    runner.hsl_ema_tail_enabled = True
    runner.entry_interval_enabled = entry_intervals
    runner.recovery_distribution_enabled = True
    runner.n_recovery_samples = runner.n + 1
    runner.raw_strategy_risk_enabled = runner.raw_strategy_growth_enabled = False
    runner.max_dispatch_candidate_bars = 2_097_152
    runner._replay_state_bytes = 6048
    runner._hsl_history_bytes_per_candidate = lambda: 128
    runner.hsl_scratch_budget_bytes = 512 * 1024**2

    assert runner._history_batch_bytes(128) <= runner.hsl_scratch_budget_bytes
    limit = runner._history_batch_limit()
    assert runner._history_batch_bytes(limit) <= runner.hsl_scratch_budget_bytes
    assert runner._history_batch_bytes(limit + 1) > runner.hsl_scratch_budget_bytes
    # Before scope-local workspaces, these unchanged recovery/interval collectors
    # forced the same requested batch to split under the shared allowance.
    previous_tail = scope_count * (48 * runner.n + 64)
    extra = previous_tail - ema_tail.drawdown_ema_tail_history_bytes(runner.n, scope_count)
    assert runner._history_batch_bytes(128) + 128 * extra > runner.hsl_scratch_budget_bytes
