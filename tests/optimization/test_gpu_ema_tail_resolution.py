"""Native EMA tails use actual eligible observations, not cutoff-bin means."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from optimization.gpu.ema_tail import drawdown_ema_tail_from_samples


def _reference(samples):
    result = []
    for candidate in samples:
        scopes = []
        for scope in candidate:
            values = np.abs(scope[np.isfinite(scope)]).astype(np.float64)
            values.sort()
            scopes.append(values[-max(len(values) // 100, 1):].mean() if len(values) else 0.0)
        result.append(scopes)
    return np.asarray(result, dtype=np.float32)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("scope_count", [2, 3])
def test_exact_observed_tail_counts_missing_bars_and_partial_cutoff(device, scope_count):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    rng = np.random.default_rng(43)
    samples = np.full((9, scope_count, 1003), np.nan, dtype=np.float32)
    for row, count in enumerate((0, 1, 99, 100, 199, 200, 999, 1000, 1003)):
        # Sparse eligible clocks, signed samples, repeated cutoff-bin values.
        indices = rng.permutation(1003)[:count]
        samples[row, :, indices] = rng.uniform(-.002, .002, (count, scope_count))
    samples[1, 0, 2], samples[1, 1, 2] = np.inf, -np.inf
    actual = drawdown_ema_tail_from_samples(torch.tensor(samples, device=device))
    assert actual.device.type == device
    np.testing.assert_allclose(actual.cpu(), _reference(samples), atol=1e-9, rtol=2e-7)
    # All values fit one legacy histogram bin, whose average loses the actual top.
    clustered = np.full((1, scope_count, 200), .001, dtype=np.float32)
    clustered[0, :, 0], clustered[0, :, 1] = .0013, .0012
    np.testing.assert_allclose(
        drawdown_ema_tail_from_samples(torch.tensor(clustered, device=device)).cpu(),
        np.full((1, scope_count), .00125, dtype=np.float32), atol=1e-10, rtol=0,
    )


def test_ema_tail_handles_empty_and_noncontiguous_input():
    samples = torch.arange(1800, dtype=torch.float32).reshape(2, 3, 300)[:, :, ::2]
    assert not samples.is_contiguous()
    np.testing.assert_allclose(drawdown_ema_tail_from_samples(samples), _reference(samples.numpy()))
    assert drawdown_ema_tail_from_samples(torch.empty((0, 3, 20))).shape == (0, 3)
    assert drawdown_ema_tail_from_samples(torch.empty((2, 3, 0))).tolist() == [[0.] * 3] * 2


def _clone(output):
    return {key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in output.items()}


def _equal(actual, expected, *, skip=()):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        if key in skip:
            continue
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0, equal_nan=True)
        else:
            assert actual[key] == value, key


TAILS = tuple(f"hsl_drawdown_ema_mean_worst_1pct_{s}" for s in ("long", "short", "portfolio"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("signal_mode", [0, 1, 2], ids=["unified", "pside", "coin"])
def test_native_capture_uses_reporting_clock_and_preserves_other_metrics(strategy, sides, signal_mode):
    from test_gpu_weighted_equity_capture import _runner_context
    baseline, params = _runner_context(strategy, sides, shock=True, factual_hsl=True, count=512)
    runner, _ = _runner_context(strategy, sides, shock=True, factual_hsl=True, hsl_tail=True, count=512)
    for offset in range(0, params.shape[1], len(runner.parameter_keys)):
        params[:, offset + runner.parameter_keys.index("hsl_signal_mode")] = signal_mode
    expected = _clone(baseline.run(params))
    actual = _clone(runner.run(params))
    _equal(actual, expected, skip=TAILS)
    assert not baseline._hsl_ema_tail_buffers
    from optimization.gpu.ema_tail import drawdown_ema_tail_history_bytes
    assert runner._history_bytes_per_candidate() - baseline._history_bytes_per_candidate() == (
        drawdown_ema_tail_history_bytes(runner.n, len(runner._hsl_ema_tail_scopes()))
    )
    samples = runner._hsl_ema_tail_buffers[3]
    reference = _reference(samples.cpu().numpy())
    assert torch.isfinite(samples[:, -1]).any(dim=1).all()
    # Unified long/short hedges can have zero account drawdown while both
    # sides trade. Coin reporting still demonstrates a nonzero observed tail.
    if signal_mode == 2:
        assert reference[:, -1].min() > 0
    assert torch.isnan(samples[:, :, :31]).all()
    for index, scope in enumerate(runner._hsl_ema_tail_scopes()):
        name = f"hsl_drawdown_ema_mean_worst_1pct_{scope}"
        np.testing.assert_allclose(actual[name].cpu(), reference[:, index], rtol=2e-7, atol=1e-9)
    if sides != "both":
        inactive = int(sides == "long")
        assert samples.shape[1] == 2  # Inactive side has no capture or reduction.
        assert actual[TAILS[inactive]].eq(0).all()
    # Partial replays and a changed temporal schedule must discard old samples.
    partial_end = runner.n // 2
    ends = np.asarray([2, partial_end, runner.n], dtype=np.int32)
    full = _clone(runner.run(params, end_steps=ends))
    assert torch.isnan(runner._hsl_ema_tail_buffers[3][0]).all()
    assert torch.isnan(runner._hsl_ema_tail_buffers[3][1, :, partial_end:]).all()
    runner.max_dispatch_candidate_bars = 47 * runner.n_coins * runner.replay_sides * 3
    chunked = _clone(runner.run(params, end_steps=ends))
    assert runner._last_temporal_dispatch["dispatch_count"] > 1
    _equal(chunked, full)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_native_tail_is_compact_after_retry_and_budget_split(strategy, monkeypatch):
    from test_gpu_weighted_equity_capture import _runner_context
    from optimization.gpu import mps_kernel
    from optimization.gpu.ema_tail import drawdown_ema_tail_history_bytes
    runner, params = _runner_context(strategy, "both", shock=True, factual_hsl=True, hsl_tail=True, count=512)
    expected = _clone(runner.run(params))
    real_reduce = mps_kernel.drawdown_ema_tail_from_samples
    accepted = []

    def reduced(samples):
        accepted.append(samples.shape[0])
        return real_reduce(samples)

    monkeypatch.setattr(mps_kernel, "drawdown_ema_tail_from_samples", reduced)
    runner.hsl_fact_capacity_learned = 1
    actual = _clone(runner.run(params))
    assert runner.last_hsl_fact_retries > 0
    assert accepted == [3]  # Rejected attempts never reduce or publish fitness.
    _equal(actual, expected)
    cost = runner._history_bytes_per_candidate()
    assert cost >= drawdown_ema_tail_history_bytes(runner.n, len(runner._hsl_ema_tail_scopes()))
    runner.hsl_scratch_budget_bytes = cost
    accepted.clear()
    _equal(runner.run(params), expected)
    assert accepted == [1, 1, 1]
    assert set(runner._hsl_ema_tail_buffers) == {1}
    assert all(value.ndim == 1 for key, value in actual.items() if key in TAILS)
    assert not any("tail_samples" in key for key in actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_disabled_native_hsl_releases_optional_tail_capture(strategy):
    from test_gpu_weighted_equity_capture import _runner_context
    runner, params = _runner_context(strategy, "both", shock=True, factual_hsl=True, hsl_tail=True, count=512)
    runner.run(params)
    assert runner._hsl_ema_tail_buffers
    active_cost = runner._history_bytes_per_candidate()
    keys = runner.parameter_keys
    for offset in (0, len(keys)):
        params[:, offset + keys.index("hsl_enabled")] = 0
    output = runner.run(params)
    assert not runner._hsl_ema_tail_buffers
    assert not runner._hsl_ema_tail_samples_enabled()
    assert runner._history_bytes_per_candidate() < active_cost
    assert all(output[key].eq(0).all() for key in TAILS)
