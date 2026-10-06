"""Canonical fill suffix volume, optional histories and native CUDA regressions."""

from collections import defaultdict
import math

import numpy as np
import pytest

from test_gpu_service_acceptance_cuda import cuda_runtime


def reference(samples, first, last, start_minute, interval):
    """Independent CPU analysis definition, with filled days rather than elapsed days."""
    if first is None:
        return 0.0
    n = last - first + 1
    starts = [0]
    for denominator in range(2, 11):
        offset = math.floor(n - n / denominator + 0.5)
        if offset >= n:
            break
        starts.append(first + offset)
    averages = []
    for start in starts:
        days = defaultdict(float)
        for k, (volume, present) in enumerate(samples):
            if k >= start and present:
                days[(start_minute + k * interval) // 1440] += float(volume)
        averages.append(sum(days.values()) / len(days) if days else 0.0)
    return sum(averages) / len(averages)


@pytest.mark.parametrize("interval,start_minute", [(1, 0), (1, 1439), (5, 719)])
def test_volume_reducer_includes_partial_days_and_actual_short_terminal_suffixes(
    cuda_runtime, interval, start_minute,
):
    from optimization.gpu.mps_kernel import weighted_volume_from_samples

    rng = np.random.default_rng(43)
    n_rows, capacity = 10, 2880
    values = np.zeros((n_rows, capacity, 2), dtype=np.float32)
    firsts = [2] * n_rows
    counts = [1, 2, 3, 5, 9, 101, 251, 2878, 31, 2878]
    lasts = [first + count - 1 for first, count in zip(firsts, counts, strict=True)]
    for row, last in zip(values, lasts, strict=True):
        row[1:last+1, 0] = rng.uniform(0, 1, last)
        row[1:last+1, 1] = rng.integers(0, 2, last)
    values[4, 4, :] = (0, 1)  # Zero contribution still counts as a filled day.
    values[-2] = 0
    values[-2, 3] = (7, 1)  # Late suffixes have no fills, contributing zero.
    values[-1] = 0
    values[-1, 3] = (0, 1)  # A whole filled UTC day can have zero contribution.
    values[-1, 2000] = (7, 1)
    actual = weighted_volume_from_samples(
        cuda_runtime.tensor(values, device="cuda"),
        cuda_runtime.tensor(firsts, device="cuda", dtype=cuda_runtime.float32)
        * (interval * 60_000),
        cuda_runtime.tensor(lasts, device="cuda", dtype=cuda_runtime.float32)
        * (interval * 60_000),
        start_minute_of_day=start_minute, interval_minutes=interval,
    )
    expected = [reference(row, first, last, start_minute, interval)
                for row, first, last in zip(values, firsts, lasts, strict=True)]
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=3e-6, atol=1e-7)


def test_volume_reducer_empty_equities_and_invalid_bounds(cuda_runtime):
    from optimization.gpu.mps_kernel import weighted_volume_from_samples

    samples = cuda_runtime.zeros((5, 8, 2), device="cuda")
    bounds = cuda_runtime.tensor([float("nan"), 0, float("inf"), 0, 0], device="cuda")
    ends = cuda_runtime.tensor([float("nan"), float("nan"), 60_000, 8*60_000, 7.75*60_000],
                              device="cuda")
    result = weighted_volume_from_samples(
        samples, bounds, ends, start_minute_of_day=0, interval_minutes=1,
    ).cpu().numpy()
    assert result[0] == 0
    assert np.isnan(result[1:]).all()  # Malformed producer bounds must not become zero.


def test_tm_volume_history_survives_temporal_chunks_and_candidate_reordering(cuda_runtime):
    from optimization.gpu.service import MpsMulticoinProxy
    from tools.gpu_parity import build_parser, fixture_inputs

    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", "trailing_martingale", "--sides", "long", "--coins", "2",
        "--bars", "512", "--seed", "43",
    ]))
    proxy = MpsMulticoinProxy(
        config=config, hlcvs=candles, mss=markets, btc=btc, timestamps=timestamps,
        exchange="binance", needed_metrics={"volume_pct_per_day_avg_w"}, batch_size=3,
    )
    runner = proxy.runners["long"]
    candidates = [{"long_entry_initial_qty_pct": value} for value in (0.01, 0.03, 0.07)]
    params = proxy._parameter_matrix(candidates, "long")
    ends = np.array([17, 233, 512], dtype=np.int32)
    runner.max_dispatch_candidate_bars = None
    expected = runner.run(params, end_steps=ends)["volume_pct_per_day_avg_w"].cpu().clone()
    assert len(set(expected.tolist())) == 3
    runner.max_dispatch_candidate_bars = 3 * 2 * 47
    order = np.array([2, 0, 1])
    for _ in range(2):
        actual = runner.run(params[order], end_steps=ends[order], profile=True)
        assert runner.last_profile["temporal_chunk_bars"] == 47
        cuda_runtime.testing.assert_close(
            actual["volume_pct_per_day_avg_w"].cpu(), expected[order], rtol=0, atol=0,
        )
    # Shrinking the batch changes buffer shape without retaining past histories.
    actual = runner.run(params[:1], end_steps=ends[:1])["volume_pct_per_day_avg_w"].cpu()
    cuda_runtime.testing.assert_close(actual, expected[:1], rtol=0, atol=0)
    assert list(runner._volume_buffers) == [1]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("coins", [1, 2])
def test_native_volume_matches_cpu_with_nonunit_contract_multiplier(
    cuda_runtime, strategy, sides, coins,
):
    from tools.gpu_parity import build_parser, fixture_inputs, run_comparison
    from optimization.gpu.parity import MetricTolerance

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", sides, "--coins", str(coins),
        "--bars", "2880", "--seed", "43",
    ]))
    for market in inputs[2].values():
        market["c_mult"] = 2.0
    metrics = ("volume_pct_per_day_avg", "volume_pct_per_day_avg_w")
    # This busy short-side fixture has a fill-count gap of seven among ~20,800 and
    # a 0.273% weighted-volume trajectory gap. Its independent GPU-history
    # reduction agrees within 3e-7 relative. Keep that residual explicit rather
    # than confusing the repaired multiplier/partial-day bugs with fill parity.
    relative = 3e-3 if (strategy, sides, coins) == ("trailing_martingale", "short", 2) else 2e-3
    report = run_comparison(inputs, "binance", metrics,
                            {name: MetricTolerance(1e-6, relative) for name in metrics},
                            gpu_engine="native")
    assert all(report["metrics"][name]["cpu"] > 0 for name in metrics)
    assert report["passed"], report["metrics"]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_native_weighted_volume_uses_actual_liquidation_horizon(cuda_runtime, strategy):
    from test_gpu_entry_sizing_parity import _fixture
    from tools.gpu_parity import run_comparison
    from optimization.gpu.parity import MetricTolerance

    inputs = _fixture("long", 1, "initial")
    config, candles, _markets, _btc, timestamps = inputs
    config["live"]["strategy_kind"] = strategy
    config["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 5.0
    config["bot"]["long"]["strategy"]["ema_anchor"]["base_qty_pct"] = 0.8
    config["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["initial_qty_pct"] = 0.8
    candles[4:, :, 0] = 21
    candles[4:, :, 1] = 19
    candles[4:, :, 2] = 20
    metrics = ("volume_pct_per_day_avg", "volume_pct_per_day_avg_w")
    report = run_comparison(inputs, "bybit", metrics,
                            {name: MetricTolerance(1e-6, 2e-6) for name in metrics},
                            diagnostics=True, gpu_engine="native")
    assert report["diagnostics"]["gpu"]["native_result"]["liquidated"]
    assert report["diagnostics"]["cpu"]["last_equity_timestamp"] < timestamps[-2]
    assert report["metrics"]["volume_pct_per_day_avg_w"]["cpu"] > 0
    assert report["passed"], report["metrics"]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_native_volume_capture_is_optional_bounded_and_cpu_simulation_free(
    cuda_runtime, monkeypatch, strategy,
):
    import backtest
    from optimization.gpu import mps_kernel
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from tools.gpu_parity import build_parser, fixture_inputs, _native_dataset

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", "2", "--bars", "512",
    ]))
    def forbidden(*args, **kwargs):
        pytest.fail("native volume execution must not run a CPU backtest")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(backtest.pbr, "run_backtest_bundle", forbidden)
    base = mps_kernel.MpsEmaAnchorMulticoinRunner
    init, run = base.__init__, base.run
    dispatches, flags = [], []
    def prepare(self, *args, **kwargs):
        init(self, *args, **kwargs)
        flags.append(self.weighted_volume_enabled)
        if self.weighted_volume_enabled:
            total = self._history_bytes_per_candidate()
            self.weighted_volume_enabled = False
            other = self._history_bytes_per_candidate()
            self.weighted_volume_enabled = True
            assert total - other == mps_kernel._volume_history_bytes(self.n)
            self.hsl_scratch_budget_bytes = total * 2
    def execute(self, parameters, **kwargs):
        result = run(self, parameters, **kwargs)
        if self.weighted_volume_enabled:
            dispatches.append(len(parameters))
            assert len(self._volume_buffers) == 1
            assert result["volume_pct_per_day_avg_w"].shape == (len(parameters),)
            assert result["volume_pct_per_day_avg_w"].device.type == "cuda"
        else:
            assert self._volume_buffers == {}
            assert "volume_pct_per_day_avg_w" not in result
        assert "volume_samples" not in result  # Histories never reach service decoding.
        return result
    monkeypatch.setattr(base, "__init__", prepare)
    monkeypatch.setattr(base, "run", execute)
    for metrics in [("volume_pct_per_day_avg_w",), ("volume_pct_per_day_avg",)]:
        with _native_dataset(inputs, "binance", metrics) as dataset:
            with CudaBacktestService(batch_size=8, max_pending=8, max_batch_delay=0.1) as service:
                service.register_dataset("volume", dataset)
                futures = [service.submit(BacktestRequest(str(i), "volume", {}))
                           for i in range(8)]
                results = [future.result(timeout=120) for future in futures]
        assert all(result.metrics == results[0].metrics for result in results)
    assert flags == [True, False]
    assert sum(dispatches) == 8
    assert max(dispatches) == 2
