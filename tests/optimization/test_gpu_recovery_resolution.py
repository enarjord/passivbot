"""Recovery sampling, dispatch ownership and native scratch-budget regressions."""

import numpy as np
import pytest

from test_gpu_service_acceptance_cuda import cuda_runtime


METRICS = tuple(f"strategy_eq_recovery_days_{suffix}" for suffix in (
    "mean", "median", "p95", "p99", "mean_worst_5pct", "mean_worst_1pct",
))


def reference(values):
    """Independent quadratic definition, including strict plateaus and open tails."""
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return np.zeros(7)
    durations = []
    for index in finite:
        later = finite[(finite > index) & (values[finite] > values[index])]
        durations.append(int(later[0] if len(later) else finite[-1]) - index)
    ordered = np.sort(durations)
    return np.array([
        np.mean(ordered), *np.percentile(ordered, [50, 95, 99]),
        np.mean(ordered[-max(1, int(len(ordered) * 0.05)):]),
        np.mean(ordered[-max(1, int(len(ordered) * 0.01)):]), ordered[-1],
    ])


@pytest.mark.parametrize("interval_minutes", [1, 5])
def test_recovery_reducer_preserves_short_plateaus_terminal_and_sparse_samples(
    cuda_runtime, interval_minutes,
):
    from optimization.gpu.mps_kernel import strategy_eq_recovery_distribution_from_samples

    rng = np.random.default_rng(43)
    values = rng.integers(90, 110, size=(6, 257)).astype(np.float32)
    values[0] = 100  # equal equity never resolves until strictly higher
    values[1] = np.arange(257)
    values[2] = -np.arange(257)
    values[3, 70:] = np.nan  # liquidation/truncation padding is not elapsed time
    values[4, 2::3] = np.nan  # elapsed slots still count across missing observations
    values[5] = np.nan
    interval_days = interval_minutes / 1440
    actual = strategy_eq_recovery_distribution_from_samples(
        cuda_runtime.as_tensor(values, device="cuda"), sample_interval_days=interval_days,
    ).cpu().numpy()
    np.testing.assert_allclose(
        actual, np.array([reference(row) for row in values]) * interval_days,
        rtol=2e-6, atol=1e-8,
    )


def test_recovery_reductions_do_not_share_mutable_scratch_between_streams(
    cuda_runtime, monkeypatch,
):
    from optimization.gpu import mps_kernel

    library = mps_kernel._strategy_eq_recovery_distribution_shader_library()
    workspaces = []
    class ObservedLibrary:
        def passivbot_strategy_eq_recovery_distribution(self, *args, **kwargs):
            # Retain dispatched scratch until both streams complete. This catches
            # process-global buffer reuse even when kernel timing hides its race.
            workspaces.append(args[1:4])
            library.passivbot_strategy_eq_recovery_distribution(*args, **kwargs)
    monkeypatch.setattr(mps_kernel, "_strategy_eq_recovery_distribution_shader_library",
                        lambda: ObservedLibrary())
    results = []
    values = [np.arange(257, dtype=np.float32), -np.arange(257, dtype=np.float32)]
    streams = [cuda_runtime.cuda.Stream(), cuda_runtime.cuda.Stream()]
    for stream, row in zip(streams, values, strict=True):
        with cuda_runtime.cuda.stream(stream):
            results.append(mps_kernel.strategy_eq_recovery_distribution_from_samples(
                cuda_runtime.tensor(row[None, :], device="cuda"), sample_interval_days=1/1440,
            ))
    for stream in streams:
        stream.synchronize()
    for left, right in zip(*workspaces, strict=True):
        assert left.data_ptr() != right.data_ptr()
    for actual, row in zip(results, values, strict=True):
        np.testing.assert_allclose(actual.cpu().numpy()[0], reference(row) / 1440,
                                   rtol=2e-6, atol=1e-8)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("coins", [1, 2])
def test_native_recovery_matches_cpu_on_subhour_replay(cuda_runtime, strategy, sides, coins):
    from test_gpu_entry_sizing_parity import _fixture
    from tools.gpu_parity import run_comparison
    from optimization.gpu.parity import MetricTolerance

    config, candles, markets, btc, timestamps = _fixture(
        "long" if sides == "both" else sides, coins, "initial",
    )
    config["live"].update(strategy_kind=strategy, hedge_mode=sides == "both")
    if sides == "both":
        config["bot"]["short"]["risk"].update(
            n_positions=coins, total_wallet_exposure_limit=float(coins),
        )
    report = run_comparison(
        (config, candles, markets, btc, timestamps), "bybit", METRICS,
        {name: MetricTolerance(1e-8, 2e-6) for name in METRICS}, gpu_engine="native",
    )
    # Positive sub-hour observations prevent a vacuous agreement on empty series.
    assert 0 < report["metrics"][METRICS[0]]["cpu"] < 1/24
    assert report["passed"], report["metrics"]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "both"])
@pytest.mark.parametrize("terminal_fill", [False, True], ids=["mark", "panic-fill"])
@pytest.mark.parametrize("weighted_capture", [False, True], ids=["recovery-only", "raw-capture"])
def test_native_recovery_observes_raw_strategy_liquidation_equity(
    cuda_runtime, monkeypatch, strategy, sides, terminal_fill, weighted_capture,
):
    import backtest
    from test_gpu_hsl_ordering import _liquidation_inputs
    from optimization.gpu import mps_kernel
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from tools.gpu_parity import _native_dataset

    inputs = _liquidation_inputs(strategy, sides, terminal_fill)
    config, candles, markets, btc, timestamps = inputs
    payload = backtest.build_backtest_payload(
        candles, markets, config, "bybit", btc, timestamps,
        metrics_only=False, skip_btc_analysis=True,
    )
    _, equities, analysis = backtest.execute_backtest(payload, config)
    assert analysis["liquidated"]
    # Account equity stops at the liquidation floor. The strategy curve retains
    # the factual loss, independently of requested weighted-history capture.
    assert equities[-1, 3] < 0 < equities[-1, 1]
    expected = equities[:, 3]
    observed = []
    base = mps_kernel._MulticoinReplayRunner
    original = base.run

    def capture(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        row = output["strategy_eq_recovery_samples"][0]
        observed.append(row[row.isfinite()].cpu().numpy().copy())
        return output

    def forbidden(*args, **kwargs):
        pytest.fail("native execution must not invoke CPU simulations")

    monkeypatch.setattr(base, "run", capture)
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(backtest.pbr, "run_backtest_bundle", forbidden)
    metrics = (*METRICS, "adg_strategy_eq_w") if weighted_capture else METRICS
    with _native_dataset(inputs, "bybit", metrics) as dataset:
        with CudaBacktestService(batch_size=1, tuning_mode="off") as service:
            service.register_dataset("raw-recovery", dataset)
            result = service.submit(BacktestRequest("raw-recovery", "raw-recovery", {})).result()
    assert result.liquidated
    assert len(observed) == 1
    np.testing.assert_allclose(observed[0], expected, rtol=1e-5, atol=1e-4)
    expected_recovery = reference(expected) / 1440
    np.testing.assert_allclose(
        [result.metrics[name] for name in METRICS], expected_recovery[:6],
        rtol=2e-6, atol=1e-8,
    )


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_native_dispatch_accounts_for_opt_in_recovery_memory(
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
        pytest.fail("native recovery execution must not run a CPU backtest")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(backtest.pbr, "run_backtest_bundle", forbidden)
    base = mps_kernel._MulticoinReplayRunner
    original_init, original_run = base.__init__, base.run
    dispatches = []
    def prepare(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        assert self.recovery_stride == 1
        assert self.n_recovery_samples == self.n + 1
        total_bytes = self._history_bytes_per_candidate()
        # The fixture can also require fill-PnL history. Recovery must add its
        # own samples/reduction storage instead of displacing that consumer.
        self.recovery_distribution_enabled = False
        other_bytes = self._history_bytes_per_candidate()
        self.recovery_distribution_enabled = True
        assert total_bytes - other_bytes == (self.n + 1) * 16 + 56
        self.hsl_scratch_budget_bytes = self._history_bytes_per_candidate() * 2
    def run(self, parameters, **kwargs):
        dispatches.append(len(parameters))
        result = original_run(self, parameters, **kwargs)
        assert len(self._recovery_buffers) == 1
        finite = result["strategy_eq_recovery_samples"].isfinite().sum(dim=1)
        count = ((result["last_eq_ts"] - result["first_eq_ts"]) / 60_000 + 1)
        np.testing.assert_array_equal(finite.cpu().numpy(), count.cpu().numpy())
        return result
    monkeypatch.setattr(base, "__init__", prepare)
    monkeypatch.setattr(base, "run", run)
    with _native_dataset(inputs, "binance", METRICS) as dataset:
        with CudaBacktestService(batch_size=8, max_pending=8, max_batch_delay=0.1) as service:
            service.register_dataset("recovery", dataset)
            futures = [service.submit(BacktestRequest(str(i), "recovery", {})) for i in range(8)]
            results = [future.result(timeout=120) for future in futures]
    assert sum(dispatches) == 8
    assert max(dispatches) == 2
    assert all(result.metrics == results[0].metrics for result in results)
