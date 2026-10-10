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
    original = mps_kernel.strategy_eq_recovery_distribution_from_samples

    def capture(samples, **kwargs):
        row = samples[0]
        observed.append(row[row.isfinite()].cpu().numpy().copy())
        return original(samples, **kwargs)

    def forbidden(*args, **kwargs):
        pytest.fail("native execution must not invoke CPU simulations")

    monkeypatch.setattr(mps_kernel, "strategy_eq_recovery_distribution_from_samples", capture)
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
        assert "strategy_eq_recovery_samples" not in result
        assert result["strategy_eq_recovery_distribution"].shape == (len(parameters), 7)
        finite = self._recovery_buffers[len(parameters)].isfinite().sum(dim=1)
        count = ((result["last_eq_ts"] - result["first_eq_ts"]) / 60_000 + 1)
        np.testing.assert_array_equal(finite.cpu().numpy(), count.cpu().numpy())
        assert len(parameters) * self._history_bytes_per_candidate() <= self.hsl_scratch_budget_bytes
        return result
    monkeypatch.setattr(base, "__init__", prepare)
    monkeypatch.setattr(base, "run", run)
    with _native_dataset(inputs, "binance", METRICS) as dataset:
        with CudaBacktestService(batch_size=8, max_pending=8, max_batch_delay=0.1) as service:
            service.register_dataset("recovery", dataset)
            futures = [service.submit(BacktestRequest(str(i), "recovery", {})) for i in range(8)]
            results = [future.result(timeout=120) for future in futures]
    assert sum(dispatches) == 8
    # Inactive native HSL releases its initial factual allowance; later work may
    # grow beyond the first width while respecting the effective history envelope.
    assert min(dispatches) >= 1
    assert all(result.metrics == results[0].metrics for result in results)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_native_recovery_policy_switch_combines_only_compact_results(cuda_runtime, monkeypatch, strategy):
    import weakref
    import backtest
    from optimization.gpu import mps_kernel
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from optimization.gpu.service import MpsMulticoinProxy
    from tools.gpu_parity import build_parser, fixture_inputs, _native_dataset

    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", "2", "--bars", "512",
        "--hsl", "unified", "--hsl-red-threshold", ".99",
    ]))
    if strategy == "trailing_martingale":
        # Keep factual activity below the retained estimate: this control isolates
        # result storage, while overflow/capacity learning have separate coverage.
        for side in ("long", "short"):
            params = inputs[0]["bot"][side]["strategy"][strategy]
            for section in ("entry", "close"):
                params[section].update(
                    threshold_base_pct=.9, threshold_we_weight=0.,
                    threshold_volatility_1m_weight=0., threshold_volatility_1h_weight=0.,
                )
    base = mps_kernel._MulticoinReplayRunner
    initial, run, concatenate = base.__init__, base.run, cuda_runtime.cat
    original_dispatch = base._dispatch_replay
    current, dispatches, raw_joins, compact_joins = {}, [], [], []
    physical_dispatches = []
    raw_control = False

    def prepare(self, *args, **kwargs):
        if raw_control:
            kwargs["compact_recovery_output"] = False
        initial(self, *args, **kwargs)
        self.hsl_scratch_budget_bytes = 200_000
        current["owner"] = weakref.ref(self)

    def execute(self, parameters, **kwargs):
        result = run(self, parameters, **kwargs)
        if self.hsl_fact_capacity == 0:
            # Seed a legal retained estimate to isolate policy-changing admission.
            # Capacity learning/residency are covered independently.
            self.hsl_fact_capacity_learned = 512
        dispatches.append(len(parameters))
        return result

    def capture(values, *args, **kwargs):
        result = concatenate(values, *args, **kwargs)
        if len(values) > 1 and all(isinstance(v, cuda_runtime.Tensor)
                                  and v.device.type == "cuda" and v.ndim == 2 for v in values):
            if result.shape[1] == 513:
                owner = current["owner"]()
                factual = sum(v.numel() * v.element_size()
                              for pair in owner._hsl_scratch_buffers.values() for v in pair)
                recovery = sum(v.numel() * v.element_size() for v in owner._recovery_buffers.values())
                clones = sum(v.numel() * v.element_size() for v in values)
                joined = result.numel() * result.element_size()
                raw_joins.append(factual + recovery + clones + joined)
            elif result.shape == (24, 7):
                compact_joins.append(result.numel() * result.element_size())
        return result

    def dispatch(self, library, kernel_args, end_steps, *, batch_size):
        assert batch_size * self._history_bytes_per_candidate() <= self.hsl_scratch_budget_bytes
        physical_dispatches.append(batch_size)
        return original_dispatch(self, library, kernel_args, end_steps, batch_size=batch_size)

    def forbidden(*args, **kwargs):
        pytest.fail("native recovery execution must not invoke CPU simulation")
    monkeypatch.setattr(base, "__init__", prepare)
    monkeypatch.setattr(base, "run", execute)
    monkeypatch.setattr(base, "_dispatch_replay", dispatch)
    monkeypatch.setattr(cuda_runtime, "cat", capture)
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(backtest.pbr, "run_backtest_bundle", forbidden)
    metrics = (*METRICS, "hard_stop_time_in_red_pct")
    with _native_dataset(inputs, "binance", metrics) as dataset:
        with CudaBacktestService(batch_size=24, max_pending=32, max_batch_delay=.1, tuning_mode="off") as service:
            service.register_dataset("policy", dataset)
            service.submit(BacktestRequest("off", "policy", {"long_hsl_enabled": 0., "short_hsl_enabled": 0.})).result(timeout=600)
            futures = [service.submit(BacktestRequest(str(i), "policy", {})) for i in range(24)]
            results = [future.result(timeout=600) for future in futures]
        assert sum(dispatches) >= 25
        assert not raw_joins, f"Raw recovery histories escaped physical admission: {raw_joins} > 200000"
        assert all(result.metrics == results[0].metrics for result in results)
        # Service admission may split the pending cohort before calling a runner.
        # Exercise the runner's compact logical assembly directly, rather than
        # assuming that the service must hide 24 serial replays behind one batch.
        config, candles, markets, btc, timestamps = inputs
        direct = MpsMulticoinProxy(
            config=config, hlcvs=candles, mss=markets, btc=btc, timestamps=timestamps,
            exchange="binance", batch_size=24, needed_metrics=metrics, factual_hsl=True,
            max_dispatch_candidate_bars=500_000_000,
        )
        owner = direct.fused_runner
        owner.hsl_fact_capacity = owner.hsl_fact_capacity_learned = 512
        parameters = np.concatenate(
            [direct._parameter_matrix([{}] * 24, side) for side in ("long", "short")], axis=1,
        )
        single = {key: value.clone() if isinstance(value, cuda_runtime.Tensor) else value
                  for key, value in owner.run(parameters[:1]).items()}
        joined_before = len(compact_joins)
        grouped = owner.run(parameters)
        assert compact_joins[joined_before:] == [24 * 7 * 4]
        assert physical_dispatches and max(physical_dispatches) < 24
        assert not raw_joins
        for key, value in grouped.items():
            expected = single[key]
            if isinstance(value, cuda_runtime.Tensor):
                cuda_runtime.testing.assert_close(value, expected.expand_as(value), rtol=0, atol=0,
                                                 equal_nan=True)
            else:
                assert value == expected
        del owner, direct, parameters, single, grouped
        # A separate raw diagnostic mode remains available and gives the same metrics.
        raw_control = True
        with CudaBacktestService(batch_size=1, tuning_mode="off") as service:
            service.register_dataset("raw", dataset)
            control = service.submit(BacktestRequest("raw", "raw", {})).result(timeout=600)
        assert control.metrics == results[0].metrics
