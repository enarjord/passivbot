"""Factual weighted histories stay execution-owned and produce compact results."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from optimization.gpu.metric_registry import (
    WEIGHTED_RAW_EQUITY_METRICS, WEIGHTED_ACCOUNT_EQUITY_METRICS,
    WEIGHTED_EQUITY_METRICS, weighted_equity_capture_metrics,
)
from optimization.gpu.weighted_equity import weighted_equity_history_bytes


def test_capture_dependencies_keep_account_exposure_separate_from_raw():
    assert weighted_equity_capture_metrics({"adg_strategy_eq_w"}) == {"adg_strategy_eq_w"}
    assert weighted_equity_capture_metrics({"adg_w_per_exposure_short_usd"}) == {"adg_w_usd"}
    assert weighted_equity_capture_metrics({"mdg_w_per_exposure_long_usd", "mdg_strategy_eq_w"}) == {
        "mdg_w_usd", "mdg_strategy_eq_w"
    }
    assert not weighted_equity_capture_metrics({"adg_usd", "drawdown_worst_usd"})
    assert weighted_equity_history_bytes(1000, 2, ()) == 0


def test_runner_restores_long_relative_clocks_before_absolute_utc(monkeypatch):
    from optimization.gpu import mps_kernel
    runner = object.__new__(mps_kernel.MpsEmaAnchorMulticoinRunner)
    first_step = 321234
    actual_first = 1704067200000 + 23 * 3_600_000 + 59 * 60_000
    runner.run_config = SimpleNamespace(
        interval_ms=60_000, first_ts_ms=actual_first - first_step * 60_000
    )
    runner.n_days = 2
    runner.weighted_equity_cols = 1
    runner.weighted_raw_equity_enabled = True
    runner.weighted_account_equity_enabled = False
    runner.weighted_equity_metrics = frozenset({"adg_strategy_eq_w"})
    observations = []
    def capture(samples, **kwargs):
        observations.append(kwargs)
        return {"adg_strategy_eq_w": torch.tensor([.1, 0.])}
    monkeypatch.setattr(mps_kernel, "weighted_equity_from_samples", capture)
    output = {
        "first_eq_ts": torch.tensor([first_step * 60_000, float("nan")], dtype=torch.float32),
        "last_eq_ts": torch.tensor([(first_step + 2) * 60_000, float("nan")], dtype=torch.float32),
    }
    runner._reduce_weighted_equity(torch.empty((2, 3, 1)), output)
    assert observations[0]["sample_counts"].tolist() == [3, 0]
    assert observations[0]["first_timestamps_ms"][0].item() == actual_first
    assert observations[0]["n_days"] == 2
    output["last_eq_ts"][1] = 60_000
    with pytest.raises(RuntimeError, match="clocks disagree"):
        runner._reduce_weighted_equity(torch.empty((2, 3, 1)), output)


def test_compact_account_results_precede_aliases_and_raw_replacements(monkeypatch):
    from optimization.gpu import metrics
    from optimization.gpu.model import GAP_BINS

    zero = torch.zeros(1)
    out = dict(
        day_end_eq=torch.tensor([[100., 101.]]), day_min_eq=torch.tensor([[99., 100.]]),
        day_max_dd=torch.tensor([[.01, .01]]), day_volume=torch.zeros((1, 2)),
        day_has_fill=torch.ones((1, 2), dtype=torch.bool), max_dd=zero, fill_count=zero,
        held_max_ms=zero, gap_max_ms=zero, gap_hist=torch.zeros((1, GAP_BINS)),
        first_fill_ts=zero, last_fill_ts=zero, first_eq_ts=zero,
        last_eq_ts=torch.tensor([86_400_000.]), last_high_ts=zero,
        recovery_max_ms=zero, liq_step=torch.tensor([-1]),
        candidate_total_wallet_exposure_limit_long=torch.tensor([2.]),
        candidate_total_wallet_exposure_limit_short=torch.tensor([4.]),
    )
    # Distinct values expose wrong source/alias ordering even when traces agree.
    for index, name in enumerate(sorted(WEIGHTED_RAW_EQUITY_METRICS)):
        out[name] = torch.tensor([10. + index])
    for index, name in enumerate(sorted(WEIGHTED_ACCOUNT_EQUITY_METRICS)):
        out[name] = torch.tensor([20. + index])
    needed = WEIGHTED_EQUITY_METRICS | {
        "adg_w_per_exposure_long_usd", "mdg_w_per_exposure_short_usd"
    }
    def forbidden(*args, **kwargs):
        pytest.fail("daily proxy suffix approximation ran despite authoritative compact outputs")
    monkeypatch.setattr(metrics, "_weighted_strategy_eq_metrics", forbidden)
    result = metrics.compute_objectives(out,
        SimpleNamespace(interval_ms=60_000, requested_start_ts_ms=0),
        {"ts0": 0, "n": 2880}, needed=needed)
    for name in WEIGHTED_EQUITY_METRICS:
        torch.testing.assert_close(result[name], out[name].to(torch.float64), rtol=0, atol=0)
    assert float(result["adg_w_per_exposure_long_usd"][0]) == float(out["adg_w_usd"][0]) / 2
    assert float(result["mdg_w_per_exposure_short_usd"][0]) == float(out["mdg_w_usd"][0]) / 4


def _runner_context(strategy, sides, *, requested=(), chunked=False,
                    raw_growth=False, raw_risk=False, btc_risk=False, hsl_tail=False, shock=False):
    from test_gpu_mps import _multicoin_exposure_fixture
    from optimization.gpu.mps_kernel import (
        MpsEmaAnchorMulticoinRunner, MpsEmaAnchorMulticoinFusedRunner,
        MpsTrailingMartingaleMulticoinRunner, MpsTrailingMartingaleMulticoinFusedRunner,
    )
    count = 1513
    closes = np.tile([100., 120.], (count, 1))
    if shock:
        closes[100:700, 0] *= .7
        closes[200:800, 1] *= 1.3
    _, row, run, data = _multicoin_exposure_fixture(
        strategy, "long" if sides == "both" else sides,
        count=count, requested_start_index=31, return_context=True, closes=closes,
    )
    cls = {
        ("ema_anchor", False): MpsEmaAnchorMulticoinRunner,
        ("ema_anchor", True): MpsEmaAnchorMulticoinFusedRunner,
        ("trailing_martingale", False): MpsTrailingMartingaleMulticoinRunner,
        ("trailing_martingale", True): MpsTrailingMartingaleMulticoinFusedRunner,
    }[strategy, sides == "both"]
    if shock:
        from optimization.gpu.model import EMA_ANCHOR_MULTICOIN_PARAM_KEYS, TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
        keys = EMA_ANCHOR_MULTICOIN_PARAM_KEYS if strategy == "ema_anchor" else TRAILING_MARTINGALE_MULTICOIN_PARAM_KEYS
        for key, value in {"hsl_enabled": 1, "hsl_signal_mode": 2,
                           "hsl_red_threshold": .01, "hsl_ema_span_minutes": 2.5,
                           "hsl_cooldown_minutes_after_red": 5, "hsl_slot_count": 2,
                           "hsl_restart_policy": 0}.items():
            row[keys.index(key)] = value
    kwargs = dict(weighted_equity_metrics=requested, recovery_distribution_enabled=True,
                  hsl_ema_tail_enabled=hsl_tail,
                  raw_strategy_growth_enabled=raw_growth, raw_strategy_risk_enabled=raw_risk,
                  btc_risk_enabled=btc_risk,
                  btc_prices=np.full(count, 30000.0) if btc_risk else None)
    if shock:
        kwargs["pnl_lookback_bars"] = 1440
    if sides != "both":
        kwargs["side"] = sides
    if chunked:
        assert strategy == "trailing_martingale" and sides != "both"
        kwargs["max_dispatch_candidate_bars"] = 47 * 2 * 3
    params = np.asarray([row] * 3, dtype=np.float64)
    if sides == "both":
        params = np.concatenate((params, params), axis=1)
    return cls(run, data, **kwargs), params


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("family", ["raw", "account", "both"])
def test_weighted_capture_ablation_preserves_replay_and_recovery(strategy, sides, family):
    requested = {
        "raw": WEIGHTED_RAW_EQUITY_METRICS,
        "account": WEIGHTED_ACCOUNT_EQUITY_METRICS,
        "both": WEIGHTED_EQUITY_METRICS,
    }[family]
    baseline, params = _runner_context(strategy, sides)
    enabled, _ = _runner_context(strategy, sides, requested=requested)
    expected = baseline.run(params)
    actual = enabled.run(params)
    assert set(actual) == set(expected) | requested
    assert not any("weighted_equity_samples" in name for name in actual)
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0, equal_nan=True,
                msg=lambda message: f"{name}: {message}")
        else:
            assert actual[name] == value
    assert enabled._history_bytes_per_candidate() - baseline._history_bytes_per_candidate() == (
        weighted_equity_history_bytes(enabled.n, enabled.n_days, requested)
    )
    columns = 2 if family == "both" else 1
    assert enabled._weighted_equity_buffers[3].shape == (3, enabled.n, columns)
    assert all(actual[name].shape == (3,) for name in requested)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("side", ["long", "short"])
def test_temporal_capture_preserves_clocks_and_partial_histories(side):
    generic, params = _runner_context("trailing_martingale", side, requested=WEIGHTED_EQUITY_METRICS)
    temporal, _ = _runner_context("trailing_martingale", side,
        requested=WEIGHTED_EQUITY_METRICS, chunked=True)
    ends = np.asarray([2, 799, 1512], dtype=np.int32)
    expected = {key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in generic.run(params, end_steps=ends).items()}
    actual = temporal.run(params, end_steps=ends)
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0, equal_nan=True,
                msg=lambda message: f"{key}: {message}")
        else:
            assert actual[key] == value
    # A later full replay must clear padding and preserve previously returned metrics.
    prior = {key: actual[key].clone() for key in WEIGHTED_EQUITY_METRICS}
    temporal.run(params)
    for key in prior:
        torch.testing.assert_close(actual[key], prior[key], rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_history_admission_reduces_before_combining_subbatches():
    runner, params = _runner_context("ema_anchor", "long", requested=WEIGHTED_EQUITY_METRICS)
    expected = {key: value.clone() if isinstance(value, torch.Tensor) else value
                for key, value in runner.run(params).items()}
    runner.hsl_scratch_budget_bytes = runner._history_bytes_per_candidate()
    # Forget the previous allocation to observe bounded one-candidate ownership.
    runner._weighted_equity_buffers.clear()
    actual = runner.run(params)
    assert set(runner._weighted_equity_buffers) == {1}
    assert set(actual) == set(expected)
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=0, equal_nan=True,
                msg=lambda message: f"{key}: {message}")
        else:
            assert actual[key] == value
    assert all(actual[name].shape == (3,) for name in WEIGHTED_EQUITY_METRICS)
    runner.hsl_scratch_budget_bytes -= 1
    with pytest.raises(ValueError, match="scratch budget"):
        runner.run(params[:1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "both"])
@pytest.mark.parametrize("terminal_fill", [False, True], ids=["mark", "panic-fill"])
def test_native_weighted_liquidation_retains_distinct_curves(strategy, sides, terminal_fill):
    from test_gpu_hsl_ordering import _liquidation_inputs
    from tools.gpu_parity import run_comparison
    from optimization.gpu.parity import MetricTolerance

    inputs = list(_liquidation_inputs(strategy, sides, terminal_fill))
    origin = int(inputs[4][0]) // 86_400_000 * 86_400_000 + 23 * 3_600_000 + 56 * 60_000
    inputs[4] = origin + np.arange(len(inputs[4]), dtype=np.int64) * 60_000
    names = WEIGHTED_EQUITY_METRICS | {"adg_w_per_exposure_long_usd"}
    report = run_comparison(tuple(inputs), "bybit", tuple(sorted(names)),
        {name: MetricTolerance(1e-8, 1e-5) for name in names},
        diagnostics=True, gpu_engine="native")
    assert report["diagnostics"]["gpu"]["native_result"]["liquidated"]
    assert report["metrics"]["adg_strategy_eq_w"]["cpu"] != report["metrics"]["adg_w_usd"]["cpu"]
    assert report["passed"], report["metrics"]


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_native_weighted_service_adapts_admission_without_cpu(monkeypatch, strategy):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    import backtest
    from rust_utils import verify_loaded_runtime_extension
    from optimization.gpu import mps_kernel
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from tools.gpu_parity import build_parser, fixture_inputs, _native_dataset

    verify_loaded_runtime_extension()
    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", "2", "--bars", "512",
    ]))
    def forbidden(*args, **kwargs):
        pytest.fail("native weighted service must not execute CPU simulations")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(backtest.pbr, "run_backtest_bundle", forbidden)
    base = mps_kernel._MulticoinReplayRunner
    init, run = base.__init__, base.run
    flags, dispatches = [], []
    def prepare(self, *args, **kwargs):
        init(self, *args, **kwargs)
        flags.append(self.weighted_equity_metrics)
        if self.weighted_equity_metrics:
            self.hsl_scratch_budget_bytes = self._history_bytes_per_candidate() * 2
    def execute(self, parameters, **kwargs):
        output = run(self, parameters, **kwargs)
        if self.weighted_equity_metrics:
            dispatches.append(len(parameters))
            assert len(self._weighted_equity_buffers) == 1
            assert all(output[name].shape == (len(parameters),)
                       and output[name].device.type == "cuda" for name in self.weighted_equity_metrics)
        else:
            assert self._weighted_equity_buffers == {}
            assert not set(output) & WEIGHTED_EQUITY_METRICS
        assert "weighted_equity_samples" not in output
        return output
    monkeypatch.setattr(base, "__init__", prepare)
    monkeypatch.setattr(base, "run", execute)
    for requested in (WEIGHTED_EQUITY_METRICS, {"adg_usd"}):
        with _native_dataset(inputs, "binance", tuple(sorted(requested))) as dataset:
            with CudaBacktestService(batch_size=8, max_pending=8, max_batch_delay=.1) as service:
                service.register_dataset("weighted", dataset)
                futures = [service.submit(BacktestRequest(str(i), "weighted", {})) for i in range(8)]
                completions = [future.result(timeout=120) for future in futures]
        assert all(set(completion.metrics) == requested for completion in completions)
        assert all(completion.metrics == completions[0].metrics for completion in completions)
    assert flags == [WEIGHTED_EQUITY_METRICS, frozenset()]
    assert sum(dispatches) == 8 and max(dispatches) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
@pytest.mark.parametrize("hsl", ["disabled", "coin", "pside", "unified"])
def test_native_weighted_metrics_on_active_shock_replay(strategy, sides, hsl):
    from test_gpu_side_equity_sampling import _side_equity_inputs
    from tools.gpu_parity import run_comparison
    from optimization.gpu.parity import MetricTolerance

    names = WEIGHTED_EQUITY_METRICS | {
        "adg_w_per_exposure_long_usd", "mdg_w_per_exposure_short_usd"
    }
    policies = {name: MetricTolerance(1e-6, 1e-4) for name in names}
    # This two-day public shock is ill-conditioned for suffix ratios. Current
    # Rust producers agree with GPU reductions on the same factual samples;
    # f32 quantization alone changes TM long coin-HSL Sortino by .1314%.
    # Existing replay differences amplify other ratios, with the largest .8437%
    # in EMA shared disabled-HSL Sortino. Bounds stay local to this fixture:
    # they neither change replay behavior nor weaken general comparator policy.
    if strategy == "ema_anchor":
        ratio_relative = (.009 if hsl == "disabled" else .0035) if sides == "both" else (
            .0071 if sides == "short" else .002
        )
        growth_absolute = 3e-6 if sides == "both" else 1e-6
    else:
        ratio_relative = (
            .0001 if hsl == "disabled" else .0016
        ) if sides == "long" else .003
        # Existing short/shared TM trajectories differ in executable quantity
        # quanta. Their largest weighted growth gap is .163 basis points.
        growth_absolute = 1e-6 if sides == "long" else 2e-5
    for name in names:
        policies[name] = MetricTolerance(
            growth_absolute, 1e-4
        ) if name.startswith(("adg", "mdg")) else MetricTolerance(1e-6, ratio_relative)
    report = run_comparison(
        _side_equity_inputs(strategy, sides, hsl), "binance", tuple(sorted(names)), policies,
        diagnostics=True, gpu_engine="native",
    )
    assert not report["diagnostics"]["gpu"]["native_result"]["liquidated"]
    assert report["passed"], report["metrics"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("hsl", ["disabled", "coin", "pside", "unified"])
def test_weighted_shock_capture_preserves_existing_shared_trace(strategy, hsl):
    from test_gpu_side_equity_sampling import _side_equity_inputs
    from optimization.gpu.service import MpsMulticoinProxy

    config, candles, markets, btc, timestamps = _side_equity_inputs(strategy, "both", hsl)
    proxy = MpsMulticoinProxy(config=config, hlcvs=candles, mss=markets, btc=btc,
        timestamps=timestamps, exchange="binance", batch_size=1,
        needed_metrics=WEIGHTED_EQUITY_METRICS)
    cls, kwargs = proxy._runner_specs["fused"]
    baseline = cls(proxy.run, proxy.data, **{**kwargs, "weighted_equity_metrics": ()})
    params = np.concatenate([
        proxy._parameter_matrix([{}], side=side) for side in ("long", "short")
    ], axis=1)
    expected = baseline.run(params)
    actual = proxy.fused_runner.run(params)
    assert set(actual) == set(expected) | WEIGHTED_EQUITY_METRICS
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[name], value, rtol=0, atol=0, equal_nan=True,
                msg=lambda message: f"{name}: {message}")
        else:
            assert actual[name] == value
