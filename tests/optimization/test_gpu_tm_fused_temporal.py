"""Shared-account replay continuity and interrupts at actual CUDA boundaries."""
from copy import deepcopy

import numpy as np
import pytest


@pytest.fixture
def cuda_runtime():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA required")
    import passivbot_rust
    from rust_utils import verify_loaded_runtime_extension

    verify_loaded_runtime_extension()
    return torch


def _replay(hsl, hedge_mode, monkeypatch):
    import backtest
    from optimization.gpu.service import MpsMulticoinProxy
    from tools.gpu_parity import build_parser, fixture_inputs

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU backtests forbidden during fused replay checks")

    for obj, name in ((backtest, "execute_backtest"), (backtest, "run_backtest"),
                      (backtest.pbr, "run_backtest_bundle")):
        monkeypatch.setattr(obj, name, forbidden)
    options = ["--fixture", "trailing_martingale", "--sides", "both",
               "--coins", "3", "--bars", "3137", "--seed", "43",
               "--hsl", hsl, "--unstuck", "--price-shock", "0", "1500", ".7",
               "--price-shock", "1", "1800", "1.3"]
    if hsl != "disabled":
        options += ["--hsl-red-threshold", ".002", "--hsl-ema-span-minutes", "2.5",
                    "--hsl-cooldown-minutes", "10000", "--hsl-lookback-days", "1"]
    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args(options))
    config["live"]["hedge_mode"] = hedge_mode
    metrics = {
        "adg_strategy_eq_w", "calmar_ratio_w_usd", "drawdown_worst_strategy_eq",
        "drawdown_worst_mean_1pct_strategy_eq", "strategy_eq_recovery_days_p95",
        "drawdown_worst_mean_1pct_ema_strategy_eq", "drawdown_worst_mean_1pct_ema_strategy_eq_long",
        "drawdown_worst_mean_1pct_strategy_eq_long", "drawdown_worst_strategy_eq_short",
        "hard_stop_time_in_red_pct", "entry_interval_hours_p99", "volume_pct_per_day_avg_w",
        "equity_balance_diff_neg_mean_btc", "drawdown_worst_btc",
    }
    replay = MpsMulticoinProxy(config=deepcopy(config), hlcvs=candles, mss=markets,
        btc=btc, timestamps=timestamps, exchange="binance", batch_size=3,
        needed_metrics=metrics)
    assert replay.fused_runner is not None
    candidates = [{}, {"long_entry_initial_qty_pct": .03}, {}]
    params = np.concatenate([replay._parameter_matrix(candidates, side)
                             for side in ("long", "short")], axis=1)
    return replay, params


def _snapshot(torch, output):
    return {key: value.clone() if isinstance(value, torch.Tensor) else value
            for key, value in output.items()}


def _assert_exact(torch, expected, actual):
    assert expected.keys() == actual.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, actual[key], rtol=0, atol=0, equal_nan=True,
                msg=lambda message: f"{key}: {message}")
        else:
            assert value == actual[key], key


@pytest.mark.parametrize("hsl", ["disabled", "coin", "unified"])
@pytest.mark.parametrize("hedge_mode", [False, True])
def test_cuda_fused_temporal_preserves_all_outputs_and_unequal_ends(
    cuda_runtime, monkeypatch, hsl, hedge_mode
):
    torch = cuda_runtime
    replay, params = _replay(hsl, hedge_mode, monkeypatch)
    runner = replay.fused_runner
    ends = np.array([1, 1439, 3137], dtype=np.int32)
    runner.max_dispatch_candidate_bars = None
    expected = _snapshot(torch, runner.run(params, end_steps=ends))
    for bars in (97, 313, 97):
        runner.max_dispatch_candidate_bars = len(params) * runner.n_coins * 2 * bars
        actual = _snapshot(torch, runner.run(params, end_steps=ends))
        assert runner._last_temporal_dispatch["dispatch_count"] > 1
        assert runner._last_temporal_dispatch["temporal_chunk_bars"] == bars
        assert runner._replay_state_bytes > 0
        _assert_exact(torch, expected, actual)


def test_cuda_fused_interrupt_after_first_chunk_never_returns_partial_metrics(
    cuda_runtime, monkeypatch
):
    torch = cuda_runtime
    replay, params = _replay("coin", True, monkeypatch)
    runner = replay.fused_runner
    runner.max_dispatch_candidate_bars = None
    expected = _snapshot(torch, runner.run(params))
    runner.max_dispatch_candidate_bars = len(params) * runner.n_coins * 2 * 97
    checks = []

    def interrupt():
        checks.append(len(checks))
        if len(checks) == 2:
            raise KeyboardInterrupt("interrupt after one completed fused chunk")

    runner.interrupt_check = interrupt
    with pytest.raises(KeyboardInterrupt, match="after one completed fused chunk"):
        runner.run(params)
    assert len(checks) == 2
    assert runner._replay_state_bytes > 0
    runner.interrupt_check = lambda: None
    # A new replay must reset a discarded partial state and external HSL/PNL buffers.
    _assert_exact(torch, expected, _snapshot(torch, runner.run(params)))


@pytest.mark.parametrize("marker,error,message", [
    (-2, ValueError, "held-position valuation"),
    (-3, RuntimeError, "PnL history overflow"),
    (-4, ValueError, "HSL controller inputs"),
])
def test_cuda_fused_continuation_preserves_fatal_marker_and_resets_next_replay(
    cuda_runtime, monkeypatch, marker, error, message
):
    torch = cuda_runtime
    replay, params = _replay("coin", True, monkeypatch)
    runner = replay.fused_runner
    runner.max_dispatch_candidate_bars = None
    expected = _snapshot(torch, runner.run(params))
    runner.max_dispatch_candidate_bars = len(params) * runner.n_coins * 2 * 97
    checks = 0

    def inject_failure():
        nonlocal checks
        checks += 1
        if checks == 2:
            # Emulate a producer failure in the completed first real GPU chunk.
            runner._buffers[len(params)][1][1, 9] = marker

    runner.interrupt_check = inject_failure
    with pytest.raises(error, match=message):
        runner.run(params)
    assert checks > 2
    assert runner._buffers[len(params)][1][1, 9].item() == marker
    runner.interrupt_check = lambda: None
    _assert_exact(torch, expected, _snapshot(torch, runner.run(params)))


def test_cuda_prepared_service_fused_long_history_interrupts_between_real_chunks(
    cuda_runtime, monkeypatch
):
    import backtest
    from optimization.gpu import mps_kernel
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import CudaBacktestService
    from tools.gpu_parity import build_parser, fixture_inputs, _native_dataset, DEFAULT_METRICS

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU backtests forbidden during native service interruption")

    for obj, name in ((backtest, "execute_backtest"), (backtest, "run_backtest"),
                      (backtest.pbr, "run_backtest_bundle")):
        monkeypatch.setattr(obj, name, forbidden)
    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", "trailing_martingale", "--sides", "both", "--coins", "3",
        "--bars", "8237", "--seed", "43", "--hsl", "coin", "--unstuck",
    ]))
    completed_chunks = []
    original = mps_kernel.record_replay_chunk

    def record(count, bars, total_bars, seconds, **kwargs):
        completed_chunks.append((count, bars, total_bars))
        return original(count, bars, total_bars, seconds, **kwargs)

    monkeypatch.setattr(mps_kernel, "record_replay_chunk", record)

    def interrupt():
        if completed_chunks:
            raise KeyboardInterrupt("native fused interrupt after completed chunk")

    with _native_dataset(inputs, "binance", DEFAULT_METRICS) as dataset:
        with CudaBacktestService(batch_size=1, max_pending=1, interrupt_check=interrupt) as service:
            service.register_dataset("history", dataset)
            future = service.submit(BacktestRequest("interrupted", "history", {}))
            with pytest.raises(KeyboardInterrupt, match="native fused interrupt"):
                future.result(timeout=180)
    # The service preserves its existing exclusive final-bar replay endpoint.
    assert completed_chunks == [(1, 8192, 8235)]
