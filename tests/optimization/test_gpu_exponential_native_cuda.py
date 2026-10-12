"""Bounded public-fixture native CUDA EW/legacy capture ablation."""

import math

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA required")


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_native_ew_only_omits_legacy_history_and_nested_analysis(monkeypatch, strategy, service_class=None):
    import backtest
    from config.ew import EW_METRICS
    from optimization.gpu import metrics, mps_kernel, weighted_equity
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.metric_registry import WEIGHTED_EQUITY_METRICS
    from optimization.gpu.native import CudaBacktestService
    from rust_utils import verify_loaded_runtime_extension
    from tools.gpu_parity import build_parser, fixture_inputs, _native_dataset

    verify_loaded_runtime_extension()
    service_class = service_class or CudaBacktestService
    inputs = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--sides", "both", "--coins", "2",
        "--bars", "5760", "--seed", "209",
    ]))
    inputs[0]["backtest"]["ew_half_life_days"] = 180.5

    def no_cpu(*args, **kwargs):
        pytest.fail("native EW acceptance must never execute a CPU simulation")
    for owner, name in ((backtest, "execute_backtest"), (backtest, "run_backtest"),
                        (backtest.pbr, "run_backtest_bundle")):
        monkeypatch.setattr(owner, name, no_cpu)

    phase = {"legacy": False}
    legacy_calls = []
    old_reduce = mps_kernel.weighted_equity_from_samples
    def legacy_reduce(*args, **kwargs):
        assert phase["legacy"], "EW-only replay executed legacy ten-window analysis"
        legacy_calls.append(kwargs["curve"])
        return old_reduce(*args, **kwargs)
    monkeypatch.setattr(mps_kernel, "weighted_equity_from_samples", legacy_reduce)
    for name in ("_weighted_subsets", "_weighted_subset_context", "_weighted_strategy_eq_metrics",
                 "_weighted_daily_series_metrics", "_weighted_pnl_metrics", "_weighted_adg"):
        # Authoritative resident legacy outputs also bypass daily approximations;
        # the positive control above proves the actual ten-window path is active.
        monkeypatch.setattr(metrics, name, no_cpu)

    snapshots = {False: [], True: []}
    original_run = mps_kernel._MulticoinReplayRunner.run
    def observed_run(self, parameters, **kwargs):
        legacy = phase["legacy"]
        result = original_run(self, parameters, **kwargs)
        if legacy:
            assert self.weighted_equity_metrics
            assert self.weighted_equity_cols == 2
            assert self._weighted_equity_buffers
            assert weighted_equity.weighted_equity_history_bytes(
                self.n, self.n_days, self.weighted_equity_metrics) > 0
            assert set(result) & WEIGHTED_EQUITY_METRICS
        else:
            assert not self.weighted_equity_metrics
            assert self.weighted_equity_cols == 0
            assert self._weighted_equity_buffers == {}
            assert self._weighted_equity_sample_buffer(len(parameters)) is None
            assert weighted_equity.weighted_equity_history_bytes(self.n, self.n_days, ()) == 0
            # The throughput change separates shared reduction scratch from
            # resident history. This assertion applies when that API is present.
            workspace_bytes = getattr(weighted_equity, "weighted_equity_workspace_bytes", None)
            if workspace_bytes is not None:
                assert workspace_bytes(self.n, self.n_days, ()) == 0
            assert not set(result) & WEIGHTED_EQUITY_METRICS
        snapshots[legacy].append({name: value.detach().clone() if isinstance(value, torch.Tensor)
                                  else value for name, value in result.items()})
        return result
    monkeypatch.setattr(mps_kernel._MulticoinReplayRunner, "run", observed_run)
    completions = {}
    legacy_names = {"adg_strategy_eq_w", "adg_w_usd"}
    for legacy in (False, True):
        phase["legacy"] = legacy
        requested = EW_METRICS | legacy_names if legacy else EW_METRICS
        with _native_dataset(inputs, "binance", tuple(sorted(requested))) as dataset:
            with service_class(batch_size=2, max_pending=2, max_batch_delay=.05) as service:
                service.register_dataset("ew-ablation", dataset)
                futures = [service.submit(BacktestRequest(str(i), "ew-ablation", {})) for i in range(2)]
                # Native variants compile lazily; permit one bounded cold
                # compile before measuring the capture/reduction contract.
                completions[legacy] = [future.result(timeout=540) for future in futures]
        assert all(set(row.metrics) == requested for row in completions[legacy])
        assert all(not row.liquidated for row in completions[legacy])
        assert all(math.isfinite(value) for row in completions[legacy] for value in row.metrics.values())
    assert {"raw_strategy", "account"} <= set(legacy_calls)
    for ew_row, legacy_row in zip(completions[False], completions[True]):
        for metric in EW_METRICS:
            assert ew_row.metrics[metric] == legacy_row.metrics[metric], metric
        assert ew_row.metadata == legacy_row.metadata
    # Admission may select different cohorts, so concatenate candidates before
    # comparison. Capture flags must leave every existing trading tensor exact.
    for name, value in snapshots[False][0].items():
        if isinstance(value, torch.Tensor):
            ew_tensor = torch.cat([row[name] for row in snapshots[False]], dim=0)
            legacy_tensor = torch.cat([row[name] for row in snapshots[True]], dim=0)
            torch.testing.assert_close(ew_tensor, legacy_tensor, rtol=0, atol=0,
                                       equal_nan=True, msg=lambda message: f"{name}: {message}")
        else:
            assert all(row[name] == value for row in snapshots[True])
