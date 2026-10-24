"""Actual-device regressions using public synthetic benchmark recipes."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not (torch.backends.mps.is_available() or torch.cuda.is_available()),
    reason="GPU unavailable",
)


def test_native_coin_hsl_forager_preserves_unfilled_incumbents(tmp_path):
    from tools.optimizer_benchmark.workloads import freeze
    from tools.optimizer_benchmark.worker import cpu_candidate, gpu_runs

    directory = tmp_path / "workload"
    entry = freeze(directory,
        patterns=["trailing_martingale/both/forager/hsl_coin/shock"],
        candidates=2, seeds=[7], bars=256)["entries"][0]
    with np.load(directory / entry["data_file"]) as data:
        arrays = tuple(data[k] for k in ("hlcvs", "btc", "timestamps"))
    cpu = [cpu_candidate((config, arrays[0], entry["markets"], arrays[1], arrays[2],
                          entry["exchange"], entry["metrics"]))
           for config in entry["configs"]]
    gpu = gpu_runs(entry, arrays, "8", 1, 0, 0, 50_000_000)["samples"][0]["results"]
    for expected, actual in zip(cpu, gpu, strict=True):
        assert expected["metrics"]["hard_stop_triggers_per_year"] > 0
        assert actual["liquidated"] == expected["liquidated"]
        for metric in entry["metrics"]:
            # Drawdown needs float32 slack; discrete HSL/fill chronology does not.
            tolerance = 2e-6 if metric == "drawdown_worst_strategy_eq" else 1e-9
            assert actual["metrics"][metric] == pytest.approx(
                expected["metrics"][metric], abs=tolerance, rel=0), metric


@pytest.mark.parametrize("strategy,temporal", [
    ("ema_anchor", False), ("trailing_martingale", False), ("trailing_martingale", True),
])
def test_single_coin_native_pipeline_profile_preserves_replay(strategy, temporal, tmp_path, monkeypatch):
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu.native import NativeBacktestService
    from optimization.gpu.parameters import prepare_candidate_parameters
    from optimization.gpu import service as replay_module
    from rust_utils import verify_loaded_runtime_extension
    from tools.gpu_parity import _native_dataset
    from tools.optimizer_benchmark.workloads import freeze
    from tools.optimizer_benchmark.worker import instrument_replay

    verify_loaded_runtime_extension()
    directory = tmp_path / "workload"
    entry = freeze(directory, profile="standard",
        patterns=[f"{strategy}/both/single/base/oscillating"],
        candidates=2, seeds=[7], bars=256)["entries"][0]
    with np.load(directory / entry["data_file"]) as data:
        candles, btc, timestamps = (data[key] for key in ("hlcvs", "btc", "timestamps"))
    inputs = (entry["configs"][0], candles, entry["markets"], btc, timestamps)
    parameters = [prepare_candidate_parameters(config, entry["markets"], entry["exchange"])
                  for config in entry["configs"]]

    def evaluate(dataset, tag):
        with NativeBacktestService(batch_size=2, tuning_mode="off") as service:
            service.register_dataset("one", dataset)
            futures = [service.submit(BacktestRequest(f"{tag}:{i}", "one", parameter))
                       for i, parameter in enumerate(parameters)]
            return [(result.metrics, result.liquidated)
                    for result in (future.result(timeout=120) for future in futures)]

    profiles, dispatches, temporal_dispatch_counts = [], [], []
    original_init = replay_module.SharedAccountReplay.__init__

    def observed_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        def configure(runner):
            if temporal:
                runner.max_dispatch_candidate_bars = 64
            original_run = runner.run
            def run(*args, **kwargs):
                result = original_run(*args, **kwargs)
                temporal_dispatch_counts.append(
                    (runner._last_temporal_dispatch or {}).get("dispatch_count", 1))
                return result
            runner.run = run
            return runner
        for runner in [self.fused_runner, *self.runners.values()]:
            if runner is not None:
                configure(runner)
        original_construct = self._construct_runner
        def construct(*args, **kwargs):
            return configure(original_construct(*args, **kwargs))
        self._construct_runner = construct
        instrument_replay(self, dispatches, profiles=profiles)

    with _native_dataset(inputs, entry["exchange"], entry["metrics"]) as dataset:
        expected = evaluate(dataset, "baseline")
        monkeypatch.setattr(replay_module.SharedAccountReplay, "__init__", observed_init)
        assert evaluate(dataset, "profiled") == expected
    assert profiles and all(row["pipeline_preparation_measured"] for row in profiles)
    assert all(row["timings_seconds"]["pipeline_preparation"] >= 0 for row in profiles)
    assert dispatches and all("pipeline_seconds" in row["runner_profile"] for row in dispatches)
    if temporal:
        assert any(count > 1 for count in temporal_dispatch_counts)
