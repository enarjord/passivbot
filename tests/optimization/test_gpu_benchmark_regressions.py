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
def test_single_coin_pipeline_profile_preserves_replay(strategy, temporal, tmp_path):
    from tools.optimizer_benchmark.workloads import freeze
    from tools.optimizer_benchmark.worker import instrument_replay, legacy_parameters
    from optimization.gpu.service import MpsSingleCoinProxy

    directory = tmp_path / "workload"
    entry = freeze(directory, profile="standard",
        patterns=[f"{strategy}/both/single/base/oscillating"],
        candidates=2, seeds=[7], bars=256)["entries"][0]
    with np.load(directory / entry["data_file"]) as data:
        candles, btc, timestamps = (data[key] for key in ("hlcvs", "btc", "timestamps"))
    inputs = (candles, entry["markets"], btc, timestamps, entry["exchange"])
    parameters = [legacy_parameters(config, inputs, entry["metrics"]) for config in entry["configs"]]
    replay = MpsSingleCoinProxy(config=entry["configs"][0], hlcvs=candles,
        mss=entry["markets"], btc=btc, timestamps=timestamps, exchange=entry["exchange"],
        batch_size=2, needed_metrics=set(entry["metrics"]),
        max_dispatch_candidate_bars=50_000_000)
    expected = replay.evaluate(parameters)
    if temporal:
        runners = [getattr(replay, "runner", None), *getattr(replay, "runners", {}).values()]
        for runner in runners:
            if runner is not None:
                runner.tuning_chunk_bars = 64
    profiles, dispatches = [], []
    instrument_replay(replay, dispatches, profiles=profiles)
    assert replay.evaluate(parameters) == expected
    assert profiles and profiles[0]["pipeline_preparation_measured"]
    assert profiles[0]["timings_seconds"]["pipeline_preparation"] > 0
    assert all(row["runner_profile"]["pipeline_seconds"] > 0 for row in dispatches)
    if temporal:
        assert any(row["runner_profile"]["dispatch_count"] > 1 for row in dispatches)
