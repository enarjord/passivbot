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
