"""Native Metal ownership and device identity; no account or network access."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from optimization.gpu.native import NativeBacktestService
from optimization.gpu.residency import MetalSuiteResidency


def test_metal_residency_switches_owners_and_enforces_remaining_working_set(monkeypatch):
    from test_gpu_residency import proxy
    allocations = []
    memory = {"used": 100}
    mps = SimpleNamespace(recommended_max_memory=lambda: 2000,
                          driver_allocated_memory=lambda: memory["used"],
                          synchronize=lambda: None, empty_cache=lambda: None)
    def upload(array, *, device):
        assert device == "mps"
        allocations.append(array.copy())
        return SimpleNamespace(contiguous=lambda: allocations[-1])
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(mps=mps, as_tensor=upload))
    manager = MetalSuiteResidency()
    try:
        first, other = proxy(manager, "one", 12), proxy(manager, "two", -7)
        manager.activate(first)
        manager.activate(other)
        assert not first.runners and isinstance(first.data["bars"], Path)
        manager.activate(first)
        assert not other.runners and len(allocations) == 3
        oversized = proxy(manager, "large", 0, invariant_bytes=856)
        with pytest.raises(MemoryError, match="45%"):
            manager.activate(oversized)
        memory["used"] = 1950
        with pytest.raises(MemoryError, match="working set"):
            manager.activate(other)
    finally:
        directory = manager._directory.name
        manager.close()
    assert not Path(directory).exists()


def test_metal_registration_is_lazy_and_unused_close_does_not_import_gpu(monkeypatch):
    import builtins
    from test_gpu_datasets import inputs
    from optimization.gpu.datasets import PreparedGpuDataset
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name in {"torch", "cupy", "optimization.gpu.runtime", "optimization.gpu.residency"}:
            pytest.fail("unused Metal facade must not import device dependencies")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with NativeBacktestService() as service:
        service.register_dataset("unused", PreparedGpuDataset(**inputs()))


def test_metal_saved_fitness_is_distinct_without_device_imports(monkeypatch):
    from config.schema import get_template_config
    from optimization import evaluation_contract as contract
    from optimization.native_checkpoint import validate_checkpoint, CHECKPOINT_VERSION
    monkeypatch.setattr(contract, "evaluation_implementation_identity", lambda: {"source": "fixture"})
    config = get_template_config()
    config["optimize"]["backend"] = "gpu"
    monkeypatch.setattr(contract.platform, "system", lambda: "Linux")
    cuda = contract.build_evaluation_contract(config)
    monkeypatch.setattr(contract.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(contract.platform, "machine", lambda: "arm64")
    metal = contract.build_evaluation_contract(config)
    assert metal["execution"]["engine"] == "metal_native"
    assert cuda["execution"]["engine"] == "cuda_native"
    assert metal["execution"]["precision"] != cuda["execution"]["precision"]
    with pytest.raises(ValueError, match="evaluation contract changed"):
        validate_checkpoint({"version": CHECKPOINT_VERSION, "backend": "gpu",
                             contract.CONTRACT_KEY: cuda}, config)


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
def test_metal_prepared_service_switches_subset_datasets_and_reuses_results(monkeypatch, strategy):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Metal required")
    import backtest
    from tools.gpu_parity import build_parser, fixture_inputs, _native_dataset
    from optimization.gpu.executor import BacktestRequest
    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", strategy, "--coins", "3", "--sides", "both", "--bars", "512",
    ]))
    requested = ("adg_strategy_eq_w", "backtest_completion_ratio", "fills_per_day")
    subset_config = deepcopy(config)
    subset_config["backtest"]["coins"]["binance"] = ["COIN00", "COIN02"]
    def forbidden(*args, **kwargs):
        pytest.fail("prepared Metal service cannot run CPU backtests")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(backtest.pbr, "run_backtest_bundle", forbidden)
    with _native_dataset((config, candles, markets, btc, timestamps), "binance", requested) as full:
        from optimization.gpu.datasets import PreparedGpuDataset
        subset = PreparedGpuDataset(config=subset_config, markets=markets, exchange="binance",
            hlcvs=full.hlcvs, btc=full.btc, timestamps=full.timestamps,
            candle_coins=full.candle_coins, coin_indices=(0, 2), metrics=requested)
        with NativeBacktestService(batch_size=2, max_pending=4, tuning_mode="off") as service:
            service.register_dataset("full", full)
            service.register_dataset("subset", subset)
            expected = {}
            for sequence, label in enumerate(("full", "subset", "full", "subset")):
                result = service.submit(BacktestRequest(str(sequence), label, {})).result(timeout=180)
                assert result.dataset_id == label and result.liquidated is False
                assert set(result.metrics) == set(requested)
                if label in expected:
                    assert result.metrics == expected[label]
                else:
                    expected[label] = result.metrics
        assert service._residency is None
        assert not service._prepared_cache and not service._subset_cache


@pytest.mark.parametrize("failure", [RuntimeError("producer failed"), KeyboardInterrupt("interrupted")])
def test_metal_worker_failure_releases_owner_and_preserves_original(monkeypatch, failure):
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("Apple Metal required")
    from tools.gpu_parity import build_parser, fixture_inputs, _native_dataset
    from optimization.gpu.executor import BacktestRequest
    from optimization.gpu import service as replay_module
    inputs = fixture_inputs(build_parser().parse_args(["--fixture", "ema_anchor", "--bars", "128"]))
    def fail(**kwargs):
        raise failure
    monkeypatch.setattr(replay_module, "MpsMulticoinProxy", fail)
    with _native_dataset(inputs, "binance", ("fills_per_day",)) as dataset:
        service = NativeBacktestService()
        service.register_dataset("market", dataset)
        with pytest.raises(type(failure)) as caught:
            service.submit(BacktestRequest("candidate", "market", {})).result(timeout=10)
        assert caught.value is failure
        service.close()
        assert service._residency is None
        assert not service._prepared_cache and not service._subset_cache
