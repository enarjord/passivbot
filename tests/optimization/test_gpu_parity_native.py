import json
import sys
from types import SimpleNamespace
import pytest

from tools import gpu_parity


def fixture(*options):
    return gpu_parity.fixture_inputs(gpu_parity.build_parser().parse_args([
        "--fixture", "trailing_martingale", "--bars", "128", *options,
    ]))


@pytest.mark.parametrize("selected,engine", [(None, "native"), ("legacy", "legacy"), ("native", "native")])
def test_cli_selects_the_native_optimizer_service_by_default(monkeypatch, capsys, selected, engine):
    monkeypatch.setitem(sys.modules, "optimization.gpu.metrics", SimpleNamespace(
        validate_gpu_metric_names=lambda names: names,
    ))
    def compared(*_args, **kwargs):
        assert kwargs["gpu_engine"] == engine
        return {"passed": True, "gpu_engine": engine}
    monkeypatch.setattr(gpu_parity, "run_comparison", compared)
    flags = ["--gpu-engine", selected] if selected is not None else []
    assert gpu_parity.main(["--fixture", "trailing_martingale", "--bars", "128", *flags]) == 0
    assert json.loads(capsys.readouterr().out)["gpu_engine"] == engine


@pytest.mark.parametrize("failure", ["allocation", "consumer", "cleanup", "consumer_and_cleanup"])
def test_native_dataset_attempts_all_cleanup_and_preserves_primary_error(monkeypatch, failure):
    import shared_arrays
    real = shared_arrays.SharedArrayManager
    owned, cleaned = [], []
    class Manager(real):
        def create_from(self, value):
            if failure == "allocation" and len(owned) == 2:
                raise RuntimeError("allocation failed")
            result = super().create_from(value)
            owned.append(result[0])
            return result
        def cleanup(self, specs=None):
            cleaned.extend(specs)
            super().cleanup(specs)
            if "cleanup" in failure and len(cleaned) == 1:
                raise RuntimeError("cleanup failed")
    monkeypatch.setattr(shared_arrays, "SharedArrayManager", Manager)
    message = "allocation" if failure == "allocation" else "consumer" if "consumer" in failure else "cleanup"
    with pytest.raises(RuntimeError, match=message):
        with gpu_parity._native_dataset(fixture(), "binance", gpu_parity.DEFAULT_METRICS):
            if "consumer" in failure:
                raise RuntimeError("consumer failed")
    assert set(owned) == set(cleaned)
    for spec in owned:
        with pytest.raises(FileNotFoundError):
            shared_arrays.attach_shared_array(spec)


def test_native_service_failure_is_not_replaced_by_legacy_comparison(monkeypatch, capsys):
    import optimization.gpu.native as native
    import rust_utils
    monkeypatch.setattr(rust_utils, "verify_loaded_runtime_extension", lambda: {
        "runtime_compiled_source_stamp":"test", "expected_source_fingerprint":"test",
    })
    monkeypatch.setitem(sys.modules, "backtest", SimpleNamespace(
        build_backtest_payload=lambda *_args, **_kwargs: None,
        execute_backtest=lambda *_args: ([], [], {name: 0.1 for name in gpu_parity.DEFAULT_METRICS}),
    ))
    def unavailable(**_kwargs):
        raise RuntimeError("native CUDA unavailable")
    monkeypatch.setattr(native, "CudaBacktestService", unavailable)
    monkeypatch.setitem(sys.modules, "optimization.gpu.metrics", SimpleNamespace(
        validate_gpu_metric_names=lambda names: names,
    ))
    capsys.readouterr()  # Ignore import-time native-extension status text.
    assert gpu_parity.main(["--fixture", "trailing_martingale", "--bars", "128",
                            "--gpu-engine", "native"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["gpu_engine"] == "native"
    assert report["status"] == "execution_failed"
    assert report["error"]["message"] == "native CUDA unavailable"


@pytest.mark.parametrize("strategy", ["ema_anchor", "trailing_martingale"])
@pytest.mark.parametrize("coins", [1, 2])
@pytest.mark.parametrize("sides", ["long", "short", "both"])
def test_native_parity_uses_actual_service_shared_replay_and_one_cpu_simulation(monkeypatch, strategy, coins, sides):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    import backtest
    import optimization.gpu.native as native
    import optimization.gpu.service as replays
    from config.metrics import resolve_metric_value
    from optimization.gpu.service import MpsMulticoinProxy
    from rust_utils import verify_loaded_runtime_extension
    assert not verify_loaded_runtime_extension().get("skipped")
    inputs = gpu_parity.fixture_inputs(gpu_parity.build_parser().parse_args([
        "--fixture", strategy, "--coins", str(coins), "--sides", sides,
    ]))
    cpu_rows, registered = [], []
    cpu = backtest.execute_backtest
    def observed_cpu(*args):
        output = cpu(*args)
        cpu_rows.append(output[2])
        return output
    monkeypatch.setattr(backtest, "execute_backtest", observed_cpu)
    class Service(native.CudaBacktestService):
        def register_dataset(self, identity, dataset):
            registered.append(dataset)
            return super().register_dataset(identity, dataset)
    monkeypatch.setattr(native, "CudaBacktestService", Service)
    def wrong_engine(**_kwargs):
        pytest.fail("native parity must not select the legacy single-coin replay")
    monkeypatch.setattr(replays, "MpsSingleCoinProxy", wrong_engine)
    report = gpu_parity.run_comparison(
        inputs, "binance", gpu_parity.DEFAULT_METRICS, gpu_parity.DEFAULT_TOLERANCES,
        diagnostics=True,
    )
    assert len(cpu_rows) == len(registered) == 1
    assert len(registered[0].coin_indices) == coins
    assert report["gpu_engine"] == "native" and report["gpu_replay"] == "shared_account"
    assert report["timings_seconds"]["gpu_prepare"] is None
    assert report["timings_seconds"]["gpu_cold"] > 0
    assert set(report["diagnostics"]["gpu"]) == {"native_result"}
    config, candles, markets, btc, timestamps = inputs
    expected = MpsMulticoinProxy(config=config, hlcvs=candles, mss=markets, btc=btc,
                                timestamps=timestamps, exchange="binance", batch_size=1,
                                needed_metrics=gpu_parity.DEFAULT_METRICS,
                                factual_hsl=True).evaluate_results([{}])[0]
    assert report["diagnostics"]["gpu"]["native_result"]["liquidated"] == expected.liquidated
    for name in gpu_parity.DEFAULT_METRICS:
        assert report["metrics"][name]["cpu"] == resolve_metric_value(cpu_rows[0], name)
        assert report["metrics"][name]["gpu"] == expected.metrics[name]
        assert report["metrics"][name]["status"] in {"match", "mismatch"}


@pytest.mark.parametrize("coins", [1, 2])
def test_real_native_parity_cli_returns_comparison_and_engine_identity(capsys, coins):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    code = gpu_parity.main(["--fixture", "trailing_martingale", "--coins", str(coins),
                            "--diagnostics", "--compact"])
    report = json.loads(capsys.readouterr().out)
    assert code in (0, 1), report
    assert report["gpu_engine"] == "native" and report["gpu_replay"] == "shared_account"
    assert report["rust_source_fingerprint"]
    assert report["feasibility"]["passed"]
    assert all(report["metrics"][name]["cpu"] is not None for name in gpu_parity.DEFAULT_METRICS)
