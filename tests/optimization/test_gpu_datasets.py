import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from optimization.gpu.datasets import PreparedGpuDataset
from shared_arrays import SharedArrayManager, SharedArraySpec


def inputs():
    return dict(
        config={"backtest": {"coins": {"binance": ["A", "B"]}}},
        markets={"A": {"price_step": 0.01}, "B": {"price_step": 0.1}},
        hlcvs=SharedArraySpec("candles", (10, 3, 4), "<f8"),
        candle_coins=["B", "C", "A"],
        btc=SharedArraySpec("btc", (10,), "<f8"),
        timestamps=SharedArraySpec("timestamps", (10,), "<i8"),
        exchange="binance", metrics=["fills_per_day"],
        time_range=[2, 8], coin_indices=[2, 0],
    )


def test_prepared_dataset_snapshots_metadata_and_selection_without_attaching(monkeypatch):
    import optimization.gpu.datasets as module
    def forbidden(*_args):
        pytest.fail("metadata preparation must not attach or initialize a device")
    monkeypatch.setattr(module, "attach_shared_array", forbidden)
    values = inputs()
    dataset = PreparedGpuDataset(**values)
    values["config"]["backtest"]["coins"]["binance"].reverse()
    values["markets"]["A"]["price_step"] = 99
    values["time_range"][0] = 0
    values["coin_indices"].reverse()
    values["candle_coins"].reverse()
    values["metrics"].append("adg_strategy_eq")
    assert json.loads(dataset.config_json)["backtest"]["coins"]["binance"] == ["A", "B"]
    assert json.loads(dataset.markets_json)["A"]["price_step"] == 0.01
    assert dataset.time_range == (2, 8)
    assert dataset.coin_indices == (2, 0)
    assert dataset.candle_coins == ("B", "C", "A")
    assert dataset.metrics == ("fills_per_day",)
    with pytest.raises(FrozenInstanceError):
        dataset.exchange = "bybit"


@pytest.mark.parametrize("change", [
    {"time_range": (0, 11)}, {"time_range": (2, 4)}, {"time_range": (False, 8)},
    {"coin_indices": (0, 0)}, {"coin_indices": (0, 3)}, {"coin_indices": (True, 0)},
    {"coin_indices": (0, 2)}, {"candle_coins": ["B", "C", "D"]},
    {"candle_coins": ["B", "C"]}, {"candle_coins": ["B", "C", "B"]}, {"candle_coins": "BCA"},
    {"metrics": []}, {"metrics": "fills_per_day"}, {"exchange": ""},
    {"timestamps": SharedArraySpec("ts", (9,), "<i8")},
    {"hlcvs": SharedArraySpec("candles", (10, 3, 3), "<f8")},
    {"btc": SharedArraySpec("btc", (10,), "O")},
    {"markets": {"A": {}}},
    {"config": {"backtest": {"coins": {"binance": ["B", "A"]}}}},
])
def test_prepared_dataset_rejects_invalid_reference_layout_or_metadata(change):
    with pytest.raises((ValueError, TypeError)):
        PreparedGpuDataset(**(inputs() | change))


def test_attachment_reads_selected_time_without_copying_and_releases_partial_setup(monkeypatch):
    import optimization.gpu.datasets as module
    manager = SharedArrayManager()
    try:
        candles = np.arange(10 * 3 * 4, dtype=np.float64).reshape(10, 3, 4)
        specs = [manager.create_from(array)[0] for array in
                 (candles, np.arange(10, dtype=np.float64), np.arange(10, dtype=np.int64))]
        dataset = PreparedGpuDataset(**(inputs() | dict(zip(("hlcvs", "btc", "timestamps"), specs))))
        original = module.attach_shared_array
        attached, closed = [], []
        def tracked(spec):
            attachment = original(spec)
            attached.append(spec.name)
            close = attachment.close
            def cleanup():
                closed.append(spec.name)
                close()
            attachment.close = cleanup
            return attachment
        monkeypatch.setattr(module, "attach_shared_array", tracked)
        with dataset.attach() as arrays:
            np.testing.assert_array_equal(arrays[0], candles[2:8])
            assert all(not array.flags.writeable and not array.flags.owndata for array in arrays)
            with pytest.raises(ValueError, match="read-only"):
                arrays[0][0, 0, 0] = 1
        assert closed == list(reversed(attached))
        attached.clear()
        closed.clear()
        def failing(spec):
            if spec == dataset.btc:
                raise FileNotFoundError("prepared segment disappeared")
            return tracked(spec)
        monkeypatch.setattr(module, "attach_shared_array", failing)
        with pytest.raises(FileNotFoundError):
            with dataset.attach():
                pytest.fail("partial attachment must not produce input")
        assert closed == attached == [dataset.hlcvs.name]
    finally:
        manager.cleanup()


def test_cuda_facade_registration_and_unused_close_do_not_import_device(monkeypatch):
    import builtins
    from optimization.gpu.native import CudaBacktestService
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name in {"torch", "optimization.gpu.runtime", "optimization.gpu.service",
                    "optimization.gpu.residency"}:
            pytest.fail("unused prepared service must not initialize device dependencies")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with CudaBacktestService() as service:
        service.register_dataset("unused", PreparedGpuDataset(**inputs()))


def test_cuda_facade_rejects_other_device_before_attachment(monkeypatch):
    from optimization.gpu.native import CudaBacktestService
    from optimization.gpu.executor import BacktestRequest
    import optimization.gpu.datasets as datasets
    import optimization.gpu.runtime as runtime
    monkeypatch.setattr(runtime, "gpu_device", lambda: "mps")
    def forbidden(*_args):
        pytest.fail("unsupported device must not attach a prepared dataset")
    monkeypatch.setattr(datasets, "attach_shared_array", forbidden)
    with CudaBacktestService() as service:
        service.register_dataset("market", PreparedGpuDataset(**inputs()))
        with pytest.raises(RuntimeError, match="NVIDIA CUDA"):
            service.submit(BacktestRequest("candidate", "market", {})).result(timeout=3)
    assert service._residency is None
    assert not service._prepared_cache and not service._subset_cache


@pytest.mark.parametrize("phase", ["setup", "body", "cleanup"])
def test_attachment_cleanup_preserves_primary_failure_and_attempts_every_close(monkeypatch, caplog, phase):
    from types import SimpleNamespace
    import optimization.gpu.datasets as module
    dataset = PreparedGpuDataset(**inputs())
    opened, closed = [], []
    def attach(spec):
        if phase == "setup" and spec == dataset.btc:
            raise FileNotFoundError("primary setup")
        opened.append(spec.name)
        def close():
            closed.append(spec.name)
            raise RuntimeError("cleanup " + spec.name)
        return SimpleNamespace(array=np.zeros(spec.shape, dtype=spec.dtype), close=close)
    monkeypatch.setattr(module, "attach_shared_array", attach)
    expected = FileNotFoundError if phase == "setup" else ValueError if phase == "body" else RuntimeError
    message = "primary setup" if phase == "setup" else "primary body" if phase == "body" else "cleanup timestamps"
    with pytest.raises(expected, match=message):
        with dataset.attach():
            if phase == "body":
                raise ValueError("primary body")
    assert closed == list(reversed(opened))
    assert "cleanup candles" in caplog.text
