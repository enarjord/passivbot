from copy import deepcopy
from types import SimpleNamespace
import json

import numpy as np
import pytest

from optimization.native_datasets import NativeDatasetRegistry
from optimize import Evaluator, SuiteEvaluator
from shared_arrays import SharedArrayManager, attach_shared_array
from tools.gpu_parity import build_parser, fixture_inputs


def fixture(manager):
    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", "trailing_martingale", "--sides", "both", "--coins", "3", "--bars", "128",
    ]))
    config["optimize"]["scoring"] = [{"metric": "adg_strategy_eq", "goal": "max"}]
    config["optimize"]["limits"] = []
    for side in ("long", "short"):
        config["bot"][side]["risk"]["n_positions"] = 2
    specs = [manager.create_from(array)[0] for array in (candles, btc)]
    base = Evaluator({"binance": specs[0]}, {"binance": specs[1]}, {"binance": markets},
                     config, timestamps={"binance": timestamps})
    return base, candles, btc, timestamps


def contexts(base, timestamps, *, lazy):
    config = deepcopy(base.config)
    config["backtest"]["coins"]["binance"] = ["COIN00", "COIN02"]
    for side in ("long", "short"):
        config["live"]["approved_coins"][side] = ["COIN00", "COIN02"]
    return [SimpleNamespace(
        label=label, exchanges=["binance"], config=deepcopy(config), msss=base.msss,
        timestamps={"binance": timestamps[32:96] if lazy else timestamps}, overrides={},
        hlcvs_specs=base.hlcvs_specs, btc_usd_specs=base.btc_usd_specs,
        master_hlcvs_specs=base.hlcvs_specs if lazy else None,
        master_btc_specs=base.btc_usd_specs if lazy else None,
        time_slice={"binance": (32, 96)} if lazy else None,
        coin_slice_indices={"binance": [0, 2]} if lazy else None,
        coin_indices={"binance": None},
        candle_coins={"binance": ("COIN00", "COIN01", "COIN02") if lazy else ("COIN00", "COIN02")},
    ) for label in ("base", "stress")]


def test_standalone_registry_borrows_histories_and_owns_only_reusable_timestamps(monkeypatch):
    manager = SharedArrayManager()
    try:
        base, candles, btc, timestamps = fixture(manager)
        allocated = []
        create = SharedArrayManager.create_from
        def metadata_only(self, values):
            assert values.ndim == 1 and values.dtype.kind in "iu"
            spec, view = create(self, values)
            allocated.append(spec)
            return spec, view
        monkeypatch.setattr(SharedArrayManager, "create_from", metadata_only)
        with NativeDatasetRegistry(base, standalone_candle_coins={
            "binance": ("COIN00", "COIN01", "COIN02"),
        }) as registry:
            dataset = registry.bindings[0].dataset
            assert dataset.hlcvs == base.hlcvs_specs["binance"]
            assert dataset.btc == base.btc_usd_specs["binance"]
            assert dataset.metrics == ("adg_strategy_eq",)
            with dataset.attach() as arrays:
                for actual, expected in zip(arrays, (candles, btc, timestamps), strict=True):
                    np.testing.assert_array_equal(actual, expected)
            registered = []
            registry.register(SimpleNamespace(register_dataset=lambda *args: registered.append(args)))
            assert len(registered) == 1 and registered[0][0] == "native:0"
            worker_dataset = registered[0][1]
            assert worker_dataset.hlcvs == dataset.hlcvs
            assert worker_dataset.btc == dataset.btc
            assert worker_dataset.timestamps == dataset.timestamps
            assert worker_dataset.coin_indices == dataset.coin_indices
            assert worker_dataset.metrics == dataset.metrics
            worker_config = json.loads(worker_dataset.config_json)
            assert "optimize" not in worker_config
            original_config = json.loads(dataset.config_json)
            for section in ("bot", "live", "backtest", "coin_overrides"):
                assert worker_config.get(section) == original_config.get(section)
        assert len(allocated) == 1
        with pytest.raises(FileNotFoundError):
            attach_shared_array(allocated[0])
        with pytest.raises(RuntimeError, match="closed"):
            registry.register(object())
        borrowed = attach_shared_array(base.hlcvs_specs["binance"])
        borrowed.close()
    finally:
        manager.cleanup()


@pytest.mark.parametrize("lazy", [False, True])
def test_suite_registry_preserves_source_identity_slices_and_timestamp_reuse(lazy, monkeypatch):
    import backtest
    def forbidden(*_args, **_kwargs):
        pytest.fail("dataset preparation must not run CPU simulations")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    manager = SharedArrayManager()
    try:
        base, candles, btc, timestamps = fixture(manager)
        if not lazy:
            base.hlcvs_specs["binance"] = manager.create_from(candles[:, [0, 2], :])[0]
        prepared = contexts(base, timestamps, lazy=lazy)
        suite = SuiteEvaluator(base, prepared, {"default": "mean"})
        monkeypatch.setattr(base, "evaluate", forbidden)
        monkeypatch.setattr(suite, "evaluate", forbidden)
        with NativeDatasetRegistry(suite) as registry:
            assert len(registry.bindings) == 2
            datasets = [binding.dataset for binding in registry.bindings]
            assert datasets[0].timestamps == datasets[1].timestamps
            assert datasets[0].time_range == ((32, 96) if lazy else (0, 128))
            assert datasets[0].timestamp_range == ((0, 64) if lazy else (0, 128))
            assert datasets[0].coin_indices == ((0, 2) if lazy else (0, 1))
            for dataset in datasets:
                with dataset.attach() as arrays:
                    expected = candles[32:96] if lazy else candles[:, [0, 2], :]
                    np.testing.assert_array_equal(arrays[0], expected)
                    np.testing.assert_array_equal(arrays[2], timestamps[32:96] if lazy else timestamps)
            from optimize import config_to_individual
            vector = config_to_individual(base.config, base.bounds, optimization_shape=base.optimization_shape)
            plan = registry.planner.prepare("candidate", vector)
            assert {slot.scenario for slot in plan.slots} == {"base", "stress"}
    finally:
        manager.cleanup()


@pytest.mark.parametrize("problem", ["columns", "timestamps", "metrics"])
def test_registry_rejects_incomplete_inputs_and_releases_earlier_metadata(problem, monkeypatch):
    manager = SharedArrayManager()
    allocated = []
    try:
        base, _candles, _btc, timestamps = fixture(manager)
        prepared = contexts(base, timestamps, lazy=True)
        if problem == "columns":
            prepared[1].candle_coins = None
        elif problem == "timestamps":
            prepared[1].timestamps["binance"] = None
        suite = SuiteEvaluator(base, prepared, {"default": "mean"})
        create = SharedArrayManager.create_from
        def track(self, array):
            spec, view = create(self, array)
            allocated.append(spec)
            return spec, view
        monkeypatch.setattr(SharedArrayManager, "create_from", track)
        with pytest.raises(ValueError):
            NativeDatasetRegistry(suite, metrics=["fills_per_day"] if problem == "metrics" else None)
        assert allocated
        for spec in allocated:
            with pytest.raises(FileNotFoundError):
                attach_shared_array(spec)
        borrowed = attach_shared_array(base.hlcvs_specs["binance"])
        borrowed.close()
    finally:
        manager.cleanup()


@pytest.mark.parametrize("body_failure", [False, True])
def test_registry_closes_every_owned_window_and_preserves_original_failure(body_failure, monkeypatch, caplog):
    manager = SharedArrayManager()
    try:
        base, _candles, _btc, timestamps = fixture(manager)
        prepared = contexts(base, timestamps, lazy=True)
        prepared[1].timestamps["binance"] = timestamps[48:112]
        prepared[1].time_slice["binance"] = (48, 112)
        registry = NativeDatasetRegistry(SuiteEvaluator(base, prepared, {"default": "mean"}))
        specs = list(registry._timestamps.values())
        assert len(specs) == 2
        closed = []
        cleanup = registry._arrays.cleanup
        first = RuntimeError("first owned cleanup")
        primary = ValueError("caller failure")
        def failing(values):
            closed.extend(values)
            cleanup(values)
            raise first if len(closed) == 1 else RuntimeError("later owned cleanup")
        monkeypatch.setattr(registry._arrays, "cleanup", failing)
        with pytest.raises(ValueError if body_failure else RuntimeError) as raised:
            with registry:
                if body_failure:
                    raise primary
        assert raised.value is (primary if body_failure else first)
        assert closed == specs
        assert "later owned cleanup" in caplog.text
        for spec in specs:
            with pytest.raises(FileNotFoundError):
                attach_shared_array(spec)
    finally:
        manager.cleanup()
