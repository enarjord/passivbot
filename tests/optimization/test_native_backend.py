from copy import deepcopy
from contextlib import contextmanager
import pickle
import time
from types import SimpleNamespace

import numpy as np
import pytest

import optimize
from optimization.backends.gpu_native_backend import run_backend
from optimization.evaluation_contract import CONTRACT_KEY, build_evaluation_contract
from optimization.gpu.executor import GpuBacktestService, ReplayResult
from optimization.native_checkpoint import load_checkpoint
from shared_arrays import SharedArrayManager
from tools.gpu_parity import build_parser, fixture_inputs


@contextmanager
def managed_arrays():
    manager = SharedArrayManager()
    try:
        yield manager
    finally:
        manager.cleanup()


def inputs(manager, *, n_obj=2):
    config, candles, markets, btc, timestamps = fixture_inputs(build_parser().parse_args([
        "--fixture", "trailing_martingale", "--sides", "both", "--coins", "3", "--bars", "128",
    ]))
    config["optimize"].update(backend="gpu_native", population_size=4, iters=8, seed=12)
    config["optimize"]["pymoo"]["algorithm"] = "nsga2" if n_obj == 2 else "nsga3"
    config["optimize"]["bounds"] = {}
    for side in ("long", "short"):
        for key in ("n_positions", "total_wallet_exposure_limit"):
            value = config["bot"][side]["risk"][key]
            config["optimize"]["bounds"][f"{side}_{key}"] = [value, value]
        config["optimize"]["bounds"][f"{side}_entry_initial_qty_pct"] = [0.01, 0.05]
    metrics = ["adg_strategy_eq", "drawdown_worst_strategy_eq", "fills_per_day", "volume_pct_per_day_avg"]
    config["optimize"]["scoring"] = [dict(metric=name, goal="max") for name in metrics[:n_obj]]
    config["optimize"]["limits"] = [dict(metric="drawdown_worst_strategy_eq", penalize_if="greater_than", value=0.2)]
    config["optimize"]["gpu"].update(batch_size=1, checkpoint_interval_seconds=0)
    specs = [manager.create_from(array)[0] for array in (candles, btc)]
    base = optimize.Evaluator({"binance": specs[0]}, {"binance": specs[1]}, {"binance": markets},
                              config, timestamps={"binance": timestamps})
    return base


class FakeService(GpuBacktestService):
    rows = []
    failure = None
    def __init__(self, **kwargs):
        kwargs.pop("max_dispatch_candidate_bars")
        kwargs.pop("interrupt_check")
        kwargs.pop("tuning_mode")
        super().__init__(**kwargs)

    def register_dataset(self, dataset_id, dataset):
        assert dataset.candle_coins == ("COIN00", "COIN01", "COIN02")
        def evaluate(candidates):
            if self.failure is not None:
                raise self.failure
            self.rows.extend(deepcopy(candidates))
            return [ReplayResult(metrics={name: 0.1 + row["long_entry_initial_qty_pct"]
                                         for name in dataset.metrics}, liquidated=False)
                    for row in candidates]
        super().register_dataset(dataset_id, SimpleNamespace(evaluate=evaluate))


def execute(base, recorder, checkpoint, *, resume=False, starts=(), interrupt_check=lambda: None):
    return run_backend(
        config=base.config, evaluator_for_pool=base, recorder=recorder, overrides_list=[],
        starting_configs_path=None, get_starting_configs=lambda _path: list(starts),
        configs_to_individuals=lambda configs, _bounds, _digits: list(configs),
        build_config_fn=optimize.individual_to_config, overrides_fn=optimize.optimizer_overrides,
        checkpoint_path=str(checkpoint), resume=resume, interrupt_check=interrupt_check,
        standalone_candle_coins={"binance": ("COIN00", "COIN01", "COIN02")},
    )


def guard_cpu(monkeypatch):
    import backtest
    import optimization.gpu.native as native
    def forbidden(*_args, **_kwargs):
        pytest.fail("native search must never run CPU backtests or construct CPU worker pools")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(backtest.pbr, "run_backtest_bundle", forbidden)
    monkeypatch.setattr(optimize.Evaluator, "evaluate", forbidden)
    monkeypatch.setattr(optimize.multiprocessing, "Pool", forbidden)
    monkeypatch.setattr(optimize.multiprocessing, "Manager", forbidden)
    monkeypatch.setattr(native, "CudaBacktestService", FakeService)
    FakeService.rows = []
    FakeService.failure = None


@pytest.mark.parametrize("n_obj,constrained", [(2, True), (2, False), (4, True)])
def test_native_ask_tell_evolution_records_results_and_resumes_without_cpu(monkeypatch, tmp_path, n_obj, constrained):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base = inputs(manager, n_obj=n_obj)
        if not constrained:
            base.config["optimize"]["limits"] = []
            base.build_limit_checks()
        records = []
        recorder = SimpleNamespace(record=records.append)
        checkpoint = tmp_path / "checkpoint.pkl"
        result = execute(base, recorder, checkpoint)
        assert result == dict(pool=None, pool_terminated=False)
        assert len(records) == 8
        assert all(row[CONTRACT_KEY]["execution"]["engine"] == "cuda_native" for row in records)
        assert all(row["metrics"]["constraint_violation"] == 0 for row in records)
        state = load_checkpoint(checkpoint, base.config)
        assert state["phase"] == "idle" and state["population"] is None
        assert state["completed"] == len(records)
        assert state["algorithm"].pop.get("F").shape == (4, n_obj)
        assert state["algorithm"].pop.get("G").shape == (4, int(constrained))
        done = len(FakeService.rows)
        execute(base, recorder, checkpoint, resume=True)
        assert len(FakeService.rows) == done and len(records) == 8
        base.config["optimize"]["iters"] = 12
        execute(base, recorder, checkpoint, resume=True)
        assert len(records) == 12


@pytest.mark.parametrize("seeds", [False, True])
def test_native_interrupt_checkpoint_retains_full_results_and_resumes_pending_gpu(monkeypatch, tmp_path, seeds):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base = inputs(manager)
        base.config["optimize"]["iters"] = 4
        vector = optimize.config_to_individual(base.config, base.bounds, optimization_shape=base.optimization_shape)
        starts = [list(vector) for _ in range(6)] if seeds else []
        # Distinct seed requests exercise survival after more seeds than population.
        if seeds:
            index = [key for key, _path in base.key_paths].index("long_entry_initial_qty_pct")
            for i, row in enumerate(starts):
                row[index] = 0.01 + i * 0.004
        records = []
        recorder = SimpleNamespace(record=records.append)
        checkpoint = tmp_path / "checkpoint.pkl"
        def interrupt():
            if records:
                raise KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            execute(base, recorder, checkpoint, starts=starts, interrupt_check=interrupt)
        state = load_checkpoint(checkpoint, base.config)
        evaluated = sum({"F", "G", "H"} <= item.evaluated for item in state["population"])
        assert evaluated == len(records) == state["completed"]
        assert state["phase"] == ("seeds" if seeds else "generation")
        assert FakeService.rows
        execute(base, recorder, checkpoint, resume=True)
        assert len(records) == (6 if seeds else 4)
        assert load_checkpoint(checkpoint, base.config)["phase"] == "idle"


def test_native_producer_failure_preserves_original_and_checkpoint(monkeypatch, tmp_path):
    guard_cpu(monkeypatch)
    FakeService.failure = RuntimeError("GPU producer original")
    with managed_arrays() as manager:
        base = inputs(manager)
        records = []
        checkpoint = tmp_path / "checkpoint.pkl"
        with pytest.raises(RuntimeError, match="GPU producer original") as raised:
            execute(base, SimpleNamespace(record=records.append), checkpoint)
        assert raised.value is FakeService.failure
        assert records == []
        state = load_checkpoint(checkpoint, base.config)
        assert state["completed"] == 0
        assert optimize._gpu_checkpoint_allows_empty_results(str(checkpoint), base.config)
        changed = deepcopy(base.config)
        changed["optimize"]["population_size"] = 8
        with pytest.raises(ValueError, match="critical run configuration changed"):
            optimize._gpu_checkpoint_allows_empty_results(str(checkpoint), changed)
        FakeService.failure = None
        execute(base, SimpleNamespace(record=records.append), checkpoint, resume=True)
        assert len(records) == 8


def test_native_contract_and_checkpoint_reject_other_engines_or_precision(monkeypatch, tmp_path):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base = inputs(manager)
        checkpoint = tmp_path / "checkpoint.pkl"
        execute(base, SimpleNamespace(record=lambda _row: None), checkpoint)
        state = load_checkpoint(checkpoint, base.config)
        state[CONTRACT_KEY]["execution"]["precision"] = "other"
        checkpoint.write_bytes(pickle.dumps(state))
        with pytest.raises(ValueError, match="contract changed"):
            load_checkpoint(checkpoint, base.config)
        cpu = deepcopy(base.config)
        cpu["optimize"]["backend"] = "pymoo"
        assert "execution" not in build_evaluation_contract(cpu)
        assert optimize._resume_config_mismatches({**cpu, CONTRACT_KEY: build_evaluation_contract(cpu)}, base.config)


@pytest.mark.parametrize("fail_preparation", [False, True])
def test_cpu_pipeline_starts_gpu_and_records_before_preparing_full_window(monkeypatch, tmp_path, fail_preparation):
    guard_cpu(monkeypatch)
    import optimization.backends.gpu_native_backend as backend
    import optimization.gpu.native as native
    from optimization.native_planning import NativeCandidatePlanner

    clock = [0.0]
    monkeypatch.setattr(backend, "time", SimpleNamespace(
        monotonic=time.monotonic, perf_counter=lambda: clock[0],
    ))
    submitted = []
    class ObservedService(FakeService):
        def submit(self, request):
            future = super().submit(request)
            submitted.append(future)
            return future
    monkeypatch.setattr(native, "CudaBacktestService", ObservedService)
    original = NativeCandidatePlanner.prepare
    prepared, records = [], []
    failure = ValueError("candidate preparation failed after GPU work started")
    def prepare(self, candidate_id, vector, **kwargs):
        if candidate_id != "preparation":
            if prepared:
                # Deterministic barrier: submitting the first candidate must not
                # wait for the rest of the admission window to be prepared.
                assert submitted
                submitted[0].result(timeout=5)
            if len(prepared) == 3:
                assert records  # Persist full successes before preparing more.
                if fail_preparation:
                    raise failure
            prepared.append(candidate_id)
            clock[0] += 0.03
        return original(self, candidate_id, vector, **kwargs)
    monkeypatch.setattr(NativeCandidatePlanner, "prepare", prepare)
    with managed_arrays() as manager:
        base = inputs(manager)
        base.config["optimize"].update(population_size=16, iters=16)
        checkpoint = tmp_path / "checkpoint.pkl"
        if fail_preparation:
            with pytest.raises(ValueError) as raised:
                execute(base, SimpleNamespace(record=records.append), checkpoint)
            assert raised.value is failure
            state = load_checkpoint(checkpoint, base.config)
            assert state["completed"] == len(records) > 0
            assert state["phase"] == "generation"
            monkeypatch.setattr(NativeCandidatePlanner, "prepare", original)
            execute(base, SimpleNamespace(record=records.append), checkpoint, resume=True)
        else:
            execute(base, SimpleNamespace(record=records.append), checkpoint)
        assert len(records) == 16
        assert load_checkpoint(checkpoint, base.config)["phase"] == "idle"
