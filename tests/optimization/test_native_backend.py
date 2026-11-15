from copy import deepcopy
from contextlib import contextmanager
import pickle
import logging
import time
from types import SimpleNamespace
from threading import Event

import numpy as np
import pytest

import optimize
from optimization.backends.gpu_backend import run_backend
from config import prepare_config
from config_utils import clean_config
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
    config["optimize"].update(backend="gpu", population_size=4, iters=8, seed=12)
    config["optimize"]["pymoo"]["algorithm"] = "nsga2" if n_obj == 2 else "nsga3"
    # This tiny search varies two coordinates among many fixed policy leaves.
    config["optimize"]["pymoo"]["shared"].update(
        mutation_probability=1.0, mutation_probability_per_variable=0.5,
    )
    config["optimize"]["bounds"] = {
        side: {
            "risk": {
                key: [config["bot"][side]["risk"][key]] * 2
                for key in ("max_positions", "total_wallet_exposure_limit_ratio")
            },
            "strategy": {"trailing_martingale": {"entry": {"initial_qty_fraction": [0.01, 0.05]}}},
        }
        for side in ("long", "short")
    }
    metrics = ["cagr_fraction_strategy_equity", "ulcer_index_fraction_strategy_equity", "sharpe_ratio_strategy_equity", "sortino_ratio_strategy_equity"]
    config["optimize"]["scoring"] = [dict(metric=name, goal="min" if name == "ulcer_index_fraction_strategy_equity" else "max") for name in metrics[:n_obj]]
    config["optimize"]["limits"] = [dict(metric="drawdown_max_fraction_strategy_equity", penalize_if="greater_than", value=0.2)]
    config["optimize"]["gpu"].update(batch_size=1, checkpoint_interval_seconds=0)
    coins = deepcopy(config["backtest"]["coins"])
    config = prepare_config(clean_config(config), verbose=False)
    config["backtest"]["coins"] = coins
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
        kwargs.pop("auto_request_capacity")
        kwargs.pop("auto_batch_size_limit", None)
        kwargs.pop("max_cohort_width", None)
        super().__init__(**kwargs)

    cohort_width = None
    accepted_cohort_width = None

    def register_dataset(self, dataset_id, dataset):
        assert dataset.candle_coins == ("COIN00", "COIN01", "COIN02")
        def evaluate(candidates):
            if self.failure is not None:
                raise self.failure
            self.rows.extend(deepcopy(candidates))
            return [ReplayResult(metrics={name: 0.1 + row["long_entry_initial_qty_fraction"]
                                         for name in dataset.metrics}, liquidated=False,
                                 metadata={"metric_status": {}, "metric_metadata": {"definition_version": 2}})
                    for row in candidates]
        super().register_dataset(dataset_id, SimpleNamespace(evaluate=evaluate))


def execute(base, recorder, checkpoint, *, resume=False, starts=(), interrupt_check=lambda: None, evaluator=None):
    return run_backend(
        config=base.config, evaluator_for_pool=base if evaluator is None else evaluator, recorder=recorder, overrides_list=[],
        starting_configs_path=None, get_starting_configs=lambda _path: list(starts),
        configs_to_individuals=lambda configs, _bounds, _digits: list(configs),
        build_config_fn=optimize.individual_to_config, overrides_fn=optimize.optimizer_overrides,
        checkpoint_path=str(checkpoint), resume=resume, interrupt_check=interrupt_check,
        standalone_candle_coins={"binance": ("COIN00", "COIN01", "COIN02")},
    )


def guard_cpu(monkeypatch):
    import backtest
    import optimization.gpu.native as native
    import optimization.gpu.runtime as runtime
    def forbidden(*_args, **_kwargs):
        pytest.fail("native search must never run CPU backtests or construct CPU worker pools")
    monkeypatch.setattr(backtest, "execute_backtest", forbidden)
    monkeypatch.setattr(backtest, "run_backtest", forbidden)
    monkeypatch.setattr(backtest.pbr, "run_backtest_bundle", forbidden)
    monkeypatch.setattr(optimize.Evaluator, "evaluate", forbidden)
    monkeypatch.setattr(optimize.multiprocessing, "Pool", forbidden)
    monkeypatch.setattr(optimize.multiprocessing, "Manager", forbidden)
    monkeypatch.setattr(native, "NativeBacktestService", FakeService)
    # The mocked service has no device. Default to CUDA for legacy max tests.
    monkeypatch.setattr(runtime, "gpu_device", lambda _torch=None: "cuda")
    FakeService.rows = []
    FakeService.failure = None


@pytest.mark.parametrize("state", [
    {"backend": "gpu_native", "version": 2},
    {"backend": "gpu", "version": 2},
    {"seed_bootstrap_complete": False},
])
def test_pre_cutover_checkpoint_cannot_supply_gpu_fitness(tmp_path, state):
    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(pickle.dumps(state))
    with pytest.raises(ValueError, match="compatible native checkpoint"):
        load_checkpoint(checkpoint, {})


def test_removed_checkpoint_class_has_a_fresh_run_instruction(tmp_path):
    checkpoint = tmp_path / "checkpoint.pkl"
    checkpoint.write_bytes(b"coptimization.backends.gpu_native_backend\nNativeSearchProblem\n.")
    with pytest.raises(ValueError, match="start a fresh run.*saved configs"):
        load_checkpoint(checkpoint, {})


@pytest.mark.parametrize("seed", [1, 19, 2026])
def test_native_fresh_population_has_no_implicit_input_config(monkeypatch, tmp_path, seed):
    import optimization.backends.gpu_backend as backend

    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base = inputs(manager)
        base.config["optimize"]["seed"] = seed
        population_size = base.config["optimize"]["population_size"]
        rng_state = np.random.get_state()
        try:
            np.random.seed(seed)
            expected = backend._build_random_sampling(base.bounds, population_size)
            np.random.seed(seed)
            original_build = backend._build_algorithm
            received = []

            def check_initial_population(**kwargs):
                received.append(kwargs["sampling"].copy())
                np.testing.assert_array_equal(kwargs["sampling"], expected)
                return original_build(**kwargs)

            monkeypatch.setattr(backend, "_build_algorithm", check_initial_population)
            execute(base, SimpleNamespace(record=lambda _row: None), tmp_path / "checkpoint.pkl")
        finally:
            np.random.set_state(rng_state)
        assert len(received) == 1
        input_vector = optimize.config_to_individual(
            base.config, base.bounds, optimization_shape=base.optimization_shape,
        )
        assert not np.any(np.all(received[0] == input_vector, axis=1))


def test_native_fresh_population_is_reproducible_and_changes_with_seed():
    from optimization.backends.gpu_backend import _build_random_sampling
    from optimization.bounds import Bound

    bounds = [Bound(0.0, 1.0), Bound(0.0, 1.0, 0.01)]
    rng_state = np.random.get_state()
    try:
        np.random.seed(19)
        first = _build_random_sampling(bounds, 8)
        np.random.seed(19)
        repeated = _build_random_sampling(bounds, 8)
        np.random.seed(20)
        different = _build_random_sampling(bounds, 8)
    finally:
        np.random.set_state(rng_state)
    np.testing.assert_array_equal(first, repeated)
    assert np.all(np.any(first != different, axis=1))


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
        assert all(row[CONTRACT_KEY]["execution"]["engine"] == build_evaluation_contract(base.config)["execution"]["engine"] for row in records)
        assert all(row["metrics"]["constraint_violation"] == 0 for row in records)
        state = load_checkpoint(checkpoint, base.config)
        assert state["phase"] == "idle" and state["population"] is None
        assert state["completed"] == len(records)
        assert state["algorithm"].pop.get("F").shape == (4, n_obj)
        # Selected metric availability constrains every search, including a
        # configuration without numeric limits.
        constraints = state["algorithm"].pop.get("G")
        assert constraints.shape == (4, 1)
        assert np.all(constraints <= 0)
        done = len(FakeService.rows)
        execute(base, recorder, checkpoint, resume=True)
        assert len(FakeService.rows) == done and len(records) == 8
        base.config["optimize"]["iters"] = 12
        execute(base, recorder, checkpoint, resume=True)
        assert len(records) == 12


@pytest.mark.parametrize("authored", [False, True])
@pytest.mark.parametrize("cohort_limit", [None, 6])
def test_cuda_max_widens_search_without_growing_survivors(monkeypatch, tmp_path, authored, cohort_limit):
    import optimization.backends.gpu_backend as backend

    guard_cpu(monkeypatch)
    phases = []
    evaluate = backend._Search.evaluate_population

    def observe(self):
        phases.append((self.state["phase"], len(self.state["population"]), self.session.max_candidates))
        return evaluate(self)

    monkeypatch.setattr(backend._Search, "evaluate_population", observe)
    with managed_arrays() as manager:
        base = inputs(manager)
        base.config["optimize"]["iters"] = 21
        gpu = base.config["optimize"]["gpu"]
        gpu.update(tuning_mode="max", batch_size=8, search_cohort_size=cohort_limit)
        vector = optimize.config_to_individual(base.config, base.bounds, optimization_shape=base.optimization_shape)
        starts = [vector] if authored else []
        checkpoint = tmp_path / "checkpoint.pkl"
        records = []
        recorder = SimpleNamespace(record=records.append)

        execute(base, recorder, checkpoint, starts=starts)
        state = load_checkpoint(checkpoint, base.config)
        assert state["max_search_budget"].target_width == 8
        assert state["max_search_budget"].cohort_limit == cohort_limit
        assert state["max_search_allocated"] == 21
        assert len(state["algorithm"].pop) == base.config["optimize"]["population_size"] == 4
        width = cohort_limit or 8
        assert phases[:2] == [("seeds", width, width), ("generation", 4, width)]
        full, tail = divmod(21 - width + authored, width)
        assert phases[2:] == [("generation", width, width)] * full + [("generation", tail, tail)]
        assert len(records) == 21 + authored
        assert len(FakeService.rows) == len(records)

        execute(base, recorder, checkpoint, resume=True)
        assert len(records) == 21 + authored
        gpu["search_cohort_size"] = width + 1
        with pytest.raises(ValueError, match="search_cohort_size"):
            execute(base, recorder, checkpoint, resume=True)
        gpu["search_cohort_size"] = cohort_limit
        base.config["optimize"]["iters"] = 20
        with pytest.raises(ValueError, match="iters below already allocated search attempts"):
            execute(base, recorder, checkpoint, resume=True)
        assert len(records) == 21 + authored
        base.config["optimize"]["iters"] = 24
        execute(base, recorder, checkpoint, resume=True)
        assert phases[-1] == ("generation", 3, 3)
        assert len(records) == 24 + authored


def test_cuda_max_resume_keeps_pending_trial_vectors_and_restarts_from_accepted_width(monkeypatch, tmp_path):
    import optimization.backends.gpu_backend as backend
    import optimization.gpu.native as native

    guard_cpu(monkeypatch)
    monkeypatch.setattr(native, "cuda_initial_candidate_width", lambda: 8)
    requested_baselines = []

    class Service(FakeService):
        def __init__(self, *, max_cohort_width, **kwargs):
            requested_baselines.append(max_cohort_width)
            self.accepted_cohort_width = self.cohort_width = max_cohort_width
            kwargs["batch_size"] = 8
            super().__init__(**kwargs)

    monkeypatch.setattr(native, "NativeBacktestService", Service)
    evaluate = backend._Search.evaluate_population
    cohorts = []

    def observed_population(self):
        population = self.state["population"].get("X").copy()
        cohorts.append(population)
        if len(cohorts) == 3:
            assert len(population) == 16
            self.checkpoint(force=True)
            raise KeyboardInterrupt
        evaluate(self)
        if len(cohorts) == 1:
            self.session.service.accepted_cohort_width = 12
            self.session.service.cohort_width = 16

    monkeypatch.setattr(backend._Search, "evaluate_population", observed_population)
    with managed_arrays() as manager:
        base = inputs(manager)
        base.config["optimize"]["iters"] = 40
        base.config["optimize"]["gpu"].update(tuning_mode="max", batch_size=None)
        checkpoint = tmp_path / "checkpoint.pkl"
        records = []
        recorder = SimpleNamespace(record=records.append)

        with pytest.raises(KeyboardInterrupt):
            execute(base, recorder, checkpoint)
        paused = load_checkpoint(checkpoint, base.config)
        assert paused["phase"] == "generation" and paused["max_search_allocated"] == 24
        assert paused["max_search_budget"].target_width == 12
        np.testing.assert_array_equal(paused["population"].get("X"), cohorts[2])

        execute(base, recorder, checkpoint, resume=True)
        done = load_checkpoint(checkpoint, base.config)
        assert requested_baselines == [8, 12]
        np.testing.assert_array_equal(cohorts[3], cohorts[2])
        assert [len(cohort) for cohort in cohorts[4:]] == [12, 4]
        assert done["phase"] == "idle" and done["max_search_allocated"] == 40
        assert done["max_search_budget"].target_width == 12
        assert len(records) == 40 and len(done["algorithm"].pop) == 4


@pytest.mark.parametrize("cohort_limit,accepted_width,seed_width,offspring_widths", [
    (None, 32, 64, [32, 32, 2]),
    (56, 60, 56, [56, 18]),
])
def test_metal_max_bootstrap_and_offspring_use_metal_width_and_budget(
    monkeypatch, tmp_path, cohort_limit, accepted_width, seed_width, offspring_widths,
):
    from optimization import native_planner_capacity
    from optimization.backends import gpu_backend
    from optimization.gpu import native, runtime
    from optimization.gpu.residency import MpsInputMemoryBudget

    guard_cpu(monkeypatch)
    lookups = []
    capacities = []

    def widths(device):
        lookups.append(device)
        return 64, 96

    budget = MpsInputMemoryBudget(2**30, 2**30, max_throughput=True)

    def service(**kwargs):
        capacities.append((kwargs["max_cohort_width"], kwargs["max_pending"], kwargs["auto_request_capacity"]))
        assert kwargs.pop("input_memory_budget") is budget
        return FakeService(**{**kwargs, "batch_size": 64})

    def forbidden(*_args):
        pytest.fail("search should use the captured Metal budget and its device")

    monkeypatch.setattr(native, "max_candidate_widths", widths)
    monkeypatch.setattr(native, "cuda_initial_candidate_width", forbidden)
    monkeypatch.setattr(runtime, "gpu_device", forbidden)
    monkeypatch.setattr(native, "NativeBacktestService", service)
    monkeypatch.setattr(native_planner_capacity, "mps_plan_worker_count", lambda *_args, **_kwargs: 0)
    observed = []
    evaluate = gpu_backend._Search.evaluate_population

    def observe(self):
        observed.append((self.state["phase"], len(self.state["population"]), self.session.max_candidates))
        evaluate(self)
        if self.state["phase"] == "seeds":
            self.session.service.cohort_width = self.session.service.accepted_cohort_width = accepted_width

    monkeypatch.setattr(gpu_backend._Search, "evaluate_population", observe)
    with managed_arrays() as manager:
        base = inputs(manager)
        base._gpu_input_memory_budget = budget
        base.config["optimize"]["iters"] = 80
        base.config["optimize"]["gpu"].update(
            tuning_mode="max", batch_size=None, search_cohort_size=cohort_limit,
        )
        template = optimize.config_to_individual(base.config, base.bounds, optimization_shape=base.optimization_shape)
        keys = [key for key, _path in base.key_paths]
        quantity = keys.index("bot.long.strategy.trailing_martingale.entry.initial_qty_fraction")
        starts = []
        for value in np.linspace(0.012, 0.03, 50):
            vector = list(template)
            vector[quantity] = float(value)
            starts.append(vector)
        records = []
        checkpoint = tmp_path / "metal-max.pkl"
        execute(base, SimpleNamespace(record=records.append), checkpoint, starts=starts)
        state = load_checkpoint(checkpoint, base.config)

    assert lookups == ["mps"]
    assert capacities == [(64, 96, 96)]
    assert observed[:2] == [("seeds", seed_width, seed_width), ("generation", 4, seed_width)]
    assert observed[2:] == [("generation", width, width) for width in offspring_widths]
    assert len(records) == 130
    assert state["max_search_allocated"] == 80
    assert state["max_search_budget"].target_width == accepted_width
    assert state["max_search_budget"].cohort_limit == cohort_limit
    assert len(state["algorithm"].pop) == 4


def test_cuda_max_tiny_budget_never_pads_evaluated_parents(monkeypatch, tmp_path):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base = inputs(manager)
        base.config["optimize"]["iters"] = 1
        base.config["optimize"]["gpu"].update(tuning_mode="max", batch_size=8)
        checkpoint = tmp_path / "checkpoint.pkl"
        records = []
        recorder = SimpleNamespace(record=records.append)
        execute(base, recorder, checkpoint)
        assert len(records) == len(FakeService.rows) == 1
        state = load_checkpoint(checkpoint, base.config)
        assert len(state["algorithm"].pop) == state["max_search_allocated"] == 1
        execute(base, recorder, checkpoint, resume=True)
        assert len(records) == 1


def test_cohort_diagnostics_exclude_restored_attempts_and_preserve_result_lineage(monkeypatch, tmp_path, caplog):
    guard_cpu(monkeypatch)
    caplog.set_level(logging.DEBUG)
    with managed_arrays() as manager:
        base = inputs(manager)
        records = []
        checkpoint = tmp_path / "checkpoint.pkl"
        execute(base, SimpleNamespace(record=records.append), checkpoint)
        summaries = [record.getMessage() for record in caplog.records
                     if record.levelno == logging.INFO and record.getMessage().startswith("GPU cohort |")]
        assert len(summaries) == 2
        def facts(message):
            return dict(token.split("=", 1) for token in message.split() if "=" in token)
        assert sum(int(facts(message)["attempts"]) for message in summaries) == len(records) == 8
        assert sum(int(facts(message)["gpu_rows"]) for message in summaries) == len(FakeService.rows)
        assert all("status=complete" in message and "phase=generation" in message for message in summaries)
        assert "operation=ask" in caplog.text and "operation=tell" in caplog.text
        state = load_checkpoint(checkpoint, base.config)
        assert not any("diagnostic" in key for key in state)
        caplog.clear()
        execute(base, SimpleNamespace(record=records.append), checkpoint, resume=True)
        assert not any(record.getMessage().startswith("GPU cohort |") for record in caplog.records)
        before_rows = len(FakeService.rows)
        base.config["optimize"]["iters"] = 12
        execute(base, SimpleNamespace(record=records.append), checkpoint, resume=True)
        summaries = [record.getMessage() for record in caplog.records
                     if record.levelno == logging.INFO and record.getMessage().startswith("GPU cohort |")]
        assert len(summaries) == 1 and facts(summaries[0])["attempts"] == "4"
        assert int(facts(summaries[0])["gpu_rows"]) == len(FakeService.rows) - before_rows
        assert len(records) == 12


def test_native_backend_validates_effective_fixed_runtime_config(monkeypatch, tmp_path):
    guard_cpu(monkeypatch)
    registered_collateral = []
    register = FakeService.register_dataset
    def observe_registration(self, dataset_id, dataset):
        import json
        registered_collateral.append(json.loads(dataset.config_json)["backtest"]["btc_collateral_target_ratio"])
        return register(self, dataset_id, dataset)
    monkeypatch.setattr(FakeService, "register_dataset", observe_registration)
    with managed_arrays() as manager:
        base = inputs(manager)
        base.config["backtest"]["btc_collateral_target_ratio"] = 0.5
        base.config["optimize"]["fixed_runtime_overrides"] = {"backtest.btc_collateral_target_ratio": 0.0}
        # CLI preflight and native dataset preparation both validate the effective
        # runtime contract; the raw template must not introduce a second gate.
        optimize._run_gpu_preparation_preflight(base.config, {})
        records = []
        execute(base, SimpleNamespace(record=records.append), tmp_path / "checkpoint.pkl")
        assert registered_collateral and all(value == 0.0 for value in registered_collateral)
        assert len(records) == len(FakeService.rows) == 8
        assert all(row["backtest"]["btc_collateral_target_ratio"] == 0.0 for row in records)


def test_native_effective_unsupported_config_is_rejected_before_service(monkeypatch, tmp_path):
    guard_cpu(monkeypatch)
    import optimization.gpu.native as native
    def forbidden_service(**_kwargs):
        pytest.fail("unsupported effective config must fail before device service construction")
    monkeypatch.setattr(native, "NativeBacktestService", forbidden_service)
    with managed_arrays() as manager:
        base = inputs(manager)
        base.config["backtest"]["btc_collateral_target_ratio"] = 0.0
        base.config["optimize"]["fixed_runtime_overrides"] = {"backtest.btc_collateral_target_ratio": 0.5}
        with pytest.raises(ValueError, match="btc_collateral_target_ratio=0"):
            optimize._run_gpu_preparation_preflight(base.config, {})
        with pytest.raises(ValueError, match="btc_collateral_target_ratio=0"):
            execute(base, SimpleNamespace(record=lambda _row: None), tmp_path / "checkpoint.pkl")
        assert FakeService.rows == []


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
            index = [key for key, _path in base.key_paths].index("bot.long.strategy.trailing_martingale.entry.initial_qty_fraction")
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


@pytest.mark.parametrize("changed_execution", [{"precision": "other"}, {"version": 1}])
def test_native_contract_and_checkpoint_reject_other_engines_or_precision(monkeypatch, tmp_path, changed_execution):
    guard_cpu(monkeypatch)
    with managed_arrays() as manager:
        base = inputs(manager)
        checkpoint = tmp_path / "checkpoint.pkl"
        execute(base, SimpleNamespace(record=lambda _row: None), checkpoint)
        state = load_checkpoint(checkpoint, base.config)
        assert state[CONTRACT_KEY]["execution"]["version"] == 2
        state[CONTRACT_KEY]["execution"].update(changed_execution)
        checkpoint.write_bytes(pickle.dumps(state))
        with pytest.raises(ValueError, match="contract changed"):
            load_checkpoint(checkpoint, base.config)
        cpu = deepcopy(base.config)
        cpu["optimize"]["backend"] = "pymoo"
        assert "execution" not in build_evaluation_contract(cpu)
        assert optimize._resume_config_mismatches({**cpu, CONTRACT_KEY: build_evaluation_contract(cpu)}, base.config)


@pytest.mark.parametrize("fail_preparation", [False, True])
@pytest.mark.parametrize("duplicate_seed", [False, True])
def test_cpu_pipeline_starts_gpu_and_records_before_preparing_full_window(
    monkeypatch, tmp_path, fail_preparation, duplicate_seed
):
    guard_cpu(monkeypatch)
    import optimization.backends.gpu_backend as backend
    import optimization.gpu.native as native
    from optimization.native_planning import NativeCandidatePlanner

    def sampling(bounds, population_size):
        # Control cohort identity: pymoo legitimately removes duplicate seeds,
        # so an uncontrolled random sample need not yield population_size rows.
        population = np.asarray([
            [bound.low + (bound.high - bound.low) * (i + 1) / (population_size + 1)
             for bound in bounds] for i in range(population_size)
        ])
        if duplicate_seed:
            population[-1] = population[0]
        return population
    monkeypatch.setattr(backend, "_build_random_sampling", sampling)

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
    monkeypatch.setattr(native, "NativeBacktestService", ObservedService)
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
        assert len(records) == 16 - int(duplicate_seed)
        assert load_checkpoint(checkpoint, base.config)["phase"] == "idle"


@pytest.mark.parametrize("fail_preparation", [False, True])
def test_preparing_hint_seals_full_cpu_windows_and_clears_after_failure(
    monkeypatch, tmp_path, fail_preparation,
):
    import optimization.gpu.native as native
    from optimization.native_planning import NativeCandidatePlanner
    from optimization.native_session import NativeEvaluationSession

    guard_cpu(monkeypatch)
    hints, services, full_polls = [], [], []
    class ObservedService(FakeService):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.release = Event()
            services.append(self)
        def register_dataset(self, dataset_id, dataset):
            super().register_dataset(dataset_id, dataset)
            replay = self._replays[dataset_id]
            evaluate = replay.evaluate
            def wait_for_cpu_window(candidates):
                assert self.release.wait(5), "CPU admission must seal before waiting for a completion"
                return evaluate(candidates)
            replay.evaluate = wait_for_cpu_window
        def set_preparing(self, dataset_ids):
            hints.append(frozenset(dataset_ids))
            super().set_preparing(dataset_ids)
            if not dataset_ids:
                self.release.set()
    monkeypatch.setattr(native, "NativeBacktestService", ObservedService)
    original_poll = NativeEvaluationSession.poll
    def poll(self, **kwargs):
        if len(self.active_candidate_ids) == self.max_candidates:
            full_polls.append(True)
            assert not self.service._preparing
        return original_poll(self, **kwargs)
    monkeypatch.setattr(NativeEvaluationSession, "poll", poll)
    original_prepare = NativeCandidatePlanner.prepare
    failure = ValueError("later CPU plan failed")
    def prepare(self, candidate_id, vector, **kwargs):
        if fail_preparation and candidate_id == "candidate:2":
            assert hints[-1]
            raise failure
        return original_prepare(self, candidate_id, vector, **kwargs)
    monkeypatch.setattr(NativeCandidatePlanner, "prepare", prepare)
    with managed_arrays() as manager:
        base = inputs(manager)
        records = []
        run = lambda: execute(base, SimpleNamespace(record=records.append), tmp_path / "checkpoint.pkl")
        if fail_preparation:
            with pytest.raises(ValueError) as raised:
                run()
            assert raised.value is failure
        else:
            run()
            assert len(records) == 8 and full_polls
        assert any(hints) and hints[-1] == frozenset()
        assert services and all(service._preparing == frozenset() for service in services)


@pytest.mark.parametrize("capacity", [2, 4])
def test_normal_window_submits_ready_rows_before_sealing_with_backpressure(monkeypatch, tmp_path, capacity):
    import optimization.backends.gpu_backend as backend
    import optimization.gpu.native as native

    guard_cpu(monkeypatch)
    # Keep this preparation burst within the existing soft cadence without a
    # wall-clock race. Every candidate still passes the ordinary planner.
    monkeypatch.setattr(backend, "time", SimpleNamespace(
        monotonic=time.monotonic, perf_counter=lambda: 0.0,
    ))
    def sampling(bounds, population_size):
        return np.asarray([
            [bound.low + (bound.high - bound.low) * (i + 1) / (population_size + 1)
             for bound in bounds] for i in range(population_size)
        ])
    monkeypatch.setattr(backend, "_build_random_sampling", sampling)
    batches, submitted, services = [], [], []
    class WarmService(FakeService):
        def __init__(self, **kwargs):
            kwargs.update(max_pending=capacity, max_batch_delay=None)
            super().__init__(**kwargs)
            self.entered, self.release = Event(), Event()
            self.sealed = False
            services.append(self)
        def submit(self, request):
            future = super().submit(request)
            submitted.append(request.request_id)
            return future
        def register_dataset(self, dataset_id, dataset):
            super().register_dataset(dataset_id, dataset)
            replay = self._replays[dataset_id]
            evaluate = replay.evaluate
            def blocked(candidates):
                batches.append(tuple(row["long_entry_initial_qty_fraction"] for row in candidates))
                self.entered.set()
                assert self.release.wait(5), "normal window did not seal"
                return evaluate(candidates)
            replay.evaluate = blocked
        def set_preparing(self, dataset_ids):
            super().set_preparing(dataset_ids)
            if not dataset_ids and not self.sealed and self._outstanding:
                self.sealed = True
                # Let the worker observe this seal before the following poll.
                # The old ordering deterministically dispatched its first row.
                assert self.entered.wait(5), "sealed prepared work did not dispatch"
                self.release.set()
    monkeypatch.setattr(native, "NativeBacktestService", WarmService)
    with managed_arrays() as manager:
        base = inputs(manager)
        base.config["optimize"].update(population_size=4, iters=4)
        base.config["optimize"]["gpu"].update(batch_size=4, tuning_mode="auto")
        records, checkpoint = [], tmp_path / "checkpoint.pkl"
        try:
            execute(base, SimpleNamespace(record=records.append), checkpoint)
        finally:
            for service in services:
                service.release.set()
        assert len(batches[0]) == capacity
        replayed = [value for batch in batches for value in batch]
        recorded = [row["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["initial_qty_fraction"] for row in records]
        assert len(replayed) == len(recorded) == len(set(submitted)) == 4
        assert sorted(replayed) == sorted(recorded)
        assert all(name.startswith("candidate:") for name in submitted)
        assert load_checkpoint(checkpoint, base.config)["completed"] == 4


@pytest.mark.parametrize("tuning_mode,seed_count,expected_window", [
    ("conservative", 12, 12), ("conservative", 1100, 1024), ("max", 1100, 1100),
])
def test_seed_admission_uses_available_cohort_on_fresh_and_resumed_runs(
    monkeypatch, tmp_path, tuning_mode, seed_count, expected_window,
):
    import optimization.backends.gpu_backend as backend

    guard_cpu(monkeypatch)
    observed = []
    class StopBeforeEvaluation(Exception):
        pass
    def inspect_seed_window(self):
        observed.append((self.state["phase"], len(self.state["population"]),
                         self.session.max_candidates))
        raise StopBeforeEvaluation
    monkeypatch.setattr(backend._Search, "evaluate_population", inspect_seed_window)

    with managed_arrays() as manager:
        base = inputs(manager)
        base.config["optimize"]["gpu"]["tuning_mode"] = tuning_mode
        vector = optimize.config_to_individual(
            base.config, base.bounds, optimization_shape=base.optimization_shape,
        )
        columns = [key for key, _path in base.key_paths]
        long_column = columns.index("bot.long.strategy.trailing_martingale.entry.initial_qty_fraction")
        short_column = columns.index("bot.short.strategy.trailing_martingale.entry.initial_qty_fraction")
        starts = [list(vector) for _ in range(seed_count)]
        for i, row in enumerate(starts):
            row[long_column] = 0.01 + 0.001 * (i % 34)
            row[short_column] = 0.01 + 0.001 * (i // 34)
        checkpoint = tmp_path / "checkpoint.pkl"
        recorder = SimpleNamespace(record=lambda _row: pytest.fail("no candidate has run"))
        with pytest.raises(StopBeforeEvaluation):
            execute(base, recorder, checkpoint, starts=starts)
        fresh = load_checkpoint(checkpoint, base.config)
        np.testing.assert_array_equal(fresh["population"].get("X"), starts)
        with pytest.raises(StopBeforeEvaluation):
            execute(base, recorder, checkpoint, resume=True)
        restored = load_checkpoint(checkpoint, base.config)
        np.testing.assert_array_equal(restored["population"].get("X"), starts)
        assert observed == [("seeds", seed_count, expected_window)] * 2
        assert restored["completed"] == restored["sequence"] == 0
        assert FakeService.rows == []


def test_seed_cohort_fills_batch_beyond_evolutionary_population(monkeypatch, tmp_path):
    import optimization.backends.gpu_backend as backend
    import optimization.gpu.native as native

    guard_cpu(monkeypatch)
    monkeypatch.setattr(backend, "time", SimpleNamespace(
        monotonic=time.monotonic, perf_counter=lambda: 0.0,
    ))
    batches, hints, services, windows = [], [], [], []
    evaluate_population = backend._Search.evaluate_population
    def observe_window(self):
        windows.append((self.state["phase"], self.session.max_candidates))
        return evaluate_population(self)
    monkeypatch.setattr(backend._Search, "evaluate_population", observe_window)
    class WideService(FakeService):
        def __init__(self, **kwargs):
            kwargs.update(max_pending=8, max_batch_delay=None)
            super().__init__(**kwargs)
            services.append(self)
        def register_dataset(self, dataset_id, dataset):
            super().register_dataset(dataset_id, dataset)
            replay = self._replays[dataset_id]
            evaluate = replay.evaluate
            def record_batch(candidates):
                batches.append(tuple(row["long_entry_initial_qty_fraction"] for row in candidates))
                return evaluate(candidates)
            replay.evaluate = record_batch
        def set_preparing(self, dataset_ids):
            hints.append(frozenset(dataset_ids))
            super().set_preparing(dataset_ids)
    monkeypatch.setattr(native, "NativeBacktestService", WideService)

    with managed_arrays() as manager:
        base = inputs(manager)
        base.config["optimize"].update(population_size=4, iters=4)
        base.config["optimize"]["gpu"].update(batch_size=8, tuning_mode="auto")
        vector = optimize.config_to_individual(
            base.config, base.bounds, optimization_shape=base.optimization_shape,
        )
        column = [key for key, _path in base.key_paths].index(
            "bot.long.strategy.trailing_martingale.entry.initial_qty_fraction",
        )
        starts = [list(vector) for _ in range(18)]
        for i, row in enumerate(starts):
            row[column] = (10 + i) / 1000
        records, checkpoint = [], tmp_path / "checkpoint.pkl"
        execute(base, SimpleNamespace(record=records.append), checkpoint, starts=starts)

        assert len(batches[0]) == 8 > base.config["optimize"]["population_size"]
        replayed = [value for batch in batches for value in batch]
        assert sorted(replayed) == sorted(row[column] for row in starts)
        assert len(records) == len(FakeService.rows) == 18
        state = load_checkpoint(checkpoint, base.config)
        assert state["completed"] == 18 and state["phase"] == "idle"
        assert len(state["algorithm"].pop) == 4
        assert any(hints) and hints[-1] == frozenset()
        assert all(service._preparing == frozenset() for service in services)
        assert windows == [("seeds", 18), ("generation", 4)]

        # The drained seed window must not survive into ordinary evolution,
        # either in this run or when a later generation resumes from disk.
        base.config["optimize"]["iters"] = 8
        execute(base, SimpleNamespace(record=records.append), checkpoint, resume=True)
        assert windows[2:] == [("generation", 4)]
        restored = load_checkpoint(checkpoint, base.config)
        assert restored["completed"] == len(records) > 18
        assert restored["phase"] == "idle" and len(restored["algorithm"].pop) == 4
