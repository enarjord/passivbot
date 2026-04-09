import copy

import numpy as np

from optimization.bounds import Bound
from optimization.backends import pymoo_backend
from opt_utils import load_results
from optimize import ResultRecorder
from tools.pareto_dash import load_pareto_dataframe


class FakePool:
    def __init__(self, processes, initializer=None):
        self.processes = processes
        self.closed = False
        self.joined = False
        self.terminated = False
        if initializer is not None:
            initializer()

    def starmap(self, fn, iterable):
        return [fn(*args) for args in iterable]

    def apply_async(self, fn, args=()):
        class _Result:
            def __init__(self, value):
                self._value = value

            def ready(self):
                return True

            def get(self):
                return self._value

        return _Result(fn(*args))

    def close(self):
        self.closed = True

    def join(self):
        self.joined = True

    def terminate(self):
        self.terminated = True


class FakeRecorder:
    def __init__(self):
        self.entries = []

    def record(self, entry):
        self.entries.append(entry)


class FakeEvaluator:
    def __init__(self, scoring_keys):
        self.bounds = [Bound(0.0, 1.0, 0.1), Bound(0.0, 1.0, 0.1)]
        self.limit_checks = []
        self.use_duplicate_guard = True
        self.config = {
            "backtest": {"coins": {"binance": ["BTC/USDT:USDT"]}},
            "bot": {"long": {"a": 0.0, "b": 0.0}},
            "optimize": {"scoring": list(scoring_keys), "backend": "pymoo"},
        }

    def evaluate(self, vector, overrides_list):
        x0 = float(vector[0])
        x1 = float(vector[1])
        objectives = [-x0]
        if len(self.config["optimize"]["scoring"]) > 1:
            objectives.append(x1)
        return (
            tuple(objectives),
            0.0,
            {
                "objectives": {f"w_{idx}": value for idx, value in enumerate(objectives)},
                "constraint_violation": 0.0,
            },
        )


def _build_config(vector, overrides_fn, overrides_list, template):
    config = copy.deepcopy(template)
    config["bot"]["long"]["a"] = float(vector[0])
    config["bot"]["long"]["b"] = float(vector[1])
    return config


def _get_starting_configs(_path):
    return []


def _configs_to_individuals(_cfgs, _bounds, _sig_digits):
    return []


def _many_starting_configs(_path):
    return [{"seed": idx} for idx in range(6)]


def _many_starting_individuals(_cfgs, _bounds, _sig_digits):
    return [
        [0.1, 0.1],
        [0.2, 0.2],
        [0.3, 0.3],
        [0.4, 0.4],
        [0.5, 0.5],
        [0.6, 0.6],
    ]


def _ignore_sigint():
    return None


def _noop_overrides(_overrides_list, config, _pside):
    return config


def test_build_algorithm_uses_nsga3_with_auto_reference_directions():
    bounds = [Bound(0.0, 1.0, 0.1) for _ in range(6)]
    config = {
        "optimize": {
            "backend": "pymoo",
            "population_size": 500,
            "pymoo": {
                "algorithm": "nsga3",
                "shared": {
                    "crossover_eta": 20.0,
                    "crossover_prob_var": 0.5,
                    "mutation_eta": 20.0,
                    "mutation_prob_var": "auto",
                    "eliminate_duplicates": True,
                },
                "algorithms": {
                    "nsga3": {
                        "ref_dirs": {
                            "method": "das_dennis",
                            "n_partitions": "auto",
                        }
                    }
                },
            },
        }
    }
    population_plan = pymoo_backend._resolve_pymoo_population_plan(config, n_obj=8)
    algorithm = pymoo_backend._build_algorithm(
        config=config,
        sampling=np.zeros((500, len(bounds)), dtype=np.float64),
        bounds=bounds,
        sig_digits=4,
        population_plan=population_plan,
    )

    assert algorithm.__class__.__name__ == "NSGA3"
    assert algorithm.pop_size == 500
    assert len(algorithm.ref_dirs) == 330


def test_resolve_algorithm_auto_selects_nsga2_for_three_or_fewer_objectives():
    config = {"optimize": {"backend": "pymoo", "pymoo": {"algorithm": "auto"}}}

    assert pymoo_backend._resolve_pymoo_algorithm_name(config, n_obj=1) == "nsga2"
    assert pymoo_backend._resolve_pymoo_algorithm_name(config, n_obj=3) == "nsga2"


def test_resolve_algorithm_auto_selects_nsga3_for_many_objectives():
    config = {"optimize": {"backend": "pymoo", "pymoo": {"algorithm": "auto"}}}

    assert pymoo_backend._resolve_pymoo_algorithm_name(config, n_obj=4) == "nsga3"
    assert pymoo_backend._resolve_pymoo_algorithm_name(config, n_obj=8) == "nsga3"


def test_resolve_population_plan_uses_auto_nsga3_population_when_null():
    plan = pymoo_backend._resolve_pymoo_population_plan(
        config={
            "optimize": {
                "backend": "pymoo",
                "population_size": None,
                "pymoo": {
                    "algorithm": "auto",
                    "algorithms": {
                        "nsga3": {
                            "ref_dirs": {
                                "method": "das_dennis",
                                "n_partitions": "auto",
                            }
                        }
                    },
                },
            }
        },
        n_obj=8,
    )

    assert plan["requested_population_size"] is None
    assert plan["actual_population_size"] == 330
    assert plan["n_partitions"] == 4
    assert len(plan["ref_dirs"]) == 330


def test_build_algorithm_falls_back_to_nsga2_for_single_objective():
    bounds = [Bound(0.0, 1.0, 0.1) for _ in range(4)]
    config = {
        "optimize": {
                "backend": "pymoo",
                "population_size": 16,
                "pymoo": {
                    "algorithm": "auto",
                },
            }
        }
    population_plan = pymoo_backend._resolve_pymoo_population_plan(config, n_obj=1)
    algorithm = pymoo_backend._build_algorithm(
        config=config,
        sampling=np.zeros((16, len(bounds)), dtype=np.float64),
        bounds=bounds,
        sig_digits=4,
        population_plan=population_plan,
    )

    assert algorithm.__class__.__name__ == "NSGA2"


def test_run_backend_records_entries(monkeypatch):
    monkeypatch.setattr(pymoo_backend.multiprocessing, "Pool", FakePool)
    evaluator = FakeEvaluator(["adg", "drawdown_worst"])
    recorder = FakeRecorder()

    result = pymoo_backend.run_backend(
        config={
            "optimize": {
                "backend": "pymoo",
                "population_size": 6,
                "iters": 12,
                "n_cpus": 1,
                "round_to_n_significant_digits": 4,
                "scoring": ["adg", "drawdown_worst"],
                "pymoo": {
                    "algorithm": "nsga3",
                    "shared": {
                        "crossover_prob_var": 0.5,
                        "crossover_eta": 20.0,
                        "mutation_eta": 20.0,
                        "mutation_prob_var": 0.5,
                        "eliminate_duplicates": True,
                    },
                },
            }
        },
        evaluator=evaluator,
        evaluator_for_pool=evaluator,
        recorder=recorder,
        overrides_list=[],
        duplicate_counter={},
        starting_configs_path=None,
        constraint_fitness_cls=None,
        ignore_sigint_in_worker=_ignore_sigint,
        get_starting_configs=_get_starting_configs,
        configs_to_individuals=_configs_to_individuals,
        record_individual_result=None,
        run_evolution=None,
        build_config_fn=_build_config,
        overrides_fn=_noop_overrides,
    )

    assert result["pool_terminated"] is False
    assert result["pool"].terminated is False
    assert evaluator.use_duplicate_guard is False
    assert recorder.entries
    assert "metrics" in recorder.entries[0]
    assert "objectives" in recorder.entries[0]["metrics"]


def test_run_backend_supports_single_objective(monkeypatch):
    monkeypatch.setattr(pymoo_backend.multiprocessing, "Pool", FakePool)
    evaluator = FakeEvaluator(["adg"])
    recorder = FakeRecorder()

    result = pymoo_backend.run_backend(
        config={
            "optimize": {
                "backend": "pymoo",
                "population_size": 4,
                "iters": 8,
                "n_cpus": 1,
                "round_to_n_significant_digits": 4,
                "scoring": ["adg"],
                "pymoo": {
                    "algorithm": "nsga3",
                    "shared": {
                        "crossover_prob_var": 0.5,
                        "crossover_eta": 20.0,
                        "mutation_eta": 20.0,
                        "mutation_prob_var": 0.5,
                        "eliminate_duplicates": True,
                    },
                },
            }
        },
        evaluator=evaluator,
        evaluator_for_pool=evaluator,
        recorder=recorder,
        overrides_list=[],
        duplicate_counter={},
        starting_configs_path=None,
        constraint_fitness_cls=None,
        ignore_sigint_in_worker=_ignore_sigint,
        get_starting_configs=_get_starting_configs,
        configs_to_individuals=_configs_to_individuals,
        record_individual_result=None,
        run_evolution=None,
        build_config_fn=_build_config,
        overrides_fn=_noop_overrides,
    )

    assert result["pool_terminated"] is False
    assert recorder.entries
    assert list(recorder.entries[0]["metrics"]["objectives"]) == ["w_0"]


def test_run_backend_evaluates_all_starting_configs_before_trim(monkeypatch):
    monkeypatch.setattr(pymoo_backend.multiprocessing, "Pool", FakePool)
    captured = {}

    def _fake_minimize(problem, algorithm, termination, seed, verbose):
        del problem, termination, seed, verbose
        captured["sampling"] = np.asarray(algorithm.initialization.sampling, dtype=np.float64)
        return None

    monkeypatch.setattr(pymoo_backend, "pymoo_minimize", _fake_minimize)
    evaluator = FakeEvaluator(["adg", "drawdown_worst"])
    recorder = FakeRecorder()

    result = pymoo_backend.run_backend(
        config={
            "optimize": {
                "backend": "pymoo",
                "population_size": 4,
                "iters": 8,
                "n_cpus": 1,
                "max_pending_starting_evals_per_cpu": 1,
                "round_to_n_significant_digits": 4,
                "scoring": ["adg", "drawdown_worst"],
                "pymoo": {
                    "algorithm": "nsga2",
                    "shared": {
                        "crossover_prob_var": 0.5,
                        "crossover_eta": 20.0,
                        "mutation_eta": 20.0,
                        "mutation_prob_var": 0.5,
                        "eliminate_duplicates": True,
                    },
                },
            }
        },
        evaluator=evaluator,
        evaluator_for_pool=evaluator,
        recorder=recorder,
        overrides_list=[],
        duplicate_counter={},
        starting_configs_path="dummy",
        constraint_fitness_cls=None,
        ignore_sigint_in_worker=_ignore_sigint,
        get_starting_configs=_many_starting_configs,
        configs_to_individuals=_many_starting_individuals,
        record_individual_result=None,
        run_evolution=None,
        build_config_fn=_build_config,
        overrides_fn=_noop_overrides,
    )

    assert result["pool_terminated"] is False
    assert len(recorder.entries) == 6
    assert captured["sampling"].shape == (4, 2)


def test_starting_payloads_are_slimmed_after_recording(monkeypatch):
    monkeypatch.setattr(pymoo_backend.multiprocessing, "Pool", FakePool)
    evaluator = FakeEvaluator(["adg", "drawdown_worst"])
    recorder = FakeRecorder()
    runner = pymoo_backend.PymooAsyncRecordingRunner(
        evaluator=evaluator,
        has_constraints=False,
        n_obj=2,
        pool=FakePool(processes=1),
        recorder=recorder,
        template=evaluator.config,
        build_config_fn=_build_config,
        overrides_fn=_noop_overrides,
        overrides_list=[],
    )

    payloads = pymoo_backend._evaluate_starting_individuals(
        starting_individuals=_many_starting_individuals(None, None, None),
        config={"optimize": {"n_cpus": 1, "max_pending_starting_evals_per_cpu": 1}},
        evaluator=evaluator,
        overrides_list=[],
        runner=runner,
        n_obj=2,
        has_constraints=False,
    )

    assert len(recorder.entries) == 6
    assert payloads
    assert all(set(payload) == {"F"} for payload in payloads)

    pop = pymoo_backend._population_from_payloads(
        np.asarray(_many_starting_individuals(None, None, None), dtype=np.float64),
        payloads,
        has_constraints=False,
    )
    assert all(not individual.data for individual in pop)


def test_run_backend_writes_readable_result_artifacts(monkeypatch, tmp_path):
    monkeypatch.setattr(pymoo_backend.multiprocessing, "Pool", FakePool)
    evaluator = FakeEvaluator(["adg", "drawdown_worst"])
    recorder = ResultRecorder(
        results_dir=str(tmp_path),
        sig_digits=4,
        flush_interval=60,
        scoring_keys=["adg", "drawdown_worst"],
        compress=False,
        write_all_results=True,
        bounds=evaluator.bounds,
    )

    result = pymoo_backend.run_backend(
        config={
            "optimize": {
                "backend": "pymoo",
                "population_size": 6,
                "iters": 12,
                "n_cpus": 1,
                "round_to_n_significant_digits": 4,
                "scoring": ["adg", "drawdown_worst"],
                "pymoo": {
                    "algorithm": "nsga3",
                    "shared": {
                        "crossover_prob_var": 0.5,
                        "crossover_eta": 20.0,
                        "mutation_eta": 20.0,
                        "mutation_prob_var": 0.5,
                        "eliminate_duplicates": True,
                    },
                },
            }
        },
        evaluator=evaluator,
        evaluator_for_pool=evaluator,
        recorder=recorder,
        overrides_list=[],
        duplicate_counter={},
        starting_configs_path=None,
        constraint_fitness_cls=None,
        ignore_sigint_in_worker=_ignore_sigint,
        get_starting_configs=_get_starting_configs,
        configs_to_individuals=_configs_to_individuals,
        record_individual_result=None,
        run_evolution=None,
        build_config_fn=_build_config,
        overrides_fn=_noop_overrides,
    )
    recorder.flush()
    recorder.close()
    result["pool"].close()
    result["pool"].join()

    records = list(load_results(tmp_path / "all_results.bin"))
    assert records
    run_data = load_pareto_dataframe(str(tmp_path))
    assert not run_data.dataframe.empty
