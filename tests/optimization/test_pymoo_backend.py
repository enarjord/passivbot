import copy

from optimization.bounds import Bound
from optimization.backends import pymoo_backend


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


def _ignore_sigint():
    return None


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
                "crossover_probability": 0.7,
                "crossover_eta": 20.0,
                "mutation_eta": 20.0,
                "mutation_indpb": 0.5,
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
        overrides_fn=object(),
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
                "crossover_probability": 0.7,
                "crossover_eta": 20.0,
                "mutation_eta": 20.0,
                "mutation_indpb": 0.5,
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
        overrides_fn=object(),
    )

    assert result["pool_terminated"] is False
    assert recorder.entries
    assert list(recorder.entries[0]["metrics"]["objectives"]) == ["w_0"]
