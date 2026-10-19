from copy import deepcopy
import pickle

import numpy as np
import pytest
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.core.problem import Problem
from pymoo.optimize import minimize
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.util.ref_dirs import get_reference_directions

from optimization import pymoo_checkpoint
from optimization.backends import pymoo_backend
from optimization.bounds import Bound
from optimization.evaluation_contract import CONTRACT_CACHE_KEY, CONTRACT_VERSION


def _config():
    config = {
        "backtest": {}, "bot": {"long": {}, "short": {}},
        "optimize": {"backend": "pymoo", "bounds": {"long_n_positions": [1, 1, 1]},
                     "scoring": [{"metric": "adg_strategy_eq", "goal": "max"}],
                     "objective_scenario": None},
    }
    config[CONTRACT_CACHE_KEY] = {
        "version": CONTRACT_VERSION,
        "implementation": {"source": "fixture-source", "runtime": "fixture-runtime"},
        "prepared_data": {"sha256": "fixture-data"},
    }
    return config


@pytest.mark.parametrize("algorithm", [NSGA2(pop_size=4), NSGA3(
    pop_size=4, ref_dirs=get_reference_directions("das-dennis", 2, n_partitions=1),
    selection=TournamentSelection(func_comp=pymoo_backend._nsga3_comp_by_cv_then_random),
)])
def test_checkpoint_roundtrip_retains_algorithm_population_and_rng(tmp_path, algorithm):
    algorithm.setup(Problem(n_var=1, n_obj=2, xl=0.0, xu=1.0), seed=15)
    algorithm.pop = algorithm.ask()
    saved_rng = deepcopy(algorithm.random_state.bit_generator.state)
    path = tmp_path / "checkpoint.pkl"
    config = _config()
    pymoo_checkpoint.save_checkpoint(path, algorithm, config)

    loaded = pymoo_checkpoint.load_checkpoint(path, config)

    assert type(loaded) is type(algorithm)
    np.testing.assert_array_equal(loaded.pop.get("X"), algorithm.pop.get("X"))
    assert loaded.random_state.bit_generator.state == saved_rng
    assert not path.with_name("checkpoint.pkl.tmp").exists()


@pytest.mark.parametrize("old_state", [NSGA2(pop_size=4), {"population": [], "gen": 3},
                                        {"backend": "deap", "version": 1},
                                        {"backend": "pymoo", "version": 0}])
def test_checkpoint_rejects_old_bare_or_retired_formats(tmp_path, old_state):
    path = tmp_path / "checkpoint.pkl"
    path.write_bytes(pickle.dumps(old_state))
    with pytest.raises(ValueError, match="compatible versioned checkpoint"):
        pymoo_checkpoint.load_checkpoint(path, _config())


@pytest.mark.parametrize("section,key,value", [
    ("implementation", "source", "changed-source"),
    ("implementation", "runtime", "changed-runtime"),
    ("prepared_data", "sha256", "changed-data"),
])
def test_checkpoint_rejects_changed_evaluation_identity(tmp_path, section, key, value):
    config = _config()
    path = tmp_path / "checkpoint.pkl"
    pymoo_checkpoint.save_checkpoint(path, NSGA2(pop_size=4), config)
    config[CONTRACT_CACHE_KEY][section][key] = value
    with pytest.raises(ValueError, match="evaluation contract changed"):
        pymoo_checkpoint.load_checkpoint(path, config)


def test_checkpoint_rejects_changed_search_but_allows_new_budget_and_workers(tmp_path):
    config = _config()
    path = tmp_path / "checkpoint.pkl"
    pymoo_checkpoint.save_checkpoint(path, NSGA2(pop_size=4), config)
    config["optimize"].update(iters=100, n_cpus=1, seed=72)
    assert isinstance(pymoo_checkpoint.load_checkpoint(path, config), NSGA2)
    config["optimize"]["bounds"]["long_n_positions"] = [2, 3, 1]
    with pytest.raises(ValueError, match="critical run configuration changed"):
        pymoo_checkpoint.load_checkpoint(path, config)


def test_failed_checkpoint_preserves_previous_file_and_removes_partial(tmp_path, monkeypatch):
    config = _config()
    path = tmp_path / "checkpoint.pkl"
    pymoo_checkpoint.save_checkpoint(path, NSGA2(pop_size=4), config)
    previous = path.read_bytes()
    def interrupted_dump(state, target, **kwargs):
        target.write(b"incomplete")
        raise OSError("fixture write failure")
    monkeypatch.setattr(pymoo_checkpoint.pickle, "dump", interrupted_dump)
    with pytest.raises(OSError, match="fixture write failure"):
        pymoo_checkpoint.save_checkpoint(path, NSGA2(pop_size=4), config)
    assert path.read_bytes() == previous
    assert not path.with_name("checkpoint.pkl.tmp").exists()


def test_incompatible_resume_is_rejected_before_starting_workers(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pkl"
    path.write_bytes(pickle.dumps({"backend": "deap", "version": 1}))
    def forbidden_pool(*args, **kwargs):
        pytest.fail("incompatible resume must not start workers")
    monkeypatch.setattr(pymoo_backend.multiprocessing, "Pool", forbidden_pool)
    with pytest.raises(ValueError, match="compatible versioned checkpoint"):
        pymoo_backend.run_backend(
            config=_config(), evaluator=None, evaluator_for_pool=None, recorder=None,
            overrides_list=[], duplicate_counter={}, starting_configs_path=None,
            ignore_sigint_in_worker=None, get_starting_configs=None, configs_to_individuals=None,
            build_config_fn=None, overrides_fn=None, checkpoint_path=path, resume=True,
        )


class ResumeProblem(Problem):
    def __init__(self, n_obj):
        super().__init__(n_var=2, n_obj=n_obj, xl=0.0, xu=1.0)

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = np.column_stack([
            np.sum((x - index / self.n_obj) ** 2, axis=1)
            for index in range(self.n_obj)
        ])


@pytest.mark.parametrize("algorithm_name,n_obj", [("nsga2", 2), ("nsga3", 4)])
@pytest.mark.parametrize("final_generations", [2, 4])
def test_checkpoint_resume_matches_uninterrupted_search(tmp_path, algorithm_name, n_obj, final_generations):
    config = _config()
    config["optimize"].update(population_size=8, pymoo={"algorithm": algorithm_name})
    bounds = [Bound(0.0, 1.0), Bound(0.0, 1.0)]
    plan = pymoo_backend._resolve_pymoo_population_plan(config, n_obj=n_obj)
    def algorithm():
        return pymoo_backend._build_algorithm(
            config=config, sampling=np.random.default_rng(9).random((8, 2)),
            bounds=bounds, sig_digits=5, population_plan=plan,
        )
    whole = algorithm()
    minimize(ResumeProblem(n_obj), whole, ("n_gen", final_generations), seed=17,
             verbose=False, copy_algorithm=False)
    partial = algorithm()
    path = str(tmp_path / "checkpoint.pkl")
    minimize(ResumeProblem(n_obj), partial, ("n_gen", 2), seed=17,
             verbose=False, copy_algorithm=False,
             callback=pymoo_backend.PymooCheckpointCallback(path, config))
    resumed = pymoo_checkpoint.load_checkpoint(path, config)
    problem = ResumeProblem(n_obj)
    termination = pymoo_backend.get_termination("n_gen", final_generations)
    pymoo_backend._prepare_resumed_algorithm(
        resumed, problem=problem, termination=termination, seed=99, callback=pymoo_backend.Callback(),
    )
    minimize(problem, resumed, termination, seed=99, verbose=False, copy_algorithm=False)
    np.testing.assert_array_equal(resumed.pop.get("X"), whole.pop.get("X"))
    np.testing.assert_array_equal(resumed.pop.get("F"), whole.pop.get("F"))
    assert resumed.evaluator.n_eval == whole.evaluator.n_eval


@pytest.mark.parametrize("algorithm_name,n_obj", [("nsga2", 2), ("nsga3", 4)])
def test_supported_search_keeps_constraints_and_fixed_stepped_bounds(algorithm_name, n_obj):
    from optimization.backends.pymoo_backend import _build_algorithm, _resolve_pymoo_population_plan
    from optimization.problem import PassivbotProblem, PymooEvaluatorAdapter

    class Evaluator:
        limit_checks = ["constraint"]
        def evaluate(self, vector, overrides):
            x, fixed = vector
            return tuple((x - 0.5) ** 2 + index * fixed for index in range(n_obj)), max(0.0, 0.5 - x), {}

    bounds = [Bound(0.0, 1.0, 0.25), Bound(0.3, 0.3)]
    problem = PassivbotProblem(bounds=bounds, scoring_keys=[f"obj{n}" for n in range(n_obj)],
                               evaluator_adapter=PymooEvaluatorAdapter(Evaluator()))
    config = {"optimize": {"population_size": 8, "pymoo": {"algorithm": algorithm_name}}}
    initial = np.asarray([[0.0, 0.3], [0.25, 0.3], [0.5, 0.3], [0.75, 0.3], [1.0, 0.3]])
    algorithm = _build_algorithm(config=config, sampling=initial, bounds=bounds, sig_digits=4,
                                 population_plan=_resolve_pymoo_population_plan(config, n_obj=n_obj))
    result = minimize(problem, algorithm, ("n_gen", 3), seed=7, verbose=False, copy_algorithm=False)
    assert result.F is not None
    assert np.all(result.X[:, 0] >= 0.5)
    assert np.all(result.X[:, 1] == 0.3)
    np.testing.assert_allclose(algorithm.pop.get("X")[:, 0] / 0.25,
                               np.round(algorithm.pop.get("X")[:, 0] / 0.25))
