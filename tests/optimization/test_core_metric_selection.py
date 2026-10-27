"""Selected unavailable metrics remain infeasible without user numeric limits."""
from types import SimpleNamespace

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2, RankAndCrowding
from pymoo.core.population import Population
from pymoo.optimize import minimize

from optimization.bounds import Bound
from optimization.problem import PassivbotProblem, PymooEvaluatorAdapter

METRIC = "ulcer_index_fraction_strategy_equity"


class AvailabilityEvaluator:
    limit_checks = []
    config = {"optimize": {"prefilter": {"checks": {}}}}

    def evaluate(self, vector, overrides):
        valid = bool(vector[0])
        return {"fitness": (.75 if valid else 0.0,),
                "constraint_violation": 0.0 if valid else 1e18,
                "metrics": {"fitness_status": "valid" if valid else "invalid"},
                "evaluation_vector": vector}


def test_cpu_search_selects_valid_candidate_over_better_invalid_placeholder():
    adapter = PymooEvaluatorAdapter(AvailabilityEvaluator())
    assert adapter.has_constraints
    problem = PassivbotProblem(bounds=[Bound(0.0, 1.0)], scoring_keys=[METRIC],
                              evaluator_adapter=adapter)
    algorithm = NSGA2(pop_size=2, sampling=np.array([[0.0], [1.0]]))
    result = minimize(problem, algorithm, ("n_gen", 1), seed=7)
    assert result.X.tolist() == [1.0]
    assert result.F.tolist() == [.75]


def test_gpu_completion_selects_valid_candidate_without_numeric_limits(monkeypatch):
    from optimization.backends import gpu_backend

    base = SimpleNamespace(bounds=[Bound(0.0, 1.0)], scoring_specs=[METRIC],
                           limit_checks=[], config=AvailabilityEvaluator.config)
    problem = gpu_backend.NativeSearchProblem(base)
    assert problem.n_ieq_constr == 1
    population = Population.new("X", np.array([[0.0], [1.0]]))
    search = gpu_backend._Search.__new__(gpu_backend._Search)
    search.state = {"algorithm": SimpleNamespace(problem=problem,
        evaluator=SimpleNamespace(n_eval=0)), "population": population, "completed": 0}
    search._reuse_record_configs = False
    search.recorder = SimpleNamespace(record=lambda entry: None)
    search.template = {}
    search.build_config_fn = search.overrides_fn = None
    search.overrides_list = []
    search.checkpoint = lambda: None
    monkeypatch.setattr(gpu_backend, "build_pymoo_record_entry", lambda **kwargs: kwargs)
    for index in range(2):
        payload = AvailabilityEvaluator().evaluate([float(index)], [])
        search.apply_terminal(index, payload)
    selected = RankAndCrowding().do(problem, population, n_survive=1)
    assert selected.get("X").tolist() == [[1.0]]
    assert search.state["completed"] == 2
