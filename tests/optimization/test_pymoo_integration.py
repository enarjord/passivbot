import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from optimization.bounds import Bound
from optimization.problem import PassivbotProblem, PymooEvaluatorAdapter
from optimization.repair import BoundsRepair


class QuadraticEvaluator:
    def __init__(self):
        self.limit_checks = []

    def evaluate(self, vector, overrides_list):
        x = np.asarray(vector, dtype=np.float64)
        return (
            (-float(np.sum((x - 1.0) ** 2)), float(np.sum(x**2))),
            0.0,
            {
                "objectives": {
                    "w_0": -float(np.sum((x - 1.0) ** 2)),
                    "w_1": float(np.sum(x**2)),
                },
                "constraint_violation": 0.0,
            },
        )


def test_pymoo_nsga2_runs_with_passivbot_problem():
    problem = PassivbotProblem(
        bounds=[Bound(0.0, 1.0, 0.1), Bound(0.0, 1.0, 0.1)],
        scoring_keys=["obj1", "obj2"],
        evaluator_adapter=PymooEvaluatorAdapter(QuadraticEvaluator()),
    )
    algorithm = NSGA2(
        pop_size=8,
        sampling=np.asarray([[0.1, 0.2]] * 8, dtype=np.float64),
        repair=BoundsRepair(problem.bounds, sig_digits=4),
        eliminate_duplicates=True,
    )

    result = minimize(
        problem,
        algorithm,
        get_termination("n_gen", 3),
        seed=1,
        verbose=False,
    )

    assert result.F is not None
    assert result.F.shape[1] == 2
    assert np.all(result.X >= 0.0)
    assert np.all(result.X <= 1.0)
    stepped = np.round(result.X / 0.1) * 0.1
    assert np.allclose(result.X, stepped)
