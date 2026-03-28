import numpy as np
import pytest

from optimization.bounds import Bound
from optimization.problem import PassivbotProblem, PymooEvaluatorAdapter


class FakeEvaluator:
    def __init__(self, has_constraints=True):
        self.limit_checks = [object()] if has_constraints else []
        self.calls = []

    def evaluate(self, vector, overrides_list):
        self.calls.append((list(vector), list(overrides_list)))
        vector[0] = 0.25
        return (
            (-1.5, 0.2),
            0.3,
            {
                "objectives": {"w_0": -1.5, "w_1": 0.2},
                "constraint_violation": 0.3,
                "stats": {"adg": {"mean": 1.5}},
            },
        )


class SingleObjectiveEvaluator:
    def __init__(self):
        self.limit_checks = []

    def evaluate(self, vector, overrides_list):
        return (
            (-1.5,),
            0.0,
            {
                "objectives": {"w_0": -1.5},
                "constraint_violation": 0.0,
            },
        )


class InvalidMetricsEvaluator:
    def __init__(self):
        self.limit_checks = []

    def evaluate(self, vector, overrides_list):
        return ((-1.0,), 0.0, None)


def test_problem_evaluate_passthroughs_metrics_and_vector():
    evaluator = FakeEvaluator(has_constraints=True)
    adapter = PymooEvaluatorAdapter(evaluator, overrides_list=["foo"])
    problem = PassivbotProblem(
        bounds=[Bound(0.0, 1.0), Bound(0.0, 2.0)],
        scoring_keys=["adg", "drawdown_worst"],
        evaluator_adapter=adapter,
    )

    out = {}
    problem._evaluate(np.asarray([0.7, 1.1]), out)

    assert out["F"].tolist() == [-1.5, 0.2]
    assert out["G"].tolist() == [0.3]
    assert out["metrics"]["constraint_violation"] == 0.3
    assert out["evaluation_vector"].tolist() == [0.25, 1.1]
    assert evaluator.calls == [([0.7, 1.1], ["foo"])]


def test_problem_without_constraints_omits_g():
    evaluator = SingleObjectiveEvaluator()
    adapter = PymooEvaluatorAdapter(evaluator)
    problem = PassivbotProblem(
        bounds=[Bound(0.0, 1.0)],
        scoring_keys=["adg"],
        evaluator_adapter=adapter,
    )

    out = {}
    problem._evaluate(np.asarray([0.7]), out)

    assert "G" not in out
    assert out["F"].tolist() == [-1.5]


def test_problem_rejects_missing_metrics_dict():
    evaluator = InvalidMetricsEvaluator()
    adapter = PymooEvaluatorAdapter(evaluator)
    problem = PassivbotProblem(
        bounds=[Bound(0.0, 1.0)],
        scoring_keys=["adg"],
        evaluator_adapter=adapter,
    )

    with pytest.raises(TypeError, match="metrics dict"):
        problem._evaluate(np.asarray([0.7]), {})
