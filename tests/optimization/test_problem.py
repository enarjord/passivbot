import numpy as np
from unittest.mock import MagicMock

from optimization.bounds import Bound
from optimization.problem import PassivbotProblem, PymooAsyncRecordingRunner, PymooEvaluatorAdapter


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


class FakeAsyncResult:
    def __init__(self, value):
        self._value = value

    def ready(self):
        return True

    def get(self):
        return self._value


class FakeAsyncPool:
    def apply_async(self, fn, args=()):
        return FakeAsyncResult(fn(*args))


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


def test_async_recording_runner_records_and_strips_metrics():
    recorder = MagicMock()
    runner = PymooAsyncRecordingRunner(
        pool=FakeAsyncPool(),
        recorder=recorder,
        template={"optimize": {"backend": "pymoo"}},
        build_config_fn=lambda vector, overrides_fn, overrides_list, template: {
            "bot": {"long": {"a": float(vector[0])}},
            "backtest": {"coins": {"binance": ["BTC/USDT:USDT"]}},
            **template,
        },
        overrides_fn=object(),
        overrides_list=["x"],
    )

    def fake_eval(x):
        return {
            "F": np.asarray([-float(x[0])]),
            "G": np.asarray([0.0]),
            "metrics": {
                "objectives": {"w_0": -float(x[0])},
                "constraint_violation": 0.0,
            },
            "evaluation_vector": np.asarray([float(x[0])]),
        }

    results = runner(fake_eval, [np.asarray([0.25]), np.asarray([0.5])])

    assert len(results) == 2
    assert "metrics" not in results[0]
    assert "evaluation_vector" not in results[0]
    assert recorder.record.call_count == 2
