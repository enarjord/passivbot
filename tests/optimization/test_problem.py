import numpy as np
from multiprocessing.reduction import ForkingPickler
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


class PayloadEvaluator:
    limit_checks = []

    def evaluate(self, vector, overrides_list):
        return {
            "fitness": (-2.0,),
            "constraint_violation": 0.0,
            "metrics": {"objectives": {"w_0": -2.0}},
            "evaluation_vector": [0.42, float(vector[1])],
        }


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


class PickleCheckingPool(FakeAsyncPool):
    def apply_async(self, fn, args=()):
        ForkingPickler.dumps((fn, args))
        return super().apply_async(fn, args)


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


def test_problem_accepts_shared_evaluation_payload_vector():
    adapter = PymooEvaluatorAdapter(PayloadEvaluator())
    problem = PassivbotProblem(
        bounds=[Bound(0.0, 1.0), Bound(0.0, 2.0)],
        scoring_keys=["adg"],
        evaluator_adapter=adapter,
    )

    out = {}
    problem._evaluate(np.asarray([0.7, 1.1]), out)

    assert out["F"].tolist() == [-2.0]
    assert out["evaluation_vector"].tolist() == [0.42, 1.1]


def test_async_recording_runner_records_and_strips_metrics():
    evaluator = FakeEvaluator(has_constraints=True)
    recorder = MagicMock()
    runner = PymooAsyncRecordingRunner(
        evaluator=evaluator,
        has_constraints=True,
        n_obj=2,
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

    results = runner(object(), [np.asarray([0.25]), np.asarray([0.5])])

    assert len(results) == 2
    assert "metrics" not in results[0]
    assert "evaluation_vector" not in results[0]
    assert recorder.record.call_count == 2


def test_async_recording_runner_does_not_reseed_parent_numpy_rng():
    evaluator = FakeEvaluator(has_constraints=True)
    recorder = MagicMock()
    runner = PymooAsyncRecordingRunner(
        evaluator=evaluator,
        has_constraints=True,
        n_obj=2,
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
    np.random.seed(42)
    before = np.random.get_state()

    runner(object(), [np.asarray([0.25])])

    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_async_recording_runner_profiles_record_result_when_enabled(monkeypatch, caplog):
    evaluator = FakeEvaluator(has_constraints=True)
    recorder = MagicMock()
    runner = PymooAsyncRecordingRunner(
        evaluator=evaluator,
        has_constraints=True,
        n_obj=2,
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

    monkeypatch.setenv("PASSIVBOT_OPTIMIZE_PROFILE", "1")
    with caplog.at_level("INFO"):
        results = runner(object(), [np.asarray([0.25])])

    assert len(results) == 1
    assert recorder.record.call_count == 1
    assert any("[opt-profile] record_result_ms=" in record.message for record in caplog.records)


def test_async_recording_runner_uses_picklable_worker_target():
    evaluator = FakeEvaluator(has_constraints=True)
    recorder = MagicMock()
    runner = PymooAsyncRecordingRunner(
        evaluator=evaluator,
        has_constraints=True,
        n_obj=2,
        pool=PickleCheckingPool(),
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

    results = runner(lambda x: x, [np.asarray([0.25, 0.5])])

    assert len(results) == 1
    assert results[0]["F"].tolist() == [-1.5, 0.2]
    assert results[0]["G"].tolist() == [0.3]
