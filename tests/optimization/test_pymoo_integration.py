import math
import pickle

import numpy as np
from unittest.mock import MagicMock
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.termination import get_termination
from pymoo.util.ref_dirs import get_reference_directions

from optimization.problem import PassivbotProblem
from optimize import compute_n_partitions
from optimization.repair import SignificantDigitsRepair


class TestPymooIntegration:
    def test_minimize_with_mock_evaluator(self):
        """Full pymoo loop with a mock evaluator returning dummy objectives."""
        config = {
            "bot": {
                "long": {"a": 0.5, "b": 0.5},
            },
            "optimize": {
                "bounds": {
                    "long_a": [0.0, 1.0],
                    "long_b": [0.0, 1.0],
                },
                "scoring": ["obj1", "obj2"],
                "limits": [],
            },
        }

        def mock_evaluate(individual, overrides_list):
            x = np.array(individual)
            obj1 = float(np.sum(x ** 2))
            obj2 = float(np.sum((x - 1) ** 2))
            return {
                "stats": {
                    "obj1": {"mean": obj1, "min": obj1, "max": obj1, "std": 0},
                    "obj2": {"mean": obj2, "min": obj2, "max": obj2, "std": 0},
                },
            }

        evaluator = MagicMock()
        evaluator.evaluate = mock_evaluate

        problem = PassivbotProblem(config, evaluator)

        ref_dirs = get_reference_directions(
            "das-dennis", 2, n_partitions=compute_n_partitions(2, 10)
        )
        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=10,
            crossover=SBX(eta=20, prob=0.7),
            mutation=PM(eta=20, prob=0.3),
            repair=SignificantDigitsRepair(3),
        )

        res = minimize(
            problem,
            algorithm,
            get_termination("n_gen", 5),
            verbose=False,
            seed=1,
        )

        assert res.F is not None
        assert res.F.shape[1] == 2
        assert len(res.X) > 0
        # All solutions within bounds
        assert np.all(res.X >= 0.0)
        assert np.all(res.X <= 1.0)

    def test_with_constraints(self):
        """Verify pymoo constraint handling works with our limit mapping."""
        config = {
            "bot": {"long": {"x": 0.5, "y": 0.5}},
            "optimize": {
                "bounds": {"long_x": [0.0, 1.0], "long_y": [0.0, 1.0]},
                "scoring": ["obj1", "obj2"],
                "limits": [
                    {"metric": "obj1", "penalize_if": "greater_than", "value": 0.5},
                ],
            },
        }

        def mock_evaluate(individual, overrides_list):
            obj1 = float(sum(x**2 for x in individual))
            obj2 = float(sum((x - 1) ** 2 for x in individual))
            return {
                "stats": {
                    "obj1": {"mean": obj1, "min": obj1, "max": obj1, "std": 0},
                    "obj2": {"mean": obj2, "min": obj2, "max": obj2, "std": 0},
                },
            }

        evaluator = MagicMock()
        evaluator.evaluate = mock_evaluate

        problem = PassivbotProblem(config, evaluator)
        assert problem.n_ieq_constr == 1

        ref_dirs = get_reference_directions(
            "das-dennis", 2, n_partitions=compute_n_partitions(2, 20)
        )
        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=20,
            crossover=SBX(eta=20, prob=0.7),
            mutation=PM(eta=20, prob=0.3),
            repair=SignificantDigitsRepair(3),
        )

        res = minimize(
            problem,
            algorithm,
            get_termination("n_gen", 10),
            verbose=False,
            seed=1,
        )

        assert res.F is not None

    def test_with_seeding(self):
        """Verify X_init seeding works."""
        config = {
            "bot": {"long": {"a": 0.5, "b": 0.5}},
            "optimize": {
                "bounds": {"long_a": [0.0, 1.0], "long_b": [0.0, 1.0]},
                "scoring": ["obj1", "obj2"],
                "limits": [],
            },
        }

        def mock_evaluate(individual, overrides_list):
            x = np.array(individual)
            obj1 = float(np.sum(x ** 2))
            obj2 = float(np.sum((x - 1) ** 2))
            return {
                "stats": {
                    "obj1": {"mean": obj1, "min": obj1, "max": obj1, "std": 0},
                    "obj2": {"mean": obj2, "min": obj2, "max": obj2, "std": 0},
                },
            }

        evaluator = MagicMock()
        evaluator.evaluate = mock_evaluate

        problem = PassivbotProblem(config, evaluator)

        # Seed with known values (2 decision variables)
        X_init = np.array([[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5], [0.5, 0.6]])

        ref_dirs = get_reference_directions(
            "das-dennis", 2, n_partitions=compute_n_partitions(2, 5)
        )
        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=5,
            sampling=X_init,
            crossover=SBX(eta=20, prob=0.7),
            mutation=PM(eta=20, prob=0.3),
            repair=SignificantDigitsRepair(3),
        )

        res = minimize(
            problem,
            algorithm,
            get_termination("n_gen", 3),
            verbose=False,
            seed=1,
        )

        assert res.F is not None
        assert np.all(res.X >= 0.0)
        assert np.all(res.X <= 1.0)

    def test_with_starmap_parallelization(self):
        """Verify StarmapParallelization runner works with ElementwiseProblem."""
        from multiprocessing.pool import ThreadPool
        from pymoo.parallelization.starmap import StarmapParallelization

        config = {
            "bot": {"long": {"a": 0.5, "b": 0.5}},
            "optimize": {
                "bounds": {"long_a": [0.0, 1.0], "long_b": [0.0, 1.0]},
                "scoring": ["obj1", "obj2"],
                "limits": [],
            },
        }

        def mock_evaluate(individual, overrides_list):
            x = np.array(individual)
            obj1 = float(np.sum(x ** 2))
            obj2 = float(np.sum((x - 1) ** 2))
            return {
                "stats": {
                    "obj1": {"mean": obj1, "min": obj1, "max": obj1, "std": 0},
                    "obj2": {"mean": obj2, "min": obj2, "max": obj2, "std": 0},
                },
            }

        evaluator = MagicMock()
        evaluator.evaluate = mock_evaluate

        pool = ThreadPool(2)
        runner = StarmapParallelization(pool.starmap)
        problem = PassivbotProblem(config, evaluator, elementwise_runner=runner)

        ref_dirs = get_reference_directions(
            "das-dennis", 2, n_partitions=compute_n_partitions(2, 10)
        )
        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=10,
            crossover=SBX(eta=20, prob=0.7),
            mutation=PM(eta=20, prob=0.3),
            repair=SignificantDigitsRepair(3),
        )

        res = minimize(
            problem,
            algorithm,
            get_termination("n_gen", 5),
            verbose=False,
            seed=1,
        )

        pool.close()
        pool.join()

        assert res.F is not None
        assert res.F.shape[1] == 2
        assert len(res.X) > 0

    def test_metrics_passthrough_to_individuals(self):
        """Verify out['metrics'] propagates to ind.data for ParetoWriterCallback."""
        config = {
            "bot": {"long": {"a": 0.5, "b": 0.5}},
            "optimize": {
                "bounds": {"long_a": [0.0, 1.0], "long_b": [0.0, 1.0]},
                "scoring": ["obj1"],
                "limits": [],
            },
        }

        def mock_evaluate(individual, overrides_list):
            x = np.array(individual)
            obj1 = float(np.sum(x ** 2))
            return {
                "stats": {
                    "obj1": {"mean": obj1, "min": obj1, "max": obj1, "std": 0},
                },
                "custom_key": "test_value",
            }

        evaluator = MagicMock()
        evaluator.evaluate = mock_evaluate

        problem = PassivbotProblem(config, evaluator)

        ref_dirs = get_reference_directions(
            "das-dennis", 1, n_partitions=1
        )
        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=4,
            crossover=SBX(eta=20, prob=0.7),
            mutation=PM(eta=20, prob=0.3),
        )

        res = minimize(
            problem,
            algorithm,
            get_termination("n_gen", 2),
            verbose=False,
            seed=1,
        )

        # Check that metrics dict is accessible on optimal individuals
        for ind in res.opt:
            metrics = ind.data.get("metrics")
            assert metrics is not None, "metrics not found on individual"
            assert "stats" in metrics
            assert "custom_key" in metrics

    def test_sig_digits_repair_in_pipeline(self):
        """Verify SignificantDigitsRepair actually rounds during optimization."""
        config = {
            "bot": {"long": {"a": 0.5, "b": 0.5}},
            "optimize": {
                "bounds": {"long_a": [0.001, 1.0], "long_b": [0.001, 1.0]},
                "scoring": ["obj1", "obj2"],
                "limits": [],
            },
        }

        evaluated_values = []

        def mock_evaluate(individual, overrides_list):
            evaluated_values.append(individual[0])
            x = np.array(individual)
            obj1 = float(np.sum(x ** 2))
            obj2 = float(np.sum((x - 1) ** 2))
            return {
                "stats": {
                    "obj1": {"mean": obj1, "min": obj1, "max": obj1, "std": 0},
                    "obj2": {"mean": obj2, "min": obj2, "max": obj2, "std": 0},
                },
            }

        evaluator = MagicMock()
        evaluator.evaluate = mock_evaluate

        problem = PassivbotProblem(config, evaluator)

        ref_dirs = get_reference_directions(
            "das-dennis", 2, n_partitions=compute_n_partitions(2, 10)
        )
        algorithm = NSGA3(
            ref_dirs=ref_dirs,
            pop_size=10,
            crossover=SBX(eta=20, prob=0.9),
            mutation=PM(eta=20, prob=0.9),
            repair=SignificantDigitsRepair(3),
        )

        res = minimize(
            problem,
            algorithm,
            get_termination("n_gen", 3),
            verbose=False,
            seed=42,
        )

        # Check that evaluated values are rounded to 3 sig digits
        for v in evaluated_values:
            if v > 0:
                digits = -int(math.floor(math.log10(abs(v)))) + 2
                rounded = round(v, digits)
                assert abs(v - rounded) < 1e-10, f"Value {v} not rounded to 3 sig digits"


def test_suite_evaluator_del_logs_close_errors(caplog):
    """__del__ must log errors instead of silently swallowing them."""
    import logging

    from optimization.evaluator import SuiteEvaluator, Evaluator
    from optimize_suite import ScenarioEvalContext

    class BadAttachment:
        def close(self):
            raise RuntimeError("shm close failed")

    evaluator = Evaluator.__new__(Evaluator)
    evaluator.hlcvs_specs = {}
    evaluator.btc_usd_specs = {}
    evaluator.msss = {}
    evaluator.timestamps = {}
    evaluator.exchanges = []
    evaluator.shared_array_manager = None
    evaluator.shared_hlcvs_np = {}
    evaluator.shared_btc_np = {}
    evaluator._attachments = {"hlcvs": {}, "btc": {}}
    evaluator.config = {}

    ctx = ScenarioEvalContext.__new__(ScenarioEvalContext)
    ctx.attachments = {"hlcvs": {"test": BadAttachment()}, "btc": {}}

    suite_eval = SuiteEvaluator(
        base_evaluator=evaluator,
        scenario_contexts=[ctx],
        aggregate_cfg={},
    )

    with caplog.at_level(logging.ERROR):
        suite_eval.__del__()

    assert "shm close failed" in caplog.text


def test_suite_evaluator_pickle_strips_shared_memory():
    """SuiteEvaluator must survive pickle roundtrip (StarmapParallelization)."""
    from optimization.evaluator import SuiteEvaluator, Evaluator

    evaluator = Evaluator.__new__(Evaluator)
    evaluator.hlcvs_specs = {}
    evaluator.btc_usd_specs = {}
    evaluator.msss = {}
    evaluator.timestamps = {}
    evaluator.exchanges = []
    evaluator.shared_array_manager = None
    evaluator.shared_hlcvs_np = {}
    evaluator.shared_btc_np = {}
    evaluator._attachments = {"hlcvs": {}, "btc": {}}
    evaluator.config = {}

    suite_eval = SuiteEvaluator(
        base_evaluator=evaluator,
        scenario_contexts=[],
        aggregate_cfg={},
    )
    suite_eval._master_attachments = {"hlcvs": {"fake": object()}, "btc": {}}
    suite_eval._master_arrays = {"hlcvs": {"fake": object()}, "btc": {}}

    data = pickle.dumps(suite_eval)
    restored = pickle.loads(data)

    assert restored._master_attachments == {"hlcvs": {}, "btc": {}}
    assert restored._master_arrays == {"hlcvs": {}, "btc": {}}
    assert restored.contexts == []
    assert restored.aggregate_cfg == {}
