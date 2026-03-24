import numpy as np
import pytest
from unittest.mock import MagicMock, patch
from optimization.problem import PassivbotProblem


class TestPassivbotProblemInit:
    def _make_config(self):
        return {
            "bot": {
                "long": {"ema_span_0": 200.0, "n_positions": 5},
                "short": {"ema_span_0": 100.0, "n_positions": 3},
            },
            "optimize": {
                "bounds": {
                    "long_ema_span_0": [200, 1440],
                    "long_n_positions": [1, 20],
                    "short_ema_span_0": [100, 1440],
                    "short_n_positions": [1, 10],
                },
                "scoring": ["adg_pnl", "drawdown_worst"],
                "limits": [
                    {"metric": "drawdown_worst", "penalize_if": "greater_than", "value": 0.85},
                ],
            },
        }

    def test_bounds_setup(self):
        config = self._make_config()
        problem = PassivbotProblem(config, evaluator=MagicMock())
        assert problem.n_var == 4
        assert problem.n_obj == 2
        assert problem.n_ieq_constr == 1
        np.testing.assert_array_equal(problem.xl, [200, 1, 100, 1])
        np.testing.assert_array_equal(problem.xu, [1440, 20, 1440, 10])

    def test_keys_order(self):
        config = self._make_config()
        problem = PassivbotProblem(config, evaluator=MagicMock())
        assert problem.keys == [
            "long_ema_span_0",
            "long_n_positions",
            "short_ema_span_0",
            "short_n_positions",
        ]

    def test_fixed_bounds(self):
        config = {
            "bot": {"long": {"a": 1.0}},
            "optimize": {
                "bounds": {"long_a": [5.0, 5.0]},
                "scoring": ["adg_pnl"],
                "limits": [],
            },
        }
        problem = PassivbotProblem(config, evaluator=MagicMock())
        assert problem.xl[0] == problem.xu[0] == 5.0

    def test_evaluate_sequential(self):
        """Test _evaluate with elementwise interface (single individual)."""
        config = {
            "bot": {"long": {"a": 0.5, "b": 0.5}},
            "optimize": {
                "bounds": {"long_a": [0, 1], "long_b": [0, 1]},
                "scoring": ["obj1", "obj2"],
                "limits": [],
            },
        }

        def mock_evaluate(individual, overrides_list):
            return {
                "stats": {
                    "obj1": {"mean": individual[0], "min": 0, "max": 1, "std": 0},
                    "obj2": {"mean": individual[1], "min": 0, "max": 1, "std": 0},
                },
            }

        evaluator = MagicMock()
        evaluator.evaluate = mock_evaluate

        problem = PassivbotProblem(config, evaluator)
        x = np.array([0.3, 0.7])
        out = {}
        problem._evaluate(x, out)

        assert isinstance(out["F"], list)
        assert len(out["F"]) == 2

    def test_evaluate_propagates_metrics(self):
        """Test that _evaluate populates out['metrics'] for callback access."""
        config = {
            "bot": {"long": {"a": 0.5, "b": 0.5}},
            "optimize": {
                "bounds": {"long_a": [0, 1], "long_b": [0, 1]},
                "scoring": ["obj1", "obj2"],
                "limits": [],
            },
        }

        metrics_payload = {"stats": {"adg": {"mean": 0.01}}}

        def mock_evaluate(individual, overrides_list):
            return metrics_payload

        evaluator = MagicMock()
        evaluator.evaluate = mock_evaluate

        problem = PassivbotProblem(config, evaluator)
        x = np.array([0.3, 0.7])
        out = {}
        problem._evaluate(x, out)

        assert "metrics" in out
        assert out["metrics"] == metrics_payload

    def test_evaluate_failed_individual_gets_empty_metrics(self):
        """Test that failed evaluations get empty metrics, not None."""
        config = {
            "bot": {"long": {"a": 0.5}},
            "optimize": {
                "bounds": {"long_a": [0, 1]},
                "scoring": ["obj1"],
                "limits": [],
            },
        }

        def mock_evaluate(individual, overrides_list):
            raise RuntimeError("backtest crashed")

        evaluator = MagicMock()
        evaluator.evaluate = mock_evaluate

        problem = PassivbotProblem(config, evaluator)
        x = np.array([0.3])
        out = {}
        problem._evaluate(x, out)

        assert out["metrics"] == {}
        assert out["F"] == [1e18]

    def test_evaluate_successful_individual_gets_metrics(self):
        """Test that successful evaluations get metrics populated."""
        config = {
            "bot": {"long": {"a": 0.5}},
            "optimize": {
                "bounds": {"long_a": [0, 1]},
                "scoring": ["obj1"],
                "limits": [],
            },
        }

        metrics_payload = {"stats": {"adg": {"mean": 0.02}}}

        def mock_evaluate(individual, overrides_list):
            return metrics_payload

        evaluator = MagicMock()
        evaluator.evaluate = mock_evaluate

        problem = PassivbotProblem(config, evaluator)
        x = np.array([0.7])
        out = {}
        problem._evaluate(x, out)

        assert out["metrics"] == metrics_payload

    def test_no_constraints(self):
        config = {
            "bot": {"long": {"a": 0.5}},
            "optimize": {
                "bounds": {"long_a": [0, 1]},
                "scoring": ["obj1"],
                "limits": [],
            },
        }
        evaluator = MagicMock()
        evaluator.evaluate = lambda ind, ovr: {"stats": {}}

        problem = PassivbotProblem(config, evaluator)
        assert problem.n_ieq_constr == 0

        x = np.array([0.5])
        out = {}
        problem._evaluate(x, out)
        assert "G" not in out


class TestProblemScoring:
    """Test that PassivbotProblem resolves metrics and applies sign-flip."""

    def test_maximize_metric_is_negated(self):
        """Metrics like adg (higher=better) should be negated for pymoo minimization."""
        config = {
            "bot": {"long": {"a": 0.5}},
            "optimize": {
                "bounds": {"long_a": [0, 1]},
                "scoring": ["adg"],
                "limits": [],
            },
        }

        def mock_evaluate(individual, overrides_list):
            return {"stats": {"adg": {"mean": 0.005, "min": 0.001, "max": 0.01, "std": 0.002}}}

        evaluator = MagicMock()
        evaluator.evaluate = mock_evaluate

        problem = PassivbotProblem(config, evaluator)
        x = np.array([0.5])
        out = {}
        problem._evaluate(x, out)

        assert out["F"][0] < 0

    def test_minimize_metric_is_positive(self):
        """Metrics like drawdown_worst (lower=better) should stay positive."""
        config = {
            "bot": {"long": {"a": 0.5}},
            "optimize": {
                "bounds": {"long_a": [0, 1]},
                "scoring": ["drawdown_worst"],
                "limits": [],
            },
        }

        def mock_evaluate(individual, overrides_list):
            return {"stats": {"drawdown_worst": {"mean": 0.3, "min": 0.1, "max": 0.5, "std": 0.1}}}

        evaluator = MagicMock()
        evaluator.evaluate = mock_evaluate

        problem = PassivbotProblem(config, evaluator)
        x = np.array([0.5])
        out = {}
        problem._evaluate(x, out)

        assert out["F"][0] > 0

    def test_missing_metric_defaults_to_zero(self):
        config = {
            "bot": {"long": {"a": 0.5}},
            "optimize": {
                "bounds": {"long_a": [0, 1]},
                "scoring": ["nonexistent_metric"],
                "limits": [],
            },
        }

        def mock_evaluate(individual, overrides_list):
            return {"stats": {}}

        evaluator = MagicMock()
        evaluator.evaluate = mock_evaluate

        problem = PassivbotProblem(config, evaluator)
        x = np.array([0.5])
        out = {}
        problem._evaluate(x, out)

        assert out["F"][0] == 0.0

    def test_currency_suffix_resolution(self):
        """scoring key 'adg' should resolve to 'adg_usd_mean' in flat_stats."""
        config = {
            "bot": {"long": {"a": 0.5}},
            "optimize": {
                "bounds": {"long_a": [0, 1]},
                "scoring": ["adg"],
                "limits": [],
            },
        }

        def mock_evaluate(individual, overrides_list):
            return {"stats": {"adg_usd": {"mean": 0.003, "min": 0.001, "max": 0.005, "std": 0.001}}}

        evaluator = MagicMock()
        evaluator.evaluate = mock_evaluate

        problem = PassivbotProblem(config, evaluator)
        x = np.array([0.5])
        out = {}
        problem._evaluate(x, out)

        assert out["F"][0] == pytest.approx(-0.003)

    def test_suite_evaluator_flat_stats_override(self):
        """SuiteEvaluator returns flat_stats with _mean overrides; Problem should use them."""
        config = {
            "bot": {"long": {"a": 0.5}},
            "optimize": {
                "bounds": {"long_a": [0, 1]},
                "scoring": ["adg"],
                "limits": [],
            },
        }

        def mock_evaluate(individual, overrides_list):
            return {
                "stats": {"adg": {"mean": 0.005}},
                "flat_stats_override": {"adg_mean": 0.008},
            }

        evaluator = MagicMock()
        evaluator.evaluate = mock_evaluate

        problem = PassivbotProblem(config, evaluator)
        x = np.array([0.5])
        out = {}
        problem._evaluate(x, out)

        assert out["F"][0] == pytest.approx(-0.008)
