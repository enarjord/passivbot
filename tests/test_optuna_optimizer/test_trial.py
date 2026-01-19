"""Tests for trial.py - parameter sampling and constraint checking."""
from unittest.mock import MagicMock

import pytest


class TestSampleParams:
    def test_fixed_bound(self):
        """Fixed bounds (low==high) still go through suggest_float for recording."""
        from optuna_optimizer.models import Bound
        from optuna_optimizer.trial import sample_params

        trial = MagicMock()
        trial.suggest_float.return_value = 0.5
        bounds = {"param": Bound(low=0.5, high=0.5)}

        result = sample_params(trial, bounds)

        assert result["param"] == 0.5
        # Fixed bounds now use suggest_float so Optuna records them in trial.params
        trial.suggest_float.assert_called_once_with("param", 0.5, 0.5)

    def test_stepped_bound(self):
        from optuna_optimizer.models import Bound
        from optuna_optimizer.trial import sample_params

        trial = MagicMock()
        trial.suggest_float.return_value = 500.0
        bounds = {"param": Bound(low=200, high=1440, step=10)}

        result = sample_params(trial, bounds)

        assert result["param"] == 500.0
        trial.suggest_float.assert_called_once_with("param", 200, 1440, step=10)

    def test_continuous_bound(self):
        from optuna_optimizer.models import Bound
        from optuna_optimizer.trial import sample_params

        trial = MagicMock()
        trial.suggest_float.return_value = 0.025
        bounds = {"param": Bound(low=0.01, high=0.05)}

        result = sample_params(trial, bounds)

        assert result["param"] == 0.025
        trial.suggest_float.assert_called_once_with("param", 0.01, 0.05)

    def test_fixed_params_override(self):
        from optuna_optimizer.models import Bound
        from optuna_optimizer.trial import sample_params

        trial = MagicMock()
        trial.suggest_float.return_value = 50.0
        bounds = {
            "tuned": Bound(low=0, high=100, step=1),
            "fixed": Bound(low=0, high=100, step=1),
        }
        fixed_params = {"fixed": 42.0}

        result = sample_params(trial, bounds, fixed_params=fixed_params)

        assert result["fixed"] == 42.0
        # Only "tuned" should be sampled (stepped bounds use suggest_float with step)
        assert trial.suggest_float.call_count == 1
        assert trial.suggest_float.call_args[0][0] == "tuned"

    def test_multiple_params(self):
        from optuna_optimizer.models import Bound
        from optuna_optimizer.trial import sample_params

        trial = MagicMock()
        # All bounds use suggest_float (including fixed bounds now)
        trial.suggest_float.side_effect = [0.03, 1.0, 500.0]

        bounds = {
            "continuous": Bound(low=0.01, high=0.05),
            "fixed": Bound(low=1.0, high=1.0),
            "stepped": Bound(low=200, high=1440, step=10),
        }

        result = sample_params(trial, bounds)

        assert result["continuous"] == 0.03
        assert result["fixed"] == 1.0
        assert result["stepped"] == 500.0
        # All 3 bounds now use suggest_float
        assert trial.suggest_float.call_count == 3


class TestCheckConstraints:
    def test_no_violations(self):
        from optuna_optimizer.models import Constraint
        from optuna_optimizer.trial import check_constraints

        metrics = {"drawdown": 0.3, "adg": 0.002}
        constraints = [
            Constraint(metric="drawdown", max=0.5),
            Constraint(metric="adg", min=0.001),
        ]

        violations = check_constraints(metrics, constraints)

        assert violations == [0.0, 0.0]

    def test_max_violation(self):
        from optuna_optimizer.models import Constraint
        from optuna_optimizer.trial import check_constraints

        metrics = {"drawdown": 0.7}
        constraints = [Constraint(metric="drawdown", max=0.5)]

        violations = check_constraints(metrics, constraints)

        assert violations == pytest.approx([0.2])  # 0.7 - 0.5

    def test_min_violation(self):
        from optuna_optimizer.models import Constraint
        from optuna_optimizer.trial import check_constraints

        metrics = {"adg": 0.0005}
        constraints = [Constraint(metric="adg", min=0.001)]

        violations = check_constraints(metrics, constraints)

        assert violations == pytest.approx([0.0005])  # 0.001 - 0.0005

    def test_missing_metric(self):
        from optuna_optimizer.models import Constraint
        from optuna_optimizer.trial import check_constraints

        metrics = {}
        constraints = [Constraint(metric="missing", max=0.5)]

        violations = check_constraints(metrics, constraints)

        # Missing metric defaults to 0.0, which is < 0.5, so no violation
        assert violations == [0.0]

    def test_empty_constraints(self):
        from optuna_optimizer.trial import check_constraints

        metrics = {"anything": 100}
        constraints = []

        violations = check_constraints(metrics, constraints)

        assert violations == []


class TestResolveMetric:
    def test_exact_match(self):
        from optuna_optimizer.trial import resolve_metric

        flat_stats = {"mdg": 0.5, "sharpe_ratio": 1.2}
        assert resolve_metric("mdg", flat_stats) == 0.5

    def test_mean_suffix(self):
        from optuna_optimizer.trial import resolve_metric

        flat_stats = {"mdg_mean": 0.5}
        assert resolve_metric("mdg", flat_stats) == 0.5

    def test_btc_suffix_preferred(self):
        from optuna_optimizer.trial import resolve_metric

        flat_stats = {"mdg_w_btc_mean": 0.5, "mdg_w_usd_mean": 0.6}
        assert resolve_metric("mdg_w", flat_stats) == 0.5

    def test_usd_suffix_fallback(self):
        from optuna_optimizer.trial import resolve_metric

        flat_stats = {"mdg_w_usd_mean": 0.6}
        assert resolve_metric("mdg_w", flat_stats) == 0.6

    def test_missing_returns_zero(self):
        from optuna_optimizer.trial import resolve_metric

        flat_stats = {"other_metric": 1.0}
        assert resolve_metric("mdg", flat_stats) == 0.0

    def test_exact_match_takes_priority(self):
        from optuna_optimizer.trial import resolve_metric

        flat_stats = {"mdg": 0.1, "mdg_mean": 0.2, "mdg_btc_mean": 0.3}
        assert resolve_metric("mdg", flat_stats) == 0.1


class TestComputeScores:
    def test_single_objective_maximize(self):
        from optuna_optimizer.models import Objective
        from optuna_optimizer.trial import compute_scores

        objectives = [Objective(metric="mdg", direction="maximize")]
        flat_stats = {"mdg_mean": 0.5}

        scores = compute_scores(flat_stats, objectives)

        # Maximize -> sign = -1, so score = -0.5
        assert scores == (-0.5,)

    def test_single_objective_minimize(self):
        from optuna_optimizer.models import Objective
        from optuna_optimizer.trial import compute_scores

        objectives = [Objective(metric="drawdown", direction="minimize")]
        flat_stats = {"drawdown_mean": 0.3}

        scores = compute_scores(flat_stats, objectives)

        # Minimize -> sign = 1, so score = 0.3
        assert scores == (0.3,)

    def test_multiple_objectives(self):
        from optuna_optimizer.models import Objective
        from optuna_optimizer.trial import compute_scores

        objectives = [
            Objective(metric="mdg", direction="maximize"),
            Objective(metric="sharpe", direction="maximize"),
        ]
        flat_stats = {"mdg_mean": 0.5, "sharpe_mean": 1.2}

        scores = compute_scores(flat_stats, objectives)

        assert scores == (-0.5, -1.2)
