"""Tests for Pareto extraction functionality."""
from unittest.mock import MagicMock

from optuna_optimizer.models import Objective


class TestTrialToConfig:
    def test_routes_long_params_correctly(self):
        from optuna_optimizer.pareto import trial_to_config

        trial = MagicMock()
        trial.params = {"long_entry_qty": 0.1}
        base = {}
        result = trial_to_config(trial, base)
        assert result["bot"]["long"]["entry_qty"] == 0.1

    def test_routes_short_params_correctly(self):
        from optuna_optimizer.pareto import trial_to_config

        trial = MagicMock()
        trial.params = {"short_entry_qty": 0.2}
        base = {}
        result = trial_to_config(trial, base)
        assert result["bot"]["short"]["entry_qty"] == 0.2

    def test_routes_other_params_to_live(self):
        from optuna_optimizer.pareto import trial_to_config

        trial = MagicMock()
        trial.params = {"leverage": 5}
        base = {}
        result = trial_to_config(trial, base)
        assert result["live"]["leverage"] == 5

    def test_preserves_base_config(self):
        from optuna_optimizer.pareto import trial_to_config

        trial = MagicMock()
        trial.params = {"long_qty": 0.1}
        base = {"backtest": {"start": "2024-01-01"}}
        result = trial_to_config(trial, base)
        assert result["backtest"]["start"] == "2024-01-01"


class TestExtractPareto:
    def test_creates_directory(self, tmp_path):
        from optuna_optimizer.pareto import extract_pareto
        study_dir = tmp_path / "study"
        study_dir.mkdir()
        study = MagicMock()
        study.best_trials = []
        objectives = [Objective(metric="adg_pnl", direction="maximize")]
        extract_pareto(study, study_dir, objectives, {})
        assert (study_dir / "pareto").exists()

    def test_writes_configs(self, tmp_path):
        from optuna_optimizer.pareto import extract_pareto
        study_dir = tmp_path / "study"
        study_dir.mkdir()
        trial = MagicMock()
        trial.number = 42
        # Optuna stores minimized value: -1 * 0.001 = -0.001 for maximize
        trial.values = [-0.001]
        trial.params = {"param": 1}
        study = MagicMock()
        study.best_trials = [trial]
        objectives = [Objective(metric="adg_pnl", direction="maximize")]
        extract_pareto(study, study_dir, objectives, {})
        files = list((study_dir / "pareto").glob("*.json"))
        assert len(files) == 1
        assert "0042_" in files[0].name
        # Filename should show actual value (undone sign transformation)
        assert "adg_pnl0.001" in files[0].name
