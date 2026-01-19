"""Tests for Optuna optimizer orchestrator (optuna_optimize.py).

Regression tests for:
- Sampler recreation on resume
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestSamplerRecreation:
    """Test that sampler is properly recreated from config on resume."""

    def test_resume_uses_nsgaii_sampler(self, tmp_path):
        """Resume should recreate NSGA-II sampler from config."""
        study_dir = tmp_path / "test_study"
        study_dir.mkdir()

        config = {
            "backtest": {"exchanges": ["binance"]},
            "optimize": {
                "bounds": {"long_ema_span_0": [200, 1440]},
                "objectives": [{"metric": "adg_pnl", "direction": "maximize"}],
                "constraints": [],
                "optuna": {
                    "n_trials": 10,
                    "n_cpus": 1,
                    "sampler": {"name": "nsgaii", "population_size": 100},
                },
            },
        }
        (study_dir / "config.json").write_text(json.dumps(config))
        (study_dir / "study.db").write_text("")  # Changed from journal.log to study.db

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        captured_sampler = None

        def capture_load_study(*args, **kwargs):
            nonlocal captured_sampler
            captured_sampler = kwargs.get("sampler")
            mock = MagicMock()
            mock.trials = []
            mock.user_attrs = {}
            return mock

        with patch("optuna_optimize.load_config") as mock_load_config:
            mock_load_config.return_value = config

            with patch("optuna_optimize.load_from_sqlite") as mock_load_sqlite:
                mock_study = MagicMock()
                mock_study.trials = []
                mock_study.user_attrs = {}
                mock_backend = MagicMock()

                def capture_sampler(*args, **kwargs):
                    nonlocal captured_sampler
                    # load_from_sqlite receives (path, name, sampler)
                    if len(args) >= 3:
                        captured_sampler = args[2]
                    return mock_backend, mock_study

                mock_load_sqlite.side_effect = capture_sampler

                with patch("optuna_optimize._run_optimization_core", new_callable=AsyncMock):
                    from optuna_optimize import resume_optimization
                    import asyncio

                    try:
                        asyncio.run(resume_optimization(study_dir, n_trials=1))
                    except Exception:
                        pass

        assert captured_sampler is not None, "Sampler should be passed to load_from_sqlite"

        from optuna.samplers import NSGAIISampler
        assert isinstance(captured_sampler, NSGAIISampler), \
            f"Expected NSGAIISampler, got {type(captured_sampler).__name__}"

    def test_resume_uses_nsgaiii_sampler_when_configured(self, tmp_path):
        """Resume should use NSGA-III sampler when that's what config specifies."""
        study_dir = tmp_path / "test_study"
        study_dir.mkdir()

        config = {
            "backtest": {"exchanges": ["binance"]},
            "optimize": {
                "bounds": {"long_ema_span_0": [200, 1440]},
                "objectives": [{"metric": "adg_pnl", "direction": "maximize"}],
                "constraints": [],
                "optuna": {
                    "n_trials": 10,
                    "n_cpus": 1,
                    "sampler": {"name": "nsgaiii", "population_size": 100},
                },
            },
        }
        (study_dir / "config.json").write_text(json.dumps(config))
        (study_dir / "study.db").write_text("")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        captured_sampler = None

        with patch("optuna_optimize.load_config") as mock_load_config:
            mock_load_config.return_value = config

            with patch("optuna_optimize.load_from_sqlite") as mock_load_sqlite:
                mock_study = MagicMock()
                mock_study.trials = []
                mock_study.user_attrs = {}
                mock_backend = MagicMock()

                def capture_sampler(*args, **kwargs):
                    nonlocal captured_sampler
                    if len(args) >= 3:
                        captured_sampler = args[2]
                    return mock_backend, mock_study

                mock_load_sqlite.side_effect = capture_sampler

                with patch("optuna_optimize._run_optimization_core", new_callable=AsyncMock):
                    from optuna_optimize import resume_optimization
                    import asyncio

                    try:
                        asyncio.run(resume_optimization(study_dir, n_trials=1))
                    except Exception:
                        pass

        from optuna.samplers import NSGAIIISampler
        assert isinstance(captured_sampler, NSGAIIISampler), \
            f"Expected NSGAIIISampler, got {type(captured_sampler).__name__}"
