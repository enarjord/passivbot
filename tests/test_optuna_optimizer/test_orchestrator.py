"""Tests for Optuna optimizer orchestrator (optuna_optimize.py).

Regression tests for:
- Lock file cleanup on interrupt
- Sampler recreation on resume
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestLockFileCleanup:
    """Test that journal lock file is cleaned up in finally block."""

    @pytest.fixture
    def study_dir(self, tmp_path):
        """Create a temporary study directory with config and journal."""
        study_dir = tmp_path / "test_study"
        study_dir.mkdir()

        # Minimal config
        config = {
            "backtest": {"exchanges": ["binance"]},
            "optimize": {
                "bounds": {"long_ema_span_0": [200, 1440]},
                "objectives": [{"metric": "adg_pnl", "direction": "maximize"}],
                "constraints": [],
                "optuna": {"n_trials": 10, "n_cpus": 1},
            },
        }
        (study_dir / "config.json").write_text(json.dumps(config))

        # Create empty journal (Optuna will populate it)
        (study_dir / "journal.log").write_text("")

        return study_dir

    def test_lock_file_removed_on_normal_exit(self, study_dir):
        """Lock file should be removed when optimization completes normally."""
        # Create a stale lock file (symlink like Optuna creates)
        lock_path = study_dir / "journal.log.lock"
        lock_path.symlink_to(study_dir / "journal.log")
        assert lock_path.exists() or lock_path.is_symlink()

        # Import here to avoid import errors when module dependencies missing
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        from optuna_optimize import _run_optimization_core
        from optuna_optimizer import SharedArrayManager

        # Mock everything heavy
        with patch("optuna_optimize._load_shared_data", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = ({}, {}, {}, {})

            with patch("optuna_optimize.Pool") as mock_pool:
                mock_pool.return_value.__enter__ = MagicMock()
                mock_pool.return_value.__exit__ = MagicMock(return_value=False)

                with patch("optuna_optimize.extract_pareto"):
                    with patch("optuna.load_study") as mock_study:
                        mock_study.return_value.trials = []

                        # Run with mocked internals - should clean up lock in finally
                        import asyncio
                        from optuna_optimizer.models import NSGAIISamplerConfig
                        try:
                            asyncio.run(_run_optimization_core(
                                config={},
                                study_dir=study_dir,
                                study=MagicMock(),
                                bounds={},
                                constraints=[],
                                objectives=[],
                                sampler_config=NSGAIISamplerConfig(),
                                n_trials=1,
                                n_cpus=1,
                                fixed_params=None,
                                penalty_weight=1000,
                                max_best_trials=10,
                                debug_level=1,
                            ))
                        except Exception:
                            pass  # We expect it may fail, but finally should still run

        # Lock file should be cleaned up
        assert not lock_path.exists() and not lock_path.is_symlink(), \
            "Lock file should be removed in finally block"

    def test_lock_file_removed_on_exception(self, study_dir):
        """Lock file should be removed even when exception occurs."""
        lock_path = study_dir / "journal.log.lock"
        lock_path.symlink_to(study_dir / "journal.log")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        from optuna_optimize import _run_optimization_core

        with patch("optuna_optimize._load_shared_data", new_callable=AsyncMock) as mock_load:
            # Simulate error during data loading
            mock_load.side_effect = RuntimeError("Simulated failure")

            with patch("optuna_optimize.extract_pareto"):
                with patch("optuna.load_study") as mock_study:
                    mock_study.return_value.trials = []

                    import asyncio
                    from optuna_optimizer.models import NSGAIISamplerConfig
                    with pytest.raises(RuntimeError, match="Simulated failure"):
                        asyncio.run(_run_optimization_core(
                            config={},
                            study_dir=study_dir,
                            study=MagicMock(),
                            bounds={},
                            constraints=[],
                            objectives=[],
                            sampler_config=NSGAIISamplerConfig(),
                            n_trials=1,
                            n_cpus=1,
                            fixed_params=None,
                            penalty_weight=1000,
                            max_best_trials=10,
                            debug_level=1,
                        ))

        # Lock should still be cleaned up despite exception
        assert not lock_path.exists() and not lock_path.is_symlink(), \
            "Lock file should be removed even on exception"


class TestSamplerRecreation:
    """Test that sampler is properly recreated from config on resume."""

    def test_resume_uses_config_sampler(self, tmp_path):
        """Resume should recreate sampler from config, not use Optuna default."""
        study_dir = tmp_path / "test_study"
        study_dir.mkdir()

        # Config with TPE sampler
        config = {
            "backtest": {"exchanges": ["binance"]},
            "optimize": {
                "bounds": {"long_ema_span_0": [200, 1440]},
                "objectives": [{"metric": "adg_pnl", "direction": "maximize"}],
                "constraints": [],
                "optuna": {
                    "n_trials": 10,
                    "n_cpus": 1,
                    "sampler": {"name": "tpe", "n_startup_trials": 50},
                },
            },
        }
        (study_dir / "config.json").write_text(json.dumps(config))
        (study_dir / "journal.log").write_text("")

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

        # Track what sampler is passed to load_study
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

            with patch("optuna.load_study", side_effect=capture_load_study):
                with patch("optuna_optimize._run_optimization_core", new_callable=AsyncMock):
                    from optuna_optimize import resume_optimization
                    import asyncio

                    try:
                        asyncio.run(resume_optimization(study_dir, n_trials=1))
                    except Exception:
                        pass  # May fail due to mocking, but we captured the sampler

        # Verify TPESampler was used, not default NSGA-II
        assert captured_sampler is not None, "Sampler should be passed to load_study"

        from optuna.samplers import TPESampler
        assert isinstance(captured_sampler, TPESampler), \
            f"Expected TPESampler, got {type(captured_sampler).__name__}"

    def test_resume_uses_gp_sampler_when_configured(self, tmp_path):
        """Resume should use GP sampler when that's what config specifies."""
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
                    "sampler": {"name": "gp"},
                },
            },
        }
        (study_dir / "config.json").write_text(json.dumps(config))
        (study_dir / "journal.log").write_text("")

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

            with patch("optuna.load_study", side_effect=capture_load_study):
                with patch("optuna_optimize._run_optimization_core", new_callable=AsyncMock):
                    from optuna_optimize import resume_optimization
                    import asyncio

                    try:
                        asyncio.run(resume_optimization(study_dir, n_trials=1))
                    except Exception:
                        pass

        from optuna.samplers import GPSampler
        assert isinstance(captured_sampler, GPSampler), \
            f"Expected GPSampler, got {type(captured_sampler).__name__}"
