"""Tests for optuna_optimizer package exports."""
import optuna_optimizer


class TestAllExports:
    def test_all_exports_are_importable(self):
        """Verify every name in __all__ is actually exported."""
        for name in optuna_optimizer.__all__:
            assert hasattr(optuna_optimizer, name), f"Missing export: {name}"

    def test_all_exports_are_not_none(self):
        """Verify exported names are not accidentally None."""
        for name in optuna_optimizer.__all__:
            obj = getattr(optuna_optimizer, name)
            assert obj is not None, f"Export is None: {name}"

    def test_expected_exports_present(self):
        """Verify key exports are in __all__."""
        expected = [
            # Models
            "Bound",
            "Constraint",
            "Objective",
            "OptunaConfig",
            # Config functions
            "extract_bounds",
            "apply_params_to_config",
            "load_seed_configs",
            # Samplers
            "create_sampler",
            "get_sampler_config_by_name",
            # Trial
            "sample_params",
            "resolve_metric",
            # Shared arrays
            "SharedArrayManager",
            "attach_shared_array",
            # Worker
            "init_worker",
            "get_context",
        ]
        for name in expected:
            assert name in optuna_optimizer.__all__, f"Expected export missing: {name}"
