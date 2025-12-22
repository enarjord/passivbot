"""
Characterization tests for src/optimize.py

These tests capture the current behavior of optimize.py functions and classes
to enable safe refactoring. They document how the code actually works today.
"""

import math
import os
import tempfile
from copy import deepcopy
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

import optimize
from optimize import (
    _apply_config_overrides,
    _looks_like_bool_token,
    _normalize_optional_bool_flag,
    _format_objectives,
    individual_to_config,
    config_to_individual,
    validate_array,
    apply_fine_tune_bounds,
    extract_configs,
    get_starting_configs,
    configs_to_individuals,
    ConstraintAwareFitness,
    ResultRecorder,
)
from optimization.bounds import Bound


class TestApplyConfigOverrides:
    """Test _apply_config_overrides function."""

    def test_empty_overrides_does_nothing(self):
        config = {"bot": {"long": {"value": 1.0}}}
        original = deepcopy(config)
        _apply_config_overrides(config, {})
        assert config == original

    def test_none_overrides_does_nothing(self):
        config = {"bot": {"long": {"value": 1.0}}}
        original = deepcopy(config)
        _apply_config_overrides(config, None)
        assert config == original

    def test_simple_override(self):
        config = {"bot": {"long": {"value": 1.0}}}
        _apply_config_overrides(config, {"bot.long.value": 2.0})
        assert config["bot"]["long"]["value"] == 2.0

    def test_nested_override_creates_path(self):
        config = {}
        _apply_config_overrides(config, {"bot.long.value": 5.0})
        assert config["bot"]["long"]["value"] == 5.0

    def test_override_creates_missing_dicts(self):
        config = {"bot": {}}
        _apply_config_overrides(config, {"bot.short.param": 10})
        assert config["bot"]["short"]["param"] == 10

    def test_non_string_keys_skipped(self):
        config = {"bot": {}}
        _apply_config_overrides(config, {123: "value"})
        assert config == {"bot": {}}

    def test_empty_dotted_path_creates_key(self):
        # Empty string dotted path actually creates an empty-string key
        config = {"bot": {}}
        _apply_config_overrides(config, {"": "value"})
        # Empty path creates a key with empty string
        assert config[""] == "value"

    def test_replaces_non_dict_intermediate_values(self):
        config = {"bot": {"long": "not_a_dict"}}
        _apply_config_overrides(config, {"bot.long.value": 3.0})
        assert config["bot"]["long"]["value"] == 3.0


class TestLooksLikeBoolToken:
    """Test _looks_like_bool_token function."""

    def test_numeric_string_booleans(self):
        assert _looks_like_bool_token("1") is True
        assert _looks_like_bool_token("0") is True

    def test_text_booleans(self):
        assert _looks_like_bool_token("true") is True
        assert _looks_like_bool_token("false") is True
        assert _looks_like_bool_token("TRUE") is True
        assert _looks_like_bool_token("FALSE") is True

    def test_short_forms(self):
        assert _looks_like_bool_token("t") is True
        assert _looks_like_bool_token("f") is True
        assert _looks_like_bool_token("T") is True
        assert _looks_like_bool_token("F") is True

    def test_yes_no(self):
        assert _looks_like_bool_token("yes") is True
        assert _looks_like_bool_token("no") is True
        assert _looks_like_bool_token("y") is True
        assert _looks_like_bool_token("n") is True
        assert _looks_like_bool_token("YES") is True
        assert _looks_like_bool_token("NO") is True

    def test_non_bool_values(self):
        assert _looks_like_bool_token("maybe") is False
        assert _looks_like_bool_token("2") is False
        assert _looks_like_bool_token("hello") is False
        assert _looks_like_bool_token("") is False


class TestNormalizeOptionalBoolFlag:
    """Test _normalize_optional_bool_flag function."""

    def test_flag_without_value_at_end(self):
        # Flag at end has no next token, so not converted
        argv = ["--suite"]
        result = _normalize_optional_bool_flag(argv, "--suite")
        assert result == ["--suite"]

    def test_flag_with_following_flag(self):
        # Next token starts with -, so not converted
        argv = ["--suite", "--other-flag"]
        result = _normalize_optional_bool_flag(argv, "--suite")
        assert result == ["--suite", "--other-flag"]

    def test_flag_with_bool_token_unchanged(self):
        # Bool token, so not converted
        argv = ["--suite", "false"]
        result = _normalize_optional_bool_flag(argv, "--suite")
        assert result == ["--suite", "false"]

    def test_flag_with_bool_token_yes(self):
        # Bool token, so not converted
        argv = ["--suite", "yes"]
        result = _normalize_optional_bool_flag(argv, "--suite")
        assert result == ["--suite", "yes"]

    def test_flag_not_present(self):
        argv = ["--other", "value"]
        result = _normalize_optional_bool_flag(argv, "--suite")
        assert result == ["--other", "value"]

    def test_multiple_flags(self):
        # Both flags followed by other flags or values
        argv = ["--suite", "--suite", "value"]
        result = _normalize_optional_bool_flag(argv, "--suite")
        # First --suite followed by another flag, second --suite has "value" which is not a bool token
        assert result == ["--suite", "--suite=true", "value"]

    def test_flag_with_non_bool_value_gets_converted(self):
        # Non-bool value after flag causes conversion
        argv = ["--suite", "custom_value"]
        result = _normalize_optional_bool_flag(argv, "--suite")
        assert result == ["--suite=true", "custom_value"]


class TestFormatObjectives:
    """Test _format_objectives function."""

    def test_empty_list(self):
        assert _format_objectives([]) == "[]"

    def test_single_value(self):
        result = _format_objectives([1.234567])
        assert result == "[1.23]"

    def test_multiple_values(self):
        result = _format_objectives([1.234, 5.678, 9.012])
        assert result == "[1.23, 5.68, 9.01]"

    def test_numpy_array(self):
        values = np.array([1.234, 5.678])
        result = _format_objectives(values)
        assert result == "[1.23, 5.68]"

    def test_large_numbers(self):
        result = _format_objectives([1234.5678])
        # 3 significant figures
        assert result == "[1.23e+03]"

    def test_small_numbers(self):
        result = _format_objectives([0.001234])
        assert result == "[0.00123]"

    def test_mixed_magnitude(self):
        result = _format_objectives([1000, 0.001, 10])
        assert result == "[1e+03, 0.001, 10]"


class TestIndividualToConfig:
    """Test individual_to_config function."""

    def test_basic_conversion(self):
        individual = [1.0, 2.0, 3.0, 4.0]
        template = {
            "bot": {
                "long": {"param1": 0.0, "param2": 0.0},
                "short": {"param1": 0.0, "param2": 0.0},
            }
        }
        overrides_list = []
        mock_overrides = lambda x, y, z: y

        result = individual_to_config(individual, mock_overrides, overrides_list, template)

        assert result["bot"]["long"]["param1"] == 1.0
        assert result["bot"]["long"]["param2"] == 2.0
        assert result["bot"]["short"]["param1"] == 3.0
        assert result["bot"]["short"]["param2"] == 4.0

    def test_sorted_keys(self):
        # Keys should be sorted alphabetically
        individual = [10.0, 20.0, 30.0, 40.0]
        template = {
            "bot": {
                "long": {"z_param": 0.0, "a_param": 0.0},
                "short": {"z_param": 0.0, "a_param": 0.0},
            }
        }
        overrides_list = []
        mock_overrides = lambda x, y, z: y

        result = individual_to_config(individual, mock_overrides, overrides_list, template)

        # sorted: long.a_param, long.z_param, short.a_param, short.z_param
        assert result["bot"]["long"]["a_param"] == 10.0
        assert result["bot"]["long"]["z_param"] == 20.0
        assert result["bot"]["short"]["a_param"] == 30.0
        assert result["bot"]["short"]["z_param"] == 40.0

    def test_does_not_modify_template(self):
        individual = [1.0, 2.0, 3.0, 4.0]
        template = {
            "bot": {
                "long": {"param1": 99.0, "param2": 99.0},
                "short": {"param1": 99.0, "param2": 99.0},
            }
        }
        original_template = deepcopy(template)
        overrides_list = []
        mock_overrides = lambda x, y, z: y

        individual_to_config(individual, mock_overrides, overrides_list, template)

        # Template should be unchanged
        assert template == original_template


class TestConfigToIndividual:
    """Test config_to_individual function."""

    def test_basic_conversion(self):
        config = {
            "bot": {
                "long": {"param1": 1.5, "param2": 2.5},
                "short": {"param1": 3.5, "param2": 4.5},
            }
        }
        bounds = [Bound(0.0, 10.0) for _ in range(4)]
        sig_digits = 6

        result = config_to_individual(config, bounds, sig_digits)

        # Values should be extracted in sorted order
        assert len(result) == 4
        assert result[0] == pytest.approx(1.5)
        assert result[1] == pytest.approx(2.5)
        assert result[2] == pytest.approx(3.5)
        assert result[3] == pytest.approx(4.5)

    def test_sorted_key_order(self):
        config = {
            "bot": {
                "long": {"z_param": 100.0, "a_param": 200.0},
                "short": {"z_param": 300.0, "a_param": 400.0},
            }
        }
        bounds = [Bound(0.0, 1000.0)] * 4
        sig_digits = 6

        result = config_to_individual(config, bounds, sig_digits)

        # Order: long.a_param, long.z_param, short.a_param, short.z_param
        assert result[0] == pytest.approx(200.0)
        assert result[1] == pytest.approx(100.0)
        assert result[2] == pytest.approx(400.0)
        assert result[3] == pytest.approx(300.0)


class TestValidateArray:
    """Test validate_array function."""

    def test_valid_array(self):
        arr = np.array([1.0, 2.0, 3.0])
        # Should not raise
        validate_array(arr, "test_array")

    def test_array_with_nan_allowed(self):
        arr = np.array([1.0, np.nan, 3.0])
        # Should not raise when allow_nan=True (default)
        validate_array(arr, "test_array", allow_nan=True)

    def test_array_with_nan_not_allowed(self):
        arr = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="contains NaN values"):
            validate_array(arr, "test_array", allow_nan=False)

    def test_array_with_inf_raises(self):
        arr = np.array([1.0, np.inf, 3.0])
        with pytest.raises(ValueError, match="contains inf values"):
            validate_array(arr, "test_array")

    def test_array_with_neg_inf_raises(self):
        arr = np.array([1.0, -np.inf, 3.0])
        with pytest.raises(ValueError, match="contains inf values"):
            validate_array(arr, "test_array")

    def test_array_all_nan_raises(self):
        arr = np.array([np.nan, np.nan, np.nan])
        with pytest.raises(ValueError, match="is entirely NaN"):
            validate_array(arr, "test_array", allow_nan=True)

    def test_2d_array(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        # Should not raise
        validate_array(arr, "test_array")

    def test_empty_array_raises(self):
        # Empty array is treated as "all NaN"
        arr = np.array([])
        with pytest.raises(ValueError, match="is entirely NaN"):
            validate_array(arr, "test_array")


class TestApplyFineTuneBounds:
    """Test apply_fine_tune_bounds function."""

    def test_no_fine_tune_params_no_change(self):
        config = {
            "optimize": {
                "bounds": {
                    "long_param1": [0.0, 1.0],
                    "short_param1": [0.0, 1.0],
                }
            },
            "bot": {
                "long": {"param1": 0.5},
                "short": {"param1": 0.5},
            },
        }
        original = deepcopy(config)
        apply_fine_tune_bounds(config, [], set())
        assert config == original

    def test_fix_non_tuned_params(self):
        config = {
            "optimize": {
                "bounds": {
                    "long_param1": [0.0, 1.0],
                    "long_param2": [0.0, 1.0],
                }
            },
            "bot": {
                "long": {"param1": 0.5, "param2": 0.7},
            },
        }
        # Only tune param1, param2 should be fixed
        apply_fine_tune_bounds(config, ["long_param1"], set())

        assert config["optimize"]["bounds"]["long_param1"] == [0.0, 1.0]
        assert config["optimize"]["bounds"]["long_param2"] == [0.7, 0.7]

    def test_cli_override_single_value_becomes_fixed(self):
        config = {
            "optimize": {
                "bounds": {
                    "long_param1": [5.0],  # Single value
                }
            },
            "bot": {
                "long": {"param1": 0.5},
            },
        }
        apply_fine_tune_bounds(config, [], {"long_param1"})

        assert config["optimize"]["bounds"]["long_param1"] == [5.0, 5.0]

    def test_cli_override_scalar_becomes_fixed(self):
        config = {
            "optimize": {
                "bounds": {
                    "long_param1": 5.0,  # Scalar
                }
            },
            "bot": {
                "long": {"param1": 0.5},
            },
        }
        apply_fine_tune_bounds(config, [], {"long_param1"})

        assert config["optimize"]["bounds"]["long_param1"] == [5.0, 5.0]

    def test_missing_bot_value_logs_warning(self):
        # When bot value is missing for a non-tuned param, it should log warning
        config = {
            "optimize": {
                "bounds": {
                    "long_param1": [0.0, 1.0],
                }
            },
            "bot": {
                "long": {},  # Missing param1
            },
        }
        # This should log a warning but not crash
        apply_fine_tune_bounds(config, [], set())

        # Bound should remain unchanged
        assert config["optimize"]["bounds"]["long_param1"] == [0.0, 1.0]

    def test_unparseable_key_keeps_bound(self):
        # Keys without underscore can't be parsed as pside_param
        config = {
            "optimize": {
                "bounds": {
                    "invalidkey": [0.0, 1.0],
                }
            },
            "bot": {},
        }
        # This should log a warning but not crash
        apply_fine_tune_bounds(config, [], set())
        # Bound should still exist
        assert config["optimize"]["bounds"]["invalidkey"] == [0.0, 1.0]

    def test_fine_tune_missing_key_logs(self):
        config = {
            "optimize": {"bounds": {}},
            "bot": {
                "long": {"param1": 0.5},
            },
        }
        # Requesting a non-existent key should log warning
        apply_fine_tune_bounds(config, ["long_nonexistent"], set())


class TestExtractConfigs:
    """Test extract_configs function."""

    def test_nonexistent_path(self):
        result = extract_configs("/nonexistent/path")
        assert result == []

    def test_all_results_bin_skipped(self):
        with tempfile.NamedTemporaryFile(suffix="_all_results.bin", delete=False) as f:
            path = f.name
        try:
            result = extract_configs(path)
            assert result == []
        finally:
            os.unlink(path)

    def test_json_file_loaded(self):
        from config_utils import get_template_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            # Write a minimal valid config
            template = get_template_config()
            import json

            f.write(json.dumps(template))
            path = f.name
        try:
            result = extract_configs(path)
            assert len(result) == 1
            assert "bot" in result[0]
        finally:
            os.unlink(path)

    def test_invalid_json_returns_empty(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("invalid json{")
            path = f.name
        try:
            result = extract_configs(path)
            assert result == []
        finally:
            os.unlink(path)

    def test_pareto_txt_file(self):
        from config_utils import get_template_config
        import json

        with tempfile.NamedTemporaryFile(mode="w", suffix="_pareto.txt", delete=False) as f:
            template = get_template_config()
            f.write(json.dumps(template) + "\n")
            f.write(json.dumps(template) + "\n")
            path = f.name
        try:
            result = extract_configs(path)
            assert len(result) == 2
        finally:
            os.unlink(path)


class TestGetStartingConfigs:
    """Test get_starting_configs function."""

    def test_none_returns_empty(self):
        result = get_starting_configs(None)
        assert result == []

    def test_single_file(self):
        from config_utils import get_template_config
        import json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            template = get_template_config()
            f.write(json.dumps(template))
            path = f.name
        try:
            result = get_starting_configs(path)
            assert len(result) == 1
        finally:
            os.unlink(path)

    def test_directory(self):
        from config_utils import get_template_config
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two JSON files
            template = get_template_config()
            with open(os.path.join(tmpdir, "config1.json"), "w") as f:
                f.write(json.dumps(template))
            with open(os.path.join(tmpdir, "config2.json"), "w") as f:
                f.write(json.dumps(template))

            result = get_starting_configs(tmpdir)
            assert len(result) == 2


class TestConfigsToIndividuals:
    """Test configs_to_individuals function."""

    def test_empty_configs(self):
        result = configs_to_individuals([], [], 6)
        assert result == []

    def test_single_config(self):
        from config_utils import get_template_config

        config = get_template_config()
        # Use actual bounds from the template
        from optimization.config_adapter import extract_bounds_tuple_list_from_config

        bounds = extract_bounds_tuple_list_from_config(config)

        result = configs_to_individuals([config], bounds, 6)

        # Should return 2 individuals: original + one with lowered TWE
        assert len(result) >= 1
        assert len(result[0]) == len(bounds)

    def test_creates_twe_variant(self):
        from config_utils import get_template_config

        config = get_template_config()
        from optimization.config_adapter import extract_bounds_tuple_list_from_config

        bounds = extract_bounds_tuple_list_from_config(config)

        result = configs_to_individuals([config], bounds, 6)

        # Should create duplicate with 0.75x TWE
        assert len(result) == 2

    def test_invalid_config_logged(self):
        invalid_config = {"invalid": "structure"}
        bounds = [(0.0, 10.0, 0.0)] * 4
        result = configs_to_individuals([invalid_config], bounds, 6)

        assert len(result) == 0


class TestConstraintAwareFitness:
    """Test ConstraintAwareFitness class."""

    def test_lower_violation_dominates(self):
        # Create a concrete fitness class from ConstraintAwareFitness
        # This is how deap requires it to be created
        pytest.importorskip("deap")
        from deap import creator, base

        # Clean up if it exists from previous test
        if hasattr(creator, "TestFitness1"):
            del creator.TestFitness1

        creator.create("TestFitness1", ConstraintAwareFitness, weights=(-1.0, -1.0))

        fit1 = creator.TestFitness1((10.0, 10.0))
        fit1.constraint_violation = 0.5

        fit2 = creator.TestFitness1((1.0, 1.0))
        fit2.constraint_violation = 1.0

        # Lower violation dominates regardless of objective values
        assert fit1.dominates(fit2)
        assert not fit2.dominates(fit1)

        # Cleanup
        del creator.TestFitness1

    def test_dominates_with_same_violation(self):
        # With same constraint violation, regular dominance applies
        pytest.importorskip("deap")
        from deap import creator, base

        # Clean up if it exists
        if hasattr(creator, "TestFitness2"):
            del creator.TestFitness2

        creator.create("TestFitness2", ConstraintAwareFitness, weights=(-1.0, -1.0))

        fit1 = creator.TestFitness2((1.0, 2.0))
        fit1.constraint_violation = 0.0

        fit2 = creator.TestFitness2((2.0, 3.0))
        fit2.constraint_violation = 0.0

        # With minimization weights, lower values dominate
        assert fit1.dominates(fit2)
        assert not fit2.dominates(fit1)

        # Cleanup
        del creator.TestFitness2


class TestResultRecorder:
    """Test ResultRecorder class."""

    def test_initialization(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bounds = [(0.0, 1.0, 0.0)] * 4
            recorder = ResultRecorder(
                results_dir=tmpdir,
                sig_digits=6,
                flush_interval=60,
                scoring_keys=["metric1", "metric2"],
                compress=False,
                write_all_results=False,
                pareto_max_size=300,
                bounds=bounds,
            )

            assert recorder.write_all is False
            assert recorder.compress is False
            assert recorder.scoring_keys == ["metric1", "metric2"]
            assert recorder.results_file is None

    def test_initialization_with_write_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bounds = [(0.0, 1.0, 0.0)] * 4
            recorder = ResultRecorder(
                results_dir=tmpdir,
                sig_digits=6,
                flush_interval=60,
                scoring_keys=["metric1"],
                compress=False,
                write_all_results=True,
                bounds=bounds,
            )

            assert recorder.write_all is True
            assert recorder.results_file is not None
            assert os.path.exists(os.path.join(tmpdir, "all_results.bin"))

            recorder.close()

    def test_record_data_without_write_all(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bounds = [(0.0, 1.0, 0.0)] * 4
            recorder = ResultRecorder(
                results_dir=tmpdir,
                sig_digits=6,
                flush_interval=60,
                scoring_keys=["metric1"],
                compress=False,
                write_all_results=False,
                bounds=bounds,
            )

            data = {
                "bot": {"long": {}, "short": {}},
                "metrics": {
                    "objectives": {"metric1": 0.5},
                    "constraint_violation": 0.0,
                },
            }

            # Should not raise
            recorder.record(data)
            recorder.close()

    def test_record_updates_pareto_store(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bounds = [(0.0, 1.0, 0.0)] * 4
            recorder = ResultRecorder(
                results_dir=tmpdir,
                sig_digits=6,
                flush_interval=60,
                scoring_keys=["metric1"],
                compress=False,
                write_all_results=False,
                bounds=bounds,
            )

            data = {
                "bot": {"long": {"param": 0.5}, "short": {"param": 0.5}},
                "metrics": {
                    "objectives": {"metric1": 0.5},
                    "constraint_violation": 0.0,
                },
            }

            recorder.record(data)

            # Store should have entries
            assert recorder.store.n_iters > 0

            recorder.close()

    def test_flush_and_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            bounds = [(0.0, 1.0, 0.0)] * 4
            recorder = ResultRecorder(
                results_dir=tmpdir,
                sig_digits=6,
                flush_interval=60,
                scoring_keys=["metric1"],
                compress=False,
                write_all_results=True,
                bounds=bounds,
            )

            data = {
                "bot": {"long": {"param": 0.5}, "short": {"param": 0.5}},
                "metrics": {
                    "objectives": {"metric1": 0.5},
                    "constraint_violation": 0.0,
                },
            }

            recorder.record(data)
            recorder.flush()
            recorder.close()

            # File should be closed
            assert recorder.results_file.closed


class TestEvaluator:
    """Test Evaluator class initialization and basic methods."""

    def test_perturb_step_digits(self):
        """Test perturbation with different change probabilities."""
        from optimize import Evaluator
        from config_utils import get_template_config

        # Create minimal evaluator for testing perturbation methods
        mock_config = get_template_config()
        # Make sure limits is a list, not a string
        mock_config["optimize"]["limits"] = []

        evaluator = Evaluator(
            hlcvs_specs={},
            btc_usd_specs={},
            msss={},
            config=mock_config,
        )

        # Use a simple individual matching bounds length
        individual = [1.0] * len(evaluator.bounds)
        perturbed = evaluator.perturb_step_digits(individual, change_chance=0.5)

        # Should have same length
        assert len(perturbed) == len(individual)

    def test_perturb_x_pct(self):
        """Test percentage-based perturbation."""
        from optimize import Evaluator
        from config_utils import get_template_config

        mock_config = get_template_config()
        # Make sure limits is a list, not a string
        mock_config["optimize"]["limits"] = []

        evaluator = Evaluator(
            hlcvs_specs={},
            btc_usd_specs={},
            msss={},
            config=mock_config,
        )

        individual = [5.0] * len(evaluator.bounds)
        perturbed = evaluator.perturb_x_pct(individual, magnitude=0.01)

        assert len(perturbed) == len(individual)

    def test_perturb_random_subset(self):
        """Test random subset perturbation."""
        from optimize import Evaluator
        from config_utils import get_template_config

        mock_config = get_template_config()
        # Make sure limits is a list, not a string
        mock_config["optimize"]["limits"] = []

        evaluator = Evaluator(
            hlcvs_specs={},
            btc_usd_specs={},
            msss={},
            config=mock_config,
        )

        individual = [1.0] * len(evaluator.bounds)
        perturbed = evaluator.perturb_random_subset(individual, frac=0.4)

        assert len(perturbed) == len(individual)

    def test_build_limit_checks(self):
        """Test limit checks building."""
        from optimize import Evaluator
        from config_utils import get_template_config

        mock_config = get_template_config()
        mock_config["optimize"]["limits"] = [
            {
                "metric": "drawdown_worst_usd",
                "penalize_if": "greater_than",
                "value": 0.3,
            }
        ]

        evaluator = Evaluator(
            hlcvs_specs={},
            btc_usd_specs={},
            msss={},
            config=mock_config,
        )

        assert len(evaluator.limit_checks) > 0
        # Check that metric key is created (could be max, mean, etc.)
        assert "drawdown_worst_usd" in evaluator.limit_checks[0]["metric_key"]
