"""
Characterization tests for src/optimize.py

These tests capture the current behavior of optimize.py functions and classes
to enable safe refactoring. They document how the code actually works today.
"""

import math
import os
import argparse
import json
from multiprocessing.reduction import ForkingPickler
import tempfile
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import numpy as np
import pytest

import optimize
from optimize import (
    _apply_config_overrides,
    _analysis_indicates_liquidation,
    _clear_candidate_metrics,
    _looks_like_bool_token,
    _normalize_optional_bool_flag,
    _record_individual_result,
    _resolve_cli_limits_override,
    _set_candidate_metrics,
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
from multiprocessing_utils import ignore_sigint_in_worker
from optimization.bounds import Bound
from optimization.callback import build_pymoo_record_entry
from optimization.config_adapter import extract_bounds_tuple_list_from_config
from optimization.config_adapter import get_optimization_key_paths
from optimization.shape import build_optimization_shape
from optimize_suite import ScenarioEvalContext
from config import load_prepared_config


def test_worker_initializer_is_pickleable_for_spawn():
    ForkingPickler.dumps(ignore_sigint_in_worker)


def test_candidate_metrics_sidecars_only_attach_to_objects():
    class Candidate:
        pass

    obj = Candidate()
    _set_candidate_metrics(obj, {"foo": 1})
    assert obj.evaluation_metrics == {"foo": 1}
    _clear_candidate_metrics(obj)
    assert not hasattr(obj, "evaluation_metrics")

    plain_vector = [1.0, 2.0]
    _set_candidate_metrics(plain_vector, {"foo": 1})
    assert not hasattr(plain_vector, "evaluation_metrics")
    _clear_candidate_metrics(plain_vector)
    assert plain_vector == [1.0, 2.0]


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


class TestLiquidationHelpers:
    def test_analysis_indicates_liquidation_uses_explicit_flag(self):
        from config_utils import get_template_config

        config = get_template_config()
        config["backtest"]["liquidation_threshold"] = 0.05
        assert _analysis_indicates_liquidation({"liquidated": True}, config) is True
        assert _analysis_indicates_liquidation({"liquidated": False}, config) is False
        assert (
            _analysis_indicates_liquidation({"drawdown_worst": 0.999, "liquidated": False}, config)
            is False
        )


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


class TestResolveCliLimitsOverride:
    def test_returns_none_when_no_limit_flags_present(self):
        args = argparse.Namespace(**{"optimize.limits": None, "limit_entries": None, "clear_limits": False})
        assert _resolve_cli_limits_override(args) is None

    def test_limit_entries_append_to_existing_config_limits_by_default(self):
        args = argparse.Namespace(
            **{
                "optimize.limits": None,
                "limit_entries": ["adg > 0.0008"],
                "clear_limits": False,
            }
        )

        result = _resolve_cli_limits_override(
            args,
            existing_limits=[{"metric": "drawdown_worst", "penalize_if": ">", "value": 0.35}],
        )

        assert result == [
            {"metric": "drawdown_worst_usd", "penalize_if": "greater_than", "value": 0.35},
            {
                "metric": "adg_usd",
                "penalize_if": "less_than_or_equal",
                "value": 0.0008,
            },
        ]

    def test_combines_limits_payload_and_repeatable_limit_entries(self):
        args = argparse.Namespace(
            **{
                "optimize.limits": '[{"metric":"drawdown_worst","penalize_if":">","value":0.35}]',
                "limit_entries": ["adg > 0.0008 stat=mean"],
                "clear_limits": False,
            }
        )

        result = _resolve_cli_limits_override(args)

        assert result == [
            {"metric": "drawdown_worst_usd", "penalize_if": "greater_than", "value": 0.35},
            {
                "metric": "adg_usd",
                "penalize_if": "less_than_or_equal",
                "value": 0.0008,
                "stat": "mean",
            },
        ]

    def test_limits_payload_replaces_existing_config_limits_before_appending(self):
        args = argparse.Namespace(
            **{
                "optimize.limits": '[{"metric":"backtest_completion_ratio","penalize_if":"<","value":1.0}]',
                "limit_entries": ["adg > 0.0008"],
                "clear_limits": False,
            }
        )

        result = _resolve_cli_limits_override(
            args,
            existing_limits=[{"metric": "drawdown_worst", "penalize_if": ">", "value": 0.35}],
        )

        assert result == [
            {
                "metric": "backtest_completion_ratio",
                "penalize_if": "less_than",
                "value": 1.0,
            },
            {
                "metric": "adg_usd",
                "penalize_if": "less_than_or_equal",
                "value": 0.0008,
            },
        ]

    def test_clear_limits_returns_empty_list(self):
        args = argparse.Namespace(
            **{"optimize.limits": None, "limit_entries": None, "clear_limits": True}
        )
        assert _resolve_cli_limits_override(args) == []

    def test_clear_limits_discards_existing_config_limits_before_appending(self):
        args = argparse.Namespace(
            **{
                "optimize.limits": None,
                "limit_entries": ["adg > 0.0008"],
                "clear_limits": True,
            }
        )

        result = _resolve_cli_limits_override(
            args,
            existing_limits=[{"metric": "drawdown_worst", "penalize_if": ">", "value": 0.35}],
        )

        assert result == [
            {
                "metric": "adg_usd",
                "penalize_if": "less_than_or_equal",
                "value": 0.0008,
            }
        ]


def test_optimize_parser_accepts_short_limit_alias():
    parser = optimize.build_command_parser(
        prog="passivbot optimize",
        description="run optimizer",
        usage="%(prog)s [config_path] [options]",
        epilog="",
    )
    template_config = optimize.get_template_config()
    del template_config["bot"]
    keep_live_keys = {"approved_coins", "minimum_coin_age_days"}
    for key in sorted(template_config["live"]):
        if key not in keep_live_keys:
            del template_config["live"][key]
    group_map = {
        "Coin Selection": parser.add_argument_group("Coin Selection"),
        "Date Range": parser.add_argument_group("Date Range"),
        "Optimizer": parser.add_argument_group("Optimizer"),
        "Suite": parser.add_argument_group("Suite"),
        "Logging": parser.add_argument_group("Logging"),
        "Backtest Runtime": parser.add_argument_group("Backtest Runtime"),
        "Optimize Common": parser.add_argument_group("Optimize Common"),
        "Optimize Bounds": parser.add_argument_group("Optimize Bounds"),
        "Optimize DEAP": parser.add_argument_group("Optimize DEAP"),
        "Optimize Pymoo": parser.add_argument_group("Optimize Pymoo"),
        "Advanced Overrides": parser.add_argument_group("Advanced Overrides"),
    }
    optimize.add_config_arguments(
        parser,
        template_config,
        command="optimize",
        help_all=False,
        group_map=group_map,
    )
    group_map["Optimize Common"].add_argument(
        "-l",
        "--limit",
        action="append",
        dest="limit_entries",
        default=None,
        metavar="SPEC",
    )
    group_map["Optimize Common"].add_argument(
        "--clear-limits",
        action="store_true",
        dest="clear_limits",
    )

    args = parser.parse_args(["-l", "adg_strategy_pnl_rebased > 0.0"])

    assert args.limit_entries == ["adg_strategy_pnl_rebased > 0.0"]


def test_optimize_parser_accepts_nested_bound_flags_alongside_limit_alias():
    parser = optimize.build_command_parser(
        prog="passivbot optimize",
        description="run optimizer",
        usage="%(prog)s [config_path] [options]",
        epilog="",
    )
    template_config = optimize.get_template_config()
    del template_config["bot"]
    keep_live_keys = {"approved_coins", "minimum_coin_age_days"}
    for key in sorted(template_config["live"]):
        if key not in keep_live_keys:
            del template_config["live"][key]
    group_map = {
        "Coin Selection": parser.add_argument_group("Coin Selection"),
        "Date Range": parser.add_argument_group("Date Range"),
        "Optimizer": parser.add_argument_group("Optimizer"),
        "Suite": parser.add_argument_group("Suite"),
        "Logging": parser.add_argument_group("Logging"),
        "Backtest Runtime": parser.add_argument_group("Backtest Runtime"),
        "Optimize Common": parser.add_argument_group("Optimize Common"),
        "Optimize Bounds": parser.add_argument_group("Optimize Bounds"),
        "Optimize DEAP": parser.add_argument_group("Optimize DEAP"),
        "Optimize Pymoo": parser.add_argument_group("Optimize Pymoo"),
        "Advanced Overrides": parser.add_argument_group("Advanced Overrides"),
    }
    optimize.add_config_arguments(
        parser,
        template_config,
        command="optimize",
        help_all=False,
        group_map=group_map,
    )
    group_map["Optimize Common"].add_argument(
        "-l",
        "--limit",
        action="append",
        dest="limit_entries",
        default=None,
        metavar="SPEC",
    )
    group_map["Optimize Common"].add_argument(
        "--clear-limits",
        action="store_true",
        dest="clear_limits",
    )

    args = parser.parse_args(
        [
            "--optimize.bounds.long.risk.total_wallet_exposure_limit",
            "0",
            "-l",
            "adg_strategy_pnl_rebased>0.0",
        ]
    )

    assert args.limit_entries == ["adg_strategy_pnl_rebased>0.0"]
    assert getattr(args, "optimize.bounds.long.risk.total_wallet_exposure_limit") == [0.0]


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
            },
            "optimize": {"bounds": {}},
        }
        original_template = deepcopy(template)
        overrides_list = []
        mock_overrides = lambda x, y, z: y

        individual_to_config(individual, mock_overrides, overrides_list, template)

        # Template should be unchanged
        assert template == original_template

    def test_includes_live_hsl_values_when_present_in_bounds(self):
        individual = [60.0, 1.0, 2.0, 0.22, 3.0, 4.0]
        template = {
            "bot": {
                "long": {
                    "param1": 0.0,
                    "param2": 0.0,
                    "hsl_ema_span_minutes": 120.0,
                },
                "short": {
                    "param1": 0.0,
                    "param2": 0.0,
                    "hsl_red_threshold": 0.25,
                },
            },
            "optimize": {
                "bounds": {
                    "long_param1": [0.0, 10.0],
                    "long_param2": [0.0, 10.0],
                    "short_param1": [0.0, 10.0],
                    "short_param2": [0.0, 10.0],
                    "long_hsl_ema_span_minutes": [30.0, 180.0, 5.0],
                    "short_hsl_red_threshold": [0.15, 0.35, 0.01],
                }
            },
        }
        overrides_list = []
        mock_overrides = lambda x, y, z: y

        result = individual_to_config(individual, mock_overrides, overrides_list, template)

        assert result["bot"]["long"]["hsl_ema_span_minutes"] == pytest.approx(60.0)
        assert result["bot"]["short"]["hsl_red_threshold"] == pytest.approx(0.22)

    def test_normalizes_common_hsl_no_restart_to_red_threshold_floor(self):
        individual = [0.20, 0.25, 1.0, 2.0, 3.0, 4.0]
        template = {
            "bot": {
                "long": {
                    "param1": 0.0,
                    "param2": 0.0,
                    "hsl_red_threshold": 0.20,
                    "hsl_no_restart_drawdown_threshold": 0.40,
                },
                "short": {"param1": 0.0, "param2": 0.0},
            },
            "optimize": {
                "bounds": {
                    "long_param1": [0.0, 10.0],
                    "long_param2": [0.0, 10.0],
                    "short_param1": [0.0, 10.0],
                    "short_param2": [0.0, 10.0],
                    "long_hsl_no_restart_drawdown_threshold": [0.20, 0.90, 0.01],
                    "long_hsl_red_threshold": [0.15, 0.35, 0.01],
                }
            },
        }
        overrides_list = []
        mock_overrides = lambda x, y, z: y

        result = individual_to_config(individual, mock_overrides, overrides_list, template)

        assert result["bot"]["long"]["hsl_red_threshold"] == pytest.approx(0.25)
        assert result["bot"]["long"]["hsl_no_restart_drawdown_threshold"] == pytest.approx(0.25)

    def test_applies_optimize_fixed_runtime_overrides_without_mutating_template(self):
        individual = [0.25, 1.0, 2.0, 3.0, 4.0]
        template = {
            "bot": {
                "long": {
                    "param1": 0.0,
                    "param2": 0.0,
                    "hsl_red_threshold": 0.20,
                    "hsl_no_restart_drawdown_threshold": 0.30,
                },
                "short": {"param1": 0.0, "param2": 0.0},
            },
            "optimize": {
                "bounds": {
                    "long_param1": [0.0, 10.0],
                    "long_param2": [0.0, 10.0],
                    "short_param1": [0.0, 10.0],
                    "short_param2": [0.0, 10.0],
                    "long_hsl_red_threshold": [0.15, 0.35, 0.01],
                },
                "fixed_runtime_overrides": {
                    "bot.long.hsl_no_restart_drawdown_threshold": 1.0
                },
            },
        }
        original_template = deepcopy(template)
        result = individual_to_config(individual, lambda x, y, z: y, [], template)

        assert result["bot"]["long"]["hsl_red_threshold"] == pytest.approx(0.25)
        assert result["bot"]["long"]["hsl_no_restart_drawdown_threshold"] == pytest.approx(1.0)
        assert template == original_template

    def test_mirror_short_from_long_override(self):
        individual = [
            0.018,
            100.0,
            570.0,
            0.015,
            0.7,
            0.47,
            0.011,
            63.0,
            636.0,
            0.0095,
            0.33,
            0.19,
        ]
        template = {
            "live": {"strategy_kind": "ema_anchor"},
            "bot": {
                "long": {
                    "risk": {
                        "n_positions": 1,
                        "total_wallet_exposure_limit": 0.55,
                    },
                    "forager": {
                        "volume_ema_span": 520,
                        "volatility_ema_span": 225,
                        "volume_drop_pct": 0.57,
                        "score_weights": {
                            "volume": 0.0,
                            "ema_readiness": 0.0,
                            "volatility": 1.0,
                        },
                    },
                    "hsl": {
                        "enabled": False,
                        "red_threshold": 0.2,
                    },
                    "unstuck": {
                        "close_pct": 0.05,
                        "ema_dist": -0.05,
                    },
                    "strategy": {
                        "ema_anchor": {
                            "base_qty_pct": 0.012,
                            "ema_span_0": 72.0,
                            "ema_span_1": 432.0,
                            "offset": 0.009,
                            "offset_psize_weight": 0.4,
                        }
                    },
                },
                "short": {
                    "risk": {
                        "n_positions": 3,
                        "total_wallet_exposure_limit": 0.25,
                    },
                    "forager": {
                        "volume_ema_span": 360,
                        "volatility_ema_span": 10,
                        "volume_drop_pct": 0.5,
                        "score_weights": {
                            "volume": 0.2,
                            "ema_readiness": 0.3,
                            "volatility": 0.5,
                        },
                    },
                    "hsl": {
                        "enabled": True,
                        "red_threshold": 0.1,
                    },
                    "unstuck": {
                        "close_pct": 0.11,
                        "ema_dist": -0.11,
                    },
                    "strategy": {
                        "ema_anchor": {
                            "base_qty_pct": 0.004,
                            "ema_span_0": 48.0,
                            "ema_span_1": 288.0,
                            "offset": 0.02,
                            "offset_psize_weight": 0.2,
                        }
                    },
                },
            },
            "optimize": {
                "enable_overrides": ["mirror_short_from_long"],
                "bounds": {
                    "long": {
                        "risk": {
                            "total_wallet_exposure_limit": [0.35, 0.8, 0.01],
                        },
                        "strategy": {
                            "ema_anchor": {
                                "base_qty_pct": [0.006, 0.03, 0.0005],
                                "ema_span_0": [48, 120, 1],
                                "ema_span_1": [288, 720, 1],
                                "offset": [0.006, 0.02, 0.0005],
                                "offset_psize_weight": [0.2, 0.8, 0.01],
                            }
                        },
                    },
                    "short": {
                        "risk": {
                            "total_wallet_exposure_limit": [0.15, 0.5, 0.01],
                        },
                        "strategy": {
                            "ema_anchor": {
                                "base_qty_pct": [0.004, 0.02, 0.0005],
                                "ema_span_0": [48, 120, 1],
                                "ema_span_1": [288, 720, 1],
                                "offset": [0.007, 0.02, 0.0005],
                                "offset_psize_weight": [0.2, 0.8, 0.01],
                            }
                        },
                    },
                },
            },
        }

        result = individual_to_config(
            individual,
            optimize.optimizer_overrides,
            ["mirror_short_from_long"],
            template,
        )

        assert result["bot"]["short"]["risk"] == result["bot"]["long"]["risk"]
        assert result["bot"]["short"]["forager"] == result["bot"]["long"]["forager"]
        assert result["bot"]["short"]["hsl"] == result["bot"]["long"]["hsl"]
        assert result["bot"]["short"]["unstuck"] == result["bot"]["long"]["unstuck"]
        assert (
            result["bot"]["short"]["strategy"]["ema_anchor"]
            == result["bot"]["long"]["strategy"]["ema_anchor"]
        )

    def test_mirror_short_from_long_override_supports_trailing_grid(self):
        config = {
            "live": {"strategy_kind": "trailing_grid"},
            "bot": {
                "long": {
                    "risk": {"n_positions": 1, "total_wallet_exposure_limit": 0.9},
                    "forager": {
                        "volume_ema_span": 520,
                        "volatility_ema_span": 225,
                        "score_weights": {"volume": 0.0, "ema_readiness": 0.0, "volatility": 1.0},
                    },
                    "strategy": {
                        "trailing_grid": {"ema_span_0": 100.0},
                        "ema_anchor": {"ema_span_0": 999.0},
                    },
                },
                "short": {
                    "risk": {"n_positions": 2, "total_wallet_exposure_limit": 0.1},
                    "forager": {
                        "volume_ema_span": 10,
                        "volatility_ema_span": 20,
                        "score_weights": {"volume": 0.1, "ema_readiness": 0.2, "volatility": 0.7},
                    },
                    "strategy": {
                        "trailing_grid": {"ema_span_0": 200.0},
                        "ema_anchor": {"ema_span_0": 123.0},
                    },
                },
            },
        }

        result = optimize.optimizer_overrides(["mirror_short_from_long"], deepcopy(config), None)

        assert result["bot"]["short"]["risk"] == result["bot"]["long"]["risk"]
        assert result["bot"]["short"]["forager"] == result["bot"]["long"]["forager"]
        assert (
            result["bot"]["short"]["strategy"]["trailing_grid"]
            == result["bot"]["long"]["strategy"]["trailing_grid"]
        )
        assert result["bot"]["short"]["strategy"]["ema_anchor"] == {"ema_span_0": 123.0}

    @pytest.mark.parametrize(
        ("override_name", "expected_start", "expected_end"),
        [
            ("lossless_close_trailing", 0.0015, 0.0015),
            ("forward_tp_grid", 0.0015, 0.009),
            ("backward_tp_grid", 0.009, 0.0015),
        ],
    )
    def test_legacy_trailing_grid_overrides_support_canonical_shape(
        self,
        override_name,
        expected_start,
        expected_end,
    ):
        config = load_prepared_config(
            "configs/examples/default_trailing_grid_long_npos10.json",
            verbose=False,
        )
        long_strategy = config["bot"]["long"]["strategy"]["trailing_grid"]
        long_strategy["close_grid_markup_start"] = 0.009
        long_strategy["close_grid_markup_end"] = 0.0015
        long_strategy["close_trailing_threshold_pct"] = 0.001
        long_strategy["close_trailing_retracement_pct"] = 0.004

        result = optimize.optimizer_overrides([override_name], deepcopy(config), "long")
        strategy = result["bot"]["long"]["strategy"]["trailing_grid"]

        if override_name == "lossless_close_trailing":
            assert strategy["close_trailing_threshold_pct"] == pytest.approx(0.004)
            assert strategy["close_trailing_retracement_pct"] == pytest.approx(0.004)
        else:
            assert strategy["close_grid_markup_start"] == pytest.approx(expected_start)
            assert strategy["close_grid_markup_end"] == pytest.approx(expected_end)

    def test_legacy_trailing_grid_overrides_reject_non_trailing_grid_strategy(self):
        config = load_prepared_config("configs/examples/ema_anchor.json", verbose=False)

        with pytest.raises(ValueError, match="live.strategy_kind = 'trailing_grid'"):
            optimize.optimizer_overrides(["forward_tp_grid"], deepcopy(config), "long")

    def test_accepts_precomputed_key_paths(self):
        individual = [10.0, 20.0, 30.0, 40.0]
        template = {
            "bot": {
                "long": {"z_param": 0.0, "a_param": 0.0},
                "short": {"z_param": 0.0, "a_param": 0.0},
            }
        }
        key_paths = get_optimization_key_paths(template)

        result = individual_to_config(
            individual,
            lambda x, y, z: y,
            [],
            template,
            key_paths=key_paths,
        )

        assert result["bot"]["long"]["a_param"] == 10.0
        assert result["bot"]["long"]["z_param"] == 20.0
        assert result["bot"]["short"]["a_param"] == 30.0
        assert result["bot"]["short"]["z_param"] == 40.0
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
            },
            "optimize": {"bounds": {}},
        }
        bounds = [Bound(0.0, 1000.0)] * 4
        sig_digits = 6

        result = config_to_individual(config, bounds, sig_digits)

        # Order: long.a_param, long.z_param, short.a_param, short.z_param
        assert result[0] == pytest.approx(200.0)
        assert result[1] == pytest.approx(100.0)
        assert result[2] == pytest.approx(400.0)
        assert result[3] == pytest.approx(300.0)

    def test_appends_live_hsl_values_when_present_in_bounds(self):
        config = {
            "bot": {
                "long": {"a_param": 1.0, "z_param": 2.0, "hsl_ema_span_minutes": 60.0},
                "short": {"a_param": 3.0, "z_param": 4.0, "hsl_red_threshold": 0.22},
            },
            "optimize": {
                "bounds": {
                    "long_a_param": [0.0, 1000.0],
                    "long_z_param": [0.0, 1000.0],
                    "short_a_param": [0.0, 1000.0],
                    "short_z_param": [0.0, 1000.0],
                    "long_hsl_ema_span_minutes": [30.0, 180.0, 5.0],
                    "short_hsl_red_threshold": [0.15, 0.35, 0.01],
                }
            },
        }
        bounds = [Bound(0.0, 1000.0)] * 6
        sig_digits = 6

        result = config_to_individual(config, bounds, sig_digits)

        assert result == pytest.approx([1.0, 60.0, 2.0, 3.0, 0.22, 4.0])

    def test_uses_precomputed_key_paths(self):
        config = {
            "bot": {
                "long": {"param1": 1.5, "param2": 2.5},
                "short": {"param1": 3.5, "param2": 4.5},
            }
        }
        bounds = [Bound(0.0, 10.0) for _ in range(4)]
        key_paths = [
            ("long_param2", ("bot", "long", "param2")),
            ("long_param1", ("bot", "long", "param1")),
            ("short_param2", ("bot", "short", "param2")),
            ("short_param1", ("bot", "short", "param1")),
        ]

        result = config_to_individual(config, bounds, sig_digits=6, key_paths=key_paths)

        assert result == pytest.approx([2.5, 1.5, 4.5, 3.5])

    def test_uses_optimization_shape(self):
        config = load_prepared_config("configs/examples/ema_anchor.json", verbose=False)
        shape = build_optimization_shape(config)

        result = config_to_individual(config, shape.bounds, optimization_shape=shape)

        assert len(result) == len(shape.bounds)


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

    def test_nested_bounds_fix_non_tuned_params(self):
        config = {
            "live": {"strategy_kind": "ema_anchor"},
            "optimize": {
                "bounds": {
                    "long": {
                        "risk": {
                            "total_wallet_exposure_limit": [0.35, 0.8, 0.01],
                        },
                        "strategy": {
                            "ema_anchor": {
                                "base_qty_pct": [0.006, 0.03, 0.0005],
                                "offset": [0.006, 0.02, 0.0005],
                            }
                        },
                    }
                }
            },
            "bot": {
                "long": {
                    "risk": {"total_wallet_exposure_limit": 0.55},
                    "strategy": {"ema_anchor": {"base_qty_pct": 0.012, "offset": 0.009}},
                }
            },
        }

        apply_fine_tune_bounds(config, ["long_offset"], set())

        assert config["optimize"]["bounds"]["long"]["strategy"]["ema_anchor"]["offset"] == [
            0.006,
            0.02,
            0.0005,
        ]
        assert config["optimize"]["bounds"]["long"]["strategy"]["ema_anchor"]["base_qty_pct"] == [
            0.012,
            0.012,
        ]
        assert config["optimize"]["bounds"]["long"]["risk"]["total_wallet_exposure_limit"] == [
            0.55,
            0.55,
        ]

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

    def test_config_fixed_params_fix_only_listed_bounds(self):
        config = {
            "optimize": {
                "bounds": {
                    "long_param1": [0.0, 1.0],
                    "long_param2": [0.0, 1.0],
                },
                "fixed_params": ["long_param2"],
            },
            "bot": {
                "long": {"param1": 0.5, "param2": 0.7},
            },
        }
        apply_fine_tune_bounds(config, [], set())

        assert config["optimize"]["bounds"]["long_param1"] == [0.0, 1.0]
        assert config["optimize"]["bounds"]["long_param2"] == [0.7, 0.7]

    def test_config_fixed_params_support_pside_hsl_keys(self):
        config = {
            "optimize": {
                "bounds": {
                    "long_hsl_red_threshold": [0.1, 0.3],
                },
                "fixed_params": ["long_hsl_red_threshold"],
            },
            "bot": {
                "long": {
                    "hsl_red_threshold": 0.22,
                },
                "short": {},
            },
        }
        apply_fine_tune_bounds(config, [], set())

        assert config["optimize"]["bounds"]["long_hsl_red_threshold"] == [0.22, 0.22]

    def test_fine_tune_and_fixed_params_share_single_effective_fixing_path(self):
        config = {
            "optimize": {
                "bounds": {
                    "long_param1": [0.0, 1.0],
                    "long_param2": [0.0, 1.0],
                    "long_param3": [0.0, 1.0],
                },
                "fixed_params": ["long_param3"],
            },
            "bot": {
                "long": {"param1": 0.5, "param2": 0.7, "param3": 0.9},
            },
        }
        apply_fine_tune_bounds(config, ["long_param1", "long_param3"], set())

        assert config["optimize"]["bounds"]["long_param1"] == [0.0, 1.0]
        assert config["optimize"]["bounds"]["long_param2"] == [0.7, 0.7]
        assert config["optimize"]["bounds"]["long_param3"] == [0.9, 0.9]


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
            assert "live" not in result[0]
        finally:
            os.unlink(path)

    def test_json_file_with_malformed_optimize_limits_still_loads_bot(self):
        from config_utils import get_template_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            template = get_template_config()
            import json

            template["optimize"]["limits"] = [{"metric": "drawdown_worst_hsl", "enabled": False}]
            f.write(json.dumps(template))
            path = f.name
        try:
            result = extract_configs(path)
            assert len(result) == 1
            assert "bot" in result[0]
            assert result[0]["bot"]["long"]["forager"]["score_weights"]["volatility"] == 1.0
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
            assert "bot" in result[0]
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

    def test_bot_only_config(self):
        from config_utils import get_template_config

        config = get_template_config()
        from optimization.config_adapter import extract_bounds_tuple_list_from_config

        bounds = extract_bounds_tuple_list_from_config(config)
        result = configs_to_individuals([deepcopy(config["bot"])], bounds, 6)

        assert len(result) >= 1
        assert len(result[0]) == len(bounds)

    def test_deduplicates_quantized_seed_individuals(self):
        from config_utils import get_template_config

        config = get_template_config()
        from optimization.config_adapter import extract_bounds_tuple_list_from_config

        bounds = extract_bounds_tuple_list_from_config(config)

        result = configs_to_individuals([config, deepcopy(config)], bounds, 6)

        assert len(result) == 1

    def test_invalid_config_logged(self):
        invalid_config = {"invalid": "structure"}
        bounds = [(0.0, 10.0, 0.0)] * 4
        result = configs_to_individuals([invalid_config], bounds, 6)

        assert len(result) == 0

    def test_full_config_seed_preserves_strategy_kind(self):
        config = load_prepared_config("configs/examples/ema_anchor.json", verbose=False)
        from optimization.config_adapter import extract_bounds_tuple_list_from_config

        bounds = extract_bounds_tuple_list_from_config(config)

        result = configs_to_individuals([config], bounds, 6)

        assert len(result) >= 1
        assert len(result[0]) == len(bounds)

    def test_old_starting_seed_inherits_current_template_bounds_and_defaults(self):
        config = load_prepared_config("configs/examples/ema_anchor.json", verbose=False)
        shape = build_optimization_shape(config)

        stale_seed = deepcopy(config)
        stale_seed["optimize"]["bounds"]["long"]["risk"].pop("entry_cooldown_minutes")
        stale_seed["optimize"]["bounds"]["short"]["risk"].pop("entry_cooldown_minutes")
        stale_seed["optimize"]["bounds"]["long"]["strategy"]["ema_anchor"].pop(
            "entry_double_down_factor"
        )
        stale_seed["optimize"]["bounds"]["short"]["strategy"]["ema_anchor"].pop(
            "entry_double_down_factor"
        )
        stale_seed["bot"]["long"]["risk"].pop("entry_cooldown_minutes")
        stale_seed["bot"]["short"]["risk"].pop("entry_cooldown_minutes")
        stale_seed["bot"]["long"]["strategy"]["ema_anchor"].pop("entry_double_down_factor")
        stale_seed["bot"]["short"]["strategy"]["ema_anchor"].pop("entry_double_down_factor")

        result = configs_to_individuals(
            [stale_seed],
            shape.bounds,
            6,
            optimization_shape=shape,
        )

        assert len(result) == 1
        assert len(result[0]) == len(shape.bounds)


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

    def test_pymoo_record_entry_is_canonical_and_mirrored(self):
        template = load_prepared_config("configs/examples/ema_anchor.json", verbose=False)
        bounds = extract_bounds_tuple_list_from_config(template)
        vector = config_to_individual(template, bounds, sig_digits=6)

        entry = build_pymoo_record_entry(
            vector=vector,
            metrics={"objectives": {"metric1": 0.5}, "constraint_violation": 0.0},
            template=template,
            build_config_fn=individual_to_config,
            overrides_fn=optimize.optimizer_overrides,
            overrides_list=["mirror_short_from_long"],
        )
        strategy_kind = entry["live"]["strategy_kind"]

        assert entry["bot"]["long"]["risk"] == entry["bot"]["short"]["risk"]
        assert entry["bot"]["long"]["forager"] == entry["bot"]["short"]["forager"]
        assert entry["bot"]["long"]["hsl"] == entry["bot"]["short"]["hsl"]
        assert entry["bot"]["long"]["unstuck"] == entry["bot"]["short"]["unstuck"]
        assert entry["bot"]["long"]["strategy"][strategy_kind] == entry["bot"]["short"]["strategy"][
            strategy_kind
        ]
        assert sorted(entry["bot"]["long"]["strategy"]) == [strategy_kind]
        assert sorted(entry["bot"]["short"]["strategy"]) == [strategy_kind]

    def test_recorded_pareto_entry_is_canonical_and_mirrored(self):
        class Candidate(list):
            pass

        config = load_prepared_config("configs/examples/ema_anchor.json", verbose=False)
        bounds = extract_bounds_tuple_list_from_config(config)
        individual = Candidate(config_to_individual(config, bounds, sig_digits=6))

        individual.evaluation_metrics = {
            "objectives": {"metric1": 0.5},
            "constraint_violation": 0.0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ResultRecorder(
                results_dir=tmpdir,
                sig_digits=6,
                flush_interval=60,
                scoring_keys=["metric1"],
                compress=False,
                write_all_results=False,
                bounds=None,
            )

            _record_individual_result(
                individual,
                config,
                config["optimize"]["enable_overrides"],
                recorder,
            )

            pareto_files = list((Path(tmpdir) / "pareto").glob("*.json"))
            assert len(pareto_files) == 1
            saved = json.loads(pareto_files[0].read_text())
            strategy_kind = saved["live"]["strategy_kind"]

            assert saved["bot"]["long"]["risk"] == saved["bot"]["short"]["risk"]
            assert saved["bot"]["long"]["forager"] == saved["bot"]["short"]["forager"]
            assert saved["bot"]["long"]["hsl"] == saved["bot"]["short"]["hsl"]
            assert saved["bot"]["long"]["unstuck"] == saved["bot"]["short"]["unstuck"]
            assert (
                saved["bot"]["long"]["strategy"][strategy_kind]
                == saved["bot"]["short"]["strategy"][strategy_kind]
            )
            assert sorted(saved["bot"]["long"]["strategy"]) == [strategy_kind]
            assert sorted(saved["bot"]["short"]["strategy"]) == [strategy_kind]

    def test_recorded_pareto_entry_with_bounds_preserves_mirror_override(self):
        template = load_prepared_config("configs/examples/ema_anchor.json", verbose=False)
        bounds = extract_bounds_tuple_list_from_config(template)
        vector = config_to_individual(template, bounds, sig_digits=6)

        entry = build_pymoo_record_entry(
            vector=vector,
            metrics={"objectives": {"metric1": 0.5}, "constraint_violation": 0.0},
            template=template,
            build_config_fn=individual_to_config,
            overrides_fn=optimize.optimizer_overrides,
            overrides_list=["mirror_short_from_long"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = ResultRecorder(
                results_dir=tmpdir,
                sig_digits=6,
                flush_interval=60,
                scoring_keys=["metric1"],
                compress=False,
                write_all_results=False,
                bounds=bounds,
            )
            recorder.record(entry)

            pareto_files = list((Path(tmpdir) / "pareto").glob("*.json"))
            assert len(pareto_files) == 1
            saved = json.loads(pareto_files[0].read_text())
            strategy_kind = saved["live"]["strategy_kind"]

            assert saved["bot"]["long"]["risk"] == saved["bot"]["short"]["risk"]
            assert saved["bot"]["long"]["forager"] == saved["bot"]["short"]["forager"]
            assert saved["bot"]["long"]["hsl"] == saved["bot"]["short"]["hsl"]
            assert saved["bot"]["long"]["unstuck"] == saved["bot"]["short"]["unstuck"]
            assert (
                saved["bot"]["long"]["strategy"][strategy_kind]
                == saved["bot"]["short"]["strategy"][strategy_kind]
            )

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

    def test_evaluate_converts_recoverable_backtest_panic_to_penalty(self):
        from optimize import Evaluator, INVALID_BACKTEST_CANDIDATE_PENALTY
        from config_utils import get_template_config

        class PanicException(Exception):
            pass

        class DummyIndividual(list):
            pass

        mock_config = get_template_config()
        mock_config["optimize"]["limits"] = []
        mock_config["optimize"]["scoring"] = ["adg_pnl_w", "drawdown_worst_usd"]

        evaluator = Evaluator(
            hlcvs_specs={"binance": object()},
            btc_usd_specs={},
            msss={"binance": {}},
            config=mock_config,
            timestamps={"binance": None},
        )
        evaluator.shared_hlcvs_np["binance"] = np.zeros((1, 1, 5))
        evaluator.shared_btc_np["binance"] = None

        individual = DummyIndividual(
            config_to_individual(mock_config, evaluator.bounds, evaluator.sig_digits)
        )

        with patch("optimize.build_backtest_payload", return_value=object()), patch(
            "optimize.execute_backtest",
            side_effect=PanicException(
                "hard-stop evaluation failed at k 1 ts 2 equity -1 peak_strategy_equity 10: equity must be finite and > 0"
            ),
        ):
            objectives, penalty, metrics = evaluator.evaluate(individual, [])

        assert objectives == (0.0, 0.0)
        assert penalty == INVALID_BACKTEST_CANDIDATE_PENALTY
        assert metrics["constraint_violation"] == INVALID_BACKTEST_CANDIDATE_PENALTY
        assert "PanicException" in metrics["error"]
        assert metrics["stats"] == {}

    def test_suite_evaluate_converts_recoverable_backtest_panic_to_penalty(self):
        from optimize import Evaluator, SuiteEvaluator, INVALID_BACKTEST_CANDIDATE_PENALTY
        from config_utils import get_template_config

        class PanicException(Exception):
            pass

        class DummyIndividual(list):
            pass

        mock_config = get_template_config()
        mock_config["optimize"]["limits"] = []
        mock_config["optimize"]["scoring"] = ["adg_pnl_w"]

        base = Evaluator(
            hlcvs_specs={},
            btc_usd_specs={},
            msss={},
            config=mock_config,
        )
        ctx = ScenarioEvalContext(
            label="test",
            config=deepcopy(mock_config),
            exchanges=["binance"],
            hlcvs_specs={},
            btc_usd_specs={},
            msss={"binance": {}},
            timestamps={"binance": None},
            shared_hlcvs_np={"binance": np.zeros((1, 1, 5))},
            shared_btc_np={},
            attachments={"hlcvs": {}, "btc": {}},
            coin_indices={"binance": None},
            overrides={},
        )
        ctx.config["backtest"]["coins"] = {}
        evaluator = SuiteEvaluator(base, [ctx], {})
        individual = DummyIndividual(config_to_individual(mock_config, base.bounds, base.sig_digits))

        with patch("optimize.build_backtest_payload", return_value=object()), patch(
            "optimize.execute_backtest",
            side_effect=PanicException(
                "hard-stop evaluation failed at k 1 ts 2 equity -1 peak_strategy_equity 10: equity must be finite and > 0"
            ),
        ):
            objectives, penalty, metrics = evaluator.evaluate(individual, [])

        assert objectives == (0.0,)
        assert penalty == INVALID_BACKTEST_CANDIDATE_PENALTY
        assert metrics["constraint_violation"] == INVALID_BACKTEST_CANDIDATE_PENALTY
        assert "PanicException" in metrics["error"]
        assert metrics["suite_metrics"] == {}
