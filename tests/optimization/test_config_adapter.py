from optimization.bounds import Bound
from optimization.config_adapter import extract_bounds_tuple_list_from_config
from optimization.config_adapter import get_optimization_key_paths
import pytest


class TestConfigAdapter:
    """Test configuration adapter functions."""

    def test_extract_bounds_tuple_list_from_config(self):
        config = {
            "optimize": {
                "bounds": {
                    "long_n_positions": [1.0, 1.0],
                    "long_total_wallet_exposure_limit": [0.1, 0.1],
                    "short_n_positions": [0.0, 0.0],
                    "short_total_wallet_exposure_limit": [0.0, 0.0],
                }
            }
        }
        # We need more bounds to satisfy extract_bounds_tuple_list_from_config
        # because it iterates over template_config["bot"]
        from config_utils import get_template_config

        template = get_template_config()
        for pside in template["bot"]:
            for key in template["bot"][pside]:
                value = template["bot"][pside][key]
                if isinstance(value, dict) or isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                bound_key = f"{pside}_{key}"
                if bound_key not in config["optimize"]["bounds"]:
                    if bound_key == "short_unstuck_ema_dist":
                        config["optimize"]["bounds"][bound_key] = [0.0, 0.99]
                    else:
                        config["optimize"]["bounds"][bound_key] = [0.0, 1.0]

        bounds = extract_bounds_tuple_list_from_config(config)
        assert len(bounds) > 0
        assert isinstance(bounds[0], Bound)

        # Find index for a short parameter
        short_param_idx = None
        current_idx = 0
        for pside in sorted(template["bot"]):
            for key in sorted(template["bot"][pside]):
                if pside == "short":
                    short_param_idx = current_idx
                    break
                current_idx += 1
            if short_param_idx is not None:
                break

        # Short should be disabled (fixed to low)
        assert bounds[short_param_idx].low == bounds[short_param_idx].high

    def test_get_optimization_key_paths_includes_pside_hsl_keys_when_bounded(self):
        config = {
            "bot": {
                "long": {
                    "n_positions": 1.0,
                    "total_wallet_exposure_limit": 1.0,
                    "hsl_red_threshold": 0.25,
                    "hsl_ema_span_minutes": 60.0,
                },
                "short": {
                    "n_positions": 1.0,
                    "total_wallet_exposure_limit": 1.0,
                    "hsl_red_threshold": 0.25,
                    "hsl_ema_span_minutes": 60.0,
                },
            },
            "optimize": {
                "bounds": {
                    "long_n_positions": [1.0, 2.0, 1.0],
                    "long_total_wallet_exposure_limit": [1.0, 2.0, 0.1],
                    "short_n_positions": [1.0, 2.0, 1.0],
                    "short_total_wallet_exposure_limit": [1.0, 2.0, 0.1],
                    "long_hsl_red_threshold": [0.15, 0.35, 0.01],
                    "short_hsl_ema_span_minutes": [30.0, 180.0, 5.0],
                }
            },
        }

        key_paths = get_optimization_key_paths(config)

        assert (
            "long_hsl_red_threshold",
            ("bot", "long", "hsl_red_threshold"),
        ) in key_paths
        assert (
            "short_hsl_ema_span_minutes",
            ("bot", "short", "hsl_ema_span_minutes"),
        ) in key_paths

    def test_get_optimization_key_paths_skips_missing_side_configs(self):
        config = {
            "bot": {
                "long": {
                    "a": 0.1,
                    "b": 0.2,
                }
            },
            "optimize": {"bounds": {}},
        }

        key_paths = get_optimization_key_paths(config)

        assert key_paths == [
            ("long_a", ("bot", "long", "a")),
            ("long_b", ("bot", "long", "b")),
        ]

    @pytest.mark.parametrize(
        ("bound_key", "bound_value", "expected"),
        [
            (
                "long_unstuck_ema_dist",
                [-1.0, -0.5, 0.01],
                r"optimize\.bounds\.long_unstuck_ema_dist lower bound must be > -1\.0",
            ),
            (
                "short_unstuck_ema_dist",
                [-0.99, 1.0, 0.01],
                r"optimize\.bounds\.short_unstuck_ema_dist upper bound must be < 1\.0",
            ),
        ],
    )
    def test_get_optimization_key_paths_rejects_invalid_unstuck_ema_dist_bounds(
        self, bound_key, bound_value, expected
    ):
        config = {
            "bot": {
                "long": {
                    "n_positions": 1.0,
                    "total_wallet_exposure_limit": 1.0,
                    "unstuck_ema_dist": -0.2,
                },
                "short": {
                    "n_positions": 1.0,
                    "total_wallet_exposure_limit": 1.0,
                    "unstuck_ema_dist": -0.2,
                },
            },
            "optimize": {
                "bounds": {
                    "long_n_positions": [1.0, 2.0, 1.0],
                    "long_total_wallet_exposure_limit": [1.0, 2.0, 0.1],
                    "short_n_positions": [1.0, 2.0, 1.0],
                    "short_total_wallet_exposure_limit": [1.0, 2.0, 0.1],
                    bound_key: bound_value,
                }
            },
        }

        with pytest.raises(ValueError, match=expected):
            get_optimization_key_paths(config)
