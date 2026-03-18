from optimization.bounds import Bound
from optimization.config_adapter import extract_bounds_tuple_list_from_config
from optimization.config_adapter import get_optimization_key_paths
from config_utils import get_template_config


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
        template = get_template_config()
        for pside in template["bot"]:
            for key, value in template["bot"][pside].items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                bound_key = f"{pside}_{key}"
                if bound_key not in config["optimize"]["bounds"]:
                    config["optimize"]["bounds"][bound_key] = [0.0, 1.0]
        for pside in ("long", "short"):
            for weight_key in ("volume", "ema_readiness", "volatility"):
                bound_key = f"{pside}_forager_score_weights_{weight_key}"
                if bound_key not in config["optimize"]["bounds"]:
                    config["optimize"]["bounds"][bound_key] = [0.0, 1.0]

        bounds = extract_bounds_tuple_list_from_config(config)
        assert len(bounds) > 0
        assert isinstance(bounds[0], Bound)

        key_paths = get_optimization_key_paths(config)
        short_param_idx = next(
            idx for idx, (_bound_key, path) in enumerate(key_paths) if path[:2] == ("bot", "short")
        )

        # Short should be disabled (fixed to low)
        assert bounds[short_param_idx].low == bounds[short_param_idx].high

    def test_get_optimization_key_paths_includes_pside_hsl_keys_when_bounded(self):
        config = {
            "bot": {
                "long": {
                    "n_positions": 1.0,
                    "total_wallet_exposure_limit": 1.0,
                    "hsl_red_threshold": 0.2,
                    "hsl_ema_span_minutes": 60.0,
                },
                "short": {
                    "n_positions": 1.0,
                    "total_wallet_exposure_limit": 1.0,
                    "hsl_red_threshold": 0.25,
                    "hsl_ema_span_minutes": 90.0,
                },
            },
            "optimize": {
                "bounds": {
                    "long_n_positions": [1.0, 2.0, 1.0],
                    "long_total_wallet_exposure_limit": [1.0, 2.0, 0.1],
                    "long_hsl_red_threshold": [0.15, 0.35, 0.01],
                    "long_hsl_ema_span_minutes": [30.0, 180.0, 5.0],
                    "short_n_positions": [1.0, 2.0, 1.0],
                    "short_total_wallet_exposure_limit": [1.0, 2.0, 0.1],
                    "short_hsl_red_threshold": [0.15, 0.35, 0.01],
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

    def test_get_optimization_key_paths_maps_forager_weight_subkeys(self):
        config = {
            "bot": {
                "long": {
                    "forager_score_weights": {
                        "volume": 0.1,
                        "ema_readiness": 0.2,
                        "volatility": 0.7,
                    }
                },
                "short": {
                    "forager_score_weights": {
                        "volume": 0.3,
                        "ema_readiness": 0.4,
                        "volatility": 0.3,
                    }
                },
            },
            "optimize": {
                "bounds": {
                    "long_forager_score_weights_volume": [0.0, 1.0, 0.05],
                    "short_forager_score_weights_ema_readiness": [0.0, 1.0, 0.05],
                }
            },
        }

        key_paths = get_optimization_key_paths(config)

        assert (
            "long_forager_score_weights_volume",
            ("bot", "long", "forager_score_weights", "volume"),
        ) in key_paths
        assert (
            "short_forager_score_weights_ema_readiness",
            ("bot", "short", "forager_score_weights", "ema_readiness"),
        ) in key_paths
