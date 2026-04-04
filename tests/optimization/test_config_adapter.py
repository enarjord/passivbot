import passivbot_rust as pbr

from optimization.bounds import Bound
from optimization.config_adapter import extract_bounds_tuple_list_from_config
from optimization.config_adapter import get_strategy_spec
from optimization.config_adapter import get_optimization_key_paths


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
                if isinstance(template["bot"][pside][key], dict):
                    continue
                bound_key = f"{pside}_{key}"
                if bound_key not in config["optimize"]["bounds"]:
                    config["optimize"]["bounds"][bound_key] = [0.0, 1.0]

        bounds = extract_bounds_tuple_list_from_config(config)
        key_paths = get_optimization_key_paths(config)
        assert len(bounds) > 0
        assert isinstance(bounds[0], Bound)

        # Find index for a short parameter
        short_param_idx = next(
            idx for idx, (_, path) in enumerate(key_paths) if path[:2] == ("bot", "short")
        )

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

    def test_get_strategy_spec_exposes_trailing_grid_metadata(self):
        spec = get_strategy_spec("trailing_grid")

        assert spec["strategy_kind"] == "trailing_grid"
        assert spec["defaults"]["long"]["ema_span_0"] == 770.0
        assert spec["defaults"]["short"]["entry_grid_spacing_pct"] == 0.025
        assert spec["optimize_bounds"]["long_ema_span_0"] == [200.0, 1440.0, 10.0]
        assert any(
            param["config_path"] == ["strategy", "long", "ema_span_0"]
            and param["legacy_config_paths"] == ["bot.long.ema_span_0"]
            for param in spec["parameters"]
        )

    def test_rust_strategy_spec_python_api_matches_adapter_cache(self):
        rust_spec = pbr.get_strategy_spec("trailing_grid")
        cached_spec = get_strategy_spec("trailing_grid")

        assert rust_spec == cached_spec

    def test_get_strategy_spec_exposes_ema_anchor_metadata(self):
        spec = get_strategy_spec("ema_anchor")

        assert spec["strategy_kind"] == "ema_anchor"
        assert spec["defaults"]["long"]["base_qty_pct"] == 0.01
        assert spec["defaults"]["short"]["offset"] == 0.002
        assert spec["optimize_bounds"]["long_offset"] == [0.0, 0.05, 0.0001]
        assert any(
            param["config_path"] == ["strategy", "long", "base_qty_pct"]
            and param["legacy_config_paths"] == []
            for param in spec["parameters"]
        )

    def test_get_optimization_key_paths_routes_strategy_fields_to_strategy_section(self):
        config = {
            "live": {"strategy_kind": "trailing_grid"},
            "bot": {
                "long": {
                    "n_positions": 1.0,
                    "total_wallet_exposure_limit": 1.0,
                    "hsl_red_threshold": 0.25,
                    "strategy": {
                        "trailing_grid": {"ema_span_0": 10.0, "entry_grid_spacing_pct": 0.02}
                    },
                },
                "short": {
                    "n_positions": 1.0,
                    "total_wallet_exposure_limit": 1.0,
                    "strategy": {
                        "trailing_grid": {"ema_span_0": 11.0, "entry_grid_spacing_pct": 0.03}
                    },
                },
            },
            "optimize": {
                "bounds": {
                    "long_n_positions": [1.0, 2.0, 1.0],
                    "long_total_wallet_exposure_limit": [1.0, 2.0, 0.1],
                    "short_n_positions": [1.0, 2.0, 1.0],
                    "short_total_wallet_exposure_limit": [1.0, 2.0, 0.1],
                    "long_ema_span_0": [200.0, 1440.0, 10.0],
                    "short_entry_grid_spacing_pct": [0.01, 0.04, 1e-05],
                    "long_hsl_red_threshold": [0.15, 0.35, 0.01],
                }
            },
        }

        key_paths = get_optimization_key_paths(config)

        assert (
            "long_ema_span_0",
            ("bot", "long", "strategy", "trailing_grid", "ema_span_0"),
        ) in key_paths
        assert (
            "short_entry_grid_spacing_pct",
            ("bot", "short", "strategy", "trailing_grid", "entry_grid_spacing_pct"),
        ) in key_paths
        assert ("long_hsl_red_threshold", ("bot", "long", "hsl_red_threshold")) in key_paths

    def test_get_optimization_key_paths_routes_ema_anchor_fields_to_strategy_section(self):
        config = {
            "live": {"strategy_kind": "ema_anchor"},
            "bot": {
                "long": {
                    "n_positions": 1.0,
                    "total_wallet_exposure_limit": 1.0,
                    "strategy": {
                        "ema_anchor": {
                            "base_qty_pct": 0.01,
                            "ema_span_0": 55.0,
                            "ema_span_1": 144.0,
                            "offset": 0.002,
                            "offset_psize_weight": 0.1,
                        }
                    },
                },
                "short": {
                    "n_positions": 1.0,
                    "total_wallet_exposure_limit": 1.0,
                    "strategy": {
                        "ema_anchor": {
                            "base_qty_pct": 0.02,
                            "ema_span_0": 34.0,
                            "ema_span_1": 89.0,
                            "offset": 0.003,
                            "offset_psize_weight": 0.2,
                        }
                    },
                },
            },
            "optimize": {
                "bounds": {
                    "long_n_positions": [1.0, 2.0, 1.0],
                    "long_total_wallet_exposure_limit": [1.0, 2.0, 0.1],
                    "short_n_positions": [1.0, 2.0, 1.0],
                    "short_total_wallet_exposure_limit": [1.0, 2.0, 0.1],
                    "long_base_qty_pct": [0.001, 0.05, 0.0001],
                    "short_offset": [0.0, 0.05, 0.0001],
                }
            },
        }

        key_paths = get_optimization_key_paths(config)

        assert (
            "long_base_qty_pct",
            ("bot", "long", "strategy", "ema_anchor", "base_qty_pct"),
        ) in key_paths
        assert ("short_offset", ("bot", "short", "strategy", "ema_anchor", "offset")) in key_paths
