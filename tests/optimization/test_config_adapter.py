import passivbot_rust as pbr
import pytest

import config.strategy_spec as strategy_spec
from config.strategy_paths import native_strategy_path_to_public
from config import prepare_config
from config.schema import get_template_config
from config.strategy_spec import get_supported_strategy_kinds
from optimization.bounds import Bound
from optimization.config_adapter import extract_bounds_tuple_list_from_config
from optimization.config_adapter import get_optimization_key_paths
from optimization.config_adapter import get_strategy_spec
from optimization.config_adapter import resolve_optimization_bound_path


def _get_path(mapping, path):
    current = mapping
    for part in path:
        current = current[part]
    return current


def _canonical_config_with_bounds(bounds, *, strategy_kind="trailing_martingale"):
    config = get_template_config()
    config['bot']['global']['strategy_kind'] = strategy_kind
    config["optimize"]["bounds"] = bounds
    for side in ("long", "short"):
        config["bot"][side]["risk"].update(max_positions=1.0, total_wallet_exposure_limit=1.0)
    return config


class TestConfigAdapter:
    """Test configuration adapter functions."""

    def test_existing_bot_does_not_construct_a_fallback_template(self, monkeypatch):
        import optimization.config_adapter as adapter

        config = get_template_config()
        expected = get_optimization_key_paths(config)

        def unexpected_template():
            raise AssertionError("existing bot config must not request fallback defaults")

        monkeypatch.setattr(adapter, "get_template_config", unexpected_template)
        assert get_optimization_key_paths(config) == expected

    def test_extract_bounds_tuple_list_from_config(self):
        config = _canonical_config_with_bounds({
            "long": {"risk": {"max_positions": [1.0, 1.0],
                              "total_wallet_exposure_limit": [0.1, 0.1]},
                     "unstuck": {"close_fraction": [0.01, 0.12]}},
            "short": {"risk": {"max_positions": [0.0, 0.0],
                               "total_wallet_exposure_limit": [0.0, 0.0]},
                      "unstuck": {"close_fraction": [0.01, 0.12]}},
        })
        bounds = extract_bounds_tuple_list_from_config(config)
        key_paths = get_optimization_key_paths(config)
        assert len(bounds) > 0
        assert isinstance(bounds[0], Bound)

        short_param_idx = next(
            idx for idx, (key, _) in enumerate(key_paths) if key == "short_unstuck_close_pct"
        )
        assert bounds[short_param_idx].low == bounds[short_param_idx].high == 0.01
        long_param_idx = next(idx for idx, (key, _) in enumerate(key_paths)
                              if key == "long_unstuck_close_pct")
        assert bounds[long_param_idx].high == 0.12

    def test_get_optimization_key_paths_includes_pside_hsl_keys_when_bounded(self):
        config = _canonical_config_with_bounds({
            "long": {"hsl": {"drawdown_ratio_threshold": [0.15, 0.35, 0.01],
                             "memory_ema_span_minutes": [100.0, 200.0, 1.0]}},
            "short": {"hsl": {"drawdown_ratio_ema_span_minutes": [30.0, 180.0, 5.0]}},
        })
        config["bot"]["global"]["hsl"]["scope"] = "position_side"

        key_paths = get_optimization_key_paths(config)

        assert ("long_hsl_red_threshold", ("bot", "long", "hsl", "drawdown_ratio_threshold")) in key_paths
        assert ("long_hsl_memory_ema_span_minutes",
                ("bot", "long", "hsl", "memory_ema_span_minutes")) in key_paths
        assert ("short_hsl_drawdown_ratio_ema_span_minutes",
                ("bot", "short", "hsl", "drawdown_ratio_ema_span_minutes")) in key_paths

    def test_get_optimization_key_paths_rejects_empty_bounds(self):
        config = {
            "bot": {
                "long": {
                    "a": 0.1,
                    "b": 0.2,
                }
            },
            "optimize": {"bounds": {}},
        }

        with pytest.raises(
            ValueError, match="config.optimize.bounds must contain at least one optimizer bound"
        ):
            get_optimization_key_paths(config)
        with pytest.raises(
            ValueError, match="config.optimize.bounds must contain at least one optimizer bound"
        ):
            extract_bounds_tuple_list_from_config(config)

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
        side = bound_key.split("_", 1)[0]
        config = _canonical_config_with_bounds({side: {"unstuck": {"ema_distance_fraction": bound_value}}})

        with pytest.raises(ValueError, match=expected):
            get_optimization_key_paths(config)

    def test_get_strategy_spec_exposes_trailing_martingale_metadata(self):
        spec = get_strategy_spec("trailing_martingale")
        template = get_template_config()
        bot_template = template["bot"]

        assert spec["strategy_kind"] == "trailing_martingale"
        assert (
            spec["defaults"]["long"]["ema_span_0"]
            == bot_template["long"]["strategy"]["trailing_martingale"]["entry"][
                "ema_span_0"
            ]
        )
        assert spec["defaults"]["short"]["entry_threshold_base_pct"] == bot_template[
            "short"
        ]["strategy"]["trailing_martingale"]["entry"]["threshold_base_fraction"]
        assert (
            spec["optimize_bounds"]["long_ema_span_0"]
            == template["optimize"]["bounds"]["long"]["strategy"][
                "trailing_martingale"
            ]["entry"]["ema_span_0_minutes"]
        )
        assert any(
            param["config_path"] == ["strategy", "long", "entry", "ema_span_0"]
            and param["legacy_config_paths"] == []
            for param in spec["parameters"]
        )

    def test_rust_strategy_spec_python_api_matches_adapter_cache(self):
        rust_spec = pbr.get_strategy_spec("trailing_martingale")
        cached_spec = get_strategy_spec("trailing_martingale")

        from config.strategy_paths import project_strategy_spec

        assert project_strategy_spec("trailing_martingale", rust_spec) == cached_spec

    def test_rust_strategy_kind_api_is_python_source_of_truth(self):
        assert tuple(pbr.get_strategy_kinds()) == get_supported_strategy_kinds()

    def test_get_strategy_spec_exposes_ema_anchor_metadata(self):
        spec = get_strategy_spec("ema_anchor")

        assert spec["strategy_kind"] == "ema_anchor"
        assert spec["defaults"]["long"]["base_qty_pct"] == 0.01
        assert spec["defaults"]["short"]["offset"] == 0.002
        assert spec["defaults"]["long"]["offset_volatility_ema_span_1m"] == 60.0
        assert spec["optimize_bounds"]["long_offset"] == [0.0, 0.05, 0.0001]
        assert spec["optimize_bounds"]["long_offset_volatility_1m_weight"] == [0.0, 40.0, 0.1]
        assert any(
            param["config_path"] == ["strategy", "long", "base_qty_fraction"]
            and param["legacy_config_paths"] == []
            for param in spec["parameters"]
        )

    def test_get_optimization_key_paths_routes_strategy_fields_to_strategy_section(self):
        config = _canonical_config_with_bounds({
            "long": {"strategy": {"trailing_martingale": {"entry": {
                "ema_span_0": [200.0, 1440.0, 10.0],
            }}}, "hsl": {"drawdown_ratio_threshold": [0.15, 0.35, 0.01]}},
            "short": {"strategy": {"trailing_martingale": {"entry": {
                "threshold_base_fraction": [0.01, 0.04, 1e-5],
            }}}},
        })

        key_paths = get_optimization_key_paths(config)

        assert (
            "long_ema_span_0",
            ("bot", "long", "strategy", "trailing_martingale", "entry", "ema_span_0"),
        ) in key_paths
        assert (
            "short_entry_threshold_base_pct",
            (
                "bot",
                "short",
                "strategy",
                "trailing_martingale",
                "entry",
                "threshold_base_fraction",
            ),
        ) in key_paths
        assert ("long_hsl_red_threshold", ("bot", "long", "hsl", "drawdown_ratio_threshold")) in key_paths

    def test_get_optimization_key_paths_routes_ema_anchor_fields_to_strategy_section(self):
        config = _canonical_config_with_bounds({
            "long": {"strategy": {"ema_anchor": {
                "base_qty_fraction": [0.001, 0.05, 0.0001],
                "offset_volatility_1m_weight": [0.0, 10.0, 0.1],
            }}},
            "short": {"strategy": {"ema_anchor": {"offset_fraction": [0.0, 0.05, 0.0001]}}},
        }, strategy_kind="ema_anchor")

        key_paths = get_optimization_key_paths(config)

        assert ("long_base_qty_pct", ("bot", "long", "strategy", "ema_anchor", "base_qty_fraction")) in key_paths
        assert (
            "long_offset_volatility_1m_weight",
            ("bot", "long", "strategy", "ema_anchor", "offset_volatility_1m_weight"),
        ) in key_paths
        assert ("short_offset", ("bot", "short", "strategy", "ema_anchor", "offset_fraction")) in key_paths

    @pytest.mark.parametrize("strategy_kind", list(get_supported_strategy_kinds()))
    def test_strategy_optimizer_key_path_map_is_exhaustive_for_rust_spec(self, strategy_kind):
        config = get_template_config()
        config['bot']['global']['strategy_kind'] = strategy_kind
        prepared = prepare_config(config, verbose=False, target="canonical", runtime=None)
        spec = pbr.get_strategy_spec(strategy_kind)

        for param in spec["parameters"]:
            optimize_key = param["optimize_key"]
            config_path = param["config_path"]
            expected = (
                "bot",
                config_path[1],
                "strategy",
                strategy_kind,
                *native_strategy_path_to_public(strategy_kind, config_path[2:]),
            )

            resolved = resolve_optimization_bound_path(prepared, optimize_key)

            assert resolved == expected
            value = _get_path(prepared, resolved)
            assert not isinstance(value, bool)
            assert isinstance(value, (int, float))

    def test_missing_rust_strategy_spec_api_fails_loudly(self, monkeypatch):
        strategy_spec.get_strategy_spec.cache_clear()
        monkeypatch.setattr(strategy_spec.pbr, "get_strategy_spec", None)

        with pytest.raises(RuntimeError, match="Rust strategy metadata is unavailable or stale"):
            strategy_spec.get_strategy_spec("trailing_martingale")

        strategy_spec.get_strategy_spec.cache_clear()
