"""Tests for Optuna optimizer config loading."""
import json
from unittest.mock import patch


class TestExtractBounds:
    def test_extract(self):
        from optuna_optimizer.config import extract_bounds
        config = {"optimize": {"bounds": {
            "long_ema_span_0": [200, 1440, 10],
            "short_enabled": [0, 0],
        }}}
        bounds = extract_bounds(config)
        assert bounds["long_ema_span_0"].step == 10
        assert bounds["short_enabled"].is_fixed


class TestExtractConstraints:
    def test_extract(self):
        from optuna_optimizer.config import extract_constraints
        config = {"optimize": {"constraints": [
            {"metric": "drawdown_worst", "max": 0.5},
        ]}}
        constraints = extract_constraints(config)
        assert len(constraints) == 1
        assert constraints[0].max == 0.5


class TestExtractOptunaConfig:
    def test_extract(self):
        from optuna_optimizer.config import extract_optuna_config
        config = {"optimize": {"optuna": {"n_trials": 5000, "sampler": {"name": "nsgaii", "population_size": 100}}}}
        cfg = extract_optuna_config(config)
        assert cfg.n_trials == 5000
        assert cfg.sampler.name == "nsgaii"


class TestExtractObjectives:
    def test_extract(self):
        from optuna_optimizer.config import extract_objectives
        config = {"optimize": {"objectives": [
            {"metric": "adg_pnl", "direction": "maximize"},
            {"metric": "drawdown_worst", "direction": "minimize"},
        ]}}
        objectives = extract_objectives(config)
        assert len(objectives) == 2
        assert objectives[0].metric == "adg_pnl"
        assert objectives[0].direction == "maximize"
        assert objectives[0].sign == -1.0
        assert objectives[1].metric == "drawdown_worst"
        assert objectives[1].direction == "minimize"
        assert objectives[1].sign == 1.0

    def test_empty(self):
        from optuna_optimizer.config import extract_objectives
        config = {"optimize": {}}
        objectives = extract_objectives(config)
        assert objectives == []


class TestApplyParamsToConfig:
    def test_long_prefix_goes_to_bot_long(self):
        from optuna_optimizer.config import apply_params_to_config
        params = {"long_entry_grid_spacing_pct": 0.01}
        base = {}
        result = apply_params_to_config(params, base)
        assert result["bot"]["long"]["entry_grid_spacing_pct"] == 0.01

    def test_short_prefix_goes_to_bot_short(self):
        from optuna_optimizer.config import apply_params_to_config
        params = {"short_entry_grid_spacing_pct": 0.02}
        base = {}
        result = apply_params_to_config(params, base)
        assert result["bot"]["short"]["entry_grid_spacing_pct"] == 0.02

    def test_no_prefix_goes_to_live(self):
        from optuna_optimizer.config import apply_params_to_config
        params = {"leverage": 5}
        base = {}
        result = apply_params_to_config(params, base)
        assert result["live"]["leverage"] == 5

    def test_preserves_existing_config(self):
        from optuna_optimizer.config import apply_params_to_config
        params = {"long_qty": 0.1}
        base = {"backtest": {"start_date": "2024-01-01"}, "live": {"other": "value"}}
        result = apply_params_to_config(params, base)
        assert result["backtest"]["start_date"] == "2024-01-01"
        assert result["live"]["other"] == "value"
        assert result["bot"]["long"]["qty"] == 0.1

    def test_does_not_mutate_base(self):
        from optuna_optimizer.config import apply_params_to_config
        params = {"long_qty": 0.1}
        base = {"live": {"existing": "value"}}
        result = apply_params_to_config(params, base)
        assert "bot" not in base
        assert result is not base

    def test_mixed_params(self):
        from optuna_optimizer.config import apply_params_to_config
        params = {
            "long_entry_qty": 0.1,
            "short_entry_qty": 0.2,
            "leverage": 10,
        }
        base = {}
        result = apply_params_to_config(params, base)
        assert result["bot"]["long"]["entry_qty"] == 0.1
        assert result["bot"]["short"]["entry_qty"] == 0.2
        assert result["live"]["leverage"] == 10


class TestExtractParamsFromConfig:
    def test_extracts_long_params(self):
        from optuna_optimizer.config import extract_params_from_config
        from optuna_optimizer.models import Bound

        config = {"bot": {"long": {"ema_span_0": 500, "n_positions": 10}}}
        bounds = {
            "long_ema_span_0": Bound(low=200, high=1000),
            "long_n_positions": Bound(low=1, high=20),
        }

        result = extract_params_from_config(config, bounds)

        assert result == {"long_ema_span_0": 500, "long_n_positions": 10}

    def test_extracts_short_params(self):
        from optuna_optimizer.config import extract_params_from_config
        from optuna_optimizer.models import Bound

        config = {"bot": {"short": {"ema_span_0": 300}}}
        bounds = {"short_ema_span_0": Bound(low=100, high=500)}

        result = extract_params_from_config(config, bounds)

        assert result == {"short_ema_span_0": 300}

    def test_filters_to_bounds_only(self):
        from optuna_optimizer.config import extract_params_from_config
        from optuna_optimizer.models import Bound

        config = {"bot": {"long": {"ema_span_0": 500, "unknown_param": 999}}}
        bounds = {"long_ema_span_0": Bound(low=200, high=1000)}

        result = extract_params_from_config(config, bounds)

        assert result == {"long_ema_span_0": 500}
        assert "long_unknown_param" not in result

    def test_clamps_to_bounds(self):
        from optuna_optimizer.config import extract_params_from_config
        from optuna_optimizer.models import Bound

        config = {"bot": {"long": {"ema_span_0": 9999}}}  # Way above max
        bounds = {"long_ema_span_0": Bound(low=200, high=1000)}

        result = extract_params_from_config(config, bounds)

        assert result == {"long_ema_span_0": 1000}  # Clamped to max

    def test_returns_none_for_missing_bot(self):
        from optuna_optimizer.config import extract_params_from_config
        from optuna_optimizer.models import Bound

        config = {"live": {"some_param": 1}}
        bounds = {"long_ema_span_0": Bound(low=200, high=1000)}

        result = extract_params_from_config(config, bounds)

        assert result is None

    def test_handles_mixed_long_short(self):
        from optuna_optimizer.config import extract_params_from_config
        from optuna_optimizer.models import Bound

        config = {
            "bot": {
                "long": {"ema_span_0": 500},
                "short": {"ema_span_0": 300},
            }
        }
        bounds = {
            "long_ema_span_0": Bound(low=200, high=1000),
            "short_ema_span_0": Bound(low=100, high=500),
        }

        result = extract_params_from_config(config, bounds)

        assert result == {"long_ema_span_0": 500, "short_ema_span_0": 300}


class TestLoadSeedConfigs:
    def test_loads_single_json_file(self, tmp_path):
        from optuna_optimizer.config import load_seed_configs
        from unittest.mock import MagicMock

        config_file = tmp_path / "config.json"
        config_file.write_text("{}")  # Content doesn't matter, we mock the loader

        mock_loader = MagicMock(return_value={"bot": {"long": {"ema_span_0": 500}}})
        result = load_seed_configs(config_file, mock_loader)

        mock_loader.assert_called_once_with(str(config_file))
        assert len(result) == 1
        assert result[0]["bot"]["long"]["ema_span_0"] == 500

    def test_loads_directory_of_configs(self, tmp_path):
        from optuna_optimizer.config import load_seed_configs
        from unittest.mock import MagicMock

        (tmp_path / "a.json").write_text("{}")
        (tmp_path / "b.json").write_text("{}")

        mock_loader = MagicMock(side_effect=[
            {"bot": {"long": {"x": 1}}},
            {"bot": {"long": {"x": 2}}},
        ])
        result = load_seed_configs(tmp_path, mock_loader)

        assert mock_loader.call_count == 2
        assert len(result) == 2

    def test_skips_non_json_files(self, tmp_path):
        from optuna_optimizer.config import load_seed_configs
        from unittest.mock import MagicMock

        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "readme.txt").write_text("not json")
        (tmp_path / "data.bin").write_bytes(b"\x00\x01\x02")

        mock_loader = MagicMock(return_value={"bot": {}})
        result = load_seed_configs(tmp_path, mock_loader)

        # Only the .json file should be loaded
        mock_loader.assert_called_once()
        assert len(result) == 1

    def test_recurses_into_subdirectories(self, tmp_path):
        from optuna_optimizer.config import load_seed_configs
        from unittest.mock import MagicMock

        subdir = tmp_path / "pareto"
        subdir.mkdir()
        (subdir / "best.json").write_text("{}")

        mock_loader = MagicMock(return_value={"bot": {"long": {"x": 1}}})
        result = load_seed_configs(tmp_path, mock_loader)

        mock_loader.assert_called_once()
        assert len(result) == 1

    def test_returns_empty_for_nonexistent_path(self, tmp_path):
        from optuna_optimizer.config import load_seed_configs
        from unittest.mock import MagicMock

        mock_loader = MagicMock()
        result = load_seed_configs(tmp_path / "does_not_exist", mock_loader)

        assert result == []

    def test_skips_invalid_config(self, tmp_path):
        from optuna_optimizer.config import load_seed_configs
        from unittest.mock import MagicMock

        (tmp_path / "good.json").write_text("{}")
        (tmp_path / "bad.json").write_text("{}")

        mock_loader = MagicMock(side_effect=[
            {"bot": {}},
            Exception("invalid config"),
        ])
        result = load_seed_configs(tmp_path, mock_loader)

        assert len(result) == 1

    def test_supports_hjson_extension(self, tmp_path):
        from optuna_optimizer.config import load_seed_configs
        from unittest.mock import MagicMock

        (tmp_path / "config.hjson").write_text("{}")

        mock_loader = MagicMock(return_value={"bot": {}})
        result = load_seed_configs(tmp_path, mock_loader)

        mock_loader.assert_called_once()
        assert len(result) == 1
