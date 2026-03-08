from optimization.bounds import Bound
from optimization.config_adapter import extract_bounds_tuple_list_from_config
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
                bound_key = f"{pside}_{key}"
                if bound_key not in config["optimize"]["bounds"]:
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

    def test_get_optimization_key_paths_includes_live_hsl_keys_when_bounded(self):
        config = {
            "bot": {
                "long": {"n_positions": 1.0, "total_wallet_exposure_limit": 1.0},
                "short": {"n_positions": 1.0, "total_wallet_exposure_limit": 1.0},
            },
            "optimize": {
                "bounds": {
                    "long_n_positions": [1.0, 2.0, 1.0],
                    "long_total_wallet_exposure_limit": [1.0, 2.0, 0.1],
                    "short_n_positions": [1.0, 2.0, 1.0],
                    "short_total_wallet_exposure_limit": [1.0, 2.0, 0.1],
                    "common_equity_hard_stop_loss_red_threshold": [0.15, 0.35, 0.01],
                    "common_equity_hard_stop_loss_ema_span_minutes": [30.0, 180.0, 5.0],
                }
            },
        }

        key_paths = get_optimization_key_paths(config)

        assert (
            "common_equity_hard_stop_loss_red_threshold",
            ("bot", "common", "equity_hard_stop_loss", "red_threshold"),
        ) in key_paths
        assert (
            "common_equity_hard_stop_loss_ema_span_minutes",
            ("bot", "common", "equity_hard_stop_loss", "ema_span_minutes"),
        ) in key_paths
