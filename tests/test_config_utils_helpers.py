import pytest

from config_utils import (
    _apply_backward_compatibility_renames,
    _apply_non_live_adjustments,
    _ensure_bot_defaults_and_bounds,
    _ensure_enforce_exposure_limit_bool,
    _normalize_position_counts,
    _rename_config_keys,
    _sync_with_template,
    get_template_config,
)


def test_ensure_bot_defaults_and_bounds_adds_missing_values():
    config = get_template_config("v7")
    config["bot"]["long"].pop("close_trailing_qty_pct", None)
    config["optimize"]["bounds"].pop("long_close_trailing_qty_pct", None)

    _ensure_bot_defaults_and_bounds(config, verbose=False)

    assert config["bot"]["long"]["close_trailing_qty_pct"] == pytest.approx(1.0)
    assert config["optimize"]["bounds"]["long_close_trailing_qty_pct"] == [0.05, 1.0]


def test_rename_config_keys_moves_legacy_fields():
    config = {
        "live": {
            "minimum_market_age_days": 12,
            "noisiness_rolling_mean_window_size": 34,
        },
        "backtest": {"exchange": "binance"},
    }

    _rename_config_keys(config, verbose=False)

    assert "minimum_market_age_days" not in config["live"]
    assert config["live"]["minimum_coin_age_days"] == 12
    assert config["live"]["ohlcv_rolling_window"] == 34
    assert config["backtest"]["exchanges"] == ["binance"]
    assert "exchange" not in config["backtest"]


def test_sync_with_template_adds_missing_and_removes_extras():
    template = get_template_config("v7")
    result = {
        "live": {},
        "backtest": {},
        "bot": {"long": {}, "short": {}, "extra_side": {}},
        "optimize": {"bounds": {}, "limits": "", "scoring": []},
        "coin_overrides": {},
    }

    _sync_with_template(template, result, base_config_path="/tmp/base_config.json", verbose=False)

    assert "extra_side" not in result["bot"]
    assert result["live"]["base_config_path"] == "/tmp/base_config.json"
    # ensure key from template was added
    assert "close_grid_markup_end" in result["bot"]["long"]


def test_normalize_position_counts_rounds_values():
    config = {
        "bot": {
            "long": {"n_positions": 3.6},
            "short": {"n_positions": 1.2},
        }
    }

    _normalize_position_counts(config)

    assert config["bot"]["long"]["n_positions"] == 4
    assert config["bot"]["short"]["n_positions"] == 1


def test_apply_non_live_adjustments_sorts_and_filters():
    config = get_template_config("v7")
    config["live"]["approved_coins"] = "btc,eth"
    config["live"]["ignored_coins"] = {"long": ["eth"], "short": []}
    config["backtest"]["end_date"] = "2023-01-01"
    config["backtest"]["use_btc_collateral"] = False
    config["optimize"]["scoring"] = ["btc_metric", "adg"]
    config["optimize"]["limits"] = "--lower_bound_drawdown 0.3 --btc_penalty 0.1"
    config["optimize"]["bounds"]["long_entry_grid_spacing_pct"] = [0.1, 0.05]

    _apply_non_live_adjustments(config, verbose=False)

    assert config["live"]["approved_coins"]["long"] == ["btc"]
    assert config["optimize"]["scoring"] == ["adg", "metric"]
    limits = config["optimize"]["limits"]
    assert isinstance(limits, dict)
    assert "lower_bound_drawdown" not in limits
    assert "penalize_if_greater_than_drawdown" in limits
    assert limits["penalty"] == pytest.approx(0.1)
    assert config["optimize"]["bounds"]["long_entry_grid_spacing_pct"] == [0.05, 0.1]
    assert config["live"]["approved_coins"]["short"] == ["btc", "eth"]


def test_ensure_enforce_exposure_limit_bool_casts_values():
    config = {
        "bot": {
            "long": {"enforce_exposure_limit": 1},
            "short": {"enforce_exposure_limit": 0},
        }
    }

    _ensure_enforce_exposure_limit_bool(config)

    assert config["bot"]["long"]["enforce_exposure_limit"] is True
    assert config["bot"]["short"]["enforce_exposure_limit"] is False


def test_apply_backward_compatibility_renames_moves_filter_keys():
    config = {
        "bot": {
            "long": {
                "filter_noisiness_rolling_window": 42,
                "filter_log_range_ema_span": 84,
                "filter_volume_rolling_window": 21,
            },
            "short": {"filter_volume_rolling_window": 11},
        },
        "optimize": {
            "bounds": {
                "long_filter_noisiness_rolling_window": [10, 20],
                "short_filter_volume_rolling_window": [30, 40],
            }
        },
    }

    _apply_backward_compatibility_renames(config, verbose=False)

    assert "filter_noisiness_rolling_window" not in config["bot"]["long"]
    assert config["bot"]["long"]["filter_log_range_ema_span"] == 84
    assert config["bot"]["long"]["filter_volume_ema_span"] == 21
    assert config["bot"]["short"]["filter_volume_ema_span"] == 11
    bounds = config["optimize"]["bounds"]
    assert "long_filter_noisiness_rolling_window" not in bounds
    assert bounds["long_filter_log_range_ema_span"] == [10, 20]
    assert "short_filter_volume_rolling_window" not in bounds
    assert bounds["short_filter_volume_ema_span"] == [30, 40]
