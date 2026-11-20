"""Structural transforms used by the explicit released-v8 converter."""

import logging

import pytest

from config.migrations.released_v8_layout import (
    migrate_btc_collateral_settings,
    migrate_empty_means_all_approved,
)
from config.migrations.renames import apply_backward_compatibility_renames, rename_config_keys
from config_transform import ConfigTransformTracker


def test_rename_config_keys_moves_legacy_fields():
    config = {
        "live": {
            "minimum_market_age_days": 12,
            "noisiness_rolling_mean_window_size": 34,
        },
        "backtest": {"exchange": "binance", "panic_market_slippage_pct": 0.0015},
    }

    rename_config_keys(config, verbose=False)

    assert "minimum_market_age_days" not in config["live"]
    assert config["live"]["minimum_coin_age_days"] == 12
    assert config["live"]["ohlcv_rolling_window"] == 34
    assert config["backtest"]["exchanges"] == ["binance"]
    assert config["backtest"]["market_order_slippage_pct"] == pytest.approx(0.0015)
    assert "exchange" not in config["backtest"]
    assert "panic_market_slippage_pct" not in config["backtest"]


def test_rename_config_keys_records_tracker_events():
    config = {
        "live": {"minimum_market_age_days": 5},
        "backtest": {"exchange": "binance", "panic_market_slippage_pct": 0.0015},
    }
    tracker = ConfigTransformTracker()

    rename_config_keys(config, verbose=False, tracker=tracker)

    summary = tracker.summary()
    assert any(
        event["action"] == "rename"
        and event["from"] == "live.minimum_market_age_days"
        and event["to"] == "live.minimum_coin_age_days"
        for event in summary
    )
    assert any(
        event["action"] == "rename"
        and event["from"] == "backtest.exchange"
        and event["to"] == "backtest.exchanges"
        for event in summary
    )
    assert any(
        event["action"] == "rename"
        and event["from"] == "backtest.panic_market_slippage_pct"
        and event["to"] == "backtest.market_order_slippage_pct"
        for event in summary
    )


def test_migrate_empty_means_all_approved_converts_global_empty_to_all(caplog):
    config = {
        "live": {
            "approved_coins": [],
            "ignored_coins": {"long": [], "short": []},
            "empty_means_all_approved": True,
        }
    }
    tracker = ConfigTransformTracker()

    with caplog.at_level(logging.WARNING):
        migrate_empty_means_all_approved(config, verbose=False, tracker=tracker)

    assert config["live"]["approved_coins"] == "all"
    assert "empty_means_all_approved" not in config["live"]
    summary = tracker.summary()
    assert any(
        event["action"] == "remove" and event["path"] == "live.empty_means_all_approved"
        for event in summary
    )
    assert any(
        event["action"] == "update"
        and event["path"] == "live.approved_coins"
        and event["new"] == "all"
        for event in summary
    )
    assert any("Converting released-v8" in rec.message for rec in caplog.records)


def test_migrate_empty_means_all_approved_keeps_explicit_per_side_values():
    config = {
        "live": {
            "approved_coins": {"long": ["BTC"], "short": []},
            "ignored_coins": {"long": [], "short": []},
            "empty_means_all_approved": True,
        }
    }

    migrate_empty_means_all_approved(config, verbose=False)

    assert config["live"]["approved_coins"] == {"long": ["BTC"], "short": []}
    assert "empty_means_all_approved" not in config["live"]


def test_migrate_btc_collateral_settings_converts_bool():
    config = {"backtest": {"use_btc_collateral": True}}
    migrate_btc_collateral_settings(config, verbose=False)
    assert config["backtest"]["btc_collateral_cap"] == pytest.approx(1.0)
    assert config["backtest"]["btc_collateral_ltv_cap"] is None
    assert "use_btc_collateral" not in config["backtest"]

    config = {"backtest": {"use_btc_collateral": False}}
    migrate_btc_collateral_settings(config, verbose=False)
    assert config["backtest"]["btc_collateral_cap"] == pytest.approx(0.0)
    assert config["backtest"]["btc_collateral_ltv_cap"] is None


def test_apply_backward_compatibility_renames_moves_filter_keys():
    config = {
        "bot": {
            "long": {
                "filter_noisiness_rolling_window": 42,
                "filter_volatility_ema_span_1m": 84,
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

    apply_backward_compatibility_renames(config, verbose=False)

    assert "filter_noisiness_rolling_window" not in config["bot"]["long"]
    assert config["bot"]["long"]["forager_volatility_ema_span_1m"] == 84
    assert config["bot"]["long"]["forager_volume_ema_span_1m"] == 21
    assert config["bot"]["short"]["forager_volume_ema_span_1m"] == 11
    bounds = config["optimize"]["bounds"]
    assert "long_filter_noisiness_rolling_window" not in bounds
    assert bounds["long_forager_volatility_ema_span_1m"] == [10, 20]
    assert "short_filter_volume_rolling_window" not in bounds
    assert bounds["short_forager_volume_ema_span_1m"] == [30, 40]
