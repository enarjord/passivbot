"""Test that market_settings_sources is preserved during config formatting."""

import copy

from config_utils import format_config, get_template_config
from config.schema import CONFIG_SCHEMA_VERSION


def _base_config():
    return {
        "config_version": CONFIG_SCHEMA_VERSION,
        "backtest": {
            "base_dir": "backtests",
            "compress_cache": True,
            "end_date": "2025-01-01",
            "start_date": "2024-01-01",
            "exchanges": ["binance"],
            "btc_collateral_cap": 1.0,
            "gap_tolerance_ohlcvs_minutes": 120,
        },
        "bot": {
            "long": {
                "risk": {"max_positions": 4},
                "universe": {"approved_coins": [], "ignored_coins": []},
            },
            "short": {
                "risk": {"max_positions": 4},
                "universe": {"approved_coins": [], "ignored_coins": []},
            },
            "global": {
                "universe": {"minimum_coin_age_days": 30},
                "warmup": {'span_multiplier': 0.2, "max_minutes": 0},
            },
        },
        "live": {"account": {}, "candles": {}},
        "optimize": {"bounds": {}, "scoring": []},
    }


def test_format_config_preserves_market_settings_sources():
    """market_settings_sources should be preserved during format_config."""
    cfg = _base_config()
    cfg["backtest"]["market_settings_sources"] = {"BTC": "binance", "ETH": "bybit", "SOL": "gateio"}
    out = format_config(copy.deepcopy(cfg), verbose=False)
    assert out["backtest"]["market_settings_sources"] == cfg["backtest"]["market_settings_sources"]
    assert out["backtest"]["market_settings_sources"]["BTC"] == "binance"
    assert out["backtest"]["market_settings_sources"]["ETH"] == "bybit"
    assert out["backtest"]["market_settings_sources"]["SOL"] == "gateio"


def test_enable_archive_candle_fetch_in_template():
    """enable_archive_candle_fetch should be present in live template."""
    template = get_template_config()
    assert "enable_archive_candle_fetch" in template["live"].get("candles", {})
    assert template["live"]["candles"]["enable_archive_candle_fetch"] is False


def test_enable_archive_candle_fetch_preserved():
    """enable_archive_candle_fetch should be preserved during format_config."""
    cfg = _base_config()
    cfg["live"]["candles"]["enable_archive_candle_fetch"] = True
    out = format_config(copy.deepcopy(cfg), verbose=False)
    assert out["live"]["candles"]["enable_archive_candle_fetch"] is True


def test_forager_ws_candle_defaults_in_template():
    template = get_template_config()
    assert template["live"]["candles"]["enable_forager_ws_candles"] is True
    assert template["live"]["candles"]["forager_ws_candle_rest_audit_minutes"] == 30


def test_margin_mode_preference_in_template():
    template = get_template_config()
    assert template["live"]["account"]["margin_mode_preference"] == "cross"


def test_margin_mode_preference_preserved():
    cfg = _base_config()
    cfg["live"]["account"]["margin_mode_preference"] = "cross"
    out = format_config(copy.deepcopy(cfg), verbose=False)
    assert out["live"]["account"]["margin_mode_preference"] == "cross"
