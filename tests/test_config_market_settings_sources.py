"""Test that market_settings_sources is preserved during config formatting."""
import copy

from config_utils import format_config, get_template_config


def _base_config():
    return {
        "backtest": {
            "base_dir": "backtests",
            "compress_cache": True,
            "end_date": "2025-01-01",
            "start_date": "2024-01-01",
            "exchanges": ["binance"],
            "btc_collateral_cap": 1.0,
            "gap_tolerance_ohlcvs_minutes": 120,
            "max_warmup_minutes": 0,
        },
        "bot": {
            "long": {"n_positions": 4},
            "short": {"n_positions": 4},
        },
        "live": {
            "approved_coins": {"long": [], "short": []},
            "ignored_coins": {"long": [], "short": []},
            "minimum_coin_age_days": 30,
            "warmup_ratio": 0.2,
            "max_warmup_minutes": 0,
        },
        "optimize": {"bounds": {}, "scoring": []},
    }


def test_format_config_preserves_market_settings_sources():
    """market_settings_sources should be preserved during format_config."""
    cfg = _base_config()
    cfg["backtest"]["market_settings_sources"] = {
        "BTC": "binance",
        "ETH": "bybit",
        "SOL": "gateio"
    }
    out = format_config(copy.deepcopy(cfg), verbose=False)
    assert out["backtest"]["market_settings_sources"] == cfg["backtest"]["market_settings_sources"]
    assert out["backtest"]["market_settings_sources"]["BTC"] == "binance"
    assert out["backtest"]["market_settings_sources"]["ETH"] == "bybit"
    assert out["backtest"]["market_settings_sources"]["SOL"] == "gateio"


def test_enable_archive_candle_fetch_in_template():
    """enable_archive_candle_fetch should be present in live template."""
    template = get_template_config()
    assert "enable_archive_candle_fetch" in template["live"]
    assert template["live"]["enable_archive_candle_fetch"] is False


def test_enable_archive_candle_fetch_preserved():
    """enable_archive_candle_fetch should be preserved during format_config."""
    cfg = _base_config()
    cfg["live"]["enable_archive_candle_fetch"] = True
    out = format_config(copy.deepcopy(cfg), verbose=False)
    assert out["live"]["enable_archive_candle_fetch"] is True
