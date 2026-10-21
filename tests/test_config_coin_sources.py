import copy
import pytest

from backtest import get_cache_hash
from config_utils import format_config
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
            "btc_collateral_ltv_cap": None,
            "gap_tolerance_ohlcvs_minutes": 120,
            "suite_enabled": False,
            "scenarios": []
        },
        "bot": {
            "long": {
                "risk": {
                    "max_positions": 4,
                    "total_wallet_exposure_limit": 1.0
                },
                "universe": {
                    "approved_coins": [],
                    "ignored_coins": []
                }
            },
            "short": {
                "risk": {
                    "max_positions": 0,
                    "total_wallet_exposure_limit": 0.0
                },
                "universe": {
                    "approved_coins": [],
                    "ignored_coins": []
                }
            },
            "global": {
                "universe": {
                    "minimum_coin_age_days": 30
                },
                "warmup": {
                    'span_multiplier': 0.2,
                    "max_minutes": 0
                }
            }
        },
        "live": {},
        "optimize": {
            "bounds": {},
            "scoring": []
        }
    }


def test_format_config_normalizes_coin_sources():
    cfg = _base_config()
    cfg["backtest"]["coin_sources"] = {"BTC": "binance"}
    out = format_config(copy.deepcopy(cfg), verbose=False)
    assert out["backtest"]["coin_sources"] == {"BTC": "binance"}


def test_format_config_rejects_invalid_coin_source_keys_before_pruning():
    cfg = _base_config()
    cfg["backtest"]["coin_sources"] = {"BTC": "binance", 123: None}
    before = copy.deepcopy(cfg)
    with pytest.raises(TypeError, match="coin_sources keys must be strings"):
        format_config(cfg, verbose=False)
    assert cfg == before


def test_format_config_preserves_exact_coin_source_identifiers():
    cfg = _base_config()
    cfg["backtest"]["coin_sources"] = {
        "1000ABCUSDT": "bitget",
        "ABC/USDT:USDT": "bybit",
        "bitget::ABCUSDT": "bitget",
        "BTC-USDT-SWAP": "okx",
    }

    out = format_config(copy.deepcopy(cfg), verbose=False)

    assert out["backtest"]["coin_sources"] == cfg["backtest"]["coin_sources"]


def test_cache_hash_includes_coin_sources():
    cfg = _base_config()
    cfg['bot']["long"]['universe']['approved_coins'] = ["BTC/USDT:USDT"]
    cfg['bot']["short"]['universe']['approved_coins'] = []
    cfg['bot']["long"]['universe']['ignored_coins'] = []
    cfg['bot']["short"]['universe']['ignored_coins'] = []
    hash_without = get_cache_hash(cfg, "combined")
    cfg["backtest"]["coin_sources"] = {"BTC": "binance"}
    hash_with = get_cache_hash(cfg, "combined")
    assert hash_with != hash_without


def test_cache_hash_is_independent_of_coin_sources_mapping_order():
    cfg1 = _base_config()
    cfg1['bot']["long"]['universe']['approved_coins'] = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    cfg1["backtest"]["coin_sources"] = {"BTC": "okx", "ETH": "hyperliquid"}

    cfg2 = copy.deepcopy(cfg1)
    cfg2["backtest"]["coin_sources"] = {"ETH": "hyperliquid", "BTC": "okx"}

    assert get_cache_hash(cfg1, "combined") == get_cache_hash(cfg2, "combined")


def test_cache_hash_includes_market_settings_sources():
    """Verify market_settings_sources affects the cache hash."""
    cfg = _base_config()
    cfg['bot']["long"]['universe']['approved_coins'] = ["BTC/USDT:USDT"]
    cfg['bot']["short"]['universe']['approved_coins'] = []
    cfg['bot']["long"]['universe']['ignored_coins'] = []
    cfg['bot']["short"]['universe']['ignored_coins'] = []
    hash_without = get_cache_hash(cfg, "combined")
    cfg["backtest"]["market_settings_sources"] = {"BTC": "hyperliquid"}
    hash_with = get_cache_hash(cfg, "combined")
    assert hash_with != hash_without


def test_cache_hash_differs_coin_sources_vs_market_settings_sources():
    """Verify coin_sources and market_settings_sources produce different hashes."""
    cfg = _base_config()
    cfg['bot']["long"]['universe']['approved_coins'] = ["BTC/USDT:USDT"]
    cfg['bot']["short"]['universe']['approved_coins'] = []
    cfg['bot']["long"]['universe']['ignored_coins'] = []
    cfg['bot']["short"]['universe']['ignored_coins'] = []

    cfg1 = copy.deepcopy(cfg)
    cfg1["backtest"]["coin_sources"] = {"BTC": "binance"}

    cfg2 = copy.deepcopy(cfg)
    cfg2["backtest"]["market_settings_sources"] = {"BTC": "binance"}

    hash1 = get_cache_hash(cfg1, "combined")
    hash2 = get_cache_hash(cfg2, "combined")
    assert hash1 != hash2


def test_cache_hash_includes_ohlcv_source_dir():
    cfg = _base_config()
    cfg['bot']["long"]['universe']['approved_coins'] = ["BTC/USDT:USDT"]
    cfg['bot']["short"]['universe']['approved_coins'] = []
    cfg['bot']["long"]['universe']['ignored_coins'] = []
    cfg['bot']["short"]['universe']['ignored_coins'] = []
    hash_without = get_cache_hash(cfg, "combined")
    cfg["backtest"]["ohlcv_source_dir"] = "/tmp/ohlcvs-a"
    hash_with = get_cache_hash(cfg, "combined")
    assert hash_with != hash_without
