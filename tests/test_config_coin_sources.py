import copy

from backtest import get_cache_hash
from config_utils import format_config


def _base_config():
    return {
        "backtest": {
            "base_dir": "backtests",
            "compress_cache": True,
            "end_date": "2025-01-01",
            "start_date": "2024-01-01",
            "exchanges": ["binance"],
            "btc_collateral_cap": 1.0,
            "btc_collateral_ltv_cap": None,
            "gap_tolerance_ohlcvs_minutes": 120,
            "max_warmup_minutes": 0,
            "suite": {
                "enabled": False,
                "include_base_scenario": False,
                "base_label": "base",
                "scenarios": [],
            },
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
        "optimize": {"bounds": {}, "scoring": [], "suite": {"enabled": False, "scenarios": []}},
    }


def test_format_config_normalizes_coin_sources():
    cfg = _base_config()
    cfg["backtest"]["coin_sources"] = {"BTC": "binance", 123: None}
    out = format_config(copy.deepcopy(cfg), verbose=False)
    assert out["backtest"]["coin_sources"] == {"BTC": "binance"}


def test_cache_hash_includes_coin_sources():
    cfg = _base_config()
    cfg["live"]["approved_coins"]["long"] = ["BTC/USDT:USDT"]
    cfg["live"]["approved_coins"]["short"] = []
    cfg["live"]["ignored_coins"]["long"] = []
    cfg["live"]["ignored_coins"]["short"] = []
    hash_without = get_cache_hash(cfg, "combined")
    cfg["backtest"]["coin_sources"] = {"BTC": "binance"}
    hash_with = get_cache_hash(cfg, "combined")
    assert hash_with != hash_without
