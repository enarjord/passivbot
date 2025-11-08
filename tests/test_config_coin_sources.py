import copy

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
            "emit_legacy_metrics": False,
            "gap_tolerance_ohlcvs_minutes": 120,
            "max_warmup_minutes": 0,
            "suite": {"enabled": False, "include_base_scenario": False, "base_label": "base", "scenarios": []},
        },
        "bot": {
            "long": {"n_positions": 4},
            "short": {"n_positions": 4},
        },
        "live": {
            "approved_coins": {"long": [], "short": []},
            "ignored_coins": {"long": [], "short": []},
        },
        "optimize": {"bounds": {}, "scoring": [], "suite": {"enabled": False, "scenarios": []}},
    }


def test_format_config_normalizes_coin_sources():
    cfg = _base_config()
    cfg["backtest"]["coin_sources"] = {"BTC": "binance", 123: None}
    out = format_config(copy.deepcopy(cfg), verbose=False)
    assert out["backtest"]["coin_sources"] == {"BTC": "binance"}
