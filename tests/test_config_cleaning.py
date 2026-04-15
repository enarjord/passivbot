import json
from copy import deepcopy

from config_utils import (
    clean_config,
    dump_config,
    get_template_config,
    load_config,
    sanitize_prepared_config_for_dump,
    strip_config_metadata,
)


def test_clean_config_removes_internal_sections_and_keeps_user_values():
    template = get_template_config()
    config = {
        "bot": {
            "long": {"n_positions": 5, "_note": "keep me?"},
            "short": {"n_positions": 3},
        },
        "coin_overrides": {
            "BTC": {
                "_meta": "do not keep",
                "live": {"forced_mode_long": "manual"},
            }
        },
        "_transform_log": ["noise"],
        "_raw": {"bot": {}},
        "_raw_effective": {"bot": {"long": {"n_positions": 7}}},
    }

    cleaned = clean_config(deepcopy(config))

    assert "_transform_log" not in cleaned
    assert "_raw" not in cleaned
    assert "_raw_effective" not in cleaned
    assert cleaned["bot"]["long"]["n_positions"] == 5
    assert cleaned["bot"]["short"]["n_positions"] == 3
    assert cleaned["bot"]["long"]["ema_span_0"] == template["bot"]["long"]["ema_span_0"]
    assert "BTC" in cleaned["coin_overrides"]
    assert "_meta" not in cleaned["coin_overrides"]["BTC"]
    assert (
        cleaned["coin_overrides"]["BTC"]["live"]["forced_mode_long"]
        == config["coin_overrides"]["BTC"]["live"]["forced_mode_long"]
    )


def test_clean_config_fills_missing_values_from_template():
    config = {}
    cleaned = clean_config(config)
    template = get_template_config()
    assert cleaned == template


def test_strip_config_metadata_removes_known_keys_recursively():
    config = {
        "bot": {
            "long": {"n_positions": 3, "_raw": {"ignore": True}},
            "_coins_sources": {"BTC": "binance"},
        },
        "_raw": {"bot": {}},
        "_raw_effective": {"bot": {"long": {"n_positions": 5}}},
        "_transform_log": ["load"],
        "nested": {"_raw": 123, "value": 5},
        "_coins_sources": {"ADA": "bybit"},
    }
    stripped = strip_config_metadata(config)
    assert "_raw" not in stripped
    assert "_raw_effective" not in stripped
    assert "_transform_log" not in stripped
    assert "_coins_sources" not in stripped
    assert "_coins_sources" not in stripped["bot"]
    assert "_raw" not in stripped["bot"]["long"]
    assert stripped["nested"]["value"] == 5


def test_sanitize_prepared_config_for_dump_removes_analysis_and_metadata():
    config = {
        "analysis": {"adg": 0.1},
        "_raw": {"bot": {}},
        "_raw_effective": {"bot": {}},
        "_transform_log": ["normalize"],
        "_coins_sources": {"approved_coins": "all"},
        "disable_plotting": "all",
        "backtest": {
            "cache_dir": "tmp/cache",
            "coins": {"binance": ["BTC"]},
        },
        "bot": {"long": {"n_positions": 3}, "short": {"n_positions": 0}},
    }

    sanitized = sanitize_prepared_config_for_dump(config)

    assert "analysis" not in sanitized
    assert "_raw" not in sanitized
    assert "_raw_effective" not in sanitized
    assert "_transform_log" not in sanitized
    assert "_coins_sources" not in sanitized
    assert "disable_plotting" not in sanitized
    assert "cache_dir" not in sanitized["backtest"]
    assert "coins" not in sanitized["backtest"]


def test_dump_config_clean_preserves_backtest_aggregate_overrides(tmp_path):
    cfg_path = tmp_path / "in.json"
    out_path = tmp_path / "out.json"
    cfg_path.write_text(
        json.dumps(
            {
                "backtest": {
                    "aggregate": {"adg_pnl": "max"},
                    "base_dir": "backtests",
                    "compress_cache": True,
                    "end_date": "2025-01-01",
                    "start_date": "2024-01-01",
                    "exchanges": ["binance"],
                    "btc_collateral_cap": 1.0,
                    "gap_tolerance_ohlcvs_minutes": 120,
                    "max_warmup_minutes": 0,
                },
                "bot": {"long": {"n_positions": 4}, "short": {"n_positions": 4}},
                "live": {
                    "approved_coins": {"long": [], "short": []},
                    "ignored_coins": {"long": [], "short": []},
                    "minimum_coin_age_days": 30,
                    "warmup_ratio": 0.2,
                    "max_warmup_minutes": 0,
                },
                "optimize": {"bounds": {}, "scoring": []},
            }
        )
    )

    loaded = load_config(str(cfg_path), verbose=False)
    dump_config(loaded, str(out_path), clean=True)
    dumped = json.loads(out_path.read_text())

    assert dumped["backtest"]["aggregate"]["adg_pnl"] == "max"
    assert dumped["backtest"]["aggregate"]["default"] == "mean"


def test_clean_config_preserves_backtest_aggregate_overrides():
    config = {
        "backtest": {"aggregate": {"adg_pnl": "max", "default": "mean"}},
    }

    cleaned = clean_config(config)

    assert cleaned["backtest"]["aggregate"]["adg_pnl"] == "max"
    assert cleaned["backtest"]["aggregate"]["default"] == "mean"
