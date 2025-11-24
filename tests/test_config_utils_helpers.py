from copy import deepcopy
from types import SimpleNamespace
import json

import config_utils
import pytest

from config_transform import ConfigTransformTracker, record_transform
from config_utils import (
    _apply_backward_compatibility_renames,
    _apply_non_live_adjustments,
    _ensure_bot_defaults_and_bounds,
    _migrate_btc_collateral_settings,
    _normalize_position_counts,
    _rename_config_keys,
    _sync_with_template,
    get_template_config,
    update_config_with_args,
    parse_overrides,
    format_config,
)


def test_ensure_bot_defaults_and_bounds_adds_missing_values():
    config = get_template_config()
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


def test_rename_config_keys_records_tracker_events():
    config = {
        "live": {"minimum_market_age_days": 5},
        "backtest": {"exchange": "binance"},
    }
    tracker = ConfigTransformTracker()

    _rename_config_keys(config, verbose=False, tracker=tracker)

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


def test_sync_with_template_adds_missing_and_removes_extras():
    template = get_template_config()
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
    config = get_template_config()
    config["live"]["approved_coins"] = "btc,eth"
    config["live"]["ignored_coins"] = {"long": ["eth"], "short": []}
    config["backtest"]["end_date"] = "2023-01-01"
    config["backtest"]["btc_collateral_cap"] = 0.0
    config["optimize"]["scoring"] = ["btc_adg", "adg"]
    config["optimize"][
        "limits"
    ] = "--lower_bound_drawdown_worst 0.3 --penalize_if_lower_than_gain_btc 0.1"
    config["optimize"]["bounds"]["long_entry_grid_spacing_pct"] = [0.1, 0.05]

    _apply_non_live_adjustments(config, verbose=False)

    assert config["live"]["approved_coins"]["long"] == ["btc"]
    assert config["optimize"]["scoring"] == ["adg_btc", "adg_usd"]
    limits = config["optimize"]["limits"]
    assert isinstance(limits, list)
    gain_limit = next((entry for entry in limits if entry["metric"] == "gain_btc"), None)
    drawdown_limit = next(
        (entry for entry in limits if entry["metric"] == "drawdown_worst_usd"), None
    )
    assert drawdown_limit is not None
    assert drawdown_limit["penalize_if"] == "greater_than"
    assert drawdown_limit["value"] == pytest.approx(0.3)
    assert gain_limit is not None
    assert gain_limit["penalize_if"] == "less_than"
    assert gain_limit["value"] == pytest.approx(0.1)
    assert config["optimize"]["bounds"]["long_entry_grid_spacing_pct"] == [0.05, 0.1]
    assert config["live"]["approved_coins"]["short"] == ["btc", "eth"]


def test_apply_non_live_adjustments_supports_legacy_coins_file():
    config = get_template_config()
    config["live"]["approved_coins"] = "configs/approved_coins_topmcap.json"
    config["live"]["ignored_coins"] = {"long": [], "short": []}
    config["live"]["empty_means_all_approved"] = False
    config["optimize"]["bounds"]["long_entry_grid_spacing_pct"] = [0.1, 0.2]
    config["backtest"]["end_date"] = "2023-01-01"
    _apply_non_live_adjustments(config, verbose=False)
    with open("configs/approved_coins.json") as fp:
        expected = json.load(fp)
    assert config["live"]["approved_coins"]["long"] == expected


def test_migrate_btc_collateral_settings_converts_bool():
    config = {"backtest": {"use_btc_collateral": True}}
    _migrate_btc_collateral_settings(config, verbose=False)
    assert config["backtest"]["btc_collateral_cap"] == pytest.approx(1.0)
    assert config["backtest"]["btc_collateral_ltv_cap"] is None
    assert "use_btc_collateral" not in config["backtest"]

    config = {"backtest": {"use_btc_collateral": False}}
    _migrate_btc_collateral_settings(config, verbose=False)
    assert config["backtest"]["btc_collateral_cap"] == pytest.approx(0.0)
    assert config["backtest"]["btc_collateral_ltv_cap"] is None


def test_normalize_limit_entries_supports_new_schema():
    raw = [
        {"metric": "adg_btc", "penalize_if": "<", "value": 0.001, "stat": "mean"},
        {"metric": "loss_profit_ratio", "penalize_if": ">", "value": 0.8},
        {"metric": "omega_ratio", "penalize_if": "outside_range", "range": [1.5, 3.0]},
        {"metric": "sharpe_ratio", "penalize_if": "inside_range", "range": [0.5, 1.0]},
    ]
    normalized = config_utils.normalize_limit_entries(raw)
    assert len(normalized) == 4
    assert normalized[0]["metric"] == "adg_btc"
    assert normalized[0]["stat"] == "mean"
    assert normalized[1]["metric"] == "loss_profit_ratio"
    assert normalized[2]["range"] == [1.5, 3.0]
    assert normalized[3]["penalize_if"] == "inside_range"


def test_normalize_limit_entries_preserves_integers():
    raw = {"penalize_if_greater_than_position_held_hours_max": 2016}
    normalized = config_utils.normalize_limit_entries(raw)
    assert len(normalized) == 1
    assert normalized[0]["metric"] == "position_held_hours_max"
    assert isinstance(normalized[0]["value"], int)
    assert normalized[0]["value"] == 2016


def test_limits_structural_equal_detects_canonical_entries():
    raw = [
        {"metric": "drawdown_worst_btc", "penalize_if": "greater_than", "value": 0.3},
        {
            "metric": "loss_profit_ratio",
            "penalize_if": "outside_range",
            "range": (0.1, 0.7),
        },
    ]
    normalized = config_utils.normalize_limit_entries(raw)
    assert config_utils._limits_structurally_equal(raw, normalized)


def test_apply_backward_compatibility_renames_moves_filter_keys():
    config = {
        "bot": {
            "long": {
                "filter_noisiness_rolling_window": 42,
                "filter_volatility_ema_span": 84,
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
    assert config["bot"]["long"]["filter_volatility_ema_span"] == 84
    assert config["bot"]["long"]["filter_volume_ema_span"] == 21
    assert config["bot"]["short"]["filter_volume_ema_span"] == 11
    bounds = config["optimize"]["bounds"]
    assert "long_filter_noisiness_rolling_window" not in bounds
    assert bounds["long_filter_volatility_ema_span"] == [10, 20]
    assert "short_filter_volume_rolling_window" not in bounds
    assert bounds["short_filter_volume_ema_span"] == [30, 40]


def test_update_config_with_args_updates_coin_sources():
    config = get_template_config()
    config["_coins_sources"] = {
        "approved_coins": {"long": ["ADA"], "short": []},
        "ignored_coins": {"long": [], "short": []},
    }
    args = SimpleNamespace()
    vars(args)["live.approved_coins"] = ["BTC", "ETH"]
    update_config_with_args(config, args, verbose=False)
    assert config["live"]["approved_coins"]["long"] == ["BTC", "ETH"]
    assert config["_coins_sources"]["approved_coins"]["long"] == ["BTC", "ETH"]
    log_entry = config["_transform_log"][-1]
    assert log_entry["step"] == "update_config_with_args"
    diff = log_entry["details"]["diffs"][0]
    assert diff["path"] == "live.approved_coins"
    assert diff["new"]["long"] == ["BTC", "ETH"]


def test_update_config_with_args_replaces_path_coin_source():
    config = get_template_config()
    config["_coins_sources"] = {"ignored_coins": "configs/ignored.json"}
    args = SimpleNamespace()
    vars(args)["live.ignored_coins"] = ["DOGE"]
    update_config_with_args(config, args, verbose=False)
    assert config["live"]["ignored_coins"]["long"] == ["DOGE"]
    assert config["_coins_sources"]["ignored_coins"]["long"] == ["DOGE"]
    entry = config["_transform_log"][-1]
    assert entry["step"] == "update_config_with_args"
    assert entry["details"]["diffs"][0]["path"] == "live.ignored_coins"


def test_load_config_stores_raw_snapshot(monkeypatch):
    import copy

    raw = {"live": {"approved_coins": ["BTC"]}}

    monkeypatch.setattr(
        config_utils,
        "load_hjson_config",
        lambda path: copy.deepcopy(raw),
    )

    def fake_format(cfg, **kwargs):
        # return a new dict to simulate normalization
        result = {"live": {"approved_coins": cfg["live"]["approved_coins"][:]}}
        result["_transform_log"] = []
        record_transform(result, "format_config", {"mock": True})
        return result

    monkeypatch.setattr(config_utils, "format_config", fake_format)

    loaded = config_utils.load_config("dummy.json", verbose=False)
    assert loaded["_raw"] == raw

    # Mutating runtime view must not mutate the raw snapshot
    loaded["live"]["approved_coins"].append("ETH")
    assert loaded["_raw"]["live"]["approved_coins"] == ["BTC"]
    log_steps = [entry["step"] for entry in loaded["_transform_log"]]
    assert log_steps[0] == "load_config"
    assert "format_config" in log_steps


def test_parse_overrides_records_transform_log():
    cfg = {
        "live": {"approved_coins": []},
        "coin_overrides": {"BTC": {"bot": {"long": {"wallet_exposure_limit": 0.5}}}},
    }
    out = parse_overrides(cfg, verbose=False)
    assert out["_transform_log"][-1]["step"] == "parse_overrides"


def test_format_config_preserves_raw_snapshot_and_log():
    cfg = get_template_config()
    raw_snapshot = {"live": {"approved_coins": ["BTC"]}}
    cfg["_raw"] = deepcopy(raw_snapshot)
    cfg["_transform_log"] = [{"step": "preprocess", "ts_ms": 0}]

    out = format_config(cfg, verbose=False)

    assert out["_raw"] == raw_snapshot
    log_steps = [entry["step"] for entry in out["_transform_log"]]
    assert log_steps[:1] == ["preprocess"]
    assert log_steps[-1] == "format_config"
    details = out["_transform_log"][-1]["details"]
    assert details["changes"], "format_config should record structural changes"
    assert any(event["action"] == "add" for event in details["changes"])


def test_update_config_with_args_records_old_new_values():
    config = get_template_config()
    args = SimpleNamespace()
    vars(args)["backtest.start_date"] = "2022-01-01"

    update_config_with_args(config, args, verbose=False)

    entry = config["_transform_log"][-1]
    assert entry["step"] == "update_config_with_args"
    diff = entry["details"]["diffs"][0]
    assert diff["path"] == "backtest.start_date"
    assert diff["old"] == "2021-04-01"
    assert diff["new"] == "2022-01-01"


def test_backtest_filter_min_cost_inherits_from_live():
    cfg = get_template_config()
    cfg["live"]["filter_by_min_effective_cost"] = True
    cfg["backtest"].pop("filter_by_min_effective_cost", None)

    formatted = format_config(cfg, verbose=False)

    assert formatted["backtest"]["filter_by_min_effective_cost"] is True
