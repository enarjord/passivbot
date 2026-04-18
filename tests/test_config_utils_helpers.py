import logging
import argparse
from copy import deepcopy
from types import SimpleNamespace
import json

import config_utils
import pytest

from cli_utils import build_command_parser, expand_help_all_argv, help_all_requested
from config import load_input_config, prepare_config
from config.project import project_config
from config.runtime_compile import compile_runtime_config
from config_transform import ConfigTransformTracker, record_transform
from utils import normalize_coins_source
from config_utils import (
    _apply_backward_compatibility_renames,
    _apply_non_live_adjustments,
    _ensure_bot_defaults_and_bounds,
    _hydrate_missing_template_fields,
    _migrate_btc_collateral_settings,
    _migrate_config_version,
    _migrate_empty_means_all_approved,
    _normalize_position_counts,
    _rename_config_keys,
    _sync_with_template,
    CLI_HELP_GROUPS,
    FIELD_RUNTIME_RULES,
    RESERVED_CLI_ARGS,
    add_config_arguments,
    get_template_config,
    load_config,
    project_template_config_for_cli,
    update_config_with_args,
    parse_overrides,
    format_config,
)


def test_load_input_config_without_path_uses_schema_defaults():
    source, base_config_path, raw_snapshot = load_input_config(None, log_info=False)

    assert base_config_path == ""
    assert source == get_template_config()
    assert raw_snapshot == get_template_config()


def test_default_example_config_matches_schema_defaults():
    with open("configs/examples/default_trailing_grid_long_npos10.json", encoding="utf-8") as fh:
        example = json.load(fh)

    assert example == get_template_config()


def test_prepare_config_preserves_raw_snapshot_and_effective_input():
    source, base_config_path, raw_snapshot = load_input_config(None, log_info=False)
    source["live"]["user"] = "test_user"

    prepared = prepare_config(
        source,
        base_config_path=base_config_path,
        live_only=True,
        verbose=False,
        target="live",
        runtime="live",
        raw_snapshot=raw_snapshot,
    )

    assert "backtest" not in prepared
    assert prepared["live"]["user"] == "test_user"
    assert prepared["bot"]["long"]["filter_volume_ema_span"] == pytest.approx(
        prepared["bot"]["long"]["forager_volume_ema_span"]
    )
    assert prepared["_raw"] == raw_snapshot
    assert prepared["_raw_effective"]["live"]["user"] == "test_user"
    assert prepared["_raw"]["live"]["user"] != "test_user"


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
        "backtest": {"exchange": "binance", "panic_market_slippage_pct": 0.0015},
    }

    _rename_config_keys(config, verbose=False)

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
    assert any(
        event["action"] == "rename"
        and event["from"] == "backtest.panic_market_slippage_pct"
        and event["to"] == "backtest.market_order_slippage_pct"
        for event in summary
    )


def test_load_config_renames_legacy_panic_market_slippage_pct(tmp_path):
    raw = {
        "backtest": {"panic_market_slippage_pct": 0.0015},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }
    path = tmp_path / "legacy_slippage.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = load_config(str(path), verbose=False)

    assert loaded["backtest"]["market_order_slippage_pct"] == pytest.approx(0.0015)
    assert "panic_market_slippage_pct" not in loaded["backtest"]


def test_hydrate_then_sync_with_template_adds_missing_and_removes_extras():
    template = get_template_config()
    result = {
        "live": {},
        "backtest": {},
        "bot": {"long": {}, "short": {}, "extra_side": {}},
        "optimize": {"bounds": {}, "limits": "", "scoring": []},
        "coin_overrides": {},
    }

    _hydrate_missing_template_fields(template, result, verbose=False)
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
    assert config["optimize"]["scoring"] == [
        {"metric": "adg_btc", "goal": "max"},
        {"metric": "adg_usd", "goal": "max"},
    ]
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
    config["optimize"]["bounds"]["long_entry_grid_spacing_pct"] = [0.1, 0.2]
    config["backtest"]["end_date"] = "2023-01-01"
    _apply_non_live_adjustments(config, verbose=False)
    with open("configs/approved_coins.json") as fp:
        expected = json.load(fp)
    assert config["live"]["approved_coins"]["long"] == expected


def test_normalize_coins_source_supports_partial_per_side_dicts():
    normalized = normalize_coins_source({"long": ["BTC", "ETH"]})

    assert normalized["long"] == ["BTC", "ETH"]
    assert normalized["short"] == []


def test_normalize_coins_source_supports_explicit_all():
    normalized_global = normalize_coins_source("all")
    normalized_partial = normalize_coins_source({"long": ["BTC"], "short": "all"})

    assert normalized_global == {"long": ["all"], "short": ["all"]}
    assert normalized_partial == {"long": ["BTC"], "short": ["all"]}


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
        _migrate_empty_means_all_approved(config, verbose=False, tracker=tracker)

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
    assert any("deprecated" in rec.message for rec in caplog.records)


def test_migrate_empty_means_all_approved_keeps_explicit_per_side_values():
    config = {
        "live": {
            "approved_coins": {"long": ["BTC"], "short": []},
            "ignored_coins": {"long": [], "short": []},
            "empty_means_all_approved": True,
        }
    }

    _migrate_empty_means_all_approved(config, verbose=False)

    assert config["live"]["approved_coins"] == {"long": ["BTC"], "short": []}
    assert "empty_means_all_approved" not in config["live"]


def test_migrate_config_version_sets_current_schema_version():
    config = {}

    _migrate_config_version(config, verbose=False)

    assert config["config_version"] == get_template_config()["config_version"]


def test_max_realized_loss_pct_default_is_consistent_across_template_and_formatting():
    template = get_template_config()
    assert template["live"]["max_realized_loss_pct"] == pytest.approx(1.0)

    sparse = {
        "live": {},
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "optimize": {"bounds": {}},
        "coin_overrides": {},
    }
    formatted = format_config(sparse, verbose=False)
    assert formatted["live"]["max_realized_loss_pct"] == pytest.approx(1.0)

    loaded = load_config("configs/examples/default_trailing_grid_long_npos10.json", verbose=False)
    assert loaded["live"]["max_realized_loss_pct"] == pytest.approx(1.0)


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


def test_normalize_limit_entries_preserves_optional_fields_on_canonical_entries():
    raw = [
        {
            "metric": "drawdown_worst_btc",
            "penalize_if": "greater_than",
            "value": 0.85,
            "enabled": False,
        },
        {
            "metric": "adg_pnl",
            "penalize_if": "less_than",
            "stat": "mean",
            "value": 0,
            "enabled": False,
        },
    ]

    normalized = config_utils.normalize_limit_entries(raw)

    assert normalized[0]["enabled"] is False
    assert normalized[1]["enabled"] is False


def test_load_config_preserves_canonical_optimize_limits(tmp_path):
    cfg = get_template_config()
    cfg["optimize"]["limits"] = [
        {
            "metric": "drawdown_worst_btc",
            "penalize_if": "greater_than",
            "value": 0.85,
            "enabled": False,
        },
        {
            "metric": "adg_pnl",
            "penalize_if": "less_than",
            "stat": "mean",
            "value": 0,
            "enabled": False,
        },
    ]
    path = tmp_path / "limits.json"
    path.write_text(json.dumps(cfg))

    loaded = load_config(str(path), verbose=False)

    assert loaded["optimize"]["limits"] == cfg["optimize"]["limits"]


def test_load_config_malformed_optimize_limits_falls_back_to_template(caplog, tmp_path):
    cfg = get_template_config()
    cfg["optimize"]["limits"] = [{"metric": "adg_pnl", "value": 0}]
    path = tmp_path / "malformed_limits.json"
    path.write_text(json.dumps(cfg))

    loaded = load_config(str(path), verbose=False)

    assert loaded["optimize"]["limits"] == get_template_config()["optimize"]["limits"]
    assert any("optimize.limits malformed or unsupported" in rec.message for rec in caplog.records)


def test_load_config_disabled_sparse_optimize_limits_are_normalized(caplog, tmp_path):
    cfg = get_template_config()
    cfg["optimize"]["limits"] = [
        {"metric": "drawdown_worst_hsl", "penalize_if": "greater_than", "value": 0.9},
        {"metric": "peak_recovery_hours_hsl", "enabled": False},
        {"metric": "position_held_hours_max", "enabled": False},
    ]
    path = tmp_path / "disabled_sparse_limits.json"
    path.write_text(json.dumps(cfg))

    loaded = load_config(str(path), verbose=False)

    assert loaded["optimize"]["limits"][1]["metric"] == "peak_recovery_hours_hsl"
    assert loaded["optimize"]["limits"][1]["enabled"] is False
    assert loaded["optimize"]["limits"][2]["enabled"] is False
    assert not any("optimize.limits malformed or unsupported" in rec.message for rec in caplog.records)


def test_normalize_limit_entries_preserves_integers():
    raw = {"penalize_if_greater_than_position_held_hours_max": 2016}
    normalized = config_utils.normalize_limit_entries(raw)
    assert len(normalized) == 1
    assert normalized[0]["metric"] == "position_held_hours_max"
    assert isinstance(normalized[0]["value"], int)
    assert normalized[0]["value"] == 2016


def test_parse_limit_cli_entry_supports_scalar_syntax():
    entry = config_utils.parse_limit_cli_entry("drawdown_worst > 0.35")

    assert entry == {
        "metric": "drawdown_worst_usd",
        "penalize_if": "less_than_or_equal",
        "value": 0.35,
    }


def test_parse_limit_cli_entry_supports_scalar_syntax_without_spaces():
    entry = config_utils.parse_limit_cli_entry("drawdown_worst<=0.35")

    assert entry == {
        "metric": "drawdown_worst_usd",
        "penalize_if": "greater_than",
        "value": 0.35,
    }


def test_parse_limit_cli_entry_supports_extended_scalar_operators():
    greater_equal = config_utils.parse_limit_cli_entry("adg_strategy_pnl_rebased>=0.001")
    equal_to = config_utils.parse_limit_cli_entry("adg_strategy_pnl_rebased == 0.0")

    assert greater_equal == {
        "metric": "adg_strategy_pnl_rebased",
        "penalize_if": "less_than",
        "value": 0.001,
    }
    assert equal_to == {
        "metric": "adg_strategy_pnl_rebased",
        "penalize_if": "not_equal",
        "value": 0,
    }


def test_parse_limit_cli_entry_supports_range_and_extras():
    entry = config_utils.parse_limit_cli_entry(
        "loss_profit_ratio outside_range [0.05,0.7] stat=mean enabled=false"
    )

    assert entry == {
        "metric": "loss_profit_ratio",
        "penalize_if": "outside_range",
        "range": [0.05, 0.7],
        "stat": "mean",
        "enabled": False,
    }


def test_parse_limit_cli_entries_supports_json_object_strings():
    entries = config_utils.parse_limit_cli_entries(
        ['{"metric":"adg","penalize_if":"<","value":0.001,"stat":"mean"}']
    )

    assert entries == [
        {
            "metric": "adg_usd",
            "penalize_if": "less_than",
            "value": 0.001,
            "stat": "mean",
        }
    ]


def test_normalize_limit_entries_supports_hjson_list_payload():
    raw = """
    [
      {
        metric: drawdown_worst
        penalize_if: greater_than
        value: 0.35
      }
    ]
    """

    normalized = config_utils.normalize_limit_entries(raw)

    assert normalized == [
        {
            "metric": "drawdown_worst_usd",
            "penalize_if": "greater_than",
            "value": 0.35,
        }
    ]


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
    assert config["bot"]["long"]["forager_volatility_ema_span"] == 84
    assert config["bot"]["long"]["forager_volume_ema_span"] == 21
    assert config["bot"]["short"]["forager_volume_ema_span"] == 11
    bounds = config["optimize"]["bounds"]
    assert "long_filter_noisiness_rolling_window" not in bounds
    assert bounds["long_forager_volatility_ema_span"] == [10, 20]
    assert "short_filter_volume_rolling_window" not in bounds
    assert bounds["short_forager_volume_ema_span"] == [30, 40]


def test_compile_runtime_config_adds_internal_forager_aliases():
    config = format_config(get_template_config(), verbose=False)

    assert "filter_volume_ema_span" not in config["bot"]["long"]
    assert "long_filter_volume_ema_span" not in config["optimize"]["bounds"]

    compiled = compile_runtime_config(config, runtime="live")

    assert compiled["bot"]["long"]["filter_volume_ema_span"] == config["bot"]["long"][
        "forager_volume_ema_span"
    ]
    assert compiled["bot"]["long"]["filter_volatility_ema_span"] == config["bot"]["long"][
        "forager_volatility_ema_span"
    ]
    assert compiled["optimize"]["bounds"]["long_filter_volume_ema_span"] == config["optimize"][
        "bounds"
    ]["long_forager_volume_ema_span"]


def test_project_config_prunes_unrelated_sections():
    config = format_config(get_template_config(), verbose=False)

    projected = project_config(config, "live", record_step=False)

    assert "backtest" not in projected
    assert "optimize" not in projected
    assert "monitor" in projected
    assert "live" in projected
    assert "bot" in projected


def test_format_config_emits_coalesced_summary_without_leaf_noise(caplog):
    tmpl = get_template_config()
    lean_live = {"bot": deepcopy(tmpl["bot"]), "live": deepcopy(tmpl["live"])}

    with caplog.at_level(logging.INFO):
        format_config(lean_live, verbose=True, live_only=True)

    messages = [rec.message for rec in caplog.records]
    assert any("Added missing backtest section from defaults" in msg for msg in messages)
    assert not any("Added missing backtest.aggregate" in msg for msg in messages)
    assert not any("renaming parameter" in msg for msg in messages)


def test_load_example_config_avoids_leaf_add_remove_log_churn(caplog):
    with caplog.at_level(logging.INFO):
        load_config("configs/examples/default_trailing_grid_long_npos10.json", verbose=True)

    messages = [rec.message for rec in caplog.records]
    assert not any("Removed unused key" in msg for msg in messages)
    assert not any("Added missing optimize.bounds.long_" in msg for msg in messages)


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
    assert config["_coins_sources"]["approved_coins"] == ["BTC", "ETH"]
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
    assert config["_coins_sources"]["ignored_coins"] == ["DOGE"]
    entry = config["_transform_log"][-1]
    assert entry["step"] == "update_config_with_args"
    assert entry["details"]["diffs"][0]["path"] == "live.ignored_coins"


def test_update_config_with_args_preserves_cli_coin_file_source_for_live_reload(tmp_path):
    approved_file = tmp_path / "approved.hjson"
    approved_file.write_text('["BTC","ETH"]', encoding="utf-8")

    config = get_template_config()
    config["_coins_sources"] = {}
    args = SimpleNamespace()
    vars(args)["live.approved_coins"] = [str(approved_file)]

    update_config_with_args(config, args, verbose=False)

    assert config["live"]["approved_coins"]["long"] == ["BTC", "ETH"]
    assert config["_coins_sources"]["approved_coins"] == [str(approved_file)]

    approved_file.write_text('["BTC","XRP"]', encoding="utf-8")
    refreshed = normalize_coins_source(config["_coins_sources"]["approved_coins"])

    assert set(refreshed["long"]) == {"BTC", "XRP"}


def test_load_config_preserves_raw_and_effective_snapshots(tmp_path):
    raw = get_template_config()
    raw["live"]["approved_coins"] = ["BTC"]
    path = tmp_path / "raw_config.json"
    path.write_text(json.dumps(raw))

    loaded = config_utils.load_config(str(path), verbose=False)
    assert loaded["_raw"] == raw
    assert loaded["_raw_effective"] == raw

    # Mutating runtime view must not mutate the stored snapshots
    loaded["live"]["approved_coins"]["long"].append("ETH")
    loaded["live"]["approved_coins"]["short"].append("ETH")
    assert loaded["_raw"]["live"]["approved_coins"] == ["BTC"]
    assert loaded["_raw_effective"]["live"]["approved_coins"] == ["BTC"]
    log_steps = [entry["step"] for entry in loaded["_transform_log"]]
    assert log_steps[0] == "load_config"
    assert "normalize_config" in log_steps


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
    assert diff["old"] == "2021-01-01"
    assert diff["new"] == "2022-01-01"


def _parse_backtest_args(raw_argv):
    help_all = help_all_requested(raw_argv)
    parser = build_command_parser(
        prog="passivbot backtest",
        description="run backtest",
        usage="%(prog)s [config_path] [options]",
        epilog="test",
    )
    parser.add_argument("config_path", type=str, default=None, nargs="?")
    parser.add_argument("--suite", nargs="?", const="true", default=None, type=config_utils.str2bool)
    template_config = project_template_config_for_cli(get_template_config(), "backtest")
    if "logging" in template_config and isinstance(template_config["logging"], dict):
        template_config["logging"].pop("level", None)
    allowed_config_keys = add_config_arguments(
        parser,
        template_config,
        command="backtest",
        help_all=help_all,
        group_map={},
    )
    return parser.parse_args(expand_help_all_argv(raw_argv)), allowed_config_keys


def _make_live_only_source_config():
    return {
        "bot": {"long": {}, "short": {}},
        "live": {},
        "coin_overrides": {},
    }


@pytest.mark.parametrize("start_flag", ["-sd", "--start-date"])
def test_backtest_cli_start_date_override_creates_missing_backtest_section(start_flag):
    source_config = _make_live_only_source_config()
    assert "backtest" not in source_config

    args, allowed_config_keys = _parse_backtest_args(
        [
            "configs/hype.json",
            "--bot.long.hsl_ema_span_minutes",
            "1440",
            start_flag,
            "2025-10-11",
        ]
    )

    update_config_with_args(source_config, args, verbose=False, allowed_keys=allowed_config_keys)

    assert source_config["backtest"]["start_date"] == "2025-10-11"
    assert source_config["bot"]["long"]["hsl_ema_span_minutes"] == pytest.approx(1440.0)
    assert "backtest.start_date" in source_config["_transform_log"][-1]["details"]["keys"]


def test_backtest_cli_candle_interval_short_flag_parses_as_int():
    args, _allowed_config_keys = _parse_backtest_args(
        [
            "configs/hype.json",
            "-cim",
            "2",
        ]
    )

    assert getattr(args, "backtest.candle_interval_minutes") == 2
    assert isinstance(getattr(args, "backtest.candle_interval_minutes"), int)


def test_update_config_with_args_ignores_non_config_parser_args():
    config = {}
    args = SimpleNamespace(config_path="configs/hype.json", log_level="info", suite=True)

    update_config_with_args(config, args, verbose=False)

    assert "_transform_log" not in config
    assert config == {}


def test_update_config_with_args_logs_optimize_limits_as_diff(caplog):
    config = get_template_config()
    original_limits = deepcopy(config["optimize"]["limits"])
    args = SimpleNamespace()
    vars(args)["optimize.limits"] = original_limits + [
        {
            "metric": "adg_strategy_pnl_rebased",
            "penalize_if": "less_than_or_equal",
            "value": 0,
        }
    ]

    with caplog.at_level(logging.INFO):
        update_config_with_args(config, args, verbose=True)

    messages = [rec.message for rec in caplog.records]
    target = [msg for msg in messages if msg.startswith("[config] changed optimize.limits")]
    assert target, messages
    assert "added 1 entry" in target[-1]
    assert "adg_strategy_pnl_rebased" in target[-1]
    assert " -> " not in target[-1]


def test_backtest_filter_min_cost_inherits_from_live():
    cfg = get_template_config()
    cfg["live"]["filter_by_min_effective_cost"] = True
    cfg["backtest"]["filter_by_min_effective_cost"] = None

    formatted = format_config(cfg, verbose=False)

    assert formatted["backtest"]["filter_by_min_effective_cost"] is True


def _format_parser_help_with_config(command: str, config: dict, help_all: bool) -> str:
    config = project_template_config_for_cli(config, command)
    parser = argparse.ArgumentParser(prog=command)
    group_map = {
        title: parser.add_argument_group(title) for title in CLI_HELP_GROUPS.get(command, [])
    }
    add_config_arguments(
        parser,
        config,
        command=command,
        help_all=help_all,
        group_map=group_map,
    )
    if command == "optimize":
        group_map["Advanced Overrides"].add_argument(
            "-t",
            "--start",
            type=str,
            required=False,
            dest="starting_configs",
            default=None,
            help=(
                "Start with given live configs. Single json file or dir with multiple json files"
                if help_all
                else argparse.SUPPRESS
            ),
        )
        group_map["Advanced Overrides"].add_argument(
            "-ft",
            "--fine_tune_params",
            "--fine-tune-params",
            type=str,
            default="",
            dest="fine_tune_params",
            help=(
                "Comma-separated optimize bounds keys to tune; other parameters are fixed to their current config values"
                if help_all
                else argparse.SUPPRESS
            ),
        )
        group_map["Optimize Common"].add_argument(
            "-l",
            "--limit",
            action="append",
            dest="limit_entries",
            default=None,
            metavar="SPEC",
        )
        group_map["Optimize Common"].add_argument(
            "--clear-limits",
            action="store_true",
            dest="clear_limits",
        )
    return parser.format_help()


def test_optimize_default_help_groups_common_flags_and_hides_bounds():
    config = get_template_config()
    help_text = _format_parser_help_with_config("optimize", config, help_all=False)

    assert "Coin Selection:" in help_text
    assert "Date Range:" in help_text
    assert "Optimizer:" in help_text
    assert "--symbols CSV_OR_PATH, -s CSV_OR_PATH" in help_text
    assert "--population-size INT, -ps INT" in help_text
    assert "--backend BACKEND, -ob BACKEND" in help_text
    assert "--limits JSON_OR_HJSON" in help_text
    assert "-l SPEC, --limit SPEC" in help_text
    assert "--clear-limits" in help_text
    assert "--minimum-coin-age-days FLOAT, -mcad FLOAT" in help_text
    assert "--hedge-mode Y/N, -hm Y/N" in help_text
    assert "--market-orders-allowed Y/N, -moa Y/N" in help_text
    assert "--market-order-near-touch-threshold FLOAT, -montt FLOAT" in help_text
    assert "--max-realized-loss-pct FLOAT, -mrlp FLOAT" in help_text
    assert "--pnls-max-lookback-days FLOAT|all, -pmld FLOAT|all" in help_text
    assert "--bot.long.entry_grid_inflation_enabled" not in help_text
    assert "--bot.long.hsl_enabled" not in help_text
    assert "--optimize_population_size" not in help_text
    assert "--optimize.bounds.long_close_grid_markup_end" not in help_text
    assert "Optimize DEAP:" not in help_text
    assert "Optimize Pymoo:" not in help_text


def test_optimize_help_all_shows_hidden_bounds_flags():
    config = get_template_config()
    help_text = _format_parser_help_with_config("optimize", config, help_all=True)

    assert "Optimize Bounds:" in help_text
    assert "--optimize.bounds.long_close_grid_markup_end MIN,MAX[,STEP]" in help_text
    assert "--limits JSON_OR_HJSON" in help_text
    assert "-l SPEC, --limit SPEC" in help_text
    assert "--hedge-mode Y/N, -hm Y/N" in help_text
    assert "--market-order-near-touch-threshold FLOAT, -montt FLOAT" in help_text
    assert "--max-realized-loss-pct FLOAT, -mrlp FLOAT" in help_text
    assert "--bot.long.entry_grid_inflation_enabled Y/N" in help_text
    assert "--bot.short.entry_grid_inflation_enabled Y/N" in help_text
    assert "--bot.long.hsl_enabled Y/N" in help_text
    assert "--bot.short.hsl_enabled Y/N" in help_text
    assert "--bot.long.hsl_orange_tier_mode VALUE" in help_text
    assert "--bot.short.hsl_orange_tier_mode VALUE" in help_text
    assert "--bot.long.hsl_panic_close_order_type VALUE" in help_text
    assert "--bot.short.hsl_panic_close_order_type VALUE" in help_text


def test_live_default_help_shows_curated_groups():
    config = get_template_config()
    del config["optimize"]
    del config["backtest"]
    help_text = _format_parser_help_with_config("live", config, help_all=False)

    assert "Coin Selection:" in help_text
    assert "Behavior:" in help_text
    assert "Runtime:" in help_text
    assert "--symbols CSV_OR_PATH, -s CSV_OR_PATH" in help_text
    assert "--ignored-coins CSV_OR_PATH" in help_text
    assert "--minimum-coin-age-days FLOAT" in help_text
    assert "--hedge-mode Y/N" in help_text
    assert "--market-order-near-touch-threshold FLOAT, -montt FLOAT" in help_text
    assert "--pnls-max-lookback-days FLOAT|all, -pmld FLOAT|all" in help_text
    assert "--user VALUE, -u VALUE" in help_text
    assert "--live.auto_gs" not in help_text
    assert "--optimize.iters" not in help_text


def test_backtest_default_help_hides_optimize_flags_and_shows_suite_controls():
    config = get_template_config()
    help_text = _format_parser_help_with_config("backtest", config, help_all=False)

    assert "Coin Selection:" in help_text
    assert "Date Range:" in help_text
    assert "Backtest Runtime:" in help_text
    assert "Suite:" in help_text
    assert "--symbols CSV_OR_PATH, -s CSV_OR_PATH" in help_text
    assert "--ignored-coins CSV_OR_PATH" in help_text
    assert "--minimum-coin-age-days FLOAT, -mcad FLOAT" in help_text
    assert "--hedge-mode Y/N, -hm Y/N" in help_text
    assert "--market-orders-allowed Y/N, -moa Y/N" in help_text
    assert "--market-order-near-touch-threshold FLOAT, -montt FLOAT" in help_text
    assert "--max-realized-loss-pct FLOAT, -mrlp FLOAT" in help_text
    assert "--pnls-max-lookback-days FLOAT|all, -pmld FLOAT|all" in help_text
    assert "--aggregate-default VALUE" in help_text
    assert "--iters INT, -i INT" not in help_text


def test_live_reserved_pnls_lookback_alias_parses_short_and_long():
    config = get_template_config()
    del config["optimize"]
    del config["backtest"]
    parser = argparse.ArgumentParser(prog="live")
    group_map = {
        title: parser.add_argument_group(title) for title in CLI_HELP_GROUPS.get("live", [])
    }
    add_config_arguments(
        parser,
        config,
        command="live",
        help_all=False,
        group_map=group_map,
    )

    parsed_short = parser.parse_args(["-pmld", "14"])
    parsed_long = parser.parse_args(["--pnls-max-lookback-days", "21"])
    parsed_all = parser.parse_args(["--pnls-max-lookback-days", "all"])

    assert getattr(parsed_short, "live.pnls_max_lookback_days") == pytest.approx(14.0)
    assert getattr(parsed_long, "live.pnls_max_lookback_days") == pytest.approx(21.0)
    assert getattr(parsed_all, "live.pnls_max_lookback_days") == "all"


@pytest.mark.parametrize("command", ["backtest", "optimize"])
def test_dotted_pnls_lookback_override_accepts_all_for_non_live_commands(command):
    config = project_template_config_for_cli(get_template_config(), command)
    parser = argparse.ArgumentParser(prog=command)
    group_map = {
        title: parser.add_argument_group(title) for title in CLI_HELP_GROUPS.get(command, [])
    }
    add_config_arguments(
        parser,
        config,
        command=command,
        help_all=False,
        group_map=group_map,
    )

    parsed = parser.parse_args(["--live.pnls_max_lookback_days", "all"])

    assert getattr(parsed, "live.pnls_max_lookback_days") == "all"


def test_optimize_fixed_bot_runtime_overrides_parse():
    config = project_template_config_for_cli(get_template_config(), "optimize")
    parser = argparse.ArgumentParser(prog="optimize")
    group_map = {
        title: parser.add_argument_group(title) for title in CLI_HELP_GROUPS.get("optimize", [])
    }
    add_config_arguments(
        parser,
        config,
        command="optimize",
        help_all=False,
        group_map=group_map,
    )

    parsed = parser.parse_args(
        [
            "--bot.long.entry_grid_inflation_enabled",
            "n",
            "--bot.short.hsl_enabled",
            "y",
            "--bot.long.hsl_orange_tier_mode",
            "tp_only",
            "--bot.short.hsl_panic_close_order_type",
            "market",
        ]
    )

    assert getattr(parsed, "bot.long.entry_grid_inflation_enabled") is False
    assert getattr(parsed, "bot.short.hsl_enabled") is True
    assert getattr(parsed, "bot.long.hsl_orange_tier_mode") == "tp_only"
    assert getattr(parsed, "bot.short.hsl_panic_close_order_type") == "market"


def test_backtest_reserved_execution_live_aliases_parse_short_and_long():
    config = project_template_config_for_cli(get_template_config(), "backtest")
    parser = argparse.ArgumentParser(prog="backtest")
    group_map = {
        title: parser.add_argument_group(title) for title in CLI_HELP_GROUPS.get("backtest", [])
    }
    add_config_arguments(
        parser,
        config,
        command="backtest",
        help_all=False,
        group_map=group_map,
    )

    parsed_market = parser.parse_args(["-moa", "y"])
    parsed_hedge = parser.parse_args(["-hm", "n"])
    parsed_near_touch = parser.parse_args(["-montt", "0.002"])
    parsed_lookback = parser.parse_args(["-pmld", "14"])
    parsed_loss = parser.parse_args(["-mrlp", "0.75"])

    assert getattr(parsed_market, "live.market_orders_allowed") is True
    assert getattr(parsed_hedge, "live.hedge_mode") is False
    assert getattr(parsed_near_touch, "live.market_order_near_touch_threshold") == pytest.approx(
        0.002
    )
    assert getattr(parsed_lookback, "live.pnls_max_lookback_days") == pytest.approx(14.0)
    assert getattr(parsed_loss, "live.max_realized_loss_pct") == pytest.approx(0.75)


def test_backtest_default_help_shows_live_near_touch_threshold_override():
    config = get_template_config()
    help_text = _format_parser_help_with_config("backtest", config, help_all=False)

    assert "--market-order-near-touch-threshold FLOAT, -montt FLOAT" in help_text


def test_runtime_registry_reserved_help_metadata_stays_in_sync():
    for full_name, rule in FIELD_RUNTIME_RULES.items():
        spec = RESERVED_CLI_ARGS.get(full_name)
        if spec is None:
            continue
        assert set(rule.get("cli_exposed_on", set())).issubset(spec.get("commands", set()))
        for command, group in rule.get("help_group", {}).items():
            assert spec.get("group", {}).get(command) == group


def test_project_template_config_for_cli_backtest_keeps_inherited_live_runtime_fields():
    config = project_template_config_for_cli(get_template_config(), "backtest")

    assert "market_orders_allowed" in config["live"]
    assert "market_order_near_touch_threshold" in config["live"]
    assert "pnls_max_lookback_days" in config["live"]
    assert "user" not in config["live"]


def test_project_template_config_for_cli_optimize_keeps_inherited_live_runtime_fields():
    config = project_template_config_for_cli(get_template_config(), "optimize")

    assert "market_orders_allowed" in config["live"]
    assert "market_order_near_touch_threshold" in config["live"]
    assert "pnls_max_lookback_days" in config["live"]
    assert "user" not in config["live"]


def test_live_reserved_user_alias_parses_short_and_long():
    config = get_template_config()
    del config["optimize"]
    del config["backtest"]
    parser = argparse.ArgumentParser(prog="live")
    group_map = {
        title: parser.add_argument_group(title) for title in CLI_HELP_GROUPS.get("live", [])
    }
    add_config_arguments(
        parser,
        config,
        command="live",
        help_all=False,
        group_map=group_map,
    )

    parsed_short = parser.parse_args(["-u", "binance_01"])
    parsed_long = parser.parse_args(["--user", "bybit_02"])

    assert getattr(parsed_short, "live.user") == "binance_01"
    assert getattr(parsed_long, "live.user") == "bybit_02"
