from copy import deepcopy
import json
import logging

import passivbot_rust as pbr
import pytest

from config import (
    CONFIG_SCHEMA_VERSION,
    compile_runtime_config,
    get_template_config,
    load_prepared_config,
    prepare_config,
    project_config,
)
from config.optimize_bounds import get_optimize_bounds_defaults
from config.migrations.trailing_grid_v7 import migrate_v7_trailing_grid_config
from config.overrides import parse_overrides
from config.strategy_spec import get_supported_strategy_kinds
from tools.migrate_config_v7 import main as migrate_config_v7_main


def _set_path(mapping, path, value):
    current = mapping
    for part in path[:-1]:
        current = current.setdefault(part, {})
    current[path[-1]] = value


def _nested_strategy_values_from_spec(spec, field):
    result = {"long": {}, "short": {}}
    for param in spec["parameters"]:
        config_path = param["config_path"]
        pside = config_path[1]
        _set_path(result[pside], config_path[2:], param[field])
    return result


def _flatten_strategy_bound_items(bounds, prefix=()):
    for key, value in bounds.items():
        path = (*prefix, key)
        if isinstance(value, dict):
            yield from _flatten_strategy_bound_items(value, path)
        else:
            yield "_".join(path), value


def _strategy_side(config, pside, kind=None):
    if kind is None:
        kind = config["live"]["strategy_kind"]
    return config["bot"][pside]["strategy"][kind]


@pytest.mark.parametrize("strategy_kind", list(get_supported_strategy_kinds()))
def test_rust_strategy_spec_matches_python_template_defaults(strategy_kind):
    source = get_template_config()
    source["live"]["strategy_kind"] = strategy_kind
    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)
    expected = _nested_strategy_values_from_spec(pbr.get_strategy_spec(strategy_kind), "default")

    assert _strategy_side(prepared, "long", strategy_kind) == expected["long"]
    assert _strategy_side(prepared, "short", strategy_kind) == expected["short"]


@pytest.mark.parametrize("strategy_kind", list(get_supported_strategy_kinds()))
def test_rust_strategy_spec_matches_generated_strategy_optimize_bounds(strategy_kind):
    spec = pbr.get_strategy_spec(strategy_kind)
    expected = spec["optimize_bounds"]
    generated = get_optimize_bounds_defaults()

    flat_generated = {}
    for pside in ("long", "short"):
        strategy_bounds = generated[pside]["strategy"][strategy_kind]
        for local_key, value in _flatten_strategy_bound_items(strategy_bounds):
            flat_generated[f"{pside}_{local_key}"] = value

    assert flat_generated == expected


def _minimal_v7_trailing_grid_config():
    return {
        "config_version": "v7.12.0",
        "live": {
            "leverage": 3,
        },
        "backtest": {
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
        },
        "bot": {
            "long": {
                "n_positions": 7,
                "total_wallet_exposure_limit": 1.5,
                "risk_wel_enforcer_threshold": 0.99,
                "risk_twel_enforcer_threshold": 0.985,
                "risk_we_excess_allowance_pct": 0.1,
                "forager_volatility_ema_span": 120,
                "forager_volume_ema_span": 760,
                "ema_span_0": 385.0,
                "ema_span_1": 620.0,
                "entry_grid_double_down_factor": 1.39,
                "entry_grid_spacing_pct": 0.02312,
                "entry_grid_spacing_we_weight": 0.6766,
                "entry_grid_spacing_volatility_weight": 17.8,
                "entry_initial_ema_dist": 0.0078,
                "entry_initial_qty_pct": 0.0122,
                "entry_trailing_double_down_factor": 1.0,
                "entry_trailing_grid_ratio": -0.32,
                "entry_trailing_retracement_pct": 0.01498,
                "entry_trailing_retracement_we_weight": 4.958,
                "entry_trailing_retracement_volatility_weight": 37.9,
                "entry_trailing_threshold_pct": 0.00215,
                "entry_trailing_threshold_we_weight": 4.243,
                "entry_trailing_threshold_volatility_weight": 15.2,
                "entry_volatility_ema_span_hours": 1909.0,
                "close_grid_markup_start": 0.01041,
                "close_grid_markup_end": 0.00241,
                "close_grid_qty_pct": 0.88,
                "close_trailing_grid_ratio": -0.07,
                "close_trailing_qty_pct": 0.89,
                "close_trailing_retracement_pct": 0.00413,
                "close_trailing_threshold_pct": 0.0125,
            },
            "short": {
                "n_positions": 1,
                "total_wallet_exposure_limit": 0.0,
                "risk_wel_enforcer_threshold": 0.8,
                "risk_twel_enforcer_threshold": 0.8,
                "risk_we_excess_allowance_pct": 0.0,
                "forager_volatility_ema_span": 10,
                "forager_volume_ema_span": 360,
                "ema_span_0": 300.0,
                "ema_span_1": 700.0,
                "entry_grid_double_down_factor": 1.0,
                "entry_grid_spacing_pct": 0.02,
                "entry_grid_spacing_we_weight": 1.0,
                "entry_grid_spacing_volatility_weight": 10.0,
                "entry_initial_ema_dist": 0.01,
                "entry_initial_qty_pct": 0.01,
                "entry_trailing_double_down_factor": 1.0,
                "entry_trailing_grid_ratio": -0.7,
                "entry_trailing_retracement_pct": 0.01,
                "entry_trailing_retracement_we_weight": 1.0,
                "entry_trailing_retracement_volatility_weight": 10.0,
                "entry_trailing_threshold_pct": 0.002,
                "entry_trailing_threshold_we_weight": 1.0,
                "entry_trailing_threshold_volatility_weight": 10.0,
                "entry_volatility_ema_span_hours": 1000.0,
                "close_grid_markup_start": 0.00402,
                "close_grid_markup_end": 0.00223,
                "close_grid_qty_pct": 0.5,
                "close_trailing_grid_ratio": -0.03,
                "close_trailing_qty_pct": 0.5,
                "close_trailing_retracement_pct": 0.005,
                "close_trailing_threshold_pct": 0.005,
            },
        },
        "optimize": {
            "bounds": {
                "long_close_grid_markup_start": [0.0015, 0.012, 1e-05],
                "long_close_grid_markup_end": [-0.1, 0.012, 1e-05],
                "long_entry_trailing_grid_ratio": [-0.8, -0.2, 0.01],
                "long_forager_volatility_ema_span": [10, 720, 1],
                "long_n_positions": [2, 7, 1],
            }
        },
        "coin_overrides": {
            "XMR": {
                "bot": {
                    "long": {
                        "entry_trailing_grid_ratio": 1.0,
                        "close_grid_markup_start": 0.02,
                    }
                }
            }
        },
    }


def test_migrate_v7_trailing_grid_config_outputs_canonical_v8_strategy_shape():
    migrated, report = migrate_v7_trailing_grid_config(_minimal_v7_trailing_grid_config())

    assert migrated["config_version"] == CONFIG_SCHEMA_VERSION
    assert migrated["live"]["strategy_kind"] == "trailing_grid_v7"
    assert set(migrated["bot"]["long"]["strategy"]) == {"trailing_grid_v7"}
    long_strategy = migrated["bot"]["long"]["strategy"]["trailing_grid_v7"]
    assert long_strategy["ema_span_0"] == pytest.approx(385.0)
    assert long_strategy["entry"]["trailing_grid_ratio"] == pytest.approx(-0.32)
    assert long_strategy["entry"]["volatility_ema_span_hours"] == pytest.approx(1909.0)
    assert long_strategy["close"]["grid_markup_start"] == pytest.approx(0.01041)
    assert long_strategy["close"]["grid_markup_end"] == pytest.approx(0.00241)
    assert migrated["backtest"]["candle_interval_minutes"] == 1
    assert migrated["bot"]["long"]["risk"]["n_positions"] == 7
    assert migrated["bot"]["long"]["risk"]["entry_cooldown_minutes"] == pytest.approx(0.0)
    assert migrated["bot"]["long"]["risk"]["we_excess_allowance_mode"] == "bounded"
    assert migrated["bot"]["long"]["forager"]["volatility_ema_span_1m"] == 120
    assert migrated["bot"]["long"]["forager"]["volume_ema_span_1m"] == 760
    assert migrated["optimize"]["bounds"]["long"]["risk"]["entry_cooldown_minutes"] == [
        0.0,
        0.0,
        0.1,
    ]
    assert (
        migrated["optimize"]["bounds"]["long"]["strategy"]["trailing_grid_v7"]["close"][
            "grid_markup_start"
        ]
        == [0.0015, 0.012, 1e-05]
    )
    assert (
        migrated["optimize"]["bounds"]["long"]["strategy"]["trailing_grid_v7"]["entry"][
            "trailing_grid_ratio"
        ]
        == [-0.8, -0.2, 0.01]
    )
    assert migrated["optimize"]["bounds"]["long"]["forager"]["volatility_ema_span_1m"] == [
        10,
        720,
        1,
    ]
    override = migrated["coin_overrides"]["XMR"]["bot"]["long"]["strategy"]["trailing_grid_v7"]
    assert override["entry"]["trailing_grid_ratio"] == pytest.approx(1.0)
    assert override["close"]["grid_markup_start"] == pytest.approx(0.02)
    assert report["destination_strategy_kind"] == "trailing_grid_v7"
    assert any("entry_trailing_grid_ratio" in item for item in report["moved_fields"])
    assert any(
        "entry_cooldown_minutes was not a v7 parameter" in item
        for item in report["warnings"]
    )


def test_migrate_v7_trailing_grid_warns_when_v7_raw_excess_would_be_clamped():
    source = _minimal_v7_trailing_grid_config()
    source["bot"]["long"]["n_positions"] = 1
    source["bot"]["long"]["total_wallet_exposure_limit"] = 1.0
    source["bot"]["long"]["risk_we_excess_allowance_pct"] = 0.1

    migrated, report = migrate_v7_trailing_grid_config(source)

    assert migrated["bot"]["long"]["risk"]["we_excess_allowance_mode"] == "bounded"
    assert any(
        "bot.long.risk.we_excess_allowance_pct=0.1" in item
        and "legacy_raw" in item
        and "above side TWEL" in item
        for item in report["warnings"]
    )


def test_migrate_v7_trailing_grid_warns_for_coin_override_raw_excess_clamp():
    source = _minimal_v7_trailing_grid_config()
    source["bot"]["long"]["n_positions"] = 1
    source["bot"]["long"]["total_wallet_exposure_limit"] = 1.0
    source["bot"]["long"]["risk_we_excess_allowance_pct"] = 0.0
    source["coin_overrides"]["BTC"] = {
        "bot": {
            "long": {
                "risk_we_excess_allowance_pct": 0.1,
            }
        }
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    assert migrated["coin_overrides"]["BTC"]["bot"]["long"]["risk"][
        "we_excess_allowance_pct"
    ] == pytest.approx(0.1)
    assert any(
        "coin_overrides.BTC.bot.long.risk.we_excess_allowance_pct=0.1" in item
        and "legacy_raw" in item
        for item in report["warnings"]
    )


def test_migrate_v7_trailing_grid_warns_for_coin_override_explicit_wel_clamp():
    source = _minimal_v7_trailing_grid_config()
    source["bot"]["long"]["n_positions"] = 4
    source["bot"]["long"]["total_wallet_exposure_limit"] = 1.0
    source["bot"]["long"]["risk_we_excess_allowance_pct"] = 0.0
    source["coin_overrides"]["BTC"] = {
        "bot": {
            "long": {
                "wallet_exposure_limit": 0.8,
                "risk_we_excess_allowance_pct": 0.5,
            }
        }
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    assert migrated["coin_overrides"]["BTC"]["bot"]["long"][
        "wallet_exposure_limit"
    ] == pytest.approx(0.8)
    assert any(
        "coin_overrides.BTC.bot.long.risk.we_excess_allowance_pct=0.5" in item
        and "base WEL = 0.8" in item
        and "above side TWEL 1" in item
        for item in report["warnings"]
    )


def test_migrate_v7_trailing_grid_reports_inserted_v8_defaults():
    source = _minimal_v7_trailing_grid_config()
    source["backtest"].pop("candle_interval_minutes", None)
    source["live"].pop("approved_coins", None)
    source["bot"]["long"].pop("risk_wel_enforcer_enabled", None)
    source["bot"]["long"].pop("close_grid_markup_end", None)

    migrated, report = migrate_v7_trailing_grid_config(source)

    assert migrated["backtest"]["candle_interval_minutes"] == 1
    assert (
        migrated["bot"]["long"]["strategy"]["trailing_grid_v7"]["close"][
            "grid_markup_end"
        ]
        == pytest.approx(0.00241)
    )
    assert "backtest.candle_interval_minutes" in report["inserted_v8_defaults"]
    assert "backtest.liquidation_threshold" in report["inserted_v8_defaults"]
    assert "backtest.maker_fee_override" in report["inserted_v8_defaults"]
    assert "backtest.market_order_slippage_pct" in report["inserted_v8_defaults"]
    assert "backtest.starting_balance" in report["inserted_v8_defaults"]
    assert "backtest.taker_fee_override" in report["inserted_v8_defaults"]
    assert "live.approved_coins" in report["inserted_v8_defaults"]
    assert "live.forager_score_hysteresis_pct" in report["inserted_v8_defaults"]
    assert "live.hsl_position_during_cooldown_policy" in report["inserted_v8_defaults"]
    assert "live.hsl_signal_mode" in report["inserted_v8_defaults"]
    assert (
        "bot.long.risk.position_exposure_enforcer_enabled"
        in report["inserted_v8_defaults"]
    )
    assert (
        "bot.long.strategy.trailing_grid_v7.close.grid_markup_end"
        in report["inserted_v8_defaults"]
    )
    assert (
        "optimize.bounds.long.strategy.trailing_grid_v7.ema_span_0"
        in report["inserted_v8_defaults"]
    )
    assert (
        "optimize.bounds.long.strategy.trailing_grid_v7.entry.grid_spacing_pct"
        in report["inserted_v8_defaults"]
    )
    assert any("inserted v8 default values" in item for item in report["warnings"])


def test_migrate_v7_trailing_grid_zero_enforcer_thresholds_disable_enforcers():
    source = _minimal_v7_trailing_grid_config()
    for pside in ("long", "short"):
        source["bot"][pside].pop("risk_wel_enforcer_enabled", None)
        source["bot"][pside].pop("risk_twel_enforcer_enabled", None)
        source["bot"][pside]["risk_wel_enforcer_threshold"] = 0
        source["bot"][pside]["risk_twel_enforcer_threshold"] = 0

    migrated, report = migrate_v7_trailing_grid_config(source)

    for pside in ("long", "short"):
        risk = migrated["bot"][pside]["risk"]
        assert risk["position_exposure_enforcer_threshold"] == 0
        assert risk["total_exposure_enforcer_threshold"] == 0
        assert risk["position_exposure_enforcer_enabled"] is False
        assert risk["total_exposure_enforcer_enabled"] is False
        assert (
            f"bot.{pside}.risk.position_exposure_enforcer_enabled"
            not in report["inserted_v8_defaults"]
        )
        assert (
            f"bot.{pside}.risk.total_exposure_enforcer_enabled"
            not in report["inserted_v8_defaults"]
        )
    assert any("position_exposure_enforcer_enabled set false" in item for item in report["warnings"])
    assert any("total_exposure_enforcer_enabled set false" in item for item in report["warnings"])


def test_migrate_config_v7_cli_clean_migration_writes_output_and_returns_zero(tmp_path):
    input_path = tmp_path / "legacy.json"
    output_path = tmp_path / "migrated.json"
    input_path.write_text(
        json.dumps(_minimal_v7_trailing_grid_config()),
        encoding="utf-8",
    )

    rc = migrate_config_v7_main([str(input_path), str(output_path)])

    assert rc == 0
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert loaded["live"]["strategy_kind"] == "trailing_grid_v7"


def test_migrate_config_v7_cli_prints_migration_warnings(tmp_path, capsys):
    input_path = tmp_path / "legacy.json"
    output_path = tmp_path / "migrated.json"
    source = _minimal_v7_trailing_grid_config()
    source["bot"]["long"]["n_positions"] = 1
    source["bot"]["long"]["total_wallet_exposure_limit"] = 1.0
    source["bot"]["long"]["risk_we_excess_allowance_pct"] = 0.1
    input_path.write_text(json.dumps(source), encoding="utf-8")

    rc = migrate_config_v7_main([str(input_path), str(output_path)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "warning: bot.long.risk.we_excess_allowance_pct=0.1" in captured.err
    assert "legacy_raw" in captured.err


def test_migrate_config_v7_cli_unresolved_returns_nonzero_without_output_but_writes_report(
    tmp_path,
):
    source = _minimal_v7_trailing_grid_config()
    source["custom_top_level_section"] = {"important": 1}
    input_path = tmp_path / "legacy.json"
    output_path = tmp_path / "migrated.json"
    report_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(source), encoding="utf-8")

    rc = migrate_config_v7_main(
        [str(input_path), str(output_path), "--report", str(report_path)]
    )

    assert rc == 1
    assert not output_path.exists()
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "manual_review_required"
    assert report["output_written"] is False
    assert "custom_top_level_section" in report["manual_review_fields"]


def test_migrate_config_v7_cli_override_writes_unresolved_best_effort_output(tmp_path):
    source = _minimal_v7_trailing_grid_config()
    source["custom_top_level_section"] = {"important": 1}
    input_path = tmp_path / "legacy.json"
    output_path = tmp_path / "migrated.json"
    report_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(source), encoding="utf-8")

    rc = migrate_config_v7_main(
        [
            str(input_path),
            str(output_path),
            "--report",
            str(report_path),
            "--allow-manual-review-output",
        ]
    )

    assert rc == 0
    assert output_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "unsafe_manual_review_output_written"
    assert report["output_written"] is True
    assert report["allow_manual_review_output"] is True


def test_migrate_v7_trailing_grid_rejects_root_ema_only_config():
    source = {
        "bot": {
            "long": {"ema_span_0": 10.0, "ema_span_1": 20.0},
            "short": {"ema_span_0": 10.0, "ema_span_1": 20.0},
        }
    }

    with pytest.raises(ValueError, match="v7-distinctive entry/close"):
        migrate_v7_trailing_grid_config(source)


def test_migrated_v7_trailing_grid_config_prepares_and_validates():
    migrated, _report = migrate_v7_trailing_grid_config(_minimal_v7_trailing_grid_config())

    prepared = prepare_config(migrated, verbose=False, target="canonical", runtime=None)

    assert prepared["live"]["strategy_kind"] == "trailing_grid_v7"
    assert set(prepared["bot"]["long"]["strategy"]) == {"trailing_grid_v7"}
    assert set(prepared["optimize"]["bounds"]["long"]["strategy"]) == {"trailing_grid_v7"}


def test_migrate_v7_trailing_grid_coin_override_preserves_wallet_exposure_limit():
    source = _minimal_v7_trailing_grid_config()
    source["coin_overrides"]["BTC"] = {
        "bot": {
            "long": {
                "ema_span_0": 3.0,
                "wallet_exposure_limit": 0.25,
                "risk_we_excess_allowance_pct": 0.2,
                "risk_we_excess_allowance_mode": "LEGACY_RAW",
            }
        }
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    long_override = migrated["coin_overrides"]["BTC"]["bot"]["long"]
    assert long_override["wallet_exposure_limit"] == pytest.approx(0.25)
    assert long_override["risk"]["we_excess_allowance_pct"] == pytest.approx(0.2)
    assert long_override["risk"]["we_excess_allowance_mode"] == "LEGACY_RAW"
    assert long_override["strategy"]["trailing_grid_v7"]["ema_span_0"] == pytest.approx(3.0)
    assert (
        "coin_overrides.BTC.bot.long.wallet_exposure_limit -> "
        "coin_overrides.BTC.bot.long.wallet_exposure_limit"
    ) in report["moved_fields"]
    prepared = prepare_config(migrated, verbose=False, target="canonical", runtime=None)
    parsed = parse_overrides(prepared, verbose=False)
    parsed_override = parsed["coin_overrides"]["BTC"]["bot"]["long"]
    assert parsed_override["wallet_exposure_limit"] == pytest.approx(0.25)
    assert parsed_override["risk"]["we_excess_allowance_pct"] == pytest.approx(0.2)
    assert parsed_override["risk"]["we_excess_allowance_mode"] == "legacy_raw"


def test_migrate_v7_trailing_grid_coin_override_reports_runtime_unsupported_risk_fields():
    source = _minimal_v7_trailing_grid_config()
    source["coin_overrides"]["BTC"] = {
        "bot": {
            "long": {
                "n_positions": 1,
                "total_wallet_exposure_limit": 0.2,
                "risk_we_excess_allowance_pct": 0.2,
            }
        }
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    long_override = migrated["coin_overrides"]["BTC"]["bot"]["long"]
    assert "n_positions" not in long_override.get("risk", {})
    assert "total_wallet_exposure_limit" not in long_override.get("risk", {})
    assert long_override["risk"]["we_excess_allowance_pct"] == pytest.approx(0.2)
    assert "coin_overrides.BTC.bot.long.n_positions" in report["manual_review_fields"]
    assert (
        "coin_overrides.BTC.bot.long.total_wallet_exposure_limit"
        in report["manual_review_fields"]
    )
    prepared = prepare_config(migrated, verbose=False, target="canonical", runtime=None)
    parsed = parse_overrides(prepared, verbose=False)
    parsed_override = parsed["coin_overrides"]["BTC"]["bot"]["long"]
    assert parsed_override["risk"]["we_excess_allowance_pct"] == pytest.approx(0.2)
    assert "n_positions" not in parsed_override.get("risk", {})
    assert "total_wallet_exposure_limit" not in parsed_override.get("risk", {})


def test_migrate_v7_trailing_grid_coin_override_reports_unsupported_shared_alias_fields():
    source = _minimal_v7_trailing_grid_config()
    source["coin_overrides"]["BTC"] = {
        "bot": {
            "long": {
                "entry_trailing_grid_ratio": 0.5,
                "hsl_tier_ratio_yellow": 0.45,
                "hsl_tier_ratio_orange": 0.75,
                "forager_score_weights": {
                    "volume": 0.2,
                    "ema_readiness": 0.3,
                    "volatility": 0.5,
                },
                "forager_volatility_ema_span": 222.0,
                "filter_volume_ema_span": 333.0,
            }
        }
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    long_override = migrated["coin_overrides"]["BTC"]["bot"]["long"]
    assert "hsl" not in long_override
    assert "forager" not in long_override
    for key in (
        "hsl_tier_ratio_yellow",
        "hsl_tier_ratio_orange",
        "forager_score_weights",
        "forager_volatility_ema_span",
        "filter_volume_ema_span",
    ):
        source_path = f"coin_overrides.BTC.bot.long.{key}"
        assert source_path in report["manual_review_fields"]
        assert not any(moved.startswith(f"{source_path} ->") for moved in report["moved_fields"])

    prepared = prepare_config(migrated, verbose=False, target="canonical", runtime=None)
    parsed = parse_overrides(
        prepared,
        verbose=False,
        symbol_normalizer=lambda coin: coin,
    )
    parsed_override = parsed["coin_overrides"]["BTC"]["bot"]["long"]
    assert "hsl" not in parsed_override
    assert "forager" not in parsed_override


def test_migrate_v7_trailing_grid_coin_override_reports_unsupported_fields():
    source = _minimal_v7_trailing_grid_config()
    source["coin_overrides"]["BTC"] = {
        "foo": {"bar": 1},
        "bot": {
            "long": {
                "ema_span_0": 3.0,
                "mystery": 42,
                "risk": {"mystery": 1},
            }
        },
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    assert "coin_overrides.BTC.foo" in report["manual_review_fields"]
    assert "coin_overrides.BTC.bot.long.mystery" in report["manual_review_fields"]
    assert "coin_overrides.BTC.bot.long.risk" in report["manual_review_fields"]
    assert "mystery" not in migrated["coin_overrides"]["BTC"]["bot"]["long"]
    assert "risk" not in migrated["coin_overrides"]["BTC"]["bot"]["long"]
    assert "foo" not in migrated["coin_overrides"]["BTC"]


def test_migrate_v7_trailing_grid_coin_override_reports_unknown_live_keys():
    source = _minimal_v7_trailing_grid_config()
    source["coin_overrides"]["BTC"] = {
        "live": {
            "leverage": 5,
            "unknown_live_toggle": True,
        },
        "bot": {"long": {"entry_trailing_grid_ratio": 0.5}},
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    assert migrated["coin_overrides"]["BTC"]["live"]["leverage"] == 5
    assert "unknown_live_toggle" not in migrated["coin_overrides"]["BTC"]["live"]
    assert "coin_overrides.BTC.live.unknown_live_toggle" in report["manual_review_fields"]
    assert (
        "coin_overrides.BTC.live.leverage -> coin_overrides.BTC.live.leverage"
        in report["moved_fields"]
    )


def test_migrate_v7_trailing_grid_reports_unknown_top_level_fields():
    source = _minimal_v7_trailing_grid_config()
    source["custom_top_level_section"] = {"important": 1}

    _migrated, report = migrate_v7_trailing_grid_config(source)

    assert "custom_top_level_section" in report["manual_review_fields"]


def test_migrate_v7_trailing_grid_reports_source_strategy_subtree():
    source = _minimal_v7_trailing_grid_config()
    source["bot"]["long"]["strategy"] = {"custom_or_old_shape": {"x": 1}}

    _migrated, report = migrate_v7_trailing_grid_config(source)

    assert "bot.long.strategy" in report["manual_review_fields"]


def test_migrate_v7_trailing_grid_reports_grouped_bot_side_sections():
    source = _minimal_v7_trailing_grid_config()
    source["bot"]["long"]["risk"] = {"n_positions": 3}

    migrated, report = migrate_v7_trailing_grid_config(source)

    assert "bot.long.risk" in report["manual_review_fields"]
    assert not any(moved.startswith("bot.long.risk ->") for moved in report["moved_fields"])
    assert migrated["bot"]["long"]["risk"]["n_positions"] == 7


def test_migrate_v7_trailing_grid_reports_top_level_wallet_exposure_limit():
    source = _minimal_v7_trailing_grid_config()
    source["bot"]["long"]["wallet_exposure_limit"] = 0.123

    migrated, report = migrate_v7_trailing_grid_config(source)

    assert "bot.long.wallet_exposure_limit" in report["manual_review_fields"]
    assert not any(
        moved.startswith("bot.long.wallet_exposure_limit ->")
        for moved in report["moved_fields"]
    )
    assert "wallet_exposure_limit" not in migrated["bot"]["long"]


def test_migrate_v7_trailing_grid_old_bound_alias_prepares():
    source = _minimal_v7_trailing_grid_config()
    source["optimize"]["bounds"] = {
        "long_entry_grid_spacing_weight": [0.1, 2.0, 0.1],
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    entry_bounds = migrated["optimize"]["bounds"]["long"]["strategy"]["trailing_grid_v7"][
        "entry"
    ]
    assert entry_bounds["grid_spacing_we_weight"] == [0.1, 2.0, 0.1]
    assert "grid_spacing_weight" not in entry_bounds
    assert (
        "optimize.bounds.long_entry_grid_spacing_weight -> "
        "optimize.bounds.long_entry_grid_spacing_we_weight"
    ) in report["moved_fields"]
    prepare_config(migrated, verbose=False, target="canonical", runtime=None)


def test_migrate_v7_trailing_grid_old_filter_side_aliases_to_forager():
    source = _minimal_v7_trailing_grid_config()
    long_side = source["bot"]["long"]
    for key in (
        "forager_volatility_ema_span",
        "forager_volume_ema_span",
        "forager_volume_drop_pct",
    ):
        long_side.pop(key, None)
    long_side.update(
        {
            "filter_volatility_ema_span": 111.0,
            "filter_volume_ema_span": 555.0,
            "filter_volume_drop_pct": 0.24,
        }
    )

    migrated, report = migrate_v7_trailing_grid_config(source)

    forager = migrated["bot"]["long"]["forager"]
    assert forager["volatility_ema_span_1m"] == pytest.approx(111.0)
    assert forager["volume_ema_span_1m"] == pytest.approx(555.0)
    assert forager["volume_drop_pct"] == pytest.approx(0.24)
    for key in (
        "filter_volatility_ema_span",
        "filter_volume_ema_span",
        "filter_volume_drop_pct",
    ):
        assert f"bot.long.{key}" not in report["manual_review_fields"]
    assert (
        "bot.long.filter_volatility_ema_span -> "
        "bot.long.forager.volatility_ema_span_1m"
    ) in report["moved_fields"]
    assert (
        "bot.long.filter_volume_ema_span -> bot.long.forager.volume_ema_span_1m"
    ) in report["moved_fields"]
    prepare_config(migrated, verbose=False, target="canonical", runtime=None)


def test_migrate_v7_trailing_grid_forager_aliases_win_over_old_filter_aliases():
    source = _minimal_v7_trailing_grid_config()
    source["bot"]["long"].update(
        {
            "forager_volatility_ema_span": 120.0,
            "filter_volatility_ema_span": 999.0,
            "forager_volume_ema_span": 760.0,
            "filter_volume_ema_span": 888.0,
        }
    )

    migrated, report = migrate_v7_trailing_grid_config(source)

    forager = migrated["bot"]["long"]["forager"]
    assert forager["volatility_ema_span_1m"] == pytest.approx(120.0)
    assert forager["volume_ema_span_1m"] == pytest.approx(760.0)
    assert not any(
        "bot.long.filter_volatility_ema_span" in item
        for item in report["manual_review_fields"]
    )
    assert not any(
        "bot.long.filter_volume_ema_span" in item
        for item in report["manual_review_fields"]
    )
    assert not any(
        moved.startswith("bot.long.filter_volatility_ema_span ->")
        for moved in report["moved_fields"]
    )
    assert not any(
        moved.startswith("bot.long.filter_volume_ema_span ->")
        for moved in report["moved_fields"]
    )


def test_migrate_v7_trailing_grid_strategy_alias_collision_keeps_newer_value():
    source = _minimal_v7_trailing_grid_config()
    source["bot"]["long"]["entry_volatility_ema_span_hours"] = 111.0
    source["bot"]["long"]["entry_volatility_ema_span_1h"] = 222.0

    migrated, report = migrate_v7_trailing_grid_config(source)

    entry = migrated["bot"]["long"]["strategy"]["trailing_grid_v7"]["entry"]
    assert entry["volatility_ema_span_hours"] == pytest.approx(111.0)
    assert any(
        "bot.long.entry_volatility_ema_span_1h conflicts with "
        "bot.long.entry_volatility_ema_span_hours" in item
        for item in report["manual_review_fields"]
    )
    assert not any(
        moved.startswith("bot.long.entry_volatility_ema_span_1h ->")
        for moved in report["moved_fields"]
    )


def test_migrate_v7_trailing_grid_old_filter_side_alias_collision_reports_loser():
    source = _minimal_v7_trailing_grid_config()
    long_side = source["bot"]["long"]
    long_side.pop("forager_volatility_ema_span", None)
    long_side.update(
        {
            "filter_volatility_ema_span": 111.0,
            "filter_noisiness_rolling_window": 222.0,
            "filter_log_range_ema_span": 333.0,
        }
    )

    migrated, report = migrate_v7_trailing_grid_config(source)

    assert migrated["bot"]["long"]["forager"]["volatility_ema_span_1m"] == pytest.approx(111.0)
    assert any(
        "bot.long.filter_noisiness_rolling_window conflicts with "
        "bot.long.filter_volatility_ema_span" in item
        for item in report["manual_review_fields"]
    )
    assert any(
        "bot.long.filter_log_range_ema_span conflicts with "
        "bot.long.filter_volatility_ema_span" in item
        for item in report["manual_review_fields"]
    )
    assert not any(
        moved.startswith("bot.long.filter_noisiness_rolling_window ->")
        for moved in report["moved_fields"]
    )


def test_migrate_v7_trailing_grid_old_filter_bound_aliases_to_forager():
    source = _minimal_v7_trailing_grid_config()
    source["optimize"]["bounds"] = {
        "long_filter_volatility_ema_span": [10, 720, 1],
        "long_filter_volume_ema_span": [360, 2880, 10],
        "long_filter_volume_drop_pct": [0.1, 0.9, 0.01],
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    forager_bounds = migrated["optimize"]["bounds"]["long"]["forager"]
    assert forager_bounds["volatility_ema_span_1m"] == [10, 720, 1]
    assert forager_bounds["volume_ema_span_1m"] == [360, 2880, 10]
    assert forager_bounds["volume_drop_pct"] == [0.1, 0.9, 0.01]
    for key in source["optimize"]["bounds"]:
        assert f"optimize.bounds.{key}" not in report["manual_review_fields"]
    assert (
        "optimize.bounds.long_filter_volatility_ema_span -> "
        "optimize.bounds.long_forager_volatility_ema_span_1m"
    ) in report["moved_fields"]
    assert (
        "optimize.bounds.long_filter_volume_ema_span -> "
        "optimize.bounds.long_forager_volume_ema_span_1m"
    ) in report["moved_fields"]
    prepare_config(migrated, verbose=False, target="canonical", runtime=None)


def test_migrate_v7_trailing_grid_forager_bound_aliases_win_over_old_filter_aliases():
    source = _minimal_v7_trailing_grid_config()
    source["optimize"]["bounds"] = {
        "long_filter_volatility_ema_span": [10, 720, 1],
        "long_forager_volatility_ema_span": [20, 800, 1],
        "long_filter_volume_ema_span": [360, 2880, 10],
        "long_forager_volume_ema_span": [500, 3000, 10],
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    forager_bounds = migrated["optimize"]["bounds"]["long"]["forager"]
    assert forager_bounds["volatility_ema_span_1m"] == [20, 800, 1]
    assert forager_bounds["volume_ema_span_1m"] == [500, 3000, 10]
    assert not any(
        "optimize.bounds.long_filter_volatility_ema_span" in item
        for item in report["manual_review_fields"]
    )
    assert not any(
        "optimize.bounds.long_filter_volume_ema_span" in item
        for item in report["manual_review_fields"]
    )
    assert not any(
        moved.startswith("optimize.bounds.long_filter_volatility_ema_span ->")
        for moved in report["moved_fields"]
    )
    assert not any(
        moved.startswith("optimize.bounds.long_filter_volume_ema_span ->")
        for moved in report["moved_fields"]
    )
    prepare_config(migrated, verbose=False, target="canonical", runtime=None)


def test_migrate_v7_trailing_grid_flat_bound_alias_collision_keeps_canonical():
    source = _minimal_v7_trailing_grid_config()
    source["optimize"]["bounds"] = {
        "long_entry_volatility_ema_span_1h": [222, 333, 1],
        "long_entry_volatility_ema_span_hours": [111, 222, 1],
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    entry_bounds = migrated["optimize"]["bounds"]["long"]["strategy"]["trailing_grid_v7"][
        "entry"
    ]
    assert entry_bounds["volatility_ema_span_hours"] == [111, 222, 1]
    assert any(
        "optimize.bounds.long_entry_volatility_ema_span_1h conflicts with "
        "optimize.bounds.long_entry_volatility_ema_span_hours" in item
        for item in report["manual_review_fields"]
    )
    assert not any(
        moved.startswith("optimize.bounds.long_entry_volatility_ema_span_1h ->")
        for moved in report["moved_fields"]
    )
    prepare_config(migrated, verbose=False, target="canonical", runtime=None)


def test_migrate_v7_trailing_grid_old_filter_bound_alias_collision_reports_loser():
    source = _minimal_v7_trailing_grid_config()
    source["optimize"]["bounds"] = {
        "long_filter_volatility_ema_span": [10, 720, 1],
        "long_filter_noisiness_rolling_window": [20, 300, 1],
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    forager_bounds = migrated["optimize"]["bounds"]["long"]["forager"]
    assert forager_bounds["volatility_ema_span_1m"] == [10, 720, 1]
    assert any(
        "optimize.bounds.long_filter_noisiness_rolling_window conflicts with "
        "optimize.bounds.long_filter_volatility_ema_span" in item
        for item in report["manual_review_fields"]
    )
    assert not any(
        moved.startswith("optimize.bounds.long_filter_noisiness_rolling_window ->")
        for moved in report["moved_fields"]
    )
    prepare_config(migrated, verbose=False, target="canonical", runtime=None)


def test_migrate_v7_trailing_grid_uncertain_markup_bounds_require_manual_review():
    source = _minimal_v7_trailing_grid_config()
    source["optimize"]["bounds"] = {
        "long_min_markup": [0.001, 0.01, 0.001],
        "long_close_grid_min_markup": [0.002, 0.02, 0.001],
        "long_markup_range": [0.001, 0.01, 0.001],
        "long_close_grid_markup_range": [0.001, 0.02, 0.001],
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    for key in source["optimize"]["bounds"]:
        assert f"optimize.bounds.{key}" in report["manual_review_fields"]
        assert not any(key in moved for moved in report["moved_fields"])
    close_bounds = migrated["optimize"]["bounds"]["long"]["strategy"]["trailing_grid_v7"][
        "close"
    ]
    assert close_bounds["grid_markup_start"] != [0.001, 0.01, 0.001]
    assert close_bounds["grid_markup_end"] != [0.002, 0.02, 0.001]
    prepare_config(migrated, verbose=False, target="canonical", runtime=None)


def test_migrate_v7_trailing_grid_unknown_flat_bound_is_reported_not_emitted():
    source = _minimal_v7_trailing_grid_config()
    source["optimize"]["bounds"] = {
        "long_entry_totally_unknown": [0.1, 2.0, 0.1],
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    entry_bounds = migrated["optimize"]["bounds"]["long"]["strategy"]["trailing_grid_v7"][
        "entry"
    ]
    assert "totally_unknown" not in entry_bounds
    assert "optimize.bounds.long_entry_totally_unknown" in report["manual_review_fields"]
    assert not any(
        "long_entry_totally_unknown" in moved for moved in report["moved_fields"]
    )
    prepare_config(migrated, verbose=False, target="canonical", runtime=None)


def test_migrate_v7_trailing_grid_unknown_nested_strategy_bound_is_reported_not_emitted():
    source = _minimal_v7_trailing_grid_config()
    source["optimize"]["bounds"] = {
        "long": {
            "strategy": {
                "trailing_grid": {
                    "entry": {
                        "grid_spacing_pct": [0.01, 0.03, 0.001],
                        "totally_unknown": [0.1, 2.0, 0.1],
                    }
                }
            }
        }
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    entry_bounds = migrated["optimize"]["bounds"]["long"]["strategy"]["trailing_grid_v7"][
        "entry"
    ]
    assert entry_bounds["grid_spacing_pct"] == [0.01, 0.03, 0.001]
    assert "totally_unknown" not in entry_bounds
    assert (
        "optimize.bounds.long.strategy.trailing_grid.entry.totally_unknown"
        in report["manual_review_fields"]
    )
    assert not any("totally_unknown" in moved for moved in report["moved_fields"])
    prepare_config(migrated, verbose=False, target="canonical", runtime=None)


def test_migrate_v7_trailing_grid_unknown_nested_shared_bound_is_reported_not_emitted():
    source = _minimal_v7_trailing_grid_config()
    source["optimize"]["bounds"] = {
        "long": {
            "foo": {"bar": [1, 2, 1]},
        }
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    assert "foo" not in migrated["optimize"]["bounds"]["long"]
    assert "optimize.bounds.long.foo.bar" in report["manual_review_fields"]
    assert not any("optimize.bounds.long.foo" in moved for moved in report["moved_fields"])
    prepare_config(migrated, verbose=False, target="canonical", runtime=None)


def test_migrate_v7_trailing_grid_valid_nested_shared_bound_is_moved():
    source = _minimal_v7_trailing_grid_config()
    source["optimize"]["bounds"] = {
        "long": {
            "risk": {"n_positions": [2, 5, 1]},
        }
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    assert migrated["optimize"]["bounds"]["long"]["risk"]["n_positions"] == [2, 5, 1]
    assert (
        "optimize.bounds.long.risk.n_positions -> "
        "optimize.bounds.long.risk.n_positions"
    ) in report["moved_fields"]
    assert "optimize.bounds.long.risk.n_positions" not in report["manual_review_fields"]
    prepare_config(migrated, verbose=False, target="canonical", runtime=None)


def test_migrate_v7_trailing_grid_nested_strategy_bound_alias_is_canonicalized():
    source = _minimal_v7_trailing_grid_config()
    source["optimize"]["bounds"] = {
        "long": {
            "strategy": {
                "trailing_grid": {
                    "entry": {
                        "volatility_ema_span_1h": [4.0, 20.0, 1.0],
                    }
                }
            }
        }
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    entry_bounds = migrated["optimize"]["bounds"]["long"]["strategy"]["trailing_grid_v7"][
        "entry"
    ]
    assert entry_bounds["volatility_ema_span_hours"] == [4.0, 20.0, 1.0]
    assert "volatility_ema_span_1h" not in entry_bounds
    assert (
        "optimize.bounds.long.strategy.trailing_grid.entry.volatility_ema_span_1h -> "
        "optimize.bounds.long.strategy.trailing_grid_v7.entry.volatility_ema_span_hours"
    ) in report["moved_fields"]
    prepare_config(migrated, verbose=False, target="canonical", runtime=None)


def test_migrate_v7_trailing_grid_nested_strategy_bound_alias_conflict_reports_loser():
    source = _minimal_v7_trailing_grid_config()
    source["optimize"]["bounds"] = {
        "long": {
            "strategy": {
                "trailing_grid_v7": {
                    "entry": {
                        "grid_spacing_weight": [0.1, 1.0, 0.1],
                        "grid_spacing_we_weight": [2.0, 2.0, 0.1],
                    }
                }
            }
        }
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    entry_bounds = migrated["optimize"]["bounds"]["long"]["strategy"]["trailing_grid_v7"][
        "entry"
    ]
    assert entry_bounds["grid_spacing_we_weight"] == [0.1, 1.0, 0.1]
    assert any(
        "optimize.bounds.long.strategy.trailing_grid_v7.entry.grid_spacing_we_weight "
        "conflicts with "
        "optimize.bounds.long.strategy.trailing_grid_v7.entry.grid_spacing_weight"
        in item
        for item in report["manual_review_fields"]
    )
    assert not any(
        moved.startswith(
            "optimize.bounds.long.strategy.trailing_grid_v7.entry.grid_spacing_we_weight ->"
        )
        for moved in report["moved_fields"]
    )
    prepare_config(migrated, verbose=False, target="canonical", runtime=None)


def test_migrate_v7_trailing_grid_nested_strategy_namespace_conflict_reports_loser():
    source = _minimal_v7_trailing_grid_config()
    source["optimize"]["bounds"] = {
        "long": {
            "strategy": {
                "trailing_grid": {
                    "entry": {
                        "grid_spacing_we_weight": [1.0, 1.0, 1.0],
                    }
                },
                "trailing_grid_v7": {
                    "entry": {
                        "grid_spacing_we_weight": [2.0, 2.0, 1.0],
                    }
                },
            }
        }
    }

    migrated, report = migrate_v7_trailing_grid_config(source)

    entry_bounds = migrated["optimize"]["bounds"]["long"]["strategy"]["trailing_grid_v7"][
        "entry"
    ]
    assert entry_bounds["grid_spacing_we_weight"] == [1.0, 1.0, 1.0]
    assert any(
        "optimize.bounds.long.strategy.trailing_grid_v7.entry.grid_spacing_we_weight "
        "conflicts with "
        "optimize.bounds.long.strategy.trailing_grid.entry.grid_spacing_we_weight"
        in item
        for item in report["manual_review_fields"]
    )
    assert not any(
        moved.startswith(
            "optimize.bounds.long.strategy.trailing_grid_v7.entry.grid_spacing_we_weight ->"
        )
        for moved in report["moved_fields"]
    )
    prepare_config(migrated, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize(
    ("target", "expected_sections"),
    [
        (
            "canonical",
            {
                "backtest",
                "bot",
                "coin_overrides",
                "live",
                "logging",
                "monitor",
                "optimize",
            },
        ),
        ("live", {"bot", "coin_overrides", "live", "logging", "monitor"}),
        ("backtest", {"backtest", "bot", "coin_overrides", "live", "logging"}),
        (
            "optimize",
            {"backtest", "bot", "coin_overrides", "live", "logging", "optimize"},
        ),
        ("monitor", {"live", "logging", "monitor"}),
    ],
)
def test_project_config_keeps_only_target_sections_and_metadata(
    target, expected_sections
):
    cfg = get_template_config()
    cfg["_raw"] = {"live": {"user": "raw_user"}}
    cfg["_raw_effective"] = {"live": {"user": "effective_user"}}
    cfg["_transform_log"] = [{"step": "seed"}]
    cfg["_coins_sources"] = {"approved_coins": "configs/approved_coins.json"}

    projected = project_config(cfg, target)

    metadata_keys = {
        "_raw",
        "_raw_effective",
        "_transform_log",
        "_coins_sources",
        "config_version",
    }
    assert set(projected) == expected_sections | metadata_keys
    for section in (
        "backtest",
        "bot",
        "coin_overrides",
        "live",
        "logging",
        "monitor",
        "optimize",
    ):
        if section in expected_sections:
            assert section in projected
        else:
            assert section not in projected


def test_prepare_config_canonical_omits_runtime_aliases():
    prepared = prepare_config(
        get_template_config(),
        verbose=False,
        target="canonical",
        runtime=None,
    )

    assert "n_positions" not in prepared["bot"]["long"]
    assert "total_wallet_exposure_limit" not in prepared["bot"]["long"]
    assert "risk_wel_enforcer_threshold" not in prepared["bot"]["long"]
    assert "unstuck_threshold" not in prepared["bot"]["long"]
    assert "hsl_red_threshold" not in prepared["bot"]["long"]
    assert "forager_volume_ema_span_1m" not in prepared["bot"]["long"]
    assert "filter_volume_ema_span_1m" not in prepared["bot"]["long"]
    assert "filter_volatility_ema_span_1m" not in prepared["bot"]["long"]
    assert "long_filter_volume_ema_span_1m" not in prepared["optimize"]["bounds"]
    assert "long_filter_volatility_ema_span_1m" not in prepared["optimize"]["bounds"]
    assert prepared["live"]["strategy_kind"] == "trailing_martingale"
    assert "ema_span_0" not in get_template_config()["bot"]["long"]
    assert "entry_grid_spacing_pct" not in get_template_config()["bot"]["short"]
    assert _strategy_side(prepared, "long")["ema_span_0"] == _strategy_side(
        get_template_config(), "long", "trailing_martingale"
    )["ema_span_0"]


def test_prepare_config_rejects_v7_flat_trailing_grid_fields_with_trailing_martingale():
    source = get_template_config()
    source["live"]["strategy_kind"] = "trailing_martingale"
    source["bot"]["long"]["entry_grid_spacing_pct"] = 0.01
    source["bot"]["long"]["entry_trailing_grid_ratio"] = -0.5
    source["bot"]["long"]["close_grid_markup_start"] = 0.01

    with pytest.raises(ValueError, match="passivbot tool migrate-config-v7"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize("strategy_kind", ["trailing_grid_v7", "ema_anchor"])
@pytest.mark.parametrize("flat_key", ["entry_grid_spacing_pct", "entry_volatility_ema_span_1h"])
def test_prepare_config_rejects_v7_flat_trailing_grid_fields_with_any_strategy_kind(
    strategy_kind,
    flat_key,
):
    source = get_template_config()
    source["live"]["strategy_kind"] = strategy_kind
    source["bot"]["long"][flat_key] = 0.5

    with pytest.raises(
        ValueError,
        match=rf"bot\.long\.{flat_key}.*passivbot tool migrate-config-v7",
    ):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_rejects_old_v7_flat_aliases_with_trailing_martingale():
    source = get_template_config()
    source["live"]["strategy_kind"] = "trailing_martingale"
    source["bot"]["long"]["entry_grid_spacing_weight"] = 0.5
    source["bot"]["long"]["entry_grid_spacing_log_weight"] = 0.6
    source["bot"]["long"]["entry_grid_spacing_log_span_hours"] = 12.0

    with pytest.raises(
        ValueError,
        match=(
            "bot.long.entry_grid_spacing_log_span_hours.*"
            "bot.long.entry_grid_spacing_log_weight.*"
            "bot.long.entry_grid_spacing_weight.*passivbot tool migrate-config-v7"
        ),
    ):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize(
    "flat_key",
    [
        "entry_grid_spacing_weight",
        "entry_grid_spacing_pct",
        "entry_volatility_ema_span_1h",
    ],
)
def test_parse_overrides_rejects_v7_flat_strategy_override_keys(flat_key):
    source = get_template_config()
    source["live"]["strategy_kind"] = "trailing_martingale"
    source["coin_overrides"] = {
        "BTC": {
            "bot": {
                "long": {
                    flat_key: 0.5,
                }
            }
        }
    }
    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    with pytest.raises(
        ValueError,
        match=(
            rf"coin_overrides\.BTC\.bot\.long.*unsupported flat strategy.*"
            rf"{flat_key}.*passivbot tool migrate-config-v7"
        ),
    ):
        parse_overrides(
            prepared,
            verbose=False,
            symbol_normalizer=lambda coin: coin,
        )


@pytest.mark.parametrize(
    "bound_key",
    [
        "long_hsl_enabled",
        "short_hsl_enabled",
        "long_hsl_orange_tier_mode",
        "short_hsl_orange_tier_mode",
        "long_hsl_panic_close_order_type",
        "short_hsl_panic_close_order_type",
    ],
)
def test_prepare_config_rejects_nontunable_bot_bounds(bound_key):
    cfg = get_template_config()
    cfg["optimize"]["bounds"][bound_key] = [0.0, 1.0]

    with pytest.raises(
        KeyError, match=rf"optimize bound {bound_key} must map to a numeric bot\."
    ):
        prepare_config(cfg, verbose=False, target="canonical", runtime=None)


def test_compile_runtime_config_adds_runtime_aliases_without_removing_canonical_keys():
    canonical = prepare_config(
        get_template_config(),
        verbose=False,
        target="optimize",
        runtime=None,
    )

    compiled = compile_runtime_config(canonical, runtime="optimize")

    assert compiled["bot"]["long"]["n_positions"] == canonical["bot"]["long"]["risk"][
        "n_positions"
    ]
    assert compiled["bot"]["long"]["total_wallet_exposure_limit"] == canonical["bot"]["long"][
        "risk"
    ]["total_wallet_exposure_limit"]
    assert compiled["bot"]["long"]["risk_wel_enforcer_threshold"] == canonical["bot"]["long"][
        "risk"
    ]["position_exposure_enforcer_threshold"]
    assert compiled["bot"]["long"]["unstuck_threshold"] == canonical["bot"]["long"]["unstuck"][
        "threshold"
    ]
    assert compiled["bot"]["long"]["hsl_red_threshold"] == canonical["bot"]["long"]["hsl"][
        "red_threshold"
    ]
    assert compiled["bot"]["long"]["forager_volume_ema_span_1m"] == canonical["bot"]["long"]["forager"][
        "volume_ema_span_1m"
    ]
    assert compiled["bot"]["long"]["filter_volume_ema_span_1m"] == canonical["bot"]["long"]["forager"][
        "volume_ema_span_1m"
    ]
    assert compiled["bot"]["long"]["filter_volatility_ema_span_1m"] == canonical["bot"]["long"][
        "forager"
    ]["volatility_ema_span_1m"]
    assert compiled["bot"]["long"]["filter_volatility_drop_pct"] == pytest.approx(0.0)
    assert compiled["optimize"]["bounds"] == canonical["optimize"]["bounds"]
    assert _strategy_side(compiled, "long")["ema_span_0"] == _strategy_side(canonical, "long")[
        "ema_span_0"
    ]


def test_prepare_config_preserves_nested_strategy_namespace():
    source = get_template_config()
    source["bot"]["long"]["strategy"]["trailing_martingale"]["ema_span_0"] = 321.0
    source["bot"]["short"]["strategy"]["trailing_martingale"]["entry"]["threshold_base_pct"] = 0.0123
    source["live"].pop("strategy_kind", None)

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["live"]["strategy_kind"] == "trailing_martingale"
    assert _strategy_side(prepared, "long")["ema_span_0"] == pytest.approx(321.0)
    assert _strategy_side(prepared, "short")["entry"]["threshold_base_pct"] == pytest.approx(0.0123)


def test_prepare_config_supports_ema_anchor_canonical_strategy_section():
    source = get_template_config()
    source["live"]["strategy_kind"] = "ema_anchor"
    source["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 2.5
    source["bot"]["long"]["strategy"]["ema_anchor"] = {
            "base_qty_pct": 0.02,
            "ema_span_0": 55.0,
            "ema_span_1": 144.0,
            "entry_double_down_factor": 0.8,
            "offset": 0.003,
            "offset_psize_weight": 0.2,
    }
    source["bot"]["short"]["strategy"]["ema_anchor"] = {
            "base_qty_pct": 0.03,
            "ema_span_0": 34.0,
            "ema_span_1": 89.0,
            "entry_double_down_factor": 0.4,
            "offset": 0.004,
            "offset_psize_weight": 0.1,
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)
    compiled = compile_runtime_config(prepared, runtime="backtest")

    assert prepared["live"]["strategy_kind"] == "ema_anchor"
    assert prepared["bot"]["long"]["risk"]["entry_cooldown_minutes"] == pytest.approx(2.5)
    assert _strategy_side(prepared, "long")["base_qty_pct"] == pytest.approx(0.02)
    assert _strategy_side(prepared, "long")["entry_double_down_factor"] == pytest.approx(0.8)
    assert _strategy_side(prepared, "short")["offset"] == pytest.approx(0.004)
    assert _strategy_side(prepared, "short")["entry_double_down_factor"] == pytest.approx(0.4)
    assert "base_qty_pct" not in compiled["bot"]["long"]
    assert "offset" not in compiled["bot"]["short"]
    assert compiled["bot"]["long"]["risk_entry_cooldown_minutes"] == pytest.approx(2.5)


def test_prepare_config_hydrates_ema_anchor_defaults_when_strategy_section_missing():
    source = get_template_config()
    source["live"]["strategy_kind"] = "ema_anchor"
    source["bot"]["long"]["strategy"] = {}
    source["bot"]["short"]["strategy"] = {}

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["live"]["strategy_kind"] == "ema_anchor"
    assert _strategy_side(prepared, "long")["base_qty_pct"] == pytest.approx(0.01)
    assert _strategy_side(prepared, "long")["ema_span_0"] == pytest.approx(200.0)
    assert _strategy_side(prepared, "long")["entry_double_down_factor"] == pytest.approx(0.0)
    assert _strategy_side(prepared, "short")["offset"] == pytest.approx(0.002)
    assert "base_qty_pct" not in prepared["bot"]["long"]


def test_prepare_config_rejects_negative_entry_cooldown_minutes():
    source = get_template_config()
    source["bot"]["long"]["risk"]["entry_cooldown_minutes"] = -0.1

    with pytest.raises(ValueError, match="bot.long.risk.entry_cooldown_minutes"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_normalizes_we_excess_allowance_mode_and_runtime_flattening():
    source = get_template_config()
    source["bot"]["long"]["risk"]["we_excess_allowance_mode"] = "LEGACY_RAW"

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)
    compiled = compile_runtime_config(prepared, runtime="backtest")

    assert prepared["bot"]["long"]["risk"]["we_excess_allowance_mode"] == "legacy_raw"
    assert compiled["bot"]["long"]["risk_we_excess_allowance_mode"] == "legacy_raw"


def test_prepare_config_normalizes_coin_override_we_excess_allowance_mode():
    source = get_template_config()
    source["coin_overrides"] = {
        "BTC": {
            "bot": {
                "long": {
                    "risk": {"we_excess_allowance_mode": "LEGACY_RAW"},
                },
                "short": {
                    "risk_we_excess_allowance_mode": "LEGACY_RAW",
                },
            }
        }
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["coin_overrides"]["BTC"]["bot"]["long"]["risk"][
        "we_excess_allowance_mode"
    ] == "legacy_raw"
    assert (
        prepared["coin_overrides"]["BTC"]["bot"]["short"][
            "risk_we_excess_allowance_mode"
        ]
        == "legacy_raw"
    )


def test_parse_overrides_normalizes_coin_override_we_excess_allowance_mode():
    source = get_template_config()
    source["coin_overrides"] = {
        "BTC": {
            "bot": {
                "long": {
                    "risk": {
                        "we_excess_allowance_mode": "LEGACY_RAW",
                    }
                }
            }
        }
    }

    parsed = parse_overrides(source, verbose=False)

    assert parsed["coin_overrides"]["BTC"]["bot"]["long"]["risk"][
        "we_excess_allowance_mode"
    ] == "legacy_raw"


def test_prepare_config_rejects_invalid_we_excess_allowance_mode():
    source = get_template_config()
    source["bot"]["long"]["risk"]["we_excess_allowance_mode"] = "raw"

    with pytest.raises(ValueError, match="bot.long.risk.we_excess_allowance_mode"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_rejects_invalid_coin_override_we_excess_allowance_mode():
    source = get_template_config()
    source["coin_overrides"] = {
        "BTC": {
            "bot": {
                "long": {
                    "risk": {"we_excess_allowance_mode": "raw"},
                }
            }
        }
    }

    with pytest.raises(
        ValueError,
        match="coin_overrides.BTC.bot.long.risk.we_excess_allowance_mode",
    ):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_target_and_runtime_preserve_raw_metadata_and_record_steps():
    source = get_template_config()
    source["live"]["user"] = "override_user"
    raw_snapshot = deepcopy(get_template_config())

    prepared = prepare_config(
        source,
        verbose=False,
        target="live",
        runtime="live",
        raw_snapshot=raw_snapshot,
        effective_snapshot=deepcopy(source),
    )

    assert prepared["_raw"]["live"]["user"] == raw_snapshot["live"]["user"]
    assert prepared["_raw_effective"]["live"]["user"] == "override_user"
    steps = [entry["step"] for entry in prepared["_transform_log"]]
    assert "normalize_config" in steps
    assert "project_config" in steps
    assert "compile_runtime_config" in steps
    assert "backtest" not in prepared
    assert "optimize" not in prepared
    assert "monitor" in prepared
    assert "strategy" in prepared["bot"]["long"]


def test_load_prepared_config_without_path_uses_schema_defaults_pipeline():
    prepared = load_prepared_config(
        None,
        verbose=False,
        target="backtest",
        runtime="backtest",
        log_info=False,
    )

    template = get_template_config()
    assert prepared["backtest"]["market_order_slippage_pct"] == template["backtest"]["market_order_slippage_pct"]
    assert prepared["bot"]["long"]["filter_volume_ema_span_1m"] == template["bot"]["long"]["forager"][
        "volume_ema_span_1m"
    ]
    assert _strategy_side(prepared, "long")["ema_span_0"] == _strategy_side(
        template, "long", "trailing_martingale"
    )["ema_span_0"]
    assert prepared["backtest"]["visible_metrics"] is None
    assert prepared["_raw"] == template
    assert prepared["_raw_effective"] == template


def test_load_prepared_config_accepts_rounded_forager_weights_from_saved_artifact(
    tmp_path,
):
    source = get_template_config()
    source["live"]["strategy_kind"] = "ema_anchor"
    rounded = {
        "volume": 0.323,
        "ema_readiness": 0.434,
        "volatility": 0.242,
    }
    source["bot"]["long"]["forager"]["score_weights"] = rounded
    source["bot"]["short"]["forager"]["score_weights"] = rounded
    path = tmp_path / "rounded_artifact_like.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    prepared = load_prepared_config(str(path), verbose=False, log_info=False)

    assert prepared["bot"]["long"]["forager"]["score_weights"]["volume"] == pytest.approx(
        0.3233233233233233
    )
    assert prepared["bot"]["short"]["forager"]["score_weights"]["ema_readiness"] == pytest.approx(
        0.4344344344344344
    )


def test_prepare_config_preserves_backtest_visible_metrics():
    source = {
        "backtest": {
            "visible_metrics": [
                "gain",
                "drawdown_worst_hsl",
                "hard_stop_restarts_short",
            ]
        },
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["backtest"]["visible_metrics"] == [
        "gain",
        "drawdown_worst_hsl",
        "hard_stop_restarts_short",
    ]


def test_prepare_config_rejects_unknown_backtest_visible_metrics():
    source = {
        "backtest": {"visible_metrics": ["not_a_metric"]},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(ValueError, match="unknown backtest.visible_metrics entries"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_rejects_invalid_backtest_visible_metrics_type():
    source = {
        "backtest": {"visible_metrics": "adg"},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(
        ValueError, match="backtest.visible_metrics must be null, \\[\\], or"
    ):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_assigns_current_schema_version_to_legacy_configs():
    source = {
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["config_version"] == CONFIG_SCHEMA_VERSION


def test_prepare_config_rejects_future_config_version():
    source = {
        "config_version": "v9.0.0",
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(ValueError, match="newer than supported schema"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_rejects_malformed_config_version():
    source = {
        "config_version": "banana",
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(ValueError, match="must be a semantic version"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_keeps_live_pnls_max_lookback_days_without_backtest_override():
    source = {
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {"pnls_max_lookback_days": 30.0},
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["live"]["pnls_max_lookback_days"] == pytest.approx(30.0)
    assert "pnls_max_lookback_days" not in prepared["backtest"]


def test_prepare_config_normalizes_all_pnls_max_lookback_days():
    source = {
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {"pnls_max_lookback_days": "ALL"},
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["live"]["pnls_max_lookback_days"] == "all"


def test_prepare_config_rejects_invalid_pnls_max_lookback_days_string():
    source = {
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {"pnls_max_lookback_days": "everything"},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(
        ValueError, match="live\\.pnls_max_lookback_days must be >= 0 or 'all'"
    ):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize(
    ("pside", "value", "expected"),
    [
        ("long", -1.0, r"bot\.long\.unstuck_ema_dist must be > -1\.0"),
        ("short", 1.0, r"bot\.short\.unstuck_ema_dist must be < 1\.0"),
    ],
)
def test_prepare_config_rejects_invalid_unstuck_ema_dist(pside, value, expected):
    source = {
        "backtest": {},
        "bot": {
            "long": {},
            "short": {},
        },
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }
    source["bot"][pside]["unstuck_ema_dist"] = value

    with pytest.raises(ValueError, match=expected):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_keeps_live_market_orders_allowed_without_backtest_override():
    source = {
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {"market_orders_allowed": True},
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["live"]["market_orders_allowed"] is True
    assert "market_orders_allowed" not in prepared["backtest"]


def test_template_config_defaults_market_orders_disabled():
    template = get_template_config()
    assert template["live"]["market_orders_allowed"] is False


def test_prepare_config_rejects_cancellations_not_greater_than_creations():
    source = get_template_config()
    source["live"]["max_n_creations_per_batch"] = 3
    source["live"]["max_n_cancellations_per_batch"] = 3

    with pytest.raises(
        ValueError,
        match=(
            "config\\.live\\.max_n_cancellations_per_batch must be greater than "
            "config\\.live\\.max_n_creations_per_batch"
        ),
    ):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        (
            "market_snapshot_ticker_strategy",
            "fast",
            "config\\.live\\.market_snapshot_ticker_strategy must be one of",
        ),
        (
            "forager_score_hysteresis_pct",
            -0.1,
            "config\\.live\\.forager_score_hysteresis_pct must be finite and >= 0\\.0",
        ),
        (
            "max_active_candle_tail_gap_minutes",
            0,
            "config\\.live\\.max_active_candle_tail_gap_minutes must be finite and > 0\\.0",
        ),
        (
            "max_forager_candle_refresh_seconds",
            -1,
            "config\\.live\\.max_forager_candle_refresh_seconds must be finite and > 0\\.0",
        ),
        (
            "max_forager_candle_refresh_seconds",
            0,
            "config\\.live\\.max_forager_candle_refresh_seconds must be finite and > 0\\.0",
        ),
        (
            "max_forager_candle_refresh_seconds",
            float("inf"),
            "config\\.live\\.max_forager_candle_refresh_seconds must be finite and > 0\\.0",
        ),
        (
            "max_forager_candle_refresh_seconds",
            float("nan"),
            "config\\.live\\.max_forager_candle_refresh_seconds must be finite and > 0\\.0",
        ),
        (
            "max_forager_candle_refresh_seconds",
            "not-a-number",
            "config\\.live\\.max_forager_candle_refresh_seconds must be numeric",
        ),
    ],
)
def test_prepare_config_rejects_invalid_staged_live_controls(field, value, match):
    source = get_template_config()
    source["live"][field] = value

    with pytest.raises((TypeError, ValueError), match=match):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_preserves_live_candle_budget_controls():
    source = get_template_config()
    source["live"]["defer_broad_candle_warmup"] = False
    source["live"]["max_forager_candle_staleness_minutes"] = 7.5

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["live"]["defer_broad_candle_warmup"] is False
    assert prepared["live"]["max_forager_candle_staleness_minutes"] == 7.5


def test_prepare_config_preserves_live_fill_refresh_economy_controls():
    source = get_template_config()
    source["live"]["fills_recent_overlap_minutes"] = 3.5
    source["live"]["fills_confirmation_overlap_minutes"] = 45.0

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["live"]["fills_recent_overlap_minutes"] == pytest.approx(3.5)
    assert prepared["live"]["fills_confirmation_overlap_minutes"] == pytest.approx(45.0)


def test_prepare_config_preserves_live_custom_endpoints_path():
    source = get_template_config()
    source["live"]["custom_endpoints_path"] = "configs/custom_endpoints.json"

    prepared = prepare_config(source, verbose=False, target="live", runtime="live")

    assert prepared["live"]["custom_endpoints_path"] == "configs/custom_endpoints.json"


@pytest.mark.parametrize(
    "field,value",
    [
        ("market_orders_allowed", False),
        ("market_order_near_touch_threshold", 0.0),
        ("pnls_max_lookback_days", 0.0),
    ],
)
def test_prepare_config_rejects_backtest_inherited_live_fields(field, value):
    source = {
        "backtest": {field: value},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    with pytest.raises(ValueError, match=f"backtest\\.{field}"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_prepare_config_migrates_legacy_backtest_market_slippage_key():
    source = {
        "backtest": {"panic_market_slippage_pct": 0.0015},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {},
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["backtest"]["market_order_slippage_pct"] == pytest.approx(0.0015)
    assert "panic_market_slippage_pct" not in prepared["backtest"]


def test_prepare_config_removes_empty_means_all_approved_from_canonical_shape():
    source = {
        "backtest": {},
        "bot": {"long": {}, "short": {}},
        "coin_overrides": {},
        "live": {
            "approved_coins": [],
            "ignored_coins": {"long": [], "short": []},
            "empty_means_all_approved": True,
        },
        "optimize": {"bounds": {}},
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert "empty_means_all_approved" not in prepared["live"]
    assert prepared["live"]["approved_coins"] == {"long": ["all"], "short": ["all"]}
    assert prepared["_coins_sources"]["approved_coins"] == "all"
    assert prepared["_raw"]["live"]["empty_means_all_approved"] is True
    assert prepared["_raw_effective"]["live"]["empty_means_all_approved"] is True


def test_prepare_config_warns_and_removes_entry_grid_inflation_enabled(caplog):
    source = get_template_config()
    source["bot"]["long"]["entry_grid_inflation_enabled"] = True
    source["bot"]["short"]["entry_grid_inflation_enabled"] = True

    with caplog.at_level(logging.WARNING):
        prepared = prepare_config(
            source, verbose=False, target="canonical", runtime=None
        )

    assert "entry_grid_inflation_enabled" not in prepared["bot"]["long"]
    assert "entry_grid_inflation_enabled" not in prepared["bot"]["short"]
    assert any(
        "entry_grid_inflation_enabled" in rec.message
        and "has no effect; removing it" in rec.message
        for rec in caplog.records
    )


def test_prepare_config_legacy_bot_omissions_do_not_backfill_schema_defaults(caplog):
    source = get_template_config()
    trailing_martingale = source["bot"]["long"]["strategy"]["trailing_martingale"]
    risk = source["bot"]["long"]["risk"]
    trailing_martingale.pop("volatility_ema_span_1h")
    trailing_martingale.pop("volatility_ema_span_1m")
    for key in ("threshold_volatility_1h_weight", "threshold_volatility_1m_weight", "threshold_we_weight"):
        trailing_martingale["entry"].pop(key)
    for key in (
        "total_exposure_entry_gate_enabled",
        "total_exposure_enforcer_policy",
        "total_exposure_enforcer_threshold",
        "we_excess_allowance_pct",
        "position_exposure_enforcer_threshold",
    ):
        risk.pop(key)

    with caplog.at_level(logging.INFO):
        prepared = prepare_config(
            source, verbose=True, target="canonical", runtime=None
        )

    long_strategy = _strategy_side(prepared, "long")
    long_risk = prepared["bot"]["long"]["risk"]
    assert long_strategy["volatility_ema_span_1h"] == pytest.approx(1787)
    assert long_strategy["volatility_ema_span_1m"] == pytest.approx(44.0)
    assert long_strategy["entry"]["threshold_volatility_1h_weight"] == pytest.approx(1.5)
    assert long_strategy["entry"]["threshold_volatility_1m_weight"] == pytest.approx(4.66)
    assert long_strategy["entry"]["threshold_we_weight"] == pytest.approx(3.578)
    assert long_risk["total_exposure_enforcer_threshold"] == pytest.approx(0.985)
    assert long_risk["total_exposure_entry_gate_enabled"] is True
    assert long_risk["total_exposure_enforcer_policy"] == "reduce_overweight"
    assert long_risk["we_excess_allowance_pct"] == pytest.approx(0.1)
    assert long_risk["position_exposure_enforcer_threshold"] == pytest.approx(0.99)


def test_prepare_config_normalizes_twel_policy_and_runtime_flattening():
    source = get_template_config()
    source["bot"]["long"]["risk"]["total_exposure_enforcer_policy"] = "REDUCE_PORTFOLIO"

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)
    compiled = compile_runtime_config(prepared)

    assert prepared["bot"]["long"]["risk"]["total_exposure_enforcer_policy"] == "reduce_portfolio"
    assert compiled["bot"]["long"]["risk_twel_enforcer_policy"] == "reduce_portfolio"
    assert compiled["bot"]["long"]["risk_twel_entry_gate_enabled"] is True


def test_prepare_config_rejects_invalid_twel_policy():
    source = get_template_config()
    source["bot"]["long"]["risk"]["total_exposure_enforcer_policy"] = "least_stuck"

    with pytest.raises(ValueError, match="bot.long.risk.total_exposure_enforcer_policy"):
        prepare_config(source, verbose=False, target="canonical", runtime=None)


def test_twel_policy_and_entry_gate_are_not_optimizer_bounds():
    bounds = get_optimize_bounds_defaults()

    for pside in ("long", "short"):
        risk_bounds = bounds[pside]["risk"]
        assert "total_exposure_entry_gate_enabled" not in risk_bounds
        assert "total_exposure_enforcer_policy" not in risk_bounds


def test_load_fake_live_hsl_config_keeps_disabled_sparse_side_loadable():
    prepared = load_prepared_config(
        "configs/fake_live_hsl_btc.hjson", verbose=False, target="live"
    )

    assert prepared["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 0.0
    assert _strategy_side(prepared, "short")["entry"]["double_down_factor"] == pytest.approx(0.5)


def test_prepare_config_silently_removes_disabled_entry_grid_inflation_flag(caplog):
    source = get_template_config()
    source["bot"]["long"]["entry_grid_inflation_enabled"] = False
    source["bot"]["short"]["entry_grid_inflation_enabled"] = False

    with caplog.at_level(logging.WARNING):
        prepared = prepare_config(
            source, verbose=False, target="canonical", runtime=None
        )

    assert "entry_grid_inflation_enabled" not in prepared["bot"]["long"]
    assert "entry_grid_inflation_enabled" not in prepared["bot"]["short"]
    assert not any(
        "entry_grid_inflation_enabled" in rec.message for rec in caplog.records
    )


def test_prepare_config_removes_entry_grid_inflation_flag_in_coin_overrides():
    source = get_template_config()
    source["coin_overrides"] = {
        "BTC": {"bot": {"long": {"entry_grid_inflation_enabled": "false"}}}
    }

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert (
        "entry_grid_inflation_enabled"
        not in prepared["coin_overrides"]["BTC"]["bot"]["long"]
    )


def test_prepare_config_warns_and_removes_coin_override_entry_grid_inflation(caplog):
    source = get_template_config()
    source["coin_overrides"] = {
        "BTC": {"bot": {"long": {"entry_grid_inflation_enabled": True}}}
    }

    with caplog.at_level(logging.WARNING):
        prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert any(
        "coin_overrides.BTC.bot.long.entry_grid_inflation_enabled" in rec.message
        and "has no effect; removing it" in rec.message
        for rec in caplog.records
    )


def test_prepare_config_normalizes_all_zero_long_forager_weights_to_ema_readiness_only():
    source = get_template_config()
    source["bot"]["long"]["forager"]["score_weights"] = {
        "volume": 0.0,
        "ema_readiness": 0.0,
        "volatility": 0.0,
    }
    source["bot"]["long"]["forager"]["volume_ema_span_1m"] = 0.0
    source["bot"]["long"]["forager"]["volatility_ema_span_1m"] = 0.0
    source["bot"]["long"]["forager"]["volume_drop_pct"] = 0.0

    prepared = prepare_config(source, verbose=False, target="canonical", runtime=None)

    assert prepared["bot"]["long"]["forager"]["score_weights"] == {
        "volume": 0.0,
        "ema_readiness": 1.0,
        "volatility": 0.0,
    }
    assert prepared["bot"]["long"]["forager"]["volume_ema_span_1m"] == 0.0
    assert prepared["bot"]["long"]["forager"]["volatility_ema_span_1m"] == 0.0
