"""V9 policy migration is explicit, offline, and preserves original inputs."""

from copy import deepcopy
import json
from pathlib import Path
import stat

import pytest
from released_v8_fixtures import released_v8_policy

from config import compile_runtime_config, prepare_config
from config.migrations.v9 import MigrationResult, migrate_v8_to_v9, require_v9_runtime_config
from config.risk_memory import (
    normalize_memory_ema_span_minutes,
    normalize_risk_bootstrap_max_lookback_minutes,
    risk_bootstrap_horizon_minutes,
)
from config.schema import get_template_config
from tools.migrate_config_v9 import main, migrate
from released_v8_fixtures import released_v8_policy


def current():
    return json.loads((Path(__file__).resolve().parents[1] /
                       "configs/examples/default_trailing_martingale_long.json").read_text())


def legacy():
    """Released-v8 policy; no evolving v9 defaults or layouts participate."""
    return released_v8_policy()


@pytest.mark.parametrize("version", ["v9.0.0", "v9.1.0"])
@pytest.mark.parametrize("field", ["lookback_ema_span_minutes", "max_realized_loss_pct", "hsl_signal_mode"])
def test_intermediate_v9_documents_require_regeneration_from_released_v8(version, field):
    source = current()
    source["config_version"] = version
    source["live"][field] = "coin" if field == "hsl_signal_mode" else 12.5
    before = deepcopy(source)
    with pytest.raises(ValueError, match="(?:final canonical|newer than supported)"):
        migrate(source, memory_span_minutes=123.5,
                acknowledge_realized_loss_gate_memory_change=True)
    assert source == before


@pytest.mark.parametrize("window", [-1.0, 0.0, 1 / 128, 43200.0])
@pytest.mark.parametrize("fraction", [0.0, 0.1])
def test_v8_gate_window_removal_requires_separate_ack_and_explicit_memory(window, fraction):
    source = legacy()
    source["live"]["max_realized_loss_pct"] = fraction
    source["live"]["realized_loss_gate_lookback_minutes"] = window
    before = deepcopy(source)
    with pytest.raises(ValueError, match="acknowledge-realized-loss-gate-memory-change"):
        migrate(source, memory_span_minutes=12.5)
    result = migrate(source, memory_span_minutes=12.5,
                     acknowledge_realized_loss_gate_memory_change=True)
    assert result.config["bot"]["global"]["loss_gate"] == {
        "max_drawdown_ratio": fraction, "memory_ema_span_minutes": 12.5}
    assert "realized_loss_gate_lookback_minutes" not in json.dumps(result.config)
    assert any(item["path"] == "live.realized_loss_gate_lookback_minutes" and item["old"] == window
               for item in result.changes)
    assert source == before


@pytest.mark.parametrize("location", ["root", "fixed", "scenario", "suite"])
def test_v8_gate_ack_audits_all_effective_contexts_including_dormant_suite(location):
    source = legacy()
    patch = {"live.max_realized_loss_pct": 0.1}
    if location == "root":
        source["live"]["max_realized_loss_pct"] = 0.1
    elif location == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = patch
    elif location == "scenario":
        source["backtest"]["scenarios"] = [{"label": "gate", "overrides": patch}]
    else:
        source["backtest"]["suite"] = {"enabled": False, "scenarios": [{"label": "gate", "overrides": patch}]}
    before = deepcopy(source)
    with pytest.raises(ValueError, match="acknowledge-realized-loss-gate-memory-change"):
        migrate(source, memory_span_minutes=12.5)
    result = migrate(source, memory_span_minutes=12.5,
                     acknowledge_realized_loss_gate_memory_change=True)
    assert result.config["config_version"] == "v9.0.0"
    assert source == before


def test_v8_conversion_preserves_authored_dates_fees_exchanges_and_scenarios():
    source = legacy()
    source["backtest"].update(start_date="2020-02-03", end_date="2021-04-05",
        maker_fee_override=0.000123, taker_fee_override=0.000456, exchanges=["binance", "bybit"],
        scenarios=[{"label": "authored", "start_date": "2020-03-04", "end_date": "2020-05-06",
                    "coins": ["BTC", "ETH"], "exchanges": ["bybit"],
                    "overrides": {"backtest.maker_fee_override": 0.000789}}])
    before = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    for key in ("start_date", "end_date", "maker_fee_override", "taker_fee_override", "exchanges", "scenarios"):
        assert result.config["backtest"][key] == before["backtest"][key]
    assert source == before
    assert migrate(result.config).config == result.config
    assert migrate(result.config).changes == []


@pytest.mark.parametrize("end_date", ["now", "today", "2020", "2020-02", "2020-02-03T04:05:06"])
def test_conversion_preserves_dynamic_and_shorthand_dates_after_validation(end_date):
    source = legacy()
    source["backtest"].update(start_date="2019", end_date=end_date,
        scenarios=[{"label": "dates", "start_date": "2020-01", "end_date": "now",
                    "overrides": {"backtest.end_date": "today"}}])
    source["optimize"]["fixed_runtime_overrides"] = {"backtest.end_date": "now"}
    before = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    assert result.config["backtest"]["start_date"] == "2019"
    assert result.config["backtest"]["end_date"] == end_date
    assert result.config["backtest"]["scenarios"] == before["backtest"]["scenarios"]
    assert result.config["optimize"]["fixed_runtime_overrides"] == before["optimize"]["fixed_runtime_overrides"]
    assert migrate(result.config).config == result.config
    assert source == before


@pytest.mark.parametrize("source_shape", ["file", "side_file", "list"])
def test_conversion_preserves_reloadable_coin_sources_and_idempotence(tmp_path, source_shape):
    approved, ignored = tmp_path / "approved.json", tmp_path / "ignored.json"
    approved.write_text(json.dumps(["BTC", "ETH"]))
    ignored.write_text(json.dumps(["ETH"]))
    source = legacy()
    values = ({"approved_coins": str(approved), "ignored_coins": str(ignored)} if source_shape == "file" else
              {"approved_coins": {"long": str(approved), "short": []},
               "ignored_coins": {"long": str(ignored), "short": []}} if source_shape == "side_file" else
              {"approved_coins": ["BTC", "ETH"], "ignored_coins": ["ETH"]})
    source["live"].update(values)
    source["backtest"]["scenarios"] = [{"label": "reload", "overrides": {
        "live.approved_coins": str(approved), "live.ignored_coins": str(ignored)}}]
    before = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    for key, value in values.items():
        for side in ("long", "short"):
            assert result.config["bot"][side]["universe"][key] == (value.get(side, []) if isinstance(value, dict) else value)
    patch = result.config["backtest"]["scenarios"][0]["overrides"]
    for side in ("long", "short"):
        assert patch[f"bot.{side}.universe.approved_coins"] == str(approved)
        assert patch[f"bot.{side}.universe.ignored_coins"] == str(ignored)
    approved.write_text(json.dumps(["SOL"]))
    assert migrate(result.config).config == result.config
    assert prepare_config(result.config, verbose=False)["bot"]["long"]["universe"]["approved_coins"] == (["SOL"] if source_shape != "list" else ["BTC"])
    assert source == before


@pytest.mark.parametrize("location", ["root", "scenario"])
@pytest.mark.parametrize("inline_shadow", [False, True])
@pytest.mark.parametrize("document", [
    {"bot": {"long": {"hsl": {"red_thresold": 0.1}}}},
    {"live": {"levrage": 3}},
])
def test_referenced_v8_patch_rejects_unknown_fields_before_extraction(
    tmp_path, location, inline_shadow, document
):
    source = legacy()
    file = tmp_path / "coin.json"
    file.write_text(json.dumps(document))
    before_file = file.read_bytes()
    patch = {"override_config_path": "coin.json"}
    if inline_shadow:
        patch.update(bot={"long": {"hsl": {"red_threshold": 0.2}}}, live={"leverage": 4})
    coin_patch = {"BTC": patch}
    if location == "root":
        source["coin_overrides"] = coin_patch
    else:
        source["backtest"]["scenarios"] = [{"label": "typo", "overrides": {"coin_overrides": coin_patch}}]
    before = deepcopy(source)
    with pytest.raises(ValueError, match="(?:red_thresold|levrage).*unknown configuration field"):
        migrate(source, memory_span_minutes=123.5, base_config_path=str(tmp_path / "base.json"))
    assert source == before
    assert file.read_bytes() == before_file


@pytest.mark.parametrize("location", ["file", "inline_with_file"])
@pytest.mark.parametrize("flat,group,leaf,old,new", [
    ("unstuck_enabled", "unstuck", "enabled", False, True),
    ("risk_we_excess_allowance_pct", "risk", "we_excess_allowance_pct", 0.1, 0.2),
    ("hsl_red_threshold", "hsl", "red_threshold", 0.1, 0.2),
])
@pytest.mark.parametrize("flat_first", [False, True])
def test_v8_sparse_shared_alias_conflicts_fail_before_file_inline_merge(
    tmp_path, location, flat, group, leaf, old, new, flat_first
):
    entries = [(flat, old), (group, {leaf: new})]
    side = dict(entries if flat_first else reversed(entries))
    document = {"bot": {"long": side}}
    file = tmp_path / "coin.json"
    file.write_text(json.dumps(document if location == "file" else {}))
    patch = {"override_config_path": "coin.json"}
    if location == "inline_with_file":
        patch.update(document)
    source = legacy()
    source["coin_overrides"] = {"BTC": patch}
    before, file_before = deepcopy(source), file.read_bytes()
    with pytest.raises(ValueError, match="conflicts"):
        migrate(source, memory_span_minutes=123.5, base_config_path=str(tmp_path / "base.json"))
    assert source == before
    assert file.read_bytes() == file_before


@pytest.mark.parametrize("inline_value", [False, True])
def test_v8_equal_shared_aliases_keep_file_then_inline_precedence(tmp_path, inline_value):
    file = tmp_path / "coin.json"
    file.write_text(json.dumps({"bot": {"long": {
        "unstuck_enabled": False, "unstuck": {"enabled": False},
    }}}))
    source = legacy()
    source["coin_overrides"] = {"BTC": {
        "override_config_path": "coin.json",
        "bot": {"long": {"unstuck_enabled": inline_value, "unstuck": {"enabled": inline_value}}},
    }}
    result = migrate(source, memory_span_minutes=123.5, base_config_path=str(tmp_path / "base.json"))
    assert result.config["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["enabled"] is inline_value


@pytest.mark.parametrize("section,old,new,old_value,new_value", [
    ("backtest", "panic_market_slippage_pct", "market_order_slippage_pct", 0.02, 0.001),
    ("live", "minimum_market_age_days", "minimum_coin_age_days", 15.0, 20.0),
    ("live", "ohlcvs_1m_update_after_minutes", "inactive_coin_candle_ttl_minutes", 15.0, 20.0),
    ("backtest", "exchange", "exchanges", "binance", ["bybit"]),
])
def test_v8_released_section_aliases_reject_conflicting_authored_values(
    section, old, new, old_value, new_value
):
    source = legacy()
    source[section].update({old: old_value, new: new_value})
    before = deepcopy(source)
    with pytest.raises(ValueError, match="conflicts"):
        migrate(source, memory_span_minutes=123.5)
    assert source == before


def test_v8_cross_section_logging_alias_rejects_conflicting_values():
    source = legacy()
    source["live"]["memory_snapshot_interval_minutes"] = 5.0
    source["logging"]["memory_snapshot_interval_minutes"] = 10.0
    with pytest.raises(ValueError, match="conflicts"):
        migrate(source, memory_span_minutes=123.5)


@pytest.mark.parametrize("section,old,new,value", [
    ("backtest", "panic_market_slippage_pct", "market_order_slippage_pct", 0.02),
    ("live", "minimum_market_age_days", "minimum_coin_age_days", 20.0),
    ("live", "ohlcvs_1m_update_after_minutes", "inactive_coin_candle_ttl_minutes", 20.0),
])
def test_v8_equal_released_section_aliases_preserve_authored_value(section, old, new, value):
    source = legacy()
    source[section].update({old: value, new: value})
    result = migrate(source, memory_span_minutes=123.5)
    if new == "minimum_coin_age_days":
        assert result.config["bot"]["global"]["universe"][new] == value
    elif new == "inactive_coin_candle_ttl_minutes":
        assert result.config["live"]["candles"][new] == value
    else:
        assert result.config[section]["market_order_slippage_fraction" if new == "market_order_slippage_pct" else new] == value
    assert old not in result.config[section]


def test_v8_equal_exchange_and_cross_section_logging_aliases_are_supported():
    source = legacy()
    source["backtest"].update(exchange="binance", exchanges=["binance"])
    source["live"]["memory_snapshot_interval_minutes"] = 5.0
    source["logging"]["memory_snapshot_interval_minutes"] = 5.0
    result = migrate(source, memory_span_minutes=123.5)
    assert result.config["backtest"]["exchanges"] == ["binance"]
    assert result.config["logging"]["memory_snapshot_interval_minutes"] == 5.0
    assert "exchange" not in result.config["backtest"]
    assert "memory_snapshot_interval_minutes" not in result.config["live"]


def test_v8_multiple_released_aliases_cannot_choose_by_insertion_order():
    source = legacy()
    source["bot"]["long"].update({
        "filter_noisiness_rolling_window": 15.0, "filter_log_range_ema_span": 20.0,
    })
    with pytest.raises(ValueError, match="conflicts"):
        migrate(source, memory_span_minutes=123.5)


@pytest.mark.parametrize("location", ["root", "nested", "inline", "file", "scenario", "fixed"])
def test_v8_unknown_underscore_fields_are_not_silently_removed(tmp_path, location):
    source = legacy()
    if location == "root":
        source["_typo"] = 1
    elif location == "nested":
        source["bot"]["long"]["risk"]["_raw"] = {"max_positions": 1}
    elif location == "inline":
        source["coin_overrides"] = {"BTC": {"bot": {"long": {"unstuck": {"_enabled": True}}}}}
    elif location == "file":
        file = tmp_path / "coin.json"
        file.write_text(json.dumps({"bot": {"long": {"hsl": {"_red_threshold": 0.1}}}}))
        source["coin_overrides"] = {"BTC": {"override_config_path": "coin.json"}}
    else:
        patch = {"bot.long.unstuck._enabled": True}
        if location == "scenario":
            source["backtest"]["scenarios"] = [{"label": "invalid", "overrides": patch}]
        else:
            source["optimize"]["fixed_runtime_overrides"] = patch
    before = deepcopy(source)
    with pytest.raises(ValueError, match="(?:unknown configuration field|not overridable)"):
        migrate(source, memory_span_minutes=123.5, base_config_path=str(tmp_path / "base.json"))
    assert source == before


def test_v8_known_root_metadata_and_result_wrapper_remain_non_authoritative():
    source = legacy()
    expected = migrate(source, memory_span_minutes=123.5).config
    for key in ("_raw", "_raw_effective", "_transform_log", "_coins_sources"):
        source[key] = {"bot": {"long": {"unstuck_enabled": False, "unstuck": {"enabled": True}}}}
    wrapped = {"config": source, "analysis": {"unrelated_result_value": 1.0}}
    before = deepcopy(wrapped)
    assert migrate(wrapped, memory_span_minutes=123.5).config == expected
    assert wrapped == before


@pytest.mark.parametrize("kind", ["cooldown", "ema"])
@pytest.mark.parametrize("location", ["fixed", "scenario", "suite"])
def test_v8_conflicting_patch_aliases_fail_before_structural_normalization(kind, location):
    source = legacy()
    patch = ({"bot.long.risk.entry_cooldown_minutes": 2.5,
              "long.entry_cooldown.base_duration_minutes": 7.5} if kind == "cooldown" else
             {"bot.long.strategy.trailing_martingale.ema_span_0": 12.5,
              "long.strategy.entry.ema_span_0": 25.0})
    if location == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = patch
    elif location == "scenario":
        source["backtest"]["scenarios"] = [{"label": "conflict", "overrides": patch}]
    else:
        source["backtest"]["suite"] = {"enabled": False, "scenarios": [{"label": "conflict", "overrides": patch}]}
    before = deepcopy(source)
    with pytest.raises(ValueError, match="conflicts with canonical path"):
        migrate(source, memory_span_minutes=123.5)
    assert source == before


def test_v8_active_gate_requires_both_memory_choice_and_semantic_ack():
    source = legacy()
    source["live"]["max_realized_loss_pct"] = 0.1
    with pytest.raises(ValueError, match="explicitly choose"):
        migrate_v8_to_v9(source, acknowledge_realized_loss_gate_memory_change=True)
    with pytest.raises(ValueError, match="acknowledge-realized-loss-gate-memory-change"):
        migrate_v8_to_v9(source, memory_span_minutes=12.5)
    output = migrate_v8_to_v9(source, memory_span_minutes=12.5,
                             acknowledge_realized_loss_gate_memory_change=True).config
    assert output["bot"]["global"]["unstuck"]["memory_ema_span_minutes"] == 12.5
    assert "realized_loss_gate_lookback_minutes" not in output["live"]


@pytest.mark.parametrize("location, entry_kind", [
    ("scenario", "cooldown"), ("suite", "ema"),
])
@pytest.mark.parametrize("restart, enabled", [(None, False), (None, True), ("never", False)])
def test_equal_released_entry_aliases_preserve_referenced_hsl_authority_audits(
    tmp_path, location, entry_kind, restart, enabled
):
    source = legacy()
    source["bot"]["long"]["hsl"]["enabled"] = False
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps({"bot": {"long": {"hsl": {
        "restart_after_red_policy": restart,
    }}}}))
    original_bytes = coin_file.read_bytes()
    source["coin_overrides"] = {"AAA": {"override_config_path": "coin.json"}}
    if entry_kind == "cooldown":
        overrides = {
            "bot": {"long": {"risk": {"entry_cooldown_minutes": 0.05}}},
            "long.entry_cooldown.base_duration_minutes": 0.05,
        }
        canonical = "bot.long.entry_cooldown.base_duration_minutes"
        expected = 0.05
    else:
        overrides = {
            "bot.long.strategy.trailing_martingale.ema_span_0": 91.5,
            "long.strategy.entry.ema_span_0": 91.5,
        }
        canonical = "bot.long.strategy.trailing_martingale.entry.ema_span_0_minutes"
        expected = 91.5
    overrides["bot.long.hsl.enabled"] = enabled
    scenarios = [{"label": "entry-alias-audit", "overrides": overrides}]
    if location == "suite":
        source["backtest"]["suite"] = {"enabled": True, "scenarios": scenarios}
    else:
        source["backtest"]["scenarios"] = scenarios
    before = deepcopy(source)
    kwargs = dict(memory_span_minutes=43200, base_config_path=str(tmp_path / "base.json"))
    if restart is None and enabled:
        with pytest.raises(ValueError, match="no v9 policy equivalent"):
            migrate(source, acknowledge_never_removal=True, **kwargs)
    else:
        if restart == "never":
            with pytest.raises(ValueError, match="acknowledge-never-removal"):
                migrate(source, **kwargs)
        output = migrate(source, acknowledge_never_removal=restart == "never", **kwargs).config
        assert output["backtest"]["scenarios"][0]["overrides"][canonical] == expected
        assert "restart_after_red_policy" not in json.dumps(output)
        assert "enabled" not in output["coin_overrides"]["AAA"].get(
            "bot", {}).get("long", {}).get("hsl", {})
    assert source == before
    assert coin_file.read_bytes() == original_bytes


@pytest.mark.parametrize("location", ["scenario", "optimizer"])
@pytest.mark.parametrize("reverse", [False, True])
def test_entry_audit_normalization_keeps_unrelated_canonical_duplicates_invalid(location, reverse):
    source = legacy()
    items = [("bot.long.hsl.enabled", False), ("long.hsl.enabled", True)]
    patch = dict(reversed(items) if reverse else items)
    if location == "scenario":
        # A supported entry collision must not turn unrelated HSL aliases into
        # an insertion-order choice or silently resolve contradictory facts.
        patch.update({"bot.long.risk.entry_cooldown_minutes": 9.0,
                      "long.entry_cooldown.base_duration_minutes": 0.05})
        source["backtest"]["scenarios"] = [{"label": "ambiguous", "overrides": patch}]
    else:
        source["optimize"]["fixed_runtime_overrides"] = patch
    before = deepcopy(source)
    with pytest.raises(ValueError, match="(?:duplicates canonical path|conflicts with canonical path) bot.long.hsl.enabled"):
        migrate(source, memory_span_minutes=43200)
    assert source == before


@pytest.mark.parametrize("value", [False, 0, -1, float("inf"), float("nan"), 1e30])
def test_risk_span_rejects_invalid_or_nondecaying_f32(value):
    with pytest.raises(ValueError):
        normalize_memory_ema_span_minutes(value)


def test_spans_and_bootstrap_are_independent_float_settings():
    cfg = get_template_config()
    prepared = prepare_config(cfg, verbose=False)
    assert prepared["config_version"] == "v9.0.0"
    assert prepared["bot"]["global"]["unstuck"]["memory_ema_span_minutes"] == 43200.0
    assert isinstance(prepared["bot"]["global"]["unstuck"]["memory_ema_span_minutes"], float)
    assert risk_bootstrap_horizon_minutes(prepared) == (129600.0, True)
    assert "realized_loss_gate_lookback_minutes" not in prepared["live"]
    with pytest.raises(ValueError):
        normalize_risk_bootstrap_max_lookback_minutes(129601.0)


@pytest.mark.parametrize("minutes", [-1.0, 0.0, 1 / 128, 43200.0, 1e308])
def test_runtime_rejects_retired_gate_window_even_when_gate_disabled(minutes):
    config = current()
    config["live"]["realized_loss_gate_lookback_minutes"] = minutes
    with pytest.raises(ValueError, match="realized_loss_gate_lookback_minutes.*removed"):
        prepare_config(config, verbose=False)


def test_realized_gate_cli_rejects_retired_duration(capsys):
    from argparse import ArgumentParser
    from config_utils import add_config_arguments

    parser = ArgumentParser()
    add_config_arguments(parser, get_template_config(), command="backtest")
    with pytest.raises(SystemExit) as error:
        parser.parse_args(["--realized-loss-gate-lookback-minutes", "1e308"])
    assert error.value.code == 2
    assert "realized-loss-gate-lookback-minutes" in capsys.readouterr().err


def test_realized_gate_scenario_rejects_retired_duration_before_consumption():
    from suite_runner import apply_scenario_overrides

    config = prepare_config(current(), verbose=False)
    with pytest.raises(ValueError, match="realized_loss_gate_lookback_minutes.*removed"):
        apply_scenario_overrides(config, {"live.realized_loss_gate_lookback_minutes": 1e308})


@pytest.mark.parametrize("location", ["root_days", "scenario_minutes", "optimizer_minutes"])
def test_migration_rejects_unrepresentable_realized_gate_duration(location):
    source = legacy()
    if location == "root_days":
        source["live"]["pnls_max_lookback_days"] = 1e12
    elif location == "scenario_minutes":
        source["backtest"]["scenarios"] = [{"label": "oversized-gate", "overrides": {
            "live.realized_loss_gate_lookback_minutes": 1e308,
        }}]
    else:
        source["optimize"]["fixed_runtime_overrides"] = {
            "live.realized_loss_gate_lookback_minutes": 1e308,
        }
    before = deepcopy(source)
    with pytest.raises(ValueError, match="(?:pnls_max_lookback_days|realized_loss_gate_lookback_minutes).*signed 64-bit"):
        migrate(source, memory_span_minutes=43200)
    assert source == before


def test_migration_cli_refuses_oversized_gate_before_writing(tmp_path, capsys):
    source = legacy()
    source["live"]["pnls_max_lookback_days"] = 1e12
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(source))
    original = path.read_bytes()
    with pytest.raises(SystemExit) as error:
        main([str(path), "--lookback-ema-span-minutes", "43200", "--in-place"])
    assert error.value.code == 2
    assert "pnls_max_lookback_days" in capsys.readouterr().err
    assert path.read_bytes() == original
    assert not path.with_name("legacy.pre-v9-backup.json").exists()


def test_migration_requires_a_memory_choice_and_does_not_mutate_input():
    source = legacy()
    before = deepcopy(source)
    with pytest.raises(ValueError, match="explicitly choose"):
        migrate_v8_to_v9(source)
    result = migrate_v8_to_v9(source, memory_span_minutes=1234.5)
    assert source == before
    assert result.config["bot"]["global"]["unstuck"]["memory_ema_span_minutes"] == 1234.5
    assert "realized_loss_gate_lookback_minutes" not in result.config["live"]
    assert "pnls_max_lookback_days" not in result.config["live"]
    assert "restart_after_red_policy" not in result.config["bot"]["long"]["hsl"]
    assert result.warnings and result.changes
    assert migrate_v8_to_v9(result.config).config == result.config
    assert migrate_v8_to_v9(result.config).changes == []
    with pytest.raises(ValueError, match="different memory span"):
        migrate_v8_to_v9(result.config, memory_span_minutes=9999)


@pytest.mark.parametrize("location", ["root", "coin", "scenario", "optimizer"])
def test_never_requires_acknowledgement_even_for_disabled_policy(location):
    source = legacy()
    if location == "root":
        source["bot"]["long"]["hsl"]["restart_after_red_policy"] = "never"
    elif location == "coin":
        source["coin_overrides"] = {"AAA": {"bot": {"long": {"hsl_restart_after_red_policy": "never"}}}}
    elif location == "scenario":
        source["backtest"]["scenarios"] = [{"label": "test", "overrides": {
            "bot.long.hsl.restart_after_red_policy": "never"}}]
    else:
        source["optimize"]["fixed_runtime_overrides"] = {"bot.long.hsl_restart_after_red_policy": "never"}
    with pytest.raises(ValueError, match="acknowledge-never-removal"):
        migrate_v8_to_v9(source, memory_span_minutes=43200)
    result = migrate_v8_to_v9(source, memory_span_minutes=43200, acknowledge_never_removal=True)
    require_v9_runtime_config(result.config)


@pytest.mark.parametrize("location", ["root", "flat_root", "coin", "inherited_coin", "scenario", "inherited_scenario", "optimizer"])
def test_disabled_v8_null_restart_is_removed_with_unambiguous_effective_context(location):
    source = legacy()
    source["bot"]["long"]["hsl"]["enabled"] = False
    if location == "root":
        source["bot"]["long"]["hsl"]["restart_after_red_policy"] = None
    elif location == "flat_root":
        source["bot"]["long"].pop("hsl")
        source["bot"]["long"].update(hsl_enabled=False, hsl_restart_after_red_policy=None)
    elif location in {"coin", "inherited_coin"}:
        policy = {"restart_after_red_policy": None}
        if location == "coin":
            policy["enabled"] = False
        source["coin_overrides"] = {"AAA": {"bot": {"long": {"hsl": policy}}}}
    else:
        patch = {"bot.long.hsl.restart_after_red_policy": None}
        if location != "inherited_scenario":
            patch["bot.long.hsl.enabled"] = False
        if location == "optimizer":
            source["optimize"]["fixed_runtime_overrides"] = patch
        else:
            source["backtest"]["scenarios"] = [{"label": "disabled", "overrides": patch}]
    before = deepcopy(source)
    result = migrate_v8_to_v9(source, memory_span_minutes=43200)
    assert source == before
    require_v9_runtime_config(result.config)
    assert any(row["old"] is None and "null restart" in row["reason"] for row in result.changes)
    # Full static validation must accept the supported legacy representation too.
    assert migrate(source, memory_span_minutes=43200).config["config_version"] == "v9.0.0"


@pytest.mark.parametrize("location", ["root", "coin", "scenario", "ambiguous", "conflicting"])
def test_enabled_or_ambiguous_null_restart_is_never_silently_repaired(location):
    source = legacy()
    if location == "root":
        source["bot"]["long"]["hsl"].update(enabled=True, restart_after_red_policy=None)
    elif location == "coin":
        source["coin_overrides"] = {"AAA": {"bot": {"long": {"hsl": {
            "enabled": True, "restart_after_red_policy": None}}}}}
    elif location == "scenario":
        source["backtest"]["scenarios"] = [{"label": "enabled", "overrides": {
            "bot.long.hsl.enabled": True, "bot.long.hsl.restart_after_red_policy": None}}]
    elif location == "conflicting":
        source["bot"]["long"]["hsl"].update(enabled=False, restart_after_red_policy=None)
        source["bot"]["long"]["hsl_enabled"] = True
    else:
        source["bot"]["long"]["hsl"].pop("enabled")
        source["bot"]["long"]["hsl"]["restart_after_red_policy"] = None
    error = "conflicts" if location == "conflicting" else "no v9 policy equivalent"
    with pytest.raises(ValueError, match=error):
        migrate_v8_to_v9(source, memory_span_minutes=43200)


@pytest.mark.parametrize("legacy_suite", [False, True])
@pytest.mark.parametrize("root_enabled, scenario_enabled, removable", [
    (False, None, True),
    (False, True, False),
    (True, False, True),
    (True, None, False),
])
def test_null_restart_resolves_effective_enablement_in_scenario_and_legacy_suite(
    legacy_suite, root_enabled, scenario_enabled, removable
):
    source = legacy()
    source["bot"]["long"]["hsl"]["enabled"] = root_enabled
    patch = {"bot.long.hsl.restart_after_red_policy": None}
    if scenario_enabled is not None:
        patch["bot.long.hsl.enabled"] = scenario_enabled
    scenario = {"label": "inherited-policy", "overrides": patch}
    if legacy_suite:
        source["backtest"].pop("scenarios", None)
        source["backtest"]["suite"] = {"enabled": True, "scenarios": [scenario]}
    else:
        source["backtest"]["scenarios"] = [scenario]
    before = deepcopy(source)
    if removable:
        result = migrate(source, memory_span_minutes=43200)
        require_v9_runtime_config(result.config)
        assert "restart_after_red_policy" not in json.dumps(result.config)
        assert len(result.config["backtest"]["scenarios"]) == 1
    else:
        with pytest.raises(ValueError, match="no v9 policy equivalent"):
            migrate(source, memory_span_minutes=43200)
    assert source == before


def test_distinct_scenario_window_requires_path_specific_choice():
    source = legacy()
    source["backtest"]["scenarios"] = [{"label": "test", "overrides": {
        "live.pnls_max_lookback_days": 2.5, "live.hsl_signal_mode": "unified"}}]
    path = "backtest.scenarios[0].overrides.live.pnls_max_lookback_days"
    with pytest.raises(ValueError, match="requires --memory-span"):
        migrate_v8_to_v9(source, memory_span_minutes=43200)
    result = migrate_v8_to_v9(source, memory_span_minutes=43200,
                             memory_span_choices={path: 1200.5})
    patch = result.config["backtest"]["scenarios"][0]["overrides"]
    assert patch["bot.global.unstuck.memory_ema_span_minutes"] == 1200.5
    assert "live.realized_loss_gate_lookback_minutes" not in patch
    assert patch["bot.global.hsl.scope"] == "portfolio"


def test_legacy_all_history_is_removed_without_infinite_risk_memory():
    source = legacy()
    source["live"]["pnls_max_lookback_days"] = "all"
    output = migrate_v8_to_v9(source, memory_span_minutes=43200).config
    assert "realized_loss_gate_lookback_minutes" not in output["live"]
    assert output["bot"]["global"]["unstuck"]["memory_ema_span_minutes"] == 43200.0


def test_old_bounds_are_not_converted_as_equivalent_policy():
    source = legacy()
    source["optimize"]["bounds"]["live_pnls_max_lookback_days"] = [1, 30]
    with pytest.raises(ValueError, match="requires explicit new"):
        migrate_v8_to_v9(source, memory_span_minutes=43200)


def test_retired_controls_are_reported_by_migration_and_rejected_at_runtime():
    source = legacy()
    source["bot"]["long"]["hsl"]["no_restart_drawdown_threshold"] = 0.5
    result = migrate_v8_to_v9(source, memory_span_minutes=43200)
    assert any(item["path"].endswith("no_restart_drawdown_threshold") for item in result.changes)
    assert "no_restart_drawdown_threshold" not in result.config["bot"]["long"]["hsl"]
    result.config["bot"]["long"]["hsl"]["no_restart_drawdown_threshold"] = 0.5
    with pytest.raises(ValueError, match="(?:removed in v9|retired)"):
        require_v9_runtime_config(result.config)


@pytest.mark.parametrize("version", [None, "v8.6.0", "v9.1.0", "v10.0.0", "nonsense"])
def test_runtime_rejects_non_v9_even_with_hsl_disabled(version):
    config = current()
    config["config_version"] = version
    with pytest.raises(ValueError, match="explicit migration"):
        prepare_config(config, verbose=False)


@pytest.mark.parametrize("patch", [
    {"live.pnls_max_lookback_days": 30},
    {"bot.long.hsl_restart_after_red_policy": "always"},
    {"live.hsl_signal_mode": "unified"},
])
def test_runtime_rejects_retired_scenario_paths(patch):
    config = current()
    config["backtest"]["scenarios"] = [{"label": "bad", "overrides": patch}]
    with pytest.raises(ValueError, match="(?:removed in v9|retired)"):
        prepare_config(config, verbose=False)


def test_cli_dry_run_default_output_and_atomic_backup(tmp_path):
    source = tmp_path / "config.json"
    source.write_text(json.dumps(legacy()))
    original = source.read_bytes()
    assert main([str(source), "--lookback-ema-span-minutes", "1234.5", "--dry-run", "--diff"]) == 0
    assert sorted(item.name for item in tmp_path.iterdir()) == ["config.json"]
    assert main([str(source), "--lookback-ema-span-minutes", "1234.5"]) == 0
    assert source.read_bytes() == original
    output = tmp_path / "config.v9.json"
    assert json.loads(output.read_text())["bot"]["global"]["unstuck"]["memory_ema_span_minutes"] == 1234.5
    with pytest.raises(SystemExit):
        main([str(source), "--lookback-ema-span-minutes", "1234.5"])
    assert source.read_bytes() == original
    assert main([str(source), "--lookback-ema-span-minutes", "1234.5", "--in-place"]) == 0
    assert (tmp_path / "config.pre-v9-backup.json").read_bytes() == original
    assert json.loads(source.read_text())["config_version"] == "v9.0.0"


def test_invalid_migration_never_replaces_input_or_creates_backup(tmp_path):
    source = tmp_path / "config.json"
    cfg = legacy()
    cfg["bot"]["long"]["hsl"]["restart_after_red_policy"] = "never"
    source.write_text(json.dumps(cfg))
    original = source.read_bytes()
    with pytest.raises(SystemExit):
        main([str(source), "--lookback-ema-span-minutes", "43200", "--in-place"])
    assert source.read_bytes() == original
    assert not (tmp_path / "config.pre-v9-backup.json").exists()


def test_move_output_materializes_file_with_inline_precedence(tmp_path):
    source_dir, output_dir = tmp_path / "source", tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    cfg = legacy()
    cfg["coin_overrides"] = {"AAA": {"override_config_path": "coin.json", "bot": {
        "long": {"hsl": {"red_threshold": 0.22}}}}}
    coin_file = source_dir / "coin.json"
    coin_file.write_text(json.dumps({"bot": {"long": {"hsl": {
        "restart_after_red_policy": "never", "red_threshold": 0.19}}}}))
    original = coin_file.read_bytes()
    with pytest.raises(ValueError, match="acknowledge-never-removal"):
        migrate(cfg, memory_span_minutes=43200, base_config_path=str(source_dir / "base.json"))
    result = migrate(cfg, memory_span_minutes=43200, acknowledge_never_removal=True,
                     base_config_path=str(source_dir / "base.json"))
    assert coin_file.read_bytes() == original
    assert "override_config_path" not in result.config["coin_overrides"]["AAA"]
    assert result.config["coin_overrides"]["AAA"]["bot"]["long"]["hsl"]["drawdown_ratio_threshold"] == 0.22
    coin_file.unlink()
    assert prepare_config(result.config, base_config_path=str(output_dir / "moved.json"), verbose=False)


@pytest.mark.parametrize("root_enabled, file_enabled, inline_enabled, removable", [
    (False, None, None, True),
    (True, None, False, True),
    (True, True, False, True),
    (False, False, None, True),
    (False, False, True, False),
    (False, True, None, False),
    (True, None, None, False),
])
def test_referenced_null_restart_uses_effective_enablement_without_freezing_inheritance(
    tmp_path, root_enabled, file_enabled, inline_enabled, removable
):
    source = legacy()
    source["bot"]["long"]["hsl"]["enabled"] = root_enabled
    policy = {"restart_after_red_policy": None}
    if file_enabled is not None:
        policy["enabled"] = file_enabled
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps({"bot": {"long": {"hsl": policy}}}))
    original_bytes = coin_file.read_bytes()
    inline = {"override_config_path": "coin.json"}
    if inline_enabled is not None:
        inline["bot"] = {"long": {"hsl": {"enabled": inline_enabled}}}
    source["coin_overrides"] = {"AAA": inline}
    before = deepcopy(source)
    kwargs = dict(memory_span_minutes=43200, base_config_path=str(tmp_path / "base.json"))
    if removable:
        result = migrate(source, **kwargs)
        require_v9_runtime_config(result.config)
        assert "restart_after_red_policy" not in json.dumps(result.config)
        patch = result.config["coin_overrides"]["AAA"]
        assert "override_config_path" not in patch
        if file_enabled is None and inline_enabled is None:
            assert "enabled" not in patch.get("bot", {}).get("long", {}).get("hsl", {})
    else:
        with pytest.raises(ValueError, match="no v9 policy equivalent"):
            migrate(source, **kwargs)
    assert source == before
    assert coin_file.read_bytes() == original_bytes


@pytest.mark.parametrize("location", ["root", "scenario", "suite"])
@pytest.mark.parametrize("file_flat, file_enabled, removable", [
    (False, True, True),
    (True, False, False),
])
def test_referenced_null_aliases_preserve_file_then_inline_enablement(
    tmp_path, location, file_flat, file_enabled, removable
):
    from config.overrides import parse_overrides
    from tools.migrate_config_v9 import _set_patch

    source = legacy()
    source["bot"]["long"]["hsl"]["enabled"] = True
    file_side = ({"hsl_enabled": file_enabled, "hsl_restart_after_red_policy": None}
                 if file_flat else {"hsl": {"enabled": file_enabled,
                                          "restart_after_red_policy": None}})
    inline_side = ({"hsl": {"enabled": not file_enabled}} if file_flat
                   else {"hsl_enabled": not file_enabled})
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps({"bot": {"long": file_side}}))
    original_bytes = coin_file.read_bytes()
    coin_patch = {"AAA": {"override_config_path": "coin.json",
                          "bot": {"long": inline_side}}}
    if location == "root":
        source["coin_overrides"] = coin_patch
    else:
        scenario = {"label": "file-alias", "overrides": {"coin_overrides": coin_patch}}
        if location == "suite":
            source["backtest"].pop("scenarios", None)
            source["backtest"]["suite"] = {"enabled": True, "scenarios": [scenario]}
        else:
            source["backtest"]["scenarios"] = [scenario]
    before = deepcopy(source)
    kwargs = dict(memory_span_minutes=43200, base_config_path=str(tmp_path / "base.json"))
    if removable:
        output = migrate(source, **kwargs).config
        assert "restart_after_red_policy" not in json.dumps(output)
        if location != "root":
            _set_patch(output, output["backtest"]["scenarios"][0]["overrides"], scenario=True)
        coin_file.unlink()
        output = parse_overrides(prepare_config(
            output, base_config_path=str(tmp_path / "moved.json"), verbose=False,
        ), verbose=False)
        assert output["coin_overrides"]["AAA"]["bot"]["long"]["hsl"]["enabled"] is False
    else:
        with pytest.raises(ValueError, match="no v9 policy equivalent"):
            migrate(source, **kwargs)
    assert source == before
    if coin_file.exists():
        assert coin_file.read_bytes() == original_bytes


def test_scenario_inline_null_uses_provenance_after_file_aliases_are_canonicalized(tmp_path):
    from config.overrides import parse_overrides
    from tools.migrate_config_v9 import _set_patch

    source = legacy()
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps({"bot": {"long": {"hsl": {
        "enabled": True, "restart_after_red_policy": "always",
    }}}}))
    original_bytes = coin_file.read_bytes()
    source["backtest"]["scenarios"] = [{"label": "inline-null", "overrides": {
        "coin_overrides": {"AAA": {"override_config_path": "coin.json", "bot": {
            "long": {"hsl_enabled": False, "hsl_restart_after_red_policy": None},
        }}},
    }}]
    before = deepcopy(source)
    output = migrate(source, memory_span_minutes=43200,
                     base_config_path=str(tmp_path / "base.json")).config
    assert "restart_after_red_policy" not in json.dumps(output)
    _set_patch(output, output["backtest"]["scenarios"][0]["overrides"], scenario=True)
    output = parse_overrides(prepare_config(output, verbose=False), verbose=False)
    assert output["coin_overrides"]["AAA"]["bot"]["long"]["hsl"]["enabled"] is False
    assert source == before
    assert coin_file.read_bytes() == original_bytes


@pytest.mark.parametrize("location", ["root", "scenario"])
def test_explicit_v8_converter_canonicalizes_inline_shared_coin_leaves(location):
    source = legacy()
    coins = {"AAA": {"bot": {"long": {
        "hsl_enabled": False, "unstuck_ema_span_0": 51.25,
    }}}}
    if location == "root":
        source["coin_overrides"] = coins
    else:
        source["backtest"]["scenarios"] = [{"label": "inline", "overrides": {
            "coin_overrides": coins,
        }}]
    before = deepcopy(source)
    result = migrate(source, memory_span_minutes=43200)
    output = result.config
    if location == "scenario":
        output = output["backtest"]["scenarios"][0]["overrides"]
    side = output["coin_overrides"]["AAA"]["bot"]["long"]
    assert side["hsl"]["enabled"] is False
    assert side["unstuck"]["ema_span_0_minutes"] == 51.25
    assert "hsl_enabled" not in side and "unstuck_ema_span_0" not in side
    assert any("unstuck_ema_span_0" in row["path"] or
               "unstuck_ema_span_0" in json.dumps(row["old"])
               for row in result.changes)
    assert source == before


@pytest.mark.parametrize("canonical_first", [False, True])
@pytest.mark.parametrize("inline_enabled", [None, False])
def test_referenced_null_rejects_same_layer_conflicts_even_when_inline_shadows_them(
    tmp_path, canonical_first, inline_enabled
):
    source = legacy()
    source["bot"]["long"]["hsl"]["enabled"] = False
    entries = [("hsl", {"enabled": True, "restart_after_red_policy": None}),
               ("hsl_enabled", False)]
    file_side = dict(entries if canonical_first else reversed(entries))
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps({"bot": {"long": file_side}}))
    original_bytes = coin_file.read_bytes()
    patch = {"override_config_path": "coin.json"}
    if inline_enabled is not None:
        patch["bot"] = {"long": {"hsl": {"enabled": inline_enabled}}}
    source["coin_overrides"] = {"AAA": patch}
    before = deepcopy(source)
    kwargs = dict(memory_span_minutes=43200, base_config_path=str(tmp_path / "base.json"))
    with pytest.raises(ValueError, match="conflicts"):
        migrate(source, **kwargs)
    assert source == before
    assert coin_file.read_bytes() == original_bytes


@pytest.mark.parametrize("inline_enabled", [0, None, "false"])
def test_referenced_null_does_not_treat_nonboolean_inline_enablement_as_disabled(
    tmp_path, inline_enabled
):
    source = legacy()
    source["bot"]["long"]["hsl"]["enabled"] = False
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps({"bot": {"long": {"hsl": {
        "enabled": False, "restart_after_red_policy": None,
    }}}}))
    source["coin_overrides"] = {"AAA": {"override_config_path": "coin.json", "bot": {
        "long": {"hsl_enabled": inline_enabled},
    }}}
    with pytest.raises(ValueError, match="no v9 policy equivalent"):
        migrate(source, memory_span_minutes=43200, base_config_path=str(tmp_path / "base.json"))


def test_referenced_null_cannot_use_lower_disabled_fact_over_ambiguous_inline(tmp_path):
    source = legacy()
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps({"bot": {"long": {"hsl": {
        "enabled": False, "restart_after_red_policy": None,
    }}}}))
    source["coin_overrides"] = {"AAA": {"override_config_path": "coin.json", "bot": {
        "long": {"hsl": {"enabled": True}, "hsl_enabled": False},
    }}}
    with pytest.raises(ValueError, match="no v9 policy equivalent"):
        migrate(source, memory_span_minutes=43200, base_config_path=str(tmp_path / "base.json"))


def test_referenced_null_without_any_enablement_remains_ambiguous(tmp_path):
    source = legacy()
    source["bot"]["long"]["hsl"].pop("enabled")
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps({"bot": {"long": {"hsl_restart_after_red_policy": None}}}))
    source["coin_overrides"] = {"AAA": {"override_config_path": "coin.json"}}
    with pytest.raises(ValueError, match="no v9 policy equivalent"):
        migrate(source, memory_span_minutes=43200, base_config_path=str(tmp_path / "base.json"))


@pytest.mark.parametrize("legacy_suite", [False, True])
def test_enablement_only_scenario_cannot_hide_inherited_referenced_null(tmp_path, legacy_suite):
    source = legacy()
    source["bot"]["long"]["hsl"]["enabled"] = False
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps({"bot": {"long": {"hsl_restart_after_red_policy": None}}}))
    original_bytes = coin_file.read_bytes()
    source["coin_overrides"] = {"AAA": {"override_config_path": "coin.json"}}
    scenario = {"label": "enable", "overrides": {"bot.long.hsl.enabled": True}}
    if legacy_suite:
        source["backtest"].pop("scenarios", None)
        source["backtest"]["suite"] = {"enabled": True, "scenarios": [scenario]}
    else:
        source["backtest"]["scenarios"] = [scenario]
    before = deepcopy(source)
    with pytest.raises(ValueError, match="no v9 policy equivalent"):
        migrate(source, memory_span_minutes=43200, base_config_path=str(tmp_path / "base.json"))
    assert source == before
    assert coin_file.read_bytes() == original_bytes


@pytest.mark.parametrize("fixed_enabled", [False, True])
def test_fixed_runtime_enablement_cannot_hide_inherited_referenced_null(tmp_path, fixed_enabled):
    source = legacy()
    source["bot"]["long"]["hsl"]["enabled"] = False
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps({"bot": {"long": {"hsl_restart_after_red_policy": None}}}))
    original_bytes = coin_file.read_bytes()
    source["coin_overrides"] = {"AAA": {"override_config_path": "coin.json"}}
    source["optimize"]["fixed_runtime_overrides"] = {"bot.long.hsl.enabled": fixed_enabled}
    before = deepcopy(source)
    kwargs = dict(memory_span_minutes=43200, base_config_path=str(tmp_path / "base.json"))
    if fixed_enabled:
        with pytest.raises(ValueError, match="no v9 policy equivalent"):
            migrate(source, **kwargs)
    else:
        output = migrate(source, **kwargs).config
        assert "enabled" not in output["coin_overrides"]["AAA"].get(
            "bot", {}).get("long", {}).get("hsl", {})
        assert not any(key.startswith("coin_overrides")
                       for key in output["optimize"]["fixed_runtime_overrides"])
    assert source == before
    assert coin_file.read_bytes() == original_bytes


@pytest.mark.parametrize("long_enabled, file_enabled, fixed, null_location, removable", [
    (True, None, {}, "file", False),
    (False, None, {}, "file", True),
    (True, False, {}, "file", True),
    (False, None, {"bot.long.hsl.enabled": True}, "file", False),
    (True, None, {"bot.long.hsl.enabled": False, "bot.short.hsl.enabled": True}, "file", True),
    (True, None, {}, "inline", False),
    (True, False, {}, "inline", True),
])
def test_optimizer_mirror_audits_referenced_short_null_after_fixed_overrides(
    tmp_path, long_enabled, file_enabled, fixed, null_location, removable
):
    source = legacy()
    source["bot"]["long"]["hsl"]["enabled"] = long_enabled
    source["bot"]["short"]["hsl"]["enabled"] = False
    source["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    source["optimize"]["fixed_runtime_overrides"] = fixed
    policy = {"restart_after_red_policy": None if null_location == "file" else "always"}
    if file_enabled is not None:
        policy["enabled"] = file_enabled
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps({"bot": {"short": {"hsl": policy}}}))
    original_bytes = coin_file.read_bytes()
    source["coin_overrides"] = {"AAA": {"override_config_path": "coin.json"}}
    if null_location == "inline":
        source["coin_overrides"]["AAA"]["bot"] = {
            "short": {"hsl_restart_after_red_policy": None},
        }
    before = deepcopy(source)
    kwargs = dict(memory_span_minutes=43200, base_config_path=str(tmp_path / "base.json"))
    if removable:
        output = migrate(source, **kwargs).config
        patch = output["coin_overrides"]["AAA"]
        if file_enabled is None:
            assert "enabled" not in patch.get("bot", {}).get("short", {}).get("hsl", {})
        else:
            assert patch["bot"]["short"]["hsl"]["enabled"] is False
        assert "restart_after_red_policy" not in json.dumps(output)
        assert output["bot"]["short"]["hsl"]["enabled"] is False
        assert output["optimize"]["fixed_runtime_overrides"] == fixed
    else:
        with pytest.raises(ValueError, match="no v9 policy equivalent"):
            migrate(source, **kwargs)
    assert source == before
    assert coin_file.read_bytes() == original_bytes


@pytest.mark.parametrize("scenario_disables_short", [False, True])
def test_optimizer_mirror_precedes_scenario_enablement_when_auditing_referenced_null(
    tmp_path, scenario_disables_short
):
    source = legacy()
    source["bot"]["long"]["hsl"]["enabled"] = True
    source["bot"]["short"]["hsl"]["enabled"] = False
    source["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps({"bot": {"short": {"hsl_restart_after_red_policy": None}}}))
    original_bytes = coin_file.read_bytes()
    overrides = {"coin_overrides": {"AAA": {"override_config_path": "coin.json"}}}
    if scenario_disables_short:
        overrides["bot.short.hsl.enabled"] = False
    source["backtest"]["scenarios"] = [{"label": "short-null", "overrides": overrides}]
    before = deepcopy(source)
    kwargs = dict(memory_span_minutes=43200, base_config_path=str(tmp_path / "base.json"))
    if scenario_disables_short:
        output = migrate(source, **kwargs).config
        patch = output["backtest"]["scenarios"][0]["overrides"]
        assert patch["bot.short.hsl.enabled"] is False
        assert "enabled" not in patch["coin_overrides"]["AAA"].get(
            "bot", {}).get("short", {}).get("hsl", {})
        assert "restart_after_red_policy" not in json.dumps(output)
    else:
        with pytest.raises(ValueError, match="no v9 policy equivalent"):
            migrate(source, **kwargs)
    assert source == before
    assert coin_file.read_bytes() == original_bytes


@pytest.mark.parametrize("fixed_resolves_long", [False, True])
def test_optimizer_mirror_does_not_resolve_authored_alias_conflicts(
    tmp_path, fixed_resolves_long
):
    source = legacy()
    source["bot"]["long"]["hsl"]["enabled"] = True
    source["bot"]["long"]["hsl_enabled"] = False
    source["bot"]["short"]["hsl"]["enabled"] = False
    source["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    if fixed_resolves_long:
        source["optimize"]["fixed_runtime_overrides"] = {"bot.long.hsl.enabled": False}
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps({"bot": {"short": {"hsl_restart_after_red_policy": None}}}))
    source["coin_overrides"] = {"AAA": {"override_config_path": "coin.json"}}
    kwargs = dict(memory_span_minutes=43200, base_config_path=str(tmp_path / "base.json"))
    expected = "conflicts with" if fixed_resolves_long else "no v9 policy equivalent"
    with pytest.raises(ValueError, match=expected):
        migrate(source, **kwargs)


@pytest.mark.parametrize("never_location", ["file", "inline"])
def test_referenced_never_requires_ack_despite_later_disabled_policy(tmp_path, never_location):
    source = legacy()
    file_policy = {"enabled": True, "restart_after_red_policy": (
        "never" if never_location == "file" else "always")}
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps({"bot": {"long": {"hsl": file_policy}}}))
    original_bytes = coin_file.read_bytes()
    source["coin_overrides"] = {"AAA": {"override_config_path": "coin.json", "bot": {
        "long": {"hsl_enabled": False, "hsl_restart_after_red_policy": (
            None if never_location == "file" else "never")},
    }}}
    before = deepcopy(source)
    kwargs = dict(memory_span_minutes=43200, base_config_path=str(tmp_path / "base.json"))
    with pytest.raises(ValueError, match="acknowledge-never-removal"):
        migrate(source, **kwargs)
    output = migrate(source, acknowledge_never_removal=True, **kwargs).config
    assert output["coin_overrides"]["AAA"]["bot"]["long"]["hsl"]["enabled"] is False
    assert "restart_after_red_policy" not in json.dumps(output)
    assert source == before
    assert coin_file.read_bytes() == original_bytes


@pytest.mark.parametrize("policy", ["always", "never"])
@pytest.mark.parametrize("location", ["root", "scenario_nested", "scenario_flat", "scenario_map", "suite_nested"])
def test_legacy_coin_flag_files_are_materialized_with_flags_and_authorization(
    tmp_path, policy, location
):
    from tools.migrate_config_v9 import _set_patch

    cfg = legacy()
    cfg["optimize"]["fixed_runtime_overrides"] = {}
    coin_file = tmp_path / "coin.json"
    coin_file.write_text(json.dumps({"bot": {"long": {
        "hsl": {"enabled": True, "restart_after_red_policy": policy, "red_threshold": 0.19},
        "wallet_exposure_limit": 0.1,
    }}}))
    original = coin_file.read_bytes()
    flags = {"AAA": "-lc coin.json -lm gs -sm manual -lw 0.37 -sw 0.21 -lev 4",
             "BBB": "-lm tp -lev 3"}
    if location == "root":
        cfg["live"]["coin_flags"] = flags
    else:
        # A pre-existing root flag must not hide a higher-precedence scenario.
        cfg["live"]["coin_flags"] = {"AAA": "-lm panic -lw 0.8"}
        if location == "scenario_flat":
            patch = {"live.coin_flags.AAA": flags["AAA"], "live.coin_flags.BBB": flags["BBB"]}
        elif location == "scenario_map":
            patch = {"live.coin_flags": flags}
        else:
            patch = {"live": {"coin_flags": flags}}
        patch["live.minimum_coin_age_days"] = 7.0
        scenario = {"label": "flags", "overrides": patch}
        if location == "suite_nested":
            cfg["backtest"]["suite"] = {"enabled": True, "scenarios": [scenario]}
            cfg["backtest"].pop("scenarios", None)
        else:
            cfg["backtest"]["scenarios"] = [scenario]
    before = deepcopy(cfg)
    kwargs = dict(memory_span_minutes=43200, base_config_path=str(tmp_path / "base.json"))
    if policy == "never":
        with pytest.raises(ValueError, match="acknowledge-never-removal"):
            migrate(cfg, **kwargs)
    result = migrate(cfg, acknowledge_never_removal=True, **kwargs)
    assert cfg == before
    assert coin_file.read_bytes() == original
    output = result.config
    assert not output["live"].get("coin_flags")
    if location != "root":
        scenario_patch = output["backtest"]["scenarios"][0]["overrides"]
        assert not any("coin_flags" in key for key in scenario_patch)
        output = deepcopy(output)
        _set_patch(output, scenario_patch, scenario=True)
        assert output["bot"]["global"]["universe"]["minimum_coin_age_days"] == 7.0
    patch = output["coin_overrides"]["AAA"]
    assert "override_config_path" not in patch
    assert patch["live"]["modes"]["long"] == "graceful_stop"
    assert patch["live"]["modes"]["short"] == "manual"
    assert patch["live"]["account"]["leverage"] == 4.0
    assert patch["bot"]["long"]["wallet_exposure_limit"] == 0.37
    assert patch["bot"]["short"]["wallet_exposure_limit"] == 0.21
    assert patch["bot"]["long"]["hsl"]["enabled"]
    assert patch["bot"]["long"]["hsl"]["drawdown_ratio_threshold"] == 0.19
    assert "restart_after_red_policy" not in patch["bot"]["long"]["hsl"]
    assert output["coin_overrides"]["BBB"]["live"] == {
        "modes": {"long": "tp_only"},
        "account": {"leverage": 3.0},
    }
    coin_file.unlink()
    assert prepare_config(output, base_config_path=str(tmp_path / "moved" / "base.json"), verbose=False)


def test_explicit_coin_overrides_keep_precedence_but_ignored_flag_file_never_requires_ack(tmp_path):
    cfg = legacy()
    cfg["coin_overrides"] = {"AAA": {"live": {"forced_mode_long": "normal"}}}
    cfg["live"]["coin_flags"] = {"AAA": "-lc ignored.json -lm panic", "BBB": "-lm manual"}
    coin_file = tmp_path / "ignored.json"
    coin_file.write_text(json.dumps({"bot": {"long": {"hsl": {
        "enabled": False, "restart_after_red_policy": "never",
    }}}}))
    kwargs = dict(memory_span_minutes=43200, base_config_path=str(tmp_path / "base.json"))
    with pytest.raises(ValueError, match="acknowledge-never-removal"):
        migrate(cfg, **kwargs)
    converted = migrate(cfg, acknowledge_never_removal=True, **kwargs).config
    assert not converted["live"].get("coin_flags")
    assert converted["coin_overrides"] == {
        "AAA": {"live": {"modes": {"long": "normal"}}}
    }
    coin_file.unlink()
    assert prepare_config(converted, verbose=False)


def _migrate_file_and_inline_cooldown(tmp_path, file_patch, inline_patch, location, *, acknowledge=False):
    from tools.migrate_config_v9 import _set_patch

    cfg = legacy()
    file = tmp_path / "coin.json"
    file.write_text(json.dumps({"bot": {"long": file_patch}}))
    original = file.read_bytes()
    patch = {"AAA": {"override_config_path": "coin.json", "bot": {"long": inline_patch}}}
    if location == "root":
        cfg["coin_overrides"] = patch
    else:
        cfg["backtest"]["scenarios"] = [{"label": "cooldown", "overrides": {"coin_overrides": patch}}]
    before = deepcopy(cfg)
    result = migrate(cfg, memory_span_minutes=43200.0, acknowledge_never_removal=acknowledge,
                     base_config_path=str(tmp_path / "base.json")).config
    assert cfg == before
    assert file.read_bytes() == original
    file.unlink()
    if location == "scenario":
        _set_patch(result, result["backtest"]["scenarios"][0]["overrides"], scenario=True)
    # A relocated result must be self-contained, including its scenario patch.
    result = prepare_config(result, base_config_path=str(tmp_path / "moved.json"), verbose=False)
    return result["coin_overrides"]["AAA"]["bot"]["long"]


@pytest.mark.parametrize("location", ["root", "scenario"])
@pytest.mark.parametrize("file_old,inline_old", [(False, False), (False, True), (True, False), (True, True)])
def test_file_inline_cooldown_precedence_survives_all_legacy_canonical_pairs(
    tmp_path, location, file_old, inline_old
):
    def patch(old, value):
        return ({"risk": {"entry_cooldown_minutes": value}} if old
                else {"entry_cooldown": {"base_duration_minutes": value}})

    actual = _migrate_file_and_inline_cooldown(
        tmp_path, patch(file_old, 2.0), patch(inline_old, 7.0), location,
    )
    assert actual["entry_cooldown"]["base_duration_minutes"] == 7.0
    assert "entry_cooldown_minutes" not in actual.get("risk", {})


@pytest.mark.parametrize("location", ["root", "scenario"])
@pytest.mark.parametrize("canonical_first", [False, True])
def test_within_inline_cooldown_conflicting_aliases_are_rejected_in_both_orders(
    tmp_path, location, canonical_first
):
    entries = [("risk", {"entry_cooldown_minutes": 7.0}),
               ("entry_cooldown", {"base_duration_minutes": 9.0})]
    inline = dict(reversed(entries) if canonical_first else entries)
    with pytest.raises(ValueError, match="conflicts with"):
        _migrate_file_and_inline_cooldown(
            tmp_path, {"entry_cooldown": {"base_duration_minutes": 2.0}}, inline, location,
        )


def test_inline_cooldown_canonicalization_keeps_removed_hsl_policy_authorization(tmp_path):
    file_patch = {"entry_cooldown": {"base_duration_minutes": 2.0}}
    inline = {"risk": {"entry_cooldown_minutes": 7.0},
              "hsl_enabled": False, "hsl_restart_after_red_policy": "never"}
    with pytest.raises(ValueError, match="acknowledge-never-removal"):
        _migrate_file_and_inline_cooldown(tmp_path, file_patch, inline, "root")
    actual = _migrate_file_and_inline_cooldown(tmp_path, file_patch, inline, "root", acknowledge=True)
    assert actual["entry_cooldown"]["base_duration_minutes"] == 7.0
    assert actual["hsl"]["enabled"] is False
    assert "hsl_restart_after_red_policy" not in actual
    assert "restart_after_red_policy" not in actual["hsl"]


def _synthetic_migration(monkeypatch, config=None):
    import tools.migrate_config_v9 as tool

    monkeypatch.setattr(tool, "migrate", lambda *args, **kwargs:
                        MigrationResult(config or {"config_version": "v9.0.0"}, [], ()))


def test_atomic_in_place_follows_symlink_and_preserves_mode(tmp_path, monkeypatch):
    _synthetic_migration(monkeypatch)
    target, source = tmp_path / "target.json", tmp_path / "link.json"
    target.write_text('{"config_version":"v8.6.0"}')
    target.chmod(0o640)
    source.symlink_to(target)
    original = target.read_bytes()
    assert main([str(source), "--in-place", "--lookback-ema-span-minutes", "43200"]) == 0
    assert source.is_symlink()
    assert stat.S_IMODE(target.stat().st_mode) == 0o640
    assert (tmp_path / "target.pre-v9-backup.json").read_bytes() == original


def test_atomic_replace_failure_preserves_source_and_cleans_temp(tmp_path, monkeypatch):
    import tools.migrate_config_v9 as tool

    _synthetic_migration(monkeypatch)
    source = tmp_path / "input.json"
    source.write_text('{"config_version":"v8.6.0"}')
    before = source.read_bytes()
    def denied(*args):
        raise PermissionError("synthetic replacement denied")
    monkeypatch.setattr(tool.os, "replace", denied)
    with pytest.raises(SystemExit):
        main([str(source), "--in-place", "--lookback-ema-span-minutes", "43200"])
    assert source.read_bytes() == before
    assert (tmp_path / "input.pre-v9-backup.json").read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


def test_output_race_never_overwrites_other_writer(tmp_path, monkeypatch):
    import tools.migrate_config_v9 as tool

    _synthetic_migration(monkeypatch)
    source, output = tmp_path / "input.json", tmp_path / "output.json"
    source.write_text('{"config_version":"v8.6.0"}')
    original_link = tool.os.link
    def race(temporary, target):
        Path(target).write_text("other writer")
        original_link(temporary, target)
    monkeypatch.setattr(tool.os, "link", race)
    with pytest.raises(SystemExit):
        main([str(source), str(output), "--lookback-ema-span-minutes", "43200"])
    assert output.read_text() == "other writer"
    assert not list(tmp_path.glob(".*.tmp"))


def test_report_collision_fails_before_output_or_backup(tmp_path, monkeypatch):
    _synthetic_migration(monkeypatch)
    source, report = tmp_path / "input.json", tmp_path / "report.json"
    source.write_text('{"config_version":"v8.6.0"}')
    report.write_text("preserve")
    before = source.read_bytes()
    with pytest.raises(SystemExit):
        main([str(source), "--in-place", "--report", str(report),
              "--lookback-ema-span-minutes", "43200"])
    assert source.read_bytes() == before
    assert report.read_text() == "preserve"
    assert not (tmp_path / "input.pre-v9-backup.json").exists()


def test_nonfinite_output_fails_before_any_write(tmp_path, monkeypatch):
    _synthetic_migration(monkeypatch, {"value": float("nan")})
    source = tmp_path / "input.json"
    source.write_text('{"config_version":"v8.6.0"}')
    before = source.read_bytes()
    with pytest.raises(SystemExit):
        main([str(source), "--in-place", "--lookback-ema-span-minutes", "43200"])
    assert source.read_bytes() == before
    assert list(tmp_path.iterdir()) == [source]


def test_nonfinite_change_report_fails_before_any_write(tmp_path, monkeypatch):
    import tools.migrate_config_v9 as tool

    monkeypatch.setattr(tool, "migrate", lambda *args, **kwargs: MigrationResult(
        {"config_version": "v9.0.0"}, [{"old": float("inf"), "new": None}], ()))
    source = tmp_path / "input.json"
    source.write_text('{"config_version":"v8.6.0"}')
    before = source.read_bytes()
    with pytest.raises(SystemExit):
        main([str(source), "--in-place", "--lookback-ema-span-minutes", "43200"])
    assert source.read_bytes() == before
    assert list(tmp_path.iterdir()) == [source]


def test_live_rejects_cooldown_longer_than_reconstructible_history_cap():
    from config.hsl import require_live_runtime_support

    cfg = get_template_config()
    cfg["bot"]["global"]["unstuck"]["memory_ema_span_minutes"] = 1.0
    cfg["live"]["startup"]["risk_bootstrap_max_lookback_minutes"] = 100.0
    cfg["bot"]["long"]["hsl"].update(enabled=True, cooldown_after_red_minutes=200.0)
    prepared = prepare_config(cfg, verbose=False)
    # This is a live history capability boundary, not an offline-model limit.
    assert compile_runtime_config(prepared, runtime="backtest")
    with pytest.raises(ValueError, match="cannot reconstruct its full finite cooldown"):
        compile_runtime_config(prepared, runtime="live")
    cfg["live"]["startup"]["risk_bootstrap_max_lookback_minutes"] = 200.0
    assert require_live_runtime_support(cfg) == 200.0


def test_live_history_cap_checks_effective_coin_policy_and_disabled_policies():
    from config.hsl import require_live_runtime_support

    cfg = get_template_config()
    cfg["live"]["startup"]["risk_bootstrap_max_lookback_minutes"] = 100.0
    assert require_live_runtime_support(cfg) == 0.0
    cfg["coin_overrides"] = {"AAA": {"bot": {"long": {"hsl": {
        "enabled": True, "cooldown_after_red_minutes": 101.0}}}}}
    with pytest.raises(ValueError, match="coin_overrides.AAA"):
        require_live_runtime_support(cfg)


def test_global_slow_span_bounds_survive_loader_cleanup_and_optimizer_shape():
    from config_utils import clean_config
    from optimization.shape import build_optimization_shape

    cfg = get_template_config()
    cfg["bot"]["global"]["unstuck"]["memory_ema_span_minutes"] = 4321.5
    cfg["optimize"]["bounds"].pop("live", None)
    prepared = prepare_config(cfg, verbose=False)
    fixed_shape = build_optimization_shape(prepared)
    index = [key for key, _ in fixed_shape.key_paths].index("unstuck_memory_ema_span_minutes")
    assert fixed_shape.bounds[index].low == fixed_shape.bounds[index].high == 4321.5
    prepared["optimize"]["bounds"]["global"]["unstuck"] = {"memory_ema_span_minutes": [2000.5, 6000.5, 1.0]}
    exported = clean_config(prepared)
    shape = build_optimization_shape(prepare_config(exported, verbose=False))
    index = [key for key, _ in shape.key_paths].index("unstuck_memory_ema_span_minutes")
    assert shape.key_paths[index][1] == ("bot", "global", "unstuck", "memory_ema_span_minutes")
    assert shape.bounds[index].low == 2000.5
    assert shape.bounds[index].high == 6000.5
