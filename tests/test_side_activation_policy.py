"""Canonical activation is fixed policy, independent of numeric side capacity."""

from copy import deepcopy
from itertools import product

import pytest
from released_v8_fixtures import without_metric_selection

from config.activation import side_activation, side_enabled, validate_side_activation
from config.load import prepare_config
from config.schema import get_template_config, get_expanded_config_prototype


def prepare(source, **kwargs):
    return prepare_config({"config_version": "v9.0.0", **source}, **kwargs)


@pytest.mark.parametrize("invalid", [None, True, {}, {"long": True},
    {"long": 0, "short": False}, {"long": True, "short": "false"},
    {"long": True, "short": False, "extra": True}])
def test_activation_pair_is_strict(invalid):
    with pytest.raises(TypeError):
        validate_side_activation(invalid)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("invalid", [None, 0, 1, "true", "false", [], {}])
def test_side_activation_rejects_non_boolean_authored_values(side, invalid):
    source = {"bot": {side: {"enabled": invalid}}}
    with pytest.raises((TypeError, ValueError), match="enabled"):
        prepare(source, live_only=True, verbose=False)


@pytest.mark.parametrize("side", ["long", "short"])
def test_runtime_activation_does_not_fabricate_missing_permission(side):
    with pytest.raises(KeyError, match="enabled"):
        side_enabled({"bot": {side: {}}}, side)


@pytest.mark.parametrize("long,short", product((False, True), (False, True)))
def test_only_explicit_side_flags_grant_activation(long, short):
    source = {"bot": {"long": {"enabled": long}, "short": {"enabled": short}}}
    original = deepcopy(source)
    effective = prepare(source, live_only=True, verbose=False)
    assert side_activation(effective) == {"long": long, "short": short}
    assert "enabled" not in effective["bot"]["global"]
    assert source == original


@pytest.mark.parametrize("side", ["long", "short"])
def test_absent_and_empty_side_have_identical_activation_and_parameters(side):
    absent = prepare({"bot": {}}, live_only=True, verbose=False)
    empty = prepare({"bot": {side: {}}}, live_only=True, verbose=False)
    assert absent["bot"][side] == empty["bot"][side]
    assert side_activation(absent) == {"long": False, "short": False}


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("flag,capacity", product((False, True), (0.0, 1.0)))
def test_numeric_capacity_never_rewrites_explicit_activation(side, flag, capacity):
    source = {"bot": {side: {"enabled": flag,
        "risk": {"total_wallet_exposure_limit_ratio": capacity, "max_positions": int(capacity)}}}}
    effective = prepare(source, live_only=True, verbose=False)
    assert side_enabled(effective, side) is flag
    assert effective["bot"][side]["risk"]["total_wallet_exposure_limit_ratio"] == capacity
    assert effective["bot"][side]["risk"]["max_positions"] == int(capacity)


def test_activation_defaults_are_false_fixed_policy_and_not_coin_overridable():
    from config.field_registry import describe_field

    source = get_template_config()
    assert "enabled" not in source["bot"]["long"]
    assert "enabled" not in source["bot"]["short"]
    defaults = prepare(source, live_only=True, verbose=False)
    for owner in ("long", "short"):
        assert defaults["bot"][owner]["enabled"] is False
        record = describe_field(f"bot.{owner}.enabled")
        assert record["optimizer"]["eligible"] is False
        assert record["coin_override"] == "forbidden"
    source = {"coin_overrides": {"BTC": {"bot": {"long": {"enabled": True}}}}}
    with pytest.raises((TypeError, ValueError), match="enabled"):
        prepare(source, live_only=True, verbose=False)


@pytest.mark.parametrize("owner", ["long", "short"])
def test_prepared_direct_changed_value_mutation_survives_reprepare(owner):
    effective = prepare({"bot": {"long": {"enabled": True}, "short": {"enabled": True}}},
                        live_only=True, verbose=False)
    copied = deepcopy(effective)
    copied["bot"][owner]["enabled"] = False
    again = prepare(copied, live_only=True, verbose=False)
    assert side_activation(again) == {side: side != owner for side in ("long", "short")}
    assert "enabled" not in again["bot"]["global"]


@pytest.mark.parametrize("channel", ["fixed", "scenario", "cli"])
def test_equal_false_declaration_remains_authored_after_shared_parameter_change(channel):
    from argparse import Namespace
    from config.activation import activation_source_view
    from config_utils import update_config_with_args
    from optimization.warmup import _apply_config_overrides
    from suite_runner import apply_scenario_overrides

    config = prepare({"bot": {}}, live_only=True, verbose=False)
    patch = {"bot.short.enabled": False}
    if channel == "fixed":
        _apply_config_overrides(config, patch)
    elif channel == "scenario":
        apply_scenario_overrides(config, patch)
    else:
        update_config_with_args(config, Namespace(**patch))
    config["bot"]["global"]["risk"]["max_positions"] = 2
    again = prepare(config, live_only=True, verbose=False)
    assert side_activation(again) == {"long": False, "short": False}
    assert activation_source_view(again)["bot"]["short"]["enabled"] is False


def test_parent_replacement_removes_activation_permission_and_defaults_disabled():
    from suite_runner import apply_scenario_overrides
    from config.activation import activation_source_view

    config = prepare({"bot": {"long": {"enabled": True}, "short": {"enabled": True}}},
                     live_only=True, verbose=False)
    replacement = deepcopy(config["bot"]["short"])
    replacement.pop("enabled")
    apply_scenario_overrides(config, {"bot.short": replacement})
    assert side_activation(config) == {"long": True, "short": False}
    assert "enabled" not in activation_source_view(config)["bot"]["short"]
    assert side_activation(prepare(config, live_only=True, verbose=False)) == {"long": True, "short": False}


def test_authored_projection_json_retains_omission_but_effective_json_is_snapshot():
    import json
    from config.activation import activation_source_view

    config = prepare({"bot": {"short": {"enabled": False}}},
                     live_only=True, verbose=False)
    authored = json.loads(json.dumps(activation_source_view(config)))
    inspected = json.loads(json.dumps(config))
    assert "enabled" not in authored["bot"].get("long", {})
    assert inspected["bot"]["long"]["enabled"] is False
    authored["bot"]["long"] = {"enabled": True}
    inspected["bot"]["long"]["enabled"] = True
    assert side_activation(prepare(authored, live_only=True, verbose=False)) == {"long": True, "short": False}
    assert side_activation(prepare(inspected, live_only=True, verbose=False)) == {"long": True, "short": False}


def test_foreign_mapping_and_metadata_cannot_grant_activation_source_authority():
    from config.activation import ActivationSource, PreparedActivationBot, get_activation_source

    class ForeignBot(PreparedActivationBot):
        pass

    bot = ForeignBot({owner: {"enabled": False} for owner in ("long", "short")},
                     ActivationSource(True, True))
    config = {"bot": bot, "_raw": {"bot": {"global": {"enabled": True}}},
              "_activation_source": {"global_enabled": True}}
    assert get_activation_source(config) == ActivationSource(False, False)
    with pytest.raises(ValueError, match="_activation_source.*unknown"):
        prepare(config, live_only=True, verbose=False)
    config.pop("_activation_source")
    assert side_activation(prepare(config, live_only=True, verbose=False)) == {"long": False, "short": False}


def test_compact_retains_explicit_false_and_omits_default_activation():
    from config.authoring import compact_config

    effective = prepare({"bot": {"short": {"enabled": False}}},
                        live_only=True, verbose=False)
    result = compact_config(effective)
    assert "enabled" not in result["bot"].get("global", {})
    assert result["bot"]["short"]["enabled"] is False
    assert "enabled" not in result["bot"].get("long", {})


@pytest.mark.parametrize("base,fixed,scenarios,expected", [
    (False, {}, [], False),
    (False, {"bot.long.enabled": True}, [], True),
    (True, {"bot.long.enabled": False}, [], False),
    (False, {}, [{"overrides": {"bot.long.enabled": True}}], True),
    (True, {}, [{"overrides": {"bot.long.enabled": False}}], False),
    (False, {}, [{"overrides": {"bot.short.enabled": True}}], False),
])
def test_optimizer_activation_reachability_uses_fixed_and_scenario_authority(base, fixed, scenarios, expected):
    from config.activation import optimizer_side_may_activate

    config = prepare({"bot": {"long": {"enabled": base}}}, live_only=True, verbose=False)
    config["optimize"]["fixed_runtime_overrides"] = fixed
    config["backtest"]["suite_enabled"] = bool(scenarios)
    config["backtest"]["scenarios"] = scenarios
    original = deepcopy(config)
    assert optimizer_side_may_activate(config, "long") is expected
    assert config == original


@pytest.mark.parametrize("flag", [False, True])
def test_empty_suite_uses_actual_single_base_activation(flag):
    from config.activation import optimizer_side_may_activate
    from suite_runner import extract_suite_config

    config = prepare({"bot": {"long": {"enabled": flag}},
                      "backtest": {"suite_enabled": True, "scenarios": []}},
                     live_only=True, verbose=False)
    assert extract_suite_config(config, None)["enabled"] is False
    assert optimizer_side_may_activate(config, "long") is flag


@pytest.mark.parametrize("channel", ["base", "fixed", "scenario"])
@pytest.mark.parametrize("flag", [False, True])
def test_numeric_search_never_grants_activation_but_fixed_scenario_permission_retains_ranges(channel, flag):
    from optimization.config_adapter import extract_bounds_tuple_list_from_config, get_optimization_key_paths

    source = {"bot": {"long": {"enabled": flag if channel == "base" else False}},
              "optimize": {"bounds": {"long": {
                  "risk": {"max_positions": [1, 3], "total_wallet_exposure_limit_ratio": [0.5, 1.5]},
                  "unstuck": {"close_fraction": [0.01, 0.12]}}}}}
    if channel == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = {"bot.long.enabled": flag}
    elif channel == "scenario":
        source["backtest"] = {"suite_enabled": True, "scenarios": [{"label": "one", "overrides": {"bot.long.enabled": flag}}]}
    config = prepare(source, live_only=True, verbose=False)
    pairs = dict(zip((key for key, _ in get_optimization_key_paths(config)),
                     extract_bounds_tuple_list_from_config(config)))
    bound = pairs["bot.long.unstuck.close_fraction"]
    assert bound.low == 0.01
    assert bound.high == (0.12 if flag else 0.01)
    assert side_activation(config) == {"long": flag if channel == "base" else False, "short": False}


def test_evaluation_identity_applies_explicit_fixed_side_permission():
    from optimization.evaluation_contract import build_evaluation_contract

    source = {"bot": {}, "optimize": {"fixed_runtime_overrides": {"bot.long.enabled": True}}}
    effective = prepare(source, live_only=True, verbose=False)
    contract = build_evaluation_contract(effective)
    assert contract["bot"]["long"]["enabled"] is True
    assert "enabled" not in contract["bot"].get("global", {})
    assert side_activation(prepare({"bot": contract["bot"]}, live_only=True, verbose=False)) == {"long": True, "short": False}
    effective["bot"]["short"]["enabled"] = True
    from optimization.warmup import _apply_config_overrides
    _apply_config_overrides(effective, {"bot.short.enabled": False})
    assert build_evaluation_contract(effective)["bot"]["short"]["enabled"] is False


@pytest.mark.parametrize("command", ["live", "backtest", "optimize"])
def test_canonical_cli_activation_flags_and_equal_false_pin(command):
    import argparse
    from config_utils import add_arguments_recursively, update_config_with_args

    parser = argparse.ArgumentParser()
    add_arguments_recursively(parser, get_expanded_config_prototype(), command=command)
    args = parser.parse_args(["--bot.long.enabled", "y", "--bot.short.enabled", "n"])
    config = prepare({"bot": {}}, live_only=True, verbose=False)
    update_config_with_args(config, args)
    assert side_activation(prepare(config, live_only=True, verbose=False)) == {"long": True, "short": False}


def test_explanation_reports_side_default_and_explicit_false_pin():
    from config.authoring import explain_config

    source = {"config_version": "v9.0.0", "bot": {"short": {"enabled": False}}}
    long = explain_config(source, "bot.long.enabled")
    short = explain_config(source, "bot.short.enabled")
    assert long["value"] is False
    assert long["sources"][-1] == {"kind": "default_or_derived", "path": "bot.long.enabled"}
    assert short["value"] is False
    assert short["sources"][-1]["kind"] == "authored"


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_released_v8_materializes_true_both_and_preserves_zero_capacity_search(kind):
    from config.migrations.v9 import migrate_v8_to_v9
    from config.migrations.released_v8 import get_released_v8_defaults
    from optimization.config_adapter import extract_bounds_tuple_list_from_config, get_optimization_key_paths

    raw = without_metric_selection(get_released_v8_defaults(strategy_kind=kind))
    raw["bot"].pop("short")
    raw["optimize"]["bounds"]["short"]["risk"]["n_positions"] = [1, 3]
    raw["optimize"]["bounds"]["short"]["risk"]["total_wallet_exposure_limit"] = [0.0, 1.5]
    original = deepcopy(raw)
    converted = migrate_v8_to_v9(raw, best_effort=True)
    config = prepare_config(converted.config, live_only=True, verbose=False)
    assert side_activation(config) == {"long": True, "short": True}
    assert config["bot"]["short"]["risk"]["total_wallet_exposure_limit_ratio"] == 0.0
    bounds = dict(zip((key for key, _ in get_optimization_key_paths(config)),
                      extract_bounds_tuple_list_from_config(config)))
    assert bounds["bot.short.risk.total_wallet_exposure_limit_ratio"].high > 0
    for side in ("long", "short"):
        records = [item for item in converted.changes if item["path"] == f"bot.{side}.enabled"]
        assert len(records) == 1 and records[0]["new"] is True
        assert "reachability" in records[0]["reason"]
    assert raw == original


def test_released_v8_does_not_accept_unpublished_activation_fields():
    from released_v8_fixtures import released_v8_policy
    from config.migrations.v9 import migrate_v8_to_v9
    raw = released_v8_policy()
    raw["bot"]["long"]["enabled"] = False
    with pytest.raises(ValueError, match="not a released-v8 field"):
        migrate_v8_to_v9(raw, best_effort=True)


@pytest.mark.parametrize("invalid", [{"long": 1, "short": 1}, None, {"long": True}])
def test_cached_backtest_packet_validates_boolean_types_before_equality(invalid):
    from tools.gpu_parity import build_parser, fixture_inputs
    from backtest import prep_backtest_args

    args = build_parser().parse_args(["--fixture", "ema_anchor", "--sides", "both", "--bars", "64"])
    config, _, markets, _, _ = fixture_inputs(args)
    bot, strategy, exchange, params = prep_backtest_args(config, markets, args.exchange)
    cached = {**params, "side_activation": invalid}
    with pytest.raises(TypeError, match="side_activation"):
        prep_backtest_args(config, markets, args.exchange, exchange_params=exchange, backtest_params=cached)


@pytest.mark.parametrize("value", [True, False, None])
@pytest.mark.parametrize("surface", ["loader", "cleaner", "compact", "resolved", "activation_source"])
def test_global_activation_is_rejected_by_all_authoring_surfaces(value, surface):
    from config.activation import activation_source_view
    from config.authoring import compact_config, resolved_config
    from tools.clean_config import cleanup_config

    source = {"config_version": "v9.0.0", "live": {}, "bot": {"global": {"enabled": value},
              "long": {"enabled": True}}}
    original = deepcopy(source)
    functions = {
        "loader": lambda config: prepare_config(config, live_only=True, verbose=False),
        "cleaner": cleanup_config, "compact": compact_config, "resolved": resolved_config,
        "activation_source": activation_source_view,
    }
    with pytest.raises(ValueError, match=r"bot.global.enabled is unsupported.*bot.long.enabled.*bot.short.enabled"):
        functions[surface](source)
    assert source == original


@pytest.mark.parametrize("value", [True, False, None])
@pytest.mark.parametrize("parent", ["leaf", "global", "bot"])
@pytest.mark.parametrize("channel", ["fixed", "scenario", "authored", "activation"])
def test_global_activation_rejected_in_leaf_and_parent_patches(value, parent, channel):
    from config.activation import apply_activation_patch
    from config.bot_inheritance import apply_bot_config_patch
    from optimization.warmup import _apply_config_overrides
    from suite_runner import apply_scenario_overrides

    config = prepare({"bot": {"long": {"enabled": True}}}, live_only=True, verbose=False)
    patch = {"leaf": {"bot.global.enabled": value},
             "global": {"bot.global": {"enabled": value}},
             "bot": {"bot": {"global": {"enabled": value}}}}[parent]
    functions = {"fixed": _apply_config_overrides, "scenario": apply_scenario_overrides,
                 "authored": apply_bot_config_patch, "activation": apply_activation_patch}
    with pytest.raises(ValueError, match=r"bot.global.enabled is unsupported.*bot.long.enabled"):
        functions[channel](config, patch)


@pytest.mark.parametrize("value", [True, False, None])
@pytest.mark.parametrize("location", ["fixed", "scenario"])
def test_loader_rejects_global_activation_inside_inactive_declarations(value, location):
    source = {"config_version": "v9.0.0", "bot": {"long": {"enabled": True}}}
    patch = {"bot.global.enabled": value}
    if location == "fixed":
        source["optimize"] = {"fixed_runtime_overrides": patch}
    else:
        source["backtest"] = {"suite_enabled": False,
                              "scenarios": [{"label": "inactive", "overrides": patch}]}
    with pytest.raises(ValueError, match="bot.global.enabled is unsupported"):
        prepare_config(source, live_only=True, verbose=False)


@pytest.mark.parametrize("value", [True, False, None])
def test_global_activation_direct_edit_cannot_hide_in_retained_policy(value):
    from config.activation import activation_source_view
    from config.bot_inheritance import bot_policy_source_view
    from config_utils import clean_config
    config = prepare({"bot": {"long": {"enabled": True}}}, live_only=True, verbose=False)
    config["bot"]["global"]["enabled"] = value
    for export in (activation_source_view, bot_policy_source_view, clean_config):
        with pytest.raises(ValueError, match="bot.global.enabled is unsupported"):
            export(config)


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_nested_feature_switches_still_inherit_without_enabling_sides(kind):
    config = prepare({"bot": {"portfolio": {"strategy_kind": kind}, "global": {
        "hsl": {"enabled": True}, "unstuck": {"enabled": False},
        "risk": {"position_exposure_enforcer": {"enabled": True}}},
        "short": {"unstuck": {"enabled": True}}}}, live_only=True, verbose=False)
    assert side_activation(config) == {"long": False, "short": False}
    assert config["bot"]["long"]["hsl"]["enabled"] is True
    assert config["bot"]["short"]["hsl"]["enabled"] is True
    assert config["bot"]["long"]["unstuck"]["enabled"] is False
    assert config["bot"]["short"]["unstuck"]["enabled"] is True
    assert config["bot"]["long"]["risk"]["position_exposure_enforcer"]["enabled"] is True


@pytest.mark.parametrize("export_kind", ["clean", "compact", "resolved", "artifact", "effective", "candidate"])
@pytest.mark.parametrize("explicit", [False, True])
def test_source_and_result_exports_never_emit_global_activation(export_kind, explicit):
    from config.authoring import compact_config, resolved_config
    from config.bot_inheritance import bot_policy_source_view
    from config_utils import clean_config
    from session_artifacts import effective_setup_config

    source = {"bot": {"long": {"enabled": True}}}
    if explicit:
        source["bot"]["short"] = {"enabled": False}
    config = prepare(source, live_only=True, verbose=False)
    exports = {"clean": clean_config, "compact": compact_config, "resolved": resolved_config,
               "artifact": effective_setup_config, "effective": deepcopy,
               "candidate": lambda config: bot_policy_source_view(config, candidate=True)}
    result = exports[export_kind](config)
    assert "enabled" not in result["bot"].get("global", {})
    assert result["bot"]["long"]["enabled"] is True
    if export_kind == "artifact":
        assert result["bot"].get("short", {}).get("enabled", False) is False
    else:
        assert side_activation(prepare(result, live_only=True, verbose=False)) == {"long": True, "short": False}
    if export_kind in {"clean", "compact", "resolved", "candidate"}:
        assert ("enabled" in result["bot"].get("short", {})) is explicit


def test_editor_schema_and_registry_expose_activation_only_for_sides():
    from config.field_registry import get_field_registry
    from config.json_schema import generate_schema
    from jsonschema import Draft202012Validator, ValidationError

    registry = get_field_registry()
    assert registry.find(("bot", "global", "enabled")) is None
    assert registry.find(("bot", "long", "enabled")) is not None
    assert registry.find(("bot", "short", "enabled")) is not None
    validator = Draft202012Validator(generate_schema())
    validator.validate({"config_version": "v9.0.0", "bot": {"long": {"enabled": True}}})
    for value in (True, False, None):
        with pytest.raises(ValidationError):
            validator.validate({"config_version": "v9.0.0", "bot": {"global": {"enabled": value}}})


@pytest.mark.parametrize("parent", ["leaf", "side", "bot"])
def test_unprepared_activation_patch_rejects_explicit_null_at_any_depth(parent):
    from config.activation import apply_activation_patch, resolve_side_activation
    config = {}
    resolve_side_activation(config)
    patch = {"leaf": {"bot.long.enabled": None}, "side": {"bot.long": {"enabled": None}},
             "bot": {"bot": {"long": {"enabled": None}}}}[parent]
    with pytest.raises(TypeError, match="bot.long.enabled must be a boolean"):
        apply_activation_patch(config, patch)


@pytest.mark.parametrize("value", [True, False, None])
def test_explicit_activation_source_cannot_hide_unsupported_global_value(value):
    from config.activation import ActivationSource, resolve_side_activation
    with pytest.raises(ValueError, match="bot.global.enabled is unsupported"):
        resolve_side_activation({"bot": {"global": {"enabled": value}}}, ActivationSource(True, False))
