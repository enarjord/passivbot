"""Canonical activation is fixed policy, independent of numeric side capacity."""

from copy import deepcopy
from itertools import product

import pytest

from config.activation import side_activation, side_enabled, validate_side_activation
from config.load import prepare_config
from config.schema import get_template_config


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


@pytest.mark.parametrize("global_flag,pin", product((False, True), (None, False, True)))
def test_global_activation_is_inherited_unless_side_is_explicitly_pinned(global_flag, pin):
    source = {"bot": {"global": {"enabled": global_flag}}}
    if pin is not None:
        source["bot"]["short"] = {"enabled": pin}
    original = deepcopy(source)
    effective = prepare(source, live_only=True, verbose=False)
    assert side_activation(effective) == {"long": global_flag,
        "short": global_flag if pin is None else pin}
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
        "risk": {"total_wallet_exposure_limit": capacity, "max_positions": int(capacity)}}}}
    effective = prepare(source, live_only=True, verbose=False)
    assert side_enabled(effective, side) is flag
    assert effective["bot"][side]["risk"]["total_wallet_exposure_limit"] == capacity
    assert effective["bot"][side]["risk"]["max_positions"] == int(capacity)


def test_activation_defaults_are_false_fixed_policy_and_not_coin_overridable():
    from config.field_registry import describe_field

    defaults = get_template_config()
    for owner in ("global", "long", "short"):
        assert defaults["bot"][owner]["enabled"] is False
        record = describe_field(f"bot.{owner}.enabled")
        assert record["optimizer"]["eligible"] is False
        assert record["coin_override"] == "forbidden"
    source = {"coin_overrides": {"BTC": {"bot": {"long": {"enabled": True}}}}}
    with pytest.raises((TypeError, ValueError), match="enabled"):
        prepare(source, live_only=True, verbose=False)


@pytest.mark.parametrize("owner", ["global", "long", "short"])
def test_prepared_direct_changed_value_mutation_survives_reprepare(owner):
    effective = prepare({"bot": {"global": {"enabled": True}}}, live_only=True, verbose=False)
    copied = deepcopy(effective)
    copied["bot"][owner]["enabled"] = False
    again = prepare(copied, live_only=True, verbose=False)
    assert again["bot"][owner]["enabled"] is False
    if owner == "global":
        assert side_activation(again) == {"long": False, "short": False}
    else:
        assert again["bot"]["global"]["enabled"] is True


@pytest.mark.parametrize("channel", ["fixed", "scenario", "cli"])
def test_equal_false_declaration_remains_pin_after_later_global_change(channel):
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
    config["bot"]["global"]["enabled"] = True
    again = prepare(config, live_only=True, verbose=False)
    assert side_activation(again) == {"long": True, "short": False}
    assert activation_source_view(again)["bot"]["short"]["enabled"] is False


def test_parent_replacement_removes_activation_pin_and_inherits():
    from suite_runner import apply_scenario_overrides
    from config.activation import activation_source_view

    config = prepare({"bot": {"global": {"enabled": True}, "short": {"enabled": False}}},
                     live_only=True, verbose=False)
    replacement = deepcopy(config["bot"]["short"])
    replacement.pop("enabled")
    apply_scenario_overrides(config, {"bot.short": replacement})
    assert side_activation(config) == {"long": True, "short": True}
    assert "enabled" not in activation_source_view(config)["bot"]["short"]
    assert side_activation(prepare(config, live_only=True, verbose=False)) == {"long": True, "short": True}


def test_authored_projection_json_retains_omission_but_effective_json_is_snapshot():
    import json
    from config.activation import activation_source_view

    config = prepare({"bot": {"global": {"enabled": True}, "short": {"enabled": False}}},
                     live_only=True, verbose=False)
    authored = json.loads(json.dumps(activation_source_view(config)))
    inspected = json.loads(json.dumps(config))
    assert "enabled" not in authored["bot"]["long"]
    assert inspected["bot"]["long"]["enabled"] is True
    authored["bot"]["global"]["enabled"] = False
    inspected["bot"]["global"]["enabled"] = False
    assert side_activation(prepare(authored, live_only=True, verbose=False)) == {"long": False, "short": False}
    assert side_activation(prepare(inspected, live_only=True, verbose=False)) == {"long": True, "short": False}


def test_foreign_mapping_and_metadata_cannot_grant_activation_source_authority():
    from config.activation import ActivationSource, PreparedActivationBot, get_activation_source

    class ForeignBot(PreparedActivationBot):
        pass

    bot = ForeignBot({owner: {"enabled": False} for owner in ("global", "long", "short")},
                     ActivationSource(True, None, None))
    config = {"bot": bot, "_raw": {"bot": {"global": {"enabled": True}}},
              "_activation_source": {"global_enabled": True}}
    assert get_activation_source(config) == ActivationSource(False, False, False)
    with pytest.raises(ValueError, match="_activation_source.*unknown"):
        prepare(config, live_only=True, verbose=False)
    config.pop("_activation_source")
    assert side_activation(prepare(config, live_only=True, verbose=False)) == {"long": False, "short": False}


def test_compact_retains_explicit_false_and_omits_derived_inheritance():
    from config.authoring import compact_config

    effective = prepare({"bot": {"global": {"enabled": True}, "short": {"enabled": False}}},
                        live_only=True, verbose=False)
    result = compact_config(effective)
    assert result["bot"]["global"]["enabled"] is True
    assert result["bot"]["short"]["enabled"] is False
    assert "enabled" not in result["bot"]["long"]


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

    config = prepare({"bot": {"global": {"enabled": base}}}, live_only=True, verbose=False)
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

    config = prepare({"bot": {"global": {"enabled": flag}},
                      "backtest": {"suite_enabled": True, "scenarios": []}},
                     live_only=True, verbose=False)
    assert extract_suite_config(config, None)["enabled"] is False
    assert optimizer_side_may_activate(config, "long") is flag


@pytest.mark.parametrize("channel", ["base", "fixed", "scenario"])
@pytest.mark.parametrize("flag", [False, True])
def test_numeric_search_never_grants_activation_but_fixed_scenario_permission_retains_ranges(channel, flag):
    from optimization.config_adapter import extract_bounds_tuple_list_from_config, get_optimization_key_paths

    source = {"bot": {"global": {"enabled": flag if channel == "base" else False}},
              "optimize": {"bounds": {"long": {
                  "risk": {"max_positions": [1, 3], "total_wallet_exposure_limit": [0.5, 1.5]},
                  "unstuck": {"close_fraction": [0.01, 0.12]}}}}}
    if channel == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = {"bot.global.enabled": flag}
    elif channel == "scenario":
        source["backtest"] = {"suite_enabled": True, "scenarios": [{"label": "one", "overrides": {"bot.global.enabled": flag}}]}
    config = prepare(source, live_only=True, verbose=False)
    pairs = dict(zip((key for key, _ in get_optimization_key_paths(config)),
                     extract_bounds_tuple_list_from_config(config)))
    bound = pairs["long_unstuck_close_pct"]
    assert bound.low == 0.01
    assert bound.high == (0.12 if flag else 0.01)
    assert side_activation(config) == dict.fromkeys(("long", "short"), flag if channel == "base" else False)


def test_evaluation_identity_applies_global_fixed_permission_without_pinning_inherited_false():
    from optimization.evaluation_contract import build_evaluation_contract

    source = {"bot": {}, "optimize": {"fixed_runtime_overrides": {"bot.global.enabled": True}}}
    effective = prepare(source, live_only=True, verbose=False)
    contract = build_evaluation_contract(effective)
    assert {side: contract["bot"][side]["enabled"] for side in ("long", "short")} == {"long": True, "short": True}
    effective["bot"]["short"]["enabled"] = True
    from optimization.warmup import _apply_config_overrides
    _apply_config_overrides(effective, {"bot.short.enabled": False})
    assert build_evaluation_contract(effective)["bot"]["short"]["enabled"] is False


@pytest.mark.parametrize("command", ["live", "backtest", "optimize"])
def test_canonical_cli_activation_flags_and_equal_false_pin(command):
    import argparse
    from config_utils import add_arguments_recursively, update_config_with_args

    parser = argparse.ArgumentParser()
    add_arguments_recursively(parser, get_template_config(), command=command)
    args = parser.parse_args(["--bot.global.enabled", "y", "--bot.short.enabled", "n"])
    config = prepare({"bot": {}}, live_only=True, verbose=False)
    update_config_with_args(config, args)
    assert side_activation(prepare(config, live_only=True, verbose=False)) == {"long": True, "short": False}


def test_explanation_reports_global_inheritance_and_explicit_false_pin():
    from config.authoring import explain_config

    source = {"config_version": "v9.0.0", "bot": {"global": {"enabled": True}, "short": {"enabled": False}}}
    long = explain_config(source, "bot.long.enabled")
    short = explain_config(source, "bot.short.enabled")
    assert long["value"] is True
    assert long["sources"][-1] == {"kind": "inherited", "path": "bot.global.enabled", "value": True}
    assert short["value"] is False
    assert short["sources"][-1]["kind"] == "authored"


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_released_v8_materializes_true_both_and_preserves_zero_capacity_search(kind):
    from config.migrations.v9 import migrate_v8_to_v9
    from config.migrations.released_v8 import get_released_v8_defaults
    from optimization.config_adapter import extract_bounds_tuple_list_from_config, get_optimization_key_paths

    raw = get_released_v8_defaults(strategy_kind=kind)
    raw["bot"].pop("short")
    raw["optimize"]["bounds"]["short"]["risk"]["n_positions"] = [1, 3]
    raw["optimize"]["bounds"]["short"]["risk"]["total_wallet_exposure_limit"] = [0.0, 1.5]
    original = deepcopy(raw)
    converted = migrate_v8_to_v9(raw, best_effort=True)
    config = prepare_config(converted.config, live_only=True, verbose=False)
    assert side_activation(config) == {"long": True, "short": True}
    assert config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 0.0
    bounds = dict(zip((key for key, _ in get_optimization_key_paths(config)),
                      extract_bounds_tuple_list_from_config(config)))
    assert bounds["short_total_wallet_exposure_limit"].high > 0
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
