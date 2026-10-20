"""Offline editor contracts, including strict authoring and sparse authority."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest
from jsonschema import Draft202012Validator, ValidationError

from config import prepare_config
from config.field_registry import (
    describe_field, get_coin_override_paths, get_field_registry,
    get_shared_optimizer_bindings, get_special_optimizer_bindings,
)
from config.field_spec import CoinOverridePolicy
from config.hsl import generated_template
from config.optimize_bounds import flatten_optimize_bounds
from config.overrides import get_allowed_modifications, parse_overrides
from config.param_paths import OPTIMIZABLE_BOT_KEY_PATHS, resolve_optimizer_key_path
from config.schema import get_template_config
from config.json_schema import generate_schema
from tools.config_schema import DEFAULT_OUTPUT, main, render_schema


@pytest.fixture(scope="module")
def validator():
    schema = generate_schema()
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def source(**sections):
    return {"config_version": "v9.0.0", **sections}


def test_generated_schema_is_current_and_checks_do_not_write(tmp_path, capsys):
    assert DEFAULT_OUTPUT.read_text() == render_schema()
    assert main(["--check"]) == 0
    destination = tmp_path / "schema.json"
    assert main(["--output", str(destination)]) == 0
    original = destination.read_bytes()
    assert main(["--check", "--output", str(destination)]) == 0
    assert destination.read_bytes() == original
    destination.write_text("{}\n")
    assert main(["--check", "--output", str(destination)]) == 1
    assert destination.read_text() == "{}\n"


def test_all_schema_references_are_local_and_resolve(validator):
    schema = validator.schema
    def visit(value):
        if isinstance(value, dict):
            if "$ref" in value:
                pointer = value["$ref"]
                assert pointer.startswith("#/")
                target = schema
                for part in pointer[2:].split("/"):
                    target = target[part.replace("~1", "/").replace("~0", "~")]
                assert isinstance(target, (dict, bool))
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    visit(schema)


def test_schema_generation_and_metadata_are_offline(monkeypatch):
    import socket

    def forbidden(*args, **kwargs):
        raise AssertionError("editor metadata must not access the network")
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    schema = generate_schema()
    Draft202012Validator(schema).validate(get_template_config())
    assert describe_field("bot.global.hsl.enabled")["default"] == {"declared": False}


def test_all_intentional_examples_match_editor_schema(validator):
    for path in sorted(Path("configs/examples").glob("*.json")):
        payload = json.loads(path.read_text())
        validator.validate(payload)


@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("scope", ["coin", "position_side", "portfolio"])
def test_templates_and_canonical_roundtrips_match_schema(validator, strategy, scope):
    payload = generated_template(get_template_config(), scope)
    payload["bot"]["global"]["strategy_kind"] = strategy
    validator.validate(payload)
    prepared = prepare_config(payload, verbose=False)
    prepared = {key: value for key, value in prepared.items() if not key.startswith("_")}
    prepared["live"].pop("base_config_path", None)
    validator.validate(prepared)


@pytest.mark.parametrize("payload", [
    source(), source(bot={}), source(bot={"long": {}}),
    source(backtest={"filter_by_min_effective_cost": None, "scenarios": [{
        "label": None, "start_date": None, "end_date": None, "exchanges": "bybit",
        "coins": {"long": ["BTC"], "short": None}, "ignored_coins": "ETH",
        "coin_sources": {"BTC": None},
    }]}),
    source(optimize={"scoring": ["adg", {"metric": "adg_strategy_eq", "goal": "max", "scenario": None}],
                     "limits": [{"metric": "adg", "penalize_if": "less_than", "value": 0.0}]}),
    source(live={"startup": {"startup_phase_budgets": {"account": {"elapsed_ms": 123, "since_previous_ms": 0}}}}),
    source(backtest={"market_settings": {"overrides": {"BTC": {"min_cost": 0.1}},
                                        "overrides_by_exchange": {"bybit": {"ETH": {"c_mult": 1.0}}}}}),
])
def test_sparse_dynamic_forms_validate_without_default_insertion(validator, payload):
    original = deepcopy(payload)
    validator.validate(payload)
    assert payload == original


@pytest.mark.parametrize("payload", [
    {}, {"config_version": "v8.6.0"}, source(unknown=1), source(fitness=0.5),
    {"config": source()}, source(_raw={}), source(bot={"risk": {}}),
    source(bot={"long": {"not_a_field": 1.0}}),
    source(live={"startup": {"startup_phase_budgets": {"wrong_phase": {"elapsed_ms": 1}}}}),
    source(live={"startup": {"startup_phase_budgets": {"account": {"elapsed_ms": 1.5}}}}),
    source(live={"startup": {"startup_phase_budgets": {"account": {"bad_key": 1}}}}),
    source(bot={"long": {"universe": {"approved_coins": {"long": ["BTC"], "wrong_side": []}}}}),
    source(optimize={"scoring": [{"metric": "adg", "surprise": True}]}),
    source(backtest={"scenarios": [{"unknown": True}]}),
    source(backtest={"market_settings": {"overrides": {"BTC": {"minimum_cost": 1}}}}),
])
def test_editor_schema_rejects_unknown_fields_results_and_wrong_record_types(validator, payload):
    with pytest.raises(ValidationError):
        validator.validate(payload)


@pytest.mark.parametrize("value", [0.5, [0.5], [0.0, 1.0], [0.0, 1.0, None], [0.0, 1.0, 0.01]])
def test_canonical_nested_bounds_support_documented_forms(validator, value):
    validator.validate(source(optimize={"bounds": {"long": {"forager": {"score_weights": {"volume": value}}}}}))


@pytest.mark.parametrize("bounds", [
    {"risk": {}}, {"unstuck": {}}, {"long_forager_score_weights_volume": [0, 1]},
    {"long": {"forager": {"score_weights_volume": [0, 1]}}},
    {"long": {"risk": {"position_exposure_enforcer.enabled": [0, 1]}}},
    {"long": {"forager": {"score_weights": {"volume": []}}}},
])
def test_bounds_do_not_open_aliases_or_non_numeric_genes(validator, bounds):
    with pytest.raises(ValidationError):
        validator.validate(source(optimize={"bounds": bounds}))


@pytest.mark.parametrize("patch", [
    {"bot.long.unstuck.ema_span_0": 60.5},
    {"bot": {"long": {"unstuck.ema_span_0": 60.5}}},
    {"coin_overrides.BTC.bot.long.unstuck.ema_span_0": 60.5},
    {"bot.long.entry_cooldown.max_duration_minutes": None},
])
@pytest.mark.parametrize("owner", ["scenario", "fixed"])
def test_literal_and_nested_overrides_reuse_known_fields(validator, patch, owner):
    payload = source(**({"backtest": {"scenarios": [{"overrides": patch}]}} if owner == "scenario"
                       else {"optimize": {"fixed_runtime_overrides": patch}}))
    validator.validate(payload)


@pytest.mark.parametrize("patch", [
    {"bot.long.not_a_field": 1}, {"long.unstuck.ema_span_0": 1},
    {"coin_overrides.BTC.bot.long.risk.max_positions": 1},
    {"coin_overrides.BTC.bot.global.loss_gate.max_drawdown_ratio": 0.5},
    {"coin_overrides.BTC.override_config_path": "policy.json"},
])
def test_overrides_cannot_expand_canonical_paths_or_coin_permissions(validator, patch):
    with pytest.raises(ValidationError):
        validator.validate(source(backtest={"scenarios": [{"overrides": patch}]}))


@pytest.mark.parametrize("patch", [
    {"bot.long.unstuck": {"close_pct": 0.1}},
    {"bot": {"long.unstuck": {"close_pct": 0.1}}},
    {"optimize.prefilter.penalty_weight": 1.0},
    {"optimize": {"prefilter": {"penalty_weight": 1.0}}},
    {"coin_overrides": {"BTC": {"live": {"account": {"leverage": 2}}}}},
    {"coin_overrides.BTC.bot.long.unstuck": {"close_pct": 0.1}},
])
def test_fixed_runtime_schema_reuses_paths_but_rejects_object_pins_and_prefilter_policy(validator, patch):
    with pytest.raises(ValidationError):
        validator.validate(source(optimize={"fixed_runtime_overrides": patch}))


def test_inline_coin_permissions_and_hsl_context_remain_narrow(validator):
    payload = source(coin_overrides={"BTC": {"override_config_path": "policy.json", "bot": {"long": {
        "wallet_exposure_limit": 0.5, "hsl": {"enabled": False},
        "unstuck": {"ema_span_0": 15.5},
    }}}})
    validator.validate(payload)
    for scope in ("position_side", "portfolio"):
        payload["bot"] = {"global": {"hsl": {"scope": scope}}}
        with pytest.raises(ValidationError):
            validator.validate(payload)
    payload = source(coin_overrides={"BTC": {"bot": {"long": {"risk": {"max_positions": 2}}}}})
    with pytest.raises(ValidationError):
        validator.validate(payload)


def test_shared_metadata_matches_adapters_without_changing_special_validation_membership():
    registry = get_field_registry()
    assert get_special_optimizer_bindings() == OPTIMIZABLE_BOT_KEY_PATHS
    assert "long_unstuck_ema_span_0" not in OPTIMIZABLE_BOT_KEY_PATHS
    for scope in ("coin", "position_side", "portfolio"):
        config = generated_template(get_template_config(), scope)
        for key, path in get_shared_optimizer_bindings().items():
            if key.startswith("hsl_") and scope != "portfolio":
                continue
            assert resolve_optimizer_key_path(config, key) == path
            field = registry.find(path)
            assert field.optimize_key == key
            assert field.optimizable
    field = registry.find(("bot", "long", "hsl", "drawdown_ratio_threshold"))
    assert field.recommended_bounds == (0.15, 0.15, 0.001)
    assert field.domain.minimum == 0 and field.domain.maximum == 1
    assert field.domain.exclusive_minimum
    assert field.coin_override is CoinOverridePolicy.COIN_HSL_ONLY
    assert field.denominator == "scope equity plus remembered drawdown"
    assert registry.find(("bot", "long", "risk", "total_wallet_exposure_limit")).coin_override is CoinOverridePolicy.FORBIDDEN


def test_metadata_report_is_separate_and_optional_authority_has_no_default(capsys):
    report = describe_field("bot.global.hsl.enabled")
    assert report["default"] == {"declared": False}
    assert report["optimizer"]["eligible"] is False
    assert main(["--field", "bot.global.hsl.enabled"]) == 0
    assert json.loads(capsys.readouterr().out) == json.loads(json.dumps(report))
    with pytest.raises(KeyError):
        describe_field("bot.global.unknown")
    patch = describe_field("coin_overrides.BTC.bot.long.unstuck.ema_span_0")
    assert patch["default"] == {"declared": False}
    assert patch["unit"] == "minutes"
    assert patch["optimizer"]["eligible"] is False
    with pytest.raises(KeyError):
        describe_field("coin_overrides.BTC.bot.long.risk.max_positions")


def test_schema_cli_is_offline_and_registered():
    completed = subprocess.run([sys.executable, "-c", "from passivbot_cli.main import main; raise SystemExit(main())",
                                "tool", "config-schema", "--check"],
                               text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("path", [
    "bot.global.strategy_kind", "bot.global.hedge_mode",
    "bot.global.universe.minimum_coin_age_days", "bot.global.universe.filter_by_min_effective_cost",
    "bot.global.execution.market_orders_allowed", "bot.global.execution.market_order_near_touch_threshold",
    "bot.global.forager.score_hysteresis_pct", "bot.global.warmup.ratio", "bot.global.warmup.max_minutes",
    "bot.long.universe.approved_coins", "bot.short.universe.ignored_coins",
    "bot.long.forager.always_include",
])
def test_shared_policy_relocation_does_not_grant_search_or_coin_authority(path):
    report = describe_field(path)
    assert report["optimizer"]["eligible"] is False
    assert report["optimizer"]["key"] is None
    assert report["coin_override"] == "forbidden"


def test_side_owned_coin_sources_and_persistent_selection_have_distinct_shapes(validator):
    validator.validate(source(bot={"long": {
        "universe": {"approved_coins": "coins.json", "ignored_coins": None},
        "forager": {"always_include": ["BTC", "bybit::ETH/USDT:USDT"]},
    }}))
    for patch in (
        {"universe": {"approved_coins": {"long": ["BTC"]}}},
        {"forager": {"always_include": "coins.json"}},
        {"forager": {"always_include": [True]}},
    ):
        with pytest.raises(ValidationError):
            validator.validate(source(bot={"long": patch}))
