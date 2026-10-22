"""Cooldown grouping must not change effective policy, bounds, or overrides."""

import argparse
from copy import deepcopy
import json

import pytest

from config import get_template_config, prepare_config, compile_runtime_config
from config.overrides import parse_overrides
from config.optimize_bounds import flatten_optimize_bounds
from config.param_paths import resolve_bound_selectors, require_existing_config_path
from config_utils import (
    clean_config,
    add_config_arguments,
    create_acronym,
    update_config_with_args,
)
from tools.migrate_config_v9 import migrate as migrate_config_v9


def legacy_config():
    config = clean_config(get_template_config())
    config["config_version"] = "v8.4.0"
    config["bot"]["global"].pop("enabled")
    for side in ("long", "short"):
        config["bot"][side].pop("enabled")
    for key in ("lookback_ema_span_minutes", "risk_bootstrap_max_lookback_minutes",
                "realized_loss_gate_lookback_minutes"):
        config["live"].pop(key, None)
    config["live"]["pnls_max_lookback_days"] = 30.0
    config["optimize"]["bounds"].pop("live", None)
    # This fixture migrates unrelated strategy fields with old HSL disabled,
    # not newly generated current-schema HSL restart authorization.
    for side in ("long", "short"):
        config["bot"][side]["hsl"].update(enabled=False, restart_after_red_policy=None)
    config["optimize"]["fixed_runtime_overrides"] = {
        key: value
        for key, value in config["optimize"]["fixed_runtime_overrides"].items()
        if ".hsl." not in key
    }
    for root in (config["bot"], config["optimize"]["bounds"]):
        for side in ("long", "short"):
            root[side]["risk"]["entry_cooldown_minutes"] = root[side].pop(
                "entry_cooldown"
            )["base_duration_minutes"]
    return config


def prepare_migrated_legacy(config):
    original = deepcopy(config)
    with pytest.raises(ValueError, match="migrate-config-v9"):
        prepare_config(config, verbose=False)
    migrated = migrate_config_v9(config, memory_span_minutes=43200.0).config
    assert config == original
    return prepare_config(migrated, verbose=False)


@pytest.mark.parametrize("duration", [0.0, 0.05, 2.5, 24.1])
def test_migration_preserves_runtime_optimizer_and_saved_roundtrip(duration):
    old = legacy_config()
    old["bot"]["long"]["risk"]["entry_cooldown_minutes"] = duration
    old["optimize"]["bounds"]["long"]["risk"]["entry_cooldown_minutes"] = [
        0.05,
        17.5,
        0.05,
    ]
    prepared = prepare_migrated_legacy(old)
    assert prepared["bot"]["long"]["entry_cooldown"] == {
        "base_duration_minutes": duration,
        "min_duration_minutes": 0.0,
        "max_duration_minutes": None,
        "weights_minutes": {"exposure_ratio": 0.0, "adverse_directionality": 0.0},
    }
    assert "entry_cooldown_minutes" not in prepared["bot"]["long"]["risk"]
    runtime = compile_runtime_config(prepared)
    assert runtime["bot"]["long"]["risk_entry_cooldown_minutes"] == duration
    bounds = flatten_optimize_bounds(
        prepared["optimize"]["bounds"], strategy_kind="trailing_martingale"
    )
    assert bounds["long_risk_entry_cooldown_minutes"] == [0.05, 17.5, 0.05]
    saved = clean_config(prepared)
    assert clean_config(prepare_config(saved, verbose=False)) == saved
    canonical = deepcopy(old)
    for root in (canonical["bot"], canonical["optimize"]["bounds"]):
        for side in ("long", "short"):
            root[side]["entry_cooldown"] = {
                "base_duration_minutes": root[side]["risk"].pop(
                    "entry_cooldown_minutes"
                )
            }
    assert (
        compile_runtime_config(prepare_migrated_legacy(canonical))["bot"]
        == runtime["bot"]
    )


@pytest.mark.parametrize(
    "file_old,inline_old", [(True, False), (False, True), (True, True), (False, False)]
)
def test_external_and_inline_precedence_survives_mixed_spellings(
    tmp_path, file_old, inline_old
):
    def patch(old, value):
        return {
            "bot": {
                "long": (
                    {"risk": {"entry_cooldown_minutes": value}}
                    if old
                    else {"entry_cooldown": {"base_duration_minutes": value}}
                )
            }
        }

    config = get_template_config()
    file = tmp_path / "coin.json"
    file.write_text(json.dumps(patch(file_old, 8.0)))
    config["coin_overrides"] = {
        "BTC": {"override_config_path": str(file), **patch(inline_old, 0.05)}
    }
    if file_old or inline_old:
        # Inline scope validation runs first when that authored patch is retired;
        # otherwise the external source is rejected before file values are pruned.
        error = (r"coin_overrides\.BTC\.bot\.long\.risk\.entry_cooldown_minutes is not overridable"
                 if inline_old else
                 r"coin_overrides\.BTC\.override_config_path.*config\.bot\.long\.risk\.entry_cooldown_minutes is an unknown configuration field")
        with pytest.raises(ValueError, match=error):
            parse_overrides(prepare_config(config, verbose=False), verbose=False)
        released = legacy_config()
        released["coin_overrides"] = deepcopy(config["coin_overrides"])
        prepared = prepare_migrated_legacy(released)
    else:
        prepared = parse_overrides(prepare_config(config, verbose=False), verbose=False)
    assert prepared["coin_overrides"]["BTC"]["bot"]["long"]["entry_cooldown"] == {
        "base_duration_minutes": 0.05
    }


def test_current_config_rejects_conflicting_retired_cooldown_path():
    config = get_template_config()
    config["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 9.0
    original = deepcopy(config)
    with pytest.raises(ValueError, match="entry_cooldown_minutes.*unknown configuration field"):
        prepare_config(config, verbose=False)
    assert config == original


@pytest.mark.parametrize("reverse", [False, True])
def test_scenario_alias_and_nested_legacy_collision(reverse):
    config = legacy_config()
    items = [
        ("bot", {"long": {"risk": {"entry_cooldown_minutes": 9.0}}}),
        ("long.entry_cooldown.base_duration_minutes", 0.05),
    ]
    config["backtest"]["scenarios"] = [
        {"label": "mixed", "overrides": dict(reversed(items) if reverse else items)}
    ]
    original = deepcopy(config)
    with pytest.raises(ValueError, match="conflicts.*canonical"):
        migrate_config_v9(config, memory_span_minutes=43200.0)
    assert config == original

@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("spelling", ["dotted", "underscores", "short"])
def test_retired_cli_flags_reject_and_canonical_path_applies(side, spelling):
    config = get_template_config()
    parser = argparse.ArgumentParser()
    allowed = add_config_arguments(
        parser, config, command="backtest", help_all=True, group_map={}
    )
    old = f"bot.{side}.risk.entry_cooldown_minutes"
    flag = {
        "dotted": "--" + old,
        "underscores": "--" + old.replace(".", "_"),
        "short": "-" + create_acronym(old, set()),
    }[spelling]
    with pytest.raises(SystemExit):
        parser.parse_args([flag, "0.05"])
    canonical = f"bot.{side}.entry_cooldown.base_duration_minutes"
    canonical_flag = "--" + (canonical.replace(".", "_") if spelling == "underscores" else canonical)
    if spelling == "short":
        canonical_flag = next(option for action in parser._actions if action.dest == canonical
                              for option in action.option_strings
                              if option.startswith("-") and not option.startswith("--"))
    args = parser.parse_args([canonical_flag, "0.05"])
    update_config_with_args(config, args, verbose=False, allowed_keys=allowed)
    assert (
        prepare_config(config, verbose=False)["bot"][side]["entry_cooldown"][
            "base_duration_minutes"
        ]
        == 0.05
    )


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("spelling", ["dotted", "underscores", "short"])
def test_retired_bound_cli_flags_reject_and_canonical_optimizer_values_apply(side, spelling):
    config = get_template_config()
    parser = argparse.ArgumentParser()
    allowed = add_config_arguments(
        parser, config, command="optimize", help_all=True, group_map={}
    )
    old = f"optimize.bounds.{side}.risk.entry_cooldown_minutes"
    flag = {
        "dotted": "--" + old,
        "underscores": "--" + old.replace(".", "_"),
        "short": "-" + create_acronym(old, set()),
    }[spelling]
    if spelling == "short":
        legacy_parser = argparse.ArgumentParser()
        add_config_arguments(
            legacy_parser,
            legacy_config(),
            command="optimize",
            help_all=True,
            group_map={},
        )
        flag = next(
            option
            for action in legacy_parser._actions
            if action.dest == old
            for option in action.option_strings
            if option.startswith("-") and not option.startswith("--")
        )
    with pytest.raises(SystemExit):
        parser.parse_args([flag, "0.05,17.5,0.05"])
    canonical = f"optimize.bounds.{side}.entry_cooldown.base_duration_minutes"
    canonical_flag = "--" + (canonical.replace(".", "_") if spelling == "underscores" else canonical)
    if spelling == "short":
        canonical_flag = next(option for action in parser._actions if action.dest == canonical
                              for option in action.option_strings
                              if option.startswith("-") and not option.startswith("--"))
    args = parser.parse_args([canonical_flag, "0.05,17.5,0.05"])
    update_config_with_args(config, args, verbose=False, allowed_keys=allowed)
    bounds = prepare_config(config, verbose=False)["optimize"]["bounds"][side]
    assert bounds["entry_cooldown"]["base_duration_minutes"] == [0.05, 17.5, 0.05]


def test_retired_leaf_selectors_reject_and_canonical_gene_identity_is_preserved():
    config = prepare_config(get_template_config(), verbose=False)
    bounds = flatten_optimize_bounds(config["optimize"]["bounds"], strategy_kind="trailing_martingale")
    for selector in ("long.risk.entry_cooldown_minutes", "bot.long.risk_entry_cooldown_minutes"):
        with pytest.raises(KeyError, match="missing"):
            require_existing_config_path(config, selector)
        assert resolve_bound_selectors(config, [selector], bounds) == {}
    canonical = "bot.long.entry_cooldown.base_duration_minutes"
    assert require_existing_config_path(config, canonical) == (
        "bot", "long", "entry_cooldown", "base_duration_minutes",
    )
    assert set(resolve_bound_selectors(config, [canonical], bounds)) == {"long_risk_entry_cooldown_minutes"}
    assert set(resolve_bound_selectors(config, ["*.entry_cooldown"], bounds)) == {
        f"{side}_{key}"
        for side in ("long", "short")
        for key in ("risk_entry_cooldown_minutes", "entry_cooldown_min_duration_minutes",
                    "entry_cooldown_weights_minutes_exposure_ratio",
                    "entry_cooldown_weights_minutes_adverse_directionality")
    }


def test_risk_group_excludes_cooldown_and_canonical_group_retains_it():
    config = prepare_config(get_template_config(), verbose=False)
    bounds = flatten_optimize_bounds(
        config["optimize"]["bounds"], strategy_kind="trailing_martingale"
    )
    risk = resolve_bound_selectors(config, ["long.risk"], bounds)
    assert "long_risk_entry_cooldown_minutes" not in risk
    assert {"long_n_positions", "long_total_wallet_exposure_limit"} <= set(risk)
    assert "long_risk_entry_cooldown_minutes" in resolve_bound_selectors(
        config, ["long.entry_cooldown"], bounds
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_dotted_scenario_group_conflict_rejects_in_both_orders(reverse):
    config = legacy_config()
    risk = deepcopy(config["bot"]["long"]["risk"])
    risk["entry_cooldown_minutes"] = 19.0
    items = [
        ("bot.long.risk", risk),
        ("long.entry_cooldown", {"base_duration_minutes": 7.0}),
    ]
    config["backtest"]["scenarios"] = [
        {"label": "groups", "overrides": dict(reversed(items) if reverse else items)}
    ]
    original = deepcopy(config)
    with pytest.raises(ValueError, match="conflicts.*canonical"):
        migrate_config_v9(config, memory_span_minutes=43200.0)
    assert config == original
