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


def legacy_config():
    config = clean_config(get_template_config())
    config["config_version"] = "v8.4.0"
    for root in (config["bot"], config["optimize"]["bounds"]):
        for side in ("long", "short"):
            root[side]["risk"]["entry_cooldown_minutes"] = root[side].pop(
                "entry_cooldown"
            )["base_duration_minutes"]
    return config


@pytest.mark.parametrize("duration", [0.0, 0.05, 2.5, 24.1])
def test_migration_preserves_runtime_optimizer_and_saved_roundtrip(duration):
    old = legacy_config()
    old["bot"]["long"]["risk"]["entry_cooldown_minutes"] = duration
    old["optimize"]["bounds"]["long"]["risk"]["entry_cooldown_minutes"] = [
        0.05,
        17.5,
        0.05,
    ]
    prepared = prepare_config(old, verbose=False)
    assert prepared["bot"]["long"]["entry_cooldown"] == {
        "base_duration_minutes": duration
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
        compile_runtime_config(prepare_config(canonical, verbose=False))["bot"]
        == runtime["bot"]
    )


@pytest.mark.parametrize(
    "file_old,inline_old", [(True, False), (False, True), (True, True)]
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
    prepared = parse_overrides(prepare_config(config, verbose=False), verbose=False)
    assert prepared["coin_overrides"]["BTC"]["bot"]["long"]["entry_cooldown"] == {
        "base_duration_minutes": 0.05
    }


def test_conflicting_canonical_value_wins_with_warning(caplog):
    config = get_template_config()
    config["bot"]["long"]["risk"]["entry_cooldown_minutes"] = 9.0
    assert (
        prepare_config(config, verbose=False)["bot"]["long"]["entry_cooldown"][
            "base_duration_minutes"
        ]
        == 24.1
    )
    assert "Conflicting entry cooldown" in caplog.text


@pytest.mark.parametrize("reverse", [False, True])
def test_scenario_alias_and_nested_legacy_collision(reverse, caplog):
    config = legacy_config()
    items = [
        ("bot", {"long": {"risk": {"entry_cooldown_minutes": 9.0}}}),
        ("long.entry_cooldown.base_duration_minutes", 0.05),
    ]
    config["backtest"]["scenarios"] = [
        {"label": "mixed", "overrides": dict(reversed(items) if reverse else items)}
    ]
    result = prepare_config(config, verbose=False)["backtest"]["scenarios"][0][
        "overrides"
    ]
    assert result == {"bot.long.entry_cooldown.base_duration_minutes": 0.05}
    assert "Conflicting entry cooldown" in caplog.text


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("spelling", ["dotted", "underscores", "short"])
def test_old_cli_flags_apply_to_new_path(side, spelling):
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
    args = parser.parse_args([flag, "0.05"])
    update_config_with_args(config, args, verbose=False, allowed_keys=allowed)
    assert (
        prepare_config(config, verbose=False)["bot"][side]["entry_cooldown"][
            "base_duration_minutes"
        ]
        == 0.05
    )


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("spelling", ["dotted", "underscores", "short"])
def test_old_bound_cli_flags_keep_optimizer_values(side, spelling):
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
    args = parser.parse_args([flag, "0.05,17.5,0.05"])
    update_config_with_args(config, args, verbose=False, allowed_keys=allowed)
    bounds = prepare_config(config, verbose=False)["optimize"]["bounds"][side]
    assert bounds["entry_cooldown"]["base_duration_minutes"] == [0.05, 17.5, 0.05]


def test_old_and_new_leaf_selectors_preserve_gene_identity():
    config = prepare_config(get_template_config(), verbose=False)
    bounds = flatten_optimize_bounds(
        config["optimize"]["bounds"], strategy_kind="trailing_martingale"
    )
    for selector in [
        "long.risk.entry_cooldown_minutes",
        "bot.long.risk_entry_cooldown_minutes",
        "long.entry_cooldown.base_duration_minutes",
    ]:
        assert require_existing_config_path(config, selector) == (
            "bot",
            "long",
            "entry_cooldown",
            "base_duration_minutes",
        )
        assert set(resolve_bound_selectors(config, [selector], bounds)) == {
            "long_risk_entry_cooldown_minutes"
        }
    for selector in [
        "*.risk.entry_cooldown_minutes",
        "entry_cooldown_minutes",
        "*.entry_cooldown",
    ]:
        assert set(resolve_bound_selectors(config, [selector], bounds)) == {
            "long_risk_entry_cooldown_minutes",
            "short_risk_entry_cooldown_minutes",
        }


def test_legacy_risk_group_selector_still_includes_cooldown():
    config = prepare_config(get_template_config(), verbose=False)
    bounds = flatten_optimize_bounds(
        config["optimize"]["bounds"], strategy_kind="trailing_martingale"
    )
    assert "long_risk_entry_cooldown_minutes" in resolve_bound_selectors(
        config, ["long.risk"], bounds
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_dotted_scenario_group_replacement_migrates_with_canonical_precedence(reverse):
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
    result = prepare_config(config, verbose=False)["backtest"]["scenarios"][0][
        "overrides"
    ]
    assert result["bot.long.entry_cooldown.base_duration_minutes"] == 7.0
    assert "entry_cooldown_minutes" not in result["bot.long.risk"]
    assert result["bot.long.risk"]["n_positions"] == risk["n_positions"]
