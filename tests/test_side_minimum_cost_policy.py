"""Directional minimum-cost admission stays explicit through configuration and Rust."""
from copy import deepcopy
from itertools import product

import pytest

from config import prepare_config
from config.authoring import compact_config, explain_config, resolved_config
from config.policy import effective_minimum_cost_filters, validate_minimum_cost_filters
from config.schema import get_template_config
from config.migrations.v9 import migrate_v8_to_v9
from released_v8_fixtures import released_v8_policy


@pytest.mark.parametrize("long,short,override", product((False, True), (False, True), (None, False, True)))
def test_independent_inheritance_and_authored_export_roundtrip(long, short, override):
    source = get_template_config()
    source["optimize"]["bounds"] = {}
    for side, flag in (("long", long), ("short", short)):
        source["bot"][side]["universe"]["filter_by_min_effective_cost"] = flag
    source["backtest"]["filter_by_min_effective_cost"] = override
    original = deepcopy(source)
    expected = {"long": long, "short": short} if override is None else dict.fromkeys(("long", "short"), override)
    assert effective_minimum_cost_filters(prepare_config(source, verbose=False)) == expected
    for output in (compact_config(source), resolved_config(source)):
        assert output["backtest"]["filter_by_min_effective_cost"] is override
        assert effective_minimum_cost_filters(prepare_config(output, verbose=False)) == expected
    report = explain_config(source, "backtest.filter_by_min_effective_cost")
    assert report["value"] == (expected if override is None else override)
    if override is None:
        assert report["sources"][-2:] == [
            {"kind": "inherited", "path": f"bot.{side}.universe.filter_by_min_effective_cost", "value": expected[side]}
            for side in ("long", "short")
        ]
    assert source == original


@pytest.mark.parametrize("invalid", [None, True, False, {}, {"long": True}, {"long": True, "short": 1},
                                         {"long": None, "short": False}, {"long": True, "short": False, "extra": False}])
def test_native_transport_pair_is_complete_and_strict(invalid):
    with pytest.raises(TypeError):
        validate_minimum_cost_filters(invalid)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("invalid", [None, 0, 1, "false"])
def test_canonical_side_flag_requires_boolean(side, invalid):
    source = get_template_config()
    source["bot"][side]["universe"]["filter_by_min_effective_cost"] = invalid
    with pytest.raises((ValueError, TypeError)):
        prepare_config(source, verbose=False)


@pytest.mark.parametrize("location", ["root", "fixed", "scenario"])
def test_retired_development_global_path_is_not_a_loader_alias(location):
    source = get_template_config()
    path = "bot.global.universe.filter_by_min_effective_cost"
    if location == "root":
        source["bot"]["global"]["universe"]["filter_by_min_effective_cost"] = False
    elif location == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = {path: False}
    else:
        source["backtest"]["scenarios"] = [{"label": "old", "overrides": {path: False}}]
    with pytest.raises((ValueError, KeyError)):
        prepare_config(source, verbose=False)


@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("location", ["root", "fixed", "scenario"])
def test_released_shared_flag_fans_out_without_changing_simulation_override(value, location):
    source = released_v8_policy()
    source["backtest"]["filter_by_min_effective_cost"] = None
    path = "live.filter_by_min_effective_cost"
    if location == "root":
        source["live"]["filter_by_min_effective_cost"] = value
    elif location == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = {path: value}
    else:
        source["backtest"]["scenarios"] = [{"label": "one", "overrides": {path: value}}]
    original = deepcopy(source)
    migrated = migrate_v8_to_v9(source, memory_span_minutes=120.0)
    result = migrated.config
    if location == "root":
        assert {side: result["bot"][side]["universe"]["filter_by_min_effective_cost"] for side in ("long", "short")} == dict.fromkeys(("long", "short"), value)
    else:
        patch = result["optimize"]["fixed_runtime_overrides"] if location == "fixed" else result["backtest"]["scenarios"][0]["overrides"]
        assert patch == {f"bot.{side}.universe.filter_by_min_effective_cost": value for side in ("long", "short")}
    assert result["backtest"]["filter_by_min_effective_cost"] is None
    context = {"optimize": True} if location == "fixed" else {"scenario": "one"} if location == "scenario" else {}
    report = explain_config(result, "backtest.filter_by_min_effective_cost", **context)
    assert report["value"] == {"long": value, "short": value}
    assert any("minimum-cost" in change["reason"] or "live/backtest policy" in change["reason"] for change in migrated.changes)
    assert source == original


def test_effective_evaluation_identity_records_both_independent_flags():
    from optimization.evaluation_contract import build_evaluation_contract
    config = prepare_config(get_template_config(), verbose=False)
    original = build_evaluation_contract(config)
    assert original["backtest_policy"]["filter_by_min_effective_cost"] == {"long": True, "short": True}
    for side in ("long", "short"):
        changed = deepcopy(config)
        changed["bot"][side]["universe"]["filter_by_min_effective_cost"] = False
        current = build_evaluation_contract(changed)
        assert current != original
        assert current["backtest_policy"]["filter_by_min_effective_cost"][side] is False


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_optimizer_mirroring_preserves_directional_admission_policy(kind):
    from optimizer_overrides import optimizer_overrides
    source = get_template_config()
    source["bot"]["global"]["strategy_kind"] = kind
    source["bot"]["long"]["universe"]["filter_by_min_effective_cost"] = False
    source["bot"]["short"]["universe"]["filter_by_min_effective_cost"] = True
    effective = optimizer_overrides(["mirror_short_from_long"], source)
    assert effective_minimum_cost_filters(effective) == {"long": False, "short": True}


@pytest.mark.parametrize("location", ["fixed", "scenario"])
def test_released_fanout_rejects_conflicting_explicit_side_authority(location):
    source = released_v8_policy()
    patch = {"live.filter_by_min_effective_cost": False,
             "bot.short.universe.filter_by_min_effective_cost": True}
    if location == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = patch
    else:
        source["backtest"]["scenarios"] = [{"label": "one", "overrides": patch}]
    with pytest.raises(ValueError, match="conflicts"):
        migrate_v8_to_v9(source, memory_span_minutes=120.0)


def test_actual_cli_exports_and_explains_independent_inheritance(tmp_path, capsys):
    import json
    from passivbot_cli import main as cli
    source = get_template_config()
    source["bot"]["long"]["universe"]["filter_by_min_effective_cost"] = False
    source["backtest"]["filter_by_min_effective_cost"] = None
    filename = tmp_path / "source.json"
    filename.write_text(json.dumps(source))
    for mode in ("compact", "resolved"):
        output = tmp_path / f"{mode}.json"
        assert cli.main(["tool", "clean-config", str(filename), str(output), "--mode", mode]) == 0
        exported = json.loads(output.read_text())
        assert exported["backtest"]["filter_by_min_effective_cost"] is None
        assert effective_minimum_cost_filters(prepare_config(exported, verbose=False)) == {"long": False, "short": True}
    capsys.readouterr()
    assert cli.main(["tool", "config-explain", str(filename), "backtest.filter_by_min_effective_cost", "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["value"] == {"long": False, "short": True}


@pytest.mark.parametrize("command", ["live", "backtest", "optimize"])
def test_canonical_side_cli_flags_parse_independently_and_global_flag_rejects(command):
    import argparse
    from config_utils import add_arguments_recursively
    parser = argparse.ArgumentParser()
    add_arguments_recursively(parser, get_template_config(), command=command)
    args = vars(parser.parse_args(["--bot.long.universe.filter_by_min_effective_cost", "n",
                                  "--bot.short.universe.filter_by_min_effective_cost", "y"]))
    assert args["bot.long.universe.filter_by_min_effective_cost"] is False
    assert args["bot.short.universe.filter_by_min_effective_cost"] is True
    with pytest.raises(SystemExit):
        parser.parse_args(["--bot.global.universe.filter_by_min_effective_cost", "n"])


@pytest.mark.parametrize("version", ["v8.6.0", "v9.0.0"])
def test_explicit_migration_does_not_translate_unpublished_global_flag(version):
    from tools.migrate_config_v9 import migrate
    source = released_v8_policy() if version.startswith("v8") else get_template_config()
    source["bot"].setdefault("global", {}).setdefault("universe", {})["filter_by_min_effective_cost"] = False
    with pytest.raises(ValueError, match="filter_by_min_effective_cost"):
        migrate(source, memory_span_minutes=120.0)
