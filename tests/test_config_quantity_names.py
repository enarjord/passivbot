"""Public quantity names preserve values and the stable native transport."""

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from backtest import _resolve_backtest_hsl
from config import get_template_config, prepare_config
from config.hsl import generated_template, native_hsl_mode, native_hsl_policy
from config.optimize_bounds import flatten_optimize_bounds, set_flat_optimize_bound
from config.param_paths import resolve_bound_selectors, resolve_optimizer_key_path
from released_v8_fixtures import released_v8_policy
from tools.migrate_config_v9 import migrate


def test_public_position_side_scope_is_not_truncated_in_console_diagnostics():
    from live.event_bus import EventTypes, LiveEvent, format_console_event

    event = LiveEvent(EventTypes.HSL_STATUS, data={
        "engine": "hsl", "signal_mode": "position_side", "observation_status": "current",
        "counts": {name: 0 for name in ("green", "red", "inactive", "unavailable", "estimated")},
    })
    assert "mode=position_side " in format_console_event(event)


@pytest.mark.parametrize("scope,native", [
    ("coin", "coin"), ("position_side", "pside"), ("portfolio", "portfolio"),
])
def test_public_scope_and_policy_are_adapted_only_at_the_native_boundary(scope, native):
    config = prepare_config(generated_template(get_template_config(), scope), verbose=False)
    policies = {"TEST": [deepcopy(config["bot"][side]["hsl"]) for side in ("long", "short")]}
    original = deepcopy((config, policies))
    payload = _resolve_backtest_hsl(config, policies)
    assert native_hsl_mode(scope) == native == payload["mode"]
    transported = list(payload["sides"])
    if scope == "portfolio":
        transported.append(payload["portfolio"])
    elif scope == "coin":
        transported.extend(payload["coins"]["TEST"])
    for policy in transported:
        assert "red_threshold" in policy
        assert "drawdown_ratio_threshold" not in policy
    public = config["bot"]["long"]["hsl"]
    assert native_hsl_policy(public)["red_threshold"] == public["drawdown_ratio_threshold"]
    if scope == "coin":
        from optimization.gpu.hsl import project_bot

        request = SimpleNamespace(bot_params_list=[{"long": {}}],
                                  backtest_params={"equity_hard_stop_loss": payload, "coins": ["TEST"]})
        request.bot_params_list[0]["long"]["n_positions"] = 1
        request.backtest_params["dynamic_wel_by_tradability"] = False
        projected = project_bot(request, 0, "long", config, base=True)
        assert projected["hsl_red_threshold"] == public["drawdown_ratio_threshold"]
        assert "hsl_drawdown_ratio_threshold" not in projected
    assert (config, policies) == original


@pytest.mark.parametrize("scope", ["coin", "position_side", "portfolio"])
def test_public_bounds_keep_stable_optimizer_ids_and_roundtrip_through_the_writer(scope):
    config = prepare_config(generated_template(get_template_config(), scope), verbose=False)
    flat = flatten_optimize_bounds(config["optimize"]["bounds"], strategy_kind="trailing_martingale")
    key = "hsl_red_threshold" if scope == "portfolio" else "long_hsl_red_threshold"
    owner = "portfolio" if scope == "portfolio" else "long"
    assert key in flat
    assert resolve_optimizer_key_path(config, key) == ("bot", owner, "hsl", "drawdown_ratio_threshold")
    assert resolve_optimizer_key_path(config, "long_unstuck_ema_dist") == (
        "bot", "long", "unstuck", "ema_distance_fraction",
    )
    result = {}
    set_flat_optimize_bound(result, "trailing_martingale", key, [0.05, 0.1])
    assert result[owner]["hsl"] == {"drawdown_ratio_threshold": [0.05, 0.1]}
    assert flatten_optimize_bounds(result, strategy_kind="trailing_martingale") == {key: [0.05, 0.1]}


@pytest.mark.parametrize("selector,keys", [
    ("red_threshold", {"long_hsl_red_threshold", "short_hsl_red_threshold"}),
    ("hsl.red_threshold", {"long_hsl_red_threshold", "short_hsl_red_threshold"}),
    ("long.hsl.red_threshold", {"long_hsl_red_threshold"}),
    ("bot.*.hsl.red_threshold", {"long_hsl_red_threshold", "short_hsl_red_threshold"}),
    ("ema_dist", {"long_unstuck_ema_dist", "short_unstuck_ema_dist"}),
    ("unstuck.ema_dist", {"long_unstuck_ema_dist", "short_unstuck_ema_dist"}),
    ("short.unstuck.ema_dist", {"short_unstuck_ema_dist"}),
    ("bot.*.unstuck.ema_dist", {"long_unstuck_ema_dist", "short_unstuck_ema_dist"}),
])
def test_released_quantity_selector_suffixes_preserve_exact_gene_authority(selector, keys):
    source = released_v8_policy()
    source["optimize"]["fixed_params"] = [selector]
    original = deepcopy(source)
    migration = migrate(source, memory_span_minutes=123.5)
    config = migration.config
    flat = flatten_optimize_bounds(config["optimize"]["bounds"], strategy_kind="trailing_martingale")
    assert set(resolve_bound_selectors(config, config["optimize"]["fixed_params"], flat)) == keys
    assert source == original
    assert any(change["path"] == "optimize.fixed_params" and change["old"] == [selector]
               for change in migration.changes)


@pytest.mark.parametrize("old,new", [("coin", "coin"), ("pside", "position_side")])
def test_released_scope_conversion_reports_the_authored_value_without_rescaling(old, new):
    source = released_v8_policy()
    source["live"]["hsl_signal_mode"] = old
    migration = migrate(source, memory_span_minutes=123.5)
    assert migration.config["bot"]["portfolio"]["hsl"]["scope"] == new
    assert any(change["path"] == "live.hsl_signal_mode" and change["old"] == old
               and change["new"] == {"bot.portfolio.hsl.scope": new} for change in migration.changes)
    for side in ("long", "short"):
        assert migration.config["bot"][side]["hsl"]["drawdown_ratio_threshold"] == source["bot"][side]["hsl"]["red_threshold"]
        assert migration.config["bot"][side]["unstuck"]["ema_distance_fraction"] == source["bot"][side]["unstuck"]["ema_dist"]


@pytest.mark.parametrize("alias", ["unstuck_ema_dist", "unstuck.ema_dist", "unstuck.ema_distance_fraction"])
def test_released_override_conversion_never_rewrites_a_new_quantity_name_twice(alias):
    source = released_v8_policy()
    source["optimize"]["fixed_runtime_overrides"] = {f"bot.long.{alias}": 0.125}
    config = migrate(source, memory_span_minutes=123.5).config
    assert config["optimize"]["fixed_runtime_overrides"] == {
        "bot.long.unstuck.ema_distance_fraction": 0.125,
    }


@pytest.mark.parametrize("section", ["bot", "bounds", "coin", "scenario"])
@pytest.mark.parametrize("group,old", [("hsl", "red_threshold"), ("unstuck", "ema_dist")])
def test_retired_quantity_names_are_converter_inputs_only(section, group, old):
    config = get_template_config()
    if section == "bot":
        config["bot"]["long"][group][old] = 0.1
    elif section == "bounds":
        config["optimize"]["bounds"]["long"][group][old] = [0.1, 0.2]
    elif section == "coin":
        config["coin_overrides"] = {"TEST": {"bot": {"long": {group: {old: 0.1}}}}}
    else:
        config["backtest"]["scenarios"] = [{"label": "old", "overrides": {f"bot.long.{group}.{old}": 0.1}}]
    with pytest.raises((ValueError, KeyError), match="unknown|Unknown|invalid|unsupported|inactive|not overridable"):
        prepare_config(config, verbose=False)


def test_retired_pside_scope_is_not_a_current_loading_alias():
    config = get_template_config()
    config["bot"]["portfolio"]["hsl"]["scope"] = "pside"
    with pytest.raises(ValueError, match="position_side"):
        prepare_config(config, verbose=False)


@pytest.mark.parametrize("selector", ["red_threshold", "hsl.red_threshold", "bot.hsl.red_threshold"])
def test_released_portfolio_threshold_selectors_preserve_the_global_gene(selector):
    source = released_v8_policy()
    source["live"]["hsl_signal_mode"] = "unified"
    source["bot"]["hsl"] = deepcopy(source["bot"]["long"]["hsl"])
    source["optimize"]["bounds"]["hsl"] = deepcopy(source["optimize"]["bounds"]["long"]["hsl"])
    for side in ("long", "short"):
        source["optimize"]["bounds"][side].pop("hsl", None)
    source["optimize"]["fixed_params"] = [selector]
    config = migrate(source, memory_span_minutes=123.5).config
    flat = flatten_optimize_bounds(config["optimize"]["bounds"], strategy_kind="trailing_martingale")
    assert set(resolve_bound_selectors(config, config["optimize"]["fixed_params"], flat)) == {"hsl_red_threshold"}


@pytest.mark.parametrize("owner", ["fixed", "scenario"])
def test_released_scope_override_reports_exact_old_value(owner):
    source = released_v8_policy()
    patch = {"live.hsl_signal_mode": "pside"}
    if owner == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = patch
        prefix = "optimize.fixed_runtime_overrides"
    else:
        source["backtest"]["scenarios"] = [{"label": "scope", "overrides": patch}]
        prefix = "backtest.scenarios[0].overrides"
    migration = migrate(source, memory_span_minutes=123.5)
    target = (migration.config["optimize"]["fixed_runtime_overrides"] if owner == "fixed" else
              migration.config["backtest"]["scenarios"][0]["overrides"])
    assert target == {"bot.portfolio.hsl.scope": "position_side"}
    assert any(change["path"] == prefix + ".live.hsl_signal_mode" and change["old"] == "pside"
               and change["new"] == target for change in migration.changes)


@pytest.mark.parametrize("group,old,new", [
    ("hsl", "red_threshold", "drawdown_ratio_threshold"),
    ("unstuck", "ema_dist", "ema_distance_fraction"),
])
@pytest.mark.parametrize("reverse", [False, True])
def test_released_quantity_alias_conflicts_fail_in_both_authoring_orders(group, old, new, reverse):
    source = released_v8_policy()
    values = [(old, 0.1), (new, 0.2)]
    source["bot"]["long"][group] = dict(reversed(values) if reverse else values)
    original = deepcopy(source)
    with pytest.raises(ValueError, match="conflicts"):
        migrate(source, memory_span_minutes=123.5)
    assert source == original


@pytest.mark.parametrize("group,old,new", [
    ("hsl", "red_threshold", "drawdown_ratio_threshold"),
    ("unstuck", "ema_dist", "ema_distance_fraction"),
])
def test_equal_released_quantity_aliases_are_reported_once_without_mutating_input(group, old, new):
    source = released_v8_policy()
    source["bot"]["long"][group][new] = source["bot"]["long"][group][old]
    original = deepcopy(source)
    migration = migrate(source, memory_span_minutes=123.5)
    assert source == original
    assert old not in migration.config["bot"]["long"][group]
    assert migration.config["bot"]["long"][group][new] == original["bot"]["long"][group][old]
    assert any(change["path"] == f"bot.long.{group}.{old}" for change in migration.changes)


@pytest.mark.parametrize("group,old,new", [
    ("hsl", "red_threshold", "drawdown_ratio_threshold"),
    ("unstuck", "ema_dist", "ema_distance_fraction"),
])
def test_released_inline_quantities_replace_file_values_after_independent_conversion(tmp_path, group, old, new):
    source = released_v8_policy()
    document = {"bot": {"long": {group: {old: 0.1}}}}
    file = tmp_path / "coin.json"
    file.write_text(json.dumps(document))
    before_file = file.read_bytes()
    source["coin_overrides"] = {"TEST": {"override_config_path": file.name,
                                          "bot": {"long": {group: {old: 0.2}}}}}
    original = deepcopy(source)
    migration = migrate(source, memory_span_minutes=123.5,
                        base_config_path=str(tmp_path / "input.json"))
    assert source == original
    assert file.read_bytes() == before_file
    block = migration.config["coin_overrides"]["TEST"]["bot"]["long"][group]
    assert block[new] == 0.2
    assert old not in block
    assert any(change["path"] == f"coin_overrides.TEST.bot.long.{group}.{old}"
               and change["old"] == 0.2 for change in migration.changes)


def test_released_position_side_scope_validates_materialized_non_hsl_coin_policy(tmp_path):
    source = released_v8_policy()
    source["live"]["hsl_signal_mode"] = "pside"
    file = tmp_path / "coin.json"
    file.write_text(json.dumps({"bot": {"long": {"unstuck": {"ema_dist": 0.125}}}}))
    source["coin_overrides"] = {"TEST": {"override_config_path": file.name}}
    config = migrate(source, memory_span_minutes=123.5,
                     base_config_path=str(tmp_path / "input.json")).config
    assert config["bot"]["portfolio"]["hsl"]["scope"] == "position_side"
    assert config["coin_overrides"]["TEST"]["bot"]["long"]["unstuck"]["ema_distance_fraction"] == 0.125


def test_smoke_config_checks_recognize_the_public_position_side_scope(tmp_path):
    from live.smoke_report import _smoke_config_check_records

    config = generated_template(get_template_config(), "position_side")
    config["bot"]["long"]["hsl"]["enabled"] = True
    config["live"]["account"]["balance_override"] = 1000.0
    file = tmp_path / "policy.json"
    file.write_text(json.dumps(config))
    report = _smoke_config_check_records([{"config_path": file.name}], config_base_dir=tmp_path)
    assert report["hard_failures"] == 1
    assert report["issues"][0]["code"] == "hsl_balance_override_account_level_replay_unsafe"
    assert report["issues"][0]["hsl_signal_mode"] == "position_side"
