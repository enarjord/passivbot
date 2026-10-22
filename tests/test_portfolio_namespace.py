"""Account-wide policy has one canonical scope, distinct from directional defaults."""

from copy import deepcopy
import json

import pytest
from jsonschema import Draft202012Validator

from config import prepare_config
from config.authoring import compact_config, resolved_config
from config.field_registry import get_field_registry
from config.hsl import generated_template
from config.json_schema import generate_schema
from config.optimize_bounds import flatten_optimize_bounds
from config.param_paths import resolve_optimizer_key_path
from config.schema import get_template_config
from released_v8_fixtures import released_v8_policy
from tools.clean_config import cleanup_config
from tools.migrate_config_v9 import migrate


PORTFOLIO_PATHS = (
    "strategy_kind", "hedge_mode", "execution.market_orders_allowed",
    "execution.market_order_near_touch_fraction", "warmup.span_multiplier",
    "warmup.max_minutes", "loss_gate.max_drawdown_ratio",
    "loss_gate.memory_ema_span_minutes", "unstuck.memory_ema_span_minutes",
    "hsl.scope", "hsl.enabled", "hsl.drawdown_ratio_threshold",
    "hsl.drawdown_ratio_ema_span_minutes", "hsl.memory_ema_span_minutes",
    "hsl.panic_close_order_type", "hsl.cooldown_after_red_minutes",
    "hsl.scale_budget_with_excess_allowance",
)


def read(node, path):
    for part in path.split("."):
        node = node[part]
    return node


def write(node, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def test_registry_declares_exact_portfolio_scope_and_remaining_global_defaults():
    fields = get_field_registry()
    assert {".".join(field.path[2:]) for field in fields
            if field.path[:2] == ("bot", "portfolio")} == set(PORTFOLIO_PATHS)
    assert {".".join(field.path[2:]) for field in fields
            if field.path[:2] == ("bot", "global")} == {
                "universe.approved_coins", "forager.score_hysteresis_margin"}
    assert all(field.scope == "portfolio" for field in fields
               if field.path[:2] == ("bot", "portfolio"))


@pytest.mark.parametrize("local", PORTFOLIO_PATHS)
@pytest.mark.parametrize("owner", ["root", "scenario", "fixed"])
def test_retired_global_portfolio_paths_are_rejected_without_mutation(local, owner):
    source = generated_template(get_template_config(), "portfolio")
    path = "bot.global." + local
    value = read(source, "bot.portfolio." + local)
    if owner == "root":
        write(source, path, deepcopy(value))
    elif owner == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = {path: value}
    else:
        source["backtest"]["scenarios"] = [{"label": "retired", "overrides": {path: value}}]
    original = deepcopy(source)
    assert list(Draft202012Validator(generate_schema()).iter_errors(source))
    with pytest.raises((ValueError, KeyError), match="(?:unknown|Unknown|unsupported)"):
        prepare_config(source, verbose=False, live_only=True)
    assert source == original


@pytest.mark.parametrize("scope", ["coin", "position_side", "portfolio"])
def test_portfolio_bounds_mirror_bot_paths_and_keep_native_optimizer_ids(scope):
    source = generated_template(get_template_config(), scope)
    source["optimize"]["bounds"].setdefault("portfolio", {}).update({
        "unstuck": {"memory_ema_span_minutes": [123.5, 456.75]},
        "loss_gate": {"memory_ema_span_minutes": [234.5, 567.75]},
    })
    config = prepare_config(source, verbose=False, live_only=True)
    flat = flatten_optimize_bounds(config["optimize"]["bounds"], strategy_kind="trailing_martingale")
    for key in ("unstuck_memory_ema_span_minutes", "loss_gate_memory_ema_span_minutes"):
        path = resolve_optimizer_key_path(config, key)
        assert path[:2] == ("bot", "portfolio")
        assert read(config, "optimize.bounds." + ".".join(path[1:])) == flat[key]
    retired = deepcopy(source)
    retired["optimize"]["bounds"]["global"] = retired["optimize"]["bounds"].pop("portfolio")
    with pytest.raises(ValueError, match="unknown configuration field"):
        prepare_config(retired, verbose=False, live_only=True)


def test_directional_hsl_never_authorizes_portfolio_policy():
    source = get_template_config()
    source["bot"]["portfolio"]["hsl"]["scope"] = "portfolio"
    source["bot"]["long"]["hsl"]["enabled"] = True
    with pytest.raises(ValueError, match="(?:explicit|missing|required)"):
        prepare_config(source, verbose=False, live_only=True)
    source = generated_template(get_template_config(), "portfolio")
    source["bot"]["portfolio"]["hsl"]["scale_budget_with_excess_allowance"] = True
    with pytest.raises(ValueError, match="requires coin HSL mode"):
        prepare_config(source, verbose=False, live_only=True)


@pytest.mark.parametrize("mode", [compact_config, resolved_config, cleanup_config])
def test_exports_retain_explicit_portfolio_policy_values_and_side_independence(mode):
    source = generated_template(get_template_config(), "portfolio")
    source["bot"]["portfolio"]["hsl"].update(enabled=True, cooldown_after_red_minutes=17.25)
    source["bot"]["portfolio"]["loss_gate"]["memory_ema_span_minutes"] = 123.5
    source["bot"]["portfolio"]["unstuck"]["memory_ema_span_minutes"] = 234.75
    source["bot"]["global"]["forager"]["score_hysteresis_margin"] = 0.125
    source["bot"]["long"]["hsl"]["cooldown_after_red_minutes"] = 35.5
    original = deepcopy(source)
    output = mode(source)
    if isinstance(output, tuple):
        output = output[0]
    reloaded = prepare_config(json.loads(json.dumps(output)), verbose=False, live_only=True)
    for path in ("bot.portfolio.hsl", "bot.portfolio.loss_gate", "bot.portfolio.unstuck",
                 "bot.global.forager", "bot.long.hsl"):
        assert read(reloaded, path) == read(source, path)
    assert source == original


@pytest.mark.parametrize("owner", ["scenario", "fixed"])
def test_released_v8_selectors_land_directly_at_portfolio_paths_and_are_reported(owner):
    source = released_v8_policy()
    old = {"live.hedge_mode": True, "live.warmup_ratio": 0.625,
           "live.market_order_near_touch_threshold": 0.0125}
    if owner == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = old
    else:
        source["backtest"]["scenarios"] = [{"label": "policy", "overrides": old}]
    original = deepcopy(source)
    migration = migrate(source, memory_span_minutes=123.5)
    patch = (migration.config["optimize"]["fixed_runtime_overrides"] if owner == "fixed" else
             migration.config["backtest"]["scenarios"][0]["overrides"])
    expected = {"bot.portfolio.hedge_mode": True, "bot.portfolio.warmup.span_multiplier": 0.625,
                "bot.portfolio.execution.market_order_near_touch_fraction": 0.0125}
    assert patch == expected
    for path in expected:
        assert any(path in str(row["new"]) for row in migration.changes)
    assert source == original
    assert prepare_config(migration.config, verbose=False, live_only=True)
