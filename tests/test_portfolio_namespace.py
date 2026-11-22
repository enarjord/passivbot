"""Account-wide policy has one canonical scope, distinct from directional defaults."""

from copy import deepcopy
import json

import pytest
from released_v8_fixtures import without_metric_selection
from jsonschema import Draft202012Validator

from config import prepare_config
from config.authoring import compact_config, resolved_config
from config.field_registry import get_field_registry
from config.hsl import generated_template
from config.migrations.released_v8 import get_released_v8_defaults
from config.migrations.released_v8_strategy import RELEASED_V8_STRATEGY_PATH_RENAMES
from config.json_schema import generate_schema
from config.optimize_bounds import flatten_optimize_bounds
from config.param_paths import resolve_optimizer_key_path
from config.schema import get_template_config
from released_v8_fixtures import released_v8_policy
from tools.clean_config import cleanup_config
from tools.migrate_config_v9 import migrate


PORTFOLIO_PATHS = (
    "strategy_kind", "hedge_mode", "execution.market_orders_allowed",
    "execution.market_order_near_touch_fraction", "warmup.activation_span_fraction",
    "warmup.max_minutes", "risk.loss_realization.loss_gate.enabled",
    "risk.loss_realization.loss_gate.max_realized_drawdown_fraction",
    "risk.loss_realization.memory_ema_span_minutes",
    "risk.hsl.scope", "risk.hsl.enabled", "risk.hsl.drawdown_threshold_fraction",
    "risk.hsl.drawdown_fraction_ema_span_minutes", "risk.hsl.memory_ema_span_minutes",
    "risk.hsl.panic_close_order_type", "risk.hsl.cooldown_after_red_minutes",
    "risk.hsl.scale_budget_with_excess_allowance",
)

DIRECTIONAL_HSL_PATHS = tuple(path for path in PORTFOLIO_PATHS
                              if path.startswith("risk.hsl.") and path != "risk.hsl.scope")
PORTFOLIO_ONLY_PATHS = tuple(path for path in PORTFOLIO_PATHS
                             if path not in DIRECTIONAL_HSL_PATHS)


def read(node, path):
    for part in path.split("."):
        node = node[part]
    return node


def write(node, path, value):
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def expected_released_strategy_bounds(kind, bounds):
    """Use frozen released paths, independent of current native strategy names."""
    names = RELEASED_V8_STRATEGY_PATH_RENAMES[kind]
    expected = {}

    def visit(node, prefix=()):
        for key, value in node.items():
            path = (*prefix, key)
            if isinstance(value, dict):
                visit(value, path)
            else:
                write(expected, ".".join(names.get(path, path)), deepcopy(value))

    visit(bounds)
    return expected


def test_registry_declares_exact_portfolio_scope_and_remaining_global_defaults():
    fields = get_field_registry()
    assert {".".join(field.path[2:]) for field in fields
            if field.path[:2] == ("bot", "portfolio")} == set(PORTFOLIO_PATHS)
    common = {field.path[2:] for field in fields if field.path[:2] == ("bot", "global")}
    assert ("enabled",) not in common
    for side in ("long", "short"):
        directional = {field.path[2:] for field in fields
                       if field.path[:2] == ("bot", side)}
        assert ("enabled",) in directional
        assert common == directional - {("enabled",)}
    assert all(field.scope == "portfolio" for field in fields
               if field.path[:2] == ("bot", "portfolio"))


@pytest.mark.parametrize("local", PORTFOLIO_ONLY_PATHS)
@pytest.mark.parametrize("owner", ["root", "scenario", "fixed"])
def test_retired_global_portfolio_paths_are_rejected_without_mutation(local, owner):
    source = generated_template(get_template_config(), "portfolio")
    path = "bot.global." + local
    value = (0.05 if local == "risk.loss_realization.loss_gate.max_realized_drawdown_fraction"
             else read(source, "bot.portfolio." + local))
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
    source["bot"]["portfolio"]["risk"]["loss_realization"]["loss_gate"] = {
        "enabled": True, "max_realized_drawdown_fraction": 0.05,
    }
    loss_bounds = source["optimize"]["bounds"].setdefault("portfolio", {}).setdefault("risk", {})
    loss_bounds["loss_realization"] = {
        "memory_ema_span_minutes": [123.5, 456.75],
        "loss_gate": {"max_realized_drawdown_fraction": [0.01, 0.1]},
    }
    config = prepare_config(source, verbose=False, live_only=True)
    flat = flatten_optimize_bounds(config["optimize"]["bounds"], strategy_kind="trailing_martingale")
    for key in ("realized_loss_memory_ema_span_minutes", "loss_gate_max_drawdown_fraction"):
        path = resolve_optimizer_key_path(config, key)
        assert path[:2] == ("bot", "portfolio")
        assert read(config, "optimize.bounds." + ".".join(path[1:])) == flat[key]
    retired = deepcopy(source)
    retired["optimize"]["bounds"]["global"] = retired["optimize"]["bounds"].pop("portfolio")
    with pytest.raises(ValueError, match="unknown configuration field"):
        prepare_config(retired, verbose=False, live_only=True)


@pytest.mark.parametrize("owner", ["global", "long", "short"])
def test_directional_hsl_never_authorizes_portfolio_policy(owner):
    source = get_template_config()
    source["bot"]["portfolio"]["risk"]["hsl"]["scope"] = "portfolio"
    directional = deepcopy(source["bot"]["global"]["risk"]["hsl"])
    directional["enabled"] = True
    write(source, f"bot.{owner}.risk.hsl", directional)
    original = deepcopy(source)
    with pytest.raises(ValueError, match="(?:explicit|missing|required)"):
        prepare_config(source, verbose=False, live_only=True)
    assert source == original


def test_portfolio_hsl_rejects_coin_only_budget_scaling():
    source = generated_template(get_template_config(), "portfolio")
    source["bot"]["portfolio"]["risk"]["hsl"]["scale_budget_with_excess_allowance"] = True
    with pytest.raises(ValueError, match="requires coin HSL mode"):
        prepare_config(source, verbose=False, live_only=True)


@pytest.mark.parametrize("local", DIRECTIONAL_HSL_PATHS)
@pytest.mark.parametrize("owner", ["root", "scenario", "fixed"])
def test_directional_hsl_defaults_remain_valid_with_explicit_portfolio_policy(local, owner):
    source = generated_template(get_template_config(), "portfolio")
    path = "bot.global." + local
    value = read(source, "bot.portfolio." + local)
    if owner == "root":
        write(source, path, deepcopy(value))
    elif owner == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = {path: value}
    else:
        source["backtest"]["scenarios"] = [{"label": "directional", "overrides": {path: value}}]
    original = deepcopy(source)
    assert not list(Draft202012Validator(generate_schema()).iter_errors(source))
    assert prepare_config(source, verbose=False, live_only=True)
    assert source == original


@pytest.mark.parametrize("mode", [compact_config, resolved_config, cleanup_config])
def test_exports_retain_explicit_portfolio_policy_values_and_side_independence(mode):
    source = generated_template(get_template_config(), "portfolio")
    source["bot"]["portfolio"]["risk"]["hsl"].update(enabled=True, cooldown_after_red_minutes=17.25)
    source["bot"]["portfolio"]["risk"]["loss_realization"] = {
        "memory_ema_span_minutes": 123.5,
        "loss_gate": {"enabled": False, "max_realized_drawdown_fraction": 0.0},
    }
    source["bot"]["global"]["forager"]["score_hysteresis_margin"] = 0.125
    write(source, "bot.long.risk.hsl.cooldown_after_red_minutes", 35.5)
    original = deepcopy(source)
    expected = prepare_config(source, verbose=False, live_only=True)
    output = mode(source)
    if isinstance(output, tuple):
        output = output[0]
    reloaded = prepare_config(json.loads(json.dumps(output)), verbose=False, live_only=True)
    for path in ("bot.portfolio.risk.hsl", "bot.portfolio.risk.loss_realization",
                 "bot.global.forager", "bot.long.risk.hsl"):
        assert read(reloaded, path) == read(expected, path)
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
    expected = {"bot.portfolio.hedge_mode": True, "bot.portfolio.warmup.activation_span_fraction": 0.625,
                "bot.portfolio.execution.market_order_near_touch_fraction": 0.0125}
    assert patch == expected
    for path in expected:
        assert any(path in str(row["new"]) for row in migration.changes)
    assert source == original
    assert prepare_config(migration.config, verbose=False, live_only=True)


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("scope", ["coin", "position_side", "portfolio"])
def test_released_missing_bounds_use_portfolio_strategy_and_scope_without_changing_defaults(kind, scope):
    source = without_metric_selection(get_released_v8_defaults(strategy_kind=kind))
    source["optimize"]["bounds"] = {}
    source["live"]["hsl_signal_mode"] = {"portfolio": "unified", "position_side": "pside"}.get(scope, scope)
    if scope == "portfolio":
        source["bot"]["hsl"] = deepcopy(source["bot"]["long"]["hsl"])
    original = deepcopy(source)
    output = migrate(source, memory_span_minutes=123.5).config
    defaults = get_released_v8_defaults(strategy_kind=kind)
    for side in ("long", "short"):
        expected = expected_released_strategy_bounds(kind, defaults["optimize"]["bounds"][side]["strategy"][kind])
        assert output["optimize"]["bounds"][side]["strategy"] == {kind: expected}
        assert ("hsl" in output["optimize"]["bounds"][side].get("risk", {})) == (scope != "portfolio")
    assert output["bot"]["portfolio"]["risk"]["hsl"]["scope"] == scope
    assert source == original
