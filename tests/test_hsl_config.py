"""Canonical configuration and explicit migration for the sole HSL engine."""

from argparse import Namespace
from copy import deepcopy
import json

import pytest

from config import prepare_config
from config.schema import get_template_config
from config.hsl import generated_template, require_runtime_support
from config.optimize_bounds import flatten_optimize_bounds
from config_utils import update_config_with_args
from optimization.config_adapter import get_optimization_key_paths
from released_v8_fixtures import released_v8_policy


def source(mode="coin", enabled=True):
    cfg = generated_template(get_template_config(), mode)
    block = cfg["bot"]["global"]["hsl"] if mode == "portfolio" else cfg["bot"]["long"]["hsl"]
    block["enabled"] = enabled
    return cfg


def as_legacy(cfg, version="v8.6.0"):
    """Use frozen v8 authoring with the requested HSL scope and enablement."""
    mode = cfg["bot"]["global"]["hsl"]["scope"]
    enabled = {side: cfg["bot"][side]["hsl"]["enabled"] for side in ("long", "short")}
    portfolio_enabled = cfg["bot"]["global"]["hsl"].get("enabled", False)
    legacy = released_v8_policy()
    legacy["config_version"] = version
    legacy["live"]["hsl_signal_mode"] = {
        "portfolio": "unified", "position_side": "pside"
    }.get(mode, mode)
    for side in ("long", "short"):
        policy = legacy["bot"][side]["hsl"]
        policy["enabled"] = enabled[side]
        policy["restart_after_red_policy"] = "always" if enabled[side] else None
    if mode == "portfolio":
        policy = deepcopy(legacy["bot"]["long"]["hsl"])
        policy["enabled"] = portfolio_enabled
        policy["restart_after_red_policy"] = "always" if portfolio_enabled else None
        legacy["bot"]["hsl"] = policy
        # The authored unified search has one shared HSL domain.
        bounds = legacy["optimize"]["bounds"]
        bounds["hsl"] = bounds["long"].pop("hsl")
        bounds["short"].pop("hsl", None)
    cfg.clear()
    cfg.update(legacy)
    return cfg


def prepared(cfg):
    return prepare_config(cfg, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
def test_explicit_hsl_config_roundtrips_without_legacy_fields(mode):
    cfg = source(mode)
    result = prepared(cfg)
    result2 = prepared(result)
    assert result2["bot"] == result["bot"]
    assert "hsl_engine" not in result["live"]
    block = result["bot"]["global"]["hsl"] if mode == "portfolio" else result["bot"]["long"]["hsl"]
    assert "restart_after_red_policy" not in block
    assert block["cooldown_minutes_after_red"] >= 0.0
    assert not (
        {"tier_ratios", "orange_tier_mode", "no_restart_drawdown_threshold"}
        & block.keys()
    )


def test_missing_selector_uses_sole_engine_without_authorizing_restart():
    cfg = get_template_config()
    cfg["live"].pop("hsl_engine", None)
    result = prepared(cfg)
    assert "hsl_engine" not in result["live"]
    assert "tier_ratios" not in result["bot"]["long"]["hsl"]


def test_explicit_legacy_selector_is_rejected_before_runtime():
    cfg = source()
    cfg["live"]["hsl_engine"] = "legacy"
    with pytest.raises(ValueError, match="hsl_engine is removed in v9.*migrate-config-v9"):
        prepared(cfg)


@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
@pytest.mark.parametrize("policy", [None, "always", "never", "threshold", [], {}])
def test_current_hsl_rejects_every_authored_restart_selector(mode, policy):
    cfg = source(mode)
    block = cfg["bot"]["global"]["hsl"] if mode == "portfolio" else cfg["bot"]["long"]["hsl"]
    block["restart_after_red_policy"] = policy
    with pytest.raises(ValueError, match="restart_after_red_policy"):
        prepared(cfg)


@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
def test_cli_enablement_preserves_finite_cooldown_without_restart_selector(mode):
    cfg = prepared(source(mode, enabled=False))
    path = "bot.global.hsl" if mode == "portfolio" else "bot.long.hsl"
    update_config_with_args(cfg, Namespace(**{f"{path}.enabled": True}), verbose=False)
    result = prepared(cfg)
    block = result["bot"]["global"]["hsl"] if mode == "portfolio" else result["bot"]["long"]["hsl"]
    assert block["enabled"]
    assert block["cooldown_minutes_after_red"] >= 0.0
    assert "restart_after_red_policy" not in block
    with pytest.raises(ValueError, match="removed|unknown"):
        update_config_with_args(cfg, Namespace(**{f"{path}.restart_after_red_policy": "never"}), verbose=False)


def test_portfolio_missing_portfolio_block_is_not_hydrated_even_with_equal_sides():
    cfg = source("portfolio", enabled=False)
    cfg["bot"]["global"]["hsl"] = {"scope": "portfolio"}
    cfg["bot"]["short"]["hsl"] = deepcopy(cfg["bot"]["long"]["hsl"])
    with pytest.raises(ValueError, match="bot.global.hsl missing required fields"):
        prepared(cfg)


def test_cli_mode_change_rechecks_portfolio_presence():
    cfg = prepared(source("position_side"))
    update_config_with_args(
        cfg, Namespace(**{"bot.global.hsl.scope": "portfolio"}), verbose=False
    )
    with pytest.raises(ValueError, match="explicit bot.global.hsl"):
        prepared(cfg)


@pytest.mark.parametrize("span", [0, -1, 0.99, 1e30, float("inf"), float("nan"), True])
def test_slow_memory_rejects_invalid_or_nondecaying_spans(span):
    cfg = source()
    cfg["bot"]["long"]["hsl"]["memory_ema_span_minutes"] = span
    with pytest.raises(ValueError, match="memory_ema_span_minutes"):
        prepared(cfg)


@pytest.mark.parametrize("span", [1.0, 1.5, 43200.0, 200000.0])
def test_independent_slow_and_fast_spans_preserve_fractional_minutes(span):
    cfg = source()
    cfg["bot"]["long"]["hsl"]["memory_ema_span_minutes"] = span
    cfg["bot"]["long"]["hsl"]["drawdown_ratio_ema_span_minutes"] = 2.5
    result = prepared(cfg)
    assert result["bot"]["long"]["hsl"]["memory_ema_span_minutes"] == span
    assert result["bot"]["long"]["hsl"]["drawdown_ratio_ema_span_minutes"] == 2.5


@pytest.mark.parametrize("cap", [0.0, -1.0, 0.99, 129600.01, float("inf"), float("nan"), True])
def test_startup_coverage_cap_is_bounded_independently_of_memory_age(cap):
    cfg = source()
    cfg["live"]["startup"]["risk_bootstrap_max_lookback_minutes"] = cap
    with pytest.raises(ValueError, match="risk_bootstrap_max_lookback_minutes"):
        prepared(cfg)


@pytest.mark.parametrize(
    "policy", ["panic", "normal", "manual", "tp_only", "graceful_stop"]
)
def test_removed_intervention_policy_is_rejected_at_v9_boundary(policy):
    cfg = source()
    cfg["live"]["hsl_position_during_cooldown_policy"] = policy
    with pytest.raises(ValueError, match="hsl_position_during_cooldown_policy.*removed in v9"):
        prepared(cfg)


@pytest.mark.parametrize("section", ["bounds", "fixed_runtime_overrides"])
@pytest.mark.parametrize(
    "field", ["no_restart_drawdown_threshold", "orange_tier_mode", "tier_ratios.yellow"]
)
def test_removed_optimizer_paths_are_rejected_before_pruning(section, field):
    cfg = source()
    key = (
        f"bot.long.hsl.{field}"
        if section == "fixed_runtime_overrides"
        else f"long_hsl_{field}"
    )
    cfg["optimize"][section][key] = [0, 1]
    with pytest.raises(ValueError, match="removed.*(v9|HSL)|unknown configuration field"):
        prepared(cfg)


@pytest.mark.parametrize("section", ["scoring", "limits"])
@pytest.mark.parametrize("tier", ["yellow", "orange"])
def test_removed_metric_is_rejected_in_keys_and_values(section, tier):
    cfg = source()
    cfg["optimize"][section] = [{"metric": f"hard_stop_time_in_{tier}_pct", **({"goal": "min"} if section == "scoring" else {"penalize_if": "greater_than", "value": 0.5})}]
    with pytest.raises(ValueError, match="removed for HSL"):
        prepared(cfg)


def test_portfolio_optimizer_paths_target_portfolio_only():
    cfg = prepared(source("portfolio"))
    keys = dict(get_optimization_key_paths(cfg))
    assert keys["hsl_red_threshold"] == ("bot", "global", "hsl", "drawdown_ratio_threshold")
    assert keys["hsl_drawdown_ratio_ema_span_minutes"] == ("bot", "global", "hsl", "drawdown_ratio_ema_span_minutes")
    assert not any(k.startswith(("long_hsl_", "short_hsl_")) for k in keys)
    assert "hsl_red_threshold" in flatten_optimize_bounds(
        cfg["optimize"]["bounds"], strategy_kind=cfg['bot']['global']['strategy_kind']
    )


@pytest.mark.parametrize("section", ["bounds", "fixed_runtime_overrides"])
def test_portfolio_rejects_inactive_side_optimizer_paths(section):
    cfg = source("portfolio")
    if section == "bounds":
        cfg["optimize"][section]["long"]["hsl"] = {"drawdown_ratio_threshold": [0.1, 0.2]}
    else:
        cfg["optimize"][section]["bot.short.hsl.drawdown_ratio_threshold"] = 0.2
    with pytest.raises(ValueError, match="inactive in portfolio"):
        prepared(cfg)


def test_enabling_coin_override_inherits_finite_policy():
    from config.overrides import parse_overrides

    cfg = source(enabled=False)
    cfg["coin_overrides"] = {"TEST": {"bot": {"long": {"hsl": {"enabled": True}}}}}
    result = parse_overrides(prepared(cfg), verbose=False)
    assert result["coin_overrides"]["TEST"]["bot"]["long"]["hsl"]["enabled"]
    assert "restart_after_red_policy" not in result["bot"]["long"]["hsl"]


@pytest.mark.parametrize("mode", ["position_side", "portfolio"])
def test_inactive_coin_hsl_override_is_not_silently_ignored(mode):
    cfg = source(mode)
    cfg["coin_overrides"] = {"TEST": {"bot": {"long": {"hsl": {"drawdown_ratio_threshold": 0.2}}}}}
    with pytest.raises(ValueError, match="inactive in portfolio|overridable only.*scope"):
        prepared(cfg)


@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
def test_unimplemented_runtime_mode_is_explicitly_rejected(mode):
    with pytest.raises(ValueError, match="runtime integration is not available"):
        require_runtime_support(prepared(source(mode)))
    require_runtime_support(prepared(get_template_config()), supported_modes=("coin",))


@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
def test_live_constructor_accepts_hsl_before_offline_credential_sentinel(
    monkeypatch, mode
):
    import passivbot

    class OfflineBoundary(Exception):
        pass

    def stop_before_credentials(*args):
        raise OfflineBoundary()

    monkeypatch.setattr(passivbot, "load_user_info", stop_before_credentials)
    with pytest.raises(OfflineBoundary):
        passivbot.Passivbot(prepared(source(mode)))


def test_live_constructor_rejects_unknown_engine_before_credentials(monkeypatch):
    import passivbot

    def forbidden(*args):
        pytest.fail("unknown engine reached credential lookup")

    monkeypatch.setattr(passivbot, "load_user_info", forbidden)
    cfg = prepared(source())
    cfg["live"]["hsl_engine"] = "unknown"
    with pytest.raises(ValueError, match="obsolete"):
        passivbot.Passivbot(cfg)


@pytest.mark.parametrize(
    "path",
    ["bot.long.hsl.drawdown_ratio_threshold", "bot.short.hsl.no_restart_drawdown_threshold"],
)
def test_cli_cannot_target_inactive_or_removed_hsl_settings(path):
    cfg = prepared(source("portfolio"))
    with pytest.raises(ValueError, match="inactive|removed"):
        update_config_with_args(cfg, Namespace(**{path: 0.3}), verbose=False)


def test_hsl_coin_override_parsing_keeps_explicit_threshold():
    from config.overrides import parse_overrides

    cfg = source()
    cfg["coin_overrides"] = {
        "TEST": {"bot": {"long": {"hsl": {"drawdown_ratio_threshold": 0.12}}}}
    }
    result = parse_overrides(prepared(cfg), verbose=False)
    assert result["coin_overrides"]["TEST"]["bot"]["long"]["hsl"] == {
        "drawdown_ratio_threshold": 0.12
    }


@pytest.mark.parametrize("change", ["enable", "mode", "removed", "inactive"])
def test_scenario_rechecks_effective_hsl_contract(change):
    from suite_runner import SuiteScenario, apply_scenario

    cfg = source("portfolio" if change == "inactive" else "coin", enabled=False)
    cfg = prepared(cfg)
    overrides = {
        "enable": {"bot.long.hsl.enabled": True},
        "mode": {"bot.global.hsl.scope": "portfolio"},
        "removed": {"bot.long.hsl.no_restart_drawdown_threshold": 0.3},
        "inactive": {"bot.long.hsl.drawdown_ratio_threshold": 0.3},
    }[change]
    scenario = SuiteScenario(
        "test", None, None, ["TEST"], [], exchanges=["binance"], overrides=overrides
    )
    if change == "enable":
        result, *_ = apply_scenario(cfg, scenario, ["TEST"], [], ["binance"], {"TEST"}, quiet=True)
        assert result["bot"]["long"]["hsl"]["enabled"]
        assert "restart_after_red_policy" not in result["bot"]["long"]["hsl"]
    else:
        with pytest.raises(ValueError, match="explicit|removed|inactive"):
            apply_scenario(cfg, scenario, ["TEST"], [], ["binance"], {"TEST"}, quiet=True)


def test_optimizer_finalization_rechecks_fixed_enablement():
    from optimization.warmup import _finalize_optimizer_vector_config

    cfg = source(enabled=False)
    cfg["optimize"]["fixed_runtime_overrides"] = {"bot.long.hsl.enabled": True}
    cfg = prepared(cfg)
    _finalize_optimizer_vector_config(cfg)
    assert cfg["bot"]["long"]["hsl"]["enabled"]
    assert "restart_after_red_policy" not in cfg["bot"]["long"]["hsl"]


def test_legacy_saved_fitness_is_incompatible_with_sole_engine():
    from optimization.evaluation_contract import CONTRACT_KEY, build_evaluation_contract
    from optimize import _resume_config_mismatches

    cfg = prepared(source(enabled=False))
    old = {**deepcopy(cfg), CONTRACT_KEY: build_evaluation_contract(cfg)}
    old[CONTRACT_KEY]["version"] = 1
    assert any(
        "unsupported or malformed snapshot" in item
        for item in _resume_config_mismatches(old, cfg)
    )


@pytest.mark.parametrize(
    "field,bounds",
    [
        ("drawdown_ratio_threshold", [0, 0.2]),
        ("drawdown_ratio_ema_span_minutes", [0.5, 2]),
        ("cooldown_minutes_after_red", [-1, 2]),
    ],
)
@pytest.mark.parametrize("mode", ["coin", "portfolio"])
def test_hsl_numeric_bounds_rejected_before_candidate_generation(field, bounds, mode):
    cfg = source(mode)
    target = (
        cfg["optimize"]["bounds"]["global"]
        if mode == "portfolio"
        else cfg["optimize"]["bounds"]["long"]
    )
    target["hsl"][field] = bounds
    with pytest.raises(ValueError, match="optimize.bounds"):
        prepared(cfg)


def test_engine_cannot_be_changed_by_optimizer_override():
    cfg = get_template_config()
    cfg["optimize"]["fixed_runtime_overrides"]["live.hsl_engine"] = "hsl"
    with pytest.raises(ValueError, match="removed.*v9|startup selector"):
        prepared(cfg)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
async def test_live_entrypoint_accepts_hsl_without_external_setup(monkeypatch, mode):
    import passivbot

    cfg = source(mode)
    cfg["live"]["account"]["user"] = "offline_test"
    monkeypatch.setattr(passivbot.sys, "argv", ["passivbot"])
    monkeypatch.setattr(passivbot, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(
        passivbot, "load_input_config", lambda *args: (cfg, "", deepcopy(cfg))
    )

    class OfflineBoundary(Exception):
        pass

    def stop_after_config(config, **kwargs):
        assert "hsl_engine" not in config["live"]
        assert config["bot"]["global"]["hsl"]["scope"] == mode
        raise OfflineBoundary()

    def forbidden(*args, **kwargs):
        pytest.fail("offline configuration test reached external setup")

    for name in (
        "load_user_info",
        "load_markets",
        "setup_bot",
        "configure_custom_endpoint_loader",
    ):
        monkeypatch.setattr(passivbot, name, forbidden)
    monkeypatch.setattr(passivbot, "resolve_live_log_file_settings", stop_after_config)
    with pytest.raises(OfflineBoundary):
        await passivbot._run_live({})


@pytest.mark.parametrize(
    "field",
    [
        "hsl_unavailable_grace_seconds",
        "hsl_accept_incomplete_history",
        "risk_input_max_attempts",
    ],
)
@pytest.mark.parametrize("section", ["bounds", "fixed_runtime_overrides"])
def test_retired_recovery_controls_cannot_be_optimizer_parameters(field, section):
    cfg = source()
    cfg["optimize"][section][f"live.{field}"] = [0, 1] if section == "bounds" else 1
    with pytest.raises(ValueError, match="removed.*v9|removed HSL recovery control"):
        prepared(cfg)


@pytest.mark.parametrize("mode", ["position_side", "portfolio"])
def test_aggregate_balance_override_constructor_rejects_before_credentials(
    mode, monkeypatch
):
    import passivbot

    config = prepared(source(mode))
    config["live"]["account"]["balance_override"] = 1000.0
    monkeypatch.setattr(
        passivbot,
        "load_user_info",
        lambda *args: pytest.fail("unsupported override reached credentials"),
    )
    with pytest.raises(
        ValueError, match="does not support live.account.balance_override"
    ):
        passivbot.Passivbot(config)


@pytest.mark.parametrize(
    "version",
    [None, "v8.0.0", "v8.1.0", "v8.2.0", "v8.3.0", "v8.4.0", "v8.5.0", "8.5.0"],
)
@pytest.mark.parametrize("policy", ["always", "never"])
@pytest.mark.parametrize("location", ["base", "coin", "fixed", "scenario"])
def test_old_selectorless_hsl_requires_explicit_migration(version, policy, location):
    cfg = source(enabled=location == "base")
    as_legacy(cfg, version)
    cfg["optimize"]["fixed_runtime_overrides"] = {}
    for side in ("long", "short"):
        cfg["bot"][side]["hsl"]["restart_after_red_policy"] = None
    if location == "base":
        cfg["bot"]["long"]["hsl"]["restart_after_red_policy"] = policy
    elif location == "coin":
        cfg["coin_overrides"] = {
            "TEST": {
                "bot": {
                    "long": {
                        "hsl": {"enabled": True, "restart_after_red_policy": policy}
                    }
                }
            }
        }
    elif location == "fixed":
        cfg["optimize"]["fixed_runtime_overrides"] = {
            "bot.long.hsl.enabled": True,
            "bot.long.hsl.restart_after_red_policy": policy,
        }
    elif location == "scenario":
        cfg["backtest"]["scenarios"] = [
            {
                "label": "hsl",
                "overrides": {
                    "bot.long.hsl.enabled": True,
                    "bot.long.hsl.restart_after_red_policy": policy,
                },
            }
        ]
    with pytest.raises(ValueError, match="migrate-config-v9"):
        prepared(cfg)
    from tools.migrate_config_v9 import migrate

    if version is None:
        with pytest.raises(ValueError, match="versioned v8"):
            migrate(cfg, memory_span_minutes=43200.0, acknowledge_never_removal=True)
        return
    if policy == "never":
        with pytest.raises(ValueError, match="acknowledge-never-removal"):
            migrate(cfg, memory_span_minutes=43200.0)
    converted = migrate(cfg, memory_span_minutes=43200.0, acknowledge_never_removal=True).config
    assert prepared(converted)["config_version"] == "v9.0.0"


def test_old_disabled_hsl_requires_explicit_migration_too():
    cfg = source(enabled=False)
    as_legacy(cfg, "v8.5.0")
    cfg["optimize"]["fixed_runtime_overrides"] = {}
    for side in ("long", "short"):
        cfg["bot"][side]["hsl"]["restart_after_red_policy"] = "threshold"
    from tools.migrate_config_v9 import migrate
    with pytest.raises(ValueError, match="migrate-config-v9"):
        prepared(cfg)
    with pytest.raises(ValueError, match="no v9 policy equivalent"):
        migrate(cfg, memory_span_minutes=43200.0)
    for side in ("long", "short"):
        del cfg["bot"][side]["hsl"]["restart_after_red_policy"]
    result = migrate(cfg, memory_span_minutes=43200.0).config
    assert prepared(result)["config_version"] == "v9.0.0"
    assert "restart_after_red_policy" not in result["bot"]["long"]["hsl"]


@pytest.mark.parametrize("location", ["root", "scenario"])
@pytest.mark.parametrize("policy", ["always", "never"])
def test_old_root_file_backed_hsl_requires_migration_before_hydration(
    tmp_path, location, policy
):
    from tools.migrate_config_v9 import migrate

    cfg = source(enabled=False)
    as_legacy(cfg, "v8.5.0")
    cfg["optimize"]["fixed_runtime_overrides"] = {}
    for side in ("long", "short"):
        cfg["bot"][side]["hsl"]["restart_after_red_policy"] = None
    (tmp_path / "coin.json").write_text(
        json.dumps(
            {
                "bot": {
                    "long": {
                        "hsl": {"enabled": True, "restart_after_red_policy": policy}
                    }
                }
            }
        )
    )
    patch = {"TEST": {"override_config_path": "coin.json"}}
    if location == "root":
        cfg["coin_overrides"] = patch
    else:
        cfg["backtest"]["scenarios"] = [
            {"label": "file", "overrides": {"coin_overrides": patch}}
        ]
    with pytest.raises(ValueError, match="migrate-config-v9"):
        prepare_config(
            cfg, base_config_path=str(tmp_path / "config.json"), verbose=False
        )
    result = migrate(cfg, base_config_path=str(tmp_path / "config.json"), memory_span_minutes=43200.0, acknowledge_never_removal=True).config
    assert result["config_version"] == "v9.0.0"
    assert not any(key.startswith("_hsl") for key in result)
    assert prepare_config(result, verbose=False)["config_version"] == "v9.0.0"


@pytest.mark.parametrize("policy", ["always", "never"])
@pytest.mark.parametrize("location", ["root", "scenario"])
def test_old_legacy_coin_flag_file_requires_migration(tmp_path, policy, location):
    from config.overrides import parse_overrides
    from tools.migrate_config_v9 import migrate

    cfg = source(enabled=False)
    as_legacy(cfg, "v8.5.0")
    cfg["optimize"]["fixed_runtime_overrides"] = {}
    for side in ("long", "short"):
        cfg["bot"][side]["hsl"]["restart_after_red_policy"] = None
    (tmp_path / "coin.json").write_text(
        json.dumps(
            {
                "bot": {
                    "long": {
                        "hsl": {
                            "enabled": True,
                            "restart_after_red_policy": policy,
                        }
                    }
                }
            }
        )
    )
    flags = {"live": {"coin_flags": {"TEST": "-lc coin.json"}}}
    if location == "root":
        cfg["live"].update(flags["live"])
    else:
        cfg["backtest"]["scenarios"] = [{"label": "legacy", "overrides": flags}]
    base_path = str(tmp_path / "base.json")
    with pytest.raises(ValueError, match="migrate-config-v9"):
        prepare_config(cfg, base_config_path=base_path, verbose=False)
    if location == "root":
        converted = migrate(cfg, base_config_path=base_path, memory_span_minutes=43200.0, acknowledge_never_removal=True).config
        assert prepared(converted)["config_version"] == "v9.0.0"
        block = parse_overrides(converted, verbose=False)["coin_overrides"]["TEST"]["bot"]["long"]["hsl"]
        assert block["enabled"]
        assert "restart_after_red_policy" not in block


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["backtest", "optimize"])
@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("policy", ["always", "never"])
async def test_cli_external_suite_old_hsl_rejected_before_preparation(
    tmp_path,
    monkeypatch,
    command,
    wrapped,
    policy,
):
    import importlib

    module = importlib.import_module(command)
    cfg = source(enabled=False)
    as_legacy(cfg, "v8.5.0")
    cfg["optimize"]["fixed_runtime_overrides"] = {}
    for side in ("long", "short"):
        cfg["bot"][side]["hsl"]["restart_after_red_policy"] = None
    path = tmp_path / "base.json"
    path.write_text(json.dumps({"config": cfg} if wrapped else cfg))
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "label": "external",
                        "overrides": {
                            "bot.long.hsl.enabled": True,
                            "bot.long.hsl.restart_after_red_policy": policy,
                        },
                    }
                ]
            }
        )
    )

    def unexpected_prepare(*args, **kwargs):
        pytest.fail(
            "old external HSL must be checked before preparation erases schema origin"
        )

    monkeypatch.setattr(module, "prepare_config", unexpected_prepare)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [f"passivbot {command}", str(path), "--suite-config", str(suite_path)],
    )
    with pytest.raises(ValueError, match="migrate-config-v9"):
        await module.main()


@pytest.mark.parametrize("version", ["v8.5.0", "v9.0.0"])
@pytest.mark.parametrize("patch_kind", ["inline", "file", "legacy_file", "non_hsl"])
def test_external_suite_loader_gates_old_base_and_resolves_current_files(tmp_path, version, patch_kind):
    from suite_runner import load_suite_override_config

    cfg = source(enabled=False)
    if version != "v9.0.0":
        as_legacy(cfg, version)
    hsl = {"bot": {"long": {"hsl": {"enabled": True, "cooldown_after_red_minutes": 2.5}}}}
    (tmp_path / "coin.json").write_text(json.dumps(hsl))
    patches = {
        "inline": hsl,
        "file": {"coin_overrides": {"TEST": {"override_config_path": "coin.json"}}},
        "legacy_file": {"live.coin_flags.TEST": "-lc coin.json"},
        "non_hsl": {'bot.global.universe.minimum_coin_age_days': 30},
    }
    scenarios = [{"label": "external", "overrides": patches[patch_kind]}]
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"backtest": {"scenarios": scenarios}}))
    kwargs = {"source_config": cfg, "base_config_path": str(tmp_path / "base.json")}
    if version != "v9.0.0":
        with pytest.raises(ValueError, match="migrate-config-v9"):
            load_suite_override_config(path, **kwargs)
    elif patch_kind == "legacy_file":
        with pytest.raises(ValueError, match="unknown configuration field"):
            load_suite_override_config(path, **kwargs)
    else:
        loaded = load_suite_override_config(path, **kwargs)
        assert loaded["scenarios"] == scenarios


@pytest.mark.parametrize(
    "entrypoint", ["ensure_suite_config", "run_backtest_suite_sync"]
)
def test_external_suite_file_helpers_gate_before_hydration(
    tmp_path, monkeypatch, entrypoint
):
    import optimize_suite
    import suite_runner

    cfg = source(enabled=False)
    as_legacy(cfg, "v8.5.0")
    cfg["optimize"]["fixed_runtime_overrides"] = {}
    for side in ("long", "short"):
        cfg["bot"][side]["hsl"]["restart_after_red_policy"] = None
    base = tmp_path / "base.json"
    base.write_text(json.dumps(cfg))
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "label": "external",
                        "overrides": {
                            "bot.long.hsl.enabled": True,
                            "bot.long.hsl.restart_after_red_policy": "always",
                        },
                    }
                ]
            }
        )
    )

    def unexpected_prepare(*args, **kwargs):
        pytest.fail("suite helper must gate old HSL before hydration")

    module = optimize_suite if entrypoint == "ensure_suite_config" else suite_runner
    monkeypatch.setattr(module, "prepare_config", unexpected_prepare)
    with pytest.raises(ValueError, match="migrate-config-v9"):
        if entrypoint == "ensure_suite_config":
            module.ensure_suite_config(base, suite)
        else:
            module.run_backtest_suite_sync(base, suite_config_path=suite)


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["backtest", "optimize"])
async def test_cli_current_external_suite_reaches_preparation(
    tmp_path, monkeypatch, command
):
    import importlib

    module = importlib.import_module(command)
    cfg = source(enabled=False)
    base = tmp_path / "base.json"
    base.write_text(json.dumps(cfg))
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "scenarios": [
                    {
                        "label": "external",
                        "overrides": {
                            "bot.long.hsl.enabled": True,
                            "bot.long.hsl.cooldown_after_red_minutes": 2.5,
                        },
                    }
                ]
            }
        )
    )

    class PreparedOnly(Exception):
        pass

    def capture_prepare(source, **kwargs):
        assert source["config_version"] == "v9.0.0"
        raise PreparedOnly

    monkeypatch.setattr(module, "prepare_config", capture_prepare)
    monkeypatch.setattr(
        module.sys,
        "argv",
        [f"passivbot {command}", str(base), "--suite-config", str(suite)],
    )
    with pytest.raises(PreparedOnly):
        await module.main()


@pytest.mark.parametrize("mode", ["coin", "position_side", "portfolio"])
@pytest.mark.parametrize("selector", [None, "legacy", "revised"])
def test_disabled_legacy_hsl_requires_migration_and_removes_restart_selector(mode, selector):
    from tools.migrate_config_v9 import migrate

    cfg = as_legacy(source(mode, enabled=False), "v8.5.0")
    if selector is not None:
        cfg["live"]["hsl_engine"] = selector
    original = deepcopy(cfg)
    with pytest.raises(ValueError, match="migrate-config-v9"):
        prepared(cfg)
    result = migrate(cfg, memory_span_minutes=43200.0).config
    assert cfg == original
    assert "hsl_engine" not in result["live"]
    path = "bot.global.hsl" if mode == "portfolio" else "bot.long.hsl"
    block = result["bot"]["global"]["hsl"] if mode == "portfolio" else result["bot"]["long"]["hsl"]
    assert "restart_after_red_policy" not in block
    assert block["enabled"] is False
    result = prepared(result)
    update_config_with_args(result, Namespace(**{f"{path}.enabled": True}))
    result = prepared(result)
    block = result["bot"]["global"]["hsl"] if mode == "portfolio" else result["bot"]["long"]["hsl"]
    assert block["enabled"] and block["cooldown_minutes_after_red"] >= 0.0


@pytest.mark.parametrize("enabled_before,enabled_after", [(False, True), (True, False)])
@pytest.mark.parametrize("wrapped", [False, True])
def test_legacy_hsl_guard_uses_cli_modified_config(enabled_before, enabled_after, wrapped):
    cfg = source(enabled=enabled_before)
    as_legacy(cfg, "v8.5.0")
    cfg["live"]["hsl_engine"] = "legacy"
    if wrapped:
        cfg = {"config": cfg}
    update_config_with_args(cfg, Namespace(**{"bot.long.hsl.enabled": enabled_after}))
    with pytest.raises(ValueError, match="migrate-config-v9"):
        prepared(cfg)


@pytest.mark.parametrize("location", ["coin", "fixed", "scenario", "file"])
def test_disabled_legacy_restart_choices_cannot_survive_in_patches(location, tmp_path):
    from config.overrides import parse_overrides
    cfg = source(enabled=False)
    as_legacy(cfg, "v8.5.0")
    policy = {"bot": {"long": {"hsl": {"enabled": False, "restart_after_red_policy": "never"}}}}
    if location == "coin":
        cfg["coin_overrides"] = {"TEST": policy}
    elif location == "file":
        (tmp_path / "coin.json").write_text(json.dumps(policy))
        cfg["coin_overrides"] = {"TEST": {"override_config_path": "coin.json"}}
    elif location == "fixed":
        cfg["optimize"]["fixed_runtime_overrides"] = {"bot.long.hsl.restart_after_red_policy": "never"}
    else:
        cfg["backtest"]["scenarios"] = [{"label": "off", "overrides": {"bot.long.hsl.restart_after_red_policy": "never"}}]
    from tools.migrate_config_v9 import migrate
    with pytest.raises(ValueError, match="migrate-config-v9"):
        prepare_config(cfg, verbose=False, base_config_path=str(tmp_path / "base.json"))
    result = migrate(cfg, memory_span_minutes=43200.0, acknowledge_never_removal=True,
                     base_config_path=str(tmp_path / "base.json")).config
    if location in {"coin", "file"}:
        result = parse_overrides(result, verbose=False)
        assert result["coin_overrides"]["TEST"]["bot"]["long"]["hsl"].get("restart_after_red_policy") is None
    elif location == "fixed":
        assert result["optimize"]["fixed_runtime_overrides"].get("bot.long.hsl.restart_after_red_policy") is None
    else:
        assert result["backtest"]["scenarios"][0]["overrides"].get("bot.long.hsl.restart_after_red_policy") is None
