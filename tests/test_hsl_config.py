"""Canonical configuration and explicit migration for the opt-in HSL engine."""

from argparse import Namespace
from copy import deepcopy

import pytest

from config import prepare_config
from config.schema import get_template_config
from config.hsl import generated_template, require_runtime_support
from config.optimize_bounds import flatten_optimize_bounds
from config_utils import update_config_with_args
from optimization.config_adapter import get_optimization_key_paths


def source(mode="coin", enabled=True):
    cfg = generated_template(get_template_config(), mode)
    block = cfg["bot"]["hsl"] if mode == "unified" else cfg["bot"]["long"]["hsl"]
    block["enabled"] = enabled
    return cfg


def prepared(cfg):
    return prepare_config(cfg, verbose=False, target="canonical", runtime=None)


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_explicit_hsl_config_roundtrips_without_legacy_fields(mode):
    cfg = source(mode)
    result = prepared(cfg)
    result2 = prepared(result)
    assert result2["bot"] == result["bot"]
    assert "hsl_engine" not in result["live"]
    block = result["bot"]["hsl"] if mode == "unified" else result["bot"]["long"]["hsl"]
    assert block["restart_after_red_policy"] == "always"
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
    with pytest.raises(ValueError, match="legacy HSL has been removed.*migrate-hsl"):
        prepared(cfg)


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
@pytest.mark.parametrize("policy", [None, "threshold", "omit", [], {}])
def test_enabled_hsl_scope_requires_explicit_restart_choice(mode, policy):
    cfg = source(mode)
    block = cfg["bot"]["hsl"] if mode == "unified" else cfg["bot"]["long"]["hsl"]
    if policy == "omit":
        del block["restart_after_red_policy"]
    else:
        block["restart_after_red_policy"] = policy
    with pytest.raises(ValueError, match="restart_after_red_policy"):
        prepared(cfg)


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_disabled_missing_policy_cannot_become_implicit_after_hydration_or_cli(mode):
    cfg = source(mode, enabled=False)
    path = "bot.hsl" if mode == "unified" else "bot.long.hsl"
    block = cfg["bot"]["hsl"] if mode == "unified" else cfg["bot"]["long"]["hsl"]
    del block["restart_after_red_policy"]
    cfg = prepared(cfg)
    update_config_with_args(cfg, Namespace(**{f"{path}.enabled": True}), verbose=False)
    with pytest.raises(ValueError, match="explicit choice"):
        prepared(cfg)
    update_config_with_args(
        cfg, Namespace(**{f"{path}.restart_after_red_policy": "never"}), verbose=False
    )
    assert "hsl_engine" not in prepared(cfg)["live"]


def test_unified_missing_portfolio_block_is_not_hydrated_even_with_equal_sides():
    cfg = source("unified", enabled=False)
    del cfg["bot"]["hsl"]
    cfg["bot"]["short"]["hsl"] = deepcopy(cfg["bot"]["long"]["hsl"])
    with pytest.raises(ValueError, match="explicit bot.hsl.*pside"):
        prepared(cfg)


def test_cli_mode_change_rechecks_portfolio_presence():
    cfg = prepared(source("pside"))
    update_config_with_args(
        cfg, Namespace(**{"live.hsl_signal_mode": "unified"}), verbose=False
    )
    with pytest.raises(ValueError, match="explicit bot.hsl"):
        prepared(cfg)


@pytest.mark.parametrize("days", [0, -1, 0.99, 90.01, float("inf"), float("nan"), True])
def test_enabled_lookback_rejects_outside_one_to_ninety_days(days):
    cfg = source()
    cfg["live"]["pnls_max_lookback_days"] = days
    with pytest.raises(ValueError, match="lookback"):
        prepared(cfg)


@pytest.mark.parametrize("days", [1, 1.5, 30, 90])
def test_lookback_boundaries_and_fractional_ema_are_preserved(days):
    cfg = source()
    cfg["live"]["pnls_max_lookback_days"] = days
    cfg["bot"]["long"]["hsl"]["ema_span_minutes"] = 2.5
    result = prepared(cfg)
    assert result["live"]["pnls_max_lookback_days"] == days
    assert result["bot"]["long"]["hsl"]["ema_span_minutes"] == 2.5


@pytest.mark.parametrize(
    "policy", ["panic", "normal", "manual", "tp_only", "graceful_stop"]
)
def test_removed_intervention_policy_is_stripped_from_hsl_config(policy):
    cfg = source()
    cfg["live"]["hsl_position_during_cooldown_policy"] = policy
    assert "hsl_position_during_cooldown_policy" not in prepared(cfg)["live"]


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
    with pytest.raises(ValueError, match="removed HSL"):
        prepared(cfg)


@pytest.mark.parametrize("section", ["scoring", "limits"])
@pytest.mark.parametrize("tier", ["yellow", "orange"])
def test_removed_metric_is_rejected_in_keys_and_values(section, tier):
    cfg = source()
    cfg["optimize"][section] = [{f"hard_stop_time_in_{tier}_pct": 0.5}]
    with pytest.raises(ValueError, match="removed for HSL"):
        prepared(cfg)


def test_unified_optimizer_paths_target_portfolio_only():
    cfg = prepared(source("unified"))
    keys = dict(get_optimization_key_paths(cfg))
    assert keys["hsl_red_threshold"] == ("bot", "hsl", "red_threshold")
    assert keys["hsl_ema_span_minutes"] == ("bot", "hsl", "ema_span_minutes")
    assert not any(k.startswith(("long_hsl_", "short_hsl_")) for k in keys)
    assert "hsl_red_threshold" in flatten_optimize_bounds(
        cfg["optimize"]["bounds"], strategy_kind=cfg["live"]["strategy_kind"]
    )


@pytest.mark.parametrize("section", ["bounds", "fixed_runtime_overrides"])
def test_unified_rejects_inactive_side_optimizer_paths(section):
    cfg = source("unified")
    if section == "bounds":
        cfg["optimize"][section]["long"]["hsl"] = {"red_threshold": [0.1, 0.2]}
    else:
        cfg["optimize"][section]["bot.short.hsl.restart_after_red_policy"] = "never"
    with pytest.raises(ValueError, match="inactive in unified"):
        prepared(cfg)


def test_enabling_coin_override_requires_restart_choice():
    cfg = source(enabled=False)
    del cfg["bot"]["long"]["hsl"]["restart_after_red_policy"]
    cfg["coin_overrides"] = {"TEST": {"bot": {"long": {"hsl": {"enabled": True}}}}}
    with pytest.raises(ValueError, match="explicit choice"):
        prepared(cfg)


@pytest.mark.parametrize("mode", ["pside", "unified"])
def test_inactive_coin_hsl_override_is_not_silently_ignored(mode):
    cfg = source(mode)
    cfg["coin_overrides"] = {"TEST": {"bot": {"long": {"hsl": {"red_threshold": 0.2}}}}}
    with pytest.raises(ValueError, match="inactive outside coin mode"):
        prepared(cfg)


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_unimplemented_runtime_mode_is_explicitly_rejected(mode):
    with pytest.raises(ValueError, match="runtime integration is not available"):
        require_runtime_support(prepared(source(mode)))
    require_runtime_support(prepared(get_template_config()), supported_modes=("coin",))


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
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
    ["bot.long.hsl.red_threshold", "bot.short.hsl.no_restart_drawdown_threshold"],
)
def test_cli_cannot_target_inactive_or_removed_hsl_settings(path):
    cfg = prepared(source("unified"))
    with pytest.raises(ValueError, match="inactive|removed"):
        update_config_with_args(cfg, Namespace(**{path: 0.3}), verbose=False)


def test_hsl_coin_override_parsing_keeps_explicit_restart():
    from config.overrides import parse_overrides

    cfg = source()
    cfg["coin_overrides"] = {
        "TEST": {"bot": {"long": {"hsl": {"red_threshold": 0.12}}}}
    }
    result = parse_overrides(prepared(cfg), verbose=False)
    assert result["coin_overrides"]["TEST"]["bot"]["long"]["hsl"] == {
        "red_threshold": 0.12
    }


@pytest.mark.parametrize("change", ["enable", "mode", "removed", "inactive"])
def test_scenario_rechecks_effective_hsl_contract(change):
    from suite_runner import SuiteScenario, apply_scenario

    cfg = source("unified" if change == "inactive" else "coin", enabled=False)
    cfg["bot"]["long"]["hsl"].pop("restart_after_red_policy")
    cfg = prepared(cfg)
    overrides = {
        "enable": {"bot.long.hsl.enabled": True},
        "mode": {"live.hsl_signal_mode": "unified"},
        "removed": {"bot.long.hsl.no_restart_drawdown_threshold": 0.3},
        "inactive": {"bot.long.hsl.red_threshold": 0.3},
    }[change]
    scenario = SuiteScenario(
        "test", None, None, ["TEST"], [], exchanges=["binance"], overrides=overrides
    )
    with pytest.raises(ValueError, match="explicit|removed|inactive"):
        apply_scenario(cfg, scenario, ["TEST"], [], ["binance"], {"TEST"}, quiet=True)


def test_optimizer_finalization_rechecks_fixed_enablement():
    from optimization.warmup import _finalize_optimizer_vector_config

    cfg = source(enabled=False)
    del cfg["bot"]["long"]["hsl"]["restart_after_red_policy"]
    cfg["optimize"]["fixed_runtime_overrides"] = {"bot.long.hsl.enabled": True}
    cfg = prepared(cfg)
    with pytest.raises(ValueError, match="explicit choice"):
        _finalize_optimizer_vector_config(cfg)


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
        ("red_threshold", [0, 0.2]),
        ("ema_span_minutes", [0.5, 2]),
        ("cooldown_minutes_after_red", [-1, 2]),
    ],
)
@pytest.mark.parametrize("mode", ["coin", "unified"])
def test_hsl_numeric_bounds_rejected_before_candidate_generation(field, bounds, mode):
    cfg = source(mode)
    target = (
        cfg["optimize"]["bounds"]
        if mode == "unified"
        else cfg["optimize"]["bounds"]["long"]
    )
    target["hsl"][field] = bounds
    with pytest.raises(ValueError, match="optimize.bounds"):
        prepared(cfg)


def test_engine_cannot_be_changed_by_optimizer_override():
    cfg = get_template_config()
    cfg["optimize"]["fixed_runtime_overrides"]["live.hsl_engine"] = "hsl"
    with pytest.raises(ValueError, match="startup selector"):
        prepared(cfg)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
async def test_live_entrypoint_accepts_hsl_without_external_setup(monkeypatch, mode):
    import passivbot

    cfg = source(mode)
    cfg["live"]["user"] = "offline_test"
    monkeypatch.setattr(passivbot.sys, "argv", ["passivbot"])
    monkeypatch.setattr(passivbot, "configure_logging", lambda **kwargs: None)
    monkeypatch.setattr(
        passivbot, "load_input_config", lambda *args: (cfg, "", deepcopy(cfg))
    )

    class OfflineBoundary(Exception):
        pass

    def stop_after_config(config, **kwargs):
        assert "hsl_engine" not in config["live"]
        assert config["live"]["hsl_signal_mode"] == mode
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
    with pytest.raises(ValueError, match="removed HSL recovery control"):
        prepared(cfg)


@pytest.mark.parametrize("mode", ["pside", "unified"])
def test_aggregate_balance_override_constructor_rejects_before_credentials(
    mode, monkeypatch
):
    import passivbot

    config = prepared(source(mode))
    config["live"]["balance_override"] = 1000.0
    monkeypatch.setattr(
        passivbot,
        "load_user_info",
        lambda *args: pytest.fail("unsupported override reached credentials"),
    )
    with pytest.raises(ValueError, match="does not support live.balance_override"):
        passivbot.Passivbot(config)


@pytest.mark.parametrize(
    "version",
    [None, "v8.0.0", "v8.1.0", "v8.2.0", "v8.3.0", "v8.4.0", "v8.5.0", "8.5.0"],
)
@pytest.mark.parametrize("policy", ["always", "never"])
@pytest.mark.parametrize("location", ["base", "coin", "fixed", "scenario", "disabled"])
def test_old_selectorless_hsl_requires_explicit_migration(version, policy, location):
    cfg = source(enabled=location == "base")
    cfg["config_version"] = version
    cfg["optimize"]["fixed_runtime_overrides"] = {}
    for side in ("long", "short"):
        cfg["bot"][side]["hsl"]["restart_after_red_policy"] = None
    if location in {"base", "disabled"}:
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
    with pytest.raises(ValueError, match="migrate-hsl"):
        prepared(cfg)
    from tools.migrate_hsl_config import migrate

    converted = migrate(cfg)
    assert prepared(converted)["config_version"] == "v8.6.0"


def test_old_disabled_hsl_without_restart_authorization_can_upgrade():
    cfg = source(enabled=False)
    cfg["config_version"] = "v8.5.0"
    cfg["optimize"]["fixed_runtime_overrides"] = {}
    for side in ("long", "short"):
        cfg["bot"][side]["hsl"]["restart_after_red_policy"] = "threshold"
    result = prepared(cfg)
    assert result["config_version"] == "v8.6.0"
    assert result["bot"]["long"]["hsl"]["restart_after_red_policy"] is None
