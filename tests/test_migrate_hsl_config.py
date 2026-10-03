"""Config migration is explicit, offline and preserves source files and policy choices."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from config.schema import (
    CONFIG_SCHEMA_VERSION,
    SUPPORTED_PREVIOUS_CONFIG_SCHEMA_VERSIONS,
    get_template_config,
)
from config.hsl import generated_template
from tools.migrate_hsl_config import main, migrate


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    import socket

    def denied(*args, **kwargs):
        pytest.fail("offline config migration attempted network access")

    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setattr(socket, "getaddrinfo", denied)


def legacy(mode="coin"):
    cfg = get_template_config()
    cfg["config_version"] = "v8.4.0"
    cfg["live"]["hsl_signal_mode"] = mode
    # Source fixture intentionally models the retired release, independent of defaults.
    for side in ("long", "short"):
        cfg["bot"][side]["hsl"].update(
            restart_after_red_policy="threshold",
            no_restart_drawdown_threshold=1.0,
            orange_tier_mode="tp_only_with_active_entry_cancellation",
            tier_ratios={"yellow": 0.5, "orange": 0.75},
        )
    cfg["bot"]["long"]["hsl"]["enabled"] = True
    # An authored retired optimizer override must be edited, never silently dropped.
    cfg["optimize"]["fixed_runtime_overrides"] = {}
    if mode == "unified":
        for side in ("long", "short"):
            cfg["optimize"]["bounds"][side].pop("hsl", None)
    return cfg


@pytest.mark.parametrize("mode", ["coin", "pside"])
@pytest.mark.parametrize("policy", ["always", "never"])
def test_explicit_conversion_is_idempotent_and_preserves_other_settings(mode, policy):
    source = legacy(mode)
    original = deepcopy(source)
    result = migrate(source, restart_policies={"long": policy})
    assert source == original
    assert "hsl_engine" not in result["live"]
    assert result["bot"]["long"]["hsl"]["restart_after_red_policy"] == policy
    assert (
        result["bot"]["long"]["hsl"]["red_threshold"]
        == source["bot"]["long"]["hsl"]["red_threshold"]
    )
    assert result["bot"]["long"]["risk"] == source["bot"]["long"]["risk"]
    assert "tier_ratios" not in result["bot"]["long"]["hsl"]
    assert "hsl_position_during_cooldown_policy" not in result["live"]
    assert migrate(result) == result


def test_enabled_legacy_threshold_cannot_be_guessed():
    with pytest.raises(ValueError, match="explicit choice"):
        migrate(legacy())


def test_unified_policy_is_not_copied_from_side_or_template():
    cfg = legacy("unified")
    with pytest.raises(ValueError, match="explicit bot.hsl"):
        migrate(cfg)
    policy = generated_template(get_template_config(), "unified")["bot"]["hsl"]
    policy.update(enabled=True, restart_after_red_policy="never", red_threshold=0.123)
    result = migrate(cfg, portfolio=policy)
    assert result["bot"]["hsl"] == policy
    with pytest.raises(ValueError, match="already exists"):
        migrate(result, portfolio=policy)


@pytest.mark.parametrize(
    "choices", [{"bad": "always"}, {"long": "threshold"}, {"portfolio": "always"}]
)
def test_invalid_or_inactive_choices_fail(choices):
    with pytest.raises(ValueError):
        migrate(legacy(), restart_policies=choices)


def test_retired_search_dimension_is_rejected_not_silently_removed():
    cfg = legacy()
    cfg["optimize"]["fixed_runtime_overrides"][
        "bot.long.hsl.no_restart_drawdown_threshold"
    ] = 0.25
    with pytest.raises(ValueError, match="removed"):
        migrate(cfg, restart_policies={"long": "always"})


def test_coin_override_still_requires_its_own_explicit_restart_choice():
    cfg = legacy()
    cfg["coin_overrides"] = {
        "BTC": {"bot": {"long": {"hsl": {"restart_after_red_policy": "threshold"}}}}
    }
    with pytest.raises(ValueError, match="coin_overrides.*explicit choice"):
        migrate(cfg, restart_policies={"long": "always"})


def test_nested_current_input_supported_without_mutation():
    cfg = {"config": legacy()}
    original = deepcopy(cfg)
    assert "hsl_engine" not in migrate(cfg, restart_policies={"long": "always"})["live"]
    assert cfg == original


def test_cli_writes_new_file_and_refuses_overwrite(tmp_path, capsys):
    src, dst = tmp_path / "source.json", tmp_path / "converted.json"
    src.write_text(json.dumps(legacy()))
    before = src.read_bytes()
    assert main([str(src), str(dst), "--restart-policy", "long=always"]) == 0
    assert src.read_bytes() == before
    assert "hsl_engine" not in json.loads(dst.read_text())["live"]
    assert "not equivalent" in capsys.readouterr().err
    result = dst.read_bytes()
    with pytest.raises(SystemExit):
        main([str(src), str(dst), "--restart-policy", "long=never"])
    assert dst.read_bytes() == result
    with pytest.raises(SystemExit):
        main([str(src), str(src), "--restart-policy", "long=always"])
    assert src.read_bytes() == before


def test_cli_streamlines_scenarios_without_changing_config(tmp_path):
    cfg = legacy()
    cfg["backtest"]["scenarios"] = [
        {"label": "base"},
        {"label": "recent", "start_date": "2025-10-02"},
    ]
    src, dst = tmp_path / "source.json", tmp_path / "converted.json"
    src.write_text(json.dumps(cfg))
    assert main([str(src), str(dst), "--restart-policy", "long=always"]) == 0
    text = dst.read_text()
    assert (
        '"scenarios": [\n'
        '            {"label": "base"},\n'
        '            {"label": "recent", "start_date": "2025-10-02"}\n'
        "        ]"
    ) in text
    assert text.endswith("\n")
    assert json.loads(text) == migrate(
        cfg, restart_policies={"long": "always"}, base_config_path=str(src)
    )


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_cli_nonfinite_output_does_not_create_file(tmp_path, monkeypatch, value):
    src, dst = tmp_path / "source.json", tmp_path / "converted.json"
    src.write_text(json.dumps(legacy()))
    monkeypatch.setattr("tools.migrate_hsl_config.migrate", lambda *a, **kw: {"value": value})
    with pytest.raises(SystemExit):
        main([str(src), str(dst), "--restart-policy", "long=always"])
    assert not dst.exists()


@pytest.mark.parametrize(
    "extra", [[], ["--restart-policy", "long=always", "--restart-policy", "long=never"]]
)
def test_failed_cli_does_not_create_output(tmp_path, extra):
    src, dst = tmp_path / "source.json", tmp_path / "converted.json"
    src.write_text(json.dumps(legacy()))
    with pytest.raises(SystemExit):
        main([str(src), str(dst), *extra])
    assert not dst.exists()


@pytest.mark.parametrize(
    "patch",
    [
        {"restart_after_red_policy": "threshold"},
        {"no_restart_drawdown_threshold": 0.25},
        {"red_threshold": 0.0},
    ],
)
def test_file_backed_invalid_policy_fails_before_output(tmp_path, patch):
    src, dst, override = (
        tmp_path / "source.json",
        tmp_path / "output.json",
        tmp_path / "coin.json",
    )
    override.write_text(json.dumps({"bot": {"long": {"hsl": patch}}}))
    cfg = legacy()
    cfg["coin_overrides"] = {"BTC": {"override_config_path": str(override)}}
    src.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit):
        main([str(src), str(dst), "--restart-policy", "long=always"])
    assert not dst.exists()


def test_relative_override_materialized_with_inline_precedence_and_reloaded_elsewhere(
    tmp_path, monkeypatch
):
    from config.load import load_prepared_config
    from config.overrides import parse_overrides

    source_dir, output_dir, cwd = (tmp_path / p for p in ("source", "output", "cwd"))
    for folder in (source_dir, output_dir, cwd):
        folder.mkdir()
    patch = {
        "bot": {
            "long": {
                "hsl": {"red_threshold": 0.13, "restart_after_red_policy": "never"}
            }
        }
    }
    override = source_dir / "coin.json"
    override.write_text(json.dumps(patch))
    # A different same-named file must never supply the converted policy.
    (output_dir / "coin.json").write_text(
        json.dumps({"bot": {"long": {"hsl": {"red_threshold": 0.9}}}})
    )
    cfg = legacy()
    cfg["coin_overrides"] = {
        "BTC": {
            "override_config_path": "coin.json",
            "bot": {"long": {"hsl": {"red_threshold": 0.17}}},
        }
    }
    src, dst = source_dir / "source.json", output_dir / "converted.json"
    src.write_text(json.dumps(cfg))
    original = override.read_bytes()
    monkeypatch.chdir(cwd)
    assert main([str(src), str(dst), "--restart-policy", "long=always"]) == 0
    assert override.read_bytes() == original
    output = json.loads(dst.read_text())
    assert "override_config_path" not in output["coin_overrides"]["BTC"]
    override.unlink()  # Result must be self-contained, not dependent on old files.
    reloaded = parse_overrides(
        load_prepared_config(str(dst), verbose=False), verbose=False
    )
    assert reloaded["coin_overrides"]["BTC"]["bot"]["long"]["hsl"] == {
        "red_threshold": 0.17,
        "restart_after_red_policy": "never",
    }


@pytest.mark.parametrize("mode", ["Unified", " unified "])
def test_canonical_mode_normalization_precedes_scope_choices(mode):
    cfg = legacy("unified")
    cfg["live"]["hsl_signal_mode"] = mode
    policy = generated_template(get_template_config(), "unified")["bot"]["hsl"]
    result = migrate(cfg, portfolio=policy, restart_policies={"portfolio": "never"})
    assert result["live"]["hsl_signal_mode"] == "unified"
    assert result["bot"]["hsl"]["restart_after_red_policy"] == "never"
    with pytest.raises(ValueError, match="side restart choices are inactive"):
        migrate(cfg, portfolio=policy, restart_policies={"long": "always"})


@pytest.mark.parametrize(
    "alias",
    [
        "bot.long.hsl.restart_after_red_policy",
        "long.hsl.restart_after_red_policy",
        "bot.long.hsl_restart_after_red_policy",
    ],
)
def test_explicit_restart_choice_survives_optimizer_fixed_override(alias):
    from optimization.warmup import _apply_config_overrides

    cfg = legacy()
    cfg["optimize"]["fixed_runtime_overrides"] = {alias: "always"}
    result = migrate(cfg, restart_policies={"long": "never"})
    assert result["optimize"]["fixed_runtime_overrides"][alias] == "never"
    effective = deepcopy(result)
    _apply_config_overrides(effective, effective["optimize"]["fixed_runtime_overrides"])
    assert effective["bot"]["long"]["hsl"]["restart_after_red_policy"] == "never"


@pytest.mark.parametrize(
    "selector",
    [
        "long.hsl.no_restart_drawdown_threshold",
        "*.hsl.tier_ratios",
        "long_hsl_no_restart_drawdown_threshold",
        "long.hsl.misspelled_threshold",
    ],
)
def test_inactive_fixed_parameter_selector_rejected(selector):
    cfg = legacy()
    cfg["optimize"]["fixed_params"] = [selector]
    with pytest.raises(ValueError, match="fixed_params|removed"):
        migrate(cfg, restart_policies={"long": "always"})


def test_valid_fixed_groups_and_leaf_selectors_remain_supported():
    cfg = legacy()
    cfg["optimize"]["fixed_params"] = ["long.hsl", "*.hsl.red_threshold", "short.risk"]
    result = migrate(cfg, restart_policies={"long": "always"})
    assert result["optimize"]["fixed_params"] == cfg["optimize"]["fixed_params"]


@pytest.mark.parametrize("mode,scope", [("pside", "short"), ("unified", "portfolio")])
def test_restart_choice_updates_only_matching_optimizer_scope(mode, scope):
    cfg = legacy(mode)
    policy = (
        generated_template(get_template_config(), "unified")["bot"]["hsl"]
        if mode == "unified"
        else None
    )
    chosen = (
        "bot.hsl.restart_after_red_policy"
        if scope == "portfolio"
        else "bot.short.hsl.restart_after_red_policy"
    )
    fixed = {chosen: "always", "bot.long.risk.n_positions": 3}
    cfg["optimize"]["fixed_runtime_overrides"] = fixed
    choices = {scope: "never"}
    if mode == "pside":
        choices["long"] = "always"
    result = migrate(cfg, portfolio=policy, restart_policies=choices)
    assert result["optimize"]["fixed_runtime_overrides"] == {**fixed, chosen: "never"}
    assert cfg["optimize"]["fixed_runtime_overrides"][chosen] == "always"


def test_missing_override_file_does_not_create_output(tmp_path):
    cfg = legacy()
    cfg["coin_overrides"] = {"BTC": {"override_config_path": "missing.json"}}
    src, dst = tmp_path / "source.json", tmp_path / "output.json"
    src.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit):
        main([str(src), str(dst), "--restart-policy", "long=always"])
    assert not dst.exists()


def test_valid_selector_cannot_mask_unmatched_selector():
    cfg = legacy()
    cfg["optimize"]["fixed_params"] = ["long.hsl", "short.hsl.missing"]
    with pytest.raises(ValueError, match="short.hsl.missing.*matches no active bounds"):
        migrate(cfg, restart_policies={"long": "always"})


@pytest.mark.parametrize("policy", ["threshold", None])
def test_effective_optimizer_enablement_requires_explicit_policy(policy):
    cfg = legacy()
    cfg["bot"]["long"]["hsl"].update(enabled=False, restart_after_red_policy=policy)
    cfg["optimize"]["fixed_runtime_overrides"] = {"bot.long.hsl.enabled": True}
    with pytest.raises(ValueError, match="explicit choice"):
        migrate(cfg)
    result = migrate(cfg, restart_policies={"long": "never"})
    assert result["bot"]["long"]["hsl"]["enabled"] is False
    from optimization.warmup import _finalize_optimizer_vector_config

    effective = _finalize_optimizer_vector_config(deepcopy(result))
    assert effective["bot"]["long"]["hsl"]["enabled"] is True
    assert effective["bot"]["long"]["hsl"]["restart_after_red_policy"] == "never"


@pytest.mark.parametrize(
    "patch",
    [
        {"restart_after_red_policy": "threshold"},
        {"red_threshold": 0.0},
    ],
)
def test_scenario_file_invalid_policy_creates_no_output(tmp_path, patch):
    cfg = legacy()
    override = tmp_path / "coin.json"
    override.write_text(json.dumps({"bot": {"long": {"hsl": patch}}}))
    cfg["backtest"]["scenarios"] = [
        {
            "label": "invalid",
            "overrides": {
                "coin_overrides": {"BTC": {"override_config_path": "coin.json"}}
            },
        }
    ]
    src, dst = tmp_path / "source.json", tmp_path / "converted.json"
    src.write_text(json.dumps(cfg))
    with pytest.raises(SystemExit):
        main([str(src), str(dst), "--restart-policy", "long=always"])
    assert not dst.exists()


def test_scenario_file_materializes_and_preserves_atomic_replacement(tmp_path):
    from config.load import load_prepared_config
    from config.overrides import parse_overrides
    from suite_runner import build_scenarios, apply_scenario

    source_dir, output_dir = tmp_path / "source", tmp_path / "output"
    source_dir.mkdir()
    output_dir.mkdir()
    override = source_dir / "coin.json"
    override.write_text(
        json.dumps(
            {
                "bot": {
                    "long": {
                        "hsl": {
                            "restart_after_red_policy": "never",
                            "red_threshold": 0.13,
                        }
                    }
                }
            }
        )
    )
    cfg = legacy()
    cfg["coin_overrides"] = {"ETH": {"bot": {"long": {"hsl": {"red_threshold": 0.2}}}}}
    cfg["backtest"]["scenarios"] = [
        {
            "label": "file",
            "overrides": {
                "coin_overrides": {
                    "BTC": {
                        "override_config_path": "coin.json",
                        "bot": {"long": {"hsl": {"red_threshold": 0.17}}},
                    }
                }
            },
        }
    ]
    src, dst = source_dir / "source.json", output_dir / "converted.json"
    src.write_text(json.dumps(cfg))
    assert main([str(src), str(dst), "--restart-policy", "long=always"]) == 0
    result = json.loads(dst.read_text())
    assert (
        "override_config_path"
        not in result["backtest"]["scenarios"][0]["overrides"]["coin_overrides"]["BTC"]
    )
    override.unlink()
    loaded = load_prepared_config(str(dst), verbose=False)
    scenarios, _ = build_scenarios(loaded["backtest"])
    effective, _ = apply_scenario(
        loaded, scenarios[0], ["BTC"], [], ["binance"], {"BTC"}, quiet=True
    )
    resolved = parse_overrides(effective, verbose=False)
    assert set(resolved["coin_overrides"]) == {"BTC"}
    assert resolved["coin_overrides"]["BTC"]["bot"]["long"]["hsl"] == {
        "red_threshold": 0.17,
        "restart_after_red_policy": "never",
    }
    assert migrate(result) == result


def test_scenario_and_optimizer_enablement_validate_in_combination():
    cfg = legacy()
    cfg["bot"]["long"]["hsl"].update(enabled=False, restart_after_red_policy="always")
    cfg["optimize"]["fixed_runtime_overrides"] = {"bot.long.hsl.enabled": True}
    cfg["backtest"]["scenarios"] = [
        {
            "label": "combined",
            "overrides": {"bot.long.hsl.restart_after_red_policy": "threshold"},
        }
    ]
    with pytest.raises(ValueError, match="explicit choice"):
        migrate(cfg)


@pytest.mark.parametrize("dotted", [False, True])
def test_scenario_coin_patch_forms_are_materialized(tmp_path, dotted):
    cfg = legacy()
    for filename, threshold in [("base.json", 0.2), ("scenario.json", 0.3)]:
        (tmp_path / filename).write_text(
            json.dumps({"bot": {"long": {"hsl": {"red_threshold": threshold}}}})
        )
    cfg["coin_overrides"] = {"BTC": {"override_config_path": "base.json"}}
    patch = {"override_config_path": "scenario.json"}
    overrides = (
        {"coin_overrides.BTC.override_config_path": "scenario.json"}
        if dotted
        else {"coin_overrides": {"BTC": patch}}
    )
    cfg["backtest"]["scenarios"] = [{"label": "patch", "overrides": overrides}]
    result = migrate(
        cfg,
        restart_policies={"long": "always"},
        base_config_path=str(tmp_path / "input.json"),
    )
    assert result["coin_overrides"]["BTC"]["bot"]["long"]["hsl"]["red_threshold"] == 0.2
    assert result["backtest"]["scenarios"][0]["overrides"] == {
        "coin_overrides": {"BTC": {"bot": {"long": {"hsl": {"red_threshold": 0.3}}}}}
    }


def test_scenario_empty_coin_mapping_remains_replacement():
    cfg = legacy()
    cfg["coin_overrides"] = {"BTC": {"bot": {"long": {"hsl": {"red_threshold": 0.2}}}}}
    cfg["backtest"]["scenarios"] = [
        {"label": "clear", "overrides": {"coin_overrides": {}}}
    ]
    result = migrate(cfg, restart_policies={"long": "always"})
    assert result["backtest"]["scenarios"][0]["overrides"]["coin_overrides"] == {}


def test_effective_optimizer_policy_checks_coin_patches():
    cfg = legacy()
    cfg["bot"]["long"]["hsl"].update(enabled=False, restart_after_red_policy="always")
    cfg["coin_overrides"] = {
        "BTC": {"bot": {"long": {"hsl": {"restart_after_red_policy": "threshold"}}}}
    }
    cfg["optimize"]["fixed_runtime_overrides"] = {"bot.long.hsl.enabled": True}
    with pytest.raises(
        ValueError, match="explicit choice|normalization lost.*restart_after_red_policy"
    ):
        migrate(cfg)


@pytest.mark.parametrize(
    "overrides", [{"coin_overrides": {}}, {"live.hedge_mode": True}]
)
def test_legacy_flags_respect_scenario_replacement_and_inheritance(overrides):
    cfg = legacy()
    # A genuine pre-independent-unstuck config invokes canonical flag migration.
    for root in (cfg["bot"], cfg["optimize"]["bounds"]):
        for side in ("long", "short"):
            strategy = root[side]["strategy"]["trailing_martingale"]
            for key in ("ema_span_0", "ema_span_1"):
                strategy[key] = strategy["entry"].pop(key)
                root[side]["unstuck"].pop(key)
    cfg["live"]["coin_flags"] = {"BTC": "-lm gs"}
    cfg["backtest"]["scenarios"] = [{"label": "flags", "overrides": overrides}]
    result = migrate(cfg, restart_policies={"long": "always"})
    assert (
        result["coin_overrides"]["BTC"]["live"]["forced_mode_long"] == "graceful_stop"
    )
    assert result["live"]["coin_flags"] == {}
    if "coin_overrides" in overrides:
        assert result["backtest"]["scenarios"][0]["overrides"]["coin_overrides"] == {}
    else:
        assert "coin_overrides" not in result["backtest"]["scenarios"][0]["overrides"]


def test_symlink_input_uses_callers_directory_for_relative_overrides(tmp_path):
    from config.load import load_prepared_config
    from config.overrides import parse_overrides

    real, links = tmp_path / "real", tmp_path / "links"
    real.mkdir()
    links.mkdir()
    cfg = legacy()
    cfg["bot"]["long"]["hsl"]["restart_after_red_policy"] = "always"
    cfg["coin_overrides"] = {"BTC": {"override_config_path": "coin.json"}}
    target = real / "config.json"
    target.write_text(json.dumps(cfg))
    link = links / "config.json"
    link.symlink_to(target)
    for folder, threshold in [(real, 0.2), (links, 0.3)]:
        (folder / "coin.json").write_text(
            json.dumps({"bot": {"long": {"hsl": {"red_threshold": threshold}}}})
        )
    with pytest.raises(ValueError, match="migrate-hsl"):
        load_prepared_config(str(link), verbose=False)
    out = tmp_path / "output.json"
    assert main([str(link), str(out)]) == 0
    loaded = parse_overrides(
        load_prepared_config(str(out), verbose=False), verbose=False
    )
    assert loaded["coin_overrides"]["BTC"]["bot"]["long"]["hsl"]["red_threshold"] == 0.3
    assert json.loads(out.read_text())["coin_overrides"] == loaded["coin_overrides"]


@pytest.mark.parametrize(
    "selector",
    [
        "long.hsl.no_restart_drawdown_threshold",
        "long.hsl.misspelled_threshold",
        "long.hsl.red_threshold",
    ],
)
def test_scenario_optimizer_controls_rejected_even_if_selector_is_valid(selector):
    cfg = legacy()
    cfg["backtest"]["scenarios"] = [
        {"label": "ignored-control", "overrides": {"optimize.fixed_params": [selector]}}
    ]
    with pytest.raises(ValueError, match="optimizer controls.*top-level"):
        migrate(cfg, restart_policies={"long": "always"})


@pytest.mark.parametrize("mode", ["coin", "pside"])
@pytest.mark.parametrize("short_enabled", [False, True])
def test_optimizer_mirroring_cannot_overwrite_explicit_restart_choice(
    mode, short_enabled
):
    cfg = legacy(mode)
    cfg["bot"]["short"]["hsl"]["enabled"] = short_enabled
    cfg["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    original = deepcopy(cfg)
    with pytest.raises(
        ValueError,
        match=r"optimizer.*bot.short.hsl.restart_after_red_policy.*never.*always",
    ):
        migrate(cfg, restart_policies={"long": "always", "short": "never"})
    assert cfg == original


@pytest.mark.parametrize("policy", ["always", "never"])
def test_matching_explicit_restart_choices_allow_optimizer_mirroring(policy):
    cfg = legacy()
    cfg["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    result = migrate(cfg, restart_policies={"long": policy, "short": policy})
    assert result["optimize"]["enable_overrides"] == ["mirror_short_from_long"]
    assert all(
        result["bot"][side]["hsl"]["restart_after_red_policy"] == policy
        for side in ("long", "short")
    )


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("policy,fixed", [("never", "always"), ("always", "never")])
def test_portfolio_file_restart_choice_survives_optimizer_overrides(
    enabled, policy, fixed
):
    from optimization.warmup import _finalize_optimizer_vector_config

    cfg = legacy("unified")
    cfg["optimize"]["fixed_runtime_overrides"] = {
        "bot.hsl.restart_after_red_policy": fixed
    }
    portfolio = generated_template(get_template_config(), "unified")["bot"]["hsl"]
    portfolio.update(enabled=enabled, restart_after_red_policy=policy)
    original, original_portfolio = deepcopy(cfg), deepcopy(portfolio)
    result = migrate(cfg, portfolio=portfolio)
    assert result["bot"]["hsl"]["restart_after_red_policy"] == policy
    assert (
        result["optimize"]["fixed_runtime_overrides"][
            "bot.hsl.restart_after_red_policy"
        ]
        == policy
    )
    optimized = _finalize_optimizer_vector_config(deepcopy(result))
    assert optimized["bot"]["hsl"]["restart_after_red_policy"] == policy
    assert cfg == original and portfolio == original_portfolio


def test_explicit_restart_argument_overrides_portfolio_file_choice():
    cfg = legacy("unified")
    cfg["optimize"]["fixed_runtime_overrides"] = {
        "bot.hsl.restart_after_red_policy": "never"
    }
    portfolio = generated_template(get_template_config(), "unified")["bot"]["hsl"]
    portfolio.update(enabled=True, restart_after_red_policy="never")
    result = migrate(cfg, portfolio=portfolio, restart_policies={"portfolio": "always"})
    assert result["bot"]["hsl"]["restart_after_red_policy"] == "always"
    assert (
        result["optimize"]["fixed_runtime_overrides"][
            "bot.hsl.restart_after_red_policy"
        ]
        == "always"
    )


@pytest.mark.parametrize("spelling", ["Never", " never ", "Always", " always "])
@pytest.mark.parametrize("fixed", [None, "always", "never"])
def test_portfolio_choice_uses_canonical_spelling(spelling, fixed):
    cfg = legacy("unified")
    if fixed is not None:
        cfg["optimize"]["fixed_runtime_overrides"] = {
            "bot.hsl.restart_after_red_policy": fixed
        }
    portfolio = generated_template(get_template_config(), "unified")["bot"]["hsl"]
    portfolio.update(enabled=True, restart_after_red_policy=spelling)
    result = migrate(cfg, portfolio=portfolio)
    assert result["bot"]["hsl"]["restart_after_red_policy"] == spelling.strip().lower()
    if fixed is not None:
        assert (
            result["optimize"]["fixed_runtime_overrides"][
                "bot.hsl.restart_after_red_policy"
            ]
            == spelling.strip().lower()
        )


@pytest.mark.parametrize("file_backed", [False, True])
@pytest.mark.parametrize("coin", ["BTC", "BTCUSDT"])
def test_scenario_dotted_path_uses_canonical_materialized_coin_shape(
    tmp_path, file_backed, coin
):
    from config import prepare_config
    from config.overrides import parse_overrides
    from suite_runner import apply_scenario_overrides

    cfg = legacy()
    cfg["bot"]["long"]["hsl"]["restart_after_red_policy"] = "always"
    patch = {"bot": {"long": {"hsl": {"red_threshold": 0.2}}}}
    if file_backed:
        (tmp_path / "coin.json").write_text(json.dumps(patch))
        patch = {"override_config_path": "coin.json"}
    cfg["coin_overrides"] = {coin: patch}
    scenario = {f"coin_overrides.{coin}.bot.long.hsl.red_threshold": 0.3}
    cfg["backtest"]["scenarios"] = [{"label": "canonical", "overrides": scenario}]
    source = deepcopy(cfg)
    # Reference the already-authored new semantics; the old source itself must
    # go through the explicit migration below, not the ordinary loader.
    source["config_version"] = CONFIG_SCHEMA_VERSION
    canonical = parse_overrides(
        prepare_config(
            source,
            target="canonical",
            runtime=None,
            base_config_path=str(tmp_path / "source.json"),
        ),
        verbose=False,
    )
    apply_scenario_overrides(canonical, scenario)
    result = migrate(cfg, base_config_path=str(tmp_path / "source.json"))
    effective = deepcopy(result)
    apply_scenario_overrides(effective, result["backtest"]["scenarios"][0]["overrides"])
    assert effective["coin_overrides"] == canonical["coin_overrides"]
    assert result["coin_overrides"][coin]["bot"]["long"]["hsl"]["red_threshold"] == 0.2


def test_exact_symbol_override_is_not_collapsed_to_coin():
    cfg = legacy()
    cfg["coin_overrides"] = {
        "BTCUSDT": {"bot": {"long": {"hsl": {"red_threshold": 0.2}}}}
    }
    cfg["backtest"]["scenarios"] = [
        {
            "label": "wrong-key",
            "overrides": {"coin_overrides.BTC.bot.long.hsl.red_threshold": 0.3},
        }
    ]
    with pytest.raises(ValueError, match="missing coin_overrides.BTC"):
        migrate(cfg, restart_policies={"long": "always"})


@pytest.mark.parametrize("field", ["scoring", "limits"])
def test_migrated_unified_rejects_side_optimizer_metrics(field):
    cfg = legacy("unified")
    metric = {"metric": "hard_stop_triggers_long"}
    cfg["optimize"][field] = [
        (
            {**metric, "goal": "min"}
            if field == "scoring"
            else {**metric, "penalize_if": "greater_than", "value": 1}
        )
    ]
    policy = generated_template(get_template_config(), "unified")["bot"]["hsl"]
    policy.update(enabled=True, restart_after_red_policy="always")
    with pytest.raises(ValueError, match="unified"):
        migrate(cfg, portfolio=policy)


@pytest.mark.parametrize("field", ["scoring", "limits"])
@pytest.mark.parametrize("selected", ["active", "inactive", None])
def test_optimizer_metrics_follow_scenario_selection(field, selected):
    cfg = legacy()
    metric = {"metric": "hard_stop_triggers_long", "scenario": selected}
    cfg["optimize"][field] = [
        (
            {**metric, "goal": "min"}
            if field == "scoring"
            else {**metric, "penalize_if": "greater_than", "value": 1}
        )
    ]
    cfg["backtest"]["scenarios"] = [
        {"label": "active", "overrides": {}},
        {"label": "inactive", "overrides": {"bot.long.hsl.enabled": False}},
    ]
    if selected == "active":
        migrate(cfg, restart_policies={"long": "always"})
    else:
        with pytest.raises(ValueError, match="inactive|disabled"):
            migrate(cfg, restart_policies={"long": "always"})


def test_ordered_scenario_can_introduce_coin_before_addressing_leaf():
    cfg = legacy()
    cfg["backtest"]["scenarios"] = [
        {
            "label": "ordered",
            "overrides": {
                "coin_overrides": {
                    "BTC": {"bot": {"long": {"hsl": {"red_threshold": 0.2}}}}
                },
                "coin_overrides.BTC.bot.long.hsl.red_threshold": 0.3,
            },
        }
    ]
    result = migrate(cfg, restart_policies={"long": "always"})
    assert (
        result["backtest"]["scenarios"][0]["overrides"]["coin_overrides"]["BTC"]["bot"][
            "long"
        ]["hsl"]["red_threshold"]
        == 0.3
    )


@pytest.mark.parametrize("scenario", [False, True])
def test_gpu_coarse_candle_interval_rejected_offline(scenario):
    cfg = legacy()
    cfg["optimize"]["backend"] = "gpu"
    if scenario:
        cfg["backtest"]["scenarios"] = [
            {"label": "coarse", "overrides": {"backtest.candle_interval_minutes": 5}}
        ]
    else:
        cfg["backtest"]["candle_interval_minutes"] = 5
    with pytest.raises(ValueError, match="1m candles"):
        migrate(cfg, restart_policies={"long": "always"})


@pytest.mark.parametrize("version", sorted(SUPPORTED_PREVIOUS_CONFIG_SCHEMA_VERSIONS))
@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_previous_schema_migration_requires_policy_and_reloads_offline(
    version, mode, tmp_path
):
    from config.load import load_prepared_config

    cfg = legacy(mode)
    cfg["config_version"] = version
    cfg["live"]["hsl_engine"] = "legacy"
    with pytest.raises(ValueError, match="explicit"):
        migrate(cfg)
    policy = None
    choices = {"long": "never"}
    if mode == "unified":
        policy = generated_template(get_template_config(), mode)["bot"]["hsl"]
        policy.update(enabled=True, restart_after_red_policy="never")
        choices = None
    original = deepcopy(cfg)
    migrated = migrate(cfg, restart_policies=choices, portfolio=policy)
    assert cfg == original
    assert migrated["config_version"] == CONFIG_SCHEMA_VERSION
    assert migrate(migrated) == migrated
    output = tmp_path / "migrated.json"
    output.write_text(json.dumps(migrated, allow_nan=False))
    loaded = load_prepared_config(str(output), verbose=False)
    assert loaded["config_version"] == CONFIG_SCHEMA_VERSION
    assert loaded["live"]["hsl_signal_mode"] == mode
    active = loaded["bot"]["hsl"] if mode == "unified" else loaded["bot"]["long"]["hsl"]
    assert active["enabled"] is True
    assert active["restart_after_red_policy"] == "never"
    assert "hsl_engine" not in loaded["live"]


@pytest.mark.parametrize("version", ["v8.7.0", "v9.0.0", "v8.0.999", "banana"])
def test_cli_invalid_schema_preserves_input_and_existing_output(version, tmp_path):
    source, output = tmp_path / "old.json", tmp_path / "converted.json"
    cfg = legacy()
    cfg["config_version"] = version
    source.write_text(json.dumps(cfg))
    original = source.read_bytes()
    with pytest.raises(SystemExit):
        main([str(source), str(output), "--restart-policy", "long=always"])
    assert source.read_bytes() == original
    assert not output.exists()
    output.write_bytes(b"existing file must survive")
    with pytest.raises(SystemExit):
        main([str(source), str(output), "--restart-policy", "long=always"])
    assert output.read_bytes() == b"existing file must survive"


@pytest.mark.parametrize(
    "path",
    sorted(
        (Path(__file__).resolve().parents[1] / "configs" / "examples").glob("*.json")
    ),
    ids=lambda path: path.name,
)
def test_public_examples_pass_effective_optimizer_and_migration_validation(path):
    from config.hsl import FIELDS

    source = json.loads(path.read_text())
    assert source["config_version"] == CONFIG_SCHEMA_VERSION
    result = migrate(source, base_config_path=str(path))
    assert migrate(result) == result
    assert "hsl_engine" not in result["live"]
    assert all(set(result["bot"][side]["hsl"]) == FIELDS for side in ("long", "short"))


@pytest.mark.parametrize("mode", ["coin", "pside", "unified"])
def test_master_v85_adaptive_policies_survive_hsl_migration(mode):
    source = legacy(mode)
    source["config_version"] = "v8.5.0"
    cooldown = source["bot"]["long"]["entry_cooldown"]
    cooldown.update(
        min_duration_minutes=2.0,
        max_duration_minutes=90.0,
        weights_minutes={"exposure_ratio": 4.0, "adverse_directionality": 8.0},
    )
    source["bot"]["long"]["forager"]["unilateralness_ema_span_1m"] = 3.25
    source["optimize"]["bounds"]["long"]["entry_cooldown"] = {
        "weights_minutes": {"exposure_ratio": [2.0, 6.0, 0.25]}
    }
    source["coin_overrides"] = {
        "BTC": {"bot": {"long": {"entry_cooldown": {"max_duration_minutes": 55.0}}}}
    }
    policies = {"long": "always"}
    portfolio = None
    if mode == "unified":
        portfolio = generated_template(get_template_config(), mode)["bot"]["hsl"]
        portfolio.update(enabled=True, restart_after_red_policy="always")
        policies = None
    migrated = migrate(source, restart_policies=policies, portfolio=portfolio)
    assert migrated["bot"]["long"]["entry_cooldown"] == cooldown
    assert migrated["bot"]["long"]["forager"]["unilateralness_ema_span_1m"] == 3.25
    assert migrated["coin_overrides"] == source["coin_overrides"]
    bounds = migrated["optimize"]["bounds"]["long"]["entry_cooldown"]
    assert bounds["weights_minutes"] == {
        "exposure_ratio": [2.0, 6.0, 0.25], "adverse_directionality": [8.0, 8.0],
    }
    assert bounds["min_duration_minutes"] == [2.0, 2.0]
    assert bounds["max_duration_minutes"] == [90.0, 90.0]
    assert migrate(migrated) == migrated
