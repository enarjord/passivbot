"""Explicit directional refactoring preserves retained execution and search policy."""

from copy import deepcopy
import json

import pytest

from config.load import prepare_config
from config.reshape import reshape_config
from config.universe import resolve_universe_sources


def policy():
    return {
        "config_version": "v9.0.0", "live": {},
        "bot": {
            "global": {"universe": {"approved_coins": ["BTC", "ETH"], "ignored_coins": ["ETH"]}},
            "long": {"enabled": True, "risk": {"max_positions": 3, "total_wallet_exposure_limit_ratio": 1.5}},
            "short": {"enabled": True, "risk": {"max_positions": 1, "total_wallet_exposure_limit_ratio": 0.0},
                      "unstuck": {"close_fraction": 0.15}},
        },
        "optimize": {"bounds": {
            "long": {"unstuck": {"close_fraction": [0.02, 0.06]}},
            "short": {"risk": {"total_wallet_exposure_limit_ratio": [0.0, 0.0]}},
        }},
    }


def load(source, base=""):
    return prepare_config(deepcopy(source), base_config_path=str(base), live_only=True, verbose=False)


def test_default_refactoring_preserves_source_exactly():
    source = policy()
    original = deepcopy(source)
    result = reshape_config(source)
    assert result.config == original and result.config is not source
    assert result.changes == result.warnings == []
    assert source == original


@pytest.mark.parametrize("argument,value", [("disabled_sides", "bad"), ("bot_layout", "bad")])
def test_unknown_options_fail(argument, value):
    with pytest.raises(ValueError, match="unsupported"):
        reshape_config(policy(), **{argument: value})


@pytest.mark.parametrize("action", ["disable", "omit"])
@pytest.mark.parametrize("context", ["bounds", "fixed", "scenario", "nested-scenario", "scenario-bounds", "mirror"])
def test_dormant_side_reachability_blocks_refactoring(action, context):
    source = policy()
    if context == "bounds":
        source["optimize"]["bounds"]["short"]["risk"]["total_wallet_exposure_limit_ratio"] = [0.0, 1.0]
    elif context == "fixed":
        source["optimize"]["fixed_runtime_overrides"] = {"bot.short.risk.total_wallet_exposure_limit_ratio": 1.0}
    elif context == "scenario":
        source["backtest"] = {"scenarios": [{"label": "later", "overrides": {"bot.short.risk.total_wallet_exposure_limit_ratio": 1.0}}]}
    elif context == "nested-scenario":
        source["backtest"] = {"scenarios": [{"label": "later", "overrides": {"bot": {"short": {"risk": {"total_wallet_exposure_limit_ratio": 1.0}}}}}]}
    elif context == "scenario-bounds":
        source["backtest"] = {"scenarios": [{"label": "later", "overrides": {"optimize.bounds.short.risk.total_wallet_exposure_limit_ratio": [0.0, 1.0]}}]}
    else:
        source["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    with pytest.raises(ValueError, match="short"):
        reshape_config(source, disabled_sides=action)


def test_fixed_selector_prevents_unreachable_bound_from_blocking_omission():
    source = policy()
    source["optimize"]["bounds"]["short"]["risk"]["total_wallet_exposure_limit_ratio"] = [0.0, 1.0]
    source["optimize"]["fixed_params"] = ["bot.short.risk.total_wallet_exposure_limit_ratio"]
    result = reshape_config(source, disabled_sides="omit")
    assert "short" not in result.config["bot"]
    assert result.config["optimize"]["fixed_params"] == []


def test_fixed_scalar_capacity_override_keeps_precedence_over_bounds():
    source = policy()
    source["optimize"]["bounds"]["short"]["risk"]["total_wallet_exposure_limit_ratio"] = [0.0, 1.0]
    source["optimize"]["fixed_runtime_overrides"] = {"bot.short.risk.total_wallet_exposure_limit_ratio": 0.0}
    result = reshape_config(source, disabled_sides="omit")
    assert "short" not in result.config["bot"]


def test_missing_side_can_be_omitted_without_activating_it():
    source = policy()
    del source["bot"]["short"]
    del source["optimize"]["bounds"]["short"]
    result = reshape_config(source, disabled_sides="omit", bot_layout="global")
    assert "short" not in result.config["bot"]
    assert load(result.config)["bot"]["short"]["enabled"] is False


def test_disable_retains_profile_and_omission_prunes_side_policies():
    source = policy()
    source["coin_overrides"] = {"BTC": {"bot": {
        "long": {"unstuck": {"close_fraction": 0.04}},
        "short": {"unstuck": {"close_fraction": 0.12}},
    }}}
    source["optimize"]["fixed_runtime_overrides"] = {"bot.short.unstuck.close_fraction": 0.13}
    source["backtest"] = {"scenarios": [{"label": "dormant", "overrides": {"bot.short.unstuck.close_fraction": 0.14}}]}
    disabled = reshape_config(source, disabled_sides="disable")
    assert disabled.config["bot"]["short"] == {**source["bot"]["short"], "enabled": False}
    omitted = reshape_config(source, disabled_sides="omit")
    assert "short" not in omitted.config["bot"]
    assert "short" not in omitted.config["optimize"]["bounds"]
    assert "short" not in omitted.config["coin_overrides"]["BTC"]["bot"]
    assert omitted.config["optimize"]["fixed_runtime_overrides"] == {}
    assert omitted.config["backtest"]["scenarios"][0]["overrides"] == {}
    assert any("held positions" in warning for warning in omitted.warnings)


def test_global_layout_retains_dormant_profile_and_active_side_bounds():
    source = policy()
    before = load(source)
    result = reshape_config(source, disabled_sides="disable", bot_layout="global")
    assert result.config["bot"]["long"] == {"enabled": True}
    assert result.config["bot"]["short"]["enabled"] is False
    assert result.config["bot"]["short"]["unstuck"]["close_fraction"] == 0.15
    assert result.config["optimize"]["bounds"] == source["optimize"]["bounds"]
    after = load(result.config)
    assert before["bot"]["long"] == after["bot"]["long"]
    assert {**before["bot"]["short"], "enabled": False} == after["bot"]["short"]


@pytest.mark.parametrize("layout", ["global", "side"])
def test_layout_rejects_global_bounds_instead_of_changing_coupling(layout):
    source = policy()
    source["optimize"]["bounds"]["global"] = {"unstuck": {"close_fraction": [0.02, 0.08]}}
    with pytest.raises(ValueError, match="global optimizer bounds"):
        reshape_config(source, bot_layout=layout)


def test_global_layout_rejects_two_active_profiles():
    source = policy()
    source["bot"]["short"]["risk"]["total_wallet_exposure_limit_ratio"] = 1.0
    with pytest.raises(ValueError, match="exactly one"):
        reshape_config(source, bot_layout="global")


def test_global_scenario_override_that_would_gain_authority_is_rejected():
    source = policy()
    source["backtest"] = {"scenarios": [{"label": "edit", "overrides": {
        "bot.global.risk.total_wallet_exposure_limit_ratio": 0.7,
    }}]}
    with pytest.raises(ValueError, match="retained long.*scenario"):
        reshape_config(source, disabled_sides="omit", bot_layout="global")


@pytest.mark.parametrize("layout", ["global", "side"])
def test_universe_file_bindings_survive_layout_and_reload(tmp_path, layout):
    source = policy()
    (tmp_path / "coins.json").write_text('["BTC"]')
    source["bot"]["global"]["universe"]["approved_coins"] = "./coins.json"
    source["bot"]["long"]["universe"] = {"approved_coins": None, "ignored_coins": None}
    base = tmp_path / "source.json"
    result = reshape_config(source, disabled_sides="omit", bot_layout=layout, base_config_path=str(base))
    profile = result.config["bot"]["global" if layout == "global" else "long"]
    assert profile["universe"]["approved_coins"] == "./coins.json"
    (tmp_path / "coins.json").write_text('["ETH"]')
    resolved, _ = resolve_universe_sources(load(result.config, base), "approved_coins")
    assert resolved["long"] == ["ETH"]


def test_global_layout_pins_dormant_original_file_binding(tmp_path):
    source = policy()
    (tmp_path / "old.json").write_text('["BTC"]')
    (tmp_path / "long.json").write_text('["ETH"]')
    source["bot"]["global"]["universe"]["approved_coins"] = "./old.json"
    source["bot"]["long"]["universe"] = {"approved_coins": "./long.json"}
    base = tmp_path / "source.json"
    result = reshape_config(source, disabled_sides="disable", bot_layout="global", base_config_path=str(base))
    assert result.config["bot"]["global"]["universe"]["approved_coins"] == "./long.json"
    assert result.config["bot"]["short"]["universe"]["approved_coins"] == "./old.json"


def test_omitting_mirror_source_is_rejected_even_when_its_activation_is_false():
    source = policy()
    source["bot"]["long"]["enabled"] = False
    source["bot"]["short"]["enabled"] = True
    source["bot"]["short"]["risk"]["total_wallet_exposure_limit_ratio"] = 1.0
    source["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    with pytest.raises(ValueError, match="mirror_short_from_long"):
        reshape_config(source, disabled_sides="omit")


def test_referenced_omitted_side_requires_resolved_export(tmp_path):
    source = policy()
    (tmp_path / "coin.json").write_text(json.dumps({"bot": {"short": {"unstuck": {"close_fraction": 0.12}}}}))
    source["coin_overrides"] = {"BTC": {"override_config_path": "coin.json"}}
    with pytest.raises(ValueError, match="resolved"):
        reshape_config(source, disabled_sides="omit", base_config_path=str(tmp_path / "source.json"))


def test_referenced_scenario_coin_policy_requires_resolved_export(tmp_path):
    source = policy()
    (tmp_path / "coin.json").write_text(json.dumps({"bot": {"short": {"unstuck": {"close_fraction": 0.12}}}}))
    source["backtest"] = {"scenarios": [{"label": "file", "overrides": {
        "coin_overrides": {"BTC": {"override_config_path": "coin.json"}},
    }}]}
    with pytest.raises(ValueError, match="resolved"):
        reshape_config(source, disabled_sides="omit", base_config_path=str(tmp_path / "source.json"))


def test_parent_replacement_that_enables_dormant_side_is_audited():
    source = policy()
    source["bot"]["short"]["enabled"] = False
    source["bot"]["short"]["risk"]["total_wallet_exposure_limit_ratio"] = 1.0
    source["backtest"] = {"scenarios": [{"label": "replacement", "overrides": {
        "bot.short": {"enabled": True, "risk": {"max_positions": 1, "total_wallet_exposure_limit_ratio": 1.0}},
    }}]}
    with pytest.raises(ValueError, match="short"):
        reshape_config(source, disabled_sides="omit")


def test_optimizer_scenario_context_matches_real_suite_transform_order():
    from config.bot_inheritance import bot_policy_source_view
    from config.reshape import _contexts, _prepare, _transforms
    from optimizer_overrides import apply_coupled_unstuck_ema_spans, materialize_coupled_scenario_spans
    from suite_runner import apply_scenario_overrides

    source = policy()
    source["bot"]["short"]["enabled"] = False
    source["optimize"]["enable_overrides"] = [
        "mirror_short_from_long", "couple_unstuck_ema_spans", "lossless_close_trailing",
    ]
    source["backtest"] = {"scenarios": [{"label": "after-candidate", "overrides": {
        "bot.short.risk.total_wallet_exposure_limit_ratio": 0.2,
        "bot.short.strategy.trailing_martingale.entry.ema_span_0_minutes": 123.0,
        "bot.short.strategy.trailing_martingale.close.threshold_base_fraction": 0.001,
    }}]}

    # Use the ordinary candidate export and suite patch owners as the reference.
    candidate = _transforms(_prepare(source, ""))
    materialize_coupled_scenario_spans(candidate)
    reference = load(bot_policy_source_view(candidate, candidate=True, include_defaults=True))
    apply_scenario_overrides(reference, candidate["backtest"]["scenarios"][0]["overrides"])
    apply_coupled_unstuck_ema_spans(reference)

    for upper in (False, True):
        audited = dict(_contexts(source, "", upper=upper))["optimizer scenario 1"]
        expected = reference["bot"]["short"]
        actual = audited["bot"]["short"]
        assert actual["risk"]["total_wallet_exposure_limit_ratio"] == expected["risk"]["total_wallet_exposure_limit_ratio"] == 0.2
        assert actual["strategy"]["trailing_martingale"]["close"]["threshold_base_fraction"] == expected["strategy"]["trailing_martingale"]["close"]["threshold_base_fraction"] == 0.001
        assert actual["unstuck"]["ema_span_0_minutes"] == expected["unstuck"]["ema_span_0_minutes"] == 123.0


def test_scenario_activation_retains_completed_mirrored_capacity():
    source = policy()
    source["bot"]["short"]["enabled"] = False
    source["optimize"]["enable_overrides"] = ["mirror_short_from_long"]
    source["backtest"] = {"scenarios": [{"label": "activate", "overrides": {
        "bot.short.enabled": True,
    }}]}
    with pytest.raises(ValueError, match="short.*optimizer scenario"):
        reshape_config(source, disabled_sides="omit")
