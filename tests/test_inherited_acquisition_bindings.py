"""Acquisition facts and reachable gates retain their original authoring owners."""
from copy import deepcopy

import pytest

from config import prepare_config
from config.bot_inheritance import (
    apply_bot_config_patch, bot_policy_source_view, get_bot_policy,
)
from config.inherited_bounds import get_bound_plan, optimize_bounds_source_view
from config.universe import materialize_side_universe_sources
from optimization.warmup import build_optimizer_data_config, refresh_optimizer_source_bindings


def source(kind="trailing_martingale"):
    return {"config_version": "v9.0.0", "bot": {
        "portfolio": {"strategy_kind": kind},
        "global": {"enabled": True, "risk": {
            "max_positions": 0, "total_wallet_exposure_limit": 0.0}}},
        "optimize": {"bounds": {"global": {"risk": {
            "max_positions": [0, 3], "total_wallet_exposure_limit": [0.0, 2.0]}}}}}


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("short_pin", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_acquisition_gate_genes_fan_out_without_changing_pins_or_ranges(kind, short_pin, fixed):
    raw = source(kind)
    if short_pin:
        raw["bot"]["short"] = {"risk": {
            "max_positions": 0, "total_wallet_exposure_limit": 0.0}}
    if fixed:
        raw["optimize"]["fixed_runtime_overrides"] = {
            "bot.global.risk.max_positions": 5,
            "bot.global.risk.total_wallet_exposure_limit": 4.0}
    base = prepare_config(raw, verbose=False)
    original = deepcopy(base)
    identity = get_bound_plan(base).identity()
    data = build_optimizer_data_config(base)
    expected = {"max_positions": 5 if fixed else 3,
                "total_wallet_exposure_limit": 4.0 if fixed else 2.0}
    for field, value in expected.items():
        assert data["bot"]["long"]["risk"][field] == value
        assert data["bot"]["short"]["risk"][field] == (0 if short_pin else value)
    assert base == original
    assert get_bound_plan(base).identity() == get_bound_plan(data).identity() == identity
    assert optimize_bounds_source_view(data)["optimize"]["bounds"] == raw["optimize"]["bounds"]


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_acquired_membership_requires_an_explicit_binding_refresh_without_rereading(tmp_path, kind):
    raw = source(kind)
    file = tmp_path / "approved.json"
    file.write_text('["AAA","BBB"]')
    raw["bot"]["global"]["universe"] = {"approved_coins": str(file)}
    config = prepare_config(raw, live_only=True, verbose=False)
    before = get_bound_plan(config)
    materialize_side_universe_sources(config, "approved_coins", {
        "long": ["AAA"], "short": ["BBB"]})
    policy = get_bot_policy(config)
    with pytest.raises(ValueError, match="does not match source authority"):
        before.apply_vector(policy, [binding.bound.low for binding in before.bindings])
    file.unlink()
    refresh_optimizer_source_bindings(config)
    after = get_bound_plan(config)
    assert after.identity() != before.identity()
    after.apply_vector(get_bot_policy(config), [binding.bound.low for binding in after.bindings])
    assert optimize_bounds_source_view(config)["optimize"]["bounds"] == raw["optimize"]["bounds"]
    assert list(config["bot"]["long"]["universe"]["approved_coins"]) == ["AAA"]


@pytest.mark.parametrize("field", ["approved_coins", "ignored_coins"])
def test_plain_source_replacing_equal_prepared_membership_is_an_explicit_pin(field):
    raw = source()
    raw["bot"]["global"]["universe"] = {field: ["AAA"]}
    config = prepare_config(raw, verbose=False)
    config["bot"]["long"]["universe"][field] = ["AAA"]
    assert bot_policy_source_view(config)["bot"]["long"]["universe"][field] == ["AAA"]
    apply_bot_config_patch(config, {f"bot.global.universe.{field}": ["BBB"]})
    assert config["bot"]["long"]["universe"][field] == ["AAA"]
    assert config["bot"]["short"]["universe"][field] == ["BBB"]


def test_authored_facade_preserves_prior_canonical_edits_and_rejects_invalid_aliases():
    config = prepare_config(source(), verbose=False)
    config["bot"]["long"]["risk"]["total_wallet_exposure_limit"] = 1.25
    apply_bot_config_patch(config, {"bot.global.risk.total_wallet_exposure_limit": 2.0})
    assert config["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 1.25
    invalid = deepcopy(config)
    invalid["bot"]["long"]["wallet_exposure_limit"] = 0.5
    with pytest.raises(ValueError, match="unknown"):
        apply_bot_config_patch(invalid, {"bot.global.enabled": False})
    invalid = deepcopy(config)
    invalid["bot"]["long"]["risk"]["max_positions"] = 0.5
    with pytest.raises(TypeError, match="max_positions"):
        apply_bot_config_patch(invalid, {"bot.global.enabled": False})


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_omitted_fixed_float_signature_is_stable_after_actual_vector_normalization(kind):
    from config.bot_inheritance import apply_bot_policy_patch
    from config.inherited_bounds import prepare_bound_plan
    from optimization.shape import build_optimization_shape
    from optimize import config_to_individual, individual_to_config
    from optimizer_overrides import optimizer_overrides
    raw = source(kind)
    raw["bot"]["global"]["entry_cooldown"] = {"base_duration_minutes": 0}
    raw["optimize"]["bounds"]["global"]["risk"].pop("max_positions")
    config = prepare_config(raw, verbose=False)
    shape = build_optimization_shape(config)
    vector = config_to_individual(config, shape.bounds, optimization_shape=shape)
    candidate = individual_to_config(vector, optimizer_overrides, [], config, key_paths=shape.key_paths)
    plan = get_bound_plan(config)
    policy = get_bot_policy(candidate)
    # A prepared dataset reconstructs the same canonical plan from its trusted
    # policy and logical ranges; effective float normalization cannot split it.
    assert prepare_bound_plan(candidate).identity() == plan.identity()
    plan.apply_values(policy, {})
    changed = apply_bot_policy_patch(policy, {"bot.global.entry_cooldown.base_duration_minutes": 1.0})
    with pytest.raises(ValueError, match="does not match source authority"):
        plan.apply_values(changed, {})
    changed = apply_bot_policy_patch(policy, {"bot.global.risk.max_positions": 1})
    with pytest.raises(ValueError, match="does not match source authority"):
        plan.apply_values(changed, {})
    with pytest.raises(TypeError, match="max_positions"):
        apply_bot_policy_patch(policy, {"bot.global.risk.max_positions": 0.5})


@pytest.mark.parametrize("anchored", [False, True])
def test_vector_reconciliation_cannot_erase_an_unprepared_source_edit(tmp_path, anchored):
    from optimization.shape import build_optimization_shape
    from optimization.warmup import build_optimizer_vector_config
    from optimize import install_anchored_fine_tune_plan
    original, replacement = tmp_path / "first.json", tmp_path / "second.json"
    original.write_text('["AAA"]')
    replacement.write_text('["AAA"]')
    raw = source()
    raw["bot"]["global"]["universe"] = {"approved_coins": str(original)}
    config = prepare_config(raw, verbose=False)
    if anchored:
        install_anchored_fine_tune_plan(config, ["global.risk.total_wallet_exposure_limit"],
                                      "unused", starting_configs_override=[bot_policy_source_view(config)])
    shape = build_optimization_shape(config)
    config["bot"]["global"]["universe"]["approved_coins"] = str(replacement)
    with pytest.raises(ValueError, match="does not match source authority"):
        build_optimizer_vector_config([bound.low for bound in shape.bounds], config)
    assert config["bot"]["global"]["universe"]["approved_coins"] == str(replacement)
