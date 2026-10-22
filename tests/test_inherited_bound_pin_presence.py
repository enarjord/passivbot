"""Equal values do not let an obsolete side coordinate regain source ownership."""

from copy import deepcopy

import pytest

from config.bot_inheritance import (
    apply_bot_config_patch, apply_bot_policy_patch, get_bot_policy, resolve_bot_policy,
)
from config.inherited_bounds import get_bound_plan, prepare_bound_plan
from test_inherited_bound_consumers import prepared
from test_inherited_bounds import capacity_binding, vector_for


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("replacement", ["side", "parent", "root"])
@pytest.mark.parametrize("fixed_selector", [False, True])
def test_removed_equal_pin_rejects_old_direct_values(side, replacement, fixed_selector):
    config = prepared(
        source={"global": {"risk": {"capacity": 1.0}},
                side: {"risk": {"capacity": 1.0}}},
        bounds={"global": {"risk": {"capacity": [1.0, 3.0]}}},
    )
    if fixed_selector:
        config["optimize"]["fixed_params"] = [f"bot.{side}.risk.capacity"]
        prepare_bound_plan(config)
    policy, plan = get_bot_policy(config), get_bound_plan(config)
    before = deepcopy(policy.source_view())
    if replacement == "root":
        source = policy.source_view()
        source.pop(side)
        changed = resolve_bot_policy(source, policy.catalog)
    else:
        selector = f"bot.{side}" + (".risk" if replacement == "parent" else "")
        changed = apply_bot_policy_patch(policy, {selector: {}})
    assert changed.effective_view() == policy.effective_view()
    assert policy.has_side_pin(side, ("risk", "capacity"))
    assert not changed.has_side_pin(side, ("risk", "capacity"))
    with pytest.raises(ValueError, match="does not match source authority"):
        plan.apply_values(changed, {
            capacity_binding(plan, "global").key: 3.0,
            capacity_binding(plan, side).key: 1.0,
        })
    assert policy.source_view() == before
    assert not changed.has_side_pin(side, ("risk", "capacity"))


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("replacement", ["side", "parent", "root"])
def test_canonical_parent_patch_cannot_reuse_an_equal_side_pin_plan(kind, side, replacement):
    from config import prepare_config
    from config.schema import get_template_config

    source = get_template_config()
    source["bot"]["portfolio"]["strategy_kind"] = kind
    value = source["bot"]["global"]["unstuck"]["close_fraction"]
    source["bot"].setdefault(side, {})["unstuck"] = {"close_fraction": value}
    source["optimize"]["bounds"] = {"global": {"unstuck": {"close_fraction": [0.02, 0.06]}}}
    config = prepare_config(source, verbose=False)
    plan = get_bound_plan(config)
    if replacement == "root":
        bot = get_bot_policy(config).source_view()
        bot[side].pop("unstuck")
        patch = {"bot": bot}
    else:
        patch = {f"bot.{side}" + (".unstuck" if replacement == "parent" else ""): {}}
    assert apply_bot_config_patch(config, patch)
    changed = get_bot_policy(config)
    assert not changed.has_side_pin(side, ("unstuck", "close_fraction"))
    with pytest.raises(ValueError, match="does not match source authority"):
        plan.apply_vector(changed, [
            0.055 if binding.key == "bot.global.unstuck.close_fraction" else binding.bound.low
            for binding in plan.bindings
        ])
    rebuilt = prepare_bound_plan(config)
    common = next(binding for binding in rebuilt.bindings
                  if binding.key == "bot.global.unstuck.close_fraction")
    assert common.effective_targets == tuple(
        ("bot", target, "unstuck", "close_fraction") for target in ("long", "short")
    )
    updated = rebuilt.apply_values(changed, {common.key: 0.055})
    assert [updated.effective_view()[target]["unstuck"]["close_fraction"]
            for target in ("long", "short")] == [0.055, 0.055]
    assert not updated.has_side_pin(side, ("unstuck", "close_fraction"))


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("fixed_selector", [False, True])
def test_explicit_side_bound_without_a_value_pin_remains_repeatable(side, fixed_selector):
    config = prepared(bounds={
        "global": {"risk": {"capacity": [1.0, 3.0]}},
        side: {"risk": {"capacity": [1.0, 3.0]}},
    })
    if fixed_selector:
        config["optimize"]["fixed_params"] = [f"bot.{side}.risk.capacity"]
        prepare_bound_plan(config)
    policy, plan = get_bot_policy(config), get_bound_plan(config)
    assert not policy.has_side_pin(side, ("risk", "capacity"))
    first = plan.apply_vector(policy, vector_for(plan, **{"bot.global.risk.capacity": 2.0}))
    second = plan.apply_vector(first, vector_for(plan, **{"bot.global.risk.capacity": 3.0}))
    assert first.has_side_pin(side, ("risk", "capacity"))
    assert second.effective_view()[side]["risk"]["capacity"] == 1.0
    assert second.effective_view()["short" if side == "long" else "long"]["risk"]["capacity"] == 3.0


@pytest.mark.parametrize("side", ["long", "short"])
def test_unchanged_equal_pin_and_repeated_common_gene_keep_the_same_authority(side):
    config = prepared(source={side: {"risk": {"capacity": 1.0}}},
                      bounds={"global": {"risk": {"capacity": [1.0, 3.0]}}})
    policy, plan = get_bot_policy(config), get_bound_plan(config)
    for value in (2.0, 3.0, 2.0):
        policy = plan.apply_vector(policy, vector_for(plan, **{"bot.global.risk.capacity": value}))
        assert policy.has_side_pin(side, ("risk", "capacity"))
        assert policy.effective_view()[side]["risk"]["capacity"] == 1.0
