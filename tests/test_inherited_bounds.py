"""Global search coordinates couple inheritors without erasing side pins."""

import pytest

from config.bot_inheritance import _get, apply_bot_policy_patch, resolve_bot_policy
from config.inherited_bounds import build_bound_plan
from config.universe import ResolvedCoinList
from test_bot_inheritance import catalog, real_catalog


SUFFIX = ("risk", "capacity")


def plan_for(raw=None, bounds=None):
    policy = resolve_bot_policy({} if raw is None else raw, catalog())
    plan = build_bound_plan(policy, {} if bounds is None else bounds, strategy_kind="trailing_martingale", hsl_scope="coin")
    return policy, plan


def vector_for(plan, **updates):
    return [updates.get(binding.key, binding.bound.low) for binding in plan.bindings]


def capacity_binding(plan, owner):
    return next(binding for binding in plan.bindings if binding.authoring_path == ("bot", owner, *SUFFIX))


def test_one_global_coordinate_targets_both_inheriting_sides():
    policy, plan = plan_for(bounds={"global": {"risk": {"capacity": [1.0, 3.0]}}})
    binding = capacity_binding(plan, "global")
    assert binding.effective_targets == (("bot", "long", *SUFFIX), ("bot", "short", *SUFFIX))
    assert binding.native_ids == ("long_risk_capacity", "short_risk_capacity")
    assert not any(entry.authoring_path in {("bot", side, *SUFFIX) for side in ("long", "short")}
                   for entry in plan.bindings)
    result = plan.apply_vector(policy, vector_for(plan, **{binding.key: 2.0}))
    assert [_get(result.effective_view()[side], SUFFIX) for side in ("long", "short")] == [2.0, 2.0]


@pytest.mark.parametrize("pin_kind", ["equal_value", "fixed_side_bound", "searched_side_bound"])
def test_side_authority_is_independent_of_global_coordinate(pin_kind):
    raw = {"global": {"risk": {"capacity": 1.0}}}
    bounds = {"global": {"risk": {"capacity": [1.0, 3.0]}}}
    if pin_kind == "equal_value":
        raw["short"] = {"risk": {"capacity": 1.0}}
    else:
        bounds["short"] = {"risk": {"capacity": [1.0, 1.0] if pin_kind == "fixed_side_bound" else [4.0, 6.0]}}
    policy, plan = plan_for(raw, bounds)
    global_binding = capacity_binding(plan, "global")
    short_binding = capacity_binding(plan, "short")
    assert global_binding.effective_targets == (("bot", "long", *SUFFIX),)
    assert short_binding.effective_targets == (("bot", "short", *SUFFIX),)
    vector = vector_for(plan, **{global_binding.key: 3.0, short_binding.key: short_binding.bound.high})
    result = plan.apply_vector(policy, vector)
    assert _get(result.effective_view()["long"], SUFFIX) == 3.0
    assert _get(result.effective_view()["short"], SUFFIX) == short_binding.bound.high
    # Reusable authored output retains the exact coupling on reconstruction.
    reloaded = resolve_bot_policy(result.source_view(include_defaults=True), policy.catalog)
    restored = build_bound_plan(reloaded, bounds, strategy_kind="trailing_martingale", hsl_scope="coin")
    assert restored.identity() == plan.identity()


def test_equal_effective_values_with_different_pins_have_different_identities():
    bounds = {"global": {"risk": {"capacity": [1.0, 3.0]}}}
    inherited, global_plan = plan_for(bounds=bounds)
    pinned, pinned_plan = plan_for({"short": {"risk": {"capacity": 1.0}}}, bounds)
    assert inherited.effective_view() == pinned.effective_view()
    assert global_plan.identity() != pinned_plan.identity()
    with pytest.raises(ValueError, match="does not match source authority"):
        global_plan.apply_vector(pinned, vector_for(global_plan))


def test_omitted_bounds_remain_fixed_without_recommended_search_ranges():
    policy, plan = plan_for({"global": {"risk": {"capacity": 2.0}}, "short": {"risk": {"capacity": 5.0}}})
    assert all(binding.bound.low == binding.bound.high for binding in plan.bindings)
    assert capacity_binding(plan, "global").bound.low == 2.0
    assert capacity_binding(plan, "short").bound.low == 5.0
    result = plan.apply_vector(policy, vector_for(plan))
    assert result.effective_view() == policy.effective_view()


def test_fully_pinned_global_range_is_reported_without_a_dead_coordinate():
    _, plan = plan_for({side: {"risk": {"capacity": 1.0}} for side in ("long", "short")},
                       {"global": {"risk": {"capacity": [1.0, 3.0]}}})
    assert ("optimize", "bounds", "global", *SUFFIX) in plan.unused_bounds
    assert not any(binding.authoring_path == ("bot", "global", *SUFFIX) for binding in plan.bindings)


def test_nullable_policy_has_no_numeric_coordinate_until_domain_is_authored():
    _, fixed = plan_for()
    assert not any(binding.authoring_path[-2:] == ("entry_cooldown", "maximum") for binding in fixed.bindings)
    policy, searched = plan_for(bounds={"global": {"entry_cooldown": {"maximum": [10.0, 20.0]}}})
    target = "bot.global.entry_cooldown.maximum"
    result = searched.apply_vector(policy, vector_for(searched, **{target: 15.0}))
    assert [result.effective_view()[side]["entry_cooldown"]["maximum"] for side in ("long", "short")] == [15.0, 15.0]


def test_portfolio_memory_has_one_coordinate_and_one_native_target():
    policy, plan = plan_for({"portfolio": {"unstuck": {"memory": 43200.0}}},
                            {"portfolio": {"unstuck": {"memory": [60.0, 43200.0]}}})
    binding = next(binding for binding in plan.bindings if binding.authoring_path[1] == "portfolio")
    assert binding.effective_targets == (("bot", "portfolio", "unstuck", "memory"),)
    assert binding.native_ids == ("unstuck_memory",)
    result = plan.apply_vector(policy, vector_for(plan, **{binding.key: 120.0}))
    assert result.effective_view()["portfolio"]["unstuck"]["memory"] == 120.0
    assert "memory" not in result.effective_view()["long"].get("unstuck", {})


def test_derived_overlay_does_not_create_independent_genes_or_pins():
    policy, plan = plan_for(bounds={"global": {"risk": {"capacity": [1.0, 3.0]}}})
    derived = apply_bot_policy_patch(policy, {"bot.short.risk.capacity": 9.0}, channel="derived")
    other = build_bound_plan(derived, {"global": {"risk": {"capacity": [1.0, 3.0]}}}, strategy_kind="trailing_martingale", hsl_scope="coin")
    assert other.identity() == plan.identity()
    assert not derived.has_side_pin("short", SUFFIX)
    result = plan.apply_vector(derived, vector_for(plan, **{"bot.global.risk.capacity": 2.0}))
    assert _get(result.effective_view()["short"], SUFFIX) == 2.0  # Owner must recompute derived overlay.


def test_derived_overlay_does_not_change_an_omitted_fixed_side_pin():
    policy, original = plan_for({"short": {"risk": {"capacity": 2.0}}})
    derived = apply_bot_policy_patch(policy, {"bot.short.risk.capacity": 9.0}, channel="derived")
    rebuilt = build_bound_plan(derived, {}, strategy_kind="trailing_martingale", hsl_scope="coin")
    assert rebuilt.identity() == original.identity()
    assert capacity_binding(rebuilt, "short").bound.low == 2.0
    result = rebuilt.apply_vector(derived, vector_for(rebuilt))
    assert _get(result.effective_view()["short"], SUFFIX) == 2.0


def test_omitted_fixed_value_change_invalidates_the_old_plan():
    policy, plan = plan_for({"short": {"risk": {"capacity": 2.0}}})
    changed = apply_bot_policy_patch(policy, {"bot.short.risk.capacity": 3.0})
    with pytest.raises(ValueError, match="does not match source authority"):
        plan.apply_vector(changed, vector_for(plan))


def test_same_file_with_different_frozen_membership_cannot_reuse_plan():
    def source(coin):
        return {"global": {"universe": {"approved_coins": "./coins.json"}}, **{
            side: {"universe": {"approved_coins": ResolvedCoinList([coin], None, "./coins.json")}}
            for side in ("long", "short")
        }}
    policy, plan = plan_for(source("AAA"))
    other = resolve_bot_policy(source("BBB"), policy.catalog)
    assert plan.identity() != build_bound_plan(other, {}, strategy_kind="trailing_martingale", hsl_scope="coin").identity()
    with pytest.raises(ValueError, match="does not match source authority"):
        plan.apply_vector(other, vector_for(plan))


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_current_registry_native_ids_and_omission_domains_are_preserved(kind):
    declared = real_catalog()
    field = next(field for field in declared.fields if field.suffix[:2] == ("strategy", kind) and field.optimizable)
    source = "./coins.json"
    policy = resolve_bot_policy({"global": {"universe": {"approved_coins": source}}, **{
        side: {"universe": {"approved_coins": ResolvedCoinList(["bybit::BTC/USD:BTC"], None, source)}}
        for side in ("long", "short")
    }}, declared)
    default = float(field.default_value())
    from config.bot_inheritance import _set
    bounds = {"global": {}}
    _set(bounds["global"], field.suffix, [default, default + 0.25])
    plan = build_bound_plan(policy, bounds, strategy_kind=kind, hsl_scope="coin")
    binding = next(binding for binding in plan.bindings if binding.authoring_path == ("bot", "global", *field.suffix))
    assert binding.native_ids == tuple(field.native_key(side) for side in ("long", "short"))
    assert not any(entry.authoring_path[2:4] == ("strategy", "ema_anchor" if kind == "trailing_martingale" else "trailing_martingale")
                   for entry in plan.bindings)
    result = plan.apply_vector(policy, vector_for(plan, **{binding.key: default + 0.125}))
    assert _get(result.effective_view()["short"], field.suffix) == default + 0.125
    prepared = result.effective_view()["short"]["universe"]["approved_coins"]
    assert type(prepared) is ResolvedCoinList
    assert prepared == ["bybit::BTC/USD:BTC"] and prepared.shared_source == source
    omitted = build_bound_plan(policy, {}, strategy_kind=kind, hsl_scope="coin")
    assert all(entry.bound.low == entry.bound.high for entry in omitted.bindings)


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("scope", ["coin", "position_side", "portfolio"])
def test_current_registry_hsl_bindings_follow_explicit_scope(kind, scope):
    policy = resolve_bot_policy({}, real_catalog())
    plan = build_bound_plan(policy, {}, strategy_kind=kind, hsl_scope=scope)
    side_hsl = [binding for binding in plan.bindings
                if binding.authoring_path[1] != "portfolio" and binding.authoring_path[2] == "hsl"]
    assert bool(side_hsl) is (scope != "portfolio")
    if scope == "portfolio":
        for side in ("global", "long", "short"):
            with pytest.raises(ValueError, match="inactive in portfolio HSL scope"):
                build_bound_plan(
                    policy, {side: {"hsl": {"drawdown_ratio_threshold": [0.1, 0.2]}}},
                    strategy_kind=kind, hsl_scope=scope,
                )


def test_hsl_scope_is_part_of_plan_identity_even_with_identical_native_ids():
    policy = resolve_bot_policy({}, catalog())
    coin = build_bound_plan(policy, {}, strategy_kind="trailing_martingale", hsl_scope="coin")
    side = build_bound_plan(policy, {}, strategy_kind="trailing_martingale", hsl_scope="position_side")
    assert coin.bindings == side.bindings
    assert coin.identity() != side.identity()


@pytest.mark.parametrize("bounds", [
    {"global": {"enabled": [0.0, 1.0]}},
    {"global": {"risk": {"unknown": [0.0, 1.0]}}},
    {"global": {"risk": {"capacity": [False, 1.0]}}},
    {"global": {"risk": {"capacity": [float("inf"), 1.0]}}},
])
def test_invalid_search_authority_is_not_inferred(bounds):
    with pytest.raises((ValueError, TypeError)):
        plan_for(bounds=bounds)


def test_numeric_gene_reuse_is_allowed_but_source_and_enablement_edits_are_not():
    policy, plan = plan_for(bounds={"global": {"risk": {"capacity": [1.0, 3.0]}}})
    first = plan.apply_vector(policy, vector_for(plan, **{"bot.global.risk.capacity": 2.0}))
    second = plan.apply_vector(first, vector_for(plan, **{"bot.global.risk.capacity": 3.0}))
    assert _get(second.effective_view()["long"], SUFFIX) == 3.0
    changed = apply_bot_policy_patch(policy, {"bot.global.enabled": True})
    with pytest.raises(ValueError, match="does not match source authority"):
        plan.apply_vector(changed, vector_for(plan))
