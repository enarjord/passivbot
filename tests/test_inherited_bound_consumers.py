"""Prepared optimizer consumers retain logical coordinates and raw search authority."""

from copy import deepcopy
import json

import pytest

from config.bot_inheritance import install_bot_policy, resolve_bot_policy
from config.inherited_bounds import (get_bound_plan, optimize_bounds_source_view,
                                     prepare_bound_plan, resolve_plan_selectors)
from test_bot_inheritance import catalog


def prepared(source=None, bounds=None):
    source = {} if source is None else deepcopy(source)
    source["portfolio"] = {"strategy_kind": "trailing_martingale", "hsl": {"scope": "coin"}}
    config = {"optimize": {"bounds": {} if bounds is None else deepcopy(bounds), "fixed_params": []}}
    install_bot_policy(config, resolve_bot_policy(source, catalog()))
    prepare_bound_plan(config)
    return config


def test_prepared_bounds_runtime_expansion_and_authored_projection_are_separate():
    raw = {"global": {"risk": {"capacity": [1.0, 3.0, 0.5]}}}
    config = prepared(bounds=raw)
    plan = get_bound_plan(config)
    gene = next(binding for binding in plan.bindings if binding.authoring_path[-2:] == ("risk", "capacity"))
    assert gene.native_ids == ("long_risk_capacity", "short_risk_capacity")
    assert config["optimize"]["bounds"]["long"]["risk"]["capacity"] == [1.0, 3.0, 0.5]
    assert config["optimize"]["bounds"]["short"]["risk"]["capacity"] == [1.0, 3.0, 0.5]
    assert optimize_bounds_source_view(config)["optimize"]["bounds"] == raw
    assert get_bound_plan(deepcopy(config)).identity() == plan.identity()
    assert get_bound_plan(json.loads(json.dumps(config))) is None


@pytest.mark.parametrize("selector", ["global.risk.capacity", "bot.global.risk.capacity",
    "long.risk.capacity", "bot.short.risk.capacity", "short_risk_capacity",
    "risk.capacity", "optimize.bounds.global.risk.capacity"])
def test_effective_or_native_selector_freezes_one_shared_coordinate(selector):
    config = prepared(bounds={"global": {"risk": {"capacity": [1.0, 3.0]}}})
    selected = resolve_plan_selectors(config, [selector])
    assert selected == {"bot.global.risk.capacity": ("bot", "global", "risk", "capacity")}


def test_explicit_side_domain_separates_ownership_without_duplicating_global_gene():
    config = prepared(bounds={"global": {"risk": {"capacity": [1.0, 3.0]}},
                              "short": {"risk": {"capacity": [4.0, 6.0]}}})
    assert resolve_plan_selectors(config, ["short.risk.capacity"]) == {
        "bot.short.risk.capacity": ("bot", "short", "risk", "capacity")}
    assert resolve_plan_selectors(config, ["global.risk.capacity"]) == {
        "bot.global.risk.capacity": ("bot", "global", "risk", "capacity")}


def test_direct_expanded_bound_edits_cannot_silently_change_authored_authority():
    config = prepared(bounds={"global": {"risk": {"capacity": [1.0, 3.0]}}})
    config["optimize"]["bounds"]["short"]["risk"]["capacity"] = [4.0, 6.0]
    with pytest.raises(ValueError, match="rebuild"):
        get_bound_plan(config)
    with pytest.raises(ValueError, match="rebuild"):
        optimize_bounds_source_view(config)
    plan = prepare_bound_plan(config, authored_bounds={"short": {"risk": {"capacity": [4.0, 6.0]}}})
    assert any(binding.authoring_path[:2] == ("bot", "short") for binding in plan.bindings)


def test_reprepare_keeps_original_domains_and_unused_common_authority():
    raw = {"global": {"risk": {"capacity": [1.0, 3.0]}}}
    config = prepared(source={side: {"risk": {"capacity": 1.0}} for side in ("long", "short")}, bounds=raw)
    original = get_bound_plan(config)
    rebuilt = prepare_bound_plan(config)
    assert rebuilt.identity() == original.identity()
    assert rebuilt.unused_bounds == (("optimize", "bounds", "global", "risk", "capacity"),)
    assert optimize_bounds_source_view(config)["optimize"]["bounds"] == raw


def test_prepared_source_capture_order_cannot_change_binding_identity():
    from dataclasses import replace
    from config.bot_inheritance import get_bot_policy
    from config.inherited_bounds import build_bound_plan
    from config.universe import materialize_side_universe_sources
    config = prepared(source={"global": {"universe": {"approved_coins": ["BTC", "ETH"],
                                                        "ignored_coins": ["SOL", "XRP"]}}})
    for field, values in (("approved_coins", ["BTC", "ETH"]), ("ignored_coins", ["SOL", "XRP"])):
        materialize_side_universe_sources(config, field, {side: values for side in ("long", "short")})
    policy = get_bot_policy(config)
    assert len(policy._prepared) == 4
    reverse = replace(policy, _prepared=tuple(reversed(policy._prepared)))
    kwargs = {"strategy_kind": "trailing_martingale", "hsl_scope": "coin"}
    first = build_bound_plan(policy, {}, **kwargs)
    second = build_bound_plan(reverse, {}, **kwargs)
    assert first.source_signature == second.source_signature
    assert first.identity() == second.identity()
    assert reverse.effective_view()["long"]["universe"]["approved_coins"] == ["BTC", "ETH"]


def test_fixed_target_selector_freezes_shared_gene_at_current_value_outside_prior_range():
    from config.bot_inheritance import get_bot_policy
    config = prepared(bounds={"global": {"risk": {"capacity": [4.0, 6.0]}}})
    original = optimize_bounds_source_view(config)["optimize"]["bounds"]
    config["optimize"]["fixed_params"] = ["short.risk.capacity"]
    plan = prepare_bound_plan(config)
    binding = next(binding for binding in plan.bindings if binding.key == "bot.global.risk.capacity")
    assert binding.bound.low == binding.bound.high == 1.0
    assert binding.effective_targets == (("bot", "long", "risk", "capacity"), ("bot", "short", "risk", "capacity"))
    assert optimize_bounds_source_view(config)["optimize"]["bounds"] == original
    candidate = plan.apply_vector(get_bot_policy(config), [item.bound.low for item in plan.bindings])
    assert candidate.effective_view()["long"]["risk"]["capacity"] == 1.0
    assert candidate.effective_view()["short"]["risk"]["capacity"] == 1.0
