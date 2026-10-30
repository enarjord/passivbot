"""Authored leaf presence is independent of effective values and source masks."""

from copy import deepcopy

import pytest

from config.bot_inheritance import SideFieldCatalog, apply_bot_policy_patch, resolve_bot_policy
from config.field_spec import FieldRegistry, FieldSpec, FieldType
from config.universe import ResolvedCoinList


def test_policy_path_lookup_preserves_owned_and_foreign_mapping_values():
    from collections import UserDict
    from types import MappingProxyType
    from config.bot_inheritance import ABSENT, _get

    leaf = object()
    for wrap in (dict, UserDict, MappingProxyType):
        nested = wrap({"leaf": leaf, "nullable": None})
        node = wrap({"nested": nested})
        assert _get(node, ("nested", "leaf")) is leaf
        assert _get(node, ("nested", "nullable")) is None
        assert _get(node, ("nested", "missing")) is ABSENT
        fallback = object()
        assert _get(node, ("missing",), fallback) is fallback
    for node in (None, 17, [], object()):
        assert _get(node, ()) is node
        assert _get(node, ("missing",)) is ABSENT
    assert _get({"nested": None}, ("nested", "missing")) is ABSENT


def test_policy_path_lookup_retains_custom_membership_and_indexing():
    from config.bot_inheritance import _get

    events = []

    class Virtual(dict):
        def __contains__(self, key):
            events.append(("contains", key))
            return key == "virtual"

        def __getitem__(self, key):
            events.append(("getitem", key))
            return {"leaf": 3.25}

    assert _get(Virtual(), ("virtual", "leaf")) == 3.25
    assert events == [("contains", "virtual"), ("getitem", "virtual")]

    class Broken(dict):
        def __contains__(self, key):
            raise RuntimeError("membership failed")

    with pytest.raises(RuntimeError, match="membership failed"):
        _get(Broken(), ("value",))

    class BrokenLookup(Virtual):
        def __getitem__(self, key):
            raise KeyError("indexing failed")

    with pytest.raises(KeyError, match="indexing failed"):
        _get(BrokenLookup(), ("virtual",))


def test_policy_path_lookup_keeps_typing_mapping_class_spoof_behavior():
    from config.bot_inheritance import _get

    class MappingView:
        @property
        def __class__(self):
            return dict

        def __contains__(self, key):
            return key == "value"

        def __getitem__(self, key):
            return 7.5

    assert _get(MappingView(), ("value",)) == 7.5


def catalog():
    fields = []
    defaults = {
        "risk": {"capacity": 1.0},
        "entry_cooldown": {"maximum": None},
        "forager": {"always_include": [], "score_weights": {
            "volume": 0.2, "log_range": 0.3, "ema_readiness": 0.5, "unilateralness": 0.0,
        }},
        "universe": {"approved_coins": ["AAA"], "ignored_coins": []},
    }
    declarations = [
        (("risk", "capacity"), FieldType.FLOAT, False, True),
        (("enabled",), FieldType.BOOLEAN, False, False),
        (("entry_cooldown", "maximum"), FieldType.FLOAT, True, True),
        (("forager", "always_include"), FieldType.ARRAY, False, False),
        *(( ("forager", "score_weights", name), FieldType.FLOAT, False, True)
          for name in defaults["forager"]["score_weights"]),
        *(( ("universe", name), (FieldType.ARRAY, FieldType.STRING), True, False)
          for name in ("approved_coins", "ignored_coins")),
    ]
    for side in ("long", "short"):
        for suffix, types, nullable, optimized in declarations:
            fields.append(FieldSpec(
                ("bot", side, *suffix), types if isinstance(types, tuple) else (types,),
                nullable=nullable, optimizable=optimized,
                optimize_key=f"{side}_{'_'.join(suffix)}" if optimized else None,
            ))
    fields.append(FieldSpec(("bot", "portfolio", "unstuck", "memory"),
                            (FieldType.FLOAT,), optimizable=True, optimize_key="unstuck_memory"))
    return SideFieldCatalog.from_registry(FieldRegistry(fields), common_defaults=defaults)


def real_catalog():
    from config.field_registry import get_field_registry
    from config.bot_defaults import get_common_bot_defaults
    defaults = get_common_bot_defaults()
    # An explicit test profile, not an implicit selection made by the resolver.
    defaults["universe"]["approved_coins"] = ["AAA"]
    return SideFieldCatalog.from_registry(get_field_registry(), common_defaults=defaults)


def test_leaf_precedence_preserves_false_zero_empty_and_nullable_values():
    policy = resolve_bot_policy({
        "long": {"enabled": True}, "global": { "risk": {"capacity": 2.0},
                   "forager": {"always_include": ["AAA"]},
                   "entry_cooldown": {"maximum": 30.0}},
        "short": {"enabled": False, "risk": {"capacity": 0.0},
                  "forager": {"always_include": []}, "entry_cooldown": {"maximum": None}},
    }, catalog())
    effective, origins = policy.resolve()
    assert effective["long"]["enabled"] is True
    assert effective["short"]["enabled"] is False
    assert effective["short"]["risk"]["capacity"] == 0.0
    assert effective["short"]["forager"]["always_include"] == []
    assert effective["short"]["entry_cooldown"]["maximum"] is None
    assert next(origin for origin in origins if origin.target == ("bot", "short", "enabled")).kind == "side"


def test_equal_side_pin_survives_global_edit_and_source_roundtrip():
    policy = resolve_bot_policy({"global": {"risk": {"capacity": 1.0}},
                                 "short": {"risk": {"capacity": 1.0}}}, catalog())
    changed = apply_bot_policy_patch(policy, {"bot.global.risk.capacity": 3.0})
    assert changed.effective_view()["long"]["risk"]["capacity"] == 3.0
    assert changed.effective_view()["short"]["risk"]["capacity"] == 1.0
    assert policy.effective_view()["long"]["risk"]["capacity"] == 1.0
    exported = changed.source_view(include_defaults=True)
    assert exported["short"] == {"risk": {"capacity": 1.0}}
    assert "long" not in exported
    reloaded = resolve_bot_policy(exported, policy.catalog)
    assert reloaded.effective_view() == changed.effective_view()


def test_source_nulls_have_field_specific_meanings():
    policy = resolve_bot_policy({
        "global": {"universe": {"approved_coins": ["AAA"], "ignored_coins": ["BBB"]}},
        "short": {"universe": {"approved_coins": None, "ignored_coins": None}},
    }, catalog())
    effective = policy.effective_view()
    assert effective["short"]["universe"]["approved_coins"] == ["AAA"]
    assert effective["short"]["universe"]["ignored_coins"] == []
    assert effective["long"]["universe"]["ignored_coins"] == ["BBB"]
    assert not policy.has_side_pin("short", ("universe", "approved_coins"))
    assert policy.has_side_pin("short", ("universe", "ignored_coins"))

    clear_global = resolve_bot_policy({"global": {"universe": {"ignored_coins": None}}}, catalog())
    assert clear_global.source_view()["global"]["universe"]["ignored_coins"] is None
    assert all(clear_global.effective_view()[scope]["universe"]["ignored_coins"] == []
               for scope in ("global", "long", "short"))


def test_owned_atomic_patch_does_not_refill_missing_weight_leaves():
    initial = {"global": {"forager": {"score_weights": {"volume": 0.1}}}}
    policy = resolve_bot_policy(initial, catalog())
    assert policy.effective_view()["long"]["forager"]["score_weights"]["ema_readiness"] == 0.5
    with pytest.raises(ValueError, match="missing required keys"):
        apply_bot_policy_patch(policy, {"bot.long.forager.score_weights": {"volume": 0.1}})
    complete = policy.effective_view()["long"]["forager"]["score_weights"]
    result = apply_bot_policy_patch(policy, {"bot.long.forager.score_weights": complete})
    assert result.source_view()["long"]["forager"]["score_weights"] == complete


def test_derived_values_are_not_authoring_pins_or_mutable_shared_state():
    policy = resolve_bot_policy({}, catalog())
    derived = apply_bot_policy_patch(policy, {"bot.short.risk.capacity": 4.0}, channel="derived")
    assert derived.source_view() == {}
    assert derived.effective_view()["short"]["risk"]["capacity"] == 4.0
    assert not derived.has_side_pin("short", ("risk", "capacity"))
    assert derived.candidate_view()["short"] == {"risk": {"capacity": 4.0}}
    assert "short" not in derived.source_view(include_defaults=True)
    updated = apply_bot_policy_patch(derived, {"bot.global.risk.capacity": 2.0})
    assert updated.effective_view()["short"]["risk"]["capacity"] == 2.0
    view = derived.effective_view()
    view["short"]["risk"]["capacity"] = 100.0
    assert derived.effective_view()["short"]["risk"]["capacity"] == 4.0


def test_prepared_source_objects_survive_numeric_edits_without_io(monkeypatch):
    from pathlib import Path
    source = "./approved.json"
    raw = {"global": {"universe": {"approved_coins": source}}, **{
        side: {"universe": {"approved_coins": ResolvedCoinList(["bybit::BTC/USD:BTC"], None, source)}}
        for side in ("long", "short")
    }}
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: pytest.fail("inheritance must not open source files"))
    policy = resolve_bot_policy(raw, catalog())
    changed = apply_bot_policy_patch(policy, {"bot.global.risk.capacity": 3.0})
    members = changed.effective_view()["long"]["universe"]["approved_coins"]
    assert type(members) is ResolvedCoinList and members == ["bybit::BTC/USD:BTC"]
    assert members.source is None and members.shared_source == source
    members.append("CCC")
    assert changed.effective_view()["long"]["universe"]["approved_coins"] == ["bybit::BTC/USD:BTC"]
    # An explicit same-file replacement invalidates preparation; its owner may
    # rebind once, rather than silently reusing or fabricating membership here.
    rebound = apply_bot_policy_patch(changed, {"bot.global.universe.approved_coins": source})
    assert rebound.effective_view()["long"]["universe"]["approved_coins"] == source
    assert type(rebound.effective_view()["long"]["universe"]["approved_coins"]) is str


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_real_registry_preserves_prepared_sources_across_strategy_edits(kind, monkeypatch):
    from pathlib import Path
    from config.bot_inheritance import _get

    declared = real_catalog()
    field = next(field for field in declared.fields
                 if field.suffix[:2] == ("strategy", kind) and field.optimizable)
    source = "./approved.json"
    raw = {"global": {"universe": {"approved_coins": source}}, **{
        side: {"universe": {
            "approved_coins": ResolvedCoinList(["bybit::BTC/USD:BTC"], None, source),
            "ignored_coins": ResolvedCoinList([], [], None),
        }} for side in ("long", "short")
    }}
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: pytest.fail("unexpected file read"))
    policy = resolve_bot_policy(raw, declared)
    changed = apply_bot_policy_patch(
        policy, {".".join(("bot", "global", *field.suffix)): float(field.default_value()) + 0.125},
    )
    for side in ("long", "short"):
        values = changed.effective_view()[side]
        members = values["universe"]["approved_coins"]
        assert type(members) is ResolvedCoinList
        assert members == ["bybit::BTC/USD:BTC"]
        assert members.source is None and members.shared_source == source
        assert type(values["universe"]["ignored_coins"]) is ResolvedCoinList
        assert _get(values, field.suffix) == float(field.default_value()) + 0.125
        members.clear()
        assert raw[side]["universe"]["approved_coins"] == ["bybit::BTC/USD:BTC"]
        assert changed.effective_view()[side]["universe"]["approved_coins"] == ["bybit::BTC/USD:BTC"]
    exported = changed.source_view(include_defaults=True)
    assert exported["global"]["universe"]["approved_coins"] == source
    assert all(exported[side]["universe"]["approved_coins"] is None for side in ("long", "short"))


def test_authored_patch_replaces_gene_origin_even_when_the_value_is_equal():
    policy = resolve_bot_policy({}, catalog())
    path = ("bot", "global", "risk", "capacity")
    gene = apply_bot_policy_patch(policy, {".".join(path): 2.0}, channel="gene")
    assert all(origin.kind == "gene" for origin in gene.resolve()[1]
               if origin.target[-2:] == ("risk", "capacity"))
    authored = apply_bot_policy_patch(gene, {".".join(path): 2.0})
    assert all(origin.kind == "global" for origin in authored.resolve()[1]
               if origin.target[-2:] == ("risk", "capacity"))


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_real_registry_strategy_fields_resolve_without_mutating_defaults(kind):
    from config.schema import get_template_config
    before = get_template_config()
    declared = real_catalog()
    field = next(field for field in declared.fields if field.suffix[:2] == ("strategy", kind) and field.optimizable)
    value = float(field.default_value())
    node = {}
    from config.bot_inheritance import _set
    _set(node, field.suffix, value + 0.125)
    policy = resolve_bot_policy({"global": node}, declared)
    from config.bot_inheritance import _get
    assert _get(policy.effective_view()["long"], field.suffix) == value + 0.125
    assert _get(policy.effective_view()["short"], field.suffix) == value + 0.125
    assert get_template_config() == before

    # The selected-engine parent is validated by its existing owner; this
    # structural core must not demand every inactive engine descendant too.
    replacement = apply_bot_policy_patch(policy, {"bot.long.strategy": {kind: node["strategy"][kind]}})
    assert _get(replacement.effective_view()["long"], field.suffix) == value + 0.125


@pytest.mark.parametrize("raw", [
    {"global": {"enabled": None}}, {"short": {"risk": {"capacity": True}}},
    {"global": {"risk": {"capacity": float("nan")}}}, {"global": {"retired": 1}},
    {"global": {"universe": {"approved_coins": None}}},
])
def test_invalid_or_unknown_values_fail_before_default_hydration(raw):
    with pytest.raises((TypeError, ValueError)):
        resolve_bot_policy(raw, catalog())


def test_plain_json_cannot_supply_policy_authority_or_overlapping_patches():
    policy = resolve_bot_policy({}, catalog())
    with pytest.raises(TypeError, match="typed BotPolicy"):
        apply_bot_policy_patch({"_source": {}, "catalog": {}}, {"bot.global.enabled": True})
    with pytest.raises(ValueError, match="overlapping"):
        apply_bot_policy_patch(policy, {"bot.long.risk": {"capacity": 1.0}, "bot.long.risk.capacity": 2.0})


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_coin_projection_uses_existing_permissions_after_inheritance(kind):
    from config.bot_inheritance import resolve_coin_bot_policy
    declared = real_catalog()
    policy = resolve_bot_policy({
        "global": {"unstuck": {"close_fraction": 0.5}},
        "short": {"unstuck": {"close_fraction": 0.75}},
    }, declared)
    patch = {"bot": {"short": {"unstuck": {"close_fraction": 0.125}}}}
    original = deepcopy(patch)
    effective, origins = resolve_coin_bot_policy(
        policy, patch, coin="bybit::BTC/USD:BTC", strategy_kind=kind,
    )
    assert effective["long"]["unstuck"]["close_fraction"] == 0.5
    assert effective["short"]["unstuck"]["close_fraction"] == 0.125
    assert patch == original
    origin = next(origin for origin in origins if origin.target == ("bot", "short", "unstuck", "close_fraction"))
    assert origin.kind == "coin" and origin.source[:2] == ("coin_overrides", "bybit::BTC/USD:BTC")
    with pytest.raises(ValueError):
        resolve_coin_bot_policy(policy, {"bot": {"short": {"universe": {"minimum_coin_age_days": 0.0}}}},
                                coin="BTC", strategy_kind=kind)
    with pytest.raises(ValueError, match="materialized override files"):
        resolve_coin_bot_policy(policy, {"override_config_path": "./coin.json"}, coin="BTC", strategy_kind=kind)


def test_catalog_requires_explicit_compatible_common_defaults():
    from config.field_registry import get_field_registry
    with pytest.raises(ValueError, match="missing explicit common default"):
        SideFieldCatalog.from_registry(get_field_registry(), common_defaults={})


def test_prepared_envelope_retains_one_typed_source_and_exports_sparse_pins():
    import json
    from config.bot_inheritance import (
        bot_policy_source_view, get_bot_policy, install_bot_policy,
    )

    policy = resolve_bot_policy({"short": {"risk": {"capacity": 1.0}}}, catalog())
    config = {"config_version": "v9.0.0"}
    install_bot_policy(config, policy)
    assert get_bot_policy(config) is policy
    exported = bot_policy_source_view(config, include_defaults=True)
    assert exported["bot"]["short"] == {"risk": {"capacity": 1.0}}
    assert "long" not in exported["bot"]
    assert get_bot_policy(json.loads(json.dumps(config))) is None
    exported["bot"]["short"]["risk"]["capacity"] = 9.0
    assert policy.source_view()["short"]["risk"]["capacity"] == 1.0


def test_direct_changed_values_are_adopted_before_source_projection():
    from config.bot_inheritance import bot_policy_source_view, install_bot_policy

    policy = resolve_bot_policy({}, catalog())
    config = {}
    install_bot_policy(config, policy)
    config["bot"]["long"]["enabled"] = True
    config["bot"]["global"]["risk"]["capacity"] = 3.0
    config["bot"]["short"]["risk"]["capacity"] = 2.0
    source = bot_policy_source_view(config)["bot"]
    reloaded = resolve_bot_policy(source, policy.catalog)
    assert reloaded.effective_view()["long"]["enabled"] is True
    assert reloaded.effective_view()["short"]["enabled"] is False
    assert reloaded.effective_view()["long"]["risk"]["capacity"] == 3.0
    assert reloaded.effective_view()["short"]["risk"]["capacity"] == 2.0
    assert "enabled" not in source["short"]


def test_owned_gene_and_derived_baselines_do_not_become_authored_side_pins():
    from config.bot_inheritance import bot_policy_source_view, install_bot_policy

    policy = apply_bot_policy_patch(resolve_bot_policy({}, catalog()),
                                    {"bot.global.risk.capacity": 3.0}, channel="gene")
    policy = apply_bot_policy_patch(policy, {"bot.short.risk.capacity": 2.0}, channel="derived")
    config = {}
    install_bot_policy(config, policy)
    source = bot_policy_source_view(config)["bot"]
    assert source == {"global": {"risk": {"capacity": 3.0}}}
    candidate = bot_policy_source_view(config, candidate=True)["bot"]
    assert candidate["short"]["risk"]["capacity"] == 2.0


def test_direct_prepared_membership_mutation_cannot_become_source_authority():
    from config.bot_inheritance import (
        bot_policy_source_view, install_bot_policy, with_prepared_bot_sources,
    )

    policy = resolve_bot_policy({}, catalog())
    effective = policy.effective_view()
    effective["long"]["universe"]["approved_coins"] = ResolvedCoinList(
        ["AAA"], None, ["AAA"], source_present=False,
    )
    policy = with_prepared_bot_sources(policy, effective)
    config = {}
    install_bot_policy(config, policy)
    config["bot"]["long"]["universe"]["approved_coins"].append("BBB")
    with pytest.raises(ValueError, match="prepared membership edits"):
        bot_policy_source_view(config)


def test_field_specific_prepared_sources_keep_ignored_omission_distinct_from_null(monkeypatch):
    from pathlib import Path
    from config.bot_inheritance import with_prepared_bot_sources

    source = "./ignored.json"
    policy = resolve_bot_policy({
        "global": {"universe": {"ignored_coins": source}},
        "short": {"universe": {"ignored_coins": None}},
    }, catalog())
    effective = policy.effective_view()
    effective["long"]["universe"]["ignored_coins"] = ResolvedCoinList(
        ["BBB"], None, source, source_present=False,
    )
    # A legacy prepared ignored view carried the APPROVED shared source. Its
    # irrelevant attribute must not discard an explicit null clear.
    effective["short"]["universe"]["ignored_coins"] = ResolvedCoinList([], None, ["AAA"])
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: pytest.fail("unexpected source read"))
    frozen = with_prepared_bot_sources(policy, effective)
    assert "universe" not in frozen.source_view().get("long", {})
    assert frozen.source_view()["short"]["universe"]["ignored_coins"] is None
    changed = apply_bot_policy_patch(frozen, {"bot.global.risk.capacity": 3.0})
    assert type(changed.effective_view()["long"]["universe"]["ignored_coins"]) is ResolvedCoinList
    assert changed.effective_view()["long"]["universe"]["ignored_coins"] == ["BBB"]
    assert changed.effective_view()["long"]["universe"]["ignored_coins"].source_present is False
    assert changed.effective_view()["short"]["universe"]["ignored_coins"] == []
    redeclared = apply_bot_policy_patch(changed, {"bot.global.universe.ignored_coins": source})
    assert type(redeclared.effective_view()["long"]["universe"]["ignored_coins"]) is str
    assert redeclared.effective_view()["short"]["universe"]["ignored_coins"] == []


def test_prepared_source_capture_rejects_a_different_shared_recipe():
    from config.bot_inheritance import with_prepared_bot_sources

    policy = resolve_bot_policy({"global": {"universe": {"ignored_coins": "./ignored.json"}}}, catalog())
    effective = policy.effective_view()
    effective["long"]["universe"]["ignored_coins"] = ResolvedCoinList(
        ["BBB"], None, "./approved.json", source_present=False,
    )
    with pytest.raises(ValueError, match="does not match .* source"):
        with_prepared_bot_sources(policy, effective)


def test_exact_prepared_source_input_keeps_absence_instead_of_creating_null_clear():
    source = "./ignored.json"
    policy = resolve_bot_policy({
        "global": {"universe": {"ignored_coins": source}},
        "long": {"universe": {"ignored_coins": ResolvedCoinList(
            ["BBB"], None, source, source_present=False,
        )}},
        "short": {"universe": {"ignored_coins": ResolvedCoinList([], None)}},
    }, catalog())
    assert "ignored_coins" not in policy.source_view()["long"]["universe"]
    assert policy.source_view()["short"]["universe"]["ignored_coins"] is None
    assert policy.effective_view()["long"]["universe"]["ignored_coins"] == ["BBB"]
    assert policy.effective_view()["short"]["universe"]["ignored_coins"] == []


def test_prepared_capture_requires_presence_to_match_the_original_source():
    from config.bot_inheritance import with_prepared_bot_sources

    policy = resolve_bot_policy({}, catalog())
    effective = policy.effective_view()
    effective["long"]["universe"]["approved_coins"] = ResolvedCoinList(["AAA"], None, ["AAA"])
    with pytest.raises(ValueError, match="does not match .* source"):
        with_prepared_bot_sources(policy, effective)


@pytest.mark.parametrize("source", [
    {}, [], {"nested": [None, False, 0, 1.25, "value", {"empty": []}]},
])
def test_frozen_policy_payloads_share_only_immutable_representation_on_copy(source):
    from config.bot_inheritance import _freeze, _thaw

    frozen = _freeze(source)
    memo = {}
    assert deepcopy(frozen, memo) is frozen
    assert memo[id(frozen)] is frozen
    first, second = _thaw(frozen), _thaw(frozen)
    assert first == second == source
    assert first is not second
    if isinstance(first, dict):
        first["changed"] = []
    else:
        first.append("changed")
    assert _thaw(frozen) == source


@pytest.mark.parametrize("kind", ["object", "array"])
def test_mutable_frozen_payloads_keep_copy_aliases_and_cycles(kind):
    from config.bot_inheritance import _Array, _Object

    child = []
    value = (_Object((("left", child), ("right", child))) if kind == "object"
             else _Array((child, child)))
    child.append(value)
    copied = deepcopy(value)
    left, right = ((copied.items[0][1], copied.items[1][1]) if kind == "object"
                   else copied.items)
    assert copied is not value
    assert left is right and left is not child
    assert left[0] is copied
    left.append("changed")
    assert len(child) == 1


def test_cyclic_frozen_wrapper_is_copied_without_immutable_fast_path_recursion():
    from config.bot_inheritance import _Array

    value = _Array(())
    object.__setattr__(value, "items", (value,))
    copied = deepcopy(value)
    assert copied is not value
    assert copied.items[0] is copied


@pytest.mark.parametrize("scalar_type, scalar", [(int, 3), (float, 1.25), (str, "value")])
def test_scalar_subclasses_in_frozen_payloads_keep_mutable_state_isolated(scalar_type, scalar):
    from config.bot_inheritance import _freeze, _thaw

    class Scalar(scalar_type):
        pass

    value = Scalar(scalar)
    value.extra = []
    frozen = _freeze({"value": value})
    copied = deepcopy(frozen)
    copied_value = _thaw(copied)["value"]
    assert copied is not frozen
    assert type(copied_value) is Scalar
    assert copied_value is not value
    copied_value.extra.append("changed")
    assert value.extra == []


def test_string_subclass_object_key_keeps_copy_isolation():
    from config.bot_inheritance import _freeze

    class Key(str):
        pass

    key = Key("value")
    key.extra = []
    frozen = _freeze({key: 1})
    copied = deepcopy(frozen)
    copied_key = copied.items[0][0]
    assert copied is not frozen
    assert copied_key is not key and type(copied_key) is Key
    copied_key.extra.append("changed")
    assert key.extra == []


@pytest.mark.parametrize("kind", ["scalar", "wrapper"])
def test_metaclass_equality_cannot_grant_immutable_payload_sharing(kind):
    from config.bot_inheritance import _Array

    class MasqueradingType(type):
        def __eq__(cls, other):
            return True

        __hash__ = type.__hash__

    if kind == "scalar":
        class ForeignScalar(int, metaclass=MasqueradingType):
            pass

        child = ForeignScalar(1)
        child.extra = []
        value = _Array((child,))
        copied = deepcopy(value)
        assert copied is not value and copied.items[0] is not child
        copied.items[0].extra.append("changed")
        assert child.extra == []
    else:
        class ForeignArray(_Array, metaclass=MasqueradingType):
            pass

        child = []
        value = ForeignArray((child,))
        copied = deepcopy(value)
        assert type(copied) is ForeignArray and copied is not value
        assert copied.items[0] is not child
        copied.items[0].append("changed")
        assert child == []


def test_frozen_payload_extra_state_and_subclass_slots_are_copied():
    from config.bot_inheritance import _Array

    class SlottedArray(_Array):
        __slots__ = ("extra",)

        def __init__(self, items, extra):
            super().__init__(items)
            object.__setattr__(self, "extra", extra)

    for value in (_Array((1,)), SlottedArray((1,), [])):
        object.__setattr__(value, "extra", [])
        copied = deepcopy(value)
        assert type(copied) is type(value) and copied is not value
        copied.extra.append("changed")
        assert value.extra == []


def test_frozen_wrapper_subclass_keeps_custom_reduction_and_state():
    from config.bot_inheritance import _Array

    calls = []

    class ReducedArray(_Array):
        def __reduce_ex__(self, protocol):
            calls.append(protocol)
            return (ReducedArray, (self.items,), {"extra": self.extra})

        def __setstate__(self, state):
            object.__setattr__(self, "extra", state["extra"])
            calls.append("state")

    value = ReducedArray((1,))
    object.__setattr__(value, "extra", [])
    copied = deepcopy(value)
    assert calls == [4, "state"]
    assert type(copied) is ReducedArray and copied is not value
    copied.extra.append("changed")
    assert value.extra == []


def test_frozen_wrapper_subclass_keeps_reduction_item_aliases():
    from config.bot_inheritance import _Array

    class ContainerArray(_Array):
        def __reduce_ex__(self, protocol):
            return (ContainerArray, (self.items,), {"sequence": [], "mapping": {}},
                    iter(self.sequence), iter(self.mapping.items()))

        def append(self, item):
            self.sequence.append(item)

        def __setitem__(self, key, item):
            self.mapping[key] = item

    child = []
    value = ContainerArray((child,))
    object.__setattr__(value, "sequence", [child])
    object.__setattr__(value, "mapping", {"same": child})
    copied = deepcopy(value)
    copied_child = copied.items[0]
    assert copied_child is copied.sequence[0] is copied.mapping["same"]
    assert copied_child is not child


def test_frozen_wrapper_subclass_keeps_keyword_new_arguments():
    from config.bot_inheritance import _Array

    class KeywordArray(_Array):
        def __new__(cls, items, *, token):
            value = super().__new__(cls)
            object.__setattr__(value, "token", token)
            return value

        def __init__(self, items, *, token):
            super().__init__(items)

        def __getnewargs_ex__(self):
            return (self.items,), {"token": self.token}

    value = KeywordArray((1,), token=[])
    copied = deepcopy(value)
    assert type(copied) is KeywordArray and copied is not value
    copied.token.append("changed")
    assert value.token == []


def test_frozen_wrapper_subclass_keeps_registered_and_atomic_reductions(monkeypatch):
    from copyreg import dispatch_table
    from config.bot_inheritance import _Array

    class RegisteredArray(_Array):
        def __reduce_ex__(self, protocol):
            pytest.fail("registered reduction must retain precedence")

    monkeypatch.setitem(dispatch_table, RegisteredArray,
                        lambda value: (_Array, (value.items,)))
    value = RegisteredArray(([],))
    copied = deepcopy(value)
    assert type(copied) is _Array
    assert copied.items[0] is not value.items[0]

    class AtomicArray(_Array):
        def __reduce_ex__(self, protocol):
            return "atomic"

    atomic = AtomicArray((1,))
    assert deepcopy(atomic) is atomic


def test_frozen_wrapper_subclass_does_not_add_six_element_reduction_support():
    from config.bot_inheritance import _Array

    class ExtendedArray(_Array):
        def __reduce_ex__(self, protocol):
            return (_Array, (self.items,), None, None, None, lambda value, state: None)

    with pytest.raises(TypeError):
        deepcopy(ExtendedArray((1,)))


def test_frozen_wrapper_subclass_keeps_reduce_fallback_when_reduce_ex_is_none():
    from copy import Error
    from config.bot_inheritance import _Array

    class ReducedArray(_Array):
        __reduce_ex__ = None

        def __reduce__(self):
            return (_Array, (self.items,))

    value = ReducedArray(([],))
    copied = deepcopy(value)
    assert type(copied) is _Array and copied.items[0] is not value.items[0]

    class UncopyableArray(_Array):
        __reduce_ex__ = None
        __reduce__ = None

    with pytest.raises(Error, match="un\\(deep\\)copyable"):
        deepcopy(UncopyableArray((1,)))


def test_frozen_payload_dataclass_and_pickle_shape_are_unchanged():
    from dataclasses import asdict, fields
    import pickle
    from config.bot_inheritance import _freeze

    value = _freeze({"items": [1, "value"]})
    assert [item.name for item in fields(value)] == ["items"]
    expected = {"items": (("items", {"items": (1, "value")}),)}
    assert asdict(value) == expected
    restored = pickle.loads(pickle.dumps(value))
    assert restored == value and hash(restored) == hash(value)
    assert repr(restored) == repr(value)
    assert deepcopy(restored) is restored


def test_copied_prepared_policy_retains_independent_values_and_sparse_source_authority():
    from config.bot_inheritance import bot_policy_source_view, get_bot_policy, install_bot_policy

    policy = resolve_bot_policy({"short": {"risk": {"capacity": 1.0}}}, catalog())
    config = {}
    install_bot_policy(config, policy)
    copied = deepcopy(config)
    copied_policy = get_bot_policy(copied)
    assert copied["bot"] is not config["bot"]
    assert copied_policy is not policy
    assert copied_policy._source is policy._source
    assert copied["bot"]._effective_baseline is config["bot"]._effective_baseline
    # Catalog defaults and all outward mutable value projections remain copies.
    assert copied_policy.catalog is not policy.catalog
    source = bot_policy_source_view(copied)["bot"]
    source["short"]["risk"]["capacity"] = 9.0
    copied["bot"]["global"]["risk"]["capacity"] = 3.0
    assert config["bot"]["global"]["risk"]["capacity"] == 1.0
    assert bot_policy_source_view(config)["bot"] == {"short": {"risk": {"capacity": 1.0}}}
    assert bot_policy_source_view(copied)["bot"] == {
        "global": {"risk": {"capacity": 3.0}}, "short": {"risk": {"capacity": 1.0}},
    }
