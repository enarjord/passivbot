"""Structural catalog indexes never cache policy values or mutable metadata."""

from copy import deepcopy
from dataclasses import replace
import pickle
from types import SimpleNamespace

import pytest

from config.bot_inheritance import (
    SideField, SideFieldCatalog, _freeze, resolve_bot_policy,
)
from config.field_spec import FieldSpec, FieldType


def numeric(suffix, default=1.0):
    return SideField(tuple(suffix), (FieldType.FLOAT,), False, _freeze(default),
                     ("long_value", "short_value"), True)


def test_duplicate_suffixes_keep_first_field_and_declared_order():
    first = numeric(("risk", "capacity"))
    second = replace(first, types=(FieldType.BOOLEAN,))
    declared = SideFieldCatalog((first, second))
    assert declared.fields == (first, second)
    assert declared.find(("risk", "capacity")) is first
    assert declared.find(["risk", "capacity"]) is first
    assert declared.find(("risk",)) is None
    declared.validate_tree({"risk": {"capacity": 0.0}}, ("bot", "long"))
    with pytest.raises(TypeError, match="declared serialization type"):
        declared.validate_tree({"risk": {"capacity": False}}, ("bot", "long"))


def test_parent_prefixes_keep_exact_paths_and_value_validation():
    activation = SideField(("enabled",), (FieldType.BOOLEAN,), False, False,
                           (None, None), False, inheritable=False)
    declared = SideFieldCatalog((numeric(("risk", "capacity")),
                                 numeric(("risk_extra", "capacity")), activation))
    declared.validate_tree({"risk": {}, "risk_extra": {"capacity": 2.0}}, ("bot", "global"))
    declared.validate_tree({"enabled": False}, ("bot", "short"))
    for unknown in ({"ris": {}}, {"risk": {"unknown": 2.0}}, {"risk": {"capacity": {}}}):
        with pytest.raises((TypeError, ValueError)):
            declared.validate_tree(unknown, ("bot", "long"))
    with pytest.raises(TypeError, match="bot.long.risk must be an object"):
        declared.validate_tree({"risk": 2.0}, ("bot", "long"))
    with pytest.raises(ValueError, match="bot.global.enabled"):
        declared.validate_tree({"enabled": False}, ("bot", "global"))
    with pytest.raises(TypeError, match="keys must be strings"):
        declared.validate_tree({"risk": {3: 2.0}}, ("bot", "long"))
    with pytest.raises(TypeError, match="expected str instance"):
        declared.validate_tree({3: 2.0}, ("bot", "long"))


def test_mutable_or_foreign_declarations_keep_live_lookup_behavior():
    first = numeric(("risk", "capacity"))
    fields = [first]
    declared = SideFieldCatalog(fields)
    assert declared.find(first.suffix) is first
    second = numeric(("other", "value"))
    fields.append(second)
    assert declared.find(second.suffix) is second
    declared.validate_tree({"other": {"value": 2.0}}, ("bot", "long"))
    fields[0] = numeric(("replacement", "capacity"))
    assert declared.find(first.suffix) is None
    foreign = SimpleNamespace(**first.__dict__)
    mutable_field = SideFieldCatalog((foreign,))
    assert mutable_field.find(first.suffix) is foreign
    foreign.suffix = ("changed", "capacity")
    assert mutable_field.find(first.suffix) is None
    assert mutable_field.find(foreign.suffix) is foreign
    # Constructor acceptance and lazy errors stay unchanged for invalid shapes.
    with pytest.raises(TypeError):
        SideFieldCatalog(None).find(("risk", "capacity"))


def test_deepcopy_keeps_field_metadata_and_default_values_independent():
    side = SideField(("forager", "always_include"), (FieldType.ARRAY,), False,
                     _freeze(["AAA"]), (None, None), False)
    portfolio = FieldSpec(("bot", "portfolio", "settings"), (FieldType.OBJECT,),
                          default={"values": [1]})
    declared = SideFieldCatalog((side,), (portfolio,))
    copied = deepcopy(declared)
    assert copied == declared and copied is not declared
    assert copied.find(side.suffix) is copied.fields[0]
    assert copied.fields[0] is not side
    copied.portfolio_fields[0].default["values"].append(2)
    assert portfolio.default == {"values": [1]}
    value = copied.find(side.suffix).default_value()
    value.append("BBB")
    assert side.default_value() == ["AAA"]
    with pytest.raises(TypeError):
        copied._lookup.positions[side.suffix] = 99


def test_pickle_and_replacement_rebuild_only_structural_metadata():
    first = numeric(("risk", "capacity"))
    original = SideFieldCatalog((first,))
    restored = pickle.loads(pickle.dumps(original))
    assert restored == original
    assert restored.find(first.suffix) is restored.fields[0]
    restored.validate_tree({"risk": {}}, ("bot", "long"))
    second = numeric(("new", "value"))
    changed = replace(original, fields=(second,))
    assert changed.find(first.suffix) is None
    assert changed.find(second.suffix) is second
    assert original.find(first.suffix) is first


def test_independent_catalog_defaults_never_share_policy_state():
    first = SideFieldCatalog((numeric(("risk", "capacity"), 1.0),))
    second = SideFieldCatalog((numeric(("risk", "capacity"), 5.0),))
    assert resolve_bot_policy({}, first).effective_view()["long"]["risk"]["capacity"] == 1.0
    assert resolve_bot_policy({}, second).effective_view()["long"]["risk"]["capacity"] == 5.0
    assert resolve_bot_policy({}, first).source_view() == {}
