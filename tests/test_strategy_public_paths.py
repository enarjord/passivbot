"""Public metadata projection leaves the native protocol independently owned."""

from copy import deepcopy

import passivbot_rust as pbr
import pytest

from config.strategy import build_runtime_strategy_side
from config.strategy_paths import (
    native_strategy_path_to_public, project_native_strategy_side,
    project_strategy_spec, lower_runtime_strategy_side,
)
from config.strategy_spec import get_strategy_defaults, get_strategy_spec


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_public_projection_changes_only_paths_without_mutating_native_spec(kind):
    native = pbr.get_strategy_spec(kind)
    original = deepcopy(native)
    projected = project_strategy_spec(kind, native)
    assert get_strategy_spec(kind) == projected
    native_fields = native["parameters"] + native.get("fixed_parameters", [])
    public_fields = projected["parameters"] + projected.get("fixed_parameters", [])
    paths = []
    for raw, public in zip(native_fields, public_fields, strict=True):
        assert {key: value for key, value in public.items() if key != "config_path"} == {
            key: value for key, value in raw.items() if key != "config_path"
        }
        expected = [*raw["config_path"][:2], *native_strategy_path_to_public(kind, raw["config_path"][2:])]
        assert public["config_path"] == expected
        paths.append(tuple(expected))
    assert len(paths) == len(set(paths))
    projected["parameters"][0]["config_path"].append("changed")
    projected["parameters"][0]["bounds"][0] = -123.5
    assert native == original == pbr.get_strategy_spec(kind)


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_runtime_lowering_preserves_all_native_defaults_and_precise_override(kind, side):
    source = get_strategy_defaults(kind)[side]
    original = deepcopy(source)
    expected = lower_runtime_strategy_side(kind, source)
    assert build_runtime_strategy_side(source, strategy_kind=kind, pside=side) == expected
    assert project_native_strategy_side(kind, expected) == source
    native_name = "entry_initial_qty_pct" if kind == "trailing_martingale" else "base_qty_pct"
    parameter = next(item for item in pbr.get_strategy_spec(kind)["parameters"]
                     if item["side"] == side and item["name"] == native_name)
    native_path = parameter["config_path"][2:]
    path = native_strategy_path_to_public(kind, native_path)
    override = {}
    target = override
    expected_target = expected
    for part in path[:-1]:
        target = target.setdefault(part, {})
    for part in native_path[:-1]:
        expected_target = expected_target[part]
    value = 0.012345678901234567
    target[path[-1]] = value
    expected_target[native_path[-1]] = value
    before = deepcopy(override)
    lowered = build_runtime_strategy_side(source, strategy_kind=kind, pside=side, override_side=override)
    assert lowered == expected
    assert source == original
    assert override == before


@pytest.mark.parametrize("kind,old,new", [
    ("trailing_martingale", {"entry": {"initial_qty_pct": 0.1, "initial_qty_fraction": 0.2}}, "initial_qty_fraction"),
    ("ema_anchor", {"base_qty_pct": 0.1, "base_qty_fraction": 0.2}, "base_qty_fraction"),
])
def test_tree_projection_rejects_collisions_instead_of_discarding_unknown_fields(kind, old, new):
    with pytest.raises(ValueError, match="conflicting strategy path"):
        project_native_strategy_side(kind, old)
    with pytest.raises(ValueError, match="conflicting strategy path"):
        lower_runtime_strategy_side(kind, old)
    source = {"unrecognized": {new: 0.123456789}}
    assert project_native_strategy_side(kind, source) == source
    assert lower_runtime_strategy_side(kind, source) == source


def test_future_native_field_cannot_overwrite_an_existing_projected_public_path():
    native = pbr.get_strategy_spec("ema_anchor")
    original = deepcopy(native)
    collision = deepcopy(next(field for field in native["parameters"]
                              if field["config_path"] == ["strategy", "long", "base_qty_pct"]))
    collision.update(name="future_field", optimize_key="long_future_field",
                     config_path=["strategy", "long", "base_qty_fraction"])
    native["parameters"].append(collision)
    with pytest.raises(ValueError, match="conflicting public strategy path"):
        project_strategy_spec("ema_anchor", native)
    assert pbr.get_strategy_spec("ema_anchor") == original
