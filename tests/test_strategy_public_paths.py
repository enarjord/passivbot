"""Native and authored strategy paths share one canonical spelling."""

from copy import deepcopy

import passivbot_rust as pbr
import pytest

from config.strategy import build_runtime_strategy_side
from config.strategy_spec import get_strategy_defaults, get_strategy_spec


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_native_strategy_metadata_is_canonical_and_independently_copied(kind):
    native = pbr.get_strategy_spec(kind)
    original = deepcopy(native)
    get_strategy_spec.cache_clear()
    spec = get_strategy_spec(kind)
    try:
        assert spec == native
        paths = [tuple(row["config_path"]) for row in
                 spec["parameters"] + spec.get("fixed_parameters", [])]
        assert len(paths) == len(set(paths))
        spec["parameters"][0]["config_path"].append("changed")
        spec["parameters"][0]["bounds"][0] = -123.5
        assert native == original == pbr.get_strategy_spec(kind)
    finally:
        get_strategy_spec.cache_clear()


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_canonical_runtime_tree_copies_defaults_and_precise_overrides(kind, side):
    source = get_strategy_defaults(kind)[side]
    original = deepcopy(source)
    expected = deepcopy(source)
    assert build_runtime_strategy_side(source, strategy_kind=kind, pside=side) == expected
    name = "entry_initial_qty_fraction" if kind == "trailing_martingale" else "base_qty_fraction"
    parameter = next(row for row in pbr.get_strategy_spec(kind)["parameters"]
                     if row["side"] == side and row["name"] == name)
    path = parameter["config_path"][2:]
    override = {}
    target, expected_target = override, expected
    for part in path[:-1]:
        target = target.setdefault(part, {})
        expected_target = expected_target[part]
    value = 0.012345678901234567
    target[path[-1]] = expected_target[path[-1]] = value
    before = deepcopy(override)
    runtime = build_runtime_strategy_side(source, strategy_kind=kind, pside=side, override_side=override)
    assert runtime == expected
    assert source == original and override == before
    for part in path[:-1]:
        runtime = runtime[part]
    runtime[path[-1]] = 42.0
    assert source == original and override == before


def test_duplicate_native_metadata_paths_are_rejected_without_mutating_the_source(monkeypatch):
    native = pbr.get_strategy_spec("ema_anchor")
    native["parameters"].append({**deepcopy(native["parameters"][0]),
                                "name": "future_field", "optimize_key": "long_future_field"})
    original = deepcopy(native)
    monkeypatch.setattr(pbr, "get_strategy_spec", lambda kind: native)
    get_strategy_spec.cache_clear()
    try:
        with pytest.raises(ValueError, match="conflicting strategy metadata path"):
            get_strategy_spec("ema_anchor")
        assert native == original
    finally:
        get_strategy_spec.cache_clear()
