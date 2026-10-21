"""Public metadata projection leaves the native protocol independently owned."""

from copy import deepcopy

import passivbot_rust as pbr
import pytest

from config.strategy import build_runtime_strategy_side
from config.strategy_paths import project_strategy_spec
from config.strategy_spec import get_strategy_defaults, get_strategy_spec


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_identity_projection_copies_metadata_without_mutating_native_spec(kind):
    native = pbr.get_strategy_spec(kind)
    original = deepcopy(native)
    projected = project_strategy_spec(kind, native)
    assert projected == native
    assert get_strategy_spec(kind) == native
    projected["parameters"][0]["config_path"].append("changed")
    projected["parameters"][0]["bounds"][0] = -123.5
    assert native == original == pbr.get_strategy_spec(kind)


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_identity_runtime_lowering_preserves_all_native_defaults_and_precise_override(kind, side):
    source = get_strategy_defaults(kind)[side]
    original = deepcopy(source)
    assert build_runtime_strategy_side(source, strategy_kind=kind, pside=side) == source
    native_name = "entry_initial_qty_pct" if kind == "trailing_martingale" else "base_qty_pct"
    parameter = next(item for item in pbr.get_strategy_spec(kind)["parameters"]
                     if item["side"] == side and item["name"] == native_name)
    path = parameter["config_path"][2:]
    override = {}
    target = override
    expected = deepcopy(source)
    expected_target = expected
    for part in path[:-1]:
        target = target.setdefault(part, {})
        expected_target = expected_target[part]
    value = 0.012345678901234567
    target[path[-1]] = value
    expected_target[path[-1]] = value
    before = deepcopy(override)
    lowered = build_runtime_strategy_side(source, strategy_kind=kind, pside=side, override_side=override)
    assert lowered == expected
    assert source == original
    assert override == before
