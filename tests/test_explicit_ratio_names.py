"""Public quantity names preserve native values and reject development spellings."""

import pytest

from config.field_registry import get_field_registry
from config.schema import get_template_config
from config.shared_bot import flatten_shared_bot_side
from tools.clean_config import cleanup_config


@pytest.mark.parametrize("parts,old,new", [
    (("bot", "global", "warmup"), "ratio", "span_multiplier"),
    (("bot", "global", "execution"), "market_order_near_touch_threshold", "market_order_near_touch_fraction"),
    (("bot", "long", "risk", "position_exposure_enforcer"), "threshold", "threshold_ratio"),
    (("bot", "short", "risk", "total_exposure_enforcer"), "threshold", "threshold_ratio"),
])
def test_retired_development_quantity_spelling_is_not_an_authored_alias(parts, old, new):
    source = get_template_config()
    target = source
    for part in parts:
        target = target[part]
    target[old] = target.pop(new)
    with pytest.raises(ValueError, match="unknown configuration field"):
        cleanup_config(source)


@pytest.mark.parametrize("side", ["long", "short"])
def test_ratio_thresholds_preserve_native_ids_and_full_float_precision(side):
    source = get_template_config()
    expected = {"risk_wel_enforcer_threshold": 0.7123456789012345,
                "risk_twel_enforcer_threshold": 0.9123456789012345}
    risk = source["bot"][side]["risk"]
    risk["position_exposure_enforcer"]["threshold_ratio"] = expected["risk_wel_enforcer_threshold"]
    risk["total_exposure_enforcer"]["threshold_ratio"] = expected["risk_twel_enforcer_threshold"]
    cleaned = cleanup_config(source)
    native = flatten_shared_bot_side(cleaned["bot"][side])
    assert {key: native[key] for key in expected} == expected
    assert not any("threshold_ratio" in key for key in native)


def test_ratio_units_and_denominators_are_declared_at_the_canonical_path():
    registry = get_field_registry()
    cases = {
        ("bot", "global", "warmup", "span_multiplier"): ("multiplier", "indicator span"),
        ("bot", "global", "execution", "market_order_near_touch_fraction"): ("fraction", "market price"),
        ("bot", "long", "risk", "position_exposure_enforcer", "threshold_ratio"):
            ("ratio", "effective per-position wallet exposure limit"),
    }
    for path, expected in cases.items():
        field = registry.find(path)
        assert (field.unit, field.denominator) == expected
