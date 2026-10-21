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


def test_trailing_overview_json_names_enforcer_ratios_explicitly():
    import json
    from tools import trailing_inspect

    source = get_template_config()
    risk = source["bot"]["long"]["risk"]
    risk["position_exposure_enforcer"]["threshold_ratio"] = 0.8
    risk["total_exposure_enforcer"]["threshold_ratio"] = 0.9
    overview = trailing_inspect.build_overview(
        sources={"long": {
            "params": trailing_inspect._extract_strategy_params(source, "long"),
            "context": trailing_inspect._extract_side_context(source, "long"),
        }},
        parameter_source="test",
        price_anchor=100.0,
        volatility_scenarios=(("normal", 0.005, 0.0025),),
        exposure_ratios=(0.5,),
    )
    context = json.loads(json.dumps(overview))["sides"]["long"]["context"]
    assert context["position_exposure_enforcer_threshold_ratio"] == 0.8
    assert context["total_exposure_enforcer_threshold_ratio"] == 0.9
    assert "position_exposure_enforcer_threshold" not in context
    assert "total_exposure_enforcer_threshold" not in context
