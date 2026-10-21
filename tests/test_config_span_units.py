"""Public span units retain native identifiers and floating physical horizons."""

from copy import deepcopy

import passivbot_rust as pbr
import pytest

from config.field_registry import get_field_registry
from config.load import prepare_config
from config.param_paths import resolve_optimizer_key_path
from config.schema import get_template_config
from config.shared_bot import get_grouped_bot_value
from config.strategy import build_runtime_strategy_side
from config.strategy_paths import native_strategy_path_to_public


def set_value(mapping, path, value):
    for part in path[:-1]:
        mapping = mapping.setdefault(part, {})
    mapping[path[-1]] = value


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_strategy_span_units_preserve_raw_native_metadata_and_fractional_values(kind, side):
    source = get_template_config()
    source["bot"]["global"]["strategy_kind"] = kind
    native_spec = pbr.get_strategy_spec(kind)
    native_before = deepcopy(native_spec)
    expected_native = {}
    for parameter in [*native_spec["parameters"], *native_spec["fixed_parameters"]]:
        if parameter["side"] != side:
            continue
        native = tuple(parameter["config_path"][2:])
        set_value(expected_native, native, parameter["default"])
        if "ema_span" not in native[-1]:
            continue
        public = native_strategy_path_to_public(kind, native)
        unit = "one-hour candles" if native[-1].endswith("_1h") else "minutes"
        assert public[-1].endswith("_1h_candles" if unit == "one-hour candles" else "_minutes")
        value = 4.5 if unit == "one-hour candles" else 123.75
        set_value(source["bot"][side]["strategy"][kind], public, value)
        set_value(expected_native, native, value)
        full_path = ("bot", side, "strategy", kind, *public)
        assert resolve_optimizer_key_path(source, parameter["optimize_key"]) == full_path
        assert get_field_registry().find(full_path).unit == unit
    original = deepcopy(source)
    prepared = prepare_config(source, verbose=False)
    lowered = build_runtime_strategy_side(prepared["bot"][side], strategy_kind=kind, pside=side)
    assert lowered == expected_native
    assert source == original
    assert native_spec == native_before == pbr.get_strategy_spec(kind)


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("native,group,public", [
    ("forager_volume_ema_span_1m", "forager", "volume_ema_span_minutes"),
    ("forager_volatility_ema_span_1m", "forager", "volatility_ema_span_minutes"),
    ("unilateralness_ema_span_1m", "forager", "unilateralness_ema_span_minutes"),
    ("unstuck_ema_span_0", "unstuck", "ema_span_0_minutes"),
    ("unstuck_ema_span_1", "unstuck", "ema_span_1_minutes"),
    ("hsl_cooldown_minutes_after_red", "hsl", "cooldown_after_red_minutes"),
])
def test_shared_minute_fields_keep_native_optimizer_and_runtime_bindings(side, native, group, public):
    source = get_template_config()
    source["bot"][side][group][public] = 123.75
    prepared = prepare_config(source, verbose=False)
    assert get_grouped_bot_value(prepared["bot"][side], native) == 123.75
    assert resolve_optimizer_key_path(prepared, f"{side}_{native}") == ("bot", side, group, public)


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_unreleased_strategy_span_spellings_are_not_runtime_aliases(kind):
    from tools.migrate_config_v9 import migrate

    source = get_template_config()
    source["bot"]["global"]["strategy_kind"] = kind
    native = next(parameter for parameter in pbr.get_strategy_spec(kind)["parameters"]
                  if parameter["side"] == "long" and parameter["name"] == "ema_span_0")
    set_value(source["bot"]["long"]["strategy"][kind], native["config_path"][2:], 123.75)
    with pytest.raises((KeyError, TypeError, ValueError), match="unknown|Unknown|retired|unexpected"):
        prepare_config(source, verbose=False)
    with pytest.raises((KeyError, TypeError, ValueError), match="unknown|Unknown|retired|unexpected"):
        migrate(source)
