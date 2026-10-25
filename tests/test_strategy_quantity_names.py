"""Public and native strategy quantities use the same names and preserve values."""

from copy import deepcopy
import json

import passivbot_rust as pbr
import pytest

from config import get_template_config, prepare_config
from config.authoring import compact_config, resolved_config
from config.field_registry import get_field_registry
from config.migrations.released_v8 import get_released_v8_defaults
from config.optimize_bounds import flatten_optimize_bounds
from config.param_paths import resolve_optimizer_key_path
from config.strategy import build_runtime_strategy_side

from tools.clean_config import cleanup_config
from tools.migrate_config_v9 import migrate


FIELDS = [
    ("trailing_martingale", ("entry", "double_down_factor"),
     ("entry", "reentry_qty_multiplier"), "entry_reentry_qty_multiplier", 1.375),
    ("trailing_martingale", ("entry", "threshold_we_weight"),
     ("entry", "threshold_exposure_ratio_coefficient"), "entry_threshold_exposure_ratio_coefficient", 2.125),
    ("trailing_martingale", ("entry", "retracement_we_weight"),
     ("entry", "retracement_exposure_ratio_coefficient"), "entry_retracement_exposure_ratio_coefficient", 0.375),
    ("trailing_martingale", ("close", "threshold_we_weight"),
     ("close", "threshold_exposure_ratio_coefficient"), "close_threshold_exposure_ratio_coefficient", -0.0125),
    ("ema_anchor", ("entry_double_down_factor",),
     ("entry_qty_exposure_coefficient",), "entry_qty_exposure_coefficient", 0.625),
    ("ema_anchor", ("offset_psize_weight",),
     ("inventory_offset_exposure_coefficient",), "inventory_offset_exposure_coefficient", 0.125),
]


def put(node, path, value):
    for part in path[:-1]:
        node = node.setdefault(part, {})
    node[path[-1]] = value


def at(node, path):
    for part in path:
        node = node[part]
    return node


def source(kind):
    config = get_template_config()
    config["bot"]["portfolio"]["strategy_kind"] = kind
    config["bot"]["long"]["enabled"] = True
    config["bot"]["short"]["enabled"] = True
    config["optimize"]["bounds"] = {}
    return config


@pytest.mark.parametrize("kind,old,new,native_id,value", FIELDS)
@pytest.mark.parametrize("scope", ["global", "long", "short"])
@pytest.mark.parametrize("channel", ["root", "fixed", "scenario", "bounds"])
def test_retired_strategy_spellings_are_rejected_without_mutation(kind, old, new, native_id, value, scope, channel):
    config = source(kind)
    path = ("bot", scope, "strategy", kind, *old)
    if channel == "root":
        put(config, path, value)
    elif channel == "bounds":
        put(config, ("optimize", "bounds", scope, "strategy", kind, *old), [value, value])
    else:
        patch = {".".join(path): value}
        if channel == "fixed":
            config["optimize"]["fixed_runtime_overrides"] = patch
        else:
            config["backtest"]["scenarios"] = [{"label": "invalid", "overrides": patch}]
    original = deepcopy(config)
    with pytest.raises((KeyError, TypeError, ValueError), match=old[-1]):
        prepare_config(config, verbose=False)
    assert config == original


@pytest.mark.parametrize("kind,old,new,native_id,value", FIELDS)
@pytest.mark.parametrize("scope", ["global", "long", "short"])
def test_new_strategy_quantities_roundtrip_with_unchanged_native_values(kind, old, new, native_id, value, scope):
    config = source(kind)
    path = ("bot", scope, "strategy", kind, *new)
    put(config, path, value)
    prepared = prepare_config(config, verbose=False)
    for export in (cleanup_config, compact_config, resolved_config):
        loaded = prepare_config(export(prepared), verbose=False)
        assert at(loaded, path) == value
    for side in (("long", "short") if scope == "global" else (scope,)):
        public = prepared["bot"][side]["strategy"][kind]
        native = build_runtime_strategy_side(prepared["bot"][side], strategy_kind=kind, pside=side)
        assert at(native, new) == value
        assert native == deepcopy(public)
        assert deepcopy(native) == public
        assert resolve_optimizer_key_path(prepared, f"{side}_{native_id}") == (
            "bot", side, "strategy", kind, *new)
    registry = get_field_registry()
    assert registry.find(path) is not None
    assert registry.find(("bot", scope, "strategy", kind, *old)) is None


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_v8_strategy_quantities_keep_sources_bounds_files_and_final_targets(tmp_path, kind):
    config = get_released_v8_defaults(strategy_kind=kind)
    config["backtest"]["visible_metrics"] = [m for m in config["backtest"]["visible_metrics"] if not m.endswith("_w") and not m.startswith("strategy_eq_underwater_pct_")]
    config["optimize"]["scoring"] = [m for m in config["optimize"]["scoring"] if not m["metric"].endswith("_w") and not m["metric"].startswith("strategy_eq_underwater_pct_")]
    fields = [row for row in FIELDS if row[0] == kind]
    file_patch = {"bot": {"long": {"strategy": {kind: {}}}}}
    selectors = []
    for _, old, new, native_id, value in fields:
        put(config, ("bot", "long", "strategy", kind, *old), value)
        put(config, ("optimize", "bounds", "long", "strategy", kind, *old), [value, value])
        put(file_patch, ("bot", "long", "strategy", kind, *old), value / 2)
        selectors.append(".".join(("bot", "long", "strategy", kind, *old)))
    config["optimize"]["fixed_params"] = selectors
    file = tmp_path / "coin.json"
    file.write_text(json.dumps(file_patch))
    original_file = file.read_bytes()
    config["coin_overrides"] = {"BTC": {"override_config_path": "coin.json"}}
    put(config["coin_overrides"]["BTC"], ("bot", "long", "strategy", kind, *fields[0][1]),
        fields[0][-1] * 0.75)
    original = deepcopy(config)
    converted = migrate(config, memory_span_minutes=123.5,
                        base_config_path=str(tmp_path / "base.json"))
    assert config == original and file.read_bytes() == original_file
    flat = flatten_optimize_bounds(converted.config["optimize"]["bounds"], strategy_kind=kind)
    for index, (_, old, new, native_id, value) in enumerate(fields):
        assert at(converted.config, ("bot", "long", "strategy", kind, *new)) == value
        assert at(converted.config, ("coin_overrides", "BTC", "bot", "long", "strategy", kind, *new)) == value * (0.75 if index == 0 else 0.5)
        assert flat[f"long_{native_id}"] == [value, value]
        target = ".".join(("bot", "long", "strategy", kind, *new))
        assert target in converted.config["optimize"]["fixed_params"]
        assert any(isinstance(row["new"], dict) and target in row["new"]
                   for row in converted.changes)
    file.unlink()
    prepare_config(converted.config, verbose=False)


@pytest.mark.parametrize("kind,old,new,native_id,value", FIELDS)
def test_native_metadata_uses_the_canonical_quantity_names_and_optimizer_ids(kind, old, new, native_id, value):
    records = [row for row in pbr.get_strategy_spec(kind)["parameters"]
               if row["side"] == "long" and tuple(row["config_path"][2:]) == new]
    assert len(records) == 1
    assert records[0]["name"] == native_id
    assert records[0]["optimize_key"] == f"long_{native_id}"
