"""Authored log-range quantities lower to the unchanged native volatility ABI."""

from copy import deepcopy
import json

import passivbot_rust as pbr
import pytest

from config import get_template_config, prepare_config
from config.authoring import compact_config, resolved_config
from config.bot import normalize_forager_score_weights
from config.field_registry import get_field_registry
from config.migrations.released_v8 import get_released_v8_defaults
from config.optimize_bounds import flatten_optimize_bounds
from config.param_paths import resolve_optimizer_key_path
from config.shared_bot import (
    canonicalize_shared_bot_side, flatten_shared_bot_side, get_grouped_bot_value,
    set_grouped_bot_value,
)
from config.strategy import build_runtime_strategy_side
from config.strategy_paths import lower_runtime_strategy_side, project_native_strategy_side
from tools.clean_config import cleanup_config
from tools.migrate_config_v9 import migrate


FIELDS = [
    ("trailing_martingale", ("volatility_ema_span_1m",),
     ("volatility_ema_span_minutes",), ("log_range_ema_span_minutes",), 15.375),
    ("trailing_martingale", ("volatility_ema_span_1h",),
     ("volatility_ema_span_1h_candles",), ("log_range_ema_span_1h_candles",), 4.625),
    ("ema_anchor", ("offset_volatility_ema_span_1m",),
     ("offset_volatility_ema_span_minutes",), ("offset_log_range_ema_span_minutes",), 21.375),
    ("ema_anchor", ("offset_volatility_ema_span_1h",),
     ("offset_volatility_ema_span_1h_candles",), ("offset_log_range_ema_span_1h_candles",), 3.625),
    *[("trailing_martingale", (leg, f"{role}_volatility_{horizon}_weight"),
       (leg, f"{role}_volatility_{horizon}_weight"),
       (leg, f"{role}_log_range_{horizon}_coefficient"), 0.375)
      for leg in ("entry", "close") for role in ("threshold", "retracement")
      for horizon in ("1m", "1h")],
    *[("ema_anchor", (f"offset_volatility_{horizon}_weight",),
       (f"offset_volatility_{horizon}_weight",),
       (f"offset_log_range_{horizon}_coefficient",), 0.625)
      for horizon in ("1m", "1h")],
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


@pytest.mark.parametrize("kind,native,old,new,value", FIELDS)
@pytest.mark.parametrize("scope", ["global", "long", "short"])
@pytest.mark.parametrize("channel", ["root", "fixed", "scenario", "bounds"])
def test_old_strategy_log_range_spellings_are_rejected(kind, native, old, new, value, scope, channel):
    config = source(kind)
    path = ("bot", scope, "strategy", kind, *old)
    if channel == "root":
        put(config, path, value)
    elif channel == "bounds":
        put(config, ("optimize", "bounds", scope, "strategy", kind, *old), [value, value])
    elif channel == "fixed":
        config["optimize"]["fixed_runtime_overrides"] = {".".join(path): value}
    else:
        config["backtest"]["scenarios"] = [{"label": "invalid", "overrides": {".".join(path): value}}]
    original = deepcopy(config)
    with pytest.raises((KeyError, TypeError, ValueError), match=old[-1]):
        prepare_config(config, verbose=False)
    assert config == original


@pytest.mark.parametrize("kind,native,old,new,value", FIELDS)
@pytest.mark.parametrize("scope", ["global", "long", "short"])
def test_strategy_log_range_roundtrips_preserve_native_ids_and_values(kind, native, old, new, value, scope):
    config = source(kind)
    path = ("bot", scope, "strategy", kind, *new)
    put(config, path, value)
    prepared = prepare_config(config, verbose=False)
    for export in (cleanup_config, compact_config, resolved_config):
        assert at(prepare_config(export(prepared), verbose=False), path) == value
    for side in (("long", "short") if scope == "global" else (scope,)):
        public = prepared["bot"][side]["strategy"][kind]
        lowered = build_runtime_strategy_side(prepared["bot"][side], strategy_kind=kind, pside=side)
        assert at(lowered, native) == value
        assert lowered == lower_runtime_strategy_side(kind, public)
        assert project_native_strategy_side(kind, lowered) == public
        native_id = f"{side}_{'_'.join(native)}"
        assert resolve_optimizer_key_path(prepared, native_id) == ("bot", side, "strategy", kind, *new)
    assert get_field_registry().find(path) is not None
    assert get_field_registry().find(("bot", scope, "strategy", kind, *old)) is None


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_released_strategy_log_ranges_convert_root_bounds_files_and_reports(tmp_path, kind):
    config = get_released_v8_defaults(strategy_kind=kind)
    config["backtest"]["visible_metrics"] = [m for m in config["backtest"]["visible_metrics"] if not m.endswith("_w")]
    config["optimize"]["scoring"] = [m for m in config["optimize"]["scoring"] if not m["metric"].endswith("_w")]
    fields = [row for row in FIELDS if row[0] == kind]
    patch = {"bot": {"long": {"strategy": {kind: {}}}}}
    selectors = []
    for _, native, old, new, value in fields:
        put(config, ("bot", "long", "strategy", kind, *native), value)
        put(config, ("optimize", "bounds", "long", "strategy", kind, *native), [value, value])
        put(patch, ("bot", "long", "strategy", kind, *native), value / 2)
        selectors.append(".".join(("bot", "long", "strategy", kind, *native)))
    config["optimize"]["fixed_params"] = selectors
    file = tmp_path / "coin.json"
    file.write_text(json.dumps(patch))
    original_bytes = file.read_bytes()
    config["coin_overrides"] = {"BTC": {"override_config_path": "coin.json"}}
    put(config["coin_overrides"]["BTC"], ("bot", "long", "strategy", kind, *fields[0][1]), fields[0][-1] * 0.75)
    before = deepcopy(config)
    result = migrate(config, memory_span_minutes=123.5, base_config_path=str(tmp_path / "base.json"))
    assert config == before and file.read_bytes() == original_bytes
    flat = flatten_optimize_bounds(result.config["optimize"]["bounds"], strategy_kind=kind)
    for index, (_, native, old, new, value) in enumerate(fields):
        path = ("bot", "long", "strategy", kind, *new)
        assert at(result.config, path) == value
        assert at(result.config["coin_overrides"]["BTC"], path) == value * (0.75 if index == 0 else 0.5)
        assert flat[f"long_{'_'.join(native)}"] == [value, value]
        assert ".".join(path) in result.config["optimize"]["fixed_params"]
        assert any(isinstance(row["new"], dict) and ".".join(path) in row["new"] for row in result.changes)
    file.unlink()
    prepare_config(result.config, verbose=False)


@pytest.mark.parametrize("scope", ["global", "long", "short"])
@pytest.mark.parametrize("channel", ["root", "fixed", "scenario", "bounds"])
@pytest.mark.parametrize("old,value", [(("volatility_ema_span_minutes",), 12.375), (("score_weights", "volatility"), 0.5)])
def test_old_forager_log_range_spellings_are_rejected(scope, channel, old, value):
    config = source("trailing_martingale")
    path = ("bot", scope, "forager", *old)
    if channel == "root":
        put(config, path, value)
    elif channel == "bounds":
        put(config, ("optimize", "bounds", scope, "forager", *old), [value, value])
    elif channel == "fixed":
        config["optimize"]["fixed_runtime_overrides"] = {".".join(path): value}
    else:
        config["backtest"]["scenarios"] = [{"label": "invalid", "overrides": {".".join(path): value}}]
    original = deepcopy(config)
    with pytest.raises((KeyError, TypeError, ValueError), match=old[-1]):
        prepare_config(config, verbose=False)
    assert config == original


@pytest.mark.parametrize("scope", ["global", "long", "short"])
@pytest.mark.parametrize("weights", [
    {"volume": 2.0, "ema_readiness": 3.0, "log_range": 5.0, "unilateralness": 7.0},
    {"volume": 0.0, "ema_readiness": 0.0, "log_range": 0.0, "unilateralness": 0.0},
])
def test_forager_weights_preserve_normalization_order_native_packet_and_exports(scope, weights):
    config = source("trailing_martingale")
    put(config, ("bot", scope, "forager", "score_weights"), weights)
    put(config, ("bot", scope, "forager", "log_range_ema_span_minutes"), 27.375)
    original = deepcopy(config)
    prepared = prepare_config(config, verbose=False)
    expected = normalize_forager_score_weights(weights, path="weights")
    native_weights = {("volatility" if key == "log_range" else key): value for key, value in expected.items()}
    assert config == original
    for side in (("long", "short") if scope == "global" else (scope,)):
        native = flatten_shared_bot_side(prepared["bot"][side])
        assert native["forager_score_weights"] == native_weights
        assert native["forager_volatility_ema_span_1m"] == 27.375
        assert resolve_optimizer_key_path(prepared, f"{side}_forager_score_weights_volatility") == ("bot", side, "forager", "score_weights", "log_range")
        assert get_grouped_bot_value(prepared["bot"][side], "forager_score_weights") == native_weights
    for export in (cleanup_config, compact_config, resolved_config):
        reloaded = prepare_config(export(prepared), verbose=False)
        assert reloaded["bot"]["long"]["forager"]["score_weights"] == prepared["bot"]["long"]["forager"]["score_weights"]


def test_trusted_shared_native_weight_projection_is_detached_and_bidirectional():
    weights = {"volume": 1.0, "ema_readiness": 2.0, "volatility": 3.0, "unilateralness": 4.0}
    original = deepcopy(weights)
    side = {"forager_score_weights": weights}
    canonicalize_shared_bot_side(side)
    assert side == {"forager": {"score_weights": {"volume": 1.0, "ema_readiness": 2.0, "log_range": 3.0, "unilateralness": 4.0}}}
    assert weights == original
    assert flatten_shared_bot_side(side)["forager_score_weights"] == original
    set_grouped_bot_value(side, "forager_score_weights", weights)
    assert side["forager"]["score_weights"]["log_range"] == 3.0
    assert weights == original
    assert "volatility" not in side["forager"]["score_weights"]


@pytest.mark.parametrize("layout", ["flattened", "nested"])
def test_released_forager_log_range_preserves_bounds_selectors_scenarios_and_final_reports(layout):
    config = get_released_v8_defaults()
    config["backtest"]["visible_metrics"] = [m for m in config["backtest"]["visible_metrics"] if not m.endswith("_w")]
    config["optimize"]["scoring"] = [m for m in config["optimize"]["scoring"] if not m["metric"].endswith("_w")]
    weights = {"volume": 0.2, "ema_readiness": 0.3, "volatility": 0.4, "unilateralness": 0.1}
    config["bot"]["long"]["forager"].update(score_weights=weights, volatility_ema_span_1m=37.375)
    bounds = config["optimize"]["bounds"]["long"]["forager"]
    bounds["volatility_ema_span_1m"] = [31.375, 47.625, 0.25]
    if layout == "nested":
        bounds["score_weights"] = {name: [value, value] for name, value in weights.items()}
        for name in weights:
            bounds.pop(f"score_weights_{name}", None)
    else:
        bounds.update({f"score_weights_{name}": [value, value] for name, value in weights.items()})
    selectors = ["bot.long.forager.score_weights.volatility", "bot.long.forager.volatility_ema_span_1m"]
    config["optimize"]["fixed_params"] = selectors
    config["backtest"]["scenarios"] = [{"label": "range", "overrides": {
        "bot.long.forager_score_weights.volatility": 0.375,
        "bot.long.forager_volatility_ema_span_1m": 23.625,
    }}]
    before = deepcopy(config)
    result = migrate(config, memory_span_minutes=123.5)
    assert config == before
    public = result.config["bot"]["long"]["forager"]
    assert public["score_weights"]["log_range"] == weights["volatility"]
    assert public["log_range_ema_span_minutes"] == 37.375
    flat = flatten_optimize_bounds(result.config["optimize"]["bounds"], strategy_kind="trailing_martingale")
    assert flat["long_forager_score_weights_volatility"] == [0.4, 0.4]
    assert flat["long_forager_volatility_ema_span_1m"] == [31.375, 47.625, 0.25]
    expected = ["bot.long.forager.score_weights.log_range", "bot.long.forager.log_range_ema_span_minutes"]
    assert result.config["optimize"]["fixed_params"] == expected
    assert result.config["backtest"]["scenarios"][0]["overrides"] == {
        "bot.long.forager.score_weights.log_range": 0.375,
        "bot.long.forager.log_range_ema_span_minutes": 23.625,
    }
    assert any(row["path"] == "optimize.fixed_params" and row["new"] == expected for row in result.changes)
    for target in expected:
        assert any(isinstance(row["new"], dict) and target in row["new"] for row in result.changes)


@pytest.mark.parametrize("kind,native,old,new,value", FIELDS)
def test_log_range_native_metadata_keeps_frozen_identifiers(kind, native, old, new, value):
    records = [row for row in pbr.get_strategy_spec(kind)["parameters"]
               if row["side"] == "long" and tuple(row["config_path"][2:]) == native]
    assert len(records) == 1
    assert records[0]["name"] == "_".join(native)
    assert records[0]["optimize_key"] == f"long_{'_'.join(native)}"


@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("globally_dead_only", [False, True])
@pytest.mark.parametrize("policy", ["disabled_domain", "reachable", "runtime_disabled"])
def test_disabled_retracement_dead_genes_keep_native_ids_public_paths_and_bound_floors(side, globally_dead_only, policy):
    from optimization.warmup import optimizer_dead_param_values

    config = source("trailing_martingale")
    root = ("bot", side, "strategy", "trailing_martingale", "close")
    put(config, (*root, "retracement_base_fraction"), 0.0)
    put(config, (*root, "retracement_log_range_1h_coefficient"), 0.75)
    put(config, (*root, "retracement_log_range_1m_coefficient"), 0.875)
    bounds = ("optimize", "bounds", side, "strategy", "trailing_martingale", "close")
    put(config, (*bounds, "retracement_base_fraction"),
        [0.0, 0.0] if policy == "disabled_domain" else
        [0.0, 0.1] if policy == "reachable" else [0.01, 0.02])
    put(config, (*bounds, "retracement_log_range_1h_coefficient"), [0.25, 1.0])
    put(config, (*bounds, "retracement_log_range_1m_coefficient"), [0.5, 1.0])
    if policy == "runtime_disabled":
        config["optimize"]["fixed_runtime_overrides"] = {".".join((*root, "retracement_base_fraction")): 0.0}
    prepared = prepare_config(config, verbose=False)
    before = deepcopy(prepared)
    dead = optimizer_dead_param_values(prepared, globally_dead_only=globally_dead_only)
    expected = {
        f"{side}_close_retracement_base_pct": 0.0,
        f"{side}_close_retracement_volatility_1h_weight": 0.25,
        f"{side}_close_retracement_volatility_1m_weight": 0.5,
    }
    if globally_dead_only and policy == "reachable":
        assert not (expected.keys() & dead.keys())
    else:
        assert {key: dead[key] for key in expected} == expected
    assert prepared == before
