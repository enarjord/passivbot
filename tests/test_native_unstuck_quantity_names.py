"""Native unstuck units and released-v8 conversion have separate vocabularies."""

from copy import deepcopy
import json

import pytest

from config import prepare_config
from config.migrations.released_v8_unstuck import RELEASED_V8_UNSTUCK_FLAT_PATHS
from config.optimize_bounds import flatten_optimize_bounds
from config.param_paths import resolve_bound_selectors
from config.schema import get_expanded_config_prototype
from config.shared_bot import flatten_shared_bot_side
from optimization.config_adapter import get_optimization_key_paths
from released_v8_fixtures import released_v8_policy
from tools.migrate_config_v9 import migrate


FIELDS = (
    ("unstuck_close_pct", "unstuck_close_fraction", "close_pct", "close_fraction", .175),
    ("unstuck_ema_dist", "unstuck_ema_distance_fraction", "ema_dist", "ema_distance_fraction", -.0125),
    ("unstuck_ema_span_0", "unstuck_ema_span_0_minutes", "ema_span_0", "ema_span_0_minutes", 13.75),
    ("unstuck_ema_span_1", "unstuck_ema_span_1_minutes", "ema_span_1", "ema_span_1_minutes", 47.125),
    ("unstuck_loss_allowance_pct", "unstuck_loss_allowance_fraction", "loss_allowance_pct", "loss_allowance_fraction", .0375),
    ("unstuck_threshold", "unstuck_exposure_trigger_ratio", "threshold", "exposure_trigger_ratio", .625),
)


@pytest.mark.parametrize("side", ["long", "short"])
def test_public_values_lower_to_only_current_native_ids_without_rescaling(side):
    source = get_expanded_config_prototype()
    for _, _, _, public, value in FIELDS:
        source["bot"][side]["unstuck"][public] = value
    config = prepare_config(source, verbose=False)
    flat = flatten_shared_bot_side(config["bot"][side])
    paths = dict(get_optimization_key_paths(source))
    for old, new, _, public, value in FIELDS:
        assert old not in flat
        assert flat[new] == value
        assert f"{side}_{old}" not in paths
        assert paths[f"{side}_{new}"] == ("bot", side, "unstuck", public)


@pytest.mark.parametrize("old,new,legacy,public,value", FIELDS)
def test_native_json_rejects_removed_unstuck_spelling(old, new, legacy, public, value):
    import passivbot_rust as pbr
    from test_orchestrator_json_api import make_input, make_symbol

    payload = make_input(balance=1000.0, symbols=[make_symbol(0, bid=100.0, ask=100.0)])
    bot = payload["global"]["global_bot_params"]["long"]
    bot[old] = bot.pop(new)
    with pytest.raises(ValueError, match=f"{old}|{new}"):
        pbr.compute_ideal_orders_json(json.dumps(payload))


def test_native_json_rejects_removed_loss_gate_spelling():
    import passivbot_rust as pbr
    from test_orchestrator_json_api import make_input, make_symbol

    payload = make_input(balance=1000.0, symbols=[make_symbol(0, bid=100.0, ask=100.0)])
    payload["global"]["max_realized_loss_pct"] = payload["global"].pop("loss_gate_max_drawdown_fraction")
    with pytest.raises(ValueError, match="max_realized_loss_pct|loss_gate_max_drawdown_fraction"):
        pbr.compute_ideal_orders_json(json.dumps(payload))


def test_released_v8_flat_values_bounds_fixed_and_scenario_keep_final_authority():
    source = released_v8_policy()
    source["optimize"]["bounds"] = {}
    source["optimize"]["fixed_params"] = []
    overrides = {}
    for old, _, legacy, _, value in FIELDS:
        source["bot"]["long"]["unstuck"].pop(legacy)
        source["bot"]["long"][old] = value
        source["optimize"]["bounds"][f"long_{old}"] = [value, value]
        source["optimize"]["fixed_params"].append(f"bot.long.{old}")
        overrides[f"bot.short.{old}"] = value
    source["backtest"]["scenarios"] = [{"label": "fractional", "overrides": overrides}]
    original = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    assert source == original
    converted = result.config
    bounds = flatten_optimize_bounds(converted["optimize"]["bounds"], strategy_kind="trailing_martingale")
    selected = resolve_bound_selectors(converted, converted["optimize"]["fixed_params"], bounds)
    assert set(selected) == {f"long_{new}" for _, new, *_ in FIELDS}
    for old, new, _, public, value in FIELDS:
        assert converted["bot"]["long"]["unstuck"][public] == value
        assert bounds[f"long_{new}"] == [value, value]
        target = f"bot.long.unstuck.{public}"
        assert target in converted["optimize"]["fixed_params"]
        assert converted["backtest"]["scenarios"][0]["overrides"][f"bot.short.unstuck.{public}"] == value
        assert any(change["path"] == f"bot.long.{old}" for change in result.changes)


@pytest.mark.parametrize("old,new,legacy,public,value", FIELDS)
def test_released_v8_conflicting_flat_and_grouped_values_are_rejected_without_mutation(old, new, legacy, public, value):
    source = released_v8_policy()
    source["bot"]["long"][old] = value
    source["bot"]["long"]["unstuck"][legacy] = value + .125
    original = deepcopy(source)
    with pytest.raises(ValueError, match="conflicts"):
        migrate(source, memory_span_minutes=123.5)
    assert source == original


def test_released_v8_file_then_inline_flat_span_preserves_precedence_and_source(tmp_path):
    source = released_v8_policy()
    document = {"bot": {"long": {"unstuck_ema_span_0": 17.5}}}
    path = tmp_path / "coin.json"
    path.write_text(json.dumps(document))
    original_bytes = path.read_bytes()
    source["coin_overrides"] = {"BTC": {
        "override_config_path": "coin.json", "bot": {"long": {"unstuck_ema_span_0": 23.75}},
    }}
    original = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5, base_config_path=str(tmp_path / "base.json"))
    assert source == original and path.read_bytes() == original_bytes
    assert result.config["coin_overrides"]["BTC"]["bot"]["long"]["unstuck"]["ema_span_0_minutes"] == 23.75
    assert any("coin_overrides.BTC" in change["path"] and "unstuck_ema_span_0" in change["path"]
               for change in result.changes)


def test_released_unstuck_vocabulary_is_frozen():
    assert len(RELEASED_V8_UNSTUCK_FLAT_PATHS) == 6
    with pytest.raises(TypeError):
        RELEASED_V8_UNSTUCK_FLAT_PATHS["unstuck_close_pct"] = ("unstuck", "wrong")
