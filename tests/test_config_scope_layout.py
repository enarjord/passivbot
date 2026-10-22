"""Canonical scopes and bounds address exactly one public parameter path."""

from copy import deepcopy

import pytest

from config import prepare_config
from config.hsl import generated_template
from config.optimize_bounds import flatten_optimize_bounds
from config.param_paths import resolve_optimizer_key_path
from config.schema import get_template_config
from released_v8_fixtures import released_v8_policy
from tools.migrate_config_v9 import migrate


@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("scope", ["coin", "position_side", "portfolio"])
def test_canonical_bounds_mirror_their_bot_parameter_paths(strategy, scope):
    source = generated_template(get_template_config(), scope)
    source["bot"]["portfolio"]["strategy_kind"] = strategy
    config = prepare_config(source, verbose=False)
    bounds = config["optimize"]["bounds"]
    flat = flatten_optimize_bounds(bounds, strategy_kind=strategy)
    assert flat
    for key, bound in flat.items():
        path = resolve_optimizer_key_path(config, key)
        assert path is not None, key
        target = bounds
        parameter = config["bot"]
        for part in path[1:]:
            target = target[part]
            parameter = parameter[part]
        assert target == bound
        assert not isinstance(parameter, dict)


@pytest.mark.parametrize("retired", ["risk", "unstuck"])
def test_development_v9_scope_layouts_are_rejected_without_conversion(retired):
    source = get_template_config()
    source["bot"][retired] = {}
    original = deepcopy(source)
    with pytest.raises(ValueError, match="unknown configuration field"):
        prepare_config(source, verbose=False)
    assert source == original


def test_flat_forager_bound_spelling_is_not_public_authoring_authority():
    source = get_template_config()
    source["optimize"]["bounds"]["long"]["forager"]["score_weights_volume"] = [0.0, 1.0]
    with pytest.raises(ValueError, match="unknown configuration field"):
        prepare_config(source, verbose=False)


def test_global_loss_memories_remain_independent():
    source = get_template_config()
    source["bot"]["portfolio"]["unstuck"]["memory_ema_span_minutes"] = 37.5
    source["bot"]["portfolio"]["loss_gate"]["memory_ema_span_minutes"] = 93.5
    source["bot"]["long"]["hsl"]["memory_ema_span_minutes"] = 17.5
    config = prepare_config(source, verbose=False)
    assert config["bot"]["portfolio"]["unstuck"]["memory_ema_span_minutes"] == 37.5
    assert config["bot"]["portfolio"]["loss_gate"]["memory_ema_span_minutes"] == 93.5
    assert config["bot"]["long"]["hsl"]["memory_ema_span_minutes"] == 17.5


@pytest.mark.parametrize("nested_first", [False, True])
@pytest.mark.parametrize("equal", [False, True])
def test_released_weight_bound_alias_conflicts_do_not_depend_on_insertion_order(nested_first, equal):
    source = released_v8_policy()
    original = deepcopy(source)
    forager = source["optimize"]["bounds"]["long"]["forager"]
    flat = forager.pop("score_weights_volume")
    nested = flat if equal else [0.2, 0.8, 0.1]
    items = [("score_weights_volume", flat), ("score_weights", {"volume": nested})]
    for key, value in (reversed(items) if nested_first else items):
        forager[key] = deepcopy(value)
    authored = deepcopy(source)
    if equal:
        migration = migrate(source, memory_span_minutes=123.5, acknowledge_search_backend_change=True)
        converted = migration.config
        assert converted["optimize"]["bounds"]["long"]["forager"]["score_weights"]["volume"] == flat
        assert any(
            record["path"] == "optimize.bounds.long.forager.score_weights_volume"
            and record["new"] == {"optimize.bounds.long.forager.score_weights.volume": flat}
            and not record.get("best_effort")
            for record in migration.changes
        )
    else:
        with pytest.raises(ValueError, match="score_weights_volume conflicts with .*score_weights.volume"):
            migrate(source, memory_span_minutes=123.5, acknowledge_search_backend_change=True)
    assert source == authored
    assert original["optimize"]["bounds"]["long"]["forager"]["score_weights_volume"] == flat
