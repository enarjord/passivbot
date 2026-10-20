"""Canonical scopes and bounds address exactly one public parameter path."""

from copy import deepcopy

import pytest

from config import prepare_config
from config.hsl import generated_template
from config.optimize_bounds import flatten_optimize_bounds
from config.param_paths import resolve_optimizer_key_path
from config.schema import get_template_config


@pytest.mark.parametrize("strategy", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("scope", ["coin", "pside", "portfolio"])
def test_canonical_bounds_mirror_their_bot_parameter_paths(strategy, scope):
    source = generated_template(get_template_config(), scope)
    source["live"]["strategy_kind"] = strategy
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
    source["bot"]["global"]["unstuck"]["memory_ema_span_minutes"] = 37.5
    source["bot"]["global"]["loss_gate"]["memory_ema_span_minutes"] = 93.5
    source["bot"]["long"]["hsl"]["memory_ema_span_minutes"] = 17.5
    config = prepare_config(source, verbose=False)
    assert config["bot"]["global"]["unstuck"]["memory_ema_span_minutes"] == 37.5
    assert config["bot"]["global"]["loss_gate"]["memory_ema_span_minutes"] == 93.5
    assert config["bot"]["long"]["hsl"]["memory_ema_span_minutes"] == 17.5
