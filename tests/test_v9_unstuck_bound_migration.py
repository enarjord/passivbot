"""Explicit conversion preserves newly independent unstuck search dimensions."""

from copy import deepcopy

import pytest

from config import get_template_config, prepare_config
from config.migrations.v9 import migrate_v8_to_v9
from config.optimize_bounds import flatten_optimize_bounds


def _released_config(kind="trailing_martingale"):
    source = get_template_config()
    source["config_version"] = "v8.6.0"
    source["live"]["strategy_kind"] = kind
    for side in ("long", "short"):
        for key in ("ema_span_0", "ema_span_1"):
            source["bot"][side]["unstuck"].pop(key)
            source["optimize"]["bounds"][side]["unstuck"].pop(key)
    return source


def _convert(source):
    before = deepcopy(source)
    with pytest.raises(ValueError, match="explicit migration"):
        prepare_config(source, verbose=False)
    result = migrate_v8_to_v9(source, memory_span_minutes=43200.0)
    assert source == before
    prepared = prepare_config(result.config, verbose=False)
    assert prepare_config(prepared, verbose=False)["optimize"]["bounds"] == prepared["optimize"]["bounds"]
    return prepared


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("omitted", ["bounds", "optimize"])
def test_omitted_search_policy_fixes_new_unstuck_spans_at_effective_values(kind, omitted):
    source = _released_config(kind)
    if omitted == "bounds":
        source["optimize"].pop("bounds")
    else:
        source.pop("optimize")
    strategy = source["bot"]["long"]["strategy"][kind]
    if kind == "trailing_martingale":
        strategy = strategy["entry"]
    strategy["ema_span_0"] = 403.5
    prepared = _convert(source)
    assert prepared["bot"]["long"]["unstuck"]["ema_span_0"] == 403.5
    assert prepared["optimize"]["bounds"]["long"]["unstuck"]["ema_span_0"] == [403.5, 403.5]


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("bound", [12.5, [12.5], [12.5, 12.5], [10, 20]])
def test_released_flat_bounds_preserve_fixed_or_authored_unstuck_choices(bound, explicit):
    source = _released_config()
    source["optimize"]["bounds"] = flatten_optimize_bounds(
        source["optimize"]["bounds"], strategy_kind="trailing_martingale",
    )
    source["optimize"]["bounds"]["long_ema_span_0"] = bound
    if explicit:
        source["optimize"]["bounds"]["long_unstuck_ema_span_0"] = [34.5, 34.5]
    prepared = _convert(source)
    expected = (34.5 if explicit else
                source["bot"]["long"]["strategy"]["trailing_martingale"]["entry"]["ema_span_0"]
                if bound == [10, 20] else 12.5)
    assert prepared["optimize"]["bounds"]["long"]["unstuck"]["ema_span_0"] == [expected, expected]
    assert not any(key.startswith(("long_", "short_")) for key in prepared["optimize"]["bounds"])


def test_mixed_released_flat_and_grouped_bound_conflict_is_not_overwritten():
    source = _released_config()
    source["optimize"]["bounds"]["long_ema_span_0"] = [12.5, 12.5]
    source["optimize"]["bounds"]["long"]["strategy"]["trailing_martingale"]["entry"]["ema_span_0"] = [20, 30]
    before = deepcopy(source)
    with pytest.raises(ValueError, match="conflict"):
        migrate_v8_to_v9(source, memory_span_minutes=43200.0)
    assert source == before


def test_released_flat_bound_conversion_does_not_discard_unknown_root_fields():
    source = _released_config()
    source["optimize"]["bounds"] = flatten_optimize_bounds(
        source["optimize"]["bounds"], strategy_kind="trailing_martingale",
    )
    source["optimize"]["bounds"]["long_ema_span_0"] = [12.5, 12.5]
    source["optimize"]["bounds"]["misspelled_bound"] = [1, 2]
    with pytest.raises(ValueError, match="misspelled_bound.*unknown"):
        migrate_v8_to_v9(source, memory_span_minutes=43200.0)
