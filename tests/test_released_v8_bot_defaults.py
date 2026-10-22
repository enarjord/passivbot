"""Released root omissions keep historical values before v9 inheritance begins."""

from copy import deepcopy

import pytest

from config.migrations.released_v8 import (
    get_released_v8_defaults, materialize_released_v8_bot_defaults,
    resolve_released_v8_defaults,
)
from config.migrations.v9 import migrate_v8_to_v9


def released(kind="trailing_martingale"):
    source = get_released_v8_defaults(strategy_kind=kind)
    source["backtest"]["visible_metrics"] = [name for name in source["backtest"]["visible_metrics"]
                                             if not name.endswith("_w")]
    source["optimize"]["scoring"] = [row for row in source["optimize"]["scoring"]
                                      if not row["metric"].endswith("_w")]
    return source


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
@pytest.mark.parametrize("side", ["long", "short"])
def test_root_omitted_side_is_pinned_to_the_complete_frozen_profile(kind, side):
    source = released(kind)
    source["bot"].pop(side)
    source["bot"]["short" if side == "long" else "long"].pop("forager")
    original = deepcopy(source)
    expected = resolve_released_v8_defaults(source)["bot"]
    candidate = deepcopy(source)
    changes = []
    materialize_released_v8_bot_defaults(candidate, changes=changes)
    assert candidate["bot"] == expected
    assert candidate["bot"][side]["risk"]["total_wallet_exposure_limit"] == 0.0
    assert any(row["path"] == f"bot.{side}.risk.total_wallet_exposure_limit"
               and row["new"] == 0.0 for row in changes)
    assert source == original


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_missing_unstuck_spans_still_derive_from_the_authored_strategy(kind):
    source = released(kind)
    for key in ("ema_span_0", "ema_span_1"):
        source["bot"]["long"]["unstuck"].pop(key)
        source["optimize"]["bounds"]["long"]["unstuck"].pop(key)
    strategy = source["bot"]["long"]["strategy"][kind]
    if kind == "trailing_martingale":
        strategy = strategy["entry"]
    strategy["ema_span_0"] = 403.5
    original = deepcopy(source)
    output = migrate_v8_to_v9(source, memory_span_minutes=123.5).config
    assert output["bot"]["long"]["unstuck"]["ema_span_0_minutes"] == 403.5
    assert output["optimize"]["bounds"]["long"]["unstuck"]["ema_span_0_minutes"] == [403.5, 403.5]
    assert source == original


def test_default_enablement_cannot_authorize_removing_an_ambiguous_null_restart():
    source = released()
    source["bot"]["long"]["hsl"].pop("enabled")
    source["bot"]["long"]["hsl"]["restart_after_red_policy"] = None
    original = deepcopy(source)
    with pytest.raises(ValueError, match="no v9 policy equivalent"):
        migrate_v8_to_v9(source, memory_span_minutes=123.5)
    assert source == original


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor"])
def test_omitted_short_positive_search_bounds_keep_old_runtime_policy_and_coupled_spans(kind):
    source = released(kind)
    source["bot"].pop("short")
    source["bot"]["long"].pop("forager")
    source["optimize"]["bounds"]["short"]["risk"].update(
        n_positions=[1, 4, 1], total_wallet_exposure_limit=[0.0, 1.0],
    )
    original = deepcopy(source)
    historical = resolve_released_v8_defaults(source)
    expected = migrate_v8_to_v9(historical, memory_span_minutes=123.5).config
    result = migrate_v8_to_v9(source, memory_span_minutes=123.5)
    for side in ("long", "short"):
        assert result.config["bot"][side] == expected["bot"][side]
    short = result.config["bot"]["short"]
    assert short["entry_cooldown"]["base_duration_minutes"] == 0.0
    assert short["risk"]["total_wallet_exposure_limit"] == 0.0
    assert short["risk"]["max_positions"] == 1
    assert short["enabled"] is True  # Historical search bounds can still activate capacity.
    assert short["unstuck"]["ema_span_0_minutes"] == 60.0
    assert short["unstuck"]["ema_span_1_minutes"] == 60.0
    assert result.config["optimize"]["bounds"]["short"]["risk"]["total_wallet_exposure_limit"] == [0.0, 1.0]
    assert source == original


@pytest.mark.parametrize("path,value", [
    (("n_positions",), 3),
    (("risk_entry_cooldown_minutes",), 7.25),
    (("risk", "we_excess_allowance_fraction"), 0.123),
    (("forager_volatility_ema_span_1m",), 13.5),
    (("strategy", "trailing_martingale", "entry", "ema_span_0_minutes"), 123.5),
])
def test_explicit_alias_presence_precedes_frozen_root_defaults(path, value):
    source = released()
    source["bot"]["long"] = {}
    node = source["bot"]["long"]
    for part in path[:-1]:
        node = node.setdefault(part, {})
    node[path[-1]] = value
    original = deepcopy(source)
    result = migrate_v8_to_v9(source, memory_span_minutes=123.5)
    expected_paths = {
        ("n_positions",): ("risk", "max_positions"),
        ("risk_entry_cooldown_minutes",): ("entry_cooldown", "base_duration_minutes"),
        ("risk", "we_excess_allowance_fraction"): ("risk", "we_excess_allowance_fraction"),
        ("forager_volatility_ema_span_1m",): ("forager", "volatility_ema_span_minutes"),
        ("strategy", "trailing_martingale", "entry", "ema_span_0_minutes"): path,
    }
    selected = result.config["bot"]["long"]
    for part in expected_paths[path]:
        selected = selected[part]
    assert selected == value
    assert source == original


def test_conflicting_aliases_still_fail_before_root_default_materialization():
    source = released()
    source["bot"]["long"]["n_positions"] = 3
    original = deepcopy(source)
    with pytest.raises(ValueError, match="conflicts"):
        migrate_v8_to_v9(source, memory_span_minutes=123.5)
    assert source == original


def test_sparse_coin_and_scenario_fixed_replacements_do_not_gain_root_defaults():
    source = released()
    source["coin_overrides"] = {"BTC": {"bot": {"short": {"unstuck": {"close_pct": 0.007}}}}}
    source["backtest"]["scenarios"] = [{"label": "partial", "overrides": {
        "bot.short": {"risk": {"n_positions": 2}},
    }}]
    source["optimize"]["fixed_runtime_overrides"] = {"bot.long.risk.n_positions": 3}
    result = migrate_v8_to_v9(source, memory_span_minutes=123.5).config
    assert result["coin_overrides"]["BTC"]["bot"]["short"] == {"unstuck": {"close_fraction": 0.007}}
    assert result["backtest"]["scenarios"][0]["overrides"] == {"bot.short": {"risk": {"max_positions": 2}}}
    assert result["optimize"]["fixed_runtime_overrides"] == {"bot.long.risk.max_positions": 3}
    sparse = {"config_version": "v8.6.0", "bot": {"short": {"unstuck": {"close_pct": 0.007}}}}
    output = migrate_v8_to_v9(sparse, _coin_patch=True).config
    assert output["bot"] == {"short": {"unstuck": {"close_fraction": 0.007}}}
