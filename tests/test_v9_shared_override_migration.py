"""Released shared-field selectors convert only in their owning side context."""
from copy import deepcopy

import pytest

from test_v9_config_migration import legacy
from tools.migrate_config_v9 import migrate


def put_patch(config, owner, patch):
    if owner == "fixed":
        config["optimize"]["fixed_runtime_overrides"] = patch
    else:
        scenario = {"label": "authored", "start_date": "2020-02-03",
                    "end_date": "2021-04-05", "overrides": patch}
        if owner == "suite":
            config["backtest"].pop("scenarios", None)
            config["backtest"]["suite"] = {"enabled": False, "scenarios": [scenario]}
        else:
            config["backtest"]["scenarios"] = [scenario]


def get_patch(config, owner):
    if owner == "fixed":
        return config["optimize"]["fixed_runtime_overrides"]
    return config["backtest"]["scenarios"][0]["overrides"]


@pytest.mark.parametrize("owner", ["scenario", "suite", "fixed"])
@pytest.mark.parametrize("side", ["long", "short"])
@pytest.mark.parametrize("syntax", ["dotted", "nested", "shorthand", "canonical"])
def test_shared_position_selector_converts_without_changing_authored_value(owner, side, syntax):
    source = legacy()
    canonical = f"bot.{side}.risk.max_positions"
    patch = ({f"bot.{side}.n_positions": 4} if syntax == "dotted" else
             {"bot": {side: {"n_positions": 4}}} if syntax == "nested" else
             {f"{side}.n_positions": 4} if syntax == "shorthand" else
             {canonical: 4})
    put_patch(source, owner, patch)
    before = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    assert get_patch(result.config, owner) == {canonical: 4}
    if owner != "fixed":
        scenario = result.config["backtest"]["scenarios"][0]
        assert (scenario["label"], scenario["start_date"], scenario["end_date"]) == (
            "authored", "2020-02-03", "2021-04-05")
    assert source == before
    assert migrate(result.config).config == result.config
    if syntax != "canonical":
        assert any(row["old"] == 4 and row["new"] == {canonical: 4}
                   and row["reason"] == "canonical v9 override path" for row in result.changes)


@pytest.mark.parametrize("owner", ["scenario", "suite", "fixed"])
@pytest.mark.parametrize("old,new,value", [
    ("total_wallet_exposure_limit", "risk.total_wallet_exposure_limit", 1.25),
    ("risk_we_excess_allowance_pct", "risk.we_excess_allowance_pct", 0.125),
    ("forager_score_weights.volume", "forager.score_weights.volume", 0.375),
    ("forager_volume_ema_span_1m", "forager.volume_ema_span_1m", 37.5),
    ("hsl_cooldown_minutes_after_red", "hsl.cooldown_minutes_after_red", 17.5),
    ("hsl_ema_span_minutes", "hsl.drawdown_ratio_ema_span_minutes", 6.25),
    ("unstuck_threshold", "unstuck.exposure_trigger_ratio", 0.625),
    ("risk_entry_cooldown_minutes", "entry_cooldown.base_duration_minutes", 9.5),
])
def test_other_declared_shared_fields_and_dictionary_tails_convert(owner, old, new, value):
    source = legacy()
    put_patch(source, owner, {"bot.long." + old: value})
    before = deepcopy(source)
    output = migrate(source, memory_span_minutes=123.5).config
    assert get_patch(output, owner) == {"bot.long." + new: value}
    assert source == before


@pytest.mark.parametrize("owner", ["scenario", "suite", "fixed"])
@pytest.mark.parametrize("nested", [False, True])
def test_flat_shared_dictionary_map_preserves_each_authored_leaf(owner, nested):
    source = legacy()
    weights = {"volume": 0.375, "volatility": 0.625}
    patch = ({"bot": {"long": {"forager_score_weights": weights}}} if nested else
             {"bot.long.forager_score_weights": weights})
    put_patch(source, owner, patch)
    before = deepcopy(source)
    if owner == "fixed" and not nested:
        # Fixed runtime overrides still require leaf selectors; conversion must
        # not broaden that contract by expanding an atomic dotted mapping.
        with pytest.raises(TypeError, match="must target leaf settings"):
            migrate(source, memory_span_minutes=123.5)
        assert source == before
        return
    output = migrate(source, memory_span_minutes=123.5).config
    expected = ({"bot.long.forager.score_weights.volume": 0.375,
                 "bot.long.forager.score_weights.volatility": 0.625} if nested else
                {"bot.long.forager.score_weights": weights})
    assert get_patch(output, owner) == expected
    assert source == before


@pytest.mark.parametrize("owner", ["scenario", "suite", "fixed"])
def test_canonical_strategy_leaves_and_grouped_shared_leaves_are_not_realiased(owner):
    source = legacy()
    patch = {"bot.long.strategy.trailing_martingale.entry.initial_qty_pct": 0.125,
             "bot.long.strategy.trailing_martingale.entry.threshold_base_pct": 0.025,
             "bot.long.risk.max_positions": 4,
             "bot.long.forager.score_weights.volume": 0.375}
    put_patch(source, owner, patch)
    before = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    assert get_patch(result.config, owner) == patch
    assert not any(row["reason"] == "canonical v9 override path" for row in result.changes)
    assert source == before


@pytest.mark.parametrize("owner", ["scenario", "suite", "fixed"])
@pytest.mark.parametrize("reverse", [False, True])
def test_flat_and_canonical_alias_conflicts_fail_without_precedence(owner, reverse):
    source = legacy()
    items = [("bot.long.n_positions", 4), ("bot.long.risk.max_positions", 6)]
    put_patch(source, owner, dict(reversed(items) if reverse else items))
    before = deepcopy(source)
    with pytest.raises(ValueError, match="conflicts with canonical path"):
        migrate(source, memory_span_minutes=123.5)
    assert source == before


@pytest.mark.parametrize("owner", ["scenario", "suite", "fixed"])
def test_equal_aliases_coalesce_to_one_canonical_selector(owner):
    source = legacy()
    put_patch(source, owner, {"bot.long.n_positions": 4, "bot.long.risk.max_positions": 4})
    assert get_patch(migrate(source, memory_span_minutes=123.5).config, owner) == {
        "bot.long.risk.max_positions": 4}


@pytest.mark.parametrize("owner", ["scenario", "suite", "fixed"])
@pytest.mark.parametrize("patch", [
    {"bot.long.n_postions": 4},
    {"bot": {"long": {"unknown_shared_field": 4}}},
    {"bot.long.risk.n_postions": 4},
    {"bot.long.strategy.trailing_martingale.entry.unknown_field": 0.1},
])
def test_unknown_fields_are_preserved_for_strict_rejection(owner, patch):
    source = legacy()
    put_patch(source, owner, patch)
    before = deepcopy(source)
    with pytest.raises(ValueError, match="(?:unknown configuration field|not overridable)"):
        migrate(source, memory_span_minutes=123.5)
    assert source == before


@pytest.mark.parametrize("owner", ["scenario", "suite", "fixed"])
def test_conversion_does_not_inject_unrequested_override_defaults(owner):
    source = legacy()
    put_patch(source, owner, {"bot.long.n_positions": 4})
    output = migrate(source, memory_span_minutes=123.5).config
    assert list(get_patch(output, owner)) == ["bot.long.risk.max_positions"]
    assert output["bot"]["long"]["risk"]["max_positions"] != 4


def test_fixed_parameter_selectors_use_same_context_aware_shared_conversion():
    source = legacy()
    source["optimize"]["fixed_params"] = ["bot.long.n_positions", "short.n_positions",
        "bot.long.strategy.trailing_martingale.entry.initial_qty_pct"]
    result = migrate(source, memory_span_minutes=123.5)
    assert result.config["optimize"]["fixed_params"] == ["bot.long.risk.max_positions",
        "bot.short.risk.max_positions", "bot.long.strategy.trailing_martingale.entry.initial_qty_pct"]


@pytest.mark.parametrize("owner", ["scenario", "suite", "fixed"])
def test_shared_conversion_preserves_separate_gate_acknowledgment(owner):
    source = legacy()
    put_patch(source, owner, {"bot.long.n_positions": 4, "live.max_realized_loss_pct": 0.1})
    before = deepcopy(source)
    with pytest.raises(ValueError, match="acknowledge-realized-loss-gate-memory-change"):
        migrate(source, memory_span_minutes=123.5)
    output = migrate(source, memory_span_minutes=123.5,
        acknowledge_realized_loss_gate_memory_change=True).config
    assert get_patch(output, owner) == {"bot.long.risk.max_positions": 4,
        "bot.risk.loss_gate.max_drawdown_ratio": 0.1}
    assert source == before


@pytest.mark.parametrize("selector,expected,family,sides", [
    ("n_positions", "max_positions", "n_positions", {"long", "short"}),
    ("risk.n_positions", "risk.max_positions", "n_positions", {"long", "short"}),
    ("*.n_positions", "*.max_positions", "n_positions", {"long", "short"}),
    ("long.n_positions", "bot.long.risk.max_positions", "n_positions", {"long"}),
    ("bot.*.risk.n_positions", "bot.*.risk.max_positions", "n_positions", {"long", "short"}),
    ("threshold", "exposure_trigger_ratio", "unstuck_threshold", {"long", "short"}),
    ("unstuck.threshold", "unstuck.exposure_trigger_ratio", "unstuck_threshold", {"long", "short"}),
    ("*.unstuck.threshold", "bot.*.unstuck.exposure_trigger_ratio", "unstuck_threshold", {"long", "short"}),
    ("short.unstuck_threshold", "bot.short.unstuck.exposure_trigger_ratio", "unstuck_threshold", {"short"}),
    ("ema_span_minutes", "drawdown_ratio_ema_span_minutes", "hsl_drawdown_ratio_ema_span_minutes", {"long", "short"}),
    ("hsl.ema_span_minutes", "hsl.drawdown_ratio_ema_span_minutes", "hsl_drawdown_ratio_ema_span_minutes", {"long", "short"}),
    ("*.hsl.ema_span_minutes", "bot.*.hsl.drawdown_ratio_ema_span_minutes", "hsl_drawdown_ratio_ema_span_minutes", {"long", "short"}),
    ("long.hsl_ema_span_minutes", "bot.long.hsl.drawdown_ratio_ema_span_minutes", "hsl_drawdown_ratio_ema_span_minutes", {"long"}),
    ("entry_cooldown_minutes", "entry_cooldown.base_duration_minutes", "risk_entry_cooldown_minutes", {"long", "short"}),
    ("risk.entry_cooldown_minutes", "entry_cooldown.base_duration_minutes", "risk_entry_cooldown_minutes", {"long", "short"}),
    ("long.risk.entry_cooldown_minutes", "bot.long.entry_cooldown.base_duration_minutes", "risk_entry_cooldown_minutes", {"long"}),
    ("*.risk.entry_cooldown_minutes", "bot.*.entry_cooldown.base_duration_minutes", "risk_entry_cooldown_minutes", {"long", "short"}),
])
@pytest.mark.parametrize("layout", ["grouped", "flat", "mixed"])
def test_released_atomic_shared_selectors_preserve_actual_final_bound_targets(
    selector, expected, family, sides, layout,
):
    from config.optimize_bounds import flatten_optimize_bounds
    from config.param_paths import resolve_bound_selectors

    source = legacy()
    # Released loading canonicalizes shared flat fields before selector matching.
    # The expected side sets above are goldens from that grouped consumer, not
    # invented raw flat paths or an imitation of the current resolver.
    for side in ("long", "short"):
        source["optimize"]["bounds"][side]["risk"]["n_positions"] = [1, 9, 1]
        if layout == "flat" or (layout == "mixed" and side == "long"):
            source["bot"][side]["n_positions"] = source["bot"][side]["risk"].pop("n_positions")
    source["optimize"]["fixed_params"] = [selector]
    before = deepcopy(source)
    result = migrate(source, memory_span_minutes=123.5)
    converted = result.config
    assert converted["optimize"]["fixed_params"] == [expected]
    bounds = flatten_optimize_bounds(
        converted["optimize"]["bounds"], strategy_kind=converted["live"]["strategy_kind"],
    )
    selected = resolve_bound_selectors(converted, converted["optimize"]["fixed_params"], bounds)
    assert set(selected) == {f"{side}_{family}" for side in sides}
    assert {path[1] for path in selected.values()} == sides
    assert source == before
    assert any(row["path"] == "optimize.fixed_params" and row["old"] == [selector]
               and row["new"] == [expected] for row in result.changes)
    assert migrate(converted).config == converted


@pytest.mark.parametrize("selector", [
    "bot.*.n_positions", "*.short.n_positions", "bot.*.unstuck_threshold",
    "custom_n_positions", "n_positions_extra", "risk.n_positions_extra",
    "unrelated.risk.n_positions", "unrelated.unstuck.threshold",
    "unrelated.hsl.ema_span_minutes", "bot.long.strategy.ema_anchor.n_positions",
])
def test_unmatched_shared_selectors_do_not_gain_authority(selector):
    from config.optimize_bounds import flatten_optimize_bounds
    from config.param_paths import resolve_bound_selectors

    source = legacy()
    source["optimize"]["fixed_params"] = [selector]
    converted = migrate(source, memory_span_minutes=123.5).config
    assert converted["optimize"]["fixed_params"] == [selector]
    bounds = flatten_optimize_bounds(converted["optimize"]["bounds"],
                                     strategy_kind=converted["live"]["strategy_kind"])
    assert resolve_bound_selectors(converted, [selector], bounds) == {}


@pytest.mark.parametrize("selector", ["risk", "long.risk", "bot.long.risk", "*.risk", "bot.*.risk"])
def test_changed_risk_group_selector_requires_explicit_canonical_choices(selector):
    source = legacy()
    source["optimize"]["fixed_params"] = [selector]
    before = deepcopy(source)
    with pytest.raises(ValueError, match="selector.*changed group authority.*explicit canonical"):
        migrate(source, memory_span_minutes=123.5)
    assert source == before


@pytest.mark.parametrize("selector,expected", [
    ("ema_span_minutes", "drawdown_ratio_ema_span_minutes"),
    ("hsl.ema_span_minutes", "hsl.drawdown_ratio_ema_span_minutes"),
    ("bot.hsl.ema_span_minutes", "bot.risk.hsl.drawdown_ratio_ema_span_minutes"),
])
def test_portfolio_smoothing_selectors_preserve_only_the_old_smoothing_gene(selector, expected):
    from config.optimize_bounds import flatten_optimize_bounds
    from config.param_paths import resolve_bound_selectors

    source = legacy()
    source["live"]["hsl_signal_mode"] = "unified"
    source["bot"]["hsl"] = deepcopy(source["bot"]["long"]["hsl"])
    source["optimize"]["bounds"]["hsl"] = source["optimize"]["bounds"]["long"].pop("hsl")
    source["optimize"]["bounds"]["short"].pop("hsl")
    source["optimize"]["fixed_params"] = [selector]
    converted = migrate(source, memory_span_minutes=123.5).config
    assert converted["optimize"]["fixed_params"] == [expected]
    bounds = flatten_optimize_bounds(converted["optimize"]["bounds"],
                                     strategy_kind=converted["live"]["strategy_kind"])
    assert resolve_bound_selectors(converted, [expected], bounds) == {
        "hsl_drawdown_ratio_ema_span_minutes": ("bot", "risk", "hsl", "drawdown_ratio_ema_span_minutes")
    }


def test_moved_portfolio_group_selector_does_not_freeze_new_memory_implicitly():
    source = legacy()
    source["live"]["hsl_signal_mode"] = "unified"
    source["bot"]["hsl"] = deepcopy(source["bot"]["long"]["hsl"])
    source["optimize"]["fixed_params"] = ["bot.hsl"]
    with pytest.raises(ValueError, match="bot.hsl.*explicit canonical intended fields"):
        migrate(source, memory_span_minutes=123.5)
