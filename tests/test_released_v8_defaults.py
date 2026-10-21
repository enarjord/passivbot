"""Released defaults and authored inheritance survive current-schema evolution."""

from copy import deepcopy

import pytest

from config import prepare_config
from config.migrations.released_v8 import (
    get_released_v8_defaults,
    resolve_released_v8_defaults,
    resolve_released_v8_minimum_cost_policy,
)
from config.migrations.v9 import migrate_v8_to_v9
from released_v8_fixtures import released_v8_policy
from tools.migrate_config_v9 import migrate


def test_snapshot_contains_public_released_policy_instead_of_v9_substitutes():
    config = get_released_v8_defaults()
    assert config["config_version"] == "v8.6.0"
    assert config["live"]["filter_by_min_effective_cost"] is True
    assert config["backtest"]["filter_by_min_effective_cost"] is False
    assert config["bot"]["long"]["risk"]["n_positions"] == 7.0
    assert config["bot"]["short"]["risk"]["n_positions"] == 1.0
    assert config["bot"]["long"]["hsl"]["restart_after_red_policy"] == "always"
    assert config["optimize"]["bounds"]["short"]["risk"]["n_positions"] == [5, 5, 1]
    assert config["optimize"]["gpu"]["validate_per_generation"] == 8
    assert {record["metric"] for record in config["optimize"]["scoring"]} >= {
        "adg_strategy_eq", "adg_strategy_eq_w",
    }
    assert set(config["bot"]["long"]["strategy"]) == {
        "trailing_martingale", "ema_anchor", "trailing_grid_v7",
    }
    assert "max_positions" not in config["bot"]["long"]["risk"]


@pytest.mark.parametrize("kind", ["trailing_martingale", "ema_anchor", "trailing_grid_v7"])
def test_active_strategy_projection_and_returned_copies_are_independent(kind):
    config = get_released_v8_defaults(strategy_kind=kind)
    original = deepcopy(config)
    assert config["live"]["strategy_kind"] == kind
    for side in ("long", "short"):
        assert set(config["bot"][side]["strategy"]) == {kind}
        assert set(config["optimize"]["bounds"][side]["strategy"]) == {kind}
    config["optimize"]["bounds"]["long"]["risk"]["n_positions"][0] = 999
    config["bot"]["short"]["strategy"][kind].clear()
    assert get_released_v8_defaults(strategy_kind=kind) == original


def test_sparse_released_defaults_keep_disabled_sides_and_empty_coin_universe():
    source = {
        "config_version": "v8.6.0", "bot": {"long": {}},
        "live": {"approved_coins": {"long": ["BTC"]}},
        "optimize": {
            "bounds": {"long": {"risk": {"n_positions": [2, 4, 1]}}},
            "fixed_runtime_overrides": {},
        },
    }
    original = deepcopy(source)
    config = resolve_released_v8_defaults(source)
    assert source == original
    assert config["bot"]["long"]["risk"]["total_wallet_exposure_limit"] == 1.5
    assert config["bot"]["short"]["risk"]["total_wallet_exposure_limit"] == 0.0
    assert config["live"]["approved_coins"] == {"long": ["BTC"], "short": []}
    assert config["live"]["ignored_coins"] == {"long": [], "short": []}
    assert config["optimize"]["bounds"]["long"]["risk"]["n_positions"] == [2, 4, 1]
    assert config["optimize"]["bounds"]["short"]["risk"]["n_positions"] == [5, 5, 1]
    assert config["optimize"]["fixed_runtime_overrides"] == {}


def test_default_resolution_preserves_authored_patches_unknowns_and_dynamic_values():
    source = {
        "config_version": "v8.6.0", "bot": {},
        "live": {"approved_coins": "approved.json", "startup_phase_budgets": {"candles": 10}},
        "backtest": {
            "end_date": "now", "filter_by_min_effective_cost": None,
            "reducer": {"adg_strategy_eq": "min"},
            "scenarios": [{"label": "one", "overrides": {"live.warmup_ratio": 0.5}}],
        },
        "coin_overrides": {"BTC": {"live": {"forced_mode_long": "normal"}}},
        "unrecognized": {"must_survive_for_validation": True},
    }
    original = deepcopy(source)
    config = resolve_released_v8_defaults(source)
    assert source == original
    assert config["live"]["approved_coins"] == "approved.json"
    for key, value in source["backtest"].items():
        assert config["backtest"][key] == value
    assert config["coin_overrides"] == source["coin_overrides"]
    assert config["unrecognized"] == source["unrecognized"]
    assert resolve_released_v8_defaults(config) == config


@pytest.mark.parametrize("source", [{}, {"config_version": "v9.0.0"}, {"config_version": "v8"}])
def test_default_resolver_is_only_for_versioned_released_v8(source):
    with pytest.raises(ValueError, match="versioned v8"):
        resolve_released_v8_defaults(source)


_MISSING = object()


@pytest.mark.parametrize("live", [_MISSING, False, True])
@pytest.mark.parametrize("backtest", [_MISSING, None, False, True])
def test_missing_and_null_minimum_cost_policy_have_distinct_meanings(live, backtest):
    source = {"live": {}, "backtest": {}}
    if live is not _MISSING:
        source["live"]["filter_by_min_effective_cost"] = live
    if backtest is not _MISSING:
        source["backtest"]["filter_by_min_effective_cost"] = backtest
    original = deepcopy(source)
    expected_live = True if live is _MISSING else live
    expected_backtest = False if backtest is _MISSING else expected_live if backtest is None else backtest
    assert resolve_released_v8_minimum_cost_policy(source) == {
        "live": expected_live, "backtest": expected_backtest,
    }
    assert source == original


@pytest.mark.parametrize("live,backtest", [(None, None), ("false", None), (0, None), (True, 1), (True, "false")])
def test_malformed_minimum_cost_policy_is_not_coerced(live, backtest):
    source = {"live": {"filter_by_min_effective_cost": live},
              "backtest": {"filter_by_min_effective_cost": backtest}}
    with pytest.raises(TypeError, match="must be a boolean"):
        resolve_released_v8_minimum_cost_policy(source)


@pytest.mark.parametrize("converter", [migrate_v8_to_v9, migrate])
@pytest.mark.parametrize("backtest", [_MISSING, None, False, True])
def test_explicit_migration_materializes_missing_defaults_and_retains_null(converter, backtest):
    source = released_v8_policy()
    del source["live"]["filter_by_min_effective_cost"]
    if backtest is _MISSING:
        del source["backtest"]["filter_by_min_effective_cost"]
    else:
        source["backtest"]["filter_by_min_effective_cost"] = backtest
    before = deepcopy(source)
    result = converter(source, memory_span_minutes=123.5)
    assert source == before
    assert result.config["bot"]["long"]["universe"]["filter_by_min_effective_cost"] is True
    assert result.config["backtest"]["filter_by_min_effective_cost"] is (
        False if backtest is _MISSING else backtest
    )
    reports = {item["path"]: item for item in result.changes
               if "materialized released-v8 default" in item["reason"]}
    assert set(reports) == {"live.filter_by_min_effective_cost"} | (
        {"backtest.filter_by_min_effective_cost"} if backtest is _MISSING else set()
    )
    assert all(not item.get("best_effort") for item in reports.values())
    loaded = prepare_config(result.config, verbose=False)
    from config.policy import effective_minimum_cost_filter
    assert loaded["backtest"]["filter_by_min_effective_cost"] is (False if backtest is _MISSING else backtest)
    assert effective_minimum_cost_filter(loaded, "long") is (True if backtest is None else False if backtest is _MISSING else backtest)


def test_migrated_null_tracks_shared_policy_when_authored_config_is_reused():
    source = released_v8_policy()
    source["backtest"]["filter_by_min_effective_cost"] = None
    result = migrate(source, memory_span_minutes=123.5)
    assert result.config["backtest"]["filter_by_min_effective_cost"] is None
    for enabled in (False, True):
        config = deepcopy(result.config)
        from config.policy import effective_minimum_cost_filter
        config["bot"]["long"]["universe"]["filter_by_min_effective_cost"] = enabled
        loaded = prepare_config(config, verbose=False)
        assert loaded["backtest"]["filter_by_min_effective_cost"] is None
        assert effective_minimum_cost_filter(loaded, "long") is enabled
    assert migrate(result.config).config == result.config


@pytest.mark.parametrize("structured", [False, True])
def test_migration_keeps_explicit_null_in_scenario_and_fixed_policy_patches(structured):
    source = released_v8_policy()
    patch = ({"backtest": {"filter_by_min_effective_cost": None}} if structured else
             {"backtest.filter_by_min_effective_cost": None})
    source["optimize"]["fixed_runtime_overrides"] = deepcopy(patch)
    source["backtest"]["scenarios"] = [{"label": "inherit", "overrides": deepcopy(patch)}]
    result = migrate(source, memory_span_minutes=123.5)
    expected = {"backtest.filter_by_min_effective_cost": None}
    assert result.config["optimize"]["fixed_runtime_overrides"] == expected
    assert result.config["backtest"]["scenarios"][0]["overrides"] == expected


def test_full_public_released_defaults_report_metric_and_memory_choices():
    source = get_released_v8_defaults(strategy_kind="trailing_martingale")
    before = deepcopy(source)
    with pytest.raises(ValueError, match="retired metric"):
        migrate(source, memory_span_minutes=123.5)
    result = migrate(source, best_effort=True)
    assert source == before
    assert result.config["bot"]["long"]["risk"]["max_positions"] == 7
    assert result.config["optimize"]["pymoo"] == source["optimize"]["pymoo"]
    assert result.config["backtest"]["filter_by_min_effective_cost"] is False
    guesses = [item for item in result.changes if item.get("best_effort")]
    assert any(item["old"] == "adg_strategy_eq_w" and item["new"] == "adg_strategy_eq_ew" for item in guesses)
    assert any(item["path"] == "backtest.ew_half_life_days" for item in guesses)
    assert any(item["path"] == "live.pnls_max_lookback_days" for item in guesses)
    assert migrate(result.config).changes == []
