"""
Verification tests for the Data Strategy Redesign.

These tests implement the test cases documented in docs/data_strategy_verification_tests.md.
They verify the new flattened scenario configuration system works correctly across all use cases.
"""

from copy import deepcopy

import pytest

from config_utils import format_config, get_template_config
from suite_runner import (
    SuiteScenario,
    build_scenarios,
    collect_suite_coin_sources,
    extract_suite_config,
    filter_scenarios_by_label,
)


def make_test_config(**backtest_overrides):
    """Create a test config with backtest overrides."""
    cfg = get_template_config()
    cfg["_raw"] = deepcopy(cfg)
    for key, value in backtest_overrides.items():
        cfg["backtest"][key] = value
    return cfg


# =============================================================================
# Category 1: Basic Scenario Configurations
# =============================================================================


class TestBasicScenarioConfigurations:
    """TC1.x: Basic scenario configuration tests."""

    def test_tc1_1_empty_scenarios_non_suite_mode(self):
        """TC1.1: Empty scenarios means non-suite mode (single backtest)."""
        cfg = make_test_config(exchanges=["binance"], scenarios=[])
        formatted = format_config(deepcopy(cfg), verbose=False)

        suite_cfg = extract_suite_config(formatted, None)
        # Empty scenarios = not enabled
        assert suite_cfg.get("enabled") is False or suite_cfg.get("scenarios", []) == []

    def test_tc1_2_single_scenario_inheriting_exchanges(self):
        """TC1.2: Single scenario without exchanges inherits from base."""
        cfg = make_test_config(
            exchanges=["binance", "bybit"],
            scenarios=[{"label": "default"}],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance", "bybit"])

        assert len(scenarios) == 1
        assert scenarios[0].label == "default"
        assert scenarios[0].exchanges == ["binance", "bybit"]

    def test_tc1_3_single_scenario_explicit_single_exchange(self):
        """TC1.3: Single scenario with explicit single exchange."""
        cfg = make_test_config(
            exchanges=["binance", "bybit"],
            scenarios=[{"label": "binance_only", "exchanges": ["binance"]}],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance", "bybit"])

        assert len(scenarios) == 1
        assert scenarios[0].exchanges == ["binance"]

    def test_tc1_4_single_scenario_explicit_multiple_exchanges(self):
        """TC1.4: Scenario can override base to use more exchanges."""
        cfg = make_test_config(
            exchanges=["binance"],  # Base has 1
            scenarios=[{"label": "combined", "exchanges": ["binance", "bybit"]}],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert len(scenarios) == 1
        assert scenarios[0].exchanges == ["binance", "bybit"]

    def test_tc1_5_multiple_scenarios_all_inheriting(self):
        """TC1.5: Multiple scenarios all inheriting base exchanges."""
        cfg = make_test_config(
            exchanges=["binance", "bybit"],
            scenarios=[
                {"label": "scenario_a"},
                {"label": "scenario_b"},
                {"label": "scenario_c"},
            ],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance", "bybit"])

        assert len(scenarios) == 3
        for scenario in scenarios:
            assert scenario.exchanges == ["binance", "bybit"]

    def test_tc1_6_multiple_scenarios_different_single_exchanges(self):
        """TC1.6: Multiple scenarios with different single exchanges (original failing case)."""
        cfg = make_test_config(
            exchanges=["binance", "bybit"],
            scenarios=[
                {"label": "binance", "exchanges": ["binance"]},
                {"label": "bybit", "exchanges": ["bybit"]},
            ],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance", "bybit"])

        assert len(scenarios) == 2
        assert scenarios[0].label == "binance"
        assert scenarios[0].exchanges == ["binance"]
        assert scenarios[1].label == "bybit"
        assert scenarios[1].exchanges == ["bybit"]

    def test_tc1_7_mixed_inheritance_patterns(self):
        """TC1.7: Mixed inheritance patterns in same config."""
        cfg = make_test_config(
            exchanges=["binance", "bybit"],
            scenarios=[
                {"label": "combined"},  # Inherits
                {"label": "binance_only", "exchanges": ["binance"]},
                {"label": "bybit_only", "exchanges": ["bybit"]},
                {"label": "explicit_combined", "exchanges": ["binance", "bybit"]},
            ],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance", "bybit"])

        assert len(scenarios) == 4
        assert scenarios[0].exchanges == ["binance", "bybit"]  # inherited
        assert scenarios[1].exchanges == ["binance"]
        assert scenarios[2].exchanges == ["bybit"]
        assert scenarios[3].exchanges == ["binance", "bybit"]  # explicit


# =============================================================================
# Category 2: Exchange Variations
# =============================================================================


class TestExchangeVariations:
    """TC2.x: Exchange variation tests."""

    def test_tc2_1_base_single_exchange_scenarios_inherit(self):
        """TC2.1: Base has single exchange, scenarios inherit."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[{"label": "default"}],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert len(scenarios) == 1
        assert scenarios[0].exchanges == ["binance"]

    def test_tc2_2_scenario_overrides_with_fewer_exchanges(self):
        """TC2.2: Scenario uses subset of base exchanges."""
        cfg = make_test_config(
            exchanges=["binance", "bybit", "kucoin"],
            scenarios=[{"label": "two_only", "exchanges": ["binance", "bybit"]}],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(
            suite_cfg, base_exchanges=["binance", "bybit", "kucoin"]
        )

        assert len(scenarios) == 1
        assert scenarios[0].exchanges == ["binance", "bybit"]

    def test_tc2_3_scenario_overrides_with_different_exchange(self):
        """TC2.3: Scenario uses exchange not in base."""
        cfg = make_test_config(
            exchanges=["binance", "bybit"],
            scenarios=[{"label": "kucoin_only", "exchanges": ["kucoin"]}],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance", "bybit"])

        assert len(scenarios) == 1
        assert scenarios[0].exchanges == ["kucoin"]

    def test_tc2_4_multiple_scenarios_different_subsets(self):
        """TC2.4: Multiple scenarios each picking different subset."""
        cfg = make_test_config(
            exchanges=["binance", "bybit", "kucoin", "bitget"],
            scenarios=[
                {"label": "tier1", "exchanges": ["binance", "bybit"]},
                {"label": "tier2", "exchanges": ["kucoin", "bitget"]},
                {"label": "all", "exchanges": ["binance", "bybit", "kucoin", "bitget"]},
            ],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(
            suite_cfg, base_exchanges=["binance", "bybit", "kucoin", "bitget"]
        )

        assert len(scenarios) == 3
        assert scenarios[0].exchanges == ["binance", "bybit"]
        assert scenarios[1].exchanges == ["kucoin", "bitget"]
        assert scenarios[2].exchanges == ["binance", "bybit", "kucoin", "bitget"]


# =============================================================================
# Category 3: Coin Variations
# =============================================================================


class TestCoinVariations:
    """TC3.x: Coin variation tests."""

    def test_tc3_1_scenario_with_explicit_coin_list(self):
        """TC3.1: Scenario with explicit coin list."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[{"label": "btc_eth", "coins": ["BTC", "ETH"]}],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert scenarios[0].coins == ["BTC", "ETH"]

    def test_tc3_2_scenario_with_ignored_coins(self):
        """TC3.2: Scenario with ignored_coins."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[{"label": "no_memes", "ignored_coins": ["DOGE", "SHIB"]}],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert scenarios[0].ignored_coins == ["DOGE", "SHIB"]

    def test_tc3_3_scenarios_with_overlapping_coin_lists(self):
        """TC3.3: Scenarios with overlapping coin lists."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[
                {"label": "set_a", "coins": ["BTC", "ETH", "SOL"]},
                {"label": "set_b", "coins": ["ETH", "SOL", "ADA"]},
            ],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert set(scenarios[0].coins) == {"BTC", "ETH", "SOL"}
        assert set(scenarios[1].coins) == {"ETH", "SOL", "ADA"}

    def test_tc3_4_scenarios_with_disjoint_coin_lists(self):
        """TC3.4: Scenarios with disjoint coin lists."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[
                {"label": "majors", "coins": ["BTC", "ETH"]},
                {"label": "alts", "coins": ["SOL", "ADA"]},
            ],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert set(scenarios[0].coins) == {"BTC", "ETH"}
        assert set(scenarios[1].coins) == {"SOL", "ADA"}

    def test_tc3_5_scenario_with_coin_sources(self):
        """TC3.5: Scenario with per-coin exchange assignment."""
        cfg = make_test_config(
            exchanges=["binance", "bybit"],
            scenarios=[
                {
                    "label": "mixed_sources",
                    "coin_sources": {"BTC": "binance", "ETH": "bybit"},
                }
            ],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance", "bybit"])

        assert scenarios[0].coin_sources == {"BTC": "binance", "ETH": "bybit"}

    def test_tc3_6_conflicting_coin_sources_raises_error(self):
        """TC3.6: Conflicting coin_sources across scenarios raises error."""
        config = {"backtest": {"coin_sources": {}}}
        scenarios = [
            SuiteScenario(
                "a", None, None, None, None, coin_sources={"BTC": "binance"}
            ),
            SuiteScenario(
                "b", None, None, None, None, coin_sources={"BTC": "bybit"}
            ),  # Conflict!
        ]

        with pytest.raises(ValueError, match="forces"):
            collect_suite_coin_sources(config, scenarios)


# =============================================================================
# Category 4: Date/Time Variations
# =============================================================================


class TestDateVariations:
    """TC4.x: Date/time variation tests."""

    def test_tc4_1_scenario_with_custom_start_date(self):
        """TC4.1: Scenario with custom start_date."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[{"label": "recent", "start_date": "2023-01-01"}],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert scenarios[0].start_date == "2023-01-01"

    def test_tc4_2_scenario_with_custom_end_date(self):
        """TC4.2: Scenario with custom end_date."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[{"label": "early", "end_date": "2022-01-01"}],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert scenarios[0].end_date == "2022-01-01"

    def test_tc4_3_scenarios_with_different_date_windows(self):
        """TC4.3: Scenarios with different date windows."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[
                {"label": "2021", "start_date": "2021-01-01", "end_date": "2022-01-01"},
                {"label": "2022", "start_date": "2022-01-01", "end_date": "2023-01-01"},
                {"label": "2023", "start_date": "2023-01-01", "end_date": "2024-01-01"},
            ],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert len(scenarios) == 3
        assert scenarios[0].start_date == "2021-01-01"
        assert scenarios[0].end_date == "2022-01-01"
        assert scenarios[1].start_date == "2022-01-01"
        assert scenarios[1].end_date == "2023-01-01"
        assert scenarios[2].start_date == "2023-01-01"
        assert scenarios[2].end_date == "2024-01-01"


# =============================================================================
# Category 5: Override Variations
# =============================================================================


class TestOverrideVariations:
    """TC5.x: Scenario override tests."""

    def test_tc5_1_scenario_with_bot_parameter_overrides(self):
        """TC5.1: Scenario with bot parameter overrides."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[
                {
                    "label": "conservative",
                    "overrides": {
                        "bot.long.total_wallet_exposure_limit": 0.5,
                        "bot.short.total_wallet_exposure_limit": 0.5,
                    },
                }
            ],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert (
            scenarios[0].overrides["bot.long.total_wallet_exposure_limit"] == 0.5
        )
        assert (
            scenarios[0].overrides["bot.short.total_wallet_exposure_limit"] == 0.5
        )

    def test_tc5_2_scenario_disabling_position_side(self):
        """TC5.2: Scenario disabling a position side."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[
                {
                    "label": "long_only",
                    "overrides": {"bot.short.total_wallet_exposure_limit": 0},
                },
                {
                    "label": "short_only",
                    "overrides": {"bot.long.total_wallet_exposure_limit": 0},
                },
            ],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert scenarios[0].overrides["bot.short.total_wallet_exposure_limit"] == 0
        assert scenarios[1].overrides["bot.long.total_wallet_exposure_limit"] == 0

    def test_tc5_3_scenario_with_grid_trailing_ratio_overrides(self):
        """TC5.3: Scenario with grid/trailing ratio overrides."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[
                {
                    "label": "pure_grid",
                    "overrides": {
                        "bot.long.entry_trailing_grid_ratio": 0,
                        "bot.long.close_trailing_grid_ratio": 0,
                    },
                },
                {
                    "label": "pure_trailing",
                    "overrides": {
                        "bot.long.entry_trailing_grid_ratio": 1,
                        "bot.long.close_trailing_grid_ratio": 1,
                    },
                },
            ],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert scenarios[0].overrides["bot.long.entry_trailing_grid_ratio"] == 0
        assert scenarios[1].overrides["bot.long.entry_trailing_grid_ratio"] == 1

    def test_tc5_4_scenario_with_n_positions_override(self):
        """TC5.4: Scenario with n_positions override."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[
                {"label": "n3", "overrides": {"bot.long.n_positions": 3}},
                {"label": "n5", "overrides": {"bot.long.n_positions": 5}},
                {"label": "n10", "overrides": {"bot.long.n_positions": 10}},
            ],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert scenarios[0].overrides["bot.long.n_positions"] == 3
        assert scenarios[1].overrides["bot.long.n_positions"] == 5
        assert scenarios[2].overrides["bot.long.n_positions"] == 10


# =============================================================================
# Category 6: Aggregation Variations
# =============================================================================


class TestAggregationVariations:
    """TC6.x: Aggregation variation tests."""

    def test_tc6_1_default_aggregation_mean(self):
        """TC6.1: Default aggregation (mean)."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[{"label": "a"}, {"label": "b"}],
            aggregate={"default": "mean"},
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        _, aggregate_cfg = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert aggregate_cfg["default"] == "mean"

    def test_tc6_2_custom_aggregation_mode(self):
        """TC6.2: Custom aggregation mode."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[{"label": "a"}, {"label": "b"}],
            aggregate={"default": "min"},
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        _, aggregate_cfg = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert aggregate_cfg["default"] == "min"

    def test_tc6_3_per_metric_aggregation(self):
        """TC6.3: Per-metric aggregation."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[{"label": "a"}, {"label": "b"}],
            aggregate={"default": "mean", "mdg": "min", "sharpe_ratio": "median"},
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        _, aggregate_cfg = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert aggregate_cfg["default"] == "mean"
        assert aggregate_cfg["mdg"] == "min"
        assert aggregate_cfg["sharpe_ratio"] == "median"


# =============================================================================
# Category 7: Legacy Config Migration
# =============================================================================


class TestLegacyMigration:
    """TC7.x: Legacy config migration tests."""

    def test_tc7_1_old_config_with_suite_enabled(self):
        """TC7.1: Old config with suite.enabled=true."""
        cfg = get_template_config()
        cfg["_raw"] = deepcopy(cfg)
        cfg["backtest"]["suite"] = {
            "enabled": True,
            "scenarios": [{"label": "test"}],
            "aggregate": {"default": "mean"},
        }
        # Remove new-style keys
        cfg["backtest"].pop("scenarios", None)
        cfg["backtest"].pop("aggregate", None)

        formatted = format_config(deepcopy(cfg), verbose=False)

        assert "suite" not in formatted["backtest"]
        assert formatted["backtest"]["scenarios"] == [{"label": "test"}]
        assert formatted["backtest"]["aggregate"]["default"] == "mean"

    def test_tc7_2_old_config_with_include_base_scenario_true(self):
        """TC7.2: Old config with include_base_scenario=true."""
        cfg = get_template_config()
        cfg["_raw"] = deepcopy(cfg)
        cfg["backtest"]["suite"] = {
            "enabled": True,
            "include_base_scenario": True,
            "base_label": "my_base",
            "scenarios": [{"label": "custom"}],
        }
        cfg["backtest"].pop("scenarios", None)
        cfg["backtest"].pop("aggregate", None)

        formatted = format_config(deepcopy(cfg), verbose=False)

        assert "suite" not in formatted["backtest"]
        assert len(formatted["backtest"]["scenarios"]) == 2
        assert formatted["backtest"]["scenarios"][0]["label"] == "my_base"
        assert formatted["backtest"]["scenarios"][1]["label"] == "custom"

    def test_tc7_3_old_config_with_include_base_scenario_false(self):
        """TC7.3: Old config with include_base_scenario=false."""
        cfg = get_template_config()
        cfg["_raw"] = deepcopy(cfg)
        cfg["backtest"]["suite"] = {
            "enabled": True,
            "include_base_scenario": False,
            "scenarios": [{"label": "custom"}],
        }
        cfg["backtest"].pop("scenarios", None)
        cfg["backtest"].pop("aggregate", None)

        formatted = format_config(deepcopy(cfg), verbose=False)

        assert "suite" not in formatted["backtest"]
        assert len(formatted["backtest"]["scenarios"]) == 1
        assert formatted["backtest"]["scenarios"][0]["label"] == "custom"

    def test_tc7_6_old_config_with_include_base_scenario_true_and_empty_scenarios(self):
        """TC7.6: Old config with include_base_scenario=true but no explicit scenarios."""
        cfg = get_template_config()
        cfg["_raw"] = deepcopy(cfg)
        cfg["backtest"]["suite"] = {
            "enabled": True,
            "include_base_scenario": True,
            "base_label": "base_only",
            "scenarios": [],
        }
        cfg["backtest"].pop("scenarios", None)
        cfg["backtest"].pop("aggregate", None)

        formatted = format_config(deepcopy(cfg), verbose=False)

        assert "suite" not in formatted["backtest"]
        assert len(formatted["backtest"]["scenarios"]) == 1
        assert formatted["backtest"]["scenarios"][0]["label"] == "base_only"

    def test_tc7_4_old_config_with_combine_ohlcvs_true(self):
        """TC7.4: Old config with combine_ohlcvs=true."""
        cfg = get_template_config()
        cfg["_raw"] = deepcopy(cfg)
        cfg["backtest"]["combine_ohlcvs"] = True

        formatted = format_config(deepcopy(cfg), verbose=False)

        assert "combine_ohlcvs" not in formatted["backtest"]

    def test_tc7_5_old_config_with_combine_ohlcvs_false(self):
        """TC7.5: Old config with combine_ohlcvs=false."""
        cfg = get_template_config()
        cfg["_raw"] = deepcopy(cfg)
        cfg["backtest"]["combine_ohlcvs"] = False

        formatted = format_config(deepcopy(cfg), verbose=False)

        assert "combine_ohlcvs" not in formatted["backtest"]

    def test_tc7_6_full_legacy_migration(self):
        """TC7.6: Full legacy migration with all components."""
        cfg = get_template_config()
        cfg["_raw"] = deepcopy(cfg)
        cfg["backtest"]["exchanges"] = ["binance", "bybit"]
        cfg["backtest"]["combine_ohlcvs"] = True
        cfg["backtest"]["suite"] = {
            "enabled": True,
            "include_base_scenario": True,
            "base_label": "combined",
            "aggregate": {"default": "mean", "mdg": "min"},
            "scenarios": [
                {"label": "binance", "exchanges": ["binance"]},
                {"label": "bybit", "exchanges": ["bybit"]},
            ],
        }
        cfg["backtest"].pop("scenarios", None)
        cfg["backtest"].pop("aggregate", None)

        formatted = format_config(deepcopy(cfg), verbose=False)

        assert "combine_ohlcvs" not in formatted["backtest"]
        assert "suite" not in formatted["backtest"]
        assert len(formatted["backtest"]["scenarios"]) == 3
        assert formatted["backtest"]["scenarios"][0]["label"] == "combined"
        assert formatted["backtest"]["scenarios"][1]["label"] == "binance"
        assert formatted["backtest"]["scenarios"][2]["label"] == "bybit"
        assert formatted["backtest"]["aggregate"]["default"] == "mean"
        assert formatted["backtest"]["aggregate"]["mdg"] == "min"


# =============================================================================
# Category 8: Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """TC8.x: Edge case and error handling tests."""

    def test_tc8_1_empty_exchanges_with_scenario_exchanges(self):
        """TC8.1: Empty base exchanges but scenario provides exchanges."""
        cfg = make_test_config(
            exchanges=[],
            scenarios=[{"label": "binance", "exchanges": ["binance"]}],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=[])

        assert len(scenarios) == 1
        assert scenarios[0].exchanges == ["binance"]

    def test_tc8_4_large_number_of_scenarios(self):
        """TC8.4: Large number of scenarios (stress test)."""
        scenarios_list = [{"label": f"s{i}"} for i in range(50)]
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=scenarios_list,
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert len(scenarios) == 50
        for i, scenario in enumerate(scenarios):
            assert scenario.label == f"s{i}"

    def test_tc8_5_overlapping_date_ranges_allowed(self):
        """TC8.5: Overlapping but valid date ranges are allowed."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[
                {"label": "q1", "start_date": "2023-01-01", "end_date": "2023-04-01"},
                {"label": "q2", "start_date": "2023-03-01", "end_date": "2023-06-01"},
            ],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        # Should not raise
        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance"])

        assert len(scenarios) == 2
        assert scenarios[0].start_date == "2023-01-01"
        assert scenarios[0].end_date == "2023-04-01"
        assert scenarios[1].start_date == "2023-03-01"
        assert scenarios[1].end_date == "2023-06-01"


# =============================================================================
# Additional validation tests
# =============================================================================


class TestScenarioDataStrategy:
    """Tests for correct data strategy selection based on exchange configuration."""

    def test_single_exchange_scenario_uses_single_mode(self):
        """Single exchange scenario should use single exchange data (not combined)."""
        cfg = make_test_config(
            exchanges=["binance", "bybit"],
            scenarios=[{"label": "binance_only", "exchanges": ["binance"]}],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance", "bybit"])

        # Single exchange = single exchange mode
        assert len(scenarios[0].exchanges) == 1
        assert scenarios[0].exchanges == ["binance"]

    def test_multi_exchange_scenario_uses_combined_mode(self):
        """Multi exchange scenario should use combined data."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[{"label": "combined", "exchanges": ["binance", "bybit"]}],
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance"])

        # Multiple exchanges = combined mode
        assert len(scenarios[0].exchanges) == 2
        assert scenarios[0].exchanges == ["binance", "bybit"]

    def test_inherited_multi_exchange_uses_combined_mode(self):
        """Inherited multi-exchange config should use combined mode."""
        cfg = make_test_config(
            exchanges=["binance", "bybit"],
            scenarios=[{"label": "default"}],  # No exchanges = inherit
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        scenarios, _ = build_scenarios(suite_cfg, base_exchanges=["binance", "bybit"])

        # Inherited multiple exchanges = combined mode
        assert len(scenarios[0].exchanges) == 2
        assert scenarios[0].exchanges == ["binance", "bybit"]


# =============================================================================
# Suite Enabled Config Tests
# =============================================================================


class TestSuiteEnabledConfig:
    """Tests for suite_enabled config parameter."""

    def test_suite_enabled_true_with_scenarios(self):
        """suite_enabled=true with scenarios = suite mode enabled."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[{"label": "test"}],
            suite_enabled=True,
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        assert suite_cfg["enabled"] is True

    def test_suite_enabled_false_with_scenarios(self):
        """suite_enabled=false with scenarios = suite mode disabled."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[{"label": "test"}],
            suite_enabled=False,
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        assert suite_cfg["enabled"] is False

    def test_suite_enabled_true_without_scenarios(self):
        """suite_enabled=true without scenarios = suite mode disabled (no scenarios)."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[],
            suite_enabled=True,
        )
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        assert suite_cfg["enabled"] is False

    def test_suite_enabled_default_is_true(self):
        """suite_enabled defaults to true."""
        cfg = make_test_config(
            exchanges=["binance"],
            scenarios=[{"label": "test"}],
        )
        # Don't explicitly set suite_enabled - should default to true
        formatted = format_config(deepcopy(cfg), verbose=False)
        suite_cfg = extract_suite_config(formatted, None)

        assert suite_cfg["enabled"] is True


# =============================================================================
# Scenario Filtering Tests
# =============================================================================


class TestScenarioFiltering:
    """Tests for filter_scenarios_by_label function."""

    def test_filter_single_label(self):
        """Filter to single scenario by label."""
        scenarios = [
            {"label": "base"},
            {"label": "binance_only", "exchanges": ["binance"]},
            {"label": "bybit_only", "exchanges": ["bybit"]},
        ]
        filtered = filter_scenarios_by_label(scenarios, ["base"])

        assert len(filtered) == 1
        assert filtered[0]["label"] == "base"

    def test_filter_multiple_labels(self):
        """Filter to multiple scenarios by labels."""
        scenarios = [
            {"label": "base"},
            {"label": "binance_only", "exchanges": ["binance"]},
            {"label": "bybit_only", "exchanges": ["bybit"]},
        ]
        filtered = filter_scenarios_by_label(scenarios, ["base", "bybit_only"])

        assert len(filtered) == 2
        labels = [s["label"] for s in filtered]
        assert "base" in labels
        assert "bybit_only" in labels

    def test_filter_empty_labels_returns_all(self):
        """Empty filter list returns all scenarios."""
        scenarios = [
            {"label": "base"},
            {"label": "other"},
        ]
        filtered = filter_scenarios_by_label(scenarios, [])

        assert len(filtered) == 2

    def test_filter_no_match_raises_error(self):
        """No matching labels raises ValueError with available labels."""
        scenarios = [
            {"label": "base"},
            {"label": "other"},
        ]

        with pytest.raises(ValueError, match="No scenarios match"):
            filter_scenarios_by_label(scenarios, ["nonexistent"])

    def test_filter_preserves_scenario_data(self):
        """Filtering preserves all scenario data."""
        scenarios = [
            {
                "label": "custom",
                "exchanges": ["binance"],
                "coins": ["BTC", "ETH"],
                "overrides": {"bot.long.n_positions": 5},
            },
        ]
        filtered = filter_scenarios_by_label(scenarios, ["custom"])

        assert len(filtered) == 1
        assert filtered[0]["exchanges"] == ["binance"]
        assert filtered[0]["coins"] == ["BTC", "ETH"]
        assert filtered[0]["overrides"]["bot.long.n_positions"] == 5
