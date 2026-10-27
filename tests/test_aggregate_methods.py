"""Raw metric reduction, explicit reducer selection and stored-objective integrity."""

from copy import deepcopy
import math
import operator
from types import SimpleNamespace

import numpy as np
import pytest

from config.reducers import canonicalize_reducer_mapping
from config.scoring import normalize_scoring_entries
from suite_runner import SuiteScenario, ScenarioResult, reduce_metrics, apply_scenario
from pareto_store import _resolve_reducer_mode, _suite_metrics_to_stats, _evaluate_limits, LimitSpec
from pareto_core import extract_objectives
from metrics_schema import flatten_metric_stats, MetricAggregationError

CAGR = "cagr_fraction_strategy_equity"
HOLD = "position_held_max_days"
RECOVERY = "peak_recovery_max_days_strategy_equity"
PAIN = "pain_index_fraction_strategy_equity"


class TestResolveAggregateMode:
    def test_none_cfg_returns_mean(self):
        assert _resolve_reducer_mode(CAGR, None) == "mean"

    def test_empty_cfg_returns_mean(self):
        assert _resolve_reducer_mode(CAGR, {}) == "mean"

    def test_default_applies_when_metric_absent(self):
        assert _resolve_reducer_mode(PAIN, {"default": "median"}) == "median"

    def test_exact_match(self):
        assert _resolve_reducer_mode(HOLD, {"default": "mean", HOLD: "max"}) == "max"

    def test_distinct_quantities_keep_distinct_reducers(self):
        cfg = {"default": "mean", RECOVERY: "max", HOLD: "min"}
        assert _resolve_reducer_mode(RECOVERY, cfg) == "max"
        assert _resolve_reducer_mode(HOLD, cfg) == "min"

    @pytest.mark.parametrize("metric", ["position_held_max", "position_held_max_hours", HOLD + "_mean"])
    def test_authored_unit_prefix_and_stat_aliases_are_rejected(self, metric):
        with pytest.raises(ValueError, match="unknown or retired core metric"):
            canonicalize_reducer_mapping({"reducer": {metric: "min"}}, path="backtest")

    def test_default_mean_when_default_absent(self):
        assert _resolve_reducer_mode(CAGR, {RECOVERY: "max"}) == "mean"


class TestSuiteMetricsToStatsAggregate:
    def _entry_with_metrics_format(self, aggregated_value=None):
        payload = {
            "stats": {"mean": 100.0, "min": 50.0, "max": 200.0,
                      "std": float(np.std([50.0, 50.0, 100.0, 200.0])), "median": 75.0},
            "scenarios": {},
        }
        if aggregated_value is not None:
            payload["aggregated"] = aggregated_value
        return {"suite_metrics": {"definition_version": 1, "metrics": {HOLD: payload}}}

    def test_precomputed_aggregated_used_when_present(self):
        _, aggregate = _suite_metrics_to_stats(self._entry_with_metrics_format(200.0))
        assert aggregate[HOLD] == 200.0

    def test_fallback_uses_mean_without_cfg(self):
        _, aggregate = _suite_metrics_to_stats(self._entry_with_metrics_format())
        assert aggregate[HOLD] == 100.0

    @pytest.mark.parametrize("mode,expected", [("max", 200.0), ("min", 50.0)])
    def test_explicit_reducer_selects_stored_raw_stat(self, mode, expected):
        _, aggregate = _suite_metrics_to_stats(self._entry_with_metrics_format(),
            reducer_cfg={"default": "mean", HOLD: mode})
        assert aggregate[HOLD] == expected

    def test_precomputed_takes_precedence_over_cfg(self):
        entry = self._entry_with_metrics_format(999.0)
        _, aggregate = _suite_metrics_to_stats(entry, reducer_cfg={HOLD: "max"})
        assert aggregate[HOLD] == 999.0

    def test_stats_flat_always_available(self):
        flat, _ = _suite_metrics_to_stats(self._entry_with_metrics_format(), reducer_cfg={HOLD: "max"})
        assert flat[HOLD + "_mean"] == 100.0
        assert flat[HOLD + "_max"] == 200.0

    def test_unversioned_legacy_aggregate_is_rejected(self):
        entry = {"suite_metrics": {"aggregate": {"stats": {RECOVERY: {"mean": 300.0, "max": 500.0}}}}}
        with pytest.raises(MetricAggregationError, match="version missing or stale"):
            _suite_metrics_to_stats(entry, reducer_cfg={RECOVERY: "max"})


class TestAggregateMetricsConfig:
    def _make_results(self, metric_name, values):
        return [ScenarioResult(
            scenario=SuiteScenario(f"s{i}", None, None, None, None), per_exchange={},
            metrics={"definition_version": 1, "stats": {metric_name: {"mean": value}}},
            elapsed_seconds=0.0, output_path=None,
        ) for i, value in enumerate(values)]

    def test_default_mean(self):
        summary = reduce_metrics(self._make_results(CAGR, [1.0, 3.0]), {"default": "mean"})
        assert summary["aggregated"][CAGR] == pytest.approx(2.0)

    def test_explicit_max(self):
        summary = reduce_metrics(self._make_results(HOLD, [100.0, 300.0]), {HOLD: "max"})
        assert summary["aggregated"][HOLD] == pytest.approx(300.0)

    def test_explicit_min(self):
        summary = reduce_metrics(self._make_results(HOLD, [100.0, 300.0]), {HOLD: "min"})
        assert summary["aggregated"][HOLD] == pytest.approx(100.0)

    def test_authored_stat_suffix_is_rejected(self):
        with pytest.raises(ValueError, match="unknown or retired core metric"):
            normalize_scoring_entries([{"metric": HOLD + "_max", "goal": "min"}])
        spec, _ = normalize_scoring_entries([{"metric": HOLD, "goal": "min", "reducer": "max"}])
        assert spec[0].metric == HOLD and spec[0].reducer == "max"


class TestApplyScenarioBaseCoins:
    """Scenarios without explicit coins should fall back to base_coins,
    not master_coins."""

    BASE_CONFIG = {
        "backtest": {
            "start_date": "2021-01-01",
            "end_date": "2021-01-31",
            "coins": {},
            "cache_dir": {},
            "exchanges": ["binance"],
        },
        "live": {},
        "bot": {
            "long": {"universe": {"approved_coins": [], "ignored_coins": []}},
            "short": {"universe": {"approved_coins": [], "ignored_coins": []}},
        },
    }

    def test_no_coins_uses_base_coins(self):
        scenario = SuiteScenario("base", None, None, coins=None, ignored_coins=None)
        base_coins = ["BTC", "ETH"]
        master_coins = ["BTC", "ETH", "DOGE", "SHIB", "XRP"]
        _, coins = apply_scenario(
            deepcopy(self.BASE_CONFIG),
            scenario,
            master_coins=master_coins,
            master_ignored=[],
            available_exchanges=["binance"],
            available_coins={"BTC", "ETH", "DOGE", "SHIB", "XRP"},
            base_coins=base_coins,
        )
        assert coins == ["BTC", "ETH"]

    def test_no_coins_falls_back_to_master_when_base_not_provided(self):
        scenario = SuiteScenario("base", None, None, coins=None, ignored_coins=None)
        master_coins = ["BTC", "ETH", "DOGE"]
        _, coins = apply_scenario(
            deepcopy(self.BASE_CONFIG),
            scenario,
            master_coins=master_coins,
            master_ignored=[],
            available_exchanges=["binance"],
            available_coins={"BTC", "ETH", "DOGE"},
        )
        assert coins == ["BTC", "DOGE", "ETH"]

    def test_explicit_coins_ignores_base_coins(self):
        scenario = SuiteScenario("subset", None, None, coins=["DOGE", "SHIB"], ignored_coins=None)
        base_coins = ["BTC", "ETH"]
        master_coins = ["BTC", "ETH", "DOGE", "SHIB"]
        _, coins = apply_scenario(
            deepcopy(self.BASE_CONFIG),
            scenario,
            master_coins=master_coins,
            master_ignored=[],
            available_exchanges=["binance"],
            available_coins={"BTC", "ETH", "DOGE", "SHIB"},
            base_coins=base_coins,
        )
        assert coins == ["DOGE", "SHIB"]

    def test_base_ignored_used_for_no_ignored(self):
        scenario = SuiteScenario("base", None, None, coins=["BTC"], ignored_coins=None)
        base_ignored = ["XRP"]
        master_ignored = ["XRP", "DOGE", "SHIB"]
        base_config = deepcopy(self.BASE_CONFIG)
        for pside in ("long", "short"):
            base_config["bot"][pside]["universe"]["ignored_coins"] = list(base_ignored)
        cfg, _ = apply_scenario(
            base_config,
            scenario,
            master_coins=["BTC"],
            master_ignored=master_ignored,
            available_exchanges=["binance"],
            available_coins={"BTC", "XRP", "DOGE", "SHIB"},
            base_ignored=base_ignored,
        )
        ignored_long = cfg['bot']["long"]['universe']['ignored_coins']
        assert "XRP" in ignored_long
        assert "DOGE" not in ignored_long

    @pytest.mark.parametrize("ignored", [
        {"long": [], "short": []},
        {"long": ["DOGE"], "short": ["DOGE"]},
    ])
    def test_base_ignored_helper_does_not_replace_explicit_side_sources(self, ignored):
        base_config = deepcopy(self.BASE_CONFIG)
        for pside, coins in ignored.items():
            base_config["bot"][pside]["universe"]["ignored_coins"] = list(coins)
        cfg, _ = apply_scenario(
            base_config,
            SuiteScenario("base", None, None, coins=["BTC"], ignored_coins=None),
            master_coins=["BTC"],
            master_ignored=["XRP", "DOGE", "SHIB"],
            available_exchanges=["binance"],
            available_coins={"BTC", "XRP", "DOGE", "SHIB"},
            base_ignored=["XRP"],
        )
        assert {pside: cfg["bot"][pside]["universe"]["ignored_coins"]
                for pside in ("long", "short")} == ignored

    def test_base_ignored_helper_cannot_repair_unsupported_asymmetric_sources(self):
        base_config = deepcopy(self.BASE_CONFIG)
        base_config["bot"]["long"]["universe"]["ignored_coins"] = ["XRP"]
        base_config["bot"]["short"]["universe"]["ignored_coins"] = ["DOGE"]
        before = deepcopy(base_config)
        with pytest.raises(ValueError, match="asymmetric.*ignored_coins"):
            apply_scenario(
                base_config,
                SuiteScenario("base", None, None, coins=["BTC"], ignored_coins=None),
                master_coins=["BTC"],
                master_ignored=["XRP", "DOGE", "SHIB"],
                available_exchanges=["binance"],
                available_coins={"BTC", "XRP", "DOGE", "SHIB"},
                base_ignored=["XRP"],
            )
        assert base_config == before


class TestCalcFitnessAggregateSelection:
    def test_explicit_reducer_changes_selected_raw_objective_without_overwriting_mean(self):
        from optimize import Evaluator
        from config.scoring import extract_objective_specs

        specs = extract_objective_specs([
            {"metric": HOLD, "goal": "min", "reducer": "max"},
            {"metric": CAGR, "goal": "max", "reducer": "mean"},
        ])
        flat = flatten_metric_stats({
            HOLD: {"mean": 150.0, "max": 300.0}, CAGR: {"mean": 0.001},
        })
        before = deepcopy(flat)
        evaluator = SimpleNamespace(scoring_specs=specs, limit_checks=[])
        fitness, penalty, raw = Evaluator.calc_fitness(evaluator, flat, return_raw_objectives=True)
        assert raw == {HOLD: 300.0, CAGR: 0.001}
        assert fitness == pytest.approx((2 / math.pi * math.atan(300.0), -2 / math.pi * math.atan(0.001)))
        assert penalty == 0.0
        assert flat == before


class TestObjectivesNotDoubleCorrected:
    AGGREGATE_CFG = {"default": "mean", HOLD: "max", RECOVERY: "max"}

    def _make_pareto_entry(self, scoring_keys, objective_values, mean_values, max_values):
        rows = {name: {"stats": {"mean": mean_values[name], "max": max_values[name]},
                       "aggregated": max_values[name] if self.AGGREGATE_CFG.get(name) == "max" else mean_values[name],
                       "scenarios": {}} for name in scoring_keys}
        return {
            "backtest": {"reducer": self.AGGREGATE_CFG},
            "optimize": {"scoring": [{"metric": name, "goal": "max" if name == CAGR else "min"} for name in scoring_keys]},
            "metrics": {"definition_version": 1, "objectives": dict(zip(scoring_keys, objective_values)), "constraint_violation": 0.0},
            "suite_metrics": {"definition_version": 1, "metrics": rows},
        }

    def test_named_raw_objectives_are_not_double_corrected(self):
        entry = self._make_pareto_entry([CAGR, HOLD], [0.001, 300.0],
            {CAGR: 0.001, HOLD: 150.0}, {CAGR: 0.001, HOLD: 300.0})
        before = deepcopy(entry)
        values, keys = extract_objectives(entry)
        assert keys == [CAGR, HOLD]
        assert values == pytest.approx((0.001, 300.0))
        _suite_metrics_to_stats(entry, reducer_cfg=self.AGGREGATE_CFG)
        assert extract_objectives(entry)[0] == values
        assert entry == before

    def test_limit_filtering_uses_raw_aggregate(self):
        entry = self._make_pareto_entry([HOLD], [300.0], {HOLD: 150.0}, {HOLD: 300.0})
        flat, aggregate = _suite_metrics_to_stats(entry, reducer_cfg=self.AGGREGATE_CFG)
        assert aggregate[HOLD] == 300.0
        limit = LimitSpec(metric=HOLD, field="auto", op=operator.le, value=200.0)
        assert not _evaluate_limits([limit], flat, aggregate, entry["metrics"]["objectives"], {})
        limit = LimitSpec(metric=HOLD, field="auto", op=operator.le, value=300.0)
        assert _evaluate_limits([limit], flat, aggregate, entry["metrics"]["objectives"], {})
