"""Retired formulas cannot re-enter authored objectives through an explicit goal or alias."""

import pytest

from config.limits import normalize_limit_entries
from config.reducers import reducer_mapping_from_aliases
from config.scoring import ObjectiveSpec, normalize_scoring_entries


@pytest.mark.parametrize("metric", [
    "adg_w", "adg_w_usd_mean", "adg_w_per_exposure_long_usd",
    "adg_strategy_pnl_rebased_w", "volume_pct_per_day_avg_w",
])
@pytest.mark.parametrize("surface", ["string", "object", "spec", "limit", "reducer"])
def test_retired_formula_rejected_at_every_authored_metric_boundary(metric, surface):
    with pytest.raises(ValueError, match="unknown or retired"):
        if surface == "string":
            normalize_scoring_entries([metric], unknown_goal="min")
        elif surface == "object":
            normalize_scoring_entries([{"metric": metric, "goal": "max"}])
        elif surface == "spec":
            normalize_scoring_entries([ObjectiveSpec(metric, "max")])
        elif surface == "limit":
            normalize_limit_entries([{"metric": metric, "penalize_if": "less_than", "value": 0}])
        else:
            reducer_mapping_from_aliases({"reducer": {metric: "mean"}}, path="backtest")


@pytest.mark.parametrize("metric", [
    "cagr_fraction_strategy_equity_ew", "position_held_time_weighted_mean_hours",
])
def test_explicit_cagr_ew_and_retained_duration_metric_are_supported(metric):
    specs, _ = normalize_scoring_entries([{"metric": metric, "goal": "max"}])
    assert specs[0].metric == metric
