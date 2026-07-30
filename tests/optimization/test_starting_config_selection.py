from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from optimization.starting_config_selection import select_starting_config_artifacts


def _metric_stats(*, mean: float, max_value: float | None = None) -> dict:
    maximum = mean if max_value is None else max_value
    return {
        "mean": mean,
        "min": mean,
        "max": maximum,
        "std": 0.0,
        "median": mean,
    }


def _write_candidate(
    pareto_dir: Path,
    name: str,
    *,
    adg: float,
    drawdown_mean: float,
    drawdown_max: float | None = None,
) -> None:
    payload = {
        "config_version": "v8.0.0",
        "backtest": {"aggregate": {"default": "max"}},
        "bot": {"long": {"risk": {"total_wallet_exposure_limit": 1.0 + adg}}},
        "optimize": {
            "scoring": [
                {"metric": "adg_strategy_eq", "goal": "max"},
                {"metric": "drawdown_worst_strategy_eq", "goal": "min"},
            ]
        },
        "metrics": {
            "objectives": {
                "adg_strategy_eq": adg,
                "drawdown_worst_strategy_eq": drawdown_mean,
            },
            "stats": {
                "adg_strategy_eq": _metric_stats(mean=adg),
                "drawdown_worst_strategy_eq": _metric_stats(
                    mean=drawdown_mean,
                    max_value=drawdown_max,
                ),
            },
        },
    }
    (pareto_dir / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture()
def sample_pareto_dir(tmp_path: Path) -> Path:
    pareto_dir = tmp_path / "run" / "pareto"
    pareto_dir.mkdir(parents=True)
    _write_candidate(
        pareto_dir,
        "low_drawdown_anchor",
        adg=0.001,
        drawdown_mean=0.10,
        drawdown_max=0.20,
    )
    _write_candidate(
        pareto_dir,
        "adg_anchor",
        adg=0.004,
        drawdown_mean=0.40,
        drawdown_max=0.80,
    )
    _write_candidate(
        pareto_dir,
        "middle",
        adg=0.002,
        drawdown_mean=0.25,
        drawdown_max=0.60,
    )
    _write_candidate(
        pareto_dir,
        "diverse_fill",
        adg=0.003,
        drawdown_mean=0.15,
        drawdown_max=0.30,
    )
    return pareto_dir


def test_filters_with_effective_aggregate_then_compresses_with_anchors_farthest(
    sample_pareto_dir: Path,
):
    result = select_starting_config_artifacts(
        str(sample_pareto_dir),
        limits=[
            {
                "metric": "drawdown_worst_strategy_eq",
                "penalize_if": "greater_than",
                "value": 0.30,
            }
        ],
        aggregate_cfg={"default": "mean"},
        filter_by_limits=True,
        max_count=2,
    )

    assert result.loaded_count == 4
    assert result.filtered_count == 3
    assert result.selected_count == 2
    assert [candidate.path.name for candidate in result.candidates] == [
        "diverse_fill.json",
        "low_drawdown_anchor.json",
    ]


def test_compression_without_filtering_uses_all_metric_bearing_candidates(
    sample_pareto_dir: Path,
):
    result = select_starting_config_artifacts(
        str(sample_pareto_dir),
        limits=[],
        aggregate_cfg={"default": "mean"},
        filter_by_limits=False,
        max_count=2,
    )

    assert result.filtered_count == 4
    assert result.selected_count == 2
    assert [candidate.path.name for candidate in result.candidates] == [
        "diverse_fill.json",
        "adg_anchor.json",
    ]


def test_metric_preselection_accepts_one_pareto_artifact_and_ignores_siblings(
    sample_pareto_dir: Path,
):
    (sample_pareto_dir / "ordinary_config.json").write_text(
        json.dumps({"bot": {"long": {}, "short": {}}}),
        encoding="utf-8",
    )

    result = select_starting_config_artifacts(
        str(sample_pareto_dir / "middle.json"),
        limits=[],
        aggregate_cfg={"default": "mean"},
        filter_by_limits=False,
        max_count=None,
    )

    assert result.loaded_count == 1
    assert [candidate.path.name for candidate in result.candidates] == ["middle.json"]


def test_filtering_resolves_auto_limit_direction_like_optimizer(
    sample_pareto_dir: Path,
):
    result = select_starting_config_artifacts(
        str(sample_pareto_dir),
        limits=[
            {
                "metric": "adg_strategy_eq",
                "penalize_if": "auto",
                "value": 0.0015,
            }
        ],
        aggregate_cfg={"default": "mean"},
        filter_by_limits=True,
        max_count=None,
    )

    assert result.filtered_count == 3
    assert "low_drawdown_anchor.json" not in {
        candidate.path.name for candidate in result.candidates
    }


def test_filtering_rejects_unknown_auto_limit_metric(
    sample_pareto_dir: Path,
):
    with pytest.raises(
        ValueError,
        match="unknown optimizer limit metric 'adg_strategy_eqq'",
    ):
        select_starting_config_artifacts(
            str(sample_pareto_dir),
            limits=[
                {
                    "metric": "adg_strategy_eqq",
                    "penalize_if": "auto",
                    "value": 0.0015,
                }
            ],
            aggregate_cfg={"default": "mean"},
            filter_by_limits=True,
            max_count=None,
        )


def test_inside_range_filter_preserves_boundary_candidates(
    sample_pareto_dir: Path,
):
    result = select_starting_config_artifacts(
        str(sample_pareto_dir),
        limits=[
            {
                "metric": "drawdown_worst_strategy_eq",
                "penalize_if": "inside_range",
                "range": [0.10, 0.25],
            }
        ],
        aggregate_cfg={"default": "mean"},
        filter_by_limits=True,
        max_count=None,
    )

    assert {candidate.path.name for candidate in result.candidates} == {
        "low_drawdown_anchor.json",
        "middle.json",
        "adg_anchor.json",
    }


def test_filtering_normalizes_descending_range_bounds(
    sample_pareto_dir: Path,
):
    result = select_starting_config_artifacts(
        str(sample_pareto_dir),
        limits=[
            {
                "metric": "drawdown_worst_strategy_eq",
                "penalize_if": "outside_range",
                "range": [0.30, 0.10],
            }
        ],
        aggregate_cfg={"default": "mean"},
        filter_by_limits=True,
        max_count=None,
    )

    assert {candidate.path.name for candidate in result.candidates} == {
        "low_drawdown_anchor.json",
        "middle.json",
        "diverse_fill.json",
    }


def test_filtering_recomputes_suite_aggregate_with_current_optimizer_default(
    tmp_path: Path,
):
    pareto_dir = tmp_path / "run" / "pareto"
    pareto_dir.mkdir(parents=True)
    _write_candidate(
        pareto_dir,
        "suite_candidate",
        adg=0.002,
        drawdown_mean=0.20,
        drawdown_max=0.80,
    )
    artifact_path = pareto_dir / "suite_candidate.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["suite_metrics"] = {
        "metrics": {
            "adg_strategy_eq": {
                "aggregated": 0.002,
                "stats": _metric_stats(mean=0.002),
            },
            "drawdown_worst_strategy_eq": {
                "aggregated": 0.80,
                "stats": _metric_stats(mean=0.20, max_value=0.80),
            },
        }
    }
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    result = select_starting_config_artifacts(
        str(pareto_dir),
        limits=[
            {
                "metric": "drawdown_worst_strategy_eq",
                "penalize_if": "greater_than",
                "value": 0.50,
            }
        ],
        aggregate_cfg={"default": "mean"},
        filter_by_limits=True,
        max_count=None,
    )

    assert result.selected_count == 1


def test_filtering_recomputes_aggregate_from_active_scenarios(tmp_path: Path):
    pareto_dir = tmp_path / "run" / "pareto"
    pareto_dir.mkdir(parents=True)
    _write_candidate(
        pareto_dir,
        "suite_candidate",
        adg=0.002,
        drawdown_mean=0.50,
        drawdown_max=0.80,
    )
    artifact_path = pareto_dir / "suite_candidate.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["suite_metrics"] = {
        "scenario_labels": ["base", "stress"],
        "metrics": {
            "adg_strategy_eq": {
                "aggregated": 0.002,
                "stats": _metric_stats(mean=0.002),
                "scenarios": {"base": 0.002, "stress": 0.002},
            },
            "drawdown_worst_strategy_eq": {
                "aggregated": 0.80,
                "stats": _metric_stats(mean=0.50, max_value=0.80),
                "scenarios": {"base": 0.20, "stress": 0.80},
            },
        },
    }
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    result = select_starting_config_artifacts(
        str(pareto_dir),
        limits=[
            {
                "metric": "drawdown_worst_strategy_eq",
                "penalize_if": "greater_than",
                "value": 0.50,
            }
        ],
        aggregate_cfg={"default": "max"},
        scenario_labels=["base"],
        filter_by_limits=True,
        max_count=None,
    )

    assert result.selected_count == 1


def test_filtering_aggregates_metric_from_available_active_scenarios(
    tmp_path: Path,
):
    pareto_dir = tmp_path / "run" / "pareto"
    pareto_dir.mkdir(parents=True)
    _write_candidate(
        pareto_dir,
        "suite_candidate",
        adg=0.002,
        drawdown_mean=0.20,
        drawdown_max=0.20,
    )
    artifact_path = pareto_dir / "suite_candidate.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["suite_metrics"] = {
        "scenario_labels": ["base", "stress"],
        "metrics": {
            "adg_strategy_eq": {
                "aggregated": 0.002,
                "stats": _metric_stats(mean=0.002),
                "scenarios": {"base": 0.002, "stress": 0.002},
            },
            "drawdown_worst_strategy_eq": {
                "aggregated": 0.20,
                "stats": _metric_stats(mean=0.20),
                "scenarios": {"base": 0.20},
            },
        },
    }
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    result = select_starting_config_artifacts(
        str(pareto_dir),
        limits=[
            {
                "metric": "drawdown_worst_strategy_eq",
                "penalize_if": "greater_than",
                "value": 0.50,
            }
        ],
        aggregate_cfg={"default": "max"},
        scenario_labels=["base", "stress"],
        filter_by_limits=True,
        max_count=None,
    )

    assert result.selected_count == 1


def test_filtering_rejects_artifact_missing_active_scenario_values(tmp_path: Path):
    pareto_dir = tmp_path / "run" / "pareto"
    pareto_dir.mkdir(parents=True)
    _write_candidate(
        pareto_dir,
        "suite_candidate",
        adg=0.002,
        drawdown_mean=0.20,
        drawdown_max=0.20,
    )
    artifact_path = pareto_dir / "suite_candidate.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["suite_metrics"] = {
        "scenario_labels": ["base"],
        "metrics": {
            "adg_strategy_eq": {
                "aggregated": 0.002,
                "stats": _metric_stats(mean=0.002),
                "scenarios": {"base": 0.002},
            },
            "drawdown_worst_strategy_eq": {
                "aggregated": 0.20,
                "stats": _metric_stats(mean=0.20),
                "scenarios": {"base": 0.20},
            },
        },
    }
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Limit metric 'drawdown_worst_strategy_eq' could not be resolved",
    ):
        select_starting_config_artifacts(
            str(pareto_dir),
            limits=[
                {
                    "metric": "drawdown_worst_strategy_eq",
                    "penalize_if": "greater_than",
                    "value": 0.50,
                }
            ],
            aggregate_cfg={"default": "max"},
            scenario_labels=["stress"],
            filter_by_limits=True,
            max_count=None,
        )


def test_filtering_rejects_suite_artifact_missing_effective_aggregate_stat(
    tmp_path: Path,
):
    pareto_dir = tmp_path / "run" / "pareto"
    pareto_dir.mkdir(parents=True)
    _write_candidate(
        pareto_dir,
        "incomplete_suite_candidate",
        adg=0.002,
        drawdown_mean=0.20,
        drawdown_max=0.20,
    )
    artifact_path = pareto_dir / "incomplete_suite_candidate.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["suite_metrics"] = {
        "aggregate": {
            "aggregated": {
                "adg_strategy_eq": 0.002,
                "drawdown_worst_strategy_eq": 0.20,
            }
        }
    }
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="Limit metric 'drawdown_worst_strategy_eq' could not be resolved",
    ):
        select_starting_config_artifacts(
            str(pareto_dir),
            limits=[
                {
                    "metric": "drawdown_worst_strategy_eq",
                    "penalize_if": "greater_than",
                    "value": 0.50,
                }
            ],
            aggregate_cfg={"default": "max"},
            filter_by_limits=True,
            max_count=None,
        )


def test_metric_preselection_warns_that_artifacts_are_not_verified(
    sample_pareto_dir: Path,
    caplog,
):
    caplog.set_level(logging.WARNING)

    select_starting_config_artifacts(
        str(sample_pareto_dir),
        limits=[],
        aggregate_cfg={"default": "mean"},
        filter_by_limits=False,
        max_count=2,
    )

    assert "trusts stored Pareto metrics and objectives" in caplog.text
    assert (
        "coins, exchanges, date range, scenarios, or backtest settings" in caplog.text
    )


def test_metric_preselection_rejects_ordinary_seed_configs(sample_pareto_dir: Path):
    (sample_pareto_dir / "ordinary_config.json").write_text(
        json.dumps({"bot": {"long": {}, "short": {}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="metric-bearing Pareto artifact"):
        select_starting_config_artifacts(
            str(sample_pareto_dir),
            limits=[],
            aggregate_cfg={"default": "mean"},
            filter_by_limits=False,
            max_count=2,
        )


def test_metric_preselection_rejects_directory_with_only_ordinary_configs(
    tmp_path: Path,
):
    seeds_dir = tmp_path / "seeds"
    seeds_dir.mkdir()
    (seeds_dir / "ordinary_config.json").write_text(
        json.dumps({"bot": {"long": {}, "short": {}}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="could not load complete metric-bearing Pareto artifacts"):
        select_starting_config_artifacts(
            str(seeds_dir),
            limits=[],
            aggregate_cfg={"default": "mean"},
            filter_by_limits=False,
            max_count=2,
        )


def test_metric_preselection_allows_pareto_compress_manifest(sample_pareto_dir: Path):
    (sample_pareto_dir / "selection.json").write_text(
        json.dumps({"selected_count": 4}),
        encoding="utf-8",
    )

    result = select_starting_config_artifacts(
        str(sample_pareto_dir),
        limits=[],
        aggregate_cfg={"default": "mean"},
        filter_by_limits=False,
        max_count=2,
    )

    assert result.loaded_count == 4


def test_metric_preselection_fails_when_all_candidates_are_filtered(
    sample_pareto_dir: Path,
):
    with pytest.raises(ValueError, match="No starting configs remained"):
        select_starting_config_artifacts(
            str(sample_pareto_dir),
            limits=[
                {
                    "metric": "adg_strategy_eq",
                    "penalize_if": "less_than_or_equal",
                    "value": 1.0,
                }
            ],
            aggregate_cfg={"default": "mean"},
            filter_by_limits=True,
            max_count=None,
        )


def test_metric_preselection_fails_when_limit_metric_is_missing(
    sample_pareto_dir: Path,
):
    with pytest.raises(
        ValueError, match="Limit metric 'missing_metric' could not be resolved"
    ):
        select_starting_config_artifacts(
            str(sample_pareto_dir),
            limits=[
                {
                    "metric": "missing_metric",
                    "penalize_if": "greater_than",
                    "value": 1.0,
                }
            ],
            aggregate_cfg={"default": "mean"},
            filter_by_limits=True,
            max_count=None,
        )
