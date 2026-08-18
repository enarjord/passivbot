from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest
import pareto_explorer

from pareto_explorer import (
    build_scenario_front,
    build_parser,
    filter_candidates,
    load_candidates,
    project_candidates_to_scenario,
    run_from_args,
    select_candidate,
)


def _metric_stats(value: float) -> dict:
    return {"mean": value, "min": value, "max": value, "std": 0.0, "median": value}


def _write_candidate(
    path: Path,
    name: str,
    objectives: dict[str, float],
    *,
    extra_stats: dict[str, float] | None = None,
) -> None:
    stats = {
        "metric_a": _metric_stats(objectives["metric_a"]),
        "metric_b": _metric_stats(objectives["metric_b"]),
        "metric_c": _metric_stats(objectives["metric_c"]),
    }
    for metric, value in (extra_stats or {}).items():
        stats[metric] = _metric_stats(value)
    payload = {
        "optimize": {
            "scoring": [
                {"metric": "metric_a", "goal": "max"},
                {"metric": "metric_b", "goal": "max"},
                {"metric": "metric_c", "goal": "max"},
            ]
        },
        "metrics": {
            "objectives": objectives,
            "stats": stats,
        },
    }
    with open(path / f"{name}.json", "w") as f:
        json.dump(payload, f, indent=2)


def _write_suite_candidate(
    path: Path,
    name: str,
    *,
    adg_mean: float,
    adg_min: float,
    recovery_mean: float,
    recovery_max: float,
    drawdown_mean: float,
    drawdown_max: float,
) -> None:
    payload = {
        "optimize": {
            "scoring": [
                {"metric": "adg_strategy_pnl_rebased", "goal": "max"},
                {"metric": "peak_recovery_hours_hsl", "goal": "min"},
                {"metric": "drawdown_worst_hsl", "goal": "min"},
            ]
        },
        "backtest": {
            "aggregate": {
                "default": "mean",
                "peak_recovery_hours_hsl": "mean",
                "drawdown_worst_hsl": "max",
            }
        },
        "suite_metrics": {
            "aggregate": {
                "stats": {
                    "adg_strategy_pnl_rebased": {
                        "mean": adg_mean,
                        "min": adg_min,
                        "max": max(adg_mean, adg_min),
                        "std": 0.0,
                        "median": adg_mean,
                    },
                    "peak_recovery_hours_hsl": {
                        "mean": recovery_mean,
                        "min": recovery_mean,
                        "max": recovery_max,
                        "std": 0.0,
                        "median": recovery_mean,
                    },
                    "drawdown_worst_hsl": {
                        "mean": drawdown_mean,
                        "min": drawdown_mean,
                        "max": drawdown_max,
                        "std": 0.0,
                        "median": drawdown_mean,
                    },
                },
                "aggregated": {
                    "adg_strategy_pnl_rebased": adg_mean,
                    "peak_recovery_hours_hsl": recovery_mean,
                    "drawdown_worst_hsl": drawdown_max,
                },
            }
        },
    }
    with open(path / f"{name}.json", "w") as f:
        json.dump(payload, f, indent=2)


def _write_fill_suite_candidate(
    path: Path,
    name: str,
    *,
    adg: float,
    p95_gap: float,
    p99_gap: float,
) -> None:
    payload = {
        "optimize": {
            "scoring": [
                {"metric": "adg_strategy_eq", "goal": "max"},
                {"metric": "fills_gap_p99_hours", "goal": "min"},
            ]
        },
        "suite_metrics": {
            "metrics": {
                "adg_strategy_eq": {
                    "stats": _metric_stats(adg),
                    "aggregated": adg,
                    "scenarios": {},
                },
                "fills_gap_p95_hours": {
                    "stats": _metric_stats(p95_gap),
                    "aggregated": p95_gap,
                    "scenarios": {},
                },
                "fills_gap_p99_hours": {
                    "stats": _metric_stats(p99_gap),
                    "aggregated": p99_gap,
                    "scenarios": {},
                },
            }
        },
    }
    with open(path / f"{name}.json", "w") as f:
        json.dump(payload, f, indent=2)


def _write_scenario_candidate(
    path: Path,
    name: str,
    *,
    aggregate: dict[str, float],
    scenarios: dict[str, dict[str, float]],
) -> None:
    scoring = [
        {"metric": "metric_a", "goal": "max"},
        {"metric": "metric_b", "goal": "min"},
    ]
    metric_names = set(aggregate)
    for values in scenarios.values():
        metric_names.update(values)
    suite_metrics = {}
    for metric in sorted(metric_names):
        scenario_values = {
            label: values[metric]
            for label, values in scenarios.items()
            if metric in values
        }
        suite_metrics[metric] = {
            "stats": _metric_stats(aggregate[metric]),
            "aggregated": aggregate[metric],
            "scenarios": scenario_values,
        }
    payload = {
        "optimize": {"scoring": scoring},
        "metrics": {"objectives": aggregate},
        "suite_metrics": {
            "metrics": suite_metrics,
            "scenario_labels": list(scenarios),
        },
    }
    with open(path / f"{name}.json", "w") as f:
        json.dump(payload, f, indent=2)


@pytest.fixture()
def scenario_pareto_dir(tmp_path: Path) -> Path:
    pareto_dir = tmp_path / "suite_run" / "pareto"
    pareto_dir.mkdir(parents=True)
    values = {
        "a": {
            "bull": {"metric_a": 0.9, "metric_b": 0.4, "sharpe_ratio_strategy_eq": 1.2},
            "bear": {"metric_a": 0.2, "metric_b": 0.8, "sharpe_ratio_strategy_eq": 0.2},
        },
        "b": {
            "bull": {"metric_a": 0.8, "metric_b": 0.3, "sharpe_ratio_strategy_eq": 1.0},
            "bear": {"metric_a": 0.8, "metric_b": 0.5, "sharpe_ratio_strategy_eq": 1.5},
        },
        "c_dominated": {
            "bull": {"metric_a": 0.7, "metric_b": 0.5, "sharpe_ratio_strategy_eq": 0.8},
            "bear": {"metric_a": 0.6, "metric_b": 0.7, "sharpe_ratio_strategy_eq": 0.7},
        },
        "d": {
            "bull": {"metric_a": 0.5, "metric_b": 0.1, "sharpe_ratio_strategy_eq": 0.5},
            "bear": {"metric_a": 0.9, "metric_b": 0.9, "sharpe_ratio_strategy_eq": 1.1},
        },
    }
    for name, scenario_values in values.items():
        _write_scenario_candidate(
            pareto_dir,
            name,
            aggregate={
                "metric_a": 0.5,
                "metric_b": 0.5,
                "sharpe_ratio_strategy_eq": 0.5,
            },
            scenarios={
                "bull": scenario_values["bull"],
                "bear": scenario_values["bear"],
            },
        )
    return pareto_dir


@pytest.fixture()
def sample_pareto_dir(tmp_path: Path) -> Path:
    pareto_dir = tmp_path / "run" / "pareto"
    pareto_dir.mkdir(parents=True)
    _write_candidate(
        pareto_dir,
        "a_extreme",
        {"metric_a": 1.0, "metric_b": 0.0, "metric_c": 0.0},
    )
    _write_candidate(
        pareto_dir,
        "b_extreme",
        {"metric_a": 0.0, "metric_b": 1.0, "metric_c": 0.0},
    )
    _write_candidate(
        pareto_dir,
        "c_extreme",
        {"metric_a": 0.0, "metric_b": 0.0, "metric_c": 1.0},
    )
    _write_candidate(
        pareto_dir,
        "balanced",
        {"metric_a": 0.65, "metric_b": 0.65, "metric_c": 0.65},
    )
    return pareto_dir


def test_load_candidates_accepts_run_or_pareto_dir(sample_pareto_dir: Path):
    pareto_dir, candidates, specs = load_candidates(sample_pareto_dir)
    assert pareto_dir == sample_pareto_dir.resolve()
    assert len(candidates) == 4
    assert [spec.metric for spec in specs] == ["metric_a", "metric_b", "metric_c"]

    run_dir, candidates_from_run, specs_from_run = load_candidates(sample_pareto_dir.parent)
    assert run_dir == sample_pareto_dir.resolve()
    assert len(candidates_from_run) == 4
    assert [spec.metric for spec in specs_from_run] == ["metric_a", "metric_b", "metric_c"]


def test_load_candidates_ignores_non_candidate_json_artifacts(sample_pareto_dir: Path):
    (sample_pareto_dir / "selection.json").write_text(
        json.dumps({"selected_count": 4, "selected": []}) + "\n",
        encoding="utf-8",
    )

    pareto_dir, candidates, specs = load_candidates(sample_pareto_dir)

    assert pareto_dir == sample_pareto_dir.resolve()
    assert len(candidates) == 4
    assert [candidate.path.name for candidate in candidates] == [
        "a_extreme.json",
        "b_extreme.json",
        "balanced.json",
        "c_extreme.json",
    ]
    assert [spec.metric for spec in specs] == ["metric_a", "metric_b", "metric_c"]


def test_filter_candidates_with_cli_keep_condition(sample_pareto_dir: Path):
    _pareto_dir, candidates, _specs = load_candidates(sample_pareto_dir)
    filtered, limits = filter_candidates(
        candidates,
        limits_payload=None,
        limit_entries=["metric_a>0.6"],
    )
    assert len(limits) == 1
    assert sorted(candidate.path.stem for candidate in filtered) == ["a_extreme", "balanced"]


def test_filter_candidates_raises_when_limit_metric_is_missing(sample_pareto_dir: Path):
    _pareto_dir, candidates, _specs = load_candidates(sample_pareto_dir)
    with pytest.raises(ValueError, match="Limit metric 'missing_metric' could not be resolved"):
        filter_candidates(
            candidates,
            limits_payload=None,
            limit_entries=["missing_metric>0.0"],
        )


def test_filter_candidates_uses_suite_aggregate_defaults_for_omitted_stat(tmp_path: Path):
    pareto_dir = tmp_path / "run" / "pareto"
    pareto_dir.mkdir(parents=True)
    _write_suite_candidate(
        pareto_dir,
        "passes_by_aggregate_defaults",
        adg_mean=0.02,
        adg_min=-0.05,
        recovery_mean=4000.0,
        recovery_max=9000.0,
        drawdown_mean=0.4,
        drawdown_max=0.7,
    )
    _write_suite_candidate(
        pareto_dir,
        "fails_drawdown_max",
        adg_mean=0.03,
        adg_min=0.01,
        recovery_mean=3000.0,
        recovery_max=3500.0,
        drawdown_mean=0.5,
        drawdown_max=0.9,
    )
    _pareto_dir, candidates, _specs = load_candidates(pareto_dir)

    filtered, limits = filter_candidates(
        candidates,
        limits_payload=None,
        limit_entries=[
            "adg_strategy_pnl_rebased>0.0",
            "peak_recovery_hours_hsl<5000",
            "drawdown_worst_hsl<0.8",
        ],
    )

    assert len(limits) == 3
    assert [candidate.path.stem for candidate in filtered] == ["passes_by_aggregate_defaults"]


def test_filter_candidates_explicit_stat_overrides_suite_aggregate_defaults(tmp_path: Path):
    pareto_dir = tmp_path / "run" / "pareto"
    pareto_dir.mkdir(parents=True)
    _write_suite_candidate(
        pareto_dir,
        "strict_failure",
        adg_mean=0.02,
        adg_min=-0.05,
        recovery_mean=4000.0,
        recovery_max=9000.0,
        drawdown_mean=0.4,
        drawdown_max=0.7,
    )
    _pareto_dir, candidates, _specs = load_candidates(pareto_dir)

    filtered, _limits = filter_candidates(
        candidates,
        limits_payload=None,
        limit_entries=[
            "adg_strategy_pnl_rebased>0.0 stat=min",
            "peak_recovery_hours_hsl<5000 stat=max",
            "drawdown_worst_hsl<0.8 stat=max",
        ],
    )

    assert filtered == []


def test_select_candidate_knee_prefers_balanced_candidate(sample_pareto_dir: Path):
    _pareto_dir, candidates, specs = load_candidates(sample_pareto_dir)
    result = select_candidate(candidates, specs, method="knee")
    assert result.candidate.path.stem == "balanced"


def test_select_candidate_reference_prefers_target_match(sample_pareto_dir: Path):
    _pareto_dir, candidates, specs = load_candidates(sample_pareto_dir)
    result = select_candidate(
        candidates,
        specs,
        method="reference",
        target_pairs=["metric_b=1.0", "metric_a=0.0", "metric_c=0.0"],
    )
    assert result.candidate.path.stem == "b_extreme"


def test_select_candidate_ideal_prefers_balanced_candidate(sample_pareto_dir: Path):
    _pareto_dir, candidates, specs = load_candidates(sample_pareto_dir)
    result = select_candidate(candidates, specs, method="ideal")
    assert result.candidate.path.stem == "balanced"


def test_select_candidate_utility_respects_weights(sample_pareto_dir: Path):
    _pareto_dir, candidates, specs = load_candidates(sample_pareto_dir)
    result = select_candidate(
        candidates,
        specs,
        method="utility",
        weight_pairs=["metric_b=5", "metric_a=1", "metric_c=1"],
    )
    assert result.candidate.path.stem == "b_extreme"


def test_select_candidate_lexicographic_respects_priority(sample_pareto_dir: Path):
    _pareto_dir, candidates, specs = load_candidates(sample_pareto_dir)
    result = select_candidate(
        candidates,
        specs,
        method="lexicographic",
        priority_arg="metric_c,metric_b,metric_a",
    )
    assert result.candidate.path.stem == "c_extreme"


def test_select_candidate_outranking_prefers_balanced_candidate(sample_pareto_dir: Path):
    _pareto_dir, candidates, specs = load_candidates(sample_pareto_dir)
    result = select_candidate(candidates, specs, method="outranking")
    assert result.candidate.path.stem == "balanced"


def test_build_parser_defaults_to_ideal_method():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.method == "ideal"


def test_build_parser_accepts_scenario():
    args = build_parser().parse_args(["--scenario", "bull"])
    assert args.scenario == "bull"


def test_build_parser_accepts_save_outputs_and_overwrite():
    args = build_parser().parse_args(
        [
            "-s",
            "selected.json",
            "-f",
            "filtered",
            "--overwrite",
        ]
    )

    assert args.save_selected == "selected.json"
    assert args.save_filtered == "filtered"
    assert args.overwrite is True


def test_project_and_rebuild_scenario_front(scenario_pareto_dir: Path):
    _pareto_dir, candidates, specs = load_candidates(scenario_pareto_dir)

    bull = project_candidates_to_scenario(candidates, specs, "bull")
    bull_front = build_scenario_front(bull, specs)
    bear = project_candidates_to_scenario(candidates, specs, "bear")
    bear_front = build_scenario_front(bear, specs)

    assert [candidate.path.stem for candidate in bull_front] == ["a", "b", "d"]
    assert [candidate.path.stem for candidate in bear_front] == ["b", "d"]
    assert bull_front[0].objectives == {"metric_a": 0.9, "metric_b": 0.4}


def test_scenario_front_keeps_first_exact_objective_vector(scenario_pareto_dir: Path):
    duplicate = json.loads((scenario_pareto_dir / "a.json").read_text())
    (scenario_pareto_dir / "z_duplicate.json").write_text(json.dumps(duplicate))
    _pareto_dir, candidates, specs = load_candidates(scenario_pareto_dir)

    front = build_scenario_front(
        project_candidates_to_scenario(candidates, specs, "bull"),
        specs,
    )

    assert "a" in [candidate.path.stem for candidate in front]
    assert "z_duplicate" not in [candidate.path.stem for candidate in front]


def test_scenario_projection_fails_for_non_suite_candidate(sample_pareto_dir: Path):
    _pareto_dir, candidates, specs = load_candidates(sample_pareto_dir)
    with pytest.raises(ValueError, match="requires suite Pareto artifacts"):
        project_candidates_to_scenario(candidates, specs, "bull")


def test_scenario_projection_lists_available_labels(scenario_pareto_dir: Path):
    _pareto_dir, candidates, specs = load_candidates(scenario_pareto_dir)
    with pytest.raises(ValueError, match="Available scenarios: bear, bull"):
        project_candidates_to_scenario(candidates, specs, "sideways")


def test_scenario_projection_fails_when_scoring_metric_is_missing(
    scenario_pareto_dir: Path,
):
    candidate_path = scenario_pareto_dir / "a.json"
    payload = json.loads(candidate_path.read_text())
    del payload["suite_metrics"]["metrics"]["metric_b"]["scenarios"]["bull"]
    candidate_path.write_text(json.dumps(payload))
    _pareto_dir, candidates, specs = load_candidates(scenario_pareto_dir)

    with pytest.raises(ValueError, match="missing scoring metric.*metric_b"):
        project_candidates_to_scenario(candidates, specs, "bull")


def test_scenario_limit_uses_scalar_and_rejects_non_mean_reducer(
    scenario_pareto_dir: Path,
):
    _pareto_dir, candidates, specs = load_candidates(scenario_pareto_dir)
    projected = project_candidates_to_scenario(candidates, specs, "bull")

    filtered, _limits = filter_candidates(
        projected,
        limits_payload=None,
        limit_entries=["metric_a>0.75"],
    )
    assert [candidate.path.stem for candidate in filtered] == ["a", "b"]

    with pytest.raises(ValueError, match="stores one mean value.*reducer='max'.*unavailable"):
        filter_candidates(
            projected,
            limits_payload=None,
            limit_entries=["metric_a>0.75 stat=max"],
        )


def test_limit_entry_can_select_scenario_without_projecting_candidates(
    scenario_pareto_dir: Path,
):
    _pareto_dir, candidates, _specs = load_candidates(scenario_pareto_dir)

    filtered, limits = filter_candidates(
        candidates,
        limits_payload=None,
        limit_entries=["metric_a>0.75 scenario=bull"],
    )

    assert [candidate.path.stem for candidate in filtered] == ["a", "b"]
    assert limits[0]["scenario"] == "bull"


def test_filter_candidates_resolves_auto_limit_with_optimizer_direction(
    scenario_pareto_dir: Path,
):
    _pareto_dir, candidates, _specs = load_candidates(scenario_pareto_dir)

    filtered, limits = filter_candidates(
        candidates,
        limits_payload=json.dumps(
            [
                {
                    "metric": "sharpe_ratio_strategy_eq",
                    "penalize_if": "auto",
                    "scenario": "bull",
                    "value": 1.1,
                }
            ]
        ),
        limit_entries=None,
    )

    assert [candidate.path.stem for candidate in filtered] == ["a"]
    assert limits[0]["penalize_if"] == "less_than"


def test_ordinary_filter_uses_stored_legacy_suite_aggregate(tmp_path: Path):
    pareto_dir = tmp_path / "legacy_suite" / "pareto"
    pareto_dir.mkdir(parents=True)
    payload = {
        "optimize": {
            "scoring": [
                {"metric": "metric_a", "goal": "max"},
                {"metric": "metric_b", "goal": "min"},
            ]
        },
        "backtest": {"aggregate": {"default": "max"}},
        "suite_metrics": {
            "aggregate": {
                "aggregated": {
                    "metric_a": 0.8,
                    "metric_b": 0.4,
                }
            }
        },
    }
    (pareto_dir / "candidate.json").write_text(json.dumps(payload), encoding="utf-8")
    _pareto_dir, candidates, _specs = load_candidates(pareto_dir)

    filtered, _limits = filter_candidates(
        candidates,
        limits_payload=None,
        limit_entries=["metric_b<0.5"],
    )

    assert [candidate.path.name for candidate in filtered] == ["candidate.json"]


def test_explicit_null_limit_keeps_suite_aggregate_after_scenario_projection(
    scenario_pareto_dir: Path,
):
    _pareto_dir, candidates, specs = load_candidates(scenario_pareto_dir)
    projected = project_candidates_to_scenario(candidates, specs, "bull")

    filtered, limits = filter_candidates(
        projected,
        limits_payload=None,
        limit_entries=["metric_a<0.6 scenario=null"],
    )

    assert [candidate.path.stem for candidate in filtered] == [
        "a",
        "b",
        "c_dominated",
        "d",
    ]
    assert "scenario" in limits[0]
    assert limits[0]["scenario"] is None

    filtered_max, max_limits = filter_candidates(
        projected,
        limits_payload=None,
        limit_entries=["metric_a<0.6 scenario=null stat=max"],
    )

    assert [candidate.path.stem for candidate in filtered_max] == [
        "a",
        "b",
        "c_dominated",
        "d",
    ]
    assert max_limits[0]["reducer"] == "max"


def test_run_from_args_scenario_json_reports_scope_and_uses_scenario_metrics(
    scenario_pareto_dir: Path,
    capsys,
):
    args = build_parser().parse_args(
        [
            str(scenario_pareto_dir),
            "--scenario",
            "bear",
            "--objectives",
            "sharpe_ratio_strategy_eq",
            "--json",
        ]
    )
    result = run_from_args(args)
    payload = json.loads(capsys.readouterr().out)

    assert result.candidate.path.stem == "b"
    assert payload["scenario"] == "bear"
    assert payload["front_scope"] == "saved_aggregate_pareto_members"
    assert payload["scenario_front_complete"] is False
    assert payload["loaded_count"] == 4
    assert payload["retained_count"] == 4
    assert payload["scenario_front_count"] == 2
    assert payload["selected"]["objectives"]["sharpe_ratio_strategy_eq"] == pytest.approx(1.5)


def test_run_from_args_scenario_text_documents_incomplete_front(
    scenario_pareto_dir: Path,
    capsys,
):
    args = build_parser().parse_args([str(scenario_pareto_dir), "--scenario", "bull"])
    run_from_args(args)
    output = capsys.readouterr().out

    assert "| Scenario              | bull" in output
    assert "| Scenario front        | 3" in output
    assert "saved aggregate Pareto members" in output
    assert "candidates discarded by the suite optimizer are not recoverable" in output


def test_run_from_args_prints_summary(sample_pareto_dir: Path, capsys):
    args = argparse.Namespace(
        path=str(sample_pareto_dir),
        method="ideal",
        limit_entries=[],
        limits_payload=None,
        objectives=None,
        weight=None,
        target=None,
        priority=None,
        show_top=3,
        json_output=False,
    )
    result = run_from_args(args)
    captured = capsys.readouterr().out
    assert "| Loaded candidates" in captured
    assert "| Retained after limits" in captured
    assert "| Applied limits" in captured
    assert "| Method                | ideal" in captured
    assert "| Distance              |" in captured
    assert "Method summary:" in captured
    assert "| Selected file" in captured
    assert "| Selected path" in captured
    assert "| Selected hash" not in captured
    assert "Backtest command: passivbot backtest" in captured
    assert "Active objectives:" in captured
    assert "| metric" in captured
    assert "| goal" in captured
    assert "Why this winner:" in captured
    assert "Objective table:" in captured
    assert "metric" in captured
    assert "utility" in captured
    assert "ideal" in captured
    assert "Target utilities:" not in captured
    assert "Top candidates:" in captured
    assert result.candidate.path.stem == "balanced"


def test_run_from_args_uses_latest_pareto_dir_when_path_omitted(tmp_path: Path, monkeypatch, capsys):
    root = tmp_path / "optimize_results"
    older_run = "2026-04-28T09_00_00_older"
    newer_run = "2026-04-28T10_00_00_newer"
    older = root / older_run / "pareto"
    newer = root / newer_run / "pareto"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    _write_candidate(older, "older_balanced", {"metric_a": 0.6, "metric_b": 0.6, "metric_c": 0.6})
    _write_candidate(newer, "newer_balanced", {"metric_a": 0.65, "metric_b": 0.65, "metric_c": 0.65})
    os.utime(older.resolve(), (1000, 1000))
    os.utime(newer.resolve(), (2000, 2000))
    monkeypatch.chdir(tmp_path)

    args = argparse.Namespace(
        path=None,
        method="ideal",
        limit_entries=[],
        limits_payload=None,
        objectives=None,
        weight=None,
        target=None,
        priority=None,
        show_top=1,
        json_output=False,
    )
    result = run_from_args(args)
    captured = capsys.readouterr().out
    assert str(newer.resolve()) not in captured
    assert f"optimize_results/{newer_run}/pareto" in captured
    assert (
        f"Backtest command: passivbot backtest "
        f"optimize_results/{newer_run}/pareto/newer_balanced.json"
    ) in captured
    assert result.candidate.path.stem == "newer_balanced"


def test_run_from_args_json_output(sample_pareto_dir: Path, capsys):
    args = argparse.Namespace(
        path=str(sample_pareto_dir),
        method="utility",
        limit_entries=[],
        limits_payload=None,
        objectives=None,
        weight=["metric_b=5", "metric_a=1", "metric_c=1"],
        target=None,
        priority=None,
        show_top=2,
        json_output=True,
    )
    result = run_from_args(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["method"] == "utility"
    assert "weighted normalized utility" in payload["method_description"].lower()
    assert payload["selected"]["file"] == "b_extreme.json"
    assert len(payload["top_candidates"]) == 2
    assert payload["top_candidates"][0]["file"] == "b_extreme.json"
    assert payload["selected"]["details"]["utility_contributions"]["metric_b"] > 0
    assert payload["selected"]["details"]["ideal_point"]["metric_b"] == pytest.approx(1.0)
    assert "ranking_order" not in payload["selected"]["details"]
    assert "score_vector" not in payload["selected"]["details"]
    assert result.candidate.path.stem == "b_extreme"


def test_run_from_args_saves_selected_member_exactly(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    output = tmp_path / "promoted" / "candidate.json"
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "--method",
            "utility",
            "--weight",
            "metric_b=5",
            "-s",
            str(output),
        ]
    )

    result = run_from_args(args)

    assert result.candidate.path.name == "b_extreme.json"
    assert output.read_bytes() == (sample_pareto_dir / "b_extreme.json").read_bytes()
    assert "Saved selected member:" in capsys.readouterr().out


def test_run_from_args_saves_filtered_members_and_manifest(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    output_dir = tmp_path / "filtered"
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-l",
            "metric_a>0.6",
            "-f",
            str(output_dir),
        ]
    )

    result = run_from_args(args)

    assert result.candidate.path.name == "balanced.json"
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "a_extreme.json",
        "balanced.json",
        "selection.json",
    ]
    assert (output_dir / "a_extreme.json").read_bytes() == (
        sample_pareto_dir / "a_extreme.json"
    ).read_bytes()
    manifest = json.loads((output_dir / "selection.json").read_text())
    assert manifest["tool"] == "passivbot tool pareto"
    assert manifest["mode"] == "filtered"
    assert manifest["loaded_count"] == 4
    assert manifest["retained_count"] == 2
    assert manifest["scenario"] is None
    assert manifest["method"] == "ideal"
    assert manifest["objectives"] == ["metric_a", "metric_b", "metric_c"]
    assert manifest["weights"] == {
        "metric_a": pytest.approx(1 / 3),
        "metric_b": pytest.approx(1 / 3),
        "metric_c": pytest.approx(1 / 3),
    }
    assert manifest["targets"] == {}
    assert manifest["priority"] == []
    assert manifest["selected_member"]["file"] == "balanced.json"
    assert [member["file"] for member in manifest["members"]] == [
        "a_extreme.json",
        "balanced.json",
    ]
    assert manifest["applied_limits"][0]["metric"] == "metric_a"
    assert "Saved filtered members: 2" in capsys.readouterr().out


def test_saved_outputs_are_reported_in_json(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    selected_output = tmp_path / "selected.json"
    filtered_output = tmp_path / "filtered"
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-l",
            "metric_a>0.6",
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--json",
        ]
    )

    run_from_args(args)
    payload = json.loads(capsys.readouterr().out)

    assert payload["selected"]["saved_path"] == str(selected_output.resolve())
    assert payload["saved_filtered"] == {
        "count": 2,
        "directory": str(filtered_output.resolve()),
        "manifest": str((filtered_output / "selection.json").resolve()),
        "stage": "post_limits",
    }


def test_saved_outputs_require_explicit_overwrite(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"old": true}\n')
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-s", str(selected_output)]
    )

    with pytest.raises(FileExistsError, match="use --overwrite"):
        run_from_args(args)
    assert json.loads(selected_output.read_text()) == {"old": True}

    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-s", str(selected_output), "--overwrite"]
    )
    result = run_from_args(args)
    capsys.readouterr()
    assert selected_output.read_bytes() == result.candidate.path.read_bytes()

    filtered_output = tmp_path / "filtered"
    filtered_output.mkdir()
    stale = filtered_output / "stale.json"
    stale.write_text('{"stale": true}\n')
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(filtered_output)]
    )
    with pytest.raises(FileExistsError, match="use --overwrite"):
        run_from_args(args)
    assert stale.exists()

    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-l",
            "metric_a>0.6",
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )
    run_from_args(args)
    capsys.readouterr()
    assert not stale.exists()
    assert sorted(path.name for path in filtered_output.iterdir()) == [
        "a_extreme.json",
        "balanced.json",
        "selection.json",
    ]


@pytest.mark.parametrize("output_exists", [False, True])
def test_selected_output_uses_destination_permissions(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
    output_exists: bool,
):
    selected_output = tmp_path / "selected.json"
    if output_exists:
        selected_output.write_text('{"old": true}\n')
        selected_output.chmod(0o750)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-s", str(selected_output), "--overwrite"]
    )

    previous_umask = os.umask(0o027)
    try:
        run_from_args(args)
    finally:
        os.umask(previous_umask)
    capsys.readouterr()

    expected_mode = 0o750 if output_exists else 0o640
    assert stat.S_IMODE(selected_output.stat().st_mode) == expected_mode


def test_selected_output_preserves_existing_file_metadata(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"old": true}\n')
    before = selected_output.stat()
    xattr_name = b"user.passivbot_test"
    xattr_supported = hasattr(os, "setxattr") and hasattr(os, "getxattr")
    if xattr_supported:
        try:
            os.setxattr(selected_output, xattr_name, b"preserve")
        except OSError:
            xattr_supported = False
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-s", str(selected_output), "--overwrite"]
    )

    run_from_args(args)
    capsys.readouterr()

    after = selected_output.stat()
    assert (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)
    if xattr_supported:
        assert os.getxattr(selected_output, xattr_name) == b"preserve"


@pytest.mark.skipif(sys.platform != "darwin", reason="requires macOS ACL tools")
def test_selected_output_preserves_macos_access_acl(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"old": true}\n')
    added = subprocess.run(
        ["/bin/chmod", "+a", "everyone allow read", str(selected_output)],
        capture_output=True,
        text=True,
        check=False,
    )
    if added.returncode != 0:
        pytest.skip(f"could not create test ACL: {added.stderr.strip()}")
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-s", str(selected_output), "--overwrite"]
    )

    run_from_args(args)
    capsys.readouterr()
    listed = subprocess.run(
        ["/bin/ls", "-le", str(selected_output)],
        capture_output=True,
        text=True,
        check=True,
    )

    assert " allow read" in listed.stdout


def test_selected_output_overwrite_uses_fresh_modification_time(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"old": true}\n')
    old_timestamp_ns = 946684800_000_000_000
    os.utime(selected_output, ns=(old_timestamp_ns, old_timestamp_ns))
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-s", str(selected_output), "--overwrite"]
    )

    run_from_args(args)
    capsys.readouterr()

    assert selected_output.stat().st_mtime_ns > old_timestamp_ns


def test_selected_staging_uses_normal_exclusive_file_creation(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    selected_output = tmp_path / "selected.json"
    real_open = os.open
    observed = []

    def track_open(path, flags, mode=0o777, *args, **kwargs):
        path = Path(path)
        if path.name.startswith(pareto_explorer.SELECTED_STAGING_PREFIX):
            observed.append((flags, mode))
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", track_open)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-s", str(selected_output)]
    )

    run_from_args(args)
    capsys.readouterr()

    assert observed
    flags, mode = observed[0]
    assert flags & os.O_CREAT
    assert flags & os.O_EXCL
    assert mode == 0o666


@pytest.mark.parametrize(
    ("error_number", "winerror"),
    [(errno.EOPNOTSUPP, None), (errno.EINVAL, 1)],
)
def test_selected_output_falls_back_when_hard_links_are_unsupported(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
    error_number: int,
    winerror: int | None,
):
    selected_output = tmp_path / "selected.json"

    def reject_hard_link(*args, **kwargs):
        error = OSError(error_number, "hard links unsupported")
        if winerror is not None:
            error.winerror = winerror
        raise error

    monkeypatch.setattr(os, "link", reject_hard_link)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-s", str(selected_output)]
    )

    result = run_from_args(args)
    capsys.readouterr()

    assert selected_output.read_bytes() == result.candidate.source_bytes


def test_selected_fallback_refuses_to_remove_concurrent_replacement(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    real_copy_metadata = pareto_explorer._copy_file_metadata

    def reject_hard_link(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "hard links unsupported")

    def replace_during_metadata_copy(source: Path, destination: Path):
        if destination == selected_output:
            newer = tmp_path / "newer.json"
            newer.write_text('{"selected": "newer"}\n')
            os.replace(newer, selected_output)
            raise OSError("simulated metadata copy failure")
        real_copy_metadata(source, destination)

    monkeypatch.setattr(os, "link", reject_hard_link)
    monkeypatch.setattr(
        pareto_explorer, "_copy_file_metadata", replace_during_metadata_copy
    )
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-s", str(selected_output), "--overwrite"]
    )

    with pytest.raises(RuntimeError, match="changed concurrently"):
        run_from_args(args)

    assert json.loads(selected_output.read_text()) == {"selected": "newer"}
    backups = [path for path in tmp_path.iterdir() if path.suffix == ".backup"]
    assert len(backups) == 1
    assert json.loads(backups[0].read_text()) == {"selected": "old"}
    backups[0].unlink()


def test_selected_fallback_recovers_when_exclusive_creation_is_interrupted(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    filtered_output = tmp_path / "filtered"
    real_open = Path.open
    interrupted = False

    def reject_hard_link(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "hard links unsupported")

    def interrupt_after_exclusive_create(path: Path, *args, **kwargs):
        nonlocal interrupted
        output_file = real_open(path, *args, **kwargs)
        if path == selected_output and args and args[0] == "xb" and not interrupted:
            interrupted = True
            output_file.close()
            raise KeyboardInterrupt()
        return output_file

    monkeypatch.setattr(os, "link", reject_hard_link)
    monkeypatch.setattr(Path, "open", interrupt_after_exclusive_create)
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    with pytest.raises(RuntimeError, match="unverified file retained"):
        run_from_args(args)

    assert interrupted
    assert json.loads(selected_output.read_text()) == {"selected": "old"}
    assert not filtered_output.exists()
    failed = [path for path in tmp_path.iterdir() if path.suffix == ".failed"]
    assert len(failed) == 1
    assert failed[0].read_bytes() == b""
    failed[0].unlink()


def test_selected_overwrite_detects_replacement_before_commit(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    real_replace = os.replace
    raced = False

    def replace_destination_before_move(source, destination):
        nonlocal raced
        source = Path(source)
        destination = Path(destination)
        if source == selected_output and destination.suffix == ".backup" and not raced:
            newer = tmp_path / "newer.json"
            newer.write_text('{"selected": "newer"}\n')
            real_replace(newer, selected_output)
            raced = True
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", replace_destination_before_move)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-s", str(selected_output), "--overwrite"]
    )

    with pytest.raises(RuntimeError, match="changed concurrently; refusing install"):
        run_from_args(args)

    assert raced
    assert json.loads(selected_output.read_text()) == {"selected": "newer"}
    assert not [path for path in tmp_path.iterdir() if path.suffix == ".backup"]


def test_selected_overwrite_detects_in_place_edit_before_commit(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    original_inode = selected_output.stat().st_ino
    real_replace = os.replace
    edited = False

    def edit_destination_before_move(source, destination):
        nonlocal edited
        source = Path(source)
        destination = Path(destination)
        if source == selected_output and destination.suffix == ".backup" and not edited:
            selected_output.write_text('{"selected": "newer"}\n')
            edited = True
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", edit_destination_before_move)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-s", str(selected_output), "--overwrite"]
    )

    with pytest.raises(RuntimeError, match="changed concurrently; refusing install"):
        run_from_args(args)

    assert edited
    assert selected_output.stat().st_ino == original_inode
    assert json.loads(selected_output.read_text()) == {"selected": "newer"}
    assert not [path for path in tmp_path.iterdir() if path.suffix == ".backup"]


def test_selected_output_supports_name_max_destination(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
    suffix = ".json"
    selected_output = tmp_path / ("x" * (name_max - len(suffix)) + suffix)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-s", str(selected_output)]
    )

    result = run_from_args(args)
    capsys.readouterr()

    assert selected_output.read_bytes() == result.candidate.source_bytes


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="requires POSIX FIFO support")
def test_selected_output_refuses_existing_special_node(
    sample_pareto_dir: Path,
    tmp_path: Path,
):
    selected_output = tmp_path / "selected.json"
    os.mkfifo(selected_output)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-s", str(selected_output), "--overwrite"]
    )

    with pytest.raises(ValueError, match="must be a regular file"):
        run_from_args(args)


def test_selected_output_does_not_overwrite_racing_destination_without_permission(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected_output = tmp_path / "selected.json"
    destination_created = False

    def create_destination_during_staging(candidate, destination, file_descriptor):
        nonlocal destination_created
        destination = Path(destination)
        if not destination_created and destination.suffix == ".tmp":
            selected_output.write_text('{"racing": true}\n')
            destination_created = True
        with os.fdopen(file_descriptor, "wb") as destination_file:
            destination_file.write(candidate.source_bytes)

    monkeypatch.setattr(
        "pareto_explorer._write_candidate_snapshot_fd",
        create_destination_during_staging,
    )
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-s", str(selected_output)]
    )

    with pytest.raises(FileExistsError, match="use --overwrite"):
        run_from_args(args)

    assert json.loads(selected_output.read_text()) == {"racing": True}


def test_selected_output_continues_when_post_link_staging_cleanup_fails(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    selected_output = tmp_path / "selected.json"
    filtered_output = tmp_path / "filtered"
    real_unlink = Path.unlink

    def fail_selected_staging_cleanup(path: Path, *args, **kwargs):
        if path.name.startswith(pareto_explorer.SELECTED_STAGING_PREFIX) and path.suffix == ".tmp":
            raise OSError("simulated selected staging cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_selected_staging_cleanup)
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
        ]
    )

    result = run_from_args(args)

    assert selected_output.read_bytes() == result.candidate.source_bytes
    assert (filtered_output / "selection.json").exists()
    assert "selected output installed, but temporary path could not be removed" in (
        capsys.readouterr().err
    )
    staging_paths = [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(pareto_explorer.SELECTED_STAGING_PREFIX) and path.suffix == ".tmp"
    ]
    assert len(staging_paths) == 1
    real_unlink(staging_paths[0])


def test_filtered_overwrite_refuses_non_json_entries(
    sample_pareto_dir: Path,
    tmp_path: Path,
):
    output_dir = tmp_path / "filtered"
    output_dir.mkdir()
    note = output_dir / "notes.txt"
    note.write_text("keep me\n")
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output_dir), "--overwrite"]
    )

    with pytest.raises(FileExistsError, match="non-JSON entries: notes.txt"):
        run_from_args(args)

    assert note.read_text() == "keep me\n"


def test_filtered_output_detects_destination_filesystem_filename_collision(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    output_dir = tmp_path / "filtered"
    real_open = os.open
    member_creations = 0

    def collide_on_second_member(path, flags, mode=0o777, *args, **kwargs):
        nonlocal member_creations
        path = Path(path)
        if (
            path.parent.name.startswith(pareto_explorer.FILTERED_STAGING_PREFIX)
            and path.suffix == ".json"
            and path.name != pareto_explorer.FILTERED_SELECTION_MANIFEST
            and flags & os.O_EXCL
        ):
            member_creations += 1
            if member_creations == 2:
                raise FileExistsError(path)
        return real_open(path, flags, mode, *args, **kwargs)

    monkeypatch.setattr(os, "open", collide_on_second_member)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output_dir)]
    )

    with pytest.raises(ValueError, match="collides on the destination filesystem"):
        run_from_args(args)

    assert not output_dir.exists()


@pytest.mark.parametrize("overwrite", [False, True])
def test_save_filtered_refuses_ds_store(
    sample_pareto_dir: Path,
    tmp_path: Path,
    overwrite: bool,
):
    output_dir = tmp_path / "filtered"
    output_dir.mkdir()
    ds_store = output_dir / ".DS_Store"
    ds_store.write_bytes(b"metadata")
    command = [str(sample_pareto_dir), "-f", str(output_dir)]
    if overwrite:
        command.append("--overwrite")
    args = build_parser().parse_args(command)

    with pytest.raises(FileExistsError, match=r"non-JSON entries: \.DS_Store"):
        run_from_args(args)

    assert ds_store.read_bytes() == b"metadata"


def test_filtered_overwrite_revalidates_contents_after_move_aside(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    output_dir = tmp_path / "filtered"
    output_dir.mkdir()
    stale = output_dir / "stale.json"
    stale.write_text('{"stale": true}\n')
    real_replace = os.replace
    injected = False

    def add_non_json_before_move_aside(source, destination):
        nonlocal injected
        source = Path(source)
        destination = Path(destination)
        if (
            source == output_dir
            and destination.name.startswith(pareto_explorer.FILTERED_BACKUP_PREFIX)
            and not injected
        ):
            (output_dir / "notes.txt").write_text("preserve me\n")
            injected = True
        return real_replace(source, destination)

    monkeypatch.setattr("pareto_explorer.os.replace", add_non_json_before_move_aside)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output_dir), "--overwrite"]
    )

    with pytest.raises(FileExistsError, match="non-JSON entries: notes.txt"):
        run_from_args(args)

    assert json.loads(stale.read_text()) == {"stale": True}
    assert (output_dir / "notes.txt").read_text() == "preserve me\n"
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "notes.txt",
        "stale.json",
    ]


def test_filtered_overwrite_rejects_concurrent_directory_replacement(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    output_dir = tmp_path / "filtered"
    output_dir.mkdir()
    stale = output_dir / "stale.json"
    stale.write_text('{"stale": true}\n')
    detached_old = tmp_path / "detached-old"
    real_replace = os.replace
    replaced = False

    def replace_destination_before_move_aside(source, destination):
        nonlocal replaced
        source = Path(source)
        destination = Path(destination)
        if (
            source == output_dir
            and destination.name.startswith(pareto_explorer.FILTERED_BACKUP_PREFIX)
            and not replaced
        ):
            real_replace(output_dir, detached_old)
            output_dir.mkdir()
            (output_dir / "newer.json").write_text('{"newer": true}\n')
            replaced = True
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", replace_destination_before_move_aside)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output_dir), "--overwrite"]
    )

    with pytest.raises(RuntimeError, match="changed concurrently; refusing install"):
        run_from_args(args)

    assert replaced
    assert json.loads((output_dir / "newer.json").read_text()) == {"newer": True}
    assert sorted(path.name for path in output_dir.iterdir()) == ["newer.json"]
    assert json.loads((detached_old / "stale.json").read_text()) == {"stale": True}


def test_filtered_overwrite_preserves_previous_set_when_staging_fails(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    output_dir = tmp_path / "filtered"
    output_dir.mkdir()
    stale = output_dir / "stale.json"
    stale.write_text('{"stale": true}\n')
    copies = 0

    real_write_snapshot = pareto_explorer._write_candidate_snapshot_fd

    def fail_on_second_copy(candidate, destination, file_descriptor):
        nonlocal copies
        copies += 1
        if copies == 2:
            os.close(file_descriptor)
            raise OSError("simulated copy failure")
        real_write_snapshot(candidate, destination, file_descriptor)

    monkeypatch.setattr(
        "pareto_explorer._write_candidate_snapshot_fd", fail_on_second_copy
    )
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output_dir), "--overwrite"]
    )

    with pytest.raises(OSError, match="simulated copy failure"):
        run_from_args(args)

    assert json.loads(stale.read_text()) == {"stale": True}
    assert sorted(path.name for path in output_dir.iterdir()) == ["stale.json"]
    assert not [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(pareto_explorer.FILTERED_STAGING_PREFIX)
    ]


@pytest.mark.parametrize("selected_exists", [False, True])
def test_combined_outputs_restore_selected_when_filtered_staging_fails(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
    selected_exists: bool,
):
    selected_output = tmp_path / "selected.json"
    original_inode = None
    if selected_exists:
        selected_output.write_text('{"selected": "old"}\n')
        original_inode = selected_output.stat().st_ino
    filtered_output = tmp_path / "filtered"
    filtered_output.mkdir()
    stale = filtered_output / "stale.json"
    stale.write_text('{"filtered": "old"}\n')

    real_write_snapshot = pareto_explorer._write_candidate_snapshot_fd

    def fail_during_filtered_staging(candidate, destination, file_descriptor):
        destination = Path(destination)
        if destination.name == "b_extreme.json" and destination.parent.name.startswith(
            pareto_explorer.FILTERED_STAGING_PREFIX
        ):
            os.close(file_descriptor)
            raise OSError("simulated filtered copy failure")
        real_write_snapshot(candidate, destination, file_descriptor)

    monkeypatch.setattr(
        "pareto_explorer._write_candidate_snapshot_fd",
        fail_during_filtered_staging,
    )
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    with pytest.raises(OSError, match="simulated filtered copy failure"):
        run_from_args(args)

    if selected_exists:
        assert json.loads(selected_output.read_text()) == {"selected": "old"}
        assert selected_output.stat().st_ino == original_inode
    else:
        assert not selected_output.exists()
    assert json.loads(stale.read_text()) == {"filtered": "old"}
    assert sorted(path.name for path in filtered_output.iterdir()) == ["stale.json"]
    assert not [path for path in tmp_path.iterdir() if ".backup" in path.name]


def test_combined_outputs_restore_both_destinations_when_interrupted(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    selected_inode = selected_output.stat().st_ino
    filtered_output = tmp_path / "filtered"
    filtered_output.mkdir()
    stale = filtered_output / "stale.json"
    stale.write_text('{"filtered": "old"}\n')
    real_copy_metadata = pareto_explorer._copy_directory_metadata

    def interrupt_after_filtered_move(source: Path, destination: Path):
        if source.name.startswith(pareto_explorer.FILTERED_BACKUP_PREFIX):
            raise KeyboardInterrupt()
        real_copy_metadata(source, destination)

    monkeypatch.setattr(
        pareto_explorer, "_copy_directory_metadata", interrupt_after_filtered_move
    )
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    with pytest.raises(KeyboardInterrupt):
        run_from_args(args)

    assert json.loads(selected_output.read_text()) == {"selected": "old"}
    assert selected_output.stat().st_ino == selected_inode
    assert json.loads(stale.read_text()) == {"filtered": "old"}
    assert sorted(path.name for path in filtered_output.iterdir()) == ["stale.json"]
    assert not [path for path in tmp_path.iterdir() if ".backup" in path.name]


def test_combined_outputs_restore_selected_when_install_return_is_interrupted(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    selected_inode = selected_output.stat().st_ino
    filtered_output = tmp_path / "filtered"
    filtered_output.mkdir()
    stale = filtered_output / "stale.json"
    stale.write_text('{"filtered": "old"}\n')
    real_write_selected = pareto_explorer._write_selected_output
    interrupted = False

    def interrupt_after_selected_return(*args, **kwargs):
        nonlocal interrupted
        real_write_selected(*args, **kwargs)
        interrupted = True
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        pareto_explorer, "_write_selected_output", interrupt_after_selected_return
    )
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    with pytest.raises(KeyboardInterrupt):
        run_from_args(args)

    assert interrupted
    assert json.loads(selected_output.read_text()) == {"selected": "old"}
    assert selected_output.stat().st_ino == selected_inode
    assert json.loads(stale.read_text()) == {"filtered": "old"}
    assert sorted(path.name for path in filtered_output.iterdir()) == ["stale.json"]
    assert not [path for path in tmp_path.iterdir() if ".backup" in path.name]


def test_combined_outputs_cleanup_filtered_reservation_when_return_is_interrupted(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    selected_inode = selected_output.stat().st_ino
    filtered_output = tmp_path / "filtered"
    real_ensure = pareto_explorer._ensure_filtered_output_dir
    interrupted = False

    def interrupt_after_reservation(*args, **kwargs):
        nonlocal interrupted
        result = real_ensure(*args, **kwargs)
        interrupted = True
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        pareto_explorer, "_ensure_filtered_output_dir", interrupt_after_reservation
    )
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    with pytest.raises(KeyboardInterrupt):
        run_from_args(args)

    assert interrupted
    assert json.loads(selected_output.read_text()) == {"selected": "old"}
    assert selected_output.stat().st_ino == selected_inode
    assert not filtered_output.exists()


def test_combined_outputs_record_no_hard_link_commit_before_nested_return(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    filtered_output = tmp_path / "filtered"
    real_install = pareto_explorer._install_selected_without_overwrite
    interrupted = False

    def reject_hard_link(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "hard links unsupported")

    def interrupt_after_nested_return(*args, **kwargs):
        nonlocal interrupted
        result = real_install(*args, **kwargs)
        destination = Path(args[1])
        if destination == selected_output and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(os, "link", reject_hard_link)
    monkeypatch.setattr(
        pareto_explorer,
        "_install_selected_without_overwrite",
        interrupt_after_nested_return,
    )
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    result = run_from_args(args)

    assert interrupted
    assert selected_output.read_bytes() == result.candidate.source_bytes
    assert (filtered_output / "selection.json").exists()
    assert "install committed before interruption" in capsys.readouterr().err


def test_combined_outputs_keep_committed_pair_when_filtered_return_is_interrupted(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    filtered_output = tmp_path / "filtered"
    filtered_output.mkdir()
    (filtered_output / "stale.json").write_text('{"filtered": "old"}\n')
    real_write_filtered = pareto_explorer._write_filtered_outputs
    interrupted = False

    def interrupt_after_filtered_return(*args, **kwargs):
        nonlocal interrupted
        real_write_filtered(*args, **kwargs)
        interrupted = True
        raise KeyboardInterrupt()

    monkeypatch.setattr(
        pareto_explorer, "_write_filtered_outputs", interrupt_after_filtered_return
    )
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    with pytest.raises(KeyboardInterrupt):
        run_from_args(args)

    assert interrupted
    assert selected_output.read_bytes() != b'{"selected": "old"}\n'
    assert not (filtered_output / "stale.json").exists()
    assert (filtered_output / "selection.json").exists()


def test_selected_rollback_restores_path_when_verification_is_interrupted(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    filtered_output = tmp_path / "filtered"
    real_write_snapshot = pareto_explorer._write_candidate_snapshot_fd
    real_stat = Path.stat
    interrupted = False

    def fail_filtered_staging(candidate, destination, file_descriptor):
        destination = Path(destination)
        if destination.parent.name.startswith(pareto_explorer.FILTERED_STAGING_PREFIX):
            os.close(file_descriptor)
            raise OSError("simulated filtered copy failure")
        real_write_snapshot(candidate, destination, file_descriptor)

    def interrupt_rollback_stat(path: Path, *args, **kwargs):
        nonlocal interrupted
        if path.suffix == ".rollback" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt()
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(
        pareto_explorer, "_write_candidate_snapshot_fd", fail_filtered_staging
    )
    monkeypatch.setattr(Path, "stat", interrupt_rollback_stat)
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    with pytest.raises(KeyboardInterrupt):
        run_from_args(args)

    assert interrupted
    assert selected_output.exists()
    backups = [path for path in tmp_path.iterdir() if path.suffix == ".backup"]
    rollbacks = [path for path in tmp_path.iterdir() if path.suffix == ".rollback"]
    assert len(backups) == 1
    assert len(rollbacks) == 1
    assert json.loads(backups[0].read_text()) == {"selected": "old"}
    assert selected_output.read_bytes() == rollbacks[0].read_bytes()
    backups[0].unlink()
    rollbacks[0].unlink()


def test_selected_rollback_preserves_timestamps_without_hard_links(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    old_timestamp_ns = 946684800_000_000_000
    os.utime(selected_output, ns=(old_timestamp_ns, old_timestamp_ns))
    filtered_output = tmp_path / "filtered"
    real_write_snapshot = pareto_explorer._write_candidate_snapshot_fd

    def reject_hard_link(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "hard links unsupported")

    def fail_filtered_staging(candidate, destination, file_descriptor):
        destination = Path(destination)
        if destination.parent.name.startswith(pareto_explorer.FILTERED_STAGING_PREFIX):
            os.close(file_descriptor)
            raise OSError("simulated filtered copy failure")
        real_write_snapshot(candidate, destination, file_descriptor)

    monkeypatch.setattr(os, "link", reject_hard_link)
    monkeypatch.setattr(
        pareto_explorer, "_write_candidate_snapshot_fd", fail_filtered_staging
    )
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    with pytest.raises(OSError, match="simulated filtered copy failure"):
        run_from_args(args)

    assert json.loads(selected_output.read_text()) == {"selected": "old"}
    assert selected_output.stat().st_mtime_ns == old_timestamp_ns


def test_combined_outputs_refuse_rollback_over_newer_selected_file(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    filtered_output = tmp_path / "filtered"

    real_write_snapshot = pareto_explorer._write_candidate_snapshot_fd

    def replace_selected_then_fail(candidate, destination, file_descriptor):
        destination = Path(destination)
        if destination.parent.name.startswith(pareto_explorer.FILTERED_STAGING_PREFIX):
            newer = tmp_path / "newer.json"
            newer.write_text('{"selected": "newer"}\n')
            os.replace(newer, selected_output)
            os.close(file_descriptor)
            raise OSError("simulated filtered copy failure")
        real_write_snapshot(candidate, destination, file_descriptor)

    monkeypatch.setattr(
        pareto_explorer, "_write_candidate_snapshot_fd", replace_selected_then_fail
    )
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    with pytest.raises(RuntimeError, match="changed concurrently; refusing rollback"):
        run_from_args(args)

    assert json.loads(selected_output.read_text()) == {"selected": "newer"}
    backups = [path for path in tmp_path.iterdir() if path.suffix == ".backup"]
    assert len(backups) == 1
    assert json.loads(backups[0].read_text()) == {"selected": "old"}
    backups[0].unlink()


def test_combined_outputs_refuse_rollback_over_in_place_selected_edit(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    filtered_output = tmp_path / "filtered"

    real_write_snapshot = pareto_explorer._write_candidate_snapshot_fd

    def edit_selected_then_fail(candidate, destination, file_descriptor):
        destination = Path(destination)
        if destination.parent.name.startswith(pareto_explorer.FILTERED_STAGING_PREFIX):
            selected_output.write_text('{"selected": "newer"}\n')
            os.close(file_descriptor)
            raise OSError("simulated filtered copy failure")
        real_write_snapshot(candidate, destination, file_descriptor)

    monkeypatch.setattr(
        pareto_explorer, "_write_candidate_snapshot_fd", edit_selected_then_fail
    )
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    with pytest.raises(RuntimeError, match="changed concurrently; refusing rollback"):
        run_from_args(args)

    assert json.loads(selected_output.read_text()) == {"selected": "newer"}
    backups = [path for path in tmp_path.iterdir() if path.suffix == ".backup"]
    assert len(backups) == 1
    assert json.loads(backups[0].read_text()) == {"selected": "old"}
    backups[0].unlink()


def test_combined_outputs_keep_committed_pair_when_filtered_rename_is_interrupted(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    filtered_output = tmp_path / "filtered"
    filtered_output.mkdir()
    (filtered_output / "stale.json").write_text('{"filtered": "old"}\n')
    real_replace = os.replace
    interrupted = False

    def interrupt_after_filtered_commit(source, destination):
        nonlocal interrupted
        source = Path(source)
        destination = Path(destination)
        result = real_replace(source, destination)
        if (
            not interrupted
            and source.name.startswith(pareto_explorer.FILTERED_STAGING_PREFIX)
            and source.name.endswith(".tmp")
            and destination == filtered_output
        ):
            interrupted = True
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(os, "replace", interrupt_after_filtered_commit)
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    result = run_from_args(args)

    assert interrupted
    assert selected_output.read_bytes() == result.candidate.source_bytes
    assert not (filtered_output / "stale.json").exists()
    assert (filtered_output / "selection.json").exists()
    assert "install committed before interruption" in capsys.readouterr().err
    assert not [path for path in tmp_path.iterdir() if ".backup" in path.name]


def test_combined_outputs_do_not_delete_reused_filtered_staging_path_after_commit(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    filtered_output = tmp_path / "filtered"
    filtered_output.mkdir()
    (filtered_output / "stale.json").write_text('{"filtered": "old"}\n')
    real_replace = os.replace
    reused_staging: Path | None = None

    def reuse_staging_path_after_commit(source, destination):
        nonlocal reused_staging
        source = Path(source)
        destination = Path(destination)
        result = real_replace(source, destination)
        if (
            source.name.startswith(pareto_explorer.FILTERED_STAGING_PREFIX)
            and destination == filtered_output
        ):
            source.mkdir()
            (source / "foreign.txt").write_text("preserve me\n")
            reused_staging = source
        return result

    monkeypatch.setattr(os, "replace", reuse_staging_path_after_commit)
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    result = run_from_args(args)

    assert reused_staging is not None
    assert selected_output.read_bytes() == result.candidate.source_bytes
    assert not (filtered_output / "stale.json").exists()
    assert (filtered_output / "selection.json").exists()
    assert (reused_staging / "foreign.txt").read_text() == "preserve me\n"
    shutil.rmtree(reused_staging)


def test_combined_outputs_keep_committed_pair_when_selected_rename_is_interrupted(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    filtered_output = tmp_path / "filtered"
    real_link = os.link
    interrupted = False

    def interrupt_after_selected_commit(source, destination, *args, **kwargs):
        nonlocal interrupted
        source = Path(source)
        destination = Path(destination)
        result = real_link(source, destination, *args, **kwargs)
        if (
            not interrupted
            and source.name.startswith(pareto_explorer.SELECTED_STAGING_PREFIX)
            and source.name.endswith(".tmp")
            and destination == selected_output
        ):
            interrupted = True
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(os, "link", interrupt_after_selected_commit)
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    result = run_from_args(args)

    assert interrupted
    assert selected_output.read_bytes() == result.candidate.source_bytes
    assert (filtered_output / "selection.json").exists()
    assert "selected output install committed before interruption" in (
        capsys.readouterr().err
    )
    assert not [path for path in tmp_path.iterdir() if ".backup" in path.name]


def test_combined_outputs_restore_both_when_filtered_move_aside_is_interrupted(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    selected_inode = selected_output.stat().st_ino
    filtered_output = tmp_path / "filtered"
    filtered_output.mkdir()
    stale = filtered_output / "stale.json"
    stale.write_text('{"filtered": "old"}\n')
    real_replace = os.replace
    interrupted = False

    def interrupt_after_filtered_move_aside(source, destination):
        nonlocal interrupted
        source = Path(source)
        destination = Path(destination)
        result = real_replace(source, destination)
        if (
            not interrupted
            and source == filtered_output
            and destination.name.startswith(pareto_explorer.FILTERED_BACKUP_PREFIX)
        ):
            interrupted = True
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(os, "replace", interrupt_after_filtered_move_aside)
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    with pytest.raises(KeyboardInterrupt):
        run_from_args(args)

    assert interrupted
    assert json.loads(selected_output.read_text()) == {"selected": "old"}
    assert selected_output.stat().st_ino == selected_inode
    assert json.loads(stale.read_text()) == {"filtered": "old"}
    assert sorted(path.name for path in filtered_output.iterdir()) == ["stale.json"]
    assert not [path for path in tmp_path.iterdir() if ".backup" in path.name]


def test_combined_outputs_remain_consistent_when_backup_cleanup_fails(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    filtered_output = tmp_path / "filtered"
    filtered_output.mkdir()
    (filtered_output / "stale.json").write_text('{"filtered": "old"}\n')
    real_remove = pareto_explorer._remove_output_tree

    def fail_backup_cleanup(path: Path):
        if path.name.startswith(pareto_explorer.FILTERED_BACKUP_PREFIX):
            raise OSError("simulated backup cleanup failure")
        return real_remove(path)

    monkeypatch.setattr(pareto_explorer, "_remove_output_tree", fail_backup_cleanup)
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    result = run_from_args(args)

    assert selected_output.read_bytes() == result.candidate.source_bytes
    assert not (filtered_output / "stale.json").exists()
    assert (filtered_output / "selection.json").exists()
    assert "old backup could not be removed" in capsys.readouterr().err
    backups = [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(pareto_explorer.FILTERED_BACKUP_PREFIX)
    ]
    assert len(backups) == 1
    real_remove(backups[0])


def test_combined_outputs_remain_consistent_when_backup_cleanup_is_interrupted(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    filtered_output = tmp_path / "filtered"
    filtered_output.mkdir()
    (filtered_output / "stale.json").write_text('{"filtered": "old"}\n')
    real_remove = pareto_explorer._remove_output_tree

    def interrupt_backup_cleanup(path: Path):
        if path.name.startswith(pareto_explorer.FILTERED_BACKUP_PREFIX):
            raise KeyboardInterrupt()
        return real_remove(path)

    monkeypatch.setattr(
        pareto_explorer, "_remove_output_tree", interrupt_backup_cleanup
    )
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    result = run_from_args(args)

    assert selected_output.read_bytes() == result.candidate.source_bytes
    assert not (filtered_output / "stale.json").exists()
    assert (filtered_output / "selection.json").exists()
    assert "old backup could not be removed" in capsys.readouterr().err
    backups = [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(pareto_explorer.FILTERED_BACKUP_PREFIX)
    ]
    assert len(backups) == 1
    real_remove(backups[0])


def test_combined_outputs_warn_when_selected_backup_cleanup_fails(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    selected_output = tmp_path / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    filtered_output = tmp_path / "filtered"
    real_unlink = Path.unlink
    backup_unlinks = 0

    def fail_selected_backup_cleanup(path: Path, *args, **kwargs):
        nonlocal backup_unlinks
        if path.suffix == ".backup":
            backup_unlinks += 1
            if backup_unlinks == 2:
                raise OSError("simulated selected backup cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_selected_backup_cleanup)
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    result = run_from_args(args)

    assert selected_output.read_bytes() == result.candidate.source_bytes
    assert (filtered_output / "selection.json").exists()
    assert "selected output installed, but temporary path could not be removed" in (
        capsys.readouterr().err
    )
    backups = [path for path in tmp_path.iterdir() if path.suffix == ".backup"]
    assert len(backups) == 1
    real_unlink(backups[0])


@pytest.mark.parametrize("output_exists", [False, True])
def test_filtered_output_directory_uses_destination_permissions(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
    output_exists: bool,
):
    output_dir = tmp_path / "filtered"
    if output_exists:
        output_dir.mkdir()
        output_dir.chmod(0o750)
        (output_dir / "stale.json").write_text('{"stale": true}\n')
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output_dir), "--overwrite"]
    )

    previous_umask = os.umask(0o027)
    try:
        run_from_args(args)
    finally:
        os.umask(previous_umask)
    capsys.readouterr()

    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o750


def test_filtered_output_preserves_existing_directory_metadata(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    output_dir = tmp_path / "filtered"
    output_dir.mkdir()
    (output_dir / "stale.json").write_text('{"stale": true}\n')
    before = output_dir.stat()
    xattr_name = b"user.passivbot_test"
    xattr_supported = hasattr(os, "setxattr") and hasattr(os, "getxattr")
    if xattr_supported:
        try:
            os.setxattr(output_dir, xattr_name, b"preserve")
        except OSError:
            xattr_supported = False
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output_dir), "--overwrite"]
    )

    run_from_args(args)
    capsys.readouterr()

    after = output_dir.stat()
    assert (after.st_uid, after.st_gid) == (before.st_uid, before.st_gid)
    if xattr_supported:
        assert os.getxattr(output_dir, xattr_name) == b"preserve"


def test_filtered_output_overwrite_uses_fresh_modification_time(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    output_dir = tmp_path / "filtered"
    output_dir.mkdir()
    (output_dir / "stale.json").write_text('{"stale": true}\n')
    old_timestamp_ns = 946684800_000_000_000
    os.utime(output_dir, ns=(old_timestamp_ns, old_timestamp_ns))
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output_dir), "--overwrite"]
    )

    run_from_args(args)
    capsys.readouterr()

    assert output_dir.stat().st_mtime_ns > old_timestamp_ns


def test_filtered_output_applies_directory_metadata_before_writing_members(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    output_dir = tmp_path / "filtered"
    output_dir.mkdir()
    (output_dir / "stale.json").write_text('{"stale": true}\n')
    real_copy_metadata = pareto_explorer._copy_directory_metadata
    real_write_snapshot = pareto_explorer._write_candidate_snapshot_fd
    metadata_ready = False

    def track_metadata(source: Path, destination: Path):
        nonlocal metadata_ready
        real_copy_metadata(source, destination)
        if source == output_dir and destination.name.endswith(".tmp"):
            metadata_ready = True

    def require_metadata_before_write(candidate, destination, file_descriptor):
        assert metadata_ready
        real_write_snapshot(candidate, destination, file_descriptor)

    monkeypatch.setattr(pareto_explorer, "_copy_directory_metadata", track_metadata)
    monkeypatch.setattr(
        pareto_explorer, "_write_candidate_snapshot_fd", require_metadata_before_write
    )
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output_dir), "--overwrite"]
    )

    run_from_args(args)
    capsys.readouterr()

    assert metadata_ready
    assert (output_dir / "selection.json").exists()


def test_filtered_output_rechecks_directory_created_during_reservation(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    output_dir = tmp_path / "filtered"
    real_mkdir = Path.mkdir
    raced = False

    def create_racing_destination(path: Path, *args, **kwargs):
        nonlocal raced
        if path == output_dir and not raced:
            raced = True
            os.mkdir(path, 0o750)
            path.chmod(0o750)
            raise FileExistsError(path)
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", create_racing_destination)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output_dir), "--overwrite"]
    )

    run_from_args(args)
    capsys.readouterr()

    assert raced
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o750


def test_new_filtered_output_retains_inherited_setgid_metadata(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    output_parent = tmp_path / "exports"
    output_parent.mkdir()
    output_parent.chmod(stat.S_IMODE(output_parent.stat().st_mode) | stat.S_ISGID)
    probe = output_parent / "probe"
    probe.mkdir(mode=0o777)
    inherited_setgid = bool(probe.stat().st_mode & stat.S_ISGID)
    probe.rmdir()
    if not inherited_setgid:
        pytest.skip("filesystem does not inherit setgid on child directories")
    output_dir = output_parent / "filtered"
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output_dir)]
    )

    run_from_args(args)
    capsys.readouterr()

    assert output_dir.stat().st_mode & stat.S_ISGID


def test_filtered_output_supports_name_max_destination(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    name_max = os.pathconf(tmp_path, "PC_NAME_MAX")
    output_dir = tmp_path / ("x" * name_max)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output_dir)]
    )

    run_from_args(args)
    capsys.readouterr()

    assert (output_dir / "selection.json").exists()


def test_filtered_output_builds_before_applying_read_only_destination_mode(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    output_dir = tmp_path / "filtered"
    output_dir.mkdir()
    (output_dir / "stale.json").write_text('{"stale": true}\n')
    output_dir.chmod(0o555)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output_dir), "--overwrite"]
    )

    try:
        run_from_args(args)
        capsys.readouterr()
        assert stat.S_IMODE(output_dir.stat().st_mode) == 0o555
        assert (output_dir / "selection.json").exists()
    finally:
        output_dir.chmod(0o755)


def test_save_filtered_refuses_directory_containing_resolved_source_member(
    sample_pareto_dir: Path,
    tmp_path: Path,
):
    output_dir = tmp_path / "filtered"
    output_dir.mkdir()
    resolved_source = output_dir / "linked.json"
    resolved_source.write_bytes((sample_pareto_dir / "balanced.json").read_bytes())
    (sample_pareto_dir / "linked.json").symlink_to(resolved_source)
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-l",
            "metric_a>0.8",
            "-f",
            str(output_dir),
            "--overwrite",
        ]
    )

    with pytest.raises(ValueError, match="resolved source Pareto member"):
        run_from_args(args)

    assert resolved_source.exists()


def test_save_filtered_preserves_symlink_member_name(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    resolved_source = external_dir / "balanced.json"
    resolved_source.write_bytes((sample_pareto_dir / "balanced.json").read_bytes())
    (sample_pareto_dir / "aliased_member.json").symlink_to(resolved_source)
    output_dir = tmp_path / "filtered"
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output_dir)]
    )

    run_from_args(args)
    capsys.readouterr()

    assert (output_dir / "aliased_member.json").read_bytes() == resolved_source.read_bytes()
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "a_extreme.json",
        "aliased_member.json",
        "b_extreme.json",
        "balanced.json",
        "c_extreme.json",
        "selection.json",
    ]
    manifest = json.loads((output_dir / "selection.json").read_text())
    member = next(
        entry for entry in manifest["members"] if entry["file"] == "aliased_member.json"
    )
    assert member["hash"] == "aliased_member"
    assert member["source_path"] == str(resolved_source.resolve())
    assert member["output_path"] == str((output_dir / "aliased_member.json").resolve())


def test_saved_outputs_use_candidate_snapshot_from_selection_time(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    source = sample_pareto_dir / "balanced.json"
    original_bytes = source.read_bytes()
    selected_output = tmp_path / "selected.json"
    filtered_output = tmp_path / "filtered"
    real_write_snapshot = pareto_explorer._write_candidate_snapshot_fd
    source_changed = False

    def change_source_before_export(candidate, destination, file_descriptor):
        nonlocal source_changed
        if not source_changed:
            source.write_text('{"changed": true}\n')
            source_changed = True
        return real_write_snapshot(candidate, destination, file_descriptor)

    monkeypatch.setattr(
        pareto_explorer, "_write_candidate_snapshot_fd", change_source_before_export
    )
    args = build_parser().parse_args(
        [
            str(source),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
        ]
    )

    run_from_args(args)
    capsys.readouterr()

    assert source.read_bytes() != original_bytes
    assert selected_output.read_bytes() == original_bytes
    assert (filtered_output / "balanced.json").read_bytes() == original_bytes


def test_save_filtered_refuses_manifest_filename_collision(
    sample_pareto_dir: Path,
    tmp_path: Path,
):
    _write_candidate(
        sample_pareto_dir,
        "Selection",
        {"metric_a": 0.5, "metric_b": 0.5, "metric_c": 0.5},
    )
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(tmp_path / "filtered")]
    )

    with pytest.raises(ValueError, match="reserved manifest name 'selection.json'"):
        run_from_args(args)


def test_save_outputs_refuse_source_pareto_directory(
    sample_pareto_dir: Path,
):
    selected_args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(sample_pareto_dir / "promoted.json"),
        ]
    )
    with pytest.raises(ValueError, match="inside the source Pareto directory"):
        run_from_args(selected_args)

    filtered_args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(sample_pareto_dir / "filtered")]
    )
    with pytest.raises(ValueError, match="inside the source Pareto directory"):
        run_from_args(filtered_args)


def test_save_selected_refuses_filesystem_identity_alias_of_source_member(
    sample_pareto_dir: Path,
    tmp_path: Path,
):
    source = sample_pareto_dir / "balanced.json"
    output_alias = tmp_path / "selected.json"
    os.link(source, output_alias)
    original_bytes = source.read_bytes()
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-s", str(output_alias), "--overwrite"]
    )

    with pytest.raises(ValueError, match="source Pareto member"):
        run_from_args(args)

    assert source.read_bytes() == original_bytes
    assert output_alias.read_bytes() == original_bytes


def test_save_filtered_refuses_filesystem_identity_alias_of_source_directory(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    output_alias = tmp_path / "source_alias"
    output_alias.mkdir()
    source_files = sorted(path.name for path in sample_pareto_dir.iterdir())
    real_samefile = os.path.samefile

    def report_bind_mount_alias(left, right):
        pair = {Path(left).resolve(), Path(right).resolve()}
        if pair == {output_alias.resolve(), sample_pareto_dir.resolve()}:
            return True
        return real_samefile(left, right)

    monkeypatch.setattr(os.path, "samefile", report_bind_mount_alias)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output_alias), "--overwrite"]
    )

    with pytest.raises(ValueError, match="inside the source Pareto directory"):
        run_from_args(args)

    assert sorted(path.name for path in sample_pareto_dir.iterdir()) == source_files
    assert not list(output_alias.iterdir())


@pytest.mark.parametrize("output_kind", ["selected", "filtered"])
def test_save_outputs_refuse_descendants_of_source_directory_alias(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
    output_kind: str,
):
    output_alias = tmp_path / "source-alias"
    output_alias.mkdir()
    if output_kind == "selected":
        output = output_alias / "notes.json"
        output.write_text('{"unrelated": true}\n')
        command = [str(sample_pareto_dir), "-s", str(output), "--overwrite"]
    else:
        output = output_alias / "nested"
        output.mkdir()
        (output / "notes.json").write_text('{"unrelated": true}\n')
        command = [str(sample_pareto_dir), "-f", str(output), "--overwrite"]
    real_samefile = os.path.samefile

    def report_bind_mount_alias(left, right):
        pair = {Path(left).resolve(), Path(right).resolve()}
        if pair == {output_alias.resolve(), sample_pareto_dir.resolve()}:
            return True
        return real_samefile(left, right)

    monkeypatch.setattr(os.path, "samefile", report_bind_mount_alias)
    args = build_parser().parse_args(command)

    with pytest.raises(ValueError, match="inside the source Pareto directory"):
        run_from_args(args)

    preserved = output / "notes.json" if output_kind == "filtered" else output
    assert json.loads(preserved.read_text()) == {"unrelated": True}


def test_single_file_input_allows_sibling_outputs(
    sample_pareto_dir: Path,
    capsys,
):
    source = sample_pareto_dir / "balanced.json"
    selected_output = sample_pareto_dir / "promoted.json"
    filtered_output = sample_pareto_dir / "single_filtered"
    args = build_parser().parse_args(
        [
            str(source),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
        ]
    )

    result = run_from_args(args)
    capsys.readouterr()

    assert result.candidate.path == source.resolve()
    assert selected_output.read_bytes() == source.read_bytes()
    assert sorted(path.name for path in filtered_output.iterdir()) == [
        "balanced.json",
        "selection.json",
    ]


def test_single_file_input_rejects_non_json_filtered_member_name(
    sample_pareto_dir: Path,
    tmp_path: Path,
):
    json_source = sample_pareto_dir / "balanced.json"
    source = tmp_path / "balanced.config"
    source.write_bytes(json_source.read_bytes())
    output_dir = tmp_path / "filtered"
    args = build_parser().parse_args([str(source), "-f", str(output_dir)])

    with pytest.raises(ValueError, match=r"must use \.json filenames"):
        run_from_args(args)

    assert source.exists()
    assert not output_dir.exists()


def test_single_file_input_rejects_filtered_parent_filesystem_alias(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    source = sample_pareto_dir / "balanced.json"
    output_alias = tmp_path / "source-parent-alias"
    output_alias.mkdir()
    real_samefile = os.path.samefile

    def report_bind_mount_alias(left, right):
        pair = {Path(left).resolve(), Path(right).resolve()}
        if pair == {output_alias.resolve(), sample_pareto_dir.resolve()}:
            return True
        return real_samefile(left, right)

    monkeypatch.setattr(os.path, "samefile", report_bind_mount_alias)
    args = build_parser().parse_args(
        [str(source), "-f", str(output_alias), "--overwrite"]
    )

    with pytest.raises(ValueError, match="resolved source Pareto member"):
        run_from_args(args)

    assert source.exists()
    assert not list(output_alias.iterdir())


def test_single_file_symlink_member_path_normalizes_dot_dot_without_dereferencing(
    tmp_path: Path,
):
    run_dir = tmp_path / "run"
    pareto_dir = run_dir / "pareto"
    pareto_dir.mkdir(parents=True)
    (run_dir / "other").mkdir()
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    resolved_source = external_dir / "target.json"
    _write_candidate(
        external_dir,
        "target",
        {"metric_a": 0.7, "metric_b": 0.7, "metric_c": 0.7},
    )
    member_path = pareto_dir / "member.json"
    member_path.symlink_to(resolved_source)
    spelled_path = run_dir / "other" / ".." / "pareto" / "member.json"
    args = build_parser().parse_args(
        [str(spelled_path), "-f", str(pareto_dir), "--overwrite"]
    )

    with pytest.raises(ValueError, match="resolved source Pareto member"):
        run_from_args(args)

    assert member_path.is_symlink()


def test_single_file_member_path_canonicalizes_symlinked_parent(
    tmp_path: Path,
):
    pareto_dir = tmp_path / "pareto"
    pareto_dir.mkdir()
    external_dir = tmp_path / "external"
    external_dir.mkdir()
    resolved_source = external_dir / "target.json"
    _write_candidate(
        external_dir,
        "target",
        {"metric_a": 0.7, "metric_b": 0.7, "metric_c": 0.7},
    )
    member_path = pareto_dir / "member.json"
    member_path.symlink_to(resolved_source)
    aliased_parent = tmp_path / "pareto_alias"
    aliased_parent.symlink_to(pareto_dir, target_is_directory=True)
    args = build_parser().parse_args(
        [
            str(aliased_parent / "member.json"),
            "-f",
            str(pareto_dir),
            "--overwrite",
        ]
    )

    with pytest.raises(ValueError, match="resolved source Pareto member"):
        run_from_args(args)

    assert member_path.is_symlink()


def test_save_outputs_refuse_both_overlap_directions(
    sample_pareto_dir: Path,
    tmp_path: Path,
):
    filtered_parent = tmp_path / "filtered"
    selected_inside_args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(filtered_parent / "selected.json"),
            "-f",
            str(filtered_parent),
        ]
    )
    with pytest.raises(ValueError, match="must not overlap"):
        run_from_args(selected_inside_args)

    selected_parent = tmp_path / "selected.json"
    filtered_inside_args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_parent),
            "-f",
            str(selected_parent / "filtered"),
        ]
    )
    with pytest.raises(ValueError, match="must not overlap"):
        run_from_args(filtered_inside_args)

    assert not filtered_parent.exists()
    assert not selected_parent.exists()


def test_save_outputs_refuse_filesystem_alias_overlap(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected_parent = tmp_path / "selected-parent"
    selected_parent.mkdir()
    selected_output = selected_parent / "selected.json"
    selected_output.write_text('{"selected": "old"}\n')
    filtered_alias = tmp_path / "filtered-alias"
    filtered_alias.mkdir()
    real_samefile = os.path.samefile

    def report_bind_mount_alias(left, right):
        pair = {Path(left).resolve(), Path(right).resolve()}
        if pair == {selected_parent.resolve(), filtered_alias.resolve()}:
            return True
        return real_samefile(left, right)

    monkeypatch.setattr(os.path, "samefile", report_bind_mount_alias)
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_alias),
            "--overwrite",
        ]
    )

    with pytest.raises(ValueError, match="must not overlap"):
        run_from_args(args)

    assert json.loads(selected_output.read_text()) == {"selected": "old"}
    assert not list(filtered_alias.iterdir())


def test_save_outputs_conservatively_normalize_aliased_uncreated_suffixes(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected_root = tmp_path / "selected-root"
    selected_root.mkdir()
    filtered_root = tmp_path / "filtered-root"
    filtered_root.mkdir()
    selected_output = selected_root / "New" / "selected.json"
    filtered_output = filtered_root / "new"
    real_samefile = os.path.samefile

    def report_bind_mount_alias(left, right):
        pair = {Path(left).resolve(), Path(right).resolve()}
        if pair == {selected_root.resolve(), filtered_root.resolve()}:
            return True
        return real_samefile(left, right)

    monkeypatch.setattr(os.path, "samefile", report_bind_mount_alias)
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
            "--overwrite",
        ]
    )

    with pytest.raises(ValueError, match="must not overlap"):
        run_from_args(args)

    assert not selected_output.exists()
    assert not filtered_output.exists()


def test_filtered_manifest_records_normalized_decision_inputs(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    utility_dir = tmp_path / "utility"
    utility_args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-m",
            "utility",
            "-o",
            "metric_b,metric_a",
            "--weight",
            "metric_b=5",
            "-f",
            str(utility_dir),
        ]
    )
    run_from_args(utility_args)
    capsys.readouterr()
    utility_manifest = json.loads((utility_dir / "selection.json").read_text())
    assert utility_manifest["method"] == "utility"
    assert utility_manifest["objectives"] == ["metric_b", "metric_a"]
    assert utility_manifest["weights"] == {
        "metric_a": pytest.approx(1 / 6),
        "metric_b": pytest.approx(5 / 6),
    }
    assert utility_manifest["targets"] == {}
    assert utility_manifest["priority"] == []

    reference_dir = tmp_path / "reference"
    reference_args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-m",
            "reference",
            "--target",
            "metric_a=0.75",
            "-f",
            str(reference_dir),
        ]
    )
    run_from_args(reference_args)
    capsys.readouterr()
    reference_manifest = json.loads((reference_dir / "selection.json").read_text())
    assert reference_manifest["method"] == "reference"
    assert reference_manifest["targets"] == {"metric_a": 0.75}

    priority_dir = tmp_path / "priority"
    priority_args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-m",
            "lexicographic",
            "--priority",
            "metric_c,metric_a",
            "-f",
            str(priority_dir),
        ]
    )
    run_from_args(priority_args)
    capsys.readouterr()
    priority_manifest = json.loads((priority_dir / "selection.json").read_text())
    assert priority_manifest["method"] == "lexicographic"
    assert priority_manifest["objectives"] == ["metric_c", "metric_a"]
    assert priority_manifest["priority"] == ["metric_c", "metric_a"]


def test_save_filtered_with_scenario_exports_post_limit_set(
    scenario_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    output_dir = tmp_path / "bull_filtered"
    args = build_parser().parse_args(
        [
            str(scenario_pareto_dir),
            "--scenario",
            "bull",
            "-f",
            str(output_dir),
        ]
    )

    run_from_args(args)
    capsys.readouterr()
    manifest = json.loads((output_dir / "selection.json").read_text())

    assert manifest["scenario"] == "bull"
    assert manifest["retained_count"] == 4
    assert sorted(member["file"] for member in manifest["members"]) == [
        "a.json",
        "b.json",
        "c_dominated.json",
        "d.json",
    ]
    assert (output_dir / "c_dominated.json").exists()


def test_no_saved_output_is_written_when_limits_reject_all(
    sample_pareto_dir: Path,
    tmp_path: Path,
):
    selected_output = tmp_path / "selected.json"
    filtered_output = tmp_path / "filtered"
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-l",
            "metric_a>2",
            "-s",
            str(selected_output),
            "-f",
            str(filtered_output),
        ]
    )

    with pytest.raises(ValueError, match="No Pareto candidates remained"):
        run_from_args(args)

    assert not selected_output.exists()
    assert not filtered_output.exists()


def test_select_candidate_accepts_non_scoring_metric_from_stats(sample_pareto_dir: Path):
    for entry_path, sharpe in {
        "a_extreme.json": 0.2,
        "b_extreme.json": 0.3,
        "c_extreme.json": 0.4,
        "balanced.json": 1.4,
    }.items():
        path = sample_pareto_dir / entry_path
        payload = json.loads(path.read_text())
        payload["metrics"]["stats"]["sharpe_ratio_strategy_pnl_rebased"] = _metric_stats(sharpe)
        path.write_text(json.dumps(payload, indent=2))

    _pareto_dir, candidates, specs = load_candidates(sample_pareto_dir)
    result = select_candidate(
        candidates,
        specs,
        method="utility",
        objectives_arg="sharpe_ratio_strategy_pnl_rebased,metric_a,metric_b",
        weight_pairs=["sharpe_ratio_strategy_pnl_rebased=5", "metric_a=1", "metric_b=1"],
    )
    assert result.candidate.path.stem == "balanced"
    assert "sharpe_ratio_strategy_pnl_rebased" in result.objective_values
    assert result.objective_values["sharpe_ratio_strategy_pnl_rebased"] == pytest.approx(1.4)


def test_select_candidate_accepts_non_scoring_fill_metric_from_suite_metrics(tmp_path: Path):
    pareto_dir = tmp_path / "run" / "pareto"
    pareto_dir.mkdir(parents=True)
    _write_fill_suite_candidate(
        pareto_dir,
        "low_p95_gap",
        adg=0.01,
        p95_gap=12.0,
        p99_gap=80.0,
    )
    _write_fill_suite_candidate(
        pareto_dir,
        "high_p95_gap",
        adg=0.01,
        p95_gap=48.0,
        p99_gap=80.0,
    )

    _pareto_dir, candidates, specs = load_candidates(pareto_dir)
    result = select_candidate(
        candidates,
        specs,
        method="ideal",
        objectives_arg="adg_strategy_eq,fills_gap_p95_hours",
    )

    assert result.candidate.path.stem == "low_p95_gap"
    assert result.objective_values["fills_gap_p95_hours"] == pytest.approx(12.0)


def test_run_from_args_formats_goal_for_non_scoring_metric(sample_pareto_dir: Path, capsys):
    for entry_path, sharpe in {
        "a_extreme.json": 0.2,
        "b_extreme.json": 0.3,
        "c_extreme.json": 0.4,
        "balanced.json": 1.4,
    }.items():
        path = sample_pareto_dir / entry_path
        payload = json.loads(path.read_text())
        payload["metrics"]["stats"]["sharpe_ratio_strategy_pnl_rebased"] = _metric_stats(sharpe)
        path.write_text(json.dumps(payload, indent=2))

    args = argparse.Namespace(
        path=str(sample_pareto_dir),
        method="utility",
        limit_entries=[],
        limits_payload=None,
        objectives="sharpe_ratio_strategy_pnl_rebased,metric_a,metric_b",
        weight=["sharpe_ratio_strategy_pnl_rebased=5", "metric_a=1", "metric_b=1"],
        target=None,
        priority=None,
        show_top=1,
        json_output=False,
    )
    run_from_args(args)
    captured = capsys.readouterr().out
    assert "sharpe_ratio_strategy_pnl_rebased" in captured
    assert "| max  | 1.400" in captured


def test_run_from_args_ideal_uses_distance_label_and_omits_hash(sample_pareto_dir: Path, capsys):
    args = argparse.Namespace(
        path=str(sample_pareto_dir),
        method="ideal",
        limit_entries=[],
        limits_payload=None,
        objectives=None,
        weight=None,
        target=None,
        priority=None,
        show_top=1,
        json_output=False,
    )
    run_from_args(args)
    captured = capsys.readouterr().out
    assert "| Distance" in captured
    assert "| Score" not in captured
    assert "| Selected hash" not in captured
    assert "Backtest command: passivbot backtest" in captured


def test_build_parser_accepts_short_objectives_alias():
    parser = build_parser()
    args = parser.parse_args(["-o", "metric_a,metric_b"])
    assert args.objectives == "metric_a,metric_b"
