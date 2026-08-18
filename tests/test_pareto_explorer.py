from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pytest

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


def test_filtered_overwrite_preserves_previous_set_when_staging_fails(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    output_dir = tmp_path / "filtered"
    output_dir.mkdir()
    stale = output_dir / "stale.json"
    stale.write_text('{"stale": true}\n')
    real_copy2 = shutil.copy2
    copies = 0

    def fail_on_second_copy(source, destination, *args, **kwargs):
        nonlocal copies
        copies += 1
        if copies == 2:
            raise OSError("simulated copy failure")
        return real_copy2(source, destination, *args, **kwargs)

    monkeypatch.setattr("pareto_explorer.shutil.copy2", fail_on_second_copy)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output_dir), "--overwrite"]
    )

    with pytest.raises(OSError, match="simulated copy failure"):
        run_from_args(args)

    assert json.loads(stale.read_text()) == {"stale": True}
    assert sorted(path.name for path in output_dir.iterdir()) == ["stale.json"]
    assert not [path for path in tmp_path.iterdir() if path.name.startswith(".filtered.")]


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
