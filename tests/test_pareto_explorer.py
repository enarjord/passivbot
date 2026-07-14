from __future__ import annotations

import argparse
import json
import os
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


def test_scenario_limit_uses_scalar_and_rejects_non_mean_stat(
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

    with pytest.raises(ValueError, match="stores one mean value.*stat='max'.*unavailable"):
        filter_candidates(
            projected,
            limits_payload=None,
            limit_entries=["metric_a>0.75 stat=max"],
        )


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
