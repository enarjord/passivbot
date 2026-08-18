from __future__ import annotations

import argparse
import json
import os
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


def test_build_parser_accepts_save_outputs():
    args = build_parser().parse_args(
        ["-s", "selected.json", "-f", "filtered", "--overwrite"]
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
            "--save-selected",
            str(output),
        ]
    )

    result = run_from_args(args)

    assert result.candidate.path.name == "b_extreme.json"
    assert output.read_bytes() == result.candidate.path.read_bytes()
    assert "Saved selected member:" in capsys.readouterr().out


def test_new_selected_output_does_not_require_hard_links(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    output = tmp_path / "selected.json"
    monkeypatch.setattr(
        pareto_explorer.os,
        "link",
        lambda *_args: (_ for _ in ()).throw(OSError("hard links unavailable")),
    )

    result = run_from_args(
        build_parser().parse_args([str(sample_pareto_dir), "-s", str(output)])
    )
    capsys.readouterr()
    assert output.read_bytes() == result.candidate.path.read_bytes()


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode semantics")
def test_selected_output_preserves_source_or_existing_mode(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    source = sample_pareto_dir / "balanced.json"
    output = tmp_path / "selected.json"
    source.chmod(0o640)

    run_from_args(
        build_parser().parse_args([str(sample_pareto_dir), "-s", str(output)])
    )
    capsys.readouterr()
    assert output.stat().st_mode & 0o777 == 0o640

    output.chmod(0o604)
    source.chmod(0o600)
    run_from_args(
        build_parser().parse_args(
            [str(sample_pareto_dir), "-s", str(output), "--overwrite"]
        )
    )
    capsys.readouterr()
    assert output.stat().st_mode & 0o777 == 0o604


def test_run_from_args_saves_post_limit_members_and_manifest(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    output = tmp_path / "filtered"
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-l", "metric_a>0.6", "-f", str(output)]
    )

    result = run_from_args(args)

    assert result.candidate.path.name == "balanced.json"
    assert sorted(path.name for path in output.iterdir()) == [
        "a_extreme.json",
        "balanced.json",
        "selection.json",
    ]
    for name in ["a_extreme.json", "balanced.json"]:
        assert (output / name).read_bytes() == (sample_pareto_dir / name).read_bytes()
    manifest = json.loads((output / "selection.json").read_text())
    assert manifest["loaded_count"] == 4
    assert manifest["retained_count"] == 2
    assert manifest["selected_member"] == "balanced.json"
    assert manifest["members"] == ["a_extreme.json", "balanced.json"]
    assert manifest["applied_limits"][0]["metric"] == "metric_a"
    assert "Saved filtered members: 2" in capsys.readouterr().out


def test_filtered_export_detects_destination_filesystem_name_collision(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    original_exists = Path.exists

    def collision_exists(path: Path) -> bool:
        if path.name == "b_extreme.json" and path.parent.name.startswith(
            ".pareto-filtered-"
        ):
            return True
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", collision_exists)
    output = tmp_path / "filtered"
    args = build_parser().parse_args([str(sample_pareto_dir), "-f", str(output)])

    with pytest.raises(ValueError, match="collides on the destination filesystem"):
        run_from_args(args)
    assert not output.exists()


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode semantics")
def test_filtered_output_preserves_source_or_existing_modes(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    output = tmp_path / "filtered"
    sample_pareto_dir.chmod(0o751)
    (sample_pareto_dir / "balanced.json").chmod(0o640)

    run_from_args(
        build_parser().parse_args([str(sample_pareto_dir), "-f", str(output)])
    )
    capsys.readouterr()
    assert output.stat().st_mode & 0o777 == 0o751
    assert (output / "balanced.json").stat().st_mode & 0o777 == 0o640

    output.chmod(0o750)
    sample_pareto_dir.chmod(0o700)
    run_from_args(
        build_parser().parse_args(
            [str(sample_pareto_dir), "-f", str(output), "--overwrite"]
        )
    )
    capsys.readouterr()
    assert output.stat().st_mode & 0o777 == 0o750


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode semantics")
def test_filtered_overwrite_removes_read_only_backup(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    output = tmp_path / "filtered"
    output.mkdir()
    old_output = output / "old.json"
    old_output.write_text('{"old": true}\n')
    old_output.chmod(0o444)
    output.chmod(0o555)

    run_from_args(
        build_parser().parse_args(
            [str(sample_pareto_dir), "-f", str(output), "--overwrite"]
        )
    )
    capsys.readouterr()

    assert output.stat().st_mode & 0o777 == 0o555
    assert not list(tmp_path.glob(".pareto-backup-*"))


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode semantics")
def test_read_only_filtered_stage_remains_cleanupable(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    sample_pareto_dir.chmod(0o555)
    selected = tmp_path / "selected.json"
    filtered = tmp_path / "filtered"

    def fail_selected_install(*_args, **_kwargs):
        raise OSError("simulated selected install failure")

    monkeypatch.setattr(pareto_explorer, "_install_selected", fail_selected_install)
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected),
            "-f",
            str(filtered),
        ]
    )

    with pytest.raises(OSError, match="simulated selected install failure"):
        run_from_args(args)
    assert not list(tmp_path.glob(".pareto-filtered-*"))


def test_exports_support_long_destination_names(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    selected = tmp_path / (("s" * 245) + ".json")
    run_from_args(
        build_parser().parse_args([str(sample_pareto_dir), "-s", str(selected)])
    )
    capsys.readouterr()
    assert selected.is_file()

    output = tmp_path / ("f" * 250)
    output.mkdir()
    (output / "old.json").write_text('{"old": true}\n')

    run_from_args(
        build_parser().parse_args(
            [str(sample_pareto_dir), "-f", str(output), "--overwrite"]
        )
    )
    capsys.readouterr()
    assert (output / "selection.json").is_file()
    assert not list(tmp_path.glob(".pareto-backup-*"))


def test_saved_outputs_are_reported_in_json(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    selected = tmp_path / "selected.json"
    filtered = tmp_path / "filtered"
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-l",
            "metric_a>0.6",
            "-s",
            str(selected),
            "-f",
            str(filtered),
            "--json",
        ]
    )

    run_from_args(args)
    payload = json.loads(capsys.readouterr().out)

    assert payload["selected"]["saved_path"] == str(selected.resolve())
    assert payload["saved_filtered"] == {
        "count": 2,
        "directory": str(filtered.resolve()),
        "manifest": str((filtered / "selection.json").resolve()),
    }


def test_saved_outputs_require_explicit_overwrite(
    sample_pareto_dir: Path,
    tmp_path: Path,
    capsys,
):
    selected = tmp_path / "selected.json"
    selected.write_text('{"old": true}\n')
    args = build_parser().parse_args([str(sample_pareto_dir), "-s", str(selected)])
    with pytest.raises(FileExistsError, match="use --overwrite"):
        run_from_args(args)
    assert json.loads(selected.read_text()) == {"old": True}

    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-s", str(selected), "--overwrite"]
    )
    result = run_from_args(args)
    capsys.readouterr()
    assert selected.read_bytes() == result.candidate.path.read_bytes()

    filtered = tmp_path / "filtered"
    filtered.mkdir()
    stale = filtered / "stale.json"
    stale.write_text('{"stale": true}\n')
    args = build_parser().parse_args([str(sample_pareto_dir), "-f", str(filtered)])
    with pytest.raises(FileExistsError, match="use --overwrite"):
        run_from_args(args)
    assert stale.exists()

    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-l",
            "metric_a>0.6",
            "-f",
            str(filtered),
            "--overwrite",
        ]
    )
    run_from_args(args)
    capsys.readouterr()
    assert not stale.exists()
    assert sorted(path.name for path in filtered.iterdir()) == [
        "a_extreme.json",
        "balanced.json",
        "selection.json",
    ]


def test_filtered_overwrite_refuses_non_json_entries(
    sample_pareto_dir: Path,
    tmp_path: Path,
):
    output = tmp_path / "filtered"
    output.mkdir()
    note = output / "notes.txt"
    note.write_text("keep me\n")
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output), "--overwrite"]
    )

    with pytest.raises(FileExistsError, match="non-JSON entries: notes.txt"):
        run_from_args(args)
    assert note.read_text() == "keep me\n"


def test_filtered_overwrite_refuses_current_working_directory(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    output = tmp_path / "filtered"
    output.mkdir()
    existing = output / "old.json"
    existing.write_text('{"old": true}\n')
    monkeypatch.chdir(output)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", ".", "--overwrite"]
    )

    with pytest.raises(ValueError, match="current working directory"):
        run_from_args(args)
    assert json.loads(existing.read_text()) == {"old": True}


def test_existing_path_identity_detects_directory_alias(
    tmp_path: Path,
):
    directory = tmp_path / "directory"
    directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)

    assert pareto_explorer._same_existing_path(alias, directory)


def test_filtered_overwrite_refuses_json_symlink(
    sample_pareto_dir: Path,
    tmp_path: Path,
):
    output = tmp_path / "filtered"
    output.mkdir()
    external = tmp_path / "external.json"
    external.write_text('{"external": true}\n')
    linked = output / "linked.json"
    linked.symlink_to(external)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output), "--overwrite"]
    )

    with pytest.raises(FileExistsError, match="non-JSON entries: linked.json"):
        run_from_args(args)
    assert json.loads(external.read_text()) == {"external": True}


def test_save_outputs_refuse_source_overlap(
    sample_pareto_dir: Path,
):
    selected_args = build_parser().parse_args(
        [str(sample_pareto_dir), "-s", str(sample_pareto_dir / "selected.json")]
    )
    with pytest.raises(ValueError, match="must not overlap"):
        run_from_args(selected_args)

    filtered_args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(sample_pareto_dir.parent)]
    )
    with pytest.raises(ValueError, match="must not overlap"):
        run_from_args(filtered_args)


def test_identity_overlap_detects_existing_directory_alias(
    sample_pareto_dir: Path,
    tmp_path: Path,
):
    alias = tmp_path / "pareto_alias"
    alias.symlink_to(sample_pareto_dir, target_is_directory=True)

    assert pareto_explorer._is_within_by_identity(
        alias / "filtered",
        sample_pareto_dir,
    )


def test_combined_outputs_refuse_overlap_in_either_direction(
    sample_pareto_dir: Path,
    tmp_path: Path,
):
    filtered = tmp_path / "filtered"
    selected_inside = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(filtered / "selected.json"),
            "-f",
            str(filtered),
        ]
    )
    with pytest.raises(ValueError, match="must not overlap"):
        run_from_args(selected_inside)

    selected = tmp_path / "selected.json"
    filtered_inside = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected),
            "-f",
            str(selected / "filtered"),
        ]
    )
    with pytest.raises(ValueError, match="must not overlap"):
        run_from_args(filtered_inside)
    assert not selected.exists()


def test_combined_outputs_use_identity_overlap_check(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    monkeypatch.setattr(pareto_explorer, "_is_within", lambda *_args: False)
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(filtered / "selected.json"),
            "-f",
            str(filtered),
            "--overwrite",
        ]
    )

    with pytest.raises(ValueError, match="must not overlap"):
        run_from_args(args)


def test_filtered_copy_failure_preserves_existing_output(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    output = tmp_path / "filtered"
    output.mkdir()
    existing = output / "old.json"
    existing.write_text('{"old": true}\n')

    def fail_copy(_source, _destination):
        raise OSError("simulated copy failure")

    monkeypatch.setattr(pareto_explorer.shutil, "copyfile", fail_copy)
    args = build_parser().parse_args(
        [str(sample_pareto_dir), "-f", str(output), "--overwrite"]
    )

    with pytest.raises(OSError, match="simulated copy failure"):
        run_from_args(args)
    assert json.loads(existing.read_text()) == {"old": True}
    assert sorted(path.name for path in tmp_path.iterdir()) == ["filtered", "run"]


def test_combined_copy_failure_happens_before_either_output_is_installed(
    sample_pareto_dir: Path,
    tmp_path: Path,
    monkeypatch,
):
    selected = tmp_path / "selected.json"
    filtered = tmp_path / "filtered"
    (sample_pareto_dir / "balanced.json").chmod(0o444)
    original_copyfile = pareto_explorer.shutil.copyfile
    call_count = 0

    def fail_second_copy(source, destination):
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            raise OSError("simulated filtered copy failure")
        return original_copyfile(source, destination)

    monkeypatch.setattr(pareto_explorer.shutil, "copyfile", fail_second_copy)
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-s",
            str(selected),
            "-f",
            str(filtered),
        ]
    )

    with pytest.raises(OSError, match="simulated filtered copy failure"):
        run_from_args(args)
    assert not selected.exists()
    assert not filtered.exists()
    assert not list(tmp_path.glob(".pareto-selected-*"))


def test_no_output_is_written_when_limits_reject_every_candidate(
    sample_pareto_dir: Path,
    tmp_path: Path,
):
    selected = tmp_path / "selected.json"
    filtered = tmp_path / "filtered"
    args = build_parser().parse_args(
        [
            str(sample_pareto_dir),
            "-l",
            "metric_a>2",
            "-s",
            str(selected),
            "-f",
            str(filtered),
        ]
    )

    with pytest.raises(ValueError, match="No Pareto candidates remained"):
        run_from_args(args)
    assert not selected.exists()
    assert not filtered.exists()


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
