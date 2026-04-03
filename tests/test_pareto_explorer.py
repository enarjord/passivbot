from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytest

from pareto_explorer import (
    detect_latest_pareto_dir,
    filter_candidates,
    load_candidates,
    run_from_args,
    select_candidate,
)


def _write_candidate(
    path: Path,
    name: str,
    objectives: dict[str, float],
    *,
    extra_stats: dict[str, float] | None = None,
) -> None:
    stats = {
        "metric_a": {"mean": objectives["metric_a"]},
        "metric_b": {"mean": objectives["metric_b"]},
        "metric_c": {"mean": objectives["metric_c"]},
    }
    for metric, value in (extra_stats or {}).items():
        stats[metric] = {"mean": value}
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


def test_detect_latest_pareto_dir_selects_newest(tmp_path: Path):
    older = tmp_path / "optimize_results" / "older" / "pareto"
    newer = tmp_path / "optimize_results" / "newer" / "pareto"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    older.touch()
    newer.touch()
    older.parent.touch()
    newer.parent.touch()
    older_dir = older.resolve()
    newer_dir = newer.resolve()
    os.utime(older_dir, (1000, 1000))
    os.utime(newer_dir, (2000, 2000))

    resolved = detect_latest_pareto_dir(tmp_path / "optimize_results")
    assert resolved == newer_dir


def test_filter_candidates_with_cli_keep_condition(sample_pareto_dir: Path):
    _pareto_dir, candidates, _specs = load_candidates(sample_pareto_dir)
    filtered, limits = filter_candidates(
        candidates,
        limits_payload=None,
        limit_entries=["metric_a>0.6"],
    )
    assert len(limits) == 1
    assert sorted(candidate.path.stem for candidate in filtered) == ["a_extreme", "balanced"]


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


def test_run_from_args_prints_summary(sample_pareto_dir: Path, capsys):
    args = argparse.Namespace(
        path=str(sample_pareto_dir),
        method="knee",
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
    assert "Loaded candidates: 4" in captured
    assert "Retained after limits: 4" in captured
    assert "Applied limits: 0" in captured
    assert "Method: knee" in captured
    assert "Method summary:" in captured
    assert "Selected file:" in captured
    assert "Selected hash:" in captured
    assert "Active objectives:" in captured
    assert "Why this winner:" in captured
    assert "Selected normalized utilities:" in captured
    assert "Top candidates:" in captured
    assert result.candidate.path.stem == "balanced"


def test_run_from_args_uses_latest_pareto_dir_when_path_omitted(tmp_path: Path, monkeypatch, capsys):
    root = tmp_path / "optimize_results"
    older = root / "older" / "pareto"
    newer = root / "newer" / "pareto"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    _write_candidate(older, "older_balanced", {"metric_a": 0.6, "metric_b": 0.6, "metric_c": 0.6})
    _write_candidate(newer, "newer_balanced", {"metric_a": 0.65, "metric_b": 0.65, "metric_c": 0.65})
    os.utime(older.resolve(), (1000, 1000))
    os.utime(newer.resolve(), (2000, 2000))
    monkeypatch.chdir(tmp_path)

    args = argparse.Namespace(
        path=None,
        method="knee",
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
    assert str(newer.resolve()) in captured
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
        payload["metrics"]["stats"]["sharpe_ratio_strategy_pnl_rebased"] = {"mean": sharpe}
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


def test_run_from_args_formats_goal_for_non_scoring_metric(sample_pareto_dir: Path, capsys):
    for entry_path, sharpe in {
        "a_extreme.json": 0.2,
        "b_extreme.json": 0.3,
        "c_extreme.json": 0.4,
        "balanced.json": 1.4,
    }.items():
        path = sample_pareto_dir / entry_path
        payload = json.loads(path.read_text())
        payload["metrics"]["stats"]["sharpe_ratio_strategy_pnl_rebased"] = {"mean": sharpe}
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
    assert "sharpe_ratio_strategy_pnl_rebased (max): 1.4" in captured
