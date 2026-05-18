from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from pareto_compress import build_parser, compress_candidates, compress_from_args
from pareto_explorer import load_candidates


def _write_candidate(
    pareto_dir: Path,
    name: str,
    *,
    adg: float,
    drawdown: float,
) -> None:
    payload = {
        "config_version": "v8.0.0",
        "backtest": {"aggregate": {"default": "mean"}, "starting_balance": 100000},
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
                "drawdown_worst_strategy_eq": drawdown,
            },
            "stats": {
                "adg_strategy_eq": {"mean": adg},
                "drawdown_worst_strategy_eq": {"mean": drawdown},
            },
        },
    }
    with (pareto_dir / f"{name}.json").open("w") as f:
        json.dump(payload, f)


@pytest.fixture()
def sample_pareto_dir(tmp_path: Path) -> Path:
    pareto_dir = tmp_path / "run" / "pareto"
    pareto_dir.mkdir(parents=True)
    _write_candidate(pareto_dir, "low_drawdown_anchor", adg=0.001, drawdown=0.10)
    _write_candidate(pareto_dir, "adg_anchor", adg=0.004, drawdown=0.40)
    _write_candidate(pareto_dir, "middle", adg=0.002, drawdown=0.25)
    _write_candidate(pareto_dir, "diverse_fill", adg=0.003, drawdown=0.15)
    return pareto_dir


def test_compress_includes_ideal_then_objective_anchors(sample_pareto_dir: Path):
    _pareto_dir, candidates, scoring_specs = load_candidates(sample_pareto_dir)

    members, objective_ranges, truncated = compress_candidates(candidates, scoring_specs, count=3)

    selected_files = [member.candidate.path.name for member in members]
    assert selected_files == ["diverse_fill.json", "adg_anchor.json", "low_drawdown_anchor.json"]
    assert [member.reason for member in members] == [
        "ideal_anchor",
        "objective_anchor",
        "objective_anchor",
    ]
    assert members[0].reason_details == ["closest to ideal point"]
    assert {item["metric"] for item in objective_ranges} == {
        "adg_strategy_eq",
        "drawdown_worst_strategy_eq",
    }
    assert truncated == []


def test_compress_fills_remaining_slots_by_diversity(sample_pareto_dir: Path):
    _pareto_dir, candidates, scoring_specs = load_candidates(sample_pareto_dir)

    members, _objective_ranges, truncated = compress_candidates(candidates, scoring_specs, count=4)

    assert [member.candidate.path.name for member in members] == [
        "diverse_fill.json",
        "adg_anchor.json",
        "low_drawdown_anchor.json",
        "middle.json",
    ]
    assert members[-1].reason == "diversity_fill"
    assert truncated == []


def test_compress_truncates_to_ideal_when_count_is_small(sample_pareto_dir: Path):
    _pareto_dir, candidates, scoring_specs = load_candidates(sample_pareto_dir)

    members, _objective_ranges, truncated = compress_candidates(candidates, scoring_specs, count=1)

    assert [member.candidate.path.name for member in members] == ["diverse_fill.json"]
    assert members[0].reason == "ideal_anchor"
    assert truncated == ["best adg_strategy_eq", "best drawdown_worst_strategy_eq"]


def test_compress_from_args_writes_selected_json_and_manifest(sample_pareto_dir: Path, tmp_path: Path):
    out_dir = tmp_path / "compressed"
    parser = build_parser()
    args = parser.parse_args([str(sample_pareto_dir.parent), "2", "--output-dir", str(out_dir)])

    payload = compress_from_args(args)

    assert payload["loaded_count"] == 4
    assert payload["selected_count"] == 2
    assert (out_dir / "diverse_fill.json").exists()
    assert (out_dir / "adg_anchor.json").exists()
    manifest = json.loads((out_dir / "selection.json").read_text())
    assert manifest["selected_count"] == 2
    assert [item["reason"] for item in manifest["selected"]] == ["ideal_anchor", "objective_anchor"]
    assert manifest["selected"][0]["output_path"] == str(out_dir / "diverse_fill.json")


def test_compress_writes_to_non_empty_output_dir_without_deleting_files(
    sample_pareto_dir: Path,
    tmp_path: Path,
    caplog,
):
    caplog.set_level(logging.INFO)
    out_dir = tmp_path / "compressed"
    out_dir.mkdir()
    stale_file = out_dir / "stale.json"
    stale_file.write_text('{"keep": true}\n')
    selected_file = out_dir / "diverse_fill.json"
    selected_file.write_text('{"old": true}\n')
    (out_dir / "selection.json").write_text('{"old_manifest": true}\n')
    parser = build_parser()
    args = parser.parse_args([str(sample_pareto_dir), "2", "--output-dir", str(out_dir)])

    payload = compress_from_args(args)

    assert payload["selected_count"] == 2
    assert json.loads(stale_file.read_text()) == {"keep": True}
    selected_payload = json.loads(selected_file.read_text())
    assert selected_payload["metrics"]["objectives"]["adg_strategy_eq"] == 0.003
    manifest = json.loads((out_dir / "selection.json").read_text())
    assert manifest["selected_count"] == 2
    assert "Output directory is non-empty" in caplog.text
    assert "Overwriting existing output file: diverse_fill.json" in caplog.text
    assert "Overwriting existing output file: selection.json" in caplog.text


def test_compress_applies_limits_before_selection(sample_pareto_dir: Path):
    parser = build_parser()
    args = parser.parse_args(
        [
            str(sample_pareto_dir),
            "2",
            "--limit",
            "drawdown_worst_strategy_eq<=0.30",
        ]
    )

    payload = compress_from_args(args)

    assert payload["retained_count"] == 3
    assert payload["selected_count"] == 2
    assert "adg_anchor.json" not in [item["file"] for item in payload["selected"]]


def test_compress_defaults_to_latest_optimize_results(monkeypatch, sample_pareto_dir: Path):
    parser = build_parser()
    args = parser.parse_args(["2"])
    monkeypatch.setattr("pareto_compress.detect_latest_pareto_dir", lambda: sample_pareto_dir)

    payload = compress_from_args(args)

    assert payload["selected_count"] == 2
    assert payload["pareto_dir"] == str(sample_pareto_dir.resolve())
