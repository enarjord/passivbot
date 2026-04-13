import json

import pytest

from tools.score_ema_anchor_autoresearch import load_analysis
from tools.score_ema_anchor_autoresearch import resolve_analysis_path
from tools.score_ema_anchor_autoresearch import score_analysis
from tools.score_ema_anchor_autoresearch import score_suite


def test_score_analysis_passes_reasonable_candidate():
    analysis = {
        "adg_strategy_pnl_rebased": 0.002,
        "drawdown_worst_hsl": 0.12,
        "peak_recovery_hours_hsl": 48.0,
        "fills_per_day": 1.5,
        "hours_no_fills_max": 12.0,
        "hours_no_fills_mean": 4.0,
        "hours_no_fills_median": 2.0,
        "liquidated": False,
    }

    scored = score_analysis(analysis)

    assert scored["passed"] is True
    assert scored["score"] > 0.0
    assert scored["violations"] == []


def test_score_analysis_fails_liquidated_candidate():
    analysis = {
        "adg_strategy_pnl_rebased": 0.003,
        "drawdown_worst_hsl": 0.2,
        "peak_recovery_hours_hsl": 24.0,
        "fills_per_day": 2.0,
        "hours_no_fills_max": 8.0,
        "hours_no_fills_mean": 3.0,
        "hours_no_fills_median": 2.0,
        "liquidated": True,
    }

    scored = score_analysis(analysis)

    assert scored["passed"] is False
    assert scored["score"] < 0.0
    assert any(v["metric"] == "liquidated" for v in scored["violations"])


def test_score_analysis_fails_fill_gap_violation():
    analysis = {
        "adg_strategy_pnl_rebased": 0.002,
        "drawdown_worst_hsl": 0.1,
        "peak_recovery_hours_hsl": 36.0,
        "fills_per_day": 0.4,
        "hours_no_fills_max": 240.0,
        "hours_no_fills_mean": 5.0,
        "hours_no_fills_median": 2.0,
        "liquidated": False,
    }

    scored = score_analysis(analysis)

    assert scored["passed"] is False
    assert any(v["metric"] == "hours_no_fills_max" for v in scored["violations"])


def test_resolve_and_load_analysis_from_artifact_dir(tmp_path):
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    analysis_path = artifact_dir / "analysis.json"
    payload = {
        "adg_strategy_pnl_rebased": 0.001,
        "drawdown_worst_hsl": 0.1,
        "peak_recovery_hours_hsl": 10.0,
        "fills_per_day": 1.0,
        "hours_no_fills_max": 2.0,
        "hours_no_fills_mean": 1.0,
        "hours_no_fills_median": 1.0,
        "liquidated": False,
    }
    analysis_path.write_text(json.dumps(payload), encoding="utf-8")

    assert resolve_analysis_path(artifact_dir) == analysis_path
    assert load_analysis(artifact_dir) == payload


def test_score_suite_aggregates_multiple_runs(tmp_path):
    paths = []
    for idx, adg in enumerate((0.0015, 0.0025), start=1):
        artifact_dir = tmp_path / f"artifact_{idx}"
        artifact_dir.mkdir()
        analysis_path = artifact_dir / "analysis.json"
        analysis_path.write_text(
            json.dumps(
                {
                    "adg_strategy_pnl_rebased": adg,
                    "drawdown_worst_hsl": 0.1,
                    "peak_recovery_hours_hsl": 12.0,
                    "fills_per_day": 1.2,
                    "hours_no_fills_max": 6.0,
                    "hours_no_fills_mean": 2.0,
                    "hours_no_fills_median": 1.0,
                    "liquidated": False,
                }
            ),
            encoding="utf-8",
        )
        paths.append(artifact_dir)

    scored = score_suite(paths)

    assert scored["suite_passed"] is True
    assert len(scored["results"]) == 2
    assert scored["suite_score"] == pytest.approx(
        sum(result["score"] for result in scored["results"]) / 2.0
    )


def test_score_analysis_accepts_optimize_stats_payload():
    optimize_member = {
        "metrics": {
            "stats": {
                "adg_strategy_pnl_rebased": {"mean": 0.002},
                "drawdown_worst_hsl": {"mean": 0.12},
                "peak_recovery_hours_hsl": {"mean": 24.0},
                "fills_per_day": {"mean": 1.3},
                "hours_no_fills_max": {"mean": 10.0},
                "hours_no_fills_mean": {"mean": 4.0},
                "hours_no_fills_median": {"mean": 2.0},
                "liquidated": False,
            }
        }
    }

    scored = score_analysis(optimize_member)

    assert scored["passed"] is True
    assert scored["score"] > 0.0
