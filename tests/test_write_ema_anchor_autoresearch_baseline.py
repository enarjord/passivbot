import json

import pytest

from tools.write_ema_anchor_autoresearch_baseline import build_baseline_manifest
from tools.write_ema_anchor_autoresearch_baseline import resolve_baseline_dirs
from tools.write_ema_anchor_autoresearch_baseline import write_baseline_manifest


def _write_member(path, *, adg, drawdown, recovery, fills_per_day, gap_max, gap_mean, gap_median):
    payload = {
        "metrics": {
            "stats": {
                "adg_strategy_pnl_rebased": {"mean": adg},
                "drawdown_worst_hsl": {"mean": drawdown},
                "peak_recovery_hours_hsl": {"mean": recovery},
                "fills_per_day": {"mean": fills_per_day},
                "hours_no_fills_max": {"mean": gap_max},
                "hours_no_fills_mean": {"mean": gap_mean},
                "hours_no_fills_median": {"mean": gap_median},
                "liquidated": False,
            }
        }
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_baseline_dirs_accepts_run_dir_and_pareto_dir(tmp_path):
    run_dir = tmp_path / "run"
    pareto_dir = run_dir / "pareto"
    pareto_dir.mkdir(parents=True)

    resolved_run, resolved_pareto = resolve_baseline_dirs(run_dir)
    assert resolved_run == run_dir
    assert resolved_pareto == pareto_dir

    resolved_run, resolved_pareto = resolve_baseline_dirs(pareto_dir)
    assert resolved_run == run_dir
    assert resolved_pareto == pareto_dir


def test_build_baseline_manifest_picks_best_member(tmp_path):
    run_dir = tmp_path / "run"
    pareto_dir = run_dir / "pareto"
    pareto_dir.mkdir(parents=True)
    _write_member(
        pareto_dir / "a.json",
        adg=0.0015,
        drawdown=0.15,
        recovery=36.0,
        fills_per_day=0.8,
        gap_max=10.0,
        gap_mean=4.0,
        gap_median=2.0,
    )
    _write_member(
        pareto_dir / "b.json",
        adg=0.0025,
        drawdown=0.12,
        recovery=24.0,
        fills_per_day=1.1,
        gap_max=8.0,
        gap_mean=3.0,
        gap_median=2.0,
    )

    manifest = build_baseline_manifest(run_dir)

    assert manifest["n_candidates"] == 2
    assert manifest["best_member_path"].endswith("b.json")
    assert manifest["best_passed"] is True
    assert manifest["best_score"] > 0.0
    assert manifest["candidate_command_template"].find("--baseline-pareto") != -1


def test_write_baseline_manifest_defaults_to_run_dir(tmp_path):
    run_dir = tmp_path / "run"
    pareto_dir = run_dir / "pareto"
    pareto_dir.mkdir(parents=True)
    _write_member(
        pareto_dir / "a.json",
        adg=0.002,
        drawdown=0.12,
        recovery=24.0,
        fills_per_day=1.0,
        gap_max=6.0,
        gap_mean=3.0,
        gap_median=2.0,
    )

    destination = write_baseline_manifest(run_dir)

    assert destination == run_dir / "baseline.json"
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["best_member_path"].endswith("a.json")


def test_write_baseline_manifest_rejects_failing_best_by_default(tmp_path):
    run_dir = tmp_path / "run"
    pareto_dir = run_dir / "pareto"
    pareto_dir.mkdir(parents=True)
    _write_member(
        pareto_dir / "a.json",
        adg=0.001,
        drawdown=0.12,
        recovery=600.0,
        fills_per_day=1.0,
        gap_max=6.0,
        gap_mean=3.0,
        gap_median=2.0,
    )

    with pytest.raises(ValueError, match="best baseline candidate does not satisfy constrained scorer"):
        write_baseline_manifest(run_dir)
