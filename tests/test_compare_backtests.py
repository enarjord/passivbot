import json

import pandas as pd
import pytest

from tools.compare_backtests import compare_backtest_artifacts, main


def _write_artifact(
    root,
    *,
    candidate: bool = False,
    exchange: str = "combined",
):
    root.mkdir(parents=True)
    (root / "dataset.json").write_text(
        json.dumps(
            {
                "cache_hash": "same-cache",
                "exchange": exchange,
                "coins": ["BTC"],
                "requested_start_date": "2025-01-01",
                "requested_end_date": "2026-01-01",
            }
        ),
        encoding="utf-8",
    )
    (root / "analysis.json").write_text(
        json.dumps(
            {
                "adg_strategy_eq": 0.0011 if candidate else 0.001,
                "drawdown_worst_strategy_eq": 0.21 if candidate else 0.2,
                "position_held_days_max": 12.0 if candidate else 10.0,
            }
        ),
        encoding="utf-8",
    )
    types = (
        ["entry_grid_normal_long", "close_unstuck_long", "entry_initial_normal_long"]
        if candidate
        else ["entry_grid_normal_long", "close_grid_long", "entry_initial_normal_long"]
    )
    qtys = [1.0, -1.0, 0.8 if candidate else 0.75]
    pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "coin": ["BTC", "BTC", "BTC"],
            "type": types,
            "qty": qtys,
            "price": [100.0, 105.0, 102.0],
            "pnl": [0.0, 5.0, 0.0],
            "fee_paid": [-0.1, -0.1, -0.1],
        }
    ).to_csv(root / "fills.csv", index=False)
    pd.DataFrame(
        {
            "strategy_equity": (
                [100.0, 108.0, 126.0] if candidate else [100.0, 110.0, 120.0]
            )
        }
    ).to_csv(root / "balance_and_equity.csv.gz", compression="gzip")
    return root


def test_compare_backtest_artifacts_reports_dataset_metrics_fills_and_equity(tmp_path):
    base = _write_artifact(tmp_path / "base")
    candidate = _write_artifact(tmp_path / "candidate", candidate=True)

    report = compare_backtest_artifacts(base, candidate)

    assert report["dataset_identity"]["status"] == "same"
    assert report["metrics"]["adg_strategy_eq"]["relative_delta"] == pytest.approx(0.1)
    assert report["fills"]["count_delta"] == 0
    assert report["fills"]["event_multiset_match_ratio"] == pytest.approx(2 / 3)
    assert report["fills"]["exact_multiset_match_ratio"] == pytest.approx(1 / 3)
    assert report["fills"]["common_prefix_rows"] == 1
    assert report["fills"]["type_count_delta"] == {
        "close_grid_long": -1,
        "close_unstuck_long": 1,
    }
    assert report["strategy_equity"]["final_pct_delta"] == pytest.approx(0.05)
    assert report["strategy_equity"]["common_sample_count"] == 3


def test_compare_backtest_artifacts_marks_dataset_mismatch_without_safety_verdict(
    tmp_path,
):
    base = _write_artifact(tmp_path / "base")
    candidate = _write_artifact(
        tmp_path / "candidate", candidate=True, exchange="binance"
    )

    report = compare_backtest_artifacts(base, candidate)

    assert report["dataset_identity"]["status"] == "mismatch"
    assert report["dataset_identity"]["mismatched_fields"] == ["exchange"]
    assert "does not declare a migration safe or unsafe" in report["interpretation"]


def test_compare_backtest_artifacts_marks_missing_dataset_identity_as_unproven(
    tmp_path,
):
    base = _write_artifact(tmp_path / "base")
    candidate = _write_artifact(tmp_path / "candidate", candidate=True)
    candidate_dataset_path = candidate / "dataset.json"
    candidate_dataset = json.loads(candidate_dataset_path.read_text(encoding="utf-8"))
    candidate_dataset.pop("cache_hash")
    candidate_dataset_path.write_text(json.dumps(candidate_dataset), encoding="utf-8")

    report = compare_backtest_artifacts(base, candidate)

    assert report["dataset_identity"]["status"] == "unproven"
    assert report["dataset_identity"]["missing_fields"] == ["cache_hash"]


def test_compare_backtests_cli_prints_human_summary_and_can_write_json(
    tmp_path, capsys
):
    base = _write_artifact(tmp_path / "base")
    candidate = _write_artifact(tmp_path / "candidate", candidate=True)
    output = tmp_path / "comparison.json"

    rc = main([str(base), str(candidate), "--output", str(output)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Dataset identity: same" in captured.out
    assert "Fill-event match: 66.67%" in captured.out
    assert json.loads(output.read_text(encoding="utf-8"))["fills"]["count_delta"] == 0


def test_compare_backtests_cli_returns_two_for_incomplete_artifact(tmp_path, capsys):
    rc = main([str(tmp_path / "missing-base"), str(tmp_path / "missing-candidate")])
    captured = capsys.readouterr()

    assert rc == 2
    assert "missing backtest artifact file" in captured.err
