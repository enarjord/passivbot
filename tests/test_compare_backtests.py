import json

import pandas as pd
import pytest

from tools.compare_backtests import (
    _equity_comparison,
    _metric_comparison,
    _multiset_match_fraction,
    compare_backtest_artifacts,
    main,
)


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
    assert report["metrics"]["adg_strategy_eq"] == {
        "base": 0.001,
        "candidate": 0.0011,
        "absolute_delta": pytest.approx(0.0001),
        "relative_delta_fraction": pytest.approx(0.1),
    }
    assert report["fills"]["count_delta"] == 0
    assert report["fills"]["event_multiset_match_fraction"] == pytest.approx(2 / 3)
    assert report["fills"]["exact_multiset_match_fraction"] == pytest.approx(1 / 3)
    assert report["fills"]["common_prefix_rows"] == 1
    assert report["fills"]["type_count_delta"] == {
        "close_grid_long": -1,
        "close_unstuck_long": 1,
    }
    assert report["strategy_equity"]["final_equity_relative_delta_fraction"] == pytest.approx(0.05)
    assert report["strategy_equity"]["equity_rmse_initial_equity_fraction"] == pytest.approx(
        (40 / 3) ** 0.5 / 100
    )
    assert report["strategy_equity"]["max_abs_relative_equity_delta_fraction"] == pytest.approx(
        0.05
    )
    assert report["strategy_equity"]["common_sample_count"] == 3


def test_relative_metric_delta_uses_absolute_base_and_preserves_zero_sentinel():
    metrics = _metric_comparison(
        {"adg_strategy_eq": -0.002, "drawdown_worst_strategy_eq": 0.0},
        {"adg_strategy_eq": -0.001, "drawdown_worst_strategy_eq": 0.1},
    )

    assert metrics["adg_strategy_eq"]["relative_delta_fraction"] == pytest.approx(0.5)
    assert metrics["drawdown_worst_strategy_eq"]["relative_delta_fraction"] is None


def test_equity_fraction_denominators_use_base_magnitudes_and_common_samples():
    base = pd.DataFrame(
        {"timestamp": [1, 2, 3, 4], "strategy_equity": [-100.0, 0.0, -200.0, 0.0]}
    )
    candidate = pd.DataFrame(
        {"timestamp": [1, 2, 3, 5], "strategy_equity": [-90.0, 10.0, -160.0, 20.0]}
    )

    equity = _equity_comparison(base, candidate)

    assert equity["common_sample_count"] == 3
    assert equity["final_equity_relative_delta_fraction"] is None
    assert equity["equity_rmse_initial_equity_fraction"] == pytest.approx(600 ** 0.5 / 100)
    assert equity["max_abs_relative_equity_delta_fraction"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("base_rows", "candidate_rows", "expected"),
    [([1, 1], [1, 2, 1, 3], 0.5), ([], [], 0.0)],
)
def test_multiset_match_fraction_uses_multiplicity_and_larger_row_count(
    base_rows, candidate_rows, expected
):
    assert _multiset_match_fraction(
        pd.DataFrame({"row": base_rows}), pd.DataFrame({"row": candidate_rows}), ["row"]
    ) == pytest.approx(expected)


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
    assert "Final strategy equity delta: +5.000%" in captured.out
    assert "Equity RMSE vs initial strategy equity: 3.651%" in captured.out
    assert json.loads(output.read_text(encoding="utf-8"))["fills"]["count_delta"] == 0


def test_compare_backtests_cli_json_uses_fraction_names_without_rescaling(tmp_path, capsys):
    base = _write_artifact(tmp_path / "base")
    candidate = _write_artifact(tmp_path / "candidate", candidate=True)

    assert main([str(base), str(candidate), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)

    assert report["fills"]["event_multiset_match_fraction"] == pytest.approx(2 / 3)
    assert report["strategy_equity"]["final_equity_relative_delta_fraction"] == pytest.approx(0.05)
    assert not {
        "event_multiset_match_ratio", "exact_multiset_match_ratio"
    } & report["fills"].keys()
    assert not {
        "final_pct_delta", "rmse_pct_of_initial", "max_abs_pct_delta"
    } & report["strategy_equity"].keys()


def test_compare_backtests_cli_returns_two_for_incomplete_artifact(tmp_path, capsys):
    rc = main([str(tmp_path / "missing-base"), str(tmp_path / "missing-candidate")])
    captured = capsys.readouterr()

    assert rc == 2
    assert "missing backtest artifact file" in captured.err
