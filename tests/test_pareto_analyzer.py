from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto_analyzer import DEFAULT_TOP, analyze_from_args, build_parser, format_analysis


def _write_candidate(
    pareto_dir: Path,
    name: str,
    *,
    qty_pct: float,
    twe: float,
    adg: float,
    drawdown: float,
) -> None:
    payload = {
        "config_version": "v8.0.0",
        "backtest": {"aggregate": {"default": "mean"}, "starting_balance": 100000},
        "bot": {
            "long": {
                "risk": {
                    "n_positions": 2,
                    "total_wallet_exposure_limit": twe,
                },
                "strategy": {
                    "trailing_martingale": {
                        "entry": {"initial_qty_pct": qty_pct},
                    }
                },
            }
        },
        "live": {"leverage": 10},
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
                "adg_strategy_eq": {"mean": adg, "min": adg * 0.5, "max": adg * 1.5, "std": 0.0},
                "drawdown_worst_strategy_eq": {
                    "mean": drawdown,
                    "min": drawdown * 0.8,
                    "max": drawdown * 1.2,
                    "std": 0.0,
                },
            },
        },
    }
    with (pareto_dir / f"{name}.json").open("w") as f:
        json.dump(payload, f)


@pytest.fixture()
def sample_pareto_dir(tmp_path: Path) -> Path:
    pareto_dir = tmp_path / "run" / "pareto"
    pareto_dir.mkdir(parents=True)
    _write_candidate(pareto_dir, "a", qty_pct=0.01, twe=1.0, adg=0.001, drawdown=0.30)
    _write_candidate(pareto_dir, "b", qty_pct=0.02, twe=1.5, adg=0.002, drawdown=0.35)
    _write_candidate(pareto_dir, "c", qty_pct=0.03, twe=2.0, adg=0.003, drawdown=0.45)
    return pareto_dir


def test_analyze_summarizes_config_params_and_scoring_metrics(sample_pareto_dir: Path):
    parser = build_parser()
    args = parser.parse_args([str(sample_pareto_dir), "--params", "bot.long.*", "--metrics", "scoring"])

    payload = analyze_from_args(args)

    param_by_name = {item["name"]: item for item in payload["params"]}
    assert param_by_name["bot.long.risk.total_wallet_exposure_limit"]["min"] == 1.0
    assert param_by_name["bot.long.risk.total_wallet_exposure_limit"]["max"] == 2.0
    assert param_by_name["bot.long.strategy.trailing_martingale.entry.initial_qty_pct"]["median"] == 0.02

    metric_by_name = {item["name"]: item for item in payload["metrics"]}
    assert set(metric_by_name) == {"adg_strategy_eq", "drawdown_worst_strategy_eq"}
    assert metric_by_name["adg_strategy_eq"]["mean"] == pytest.approx(0.002)
    assert metric_by_name["drawdown_worst_strategy_eq"]["median"] == pytest.approx(0.35)


def test_analyze_applies_limits_and_writes_outputs(sample_pareto_dir: Path, tmp_path: Path):
    out_dir = tmp_path / "analysis"
    parser = build_parser()
    args = parser.parse_args(
        [
            str(sample_pareto_dir.parent),
            "--limit",
            "adg_strategy_eq>0.0005",
            "--params",
            "bot.long.risk.total_wallet_exposure_limit",
            "--metrics",
            "all",
            "--output-dir",
            str(out_dir),
            "--corr",
            "5",
        ]
    )

    payload = analyze_from_args(args)

    assert payload["loaded_count"] == 3
    assert payload["retained_count"] == 3
    assert (out_dir / "params.csv").exists()
    assert (out_dir / "metrics.csv").exists()
    assert (out_dir / "correlations.csv").exists()
    assert (out_dir / "metric_correlations.csv").exists()
    assert (out_dir / "summary.json").exists()
    assert payload["correlations"]
    assert payload["metric_correlations"]


def test_analyze_default_show_rows_is_50():
    parser = build_parser()
    args = parser.parse_args([])

    assert DEFAULT_TOP == 50
    assert args.show == 50


def test_format_analysis_keeps_correlations_sorted_by_strength():
    payload = {
        "pareto_dir": "pareto",
        "loaded_count": 3,
        "retained_count": 3,
        "scoring_metrics": [],
        "metrics": [],
        "params": [],
        "correlations": [
            {"param": "z_param", "metric": "z_metric", "corr": 0.90, "abs_corr": 0.90, "count": 3},
            {"param": "a_param", "metric": "a_metric", "corr": -0.80, "abs_corr": 0.80, "count": 3},
        ],
    }

    text = format_analysis(payload, show=50, corr_limit=2)

    assert text.index("z_param") < text.index("a_param")


def test_format_analysis_shows_only_visible_metric_correlations():
    def metric_summary(name: str) -> dict[str, object]:
        return {
            "group": "metric",
            "name": name,
            "count": 3,
            "missing": 0,
            "min": 1.0,
            "max": 3.0,
            "range": 2.0,
            "mean": 2.0,
            "median": 2.0,
            "std": 1.0,
            "p05": 1.1,
            "p25": 1.5,
            "p75": 2.5,
            "p95": 2.9,
            "iqr": 1.0,
            "unique_count": 3,
            "min_file": "a.json",
            "max_file": "c.json",
        }

    payload = {
        "pareto_dir": "pareto",
        "loaded_count": 3,
        "retained_count": 3,
        "scoring_metrics": [],
        "metrics": [
            metric_summary("a_metric"),
            metric_summary("b_metric"),
            metric_summary("z_hidden_metric"),
        ],
        "params": [],
        "correlations": [],
        "metric_correlations": [
            {"metric_a": "a_metric", "metric_b": "b_metric", "corr": -0.95, "abs_corr": 0.95, "count": 3},
            {
                "metric_a": "a_metric",
                "metric_b": "z_hidden_metric",
                "corr": 0.99,
                "abs_corr": 0.99,
                "count": 3,
            },
        ],
    }

    text = format_analysis(payload, show=2, corr_limit=10)

    assert "Metric/Metric Correlations" in text
    assert "a_metric" in text
    assert "b_metric" in text
    assert "z_hidden_metric" not in text
