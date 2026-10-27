from __future__ import annotations

import json
from pathlib import Path

import pytest

from pareto_analyzer import DEFAULT_TOP, analyze_from_args, build_parser, format_analysis


def _metric_stats(value: float, *, min_value: float | None = None, max_value: float | None = None) -> dict:
    return {
        "mean": value,
        "min": value if min_value is None else min_value,
        "max": value if max_value is None else max_value,
        "std": 0.0,
        "median": value,
    }


def _write_candidate(
    pareto_dir: Path,
    name: str,
    *,
    qty_fraction: float,
    twe: float,
    adg_fraction: float,
    drawdown: float,
) -> None:
    payload = {
        "config_version": "v8.0.0",
        "backtest": {"aggregate": {"default": "mean"}, "starting_balance": 100000},
        "bot": {
            "long": {
                "risk": {
                    "n_positions": 2,
                    "total_wallet_exposure_limit_ratio": twe,
                },
                "strategy": {
                    "trailing_martingale": {
                        "entry": {"initial_qty_fraction": qty_fraction},
                    }
                },
            }
        },
        "live": {"leverage": 10},
        "optimize": {
            "scoring": [
                {"metric": "cagr_fraction_strategy_equity", "goal": "max"},
                {"metric": "drawdown_max_fraction_strategy_equity", "goal": "min"},
            ]
        },
        "metrics": {
            "definition_version": 1,
            "objectives": {
                "cagr_fraction_strategy_equity": adg_fraction,
                "drawdown_max_fraction_strategy_equity": drawdown,
            },
            "stats": {
                "cagr_fraction_strategy_equity": _metric_stats(
                    adg_fraction,
                    min_value=adg_fraction * 0.5,
                    max_value=adg_fraction * 1.5,
                ),
                "drawdown_max_fraction_strategy_equity": _metric_stats(
                    drawdown,
                    min_value=drawdown * 0.8,
                    max_value=drawdown * 1.2,
                ),
            },
        },
    }
    with (pareto_dir / f"{name}.json").open("w") as f:
        json.dump(payload, f)


@pytest.fixture()
def sample_pareto_dir(tmp_path: Path) -> Path:
    pareto_dir = tmp_path / "run" / "pareto"
    pareto_dir.mkdir(parents=True)
    _write_candidate(pareto_dir, "a", qty_fraction=0.01, twe=1.0, adg_fraction=0.001, drawdown=0.30)
    _write_candidate(pareto_dir, "b", qty_fraction=0.02, twe=1.5, adg_fraction=0.002, drawdown=0.35)
    _write_candidate(pareto_dir, "c", qty_fraction=0.03, twe=2.0, adg_fraction=0.003, drawdown=0.45)
    return pareto_dir


def test_analyze_summarizes_config_params_and_scoring_metrics(sample_pareto_dir: Path):
    parser = build_parser()
    args = parser.parse_args([str(sample_pareto_dir), "--params", "bot.long.*", "--metrics", "scoring"])

    payload = analyze_from_args(args)

    param_by_name = {item["name"]: item for item in payload["params"]}
    assert param_by_name["bot.long.risk.total_wallet_exposure_limit_ratio"]["min"] == 1.0
    assert param_by_name["bot.long.risk.total_wallet_exposure_limit_ratio"]["max"] == 2.0
    assert param_by_name["bot.long.strategy.trailing_martingale.entry.initial_qty_fraction"]["median"] == 0.02

    metric_by_name = {item["name"]: item for item in payload["metrics"]}
    assert set(metric_by_name) == {"cagr_fraction_strategy_equity", "drawdown_max_fraction_strategy_equity"}
    assert metric_by_name["cagr_fraction_strategy_equity"]["mean"] == pytest.approx(0.002)
    assert metric_by_name["drawdown_max_fraction_strategy_equity"]["median"] == pytest.approx(0.35)


def test_analyze_applies_limits_and_writes_outputs(sample_pareto_dir: Path, tmp_path: Path):
    out_dir = tmp_path / "analysis"
    parser = build_parser()
    args = parser.parse_args(
        [
            str(sample_pareto_dir.parent),
            "--limit",
            "cagr_fraction_strategy_equity>0.0005",
            "--params",
            "bot.long.risk.total_wallet_exposure_limit_ratio",
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


def test_analyzer_retains_infinite_endpoints_and_unavailable_distributions(
    tmp_path, capsys
):
    import math
    from metrics_schema import encode_metric_values
    from pareto_explorer import load_candidates
    from pareto_analyzer import resolve_metric_series_value, run_from_args

    metric = "sharpe_ratio_strategy_equity"
    short = "short_net_profit_contribution_fraction"
    pareto = tmp_path / "pareto"
    pareto.mkdir()
    for name, value in (
        ("finite", 1.5),
        ("positive", math.inf),
        ("negative", -math.inf),
    ):
        payload = {
            "optimize": {"scoring": [metric]},
            "metrics": {
                "definition_version": 1,
                "objectives": encode_metric_values({metric: value}),
                "stats": {
                    short: {
                        "mean": None,
                        "metric_status": {
                            "mean": {
                                "status": "undefined",
                                "reason": "no positive combined profit",
                            }
                        },
                    }
                },
            },
        }
        (pareto / f"{name}.json").write_text(json.dumps(payload, allow_nan=False))
    _, candidates, _ = load_candidates(pareto)
    ordered = sorted(
        candidates, key=lambda row: resolve_metric_series_value(row, metric)
    )
    assert [candidate.path.stem for candidate in ordered] == [
        "negative",
        "finite",
        "positive",
    ]
    args = build_parser().parse_args(
        [
            str(pareto),
            "--metrics",
            "all",
            "--params",
            "none",
            "--json",
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    report = run_from_args(args)
    output = capsys.readouterr().out
    json.loads(
        output,
        parse_constant=lambda value: pytest.fail(f"raw nonfinite literal {value}"),
    )
    json.dumps(report, allow_nan=False)
    summaries = {row["name"]: row for row in report["metrics"]}
    summary = summaries[metric]
    assert summary["count"] == 3 and summary["finite_count"] == 1
    assert summary["positive_infinity_count"] == summary["negative_infinity_count"] == 1
    assert summary["min"] is None and summary["max"] is None
    assert summary["metric_status"]["min"]["status"] == "negative_infinity"
    assert summary["metric_status"]["max"]["status"] == "positive_infinity"
    assert summary["mean"] is None and summary["median"] == 1.5
    assert summary["metric_status"]["mean"]["status"] == "undefined"
    for field in ("std", "p05", "p25", "p75", "p95"):
        assert (
            summary[field] is None
            and summary["metric_status"][field]["status"] == "undefined"
        )
    assert (
        summary["min_file"] == "negative.json"
        and summary["max_file"] == "positive.json"
    )
    unavailable = summaries[short]
    assert unavailable["count"] == 0 and unavailable["unavailable_count"] == 3
    assert (
        unavailable["unavailable_sources"]["finite.json"]["reason"]
        == "no positive combined profit"
    )
    assert unavailable["mean"] is None
    text = format_analysis(report, show=50, corr_limit=10)
    assert "+infinity=1" in text and "unavailable=3" in text and "undefined" in text
    json.loads(
        (tmp_path / "output" / "summary.json").read_text(),
        parse_constant=lambda value: pytest.fail(value),
    )


def test_analyzer_single_infinite_population_is_not_dropped(tmp_path):
    import math
    from metrics_schema import encode_metric_values
    from pareto_analyzer import summarize_series
    from pareto_explorer import ParetoCandidate

    metric = "omega_ratio_strategy_equity"
    candidate = ParetoCandidate(
        tmp_path / "infinite.json", {}, {metric: math.inf}, {}, {}
    )
    row = summarize_series(
        group="metric",
        name=metric,
        candidates=[candidate],
        value_maps=[encode_metric_values({metric: math.inf})],
    ).to_dict()
    assert row["count"] == 1 and row["finite_count"] == 0
    assert row["metric_status"]["mean"]["status"] == "positive_infinity"
    assert row["metric_status"]["median"]["status"] == "positive_infinity"
    assert (
        row["range"] is None and row["metric_status"]["range"]["status"] == "undefined"
    )
    json.dumps(row, allow_nan=False)
