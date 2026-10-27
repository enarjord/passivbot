import json
from pathlib import Path

import msgpack
import numpy as np
import pandas as pd

from tools import pareto_dash


def _make_suite_metrics():
    return {
        "definition_version": 1,
        "metrics": {
            "cagr_fraction_strategy_equity": {
                "aggregated": 0.1,
                "stats": {"mean": 0.1, "min": 0.05, "max": 0.2, "std": 0.02},
                "scenarios": {"base": 0.11, "stress": 0.07},
            }
        },
        "scenario_labels": ["base", "stress"],
    }


def _write_pareto_entry(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(entry))


def test_load_pareto_dataframe_handles_suite_and_params(tmp_path):
    run_dir = tmp_path / "run"
    pareto_dir = run_dir / "pareto"
    entry = {
        "bot": {"long": {"n_positions": 3}},
        "suite_metrics": _make_suite_metrics(),
        "metrics": {
            "definition_version": 1,
            "objectives": {
                "cagr_fraction_strategy_equity": 1.0,
                "omega_ratio_strategy_equity": 2.5,
            },
        },
        "optimize": {
            "scoring": ["cagr_fraction_strategy_equity", "omega_ratio_strategy_equity"]
        },
    }
    _write_pareto_entry(pareto_dir / "0001_hash.json", entry)

    run_data = pareto_dash.load_pareto_dataframe(str(run_dir))
    df = run_data.dataframe

    assert "_id" in df.columns
    assert "cagr_fraction_strategy_equity" in df.columns
    assert "cagr_fraction_strategy_equity_mean" in df.columns
    assert "base__cagr_fraction_strategy_equity" in df.columns
    assert "stress__cagr_fraction_strategy_equity" in df.columns
    assert "bot.long.n_positions" in df.columns
    assert "objective.cagr_fraction_strategy_equity" in df.columns
    assert "objective.omega_ratio_strategy_equity" in df.columns
    assert "objective.w_0" not in df.columns
    assert run_data.scenario_metrics["base"] == ["cagr_fraction_strategy_equity"]
    assert run_data.scoring_metrics == [
        "objective.cagr_fraction_strategy_equity",
        "objective.omega_ratio_strategy_equity",
    ]
    assert (
        run_data.display_labels["objective.cagr_fraction_strategy_equity"]
        == "cagr_fraction_strategy_equity"
    )
    assert (
        run_data.display_labels["objective.omega_ratio_strategy_equity"]
        == "omega_ratio_strategy_equity"
    )
    assert np.isclose(df["cagr_fraction_strategy_equity"].iloc[0], 0.1)
    assert np.isclose(df["objective.cagr_fraction_strategy_equity"].iloc[0], 1.0)


def test_load_pareto_dataframe_keeps_median_named_aggregated_metric(tmp_path):
    run_dir = tmp_path / "run"
    pareto_dir = run_dir / "pareto"
    entry = {
        "suite_metrics": {
            "definition_version": 1,
            "metrics": {
                "position_held_time_weighted_mean_hours": {
                    "aggregated": 2.5,
                    "stats": {
                        "mean": 2.5,
                        "min": 2.0,
                        "max": 3.0,
                        "std": 0.5,
                        "median": 2.5,
                    },
                    "scenarios": {"base": 2.0, "stress": 3.0},
                }
            },
            "scenario_labels": ["base", "stress"],
        }
    }
    _write_pareto_entry(pareto_dir / "0001_hash.json", entry)

    run_data = pareto_dash.load_pareto_dataframe(str(run_dir))

    assert "position_held_time_weighted_mean_hours" in run_data.aggregated_metrics
    assert (
        "position_held_time_weighted_mean_hours_mean" not in run_data.aggregated_metrics
    )


def test_load_history_dataframe_emits_iterations(tmp_path):
    run_dir = tmp_path / "run"
    pareto_dir = run_dir / "pareto"
    pareto_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "all_results.bin"

    entry = {
        "suite_metrics": _make_suite_metrics(),
        "metrics": {
            "definition_version": 1,
            "objectives": {"cagr_fraction_strategy_equity": 2.0},
        },
    }
    with history_path.open("wb") as fh:
        packer = msgpack.Packer(use_bin_type=True)
        fh.write(packer.pack(entry))
        fh.write(packer.pack(entry))

    df = pareto_dash.load_history_dataframe(str(run_dir))
    assert "iteration" in df.columns
    assert "objective.cagr_fraction_strategy_equity" in df.columns
    assert not df.empty
    assert np.isclose(df["cagr_fraction_strategy_equity"].iloc[0], 0.1)


def test_default_limit_expressions_target_scenario_columns():
    expressions = pareto_dash._limits_to_exprs(
        [
            {
                "metric": "cagr_fraction_strategy_equity",
                "penalize_if": "less_than",
                "scenario": "base",
                "value": 0.08,
            },
            {
                "metric": "cagr_fraction_strategy_equity",
                "penalize_if": "less_than",
                "reducer": "min",
                "value": 0.05,
            },
        ]
    )

    assert expressions == [
        "base__cagr_fraction_strategy_equity>=0.08",
        "cagr_fraction_strategy_equity_min>=0.05",
    ]


def test_scenario_limit_expression_supports_punctuation_in_label():
    expressions = pareto_dash._limits_to_exprs(
        [
            {
                "metric": "cagr_fraction_strategy_equity",
                "penalize_if": "greater_than",
                "scenario": "bear-market",
                "value": 0.08,
            }
        ]
    )
    dataframe = pd.DataFrame(
        {"bear-market__cagr_fraction_strategy_equity": [0.07, 0.09]}
    )

    mask = pareto_dash._apply_limits(dataframe, "\n".join(expressions))

    assert expressions == ["bear-market__cagr_fraction_strategy_equity<=0.08"]
    assert mask.tolist() == [True, False]


def test_scenario_limit_expression_quotes_or_separator_in_label():
    expressions = pareto_dash._limits_to_exprs(
        [
            {
                "metric": "cagr_fraction_strategy_equity",
                "penalize_if": "greater_than",
                "scenario": "stress||base",
                "value": 0.08,
            }
        ]
    )
    dataframe = pd.DataFrame(
        {"stress||base__cagr_fraction_strategy_equity": [0.07, 0.09]}
    )

    mask = pareto_dash._apply_limits(dataframe, "\n".join(expressions))

    assert expressions == ['"stress||base__cagr_fraction_strategy_equity"<=0.08']
    assert mask.tolist() == [True, False]


def test_scenario_inside_range_limit_expression_preserves_outside_values():
    expressions = pareto_dash._limits_to_exprs(
        [
            {
                "metric": "cagr_fraction_strategy_equity",
                "penalize_if": "inside_range",
                "scenario": "stress",
                "range": [0.05, 0.10],
            }
        ]
    )
    dataframe = pd.DataFrame(
        {"stress__cagr_fraction_strategy_equity": [0.04, 0.05, 0.075, 0.10, 0.11]}
    )

    mask = pareto_dash._apply_limits(dataframe, "\n".join(expressions))

    assert expressions == [
        "stress__cagr_fraction_strategy_equity<=0.05 || stress__cagr_fraction_strategy_equity>=0.1"
    ]
    assert mask.tolist() == [True, True, False, True, True]


def test_scenario_auto_limit_expressions_use_optimizer_directions():
    expressions = pareto_dash._limits_to_exprs(
        [
            {
                "metric": "cagr_fraction_strategy_equity",
                "penalize_if": "auto",
                "scenario": "base",
                "value": 0.001,
            },
            {
                "metric": "ulcer_index_fraction_strategy_equity",
                "penalize_if": "auto",
                "scenario": "stress",
                "value": 0.5,
            },
        ]
    )

    assert expressions == [
        "base__cagr_fraction_strategy_equity>=0.001",
        "stress__ulcer_index_fraction_strategy_equity<=0.5",
    ]


def test_scenario_equal_to_limit_expression_filters_equal_values():
    expressions = pareto_dash._limits_to_exprs(
        [
            {
                "metric": "drawdown_max_fraction_strategy_equity",
                "penalize_if": "equal_to",
                "scenario": "stress",
                "value": 0.5,
            }
        ]
    )
    dataframe = pd.DataFrame(
        {"stress__drawdown_max_fraction_strategy_equity": [0.4, 0.5, 0.6]}
    )

    mask = pareto_dash._apply_limits(dataframe, "\n".join(expressions))

    assert expressions == ["stress__drawdown_max_fraction_strategy_equity!=0.5"]
    assert mask.tolist() == [True, False, True]


def test_disabled_scenario_limit_is_not_a_dashboard_default():
    expressions = pareto_dash._limits_to_exprs(
        [
            {
                "enabled": False,
                "metric": "drawdown_max_fraction_strategy_equity",
                "penalize_if": "greater_than",
                "scenario": "stress",
                "value": 0.5,
            }
        ]
    )

    assert expressions == []


def test_dashboard_preserves_statuses_and_uses_raw_infinities_for_filters(tmp_path):
    metric = "omega_ratio_strategy_equity"
    for name, status in (("infinite", "positive_infinity"), ("undefined", "undefined")):
        entry = {
            "metrics": {
                "definition_version": 1,
                "objectives": {
                    metric: None,
                    "metric_status": {
                        metric: {"status": status, "reason": "sample support"}
                    },
                },
            },
            "optimize": {"scoring": [metric]},
        }
        _write_pareto_entry(tmp_path / "pareto" / f"{name}.json", entry)
    data = pareto_dash.load_pareto_dataframe(str(tmp_path))
    df = data.dataframe
    column = f"objective.{metric}"
    assert df[column].isna().all()
    assert len(df.attrs["metric_status"]) == 2
    assert "explicit statuses" in pareto_dash._projection_notice(df)
    assert pareto_dash._apply_limits(df, f"{column}>=10").tolist() == [True, False]
    assert pareto_dash._apply_limits(df, f"{column}!=10").tolist() == [True, False]
    assert not pareto_dash.compute_pareto_frontier(df, [column]).any()
    assert pareto_dash.compute_weighted_score(df, {column: 1}).isna().all()


def test_dashboard_rejects_stale_objectives_and_missing_filter_columns(tmp_path):
    import pytest

    _write_pareto_entry(
        tmp_path / "pareto" / "old.json", {"metrics": {"objectives": {"w_0": -1}}}
    )
    with pytest.raises(ValueError, match="definition"):
        pareto_dash.load_pareto_dataframe(str(tmp_path))
    with pytest.raises(ValueError, match="not stored"):
        pareto_dash._apply_limits(pd.DataFrame({"known": [0.1]}), "missing<1")


def test_history_retains_valid_invalid_and_prefilter_rows_with_explicit_status(
    tmp_path,
):
    metric = "cagr_fraction_strategy_equity"
    base = {"optimize": {"scoring": [metric]}}
    packets = [
        {**base, "metrics": {"definition_version": 1, "objectives": {metric: 0.4}}},
        {
            **base,
            "metrics": {
                "definition_version": 1,
                "fitness_status": "invalid",
                "objectives": {},
                "stats": {},
                "error": "insufficient observations",
            },
        },
        {
            **base,
            "metrics": {
                "definition_version": 1,
                "fitness_status": "invalid",
                "evaluation_origin": "config_prefilter",
                "objectives": {},
                "stats": {},
                "backtest_performed": False,
            },
        },
    ]
    with (tmp_path / "all_results.bin").open("wb") as stream:
        for packet in packets:
            stream.write(msgpack.packb(packet, use_bin_type=True))
    df = pareto_dash.load_history_dataframe(str(tmp_path))
    column = f"objective.{metric}"
    assert df["iteration"].tolist() == [0, 1, 2]
    assert df[column].iloc[0] == 0.4
    assert df[column].iloc[1:].isna().all()
    for iteration in (1, 2):
        assert df.attrs["metric_status"][iteration][column]["status"] == "invalid"
    assert "config_prefilter" in df.attrs["metric_status"][2][column]["reason"]
