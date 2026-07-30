import json
from pathlib import Path

import msgpack
import numpy as np
import pandas as pd

from tools import pareto_dash


def _make_suite_metrics():
    return {
        "metrics": {
            "adg": {
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
        "metrics": {"objectives": {"w_0": -1.0, "w_1": 2.5}},
        "optimize": {"scoring": ["adg", "omega_ratio"]},
    }
    _write_pareto_entry(pareto_dir / "0001_hash.json", entry)

    run_data = pareto_dash.load_pareto_dataframe(str(run_dir))
    df = run_data.dataframe

    assert "_id" in df.columns
    assert "adg" in df.columns
    assert "adg_mean" in df.columns
    assert "base__adg" in df.columns
    assert "stress__adg" in df.columns
    assert "bot.long.n_positions" in df.columns
    assert "objective.adg_usd" in df.columns
    assert "objective.omega_ratio_usd" in df.columns
    assert "objective.w_0" not in df.columns
    assert run_data.scenario_metrics["base"] == ["adg"]
    assert run_data.scoring_metrics == ["objective.adg_usd", "objective.omega_ratio_usd"]
    assert run_data.display_labels["objective.adg_usd"] == "adg_usd"
    assert run_data.display_labels["objective.omega_ratio_usd"] == "omega_ratio_usd"
    assert np.isclose(df["adg"].iloc[0], 0.1)
    assert np.isclose(df["objective.adg_usd"].iloc[0], 1.0)


def test_load_pareto_dataframe_keeps_median_named_aggregated_metric(tmp_path):
    run_dir = tmp_path / "run"
    pareto_dir = run_dir / "pareto"
    entry = {
        "suite_metrics": {
            "metrics": {
                "position_held_days_median": {
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

    assert "position_held_days_median" in run_data.aggregated_metrics
    assert "position_held_days_median_mean" not in run_data.aggregated_metrics


def test_load_history_dataframe_emits_iterations(tmp_path):
    run_dir = tmp_path / "run"
    pareto_dir = run_dir / "pareto"
    pareto_dir.mkdir(parents=True, exist_ok=True)
    history_path = run_dir / "all_results.bin"

    entry = {
        "suite_metrics": _make_suite_metrics(),
        "metrics": {"objectives": {"w_0": -2.0}},
    }
    with history_path.open("wb") as fh:
        packer = msgpack.Packer(use_bin_type=True)
        fh.write(packer.pack(entry))
        fh.write(packer.pack(entry))

    df = pareto_dash.load_history_dataframe(str(run_dir))
    assert "iteration" in df.columns
    assert "objective.w_0" in df.columns
    assert not df.empty
    assert np.isclose(df["adg"].iloc[0], 0.1)


def test_default_limit_expressions_target_scenario_columns():
    expressions = pareto_dash._limits_to_exprs(
        [
            {
                "metric": "adg",
                "penalize_if": "less_than",
                "scenario": "base",
                "value": 0.08,
            },
            {
                "metric": "adg",
                "penalize_if": "less_than",
                "stat": "min",
                "value": 0.05,
            },
        ]
    )

    assert expressions == ["base__adg_usd>=0.08", "adg_usd_min>=0.05"]


def test_scenario_limit_expression_supports_punctuation_in_label():
    expressions = pareto_dash._limits_to_exprs(
        [
            {
                "metric": "adg",
                "penalize_if": "greater_than",
                "scenario": "bear-market",
                "value": 0.08,
            }
        ]
    )
    dataframe = pd.DataFrame({"bear-market__adg_usd": [0.07, 0.09]})

    mask = pareto_dash._apply_limits(dataframe, "\n".join(expressions))

    assert expressions == ["bear-market__adg_usd<=0.08"]
    assert mask.tolist() == [True, False]


def test_scenario_limit_expression_quotes_or_separator_in_label():
    expressions = pareto_dash._limits_to_exprs(
        [
            {
                "metric": "adg",
                "penalize_if": "greater_than",
                "scenario": "stress||base",
                "value": 0.08,
            }
        ]
    )
    dataframe = pd.DataFrame({"stress||base__adg_usd": [0.07, 0.09]})

    mask = pareto_dash._apply_limits(dataframe, "\n".join(expressions))

    assert expressions == ['"stress||base__adg_usd"<=0.08']
    assert mask.tolist() == [True, False]


def test_scenario_inside_range_limit_expression_preserves_outside_values():
    expressions = pareto_dash._limits_to_exprs(
        [
            {
                "metric": "adg",
                "penalize_if": "inside_range",
                "scenario": "stress",
                "range": [0.05, 0.10],
            }
        ]
    )
    dataframe = pd.DataFrame(
        {"stress__adg_usd": [0.04, 0.05, 0.075, 0.10, 0.11]}
    )

    mask = pareto_dash._apply_limits(dataframe, "\n".join(expressions))

    assert expressions == [
        "stress__adg_usd<=0.05 || stress__adg_usd>=0.1"
    ]
    assert mask.tolist() == [True, True, False, True, True]


def test_scenario_auto_limit_expressions_use_optimizer_directions():
    expressions = pareto_dash._limits_to_exprs(
        [
            {
                "metric": "adg",
                "penalize_if": "auto",
                "scenario": "base",
                "value": 0.001,
            },
            {
                "metric": "drawdown_worst_strategy_eq",
                "penalize_if": "auto",
                "scenario": "stress",
                "value": 0.5,
            },
        ]
    )

    assert expressions == [
        "base__adg_usd>=0.001",
        "stress__drawdown_worst_strategy_eq<=0.5",
    ]


def test_scenario_equal_to_limit_expression_filters_equal_values():
    expressions = pareto_dash._limits_to_exprs(
        [
            {
                "metric": "drawdown_worst_strategy_eq",
                "penalize_if": "equal_to",
                "scenario": "stress",
                "value": 0.5,
            }
        ]
    )
    dataframe = pd.DataFrame(
        {"stress__drawdown_worst_strategy_eq": [0.4, 0.5, 0.6]}
    )

    mask = pareto_dash._apply_limits(dataframe, "\n".join(expressions))

    assert expressions == ["stress__drawdown_worst_strategy_eq!=0.5"]
    assert mask.tolist() == [True, False, True]


def test_disabled_scenario_limit_is_not_a_dashboard_default():
    expressions = pareto_dash._limits_to_exprs(
        [
            {
                "enabled": False,
                "metric": "drawdown_worst_strategy_eq",
                "penalize_if": "greater_than",
                "scenario": "stress",
                "value": 0.5,
            }
        ]
    )

    assert expressions == []
