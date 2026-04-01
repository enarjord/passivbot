import json
from pathlib import Path

import msgpack
import numpy as np

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
