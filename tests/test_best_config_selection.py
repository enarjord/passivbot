"""Tests for select_best_config (pseudo-weight selection from Pareto front)."""

import json
import os
import numpy as np
import pytest

from optimize import select_best_config


def _make_entry(objectives: dict) -> dict:
    """Build a minimal Pareto entry matching ParetoWriterCallback's on-disk format."""
    return {
        "bot": {"long": {"n_positions": 1}, "short": {"n_positions": 1}},
        "live": {},
        "backtest": {},
        "optimize": {"scoring": list(objectives.keys())},
        "metrics": {},
        "objectives": objectives,
        "constraint_violation": 0.0,
    }


class TestSelectBestConfig:
    def test_picks_balanced_solution(self, tmp_path):
        """With 3 solutions on a 2-objective front, equal weights should pick the balanced one."""
        pareto_dir = tmp_path / "pareto"
        pareto_dir.mkdir()
        # Solution A: great on obj0, bad on obj1
        a = _make_entry({"w_0": -0.9, "w_1": -0.1})
        # Solution B: balanced
        b = _make_entry({"w_0": -0.5, "w_1": -0.5})
        # Solution C: bad on obj0, great on obj1
        c = _make_entry({"w_0": -0.1, "w_1": -0.9})

        for i, entry in enumerate([a, b, c]):
            (pareto_dir / f"{i:04d}.json").write_text(json.dumps(entry))

        result_path = select_best_config(str(tmp_path), n_objectives=2)

        assert result_path is not None
        assert os.path.exists(result_path)
        with open(result_path) as f:
            best = json.load(f)
        # Balanced solution should be selected
        assert best["objectives"] == {"w_0": -0.5, "w_1": -0.5}

    def test_empty_pareto_returns_none(self, tmp_path):
        """Empty pareto dir should return None without error."""
        pareto_dir = tmp_path / "pareto"
        pareto_dir.mkdir()

        result_path = select_best_config(str(tmp_path), n_objectives=2)
        assert result_path is None

    def test_single_entry_writes_it(self, tmp_path):
        """Single entry in pareto should be written as best_config.json."""
        pareto_dir = tmp_path / "pareto"
        pareto_dir.mkdir()
        entry = _make_entry({"w_0": -0.42, "w_1": -0.58})
        (pareto_dir / "only.json").write_text(json.dumps(entry))

        result_path = select_best_config(str(tmp_path), n_objectives=2)

        assert result_path is not None
        with open(result_path) as f:
            best = json.load(f)
        assert best["objectives"]["w_0"] == -0.42

    def test_output_path_is_best_config_json(self, tmp_path):
        """Output file should be {results_dir}/best_config.json."""
        pareto_dir = tmp_path / "pareto"
        pareto_dir.mkdir()
        entry = _make_entry({"w_0": -0.5})
        (pareto_dir / "a.json").write_text(json.dumps(entry))

        result_path = select_best_config(str(tmp_path), n_objectives=1)
        assert result_path == os.path.join(str(tmp_path), "best_config.json")
