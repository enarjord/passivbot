import numpy as np
from unittest.mock import MagicMock

from optimization.output import OptimizeOutput

SCORING_KEYS = ["adg_pnl", "drawdown_worst"]


class TestOptimizeOutput:
    def _make_algorithm(self, n_gen=3, n_eval=30, opt_F=None):
        """Create a mock algorithm with the fields OptimizeOutput reads."""
        algo = MagicMock()
        algo.n_gen = n_gen
        algo.evaluator.n_eval = n_eval
        opt = MagicMock()
        if opt_F is None:
            opt_F = np.array([[0.5, 0.3], [0.8, 0.1]])
        opt.get.return_value = opt_F
        opt.__len__ = lambda self: len(opt_F)
        algo.opt = opt
        return algo

    def test_columns_exist(self):
        output = OptimizeOutput(SCORING_KEYS)
        col_names = [c.name for c in output.columns]
        assert "front" in col_names
        assert "adg_pnl" in col_names
        assert "drawdown_worst" in col_names
        assert "elapsed" in col_names

    def test_update_sets_values(self):
        output = OptimizeOutput(SCORING_KEYS)
        algo = self._make_algorithm(n_gen=2, n_eval=20, opt_F=np.array([[0.5, 0.3]]))
        output.update(algo)
        assert output.front.value is not None
        assert output.obj_columns[0].value is not None
        assert output.obj_columns[1].value is not None
        assert output.elapsed.value is not None

    def test_front_size_matches_opt_length(self):
        output = OptimizeOutput(SCORING_KEYS)
        opt_F = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        algo = self._make_algorithm(opt_F=opt_F)
        output.update(algo)
        assert output.front.value == 3

    def test_best_per_objective_absolute_values(self):
        output = OptimizeOutput(SCORING_KEYS)
        # min per column: [0.5, -0.3] -> displayed as abs: [0.5, 0.3]
        opt_F = np.array([[0.5, -0.1], [0.8, -0.3]])
        algo = self._make_algorithm(opt_F=opt_F)
        output.update(algo)
        assert output.obj_columns[0].value == "0.5000"
        assert output.obj_columns[1].value == "0.3000"

    def test_objective_values_rounded_to_4_decimals(self):
        output = OptimizeOutput(SCORING_KEYS)
        opt_F = np.array([[0.123456789, -0.987654321]])
        algo = self._make_algorithm(opt_F=opt_F)
        output.update(algo)
        assert output.obj_columns[0].value == "0.1235"
        assert output.obj_columns[1].value == "0.9877"

    def test_opt_none_shows_dash(self):
        output = OptimizeOutput(SCORING_KEYS)
        algo = MagicMock()
        algo.opt = None
        output.update(algo)
        assert output.front.value == 0
        for col in output.obj_columns:
            assert col.value == "-"
