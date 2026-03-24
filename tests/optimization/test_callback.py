import json
import os
import pytest
import numpy as np
import msgpack
from unittest.mock import MagicMock
from optimization.callback import ParetoWriterCallback
from opt_utils import load_results


class TestParetoWriterCallback:
    def _make_ind(self, x, f, g=None, metrics=None):
        ind = MagicMock()
        ind.X = np.array(x)
        ind.F = np.array(f)
        ind.G = np.array(g if g is not None else [])
        ind.data = {"metrics": metrics or {}}
        return ind

    def test_writes_pareto_configs(self, tmp_path):
        store = MagicMock()
        store.add_entry = MagicMock(return_value=True)

        template = {
            "bot": {
                "long": {"a": 0.0, "b": 0.0},
            },
        }
        scoring_keys = ["adg_pnl", "drawdown"]

        callback = ParetoWriterCallback(store, template, scoring_keys)

        ind1 = self._make_ind([1.0, 2.0], [0.5, 0.3], metrics={"stats": {}})
        algorithm = MagicMock()
        algorithm.pop = [ind1]
        algorithm.opt = [ind1]

        callback.notify(algorithm)

        assert store.add_entry.called
        entry = store.add_entry.call_args[0][0]
        assert "bot" in entry
        assert entry["bot"]["long"]["a"] == 1.0
        assert entry["bot"]["long"]["b"] == 2.0

    def test_flush_called(self):
        store = MagicMock()
        template = {"bot": {"long": {"a": 0.0}}}
        callback = ParetoWriterCallback(store, template, ["obj1"])

        ind = self._make_ind([1.0], [0.5])
        algorithm = MagicMock()
        algorithm.pop = [ind]
        algorithm.opt = [ind]

        callback.notify(algorithm)
        assert store.flush.called

    def test_none_opt_skipped(self):
        store = MagicMock()
        template = {"bot": {"long": {"a": 0.0}}}
        callback = ParetoWriterCallback(store, template, ["obj1"])

        algorithm = MagicMock()
        algorithm.pop = None
        algorithm.opt = None

        callback.notify(algorithm)
        assert not store.add_entry.called

    def test_all_results_bin_written(self, tmp_path):
        """all_results.bin is written with every population member."""
        store = MagicMock()
        template = {"bot": {"long": {"a": 0.0, "b": 0.0}}}
        scoring_keys = ["adg_pnl", "drawdown"]

        callback = ParetoWriterCallback(
            store, template, scoring_keys,
            results_dir=str(tmp_path),
            write_all_results=True,
        )

        ind1 = self._make_ind([1.0, 2.0], [0.5, 0.3])
        ind2 = self._make_ind([3.0, 4.0], [0.7, 0.1])
        algorithm = MagicMock()
        algorithm.pop = [ind1, ind2]
        algorithm.opt = [ind1]

        callback.notify(algorithm)
        callback.close()

        results_path = os.path.join(str(tmp_path), "all_results.bin")
        assert os.path.exists(results_path)
        records = list(load_results(results_path))
        assert len(records) == 2
        assert records[0]["bot"]["long"]["a"] == 1.0
        assert records[1]["bot"]["long"]["a"] == 3.0

    def test_all_results_bin_diff_compression(self, tmp_path):
        """After 100 entries a full snapshot is emitted, ensuring diff-compression resets."""
        store = MagicMock()
        template = {"bot": {"long": {"a": 0.0}}}

        callback = ParetoWriterCallback(
            store, template, ["obj"],
            results_dir=str(tmp_path),
            write_all_results=True,
        )

        # Write 101 entries across multiple notify calls
        for i in range(101):
            ind = self._make_ind([float(i)], [float(i) * 0.1])
            algorithm = MagicMock()
            algorithm.pop = [ind]
            algorithm.opt = []
            callback.notify(algorithm)

        callback.close()

        records = list(load_results(os.path.join(str(tmp_path), "all_results.bin")))
        assert len(records) == 101
        assert records[100]["bot"]["long"]["a"] == 100.0

    def test_all_results_disabled(self, tmp_path):
        """When write_all_results=False, no bin file is created."""
        store = MagicMock()
        template = {"bot": {"long": {"a": 0.0}}}

        callback = ParetoWriterCallback(
            store, template, ["obj"],
            results_dir=str(tmp_path),
            write_all_results=False,
        )

        ind = self._make_ind([1.0], [0.5])
        algorithm = MagicMock()
        algorithm.pop = [ind]
        algorithm.opt = [ind]
        callback.notify(algorithm)
        callback.close()

        assert not os.path.exists(os.path.join(str(tmp_path), "all_results.bin"))

    def test_close_idempotent(self, tmp_path):
        """Calling close() multiple times doesn't raise."""
        store = MagicMock()
        template = {"bot": {"long": {"a": 0.0}}}
        callback = ParetoWriterCallback(
            store, template, ["obj"],
            results_dir=str(tmp_path),
            write_all_results=True,
        )
        callback.close()
        callback.close()  # should not raise

    def test_template_sections_preserved(self):
        """All template sections (coin_overrides, logging, optimize bounds) survive in output."""
        store = MagicMock()
        store.add_entry = MagicMock(return_value=True)

        template = {
            "backtest": {"start_date": "2021-01-01", "end_date": "2026-03-22"},
            "bot": {"long": {"a": 0.0, "b": 0.0}},
            "coin_overrides": {"HYPE": {"long": {"total_wallet_exposure_limit": 5}}},
            "live": {"leverage": 10, "user": "hyperliquid_01"},
            "logging": {"level": 1},
            "optimize": {
                "bounds": {"long_a": [0.0, 1.0], "long_b": [0.0, 2.0]},
                "limits": [{"metric": "drawdown_worst_usd", "penalize_if": "greater_than", "value": 0.6}],
                "scoring": ["adg_pnl", "drawdown"],
                "iters": 50000,
                "population_size": 100,
                "crossover_eta": 20.0,
            },
        }
        scoring_keys = ["adg_pnl", "drawdown"]

        callback = ParetoWriterCallback(store, template, scoring_keys)

        ind = self._make_ind([1.0, 2.0], [0.5, 0.3])
        algorithm = MagicMock()
        algorithm.pop = [ind]
        algorithm.opt = [ind]

        callback.notify(algorithm)

        entry = store.add_entry.call_args[0][0]
        assert entry["coin_overrides"] == {"HYPE": {"long": {"total_wallet_exposure_limit": 5}}}
        assert entry["logging"] == {"level": 1}
        assert entry["optimize"]["bounds"] == {"long_a": [0.0, 1.0], "long_b": [0.0, 2.0]}
        assert entry["optimize"]["limits"] == template["optimize"]["limits"]
        assert entry["optimize"]["iters"] == 50000
        assert entry["optimize"]["crossover_eta"] == 20.0
        assert entry["backtest"]["start_date"] == "2021-01-01"
        assert entry["live"]["user"] == "hyperliquid_01"
