"""
pymoo Callback for writing Pareto front configs to disk each generation.
"""

import logging
import os

import msgpack
from pymoo.core.callback import Callback

from optimization.bounds import individual_to_config
from opt_utils import make_json_serializable, generate_incremental_diff
from config_utils import strip_config_metadata


class ParetoWriterCallback(Callback):
    """
    After each generation, writes the current Pareto-optimal configs
    to disk via the ParetoStore.

    Optionally writes every evaluation result to ``all_results.bin``
    (msgpack format with diff-compression) for use by downstream tools
    such as ``pareto_dash.py``.
    """

    def __init__(
        self,
        store,
        template,
        scoring_keys,
        overrides_fn=None,
        overrides_list=None,
        results_dir=None,
        write_all_results=True,
    ):
        super().__init__()
        self.store = store
        self.template = template
        self.scoring_keys = scoring_keys
        self.overrides_fn = overrides_fn
        self.overrides_list = overrides_list or []
        self.log = logging.getLogger(__name__)

        # all_results.bin writer state
        self._results_file = None
        self._packer = None
        self._prev_data = None
        self._counter = 0
        if write_all_results and results_dir:
            filepath = os.path.join(results_dir, "all_results.bin")
            self._results_file = open(filepath, "ab")
            self._packer = msgpack.Packer(use_bin_type=True)

    def _build_entry(self, ind):
        """Build a result entry dict from a pymoo Individual."""
        config = individual_to_config(
            ind.X, self.overrides_fn, self.overrides_list, self.template
        )
        objectives_map = {f"w_{i}": float(v) for i, v in enumerate(ind.F)}
        constraint_violation = (
            float(sum(max(0, g) for g in ind.G))
            if ind.G is not None and len(ind.G) > 0
            else 0.0
        )
        metrics = (
            ind.data.get("metrics", {})
            if hasattr(ind, "data") and isinstance(ind.data, dict)
            else {}
        )
        entry = dict(config)
        entry["metrics"] = metrics
        entry["objectives"] = objectives_map
        entry["constraint_violation"] = constraint_violation
        return strip_config_metadata(entry)

    def _write_result(self, entry):
        """Append a single entry to all_results.bin with diff-compression."""
        if self._results_file is None:
            return
        if self._prev_data is None or self._counter % 100 == 0:
            output_data = make_json_serializable(entry)
        else:
            diff = generate_incremental_diff(self._prev_data, entry)
            output_data = make_json_serializable(diff)
        self._counter += 1
        self._prev_data = entry
        try:
            self._results_file.write(self._packer.pack(output_data))
            self._results_file.flush()
        except Exception as exc:
            self.log.error("Error writing to all_results.bin: %s", exc)

    def notify(self, algorithm):
        pop = algorithm.pop
        opt = algorithm.opt

        # Write every individual in the population to all_results.bin
        if pop is not None:
            for ind in pop:
                entry = self._build_entry(ind)
                self._write_result(entry)

        # Write Pareto-optimal configs to the ParetoStore
        if opt is not None:
            for ind in opt:
                entry = self._build_entry(ind)
                self.store.add_entry(entry)

        self.store.flush()

    def close(self):
        """Close the all_results.bin file handle."""
        if self._results_file is not None:
            self._results_file.close()
            self._results_file = None
