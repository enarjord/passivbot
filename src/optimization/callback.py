from __future__ import annotations

from typing import Sequence

import numpy as np
from pymoo.core.callback import Callback

from config_utils import strip_config_metadata


class PymooRecorderCallback(Callback):
    def __init__(
        self,
        *,
        recorder,
        template: dict,
        build_config_fn,
        overrides_fn,
        overrides_list: Sequence[str] | None = None,
    ):
        super().__init__()
        self.recorder = recorder
        self.template = template
        self.build_config_fn = build_config_fn
        self.overrides_fn = overrides_fn
        self.overrides_list = list(overrides_list or [])

    def _build_entry(self, individual) -> dict:
        vector = None
        if hasattr(individual, "data") and isinstance(individual.data, dict):
            vector = individual.data.get("evaluation_vector")
        if vector is None:
            vector = individual.X
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
        metrics = {}
        if hasattr(individual, "data") and isinstance(individual.data, dict):
            metrics = dict(individual.data.get("metrics") or {})
        suite_metrics = metrics.pop("suite_metrics", None)
        config = self.build_config_fn(
            vector,
            self.overrides_fn,
            self.overrides_list,
            self.template,
        )
        entry = dict(config)
        if suite_metrics is not None:
            entry["suite_metrics"] = suite_metrics
            backtest = entry.get("backtest")
            if isinstance(backtest, dict):
                backtest.pop("coins", None)
        if metrics:
            entry["metrics"] = metrics
        return strip_config_metadata(entry)

    def notify(self, algorithm):
        batch = getattr(algorithm, "off", None) or getattr(algorithm, "pop", None) or []
        for individual in batch:
            self.recorder.record(self._build_entry(individual))
