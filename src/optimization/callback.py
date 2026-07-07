from __future__ import annotations

from typing import Sequence

import numpy as np

from config_utils import clean_config, strip_config_metadata


def build_pymoo_record_entry(
    *,
    vector,
    metrics,
    template: dict,
    build_config_fn,
    overrides_fn,
    overrides_list: Sequence[str] | None = None,
) -> dict:
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    metrics = dict(metrics or {})
    suite_metrics = metrics.pop("suite_metrics", None)
    config = build_config_fn(
        vector,
        overrides_fn,
        list(overrides_list or []),
        template,
    )
    anchor_meta = config.get("_optimizer_anchor")
    entry = clean_config(strip_config_metadata(config))
    if anchor_meta is not None:
        entry["optimizer_anchor"] = anchor_meta
    if callable(overrides_fn):
        entry = overrides_fn(list(overrides_list or []), entry, None)
    if suite_metrics is not None:
        entry["suite_metrics"] = suite_metrics
        backtest = entry.get("backtest")
        if isinstance(backtest, dict):
            backtest.pop("coins", None)
    if metrics:
        entry["metrics"] = metrics
    return entry
