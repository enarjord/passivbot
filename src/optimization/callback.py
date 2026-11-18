from __future__ import annotations

from copy import deepcopy
from typing import Sequence

import numpy as np

from config.result_records import build_result_record
from config.universe import anchored_universe_source_view
from config_utils import clean_config, strip_config_metadata
from optimization.evaluation_contract import CONTRACT_KEY, recorded_evaluation_contract


def build_pymoo_record_entry(
    *,
    vector,
    metrics,
    template: dict,
    build_config_fn,
    overrides_fn,
    overrides_list: Sequence[str] | None = None,
    prepared_config: dict | None = None,
) -> dict:
    """Build a result entry from a vector or its exact prepared config."""
    if isinstance(vector, np.ndarray):
        vector = vector.tolist()
    metrics = dict(metrics or {})
    suite_metrics = metrics.pop("suite_metrics", None)
    config = prepared_config if prepared_config is not None else build_config_fn(
        vector, overrides_fn, list(overrides_list or []), template,
    )
    authored = clean_config(strip_config_metadata(anchored_universe_source_view(config, candidate=True)))
    provenance = {CONTRACT_KEY: recorded_evaluation_contract(template)}
    backtest = config.get("backtest", {})
    if suite_metrics is None and "coins" in backtest:
        provenance["backtest_coins"] = deepcopy(backtest["coins"])
    anchor_meta = config.get("_optimizer_anchor")
    if anchor_meta is not None:
        provenance["optimizer_anchor"] = deepcopy(anchor_meta)
    if callable(overrides_fn):
        authored = overrides_fn(list(overrides_list or []), authored, None)
    return build_result_record(
        authored, provenance=provenance, metrics=metrics or None, suite_metrics=suite_metrics,
    )
