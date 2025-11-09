"""Shared helpers for structured metric payloads used by suites and pareto outputs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import numpy as np


MetricStats = Dict[str, float]
ScenarioMetrics = Dict[str, Dict[str, MetricStats]]


def _is_number(value: Any) -> bool:
    if isinstance(value, (int, float, np.floating)):
        return np.isfinite(value)
    return False


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _build_stats(values: Iterable[float]) -> MetricStats:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def build_scenario_metrics(analyses: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Combine per-exchange analysis dicts into structured scenario metrics."""

    stats: Dict[str, MetricStats] = {}
    metric_names = set()
    for analysis in analyses.values():
        metric_names.update(analysis.keys())

    for metric in sorted(metric_names):
        values = [
            _safe_float(analysis.get(metric))
            for analysis in analyses.values()
            if _is_number(analysis.get(metric))
        ]
        stats[metric] = _build_stats(values)

    return {"stats": stats}


def flatten_metric_stats(stats: Mapping[str, MetricStats], *, prefix: str = "") -> Dict[str, float]:
    """Convert structured stats to the legacy flat format used by scoring/limits."""

    flattened: Dict[str, float] = {}
    for metric, values in stats.items():
        for field in ("mean", "min", "max", "std"):
            key = f"{prefix}{metric}_{field}"
            flattened[key] = float(values.get(field, 0.0))
    return flattened


def merge_suite_payload(
    aggregate_stats: Mapping[str, MetricStats],
    *,
    aggregate_values: Mapping[str, float] | None = None,
    scenario_labels: Iterable[str] | None = None,
) -> Dict[str, Any]:
    payload = {
        "aggregate": {
            "aggregated": dict(aggregate_values or {}),
            "stats": dict(aggregate_stats),
        },
    }
    if scenario_labels:
        payload["scenarios"] = list(scenario_labels)
    return payload
