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
    scenario_metrics: Mapping[str, Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """
    Build the suite metrics payload embedded in Pareto members.

    Returns a structure where each metric contains aggregate stats/value
    plus per-scenario means when available.
    """

    scenario_metrics = scenario_metrics or {}
    metric_names = set(aggregate_stats.keys())
    for stats in scenario_metrics.values():
        metric_names.update(stats.get("stats", {}).keys())

    suite_metrics: Dict[str, Any] = {}
    for metric in sorted(metric_names):
        aggregate_stat = dict(aggregate_stats.get(metric, {}))
        aggregated_value = None
        if aggregate_values and metric in aggregate_values:
            aggregated_value = aggregate_values[metric]
        elif aggregate_stat:
            aggregated_value = aggregate_stat.get("mean")
        metric_entry = {
            "stats": aggregate_stat,
            "aggregated": aggregated_value,
            "scenarios": {},
        }
        for label, stats in scenario_metrics.items():
            scenario_stat = stats.get("stats", {}).get(metric)
            if scenario_stat:
                value = scenario_stat.get("mean")
                if value is not None:
                    metric_entry["scenarios"][label] = value
        suite_metrics[metric] = metric_entry

    payload: Dict[str, Any] = {"metrics": suite_metrics}
    if scenario_metrics:
        payload["scenario_labels"] = list(scenario_metrics.keys())
    return payload
