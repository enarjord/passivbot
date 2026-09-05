"""Shared helpers for structured metric payloads used by suites and pareto outputs."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

import numpy as np

from config.metrics import ANALYSIS_SHARED_KEYS, CURRENCY_METRICS, canonical_metric_name

PERIOD_KEYS = ("effective_start_date", "effective_end_date")


def evaluation_period(payload: Mapping[str, Any]) -> dict:
    """Non-aggregatable evaluation boundaries, kept separate from numeric stats."""
    return {key: payload[key] for key in PERIOD_KEYS if key in payload}


def shared_evaluation_period(payloads: Mapping[str, Mapping[str, Any]]) -> dict:
    periods = [evaluation_period(value) for value in payloads.values()]
    if periods and all(period == periods[0] for period in periods):
        return periods[0]
    return {}


def attach_result_metrics(config: dict, *, metrics=None, suite_metrics=None) -> dict:
    """Replace prior evaluation output on an already sanitized config copy."""
    config.pop("metrics", None)
    config.pop("suite_metrics", None)
    if metrics is not None:
        config["metrics"] = metrics
    if suite_metrics is not None:
        config["suite_metrics"] = suite_metrics
    return config


MetricStats = Dict[str, float]
ScenarioMetrics = Dict[str, Dict[str, MetricStats]]
KNOWN_ANALYSIS_METRICS = set(ANALYSIS_SHARED_KEYS) | set(CURRENCY_METRICS)


class MetricAggregationError(ValueError):
    """Raised when metric payloads cannot be safely aggregated for scoring."""


def _is_number(value: Any) -> bool:
    if isinstance(value, (int, float, np.integer, np.floating)):
        return np.isfinite(value)
    return False


def _is_numeric_value(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))


def _is_known_metric_key(key: Any) -> bool:
    return canonical_metric_name(str(key)) in KNOWN_ANALYSIS_METRICS


def _safe_float(value: Any) -> float:
    if not _is_number(value):
        raise MetricAggregationError(f"non-finite metric value {value!r}")
    return float(value)


def _build_stats(values: Iterable[float]) -> MetricStats:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise MetricAggregationError("cannot build stats from an empty metric sample")
    if not np.all(np.isfinite(arr)):
        raise MetricAggregationError("cannot build stats from non-finite metric values")
    return {
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
    }


def build_scenario_metrics(analyses: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """Combine per-exchange analysis dicts into structured scenario metrics."""

    if not analyses:
        raise MetricAggregationError("no analysis payloads available for scenario metrics")

    stats: Dict[str, MetricStats] = {}
    metric_names = set()
    for label, analysis in analyses.items():
        if not isinstance(analysis, Mapping):
            raise MetricAggregationError(
                f"analysis payload for {label!r} must be a mapping, got {type(analysis).__name__}"
            )
        for metric, raw_value in analysis.items():
            if _is_number(raw_value):
                metric_names.add(metric)
                continue
            if _is_numeric_value(raw_value) or _is_known_metric_key(metric):
                raise MetricAggregationError(
                    f"non-finite metric {metric!r} for {label!r}: {raw_value!r}"
                )
    if not metric_names:
        raise MetricAggregationError("analysis payloads contained no metrics")

    for metric in sorted(metric_names):
        values = []
        for label, analysis in analyses.items():
            if metric not in analysis:
                continue
            raw_value = analysis[metric]
            if not _is_number(raw_value):
                raise MetricAggregationError(
                    f"non-finite metric {metric!r} for {label!r}: {raw_value!r}"
                )
            values.append(_safe_float(raw_value))
        stats[metric] = _build_stats(values)

    payload = {"stats": stats, **shared_evaluation_period(analyses)}
    periods = {label: evaluation_period(analysis) for label, analysis in analyses.items()}
    if any(periods.values()):
        payload["exchanges"] = periods
    return payload


def build_standalone_metrics(analysis: Mapping[str, Any], exchange: str) -> Dict[str, Any]:
    """Persist standalone diagnostics without applying optimizer scoring validation."""
    period = evaluation_period(analysis)
    payload = {
        "stats": {
            metric: _build_stats([value])
            for metric, value in analysis.items()
            if _is_number(value)
        },
        **period,
    }
    if period:
        payload["exchanges"] = {exchange: period}
    diagnostics = {
        metric: str(float(value))
        for metric, value in analysis.items()
        if _is_numeric_value(value) and not _is_number(value)
    }
    if diagnostics:
        payload["nonfinite_diagnostics"] = diagnostics
    return payload


def flatten_metric_stats(stats: Mapping[str, MetricStats], *, prefix: str = "") -> Dict[str, float]:
    """Convert structured stats to the legacy flat format used by scoring/limits."""

    flattened: Dict[str, float] = {}
    required_fields = ("mean", "min", "max", "std", "median")
    for metric, values in stats.items():
        missing = [field for field in required_fields if field not in values]
        if missing:
            raise MetricAggregationError(
                f"metric {metric!r} missing stat field(s): {', '.join(missing)}"
            )
        for field in required_fields:
            key = f"{prefix}{metric}_{field}"
            flattened[key] = _safe_float(values[field])
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
        periods = {
            label: {
                **evaluation_period(metrics),
                **({"exchanges": metrics["exchanges"]} if "exchanges" in metrics else {}),
            }
            for label, metrics in scenario_metrics.items()
        }
        if any(periods.values()):
            payload["scenarios"] = periods
        payload.update(shared_evaluation_period(scenario_metrics))
    return payload
