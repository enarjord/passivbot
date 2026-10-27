"""Versioned core values, explicit statuses and requested raw reductions.

Python reduces producer values but never reimplements a trading metric formula.
All persisted payloads are JSON-safe: extended-real endpoints use null plus status.
"""

from collections.abc import Mapping
from copy import deepcopy
import math
import statistics
from numbers import Real

from config.metrics import (
    CORE_METRIC_DEFINITION_VERSION,
    KNOWN_METRICS,
    STAT_SUFFIXES,
    metric_base_name,
)

PERIOD_KEYS = ("effective_start_date", "effective_end_date")


class MetricAggregationError(ValueError):
    """A selected quantity has incomplete or undefined observation support."""

    def __init__(self, message, *, status="invalid"):
        super().__init__(message)
        self.status = status


def evaluation_period(payload):
    return {key: payload[key] for key in PERIOD_KEYS if key in payload}


def shared_evaluation_period(payloads):
    periods = [evaluation_period(value) for value in payloads.values()]
    return periods[0] if periods and all(item == periods[0] for item in periods) else {}


def attach_result_metrics(config, *, metrics=None, suite_metrics=None):
    config.pop("metrics", None)
    config.pop("suite_metrics", None)
    if metrics is not None:
        config["metrics"] = metrics
    if suite_metrics is not None:
        config["suite_metrics"] = suite_metrics
    return config


def decode_metric_value(payload, name):
    if name not in payload:
        raise MetricAggregationError(f"required metric {name!r} is missing")
    value = payload[name]
    status = payload.get("metric_status", {}).get(name, {})
    tag = status.get("status", "finite") if isinstance(status, Mapping) else None
    if tag in {"positive_infinity", "negative_infinity"}:
        if value is not None:
            raise MetricAggregationError(
                f"infinite metric {name!r} must use null plus status"
            )
        return math.inf if tag == "positive_infinity" else -math.inf
    if tag != "finite" or value is None:
        raise MetricAggregationError(
            f"required metric {name!r} unavailable: {status!r}",
            status="undefined" if tag == "undefined" else "invalid",
        )
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
    ):
        raise MetricAggregationError(f"malformed finite metric {name!r}: {value!r}")
    return float(value)


def encode_metric_values(values):
    result, statuses = {}, {}
    for name, raw in values.items():
        if isinstance(raw, bool) or not isinstance(raw, Real) or math.isnan(raw):
            raise MetricAggregationError(
                f"cannot serialize undefined raw metric {name!r}"
            )
        value = float(raw)
        result[name] = value if math.isfinite(value) else None
        if not math.isfinite(value):
            statuses[name] = {
                "status": "positive_infinity" if value > 0 else "negative_infinity"
            }
    if statuses:
        result["metric_status"] = statuses
    return result


def copy_metric_value(payload, name, *, output_name=None):
    """Copy one validated JSON-safe fact, preserving explicit unavailable statuses."""
    target = name if output_name is None else output_name
    if name not in payload:
        raise MetricAggregationError(f"required metric {name!r} is missing")
    status = payload.get("metric_status", {}).get(name)
    if (
        payload[name] is None
        and isinstance(status, Mapping)
        and status.get("status") in {"undefined", "invalid"}
    ):
        return {target: None, "metric_status": {target: deepcopy(status)}}
    encoded = encode_metric_values({target: decode_metric_value(payload, name)})
    if isinstance(status, Mapping):
        encoded.setdefault("metric_status", {})[target] = deepcopy(status)
    return encoded


def raw_reduce(values, mode):
    values = list(values)
    if not values or any(
        isinstance(x, bool) or not isinstance(x, Real) or math.isnan(x) for x in values
    ):
        raise MetricAggregationError("empty or undefined reduction population")
    if mode == "min":
        return min(values)
    if mode == "max":
        return max(values)
    if mode == "median":
        ordered = sorted(values)
        middle = len(ordered) // 2
        return (
            ordered[middle]
            if len(ordered) % 2
            else raw_reduce(ordered[middle - 1 : middle + 1], "mean")
        )
    positive, negative = math.inf in values, -math.inf in values
    if mode == "std":
        if positive or negative:
            raise MetricAggregationError(
                "standard deviation of infinite inputs is undefined", status="undefined"
            )
        value = statistics.pstdev(values)
    elif mode == "mean":
        if positive and negative:
            raise MetricAggregationError(
                "opposing infinities make the mean undefined", status="undefined"
            )
        if positive or negative:
            return math.inf if positive else -math.inf
        try:
            value = math.fsum(x / len(values) for x in values)
        except OverflowError as error:
            raise MetricAggregationError("finite reduction overflow") from error
    else:
        raise ValueError(f"unsupported metric reducer {mode!r}")
    if not math.isfinite(value):
        raise MetricAggregationError("finite reduction overflow")
    return value


def require_metric_definition(payload):
    metadata = payload.get("metric_metadata", {})
    if not isinstance(metadata, Mapping):
        raise MetricAggregationError("malformed core metric metadata")
    version = payload.get("definition_version", metadata.get("definition_version"))
    if isinstance(version, bool) or version != CORE_METRIC_DEFINITION_VERSION:
        raise MetricAggregationError(
            "core metric definition version missing or stale; re-evaluate this result"
        )


def validate_required_core_metrics(analyses, required_metrics):
    required = {metric_base_name(name) for name in required_metrics}
    for label, analysis in analyses.items():
        require_metric_definition(analysis)
        for name in sorted(required):
            if name not in KNOWN_METRICS:
                raise MetricAggregationError(f"unknown core metric {name!r}")
            try:
                decode_metric_value(analysis, name)
            except MetricAggregationError as error:
                raise MetricAggregationError(
                    f"required metric {name!r} unavailable for {label!r}: {error}"
                ) from error


def build_scenario_metrics(analyses, *, required_metrics=None, reducers=None):
    if not analyses:
        raise MetricAggregationError("no analysis sources available")
    required = (
        None
        if required_metrics is None
        else {metric_base_name(name) for name in required_metrics}
    )
    if required is not None:
        validate_required_core_metrics(analyses, required)
    else:
        for analysis in analyses.values():
            require_metric_definition(analysis)
    names = (
        required
        if required is not None
        else set().union(*(set(row) & KNOWN_METRICS for row in analyses.values()))
    )
    stats = {}
    for name in sorted(names):
        modes = set((reducers or {}).get(name, {"mean"}))
        try:
            values = [decode_metric_value(row, name) for row in analyses.values()]
            stats[name] = encode_metric_values(
                {mode: raw_reduce(values, mode) for mode in sorted(modes)}
            )
        except MetricAggregationError as error:
            if required is not None and name in required:
                raise
            stats[name] = {mode: None for mode in modes}
            stats[name]["metric_status"] = {
                mode: {"status": error.status, "reason": str(error)} for mode in modes
            }
    return {
        "definition_version": CORE_METRIC_DEFINITION_VERSION,
        "stats": stats,
        **shared_evaluation_period(analyses),
        "sources": {
            label: deepcopy(row.get("metric_metadata", {}))
            for label, row in analyses.items()
        },
    }


def build_standalone_metrics(analysis, exchange):
    return build_scenario_metrics({exchange: analysis})


def flatten_metric_stats(stats, *, prefix=""):
    values, unavailable = {}, {}
    for name, fields in stats.items():
        if name not in KNOWN_METRICS:
            raise MetricAggregationError(f"unknown core metric in statistics: {name!r}")
        for mode in STAT_SUFFIXES:
            if mode not in fields:
                continue
            key = f"{prefix}{name}_{mode}"
            try:
                values[key] = decode_metric_value(fields, mode)
            except MetricAggregationError:
                unavailable[key] = fields.get("metric_status", {}).get(
                    mode, {"status": "undefined"}
                )
    result = encode_metric_values(values)
    if unavailable:
        result.update({key: None for key in unavailable})
        result.setdefault("metric_status", {}).update(unavailable)
    return result


def merge_suite_payload(
    aggregate_stats, *, aggregate_values=None, scenario_metrics=None
):
    scenario_metrics = scenario_metrics or {}
    names = set(aggregate_stats).union(
        *(set(row.get("stats", {})) for row in scenario_metrics.values())
    )
    metric_rows = {}
    for name in sorted(names):
        row = {"stats": deepcopy(aggregate_stats.get(name, {})), "scenarios": {}}
        if aggregate_values is not None and name in aggregate_values:
            row.update(
                encode_metric_values(
                    {"aggregated": decode_metric_value(aggregate_values, name)}
                )
            )
        elif "mean" in row["stats"]:
            try:
                row.update(
                    encode_metric_values(
                        {"aggregated": decode_metric_value(row["stats"], "mean")}
                    )
                )
            except MetricAggregationError as error:
                row["aggregated"] = None
                row.setdefault("metric_status", {})["aggregated"] = {
                    "status": error.status,
                    "reason": str(error),
                }
        for label, scenario in scenario_metrics.items():
            fields = scenario.get("stats", {}).get(name)
            if fields is not None:
                try:
                    value = decode_metric_value(fields, "mean")
                    encoded = encode_metric_values({label: value})
                    statuses = encoded.pop("metric_status", {})
                    row["scenarios"].update(encoded)
                    if statuses:
                        row["scenarios"].setdefault("metric_status", {}).update(
                            statuses
                        )
                except MetricAggregationError as error:
                    row["scenarios"][label] = None
                    row["scenarios"].setdefault("metric_status", {})[label] = {
                        "status": error.status,
                        "reason": str(error),
                    }
        metric_rows[name] = row
    result = {
        "definition_version": CORE_METRIC_DEFINITION_VERSION,
        "metrics": metric_rows,
    }
    if scenario_metrics:
        result["scenario_labels"] = list(scenario_metrics)
        result["sources"] = {
            label: deepcopy(row.get("sources", {}))
            for label, row in scenario_metrics.items()
        }
        result.update(shared_evaluation_period(scenario_metrics))
    return result
