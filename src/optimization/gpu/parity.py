"""Explicit per-metric CPU/GPU comparison; no simulation or runtime imports."""

from dataclasses import dataclass
import math
from collections.abc import Mapping


@dataclass(frozen=True)
class MetricTolerance:
    absolute: float
    relative: float
    matching_infinity: bool = False

    def __post_init__(self):
        for value in (self.absolute, self.relative):
            if not math.isfinite(value) or value < 0:
                raise ValueError("metric tolerances must be finite and non-negative")
        if not isinstance(self.matching_infinity, bool):
            raise TypeError("matching_infinity must be a boolean")


def _json_number(value):
    if value is None or math.isfinite(value):
        return value
    if math.isnan(value):
        return "NaN"
    return "Infinity" if value > 0 else "-Infinity"


def compare_metrics(
    cpu: Mapping[str, float],
    gpu: Mapping[str, float],
    tolerances: Mapping[str, MetricTolerance | None],
) -> dict:
    """Compare only requested names, without inventing missing metrics or policies.

    A missing tolerance leaves a finite observation unassessed. Non-finite values
    never pass unless matching infinities are explicitly allowed for that metric.
    Reports contain standard-JSON values, including named non-finite sentinels.
    """
    if not tolerances:
        raise ValueError("at least one requested metric is required")
    rows = {}
    for metric, tolerance in tolerances.items():
        if not isinstance(metric, str) or not metric:
            raise ValueError("metric names must be non-empty strings")
        if tolerance is not None and not isinstance(tolerance, MetricTolerance):
            raise TypeError(f"invalid tolerance for {metric}")
        reference = float(cpu[metric]) if metric in cpu else None
        observed = float(gpu[metric]) if metric in gpu else None
        absolute_error = relative_error = allowed_error = None
        if reference is None or observed is None:
            status = "missing_cpu" if reference is None else "missing_gpu"
            if reference is None and observed is None:
                status = "missing_both"
        elif not (math.isfinite(reference) and math.isfinite(observed)):
            sentinel_match = (
                tolerance is not None
                and tolerance.matching_infinity
                and math.isinf(reference)
                and reference == observed
            )
            status = "sentinel_match" if sentinel_match else "non_finite"
        else:
            absolute_error = abs(observed - reference)
            scale = max(abs(reference), abs(observed))
            relative_error = absolute_error / scale if scale > 0 else 0.0
            if tolerance is None:
                status = "unassessed"
            else:
                allowed_error = tolerance.absolute + tolerance.relative * abs(reference)
                # Overflow in derived errors is not a numerical agreement.
                status = "match" if (
                    math.isfinite(absolute_error)
                    and math.isfinite(allowed_error)
                    and absolute_error <= allowed_error
                ) else "mismatch"
        rows[metric] = {
            "status": status,
            "cpu": _json_number(reference),
            "gpu": _json_number(observed),
            "absolute_error": _json_number(absolute_error),
            "relative_error": _json_number(relative_error),
            "allowed_error": _json_number(allowed_error),
        }
    passed = all(row["status"] in {"match", "sentinel_match"} for row in rows.values())
    return {"passed": passed, "metrics": rows}


def compare_limits(cpu, gpu, checks):
    """Apply canonical scalar/single-scenario limits, reporting feasibility flips.

    These are diagnostic CPU calculations, not another backtest. Missing metrics
    and suite-scoped checks remain explicitly unassessed rather than feasible.
    """
    from config.metrics import resolve_metric_value
    from limit_utils import compute_limit_violation

    def surface(metrics):
        values = dict(metrics)
        for name, value in metrics.items():
            for reducer in ("mean", "min", "max", "median"):
                values[f"{name}_{reducer}"] = value
            # A singleton finite observation has zero dispersion. Non-finite
            # source observations cannot become an apparently valid std metric.
            values[f"{name}_std"] = 0.0 if math.isfinite(value) else math.nan
        return values

    references, observations = surface(cpu), surface(gpu)
    rows = []
    cpu_penalty = gpu_penalty = 0.0
    assessed = True
    for check in checks:
        reference = resolve_metric_value(references, check["metric_key"])
        observed = resolve_metric_value(observations, check["metric_key"])
        row = {"metric": check["metric"], "mode": check["mode"]}
        if check.get("scenario") is not None:
            row["status"] = "suite_required"
            assessed = False
        elif reference is None or observed is None or not (
            math.isfinite(reference) and math.isfinite(observed)
        ):
            row["status"] = "unavailable"
            assessed = False
        else:
            cpu_violation = compute_limit_violation(check, reference)
            gpu_violation = compute_limit_violation(check, observed)
            cpu_penalty += cpu_violation
            gpu_penalty += gpu_violation
            row.update(
                cpu_violation=_json_number(cpu_violation),
                gpu_violation=_json_number(gpu_violation),
                status="match" if (cpu_violation > 0) == (gpu_violation > 0) else "flip",
            )
        rows.append(row)
    return {
        "assessed": assessed,
        "passed": assessed and all(row["status"] == "match" for row in rows),
        "cpu_feasible": cpu_penalty == 0 if assessed else None,
        "gpu_feasible": gpu_penalty == 0 if assessed else None,
        "cpu_violation": _json_number(cpu_penalty) if assessed else None,
        "gpu_violation": _json_number(gpu_penalty) if assessed else None,
        "checks": rows,
    }
