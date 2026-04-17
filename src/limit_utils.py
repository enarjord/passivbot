from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple

from config.limits import resolve_limit_stat

_BOUNDARY_VIOLATION_EPSILON = 1e-12


def expand_limit_checks(
    limits: Iterable[Dict[str, Any]],
    scoring_weights: Dict[str, float],
    *,
    penalty_weight: float,
    objective_index_map: Optional[Dict[str, List[int]]] = None,
    aggregate_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Transform normalized limit entries into executable checks.
    """
    if not limits:
        return []
    weights = scoring_weights or {}
    checks: List[Dict[str, Any]] = []
    for raw_entry in limits:
        entry = deepcopy(raw_entry)
        if isinstance(entry, dict) and not bool(entry.get("enabled", True)):
            continue
        metric = entry.get("metric")
        if not metric:
            continue
        mode = entry.get("penalize_if") or "greater_than"
        if mode == "auto":
            weight = weights.get(metric)
            if weight is None:
                continue
            mode = "less_than" if weight < 0 else "greater_than"
        if mode in {
            "greater_than",
            "greater_than_or_equal",
            "less_than",
            "less_than_or_equal",
            "equal_to",
            "not_equal",
        }:
            check = _build_single_bound_check(
                entry,
                metric,
                mode,
                penalty_weight,
                objective_index_map,
                aggregate_cfg,
            )
            if check:
                checks.append(check)
        elif mode == "outside_range":
            check = _build_range_check(
                entry,
                metric,
                penalty_weight,
                mode,
                objective_index_map,
                aggregate_cfg,
            )
            if check:
                checks.append(check)
        elif mode == "inside_range":
            check = _build_range_check(
                entry,
                metric,
                penalty_weight,
                mode,
                objective_index_map,
                aggregate_cfg,
            )
            if check:
                checks.append(check)
        else:
            raise ValueError(f"Unsupported penalize_if '{mode}' for limit on {metric}.")
    return checks


def compute_limit_violation(check: Dict[str, Any], value: Optional[float]) -> float:
    if value is None:
        return 0.0
    weight = check.get("penalty_weight", 1.0)
    mode = check["mode"]
    if mode == "greater_than":
        bound = check["bound"]
        if value > bound:
            return (value - bound) * weight
    elif mode == "greater_than_or_equal":
        bound = check["bound"]
        if value >= bound:
            return max(value - bound, _BOUNDARY_VIOLATION_EPSILON) * weight
    elif mode == "less_than":
        bound = check["bound"]
        if value < bound:
            return (bound - value) * weight
    elif mode == "less_than_or_equal":
        bound = check["bound"]
        if value <= bound:
            return max(bound - value, _BOUNDARY_VIOLATION_EPSILON) * weight
    elif mode == "equal_to":
        bound = check["bound"]
        if value == bound:
            return _BOUNDARY_VIOLATION_EPSILON * weight
    elif mode == "not_equal":
        bound = check["bound"]
        if value != bound:
            return max(abs(value - bound), _BOUNDARY_VIOLATION_EPSILON) * weight
    elif mode == "outside_range":
        low, high = check["range"]
        if value < low:
            return (low - value) * weight
        if value > high:
            return (value - high) * weight
    elif mode == "inside_range":
        low, high = check["range"]
        if low <= value <= high:
            distance = min(value - low, high - value)
            return distance * weight
    else:
        raise ValueError(f"Unsupported limit mode '{mode}'.")
    return 0.0


def _ensure_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_single_bound_check(
    entry: Dict[str, Any],
    metric: str,
    mode: str,
    penalty_weight: float,
    objective_index_map: Optional[Dict[str, List[int]]],
    aggregate_cfg: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    bound = entry.get("value")
    numeric_bound = _ensure_float(bound)
    if numeric_bound is None:
        return None
    stat = resolve_limit_stat(entry, aggregate_cfg=aggregate_cfg)
    metric_key = f"{metric}_{stat}"
    return {
        "metric": metric,
        "metric_key": metric_key,
        "mode": mode,
        "bound": numeric_bound,
        "penalty_weight": penalty_weight,
        "stat": stat,
        "objective_indexes": list(objective_index_map.get(metric, [])) if objective_index_map else [],
    }


def _build_range_check(
    entry: Dict[str, Any],
    metric: str,
    penalty_weight: float,
    mode: str,
    objective_index_map: Optional[Dict[str, List[int]]],
    aggregate_cfg: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    rng = entry.get("range")
    if not isinstance(rng, (list, tuple)) or len(rng) != 2:
        return None
    low = _ensure_float(rng[0])
    high = _ensure_float(rng[1])
    if low is None or high is None:
        return None
    if low > high:
        low, high = high, low
    stat = resolve_limit_stat(entry, aggregate_cfg=aggregate_cfg)
    metric_key = f"{metric}_{stat}"
    return {
        "metric": metric,
        "metric_key": metric_key,
        "mode": mode,
        "range": (low, high),
        "penalty_weight": penalty_weight,
        "stat": stat,
        "objective_indexes": list(objective_index_map.get(metric, [])) if objective_index_map else [],
    }
