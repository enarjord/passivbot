from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional, Tuple


ALLOWED_STATS = {"min", "max", "mean", "std"}


def expand_limit_checks(
    limits: Iterable[Dict[str, Any]],
    scoring_weights: Dict[str, float],
    *,
    penalty_weight: float,
    objective_index_map: Optional[Dict[str, List[int]]] = None,
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
        metric = entry.get("metric")
        if not metric:
            continue
        mode = entry.get("penalize_if") or "greater_than"
        if mode == "auto":
            weight = weights.get(metric)
            if weight is None:
                continue
            mode = "less_than" if weight < 0 else "greater_than"
        if mode in {"greater_than", "less_than"}:
            check = _build_single_bound_check(
                entry,
                metric,
                mode,
                penalty_weight,
                objective_index_map,
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
    elif mode == "less_than":
        bound = check["bound"]
        if value < bound:
            return (bound - value) * weight
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


def _normalize_stat(raw_stat: Optional[str], fallback: str) -> str:
    stat = (raw_stat or "").strip().lower()
    if not stat:
        return fallback
    if stat not in ALLOWED_STATS:
        raise ValueError(f"Unsupported stat '{stat}' for limit.")
    return stat


def _build_single_bound_check(
    entry: Dict[str, Any],
    metric: str,
    mode: str,
    penalty_weight: float,
    objective_index_map: Optional[Dict[str, List[int]]],
) -> Optional[Dict[str, Any]]:
    bound = entry.get("value")
    numeric_bound = _ensure_float(bound)
    if numeric_bound is None:
        return None
    fallback_stat = "min" if mode == "less_than" else "max"
    stat = _normalize_stat(entry.get("stat"), fallback_stat)
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
    fallback_stat = "mean"
    stat = _normalize_stat(entry.get("stat"), fallback_stat)
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
