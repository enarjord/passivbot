"""Parameter sampling from Optuna trials."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna

from .models import Bound, Constraint, Objective


def sample_params(
    trial: "optuna.Trial",
    bounds: dict[str, Bound],
    *,
    fixed_params: dict[str, float] | None = None,
) -> dict[str, float]:
    """
    Sample parameters from an Optuna trial.

    Args:
        trial: Optuna trial object
        bounds: Parameter bounds from config
        fixed_params: Parameters to skip sampling (use these values instead)

    Returns:
        Dict of parameter name -> sampled value
    """
    fixed = fixed_params or {}
    params = {}

    for name, bound in bounds.items():
        if name in fixed:
            # Fine-tune mode: use externally fixed value, don't record in trial
            params[name] = fixed[name]
        elif bound.step is not None:
            params[name] = trial.suggest_float(name, bound.low, bound.high, step=bound.step)
        else:
            # Always use suggest_float, even for fixed bounds (low==high).
            # This ensures Optuna records the value in trial.params for Pareto extraction.
            params[name] = trial.suggest_float(name, bound.low, bound.high)

    return params


def check_constraints(
    metrics: dict[str, float],
    constraints: list[Constraint],
) -> list[float]:
    """
    Check constraint violations.

    Args:
        metrics: Dict of metric name -> value
        constraints: List of Constraint objects

    Returns:
        List of violation amounts (0.0 = no violation)
    """
    violations = []
    for c in constraints:
        val = metrics.get(c.metric, 0.0)
        if c.max is not None and val > c.max:
            violations.append(val - c.max)
        elif c.min is not None and val < c.min:
            violations.append(c.min - val)
        else:
            violations.append(0.0)
    return violations


def compute_penalty(violations: list[float], weight: float) -> float:
    """Compute total penalty from constraint violations.

    Args:
        violations: List of violation amounts (positive = violated)
        weight: Penalty weight multiplier

    Returns:
        Sum of positive violations multiplied by weight.
    """
    if weight == 0:
        return 0.0
    return sum(v * weight for v in violations if v > 0)


def compute_scores(
    flat_stats: dict[str, float],
    objectives: list[Objective],
    violations: list[float],
    penalty_weight: float,
) -> tuple[float, ...]:
    """Compute final objective scores for Optuna (minimization).

    Args:
        flat_stats: Flattened metric statistics
        objectives: List of objectives with metrics and directions
        violations: Constraint violation amounts
        penalty_weight: Penalty multiplier (-1 = hard, 0 = disabled, >0 = soft)

    Returns:
        Tuple of scores (one per objective), ready for Optuna.
    """
    scores = [obj.sign * resolve_metric(obj.metric, flat_stats) for obj in objectives]

    if penalty_weight > 0:
        penalty = compute_penalty(violations, penalty_weight)
        scores = [s + penalty for s in scores]

    return tuple(scores)


def resolve_metric(metric: str, flat_stats: dict[str, float]) -> float:
    """Resolve a metric name to its value, trying common suffixes.

    Handles the metric naming convention where flat_stats has keys like
    'mdg_w_btc_mean' but objectives may specify just 'mdg_w'.

    Args:
        metric: Metric name to resolve (e.g., 'mdg_w')
        flat_stats: Dictionary of metric names to values

    Returns:
        The resolved metric value, or 0.0 if not found.
    """
    # Try exact match first
    if metric in flat_stats:
        return flat_stats[metric]

    # Try with _mean suffix
    if f"{metric}_mean" in flat_stats:
        return flat_stats[f"{metric}_mean"]

    # Try with currency suffix + _mean (prefer btc)
    for suffix in ("btc", "usd"):
        key = f"{metric}_{suffix}_mean"
        if key in flat_stats:
            return flat_stats[key]

    return 0.0
