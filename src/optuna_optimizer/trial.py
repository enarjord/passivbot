"""Parameter sampling from Optuna trials."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import optuna

from optuna.distributions import FloatDistribution

from .models import Bound, Constraint, Objective


def build_distributions(bounds: dict[str, Bound]) -> dict[str, FloatDistribution]:
    """Build Optuna distributions from bounds for use with study.ask(fixed_distributions=...).

    Pre-defining distributions provides ~10x speedup for suggest_float calls.

    Args:
        bounds: Parameter bounds from config

    Returns:
        Dict of param name -> FloatDistribution
    """
    distributions = {}
    for name, bound in bounds.items():
        distributions[name] = FloatDistribution(
            low=bound.low,
            high=bound.high,
            step=bound.step,
        )
    return distributions


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


def compute_scores(
    flat_stats: dict[str, float],
    objectives: list[Objective],
) -> tuple[float, ...]:
    """Compute final objective scores for Optuna (minimization)."""
    return tuple(obj.sign * resolve_metric(obj.metric, flat_stats) for obj in objectives)


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
